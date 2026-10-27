"""Urban assignments: populated blocks merged into assignments of a target
size, the way a planner groups streets for a team.

WHY GROUP
---------
`populated_areas` draws one block per street run. In a dense suburb that is
sixty blocks inside half a mile, from a 0.2 ha cul-de-sac to an 11 ha office
park - not sixty assignments. A team is handed ground of roughly one size;
this module merges neighbouring blocks into groups of about `target_ha`. The
target defaults to the corpus median for the resource type
(`norms.target_area_range_ha`, 10 ha for GROUND); the bayside-suburb test liked
11.7 ha, the size of its largest single block. A type the corpus has no
median for is sized by the median of every assignment, and the grouping's
`target_source` and a warning say so: a typo must not read as a median.

THE RULES
---------
A block is one street, both sides, out to the back fences
(`populated_areas`): one team walks it, and its edges are back fences or,
at the ends of the run, the cross street. Groups keep that: an assignment
is whole streets, bounded by back fences and streets, never one side of a
street (a hand-drawn urban search map split its assignment at the back
fence between two streets for this).
- Two blocks are NEIGHBOURS when they are within `NEIGHBOUR_GAP_M` and the
  shortest line between them crosses no ARTERIAL - a motorway, trunk,
  primary or secondary road, or their links. An arterial is an edge a
  planner draws to: a team does not work both sides of a boulevard, and a
  closed lane is where the next team starts.
- Neighbours are STREET-LINKED (`street_links`) when their streets join: two
  runs of one street, or a side street meeting the other's within
  `JUNCTION_M`, both blocks within `JUNCTION_REACH_M` of the junction - a
  cul-de-sac and the road it opens onto. Neighbours that merely back onto
  each other, a fence between them, are not.
- Pass 1 merges smallest first ALONG THE STREETS: the smallest group still
  under `SETTLED_SHARE` of the target joins the street-linked neighbour
  that keeps the merged shape most compact (area over convex-hull area),
  as long as the result stays under `MAX_SHARE` of the target and is drawn
  in one piece once the other blocks are cut out of it. A group
  over the settled share is left alone. Pass 1b: a group still under
  `FENCE_SHARE` of the target, with no street left to grow along, joins a
  block it backs onto, by the same choice.
- Pass 2 pairs what is still small across WATER: two small groups more
  than `NEIGHBOUR_GAP_M` and within `WATER_GAP_M` of each other, never across an arterial, one pair per group
  - two isles a team drives between. The pair is drawn as one polygon by a
  `LINK_HALF_M`-wide strip along the shortest gap, for the drawing only.
- The outline is the members, a `CLOSE_M` closing over the streets between
  them, and holes filled when no other group's block is inside. A group
  more than half of whose block ground the larger groups' outlines reach,
  together - standing in their courtyards or slots, closer to their blocks
  than the closing spans - is HEMMED: it joins the one whose outline
  reaches most of it (on a tie the larger), its blocks counted and drawn
  there, any part standing proud of the slot's mouth too. A hemmed water
  pair brings its strip only when its partner would otherwise be cut off -
  an isle across the water - and the host then counts the water link. A
  hemmed join is not held to `MAX_SHARE`; an assignment it carries past
  says so in its warnings. A group half reached or less keeps its blocks
  whole: a block is drawn only in the group that counts it, as an outline
  gives up every other group's block, so the larger outline goes round
  them. Street ground two outlines both reach goes to the larger group. A
  group the cut leaves in pieces keeps every piece that holds a block, and
  says so: the CalTopo export draws each piece as its own assignment.

NEIGHBOURS SHARE ONE EDGE
-------------------------
Last, the outlines are snapped together (`snap_outlines`), as a planner
draws them on CalTopo: the ground between two assignments no more than
`SNAP_GAP_M` apart - the street, the boulevard, a fence line's strip - is
split down its middle, every point going to the nearer, so the two share
one edge, vertex for vertex, with no gap and no overlap. Only ground
between two assignments is taken: an outer edge facing ground no other
assignment is near stays at its back fences. Nothing a block holds moves,
so the counts still describe the ground drawn; `area_ha` includes the
street ground each assignment took.

A group ignores the search circle: it holds whole blocks, and a block the
circle reaches is kept whole, as `populated_areas` keeps it - the ring is
extended out to the back fences beyond the nearest street, never cut.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..geo import ground_meters_per_mercator_unit, to_wgs84_geojson

#: OpenStreetMap road classes a group never crosses.
ARTERIALS = frozenset({"motorway", "motorway_link", "trunk", "trunk_link", "primary",
                       "primary_link", "secondary", "secondary_link"})
#: Blocks this close (ground metres) are neighbours: a residential street
#: and its verges are 15 to 25 m across.
NEIGHBOUR_GAP_M = 26.0
#: Closing radius that fills the street between merged blocks.
CLOSE_M = 14.0
#: A merge may not grow a group past this share of the target...
MAX_SHARE = 1.4
#: ...and a group past this share is left alone.
SETTLED_SHARE = 0.6
#: Two blocks on different streets are joined by those streets when the
#: centrelines meet within this (ground metres; OpenStreetMap draws a T
#: junction's stem ending at, or a few metres short of, the through road)...
JUNCTION_M = 15.0
#: ...and both blocks come within this of the junction: the corner lots of a
#: side street stand a right-of-way and a front yard from the through
#: road's centreline, and OpenStreetMap's line can sit 10-15 m off the
#: parcel map's.
JUNCTION_REACH_M = 40.0
#: A group still under this share of the target, with no street left to
#: grow along, may join a block it backs onto (a fence between them) - the
#: last resort for a short isolated street, never the way groups are built.
FENCE_SHARE = 0.3
#: Pass 2: small groups this close across water may pair.
WATER_GAP_M = 150.0
#: Half-width of the strip that draws a water pair as one polygon.
LINK_HALF_M = 6.0
#: Ground between two assignments this close (ground metres, across) is
#: split between them, so they share one edge: a residential street is 15
#: to 25 m, a boulevard with a rail line beside it about 80.
SNAP_GAP_M = 90.0
#: ...and only where the two outlines lie on either side of it: the
#: directions to the nearest point of each are at least this many degrees
#: apart. Where two neighbours' outer corners meet, both lie to one side,
#: and the ground beyond their outer edge is left alone.
SNAP_OPPOSITE_DEG = 120.0
#: Ground taken across a wider gap is opened by this radius, so the
#: direction test's flicker along an outer edge leaves no spikes; a gap
#: under SNAP_THIN_M across (a lot line, a fence strip) is always taken.
SNAP_OPEN_M = 3.0
SNAP_THIN_M = 8.0
#: Ground metres per cell of the grid the gaps are split on, and the
#: tolerance the shared edges are simplified to; coarser for a big town, so
#: the grid stays under SNAP_MAX_SIDE_PX a side.
SNAP_PIXEL_M = 1.0
SNAP_MAX_SIDE_PX = 3000
#: An outline's edge is drawn out to a street centreline this close
#: (ground metres, edge to centreline): half a boulevard's right-of-way
#: and a little for OpenStreetMap's line not lying in its middle.
SNAP_STREET_M = 30.0
#: ...where the street lies straight ahead of the edge: the directions from
#: a cell to the outline and to the centreline at least this many degrees
#: apart, so no finger runs along the street past the run (a junction's
#: corner is filled by SNAP_CORNER_M2, a notch by SNAP_NOTCH_M, the rest
#: of a city block by SNAP_FACE_SHARE).
SNAP_FACING_DEG = 150.0
#: A strip between two carriageways of one road no wider than this goes
#: with the ground beside it: a boulevard's median.
SNAP_MEDIAN_M = 20.0
#: A piece of ground street centrelines and outlines enclose, no bigger
#: than this (m2) - the corner of a junction - goes to the nearest of them.
SNAP_CORNER_M2 = 600.0
#: A notch in the assignments that opens onto a street - lots no run took,
#: between two runs' ends and the cross street - is filled out to the
#: street: ground a closing of this radius (ground metres) spans, no
#: farther than SNAP_NOTCH_STREET_M from a centreline.
SNAP_NOTCH_M = 45.0
SNAP_NOTCH_STREET_M = 45.0
#: Tolerance (ground metres) every outline is simplified to, as one
#: coverage: a 1 m grid's staircase gone, a lot's corner kept.
SNAP_SIMPLIFY_M = 2.0
#: A city block - ground the centrelines enclose - this much of which the
#: assignments already cover is theirs whole, each cell to the nearest: a
#: lot no run took, a church yard, the strip a rule left between two of
#: them. Its edges are then the streets round it.
SNAP_FACE_SHARE = 0.6
#: Ground metres each centreline is run on past its ends, to close the gap
#: OpenStreetMap leaves at a T junction (as JUNCTION_M).
SNAP_JUNCTION_M = 15.0
#: Ground at least this share of whose surroundings (a square SNAP_POCKET_M
#: either side) the assignments already cover is a pocket in them - a lot
#: no run took, a slit between two of them, the corner of a city block -
#: and goes to the nearest one in sight, in SNAP_POCKET_PASSES passes. A
#: straight outer edge has under half its surroundings covered and stays.
SNAP_POCKET_SHARE = 0.6
SNAP_POCKET_M = 30.0
SNAP_POCKET_PASSES = 2


def _extended(line, distance: float):
    """`line` (or each line of a MultiLineString) run on `distance` past
    both ends, straight along its end segments."""
    from shapely.geometry import LineString, MultiLineString

    def one(part):
        coords = list(part.coords)
        if len(coords) < 2:
            return part

        def past(a, b):
            dx, dy = a[0] - b[0], a[1] - b[1]
            length = math.hypot(dx, dy) or 1.0
            return (a[0] + dx / length * distance, a[1] + dy / length * distance)

        return LineString([past(coords[0], coords[1])] + coords
                          + [past(coords[-1], coords[-2])])

    if line.geom_type == "LineString":
        return one(line)
    if line.geom_type == "MultiLineString":
        return MultiLineString([one(part) for part in line.geoms])
    return line

NO_ARTERIALS_WARNING = ("the staged OpenStreetMap was fetched before road classes were "
                        "kept (query version {version}), so arterials are unknown and a "
                        "group may cross a major road; re-stage with force=True to read them")
NO_OSM_WARNING = ("osm.json is missing from the staged ground, so arterials are unknown "
                  "and a group may cross a major road")

__all__ = ["ARTERIALS", "group_blocks", "snap_outlines", "street_links", "urban_assignments",
           "urban_params", "urban_set"]


def snap_outlines(geometries: list, *, scale: float, gap_m: float = SNAP_GAP_M,
                  pixel_m: float = SNAP_PIXEL_M, streets=None) -> list:
    """`geometries` (shapely, EPSG:3857, disjoint) grown into the ground
    between them, so neighbours share one edge.

    A point outside every outline is given to the nearest one when the
    nearest two DIFFERENT outlines are within `gap_m` of it together and
    lie on either side of it (the directions to them SNAP_OPPOSITE_DEG or
    more apart): it lies across a gap no wider than that, between the two.
    Ground no outline's edge can see out of - a pocket every side of which
    is an outline - goes to the nearest too.

    `streets` (shapely lines, EPSG:3857: the OpenStreetMap centrelines) are
    the boundary wherever one runs through a gap. No gap is split across
    one - two assignments either side of a street meet on its centreline -
    and an edge that faces a street within SNAP_STREET_M is drawn out to
    the centreline, so the end of a run lies on its cross street rather
    than at the last lot. Split on a
    `pixel_m` grid, the cells taken are vectorised from one label raster
    and simplified as a coverage, so two neighbours' new edges are the same
    line. Each outline keeps all its own ground and gives up none to
    another, so what was disjoint stays disjoint and every block stays
    where it was drawn. `scale` is ground metres per map unit. Returns the
    geometries in the same order."""
    import shapely
    from affine import Affine
    from rasterio.features import rasterize
    from rasterio.features import shapes as rasterio_shapes
    from scipy import ndimage
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.geometry import shape as shapely_shape
    from shapely.ops import unary_union

    n = len(geometries)
    if n < 2:
        return list(geometries)
    west, south, east, north = unary_union(geometries).bounds
    span_m = max(east - west, north - south) * scale + 2 * gap_m
    pixel_m = max(pixel_m, span_m / SNAP_MAX_SIDE_PX)
    unit = pixel_m / scale
    pad = gap_m / scale + 2 * unit
    west, south, east, north = west - pad, south - pad, east + pad, north + pad
    width = int(math.ceil((east - west) / unit))
    height = int(math.ceil((north - south) / unit))
    transform = Affine(unit, 0.0, west, 0.0, -unit, north)
    labels = rasterize([(g, i + 1) for i, g in enumerate(geometries) if not g.is_empty],
                       out_shape=(height, width), transform=transform, fill=0,
                       dtype="int32")
    # The nearest and second-nearest outline of every cell, by distance
    # transforms over each outline's window only.
    reach_px = int(math.ceil(gap_m / pixel_m)) + 2
    shape = (height, width)
    best = np.full(shape, np.inf, dtype=np.float32)
    second = np.full(shape, np.inf, dtype=np.float32)
    nearest = np.zeros(shape, dtype=np.int32)
    # Where each cell's nearest and second-nearest outline lie, as row and
    # column offsets from the cell.
    best_dr, best_dc = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    second_dr, second_dc = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    for i in range(1, n + 1):
        rows, cols = np.nonzero(labels == i)
        if rows.size == 0:
            continue
        r0, r1 = max(rows.min() - reach_px, 0), min(rows.max() + reach_px + 1, height)
        c0, c1 = max(cols.min() - reach_px, 0), min(cols.max() + reach_px + 1, width)
        distance, (near_r, near_c) = ndimage.distance_transform_edt(
            labels[r0:r1, c0:c1] != i, return_indices=True)
        distance = distance.astype(np.float32) * pixel_m
        grid_r, grid_c = np.indices(distance.shape)
        dr, dc = (near_r - grid_r).astype(np.float32), (near_c - grid_c).astype(np.float32)
        window = (slice(r0, r1), slice(c0, c1))
        b, s, k = best[window], second[window], nearest[window]
        bdr, bdc, sdr, sdc = (best_dr[window], best_dc[window], second_dr[window],
                              second_dc[window])
        closer = distance < b
        runner_up = ~closer & (distance < s)
        # The old nearest becomes the second where this one is closer.
        s[closer], sdr[closer], sdc[closer] = b[closer], bdr[closer], bdc[closer]
        s[runner_up], sdr[runner_up], sdc[runner_up] = (
            distance[runner_up], dr[runner_up], dc[runner_up])
        b[closer], bdr[closer], bdc[closer] = distance[closer], dr[closer], dc[closer]
        k[closer] = i
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = ((best_dr * second_dr + best_dc * second_dc)
                  / (np.hypot(best_dr, best_dc) * np.hypot(second_dr, second_dc)))
    opposite = np.nan_to_num(cosine, nan=1.0) <= math.cos(math.radians(SNAP_OPPOSITE_DEG))
    street = np.zeros(shape, dtype=bool)
    if streets:
        # Each line run on past its ends by a junction's reach: OpenStreetMap
        # draws a T junction's stem ending a few metres short of the through
        # road, which would leave the city blocks either side joined.
        lines = [(_extended(line, SNAP_JUNCTION_M / scale), 1) for line in streets
                 if not line.is_empty]
        if lines:
            street = rasterize(lines, out_shape=shape, transform=transform, fill=0,
                               all_touched=True, dtype="uint8") > 0
    street &= labels == 0
    # The ground between outlines, in pieces the centrelines bound (an
    # all-touched line is watertight to 4-connected ground), and which
    # outlines and whether a street each piece touches.
    pieces, n_pieces = ndimage.label((labels == 0) & ~street)
    beside: dict[int, set[int]] = {}
    on_street = np.zeros(n_pieces + 1, dtype=bool)
    for axis in (0, 1):
        for step in (1, -1):
            other_label = np.roll(labels, step, axis=axis)
            other_street = np.roll(street, step, axis=axis)
            touching = (pieces > 0) & (other_label > 0)
            for piece, label in set(zip(pieces[touching].tolist(),
                                        other_label[touching].tolist())):
                beside.setdefault(piece, set()).add(label)
            on_street[np.unique(pieces[(pieces > 0) & other_street])] = True
    on_street[0] = False
    # Whether each cell's nearest outline is one its piece touches: an
    # outline across a centreline is not this ground's.
    # Whether the straight line from a cell to the point (dr, dc) cells
    # away crosses no centreline: ground goes only to an outline on its own
    # side of the street. A line of sight rather than which piece of ground
    # a cell is in, because a street that ends - a dead end, the edge of
    # the staged ground - leaves the pieces either side joined round its end.
    def in_sight(dr, dc, cells):
        rows, cols = np.nonzero(cells)
        clear = np.ones(rows.size, dtype=bool)
        if rows.size:
            vr, vc = dr[rows, cols], dc[rows, cols]
            # Half-cell steps: an all-touched line has no gap to slip through.
            steps = int(math.ceil(2 * max(np.abs(vr).max(), np.abs(vc).max(), 1.0)))
            for k in range(1, steps):
                t = k / steps
                sample_r = np.clip(np.rint(rows + t * vr).astype(np.int64), 0, height - 1)
                sample_c = np.clip(np.rint(cols + t * vc).astype(np.int64), 0, width - 1)
                clear &= ~street[sample_r, sample_c]
        out = np.zeros(shape, dtype=bool)
        out[rows[clear], cols[clear]] = True
        return out

    open_ground = (labels == 0) & ~street
    pairs = in_sight(best_dr, best_dc, open_ground & (best <= gap_m))

    across = best + second
    # Between two outlines: both in sight, so a gap is never bridged round
    # the mouth of the street that divides it.
    between = pairs & (across <= gap_m) & opposite
    between = in_sight(second_dr, second_dc, between)
    street_distance = np.full(shape, np.inf, dtype=np.float64)
    if street.any():
        street_distance, (street_r, street_c) = ndimage.distance_transform_edt(
            ~street, return_indices=True)
        grid_r, grid_c = np.indices(shape)
        to_street_r = (street_r - grid_r).astype(np.float32)
        to_street_c = (street_c - grid_c).astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            facing = ((best_dr * to_street_r + best_dc * to_street_c)
                      / (np.hypot(best_dr, best_dc) * np.hypot(to_street_r, to_street_c)))
        towards = (pairs
                   & (best + street_distance * pixel_m <= SNAP_STREET_M)
                   & (np.nan_to_num(facing, nan=-1.0)
                      <= math.cos(math.radians(SNAP_FACING_DEG))))
        between |= towards
        # A notch opening onto the street, filled out to it.
        notch_px = SNAP_NOTCH_M / pixel_m
        spanned = ndimage.distance_transform_edt(
            ndimage.distance_transform_edt(labels == 0) <= notch_px) > notch_px
        between |= (pairs & spanned
                    & (street_distance * pixel_m <= SNAP_NOTCH_STREET_M))
    open_px = SNAP_OPEN_M / pixel_m
    if open_px >= 1:
        eroded = ndimage.distance_transform_edt(between) > open_px
        between = ndimage.distance_transform_edt(~eroded) <= open_px
    between &= pairs
    between |= pairs & (across <= SNAP_THIN_M)
    # The corner of a junction: a small piece the centrelines cut off beside
    # one outline goes to it whole.
    if n_pieces:
        piece_m2 = np.bincount(pieces.ravel(), minlength=n_pieces + 1) * pixel_m * pixel_m
        corner = np.zeros(n_pieces + 1, dtype=bool)
        corner[1:] = piece_m2[1:] <= SNAP_CORNER_M2
        for piece in range(1, n_pieces + 1):
            if corner[piece] and not beside.get(piece):
                corner[piece] = False
        between |= (open_ground & corner[pieces]) & (pairs | (best <= gap_m))
    filled = np.where(labels > 0, labels, np.where(between, nearest, 0)).astype(np.int32)
    # Once more out to the street from the ground now drawn: a notch filled
    # above can stop a few metres short of its centreline.
    if street.any():
        drawn_distance, (drawn_r, drawn_c) = ndimage.distance_transform_edt(
            filled == 0, return_indices=True)
        grid_r, grid_c = np.indices(shape)
        to_drawn_r = (drawn_r - grid_r).astype(np.float32)
        to_drawn_c = (drawn_c - grid_c).astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            facing = ((to_drawn_r * to_street_r + to_drawn_c * to_street_c)
                      / (np.hypot(to_drawn_r, to_drawn_c) * np.hypot(to_street_r, to_street_c)))
        owner = filled[drawn_r, drawn_c]
        again = ((filled == 0) & open_ground
                 & ((drawn_distance + street_distance) * pixel_m <= SNAP_STREET_M)
                 & (np.nan_to_num(facing, nan=-1.0)
                    <= math.cos(math.radians(SNAP_FACING_DEG))))
        again = in_sight(to_drawn_r, to_drawn_c, again)
        filled[again] = owner[again]
        pairs |= again
    # A city block mostly covered already is covered whole.
    if street.any():
        faces, n_faces = ndimage.label(~street)
        size = np.bincount(faces.ravel(), minlength=n_faces + 1)
        covered = np.bincount(faces[filled > 0], minlength=n_faces + 1)
        # Not one open to the edge of the grid: that is the ground round
        # the town, however little of it the grid holds.
        border_faces = np.unique(np.concatenate([faces[0], faces[-1], faces[:, 0],
                                                 faces[:, -1]]))
        whole = np.zeros(n_faces + 1, dtype=bool)
        whole[1:] = covered[1:] >= SNAP_FACE_SHARE * size[1:]
        whole[border_faces] = False
        for face in np.flatnonzero(whole & (covered < size)):
            rows, cols = np.nonzero(faces == face)
            r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
            inside = faces[r0:r1, c0:c1] == face
            local = filled[r0:r1, c0:c1]
            drawn_here = inside & (local > 0)
            _d, (near_r, near_c) = ndimage.distance_transform_edt(~drawn_here,
                                                                  return_indices=True)
            empty = inside & (local == 0)
            local[empty] = local[near_r[empty], near_c[empty]]
            pairs[r0:r1, c0:c1] |= empty
    # Pockets, by the share of their surroundings already covered.
    window_px = int(2 * round(SNAP_POCKET_M / pixel_m) + 1)
    for _ in range(SNAP_POCKET_PASSES):
        drawn = filled > 0
        share = ndimage.uniform_filter(drawn.astype(np.float32), size=window_px,
                                       mode="constant")
        pocket = ~drawn & open_ground & (share >= SNAP_POCKET_SHARE)
        if not pocket.any():
            break
        _d, (near_r, near_c) = ndimage.distance_transform_edt(~drawn, return_indices=True)
        grid_r, grid_c = np.indices(shape)
        pocket = in_sight((near_r - grid_r).astype(np.float32),
                          (near_c - grid_c).astype(np.float32), pocket)
        filled[pocket] = filled[near_r[pocket], near_c[pocket]]
        pairs |= pocket
    # Each outline's new ground smoothed - closed, then opened - so the
    # direction tests' flicker leaves no notch or scallop on its edge. Two
    # outlines' smoothing never takes the other's ground.
    smooth_px = SNAP_OPEN_M / pixel_m
    if smooth_px >= 1:
        for i in range(1, n + 1):
            grown = filled == i
            if not grown.any():
                continue
            rows, cols = np.nonzero(grown)
            pad_px = int(math.ceil(smooth_px)) + 2
            r0, r1 = max(rows.min() - pad_px, 0), min(rows.max() + pad_px + 1, height)
            c0, c1 = max(cols.min() - pad_px, 0), min(cols.max() + pad_px + 1, width)
            local = grown[r0:r1, c0:c1]
            closed = ndimage.distance_transform_edt(
                ndimage.distance_transform_edt(~local) <= smooth_px) > smooth_px
            closed |= local
            opened = ndimage.distance_transform_edt(
                ~(ndimage.distance_transform_edt(closed) > smooth_px)) <= smooth_px
            # Thin ground along a street - out to its centreline - is kept;
            # a spike out into open ground is not.
            along = local & (street_distance[r0:r1, c0:c1] * pixel_m <= SNAP_OPEN_M + pixel_m)
            mine = (opened & closed) | along | (labels[r0:r1, c0:c1] == i)
            free = (filled[r0:r1, c0:c1] == 0) | (filled[r0:r1, c0:c1] == i)
            window = filled[r0:r1, c0:c1]
            window[local & ~mine & (labels[r0:r1, c0:c1] == 0)] = 0
            window[mine & free & pairs[r0:r1, c0:c1]] = i
    # A pocket of no one's ground with outlines all round it and no street
    # through it goes to the nearest: a lot no block took, a courtyard the
    # closing did not span.
    pockets, _count = ndimage.label(filled == 0)
    edge = np.unique(np.concatenate([pockets[0], pockets[-1], pockets[:, 0],
                                     pockets[:, -1]]))
    with_street = np.unique(pockets[street])
    enclosed = (pockets > 0) & ~np.isin(pockets, edge) & ~np.isin(pockets, with_street)
    filled[enclosed] = nearest[enclosed]
    # The centreline itself goes to an outline drawn up to it, so two that
    # meet on a street leave no line of no one's ground between them.
    if street.any():
        _d, (fill_r, fill_c) = ndimage.distance_transform_edt(filled == 0,
                                                              return_indices=True)
        line = street & (_d <= 2)
        filled[line] = filled[fill_r[line], fill_c[line]]
        # Ground near a centreline that assignments lie either side of - a
        # boulevard's median, the corner of a junction - goes with the
        # nearest. A closing of the ground drawn: never on past their ends.
        reach = SNAP_MEDIAN_M / 2 / pixel_m
        drawn = filled > 0
        spans = ndimage.distance_transform_edt(
            ndimage.distance_transform_edt(~drawn) <= reach) > reach
        median = ~drawn & spans & (street_distance * pixel_m <= SNAP_MEDIAN_M / 2)
        if median.any():
            _d, (fill_r, fill_c) = ndimage.distance_transform_edt(~drawn, return_indices=True)
            owner = filled[fill_r, fill_c]
            # Never across a centreline: a median (no outline beside it)
            # goes to either side, other ground only to its own side's.
            lonely = np.ones(n_pieces + 1, dtype=bool)
            lonely[list(beside)] = False
            lonely[0] = False
            grid_r, grid_c = np.indices(shape)
            own_side = in_sight((fill_r - grid_r).astype(np.float32),
                                (fill_c - grid_c).astype(np.float32), median & ~lonely[pieces])
            median &= lonely[pieces] | own_side
            filled[median] = owner[median]
    # One label raster vectorised: neighbours' edges are the same cell edges.
    tiles: list[list] = [[] for _ in range(n)]
    for geom, value in rasterio_shapes(filled, mask=filled > 0, transform=transform):
        tiles[int(value) - 1].append(shapely_shape(geom))
    merged = [unary_union(parts) if parts else Polygon() for parts in tiles]
    # Simplified as one coverage, so a shared edge stays one line, vertex
    # for vertex. Every outline is drawn from it alone - its own blocks
    # included - so no edge keeps the staircase of the grid it was read on.
    simplified = list(shapely.coverage_simplify(np.array(merged, dtype=object),
                                                SNAP_SIMPLIFY_M / scale))
    result = []
    for i, geometry in enumerate(geometries):
        tile = shapely.make_valid(simplified[i])
        others = unary_union([g for j, g in enumerate(simplified) if j != i])
        parts = [tile] if tile.geom_type == "Polygon" else [
            part for part in getattr(tile, "geoms", []) if part.geom_type == "Polygon"]
        # Only pieces holding some of its own ground: never a stray cell.
        keep = [part for part in parts if part.intersection(geometry).area > 0]
        # A hole with no other outline in it is a cul-de-sac's turning
        # circle or a yard the grid missed: it is this assignment's ground.
        keep = [Polygon(part.exterior, [ring for ring in part.interiors
                                        if Polygon(ring).intersects(others)])
                for part in keep]
        result.append(keep[0] if len(keep) == 1 else MultiPolygon(keep) if keep
                      else geometry)
    return result


def _arterials(payload: dict | None) -> tuple[Any, str | None]:
    """(the arterial centrelines as one shapely geometry in EPSG:3857, or
    None, and a warning when they cannot be known)."""
    from shapely.geometry import shape as shapely_shape
    from shapely.ops import unary_union

    from ..terrain.linear import OSM_QUERY_VERSION_ROUTE_TAGS

    if payload is None:
        return None, NO_OSM_WARNING
    version = int(payload.get("query_version") or 1)
    if version < OSM_QUERY_VERSION_ROUTE_TAGS:
        return None, NO_ARTERIALS_WARNING.format(version=version)
    lines = [shapely_shape(route["geometry"]) for route in payload.get("routes") or []
             if ((route.get("properties") or {}).get("highway") or "") in ARTERIALS]
    return (unary_union(lines) if lines else None), None


def _compactness(geometry) -> float:
    hull = geometry.convex_hull
    return geometry.area / hull.area if hull.area else 0.0


def street_links(shapes: list, names: list, streets: dict[str, list], *, scale: float,
                 neighbour_gap_m: float = NEIGHBOUR_GAP_M, junction_m: float = JUNCTION_M,
                 reach_m: float = JUNCTION_REACH_M) -> set:
    """Pairs `frozenset({i, j})` of neighbouring blocks joined by their
    streets: two runs of one street, or two streets meeting at a junction
    both blocks reach. `names[i]` is block i's street or None; `streets`
    maps a name to its centrelines (shapely, EPSG:3857). A block with no
    street is joined to every neighbour, as before streets were read.

    Blocks backing onto each other - the two lot rows between parallel
    streets, a fence between them - are neighbours but not linked: grouping
    them splits both streets between two teams."""
    from shapely.ops import nearest_points, unary_union

    gap = neighbour_gap_m / scale
    lines = {name: unary_union(parts) for name, parts in streets.items() if parts}
    links = set()
    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            if shapes[i].distance(shapes[j]) > gap:
                continue
            a, b = names[i], names[j]
            if a is None or b is None or a == b:
                links.add(frozenset((i, j)))
                continue
            if a not in lines or b not in lines:
                continue
            if lines[a].distance(lines[b]) * scale > junction_m:
                continue
            junction = nearest_points(lines[a], lines[b])[0]
            if (shapes[i].distance(junction) * scale <= reach_m
                    and shapes[j].distance(junction) * scale <= reach_m):
                links.add(frozenset((i, j)))
    return links


def group_blocks(shapes: list, *, target_ha: float, scale: float, arterials=None,
                 street_links=None,
                 neighbour_gap_m: float = NEIGHBOUR_GAP_M, close_m: float = CLOSE_M,
                 water_gap_m: float = WATER_GAP_M, link_half_m: float = LINK_HALF_M,
                 max_share: float = MAX_SHARE, settled_share: float = SETTLED_SHARE,
                 fence_share: float = FENCE_SHARE) -> list[dict]:
    """Groups of the block polygons `shapes` (shapely, EPSG:3857, disjoint),
    as `[{"members": [index, ...], "geometry": Polygon | MultiPolygon,
    "water_links": n, "hemmed": [index, ...]}]`, largest first; `hemmed`
    names the members that joined as hemmed, empty when none did. `scale`
    is ground metres per map unit; `arterials` a shapely geometry no group
    crosses, or None. `street_links` is the set of `frozenset({i, j})`
    pairs of blocks whose streets join (see `street_links`): pass 1 grows
    groups along those only, and blocks merely backing onto each other
    join only when a group is left under `fence_share` of the target. None
    treats every neighbour as street-joined (blocks with no street names). Every block is a member of one group and lies wholly
    inside that group's outline and no other, so counts taken over the
    members are the ground drawn. See the module docstring for the rules."""
    from shapely import STRtree, shortest_line
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.ops import unary_union

    if target_ha <= 0:
        raise ValueError(f"target_ha ({target_ha}) must be above zero")
    n = len(shapes)
    if n == 0:
        return []

    def hectares(geometry) -> float:
        return geometry.area * scale * scale / 10_000

    def crosses_arterial(line) -> bool:
        return arterials is not None and line.intersects(arterials)

    gap = neighbour_gap_m / scale
    neighbours: dict[int, set[int]] = {i: set() for i in range(n)}
    by_street: dict[int, set[int]] = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if shapes[i].distance(shapes[j]) > gap:
                continue
            if crosses_arterial(shortest_line(shapes[i], shapes[j])):
                continue
            neighbours[i].add(j)
            neighbours[j].add(i)
            if street_links is None or frozenset((i, j)) in street_links:
                by_street[i].add(j)
                by_street[j].add(i)

    groups = {i: {i} for i in range(n)}
    owner = list(range(n))
    area = {i: hectares(shapes[i]) for i in range(n)}
    links: dict[int, list] = {i: [] for i in range(n)}
    settled, ceiling = target_ha * settled_share, target_ha * max_share
    close = close_m / scale
    half = link_half_m / scale

    def outline(members, strips=()):
        union = unary_union([shapes[m] for m in members] + [s.buffer(half) for s in strips])
        closed = union.buffer(close, join_style="mitre").buffer(-close, join_style="mitre")
        return closed.union(union)

    def absorb(into: int, other: int, strip=None):
        groups[into] |= groups.pop(other)
        area[into] += area.pop(other)
        links[into] += links.pop(other) + ([strip] if strip is not None else [])
        for member in groups[into]:
            owner[member] = into

    tree = STRtree(shapes)
    inside_points = [shape.representative_point() for shape in shapes]

    def one_piece(members) -> bool:
        """Whether a group of `members` is drawn as one polygon: its outline,
        less every other block it reaches, holds all its blocks in one
        piece. Two streets that meet behind a third street's houses do not
        make one team's ground."""
        geometry = outline(members)
        others = [shapes[m] for m in tree.query(geometry, predicate="intersects")
                  if m not in members]
        if others:
            geometry = geometry.difference(unary_union(others))
        pieces = [geometry] if geometry.geom_type == "Polygon" else list(
            getattr(geometry, "geoms", []))
        return any(all(piece.covers(inside_points[m]) for m in members) for piece in pieces)

    def grow(links: dict[int, set[int]], below: float) -> None:
        """Smallest group first, while under `below`: join the partner along
        `links` that keeps the merged shape most compact, under the
        ceiling."""
        while True:
            merged = False
            for g in sorted((g for g in groups if area[g] < below),
                            key=lambda g: (area[g], g)):
                partners = {owner[j] for m in groups[g] for j in links[m]} - {g}
                options = [(_compactness(outline(groups[g] | groups[p])),
                            -(area[g] + area[p]), -p, p)
                           for p in partners if area[g] + area[p] <= ceiling
                           and one_piece(groups[g] | groups[p])]
                if options:
                    absorb(max(options)[3], g)
                    merged = True
                    break
            if not merged:
                break

    # Pass 1: along the streets - a cul-de-sac into the road it opens onto,
    # a run into the next run of its street - so each street stays one
    # team's, both sides. Pass 1b: a group still tiny with no street left
    # to grow along joins a block it backs onto.
    grow(by_street, settled)
    grow(neighbours, target_ha * fence_share)

    # Pass 2: small groups pair across water, one pair each.
    water = water_gap_m / scale
    while True:
        best = None
        for g in sorted((g for g in groups if area[g] < settled and not links[g]),
                        key=lambda g: (area[g], g)):
            shape_g = unary_union([shapes[m] for m in groups[g]])
            for p in sorted(groups):
                if (p == g or links[p] or area[p] >= settled
                        or area[g] + area[p] > ceiling):
                    continue
                shape_p = unary_union([shapes[m] for m in groups[p]])
                distance = shape_g.distance(shape_p)
                # Within a street of each other is not across water: two
                # neighbours that pass 1 kept apart (a fence, a third
                # street's houses between them) stay apart.
                if distance > water or distance <= gap:
                    continue
                strip = shortest_line(shape_g, shape_p)
                if crosses_arterial(strip):
                    continue
                # Across water means across nothing searched: a strip over
                # another group's block is a street of houses, not a lake.
                if any(strip.crosses(shapes[m]) or strip.within(shapes[m])
                       for m in tree.query(strip, predicate="intersects")
                       if owner[m] not in (g, p)):
                    continue
                if best is None or distance < best[0]:
                    best = (distance, g, p, strip)
            if best is not None:
                break
        if best is None:
            break
        _distance, g, p, strip = best
        absorb(p, g, strip)

    # A hole is kept only when another group's block is INSIDE it (its
    # inside point), not when a neighbour merely brushes its edge.
    def filled(geometry, g):
        others = [inside_points[m] for m in range(n) if owner[m] != g]
        parts = [geometry] if geometry.geom_type == "Polygon" else [
            part for part in getattr(geometry, "geoms", []) if part.geom_type == "Polygon"]
        return [Polygon(part.exterior,
                        [ring for ring in part.interiors
                         if any(Polygon(ring).contains(point) for point in others)])
                for part in parts]

    def reach(g):
        """The ground g's outline reaches, holes filled unless another
        group's block is inside."""
        return unary_union(filled(outline(groups[g], links[g]), g))

    def ground(g):
        return unary_union([shapes[m] for m in groups[g]])

    def pieces(geometry):
        return [geometry] if geometry.geom_type == "Polygon" else list(
            getattr(geometry, "geoms", []))

    def apart(geometry, members, others) -> bool:
        """Whether a piece of `geometry` holds one of `members` and none of
        `others`."""
        return any(not any(piece.contains(inside_points[m]) for m in others)
                   for piece in pieces(geometry)
                   if any(piece.contains(inside_points[m]) for m in members))

    # Hemmed groups, largest first: a group more than half of whose block
    # ground the larger groups' outlines already reach, together, stands in
    # their courtyards or slots, closer to their blocks than the closing
    # spans - as a small block does inside a neighbourhood whose lots were
    # grouped by adjacency alone (no OpenStreetMap). It joins the one whose
    # outline reaches most of it (on a tie the larger, then the one met
    # first), to be counted and drawn there. Decided by share, not by
    # whether anything is left over: a block 2 m proud of a slot's mouth,
    # drawn on its own, was the 2 m strip outside - a 0.0 ha assignment
    # carrying every house, while the houses were drawn in the larger
    # group's outline and left out of its counts - and one flush with the
    # mouth joined. Centimetres must not flip that. A hemmed join is not
    # held to the ceiling: `hemmed` names who joined, so the assignment it
    # carries past can say so.
    order = sorted(groups, key=lambda g: (-area[g], min(groups[g])))
    reached: dict[int, Any] = {}
    hemmed: dict[int, list[int]] = {}
    drawn: list[int] = []
    for g in order:
        own = ground(g)
        # Each larger outline's reach of it, read over its own ground: one
        # union of every outline, regrown after each join, would cost the
        # square of the joins in a block with a slot for every house.
        min_x, min_y, max_x, max_y = own.bounds
        hits = []
        for h in drawn:
            lo_x, lo_y, hi_x, hi_y = reached[h].bounds
            if lo_x > max_x or hi_x < min_x or lo_y > max_y or hi_y < min_y:
                continue
            part = reached[h].intersection(own)
            if not part.is_empty:
                hits.append((h, part))
        if not hits or unary_union([part for _h, part in hits]).area <= own.area / 2:
            reached[g] = reach(g)
            drawn.append(g)
            continue
        host = max(hits, key=lambda hit: (hit[1].area, area[hit[0]]))[0]
        members, earlier = sorted(groups[g]), sorted(groups[host])
        # A water pair's strip is drawn only to reach a partner the host's
        # outline would otherwise leave cut off - an isle across the water.
        # Two blocks both in the host's courtyards or slots are joined by
        # its own ground and take no strip: none is drawn and no water is
        # warned of.
        strips, links[g] = links[g], []
        hemmed.setdefault(host, []).extend(members)
        absorb(host, g)
        if reached[host].covers(own):
            # Already drawn round it: closing over blocks inside the outline
            # adds nothing to it, and a partner inside it needs no strip.
            continue
        # Its blocks are the host's now: the host's outline closes over
        # them, the part proud of the mouth too.
        reached[host] = reach(host)
        if strips and apart(reached[host], members, earlier):
            links[host] += strips
            reached[host] = reach(host)

    # Outlines, largest first. A block is drawn only in the group that
    # counts it - an outline gives up every other group's block, so a team's
    # polygon holds the houses its counts hold and no others, and a group
    # half reached or less keeps its blocks whole, the larger outline going
    # round them. Street ground two outlines both reach goes to the larger
    # group.
    claimed = None
    result = []
    for g in drawn:
        geometry = reached[g]
        foreign = [shapes[m] for m in tree.query(geometry, predicate="intersects")
                   if owner[m] != g]
        if foreign:
            geometry = geometry.difference(unary_union(foreign))
        if claimed is not None:
            geometry = geometry.difference(claimed)
        own = ground(g)
        # Every piece holding a block stays; a sliver holding none goes. A
        # piece round a block's inside point holds it however small the
        # block: a block under a square metre must not leave its group with
        # no outline at all.
        kept = [part for part in filled(geometry, g)
                if part.intersection(own).area > 1.0 / (scale * scale)
                or any(part.contains(inside_points[m]) for m in groups[g])]
        geometry = kept[0] if len(kept) == 1 else MultiPolygon(kept)
        claimed = geometry if claimed is None else claimed.union(geometry)
        result.append({"members": sorted(groups[g]), "geometry": geometry,
                       "water_links": len(links[g]), "hemmed": sorted(hemmed.get(g, []))})
    return result


def urban_params(*, target_ha: float | None, resource_type: str, max_distance_m: float | None,
                 building_markers: bool = False, marker_color: str = "#FFFFFF",
                 norms: dict | None = None) -> dict:
    """The parameters an urban set id is hashed over.

    With no `target_ha` the target is the norms' median for the resource
    type, so the norms decide the grouping, and their digest
    (`norms.norms_digest`; `norms` None is the shipped file) is hashed in.
    Without it a server restarted with another agency's `--norms` over the
    same cache read back assignments grouped toward the old corpus's median
    as current, its `target_source` still calling that "corpus median for
    GROUND" while `assignment_norms` on the same server gave the agency's
    (PR #4 review B23). A target given reads no norms and hashes none, so
    such a set keeps its id; one left to the norms changed id once, in
    0.31.11, and is grouped again on its first call.
    """
    from .norms import load_norms, norms_digest
    from .populated import marker_colour

    from .populated import METHOD

    # The grouping method and the blocks' together: a set grouped back to
    # back, or over blocks drawn the older way, is grouped again (0.35.0).
    params = {"method": METHOD,
              "target_ha": None if target_ha is None else float(target_ha),
              "resource_type": (resource_type or "GROUND").strip().upper(),
              "max_distance_m": None if max_distance_m is None else float(max_distance_m)}
    if target_ha is None:
        params["norms_digest"] = norms_digest(norms if norms is not None else load_norms())
    if building_markers:
        params["building_markers"] = True
        params["marker_color"] = marker_colour(marker_color)
    return params


def urban_assignments(directory, *, target_ha: float | None = None,
                      resource_type: str = "GROUND", norms: dict | None = None,
                      max_distance_m: float | None = None,
                      building_markers: bool = False, marker_color: str = "#FFFFFF",
                      wms_base_url: str | None = None, cache_dir: Path | None = None,
                      fetch=None) -> dict[str, Any]:
    """The populated blocks of the ground staged under `directory`, grouped
    into assignments of about `target_ha` (default: the corpus median for
    `resource_type`, or of every assignment - with a warning - for a type
    the corpus has no median for), nearest the centre first.

    Returns `{"assignments": [...], "summary", "assumptions", "center",
    "warnings"}`; each assignment carries `name`, `streets`, `blocks`,
    `facts`, `geometry` (WGS84) and `warnings`, and `markers` when asked
    for. Raises what `populated_areas` raises, and ValueError on a target
    that is not above zero.
    """
    from pyproj import Geod, Transformer
    from shapely.geometry import shape as shapely_shape
    from shapely.ops import polylabel
    from shapely.ops import transform as shapely_transform

    from .norms import load_norms, target_area_range_ha
    from .populated import populated_areas

    norms = norms if norms is not None else load_norms()
    resource = (resource_type or "GROUND").strip().upper()
    target_source = "given"
    sizing_warning = None
    if target_ha is None:
        target = target_area_range_ha(resource, norms)
        target_ha = float(target["p50"])
        if target["source"] == "polygon_area_ha":
            # The corpus has no median for this type - a typo such as DOGS
            # for DOG, or a real type nobody in the corpus used - so the
            # target is every assignment's median. Said as that, never as a
            # median for the type, and warned of: DOGS for DOG sizes the
            # assignments 70 % larger, "dog trail" for DOG_TRAIL 13 times.
            known = ", ".join((norms.get("area_ha_by_resource") or {}).get("medians") or {})
            target_source = f"corpus median of every assignment (none for {resource})"
            sizing_warning = (f"the assignment corpus has no median for resource type "
                              f"{resource!r}, so the assignments are sized toward the median "
                              f"of every assignment, {target_ha:g} ha; it has one for "
                              f"{known or 'no type'} - check the type is the one meant")
        else:
            target_source = f"corpus median for {resource}"
    if target_ha <= 0:
        raise ValueError(f"target_ha ({target_ha}) must be above zero")

    populated = populated_areas(directory, max_distance_m=max_distance_m,
                                building_markers=building_markers, marker_color=marker_color,
                                wms_base_url=wms_base_url, cache_dir=cache_dir, fetch=fetch)
    blocks = populated["blocks"]
    centre = populated["center"]
    warnings = list(populated["warnings"])
    if sizing_warning:
        warnings.append(sizing_warning)
    osm_path = Path(directory) / "osm.json"
    payload = json.loads(osm_path.read_text(encoding="utf-8")) if osm_path.is_file() else None
    arterials, arterial_warning = _arterials(payload)
    if arterial_warning:
        warnings.append(arterial_warning)

    scale = ground_meters_per_mercator_unit(centre["latitude"])
    to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
    to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform
    shapes = [shapely_transform(to_3857, shapely_shape(block["geometry"])) for block in blocks]
    from .populated import _streets

    links = street_links(shapes, [block["facts"].get("street") for block in blocks],
                         _streets(payload), scale=scale)
    groups = group_blocks(shapes, target_ha=target_ha, scale=scale, arterials=arterials,
                          street_links=links)
    street_lines = [line for lines in _streets(payload).values() for line in lines]
    for group, snapped in zip(groups, snap_outlines([g["geometry"] for g in groups],
                                                    scale=scale, streets=street_lines)):
        group["geometry"] = snapped

    geod = Geod(ellps="WGS84")
    entries = []
    for group in groups:
        members = sorted(group["members"], key=lambda m: -shapes[m].area)
        streets = []
        for m in members:
            street = blocks[m]["facts"].get("street") or "unnamed street"
            if street not in streets:
                streets.append(street)
        geometry = group["geometry"]
        main = geometry if geometry.geom_type == "Polygon" else max(geometry.geoms,
                                                                    key=lambda p: p.area)
        inside = polylabel(main, tolerance=1.0 / scale)
        longitude, latitude = to_wgs84(inside.x, inside.y)
        bearing, _back, distance_m = geod.inv(centre["longitude"], centre["latitude"],
                                              longitude, latitude)
        parts = 1 if geometry.geom_type == "Polygon" else len(geometry.geoms)
        facts = {
            "streets": streets,
            "blocks": len(members),
            "parcels": int(sum(blocks[m]["facts"]["parcels"] for m in members)),
            "buildings": int(sum(blocks[m]["facts"]["buildings"] for m in members)),
            "area_ha": round(geometry.area * scale * scale / 10_000, 2),
            "point": {"latitude": latitude, "longitude": longitude},
            "distance_from_center_m": round(distance_m, 0),
            "bearing_from_center_deg": round(bearing % 360.0, 0),
            "water_links": group["water_links"],
            "parts": parts,
        }
        group_warnings = []
        if group["water_links"]:
            group_warnings.append("parts on either side of water are joined by a thin strip "
                                  "for the drawing only; travel between them by road")
        if parts > 1:
            group_warnings.append(f"the outline is in {parts} pieces (another group's ground "
                                  "runs between them); the CalTopo export draws each piece "
                                  "as its own assignment")
        # A hemmed join is not held to the ceiling every merge keeps to:
        # the blocks standing in this group's courtyards or slots are
        # counted where they are drawn. When that is what carries it past,
        # say so - the descriptions promise no merge grows a group past it.
        hemmed = set(group["hemmed"])
        ceiling_ha = target_ha * MAX_SHARE
        block_ha = sum(shapes[m].area for m in members) * scale * scale / 10_000
        own_ha = sum(shapes[m].area for m in members if m not in hemmed) * scale * scale / 10_000
        if hemmed and own_ha <= ceiling_ha < block_ha:
            names = [blocks[m]["name"] for m in members if m in hemmed]
            listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
            group_warnings.append(
                f"{listed}, most of whose ground lies inside this assignment's outline (in a "
                f"courtyard or slot of its blocks), {'is' if len(names) == 1 else 'are'} "
                f"counted and drawn here; that takes its blocks to {block_ha:.2f} ha, past "
                f"the {ceiling_ha:.2f} ha ({MAX_SHARE * 100:g} % of the target) a merge "
                "stops at")
        entry = {"name": streets[0] + (f" +{len(streets) - 1}" if len(streets) > 1 else ""),
                 "blocks": [blocks[m]["name"] for m in members], "facts": facts,
                 "geometry": to_wgs84_geojson(geometry), "warnings": group_warnings,
                 "_distance": distance_m}
        if building_markers:
            entry["markers"] = [marker for m in members for marker in blocks[m].get("markers", [])]
            facts["markers"] = len(entry["markers"])
        entries.append(entry)
    entries.sort(key=lambda entry: (entry.pop("_distance"), entry["name"]))

    sizes = [entry["facts"]["area_ha"] for entry in entries]
    summary = {
        "count": len(entries), "blocks": len(blocks),
        "target_ha": round(target_ha, 2),
        "area_ha": {"min": min(sizes), "median": sorted(sizes)[len(sizes) // 2],
                    "max": max(sizes), "total": round(sum(sizes), 1)} if sizes else None,
        "parcels": int(sum(entry["facts"]["parcels"] for entry in entries)),
        "buildings": int(sum(entry["facts"]["buildings"] for entry in entries)),
        "parcels_left_out": populated["summary"]["parcels_left_out"],
    }
    if building_markers:
        summary["markers"] = int(sum(len(entry["markers"]) for entry in entries))
    assumptions = {
        **populated["assumptions"],
        "grouping": {
            "target_ha": round(target_ha, 2), "target_source": target_source,
            "max_ha": round(target_ha * MAX_SHARE, 2),
            "settled_ha": round(target_ha * SETTLED_SHARE, 2),
            "neighbour_gap_m": NEIGHBOUR_GAP_M, "close_m": CLOSE_M,
            "water_gap_m": WATER_GAP_M, "snap_gap_m": SNAP_GAP_M,
            "arterials": sorted(ARTERIALS) if arterials is not None or not arterial_warning
            else "unknown",
            "junction_m": JUNCTION_M, "junction_reach_m": JUNCTION_REACH_M,
            "fence_ha": round(target_ha * FENCE_SHARE, 2),
            "rule": ("each block is one street, both sides, back fence to back fence; blocks "
                     "merge smallest first along the streets that join them (a cul-de-sac "
                     "into the road it opens onto) into the most compact shape, never across "
                     "an arterial; a group still under fence_ha with no street to grow along "
                     "joins a block it backs onto; small groups then pair across water, two "
                     "at most; whole blocks only; the ground between two assignments up to "
                     "snap_gap_m apart is split down its middle, so neighbours share one "
                     "edge"),
        },
    }
    return {"assignments": entries, "summary": summary, "assumptions": assumptions,
            "center": centre, "warnings": warnings}


def urban_set(directory, params: dict, *, handle: str, norms: dict | None,
              wms_base_url: str | None, cache_dir: Path | None, fetch=None):
    """`urban_assignments` over `params` (from `urban_params`) as a
    `CandidateSet`: one `assignment` candidate per group, ids
    `urban-xxxxxx.N`, the polygon and any markers kept for `get_candidate`.

    `norms` must be the norms `params` were made under: the id is hashed
    over their digest, and a set grouped under other norms than its id
    names is refused rather than saved where the next call reads it back."""
    from datetime import datetime, timezone

    from .candidates import Candidate, CandidateSet, candidate_id, make_set_id
    from .norms import load_norms, norms_digest

    arguments = dict(params)
    arguments.pop("method", None)
    digest = arguments.pop("norms_digest", None)
    if digest is not None:
        used = norms_digest(norms if norms is not None else load_norms())
        if used != digest:
            raise ValueError(f"urban params were made under norms {digest}, and the "
                             f"assignments would be grouped under norms {used}")
    result = urban_assignments(directory, norms=norms, wms_base_url=wms_base_url,
                               cache_dir=cache_dir, fetch=fetch, **arguments)
    set_id = make_set_id("urban", params)
    records = [Candidate(id=candidate_id(set_id, index), kind="assignment", axis="urban",
                         axis_ids=[], sources=["parcel", "structures", "osm"],
                         name=entry["name"], geometry=entry["geometry"],
                         facts={**entry["facts"], "block_names": entry["blocks"]},
                         norm_flags=[], warnings=entry["warnings"],
                         markers=entry.get("markers"))
               for index, entry in enumerate(result["assignments"], start=1)]
    return CandidateSet(
        set_id=set_id, tool="urban_assignments", params=params, handle=handle,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        candidates=records, summary={**result["summary"], "center": result["center"]},
        assumptions=result["assumptions"], warnings=result["warnings"])
