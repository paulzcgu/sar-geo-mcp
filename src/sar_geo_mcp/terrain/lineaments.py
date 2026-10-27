"""Candidate linear features cut into a hillside, and the OSM subtraction.

WHAT THIS LOOKS FOR
-------------------
A cut road is a BENCH: a run of ground locally flatter than the hillside
around it, cut bank above, fill below. `bench_response` is that comparison -
slope smoothed over a neighbourhood, minus slope here - so a bench reads
positive and an undisturbed planar hillside reads zero however steep it is.

WHY ON THE DEM AND NOT ON THE RELIEF IMAGE
------------------------------------------
The float elevation is in hand. Running edge detection on an 8-bit rendered
hillshade would measure a picture of the data instead of the data.

WHAT THIS IS NOT
----------------
It is not a road detector. A fault scarp, a pipeline cut, a cattle terrace,
an old railbed and a logging grade are INDISTINGUISHABLE by this method, and
the false-positive rate on real ground has not been measured. Everything
returned is unverified evidence for a human or an agent to judge.

The tracing handles gentle curves - a contouring road - by binning component
pixels along the component's major axis and joining bin centroids. A
component that traces too winding that way - a hairpin, or a road with
spurs off it, which is one connected component - is split into its
centreline branches at every junction and each branch is judged alone
(`centreline_branches`). A coil is still dropped, by MAX_BRANCH_SINUOSITY.

WHAT WAS MEASURED, ON SYNTHETIC GROUND ONLY
-------------------------------------------
Every number below comes from a plane with a level 4 m bench cut across it,
1 m pixels. No real DEM has been through this module.

The central claim holds exactly: an undisturbed plane reads zero whatever
its steepness. Measured peak response on a bare plane was 4.8e-06 deg at
1.80 deg of slope and 5.4e-05 deg at 68.32 deg - five orders of magnitude
under MIN_RELIEF_DEG - so the one threshold really is steepness-invariant,
which was the reason for comparing neighbourhood slope against local slope
instead of thresholding slope itself.

Two limits fell out of the same measurements and are not fixed here,
because they are properties of the method rather than defects:

1. On a level tread the response equals the surrounding hillside's own
   slope, so MIN_RELIEF_DEG doubles as a floor on hillside steepness. A 4 m
   bench read 4.3 deg on a 4.32 deg plane (nothing found) and 5.7 deg on a
   5.75 deg plane (found). Benches on ground gentler than MIN_RELIEF_DEG are
   invisible to this method, by construction.

2. Roughness cuts both ways, and this is the main open risk.

   Uncorrelated noise FRAGMENTS a bench into pieces that die on the length
   floor: the same 160 px bench was found under 0.25 m of white Gaussian
   elevation noise (traced 123.6 m) and lost entirely under 0.50 m. It also
   floods the mask - 0.25 m of white noise on a bare hillside put 25% of
   pixels over MIN_RELIEF_DEG in 3266 components - though none survived the
   length, sinuosity and width filters, so nothing was reported.

   Spatially CORRELATED roughness, which is what real microtopography
   actually looks like, INVENTS benches instead. On a 51 deg plane with
   Gaussian roughness of 5 px correlation length, a single 400x400 m patch
   reported 0 lineaments at 0.5 m amplitude, 5 at 1.5 m and 10 at 3.0 m -
   around 60 per km2 on ground with nothing cut into it at all, at lengths
   of 54-98 m and widths of 8.6-13.6 m, indistinguishable in the output
   from a real bench.

   So the relief threshold is not what does the discriminating; the length,
   sinuosity and width filters are. The false-positive rate on real ground
   remains unmeasured, and a caller must treat a returned lineament as a
   place to look, never as a thing that is there.

A NOTE ON NODATA (RESOLVED)
----------------------------
`bench_response` used to fill nodata with 0 (`np.nan_to_num`) before
smoothing. That forces LOCAL slope to read flat at every nodata pixel while
the neighbourhood term is still smoothed up from the real terrain around the
hole - exactly the bench signature. Measured: a 160x4 px NaN strip cut into
an undisturbed hillside, shaped identically to `hillside_with_bench`'s real
bench, traced to a 159.0 m lineament of sinuosity 1.0 under the old fill -
not distinguishable in the output from the real thing, and a second such
strip elsewhere on a hillside that DOES have a genuine bench turned 1 found
lineament into 2.

The fix (see `bench_response`) smooths the finite-data mask with the same
Gaussian to learn what fraction of each window's mass was real ground, uses
that to divide the fill back out of the smoothed estimate, and reports NaN
- never 0 - wherever a pixel's own elevation or slope is not finite, or
wherever its window was less than half real ground (`MIN_VALID_WEIGHT`).
`find_lineaments`'s mask is `response >= min_relief_deg`, and `NaN >= x` is
`False`, so NaN pixels drop out with no special-casing there. Re-measured
after the fix: the same 160x4 px strip alone finds nothing, and the strip
placed elsewhere on a hillside with a genuine bench still finds exactly the
one real bench.

A NOTE ON COST (RESOLVED)
-------------------------
The per-label `labels == label` scan used to be quadratic in raster area
when the mask fragments, because it rescanned the WHOLE label array once
per component. Measured on white-noise hillsides at 1 m pixels: 22.5k px /
2823 components / 0.16 s, 62.5k / 7745 / 1.14 s, 160k / 20231 / 7.18 s,
360k / 46350 / 36.3 s. Extrapolated, a 2000x2000 staged DEM in that state
would have taken over an hour.

`find_lineaments` now gets each component's bounding-box slice from
`scipy.ndimage.find_objects` in one pass, and reads that component's pixels
out of its own (small) box instead of the full array - masking to the
specific label inside the box, since boxes of different components can
overlap. Re-measured on the same fragmented masks: 160k px / 20873
components, 7.185 s -> 0.067 s (107x); 360k px / 47022 components,
36.208 s -> 0.151 s (240x). Both versions agreed on which components
survive the 8-pixel floor (470 and 1097 respectively), so this changes
performance only, not results. Correlated roughness fragments far less
(317 components on 160k px, 0.17 s even under the old scan), so the
measurements above were a worst case rather than a forecast - but the
orchestration layer in Task 6 should still not hand this function a whole
tile without bounding the component count first.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from ..geo import ground_meters_per_mercator_unit
from .slope import compute_slope_aspect

#: How much flatter than its surroundings ground must read to be a bench.
#: On a level tread the response equals the hillside's own slope, so this is
#: also a floor on hillside steepness: measured, a 4 m bench read 4.3 deg on
#: a 4.32 deg plane and was not found, and 5.7 deg on a 5.75 deg plane and
#: was. Benches on ground gentler than this are invisible by construction.
MIN_RELIEF_DEG = 5.0

#: Shortest run worth offering. Below this a bench is a boulder ledge.
#: Applied to the TRACED length. Before `_trace` extended its endpoints to
#: the component's true extremes, this was only an approximate floor: bin
#: centroids sat inside the ends, so a 60 px bench traced to 48.0 m and a
#: 65 px bench to 54.0 m, pushing the real floor on the ground to nearer
#: 62 m than the 50 m written here. After that fix, the same extents trace
#: to 59.0 m and 64.0 m (a 500-row synthetic scar, extent-1 px of true
#: centre-to-centre span) - this is now the real floor, not an approximate
#: one.
MIN_LENGTH_M = 50.0

#: Fraction of a smoothing window's Gaussian mass that must land on finite
#: elevation before its smoothed neighbourhood slope is trusted. Nodata is
#: filled with 0 before smoothing (`gaussian_filter` cannot propagate NaN
#: usefully) and the fill is divided back out by this same weight, but a
#: window that is majority fill is a noisy correction rather than a
#: measurement. 0.5 is the halfway point: less than half real ground and
#: the pixel is reported as NaN, not a value - see `bench_response`.
MIN_VALID_WEIGHT = 0.5

#: Path length over end-to-end distance. A road wanders; a drainage wanders
#: more, and a component that doubles back is traced badly by the binning
#: below, so it is dropped rather than reported wrongly.
MAX_SINUOSITY = 1.6

#: The same test for one centreline branch (see `centreline_branches`),
#: looser because a walked centreline follows the ground instead of
#: squashing it onto an axis, so a switchbacking road can honestly read
#: above 1.6. Measured on real ground, 2 m pixels, 2026-10-21: branch
#: sinuosity had a median of 1.18 (Mendocino, 39 branches) and 1.20 (Tahoe
#: NF, 31); a cap of 3.0 kept 37 and 30 of them (7.7 of 8.4 km, 15.4 of
#: 15.6 km). What it drops is a coil - the worst was 577 m long at 12.73 -
#: not a road.
MAX_BRANCH_SINUOSITY = 3.0

#: The largest bend at a junction that still counts as carrying straight on.
#: Two arms meeting within this of a straight line are joined into one
#: centreline; a spur leaving at a wider angle stays its own branch. 45 deg
#: lets a road bend through a junction and keeps a hairpin - whose legs meet
#: nearly parallel, a 150-180 deg turn - apart.
MAX_JUNCTION_TURN_DEG = 45.0

#: How far back along an arm its heading into a junction is measured: far
#: enough to read the road's direction, not the last pixel's wobble.
JUNCTION_HEADING_M = 10.0

#: Component area over path length, as a multiple of the expected tread.
MIN_WIDTH_FACTOR = 0.5
MAX_WIDTH_FACTOR = 4.0

#: How far from a mapped way still counts as that way. OSM positional
#: accuracy, not a tuning knob.
DEFAULT_TOLERANCE_M = 15.0

#: Fraction of a lineament that must lie outside the buffer to be offered
#: WHOLE. Below it, the lineament is mostly a mapped way, and only its
#: unmapped stretches of at least MIN_LENGTH_M are offered, each on its own
#: (see `subtract_known_ways`).
KEEP_IF_OUTSIDE = 0.5

#: Bins along the major axis when tracing a component to a polyline. Interior
#: bin centroids preserve curvature; the first and last are pushed out to
#: the component's true extreme projected positions in `_trace` so
#: `length_m` reaches actual extent rather than stopping one bin short
#: (measured before that correction: 160 px traced 149.0 m, 400 px traced
#: 389.0 m; after: 159.0 m and 399.0 m - see `_trace`'s docstring).
TRACE_BIN_M = 10.0

LINEAMENT_ASSUMPTIONS = {
    "method": ("neighbourhood slope minus local slope; connected components "
               "traced by binning along the major axis"),
    "min_relief_deg": MIN_RELIEF_DEG,
    "min_length_m": MIN_LENGTH_M,
    "max_sinuosity": MAX_SINUOSITY,
    "min_width_factor": MIN_WIDTH_FACTOR,
    "max_width_factor": MAX_WIDTH_FACTOR,
    "known_way_tolerance_m": DEFAULT_TOLERANCE_M,
    "min_valid_weight": MIN_VALID_WEIGHT,
    "status": "unverified",
    "note": ("a fault scarp, pipeline cut, cattle terrace, old railbed and "
             "logging grade are indistinguishable by this method; nothing "
             "here is called a road, and the false-positive rate on real "
             "ground has not been measured"),
    "limit": ("a component too winding to trace along one axis (a hairpin, a "
              "road with spurs) is split into centreline branches at its "
              "junctions; traced_by says which way each lineament was traced, "
              "and a branch winding more than MAX_BRANCH_SINUOSITY (a coil) is "
              "dropped"),
    "max_branch_sinuosity": MAX_BRANCH_SINUOSITY,
    "nodata": ("filled with 0 before smoothing, then divided back out by the "
               "smoothed fraction of real ground in each window; a window "
               "under 50% real ground (MIN_VALID_WEIGHT), or a pixel where "
               "the elevation or slope itself is not finite, reads NaN, "
               "never 0 - a 160x4 px nodata strip shaped exactly like a "
               "real bench used to trace as one (159.0 m, sinuosity 1.0) "
               "under a plain nan_to_num fill and now finds nothing"),
    "measured": ("synthetic ground only, 1 m pixels: a bare plane reads "
                 "under 6e-05 deg from 1.8 to 68.3 deg of slope, so the "
                 "relief threshold is steepness-invariant; length_m traces "
                 "to within 1 m of true component extent (measured on "
                 "50/60/65/160/400 px extents); a bench survived 0.25 m of "
                 "white elevation noise and was lost under 0.50 m"),
    "false_positives": ("undisturbed ground with correlated roughness of 5 px "
                        "correlation length reported 5 lineaments per 0.16 "
                        "km2 at 1.5 m amplitude and 10 at 3.0 m, 54-98 m "
                        "long and 8.6-13.6 m wide - not separable in this "
                        "output from a real bench; a returned lineament is a "
                        "place to look, not a thing that is there"),
}


def bench_response(elevation: np.ndarray, pixel_size_m: float,
                   track_width_m: float) -> np.ndarray:
    """Degrees by which ground is flatter than its neighbourhood.

    Positive on a bench, near zero on any undisturbed planar hillside however
    steep - which is what makes one threshold work on flat and steep ground
    alike.

    NODATA
    ------
    `gaussian_filter` cannot propagate NaN usefully - one NaN in the kernel
    poisons the whole smoothed neighbourhood - so nodata has to be filled
    with something before smoothing. Filling it with 0 (as this used to do,
    via `nan_to_num`) is wrong: it forces LOCAL slope to read flat at every
    nodata pixel while `gaussian_filter` still pulls the NEIGHBOURHOOD term
    up from the real terrain around the hole, which is exactly the bench
    signature - a false bench along every nodata boundary. Measured: a
    160x4 px NaN strip cut into an otherwise undisturbed hillside, the same
    shape `hillside_with_bench` cuts a real bench into, traced to a 159.0 m
    lineament of sinuosity 1.0 - indistinguishable in the output from the
    real thing.

    The fix tracks how much of each smoothing window was real ground.
    `finite` is true only where both the elevation AND the slope computed
    from it are finite - `compute_slope_aspect` uses a central difference,
    so a NaN one pixel away also poisons the slope at pixels adjacent to
    the hole, not only inside it. Smoothing that finite mask with the same
    Gaussian gives, per output pixel, the fraction of the kernel's mass that
    landed on real ground (1.0 in the interior of good data, falling toward
    0.0 approaching a large hole). Dividing the zero-filled smoothed slope
    by that weight corrects the average back to "mean of the real values
    only" instead of being dragged toward 0 by the fill - and where finite
    ground is everywhere (every existing test fixture), the weight is 1.0
    throughout, so this is a no-op and changes no existing measurement.

    A window can still be MOSTLY fill and technically pass a weight-based
    correction with a noisy estimate, so MIN_VALID_WEIGHT gates it: below
    0.5, more than half the window's mass is fabricated rather than
    measured, and the response is set to NaN rather than trusted. NaN also
    covers every pixel where the elevation or slope itself is not finite,
    regardless of weight, so the exact nodata locations are never reported
    as a value. `find_lineaments` builds `mask = response >= min_relief_deg`
    and `NaN >= x` is `False` in numpy, so a NaN response drops the pixel
    out of the mask with no special-casing needed there - verified directly
    (`np.nan >= MIN_RELIEF_DEG` is `False`) rather than assumed.
    """
    measured = compute_slope_aspect(elevation, pixel_size_m)
    slope = measured.slope_deg.astype("float64")
    finite = np.isfinite(elevation) & np.isfinite(slope)
    slope_filled = np.where(finite, slope, 0.0)

    sigma_px = max(1.0, (track_width_m * 2.5) / pixel_size_m)
    smoothed_sum = ndimage.gaussian_filter(slope_filled, sigma_px)
    weight = ndimage.gaussian_filter(finite.astype("float64"), sigma_px)

    with np.errstate(invalid="ignore", divide="ignore"):
        neighbourhood_slope = smoothed_sum / weight

    response = neighbourhood_slope - slope_filled
    contaminated = ~finite | (weight < MIN_VALID_WEIGHT)
    response[contaminated] = np.nan
    return response


def _trace(rows: np.ndarray, cols: np.ndarray, pixel_size_m: float):
    """Component pixels to an ordered list of (col, row) points.

    Project onto the major axis, bin, take each bin's centroid. Gentle curves
    survive; a hairpin does not, and is caught by the sinuosity filter.

    A bin's centroid sits inside the component's true ends, because it is
    the mean of member pixels that all lie ON one side of the extreme (there
    is no pixel beyond the boundary to balance it). Measured before this
    correction: extents of 50/60/65/160/400 px traced to 37.0/48.0/54.0/
    149.0/389.0 m - about one TRACE_BIN_M short each time. Left alone, that
    makes the real length floor about 62 m of ground rather than the 50 m
    MIN_LENGTH_M advertises.

    The fix: after binning, push the FIRST and LAST traced points out to the
    true extreme projected positions along the major axis - shifting only
    the along-axis component of each endpoint, never the cross-axis one -
    so the polyline reaches the component's actual ends while every interior
    bin centroid, and therefore the curve's shape, is untouched. Re-measured
    after the fix, the same five extents traced to 49.0/59.0/64.0/159.0/
    399.0 m - each exactly (extent - 1) m, which is the true centre-to-
    centre span of an N-pixel run, so MIN_LENGTH_M is now the real floor
    rather than an approximate one.

    A second, un-fixed limit falls out of the same binning: at a corner or
    a junction, pixels belonging to two different arms of the component can
    land in the same along-axis bin and get averaged into one centroid that
    cuts the corner - the traced path takes a chord across the bend instead
    of following it. This makes the traced path SHORTER than the ground
    truth, which LOWERS sinuosity (length / straight-line distance), the
    OPPOSITE direction from what `MAX_SINUOSITY` guards against (which
    drops paths that trace too LONG relative to their span, such as a
    hairpin). So the sinuosity filter is not a backstop for corner-cutting;
    a corner or junction can under-report length and pass the sinuosity
    filter more easily than the true path would.
    """
    points = np.column_stack([cols.astype("float64"), rows.astype("float64")])
    centre = points.mean(axis=0)
    centred = points - centre

    _, _, basis = np.linalg.svd(centred, full_matrices=False)
    major = basis[0]
    position = centred @ major

    bin_px = max(1.0, TRACE_BIN_M / pixel_size_m)
    position_min = float(position.min())
    position_max = float(position.max())
    span = position_max - position_min
    count = max(2, int(span / bin_px))
    edges = np.linspace(position_min, position_max, count + 1)
    index = np.clip(np.digitize(position, edges) - 1, 0, count - 1)

    traced = []
    bin_positions = []
    for bin_number in range(count):
        members_mask = index == bin_number
        if members_mask.any():
            members = points[members_mask]
            traced.append([float(members[:, 0].mean()), float(members[:, 1].mean())])
            bin_positions.append(float(position[members_mask].mean()))

    if len(traced) >= 2:
        first_shift = position_min - bin_positions[0]
        traced[0][0] += first_shift * major[0]
        traced[0][1] += first_shift * major[1]
        last_shift = position_max - bin_positions[-1]
        traced[-1][0] += last_shift * major[0]
        traced[-1][1] += last_shift * major[1]

    return [(x, y) for x, y in traced]


def _path_length_m(points, pixel_size_m: float) -> float:
    total = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        total += math.hypot(x1 - x0, y1 - y0) * pixel_size_m
    return total


def _end_to_end_bearing_deg(points) -> float:
    """Compass bearing of the straight chord from the first traced point to
    the last - NOT a mean of the bearings of each traced segment.

    A curving contour road, which is exactly what this module aims to
    trace, has a heading that changes along its length; the chord from one
    end to the other can differ substantially from that heading anywhere in
    the middle, and says nothing about how much the road turns along the
    way (that is what `sinuosity` is for). It is named for what it actually
    measures so a caller cannot mistake it for an average heading.
    """
    (x0, y0), (x1, y1) = points[0], points[-1]
    east = x1 - x0
    north = -(y1 - y0)  # rows count south
    return math.degrees(math.atan2(east, north)) % 360.0


#: The eight neighbours of a pixel, in order round the ring, as (row, col)
#: offsets. Order matters: `_junctions` counts background-to-skeleton
#: transitions going round it.
_RING = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))


def _junctions(skeleton: np.ndarray) -> np.ndarray:
    """Skeleton pixels where three or more arms meet.

    A pixel is a junction when going round its eight neighbours crosses
    from background into skeleton three or more times (the crossing
    number). Counting neighbours instead over-cuts: a diagonal staircase
    step puts three skeleton pixels round a pixel that is plainly mid-line.
    """
    padded = np.pad(skeleton, 1)
    rows, cols = skeleton.shape
    ring = [padded[1 + dy:1 + dy + rows, 1 + dx:1 + dx + cols] for dy, dx in _RING]
    crossings = sum((~ring[index]) & ring[(index + 1) % 8] for index in range(8))
    return skeleton & (crossings >= 3)


def _walk(pixels: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """A junction-free skeleton branch as an ordered (row, col) path, from
    one end to the other (from any pixel, for a closed loop).

    Where a staircase step offers two unvisited neighbours, the one with
    fewer unvisited neighbours of its own is taken - it is the one on the
    line, the other is the corner the line steps past.
    """
    def around(pixel):
        row, col = pixel
        return [(row + dy, col + dx) for dy, dx in _RING if (row + dy, col + dx) in pixels]

    ends = sorted(pixel for pixel in pixels if len(around(pixel)) == 1)
    path = [ends[0] if ends else min(pixels)]
    seen = {path[0]}
    while True:
        options = [pixel for pixel in around(path[-1]) if pixel not in seen]
        if not options:
            return path
        here = path[-1]
        step = min(options, key=lambda pixel: (
            sum(1 for other in around(pixel) if other not in seen),
            abs(pixel[0] - here[0]) + abs(pixel[1] - here[1])))
        path.append(step)
        seen.add(step)


def _arm_paths(skeleton: np.ndarray, junction_zone: np.ndarray) -> list[list[tuple[int, int]]]:
    """The skeleton with every junction zone removed, as walked arm paths."""
    labels, _ = ndimage.label(skeleton & ~junction_zone, structure=np.ones((3, 3)))
    paths = []
    for label, box in enumerate(ndimage.find_objects(labels), start=1):
        if box is None:
            continue
        rows, cols = np.nonzero(labels[box] == label)
        path = _walk(set(zip((rows + box[0].start).tolist(), (cols + box[1].start).tolist())))
        if len(path) >= 2:
            paths.append(path)
    return paths


def _pair_through_junctions(paths, zones: np.ndarray, pixel_size_m: float) -> dict:
    """{(arm, end): (arm, end)} for arm ends that carry straight through a
    junction - see MAX_JUNCTION_TURN_DEG. End 0 is a path's first pixel,
    end 1 its last."""
    reach = max(2, int(round(JUNCTION_HEADING_M / pixel_size_m)))
    at_zone: dict[int, list] = {}
    for index, path in enumerate(paths):
        if len(path) < 3:
            continue
        for end in (0, 1):
            tip = path[0] if end == 0 else path[-1]
            row, col = tip
            window = zones[max(row - 1, 0):row + 2, max(col - 1, 0):col + 2]
            touching = window[window > 0]
            if touching.size == 0:
                continue
            step = min(len(path) - 1, reach)
            inner = path[step] if end == 0 else path[-1 - step]
            heading = np.array([tip[0] - inner[0], tip[1] - inner[1]], dtype=float)
            norm = float(np.hypot(*heading))
            if norm > 0:
                at_zone.setdefault(int(touching.min()), []).append((index, end, heading / norm))

    partner: dict = {}
    for ends in at_zone.values():
        options = []
        for a in range(len(ends)):
            for b in range(a + 1, len(ends)):
                (arm_a, end_a, head_a), (arm_b, end_b, head_b) = ends[a], ends[b]
                if arm_a == arm_b:
                    continue
                # Straight through: one arm heads INTO the junction the way
                # the other heads OUT of it.
                turn = math.degrees(math.acos(max(-1.0, min(1.0, float(head_a @ -head_b)))))
                if turn <= MAX_JUNCTION_TURN_DEG:
                    options.append((turn, (arm_a, end_a), (arm_b, end_b)))
        for _, first, second in sorted(options, key=lambda item: item[0]):
            if first not in partner and second not in partner:
                partner[first] = second
                partner[second] = first
    return partner


def _chains(count: int, partner: dict) -> list[list[tuple[int, bool]]]:
    """Arms linked end to end through `partner`, as [(arm, reversed), ...].
    A chain starts at an arm end with no partner; what is left after that
    is a loop, opened at its first arm."""
    used: set[int] = set()

    def follow(arm: int, enter: int):
        sequence = []
        while arm not in used:
            used.add(arm)
            sequence.append((arm, enter == 1))
            nxt = partner.get((arm, 1 - enter))
            if nxt is None:
                break
            arm, enter = nxt
        return sequence

    chains = []
    for arm in range(count):
        if arm in used:
            continue
        for end in (0, 1):
            if (arm, end) not in partner:
                chains.append(follow(arm, end))
                break
    for arm in range(count):
        if arm not in used:
            chains.append(follow(arm, 0))
    return chains


def centreline_branches(component: np.ndarray, pixel_size_m: float) -> list[dict]:
    """A component too winding or too wide for `_trace`, as its centreline:
    cut at every junction, then rejoined wherever two arms carry straight
    on through it.

    WHY
    ---
    `_trace` squashes a component onto its major axis. A road network -
    a main grade with spurs off a landing - is ONE connected component on
    the mask, and squashing a tree onto a line traces a zig-zag that fails
    MAX_SINUOSITY, so the whole network used to be dropped. Measured on
    real ground (Mendocino County, 2 m pixels, 900 m round a
    test point, 2026-10-21): 9 components were dropped that way, holding
    11.1 km of traced bench - among them a 1,036 m road with two spurs,
    plainly visible as cut and fill on the hillshade, at sinuosity 1.97.

    Cutting at every junction was not enough on its own: that same network
    has 21 junctions, and cut at all of them it came apart into 38 arms, of
    which only 7 reached 50 m - a road visible end to end on the hillshade
    arrived as pieces of 80-290 m ranked far down the list. So arms that
    carry on through a junction within MAX_JUNCTION_TURN_DEG of straight
    are joined back into one line; a spur leaving at an angle stays apart.

    HOW
    ---
    Thin the component to a one-pixel skeleton (`skimage` - see
    pyproject), remove each junction and the pixels round it so each arm is
    its own piece, walk each arm end to end, then chain arms through the
    junctions they cross straight. Width comes from the distance transform
    at the centreline: a centre pixel of an N-pixel-wide strip sits about
    (N + 1) / 2 pixels from the nearest background, so width is
    (2 * depth - 1) pixels.

    Returns dicts with `pixels` [(col, row), ...] in the COMPONENT's own
    frame, `length_m`, `mean_width_m`, and `arms`: the same three for each
    arm the line was joined from, in chain order and direction (one entry
    when it was not joined), so a caller can regroup them when the joined
    line fails a test - see `_regrouped`.
    """
    from skimage.morphology import skeletonize

    padded = np.pad(component.astype(bool), 1)
    skeleton = skeletonize(padded)
    depth = ndimage.distance_transform_edt(padded)
    # The junction pixel AND its ring: removing the centre alone leaves the
    # arms' first pixels diagonally adjacent to one another, so a three-way
    # junction stayed one piece and only its first arm was walked (measured:
    # a trunk with two spurs came back as one 161 m branch).
    junction_zone = ndimage.binary_dilation(_junctions(skeleton), structure=np.ones((3, 3)))
    zones, _ = ndimage.label(junction_zone, structure=np.ones((3, 3)))
    paths = _arm_paths(skeleton, junction_zone)

    def described(path):
        length_m = sum(math.hypot(r1 - r0, c1 - c0)
                       for (r0, c0), (r1, c1) in zip(path, path[1:])) * pixel_size_m
        width_px = float(np.mean([2.0 * depth[pixel] - 1.0 for pixel in path]))
        # Back out of the padding, into the component's frame.
        return {"pixels": [(float(col - 1), float(row - 1)) for row, col in path],
                "length_m": length_m,
                "mean_width_m": max(width_px, 1.0) * pixel_size_m}

    partner = _pair_through_junctions(paths, zones, pixel_size_m)

    branches = []
    for chain in _chains(len(paths), partner):
        joined: list[tuple[int, int]] = []
        for arm, backwards in chain:
            joined.extend(reversed(paths[arm]) if backwards else paths[arm])
        line = described(joined)
        # Each arm as it runs IN the chain - reversed where the chain walks
        # it backwards - so consecutive parts join end to end.
        parts = [described(list(reversed(paths[arm])) if backwards else paths[arm])
                 for arm, backwards in chain]
        # Width as the arms' own, weighted by length: the straight bridge
        # across a junction zone crosses the junction's widened tread.
        total = sum(part["length_m"] for part in parts)
        if total > 0:
            line["mean_width_m"] = sum(part["mean_width_m"] * part["length_m"]
                                       for part in parts) / total
        line["arms"] = parts
        branches.append(line)
    return branches


def _thin_points(points, pixel_size_m: float):
    """Every pixel of a walked branch is far more vertices than a line
    needs: keep one per TRACE_BIN_M, and always the last."""
    step = max(1, int(round(TRACE_BIN_M / pixel_size_m)))
    kept = list(points[::step])
    if kept[-1] != points[-1]:
        kept.append(points[-1])
    return kept


def _record(points, pixel_size_m: float, length_m: float, width_m: float,
            sinuosity: float, traced_by: str) -> dict:
    return {
        "pixels": points,
        "length_m": round(length_m, 1),
        "end_to_end_bearing_deg": round(_end_to_end_bearing_deg(points), 1),
        "sinuosity": round(sinuosity, 3),
        "mean_width_m": round(width_m, 1),
        "traced_by": traced_by,
    }


def _judged_branch(branch: dict, offset, pixel_size_m: float, track_width_m: float,
                   min_length_m: float) -> dict | None:
    """A centreline branch as a lineament record, or None when it fails
    the length, width or MAX_BRANCH_SINUOSITY test. `offset` is the
    component box's (col, row) origin on the raster."""
    if branch["length_m"] < min_length_m:
        return None
    if not (MIN_WIDTH_FACTOR * track_width_m <= branch["mean_width_m"]
            <= MAX_WIDTH_FACTOR * track_width_m):
        return None
    points = [(x + offset[0], y + offset[1]) for x, y in branch["pixels"]]
    straight_m = math.hypot(points[-1][0] - points[0][0],
                            points[-1][1] - points[0][1]) * pixel_size_m
    sinuosity = branch["length_m"] / straight_m if straight_m > 0 else float("inf")
    if sinuosity > MAX_BRANCH_SINUOSITY:
        return None
    record = _record(_thin_points(points, pixel_size_m), pixel_size_m, branch["length_m"],
                     branch["mean_width_m"], sinuosity, "centreline")
    record["arms_joined"] = len(branch.get("arms") or [branch])
    return record


def _joined(parts: list[dict], pixel_size_m: float) -> dict:
    """Consecutive chain parts as one branch: pixels end to end, length
    including the bridge across each junction, width weighted by length."""
    pixels: list = []
    length_m = 0.0
    for part in parts:
        if pixels:
            (x0, y0), (x1, y1) = pixels[-1], part["pixels"][0]
            length_m += math.hypot(x1 - x0, y1 - y0) * pixel_size_m
        pixels.extend(part["pixels"])
        length_m += part["length_m"]
    total = sum(part["length_m"] for part in parts) or 1.0
    return {"pixels": pixels, "length_m": length_m,
            "mean_width_m": sum(part["mean_width_m"] * part["length_m"] for part in parts) / total,
            "arms": parts}


def _regrouped(parts: list[dict], offset, pixel_size_m: float, track_width_m: float,
               min_length_m: float) -> list[dict]:
    """A failed chain's arms, in order, grouped greedily into the longest
    runs that each pass `_judged_branch`.

    Measured on real ground (Mendocino County, 2026-10-21): five arms
    chained into a 373 m U - two grades meeting at the foot of a hairpin -
    read sinuosity 3.63 and failed MAX_BRANCH_SINUOSITY. Falling back to
    single arms returned pieces of 60-114 m; regrouping returns each leg."""
    records: list[dict] = []
    group: list[dict] = []
    for part in parts:
        trial = group + [part]
        if group and _judged_branch(_joined(trial, pixel_size_m), offset, pixel_size_m,
                                    track_width_m, min_length_m) is None:
            # Adding this arm breaks the run; keep the run if it passes alone.
            done = _judged_branch(_joined(group, pixel_size_m), offset, pixel_size_m,
                                  track_width_m, min_length_m)
            if done is not None:
                records.append(done)
            group = [part]
        else:
            group = trial
    if group:
        done = _judged_branch(_joined(group, pixel_size_m), offset, pixel_size_m,
                              track_width_m, min_length_m)
        if done is not None:
            records.append(done)
    return records


def find_lineaments(elevation: np.ndarray, pixel_size_m: float, *,
                    track_width_m: float = 4.0,
                    min_relief_deg: float = MIN_RELIEF_DEG,
                    min_length_m: float = MIN_LENGTH_M) -> list[dict]:
    """Benches in the elevation, as pixel-space polylines with measurements.

    Coordinates are (column, row) floats on the DEM grid. Projecting them to
    ground is the orchestration layer's job, because only it holds the
    raster's transform.
    """
    response = bench_response(elevation, pixel_size_m, track_width_m)
    return lineaments_from_mask(response >= min_relief_deg, pixel_size_m,
                                track_width_m=track_width_m, min_length_m=min_length_m)


def lineaments_from_mask(mask: np.ndarray, pixel_size_m: float, *,
                         track_width_m: float = 4.0,
                         min_length_m: float = MIN_LENGTH_M) -> list[dict]:
    """The bench mask's components as traced, filtered polylines.

    Each component is traced along its major axis (`_trace`). One that
    traces too winding - a tree of spurs, a hairpin, a contouring road
    that bends back - is split into its centreline branches instead
    (`centreline_branches`) and each branch is judged on its own length
    and width. `traced_by` says which: "axis" or "centreline".
    """
    if not mask.any():
        return []

    labels, count = ndimage.label(mask)
    # `find_objects` returns each component's bounding-box slice in one pass
    # over the label array, instead of the `labels == label` scan below
    # rescanning the WHOLE array once per component - quadratic in area once
    # the mask fragments. Measured on a deliberately fragmented mask (white
    # elevation noise on a bare plane, 1 m pixels): at 160k px / 20873
    # components the old scan took 7.185 s against 0.067 s here (107x); at
    # 360k px / 47022 components, 36.208 s against 0.151 s (240x). Both
    # counted the same 470 and 1097 surviving components, so the rewrite
    # changes nothing about what is found.
    boxes = ndimage.find_objects(labels)
    found = []
    for label, box in enumerate(boxes, start=1):
        if box is None:
            continue
        # Pixel count first, from the box alone, so a component too small to
        # ever pass floors is dropped before the (still box-local, so cheap)
        # label comparison below - the 8-pixel floor applied as early as
        # possible.
        box_rows = box[0].stop - box[0].start
        box_cols = box[1].stop - box[1].start
        if box_rows * box_cols < 8:
            continue

        # Bounding boxes of different components can overlap, so the box's
        # own pixels must still be masked down to this specific label before
        # counting or reading coordinates out of it.
        local = labels[box] == label
        if local.sum() < 8:
            continue
        local_rows, local_cols = np.nonzero(local)
        rows = local_rows + box[0].start
        cols = local_cols + box[1].start

        points = _trace(rows, cols, pixel_size_m)
        if len(points) < 2:
            continue

        length_m = _path_length_m(points, pixel_size_m)
        if length_m < min_length_m:
            continue

        straight_m = math.hypot(points[-1][0] - points[0][0],
                                points[-1][1] - points[0][1]) * pixel_size_m
        sinuosity = length_m / straight_m if straight_m > 0 else float("inf")
        # A true mean (area / path length): a narrow tread with a wider
        # pull-out or washout reports one number between the two, so it
        # understates the tread's actual width and overstates the flare's.
        area_m2 = rows.size * pixel_size_m**2
        width_m = area_m2 / length_m
        # Too winding to trace along one axis, OR so wide for its traced
        # length that it must be several lines squashed into one: judge each
        # arm instead. The second case is a switchback road, whose legs all
        # bin onto the same axis - measured on a drawn five-leg switchback,
        # it traced 204 m straight (sinuosity 1.0) at 1,030 m of tread and
        # was dropped as too wide, never as too winding.
        if sinuosity > MAX_SINUOSITY or width_m > MAX_WIDTH_FACTOR * track_width_m:
            offset = (box[1].start, box[0].start)
            for branch in centreline_branches(local, pixel_size_m):
                record = _judged_branch(branch, offset, pixel_size_m, track_width_m,
                                        min_length_m)
                if record is not None:
                    found.append(record)
                    continue
                # The joined line failed a test - most often winding past
                # MAX_BRANCH_SINUOSITY, where a hairpin's two legs were
                # chained through the junction at its foot - so regroup its
                # arms into the longest runs that pass.
                if len(branch["arms"]) > 1:
                    found.extend(_regrouped(branch["arms"], offset, pixel_size_m,
                                            track_width_m, min_length_m))
            continue

        if not (MIN_WIDTH_FACTOR * track_width_m <= width_m
                <= MAX_WIDTH_FACTOR * track_width_m):
            continue

        found.append(_record(points, pixel_size_m, length_m, width_m, sinuosity, "axis"))

    found.sort(key=lambda item: item["length_m"], reverse=True)
    return found


def subtract_known_ways(lineaments: list[dict], routes: list[dict],
                        *, tolerance_m: float, transform,
                        latitude_deg: float,
                        min_length_m: float = MIN_LENGTH_M) -> tuple[list[dict], list[str]]:
    """Drop what OpenStreetMap already has; measure the rest against it.

    `routes` are the staged `osm.json` records, whose geometry is already in
    EPSG:3857. Returns the residue with a `geometry` in that same projection
    and a `nearest_known_way`, plus any warnings.

    A lineament at least KEEP_IF_OUTSIDE outside the buffer is offered whole.
    One that is mostly a mapped way is NOT dropped whole: each stretch of it
    outside the buffer and at least `min_length_m` long is offered as its own
    record, with `clipped_from_m` (the whole lineament's length) and
    `mapped_share`. Measured on real ground (Mendocino County, 2026-10-21):
    a 1,631 m bench followed a mapped service road for two thirds of its
    length and then ran on, unmapped, for some 500 m - and all of it,
    the unmapped run included, used to be dropped as "already mapped".
    """
    from shapely.geometry import LineString, mapping, shape
    from shapely.ops import unary_union

    warnings: list[str] = []
    ground_per_unit = ground_meters_per_mercator_unit(latitude_deg)
    tolerance_units = tolerance_m / ground_per_unit

    # Geometry and route stay paired. Filtering the geometries into their own
    # list and zipping it back against the unfiltered `routes` would slip by
    # one for every record osm.json carries without a geometry, and report a
    # real distance under the wrong way's name - the kind of wrong that reads
    # as fact on a planning map.
    pairs = [(shape(route["geometry"]), route)
             for route in routes if route.get("geometry")]
    known = [geometry for geometry, _ in pairs]
    buffered = unary_union(known).buffer(tolerance_units) if known else None
    if not known:
        warnings.append("no staged OSM ways; every lineament is offered unfiltered")

    def nearest_to(line):
        if not pairs:
            return None
        gap_units, route = min(((geometry.distance(line), route)
                                for geometry, route in pairs), key=lambda pair: pair[0])
        return {"name": route.get("properties", {}).get("name"),
                "distance_m": round(gap_units * ground_per_unit, 1)}

    min_length_units = min_length_m / ground_per_unit
    residue = []
    for lineament in lineaments:
        projected = LineString([transform * (x, y) for x, y in lineament["pixels"]])
        base = {key: value for key, value in lineament.items() if key != "pixels"}

        outside = projected.difference(buffered) if buffered is not None else projected
        share = outside.length / projected.length if projected.length > 0 else 1.0
        if share >= KEEP_IF_OUTSIDE:
            residue.append({**base, "geometry": mapping(projected),
                            "nearest_known_way": nearest_to(projected)})
            continue

        # Mostly a mapped way: offer only what runs on past it.
        for piece in _line_pieces(outside):
            if piece.length < min_length_units:
                continue
            (x0, y0), (x1, y1) = piece.coords[0], piece.coords[-1]
            chord = math.hypot(x1 - x0, y1 - y0)
            residue.append({
                **base,
                "length_m": round(piece.length * ground_per_unit, 1),
                "end_to_end_bearing_deg": round(math.degrees(math.atan2(x1 - x0, y1 - y0)) % 360.0, 1),
                "sinuosity": round(piece.length / chord, 3) if chord > 0 else None,
                "clipped_from_m": lineament["length_m"],
                "mapped_share": round(1.0 - share, 3),
                "geometry": mapping(piece),
                "nearest_known_way": nearest_to(piece),
            })

    return residue, warnings


def _line_pieces(geometry) -> list:
    """Every LineString in `geometry`, runs that meet end to end joined."""
    from shapely.geometry import MultiLineString
    from shapely.ops import linemerge

    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry]
    lines = [part for part in getattr(geometry, "geoms", []) if part.geom_type == "LineString"]
    lines += [line for part in getattr(geometry, "geoms", [])
              if part.geom_type == "MultiLineString" for line in part.geoms]
    if not lines:
        return []
    merged = linemerge(MultiLineString(lines))
    return list(merged.geoms) if merged.geom_type == "MultiLineString" else [merged]
