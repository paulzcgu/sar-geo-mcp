"""Effort per candidate, from the shared movement model.

How fast a team walks is `terrain.movement`'s, the one model every tool
reads, so a candidate's effort, its minutes from the centre and the
travel-time rings cannot disagree about the same ground:

- a LINE (a trail or road) is walked ON a mapped way: Tobler along its
  axis at the on-way factor, and vegetation does not slow it - the trail
  cuts through the brush beside it. Its time is the length over that rate,
  one way, end to end;
- a CORRIDOR (a waterway) is swept OFF any way: Tobler
  at the off-way factor times the LANDFIRE lifeform factor, area-weighted
  over the corridor, through the same `traverse_metrics` call
  `terrain_metrics` makes. Its time is sweep time per pass at the
  corridor's own sweep width.

Canopy density does not change speed (it is not staged by the hub, and
where the debrief app has it, it only narrows sweep width). Density stays
`moderate` for sweep width here and says so. Every assumption travels with
the answer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..terrain import movement
from ..terrain.evt import composition
from ..terrain.traverse import ASSUMPTIONS, tobler_rate_kmh, traverse_metrics
from .boundary import Ground

DEFAULT_DENSITY = "moderate"
DENSITY_NOTE = ("no CIR panel is staged by the hub, so vegetation density is assumed "
                "moderate for sweep width; walking speed does not depend on density - it "
                "comes from the vegetation type (terrain.movement)")
#: Ground metres between samples along a line axis.
AXIS_SAMPLE_M = 10.0


def _lifeforms(evt_path: Path | None, geometry_3857) -> tuple[dict, str]:
    if evt_path is None or not Path(evt_path).exists():
        return {}, "none; sweep width default"
    result = composition(Path(evt_path), geometry_3857)
    return result["fractions"], result["source"]


def axis_parts(axis) -> list:
    """An axis as a list of its LineString parts.

    A line candidate's axis is a MultiLineString whenever its named way forks
    or the area cuts it into pieces, so every measurement along an axis has to
    be able to walk more than one part. Empty parts and any non-lineal member
    of a collection are dropped: a clip can leave a Point where a line only
    grazes the area ring, and a point has nothing to sample along.
    """
    if axis is None or axis.is_empty:
        return []
    if axis.geom_type in ("LineString", "LinearRing"):
        return [axis]
    return [part for part in getattr(axis, "geoms", [])
            if part.geom_type in ("LineString", "LinearRing") and not part.is_empty]


def sample_along(ground: Ground, axis) -> tuple[np.ndarray, np.ndarray]:
    """Raster (rows, cols) every `AXIS_SAMPLE_M` ground metres along an axis.

    The one place an axis becomes raster indices. A line's slopes and a
    candidate's elevation profile are then read at exactly the same points,
    and the spacing is one number in one module rather than a constant each
    caller keeps its own copy of. Indices are clipped to the grid: an axis
    clipped to the subject can still round a sample onto the edge pixel.

    EVERY PART is sampled, and each gets its own two ends. Interpolating a
    MultiLineString as though it were one concatenated line divides the same
    sample budget by the total length, so a short branch of a forked candidate
    can pick up no sample at all - and the fork is then measured, and its
    walking time reported, as though that branch were not there.
    """
    from rasterio.transform import rowcol

    spacing_units = AXIS_SAMPLE_M / ground.scale
    xs: list[float] = []
    ys: list[float] = []
    for part in axis_parts(axis):
        count = max(2, int(part.length / spacing_units) + 1)
        for i in range(count):
            point = part.interpolate(min(part.length, i * spacing_units))
            xs.append(point.x)
            ys.append(point.y)
    empty = np.zeros(0, dtype=int)
    if not xs:
        return empty, empty
    rows, cols = rowcol(ground.grid.transform, xs, ys)
    shape = ground.slope_deg.shape
    rows = np.clip(np.asarray(rows, dtype=int), 0, shape[0] - 1)
    cols = np.clip(np.asarray(cols, dtype=int), 0, shape[1] - 1)
    return rows, cols


def _window(ground: Ground, geometry_3857) -> tuple[int, int, int, int]:
    """`(row0, row1, col0, col1)`: the raster window a geometry can touch,
    padded by a pixel and clipped to the grid.

    The pad is for the pixel a boundary lands inside; the clip is because a
    candidate can run to the edge of the staged raster. A geometry entirely
    off the grid clamps to a single pixel it does not cover, which rasterizes
    to an empty mask - the same answer the full raster gave, without the
    full raster.
    """
    from rasterio.transform import rowcol

    height, width = ground.slope_deg.shape
    minx, miny, maxx, maxy = geometry_3857.bounds
    rows, cols = rowcol(ground.grid.transform, [minx, maxx], [maxy, miny])
    row0 = int(np.clip(min(rows) - 1, 0, height - 1))
    row1 = int(np.clip(max(rows) + 2, row0 + 1, height))
    col0 = int(np.clip(min(cols) - 1, 0, width - 1))
    col1 = int(np.clip(max(cols) + 2, col0 + 1, width))
    return row0, row1, col0, col1


def candidate_effort(ground: Ground, geometry_3857, kind: str, *,
                     evt_path: Path | None, density_class: str = DEFAULT_DENSITY,
                     include_shared_assumptions: bool = True) -> dict:
    """Effort for one candidate geometry (Web Mercator): a line's one-way time
    or a corridor's sweep time per pass.

    `include_shared_assumptions=False` leaves out the density class, its note
    and the traverse table - the part that is the same for every candidate on
    one run - so a caller measuring a whole set can state them once beside
    the set instead of repeating a hundred copies of the same table on the
    wire. What is left is what varies per candidate: the mask note, and the
    lifeform source and fractions the corridor was read with.
    """
    from affine import Affine
    from rasterio.features import rasterize

    assumptions: dict = ({"density_class": density_class, "density_note": DENSITY_NOTE,
                          "traverse": ASSUMPTIONS,
                          "movement": movement.DEFAULT.assumptions()}
                         if include_shared_assumptions else {})

    if kind == "line":
        # Over EVERY part of the axis: a team walks all of a forked
        # assignment, so its rate is the rate over all of it and its length is
        # the total, not the longest branch's.
        rows, cols = sample_along(ground, geometry_3857)
        slopes = ground.slope_deg[rows, cols]
        # NaN slopes (a DEM void along the axis) are passed straight through:
        # tobler_rate_kmh maps NaN to flat via nan_to_num, the same handling
        # traverse_metrics gives a masked NaN pixel, rather than dropping the
        # sample and skewing the mean over what is left.
        baseline = float(tobler_rate_kmh(slopes).mean()) if slopes.size else 0.0
        rate = baseline * movement.DEFAULT.speed_factor * movement.DEFAULT.on_way_factor
        length_km = geometry_3857.length * ground.scale / 1000.0
        return {
            "kind": "line",
            "length_m": length_km * 1000.0,
            "baseline_rate_kmh": baseline,
            "rate_kmh": rate,
            "hours_one_way": (length_km / rate) if rate > 0 else None,
            "method": "tobler1993 along the axis at the on-way factor",
            "assumptions": assumptions,
        }

    lifeforms, source = _lifeforms(evt_path, geometry_3857)
    # Only the window the corridor can touch. A hasty corridor is a 30 m
    # ribbon on a raster kilometres across, and rasterizing it at full size
    # once per candidate is the whole raster's work a hundred times over for
    # the same few hundred pixels. The window holds every pixel the corridor
    # covers, in the same order, so the statistics are unchanged.
    row0, row1, col0, col1 = _window(ground, geometry_3857)
    window_shape = (row1 - row0, col1 - col0)
    window_transform = ground.grid.transform @ Affine.translation(col0, row0)
    slopes = ground.slope_deg[row0:row1, col0:col1]
    # Match boundary.py's interior-mask convention: pixels whose centre lies
    # inside the corridor, falling back to touched pixels only when the
    # corridor is thinner than a pixel and the strict mask is empty.
    mask = rasterize([(geometry_3857, 1)], out_shape=window_shape,
                     transform=window_transform, fill=0, all_touched=False).astype(bool)
    if mask.any():
        assumptions["mask_note"] = "statistics use pixels whose centre lies inside the corridor"
    else:
        mask = rasterize([(geometry_3857, 1)], out_shape=window_shape,
                         transform=window_transform, fill=0, all_touched=True).astype(bool)
        assumptions["mask_note"] = ("the corridor is thinner than a pixel; "
                                    "statistics use touched pixels")
    area_m2 = geometry_3857.area * ground.scale ** 2
    result = traverse_metrics(slope_deg=slopes, mask=mask, area_m2=area_m2,
                              density_class=density_class, lifeform_fractions=lifeforms)
    assumptions["lifeform_source"] = source
    assumptions["lifeform_fractions"] = lifeforms
    if include_shared_assumptions:
        assumptions["traverse"] = result["assumptions"]
    return {
        "kind": "corridor",
        "area_m2": area_m2,
        "baseline_rate_kmh": result["baseline_rate_kmh"],
        "rate_kmh": result["vegetation_adjusted_rate_kmh"],
        "effective_sweep_width_m": result["effective_sweep_width_m"],
        "sweep_time_hours_per_pass": result["sweep_time_hours_per_pass"],
        "method": result["method"],
        "assumptions": assumptions,
    }
