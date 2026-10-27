"""Is this a place a team can occupy and work?

WHAT THIS ANSWERS THAT NOTHING ELSE HERE DOES
---------------------------------------------
Every other tool in this package answers a version of "sweep this ground
looking for a person": corridors, coverage, sweep width, hasty candidates,
assignment norms. None of them answers "can a team stand here, reach it, and
work it", which is the rope-rescue question - and equally the helispot, the
landing zone, the incident command post and the staging area.

Two preserve studies (El Corte de Madera Creek and Purisima Creek Redwoods,
2026-09) hand-rolled the same four measurements twice each. They are the four
this returns:

  FACE          the shape of the ground - drop, span, slope, aspect
  BENCH ABOVE   the band above the high edge, where the operation stands
  LANDING BELOW the band below the low edge, where the litter arrives
  VEGETATION    on all three separately

WHY VEGETATION IS PER FOOTPRINT
-------------------------------
Because a clean face above a scrub bench is not a usable site, and one
averaged number hides exactly that. Dense coastal scrub cannot be moved
through with a litter and it hides the lip, so it disqualifies ground however
good the slope is. Measured this way it ruled out ten of Purisima's
eighty-five long faces - two of them at over 90% shrub, one of which had
already been published as a candidate on slope alone.

A missing vegetation panel is a WARNING and a null, never a zero. A shrub
fraction of 0.0 that actually means "not measured" is the most dangerous
answer this function could give.

NO VERDICT
----------
There is deliberately no `feasible` key. The thresholds that turn a shrub
fraction into a decision belong to the agency running the exercise, and the
rule in this package is that the tools measure and the agent judges. The same
reason `hasty_candidates` returns no priority.

WHAT THE NUMBERS CANNOT SEE
---------------------------
A DEM smooths anything narrower than a few pixels, so an undercut reads
shallower than it is. Bare-earth returns see no boulders, no downed timber and
no understory. LANDFIRE is 30 m pixels - 900 m2, larger than many faces worth
rigging - and it reports vegetation TYPE, not density: a redwood stand carries
no chaparral but can carry tanoak and huckleberry that is still real work with
a litter. Every answer here is a reason to go and look, not a substitute.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

#: Ground metres of the band measured above the high edge and below the low
#: edge. 35 m is about the depth of usable staging ground a rope operation
#: occupies - the anchor cluster, the operations point and the gear pile - and
#: it is what both preserve studies used.
DEFAULT_BAND_M = 35.0

#: Elevation tolerance for "level with the edge", in metres. A band is the
#: ground within `band_m` horizontally AND within this of the edge's height,
#: so a band never wraps round onto the face it is meant to sit above.
EDGE_TOLERANCE_M = 2.0

CARDINALS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def _cardinal(aspect_deg: np.ndarray) -> str | None:
    """Circular mean of an aspect sample, as a compass point."""
    values = aspect_deg[np.isfinite(aspect_deg)]
    if values.size == 0:
        return None
    radians = np.radians(values)
    mean = math.degrees(math.atan2(np.sin(radians).mean(),
                                   np.cos(radians).mean())) % 360.0
    return CARDINALS[int((mean + 22.5) // 45) % 8]


def _slope_facts(slope: np.ndarray, pixel_area_m2: float) -> dict[str, Any]:
    values = slope[np.isfinite(slope)]
    if values.size == 0:
        return {"slope_mean_deg": None, "slope_p25_deg": None,
                "slope_p75_deg": None, "slope_max_deg": None, "area_m2": 0.0}
    return {
        "slope_mean_deg": round(float(values.mean()), 1),
        "slope_p25_deg": round(float(np.percentile(values, 25)), 1),
        "slope_p75_deg": round(float(np.percentile(values, 75)), 1),
        "slope_max_deg": round(float(values.max()), 1),
        "area_m2": round(float(values.size * pixel_area_m2)),
    }


def cover_for_mask(evt_path: Path, grid, mask: np.ndarray) -> dict[str, Any] | None:
    """Vegetation composition over the cells a boolean mask selects.

    WHY NOT `evt.composition`
    ------------------------
    That takes a geometry, and the bench above a face and the landing below it
    are not geometries - they are the parts of one ring that happen to sit at
    each end's height. Passing the ring for both would report identical cover
    for two footprints whose difference is the entire point: a clean face above
    a scrub bench is not a usable site.

    So the EVT raster is sampled at the mask's own cells, through
    `evt.codes_at_cells` - the one place a DEM cell meets its EVT pixel.
    """
    from ..sources.landfire import load_lifeform_table
    from ..terrain.evt import codes_at_cells

    if not mask.any():
        return None

    rows, cols = np.nonzero(mask)
    sample = codes_at_cells(evt_path, grid.transform, rows, cols)
    sample = sample[sample >= 0]
    if sample.size == 0:
        return None

    table = load_lifeform_table()
    lifeform: dict[str, int] = {}
    names: dict[str, int] = {}
    for code, count in zip(*np.unique(sample, return_counts=True)):
        form, _, name = table.get(int(code), ("Unknown", "Unknown", f"code {code}"))
        lifeform[form] = lifeform.get(form, 0) + int(count)
        names[name] = names.get(name, 0) + int(count)
    total = max(sum(lifeform.values()), 1)

    return {
        "tree": round(lifeform.get("Tree", 0) / total, 3),
        "shrub": round(lifeform.get("Shrub", 0) / total, 3),
        "herb": round(lifeform.get("Herb", 0) / total, 3),
        "sparse": round(lifeform.get("Sparse", 0) / total, 3),
        "dominant_lifeform": max(lifeform, key=lifeform.get),
        "top_types": [[name, round(count / total, 3)]
                      for name, count in sorted(names.items(), key=lambda i: -i[1])[:3]],
        "evt_pixels": int(sample.size),
    }


def site_facts(data_dir: str | Path, geometry: dict, *,
               band_m: float = DEFAULT_BAND_M) -> dict[str, Any]:
    """Measure a candidate site on staged ground.

    `geometry` is a GeoJSON Polygon or MultiPolygon in WGS84 - a face from
    `identify_features`, a shape a planner drew, or a buffered line. It must
    fall on the staged raster; geometry off it is an error rather than a
    partial answer, because a site half-measured is not a site.
    """
    from rasterio.features import geometry_mask
    from shapely.geometry import shape as shapely_shape
    from shapely.ops import transform as shapely_transform

    from ..geo import ground_meters_per_mercator_unit
    from ..terrain.grid import load_assignment_grid
    from ..terrain.slope import compute_slope_aspect

    data_dir = Path(data_dir)
    grid = load_assignment_grid(data_dir)
    scale = grid.pixel_size_m / abs(grid.transform.a)
    pixel_area_m2 = grid.pixel_size_m ** 2
    slope_aspect = compute_slope_aspect(grid.elevation, grid.pixel_size_m)
    warnings: list[str] = []

    from pyproj import Transformer
    to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
    footprint = shapely_transform(to_3857, shapely_shape(geometry))

    def mask_of(geom) -> np.ndarray:
        return geometry_mask([geom], out_shape=grid.elevation.shape,
                            transform=grid.transform, invert=True,
                            all_touched=True)

    face_mask = mask_of(footprint) & np.isfinite(grid.elevation)
    if not face_mask.any():
        raise ValueError(
            "the geometry covers no measured cell of this staged raster - it is "
            "outside the staged extent, or over a DEM void. Stage an area that "
            "contains it, or check the coordinate order.")

    elevations = grid.elevation[face_mask]
    low, high = float(elevations.min()), float(elevations.max())
    face = {
        "vertical_extent_m": round(high - low, 1),
        "elevation_low_m": round(low, 1),
        "elevation_high_m": round(high, 1),
        "span_m": round(float(max(footprint.bounds[2] - footprint.bounds[0],
                                  footprint.bounds[3] - footprint.bounds[1]) * scale)),
        "aspect": _cardinal(slope_aspect.aspect_deg[face_mask]),
        **_slope_facts(slope_aspect.slope_deg[face_mask], pixel_area_m2),
    }

    # The two bands. `difference` keeps them off the face itself, and the
    # elevation window keeps the band above from wrapping down the sides.
    ring = footprint.buffer(band_m / scale).difference(footprint)
    ring_mask = mask_of(ring) & np.isfinite(grid.elevation)
    bands = {}
    band_masks = {}
    for name, window in (("bench_above", grid.elevation >= high - EDGE_TOLERANCE_M),
                         ("landing_below", grid.elevation <= low + EDGE_TOLERANCE_M)):
        band_masks[name] = ring_mask & window
        bands[name] = _slope_facts(slope_aspect.slope_deg[band_masks[name]],
                                   pixel_area_m2)

    vegetation: dict[str, Any] = {"face": None, "bench_above": None,
                                  "landing_below": None}
    evt_path = data_dir / "evt.tif"
    if not evt_path.exists():
        warnings.append(
            "vegetation panel missing; cover is unknown for every footprint. "
            "That is not the same as no shrub - brush disqualifies a site, so "
            "this must be resolved on the ground before the site is used.")
    else:
        try:
            for name, mask in (("face", face_mask),
                               ("bench_above", band_masks["bench_above"]),
                               ("landing_below", band_masks["landing_below"])):
                vegetation[name] = cover_for_mask(evt_path, grid, mask)
        except (OSError, ValueError) as error:
            warnings.append(f"vegetation unreadable: {type(error).__name__}")

    return {
        "face": face,
        **bands,
        "vegetation": vegetation,
        "warnings": warnings,
        "assumptions": {
            "band_m": band_m,
            "edge_tolerance_m": EDGE_TOLERANCE_M,
            "dem_resolution_m": round(grid.pixel_size_m, 3),
            "vegetation_source": "LANDFIRE LF2024 EVT, 30 m - TYPE, not density",
            "source": ("slope and aspect from the staged 3DEP raster; vegetation "
                       "from the staged LANDFIRE clip"),
            "no_verdict": ("this returns measurements only - the thresholds that "
                           "make a site feasible are the agency's, not this "
                           "library's"),
            "blind_to": ("an undercut face reads shallower than it is; bare-earth "
                         "elevation sees no boulders, downed timber or understory; "
                         "30 m vegetation pixels are coarser than many riggable "
                         "faces"),
        },
    }
