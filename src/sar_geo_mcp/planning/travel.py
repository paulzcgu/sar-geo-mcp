"""Walking time from the centre: the shared walk, read per candidate.

WHAT THIS IS
------------
Minutes on foot from one origin to every cell of the staged raster, on the
same walk the travel-time rings use (`planning.walking`, with the numbers of
`terrain.movement`): Tobler on each step's signed grade, faster on mapped
trails and roads, slowed off them by what LANDFIRE says grows there, a creek
costing minutes to cross, and rivers, open water, motorways and ground too
steep to walk closed. So a candidate's `travel_minutes_from_center` and the
ring drawn from the same centre agree, cell for cell, and its effort reads
the same factors (`effort.candidate_effort`) - on the DEFAULT model, which is
the only one walked here. A ring drawn at a pace of its own (`speed_factor`,
`off_way_factor` and the like) walks another, and plan_initial_response
warns when it shows the two side by side.

WHAT THIS IS NOT
----------------
A route. It walks the elevation model, OpenStreetMap and LANDFIRE; it knows
nothing about fences, private land, unmapped trails or brush LANDFIRE does
not show, or where a vehicle can drop a team. A DEM void is impassable -
ground nobody measured is not a shortcut - which can make a candidate
unreachable that a team could in fact walk to.

So a planner reads this as a RANK, not a promise: a drainage across a cliff
band is farther than a trail along the contour, which is the ordering that
matters on day one; it does not say a team will be there in eleven minutes.
"""

from __future__ import annotations

import math

import numpy as np

from ..terrain.movement import DEFAULT
from .boundary import Ground
from .effort import _window, sample_along
from .walking import NO_OSM_WARNING, load_walking_ground, solve
from .walking import assumptions as walking_assumptions

TRAVEL_ASSUMPTIONS = {
    **walking_assumptions(DEFAULT),
    "origin": "the centre, snapped to the cell it falls in; the walk is outward from it",
    "not_a_route": ("walking time over the elevation model, OpenStreetMap and LANDFIRE, not "
                    "a route; it ignores fences, private land and unmapped trails. Read it "
                    "as a rank, not a promise"),
}


def travel_surface(ground: Ground, origin_3857, warnings: list[str] | None = None) -> np.ndarray:
    """Minutes on foot from `origin_3857` to every cell, `inf` where unreachable.

    The origin is snapped to the cell it falls in, clipped to the grid the way
    `effort.sample_along` clips a sample: the raster is staged around the
    subject, so an origin outside it is a rounding, not a journey. An origin
    on a DEM void has no measured ground to start from and the whole surface
    is `inf`.

    `warnings`, when given, gains what the walk could not see on this ground
    (no LANDFIRE; waterway kinds unknown) - not a missing osm.json, which the
    caller already reports in its own words. Ground too fine to walk raises
    ValueError (`walking.TOO_FINE_ERROR`).
    """
    from rasterio.transform import rowcol

    walking = load_walking_ground(ground.data_dir, DEFAULT, grid=ground.grid)
    if warnings is not None:
        warnings.extend(w for w in walking.warnings
                        if w != NO_OSM_WARNING and w not in warnings)
    height, width = walking.elevation.shape
    rows, cols = rowcol(ground.grid.transform, [origin_3857.x], [origin_3857.y])
    row = int(np.clip(int(np.asarray(rows).ravel()[0]), 0, height - 1))
    col = int(np.clip(int(np.asarray(cols).ravel()[0]), 0, width - 1))
    return solve(walking, DEFAULT, (row, col))


def travel_minutes_to(surface: np.ndarray, ground: Ground, geometry_3857) -> float | None:
    """The least minutes over the cells a candidate's geometry touches.

    A line axis is read at `effort.sample_along`'s points and a corridor
    through the same windowed rasterisation `candidate_effort` measures its
    slopes in, so a candidate is measured on exactly the cells it occupies
    rather than on a bounding box or a single nearest point.

    A multi-part axis - a named way that forks, or one the area cuts into
    pieces - is the MINIMUM over all of its parts, because a team reaches the
    assignment at whichever branch is nearest. `sample_along` puts samples on
    every part, so the minimum below is over all of them and not over the
    first one; this is the whole reason that sampler walks parts.

    `None` when every cell it touches is unreachable or it touches none of
    them. Never 0.0 as a stand-in: zero minutes means the candidate covers
    the centre, which is a measurement, and "not reachable" is not.
    """
    if geometry_3857.is_empty:
        return None
    if geometry_3857.geom_type in ("LineString", "MultiLineString", "LinearRing"):
        rows, cols = sample_along(ground, geometry_3857)
        values = surface[rows, cols]
    else:
        values = _corridor_values(surface, ground, geometry_3857)
    if values.size == 0:
        return None
    best = float(np.min(values))
    return best if math.isfinite(best) else None


def _corridor_values(surface: np.ndarray, ground: Ground, geometry_3857) -> np.ndarray:
    """The surface under a corridor, in the window `candidate_effort` uses.

    Same mask convention as `effort.candidate_effort`: pixels whose centre
    lies inside the corridor, falling back to touched pixels when the corridor
    is thinner than a pixel and the strict mask is empty. The two facts on one
    candidate are then measured on the same cells.
    """
    from affine import Affine
    from rasterio.features import rasterize

    row0, row1, col0, col1 = _window(ground, geometry_3857)
    window_shape = (row1 - row0, col1 - col0)
    window_transform = ground.grid.transform @ Affine.translation(col0, row0)
    mask = rasterize([(geometry_3857, 1)], out_shape=window_shape,
                     transform=window_transform, fill=0, all_touched=False).astype(bool)
    if not mask.any():
        mask = rasterize([(geometry_3857, 1)], out_shape=window_shape,
                         transform=window_transform, fill=0, all_touched=True).astype(bool)
    return surface[row0:row1, col0:col1][mask]
