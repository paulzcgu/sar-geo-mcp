"""Quarantine for the pysheds dependency and its numpy-2 incompatibility.

pysheds 0.5 predates numpy 2 and calls the removed `np.in1d`. Rather than
scatter that knowledge, the shim is applied exactly once, here, before pysheds
is imported. Nothing else in the codebase should import pysheds directly.

If pysheds is ever replaced with a hand-rolled D8, only this module changes.
"""

from dataclasses import dataclass

import numpy as np

# Must run BEFORE pysheds is imported. np.in1d was removed in numpy 2.0; isin
# is the direct replacement with the same semantics for pysheds' usage.
if not hasattr(np, "in1d"):
    np.in1d = np.isin

from pysheds.grid import Grid  # noqa: E402
from pysheds.view import Raster, ViewFinder  # noqa: E402

_NODATA = np.float64(-999999.0)


@dataclass(frozen=True)
class FlowResult:
    """D8 routing over a filled, flat-resolved DEM."""

    flow_direction: np.ndarray
    accumulation: np.ndarray
    # Traced channel centrelines, each a list of (x, y) in the transform's CRS.
    # Populated only when network_min_cells is given.
    network: list[list[tuple[float, float]]] | None = None


def flow_accumulation(elevation: np.ndarray, transform,
                      network_min_cells: float | None = None) -> FlowResult:
    """Fill depressions, resolve flats, route D8, and accumulate.

    NaN cells are filled to the local minimum before routing: pysheds treats
    nodata as a barrier, and a NaN hole inside an assignment would silently
    truncate the channel network rather than raising.

    When `network_min_cells` is given, channel centrelines are also traced by
    following D8 flow paths. Tracing must happen here because it needs the
    pysheds Raster, which does not survive conversion to numpy.
    """
    working = np.asarray(elevation, dtype="float64").copy()
    if np.isnan(working).any():
        finite = working[np.isfinite(working)]
        fill_value = float(finite.min()) if finite.size else 0.0
        working[~np.isfinite(working)] = fill_value

    viewfinder = ViewFinder(
        affine=transform, shape=working.shape, nodata=_NODATA
    )
    grid = Grid(viewfinder=viewfinder)
    raster = Raster(working, viewfinder=viewfinder)

    filled = grid.fill_depressions(raster)
    inflated = grid.resolve_flats(filled)
    flow_direction = grid.flowdir(inflated)
    accumulation = grid.accumulation(flow_direction)

    network = None
    if network_min_cells is not None:
        # Must stay a pysheds Raster: `np.asarray(acc) > n` yields a plain
        # ndarray and extract_river_network rejects it.
        branches = grid.extract_river_network(
            flow_direction, accumulation > float(network_min_cells)
        )
        network = [
            [tuple(point) for point in feature["geometry"]["coordinates"]]
            for feature in branches.get("features", [])
            if len(feature["geometry"]["coordinates"]) >= 2
        ]

    return FlowResult(
        flow_direction=np.asarray(flow_direction),
        accumulation=np.asarray(accumulation),
        network=network,
    )
