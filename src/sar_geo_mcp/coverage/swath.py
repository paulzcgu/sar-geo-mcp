"""Coverage as a raster, not as vector swaths.

Buffering each GPS track by half the sweep width and unioning the results is
the obvious approach and it does not terminate: real sweep tracks self-cross
thousands of times, and unioning such polygons ran over ten minutes on seven
tracks before being killed. Burning the tracks into a grid and dilating runs
the same problem in under a second, and matches how the rest of the terrain
layer already works.
"""

from dataclasses import dataclass

import numpy as np
import rasterio.features
from rasterio.transform import from_origin
from scipy import ndimage

# Ground metres per cell. Fine enough that a 10 m sweep width is several
# cells across, coarse enough that a large assignment stays small.
CELL_GROUND_M = 2.0


@dataclass(frozen=True)
class CoverageGrid:
    zone_mask: np.ndarray      # True inside the assignment
    transform: object          # rasterio affine, in the zone's own units
    cell_ground_m: float
    ground_factor: float       # zone units -> ground metres

    @property
    def cell_area_m2(self) -> float:
        return self.cell_ground_m ** 2


def build_grid(zone, cell_ground_m: float = CELL_GROUND_M,
               ground_factor: float = 1.0) -> CoverageGrid:
    """A raster covering the assignment's bounding box."""
    cell = cell_ground_m / ground_factor
    min_x, min_y, max_x, max_y = zone.bounds
    width = max(1, int(np.ceil((max_x - min_x) / cell)))
    height = max(1, int(np.ceil((max_y - min_y) / cell)))
    transform = from_origin(min_x, max_y, cell, cell)
    mask = rasterio.features.rasterize(
        [(zone, 1)], out_shape=(height, width), transform=transform,
        fill=0, all_touched=True,
    ).astype(bool)
    return CoverageGrid(zone_mask=mask, transform=transform,
                        cell_ground_m=cell_ground_m, ground_factor=ground_factor)


def burn_swath(grid: CoverageGrid, geometries: list,
               sweep_width_ground_m: float) -> np.ndarray:
    """Cells within half a sweep width of any track, clipped to the zone.

    Realised via a Euclidean distance transform rather than dilation by a
    rasterised disc. The ESW is a computed value with a rationale behind it
    (see TerrainMetrics.traverse), and disc dilation with an integer cell
    radius (`round((esw/2)/cell)`) both inflated it - a 10.5 m ESW realised
    as a 14 m swath, +33% - and produced a rounding cliff where ESW 13.9 m
    and 14.0 m realised different radii entirely. The distance transform has
    no rounding step: the realised half-width tracks `sweep_width_ground_m /
    2` directly, to within one cell's discretisation.
    """
    shapes = [(g, 1) for g in geometries if not g.is_empty]
    if not shapes:
        return np.zeros_like(grid.zone_mask)
    burnt = rasterio.features.rasterize(
        shapes, out_shape=grid.zone_mask.shape, transform=grid.transform,
        fill=0, all_touched=True,
    ).astype(bool)
    distance_cells = ndimage.distance_transform_edt(~burnt)
    within = distance_cells * grid.cell_ground_m <= sweep_width_ground_m / 2
    return within & grid.zone_mask


# Below this a gap is raster noise along the boundary, not a searchable area.
MIN_GAP_M2 = 500.0


@dataclass(frozen=True)
class Gap:
    id: str
    area_m2: float
    geometry: dict             # GeoJSON mapping, in the zone's coordinates


def _labelled_regions(grid: CoverageGrid, mask: np.ndarray,
                      min_area_m2: float) -> list[tuple[float, dict]]:
    """Connected components of `mask`, largest first, each as one merged
    (area_m2, GeoJSON mapping) pair. Shared by find_gaps and close_gaps:
    both need the same "one raster label can still rasterize to several
    shapes" merge (documented GDAL polygonize behaviour) - a single
    connected region must come back as one gap/closed-area, not one per
    piece.
    """
    from shapely.geometry import mapping, shape
    from shapely.ops import unary_union

    if not mask.any():
        return []

    labels, count = ndimage.label(mask)
    if not count:
        return []
    areas = ndimage.sum(mask, labels, range(1, count + 1)) * grid.cell_area_m2

    keep = [(index + 1, area) for index, area in enumerate(areas)
            if area >= min_area_m2]
    keep.sort(key=lambda pair: -pair[1])

    regions: list[tuple[float, dict]] = []
    for label, area in keep:
        pieces = [shape(geometry) for geometry, _ in rasterio.features.shapes(
            labels.astype("int32"), mask=(labels == label),
            transform=grid.transform)]
        regions.append((float(area), mapping(unary_union(pieces))))
    return regions


def find_gaps(grid: CoverageGrid, swath: np.ndarray,
              min_area_m2: float = MIN_GAP_M2) -> list[Gap]:
    """Unsearched areas inside the zone, largest first."""
    uncovered = grid.zone_mask & ~swath
    return [
        Gap(id=f"gap_{position}", area_m2=area, geometry=geometry)
        for position, (area, geometry) in enumerate(
            _labelled_regions(grid, uncovered, min_area_m2)
        )
    ]


# --- GPS-noise gap closing (Phase 9i), 2-D analogue of corridor.py -------
#
# corridor.py closes a line gap that is short and laterally consistent with
# drift; the raster equivalent of "short" is a hole narrow enough that
# scipy.ndimage.binary_closing (dilate then erode with the same structuring
# element) can bridge it. A closing with a disc of radius r bridges any gap
# up to 2r wide - a disc radius of HALF the sweep width therefore bridges
# gaps up to one full sweep width across: the scale of a single missed or
# jittered sweep lane, which is exactly the "thin sliver between two lanes"
# scenario this exists for. A genuinely unswept block (QC1202's real gap is
# ~26000 m2, over 100 m across in every direction) is far larger than a
# sweep-width-scale disc's reach and survives untouched - see
# test_closing_does_not_swallow_a_real_unswept_block.
GAP_CLOSE_RADIUS_FACTOR = 0.5


@dataclass(frozen=True)
class ClosedArea:
    """A raster sliver folded into "covered" by closing - the area analogue
    of corridor.ClosedGap. Not filtered by MIN_GAP_M2 (that floor exists to
    hide reporting noise in REMAINING gaps; a closed region is by
    construction no wider than the closing disc, so it needs no separate
    size floor of its own)."""
    id: str
    area_m2: float
    geometry: dict             # GeoJSON mapping, in the zone's coordinates


def _closing_radius_cells(sweep_width_ground_m: float, cell_ground_m: float) -> int:
    radius = (sweep_width_ground_m * GAP_CLOSE_RADIUS_FACTOR) / cell_ground_m
    return max(1, round(radius))


def _disc(radius_cells: int) -> np.ndarray:
    """A boolean disc structuring element - scipy.ndimage has no builtin
    disc (unlike skimage.morphology.disk, which is not a project
    dependency), so build one directly from a squared-distance grid."""
    offsets = np.arange(-radius_cells, radius_cells + 1)
    y, x = np.meshgrid(offsets, offsets, indexing="ij")
    return (x * x + y * y) <= radius_cells * radius_cells


def close_gaps(grid: CoverageGrid, swath: np.ndarray,
              sweep_width_ground_m: float) -> tuple[np.ndarray, list[ClosedArea]]:
    """Binary-close `swath` within the zone; returns (closed_swath, regions).

    `closed_swath` is always a superset of `swath` - closing only adds
    cells, never removes ones already covered. Mathematically,
    morphological closing is extensive (closing(A) >= A) on an infinite
    domain, but scipy's binary_closing operates on a finite array and
    defaults to treating everything outside it as background for the
    erosion half - that shrinks already-covered cells within one radius of
    the raster's own edge for no physical reason (a real gap between
    lanes, not a boundary effect). ORing the result back with the original
    `swath` restores the guarantee directly, regardless of border handling.
    """
    radius = _closing_radius_cells(sweep_width_ground_m, grid.cell_ground_m)
    structure = _disc(radius)
    closed = (ndimage.binary_closing(swath, structure=structure) | swath) & grid.zone_mask
    gained = closed & ~swath
    regions = [
        ClosedArea(id=f"closed_area_{position}", area_m2=area, geometry=geometry)
        for position, (area, geometry) in enumerate(
            _labelled_regions(grid, gained, min_area_m2=0.0)
        )
    ]
    return closed, regions
