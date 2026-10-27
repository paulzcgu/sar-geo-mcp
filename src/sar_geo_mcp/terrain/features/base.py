"""The feature contract: one envelope, one base class, one polygonizer.

Every detector is a `FeatureDetector` subclass returning a `FeatureResult`.
The four slots exist so that warnings and assumptions have a home from day
one: missing inputs degrade into warnings, never into silence, and every
threshold a detector used travels with its answer. A detector declares the
warnings it gives for a missing input (`missing_input_warnings`), because
its empty answer then means "unknown", not "none here", and a reader that
keeps the count but not the warning beside it - a scenario's summary - must
be able to tell the two apart without parsing prose.

`FeatureShape` carries the same fields as `metrics.MetricShape`, so the
debrief document absorbs registry shapes without translation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from pydantic import BaseModel

if TYPE_CHECKING:
    from .context import TerrainContext


DEM_VOID_WARNING = (
    "the elevation model has nodata cells; the flow routines filled them, so "
    "a channel or ridge may be pulled toward a hole rather than the ground"
)


class FeatureShape(BaseModel):
    """One identified shape, citable by id."""

    id: str
    kind: str
    geometry: dict
    properties: dict


class FeatureResult(BaseModel):
    """The envelope every detector returns.

    `summary` always carries at least `count`; kinds with measurements
    (drainages) put them here. Empty shapes on good input is a real answer
    and carries no warning; a missing input always warns, with a warning
    its detector declares in `missing_input_warnings`, and its `count` of 0
    is then unknown rather than none.
    """

    kind: str
    shapes: list[FeatureShape]
    summary: dict
    warnings: list[str]
    assumptions: dict


class FeatureDetector(ABC):
    """One feature kind. Subclasses declare the three class attributes and
    implement `detect`; `list_feature_kinds` is generated from them."""

    kind: ClassVar[str]
    description: ClassVar[str]
    assumptions: ClassVar[dict]
    #: The warnings `detect` gives when an input it reads is not staged (no
    #: vegetation panel, no osm.json): the answer is then empty because
    #: nothing could be looked at, not because nothing is there. Declared
    #: beside the detector that writes them, so `input_missing` reads the
    #: detector's own word rather than matching prose; a kind that needs
    #: nothing beyond the elevation model declares none. Not part of
    #: `list_kinds`: it describes an answer, not the kind.
    missing_input_warnings: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def detect(self, ctx: TerrainContext) -> FeatureResult:
        ...

    def envelope(self, shapes=None, summary=None, warnings=None) -> FeatureResult:
        """Assemble a result, filling kind, count, and assumptions."""
        shapes = list(shapes or [])
        merged: dict[str, Any] = {"count": len(shapes)}
        merged.update(summary or {})
        return FeatureResult(
            kind=self.kind, shapes=shapes, summary=merged,
            warnings=list(warnings or []), assumptions=dict(self.assumptions),
        )


def relief_within(geom, transform, pixel_size_m: float, *,
                  elevation: np.ndarray, slope_deg: np.ndarray | None = None) -> dict:
    """Vertical extent, contour span and slope stats inside one shape.

    WHY THIS IS NOT AREA
    --------------------
    Area is the wrong unit for anything a rope goes over. A vertical face has
    almost no planimetric footprint - a 20 m drop 15 m wide covers 210 m2 at
    55 degrees and 53 m2 at 80 - so a shape's area says more about how
    SHALLOW it is than how big. `vertical_extent_m` is the number a rigger
    reads: the drop from the top of the shape to the bottom.

    `span_m` is the longer side of the shape's bounding box in ground metres,
    a cheap proxy for how many stations fit side by side on the face.

    Measured on a window around the shape rather than the whole raster: a
    metre-scale sweep finds hundreds of these, and masking 16 megapixels per
    shape would dominate the run.
    """
    from rasterio.features import geometry_mask
    from rasterio.windows import from_bounds, transform as window_transform
    from shapely.geometry import shape as shapely_shape

    scale = pixel_size_m / abs(transform.a)      # mercator units -> ground m
    bounds = shapely_shape(geom).bounds
    rows, cols = elevation.shape
    window = from_bounds(*bounds, transform=transform).round_offsets().round_lengths()
    r0 = max(0, int(window.row_off))
    c0 = max(0, int(window.col_off))
    r1 = min(rows, r0 + max(1, int(window.height)) + 1)
    c1 = min(cols, c0 + max(1, int(window.width)) + 1)
    if r1 <= r0 or c1 <= c0:
        return {}

    block = elevation[r0:r1, c0:c1]
    sub_transform = window_transform(
        ((r0, r1), (c0, c1)), transform)
    mask = geometry_mask([geom], out_shape=block.shape, transform=sub_transform,
                         invert=True, all_touched=True)
    values = block[mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}

    facts: dict[str, Any] = {
        "vertical_extent_m": float(values.max() - values.min()),
        "elevation_low_m": float(values.min()),
        "elevation_high_m": float(values.max()),
        "span_m": float(max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * scale),
    }
    if slope_deg is not None:
        slopes = slope_deg[r0:r1, c0:c1][mask]
        slopes = slopes[np.isfinite(slopes)]
        if slopes.size:
            facts["slope_mean_deg"] = float(slopes.mean())
            facts["slope_p75_deg"] = float(np.percentile(slopes, 75))
            facts["slope_max_deg"] = float(slopes.max())
    return facts


def polygonize(binary: np.ndarray, transform, pixel_size_m: float, *,
               kind: str, id_prefix: str, min_area_m2: float,
               extra_properties: dict | None = None,
               elevation: np.ndarray | None = None,
               slope_deg: np.ndarray | None = None,
               min_vertical_extent_m: float = 0.0,
               rank_by_vertical_extent: bool = False) -> list[FeatureShape]:
    """Binary mask -> citable shapes.

    Vectorize with rasterio, correct areas from Web Mercator to ground
    square metres, drop fragments below `min_area_m2`, stamp sequential
    ids. The counter starts at zero per call, which is what keeps legacy
    ids like `steep_35_0` stable.

    Pass `elevation` (and optionally `slope_deg`) and every shape also
    carries `relief_within`'s facts - vertical extent, span, slope stats.
    That is a pure addition for the existing detectors.

    `min_vertical_extent_m` drops shapes that are steep but too short to rig
    on, and `rank_by_vertical_extent` re-sorts tallest first and numbers the
    ids in that order. Both are off by default so no existing caller's ids
    or ordering move.
    """
    from rasterio.features import shapes as rasterio_shapes
    from shapely.geometry import shape as shapely_shape

    if not binary.any():
        return []

    mercator_to_ground2 = (pixel_size_m / abs(transform.a)) ** 2
    collected: list[tuple[dict, dict]] = []
    for geom, value in rasterio_shapes(binary.astype("uint8"), transform=transform):
        if value != 1:
            continue
        area_m2 = shapely_shape(geom).area * mercator_to_ground2
        if area_m2 < min_area_m2:
            continue
        properties: dict[str, Any] = {"area_m2": area_m2}
        if elevation is not None:
            properties.update(relief_within(geom, transform, pixel_size_m,
                                            elevation=elevation, slope_deg=slope_deg))
        if properties.get("vertical_extent_m", float("inf")) < min_vertical_extent_m:
            continue
        properties.update(extra_properties or {})
        collected.append((dict(geom), properties))

    if rank_by_vertical_extent:
        collected.sort(key=lambda item: -item[1].get("vertical_extent_m", 0.0))

    return [
        FeatureShape(id=f"{id_prefix}_{index}", kind=kind,
                     geometry=geom, properties=properties)
        for index, (geom, properties) in enumerate(collected)
    ]
