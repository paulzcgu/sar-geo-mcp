"""Contiguous steep ground, banded by threshold."""

import numpy as np

from ..movement import IMPASSABLE_SLOPE_DEG

from .base import FeatureDetector, FeatureResult, polygonize
from .context import TerrainContext


class SteepBandDetector(FeatureDetector):
    kind = "steep_band"
    description = ("Contiguous ground steeper than 35 or 45 degrees - the "
                   "bands that slow a ground team or stop it.")
    #: The upper band is where a ground team stops - the slope the walking
    #: tools close (`terrain.movement.IMPASSABLE_SLOPE_DEG`), read from there.
    assumptions = {"thresholds_deg": (35.0, IMPASSABLE_SLOPE_DEG), "min_area_m2": 500.0}

    def detect(self, ctx: TerrainContext) -> FeatureResult:
        grid = ctx.grid
        slope_deg = np.nan_to_num(ctx.slope_aspect.slope_deg, nan=0.0)
        shapes = []
        for threshold in self.assumptions["thresholds_deg"]:
            shapes += polygonize(
                (slope_deg >= threshold) & grid.mask,
                grid.transform, grid.pixel_size_m,
                kind=self.kind, id_prefix=f"steep_{threshold:g}",
                min_area_m2=self.assumptions["min_area_m2"],
                elevation=grid.elevation, slope_deg=slope_deg,
                extra_properties={"threshold_deg": threshold},
            )
        return self.envelope(shapes)
