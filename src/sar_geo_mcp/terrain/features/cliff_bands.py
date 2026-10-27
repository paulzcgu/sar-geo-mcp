"""Contiguous near-vertical ground: a hard stop for a ground team."""

import numpy as np

from ..landform import LANDFORM_ASSUMPTIONS
from .base import FeatureDetector, FeatureResult, polygonize
from .context import TerrainContext


class CliffBandDetector(FeatureDetector):
    kind = "cliff_band"
    description = ("Contiguous near-vertical ground - a hard stop for a "
                   "ground team, whatever the map says about the route.")
    assumptions = {
        "cliff_slope_deg": LANDFORM_ASSUMPTIONS["cliff_slope_deg"],
        "min_cliff_area_m2": LANDFORM_ASSUMPTIONS["min_cliff_area_m2"],
    }

    def detect(self, ctx: TerrainContext) -> FeatureResult:
        grid = ctx.grid
        threshold = self.assumptions["cliff_slope_deg"]
        slope_deg = np.nan_to_num(ctx.slope_aspect.slope_deg, nan=0.0)
        shapes = polygonize(
            (slope_deg >= threshold) & grid.mask,
            grid.transform, grid.pixel_size_m,
            kind=self.kind, id_prefix="cliff",
            min_area_m2=self.assumptions["min_cliff_area_m2"],
            # Pure addition: the band's own floor and ids are unchanged, but a
            # band that DOES fire now reports the drop across it. `cliff_pitch`
            # is the detector for the faces this floor discards.
            elevation=grid.elevation, slope_deg=slope_deg,
            extra_properties={"threshold_deg": threshold},
        )
        return self.envelope(shapes)
