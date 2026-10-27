"""Open patches inside the assignment: attractors and LZ candidates."""

from ..vegetation import VEGETATION_PROXY_THRESHOLD
from .base import FeatureDetector, FeatureResult, polygonize
from .context import TerrainContext

MISSING_PANEL_WARNING = "vegetation panel missing; clearings unknown"


class ClearingDetector(FeatureDetector):
    kind = "clearing"
    description = ("Open patches inside the assignment - potential "
                   "attractors and LZ candidates.")
    assumptions = {"min_area_m2": 300.0,
                   "proxy_threshold": VEGETATION_PROXY_THRESHOLD}
    missing_input_warnings = (MISSING_PANEL_WARNING,)

    def detect(self, ctx: TerrainContext) -> FeatureResult:
        proxy = ctx.vegetation_proxy
        if proxy is None:
            return self.envelope(warnings=[MISSING_PANEL_WARNING])
        grid = ctx.grid
        shapes = polygonize(
            (proxy <= self.assumptions["proxy_threshold"]) & grid.mask,
            grid.transform, grid.pixel_size_m,
            kind=self.kind, id_prefix="clearing",
            min_area_m2=self.assumptions["min_area_m2"],
        )
        return self.envelope(shapes)
