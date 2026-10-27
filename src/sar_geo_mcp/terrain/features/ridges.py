"""Ridgelines from flow accumulation on the negated elevation.

The anatomy report found 14% of boundary length on ridgelines computed this
way, and every one of them was read off the contours by a planner, because
no base map draws a ridge. A ridge has no banks, so there is no depth
measurement: length, full traced length and inverted contributing area (a
proxy for prominence) are the properties.
"""

from ..hydro import HYDRO_ASSUMPTIONS, extract_channels, upstream_is_truncated
from .base import DEM_VOID_WARNING, FeatureDetector, FeatureResult, FeatureShape
from .context import TerrainContext

NO_RIDGE_WARNING = (
    "no ridgeline above the contributing-area threshold intersects the "
    "subject; the ground may be a single slope, or the threshold too high "
    "for its size"
)
TRUNCATED_WARNING = (
    "a ridgeline reaches the DEM edge; its contributing area is understated "
    "and ridge ranking may be incomplete"
)

RIDGE_ASSUMPTIONS = {
    "method": "D8 flow accumulation on the negated elevation",
    "min_contributing_area_m2": HYDRO_ASSUMPTIONS["min_contributing_area_m2"],
    "min_channel_length_m": HYDRO_ASSUMPTIONS["min_channel_length_m"],
    "note": ("the anatomy corpus used a 2 ha threshold; this detector keeps "
             "the drainage detector's finer one"),
}


class RidgeDetector(FeatureDetector):
    kind = "ridge"
    description = ("Ridgelines ranked by inverted contributing area - the "
                   "crests planners read off the contours to bound a block "
                   "or run a thin corridor along.")
    assumptions = RIDGE_ASSUMPTIONS

    def detect(self, ctx: TerrainContext) -> FeatureResult:
        grid = ctx.grid
        flow = ctx.inverted_flow
        min_area_m2 = HYDRO_ASSUMPTIONS["min_contributing_area_m2"]

        ridges = extract_channels(
            flow.accumulation, flow.network, grid.transform,
            pixel_size_m=grid.pixel_size_m, min_area_m2=min_area_m2,
            clip_to=grid.geometry_3857, kind="ridge", id_prefix="ridge",
        )
        shapes = [FeatureShape(**ridge) for ridge in ridges]

        warnings: list[str] = []
        if not shapes:
            warnings.append(NO_RIDGE_WARNING)
        truncated = upstream_is_truncated(flow.accumulation, grid.pixel_size_m, min_area_m2)
        if truncated:
            warnings.append(TRUNCATED_WARNING)
        if ctx.has_voids:
            warnings.append(DEM_VOID_WARNING)
        return self.envelope(shapes, summary={"truncated": truncated}, warnings=warnings)
