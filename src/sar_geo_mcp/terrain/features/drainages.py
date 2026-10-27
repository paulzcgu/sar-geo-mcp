"""Ranked drainage channels with bank-top ravine depth.

The summary carries per-channel measurement dicts (id, lengths,
contributing area, depth, sidewall slope) that `metrics.DrainageMetrics`
revalidates — the detector produces plain dicts so this package's
dependency arrow keeps pointing features -> terrain math, never at
metrics.
"""

from ..hydro import (
    HYDRO_ASSUMPTIONS,
    extract_channels,
    measure_channel_depth,
    upstream_is_truncated,
)
from .base import DEM_VOID_WARNING, FeatureDetector, FeatureResult, FeatureShape
from .context import TerrainContext

NO_DRAINAGE_WARNING = (
    "no drainage above the contributing-area threshold intersects the "
    "assignment; the zone may have no channel, or the threshold may be "
    "too high for its size"
)
TRUNCATED_WARNING = (
    "a drainage enters through the DEM edge; contributing areas are "
    "understated and channel ranking may be incomplete"
)


class DrainageDetector(FeatureDetector):
    kind = "drainage"
    description = ("Drainage channels ranked by contributing area, with "
                   "bank-top ravine depth - the linear features that "
                   "channel travel or block it.")
    assumptions = HYDRO_ASSUMPTIONS

    def detect(self, ctx: TerrainContext) -> FeatureResult:
        grid = ctx.grid
        flow = ctx.flow
        min_area_m2 = HYDRO_ASSUMPTIONS["min_contributing_area_m2"]

        channels = extract_channels(
            flow.accumulation, flow.network, grid.transform,
            pixel_size_m=grid.pixel_size_m, min_area_m2=min_area_m2,
            clip_to=grid.geometry_3857,
        )

        shapes: list[FeatureShape] = []
        drainages: list[dict] = []
        for channel in channels:
            measurement = measure_channel_depth(
                channel, grid.elevation, grid.transform, grid.pixel_size_m
            )
            drainages.append({
                "id": channel["id"],
                "length_m": channel["properties"]["length_m"],
                "full_length_m": channel["properties"]["full_length_m"],
                "contributing_area_m2":
                    channel["properties"]["contributing_area_m2"],
                **measurement,
            })
            shapes.append(FeatureShape(**channel))

        warnings: list[str] = []
        if not drainages:
            warnings.append(NO_DRAINAGE_WARNING)
        truncated = upstream_is_truncated(
            flow.accumulation, grid.pixel_size_m, min_area_m2
        )
        if truncated:
            warnings.append(TRUNCATED_WARNING)
        if ctx.has_voids:
            warnings.append(DEM_VOID_WARNING)

        return self.envelope(
            shapes,
            summary={"drainages": drainages, "upstream_truncated": truncated},
            warnings=warnings,
        )
