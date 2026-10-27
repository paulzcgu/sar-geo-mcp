"""Trails, tracks, roads and waterways from the staged OSM cache.

The Overpass fetch happens at staging time (prefetch/fetch writes osm.json);
this detector only reads what was staged, so it works offline and never
hits the shared rate-limited service per question.
"""

from ..linear import LINEAR_ASSUMPTIONS
from .base import FeatureDetector, FeatureResult, FeatureShape
from .context import TerrainContext

MISSING_OSM_WARNING = "osm.json missing; trails and roads unknown"


class RouteDetector(FeatureDetector):
    kind = "route"
    description = ("Trails, tracks, roads and waterways from OpenStreetMap, "
                   "clipped to the staged extent.")
    assumptions = LINEAR_ASSUMPTIONS
    missing_input_warnings = (MISSING_OSM_WARNING,)

    def detect(self, ctx: TerrainContext) -> FeatureResult:
        payload = ctx.osm_payload
        if payload is None:
            return self.envelope(warnings=[MISSING_OSM_WARNING])
        shapes = [FeatureShape(**route) for route in payload.get("routes", [])]
        return self.envelope(shapes)
