"""Points where a route crosses the assignment boundary."""

from ..linear import ingress_candidates
from .base import FeatureDetector, FeatureResult, FeatureShape
from .context import TerrainContext
from .routes import MISSING_OSM_WARNING


class IngressDetector(FeatureDetector):
    kind = "ingress"
    description = ("Points where a route crosses the assignment boundary - "
                   "where a team can actually walk in (trails, tracks and "
                   "roads only).")
    assumptions = {"source": "OpenStreetMap via Overpass"}
    missing_input_warnings = (MISSING_OSM_WARNING,)

    def detect(self, ctx: TerrainContext) -> FeatureResult:
        payload = ctx.osm_payload
        if payload is None:
            return self.envelope(warnings=[MISSING_OSM_WARNING])
        points = ingress_candidates(
            payload.get("routes", []), ctx.grid.geometry_3857
        )
        return self.envelope([FeatureShape(**point) for point in points])
