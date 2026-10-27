"""HIFLD Hospitals: the source of every hospital `nearest_medical_facilities`
ranks.

The HIFLD "Hospitals" layer (ORNL / DHS), as kept on ArcGIS Online by FEMA.
HIFLD Open, the original home, was retired in 2025; this copy carries the
same schema and validation dates into 2025. It is the only free national
list carrying trauma level, helipad, phone and address together. Override
with `ZA_HOSPITALS_URL`.

This module returns the service's records as it gives them. Reading the
trauma field, merging one campus listed twice and correcting a children's
hospital listed as adult trauma are `planning.medical`'s.
"""

from __future__ import annotations

import math
import os

from .base import (
    Area,
    FeatureResult,
    FeatureSource,
    JsonResponse,
    SourceInfo,
    SourceUnavailable,
    http_get,
)

#: The HIFLD Hospitals layer, FEMA's maintained copy. Verified 2026-10-20:
#: 8,514 records, fields TRAUMA, HELIPAD, TELEPHONE, ADDRESS, VAL_DATE.
HOSPITALS_URL = ("https://services.arcgis.com/XG15cJAlne2vxtgt/arcgis/rest/"
                 "services/Hospitals_RAPT/FeatureServer/6")
HOSPITALS_URL_ENV = "ZA_HOSPITALS_URL"

#: Asked of HIFLD; nothing more, to keep the answer small.
HOSPITAL_FIELDS = ("ID,NAME,ADDRESS,CITY,STATE,ZIP,TELEPHONE,TYPE,STATUS,"
                   "TRAUMA,HELIPAD,LATITUDE,LONGITUDE,VAL_DATE")


def hospitals_url() -> str:
    return os.environ.get(HOSPITALS_URL_ENV, "").strip() or HOSPITALS_URL


def query_hospitals(latitude: float, longitude: float, radius_km: float,
                    where: str = "STATUS = 'OPEN'", *, get=None,
                    page_size: int = 2000, max_pages: int = 10,
                    url: str | None = None) -> list[dict]:
    """Open HIFLD hospitals within `radius_km` of the point, as raw features.

    The service buffers in its own Web Mercator, which shrinks a ground
    distance by cos(latitude), so the search distance is inflated by
    1/cos(latitude) here and the caller trims by true ground distance.
    """
    get = get or http_get
    base = url or hospitals_url()
    mercator_km = radius_km / max(math.cos(math.radians(latitude)), 0.05)
    features: list[dict] = []
    for page in range(max_pages):
        params = {
            "where": where, "geometry": f"{longitude},{latitude}",
            "geometryType": "esriGeometryPoint", "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "distance": f"{mercator_km:.3f}", "units": "esriSRUnit_Kilometer",
            "outFields": HOSPITAL_FIELDS, "returnGeometry": "true", "outSR": "4326",
            "resultOffset": str(page * page_size), "resultRecordCount": str(page_size),
            "orderByFields": "ID", "f": "json",
        }
        payload = get(f"{base}/query", params).json()
        if "error" in payload:
            raise RuntimeError("the hospital list refused the query: "
                               f"{(payload['error'] or {}).get('code', 'error')}")
        features.extend(payload.get("features") or [])
        if not payload.get("exceededTransferLimit"):
            break
    return features


class HifldHospitalsSource(FeatureSource):
    """HIFLD hospitals as a source: the open hospitals within an area's
    circle (centre to corner), as GeoJSON points carrying HIFLD's own
    attributes. `filters={"where": "<ArcGIS where clause>"}` replaces the
    default `STATUS = 'OPEN'`."""

    info = SourceInfo(
        name="hifld_hospitals",
        title="HIFLD Hospitals",
        kind="features",
        provider="ORNL / DHS HIFLD, kept on ArcGIS Online by FEMA",
        default_url=HOSPITALS_URL,
        attribution="HIFLD Hospitals (ORNL, DHS; FEMA)",
        coverage="US states, DC and territories",
        url_env=HOSPITALS_URL_ENV,
        notes=(
            "Designations are as last validated (VAL_DATE); some are a decade old.",
            "Nothing here says whether an emergency department is open or on diversion.",
        ),
    )

    def query(self, area: Area, *, filters=None, limit: int | None = None) -> FeatureResult:
        from ..geo import ground_meters_per_mercator_unit

        filters = dict(filters or {})
        where = filters.pop("where", "STATUS = 'OPEN'")
        if filters:
            raise ValueError(f"hifld_hospitals filters by where only; got {sorted(filters)}")
        latitude, longitude = area.center_wgs84
        west, south, east, north = area.bbox_3857
        half_diagonal = math.hypot(east - west, north - south) / 2
        radius_km = half_diagonal * ground_meters_per_mercator_unit(latitude) / 1000
        get = None
        if self.transport is not None:
            transport = self.transport
            get = lambda url, params=None: JsonResponse(transport.get_json(url, params))  # noqa: E731
        try:
            records = query_hospitals(latitude, longitude, radius_km, where, get=get,
                                      url=self.endpoint())
        except Exception as error:  # noqa: BLE001 - any failure is the source's
            raise SourceUnavailable.from_error(self.info.name, error) from error
        features = []
        for record in records:
            geometry = record.get("geometry") or {}
            if geometry.get("x") is None or geometry.get("y") is None:
                continue
            features.append({"type": "Feature",
                             "geometry": {"type": "Point",
                                          "coordinates": [geometry["x"], geometry["y"]]},
                             "properties": dict(record.get("attributes") or {})})
        cap = len(features) if limit is None else limit
        return FeatureResult(features=features[:cap], total=len(features),
                             attribution=self.info.attribution,
                             truncated=len(features) > cap)

