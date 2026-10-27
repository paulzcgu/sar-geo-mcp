"""US Census TIGERweb States: the source of whether a point is in the US.

HIFLD lists US hospitals only, so `nearest_medical_facilities` asks this
first and warns when the point is in another country or offshore.
"""

from __future__ import annotations

import functools
import os
from typing import Any

from .base import JsonResponse, ServiceSource, SourceInfo, SourceUnavailable, http_get

#: US Census TIGERweb "States" layer: the 50 states, DC, Puerto Rico and the
#: island areas, on land and inland water. Asked whether the point is inside
#: the US at all, because HIFLD lists US hospitals only. Verified 2026-10-21:
#: Vancouver, Windsor ON and 25 km off the California coast answer no
#: feature; Blaine WA, Guam, Puerto Rico, Hawaii and Alaska answer their code.
US_STATES_URL = ("https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
                 "State_County/MapServer/0")
US_STATES_URL_ENV = "ZA_US_STATES_URL"

#: Values the service may hold for "no code", read as none.
_UNKNOWN = {"", "NOT AVAILABLE", "N/A", "NA", "NONE", "UNKNOWN"}


def us_states_url() -> str:
    return (os.environ.get(US_STATES_URL_ENV, "").strip() or US_STATES_URL).rstrip("/")


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.upper() in _UNKNOWN else text


def point_state(latitude: float, longitude: float, *, get=None,
                url: str | None = None) -> str | None:
    """The US state, DC or territory the point is in, as its two-letter code
    ("CA", "PR", "GU"), or None when it is in none of them - another country,
    or offshore past the state's water."""
    get = get or functools.partial(http_get, timeout=10.0)
    params = {
        "geometry": f"{longitude},{latitude}", "geometryType": "esriGeometryPoint",
        "inSR": "4326", "spatialRel": "esriSpatialRelIntersects",
        "outFields": "STUSAB", "returnGeometry": "false", "f": "json",
    }
    payload = get(f"{(url or us_states_url()).rstrip('/')}/query", params).json()
    if "error" in payload:
        raise RuntimeError("the state boundary service refused the query: "
                           f"{(payload['error'] or {}).get('code', 'error')}")
    features = payload.get("features") or []
    if not features:
        return None
    return _clean((features[0].get("attributes") or {}).get("STUSAB")) or "US"


class TigerwebStatesSource(ServiceSource):
    """TIGERweb as a source: `call(latitude=..., longitude=...)` gives
    {"state": "CA"}, or {"state": None} for a point in no US state."""

    info = SourceInfo(
        name="tigerweb_states",
        title="US Census TIGERweb States",
        kind="service",
        provider="U.S. Census Bureau",
        default_url=US_STATES_URL,
        attribution="U.S. Census Bureau TIGERweb",
        coverage="US states, DC, Puerto Rico and island areas",
        url_env=US_STATES_URL_ENV,
        notes=("State polygons stop at the shoreline: a point offshore reads as outside.",),
    )

    def call(self, *, latitude: float, longitude: float, **params) -> dict[str, Any]:
        if params:
            raise ValueError(f"tigerweb_states takes latitude and longitude; got {sorted(params)}")
        get = None
        if self.transport is not None:
            transport = self.transport
            get = lambda url, query=None: JsonResponse(transport.get_json(url, query))  # noqa: E731
        try:
            return {"state": point_state(latitude, longitude, get=get, url=self.endpoint())}
        except Exception as error:  # noqa: BLE001 - any failure is the source's
            raise SourceUnavailable.from_error(self.info.name, error) from error
