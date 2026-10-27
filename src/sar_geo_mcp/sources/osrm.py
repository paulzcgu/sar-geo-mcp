"""OSRM: the source of road distance and drive time to each hospital.

OSRM's `table` service, car profile, over OpenStreetMap roads. The default
is the OSRM project's public demo, which allows light use with an
identifying User-Agent and nothing heavier; an agency running its own sets
`ZA_OSRM_URL`.
"""

from __future__ import annotations

import os
from typing import Any

from .base import JsonResponse, ServiceSource, SourceInfo, SourceUnavailable, http_get

#: OSRM's public demo server. Its usage policy allows light use with an
#: identifying User-Agent; an agency running its own sets `ZA_OSRM_URL`.
OSRM_URL = "https://router.project-osrm.org"
OSRM_URL_ENV = "ZA_OSRM_URL"


def osrm_url() -> str:
    return (os.environ.get(OSRM_URL_ENV, "").strip() or OSRM_URL).rstrip("/")


def osrm_table(origin: tuple[float, float], destinations: list[tuple[float, float]],
               *, get=None, url: str | None = None) -> dict[str, Any]:
    """Road distance and drive time from `origin` to each destination, one
    request. Points are (latitude, longitude).

    Returns `origin_snap_m` and `origin_snap` (where OSRM joined the road),
    and `legs`, one per destination: {distance_m, duration_s, snap_m}, with
    distance and duration None when OSRM found no route.
    """
    get = get or http_get
    points = [origin, *destinations]
    coordinates = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in points)
    request_url = f"{(url or osrm_url()).rstrip('/')}/table/v1/driving/{coordinates}"
    payload = get(request_url, {"sources": "0", "annotations": "distance,duration"}).json()
    if payload.get("code") != "Ok":
        raise RuntimeError(f"the road router answered {payload.get('code', 'no code')}")
    distances = (payload.get("distances") or [[]])[0]
    durations = (payload.get("durations") or [[]])[0]
    sources = payload.get("sources") or [{}]
    targets = payload.get("destinations") or []
    snapped = sources[0].get("location")
    legs = []
    for index in range(1, len(points)):
        target = targets[index] if index < len(targets) else {}
        legs.append({
            "distance_m": distances[index] if index < len(distances) else None,
            "duration_s": durations[index] if index < len(durations) else None,
            "snap_m": target.get("distance"),
        })
    return {
        "origin_snap_m": sources[0].get("distance"),
        "origin_snap": ({"latitude": snapped[1], "longitude": snapped[0]}
                        if snapped and len(snapped) == 2 else None),
        "legs": legs,
    }


class OsrmRoadSource(ServiceSource):
    """OSRM as a source: `call(origin=(lat, lon), destinations=[(lat, lon), ...])`
    gives road distance and free-flow car time from the origin to each."""

    info = SourceInfo(
        name="osrm",
        title="OSRM road router",
        kind="service",
        provider="Project OSRM, over OpenStreetMap roads",
        default_url=OSRM_URL,
        attribution="OSRM, © OpenStreetMap contributors",
        coverage="global, where OpenStreetMap has roads",
        url_env=OSRM_URL_ENV,
        shared_service=True,
        notes=(
            "Free-flow car time: no traffic, closures, or lights and sirens.",
            "Starts at the nearest routable road, which may be private or across a river.",
        ),
    )

    def call(self, *, origin, destinations, **params) -> dict[str, Any]:
        if params:
            raise ValueError(f"osrm takes origin and destinations; got {sorted(params)}")
        get = None
        if self.transport is not None:
            transport = self.transport
            get = lambda url, query=None: JsonResponse(transport.get_json(url, query))  # noqa: E731
        try:
            return osrm_table(tuple(origin), [tuple(point) for point in destinations],
                              get=get, url=self.endpoint())
        except Exception as error:  # noqa: BLE001 - any failure is the source's
            raise SourceUnavailable.from_error(self.info.name, error) from error
