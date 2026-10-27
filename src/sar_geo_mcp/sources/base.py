"""The base class every external data source is built on.

WHY A CLASS
-----------
Every service this package reads - an elevation model, a vegetation map,
OpenStreetMap, an agency inventory, a hospital list - needs the same
things: an endpoint a deployment can override, sometimes a key that must
never be shown, a cache, an attribution to show beside what it gave, and a
failure reported without leaking a URL or a credential. A source is a
subclass of one of three kinds, by the shape of what it gives:

- `RasterSource` - an area and a pixel size in, a georeferenced image out
  (elevation, vegetation, map panels, imagery);
- `FeatureSource` - an area and filters in, GeoJSON features out
  (OpenStreetMap, agency inventories, hospitals);
- `ServiceSource` - its own parameters in, JSON out (a road router, a
  package index), for services an area does not describe.

Adding a source is one module in `sources/`: subclass the kind, fill in
`info`, write the kind's one method, and list the module in
`sources.MODULES`. `tests/test_source_contract.py` then checks it like
every other source, with no tests of its own needed for the contract.

WHAT A SOURCE IS NOT
--------------------
Infra: a source returns the service's own data in the service's own
terms. It never says what the data means for a search - which way is
walkable, which hospital is a trauma center, whether a strip is a road.
That is domain logic, and it reads a source; a source never imports it.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol

from ..geo import Bbox, bbox_center_wgs84, buffer_bbox, to_web_mercator
from ..net_errors import safe_reason

Kind = Literal["raster", "features", "service"]


@dataclass(frozen=True)
class SourceInfo:
    """What a source is, said once. Everything here is safe to show anyone:
    `url_env` and `key_env` name variables, never their values."""

    name: str                 # registry key, e.g. "landfire_evt"
    title: str                # what a person calls it
    kind: Kind
    provider: str             # who publishes it
    default_url: str          # the endpoint when no override is set
    attribution: str          # shown beside anything drawn from it
    coverage: str             # "CONUS", "US", "global" ...
    url_env: str | None = None
    key_env: str | None = None   # None: no key needed
    native_resolution_m: float | None = None
    #: May Claude pull it directly through the source tools. False for a
    #: source whose terms restrict how its data may be used.
    pullable: bool = True
    #: A public, shared service with rate limits (Overpass, the OSRM demo):
    #: callers keep their requests few.
    shared_service: bool = False
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Area:
    """The ground a source is asked about: a box in Web Mercator metres.
    Every source takes the same area, so a tool builds it once."""

    bbox_3857: Bbox

    @classmethod
    def from_wgs84_bbox(cls, bbox_wgs84: Bbox) -> "Area":
        return cls(tuple(to_web_mercator(bbox_wgs84)))

    @classmethod
    def around(cls, latitude: float, longitude: float, radius_m: float) -> "Area":
        """The box holding a circle of `radius_m` ground metres."""
        point = to_web_mercator((longitude, latitude, longitude, latitude))
        return cls(tuple(buffer_bbox(point, radius_m, latitude)))

    @property
    def center_wgs84(self) -> tuple[float, float]:
        """(latitude, longitude), as `geo.bbox_center_wgs84` gives it."""
        return bbox_center_wgs84(self.bbox_3857)

    @property
    def latitude(self) -> float:
        return self.center_wgs84[0]


@dataclass
class RasterResult:
    path: Path
    bbox_3857: Bbox
    width_px: int
    height_px: int
    attribution: str
    crs: str = "EPSG:3857"
    warnings: list[str] = field(default_factory=list)


@dataclass
class FeatureResult:
    """GeoJSON features in WGS84. `total` counts every feature found;
    `truncated` says `features` holds fewer."""

    features: list[dict]
    total: int
    attribution: str
    truncated: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Readiness:
    ready: bool
    reason: str | None = None   # why not, naming the variable to set


class SourceUnavailable(RuntimeError):
    """A source could not answer. The message names the source and the
    failure's type and status (`net_errors.safe_reason`), never a URL, a
    response body or a key: it may reach a warning a person reads."""

    def __init__(self, source: str, reason: str):
        self.source = source
        self.reason = reason
        super().__init__(f"{source} unavailable: {reason}")

    @classmethod
    def from_error(cls, source: str, error: BaseException) -> "SourceUnavailable":
        return cls(source, safe_reason(error))


class Transport(Protocol):
    """What a source asks the network. A test passes a fake; production
    passes nothing, and each source keeps its own request policy (retries,
    timeouts, User-Agent), which is the service's to dictate, not ours."""

    def get_bytes(self, url: str) -> bytes: ...

    def get_json(self, url: str, params: Mapping[str, Any] | None = None) -> Any: ...

    def post_json(self, url: str, data: Mapping[str, Any]) -> Any: ...


class DataSource(ABC):
    """One external data source. See the module docstring."""

    info: ClassVar[SourceInfo]

    def __init__(self, cache_dir: Path | str | None = None, *,
                 env: Mapping[str, str] | None = None,
                 transport: Transport | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.env = os.environ if env is None else env
        self.transport = transport

    # -- configuration ---------------------------------------------------

    def endpoint(self) -> str:
        """The endpoint in use: the override when one is set, else the default."""
        if self.info.url_env:
            override = (self.env.get(self.info.url_env) or "").strip()
            if override:
                return override
        return self.info.default_url

    def key(self) -> str | None:
        """The key, for building a request. Never put it in a result,
        a warning or a log."""
        if not self.info.key_env:
            return None
        return (self.env.get(self.info.key_env) or "").strip() or None

    def readiness(self) -> Readiness:
        if self.info.key_env and self.key() is None:
            return Readiness(False, f"needs {self.info.key_env}")
        return Readiness(True)

    def describe(self) -> dict[str, Any]:
        """What the source is and whether it can be used now. Safe to show:
        names the variables, never their values, and the endpoint by its
        default only."""
        readiness = self.readiness()
        info = self.info
        return {
            "name": info.name, "title": info.title, "kind": info.kind,
            "provider": info.provider, "coverage": info.coverage,
            "attribution": info.attribution,
            "native_resolution_m": info.native_resolution_m,
            "key": info.key_env, "url_override": info.url_env,
            "endpoint_overridden": self.endpoint() != info.default_url,
            "ready": readiness.ready, "not_ready_because": readiness.reason,
            "pullable": info.pullable, "shared_service": info.shared_service,
            "notes": list(info.notes),
        }

    def require_cache(self) -> Path:
        if self.cache_dir is None:
            raise ValueError(f"{self.info.name} needs a cache directory")
        return self.cache_dir

    def folder(self) -> Path:
        """This source's own folder under the cache (`cache.layout`)."""
        from ..cache.layout import source_folder

        return source_folder(self.require_cache(), self.info.name)


class RasterSource(DataSource):
    @abstractmethod
    def fetch(self, area: Area, width_px: int, height_px: int,
              **options: Any) -> RasterResult:
        """A georeferenced image of `area` at this pixel size."""


class FeatureSource(DataSource):
    @abstractmethod
    def query(self, area: Area, *, filters: Mapping[str, Any] | None = None,
              limit: int | None = None) -> FeatureResult:
        """The source's features over `area`, GeoJSON in WGS84."""


class ServiceSource(DataSource):
    @abstractmethod
    def call(self, **params: Any) -> dict[str, Any]:
        """The service's answer to one request."""


#: How this package introduces itself to every public service it reads. The
#: OSRM demo and Overpass ask for an identifying User-Agent; the rest get
#: the same courtesy.
USER_AGENT = "sar-geo-mcp (search and rescue planning)"


def http_get(url: str, params: Mapping[str, Any] | None = None, *,
             timeout: float = 30.0):
    """A GET behind the shared retry policy (`net.get_with_retry`: three
    attempts, one second's backoff doubling) with the package's User-Agent.
    Returns the response; `.json()` reads it."""
    import functools

    import httpx

    from ..net import get_with_retry

    request = httpx.Request("GET", url, params=params)
    return get_with_retry(
        str(request.url), attempts=3, base_delay=1.0, timeout=timeout,
        get=functools.partial(httpx.get, headers={"User-Agent": USER_AGENT}))


class JsonResponse:
    """What a fetcher reads off a response (`.json()`), from a transport's
    already-parsed answer: lets a test transport stand in for `http_get`."""

    def __init__(self, payload: Any):
        self._payload = payload

    def json(self) -> Any:
        return self._payload
