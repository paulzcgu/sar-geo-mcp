"""Static satellite images, from providers that are not a WMS: Mapbox's
Static Images API and Google's Static Maps API. A landing-zone chip is cut
from them.

Each is asked for a centre, a zoom and a pixel size. The answer covers a box
this module computes from those three - Web Mercator is a fixed grid, so the
box is arithmetic, not a guess - and `lz.py` draws the patch outline with
that box's own transform, exactly as it does for a CalTopo chip.

KEYS ARE NEVER LOGGED OR RETURNED: they travel in the request URL, which is
why the package quiets httpx's request logging on import (sar_geo_mcp
__init__), why `get_static` raises nothing that carries the URL, and why a
failure's reason in `lz.py` is the exception type and status, never its
message (`net_errors.safe_reason`).

ATTRIBUTION TRAVELS WITH THE CHIP. Mapbox is asked with `attribution=false`
because its text credit would sit under the patch outline and go unread;
its terms then require the attribution elsewhere, and its logo stays on
the image (`logo` is not turned off). The text each provider requires is
`MAPBOX_ATTRIBUTION` and `GOOGLE_ATTRIBUTION`; `lz.py` puts it on every
chip it cuts, and a page that shows a chip shows it beside it. Google draws
its own logo and data credit on the image.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from urllib.parse import urlencode, urlsplit, urlunsplit

import httpx

from .base import Area, RasterResult, RasterSource, SourceInfo, SourceUnavailable

EARTH_CIRCUMFERENCE_M = 2 * math.pi * 6378137.0
MAPBOX_TILE_PX = 512
GOOGLE_TILE_PX = 256

#: The credit a page must show beside each provider's chip. Mapbox's
#: satellite style is Maxar imagery over OpenStreetMap-derived data.
MAPBOX_ATTRIBUTION = "© Mapbox © OpenStreetMap © Maxar"
GOOGLE_ATTRIBUTION = "Imagery © Google"

__all__ = ["EARTH_CIRCUMFERENCE_M", "GOOGLE_ATTRIBUTION", "GOOGLE_TILE_PX", "GetStatic",
           "GoogleSatelliteSource", "MAPBOX_ATTRIBUTION", "MAPBOX_TILE_PX",
           "MapboxSatelliteSource", "get_static", "google_request", "mapbox_request"]


def _units_per_px(zoom: float, tile_px: int) -> float:
    return EARTH_CIRCUMFERENCE_M / (tile_px * 2 ** zoom)


def _bbox(x: float, y: float, units_per_px: float, px: int) -> tuple[float, float, float, float]:
    half = units_per_px * px / 2.0
    return (x - half, y - half, x + half, y + half)


def _lonlat(x: float, y: float) -> tuple[float, float]:
    from pyproj import Transformer

    return Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform(x, y)


def mapbox_request(x: float, y: float, *, latitude: float, size_m: float, px: int,
                   token: str) -> tuple[str, tuple[float, float, float, float]]:
    """Mapbox takes a fractional zoom, so the box is `size_m` of ground to
    within the zoom's two decimals. Mapbox rounds the zoom to two decimals
    itself, so it is rounded here first and the URL and the box are both
    that zoom - the outline lands exactly. The logo stays on the image; the
    text attribution is off and travels as `MAPBOX_ATTRIBUTION` instead."""
    units = size_m / math.cos(math.radians(latitude)) / px
    zoom = round(math.log2(EARTH_CIRCUMFERENCE_M / (MAPBOX_TILE_PX * units)), 2)
    lon, lat = _lonlat(x, y)
    url = (f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/"
           f"{lon:.6f},{lat:.6f},{zoom:.2f},0/{px}x{px}?"
           + urlencode({"access_token": token, "attribution": "false"}))
    return url, _bbox(x, y, _units_per_px(zoom, MAPBOX_TILE_PX), px)


def google_request(x: float, y: float, *, latitude: float, size_m: float, px: int,
                   key: str) -> tuple[str, tuple[float, float, float, float]]:
    """Google takes only a whole zoom, so the box is the whole zoom nearest
    `size_m`, and the box returned says what it really covers."""
    units = size_m / math.cos(math.radians(latitude)) / px
    zoom = round(math.log2(EARTH_CIRCUMFERENCE_M / (GOOGLE_TILE_PX * units)))
    lon, lat = _lonlat(x, y)
    url = ("https://maps.googleapis.com/maps/api/staticmap?"
           + urlencode({"center": f"{lat:.6f},{lon:.6f}", "zoom": zoom,
                        "size": f"{px}x{px}", "maptype": "satellite", "key": key}))
    return url, _bbox(x, y, _units_per_px(zoom, GOOGLE_TILE_PX), px)


def _keyless_request(url: str) -> httpx.Request:
    """The request with its query string - where both providers take the
    key - dropped: scheme, host and path only."""
    parts = urlsplit(url)
    return httpx.Request("GET", urlunsplit((parts.scheme, parts.netloc, parts.path, "", "")))


def get_static(url: str) -> bytes:
    """One static image. Raises on a non-2xx or a body that is not an image.

    What it raises never carries `url`: httpx's own `HTTPStatusError` has
    the full URL in its message and on its request, so a refusal is raised
    afresh with the status alone and a keyless request, and a transport
    error is raised afresh as its own type with fixed text. Both are raised
    outside the `except` block, so no chained cause or context holds the
    original either. The caller turns either into a warning naming the
    source and the type and status (`net_errors.safe_reason`)."""
    status: int | None = None
    transport: type[Exception] | None = None
    try:
        response = httpx.get(url, timeout=30.0, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        status = error.response.status_code
    except (httpx.HTTPError, httpx.InvalidURL) as error:
        transport = type(error)
    if status is not None:
        request = _keyless_request(url)
        raise httpx.HTTPStatusError(f"the static image server answered {status}",
                                    request=request,
                                    response=httpx.Response(status, request=request))
    if transport is not None:
        raise transport("the static image request failed")
    if not response.content.startswith((b"\x89PNG", b"\xff\xd8")):
        raise ValueError("the answer was not an image")
    return response.content


GetStatic = Callable[[str], bytes]


#: The largest square either provider serves at scale 1.
MAX_STATIC_PX = 640


class _StaticSatelliteSource(RasterSource):
    """A keyed static image of an area's centre, sized to the area's width.
    The image is a PNG or JPEG covering `RasterResult.bbox_3857` - the box
    the provider really drew, which for Google is the nearest whole zoom.
    Cached by a hash of the request WITHOUT the key, so a key never reaches
    a file name."""

    _request: Callable

    def _cache_name(self, url: str) -> str:
        import hashlib

        keyless = str(_keyless_request(url).url) + "?" + "&".join(
            part for part in urlsplit(url).query.split("&")
            if not part.startswith(("access_token=", "key=")))
        return f"{self.info.name}_{hashlib.sha256(keyless.encode()).hexdigest()[:32]}"

    def fetch(self, area: Area, width_px: int = MAX_STATIC_PX,
              height_px: int | None = None, **options) -> RasterResult:
        from ..cache import atomic_write_bytes, note_hit, note_write
        from ..geo import ground_meters_per_mercator_unit

        if options:
            raise ValueError(f"{self.info.name} takes no options; got {sorted(options)}")
        key = self.key()
        if key is None:
            raise ValueError(f"{self.info.name} needs {self.info.key_env}")
        west, south, east, north = area.bbox_3857
        x, y = (west + east) / 2, (south + north) / 2
        size_m = max(east - west, north - south) * ground_meters_per_mercator_unit(area.latitude)
        px = max(1, min(int(width_px), MAX_STATIC_PX))
        url, bbox = self._build(x, y, latitude=area.latitude, size_m=size_m, px=px, key=key)
        cache_dir = self.require_cache()
        name = self._cache_name(url)
        existing = [path for path in cache_dir.glob(name + ".*")]
        if existing:
            note_hit(cache_dir, existing[0])
            path = existing[0]
        else:
            try:
                data = (self.transport.get_bytes(url) if self.transport else get_static(url))
            except Exception as error:  # noqa: BLE001 - reported by type and status only
                raise SourceUnavailable.from_error(self.info.name, error) from error
            path = cache_dir / (name + (".png" if data.startswith(b"\x89PNG") else ".jpg"))
            atomic_write_bytes(path, data)
            note_write(cache_dir, path, layer=self.info.name)
        return RasterResult(path=path, bbox_3857=bbox, width_px=px, height_px=px,
                            attribution=self.info.attribution)


class MapboxSatelliteSource(_StaticSatelliteSource):
    info = SourceInfo(
        name="mapbox_satellite",
        title="Mapbox Satellite",
        kind="raster",
        provider="Mapbox (Maxar imagery)",
        default_url="https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/",
        attribution=MAPBOX_ATTRIBUTION,
        coverage="global",
        key_env="ZA_MAPBOX_TOKEN",
        notes=("The Mapbox logo stays on the image; the text credit is the attribution.",),
    )

    def _build(self, x, y, *, latitude, size_m, px, key):
        return mapbox_request(x, y, latitude=latitude, size_m=size_m, px=px, token=key)


class GoogleSatelliteSource(_StaticSatelliteSource):
    info = SourceInfo(
        name="google_satellite",
        title="Google Satellite",
        kind="raster",
        provider="Google Maps Static API",
        default_url="https://maps.googleapis.com/maps/api/staticmap",
        attribution=GOOGLE_ATTRIBUTION,
        coverage="global",
        key_env="ZA_GOOGLE_MAPS_KEY",
        pullable=False,
        notes=("Google's terms restrict how static images may be used and shown: "
               "used for landing-zone chips only, never pulled on its own.",),
    )

    def _build(self, x, y, *, latitude, size_m, px, key):
        return google_request(x, y, latitude=latitude, size_m=size_m, px=px, key=key)
