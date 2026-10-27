"""Static satellite images for a landing-zone chip: the URL each provider is
asked, and the exact Web Mercator box the answer covers."""

import math

import httpx
import pytest

from sar_geo_mcp.planning.chips import (
    GOOGLE_TILE_PX, MAPBOX_TILE_PX, get_static, google_request, mapbox_request,
)

X, Y = -13607000.0, 4488000.0          # near Windy Hill, EPSG:3857
LAT = 37.36


def test_mapbox_is_asked_for_the_exact_size_at_a_fractional_zoom():
    url, bbox = mapbox_request(X, Y, latitude=LAT, size_m=200.0, px=400, token="tok")
    assert url.startswith("https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/")
    assert "access_token=tok" in url and "/400x400" in url
    # the text credit is off and travels in the envelope; the logo stays on
    assert "attribution=false" in url and "logo" not in url
    # two decimals, as Mapbox rounds it: the box below is that zoom's box
    zoom_text = url.split("/static/")[1].split(",")[2]
    assert len(zoom_text.split(".")[1]) == 2
    units_per_px = 2 * math.pi * 6378137.0 / (MAPBOX_TILE_PX * 2 ** float(zoom_text))
    assert bbox[2] - bbox[0] == pytest.approx(400 * units_per_px, rel=1e-9)
    west, south, east, north = bbox
    ground = (east - west) * math.cos(math.radians(LAT))
    assert ground == pytest.approx(200.0, rel=0.01)
    assert (west + east) / 2 == pytest.approx(X) and (south + north) / 2 == pytest.approx(Y)


def test_google_is_asked_at_a_whole_zoom_and_says_what_it_covers():
    url, bbox = google_request(X, Y, latitude=LAT, size_m=200.0, px=400, key="k")
    assert url.startswith("https://maps.googleapis.com/maps/api/staticmap?")
    assert "maptype=satellite" in url and "key=k" in url and "size=400x400" in url
    zoom = int(url.split("zoom=")[1].split("&")[0])
    units_per_px = 2 * math.pi * 6378137.0 / (GOOGLE_TILE_PX * 2 ** zoom)
    west, south, east, north = bbox
    assert east - west == pytest.approx(400 * units_per_px)
    ground = (east - west) * math.cos(math.radians(LAT))
    assert 100.0 <= ground <= 400.0          # the whole zoom nearest 200 m
    assert MAPBOX_TILE_PX == 512 and GOOGLE_TILE_PX == 256


SECRET_URL = "https://maps.googleapis.com/maps/api/staticmap?zoom=18&key=SECRET-KEY"


def _answering(monkeypatch, status, content=b""):
    def fake_get(url, **kwargs):
        return httpx.Response(status, content=content, request=httpx.Request("GET", url))
    monkeypatch.setattr(httpx, "get", fake_get)


def test_a_refused_static_image_names_the_status_and_never_the_url(monkeypatch):
    """httpx's own HTTPStatusError carries the URL - key and all - in its
    message and on its request; the one get_static raises carries neither,
    and chains nothing that would."""
    _answering(monkeypatch, 403)
    with pytest.raises(httpx.HTTPStatusError) as refused:
        get_static(SECRET_URL)
    error = refused.value
    assert error.response.status_code == 403
    assert "SECRET-KEY" not in str(error)
    assert "SECRET-KEY" not in str(error.request.url)
    assert "SECRET-KEY" not in str(error.response.request.url)
    assert error.__cause__ is None and error.__context__ is None


def test_a_transport_failure_keeps_its_type_and_loses_its_message(monkeypatch):
    def fake_get(url, **kwargs):
        raise httpx.ConnectError(f"could not reach {url}", request=httpx.Request("GET", url))
    monkeypatch.setattr(httpx, "get", fake_get)
    with pytest.raises(httpx.ConnectError) as failed:
        get_static(SECRET_URL)
    assert "SECRET-KEY" not in str(failed.value)
    assert failed.value.__cause__ is None and failed.value.__context__ is None


def test_an_answer_that_is_not_an_image_is_refused(monkeypatch):
    _answering(monkeypatch, 200, b"<html>quota exceeded</html>")
    with pytest.raises(ValueError, match="not an image") as refused:
        get_static(SECRET_URL)
    assert "SECRET-KEY" not in str(refused.value)


def test_an_image_comes_back_as_its_bytes(monkeypatch):
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
    _answering(monkeypatch, 200, png)
    assert get_static(SECRET_URL) == png
