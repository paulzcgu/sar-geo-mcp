"""Every data source in `sources.MODULES`, held to the same contract.

A new source is checked here with no tests of its own for the contract:
its `info` is complete, its endpoint can be overridden, its key never
appears in what it says about itself or in a failure, and, against a fake
network, its one method returns the shape its kind promises. Each source
needs a `FAKE_CALLS` entry - a function that runs it once offline - and the
last test fails until it has one.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from sar_geo_mcp import sources
from sar_geo_mcp.sources import (
    Area,
    DataSource,
    FeatureResult,
    FeatureSource,
    RasterResult,
    RasterSource,
    ServiceSource,
    SourceUnavailable,
)

#: A made-up key. Long and random-looking on purpose, so a leak is found by
#: plain substring search.
FAKE_KEY = "zz-test-key-7f3a91c2e4b6d8f0a1c3e5b7d9f1a3c5"  # fixture-ok

KIND_CLASS = {"raster": RasterSource, "features": FeatureSource,
              "service": ServiceSource}

#: An area round a fixture point (tests/fixture_points.py).
AREA = Area.around(37.4244, -122.3151, 400.0)

class FakeTransport:
    """Answers every request from canned bodies, and records the URLs."""

    def __init__(self, *, body: bytes = b"", json_body=None):
        self.body, self.json_body, self.urls = body, json_body, []

    def get_bytes(self, url: str) -> bytes:
        self.urls.append(url)
        return self.body

    def get_json(self, url: str, params=None):
        self.urls.append(url)
        return self.json_body

    def post_json(self, url: str, data):
        self.urls.append(url)
        return self.json_body


def _usgs_3dep(tmp_path: Path):
    from test_dem import synthetic_tile_bytes

    from sar_geo_mcp.sources.dem import Usgs3depSource

    source = Usgs3depSource(tmp_path, transport=FakeTransport(body=synthetic_tile_bytes()))
    return source.fetch(AREA, 64, 64)


def _landfire_evt(tmp_path: Path):
    from test_evt import fake_tile_bytes

    from sar_geo_mcp.sources.landfire import LandfireEvtSource

    source = LandfireEvtSource(tmp_path, transport=FakeTransport(body=fake_tile_bytes()))
    return source.fetch(AREA)


#: name -> run the source once against a fake network; returns its result.
FAKE_CALLS: dict[str, Callable[[Path], object]] = {
    "usgs_3dep": _usgs_3dep,
    "landfire_evt": _landfire_evt,
}

CLASSES = sources.source_classes()


def test_area_round_a_point_holds_the_circle():
    area = Area.around(37.4244, -122.3151, 400.0)
    latitude, longitude = area.center_wgs84
    assert latitude == pytest.approx(37.4244, abs=1e-6)
    assert longitude == pytest.approx(-122.3151, abs=1e-6)
    west, south, east, north = area.bbox_3857
    # 400 m of ground is about 504 Web Mercator metres at this latitude.
    assert (east - west) / 2 == pytest.approx(400 / 0.7945, rel=0.01)
    assert (north - south) / 2 == pytest.approx(400 / 0.7945, rel=0.01)


def test_an_unavailable_source_says_the_status_not_the_url():
    request = httpx.Request("GET", f"https://service.example/x?key={FAKE_KEY}")
    error = httpx.HTTPStatusError("502 from the url", request=request,
                                  response=httpx.Response(502, request=request))
    failure = SourceUnavailable.from_error("some_source", error)
    assert str(failure).startswith("some_source unavailable: ")
    assert "502" in str(failure)
    assert FAKE_KEY not in str(failure) and "service.example" not in str(failure)


@pytest.mark.parametrize("name", CLASSES)
def test_info_is_complete(name):
    cls = CLASSES[name]
    info = cls.info
    assert info.name == name
    assert issubclass(cls, KIND_CLASS[info.kind]), (
        f"{name} says kind {info.kind!r} but is not a {KIND_CLASS[info.kind].__name__}")
    for field in ("title", "provider", "default_url", "attribution", "coverage"):
        assert getattr(info, field).strip(), f"{name}: {field} is empty"
    assert info.default_url.startswith("https://") or "{" in info.default_url, (
        f"{name}: default_url is not https")
    for variable in (info.url_env, info.key_env):
        assert variable is None or variable.isupper(), variable


@pytest.mark.parametrize("name", [n for n, c in CLASSES.items() if c.info.url_env])
def test_the_endpoint_can_be_overridden(name):
    cls = CLASSES[name]
    override = "https://override.example/service"
    assert cls(env={cls.info.url_env: override}).endpoint() == override
    assert cls(env={}).endpoint() == cls.info.default_url
    assert cls(env={cls.info.url_env: "  "}).endpoint() == cls.info.default_url


@pytest.mark.parametrize("name", CLASSES)
def test_describe_never_shows_a_key(name):
    cls = CLASSES[name]
    env = {cls.info.key_env: FAKE_KEY} if cls.info.key_env else {}
    described = cls(env=env).describe()
    assert FAKE_KEY not in json.dumps(described)
    assert described["ready"] is True
    if cls.info.key_env:
        missing = cls(env={}).describe()
        assert missing["ready"] is False
        assert missing["not_ready_because"] == f"needs {cls.info.key_env}"


@pytest.mark.parametrize("name", CLASSES)
def test_a_fake_call_returns_the_kinds_shape(name, tmp_path):
    if name not in FAKE_CALLS:
        pytest.fail(f"{name} has no FAKE_CALLS entry in tests/test_source_contract.py")
    result = FAKE_CALLS[name](tmp_path)
    kind = CLASSES[name].info.kind
    if kind == "raster":
        assert isinstance(result, RasterResult)
        assert result.path.exists() and result.width_px > 0 and result.height_px > 0
        assert result.attribution == CLASSES[name].info.attribution
    elif kind == "features":
        assert isinstance(result, FeatureResult)
        assert result.total >= len(result.features)
        assert all(feature.get("type") == "Feature" for feature in result.features)
        assert result.attribution == CLASSES[name].info.attribution
    else:
        assert isinstance(result, dict)


def test_every_registered_source_is_a_data_source():
    assert all(issubclass(cls, DataSource) for cls in CLASSES.values())
