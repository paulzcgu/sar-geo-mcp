import pytest

from sar_geo_mcp.caltopo.wms import (
    LAYERS,
    MAX_PIXELS,
    build_getmap_url,
    cache_key,
    validate_layers,
)

BASE = "https://caltopo.com/api/SECRET1/wms"
BBOX = (-13618826.5, 4481005.8, -13614373.7, 4486604.9)


def test_catalog_contains_the_layers_the_spec_relies_on():
    for layer in ["sf", "mbt", "imagery", "ir", "cm5", "mbr", "structures"]:
        assert layer in LAYERS


def test_build_url_has_required_wms_parameters():
    url = build_getmap_url(BASE, ["sf"], BBOX, 512, 512)

    assert url.startswith(BASE + "?")
    assert "SERVICE=WMS" in url
    assert "VERSION=1.1.1" in url
    assert "REQUEST=GetMap" in url
    assert "SRS=EPSG%3A3857" in url
    assert "FORMAT=image%2Fpng" in url
    assert "WIDTH=512" in url


def test_multiple_layers_are_comma_joined_for_server_side_compositing():
    url = build_getmap_url(BASE, ["mbt", "sf"], BBOX, 512, 512)

    assert "LAYERS=mbt%2Csf" in url


def test_rejects_unknown_layer():
    with pytest.raises(ValueError, match="unknown CalTopo layer"):
        build_getmap_url(BASE, ["nope"], BBOX, 512, 512)


def test_rejects_oversize_request():
    with pytest.raises(ValueError, match="1024"):
        build_getmap_url(BASE, ["sf"], BBOX, 2048, 512)


def test_max_pixels_matches_verified_server_limit():
    assert MAX_PIXELS == 1024


def test_validate_layers_names_the_offenders():
    with pytest.raises(ValueError, match="bogus"):
        validate_layers(["sf", "bogus"])


def test_cache_key_is_stable():
    assert cache_key(["sf"], BBOX, 512, 512) == cache_key(["sf"], BBOX, 512, 512)


def test_cache_key_varies_with_every_input():
    base = cache_key(["sf"], BBOX, 512, 512)

    assert cache_key(["mbt"], BBOX, 512, 512) != base
    assert cache_key(["sf"], BBOX, 512, 256) != base
    assert cache_key(["sf"], (0.0, 0.0, 1.0, 1.0), 512, 512) != base


def test_layer_order_changes_the_key_because_it_changes_the_render():
    assert cache_key(["mbt", "sf"], BBOX, 512, 512) != cache_key(["sf", "mbt"], BBOX, 512, 512)


def test_cache_key_never_contains_the_token():
    """Cache filenames are derived from this; the credential must not leak into them."""
    key = cache_key(["sf"], BBOX, 512, 512)

    assert "SECRET1" not in key
