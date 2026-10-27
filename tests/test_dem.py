import numpy as np
import pytest

from sar_geo_mcp.terrain.dem import (
    MAX_DEM_PIXELS,
    TIFF_MAGIC_BE,
    TIFF_MAGIC_LE,
    build_export_url,
    ensure_tiff,
    fetch_dem,
)

BBOX = (-13618826.5, 4481005.8, -13614373.7, 4486604.9)


def test_build_url_requests_float32_tiff_in_web_mercator():
    url = build_export_url(BBOX, 512, 512)

    assert "bboxSR=3857" in url
    assert "imageSR=3857" in url
    assert "pixelType=F32" in url
    assert "format=tiff" in url
    assert "f=image" in url
    assert "size=512%2C512" in url


def test_build_url_rejects_oversize_request():
    with pytest.raises(ValueError, match="8000"):
        build_export_url(BBOX, 9000, 512)


def test_max_dem_pixels_matches_the_service_limit():
    assert MAX_DEM_PIXELS == 8000


def test_ensure_tiff_accepts_little_endian():
    ensure_tiff(TIFF_MAGIC_LE + b"rest of file", "http://example/dem")


def test_ensure_tiff_accepts_big_endian():
    ensure_tiff(TIFF_MAGIC_BE + b"rest of file", "http://example/dem")


def test_ensure_tiff_rejects_json_error_response():
    error = b'{"error":{"code":400,"message":"Unable to complete operation"}}'

    with pytest.raises(RuntimeError, match="did not return a GeoTIFF"):
        ensure_tiff(error, "http://example/dem")


def synthetic_tile_bytes(value: float = 100.0) -> bytes:
    """A real one-band float32 GeoTIFF, one grid tile across.

    `fetch_dem` opens what it downloads now that it mosaics tiles, so a
    stub returning TIFF magic plus padding is no longer enough - it has to
    be something rasterio can actually read.
    """
    from rasterio.io import MemoryFile
    from rasterio.transform import from_bounds

    from sar_geo_mcp.tiles import TILE_PX

    profile = {
        "driver": "GTiff", "dtype": "float32", "count": 1,
        "height": TILE_PX, "width": TILE_PX, "crs": "EPSG:3857",
        "transform": from_bounds(0, 0, 1, 1, TILE_PX, TILE_PX),
    }
    with MemoryFile() as memory:
        with memory.open(**profile) as target:
            target.write(np.full((TILE_PX, TILE_PX), value, dtype=np.float32), 1)
        return memory.read()


def test_fetch_dem_caches(tmp_path):
    calls = []

    def fake_get(url: str) -> bytes:
        calls.append(url)
        return synthetic_tile_bytes()

    first = fetch_dem(BBOX, 64, 64, cache_dir=tmp_path, get_bytes=fake_get)
    second = fetch_dem(BBOX, 64, 64, cache_dir=tmp_path, get_bytes=fake_get)

    assert first == second
    assert len(calls) == 1, "the second call must be served entirely from cache"


def test_two_overlapping_requests_share_their_tiles(tmp_path):
    """The whole point of F3, in one test.

    Before the shared grid these two hashed to different keys and
    downloaded the same ground twice - 17 DEMs and 62 MB on a real cache,
    for a handful of overlapping assignments.
    """
    calls = []

    def fake_get(url: str) -> bytes:
        calls.append(url)
        return synthetic_tile_bytes()

    fetch_dem(BBOX, 256, 256, cache_dir=tmp_path, get_bytes=fake_get)
    first_round = len(calls)

    span = BBOX[2] - BBOX[0]
    shifted = (BBOX[0] + span / 8, BBOX[1] + span / 8,
               BBOX[2] + span / 8, BBOX[3] + span / 8)
    fetch_dem(shifted, 256, 256, cache_dir=tmp_path, get_bytes=fake_get)

    assert len(calls) < first_round * 2, (
        "an overlapping request re-downloaded everything; the grid is not shared"
    )


def test_the_returned_raster_is_exactly_what_was_asked_for(tmp_path):
    """Tiling is an implementation detail. The contract - this bbox, this
    pixel size - has to survive it, because the manifest and every metric
    downstream are computed against the returned georeference."""
    import rasterio

    path = fetch_dem(BBOX, 300, 200, cache_dir=tmp_path,
                     get_bytes=lambda url: synthetic_tile_bytes(250.0))

    with rasterio.open(path) as raster:
        assert (raster.width, raster.height) == (300, 200)
        assert raster.bounds.left == pytest.approx(BBOX[0], abs=1.0)
        assert raster.bounds.top == pytest.approx(BBOX[3], abs=1.0)
        values = raster.read(1)

    assert float(np.nanmin(values)) == pytest.approx(250.0, abs=0.01), (
        "a constant surface must survive mosaic and resample unchanged"
    )


@pytest.mark.live
def test_live_dem_fetch_and_read(tmp_path):
    from sar_geo_mcp.terrain.dem import read_dem

    path = fetch_dem(BBOX, 512, 512, cache_dir=tmp_path)
    elevations, resolution_m = read_dem(path)

    assert elevations.shape == (512, 512)
    assert resolution_m > 0
    assert 0 < float(elevations.max()) < 5000
