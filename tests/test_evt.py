import pathlib

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from sar_geo_mcp.sources.landfire import (
    EVT_PIXEL_M,
    EVT_TILE_M,
    EVT_TILE_PX,
    build_tile_url,
    load_lifeform_table,
    tile_bounds,
    tiles_covering,
    write_evt_clip,
)
from sar_geo_mcp.terrain.evt import composition

# Assignment XI, and a neighbour ~400 m away — both inside one tile.
BBOX = (-13606608.2, 4490700.1, -13605941.3, 4491434.5)
NEIGHBOUR_BBOX = (-13606208.2, 4491100.1, -13605541.3, 4491834.5)


def fake_tile_bytes(code: int = 7015) -> bytes:
    """A valid in-memory GeoTIFF the size of one EVT tile."""
    from rasterio.io import MemoryFile

    codes = np.full((EVT_TILE_PX, EVT_TILE_PX), code, dtype="int16")
    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff", height=EVT_TILE_PX, width=EVT_TILE_PX, count=1,
            dtype="int16", crs="EPSG:3857",
            transform=from_origin(0, EVT_TILE_M, EVT_PIXEL_M, EVT_PIXEL_M),
        ) as dst:
            dst.write(codes, 1)
        return memfile.read()


def fake_tile_bytes_at(col: int, row: int, code: int) -> bytes:
    """A valid GeoTIFF for a SPECIFIC tile position, filled with one code."""
    from rasterio.io import MemoryFile

    left, bottom, right, top = tile_bounds(col, row)
    codes = np.full((EVT_TILE_PX, EVT_TILE_PX), code, dtype="int16")
    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff", height=EVT_TILE_PX, width=EVT_TILE_PX, count=1,
            dtype="int16", crs="EPSG:3857",
            transform=from_origin(left, top, EVT_PIXEL_M, EVT_PIXEL_M),
        ) as dst:
            dst.write(codes, 1)
        return memfile.read()


def test_clip_across_a_tile_seam_has_no_fill_pixels(tmp_path):
    """A bbox straddling two tiles must contain only real codes, never fill.

    rasterio.merge drops a column at the seam when bounds are not aligned to
    the 30 m pixel grid, leaving 0-valued pixels that composition() would
    report as a phantom 'Unknown' lifeform.
    """
    col, row = tiles_covering(BBOX)[0]
    codes_by_col = {col: 7015, col + 1: 4401}

    def fake_get(url: str) -> bytes:
        # Recover which tile was asked for from the bbox in the URL.
        from urllib.parse import parse_qs, urlparse

        left = float(parse_qs(urlparse(url).query)["bbox"][0].split(",")[0])
        requested_col = round(left / EVT_TILE_M)
        return fake_tile_bytes_at(requested_col, row, codes_by_col[requested_col])

    # Deliberately NOT aligned to the 30 m grid: 200 m either side of the seam.
    seam_x = (col + 1) * EVT_TILE_M
    bbox = (seam_x - 200.0, BBOX[1], seam_x + 200.0, BBOX[1] + 400.0)

    path = write_evt_clip(bbox, tmp_path / "evt.tif",
                          cache_dir=tmp_path / "cache", get_bytes=fake_get)

    with rasterio.open(path) as dataset:
        values = set(np.unique(dataset.read(1)).tolist())

    assert 0 not in values, f"fill pixels leaked into the clip: {sorted(values)}"
    assert values == {7015, 4401}, f"expected only the two tile codes, got {sorted(values)}"


def test_clip_across_a_row_seam_has_no_fill_pixels(tmp_path):
    """Same defect on a horizontal (row) boundary."""
    col, row = tiles_covering(BBOX)[0]
    codes_by_row = {row: 7015, row + 1: 4401}

    def fake_get(url: str) -> bytes:
        from urllib.parse import parse_qs, urlparse

        bottom = float(parse_qs(urlparse(url).query)["bbox"][0].split(",")[1])
        requested_row = round(bottom / EVT_TILE_M)
        return fake_tile_bytes_at(col, requested_row, codes_by_row[requested_row])

    seam_y = (row + 1) * EVT_TILE_M
    bbox = (BBOX[0], seam_y - 200.0, BBOX[0] + 400.0, seam_y + 200.0)

    path = write_evt_clip(bbox, tmp_path / "evt.tif",
                          cache_dir=tmp_path / "cache", get_bytes=fake_get)

    with rasterio.open(path) as dataset:
        values = set(np.unique(dataset.read(1)).tolist())

    assert 0 not in values, f"fill pixels leaked into the clip: {sorted(values)}"
    assert values == {7015, 4401}, f"expected only the two tile codes, got {sorted(values)}"


def test_lifeform_table_loads_and_decodes_known_codes():
    table = load_lifeform_table()

    assert len(table) > 1000
    lifeform, phys, name = table[7015]
    assert lifeform == "Tree"
    assert "Redwood" in name


def test_tile_grid_is_aligned_and_square():
    bounds = tile_bounds(-1772, 584)

    assert bounds[2] - bounds[0] == pytest.approx(EVT_TILE_M)
    assert bounds[3] - bounds[1] == pytest.approx(EVT_TILE_M)
    # Anchored at the Web Mercator origin, so every edge is a tile multiple.
    assert bounds[0] % EVT_TILE_M == pytest.approx(0.0)


def test_tiles_covering_a_small_bbox_is_one_tile():
    assert len(tiles_covering(BBOX)) == 1


def test_tiles_covering_a_corner_straddle_is_four():
    # Build a bbox centred exactly on a tile corner.
    col, row = tiles_covering(BBOX)[0]
    x = (col + 1) * EVT_TILE_M
    y = (row + 1) * EVT_TILE_M
    straddle = (x - 100.0, y - 100.0, x + 100.0, y + 100.0)

    assert len(tiles_covering(straddle)) == 4


def test_neighbouring_assignments_share_tile_keys():
    """The whole point: two assignments in the same area must not refetch."""
    assert tiles_covering(BBOX) == tiles_covering(NEIGHBOUR_BBOX)


def test_build_tile_url_requests_native_resolution_s16():
    col, row = tiles_covering(BBOX)[0]

    url = build_tile_url(col, row)

    assert "bboxSR=3857" in url
    assert "pixelType=S16" in url
    assert "format=tiff" in url
    assert f"size={EVT_TILE_PX}%2C{EVT_TILE_PX}" in url
    assert "interpolation=RSP_NearestNeighbor" in url


def test_second_assignment_in_the_same_area_downloads_nothing(tmp_path):
    """Two assignments, one shared tile, exactly one HTTP call."""
    calls = []

    def fake_get(url: str) -> bytes:
        calls.append(url)
        return fake_tile_bytes()

    first = write_evt_clip(BBOX, tmp_path / "a" / "evt.tif",
                           cache_dir=tmp_path / "cache", get_bytes=fake_get)
    second = write_evt_clip(NEIGHBOUR_BBOX, tmp_path / "b" / "evt.tif",
                            cache_dir=tmp_path / "cache", get_bytes=fake_get)

    assert len(calls) == 1, "the second assignment must reuse the cached tile"
    assert first.exists() and second.exists()
    assert first != second, "each assignment still gets its own clipped evt.tif"


def test_clip_covers_the_requested_bbox(tmp_path):
    def fake_get(url: str) -> bytes:
        return fake_tile_bytes()

    path = write_evt_clip(BBOX, tmp_path / "evt.tif",
                          cache_dir=tmp_path / "cache", get_bytes=fake_get)

    with rasterio.open(path) as dataset:
        bounds = dataset.bounds
        assert bounds.left <= BBOX[0] + EVT_PIXEL_M
        assert bounds.right >= BBOX[2] - EVT_PIXEL_M
        assert dataset.dtypes[0] == "int16"


def test_cache_files_are_named_by_tile_not_by_assignment(tmp_path):
    def fake_get(url: str) -> bytes:
        return fake_tile_bytes()

    cache = tmp_path / "cache"
    write_evt_clip(BBOX, tmp_path / "a.tif", cache_dir=cache, get_bytes=fake_get)
    write_evt_clip(NEIGHBOUR_BBOX, tmp_path / "b.tif", cache_dir=cache, get_bytes=fake_get)

    cached = list(cache.glob("evt_tile_*.tif"))
    assert len(cached) == 1
    col, row = tiles_covering(BBOX)[0]
    assert cached[0].name == f"evt_tile_c{col}_r{row}.tif"


def write_evt_tif(path, codes, transform):
    with rasterio.open(
        path, "w", driver="GTiff",
        height=codes.shape[0], width=codes.shape[1],
        count=1, dtype="int16", crs="EPSG:3857", transform=transform,
    ) as dst:
        dst.write(codes.astype("int16"), 1)


def test_composition_fractions_by_lifeform(tmp_path):
    # Look up a real Shrub code at test time rather than hardcoding one.
    table = load_lifeform_table()
    shrub_code = next(code for code, (lf, _, _) in table.items() if lf == "Shrub")

    codes = np.full((10, 10), 7015)
    codes[:, 6:] = shrub_code
    transform = from_origin(0, 300, 30, 30)
    tif = tmp_path / "evt.tif"
    write_evt_tif(tif, codes, transform)

    result = composition(tif, box(0, 0, 300, 300))

    assert result["fractions"]["Tree"] == pytest.approx(0.6, abs=0.05)
    assert result["fractions"]["Shrub"] == pytest.approx(0.4, abs=0.05)
    assert result["dominant_lifeform"] == "Tree"
    assert any("Redwood" in name for name, _ in result["top_types"])


def test_composition_handles_unknown_codes(tmp_path):
    codes = np.full((5, 5), -32768)
    transform = from_origin(0, 150, 30, 30)
    tif = tmp_path / "evt.tif"
    write_evt_tif(tif, codes, transform)

    result = composition(tif, box(0, 0, 150, 150))

    assert result["fractions"].get("Unknown", 0.0) == pytest.approx(1.0)
    assert result["dominant_lifeform"] == "Unknown"


def test_two_compositions_on_one_file_read_the_raster_once(tmp_path, monkeypatch):
    """A hasty set asks for composition once per corridor candidate, all
    against the SAME staged evt.tif; re-decoding the raster per candidate is
    the whole cost of the second one."""
    from sar_geo_mcp.terrain.evt import _read_evt

    codes = np.full((10, 10), 7015)
    transform = from_origin(0, 300, 30, 30)
    tif = tmp_path / "evt.tif"
    write_evt_tif(tif, codes, transform)

    _read_evt.cache_clear()
    opens = []
    real_open = rasterio.open

    def counting_open(*args, **kwargs):
        opens.append(args[0])
        return real_open(*args, **kwargs)

    monkeypatch.setattr(rasterio, "open", counting_open)

    first = composition(tif, box(0, 0, 300, 300))
    second = composition(tif, box(0, 0, 300, 300))

    assert [pathlib.Path(p) for p in opens] == [tif]
    assert first["fractions"] == second["fractions"]


def test_a_rewritten_evt_file_is_read_again(tmp_path):
    """The cache key carries the file's mtime and size, so re-staging the
    directory is never served a stale raster."""
    from sar_geo_mcp.terrain.evt import _read_evt

    table = load_lifeform_table()
    shrub_code = next(code for code, (lf, _, _) in table.items() if lf == "Shrub")
    transform = from_origin(0, 300, 30, 30)
    tif = tmp_path / "evt.tif"

    _read_evt.cache_clear()
    write_evt_tif(tif, np.full((10, 10), 7015), transform)
    assert composition(tif, box(0, 0, 300, 300))["dominant_lifeform"] == "Tree"

    write_evt_tif(tif, np.full((10, 20), shrub_code), transform)   # a different size
    assert composition(tif, box(0, 0, 300, 300))["dominant_lifeform"] == "Shrub"


def test_the_lifeform_table_is_read_from_disk_once():
    """5,000 CSV rows parsed once per process, not once per corridor."""
    assert load_lifeform_table() is load_lifeform_table()


def test_composition_reports_unknown_fraction(tmp_path):
    codes = np.full((10, 10), 7015)
    codes[:, 5:] = -32768  # half unknown
    transform = from_origin(0, 300, 30, 30)
    tif = tmp_path / "evt.tif"
    write_evt_tif(tif, codes, transform)

    result = composition(tif, box(0, 0, 300, 300))

    assert result["unknown_fraction"] == pytest.approx(0.5, abs=0.05)


@pytest.mark.live
def test_live_evt_tile_fetch_decodes_real_lifeforms(tmp_path):
    path = write_evt_clip(BBOX, tmp_path / "evt.tif", cache_dir=tmp_path / "cache")

    result = composition(path, box(*BBOX))

    # Assignment XI is redwood/oak forest: Tree must dominate.
    assert result["dominant_lifeform"] == "Tree"
    assert result["fractions"]["Tree"] > 0.5
