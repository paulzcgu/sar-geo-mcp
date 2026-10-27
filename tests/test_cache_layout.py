"""A cache from before 0.43.0 is moved into the per-source folders, never
fetched again, and the move is safe to repeat or race."""

from __future__ import annotations

import pytest

from sar_geo_mcp.cache import CacheIndex
from sar_geo_mcp.cache import layout
from sar_geo_mcp.tiles import TILE_PX

from test_dem import BBOX, synthetic_tile_bytes
from test_wms_fetch import png_bytes

PANEL_HASH = "1ef363bc6b6d6ad3f85bbbddebdd161a"
DEM_HASH = "553bc22a6bf38fe17f854815ff67ab16"


@pytest.fixture(autouse=True)
def _fresh_process(monkeypatch):
    """Each test is a new process as far as the once-per-process memo goes."""
    monkeypatch.setattr(layout, "_done", set())


def write(cache, key, data=b"x" * 10, layer="test"):
    path = cache / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    CacheIndex(cache).record(path, layer=layer)
    return path


def old_cache(cache):
    """Every source's files where 0.42 wrote them, with index rows, plus what
    is not a source's: staged ground, map reads, an unrelated file."""
    write(cache, "tiles/dem/12/-348/114.tif", synthetic_tile_bytes(), "dem")
    write(cache, f"dem_{DEM_HASH}.tif", synthetic_tile_bytes(), "dem-mosaic")
    write(cache, "tiles/wms-sf/12/-348/114.png", png_bytes(TILE_PX, TILE_PX, (255, 0, 0)), "wms:sf")
    write(cache, f"sf_{PANEL_HASH}.png", b"png", "wms-panel:sf")
    write(cache, "evt_tile_c-3404_r1120.tif", b"tif", "evt")
    write(cache, "osm_tile_v5_c-3405_r1120.json", b'{"elements": []}', "osm")
    write(cache, "osm_tile_v4_c-3405_r1120.json", b'{"elements": []}', "osm")
    write(cache, "staged/aoi-0000/manifest.json", b"{}", "staged")
    write(cache, "caltopo-state/ABC1234.json", b"{}", "caltopo-state")
    write(cache, "notes.png", b"png", "other")


def keys(cache):
    return {entry.key for entry in CacheIndex(cache).entries()}


def migrate_all(cache):
    from sar_geo_mcp.caltopo import wms
    from sar_geo_mcp.sources import dem, landfire, osm

    for module in (dem, landfire, osm, wms):
        layout.migrate(cache, module.FOLDER, module.LEGACY_CACHE)


def test_every_source_s_files_move_into_its_folder(tmp_path):
    old_cache(tmp_path)
    migrate_all(tmp_path)

    assert keys(tmp_path) == {
        "usgs_3dep/tiles/dem/12/-348/114.tif",
        f"usgs_3dep/dem_{DEM_HASH}.tif",
        "caltopo_wms/tiles/wms-sf/12/-348/114.png",
        f"caltopo_wms/sf_{PANEL_HASH}.png",
        "landfire_evt/evt_tile_c-3404_r1120.tif",
        "osm/osm_tile_v5_c-3405_r1120.json",
        "osm/osm_tile_v4_c-3405_r1120.json",
        "staged/aoi-0000/manifest.json",
        "caltopo-state/ABC1234.json",
        "notes.png",
    }
    for key in keys(tmp_path):
        assert (tmp_path / key).is_file(), key
    assert not (tmp_path / "tiles").exists()


def test_a_moved_tile_is_served_from_the_cache_not_fetched(tmp_path):
    from sar_geo_mcp.caltopo.wms import fetch_panel
    from sar_geo_mcp.sources.dem import fetch_dem

    old_cache(tmp_path)

    def no_network(url):
        raise AssertionError("a moved file was fetched again")

    fetch_dem(BBOX, 64, 64, cache_dir=tmp_path, get_bytes=no_network)
    fetch_panel("https://example.com/wms", ["sf"], BBOX, 64, 64, tmp_path,
                get_bytes=no_network)
    assert f"usgs_3dep/dem_{DEM_HASH}.tif" in keys(tmp_path)


def test_migrating_again_changes_nothing(tmp_path):
    old_cache(tmp_path)
    migrate_all(tmp_path)
    before = keys(tmp_path)
    files = sorted(path for path in tmp_path.rglob("*") if path.is_file())
    migrate_all(tmp_path)
    assert keys(tmp_path) == before
    assert sorted(path for path in tmp_path.rglob("*") if path.is_file()) == files


def test_a_file_already_in_its_folder_wins_and_the_old_copy_goes(tmp_path):
    write(tmp_path, "evt_tile_c1_r1.tif", b"old", "evt")
    write(tmp_path, "landfire_evt/evt_tile_c1_r1.tif", b"new", "evt")
    from sar_geo_mcp.sources import landfire

    layout.migrate(tmp_path, landfire.FOLDER, landfire.LEGACY_CACHE)
    assert (tmp_path / "landfire_evt/evt_tile_c1_r1.tif").read_bytes() == b"new"
    assert not (tmp_path / "evt_tile_c1_r1.tif").exists()
    assert keys(tmp_path) == {"landfire_evt/evt_tile_c1_r1.tif"}


def test_tiles_written_later_by_an_older_release_are_merged_in(tmp_path):
    """A pinned 0.42 writes at the old paths again after 0.43 moved them;
    the next 0.43 process merges its tiles into the folder already there."""
    from sar_geo_mcp.sources import dem

    write(tmp_path, "tiles/dem/12/1/1.tif", b"a", "dem")
    layout.migrate(tmp_path, dem.FOLDER, dem.LEGACY_CACHE)
    write(tmp_path, "tiles/dem/12/1/2.tif", b"b", "dem")
    layout.migrate(tmp_path, dem.FOLDER, dem.LEGACY_CACHE)
    assert keys(tmp_path) == {"usgs_3dep/tiles/dem/12/1/1.tif",
                              "usgs_3dep/tiles/dem/12/1/2.tif"}
    assert (tmp_path / "usgs_3dep/tiles/dem/12/1/2.tif").read_bytes() == b"b"


def test_a_file_that_cannot_be_moved_is_left_not_lost(tmp_path, monkeypatch):
    from sar_geo_mcp.sources import osm

    write(tmp_path, "osm_tile_v5_c1_r1.json", b"{}", "osm")

    def refuse(old, new):
        raise PermissionError("held open by another process")

    monkeypatch.setattr(layout.os, "replace", refuse)
    monkeypatch.setattr(layout.shutil, "copy2", refuse)
    layout.migrate(tmp_path, osm.FOLDER, osm.LEGACY_CACHE)
    assert (tmp_path / "osm_tile_v5_c1_r1.json").is_file()


def test_no_index_is_no_problem(tmp_path):
    (tmp_path / "evt_tile_c1_r1.tif").write_bytes(b"x")
    from sar_geo_mcp.sources import landfire

    layout.migrate(tmp_path, landfire.FOLDER, landfire.LEGACY_CACHE)
    assert (tmp_path / "landfire_evt/evt_tile_c1_r1.tif").is_file()


def test_the_wms_source_and_its_transport_name_one_folder():
    from sar_geo_mcp.caltopo import wms
    from sar_geo_mcp.sources import dem, landfire, osm
    from sar_geo_mcp.sources.caltopo_wms import CaltopoWmsSource
    from sar_geo_mcp.sources.dem import Usgs3depSource
    from sar_geo_mcp.sources.landfire import LandfireEvtSource
    from sar_geo_mcp.sources.osm import OsmSource

    assert wms.FOLDER == CaltopoWmsSource.info.name
    assert dem.FOLDER == Usgs3depSource.info.name
    assert landfire.FOLDER == LandfireEvtSource.info.name
    assert osm.FOLDER == OsmSource.info.name
