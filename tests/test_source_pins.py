"""What each data source writes to the cache, and what it asks Overpass, pinned.

The data sources are being moved into their own infra package. A move must
not change a cached file's name, its index entry or the OSM query: any of
those would make the existing cache unreadable (a full re-download, much of
it from shared, rate-limited services) or quietly drop data from staging.
A deliberate change updates the pin here in the same commit, and says why.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sar_geo_mcp.cache import CacheIndex
from sar_geo_mcp.tiles import TILE_PX

from test_dem import BBOX, synthetic_tile_bytes
from test_evt import fake_tile_bytes
from test_wms_fetch import png_bytes


def cached_files(cache_dir: Path) -> list[str]:
    return sorted(path.relative_to(cache_dir).as_posix()
                  for path in cache_dir.rglob("*")
                  if path.is_file() and path.name != "index.sqlite")


def index_rows(cache_dir: Path) -> list[tuple[str, str, str | None]]:
    return sorted((entry.key, entry.layer, entry.source_version)
                  for entry in CacheIndex(cache_dir).entries())


def test_3dep_elevation_cache_names(tmp_path):
    from sar_geo_mcp.terrain.dem import fetch_dem

    fetch_dem(BBOX, 64, 64, cache_dir=tmp_path,
              get_bytes=lambda url: synthetic_tile_bytes())

    assert cached_files(tmp_path) == [
        "dem_553bc22a6bf38fe17f854815ff67ab16.tif",
        "tiles/dem/12/-348/114.tif",
    ]
    assert [(key, layer) for key, layer, _ in index_rows(tmp_path)] == [
        ("dem_553bc22a6bf38fe17f854815ff67ab16.tif", "dem-mosaic"),
        ("tiles/dem/12/-348/114.tif", "dem"),
    ]


def test_caltopo_wms_cache_names(tmp_path):
    from sar_geo_mcp.caltopo.wms import fetch_panel

    fetch_panel("https://example.com/wms", ["sf"], BBOX, 64, 64, tmp_path,
                get_bytes=lambda url: png_bytes(TILE_PX, TILE_PX, (255, 0, 0)))

    assert cached_files(tmp_path) == [
        "sf_1ef363bc6b6d6ad3f85bbbddebdd161a.png",
        "tiles/wms-sf/12/-348/114.png",
    ]
    assert [(key, layer) for key, layer, _ in index_rows(tmp_path)] == [
        ("sf_1ef363bc6b6d6ad3f85bbbddebdd161a.png", "wms-panel:sf"),
        ("tiles/wms-sf/12/-348/114.png", "wms:sf"),
    ]


def test_landfire_vegetation_cache_names(tmp_path):
    from sar_geo_mcp.terrain import evt

    evt.fetch_tile(-3404, 1120, tmp_path, get_bytes=lambda url: fake_tile_bytes())

    assert cached_files(tmp_path) == ["evt_tile_c-3404_r1120.tif"]
    assert [(key, layer) for key, layer, _ in index_rows(tmp_path)] == [
        ("evt_tile_c-3404_r1120.tif", "evt")]


#: sha256 of each tile's Overpass query, and of the four tiles batched. The
#: text holds the tile's bounds, so it is pinned by hash.
OSM_QUERY_SHA256 = {
    (-3405, 1120): "19b89b4066d5607efa31278a7080f90d519a5a501b75182b05e7fa8f9f80906f",
    (-3404, 1120): "6042278d0a32f1e053e937105f52f54f271eec116be71ef7a7ef4eb501b5830a",
    (-3405, 1121): "6420f587c62080710183e17292d8998343b9b6bb27c413f0bde6199b25895556",
    (-3404, 1121): "325bbd96200b44414cf5af684f7822a91f221babcd3d35a52049f0ddb0c44ec9",
}
OSM_BATCH_SHA256 = "bfdbf66c104dd71463fffc5d90dfea3fafa8f114e7d40e93d6f6df6a0e260d54"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_openstreetmap_query_is_unchanged():
    from sar_geo_mcp.terrain import linear

    tiles = linear.tiles_covering_osm(BBOX)
    assert tiles == list(OSM_QUERY_SHA256)
    assert linear.OSM_QUERY_VERSION == 5
    for tile in tiles:
        assert sha256(linear.build_overpass_query(*tile)) == OSM_QUERY_SHA256[tile], tile
    assert sha256(linear.build_batched_overpass_query(tiles)) == OSM_BATCH_SHA256


def test_openstreetmap_tile_cache_names(tmp_path):
    from sar_geo_mcp.terrain import linear

    tiles = linear.tiles_covering_osm(BBOX)

    def post(query: str) -> dict:
        return {"elements": [{"type": linear.OSM_TILE_MARKER,
                              "tags": {"col": str(col), "row": str(row)}}
                             for col, row in tiles]}

    linear.fetch_osm_tiles(tiles, tmp_path, post=post)

    names = [f"osm_tile_v5_c{col}_r{row}.json" for col, row in tiles]
    assert cached_files(tmp_path) == sorted(names)
    assert [(key, layer) for key, layer, _ in index_rows(tmp_path)] == [
        (name, "osm") for name in sorted(names)]
    for name in names:
        assert json.loads((tmp_path / name).read_text())["elements"] == []
