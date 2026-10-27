"""Cache eviction (F6).

The two properties that matter: it never deletes ground a staged
assignment still rests on, and it never deletes a file it knows nothing
about. Both failures look the same from the field - a laptop that was
provisioned and now is not - and neither would surface until somebody was
already out of signal.

Since 0.27.1 the first property is a collaboration: `staging.tiles_in_use`
says which tiles staged ground rests on, the cache holds whatever keys it
is handed as `protected`, and the `evict_cache` tool wires the two. Tests
that stage something pass `protected=tiles_in_use(cache)`, exactly as the
tool does; tests that stage nothing pass `protected=()`.
"""

from __future__ import annotations

import inspect
import json
import sys
import time
from pathlib import Path

import pytest
from test_dem import synthetic_tile_bytes

from sar_geo_mcp.cache import CacheIndex
from sar_geo_mcp.cache.evict import evict, plan_eviction
from sar_geo_mcp.staging import panel_dimensions, tiles_in_use
from sar_geo_mcp.tiles import level_for, tiles_covering

BBOX = (-13617151.3, 4494360.9, -13615236.6, 4496264.4)


@pytest.fixture
def cache(tmp_path):
    return tmp_path / "cache"


def write_tile(cache: Path, key: str, size: int, index: CacheIndex,
               last_used: float | None = None) -> Path:
    path = cache / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    index.record(path, layer="dem", tier="static")
    if last_used is not None:
        with index._connect() as connection:
            connection.execute("UPDATE entries SET last_used = ? WHERE key = ?",
                               (last_used, key))
    return path


def stage(cache: Path, name: str, bbox=BBOX) -> None:
    directory = cache / "staged" / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(
        json.dumps({"bbox_3857": list(bbox), "map_id": "M1",
                    "assignment": name}), encoding="utf-8")


# ---------------------------------------------------------------------------
# Ranking.
# ---------------------------------------------------------------------------

def test_nothing_is_evicted_below_the_ceiling(cache):
    index = CacheIndex(cache)
    write_tile(cache, "tiles/dem/16/-1/1.tif", 1000, index)

    plan = plan_eviction(cache, ceiling_bytes=10_000, protected=())
    assert plan.evict == []
    assert plan.bytes_freed == 0


def test_the_least_recently_used_goes_first(cache):
    """One incident at a time: ground from last month should go before
    ground somebody is asking about now."""
    index = CacheIndex(cache)
    now = time.time()
    write_tile(cache, "tiles/dem/16/-1/1.tif", 1000, index, last_used=now - 90_000)
    write_tile(cache, "tiles/dem/16/-1/2.tif", 1000, index, last_used=now - 10)

    plan = plan_eviction(cache, ceiling_bytes=1000, protected=())
    assert plan.evict == ["tiles/dem/16/-1/1.tif"]


def test_enough_is_evicted_to_reach_the_ceiling(cache):
    index = CacheIndex(cache)
    now = time.time()
    for i in range(5):
        write_tile(cache, f"tiles/dem/16/-1/{i}.tif", 1000, index,
                   last_used=now - (100 - i))

    plan = plan_eviction(cache, ceiling_bytes=2000, protected=())
    assert plan.bytes_held <= 2000
    assert len(plan.evict) == 3


def test_map_state_is_never_evicted_by_size(cache):
    """It is kilobytes, it expires by age, and on a laptop with no signal
    it may be the only copy of a map there is."""
    index = CacheIndex(cache)
    path = cache / "caltopo-state" / "M1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"y" * 5000)
    index.record(path, layer="caltopo-state", tier="mutable")

    plan = plan_eviction(cache, ceiling_bytes=0, protected=())
    assert plan.evict == []


# ---------------------------------------------------------------------------
# What must survive.
# ---------------------------------------------------------------------------

def test_tiles_a_staged_assignment_rests_on_are_protected(cache):
    """Deleting a tile out from under a handle turns the next question
    about that assignment into a silent refetch at best, and an error in
    the middle of a debrief at worst."""
    index = CacheIndex(cache)
    width, height = panel_dimensions(BBOX)
    needed = [t.key("dem", "tif")
              for t in tiles_covering(BBOX, level_for(BBOX, width, height))]
    assert needed, "the fixture bbox covers no tiles"

    for key in needed:
        write_tile(cache, key, 1000, index, last_used=0)
    write_tile(cache, "tiles/dem/16/-999/999.tif", 1000, index, last_used=0)
    stage(cache, "M1_QA103")

    plan = plan_eviction(cache, ceiling_bytes=0, protected=tiles_in_use(cache))
    assert set(plan.evict) == {"tiles/dem/16/-999/999.tif"}
    assert plan.protected == len(needed)


def test_tiles_in_use_is_derived_from_the_manifest_not_a_stored_list(cache):
    """A stored tile list would go stale the moment a level or a buffer
    changed, and it would go stale silently."""
    stage(cache, "M1_QA103")
    held = tiles_in_use(cache)

    width, height = panel_dimensions(BBOX)
    expected = {t.key("dem", "tif")
                for t in tiles_covering(BBOX, level_for(BBOX, width, height))}
    assert held == expected


def test_an_unreadable_staged_manifest_does_not_abort_eviction(cache):
    index = CacheIndex(cache)
    write_tile(cache, "tiles/dem/16/-1/1.tif", 1000, index, last_used=0)
    directory = cache / "staged" / "broken"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text("{ not json", encoding="utf-8")

    plan = plan_eviction(cache, ceiling_bytes=0, protected=tiles_in_use(cache))
    assert plan.evict == ["tiles/dem/16/-1/1.tif"]


def test_a_file_the_index_never_saw_is_left_alone(cache):
    """It has no last_used, so there is no honest basis for ranking it.
    Deleting it would mean an upgrade predating the index quietly wiping a
    laptop's whole prefetch."""
    index = CacheIndex(cache)
    write_tile(cache, "tiles/dem/16/-1/1.tif", 1000, index, last_used=0)

    stranger = cache / "legacy_dem.tif"
    stranger.write_bytes(b"z" * 9_000_000)

    plan = plan_eviction(cache, ceiling_bytes=0, protected=())
    assert "legacy_dem.tif" not in plan.evict
    assert plan.unknown_skipped >= 1

    evict(cache, ceiling_bytes=0, execute=True, protected=())
    assert stranger.exists()


# ---------------------------------------------------------------------------
# Doing it.
# ---------------------------------------------------------------------------

def test_eviction_defaults_to_reporting_rather_than_deleting(cache):
    """Deleting cached ground on a laptop that may be about to lose
    connectivity is not something to do on a guess."""
    index = CacheIndex(cache)
    path = write_tile(cache, "tiles/dem/16/-1/1.tif", 1000, index, last_used=0)

    result = evict(cache, ceiling_bytes=0, protected=())
    assert result["executed"] is False
    assert result["would_evict"] == 1
    assert path.exists(), "a dry run deleted a file"


def test_executing_removes_the_files_and_forgets_them(cache):
    index = CacheIndex(cache)
    path = write_tile(cache, "tiles/dem/16/-1/1.tif", 1000, index, last_used=0)

    result = evict(cache, ceiling_bytes=0, execute=True, protected=())
    assert result["executed"] is True
    assert result["removed"] == 1
    assert not path.exists()
    assert index.entry(path) is None, "an evicted file stayed in the index"


def test_a_file_that_vanished_before_eviction_is_not_fatal(cache):
    index = CacheIndex(cache)
    path = write_tile(cache, "tiles/dem/16/-1/1.tif", 1000, index, last_used=0)
    path.unlink()

    result = evict(cache, ceiling_bytes=0, execute=True, protected=())
    assert result["executed"] is True
    assert result["removed"] == 0


# ---------------------------------------------------------------------------
# Who decides what is held.
# ---------------------------------------------------------------------------

def dem_keys(bbox=BBOX) -> set[str]:
    """The elevation tiles a directory staged over `bbox` rests on."""
    width, height = panel_dimensions(bbox)
    return {t.key("dem", "tif")
            for t in tiles_covering(bbox, level_for(bbox, width, height))}


def test_the_cache_evicts_without_needing_staging(cache, monkeypatch):
    """The cache is infra and must not import domain (AGENTS.md),
    so it cannot know what a staged directory is: it holds exactly what it
    is handed. With staging and handles made unimportable, eviction still
    plans and runs - and a staged manifest nobody told it about protects
    nothing, so it is not reading manifests behind the caller's back."""
    index = CacheIndex(cache)
    needed = dem_keys()
    for key in needed:
        write_tile(cache, key, 1000, index, last_used=0)
    stage(cache, "M1_QA103")

    # None in sys.modules makes `import sar_geo_mcp.staging` and `from
    # sar_geo_mcp.staging import x` fail, even inside a function and even
    # though this file imported the module already. That earlier import also
    # set the package attribute, which would still satisfy `from sar_geo_mcp
    # import staging` (or `from .. import staging` in the cache), so the
    # attributes go too.
    import sar_geo_mcp

    for name in ("staging", "handles"):
        monkeypatch.setitem(sys.modules, f"sar_geo_mcp.{name}", None)
        monkeypatch.delattr(sar_geo_mcp, name, raising=False)

    plan = plan_eviction(cache, ceiling_bytes=0, protected=())
    assert set(plan.evict) == needed
    assert plan.protected == 0

    result = evict(cache, ceiling_bytes=0, execute=True, protected=())
    assert result["removed"] == len(needed)


@pytest.mark.anyio
async def test_the_evict_cache_tool_still_protects_staged_ground(tmp_path):
    """Protection moved out of the cache in 0.27.1, so the tool is now where
    staging and the cache meet. Through MCP it must answer as it always did:
    staged tiles counted as protected, never offered, never removed."""
    pytest.importorskip("fastmcp")
    from fastmcp import Client

    from sar_geo_mcp.config import CaltopoConfig
    from sar_geo_mcp.server import build_server

    cache = tmp_path / "cache"
    index = CacheIndex(cache)
    needed = dem_keys()
    for key in needed:
        write_tile(cache, key, 1000, index, last_used=0)
    write_tile(cache, "tiles/dem/16/-999/999.tif", 1000, index, last_used=0)
    stage(cache, "M1_QA103")

    server = build_server(CaltopoConfig(cache_dir=cache))
    async with Client(server) as client:
        report = (await client.call_tool("evict_cache", {"ceiling_mb": 0})).data
        done = (await client.call_tool(
            "evict_cache", {"ceiling_mb": 0, "execute": True})).data

    assert report["executed"] is False
    assert report["protected_by_staged_assignments"] == len(needed)
    assert report["would_evict"] == 1
    assert report["examples"] == ["tiles/dem/16/-999/999.tif"]

    assert done["executed"] is True
    assert done["removed"] == 1
    assert all((cache / key).exists() for key in needed)
    assert not (cache / "tiles/dem/16/-999/999.tif").exists()


# ---------------------------------------------------------------------------
# `protected` is every caller's to state.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("function", [plan_eviction, evict],
                         ids=["plan_eviction", "evict"])
def test_protected_has_no_default_and_cannot_be_passed_by_position(cache, function):
    """A caller that forgot `protected` would quietly evict staged ground,
    so it is keyword-only and required (cache/evict.py). A "convenient"
    default of () would bring that failure back with every other test in
    this file still green."""
    parameter = inspect.signature(function).parameters["protected"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty

    with pytest.raises(TypeError, match="protected"):
        function(cache, 0)
    # Nor slipped in positionally where a reader would take it for another
    # argument.
    positional = (cache, 0, ()) if function is plan_eviction else (cache, 0, False, ())
    with pytest.raises(TypeError):
        function(*positional)


# ---------------------------------------------------------------------------
# What `tiles_in_use` makes of odd staged directories.
# ---------------------------------------------------------------------------

def write_manifest_text(cache: Path, name: str, text: str) -> None:
    directory = cache / "staged" / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(text, encoding="utf-8")


def test_a_cache_with_nothing_staged_holds_nothing(cache):
    """No `staged/` directory at all - a laptop that has only prefetched -
    is not an error, and leaves every tile to the ceiling."""
    assert tiles_in_use(cache) == set()          # the cache does not exist yet

    index = CacheIndex(cache)
    write_tile(cache, "tiles/dem/16/-1/1.tif", 1000, index, last_used=0)
    assert not (cache / "staged").exists()
    assert tiles_in_use(cache) == set()
    plan = plan_eviction(cache, ceiling_bytes=0, protected=tiles_in_use(cache))
    assert plan.evict == ["tiles/dem/16/-1/1.tif"]


@pytest.mark.parametrize("text", [
    pytest.param("{ not json", id="not-json"),
    pytest.param(json.dumps({"map_id": "M1"}), id="no-bbox"),
    pytest.param(json.dumps(["not", "an", "object"]), id="a-list"),
    pytest.param(json.dumps({"bbox_3857": None}), id="bbox-null"),
    pytest.param(json.dumps({"bbox_3857": 5}), id="bbox-a-number"),
    pytest.param(json.dumps({"bbox_3857": [0, 0, 0, 0]}), id="bbox-a-point"),
    pytest.param(json.dumps({"bbox_3857": [BBOX[2], BBOX[1], BBOX[0], BBOX[3]]}),
                 id="bbox-west-east-swapped"),
    pytest.param(json.dumps({"bbox_3857": [BBOX[0], BBOX[3], BBOX[2], BBOX[1]]}),
                 id="bbox-south-north-swapped"),
])
def test_one_odd_manifest_does_not_strip_protection_from_the_others(cache, text):
    """One damaged staged directory protects nothing itself, but it must
    neither block eviction for the whole cache nor cost a sound sibling
    its protection."""
    index = CacheIndex(cache)
    for key in dem_keys():
        write_tile(cache, key, 1000, index, last_used=0)
    write_tile(cache, "tiles/dem/16/-999/999.tif", 1000, index, last_used=0)
    stage(cache, "M1_GOOD")
    write_manifest_text(cache, "M1_BAD", text)

    assert tiles_in_use(cache) == dem_keys()
    plan = plan_eviction(cache, ceiling_bytes=0, protected=tiles_in_use(cache))
    assert plan.evict == ["tiles/dem/16/-999/999.tif"]
    assert plan.protected == len(dem_keys())


def test_a_manifest_path_that_is_a_directory_is_skipped(cache):
    stage(cache, "M1_GOOD")
    (cache / "staged" / "M1_ODD" / "manifest.json").mkdir(parents=True)

    assert tiles_in_use(cache) == dem_keys()


B25 = ("PR #4 review B25: a staged manifest whose bbox_3857 is not four "
       "numbers makes tiles_in_use (and so evict_cache) fail outright "
       "instead of being skipped")

#: bbox_3857 values that are not four numbers. The numeric strings are the
#: fixture's own extent, so whether a fix parses them (as
#: `staging._staged_bbox` does) or skips them, the answer is the same.
MALFORMED_BBOXES = [
    pytest.param([None, None, None, None], id="nulls"),
    pytest.param({"w": 0, "s": 0, "e": 1, "n": 1}, id="a-dict"),
    pytest.param(list(BBOX[:3]), id="three-numbers"),
    pytest.param([[v] for v in BBOX], id="nested-lists"),
    pytest.param("abcd", id="a-string"),
    pytest.param([str(v) for v in BBOX], id="numeric-strings"),
]


@pytest.mark.xfail(strict=True, reason=B25)
@pytest.mark.parametrize("bbox", MALFORMED_BBOXES)
def test_a_malformed_bbox_is_skipped_like_unreadable_json(cache, bbox):
    """The comment in `tiles_in_use` says an unreadable manifest protects
    nothing and is skipped; only the JSON read is broadly guarded, so these
    raise TypeError or IndexError out of it and the sound sibling's tiles
    are never computed either."""
    index = CacheIndex(cache)
    for key in dem_keys():
        write_tile(cache, key, 1000, index, last_used=0)
    write_tile(cache, "tiles/dem/16/-999/999.tif", 1000, index, last_used=0)
    stage(cache, "M1_GOOD")
    stage(cache, "M1_BAD", bbox=bbox)

    held = tiles_in_use(cache)
    assert held == dem_keys()
    plan = plan_eviction(cache, ceiling_bytes=0, protected=held)
    assert plan.evict == ["tiles/dem/16/-999/999.tif"]


@pytest.mark.xfail(strict=True, reason=B25)
@pytest.mark.anyio
async def test_evict_cache_answers_beside_a_malformed_manifest(tmp_path):
    """Through MCP the same manifest turns into "Error calling tool
    'evict_cache': unsupported operand type(s)...", which names neither the
    directory at fault nor what to do. Fails closed - nothing is deleted -
    but the operator cannot bring the cache under its ceiling at all."""
    pytest.importorskip("fastmcp")
    from fastmcp import Client

    from sar_geo_mcp.config import CaltopoConfig
    from sar_geo_mcp.server import build_server

    cache = tmp_path / "cache"
    index = CacheIndex(cache)
    for key in dem_keys():
        write_tile(cache, key, 1000, index, last_used=0)
    write_tile(cache, "tiles/dem/16/-999/999.tif", 1000, index, last_used=0)
    stage(cache, "M1_GOOD")
    stage(cache, "M1_BAD", bbox=[str(v) for v in BBOX])

    server = build_server(CaltopoConfig(cache_dir=cache))
    async with Client(server) as client:
        result = await client.call_tool(
            "evict_cache", {"ceiling_mb": 0, "execute": True}, raise_on_error=False)

    assert result.is_error is False, result.content
    assert result.data["protected_by_staged_assignments"] == len(dem_keys())
    assert result.data["removed"] == 1
    assert all((cache / key).exists() for key in dem_keys())


# ---------------------------------------------------------------------------
# Held tiles are the tiles staging actually fetched.
#
# Every test above writes the manifest by hand and works out the expected
# keys with the same default `panel_dimensions(bbox)` that `tiles_in_use`
# uses, so the expectation repeats the implementation. These stage for real
# - `prepare_area`, `stage_terrain`, `fetch_dem`, the tile cache and the
# manifest - with only the network faked, and compare against what landed
# on disk. The scenarios stage this way on every call: the LZ ground at
# 2 m (`LZ_RESOLUTION_M`), egress and rings at 10 m (`WALKING_RESOLUTION_M`).
# ---------------------------------------------------------------------------

STAGED_GROUNDS = [
    # The control: no resolution asked for, the 1,024 px default raster.
    pytest.param(1000.0, None, id="default-1000m"),
    # plan_medevac's default egress ground: coarser, but at this latitude
    # at the same level the default size would give (14).
    pytest.param(3000.0, 10.0, id="10m-3000m"),
    # plan_medevac's and plan_initial_response's LZ ground: fetched at
    # level 16, where the default size gives 15. Until 0.31.9 only level 15
    # was held (PR #4 review B3).
    pytest.param(1000.0, 2.0, id="lz-2m-1000m"),
    # 10 m walking ground near 1,500 m - rings for shorter ring_minutes
    # ([15, 30] stage 2,000 m), or a smaller egress_radius_m: fetched at
    # level 14, where the default size gives 15 (B3).
    pytest.param(1500.0, 10.0, id="walking-10m-1500m"),
]


def stage_for_real(monkeypatch, cache: Path, radius_m: float,
                   resolution_m: float | None):
    """Stage a circle through the real `prepare_area` with only the network
    faked: the 3DEP tile body, the vegetation clip and the OSM fetches.
    Returns the context and the elevation tile keys that landed in the
    cache."""
    from sar_geo_mcp.components import staging as staging_component
    from sar_geo_mcp.config import CaltopoConfig
    from sar_geo_mcp.context import ServerContext

    monkeypatch.setattr("sar_geo_mcp.net.bytes_with_retry",
                        lambda url, *args, **kwargs: synthetic_tile_bytes())
    monkeypatch.setattr("sar_geo_mcp.staging.write_evt_clip",
                        lambda *args, **kwargs: None)
    for name in ("routes_for", "parking_for", "gates_for"):
        monkeypatch.setattr(f"sar_geo_mcp.terrain.linear.{name}",
                            lambda *args, **kwargs: [])

    ctx = ServerContext.build(CaltopoConfig(cache_dir=cache))
    staging_component.prepare_area(ctx, center_latitude=37.4022,
                                   center_longitude=-122.3164,
                                   radius_m=radius_m, resolution_m=resolution_m)
    fetched = {path.relative_to(cache).as_posix()
               for path in (cache / "tiles" / "dem").rglob("*.tif")}
    assert fetched, "staging fetched no elevation tiles"
    return ctx, fetched


@pytest.mark.parametrize("radius_m, resolution_m", STAGED_GROUNDS)
def test_every_tile_staging_fetched_is_held(cache, monkeypatch, radius_m, resolution_m):
    _, fetched = stage_for_real(monkeypatch, cache, radius_m, resolution_m)

    held = tiles_in_use(cache)
    assert sorted(fetched - held) == [], (
        f"staged ground rests on tiles nothing protects; held: {sorted(held)}")
    plan = plan_eviction(cache, ceiling_bytes=0, protected=held)
    assert sorted(set(plan.evict) & fetched) == []
    assert plan.protected >= len(fetched)


@pytest.mark.parametrize("radius_m, resolution_m", STAGED_GROUNDS)
def test_evict_cache_removes_none_of_the_tiles_staged_ground_rests_on(
        cache, monkeypatch, radius_m, resolution_m):
    """The tool's promise, end to end: "Tiles that staged assignments still
    rest on are never removed". Removed, the staged handle keeps its own
    dem.tif, but the next re-stage on a laptop with no signal fails."""
    from sar_geo_mcp.components.cache import evict_cache

    ctx, fetched = stage_for_real(monkeypatch, cache, radius_m, resolution_m)

    report = evict_cache(ctx, ceiling_mb=0, execute=True)
    assert report["executed"] is True
    assert report["protected_by_staged_assignments"] == len(fetched)
    assert sorted(key for key in fetched if not (cache / key).exists()) == []


# ---------------------------------------------------------------------------
# What a manifest records, and what an older one is held at.
#
# Since 0.31.9 `stage_terrain` writes the raster size `fetch_dem` was asked
# for into the manifest (`dem_width_px`, `dem_height_px`). A manifest from
# before then carries `dem_resolution_m` and `latitude_deg`, and the size is
# worked back from those - every laptop in the field has directories like
# that, some of them staged at a chosen resolution.
# ---------------------------------------------------------------------------

def staged_manifest_path(cache: Path) -> Path:
    (path,) = (cache / "staged").glob("*/manifest.json")
    return path


def write_old_manifest(cache: Path, name: str, bbox, width_px: int,
                       latitude_deg: float = 37.4) -> None:
    """A manifest as `write_manifest` wrote it before 0.31.9, for a raster
    `width_px` wide over `bbox`: no recorded size, and the pixel width in
    ground metres as `stage_terrain` works it out from the raster."""
    from sar_geo_mcp.geo import ground_meters_per_mercator_unit
    from sar_geo_mcp.staging import write_manifest

    directory = cache / "staged" / name
    directory.mkdir(parents=True, exist_ok=True)
    resolution_m = ((bbox[2] - bbox[0]) / width_px
                    * ground_meters_per_mercator_unit(latitude_deg))
    write_manifest(directory, map_id="M1", assignment_label=name, bbox=bbox,
                   dem_bbox=bbox, dem_resolution_m=resolution_m,
                   latitude_deg=latitude_deg, panels={}, warnings=[])


def levels_of(keys) -> set[int]:
    """The pyramid levels of `tiles/dem/<level>/<col>/<row>.tif` keys."""
    return {int(key.split("/")[2]) for key in keys}


@pytest.mark.parametrize("radius_m, resolution_m", STAGED_GROUNDS)
def test_staging_records_the_raster_size_it_fetched_at(cache, monkeypatch,
                                                       radius_m, resolution_m):
    """The size in the manifest is the dem.tif's own, and at the manifest's
    extent it names exactly the tiles that landed: held, and nothing else
    with them."""
    import rasterio

    _, fetched = stage_for_real(monkeypatch, cache, radius_m, resolution_m)
    path = staged_manifest_path(cache)
    manifest = json.loads(path.read_text(encoding="utf-8"))

    with rasterio.open(path.parent / "dem.tif") as dem:
        recorded = (manifest["dem_width_px"], manifest["dem_height_px"])
        assert recorded == (dem.width, dem.height)
    assert tiles_in_use(cache) == fetched


@pytest.mark.parametrize("radius_m, resolution_m", STAGED_GROUNDS)
def test_a_manifest_from_before_the_size_was_recorded_still_holds_every_tile(
        cache, monkeypatch, radius_m, resolution_m):
    """The same grounds, their manifests stripped back to what 0.31.8 wrote.
    The tiles are still held, so still never removed."""
    from sar_geo_mcp.components.cache import evict_cache

    ctx, fetched = stage_for_real(monkeypatch, cache, radius_m, resolution_m)
    path = staged_manifest_path(cache)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    del manifest["dem_width_px"], manifest["dem_height_px"]
    path.write_text(json.dumps(manifest), encoding="utf-8")

    assert sorted(fetched - tiles_in_use(cache)) == []
    evict_cache(ctx, ceiling_mb=0, execute=True)
    assert sorted(key for key in fetched if not (cache / key).exists()) == []


#: Taller than wide: the height was the longest edge and the width was
#: rounded from it, so the pixel width leaves a few heights possible.
TALL = (BBOX[0], BBOX[1], BBOX[0] + 1914.7, BBOX[1] + 6318.51)


@pytest.mark.parametrize("max_px", [
    # At these three, rounding the extent over the pixel width on each axis
    # takes a height a pixel or so short, and with it a level coarser than
    # staging fetched (14 for 15, 13 for 14, 17 for 18).
    pytest.param(701, id="701px"),
    pytest.param(351, id="351px"),
    pytest.param(5608, id="5608px"),
    pytest.param(1024, id="default"),
])
def test_an_old_manifest_over_a_tall_extent_holds_the_level_staging_fetched(cache, max_px):
    width, height = panel_dimensions(TALL, max_px=max_px)
    staged = level_for(TALL, width, height)
    write_old_manifest(cache, "aoi-tall", TALL, width_px=width)

    held = tiles_in_use(cache)
    assert {t.key("dem", "tif") for t in tiles_covering(TALL, staged)} <= held
    # Bounded: at most the level beside it as well, never a sweep of them.
    assert levels_of(held) <= {staged - 1, staged, staged + 1}
    assert len(levels_of(held)) <= 2


@pytest.mark.parametrize("bbox", [
    pytest.param(TALL, id="tall"),
    pytest.param((BBOX[0], BBOX[1], BBOX[0] + 1200.0, BBOX[1] + 7300.0), id="taller"),
    pytest.param((TALL[0], TALL[1], TALL[0] + 6318.51, TALL[1] + 1914.7), id="wide"),
    pytest.param(BBOX, id="square"),
])
def test_the_level_worked_back_from_an_old_manifest_never_misses(bbox):
    """Every raster size staging can ask for, one to MAX_STAGE_PIXELS on the
    longest edge: the level `fetch_dem` took is always among those held, and
    at most one more beside it. Wide or square, the width was the longest
    edge, so exactly that level."""
    from sar_geo_mcp.geo import ground_meters_per_mercator_unit
    from sar_geo_mcp.staging import MAX_STAGE_PIXELS, _staged_dem_levels

    latitude = 37.4
    for max_px in range(1, MAX_STAGE_PIXELS + 1):
        width, height = panel_dimensions(bbox, max_px=max_px)
        staged = level_for(bbox, width, height)
        manifest = {"bbox_3857": list(bbox), "latitude_deg": latitude,
                    "dem_resolution_m": ((bbox[2] - bbox[0]) / width
                                         * ground_meters_per_mercator_unit(latitude))}
        levels = _staged_dem_levels(manifest, bbox)
        assert staged in levels, (max_px, levels)
        assert set(levels) <= {staged - 1, staged, staged + 1}, (max_px, levels)
        if bbox[2] - bbox[0] >= bbox[3] - bbox[1]:
            assert levels == [staged], (max_px, levels)


#: The fixture extent staged finer than the default size: 2,600 px takes
#: level 18, where 1,024 px takes 17.
FINE_PX = 2600


def fine_keys() -> set[str]:
    width, height = panel_dimensions(BBOX, max_px=FINE_PX)
    return {t.key("dem", "tif")
            for t in tiles_covering(BBOX, level_for(BBOX, width, height))}


def test_the_fixture_fine_level_is_not_the_default_one():
    """The two tests below would prove nothing if it were."""
    assert levels_of(fine_keys()) != levels_of(dem_keys())


@pytest.mark.parametrize("damage", [
    pytest.param({"dem_width_px": True, "dem_height_px": True}, id="booleans"),
    pytest.param({"dem_width_px": "2600", "dem_height_px": "2585"}, id="strings"),
    pytest.param({"dem_width_px": 0, "dem_height_px": 2585}, id="zero"),
    pytest.param({"dem_width_px": 10**9, "dem_height_px": 10**9}, id="beyond-any-staging"),
    pytest.param({"dem_width_px": 2600}, id="width-only"),
])
def test_a_damaged_recorded_size_is_not_believed(cache, damage):
    """The size is worked back from the resolution instead, as for a
    manifest that never recorded one - not taken from the damage, and not
    left to the default size either."""
    write_old_manifest(cache, "aoi-fine", BBOX, width_px=FINE_PX)
    path = staged_manifest_path(cache)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(damage)
    path.write_text(json.dumps(manifest), encoding="utf-8")

    assert tiles_in_use(cache) == fine_keys()


@pytest.mark.parametrize("damage", [
    pytest.param({"dem_resolution_m": float("nan")}, id="resolution-nan"),
    pytest.param({"dem_resolution_m": float("inf")}, id="resolution-infinite"),
    pytest.param({"dem_resolution_m": 0}, id="resolution-zero"),
    pytest.param({"dem_resolution_m": -2.0}, id="resolution-negative"),
    pytest.param({"dem_resolution_m": "2.0"}, id="resolution-a-string"),
    pytest.param({"dem_resolution_m": True}, id="resolution-a-boolean"),
    pytest.param({"dem_resolution_m": 1e-300}, id="resolution-beyond-any-staging"),
    pytest.param({"dem_resolution_m": 1e300}, id="resolution-wider-than-the-ground"),
    pytest.param({"dem_resolution_m": None}, id="resolution-null"),
    # Past a float's range, which `math.isfinite` raises on rather than
    # answering.
    pytest.param({"dem_resolution_m": 10**400}, id="resolution-a-huge-integer"),
    pytest.param({"latitude_deg": 10**400}, id="latitude-a-huge-integer"),
    pytest.param({"latitude_deg": None}, id="latitude-null"),
    pytest.param({"latitude_deg": 180.0}, id="latitude-off-the-globe"),
])
def test_a_manifest_that_cannot_say_its_size_is_held_at_the_default_one(cache, damage):
    """Neither a recorded size nor a resolution to work one back from: the
    default 1,024 px raster, what every directory was held at before
    0.31.9. Never an error, and never the sound sibling's protection."""
    write_old_manifest(cache, "aoi-fine", BBOX, width_px=FINE_PX)
    path = staged_manifest_path(cache)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(damage)
    path.write_text(json.dumps(manifest), encoding="utf-8")

    assert tiles_in_use(cache) == dem_keys()

    write_old_manifest(cache, "aoi-sound", BBOX, width_px=FINE_PX)
    assert tiles_in_use(cache) == dem_keys() | fine_keys()
