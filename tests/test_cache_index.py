"""The cache index (F1).

The property that matters most here is not what the index records - it is
that the index NEVER breaks a fetch. It is metadata about files that are
themselves the truth, so a corrupt database, a read-only directory, or a
disk that filled up must all degrade to "no bookkeeping" rather than "no
download". A field laptop that stops fetching terrain because a sqlite
file went bad is a far worse outcome than one with an inaccurate size
report.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from sar_geo_mcp.cache import CacheIndex, note_hit, note_write
from sar_geo_mcp.terrain.dem import TIFF_MAGIC_LE


@pytest.fixture
def index(tmp_path):
    return CacheIndex(tmp_path / "cache")


def written(index: CacheIndex, name: str, payload: bytes = b"x" * 100) -> Path:
    index.cache_dir.mkdir(parents=True, exist_ok=True)
    path = index.cache_dir / name
    path.write_bytes(payload)
    return path


# ---------------------------------------------------------------------------
# Recording.
# ---------------------------------------------------------------------------

def test_record_then_read_back_carries_size_layer_and_tier(index):
    path = written(index, "dem_abc.tif", b"y" * 4096)
    index.record(path, layer="dem", tier="static", source_version="3dep-2024")

    entry = index.entry(path)
    assert entry is not None
    assert entry.layer == "dem"
    assert entry.tier == "static"
    assert entry.bytes == 4096
    assert entry.source_version == "3dep-2024"
    assert entry.age_seconds < 5


def test_recording_the_same_file_twice_updates_rather_than_duplicates(index):
    path = written(index, "dem_abc.tif")
    index.record(path, layer="dem")
    written(index, "dem_abc.tif", b"z" * 900)
    index.record(path, layer="dem")

    assert len(index.entries()) == 1
    assert index.entry(path).bytes == 900


def test_touch_never_invents_a_row_for_a_file_it_has_not_seen(index):
    """A file cached before the index existed has no known fetch time.

    Inventing one would hand something downstream a fabricated age, and a
    made-up age is worse than a missing one because it will be believed.
    """
    path = written(index, "legacy_dem.tif")
    index.touch(path)

    assert index.entry(path) is None
    assert index.entries() == []


def test_touch_advances_last_used_without_moving_fetched_at(index):
    """LRU needs last_used to move; staleness needs fetched_at to stay put."""
    path = written(index, "evt_tile_c1_r1.tif")
    index.record(path, layer="evt")
    original = index.entry(path)

    time.sleep(0.02)
    index.touch(path)
    after = index.entry(path)

    assert after.last_used > original.last_used
    assert after.fetched_at == original.fetched_at


def test_keys_are_cache_relative_so_the_directory_can_move(index, tmp_path):
    """A field bundle unzips to a different absolute path on every laptop.

    Absolute keys would orphan every row on the first machine that was not
    the developer's.
    """
    path = written(index, "dem_abc.tif")
    index.record(path, layer="dem")

    moved = tmp_path / "elsewhere"
    index.cache_dir.rename(moved)
    assert CacheIndex(moved).entry(moved / "dem_abc.tif") is not None


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------

def test_summary_splits_by_tier_and_layer(index):
    index.record(written(index, "dem_a.tif", b"a" * 1000), layer="dem")
    index.record(written(index, "dem_b.tif", b"b" * 2000), layer="dem")
    index.record(written(index, "evt_c.tif", b"c" * 500), layer="evt")
    index.record(written(index, "state.json", b"d" * 10),
                 layer="caltopo-state", tier="mutable")

    summary = index.summary()
    assert summary["files"] == 4
    assert summary["bytes"] == 3510
    assert summary["by_tier"]["static"] == {"files": 3, "bytes": 3500}
    assert summary["by_tier"]["mutable"] == {"files": 1, "bytes": 10}
    assert summary["by_layer"]["dem"] == {"files": 2, "bytes": 3000}


# ---------------------------------------------------------------------------
# The index is never load-bearing.
# ---------------------------------------------------------------------------

def test_a_corrupt_index_degrades_to_no_bookkeeping_and_never_raises(index):
    """The whole safety argument in one test.

    A laptop that stops fetching terrain because a sqlite file went bad is
    a far worse failure than one with an inaccurate size report.
    """
    index.cache_dir.mkdir(parents=True, exist_ok=True)
    index.path.write_bytes(b"this is emphatically not a database")

    path = written(index, "dem_abc.tif")
    index.record(path, layer="dem")   # must not raise
    index.touch(path)                 # must not raise
    index.forget(path)                # must not raise

    assert index.entries() == []
    assert index.entry(path) is None
    assert index.summary()["files"] == 0


def test_recording_a_file_that_vanished_is_a_no_op(index):
    index.record(index.cache_dir / "never_written.tif", layer="dem")
    assert index.entries() == []


# ---------------------------------------------------------------------------
# The fetch sites actually record. Stubbed - no network.
# ---------------------------------------------------------------------------

def test_a_fetched_evt_tile_lands_in_the_index(tmp_path):
    """Wiring test: the index is only worth having if the fetchers use it."""
    from sar_geo_mcp.terrain import evt

    cache = tmp_path / "cache"
    tile = evt.fetch_tile(3, 7, cache, get_bytes=lambda url: TIFF_MAGIC_LE + b"\x00" * 64)

    entry = CacheIndex(cache).entry(tile)
    assert entry is not None, "fetch_tile wrote a file the index never heard about"
    assert entry.layer == "evt"
    assert entry.tier == "static"


def test_a_cache_hit_touches_rather_than_re_records(tmp_path):
    from sar_geo_mcp.terrain import evt

    cache = tmp_path / "cache"

    def once(url):
        raise AssertionError("second fetch should have been served from cache")

    tile = evt.fetch_tile(3, 7, cache, get_bytes=lambda url: TIFF_MAGIC_LE + b"\x00" * 64)
    first = CacheIndex(cache).entry(tile)

    time.sleep(0.02)
    evt.fetch_tile(3, 7, cache, get_bytes=once)
    second = CacheIndex(cache).entry(tile)

    assert second.fetched_at == first.fetched_at, "a cache hit is not a fetch"
    assert second.last_used > first.last_used


def test_note_helpers_are_the_same_thing_as_the_class(tmp_path):
    """The convenience functions exist so wiring a fetcher costs two lines;
    if they drifted from the class the wiring would quietly do nothing."""
    cache = tmp_path / "cache"
    cache.mkdir()
    path = cache / "osm_tile_c0_r0.json"
    path.write_text("{}", encoding="utf-8")

    note_write(cache, path, layer="osm")
    assert CacheIndex(cache).entry(path).layer == "osm"

    note_hit(cache, path)  # must not raise
