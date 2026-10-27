"""Every cache writer lands its file through a unique temp name.

WHY THIS FILE EXISTS
--------------------
An operator opening a planning session got a `PermissionError` on the
FIRST click, every time, and success on the second:

    pathlib.py:1047 in write_text
    PermissionError: [Errno 13] Permission denied:
        'cache\\caltopo-state\\TESTMAP.json.part'

Starting a session reads the map twice at once - the web thread for the
PLS echo, the staging thread a moment behind it - and on a cold cache both
fetched and both wrote the same FIXED `.part` name. The second click worked
because the cache was warm by then and only one writer was left.

`caltopo/wms.py` had already solved this for tiles and panels in 0.5.3.
These tests pin the shared helper that solved it once for all seven
writers, and the reported bug specifically.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from sar_geo_mcp.cache import atomic_output, atomic_write_bytes, atomic_write_text


# -- the helper itself ----------------------------------------------------

def test_two_threads_writing_one_destination_both_succeed(tmp_path):
    """The whole point. Two writers on the same destination at the same
    instant must both return, and the file they leave must be one complete
    copy - not a truncation of one interleaved with the other."""
    destination = tmp_path / "shared.json"
    payload = json.dumps({"features": ["x"] * 4000})
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def worker():
        try:
            barrier.wait(timeout=5)
            atomic_write_text(destination, payload)
        except BaseException as exc:  # pragma: no cover - surfaced via errors
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not errors, f"concurrent atomic write raised: {errors}"
    assert json.loads(destination.read_text(encoding="utf-8"))["features"]
    assert destination.read_text(encoding="utf-8") == payload
    leftover = list(tmp_path.rglob("*.part"))
    assert not leftover, f"leftover temp files: {leftover}"


def test_two_threads_through_the_context_manager_both_succeed(tmp_path):
    """The same guarantee for the caller that writes the temp file itself -
    rasterio is handed a PATH and opens it, so it cannot be given bytes."""
    destination = tmp_path / "shared.tif"
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def worker(marker: bytes):
        try:
            barrier.wait(timeout=5)
            with atomic_output(destination) as temporary:
                # A slow writer, so the two overlap for certain.
                with open(temporary, "wb") as handle:
                    for _ in range(20):
                        handle.write(marker * 100)
                        time.sleep(0.001)
        except BaseException as exc:  # pragma: no cover - surfaced via errors
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(m,)) for m in (b"a", b"b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not errors, f"concurrent atomic output raised: {errors}"
    content = destination.read_bytes()
    assert len(content) == 2000
    # One writer's content, whole - never a blend of the two.
    assert content in (b"a" * 2000, b"b" * 2000)
    assert not list(tmp_path.rglob("*.part"))


def test_a_failure_mid_write_leaves_no_temp_file_and_no_destination(tmp_path):
    destination = tmp_path / "never.json"

    with pytest.raises(RuntimeError, match="the network went away"):
        with atomic_output(destination) as temporary:
            temporary.write_bytes(b"half a file")
            raise RuntimeError("the network went away")

    assert not destination.exists()
    leftover = list(tmp_path.rglob("*.part"))
    assert not leftover, f"leftover temp files: {leftover}"


def test_a_failure_before_the_temp_is_written_leaves_nothing(tmp_path):
    """The caller may raise before it ever opens the temp path; cleaning up
    a file that was never created must not itself raise."""
    destination = tmp_path / "never.json"

    with pytest.raises(ValueError):
        with atomic_output(destination):
            raise ValueError("gave up early")

    assert not destination.exists()
    assert not list(tmp_path.rglob("*.part"))


def test_the_temp_name_is_unique_per_writer(tmp_path):
    """Two writers must never be handed the same temp path - that is the
    whole fix, so it is asserted directly rather than inferred from timing."""
    destination = tmp_path / "shared.json"
    seen = []
    for _ in range(5):
        with atomic_output(destination) as temporary:
            seen.append(temporary)
            temporary.write_text("x", encoding="utf-8")
    assert len({path.name for path in seen}) == 5
    assert all(path.parent == destination.parent for path in seen)


def test_the_replace_is_retried_when_windows_briefly_denies_it(tmp_path,
                                                               monkeypatch):
    """On Windows an antivirus scanner or the search indexer can hold the
    file for a few milliseconds the instant it lands, and the rename comes
    back as `PermissionError`. One failure must not lose the write."""
    destination = tmp_path / "guarded.json"
    attempts = {"count": 0}
    real_replace = Path.replace

    def flaky_replace(self, target):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise PermissionError(13, "Access is denied")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)

    atomic_write_text(destination, "survived")

    assert attempts["count"] == 2, "the denied replace was not retried"
    assert destination.read_text(encoding="utf-8") == "survived"
    assert not list(tmp_path.rglob("*.part"))


def test_a_replace_that_never_succeeds_raises_and_leaves_no_temp(tmp_path,
                                                                 monkeypatch):
    """The retry is bounded. A destination that stays locked is a real
    failure and must be reported, not spun on."""
    destination = tmp_path / "locked.json"
    attempts = {"count": 0}

    def always_denied(self, target):
        attempts["count"] += 1
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(Path, "replace", always_denied)

    with pytest.raises(PermissionError):
        atomic_write_bytes(destination, b"never lands")

    assert 1 < attempts["count"] < 20
    assert not destination.exists()
    assert not list(tmp_path.rglob("*.part"))


def test_bytes_and_text_both_land_the_exact_content(tmp_path):
    assert atomic_write_bytes(tmp_path / "b.bin", b"\x00\x01\x02") == tmp_path / "b.bin"
    assert (tmp_path / "b.bin").read_bytes() == b"\x00\x01\x02"
    atomic_write_text(tmp_path / "t.txt", "sierra é")
    assert (tmp_path / "t.txt").read_text(encoding="utf-8") == "sierra é"


# -- the reported bug -----------------------------------------------------

def test_two_threads_caching_one_map_state_both_succeed(tmp_path):
    """The field report, reproduced. Starting a session reads the map from
    the web thread and the staging thread at once; on a cold cache both
    called `write_cached` for the same map id, and the fixed
    `TESTMAP.json.part` was written by one while the other held it."""
    from sar_geo_mcp.caltopo.state import MapState, read_cached, write_cached

    features = [{"id": f"f{n}", "properties": {"class": "Assignment"},
                 "geometry": {"type": "Point", "coordinates": [-122.3, 37.4]}}
                for n in range(300)]
    state = MapState(map_id="TESTMAP", features=features, token=1_787_989_336_147,
                     fetched_at=time.time(), membership_at=time.time())

    errors: list[BaseException] = []
    # Several rounds: the collision is a race, and one pass through it is
    # not evidence that the race is gone.
    for _ in range(10):
        barrier = threading.Barrier(2)

        def worker():
            try:
                barrier.wait(timeout=5)
                write_cached(tmp_path, state)
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

    assert not errors, f"concurrent write_cached raised: {errors}"

    cached = read_cached(tmp_path, "TESTMAP")
    assert cached is not None, "the cached map state was not readable"
    assert len(cached.features) == 300
    leftover = list(tmp_path.rglob("*.part"))
    assert not leftover, f"leftover temp files: {leftover}"


# -- a second call site, so the helper is proven on more than one ---------

def test_two_threads_saving_one_candidate_set_both_succeed(tmp_path):
    """Two `hasty_candidates` calls on one handle stage into the same
    directory and save the same set id."""
    from sar_geo_mcp.planning.candidates import Candidate, CandidateSet, load_set, save_set

    cset = CandidateSet(
        set_id="hasty-abc123",
        tool="hasty_candidates",
        params={"include": ["trail"]},
        handle="aoi-0123456789abcdef",
        created_at="2026-09-08T20:00:00+00:00",
        candidates=[
            Candidate(
                id=f"hasty-abc123.{n}",
                kind="line",
                axis="trail",
                axis_ids=[f"route_{n}"],
                sources=["trail"],
                name=f"Ridge Trail {n}",
                geometry={"type": "LineString",
                          "coordinates": [[-122.31, 37.40], [-122.30 + n / 1000, 37.41]]},
                facts={"length_m": 1200.0},
                norm_flags=[],
                warnings=[],
            )
            for n in range(1, 40)
        ],
        summary={"count": 39},
        assumptions={},
        warnings=[],
    )

    errors: list[BaseException] = []
    for _ in range(10):
        barrier = threading.Barrier(2)

        def worker():
            try:
                barrier.wait(timeout=5)
                save_set(tmp_path, cset)
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

    assert not errors, f"concurrent save_set raised: {errors}"
    assert len(load_set(tmp_path, "hasty-abc123").candidates) == 39
    leftover = list(tmp_path.rglob("*.part"))
    assert not leftover, f"leftover temp files: {leftover}"
