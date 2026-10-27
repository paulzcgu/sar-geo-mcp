"""Candidate sets: stable ids, persistence inside the staged directory, and
summaries that carry no geometry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sar_geo_mcp.planning.candidates import (
    Candidate,
    CandidateSet,
    candidate_id,
    find_candidate,
    find_candidate_in,
    load_set,
    make_set_id,
    save_set,
    set_staleness,
    summaries,
)
from sar_geo_mcp.planning.boundary import _manifest_stamp

LINE = {"type": "LineString", "coordinates": [[-122.31, 37.40], [-122.30, 37.41]]}


def _candidate(index, set_id="hasty-abc123"):
    return Candidate(id=candidate_id(set_id, index), kind="line", axis="trail",
                     axis_ids=["route_0"], sources=["trail"], name="Ridge Trail",
                     geometry=LINE, facts={"length_m": 1200.0}, norm_flags=[], warnings=[])


def _set(set_id="hasty-abc123"):
    return CandidateSet(set_id=set_id, tool="hasty_candidates",
                        params={"include": ["trail"]}, handle="aoi-0123456789abcdef",
                        created_at="2026-09-07T20:00:00+00:00",
                        candidates=[_candidate(1, set_id), _candidate(2, set_id)],
                        summary={"count": 2}, assumptions={}, warnings=[])


def test_set_id_is_the_tool_plus_six_hex_of_the_params_and_is_stable():
    a = make_set_id("hasty", {"include": ["trail", "road"], "max_distance_m": None})
    b = make_set_id("hasty", {"max_distance_m": None, "include": ["trail", "road"]})
    c = make_set_id("hasty", {"include": ["trail"], "max_distance_m": None})
    assert a == b and a != c
    assert a.startswith("hasty-") and len(a) == len("hasty-") + 6


def test_candidate_ids_are_set_id_dot_one_based_index():
    assert candidate_id("hasty-abc123", 1) == "hasty-abc123.1"
    with pytest.raises(ValueError):
        candidate_id("hasty-abc123", 0)


def test_a_set_round_trips_through_the_staged_directory(tmp_path):
    path = save_set(tmp_path, _set())
    assert path == tmp_path / "candidates" / "hasty-abc123.json"
    loaded = load_set(tmp_path, "hasty-abc123")
    assert loaded == _set()
    assert json.loads(path.read_text(encoding="utf-8"))["set_id"] == "hasty-abc123"


def test_find_candidate_by_id(tmp_path):
    save_set(tmp_path, _set())
    found = find_candidate(tmp_path, "hasty-abc123.2")
    assert found.id == "hasty-abc123.2" and found.geometry == LINE
    with pytest.raises(KeyError, match="hasty-abc123.9"):
        find_candidate(tmp_path, "hasty-abc123.9")
    with pytest.raises(KeyError, match="nope"):
        find_candidate(tmp_path, "nope-000000.1")


def test_find_candidate_in_a_set_already_in_hand_refuses_in_the_same_words(tmp_path):
    """A caller that has loaded the set for another reason - its staleness -
    reads the candidate off that same read rather than loading it twice, and
    a bad index is refused in one wording, not two that can drift apart."""
    cset = _set()
    save_set(tmp_path, cset)

    assert find_candidate_in(cset, "hasty-abc123.2").id == "hasty-abc123.2"
    with pytest.raises(KeyError) as in_hand:
        find_candidate_in(cset, "hasty-abc123.9")
    with pytest.raises(KeyError) as from_disk:
        find_candidate(tmp_path, "hasty-abc123.9")
    assert str(in_hand.value) == str(from_disk.value)
    with pytest.raises(ValueError, match="not a candidate id"):
        find_candidate_in(cset, "../x")


def test_a_malformed_candidate_id_is_refused_before_touching_the_filesystem(tmp_path):
    for bad in ("", "hasty-abc123", "../x.1", "hasty-abc123.x", "hasty-ab/c.1"):
        with pytest.raises(ValueError):
            find_candidate(tmp_path, bad)


def test_summaries_carry_everything_but_geometry():
    for entry in summaries(_set()):
        assert "geometry" not in entry
        assert entry["id"].startswith("hasty-abc123.")
        assert entry["facts"]["length_m"] == 1200.0 and entry["name"] == "Ridge Trail"


def test_save_set_writes_atomically_leaving_no_part_file(tmp_path):
    save_set(tmp_path, _set())
    entries = sorted(p.name for p in (tmp_path / "candidates").iterdir())
    assert entries == ["hasty-abc123.json"]


def test_saved_set_records_the_manifest_stamp_it_was_computed_from(tmp_path):
    (tmp_path / "manifest.json").write_text('{"v": 1}', encoding="utf-8")
    save_set(tmp_path, _set())
    loaded = load_set(tmp_path, "hasty-abc123")
    assert loaded.manifest_stamp == list(_manifest_stamp(tmp_path))


def test_set_staleness_is_none_right_after_saving(tmp_path):
    (tmp_path / "manifest.json").write_text('{"v": 1}', encoding="utf-8")
    save_set(tmp_path, _set())
    loaded = load_set(tmp_path, "hasty-abc123")
    assert set_staleness(tmp_path, loaded) is None


def test_set_staleness_flags_a_directory_re_staged_after_the_set_was_saved(tmp_path):
    (tmp_path / "manifest.json").write_text('{"v": 1}', encoding="utf-8")
    save_set(tmp_path, _set())
    loaded = load_set(tmp_path, "hasty-abc123")
    (tmp_path / "manifest.json").write_text('{"v": 2, "more": "data"}', encoding="utf-8")
    assert "re-staged" in set_staleness(tmp_path, loaded)


def test_set_staleness_flags_a_set_that_never_recorded_a_stamp(tmp_path):
    (tmp_path / "manifest.json").write_text('{"v": 1}', encoding="utf-8")
    assert set_staleness(tmp_path, _set()) is not None
    assert "does not record" in set_staleness(tmp_path, _set())


def test_make_set_id_rejects_non_json_native_params():
    with pytest.raises(TypeError):
        make_set_id("hasty", {"p": Path("x")})
