from pathlib import Path

import pytest

from sar_geo_mcp.caltopo.assignment import find_assignment
from sar_geo_mcp.caltopo.session import open_map
from sar_geo_mcp.config import CaltopoConfig


def test_open_map_reports_missing_credentials_helpfully(tmp_path):
    # No _env_file escape hatch needed any more: CaltopoConfig is a plain
    # dataclass that reads nothing on construction, where the app's
    # pydantic-settings object would have picked up a real .env.
    settings = CaltopoConfig(
        caltopo_config_path=tmp_path / "absent.ini",
        caltopo_account="you@example.com",
    )

    with pytest.raises(FileNotFoundError, match="cts.ini.example"):
        open_map("MAP123", settings)


def test_find_assignment_matches_letter_case_insensitively():
    features = [
        {"id": "1", "properties": {"class": "Marker", "title": "CP"}},
        {
            "id": "2",
            "properties": {"class": "Assignment", "letter": "QA", "number": ""},
            "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
        },
    ]

    assignment = find_assignment(features, "qa")

    assert assignment.id == "2"


def test_find_assignment_lists_available_letters_when_missing():
    features = [
        {
            "id": "2",
            "properties": {"class": "Assignment", "letter": "QA", "number": ""},
            "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
        },
        {
            "id": "3",
            "properties": {"class": "Assignment", "letter": "AE", "number": ""},
            "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
        },
    ]

    with pytest.raises(LookupError, match="AE, QA"):
        find_assignment(features, "ZZ")


def test_extract_features_handles_both_response_shapes():
    from sar_geo_mcp.caltopo.session import extract_features

    assert extract_features({"state": {"features": [{"id": "a"}]}}) == [{"id": "a"}]
    assert extract_features({"features": [{"id": "b"}]}) == [{"id": "b"}]
    assert extract_features({}) == []


def test_extract_features_handles_the_public_endpoint_shape():
    from sar_geo_mcp.caltopo.session import extract_features

    payload = {"result": {"state": {"features": [{"id": "x"}]}}}

    assert extract_features(payload) == [{"id": "x"}]


def test_extract_features_returns_empty_for_unrecognised_shape():
    from sar_geo_mcp.caltopo.session import extract_features

    assert extract_features({"result": {"state": {}}}) == []


def _assignment(letter, number, feature_id):
    return {
        "id": feature_id,
        "properties": {"class": "Assignment", "letter": letter, "number": number},
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
    }


def test_find_assignment_matches_a_bare_letter_against_a_numbered_assignment():
    """Real maps label assignments 'TA 1213'; a planner types 'TA'."""
    features = [_assignment("TA", "1213", "1")]

    assert find_assignment(features, "TA").id == "1"


def test_find_assignment_still_matches_the_full_label():
    features = [_assignment("TA", "1213", "1")]

    assert find_assignment(features, "TA1213").id == "1"


def test_find_assignment_reports_ambiguity_rather_than_guessing():
    features = [_assignment("SB", "1201", "1"), _assignment("SB", "1220", "2")]

    with pytest.raises(LookupError, match="ambiguous"):
        find_assignment(features, "SB")


def test_full_label_wins_over_an_ambiguous_bare_letter():
    features = [_assignment("SB", "1201", "1"), _assignment("SB", "1220", "2")]

    assert find_assignment(features, "SB1220").id == "2"


def test_open_map_never_starts_the_library_sync_thread(tmp_path, monkeypatch):
    """`caltopo_python.CaltopoSession` defaults to `sync=True`: a daemon
    thread that polls `/since/` every ten seconds for the life of the
    process, and nothing this hub returns ever stops it. Every consumer of
    `open_map` reads the map once (`.getFeatures()`) and drops the session,
    so each call leaked one poller - a debriefer server that re-read its map
    once a minute all day was hammering CalTopo forty times a second by
    evening (2026-09-06). A one-shot read must be a one-shot read."""
    import sar_geo_mcp.caltopo.session as session_module

    ini = tmp_path / "cts.ini"
    ini.write_text("[you@example.com]\nid=x\nkey=y\naccountId=z\n", encoding="utf-8")
    seen = {}

    class FakeSession:
        def __init__(self, host, map_id, **kwargs):
            seen.update(kwargs, host=host, map_id=map_id)

    monkeypatch.setattr(session_module, "CaltopoSession", FakeSession)
    settings = CaltopoConfig(caltopo_config_path=ini, caltopo_account="you@example.com")

    open_map("MAP123", settings)

    assert seen["map_id"] == "MAP123"
    assert seen.get("sync") is False


def test_open_map_can_be_asked_for_a_watched_session(tmp_path, monkeypatch):
    """The debriefer holds ONE session per map for the life of its process
    and wants the library's incremental polling for it - opt-in, never the
    default, and the interval goes straight through."""
    import sar_geo_mcp.caltopo.session as session_module

    ini = tmp_path / "cts.ini"
    ini.write_text("[you@example.com]\nid=x\nkey=y\naccountId=z\n", encoding="utf-8")
    seen = {}

    class FakeSession:
        def __init__(self, host, map_id, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(session_module, "CaltopoSession", FakeSession)
    settings = CaltopoConfig(caltopo_config_path=ini, caltopo_account="you@example.com")

    open_map("MAP123", settings, sync=True, sync_interval=7)

    assert seen["sync"] is True
    assert seen["syncInterval"] == 7


def test_open_map_passes_timeout_and_callbacks_to_the_library(tmp_path, monkeypatch):
    """A held session needs the library's callbacks and a fetch timeout that
    matches the public path (60 s, not the library's 10 s default, which is
    too short for a 2.75 MB map over a tethered phone)."""
    import sar_geo_mcp.caltopo.session as session_module

    ini = tmp_path / "cts.ini"
    ini.write_text("[you@example.com]\nid=x\nkey=y\naccountId=z\n", encoding="utf-8")
    seen = {}

    class FakeSession:
        def __init__(self, host, map_id, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(session_module, "CaltopoSession", FakeSession)
    settings = CaltopoConfig(caltopo_config_path=ini, caltopo_account="you@example.com")

    def closed(response=None):
        pass

    open_map("MAP123", settings, sync=True, sync_timeout=60,
             mapClosedCallback=closed)

    assert seen["syncTimeout"] == 60
    assert seen["mapClosedCallback"] is closed
