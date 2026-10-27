"""Planner-chosen areas: an immutable input staged behind an `aoi-` handle."""

from __future__ import annotations

import json
import sys

import pytest

from sar_geo_mcp.config import CaltopoConfig
from sar_geo_mcp.planning.area import (
    MAX_EXTENT_M,
    PLANNING_AREA_BUFFER_M,
    AreaOfInterest,
    area_key,
    build_area,
    prepare_area,
)
from sar_geo_mcp.handles import AREA_PREFIX, is_area_handle, resolve

PLS = (37.4022, -122.3164)


def test_a_circle_becomes_a_wgs84_polygon_of_the_right_size():
    area = build_area(center_latitude=PLS[0], center_longitude=PLS[1], radius_m=1000)
    assert area.kind == "circle"
    assert area.geometry["type"] == "Polygon"
    west, south, east, north = _bounds(area.geometry)
    # 2 km across, within a few metres.
    assert abs((east - west) * 111_320 * _cos(PLS[0]) - 2000) < 20
    assert abs((north - south) * 111_320 - 2000) < 20
    assert area.extent_m == pytest.approx(2000, abs=20)


def test_a_polygon_is_taken_as_given():
    poly = _square(PLS, half_m=500)
    area = build_area(geometry=poly)
    assert area.kind == "polygon" and area.center is None and area.radius_m is None
    assert area.geometry == poly


def test_exactly_one_of_circle_or_polygon():
    with pytest.raises(ValueError, match="exactly one"):
        build_area()
    with pytest.raises(ValueError, match="exactly one"):
        build_area(center_latitude=1, center_longitude=2, radius_m=3, geometry=_square(PLS, 10))
    with pytest.raises(ValueError, match="radius"):
        build_area(center_latitude=1, center_longitude=2)


def test_the_extent_limit_is_refused_with_the_limit_named():
    with pytest.raises(ValueError) as caught:
        build_area(center_latitude=PLS[0], center_longitude=PLS[1], radius_m=MAX_EXTENT_M / 2 + 100)
    assert str(int(MAX_EXTENT_M)) in str(caught.value) and "smaller radius" in str(caught.value)


def test_key_is_stable_under_coordinate_noise_and_changes_with_the_map():
    a = area_key("circle", [PLS[0], PLS[1]], 1500.0, "MAP1")
    b = area_key("circle", [PLS[0] + 1e-9, PLS[1]], 1500.04, "MAP1")
    c = area_key("circle", [PLS[0], PLS[1]], 1500.0, None)
    assert a == b and a != c
    assert a.startswith(AREA_PREFIX) and len(a) == len(AREA_PREFIX) + 16
    assert is_area_handle(a) and not is_area_handle("MAP1234_QA103")


def test_label_is_the_key_and_the_model_round_trips():
    area = build_area(center_latitude=PLS[0], center_longitude=PLS[1], radius_m=300, map_id="M")
    assert area.label == area.key
    restored = AreaOfInterest.model_validate_json(area.model_dump_json())
    assert restored == area


# --- prepare_area and resolve ------------------------------------------------

class FakeState:
    def __init__(self, features=()):
        self.features = list(features)

    def provenance(self):
        return {"source": "test", "age_seconds": 0}


class FakeReader:
    def __init__(self, cache_dir, offline=False):
        self.cache_dir = cache_dir
        self.offline = offline
        self.reads = []

    def read(self, map_id, *, public, max_age_seconds=300.0):
        self.reads.append((map_id, public, max_age_seconds))
        if self.offline:
            raise OSError("offline")
        return FakeState()


def stub_stage(monkeypatch, staged: list):
    from sar_geo_mcp.staging import write_manifest
    from sar_geo_mcp.terrain.linear import OSM_QUERY_VERSION

    def fake_stage(subject, map_id, out_dir, *, cache_dir, buffer_m, fingerprint,
                   access="public", subject_kind="assignment", **kwargs):
        from pathlib import Path

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "subject.json").write_text(subject.model_dump_json(), encoding="utf-8")
        # A current osm.json, exactly as a real stage_terrain success writes
        # one - so a test that reuses this directory sees ground already
        # current, and `refresh_osm` costs it a file read, not a fetch.
        (Path(out_dir) / "osm.json").write_text(
            json.dumps({"routes": [], "parking": [], "gates": [],
                        "query_version": OSM_QUERY_VERSION}),
            encoding="utf-8")
        write_manifest(out_dir, map_id=map_id, assignment_label=subject.label,
                       bbox=(0, 0, 1, 1), dem_bbox=(0, 0, 1, 1), dem_resolution_m=10.0,
                       latitude_deg=37.4, panels={}, warnings=["evt unavailable: X"],
                       assignment_fingerprint=fingerprint, access=access,
                       subject_kind=subject_kind)
        staged.append((subject.label, buffer_m, subject_kind, map_id))
        return {"warnings": ["evt unavailable: X"], "dem_resolution_m": 10.0}

    monkeypatch.setattr("sar_geo_mcp.staging.stage_terrain", fake_stage)


@pytest.mark.parametrize("radius,unit,metres", [
    (0.5, "mi", 804.672), (1.2, "km", 1200.0), (2640, "ft", 804.672), (880, "yd", 804.672),
    (750, "m", 750.0), (0.5, " MI ", 804.672),
])
def test_a_radius_in_any_unit_is_the_same_ground_in_metres(radius, unit, metres):
    area = build_area(center_latitude=PLS[0], center_longitude=PLS[1],
                      radius=radius, radius_unit=unit)
    in_metres = build_area(center_latitude=PLS[0], center_longitude=PLS[1], radius_m=metres)
    assert area.radius_m == pytest.approx(metres)
    assert area.key == in_metres.key          # half a mile and 804.672 m are one handle


def test_a_radius_is_given_once_in_a_known_unit():
    with pytest.raises(ValueError, match="once"):
        build_area(center_latitude=PLS[0], center_longitude=PLS[1], radius_m=800,
                   radius=0.5, radius_unit="mi")
    with pytest.raises(ValueError, match="radius_unit"):
        build_area(center_latitude=PLS[0], center_longitude=PLS[1], radius=1,
                   radius_unit="league")


@pytest.mark.parametrize("unit", ["km", "mi", "ft"])
def test_a_unit_beside_radius_m_is_refused_not_dropped(unit):
    """radius_m is metres by its name: radius_m=2 with radius_unit "km" is
    two kilometres meant, and read as metres it stages a circle a thousand
    times too small. Metres said outright, in any case, are no conflict."""
    with pytest.raises(ValueError, match=f"radius_unit '{unit}' applies to radius, not to "
                                         "radius_m"):
        build_area(center_latitude=PLS[0], center_longitude=PLS[1], radius_m=2,
                   radius_unit=unit)
    area = build_area(center_latitude=PLS[0], center_longitude=PLS[1], radius_m=800,
                      radius_unit=" M ")
    assert area.radius_m == 800


def test_a_unit_it_does_not_know_is_refused_with_no_radius_too():
    with pytest.raises(ValueError, match="radius_unit 'furlong' is not one of"):
        build_area(center_latitude=PLS[0], center_longitude=PLS[1], radius_m=800,
                   radius_unit="furlong")


def test_prepare_area_returns_the_range_to_draw(monkeypatch, tmp_path):
    stub_stage(monkeypatch, [])
    reader = FakeReader(tmp_path / "cache")
    for _ in range(2):                         # freshly staged, then reused
        result = prepare_area(reader, center_latitude=PLS[0], center_longitude=PLS[1],
                              radius=0.5, radius_unit="mi")
        area = result["area"]
        assert area["kind"] == "circle"
        assert area["radius_m"] == pytest.approx(804.672)
        assert area["center"] == {"latitude": PLS[0], "longitude": PLS[1]}
        assert area["geometry"]["type"] == "Polygon"
        west, south, east, north = _bounds(area["geometry"])
        assert (north - south) * 111_320 == pytest.approx(2 * 804.672, abs=15)


def test_prepare_area_stages_once_and_reports(monkeypatch, tmp_path):
    staged = []
    stub_stage(monkeypatch, staged)
    reader = FakeReader(tmp_path / "cache")
    first = prepare_area(reader, center_latitude=PLS[0], center_longitude=PLS[1],
                         radius_m=1500, map_id="MAP1")
    assert first["staged"] is True and first["reason"] == "not staged before"
    assert first["handle"].startswith(AREA_PREFIX)
    assert first["extent_m"] == pytest.approx(3000, abs=30)
    assert first["warnings"] == ["evt unavailable: X"]
    assert first["map_provenance"] == {"source": "test", "age_seconds": 0}
    assert staged == [(first["handle"], PLANNING_AREA_BUFFER_M, "area", "MAP1")]

    second = prepare_area(reader, center_latitude=PLS[0], center_longitude=PLS[1],
                          radius_m=1500, map_id="MAP1")
    assert second["staged"] is False and second["reason"] == "already staged"
    assert second["handle"] == first["handle"]
    assert len(staged) == 1
    # stub_stage writes a current osm.json, so reusing it is a file read,
    # never a fetch: no "topped up" warning joins the persisted one.
    assert second["warnings"] == ["evt unavailable: X"]


def test_prepare_area_reuse_does_not_touch_overpass_when_osm_is_current(monkeypatch, tmp_path):
    """The path every reuse of every staged directory runs through must cost
    a file read, never an Overpass round trip. If a later change makes the
    query-version check eager, this fails on the forbidden fetchers below,
    not on a wrong answer."""
    from sar_geo_mcp.terrain import linear

    def boom(*args, **kwargs):
        raise AssertionError("Overpass must not be reached for a current directory")

    monkeypatch.setattr(linear, "routes_for", boom)
    monkeypatch.setattr(linear, "parking_for", boom)
    monkeypatch.setattr(linear, "gates_for", boom)
    stub_stage(monkeypatch, [])
    reader = FakeReader(tmp_path / "cache")
    handle = prepare_area(reader, geometry=_square(PLS, 400))["handle"]

    result = prepare_area(reader, geometry=_square(PLS, 400))
    assert result["reason"] == "already staged" and result["handle"] == handle


def test_prepare_area_reuse_tops_up_a_directory_left_behind_by_an_older_query(
        monkeypatch, tmp_path):
    """A directory staged before 0.11.0 holds a version-2 osm.json with no
    parking in it. Reusing it must not require force=True to fix that."""
    from sar_geo_mcp.terrain import linear

    stub_stage(monkeypatch, [])
    reader = FakeReader(tmp_path / "cache")
    handle = prepare_area(reader, geometry=_square(PLS, 400))["handle"]
    from sar_geo_mcp.planning.area import area_directory
    directory = area_directory(reader.cache_dir, handle)
    (directory / "osm.json").write_text(
        json.dumps({"routes": [{"id": "old"}], "query_version": 2}), encoding="utf-8")
    monkeypatch.setattr(linear, "routes_for", lambda bbox, cache_dir: [{"id": "route_0"}])
    monkeypatch.setattr(linear, "parking_for", lambda bbox, cache_dir: [{"id": "parking-w1"}])
    monkeypatch.setattr(linear, "gates_for", lambda bbox, cache_dir: [])

    result = prepare_area(reader, geometry=_square(PLS, 400))

    assert result["reason"] == "already staged"
    assert any("topped it up" in w for w in result["warnings"])
    payload = json.loads((directory / "osm.json").read_text(encoding="utf-8"))
    assert payload["query_version"] == linear.OSM_QUERY_VERSION
    assert payload["parking"] == [{"id": "parking-w1"}]


def test_prepare_area_without_a_map_reads_nothing(monkeypatch, tmp_path):
    staged = []
    stub_stage(monkeypatch, staged)
    reader = FakeReader(tmp_path / "cache")
    result = prepare_area(reader, geometry=_square(PLS, 400))
    assert result["map_provenance"] is None
    assert reader.reads == []


def test_prepare_area_offline_map_is_a_warning_not_a_failure(monkeypatch, tmp_path):
    stub_stage(monkeypatch, [])
    reader = FakeReader(tmp_path / "cache", offline=True)
    result = prepare_area(reader, geometry=_square(PLS, 400), map_id="MAP1")
    assert result["staged"] is True
    assert result["map_provenance"] is None
    assert any("could not read map MAP1" in w for w in result["warnings"])


def test_prepare_area_force_restages(monkeypatch, tmp_path):
    staged = []
    stub_stage(monkeypatch, staged)
    reader = FakeReader(tmp_path / "cache")
    prepare_area(reader, geometry=_square(PLS, 400))
    result = prepare_area(reader, geometry=_square(PLS, 400), force=True)
    assert result["staged"] is True and result["reason"] == "forced"
    assert len(staged) == 2


def test_prepare_area_needs_a_cache_dir(tmp_path):
    reader = FakeReader(None)
    with pytest.raises(ValueError, match="cache directory"):
        prepare_area(reader, geometry=_square(PLS, 400))


# --- osm_available: a fact from the file, not the warnings -------------------
#
# The app used to decide whether trails, roads and parking could be read
# from a staged directory by string-matching a hub warning constant - and
# the hub has THREE differently-worded constants all named
# `MISSING_OSM_WARNING` (boundary.py, parking.py, routes.py), so the app's
# match only ever caught one of them. `osm_available` replaces the guess
# with the fact `prepare_area` already has for free: whether `osm.json` is
# sitting in the directory it is about to hand back.

def stub_stage_osm_failed(monkeypatch, staged: list):
    """Like `stub_stage`, but for the fetch that died mid-way: no `osm.json`
    is written, exactly as a real `stage_terrain` leaves a directory when
    Overpass dies - routes and parking are written together or not at all.
    This is the shape of the 2026-09-21 directory, reproduced with no
    network call."""
    from sar_geo_mcp.staging import write_manifest

    def fake_stage(subject, map_id, out_dir, *, cache_dir, buffer_m, fingerprint,
                   access="public", subject_kind="area", **kwargs):
        from pathlib import Path

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "subject.json").write_text(subject.model_dump_json(), encoding="utf-8")
        write_manifest(out_dir, map_id=map_id, assignment_label=subject.label,
                       bbox=(0, 0, 1, 1), dem_bbox=(0, 0, 1, 1), dem_resolution_m=10.0,
                       latitude_deg=37.4, panels={},
                       warnings=["osm unavailable: ConnectTimeout: overpass-api.de"],
                       assignment_fingerprint=fingerprint, access=access,
                       subject_kind=subject_kind)
        staged.append((subject.label, buffer_m, subject_kind, map_id))
        return {"warnings": ["osm unavailable: ConnectTimeout: overpass-api.de"],
                "dem_resolution_m": 10.0}

    monkeypatch.setattr("sar_geo_mcp.staging.stage_terrain", fake_stage)


def test_freshly_staged_osm_available_when_the_fetch_succeeded(monkeypatch, tmp_path):
    stub_stage(monkeypatch, [])
    reader = FakeReader(tmp_path / "cache")
    result = prepare_area(reader, geometry=_square(PLS, 400))
    assert result["staged"] is True
    assert result["osm_available"] is True


def test_freshly_staged_osm_unavailable_when_the_fetch_failed(monkeypatch, tmp_path):
    stub_stage_osm_failed(monkeypatch, [])
    reader = FakeReader(tmp_path / "cache")
    result = prepare_area(reader, geometry=_square(PLS, 400))
    assert result["staged"] is True
    assert result["osm_available"] is False


def test_already_staged_osm_available_reflects_the_file_present(monkeypatch, tmp_path):
    """The already-staged branch is the one that matters most in practice:
    it is the one a retry goes through. A directory that already holds a
    current `osm.json` must say so without touching Overpass at all."""
    from sar_geo_mcp.terrain import linear

    def boom(*args, **kwargs):
        raise AssertionError("a current osm.json must not touch Overpass")

    monkeypatch.setattr(linear, "routes_for", boom)
    monkeypatch.setattr(linear, "parking_for", boom)
    monkeypatch.setattr(linear, "gates_for", boom)
    stub_stage(monkeypatch, [])
    reader = FakeReader(tmp_path / "cache")
    prepare_area(reader, geometry=_square(PLS, 400))

    result = prepare_area(reader, geometry=_square(PLS, 400))
    assert result["staged"] is False and result["reason"] == "already staged"
    assert result["osm_available"] is True


def test_already_staged_osm_unavailable_when_the_retry_still_fails(monkeypatch, tmp_path):
    """The retry a searcher's page offers goes through this same branch. If
    Overpass is still down, `refresh_osm` leaves the missing file missing,
    and the answer must still come from the file, not from whatever
    `refresh_osm` said about the attempt."""
    from sar_geo_mcp.planning.area import area_directory
    from sar_geo_mcp.staging import write_manifest
    from sar_geo_mcp.terrain import linear

    reader = FakeReader(tmp_path / "cache")
    directory = area_directory(reader.cache_dir, build_area(geometry=_square(PLS, 400)).key)
    directory.mkdir(parents=True)
    write_manifest(directory, map_id="", assignment_label="aoi-x", bbox=(0, 0, 1, 1),
                   dem_bbox=(0, 0, 1, 1), dem_resolution_m=10.0, latitude_deg=37.4,
                   panels={}, warnings=[], subject_kind="area")

    def boom(*args, **kwargs):
        raise RuntimeError("mirror unreachable")

    monkeypatch.setattr(linear, "routes_for", boom)

    result = prepare_area(reader, geometry=_square(PLS, 400))
    assert result["reason"] == "already staged"
    assert result["osm_available"] is False
    assert not (directory / "osm.json").exists()


def test_osm_available_is_read_from_the_file_not_the_warnings(monkeypatch, tmp_path):
    """THE CASE THE COMMENT IN `prepare_area` IS ABOUT. This directory
    carries a warning about an outage AND, because `refresh_osm` just
    topped it up during this very call, a complete `osm.json`. A check that
    only looked at the warnings - which is what the app used to do, via a
    hub constant - would call this ground unreadable when it just became
    readable again.

    Until 0.17.1 the outage warning also came back here, beside the field
    saying the opposite; the top-up now retires it (see
    `test_an_outage_then_a_successful_top_up_stops_repeating_the_outage`)."""
    from sar_geo_mcp.planning.area import area_directory
    from sar_geo_mcp.staging import is_osm_outage_warning, write_manifest
    from sar_geo_mcp.terrain import linear

    reader = FakeReader(tmp_path / "cache")
    directory = area_directory(reader.cache_dir, build_area(geometry=_square(PLS, 400)).key)
    directory.mkdir(parents=True)
    write_manifest(directory, map_id="", assignment_label="aoi-x", bbox=(0, 0, 1, 1),
                   dem_bbox=(0, 0, 1, 1), dem_resolution_m=10.0, latitude_deg=37.4,
                   panels={}, warnings=["osm unavailable: ConnectTimeout: overpass-api.de"],
                   subject_kind="area")
    (directory / "osm.json").write_text(
        json.dumps({"routes": [{"id": "old"}], "query_version": 2}), encoding="utf-8")
    monkeypatch.setattr(linear, "routes_for", lambda bbox, cache_dir: [{"id": "route_0"}])
    monkeypatch.setattr(linear, "parking_for", lambda bbox, cache_dir: [{"id": "parking-w1"}])
    monkeypatch.setattr(linear, "gates_for", lambda bbox, cache_dir: [])

    result = prepare_area(reader, geometry=_square(PLS, 400))

    assert result["reason"] == "already staged"
    assert any("topped it up" in w for w in result["warnings"])
    assert not any(is_osm_outage_warning(w) for w in result["warnings"])
    assert result["osm_available"] is True


def test_an_outage_then_a_successful_top_up_stops_repeating_the_outage(monkeypatch, tmp_path):
    """The Safety page shows planners every warning `prepare_area` returns.
    Ground first staged while Overpass was down must say so - until a
    top-up has fetched the file, and never again after. Every other
    warning the manifest carries is still returned."""
    from sar_geo_mcp.staging import is_osm_outage_warning, osm_outage_warning, write_manifest
    from sar_geo_mcp.terrain import linear
    from sar_geo_mcp.terrain.linear import OverpassUnavailable

    outage = osm_outage_warning("OverpassUnavailable: every Overpass endpoint refused")
    other = "evt unavailable: OSError"

    def stage_during_outage(subject, map_id, out_dir, *, cache_dir, buffer_m, fingerprint,
                            access="public", subject_kind="area", **kwargs):
        # What a real `stage_terrain` leaves when Overpass dies: no osm.json,
        # the outage recorded in the manifest beside any other warning.
        from pathlib import Path

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "subject.json").write_text(subject.model_dump_json(), encoding="utf-8")
        write_manifest(out_dir, map_id=map_id, assignment_label=subject.label,
                       bbox=(0, 0, 1, 1), dem_bbox=(0, 0, 1, 1), dem_resolution_m=10.0,
                       latitude_deg=37.4, panels={}, warnings=[outage, other],
                       assignment_fingerprint=fingerprint, access=access,
                       subject_kind=subject_kind)
        return {"warnings": [outage, other], "dem_resolution_m": 10.0}

    monkeypatch.setattr("sar_geo_mcp.staging.stage_terrain", stage_during_outage)

    def refuse(bbox, cache_dir):
        raise OverpassUnavailable("every Overpass endpoint refused", status=429)

    reader = FakeReader(tmp_path / "cache")
    area = _square(PLS, 400)

    first = prepare_area(reader, geometry=area)
    assert first["staged"] is True and first["osm_available"] is False
    assert first["warnings"] == [outage, other]

    # Overpass still down: the outage is still true, and still said.
    monkeypatch.setattr(linear, "routes_for", refuse)
    still_down = prepare_area(reader, geometry=area)
    assert still_down["osm_available"] is False
    assert outage in still_down["warnings"] and other in still_down["warnings"]

    # Overpass back: the top-up says what it did, and the outage is retired.
    monkeypatch.setattr(linear, "routes_for", lambda bbox, cache_dir: [])
    monkeypatch.setattr(linear, "parking_for", lambda bbox, cache_dir: [])
    monkeypatch.setattr(linear, "gates_for", lambda bbox, cache_dir: [])
    recovered = prepare_area(reader, geometry=area)
    assert recovered["osm_available"] is True
    assert not any(is_osm_outage_warning(w) for w in recovered["warnings"])
    assert any("was missing from this ground" in w for w in recovered["warnings"])
    assert other in recovered["warnings"]

    # And it stays retired: the file is current, so nothing is fetched and
    # the only thing left to say is the warning that is still true.
    def boom(*args, **kwargs):
        raise AssertionError("a current osm.json must not touch Overpass")

    monkeypatch.setattr(linear, "routes_for", boom)
    monkeypatch.setattr(linear, "parking_for", boom)
    monkeypatch.setattr(linear, "gates_for", boom)
    later = prepare_area(reader, geometry=area)
    assert later["warnings"] == [other]
    assert later["osm_available"] is True


def test_a_version_3_file_under_a_stale_outage_drops_the_outage_and_keeps_the_refresh_warning(
        monkeypatch, tmp_path):
    """Staged during an outage, topped up to version 3 by an older hub, now
    behind version 4 with Overpass down again. "osm unavailable" is false -
    the file is there - and goes; what `refresh_osm` says about the version
    it could not reach is true, and stays."""
    from sar_geo_mcp.planning.area import area_directory
    from sar_geo_mcp.staging import is_osm_outage_warning, write_manifest
    from sar_geo_mcp.terrain import linear
    from sar_geo_mcp.terrain.linear import OverpassUnavailable

    reader = FakeReader(tmp_path / "cache")
    directory = area_directory(reader.cache_dir, build_area(geometry=_square(PLS, 400)).key)
    directory.mkdir(parents=True)
    write_manifest(directory, map_id="", assignment_label="aoi-x", bbox=(0, 0, 1, 1),
                   dem_bbox=(0, 0, 1, 1), dem_resolution_m=10.0, latitude_deg=37.4,
                   panels={}, warnings=["osm unavailable: ConnectTimeout: overpass-api.de",
                                        "evt unavailable: OSError"],
                   subject_kind="area")
    _record_a_top_up(directory, 3)  # what that older hub's top-up to 3 wrote
    (directory / "osm.json").write_text(json.dumps(
        {"routes": [], "parking": [{"id": "p"}], "query_version": 3}), encoding="utf-8")

    def refuse(bbox, cache_dir):
        raise OverpassUnavailable("every Overpass endpoint refused", status=429)

    monkeypatch.setattr(linear, "routes_for", refuse)

    result = prepare_area(reader, geometry=_square(PLS, 400))

    assert not any(is_osm_outage_warning(w) for w in result["warnings"])
    assert "evt unavailable: OSError" in result["warnings"]
    refresh = [w for w in result["warnings"] if w.startswith("osm.json is still at query version 3")]
    assert len(refresh) == 1 and "minor_line" in refresh[0]
    assert result["osm_available"] is True


def test_ground_healed_before_0_17_1_stops_repeating_its_outage(monkeypatch, tmp_path):
    """0.17.0 topped such ground up but left the outage in its manifest, and
    a current `osm.json` is never refreshed again - so the manifest on disk
    will not be rewritten. Reading it must not return the stale warning."""
    from sar_geo_mcp.planning.area import area_directory
    from sar_geo_mcp.staging import write_manifest
    from sar_geo_mcp.terrain import linear

    reader = FakeReader(tmp_path / "cache")
    directory = area_directory(reader.cache_dir, build_area(geometry=_square(PLS, 400)).key)
    directory.mkdir(parents=True)
    write_manifest(directory, map_id="", assignment_label="aoi-x", bbox=(0, 0, 1, 1),
                   dem_bbox=(0, 0, 1, 1), dem_resolution_m=10.0, latitude_deg=37.4,
                   panels={}, warnings=["osm unavailable: ConnectTimeout: overpass-api.de",
                                        "evt unavailable: OSError"],
                   subject_kind="area")
    _record_a_top_up(directory, linear.OSM_QUERY_VERSION)  # what 0.17.0's top-up wrote
    (directory / "osm.json").write_text(json.dumps(
        {"routes": [], "parking": [], "query_version": linear.OSM_QUERY_VERSION}),
        encoding="utf-8")

    result = prepare_area(reader, geometry=_square(PLS, 400))

    assert result["warnings"] == ["evt unavailable: OSError"]
    assert result["osm_available"] is True


def test_an_areas_osm_json_is_its_own_ground_whatever_wrote_it(monkeypatch, tmp_path):
    """An area's key hashes its geometry and buffer, so its directory has one
    bbox for life: a file beside an outage there can only be an older hub's
    forced re-stage during an outage, or a staging that crashed between
    writing osm.json and the manifest - the same ground either way. No top-up
    ever wrote it (no `osm_refreshed_versions`), and still the outage is the
    stale half: the response must not say "available" and "unavailable" at
    once, and must not keep saying it on every reuse."""
    from sar_geo_mcp.planning.area import area_directory
    from sar_geo_mcp.staging import is_osm_outage_warning, write_manifest
    from sar_geo_mcp.terrain import linear

    reader = FakeReader(tmp_path / "cache")
    directory = area_directory(reader.cache_dir, build_area(geometry=_square(PLS, 400)).key)
    directory.mkdir(parents=True)
    write_manifest(directory, map_id="", assignment_label="aoi-x", bbox=(0, 0, 1, 1),
                   dem_bbox=(0, 0, 1, 1), dem_resolution_m=10.0, latitude_deg=37.4,
                   panels={}, warnings=["osm unavailable: ConnectTimeout: overpass-api.de"],
                   subject_kind="area")
    (directory / "osm.json").write_text(json.dumps(
        {"routes": [{"id": "old-boundary"}], "parking": [],
         "query_version": linear.OSM_QUERY_VERSION}), encoding="utf-8")

    for _ in range(2):
        result = prepare_area(reader, geometry=_square(PLS, 400))
        assert not any(is_osm_outage_warning(w) for w in result["warnings"])
        assert result["osm_available"] is True


def test_resolve_an_area_handle_needs_no_network(monkeypatch, tmp_path):
    stub_stage(monkeypatch, [])
    reader = FakeReader(tmp_path / "cache")
    handle = prepare_area(reader, geometry=_square(PLS, 400))["handle"]
    reader.reads.clear()
    resolved = resolve(handle, reader)
    assert resolved.status == {
        "current": True, "checked": True, "reason": "area input is immutable",
        "map": "none",
        "map_reason": ("this area was staged without a map_id, so there are "
                       "no assignments or tracks to compare against"),
    }
    assert resolved.state is None
    assert reader.reads == []
    assert (resolved.directory / "manifest.json").is_file()


def test_resolve_an_area_handle_with_a_map_reads_it_fresh_and_carries_provenance(monkeypatch, tmp_path):
    stub_stage(monkeypatch, [])
    reader = FakeReader(tmp_path / "cache")
    handle = prepare_area(reader, geometry=_square(PLS, 400), map_id="MAP1", public=False)["handle"]
    reader.reads.clear()
    resolved = resolve(handle, reader)
    assert resolved.status["current"] is True
    assert resolved.status["map_provenance"] == {"source": "test", "age_seconds": 0}
    assert resolved.state is not None
    assert reader.reads == [("MAP1", False, 0)]


def test_resolve_an_area_handle_offline_says_so(monkeypatch, tmp_path):
    stub_stage(monkeypatch, [])
    reader = FakeReader(tmp_path / "cache")
    handle = prepare_area(reader, geometry=_square(PLS, 400), map_id="MAP1")["handle"]
    reader.offline = True
    resolved = resolve(handle, reader)
    assert resolved.status["current"] is True
    assert resolved.state is None
    assert "could not read map MAP1" in resolved.status["warnings"][0]


def test_resolve_refuses_an_area_handle_that_was_never_staged(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_area"):
        resolve("aoi-0123456789abcdef", FakeReader(tmp_path))


def test_resolve_refuses_a_malformed_area_handle(tmp_path):
    with pytest.raises(ValueError, match="not an area handle"):
        resolve("aoi-../../etc", FakeReader(tmp_path))


def test_resolve_an_area_without_a_map_says_why_there_is_no_map_state(monkeypatch, tmp_path):
    stub_stage(monkeypatch, [])
    reader = FakeReader(tmp_path / "cache")
    handle = prepare_area(reader, geometry=_square(PLS, 400))["handle"]
    status = resolve(handle, reader).status
    assert status["map"] == "none"
    assert "staged without a map_id" in status["map_reason"]
    assert status["reason"] == "area input is immutable"


def test_resolve_an_area_with_a_map_marks_it_read_or_unreadable(monkeypatch, tmp_path):
    stub_stage(monkeypatch, [])
    reader = FakeReader(tmp_path / "cache")
    handle = prepare_area(reader, geometry=_square(PLS, 400), map_id="MAP1")["handle"]
    assert resolve(handle, reader).status["map"] == "read"
    reader.offline = True
    status = resolve(handle, reader).status
    assert status["map"] == "unreadable" and "could not read map MAP1" in status["map_reason"]


def test_map_id_empty_and_none_share_a_key():
    assert area_key("circle", [PLS[0], PLS[1]], 100.0, "") == area_key("circle", [PLS[0], PLS[1]], 100.0, None)


# --- the planning buffer -------------------------------------------------------

def test_the_planning_buffer_is_part_of_the_key():
    """The buffer decides how much ground is staged, so two stagings that
    differ only by it are two different rasters and must not share a
    directory: the old handle would otherwise keep serving a raster 500 m
    smaller than the one its consumer now measures against."""
    narrow = area_key("circle", [PLS[0], PLS[1]], 1500.0, "MAP1", buffer_m=500.0)
    wide = area_key("circle", [PLS[0], PLS[1]], 1500.0, "MAP1", buffer_m=1000.0)
    assert narrow != wide
    # The default is the planning buffer, so a caller who names none gets the
    # key of the staging `prepare_area` actually makes.
    assert area_key("circle", [PLS[0], PLS[1]], 1500.0, "MAP1") == area_key(
        "circle", [PLS[0], PLS[1]], 1500.0, "MAP1", buffer_m=PLANNING_AREA_BUFFER_M)


def test_a_planning_area_is_staged_with_a_kilometre_of_buffer(monkeypatch, tmp_path):
    """Measured at Windy Hill on a 1,500 m area: the four trails entering the
    circle cut off 3,073 m between them and the farthest end of any of them
    was 2,242 m from the centre, so 1,000 m of buffer covers every one with
    room to spare."""
    staged = []
    stub_stage(monkeypatch, staged)
    reader = FakeReader(tmp_path / "cache")
    result = prepare_area(reader, center_latitude=PLS[0], center_longitude=PLS[1],
                          radius_m=1500)
    assert PLANNING_AREA_BUFFER_M == 1000.0
    assert staged == [(result["handle"], 1000.0, "area", "")]


def test_the_manifest_records_the_buffer_the_raster_was_staged_with(tmp_path):
    """A directory that cannot say how much margin it holds cannot be checked
    against the buffer its consumer assumed."""
    from sar_geo_mcp.staging import write_manifest

    path = write_manifest(tmp_path, map_id="M", assignment_label="aoi-1", bbox=(0, 0, 1, 1),
                          dem_bbox=(0, 0, 1, 1), dem_resolution_m=10.0, latitude_deg=37.4,
                          panels={}, warnings=[], buffer_m=1000.0)
    assert json.loads(path.read_text(encoding="utf-8"))["buffer_m"] == 1000.0


# --- helpers -------------------------------------------------------------------

def _record_a_top_up(directory, version):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    manifest["osm_refreshed_versions"] = [version]
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _cos(latitude):
    import math
    return math.cos(math.radians(latitude))


def _bounds(geometry):
    xs = [c[0] for c in geometry["coordinates"][0]]
    ys = [c[1] for c in geometry["coordinates"][0]]
    return min(xs), min(ys), max(xs), max(ys)


def _square(center, half_m):
    lat, lon = center
    dlat = half_m / 111_320
    dlon = half_m / (111_320 * _cos(lat))
    return {"type": "Polygon", "coordinates": [[
        [lon - dlon, lat - dlat], [lon + dlon, lat - dlat],
        [lon + dlon, lat + dlat], [lon - dlon, lat + dlat], [lon - dlon, lat - dlat]]]}


# --- a forced re-stage during an outage never contradicts itself --------------

def _real_staging_with_overpass(monkeypatch, tmp_path, up: bool):
    from test_staging_subject import stub_fetchers

    from sar_geo_mcp.terrain.linear import OverpassUnavailable

    stub_fetchers(monkeypatch, tmp_path)
    if not up:
        def refuse(bbox, cache_dir):
            raise OverpassUnavailable("every Overpass endpoint refused", status=429)

        monkeypatch.setattr("sar_geo_mcp.terrain.linear.routes_for", refuse)


def test_a_forced_restage_of_the_same_ground_in_an_outage_keeps_the_file_and_says_so(
        monkeypatch, tmp_path):
    from sar_geo_mcp.staging import is_osm_kept_warning, is_osm_outage_warning

    reader = FakeReader(tmp_path / "cache")
    _real_staging_with_overpass(monkeypatch, tmp_path, up=True)
    first = prepare_area(reader, geometry=_square(PLS, 400))
    assert first["osm_available"] is True

    _real_staging_with_overpass(monkeypatch, tmp_path, up=False)
    forced = prepare_area(reader, geometry=_square(PLS, 400), force=True)

    assert forced["staged"] is True and forced["reason"] == "forced"
    assert forced["osm_available"] is True
    assert not any(is_osm_outage_warning(w) for w in forced["warnings"])
    assert any(w.startswith("the OSM refetch failed (OverpassUnavailable")
               and "for this same ground is kept" in w for w in forced["warnings"])

    kept = [w for w in forced["warnings"] if is_osm_kept_warning(w)]
    assert len(kept) == 1

    for _ in range(3):
        reused = prepare_area(reader, geometry=_square(PLS, 400))
        assert reused["osm_available"] is True
        assert not any(is_osm_outage_warning(w) for w in reused["warnings"])
        assert not any(is_osm_kept_warning(w) for w in reused["warnings"])


def test_a_forced_restage_over_other_ground_in_an_outage_reports_no_osm(monkeypatch, tmp_path):
    """The directory holds an osm.json fetched for a different box - the
    shape a moved assignment boundary leaves. It must not survive a re-stage
    that could not fetch this ground's own."""
    from sar_geo_mcp.planning.area import area_directory
    from sar_geo_mcp.staging import is_osm_outage_warning, write_manifest
    from sar_geo_mcp.terrain.linear import OSM_QUERY_VERSION

    reader = FakeReader(tmp_path / "cache")
    directory = area_directory(reader.cache_dir, build_area(geometry=_square(PLS, 400)).key)
    directory.mkdir(parents=True)
    write_manifest(directory, map_id="", assignment_label="aoi-x",
                   bbox=(0.0, 0.0, 100.0, 100.0), dem_bbox=(0.0, 0.0, 100.0, 100.0),
                   dem_resolution_m=10.0, latitude_deg=37.4, panels={}, warnings=[],
                   subject_kind="area")
    # The current version, so the only thing wrong with the file is its ground.
    (directory / "osm.json").write_text(json.dumps({"routes": [{"id": "elsewhere"}],
                                                    "query_version": OSM_QUERY_VERSION}),
                                        encoding="utf-8")

    _real_staging_with_overpass(monkeypatch, tmp_path, up=False)
    forced = prepare_area(reader, geometry=_square(PLS, 400), force=True)

    assert forced["osm_available"] is False
    assert any(is_osm_outage_warning(w) for w in forced["warnings"])
    assert not (directory / "osm.json").exists()


def test_prepare_area_over_an_unreadable_osm_json_fetches_it_again(monkeypatch, tmp_path):
    """Used to fail the whole call with JSONDecodeError."""
    from sar_geo_mcp.planning.area import area_directory
    from sar_geo_mcp.staging import write_manifest
    from sar_geo_mcp.terrain import linear

    reader = FakeReader(tmp_path / "cache")
    directory = area_directory(reader.cache_dir, build_area(geometry=_square(PLS, 400)).key)
    directory.mkdir(parents=True)
    write_manifest(directory, map_id="", assignment_label="aoi-x", bbox=(0, 0, 1, 1),
                   dem_bbox=(0, 0, 1, 1), dem_resolution_m=10.0, latitude_deg=37.4,
                   panels={}, warnings=[], subject_kind="area")
    (directory / "osm.json").write_text("{trunc", encoding="utf-8")
    monkeypatch.setattr(linear, "routes_for", lambda bbox, cache_dir: [])
    monkeypatch.setattr(linear, "parking_for", lambda bbox, cache_dir: [])
    monkeypatch.setattr(linear, "gates_for", lambda bbox, cache_dir: [])

    result = prepare_area(reader, geometry=_square(PLS, 400))

    assert result["osm_available"] is True
    assert any("could not be read" in w for w in result["warnings"])


# --- the caller's `public` on ground already staged ---------------------------
#
# The area key hashes the map id but not `public`, so a call corrected from
# public=True to public=False lands on the ground the first call staged. The
# map is then read by whichever path that ground records.

def _scenario_context(tmp_path):
    """A `ServerContext` over a recording `FakeReader`, for a scenario run
    through the real `prepare_area` (with `stub_stage`) and real handles."""
    from sar_geo_mcp.context import ServerContext

    reader = FakeReader(tmp_path / "cache")
    return ServerContext(config=CaltopoConfig(cache_dir=tmp_path / "cache"),
                         reader=reader), reader


def _stub_hasty_resolving(monkeypatch, feature_reads: list):
    """`hasty_candidates` cut down to how it starts - `ctx.staged(handle)`,
    the resolve that reads the map by the path the staged ground records -
    then an empty set. Each call appends the map reads its resolve made."""
    from sar_geo_mcp.features import hasty

    def resolving(ctx, handle, **kwargs):
        before = len(ctx.reader.reads)
        ctx.staged(handle)
        feature_reads.append(ctx.reader.reads[before:])
        return {"set_id": "hasty-000000", "candidates": [], "summary": {}, "warnings": []}

    monkeypatch.setattr(hasty, "hasty_candidates", resolving)


@pytest.mark.parametrize("public", [True, False])
def test_a_scenario_s_public_flag_reaches_every_map_read_on_new_ground(
        monkeypatch, tmp_path, public):
    """`map_id` and `public` are passed to staging as prepare_area takes them,
    and the feature's own read of the map then goes by the same path: a
    private map read by the public endpoint fails, and overlap with the
    map's assignments is lost with it."""
    from sar_geo_mcp.scenarios.initial_response import plan_initial_response

    stub_stage(monkeypatch, [])
    ctx, reader = _scenario_context(tmp_path)
    feature_reads = []
    _stub_hasty_resolving(monkeypatch, feature_reads)

    result = plan_initial_response(ctx, PLS[0], PLS[1], map_id="M", public=public,
                                   include=["hasty"])

    assert result["sections"]["hasty"]["status"] == "ok"
    search = result["staging"]["search"]
    assert search["staged"] is True
    assert search["detail_call"]["arguments"]["public"] is public
    # prepare_area's provenance read, then the feature's resolve.
    assert reader.reads == [("M", public, 0), ("M", public, 0)]
    assert feature_reads == [[("M", public, 0)]]


def test_a_corrected_public_flag_reaches_the_feature_on_ground_already_staged(
        monkeypatch, tmp_path):
    """plan_initial_response on a private map with `public` left true by
    mistake, then the corrected call. The corrected call is answered from the
    ground already staged - and its hasty section must still read the map
    the way that call asked."""
    from sar_geo_mcp.scenarios.initial_response import plan_initial_response

    stub_stage(monkeypatch, [])
    ctx, _reader = _scenario_context(tmp_path)
    feature_reads = []
    _stub_hasty_resolving(monkeypatch, feature_reads)

    first = plan_initial_response(ctx, PLS[0], PLS[1], map_id="M", public=True,
                                  include=["hasty"])
    corrected = plan_initial_response(ctx, PLS[0], PLS[1], map_id="M", public=False,
                                      include=["hasty"])

    assert corrected["staging"]["search"]["staged"] is False
    assert corrected["handles"] == first["handles"]
    assert feature_reads[-1] == [("M", False, 0)]


def test_prepare_area_records_the_access_path_of_the_latest_call(monkeypatch, tmp_path):
    """The ground does not depend on how the map is read, so the corrected
    call needs no re-stage - only the recorded access path, which every later
    resolve reads the map by."""
    from sar_geo_mcp.handles import staged_access
    from sar_geo_mcp.planning.area import area_directory

    staged = []
    stub_stage(monkeypatch, staged)
    reader = FakeReader(tmp_path / "cache")
    first = prepare_area(reader, geometry=_square(PLS, 400), map_id="M1", public=True)
    corrected = prepare_area(reader, geometry=_square(PLS, 400), map_id="M1", public=False)

    assert corrected["staged"] is False and corrected["reason"] == "already staged"
    assert corrected["handle"] == first["handle"]
    assert len(staged) == 1
    reader.reads.clear()
    resolve(first["handle"], reader)
    assert reader.reads == [("M1", False, 0)]
    assert staged_access(area_directory(reader.cache_dir, first["handle"])) == "authenticated"


def test_a_later_call_whose_own_read_failed_keeps_the_path_that_reads_the_map(
        monkeypatch, tmp_path):
    """A private map staged with public=False, then another call on the same
    ground leaving `public` at its default true - another scenario at the
    same point, say. Its own read fails and it says so; the handle must go
    on reading the map by the path that works, not be moved onto the one
    that has just failed."""
    from sar_geo_mcp.handles import staged_access
    from sar_geo_mcp.planning.area import area_directory

    class PrivateMapReader(FakeReader):
        def read(self, map_id, *, public, max_age_seconds=300.0):
            self.reads.append((map_id, public, max_age_seconds))
            if public:
                raise PermissionError(f"{map_id} is not public")
            return FakeState()

    stub_stage(monkeypatch, [])
    reader = PrivateMapReader(tmp_path / "cache")
    first = prepare_area(reader, geometry=_square(PLS, 400), map_id="M1", public=False)
    defaulted = prepare_area(reader, geometry=_square(PLS, 400), map_id="M1")

    assert defaulted["reason"] == "already staged"
    assert defaulted["map_provenance"] is None
    assert any("could not read map M1" in w for w in defaulted["warnings"])
    assert staged_access(area_directory(reader.cache_dir, first["handle"])) == "authenticated"
    reader.reads.clear()
    resolved = resolve(first["handle"], reader)
    assert reader.reads == [("M1", False, 0)]
    assert resolved.status["map"] == "read"


# --- two calls staging the same ground at once --------------------------------
#
# fastmcp runs sync tools in a threadpool, and plan_medevac and
# plan_initial_response share the 2 m landing-zone ground by design, so two
# calls at one point are two threads in `prepare_area` for one directory.

@pytest.mark.xfail(strict=True, reason=(
    "PR #4 review B50: Two parallel tool calls on the same point stage the same "
    "directory twice at the same time"))
def test_two_calls_staging_the_same_ground_at_once_stage_it_once(monkeypatch, tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import sar_geo_mcp.staging as staging_module

    staged = []
    stub_stage(monkeypatch, staged)
    write = staging_module.stage_terrain
    entered = []
    both_in = threading.Event()

    def slow_stage(*args, **kwargs):
        # A real stage spends seconds fetching before it writes its manifest.
        # Held here until a second stager arrives - or two seconds when none
        # does, which is all a lock round the check-then-stage makes of it.
        entered.append(True)
        if len(entered) >= 2:
            both_in.set()
        both_in.wait(timeout=2.0)
        return write(*args, **kwargs)

    monkeypatch.setattr("sar_geo_mcp.staging.stage_terrain", slow_stage)
    reader = FakeReader(tmp_path / "cache")
    start = threading.Barrier(2)

    def landing_zone_ground():
        start.wait(timeout=5)
        return prepare_area(reader, center_latitude=PLS[0], center_longitude=PLS[1],
                            radius_m=1000, resolution_m=2.0)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(landing_zone_ground) for _ in range(2)]
        results = [future.result(timeout=30) for future in futures]

    assert len({result["handle"] for result in results}) == 1
    assert len(staged) == 1, f"staged {len(staged)} times"
    assert sorted((result["staged"], result["reason"]) for result in results) == [
        (False, "already staged"), (True, "not staged before")]


@pytest.mark.skipif(sys.platform == "win32",
                    reason="Windows refuses to replace a file another handle holds open")
@pytest.mark.xfail(strict=True, reason=(
    "PR #4 review B50: Two parallel tool calls on the same point stage the same "
    "directory twice at the same time (dem.tif and osm.json rewritten in place, "
    "not atomically)"))
def test_a_forced_restage_never_rewrites_a_staged_file_under_its_reader(
        monkeypatch, tmp_path):
    """A reader that opened the staged ground before a re-stage must read the
    whole of what it opened: the new ground lands beside the old and replaces
    it in one rename (`cache.atomic`), never by truncating the file in place
    while another call is reading it."""
    import numpy as np
    import rasterio
    from test_staging_subject import TRANSFORM

    from sar_geo_mcp.planning.area import area_directory

    reader = FakeReader(tmp_path / "cache")
    _real_staging_with_overpass(monkeypatch, tmp_path, up=True)
    handle = prepare_area(reader, geometry=_square(PLS, 400))["handle"]
    directory = area_directory(reader.cache_dir, handle)
    names = ("dem.tif", "osm.json")
    before = {name: (directory / name).read_bytes() for name in names}

    def higher_dem(bbox, width_px, height_px, *, cache_dir):
        path = tmp_path / "higher_dem.tif"
        with rasterio.open(path, "w", driver="GTiff", height=4, width=4, count=1,
                           dtype="float32", crs="EPSG:3857", transform=TRANSFORM) as dst:
            dst.write(np.full((4, 4), 250.0, dtype="float32"), 1)
        return path

    monkeypatch.setattr("sar_geo_mcp.staging.fetch_dem", higher_dem)
    monkeypatch.setattr("sar_geo_mcp.terrain.linear.routes_for",
                        lambda bbox, cache_dir: [{"id": "route_new"}])

    held = {name: (directory / name).open("rb") for name in names}
    try:
        forced = prepare_area(reader, geometry=_square(PLS, 400), force=True)
        assert forced["staged"] is True
        for name in names:
            # The re-stage did land new ground...
            assert (directory / name).read_bytes() != before[name], name
        for name, opened in held.items():
            # ...and not underneath the reader that had the old.
            assert opened.read() == before[name], f"{name} was rewritten in place"
    finally:
        for opened in held.values():
            opened.close()
