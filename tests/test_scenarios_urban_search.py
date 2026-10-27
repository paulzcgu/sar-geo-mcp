"""plan_urban_search: the populated blocks grouped into assignments, parking,
hospitals and the CalTopo draft, in one call round a place last seen.

No network. Staging and the feature functions are stubbed with monkeypatch
where they live (`components.staging.prepare_area`,
`features.urban.urban_assignments`, ...), which is where the scenario
reaches them. The urban_assignments stub saves a real candidate set into the
staged directory, so the export section runs the real `caltopo_export` over
it and writes a real file; the medical stub runs the real medical logic over
faked HIFLD, OSRM and Overpass answers. Two tests run the real
populated_areas and urban_assignments: one with no WMS token, which must
fail the urban section in the feature's own words and nothing else, and one
over the synthetic town `test_populated.py` draws, its CalTopo panels faked,
to hold the compact section to the payloads the features actually return.
The shared machinery has its own tests in `test_scenarios_sections.py`.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import httpx
import pytest
from test_medical import FakeServices, record
from test_scenarios_initial_response import AreaStaging, _square, _stage_without_osm
from test_scenarios_sections import FakeStaging, geometry_types_in

from sar_geo_mcp.components import candidates as candidates_component
from sar_geo_mcp.components import norms as norms_component
from sar_geo_mcp.components import staging as staging_component
from sar_geo_mcp.config import CaltopoConfig
from sar_geo_mcp.context import ServerContext
from sar_geo_mcp.features import medical as medical_feature
from sar_geo_mcp.features import parking as parking_feature
from sar_geo_mcp.features import urban as urban_feature
from sar_geo_mcp.planning.candidates import (
    Candidate,
    CandidateSet,
    load_set,
    save_set,
    summaries,
)
from sar_geo_mcp.planning.medical import (HOSPITAL_COLOR, ORIGIN_COLOR, RANKED_BY_ROAD,
                                         RANKED_STRAIGHT_ROUTER_DOWN)
from sar_geo_mcp.planning.medical import nearest_medical_facilities as medical_impl
from sar_geo_mcp.planning.norms import load_norms
from sar_geo_mcp.planning.populated import NO_TOKEN
from sar_geo_mcp.scenarios import _sections
from sar_geo_mcp.scenarios.initial_response import plan_initial_response
from sar_geo_mcp.scenarios.urban_search import plan_urban_search
from exports import read_export

SAN_CARLOS = (37.5072, -122.2605)
SEQUOIA = (37.48, -122.24)
STANFORD = (37.434, -122.175)
HALF_MILE = 804.672
URBAN_SET = "urban-c0ffee"
BLOCKS_SET = "populated-b10c55"
SECTIONS = ("urban", "parking", "medical", "export")
LOTS_WARNING = ("34 built-up lot(s) were in runs of fewer than 3 along their street and are "
                "in no block")
PIECES_WARNING = ("the outline is in 2 pieces (another group's ground runs between them); the "
                  "CalTopo export draws each piece as its own assignment")
WATER_WARNING = ("parts on either side of water are joined by a thin strip for the drawing "
                 "only; travel between them by road")
#: The real urban features, held before any test stands a stub in for them.
REAL_URBAN = {"populated_areas": urban_feature.populated_areas,
              "urban_assignments": urban_feature.urban_assignments}


def urban_set(handle: str, arguments: dict) -> CandidateSet:
    """Three assignments the way urban_assignments draws them: one plain,
    one in two pieces, one joined across water - each with its polygon,
    which the section must leave out, and house markers when asked for."""
    lat, lon = SAN_CARLOS
    markers = bool(arguments.get("building_markers"))

    def houses(count, dx):
        return [{"type": "Feature",
                 "geometry": {"type": "Point", "coordinates": [lon + dx, lat + 0.0001 * i]},
                 "properties": {"class": "Marker", "title": "", "marker-symbol": "hut",
                                "marker-color": "#FFFFFF", "marker-size": 1}}
                for i in range(count)] if markers else None

    def facts(streets, blocks, parcels, buildings, area_ha, distance, bearing, names, *,
              parts=1, water=0, dx=0.0):
        return {"streets": streets, "blocks": blocks, "parcels": parcels,
                "buildings": buildings, "area_ha": area_ha,
                "point": {"latitude": lat + 0.001, "longitude": lon + dx},
                "distance_from_center_m": distance, "bearing_from_center_deg": bearing,
                "water_links": water, "parts": parts, "block_names": names}

    rows = [
        ("Laurel Street +2", {"type": "Polygon", "coordinates": [_square(lon, lat, 0.0012)]},
         facts(["Laurel Street", "Olive Street", "Walnut Street"], 4, 120, 96, 9.7, 110.0,
               30.0, ["Laurel Street (1)", "Olive Street", "Walnut Street",
                      "Laurel Street (2)"]), [], 3, 0.0),
        ("El Camino Real",
         {"type": "MultiPolygon", "coordinates": [[_square(lon + 0.004, lat, 0.001)],
                                                  [_square(lon + 0.007, lat, 0.0005)]]},
         facts(["El Camino Real"], 3, 90, 50, 11.21, 420.0, 95.0,
               ["El Camino Real (1)", "El Camino Real (2)", None], parts=2, dx=0.004),
         [PIECES_WARNING], 2, 0.004),
        ("Bayport Avenue +1", {"type": "Polygon",
                               "coordinates": [_square(lon - 0.006, lat, 0.0008)]},
         facts(["Bayport Avenue", "Terminal Way"], 2, 106, 34, 4.2, 610.0, 270.0,
               ["Bayport Avenue (1)", "Terminal Way"], water=1, dx=-0.006),
         [WATER_WARNING], 1, -0.006),
    ]
    candidates = []
    for index, (name, geometry, fact, warnings, count, dx) in enumerate(rows, start=1):
        drawn = houses(count, dx)
        if drawn is not None:
            fact["markers"] = len(drawn)
        candidates.append(Candidate(
            id=f"{URBAN_SET}.{index}", kind="assignment", axis="urban", axis_ids=[],
            sources=["parcel", "structures", "osm"], name=name, geometry=geometry,
            facts=fact, norm_flags=[], warnings=warnings, markers=drawn))
    target = arguments.get("target_ha")
    summary = {"count": 3, "blocks": 9, "target_ha": target or 10.0,
               "area_ha": {"min": 4.2, "median": 9.7, "max": 11.21, "total": 25.1},
               "parcels": 316, "buildings": 180, "parcels_left_out": 34,
               "center": {"latitude": lat, "longitude": lon}}
    if markers:
        summary["markers"] = 6
    return CandidateSet(
        set_id=URBAN_SET, tool="urban_assignments", params={"target_ha": target},
        handle=handle, created_at="2026-10-22T09:00:00+00:00", candidates=candidates,
        summary=summary,
        assumptions={"pixel_m": 1.0, "grouping": {
            "target_ha": target or 10.0,
            "target_source": "given" if target else "corpus median for GROUND",
            "max_ha": 14.0, "settled_ha": 6.0, "arterials": ["primary", "secondary"]}},
        warnings=[LOTS_WARNING], map_version=None)


def blocks_payload(arguments: dict) -> dict:
    """What populated_areas returns: block summaries and counts."""
    lat, lon = SAN_CARLOS
    blocks = [{"id": f"{BLOCKS_SET}.{i}", "kind": "block", "axis": "street",
               "axis_ids": [], "sources": ["parcel", "structures", "osm"],
               "name": f"Street {i}", "facts": {"parcels": 30, "buildings": 20},
               "norm_flags": [], "warnings": []} for i in range(1, 10)]
    return {"set_id": BLOCKS_SET, "tool": "populated_areas", "params": {}, "handle":
            arguments["handle"], "created_at": "2026-10-22T09:00:00+00:00",
            "summary": {"count": 9, "total_before_limit": 9, "truncated_to": None,
                        "parcels_built_up": 350, "parcels_in_blocks": 316,
                        "parcels_left_out": 34, "large_parcels": 2,
                        "center": {"latitude": lat, "longitude": lon}},
            "assumptions": {"pixel_m": 1.0}, "warnings": [LOTS_WARNING],
            "candidates": blocks, "reused": False, "staleness": {"current": True}}


def parking_payload(arguments: dict) -> dict:
    lat, lon = SAN_CARLOS

    def lot(index, name, distance, flags):
        point = {"latitude": lat + 0.001 * index, "longitude": lon}
        return {"id": f"parking-w{index}", "kind": "parking", "name": name, "point": point,
                "area_m2": 1400.0, "surface": "asphalt", "capacity": "40", "flags": flags,
                "distance_from_center_m": distance, "bearing_from_center_deg": 0.0,
                "nearest_way": {"name": "Laurel Street", "route_type": "ROAD",
                                "distance_m": 15.0},
                "geometry": {"type": "Polygon", "coordinates": [
                    _square(lon, point["latitude"], 0.0002)]},
                "warnings": []}

    lots = [lot(1, "Library lot", 120.0, []), lot(2, "Station garage", 300.0,
                                                    ["covered_deck", "fee"])]
    limit = arguments.get("limit") or 0
    kept = lots[:limit] if limit else lots
    return {"candidates": kept,
            "summary": {"count": len(kept), "total_before_limit": 2,
                        "truncated_to": limit if limit and limit < 2 else None},
            "query_version": 5, "center": None, "warnings": [],
            "staleness": {"current": True}, "map_provenance": None}


class Stubs:
    """Staging and every feature, recording every call. `fail` names what
    raises: "blocks" (populated_areas, with no WMS token), "grouping"
    (urban_assignments), "parking", "medical", "export" (caltopo_export);
    `fail_resolutions` grounds staging cannot fetch (None is the search
    ground's default raster)."""

    def __init__(self, monkeypatch, *, fail=(), fail_resolutions=()):
        self.fail = set(fail)
        self.staging = AreaStaging(fail_resolutions)
        self.calls = {name: [] for name in ("blocks", "grouping", "parking", "medical",
                                            "export")}
        self.services = FakeServices(
            [record("Stanford Hospital", *STANFORD, trauma="LEVEL I", helipad="Y",
                    address="300 PASTEUR DR", city="STANFORD"),
             record("Sequoia Hospital", *SEQUOIA, address="170 ALAMEDA DE LAS PULGAS",
                    city="REDWOOD CITY")],
            minutes={STANFORD: 25.0, SEQUOIA: 9.0})
        self.real_export = candidates_component.caltopo_export
        monkeypatch.setattr(staging_component, "prepare_area", self.staging)
        monkeypatch.setattr(urban_feature, "populated_areas", self.blocks)
        monkeypatch.setattr(urban_feature, "urban_assignments", self.grouping)
        monkeypatch.setattr(parking_feature, "parking_candidates", self.parking)
        monkeypatch.setattr(medical_feature, "nearest_medical_facilities", self.medical)
        monkeypatch.setattr(candidates_component, "caltopo_export", self.export)

    def blocks(self, ctx, **arguments):
        self.calls["blocks"].append(arguments)
        ctx.staged(arguments["handle"])               # as the tool does
        if "blocks" in self.fail:
            raise ValueError(NO_TOKEN)
        return blocks_payload(arguments)

    def grouping(self, ctx, **arguments):
        self.calls["grouping"].append(arguments)
        resolved = ctx.staged(arguments["handle"])
        if "grouping" in self.fail:
            raise ValueError("target_ha (0.0) must be above zero")
        cset = urban_set(arguments["handle"], arguments)
        save_set(resolved.directory, cset)
        payload = cset.model_dump(exclude={"candidates", "manifest_stamp", "map_version"})
        payload.update(candidates=summaries(cset), reused=False, staleness={"current": True})
        payload["warnings"] = [LOTS_WARNING, "arterials checked against query version 5"]
        return payload

    def parking(self, ctx, **arguments):
        self.calls["parking"].append(arguments)
        ctx.staged(arguments["handle"])
        if "parking" in self.fail:
            raise ValueError("osm.json predates parking")
        return parking_payload(arguments)

    def medical(self, ctx, **arguments):
        self.calls["medical"].append(arguments)
        if "medical" in self.fail:
            raise httpx.ConnectError("HIFLD could not be reached")
        rest = {key: value for key, value in arguments.items()
                if key not in ("latitude", "longitude")}
        return medical_impl(arguments["latitude"], arguments["longitude"], **rest,
                            query=self.services.query, route=self.services.route,
                            helipads=self.services.helipads, locate=self.services.locate)

    def export(self, ctx, **arguments):
        self.calls["export"].append(arguments)
        if "export" in self.fail:
            raise OSError(28, "No space left on device")
        return self.real_export(ctx, **arguments)

    def nothing_called(self) -> bool:
        return not self.staging.calls and not any(self.calls.values())


@pytest.fixture
def ctx(tmp_path):
    return ServerContext.build(CaltopoConfig(cache_dir=tmp_path / "cache"))


def staged_dir(ctx, handle) -> Path:
    return Path(ctx.config.cache_dir) / "staged" / handle


def statuses(result) -> dict:
    return {name: entry["status"] for name, entry in result["sections"].items()}


# --- Every section ------------------------------------------------------------


def test_every_section_answers_and_the_summary_reads_the_facts(ctx, monkeypatch):
    stubs = Stubs(monkeypatch)
    result = plan_urban_search(ctx, *SAN_CARLOS)

    assert result["status"] == "ok"
    assert statuses(result) == {name: "ok" for name in SECTIONS}
    assert all(entry["error"] is None for entry in result["sections"].values())
    assert result["place_last_seen"] == {"latitude": SAN_CARLOS[0], "longitude": SAN_CARLOS[1]}
    assert result["radius_m"] == HALF_MILE

    # One ground, half a mile round the place at the default raster, shared
    # by the blocks, the assignments and parking.
    assert stubs.staging.calls == [{"center_latitude": SAN_CARLOS[0],
                                    "center_longitude": SAN_CARLOS[1], "radius_m": HALF_MILE,
                                    "map_id": "", "public": True, "resolution_m": None}]
    handle = result["handles"]["search"]
    assert list(result["handles"]) == ["search"]
    sections = result["sections"]
    assert {name: sections[name].get("ground") for name in SECTIONS} == {
        "urban": "search", "parking": "search", "medical": None, "export": None}

    # The blocks first, then the grouping, both keeping the search circle's.
    assert stubs.calls["blocks"] == [{"handle": handle, "max_distance_m": HALF_MILE,
                                      "building_markers": False}]
    assert stubs.calls["grouping"] == [{"handle": handle, "target_ha": None,
                                        "resource_type": "GROUND", "max_distance_m": HALF_MILE,
                                        "building_markers": False}]
    urban = sections["urban"]
    assert urban["handle"] == handle and urban["set_id"] == URBAN_SET
    assert urban["created_at"] == "2026-10-22T09:00:00+00:00" and urban["reused"] is False
    assert [a["id"] for a in urban["assignments"]] == [f"{URBAN_SET}.{i}" for i in (1, 2, 3)]
    first, pieces, water = urban["assignments"]
    assert first == {"id": f"{URBAN_SET}.1", "name": "Laurel Street +2",
                     "streets": ["Laurel Street", "Olive Street", "Walnut Street"],
                     "blocks": 4, "parcels": 120, "buildings": 96, "area_ha": 9.7,
                     "point": {"latitude": SAN_CARLOS[0] + 0.001,
                               "longitude": SAN_CARLOS[1]},
                     "distance_from_center_m": 110.0, "bearing_from_center_deg": 30.0,
                     "parts": 1, "water_links": 0, "warnings": []}
    assert pieces["parts"] == 2 and pieces["warnings"] == [PIECES_WARNING]
    assert water["water_links"] == 1 and water["warnings"] == [WATER_WARNING]
    assert not any("block_names" in a or "markers" in a for a in urban["assignments"])
    assert urban["count"] == 3 and urban["target_ha"] == 10.0
    assert urban["target_source"] == "corpus median for GROUND"
    assert urban["area_ha"] == {"min": 4.2, "median": 9.7, "max": 11.21, "total": 25.1}
    assert (urban["parcels"], urban["buildings"], urban["parcels_left_out"]) == (316, 180, 34)
    assert "markers" not in urban
    assert urban["warnings"] == [LOTS_WARNING, "arterials checked against query version 5"]
    assert urban["detail_call"] == {"tool": "urban_assignments",
                                    "arguments": stubs.calls["grouping"][0]}
    # The blocks as counts; their one warning the grouping already gave.
    assert urban["blocks"] == {
        "set_id": BLOCKS_SET, "created_at": "2026-10-22T09:00:00+00:00", "reused": False,
        "count": 9, "total_before_limit": 9, "truncated_to": None, "parcels_built_up": 350,
        "parcels_in_blocks": 316, "parcels_left_out": 34, "large_parcels": 2,
        "warnings": [],
        "detail_call": {"tool": "populated_areas", "arguments": stubs.calls["blocks"][0]}}

    parking = sections["parking"]
    assert [lot["id"] for lot in parking["candidates"]] == ["parking-w1", "parking-w2"]
    assert parking["detail_call"] == {"tool": "parking_candidates", "arguments": {
        "handle": handle, "limit": 5}}

    medical = sections["medical"]
    assert [row["name"] for row in medical["categories"]["trauma_center"]] == [
        "Stanford Hospital"]
    assert medical["detail_call"]["arguments"] == {
        "latitude": SAN_CARLOS[0], "longitude": SAN_CARLOS[1], "per_category": 3}

    export = sections["export"]
    exports_dir = staged_dir(ctx, handle) / "exports"
    assert export["path"] == str(exports_dir / f"{URBAN_SET}.geojson")
    assert export["assignments"] == 4                 # the two-piece one is two
    assert [entry["label"] for entry in export["labels"]] == [
        "U01 Laurel Street +2", "U02a El Camino Real", "U02b El Camino Real",
        "U03 Bayport Avenue +1"]
    assert export["area_included"] is True
    # The drafts are of the resource the assignments were sized for, and
    # carry house markers exactly when they were asked for.
    assert export["detail_call"] == {"tool": "caltopo_export", "arguments": {
        "handle": handle, "set_id": URBAN_SET, "include_area": True,
        "resource_type": "GROUND", "include_markers": False}}
    assert export["markers_path"] == str(exports_dir / "urban_search.geojson")
    # The place, two lots, two hospitals.
    assert export["markers"] == {"Marker": 5}
    assert result["exports"] == [export["path"], export["markers_path"]]

    summary = result["summary"]
    assert set(summary) == {"urban", "parking", "medical"}
    assert summary["urban"] == {"count": 3, "target_ha": 10.0,
                                "area_ha": {"min": 4.2, "median": 9.7, "max": 11.21,
                                            "total": 25.1},
                                "parcels": 316, "buildings": 180}
    assert summary["parking"]["nearest"]["name"] == "Library lot"
    assert summary["parking"]["nearest"]["distance_from_center_m"] == 120.0
    assert summary["medical"]["nearest_trauma_center"]["name"] == "Stanford Hospital"
    assert result["warnings"] == []
    assert "No verdict" in result["note"]
    json.dumps(result)                       # and it is JSON as it stands


@pytest.mark.parametrize("building_markers", [False, True])
def test_the_answer_carries_points_but_no_lines_or_polygons(ctx, monkeypatch,
                                                            building_markers):
    Stubs(monkeypatch)
    result = plan_urban_search(ctx, *SAN_CARLOS, building_markers=building_markers)
    assert result["status"] == "ok"
    # No geometry at all: every point here - an assignment's inside point, a
    # lot's, a hospital's - is a latitude and longitude, and the house
    # markers stay with get_candidate and the drafts file.
    assert geometry_types_in(result) == set()


def test_house_markers_only_when_asked_for_and_then_in_the_drafts(ctx, monkeypatch):
    stubs = Stubs(monkeypatch)
    result = plan_urban_search(ctx, *SAN_CARLOS, building_markers=True, resource_type="DOG",
                               target_ha=6.0)
    assert stubs.calls["blocks"][0]["building_markers"] is True
    grouping = stubs.calls["grouping"][0]
    assert (grouping["building_markers"], grouping["resource_type"],
            grouping["target_ha"]) == (True, "DOG", 6.0)
    urban = result["sections"]["urban"]
    assert [a["markers"] for a in urban["assignments"]] == [3, 2, 1]
    assert urban["markers"] == 6
    assert urban["target_source"] == "given"
    export = result["sections"]["export"]
    assert export["detail_call"]["arguments"]["include_markers"] is True
    assert export["detail_call"]["arguments"]["resource_type"] == "DOG"
    drafts = read_export(export["path"])["features"]
    houses = [f for f in drafts if f["properties"].get("marker-symbol") == "hut"]
    assert len(houses) == 6
    assignments = [f for f in drafts if f["properties"]["class"] == "Assignment"]
    assert {f["properties"]["resourceType"] for f in assignments} == {"DOG"}


# --- Failures stay in their section --------------------------------------------


def test_one_section_failing_leaves_the_others_and_the_status_partial(ctx, monkeypatch):
    Stubs(monkeypatch, fail={"parking"})
    result = plan_urban_search(ctx, *SAN_CARLOS)
    assert result["status"] == "partial"
    parking = result["sections"]["parking"]
    assert parking["status"] == "failed"
    assert parking["error"] == "ValueError: osm.json predates parking"
    assert parking["detail_call"]["tool"] == "parking_candidates"
    assert {name for name, status in statuses(result).items() if status == "ok"} == {
        "urban", "medical", "export"}
    assert result["summary"]["parking"] is None
    assert result["summary"]["urban"]["count"] == 3
    assert len(result["exports"]) == 2                 # what was found is still drawn


def test_no_wms_token_fails_urban_in_the_feature_s_own_words_and_the_rest_answer(
        ctx, monkeypatch):
    """The real populated_areas and urban_assignments, on a server with no
    WMS token: lot lines and roofs cannot be read, so the urban section fails
    with the feature's own reason - and parking and the hospitals answer."""
    Stubs(monkeypatch)
    for name, function in REAL_URBAN.items():
        monkeypatch.setattr(urban_feature, name, function)
    assert ctx.config.wms_base_url() is None
    result = plan_urban_search(ctx, *SAN_CARLOS)

    assert result["status"] == "partial"
    urban = result["sections"]["urban"]
    assert urban["status"] == "failed"
    assert urban["error"] == f"ValueError: {NO_TOKEN}"
    # The call that raised - the blocks - to make again once a token is set.
    assert urban["detail_call"] == {"tool": "populated_areas", "arguments": {
        "handle": result["handles"]["search"], "max_distance_m": HALF_MILE,
        "building_markers": False}}
    assert statuses(result) == {"urban": "failed", "parking": "ok", "medical": "ok",
                                "export": "ok"}
    export = result["sections"]["export"]
    assert export["path"] is None
    assert export["warnings"][0] == ("the urban section did not answer, so there is no "
                                     "candidate set to draw as draft assignments; its error "
                                     "says why")
    assert result["exports"] == [export["markers_path"]]
    assert result["summary"]["urban"] is None


@pytest.mark.parametrize("include", [["urban"], ["urban", "export"]])
def test_no_wms_token_with_urban_the_only_finder_is_failed_and_writes_nothing(
        ctx, monkeypatch, include):
    """The likeliest set-up mistake on a new server, with urban the only
    section asked for that finds anything: the blocks are refused in the
    feature's own words and nothing else answered, so the call is "failed",
    never "partial" - and with the export asked for, no file is written and
    the top level says so."""
    stubs = Stubs(monkeypatch)
    for name, function in REAL_URBAN.items():
        monkeypatch.setattr(urban_feature, name, function)
    result = plan_urban_search(ctx, *SAN_CARLOS, include=include)

    assert result["status"] == "failed"
    urban = result["sections"]["urban"]
    assert urban["status"] == "failed" and urban["error"] == f"ValueError: {NO_TOKEN}"
    assert urban["detail_call"]["tool"] == "populated_areas"
    assert "blocks" not in urban                    # refused: nothing answered in part
    assert stubs.calls["parking"] == stubs.calls["medical"] == stubs.calls["export"] == []
    assert result["summary"] == {"urban": None}
    assert result["exports"] == []
    export = result["sections"]["export"]
    if "export" in include:
        assert export["status"] == "ok" and export["path"] is None
        assert export["markers_path"] is None and export["markers"] == {}
        assert export["warnings"] == ["the urban section did not answer, so there is no "
                                      "candidate set to draw as draft assignments; its "
                                      "error says why"]
        # Its wording is B11's (test_sections_that_did_not_answer_...).
        assert len(result["warnings"]) == 1
        assert "no CalTopo export was written" in result["warnings"][0]
    else:
        assert export == {"status": "skipped", "error": None, "warnings": []}
        assert result["warnings"] == []


@pytest.mark.xfail(strict=True, reason="PR #4 review B34: with no WMS token an urban-only "
                                       "call still stages the search ground before the "
                                       "urban section fails")
@pytest.mark.parametrize("include", [["urban"], ["urban", "export"]])
def test_no_wms_token_with_urban_the_only_ground_section_stages_nothing(ctx, monkeypatch,
                                                                        include):
    """The refusal is known before staging - the token is config, free to
    read - and nothing else asked for needs the ground: a minute of 3DEP,
    LANDFIRE and OpenStreetMap is spent for an answer that cannot use it."""
    stubs = Stubs(monkeypatch)
    for name, function in REAL_URBAN.items():
        monkeypatch.setattr(urban_feature, name, function)
    result = plan_urban_search(ctx, *SAN_CARLOS, include=include)
    assert result["status"] == "failed"
    assert result["sections"]["urban"]["error"] == f"ValueError: {NO_TOKEN}"
    assert stubs.staging.calls == []


def test_no_wms_token_with_parking_asked_for_still_stages_the_ground_once(ctx, monkeypatch):
    """Parking needs the ground whatever the token: it is staged once, the
    urban section fails with NO_TOKEN, and parking answers on it."""
    stubs = Stubs(monkeypatch)
    for name, function in REAL_URBAN.items():
        monkeypatch.setattr(urban_feature, name, function)
    result = plan_urban_search(ctx, *SAN_CARLOS, include=["urban", "parking"])
    assert [call["radius_m"] for call in stubs.staging.calls] == [HALF_MILE]
    assert result["status"] == "partial"
    assert result["sections"]["urban"]["error"] == f"ValueError: {NO_TOKEN}"
    assert result["sections"]["parking"]["status"] == "ok"
    assert stubs.calls["parking"] == [{"handle": result["handles"]["search"], "limit": 5}]


@pytest.mark.parametrize("include", [None, ["urban"]])
def test_the_grouping_failing_after_the_blocks_keeps_its_call_and_the_blocks(
        ctx, monkeypatch, include):
    """The blocks were drawn and saved before the grouping raised: they stay,
    so the section reads as answered in part - "partial", never "failed",
    even when urban is the only section asked for."""
    stubs = Stubs(monkeypatch, fail={"grouping"})
    result = plan_urban_search(ctx, *SAN_CARLOS, include=include)
    urban = result["sections"]["urban"]
    assert urban["status"] == "failed"
    assert urban["error"] == "ValueError: target_ha (0.0) must be above zero"
    assert urban["detail_call"] == {"tool": "urban_assignments",
                                    "arguments": stubs.calls["grouping"][0]}
    assert len(stubs.calls["blocks"]) == 1
    assert urban["handle"] == result["handles"]["search"]
    assert urban["warnings"] == []
    assert urban["blocks"] == {
        "set_id": BLOCKS_SET, "created_at": "2026-10-22T09:00:00+00:00", "reused": False,
        "count": 9, "total_before_limit": 9, "truncated_to": None, "parcels_built_up": 350,
        "parcels_in_blocks": 316, "parcels_left_out": 34, "large_parcels": 2,
        # Nothing else said them this time, so the blocks say their own.
        "warnings": [LOTS_WARNING],
        "detail_call": {"tool": "populated_areas", "arguments": stubs.calls["blocks"][0]}}
    assert "assignments" not in urban and "set_id" not in urban
    assert result["status"] == "partial"
    assert result["summary"]["urban"] is None
    assert geometry_types_in(result) == set()


def test_a_staging_failure_fails_urban_and_parking_but_not_the_hospitals(ctx, monkeypatch):
    stubs = Stubs(monkeypatch, fail_resolutions={None})
    result = plan_urban_search(ctx, *SAN_CARLOS)
    assert result["status"] == "partial"
    assert statuses(result) == {"urban": "failed", "parking": "failed", "medical": "ok",
                                "export": "ok"}
    assert result["staging"]["search"]["status"] == "failed"
    assert result["handles"] == {"search": None}
    for name in ("urban", "parking"):
        entry = result["sections"][name]
        assert entry["error"].startswith("StagingFailed: the search ground (805 m radius)")
        assert "RuntimeError: 3DEP could not be reached" in entry["error"]
        assert "detail_call" not in entry                # no feature call was made
    assert stubs.calls["blocks"] == stubs.calls["grouping"] == stubs.calls["parking"] == []
    assert result["staging"]["search"]["detail_call"]["tool"] == "prepare_area"
    # The hospitals were found, but there is no ground to write them beside.
    export = result["sections"]["export"]
    assert export["status"] == "ok" and export["markers_path"] is None
    assert export["warnings"][-1].startswith("no ground was staged, so there is nowhere")
    assert result["exports"] == []
    assert result["warnings"] == [
        "no CalTopo export was written, although the sections found things to draw: "
        "sections.export.warnings says why, and every section's answer stands"]


def test_every_section_failing_is_failed_and_draws_nothing(ctx, monkeypatch):
    Stubs(monkeypatch, fail={"blocks", "parking", "medical"})
    result = plan_urban_search(ctx, *SAN_CARLOS)
    assert result["status"] == "failed"
    assert statuses(result) == {"urban": "failed", "parking": "failed", "medical": "failed",
                                "export": "ok"}
    assert result["summary"] == {"urban": None, "parking": None, "medical": None}
    assert result["exports"] == []
    # One top-level warning that no file was written. Its wording is pinned
    # by test_sections_that_did_not_answer_are_not_reported_as_nothing_found
    # (PR #4 review B11): today it reads as if the search found nothing.
    assert len(result["warnings"]) == 1
    assert "no CalTopo export was written" in result["warnings"][0]


@pytest.mark.xfail(strict=True, reason="PR #4 review B11: 'nothing was found to draw' is "
                                       "reported when sections failed, not only when the "
                                       "search found nothing")
def test_sections_that_did_not_answer_are_not_reported_as_nothing_found(ctx, monkeypatch):
    """Every section failed: nothing was searched, so "nothing was found"
    would be relayed as "no blocks, lots or hospitals there". The warning
    says the sections did not answer, names them, and points to their
    errors."""
    Stubs(monkeypatch, fail={"blocks", "parking", "medical"})
    result = plan_urban_search(ctx, *SAN_CARLOS)
    assert result["status"] == "failed"
    assert result["exports"] == []
    assert not any("nothing was found" in warning for warning in result["warnings"])
    (warning,) = result["warnings"]
    assert "no CalTopo export was written" in warning and "did not answer" in warning
    assert all(name in warning for name in ("urban", "parking", "medical"))


@pytest.mark.xfail(strict=True, reason="PR #4 review B11: 'nothing was found to draw' is "
                                       "reported when the blocks were found and saved but "
                                       "the grouping failed")
def test_blocks_saved_before_the_grouping_failed_are_offered_not_called_nothing_found(
        ctx, monkeypatch):
    """The blocks were drawn and saved (the status is "partial" for them),
    then the grouping, parking and the hospitals all failed. The answer
    must not say nothing was found, nor that there is no set to draw: it
    names the sections that did not answer and the saved blocks set
    caltopo_export can draw."""
    Stubs(monkeypatch, fail={"grouping", "parking", "medical"})
    result = plan_urban_search(ctx, *SAN_CARLOS)
    assert result["status"] == "partial"
    assert result["sections"]["urban"]["blocks"]["set_id"] == BLOCKS_SET
    assert result["exports"] == []
    assert not any("nothing was found" in warning for warning in result["warnings"])
    said = [warning for warning in result["warnings"] if "did not answer" in warning]
    assert len(said) == 1
    assert all(name in said[0] for name in ("urban", "parking", "medical"))
    assert BLOCKS_SET in said[0] and "caltopo_export" in said[0]
    export = result["sections"]["export"]
    assert not any("there is no candidate set" in warning for warning in export["warnings"])


def test_sections_that_answered_and_found_nothing_say_nothing_was_found(ctx, monkeypatch):
    """The one case the wording is for: urban left out, and parking and the
    hospitals both answered - with no lot and no hospital. Nothing was found,
    so there is nothing to draw, and the answer says exactly that."""
    stubs = Stubs(monkeypatch)
    stubs.services = FakeServices([], minutes={})            # no hospital anywhere

    def no_lots(ctx, **arguments):
        stubs.calls["parking"].append(arguments)
        return {**parking_payload(arguments), "candidates": [],
                "summary": {"count": 0, "total_before_limit": 0, "truncated_to": None}}

    monkeypatch.setattr(parking_feature, "parking_candidates", no_lots)
    result = plan_urban_search(ctx, *SAN_CARLOS, include=["parking", "medical", "export"])
    assert result["status"] == "ok"
    assert statuses(result) == {"urban": "skipped", "parking": "ok", "medical": "ok",
                                "export": "ok"}
    assert result["sections"]["parking"]["candidates"] == []
    assert result["summary"] == {"parking": {"count": 0, "total_before_limit": 0,
                                             "nearest": None},
                                 "medical": {"nearest_trauma_center": None}}
    export = result["sections"]["export"]
    assert export["markers_path"] is None and export["markers"] == {}
    assert result["exports"] == []
    assert result["warnings"] == ["nothing was found to draw, so no CalTopo export was "
                                  "written"]


def test_an_export_that_raises_is_a_warning_and_keeps_its_call(ctx, monkeypatch):
    stubs = Stubs(monkeypatch, fail={"export"})
    result = plan_urban_search(ctx, *SAN_CARLOS, overrides={"export": {"label_prefix": "T"}})
    export = result["sections"]["export"]
    assert export["status"] == "ok" and export["error"] is None
    assert export["path"] is None and export["set_id"] == URBAN_SET
    assert export["warnings"][0].startswith("the draft assignments could not be written "
                                            "(OSError: [Errno 28] No space left on device)")
    assert export["detail_call"] == {"tool": "caltopo_export", "arguments": {
        "handle": result["handles"]["search"], "set_id": URBAN_SET, "include_area": True,
        "resource_type": "GROUND", "include_markers": False, "label_prefix": "T"}}
    assert stubs.calls["export"] == [export["detail_call"]["arguments"]]
    assert result["status"] == "ok"
    assert result["exports"] == [export["markers_path"]]
    # The assignments - the answer itself - are in no file: the top level
    # says so (PR #4 review B10), and never that nothing was found.
    (warning,) = result["warnings"]
    assert "draft assignments" in warning and "not written" in warning
    assert "nothing was found" not in warning


# --- What the caller chooses --------------------------------------------------


def test_include_skips_sections_and_stages_nothing_for_them(ctx, monkeypatch):
    stubs = Stubs(monkeypatch)
    result = plan_urban_search(ctx, *SAN_CARLOS, include=["medical"],
                               overrides={"urban": {"populated_areas": {"min_parcels": 2}}})
    assert result["status"] == "ok"
    for name in ("urban", "parking", "export"):
        assert result["sections"][name] == {"status": "skipped", "error": None,
                                             "warnings": []}
    assert stubs.staging.calls == [] and stubs.calls["blocks"] == []
    assert result["staging"] == {} and result["handles"] == {}
    assert set(result["summary"]) == {"medical"}
    assert result["exports"] == []
    assert result["warnings"] == ["overrides['urban'] was not used: urban was left out of "
                                  "include"]


def test_parking_alone_stages_the_search_ground_and_draws_no_blocks(ctx, monkeypatch):
    stubs = Stubs(monkeypatch)
    result = plan_urban_search(ctx, *SAN_CARLOS, include=["parking"], parking_limit=1)
    assert [call["radius_m"] for call in stubs.staging.calls] == [HALF_MILE]
    assert stubs.calls["blocks"] == stubs.calls["grouping"] == []
    assert [lot["id"] for lot in result["sections"]["parking"]["candidates"]] == [
        "parking-w1"]
    assert result["summary"] == {"parking": {
        "count": 1, "total_before_limit": 2, "nearest": {
            "id": "parking-w1", "kind": "parking", "name": "Library lot",
            "distance_from_center_m": 120.0, "bearing_from_center_deg": 0.0, "flags": []}}}


def test_with_urban_left_out_the_export_draws_only_the_markers(ctx, monkeypatch):
    stubs = Stubs(monkeypatch)
    result = plan_urban_search(ctx, *SAN_CARLOS, include=["parking", "medical", "export"],
                               overrides={"export": {"label_prefix": "T"}})
    export = result["sections"]["export"]
    assert export["status"] == "ok" and export["path"] is None
    assert export["warnings"] == ["urban was left out of include, so there is no candidate "
                                  "set to draw as draft assignments"]
    assert stubs.calls["export"] == []
    assert export["markers"] == {"Marker": 5}
    assert result["exports"] == [export["markers_path"]]
    assert result["status"] == "ok"
    assert result["warnings"] == ["overrides['export'] was not used: urban was left out of "
                                  "include, so there was no candidate set for caltopo_export "
                                  "to draw"]


def test_the_radius_takes_a_unit_and_is_converted_once(ctx, monkeypatch):
    stubs = Stubs(monkeypatch)
    result = plan_urban_search(ctx, *SAN_CARLOS, radius=0.25, radius_unit="mi")
    assert result["radius_m"] == 402.336
    assert stubs.staging.calls[0]["radius_m"] == 402.336
    # The blocks of the search circle are kept, whatever the radius.
    assert stubs.calls["blocks"][0]["max_distance_m"] == 402.336
    assert stubs.calls["grouping"][0]["max_distance_m"] == 402.336
    plan_urban_search(ctx, *SAN_CARLOS, radius_m=600, include=["parking"])
    assert stubs.staging.calls[-1]["radius_m"] == 600


def test_urban_overrides_go_by_tool_to_each_call_and_its_detail_call(ctx, monkeypatch):
    stubs = Stubs(monkeypatch)
    result = plan_urban_search(
        ctx, *SAN_CARLOS, parking_limit=0, medical_per_category=1,
        overrides={"urban": {"populated_areas": {"max_distance_m": 600, "limit": 50},
                             "urban_assignments": {"max_distance_m": 600}},
                   "parking": {"max_distance_m": 600},
                   "medical": {"search_radius_km": 150.0},
                   "export": {"label_prefix": "T", "include_area": False}})
    handle = result["handles"]["search"]
    assert stubs.calls["blocks"] == [{"handle": handle, "max_distance_m": 600,
                                      "building_markers": False, "limit": 50}]
    assert stubs.calls["grouping"] == [{"handle": handle, "target_ha": None,
                                        "resource_type": "GROUND", "max_distance_m": 600,
                                        "building_markers": False}]
    urban = result["sections"]["urban"]
    assert urban["detail_call"]["arguments"] == stubs.calls["grouping"][0]
    assert urban["blocks"]["detail_call"]["arguments"] == stubs.calls["blocks"][0]
    # The same 600 m for both, inside the search circle, and the limit only
    # lists fewer blocks: the blocks counted are the ones grouped, so
    # nothing to warn of.
    assert urban["warnings"] == [LOTS_WARNING, "arterials checked against query version 5"]
    assert stubs.calls["parking"] == [{"handle": handle, "limit": 0, "max_distance_m": 600}]
    assert stubs.calls["medical"][0]["search_radius_km"] == 150.0
    assert stubs.calls["medical"][0]["per_category"] == 1
    assert stubs.calls["export"] == [{"handle": handle, "set_id": URBAN_SET,
                                      "include_area": False, "resource_type": "GROUND",
                                      "include_markers": False, "label_prefix": "T"}]
    assert result["sections"]["export"]["labels"][0]["label"] == "T01 Laurel Street +2"
    json.dumps(result)


@pytest.mark.parametrize("urban, said", [
    ({"populated_areas": {"min_parcels": 2}}, "min_parcels 2; the assignments were grouped "
                                             "from blocks drawn with min_parcels 3"),
    ({"populated_areas": {"max_parcel_area_m2": 8094, "pixel_m": 0.5}},
     "pixel_m 0.5, max_parcel_area_m2 8094; the assignments were grouped from blocks drawn "
     "with pixel_m 1.0, max_parcel_area_m2 4047.0"),
    ({"urban_assignments": {"max_distance_m": 1200}},
     "max_distance_m 804.672; the assignments were grouped from blocks drawn with "
     "max_distance_m 1200"),
    ({"populated_areas": {"max_distance_m": None}}, "max_distance_m None; the assignments"),
])
def test_blocks_drawn_by_other_rules_than_the_grouping_s_are_said(ctx, monkeypatch, urban,
                                                                  said):
    """urban_assignments takes no block rule but max_distance_m: a rule given
    to populated_areas alone changes the blocks counted, not the blocks
    grouped. Allowed - the caller may want to compare - but said."""
    Stubs(monkeypatch)
    result = plan_urban_search(ctx, *SAN_CARLOS, overrides={"urban": urban})
    entry = result["sections"]["urban"]
    assert entry["status"] == "ok"
    warning = entry["warnings"][-1]
    assert warning.startswith("the blocks under `blocks` were drawn with ")
    assert said in warning
    assert warning.endswith("(urban_assignments takes no block rule but max_distance_m), so "
                            "the two do not count the same blocks")


@pytest.mark.parametrize("urban, said", [
    ({"populated_areas": {"max_distance_m": 1200}, "urban_assignments": {"max_distance_m": 1200}},
     "populated_areas 1,200 m; urban_assignments 1,200 m"),
    ({"urban_assignments": {"max_distance_m": None}}, "urban_assignments null (no limit)"),
])
def test_a_max_distance_past_the_search_circle_is_said_not_taken_for_a_wider_search(
        ctx, monkeypatch, urban, said):
    """Lot lines are read over the staged circle only: a max_distance_m past
    the radius keeps at most blocks straddling its edge. Passed on - that is
    a fair thing to want - but the section says it is no wider search."""
    stubs = Stubs(monkeypatch)
    result = plan_urban_search(ctx, *SAN_CARLOS, overrides={"urban": urban})
    entry = result["sections"]["urban"]
    assert entry["status"] == "ok"
    for tool, call in (("populated_areas", "blocks"), ("urban_assignments", "grouping")):
        if tool in urban:
            assert stubs.calls[call][0]["max_distance_m"] == urban[tool]["max_distance_m"]
    assert (f"max_distance_m reaches past the 805 m search radius ({said}), but lot lines are "
            "read over the staged search circle only: it can add only blocks straddling the "
            "circle's edge, never one wholly outside it - to search wider, pass a larger "
            "radius") in entry["warnings"]


def test_a_max_distance_within_the_circle_is_not_warned_of(ctx, monkeypatch):
    Stubs(monkeypatch)
    for distance in (600, HALF_MILE):
        result = plan_urban_search(ctx, *SAN_CARLOS, overrides={"urban": {
            "populated_areas": {"max_distance_m": distance},
            "urban_assignments": {"max_distance_m": distance}}})
        assert not any("reaches past" in warning
                       for warning in result["sections"]["urban"]["warnings"])


@pytest.mark.parametrize("urban, markers, warned", [
    ({"populated_areas": {"marker_color": "#FF0000"}}, True, ("#FF0000", "#FFFFFF")),
    ({"populated_areas": {"marker_color": "#FF0000"},
      "urban_assignments": {"marker_color": "00ff00"}}, True, ("#FF0000", "#00FF00")),
    ({"populated_areas": {"marker_color": "ff0000"},
      "urban_assignments": {"marker_color": "#FF0000"}}, True, None),
    # The drafts get the colour asked for; only the blocks' own set keeps white.
    ({"urban_assignments": {"marker_color": "00ff00"}}, True, None),
    ({"populated_areas": {"marker_color": "#FF0000"}}, False, None),
])
def test_a_marker_colour_the_drafts_do_not_take_is_said(ctx, monkeypatch, urban, markers,
                                                        warned):
    """The draft file carries urban_assignments' house markers: a colour
    given to populated_areas alone never reaches CalTopo, and the caller who
    asked for red houses hears so. Without markers no colour is drawn."""
    Stubs(monkeypatch)
    result = plan_urban_search(ctx, *SAN_CARLOS, building_markers=markers,
                               overrides={"urban": urban})
    warnings = result["sections"]["urban"]["warnings"]
    said = [warning for warning in warnings if "house markers" in warning]
    if warned is None:
        assert said == []
    else:
        drawn, grouped = warned
        assert said == [f"the blocks' house markers are {drawn} (populated_areas' "
                        f"marker_color) but the assignments' - and so the draft file's - are "
                        f"{grouped} (urban_assignments' marker_color): give urban_assignments "
                        "the colour the drafts should carry"]


@pytest.mark.parametrize("arguments, warned", [
    ({"radius": 0.5}, 0.5),                               # miles meant, metres read
    ({"radius_m": 49.9}, 49.9),
    ({"radius_m": 50}, None),
    ({"radius": 0.25, "radius_unit": "mi"}, None),
])
def test_a_search_radius_of_a_lot_or_two_is_warned_of(ctx, monkeypatch, arguments, warned):
    """radius_unit defaults to metres, as prepare_area's does, so radius=0.5
    alone is half a metre: answered, but with a warning naming the unit."""
    stubs = Stubs(monkeypatch)
    result = plan_urban_search(ctx, *SAN_CARLOS, **arguments)
    assert result["status"] == "ok"
    said = [warning for warning in result["warnings"] if "search radius is" in warning]
    if warned is None:
        assert said == []
    else:
        assert stubs.staging.calls[0]["radius_m"] == warned
        assert said == [f"the search radius is {warned:g} m, a lot or two across rather than "
                        "a neighbourhood: radius is read in radius_unit, which is metres unless "
                        "given - half a mile is radius=0.5, radius_unit=\"mi\""]


@pytest.mark.parametrize("arguments, message", [
    ({"radius_m": 60, "radius_unit": "yd"}, "radius_unit 'yd' applies to radius, not to radius_m"),
    ({"radius_m": 2, "radius_unit": "km"}, "radius_unit 'km' applies to radius, not to radius_m"),
    ({"radius_unit": "block"}, "radius_unit 'block' is not one of"),
])
def test_a_unit_beside_radius_m_is_refused_before_any_work(ctx, monkeypatch, arguments,
                                                           message):
    """radius_m is metres by its name: beside it, a unit is a slip - 60 yd
    meant, 60 m staged, no small-radius warning to catch it - so it is
    refused, as is a unit the scenario does not know, with no radius."""
    stubs = Stubs(monkeypatch)
    with pytest.raises(ValueError, match=message):
        plan_urban_search(ctx, *SAN_CARLOS, **arguments)
    assert stubs.nothing_called()


@pytest.mark.parametrize("overrides, message", [
    ({"hasty": {"limit": 5}}, r"no section of plan_urban_search: \['hasty'\]"),
    ({"urban": {"max_distance_m": 1200}},
     r"overrides\['urban'\] names \['max_distance_m'\], which is no tool it calls: the urban "
     r"section calls \['populated_areas', 'urban_assignments'\], so give each its own "
     r"keyword arguments, such as \{'urban': \{'urban_assignments': \{'max_distance_m': "),
    ({"urban": {"lz_candidates": {}}}, r"names \['lz_candidates'\], which is no tool"),
    ({"urban": [("populated_areas", {})]}, r"overrides\['urban'\] must be a mapping by tool"),
    ({"urban": {"populated_areas": []}},
     r"overrides\['urban'\]\['populated_areas'\] must be a mapping of populated_areas'"),
    ({"urban": {"urban_assignments": {"min_parcels": 2}}},
     r"overrides\['urban'\]\['urban_assignments'\] names \['min_parcels'\], which "
     r"urban_assignments does not take here; it accepts \['max_distance_m', 'marker_color'\]"),
    # populated_areas takes no target_ha or resource_type at all: refused as
    # a stranger, with what it does take - not as the scenario's to set.
    ({"urban": {"populated_areas": {"target_ha": 5}}},
     r"overrides\['urban'\]\['populated_areas'\] names \['target_ha'\], which "
     r"populated_areas does not take here; it accepts \['pixel_m', "),
    ({"urban": {"populated_areas": {"resource_type": "DOG"}}},
     r"\['populated_areas'\] names \['resource_type'\], which populated_areas does not take"),
    ({"urban": {"populated_areas": {"handle": "aoi-0123456789abcdef"}}},
     r"\['populated_areas'\] cannot set 'handle': the scenario stages"),
    ({"urban": {"populated_areas": {"building_markers": True}}},
     "cannot set 'building_markers': plan_urban_search sets it from its own argument "
     "building_markers"),
    ({"urban": {"urban_assignments": {"target_ha": 5}}}, "own argument target_ha"),
    ({"urban": {"urban_assignments": {"resource_type": "DOG"}}}, "own argument resource_type"),
    ({"urban": {"urban_assignments": {"building_markers": True}}},
     "own argument building_markers"),
    ({"parking": {"limit": 5}}, "own argument parking_limit"),
    ({"medical": {"per_category": 5}}, "own argument medical_per_category"),
    ({"medical": {"latitude": 1.0}}, "cannot set 'latitude'"),
    ({"export": {"set_id": "urban-000000"}}, "cannot set 'set_id': the scenario exports"),
    ({"export": {"resource_type": "DOG"}}, r"overrides\['export'\] cannot set "
                                           "'resource_type': plan_urban_search sets it from "
                                           "its own argument resource_type"),
    ({"export": {"include_markers": True}}, "own argument building_markers"),
    ({"export": {"label_prefix": "!!"}}, r"overrides\['export'\]: label_prefix \('!!'\) "
                                         "must be 1 to 6 letters or digits"),
    ({"urban": {"populated_areas": {"pixel_m": 3}}},
     r"overrides\['urban'\]\['populated_areas'\]: pixel_m \(3\) must be above 0 and at most 2"),
    ({"urban": {"populated_areas": {"min_parcels": 0}}}, r"min_parcels \(0\) must be at least 1"),
    ({"urban": {"populated_areas": {"large_parcel_min_cover": 1.5}}},
     "large_parcel_min_cover"),
    ({"urban": {"populated_areas": {"bridge_m": "wide"}}},
     r"overrides\['urban'\]\['populated_areas'\]: "),
    ({"urban": {"urban_assignments": {"max_distance_m": -1}}},
     r"overrides\['urban'\]\['urban_assignments'\]: max_distance_m \(-1\) must be above zero"),
    ({"urban": {"populated_areas": {"limit": -1}}}, r"limit \(-1\) is negative"),
    ({"parking": []}, "must be a mapping"),
])
def test_bad_overrides_are_refused_before_any_work(ctx, monkeypatch, overrides, message):
    stubs = Stubs(monkeypatch)
    with pytest.raises(ValueError, match=message):
        plan_urban_search(ctx, *SAN_CARLOS, overrides=overrides)
    assert stubs.nothing_called()


def test_a_marker_colour_is_refused_only_when_markers_are_drawn(ctx, monkeypatch):
    """The features read `marker_color` only with building markers on, and
    refuse one CalTopo would draw black only then: so does the scenario."""
    stubs = Stubs(monkeypatch)
    # An explicit null too: it would reach the feature as None, refused there
    # only once the ground was staged.
    for tool, bad in (("urban_assignments", "white"), ("urban_assignments", None),
                      ("populated_areas", None)):
        colour = {"urban": {tool: {"marker_color": bad}}}
        with pytest.raises(ValueError, match=rf"overrides\['urban'\]\['{tool}'\]: "
                                             rf"marker_color \({bad!r}\) must be a hex colour"):
            plan_urban_search(ctx, *SAN_CARLOS, building_markers=True, overrides=colour)
        assert stubs.nothing_called()
        assert plan_urban_search(ctx, *SAN_CARLOS, overrides=colour)["status"] == "ok"
        stubs.staging.calls.clear()
        for calls in stubs.calls.values():
            calls.clear()


@pytest.mark.parametrize("arguments, message", [
    ({"latitude": 95.0}, "latitude"),
    ({"longitude": float("nan")}, "longitude"),
    ({"include": ["hasty"]}, "no section of this scenario"),
    ({"include": []}, "empty"),
    ({"include": "urban"}, "must be a list"),
    ({"include": ["export"]}, "include names only export"),
    ({"radius_m": 4500.0}, "radius must be above 0 and under 4,500 m"),
    ({"radius": 3, "radius_unit": "mi"}, r"radius must be above 0 .* got 4828"),
    ({"radius_m": 0}, "radius must be above 0"),
    ({"radius": 1, "radius_m": 1000}, "give the radius once"),
    ({"radius": 1, "radius_unit": "block"}, "radius_unit 'block' is not one of"),
    # prepare_area stages it, but no lot line can be read over it.
    ({"radius_m": 4200.0}, r"radius of 4,200.0 m is too wide to read lot lines over, so "
                           r"nothing was staged: populated_areas would refuse it - the staged "
                           r"area is 8.8 km across.*or leave urban out of include"),
    ({"target_ha": 0}, "target_ha must be a number of hectares above zero"),
    ({"target_ha": -5.0}, "target_ha"),
    ({"target_ha": float("nan")}, "target_ha"),
    ({"target_ha": True}, "target_ha"),
    ({"target_ha": "10"}, "target_ha"),
    ({"resource_type": ""}, "resource_type must be a resource type"),
    ({"resource_type": 5}, "resource_type"),
    ({"building_markers": "false"}, "building_markers must be true or false"),
    # Truthiness, or `in (True, False)`, would let 1 and 0 through: markers
    # on draws another set, with a marker on every roof.
    ({"building_markers": 1}, r"building_markers must be true or false; got 1$"),
    ({"building_markers": 0}, r"building_markers must be true or false; got 0$"),
    ({"building_markers": None}, r"building_markers must be true or false; got None$"),
    ({"parking_limit": -1}, "parking_limit must be 0"),
    ({"parking_limit": 2.5}, "parking_limit"),
    ({"medical_per_category": 0}, "medical_per_category"),
    ({"medical_per_category": 11}, "medical_per_category"),
])
def test_bad_arguments_are_refused_before_any_work(ctx, monkeypatch, arguments, message):
    stubs = Stubs(monkeypatch)
    call = {"latitude": SAN_CARLOS[0], "longitude": SAN_CARLOS[1], **arguments}
    with pytest.raises(ValueError, match=message):
        plan_urban_search(ctx, **call)
    assert stubs.nothing_called()


def test_a_radius_too_wide_for_lot_lines_still_serves_parking_and_hospitals(ctx, monkeypatch):
    """Only the blocks are read over the circle at 1 to 2 m: with urban left
    out, any radius prepare_area stages will do."""
    stubs = Stubs(monkeypatch)
    result = plan_urban_search(ctx, *SAN_CARLOS, radius_m=4200.0,
                               include=["parking", "medical"])
    assert result["status"] == "ok"
    assert [call["radius_m"] for call in stubs.staging.calls] == [4200.0]


# --- The export -----------------------------------------------------------------


def test_the_export_files_are_ones_caltopo_imports(ctx, monkeypatch):
    Stubs(monkeypatch)
    result = plan_urban_search(ctx, *SAN_CARLOS)
    drafts_path, markers_path = (Path(path) for path in result["exports"])

    drafts = read_export(drafts_path)
    assert drafts["type"] == "FeatureCollection"
    assignments = [f for f in drafts["features"] if f["properties"]["class"] == "Assignment"]
    assert [f["geometry"]["type"] for f in assignments] == ["Polygon"] * 4
    assert all(f["properties"]["status"] == "DRAFT" for f in assignments)
    assert {f["properties"]["resourceType"] for f in assignments} == {"GROUND"}
    first = assignments[0]["properties"]["description"]
    assert "Streets: Laurel Street, Olive Street, Walnut Street." in first
    assert "check it on the ground before issuing" in first
    area = [f for f in drafts["features"] if f["properties"]["class"] == "Shape"]
    assert len(area) == 1 and area[0]["properties"]["title"].startswith(
        "Search area, 805 m (0.50 mi) radius")
    assert not [f for f in drafts["features"] if f["properties"].get("marker-symbol") == "hut"]

    collection = read_export(markers_path)
    assert collection["type"] == "FeatureCollection"
    features = collection["features"]
    for feature in features:
        assert feature["type"] == "Feature"
        assert feature["geometry"]["type"] == "Point"
        assert feature["properties"]["class"] == "Marker"
        assert feature["properties"]["title"] and feature["properties"]["description"]
    place = features[0]
    assert place["properties"]["title"] == "Place last seen"
    assert place["geometry"] == {"type": "Point", "coordinates": [SAN_CARLOS[1],
                                                                  SAN_CARLOS[0]]}
    assert place["properties"]["marker-color"] == ORIGIN_COLOR
    assert "From plan_urban_search; nothing in this file is a decision." in (
        place["properties"]["description"])
    assert "Nearest hospitals ranked by road time from here." in (
        place["properties"]["description"])
    lots = [f for f in features if f["properties"].get("marker-color") == "#808080"]
    assert [f["properties"]["title"] for f in lots] == ["Library lot", "Station garage"]
    assert "120 m from the place last seen." in lots[0]["properties"]["description"]
    assert "covered_deck, fee" in lots[1]["properties"]["description"]
    hospitals = [f for f in features if f["properties"].get("role") == "hospital"]
    assert {f["properties"]["marker-color"] for f in hospitals} == {HOSPITAL_COLOR}
    assert len(features) == 1 + 2 + 2


def test_a_markers_file_not_written_is_said_at_the_top_level(ctx, monkeypatch):
    """plan_initial_response's test, here: the lots and hospitals found are
    in no file when only the markers file fails, so the top level says so,
    and the export's own warning names that file, not "the CalTopo
    export", which beside the drafts reads as both."""
    Stubs(monkeypatch)

    def full_disk(path, text):
        raise OSError("No space left on device")

    monkeypatch.setattr(_sections, "atomic_write_text", full_disk)
    result = plan_urban_search(ctx, *SAN_CARLOS)
    export = result["sections"]["export"]
    assert result["status"] == "ok"
    assert export["markers_path"] is None and result["exports"] == [export["path"]]
    assert export["warnings"][-1] == (
        "the markers file urban_search.geojson could not be written (OSError: No space left "
        "on device); every section's answer stands")
    (warning,) = result["warnings"]
    assert warning.startswith("the markers file urban_search.geojson was not written")
    assert "parking and medical found" in warning and "nothing was found" not in warning


def test_a_later_call_drawing_no_markers_names_the_earlier_markers_file(ctx, monkeypatch):
    """plan_initial_response's B9 test, here: a later call on the same ground
    asking for the assignments alone writes no markers file, so the earlier
    call's lots and hospitals are still beside the new drafts. The answer
    names that file as an earlier call's; it is not removed, since the
    earlier answer gave that path."""
    Stubs(monkeypatch)
    first = plan_urban_search(ctx, *SAN_CARLOS)
    earlier = first["sections"]["export"]["markers_path"]
    second = plan_urban_search(ctx, *SAN_CARLOS, include=["urban", "export"])
    export = second["sections"]["export"]
    assert second["handles"]["search"] == first["handles"]["search"]
    assert export["markers_path"] is None and second["exports"] == [export["path"]]
    assert Path(earlier).is_file() and Path(export["path"]).parent == Path(earlier).parent
    assert second["warnings"] == [
        f"the markers file at {earlier} is an earlier call's, left as it was: this call "
        "wrote none, so it is not this answer's, and exports does not list it"]


# --- Through MCP ------------------------------------------------------------------


@pytest.mark.anyio
async def test_plan_urban_search_is_registered_as_a_scenario_and_answers(monkeypatch,
                                                                         tmp_path):
    pytest.importorskip("fastmcp")
    from fastmcp import Client
    from fastmcp.exceptions import ToolError

    from sar_geo_mcp.server import build_server

    stubs = Stubs(monkeypatch)
    server = build_server(CaltopoConfig(cache_dir=tmp_path / "cache"))
    async with Client(server) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
        tool = tools["plan_urban_search"]
        assert tool.meta["fastmcp"]["tags"] == ["scenario"]
        assert "NO VERDICT" in tool.description and "detail_call" in tool.description
        assert "BUILDING MARKERS ARE OFF UNLESS ASKED FOR" in tool.description
        properties = tool.inputSchema["properties"]
        assert properties["include"]["anyOf"][0]["items"]["enum"] == list(SECTIONS)
        assert properties["radius_unit"]["enum"] == ["m", "km", "ft", "yd", "mi"]
        assert properties["building_markers"]["default"] is False

        answer = (await client.call_tool("plan_urban_search", {
            "latitude": SAN_CARLOS[0], "longitude": SAN_CARLOS[1], "radius": 0.25,
            "radius_unit": "mi", "include": ["urban", "parking"],
            "overrides": {"urban": {"populated_areas": {"min_parcels": 2}}}})).data
        assert answer["status"] == "ok"
        assert answer["sections"]["medical"]["status"] == "skipped"
        assert answer["summary"]["urban"]["count"] == 3
        assert answer["sections"]["urban"]["blocks"]["detail_call"]["arguments"][
            "min_parcels"] == 2
        assert stubs.calls["blocks"][0]["max_distance_m"] == 402.336

        with pytest.raises(ToolError):
            await client.call_tool("plan_urban_search", {
                "latitude": SAN_CARLOS[0], "longitude": SAN_CARLOS[1], "include": ["hasty"]})


# --- Against the real features ----------------------------------------------------


class TownStaging(FakeStaging):
    """FakeStaging over the synthetic town `test_populated.py` draws: the
    staged circle as `prepare_area` writes it (`subject.json`, from
    `build_area`), and Elm Street in `osm.json`."""

    def __call__(self, ctx, **arguments):
        from test_populated import elm_street

        from sar_geo_mcp.planning.area import build_area
        from sar_geo_mcp.terrain.linear import OSM_QUERY_VERSION

        payload = super().__call__(ctx, **arguments)
        area = build_area(center_latitude=arguments["center_latitude"],
                          center_longitude=arguments["center_longitude"],
                          radius_m=arguments["radius_m"])
        directory = Path(ctx.config.cache_dir) / "staged" / payload["handle"]
        (directory / "subject.json").write_text(area.model_dump_json(), encoding="utf-8")
        (directory / "osm.json").write_text(json.dumps(
            {"routes": [elm_street()], "parking": [], "query_version": OSM_QUERY_VERSION}),
            encoding="utf-8")
        return payload


def test_the_urban_section_reads_what_the_real_features_return(tmp_path, monkeypatch):
    """The real populated_areas, urban_assignments and caltopo_export over
    the synthetic town, its parcel and structures panels drawn by the fake
    fetcher test_populated uses. Holds the compact section to the features'
    actual payloads, which the stubs above only imitate."""
    from test_populated import CENTRE, TOKEN, fake_fetch, town

    from sar_geo_mcp.caltopo import wms

    ctx = ServerContext.build(CaltopoConfig(cache_dir=tmp_path / "cache", wms_token=TOKEN))
    monkeypatch.setattr(staging_component, "prepare_area", TownStaging())
    monkeypatch.setattr(wms, "fetch_panel", fake_fetch(*town()))
    real = {}
    for name in ("populated_areas", "urban_assignments"):
        function = getattr(urban_feature, name)

        def keep(ctx, _function=function, _name=name, **arguments):
            real[_name] = copy.deepcopy(_function(ctx, **arguments))
            return real[_name]

        monkeypatch.setattr(urban_feature, name, keep)

    result = plan_urban_search(ctx, *CENTRE, radius_m=170, target_ha=2.0,
                               building_markers=True, include=["urban", "export"])

    assert result["status"] == "ok", {name: entry["error"]
                                      for name, entry in result["sections"].items()}
    urban = result["sections"]["urban"]
    grouped = real["urban_assignments"]
    (assignment,) = grouped["candidates"]
    assert set(_sections.URBAN_FACTS) <= set(assignment["facts"])
    assert {"target_ha", "target_source"} <= set(grouped["assumptions"]["grouping"])
    assert {"count", "area_ha", "parcels", "buildings", "parcels_left_out", "markers"} <= set(
        grouped["summary"])
    assert set(_sections.BLOCK_COUNTS) <= set(real["populated_areas"]["summary"])
    (compact,) = urban["assignments"]
    assert compact["name"] == "Elm Street" and compact["streets"] == ["Elm Street"]
    assert (compact["blocks"], compact["parcels"], compact["buildings"]) == (2, 13, 16)
    assert compact["markers"] == 16 and urban["markers"] == 16
    assert urban["target_ha"] == 2.0 and urban["target_source"] == "given"
    # The blocks the assignments group are the ones counted.
    assert urban["blocks"]["count"] == compact["blocks"] == 2
    assert urban["blocks"]["parcels_in_blocks"] == urban["parcels"] == 13
    assert not any("the blocks under `blocks`" in warning for warning in urban["warnings"])
    assert geometry_types_in(result) == set()

    drafts = read_export(result["exports"][0])
    assignments = [f for f in drafts["features"] if f["properties"]["class"] == "Assignment"]
    assert [f["properties"]["letter"] for f in assignments] == ["U01 Elm Street"]
    assert assignments[0]["geometry"]["type"] == "Polygon"
    houses = [f for f in drafts["features"] if f["properties"].get("marker-symbol") == "hut"]
    assert len(houses) == 16
    circle = [f for f in drafts["features"] if f["properties"]["class"] == "Shape"]
    assert circle[0]["properties"]["title"].startswith("Search area, 170 m")
    assert math.isclose(drafts["features"][-1]["geometry"]["coordinates"][1], CENTRE[0])


def test_a_wider_max_distance_reads_no_block_beyond_the_search_circle(tmp_path, monkeypatch):
    """The real features over the synthetic town: lot lines are read over the
    staged circle only, so max_distance_m 1,200 on a 15 m circle keeps the
    very blocks the radius keeps - said on the section - while a wider
    radius does reach further."""
    from test_populated import CENTRE, TOKEN, fake_fetch, town

    from sar_geo_mcp.caltopo import wms

    monkeypatch.setattr(staging_component, "prepare_area", TownStaging())
    monkeypatch.setattr(wms, "fetch_panel", fake_fetch(*town()))

    def urban(name, radius_m, overrides=None):
        ctx = ServerContext.build(CaltopoConfig(cache_dir=tmp_path / name, wms_token=TOKEN))
        result = plan_urban_search(ctx, *CENTRE, radius_m=radius_m, target_ha=0.5,
                                   include=["urban"], overrides=overrides)
        assert result["status"] == "ok", result["sections"]["urban"]["error"]
        return result["sections"]["urban"]

    def counted(entry):
        return (entry["blocks"]["count"], entry["count"], entry["parcels"],
                entry["buildings"], [a["name"] for a in entry["assignments"]])

    circle = urban("circle", 15)
    wide = urban("wide", 15, {"urban": {"populated_areas": {"max_distance_m": 1200},
                                        "urban_assignments": {"max_distance_m": 1200}}})
    assert counted(wide) == counted(circle)
    assert any(warning.startswith("max_distance_m reaches past the 15 m search radius")
               for warning in wide["warnings"])
    assert not any("reaches past" in warning for warning in circle["warnings"])
    assert urban("wider", 150)["blocks"]["count"] > circle["blocks"]["count"]


def test_the_drafts_take_urban_assignments_marker_colour(tmp_path, monkeypatch):
    """The real features and caltopo_export: the colour given to
    urban_assignments is the one every house marker in the draft file has,
    and with none given to populated_areas nothing is said of colour."""
    from test_populated import CENTRE, TOKEN, fake_fetch, town

    from sar_geo_mcp.caltopo import wms

    ctx = ServerContext.build(CaltopoConfig(cache_dir=tmp_path / "cache", wms_token=TOKEN))
    monkeypatch.setattr(staging_component, "prepare_area", TownStaging())
    monkeypatch.setattr(wms, "fetch_panel", fake_fetch(*town()))
    result = plan_urban_search(
        ctx, *CENTRE, radius_m=170, target_ha=2.0, building_markers=True,
        include=["urban", "export"],
        overrides={"urban": {"urban_assignments": {"marker_color": "ff0000"}}})
    assert result["status"] == "ok"
    assert not any("house markers" in warning
                   for warning in result["sections"]["urban"]["warnings"])
    drafts = read_export(result["exports"][0])
    houses = [f for f in drafts["features"] if f["properties"].get("marker-symbol") == "hut"]
    assert len(houses) == 16
    assert {f["properties"]["marker-color"] for f in houses} == {"#FF0000"}


def test_ground_with_no_populated_block_is_an_empty_answer_not_a_failure(tmp_path,
                                                                         monkeypatch):
    """No lot lines drawn here: the real features say so, the section is ok
    with no assignment, and the export drafts the search area alone."""
    from test_populated import CENTRE, TOKEN, fake_fetch

    from sar_geo_mcp.caltopo import wms
    from sar_geo_mcp.planning.populated import NO_PARCELS_WARNING

    ctx = ServerContext.build(CaltopoConfig(cache_dir=tmp_path / "cache", wms_token=TOKEN))
    monkeypatch.setattr(staging_component, "prepare_area", TownStaging())
    monkeypatch.setattr(wms, "fetch_panel", fake_fetch([], []))
    result = plan_urban_search(ctx, *CENTRE, radius_m=170, include=["urban", "export"])

    assert result["status"] == "ok"
    urban = result["sections"]["urban"]
    assert urban["assignments"] == [] and urban["count"] == 0 and urban["area_ha"] is None
    assert NO_PARCELS_WARNING in urban["warnings"]
    assert urban["blocks"]["count"] == 0
    assert result["summary"]["urban"] == {"count": 0, "target_ha": 10.0, "area_ha": None,
                                          "parcels": 0, "buildings": 0}
    export = result["sections"]["export"]
    assert export["assignments"] == 0 and export["features"] == {"Shape": 1, "Marker": 1}
    assert result["warnings"] == []


# --- What the server's norms decide ---------------------------------------------


def over_the_town(monkeypatch, staging=None):
    """The real features' ground: the synthetic town staged by `staging`
    (TownStaging unless given), its CalTopo panels drawn by test_populated's
    fake fetcher."""
    from test_populated import fake_fetch, town

    from sar_geo_mcp.caltopo import wms

    monkeypatch.setattr(staging_component, "prepare_area", staging or TownStaging())
    monkeypatch.setattr(wms, "fetch_panel", fake_fetch(*town()))


def town_ctx(cache_dir: Path, **config) -> ServerContext:
    """A server with the WMS token the town's panels need."""
    from test_populated import TOKEN

    return ServerContext.build(CaltopoConfig(cache_dir=cache_dir, wms_token=TOKEN, **config))


def agency_norms(path: Path, edit) -> Path:
    """The shipped norms with `edit` applied, written where `--norms` reads
    another agency's file."""
    norms = copy.deepcopy(load_norms())
    edit(norms)
    path.write_text(json.dumps(norms), encoding="utf-8")
    return path


class OnceTownStaging(TownStaging):
    """TownStaging that leaves ground already staged as it is, as the real
    prepare_area does: FakeStaging rewrites the manifest on every call, and
    a set made before a rewrite is never reused."""

    def __init__(self):
        super().__init__()
        self.staged = {}

    def __call__(self, ctx, **arguments):
        key = (str(ctx.config.cache_dir), json.dumps(arguments, sort_keys=True))
        if key in self.staged:
            self.calls.append(arguments)
            return {**self.staged[key], "staged": False, "reason": "already staged"}
        self.staged[key] = super().__call__(ctx, **arguments)
        return self.staged[key]


def test_an_unknown_resource_type_is_sized_by_the_median_of_every_assignment(tmp_path,
                                                                               monkeypatch):
    """check_resource_type takes any word: one the corpus has no median for
    is sized by the median of every assignment, not refused - while a known
    type in lower case is that type, sized by its own median."""
    from test_populated import CENTRE

    over_the_town(monkeypatch)
    norms = load_norms()
    assert "DOGS" not in norms["area_ha_by_resource"]["medians"]

    def urban(name, resource_type):
        result = plan_urban_search(town_ctx(tmp_path / name), *CENTRE, radius_m=170,
                                   resource_type=resource_type, include=["urban", "export"])
        assert result["status"] == "ok", result["sections"]["urban"]["error"]
        return result

    typo = urban("typo", "DOGS")
    section = typo["sections"]["urban"]
    assert section["target_ha"] == norms["polygon_area_ha"]["p50"] == 9.1
    assert typo["summary"]["urban"]["target_ha"] == 9.1
    assert section["detail_call"]["arguments"]["resource_type"] == "DOGS"

    lower = urban("lower", "dog")["sections"]["urban"]
    assert (lower["target_ha"], lower["target_source"]) == (
        norms["area_ha_by_resource"]["medians"]["DOG"], "corpus median for DOG")


def test_an_unknown_resource_type_says_it_fell_back_to_the_overall_median(tmp_path,
                                                                          monkeypatch):
    """DOGS for DOG sizes the assignments ~70% larger (9.1 ha, not 5.4). The
    answer must not report a median for DOGS that the corpus does not have,
    and must say the type is unknown, so the typo can be seen."""
    from test_populated import CENTRE

    over_the_town(monkeypatch)
    result = plan_urban_search(town_ctx(tmp_path / "cache"), *CENTRE, radius_m=170,
                               resource_type="DOGS", include=["urban", "export"])
    section = result["sections"]["urban"]
    assert section["status"] == "ok" and section["target_ha"] == 9.1
    assert section["target_source"] != "corpus median for DOGS"
    assert any("DOGS" in warning for warning in [*section["warnings"], *result["warnings"]])


#: An agency's GROUND median under the synthetic town's 1.55 ha of blocks,
#: so it splits Elm Street in two where the shipped 10 ha keeps it whole.
AGENCY_GROUND_HA = 1.0


def agency_ground(norms: dict) -> None:
    norms["area_ha_by_resource"]["medians"]["GROUND"] = AGENCY_GROUND_HA


def test_an_agency_norms_file_sizes_the_urban_assignments(tmp_path, monkeypatch):
    """A server started with another agency's --norms groups the blocks
    toward that file's GROUND median - the one assignment_norms on the same
    server reports - not the shipped 10 ha: the section helpers pass the
    server's ctx, and its norms, through to the feature."""
    from test_populated import CENTRE

    over_the_town(monkeypatch)
    path = agency_norms(tmp_path / "agency.json", agency_ground)
    ctx = town_ctx(tmp_path / "agency", norms_path=path)
    assert norms_component.assignment_norms(ctx, "GROUND")["target_area_ha"]["p50"] == (
        AGENCY_GROUND_HA)

    result = plan_urban_search(ctx, *CENTRE, radius_m=170, include=["urban"])
    urban = result["sections"]["urban"]
    assert urban["status"] == "ok", urban["error"]
    assert (urban["target_ha"], urban["target_source"]) == (AGENCY_GROUND_HA,
                                                            "corpus median for GROUND")
    assert result["summary"]["urban"]["target_ha"] == AGENCY_GROUND_HA
    assert [a["name"] for a in urban["assignments"]] == ["Elm Street", "Elm Street"]

    shipped = plan_urban_search(town_ctx(tmp_path / "shipped"), *CENTRE, radius_m=170,
                                include=["urban"])["sections"]["urban"]
    assert (shipped["target_ha"], shipped["target_source"]) == (10.0,
                                                                "corpus median for GROUND")
    assert [a["name"] for a in shipped["assignments"]] == ["Elm Street"]


def test_an_agency_norms_file_sets_plan_initial_response_s_corridor_widths(tmp_path,
                                                                           monkeypatch):
    """The real hasty_candidates through plan_initial_response, on a flat
    staged ground with one trail and one creek: an agency's corridor widths are the widths
    its corridors are drawn at, and the set saved - and so its id - is the
    agency's. The detail_call names no width: made again on the same server,
    it reads the same norms."""
    from test_planning_tools import PLS, _stub_staging, _trail_through

    creek = _trail_through(PLS, offset_m=100, rid="route_1")
    creek["properties"].update(route_type="WATERWAY", name="Test Creek")
    _stub_staging(monkeypatch, routes=[_trail_through(PLS), creek])
    widths = {"waterway": 40}
    path = agency_norms(tmp_path / "agency.json",
                        lambda norms: norms["corridor_width_m"]["by_axis"].update(widths))

    def hasty(name, **config):
        ctx = ServerContext.build(CaltopoConfig(cache_dir=tmp_path / name, **config))
        # Every axis: the default trail-and-road list draws no corridors.
        result = plan_initial_response(ctx, *PLS, radius_m=400, include=["hasty"],
                                       overrides={"hasty": {"include": None}})
        assert result["status"] == "ok", result["sections"]["hasty"]["error"]
        return ctx, result["sections"]["hasty"]

    def corridor_widths(section):
        return {(c["axis"], c["width_m"]) for c in section["candidates"]
                if c["kind"] == "corridor"}

    ctx, agency = hasty("agency", norms_path=path)
    _, shipped = hasty("shipped")
    assert corridor_widths(agency) == {("waterway", 40.0)}
    assert corridor_widths(shipped) == {("waterway", 30.0)}
    saved = load_set(ctx.staged(agency["handle"]).directory, agency["set_id"])
    assert saved.params["corridor_widths_m"] == widths
    assert agency["set_id"] != shipped["set_id"]
    assert agency["detail_call"] == {"tool": "hasty_candidates", "arguments": {
        "handle": agency["handle"], "limit": 20,
        "include": ["trail", "road", "waterway"]}}


def test_assignments_grouped_under_the_shipped_norms_are_not_reused_under_an_agency_s(
        tmp_path, monkeypatch):
    """Grouped under the shipped norms (10 ha), then the server restarted
    with an agency's --norms (GROUND median 1 ha) over the same cache: the
    blocks - no norm in them - may be read back, but the assignments must be
    grouped again, as a fresh cache groups them (two, not one), not returned
    as current with the old target labelled the corpus median."""
    from test_populated import CENTRE

    staging = OnceTownStaging()
    over_the_town(monkeypatch, staging)
    path = agency_norms(tmp_path / "agency.json", agency_ground)
    cache = tmp_path / "cache"

    def urban(ctx):
        result = plan_urban_search(ctx, *CENTRE, radius_m=170, include=["urban"])
        section = result["sections"]["urban"]
        assert section["status"] == "ok", section["error"]
        return section

    def grouped(section):
        return (section["target_ha"], section["count"],
                [a["name"] for a in section["assignments"]])

    first = urban(town_ctx(cache))
    assert first["target_ha"] == 10.0 and first["reused"] is False
    again = urban(town_ctx(cache, norms_path=path))
    fresh = urban(town_ctx(tmp_path / "fresh", norms_path=path))
    # The same ground, not staged again: its blocks are read back.
    assert again["handle"] == first["handle"] and again["blocks"]["reused"] is True
    assert grouped(fresh) == (AGENCY_GROUND_HA, 2, ["Elm Street", "Elm Street"])
    assert again["reused"] is False
    assert grouped(again) == grouped(fresh)


def test_hasty_flags_raised_under_the_shipped_norms_are_not_reused_under_an_agency_s(
        tmp_path, monkeypatch):
    """plan_initial_response's hasty set, made under the shipped norms, then
    the server restarted with an agency's line-length band (1 to 5 m) over
    the same cache: the corridor widths are unchanged, so the set id is the
    same, but its norm flags are read against the old band. The answer must
    carry the flags the agency's own band raises, as a fresh cache does."""
    from test_planning_tools import PLS, _stub_staging, _trail_through

    _stub_staging(monkeypatch, routes=[_trail_through(PLS)])
    path = agency_norms(tmp_path / "agency.json", lambda norms: norms["line_length_m"].update(
        p10=1, p50=3, p90=5))
    cache = tmp_path / "cache"

    def hasty(cache_dir, **config):
        ctx = ServerContext.build(CaltopoConfig(cache_dir=cache_dir, **config))
        result = plan_initial_response(ctx, *PLS, radius_m=400, include=["hasty"])
        section = result["sections"]["hasty"]
        assert section["status"] == "ok", section["error"]
        return section

    def trail_flags(section):
        (trail,) = [c for c in section["candidates"] if c["axis"] == "trail"]
        return trail["flags"]

    first = hasty(cache)
    again = hasty(cache, norms_path=path)
    fresh = hasty(tmp_path / "fresh", norms_path=path)
    assert trail_flags(first) == []
    assert any("line length" in flag for flag in trail_flags(fresh))
    assert again["handle"] == first["handle"]
    assert again["reused"] is False
    assert [c["flags"] for c in again["candidates"]] == [c["flags"] for c in fresh["candidates"]]


def test_parking_unknown_for_want_of_osm_is_null_not_none(ctx, monkeypatch):
    """The real parking feature on ground staged during an OpenStreetMap
    outage, as plan_initial_response's twin: parking is unknown, so the
    summary's counts are null - never the 0 a town with no lot gives."""
    from test_planning_tools import PLS

    _stage_without_osm(monkeypatch)
    result = plan_urban_search(ctx, *PLS, radius_m=400, include=["parking"])
    assert result["status"] == "ok"
    assert result["staging"]["search"]["osm_available"] is False
    assert any("unknown rather than absent" in warning
               for warning in result["sections"]["parking"]["warnings"])
    assert result["sections"]["parking"]["total_before_limit"] is None
    assert result["summary"]["parking"] == {"count": None, "total_before_limit": None,
                                            "nearest": None}


@pytest.mark.parametrize("route_error, ranking, ranked_by", [
    (None, RANKED_BY_ROAD, "road_time"),
    (httpx.ConnectError("OSRM unreachable"), RANKED_STRAIGHT_ROUTER_DOWN, "straight_line"),
], ids=["routed", "router-down"])
def test_the_place_last_seen_says_how_the_hospitals_were_ranked(ctx, monkeypatch,
                                                                route_error, ranking,
                                                                ranked_by):
    """PR #4 review B21 in this scenario's markers file and summary: with the
    road router down the place does not claim road-time ranking beside
    hospitals numbered "#1", and the nearest trauma center says how it is
    nearest."""
    stubs = Stubs(monkeypatch)
    stubs.services.route_error = route_error
    result = plan_urban_search(ctx, *SAN_CARLOS)
    markers_path = Path(result["sections"]["export"]["markers_path"])
    place = read_export(markers_path)["features"][0]
    assert place["properties"]["title"] == "Place last seen"
    lines = place["properties"]["description"].split("\n")
    assert [line for line in lines if "hospitals ranked" in line] == [ranking]
    assert result["summary"]["medical"]["nearest_trauma_center"]["ranked_by"] == ranked_by
