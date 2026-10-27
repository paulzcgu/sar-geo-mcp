"""plan_medevac: landing zones, the way out and the hospitals in one call.

No network. Staging and the three feature functions are stubbed with
monkeypatch where they live (`components.staging.prepare_area`,
`features.lz.lz_candidates`, ...), which is where the scenario reaches them;
the medical stub runs the real medical logic over faked HIFLD, OSRM and
Overpass answers, so its rows and markers are the real shape. One test at
the end runs the real landing-zone and egress features over flat ground
written to disk, to hold the compact sections to the payloads the features
actually return. The shared machinery has its own tests in
`test_scenarios_sections.py`.
"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path

import httpx
import pytest
from test_medical import FakeServices, record
from test_scenarios_sections import FakeStaging, geometry_types_in

from sar_geo_mcp.components import staging as staging_component
from sar_geo_mcp.config import CaltopoConfig
from sar_geo_mcp.context import ServerContext
from sar_geo_mcp.features import egress as egress_feature
from sar_geo_mcp.features import lz as lz_feature
from sar_geo_mcp.features import medical as medical_feature
from sar_geo_mcp.planning.medical import (HOSPITAL_COLOR, HOSPITAL_SYMBOL, ORIGIN_COLOR,
                                          ORIGIN_SYMBOL, WIDE_RADIUS_KM)
from sar_geo_mcp.planning.medical import nearest_medical_facilities as medical_impl
from sar_geo_mcp.scenarios import _sections
from sar_geo_mcp.scenarios.medevac import plan_medevac
from exports import read_export

PATIENT = (37.4022, -122.3164)
STANFORD = (37.434, -122.175)
SEQUOIA = (37.48, -122.24)


def _square(lon, lat, half=0.0004):
    return {"type": "Polygon", "coordinates": [[
        [lon - half, lat - half], [lon + half, lat - half], [lon + half, lat + half],
        [lon - half, lat + half], [lon - half, lat - half]]]}


def lz_payload(arguments):
    """What lz_candidates returns, three zones nearest first, the patch
    outline included - which the section must leave out."""
    lat, lon = arguments["center_latitude"], arguments["center_longitude"]

    def zone(index, grade, distance, powerline, road):
        touchdown = [lon + 0.002 * index, lat]
        return {"id": f"lz_{index}", "grade": grade,
                "touchdown": {"type": "Point", "coordinates": touchdown},
                "clear_diameter_m": 40.0 + index,
                "preferred_fit": {"fits": grade == "preferred", "angle_deg": 30.0},
                "slope_mean_deg": 2.5, "slope_max_deg": 4.1, "canopy_fraction": 0.04,
                "elevation_m": 410.0, "aspect": None, "distance_m": distance,
                "bearing_deg": 90.0, "powerline": powerline, "road": road,
                "parking_lot": None, "geometry": _square(*touchdown),
                "chips": [{"source": "caltopo_imagery", "label": "CalTopo imagery",
                           "path": f"/chips/lz_{index}.png", "attribution": "CalTopo"}],
                "chip_path": f"/chips/lz_{index}.png", "warnings": []}

    zones = [zone(1, "minimal", 180.0, None,
                  {"name": "Skyline Blvd", "route_type": "ROAD", "highway": "secondary",
                   "surface": "asphalt", "distance_m": 35.0, "on_road": True}),
             zone(2, "preferred", 420.0,
                  {"distance_m": 120.0, "name": None, "power": "minor_line"}, None),
             zone(3, "marginal", 600.0, "unknown", "unknown")]
    limit = arguments.get("limit", 10)
    kept = zones[:limit] if limit else zones
    return {"candidates": kept,
            "summary": {"count": len(kept), "total_before_limit": 3,
                        "truncated_to": limit if limit and limit < 3 else None,
                        "grades": {"preferred": 1, "minimal": 1, "marginal": 1}},
            "center": {"latitude": lat, "longitude": lon}, "resolution_m": 2.0,
            "assumptions": {"structures": "checked", "trees": "not checked",
                            "water_sources_checked": ["osm_shorelines"],
                            "powerline_kinds_checked": ["line", "minor_line"],
                            "road_class": "checked", "max_slope_deg": 5.0},
            "warnings": ["trees were not checked (no WMS token)"],
            "staleness": {"current": True}}


def egress_payload(arguments):
    """What egress_routes returns for one mode: a reachable 2WD exit, route
    line and all, and an unreachable 4x4 one."""
    lat, lon = arguments["center_latitude"], arguments["center_longitude"]
    mode = arguments["mode"]
    minutes = {"carry_out": 95.0, "walk_out": 31.0}[mode]
    end = [lon, lat + 0.012]
    two_wd = {"kind": "2wd", "reachable": True,
              "exit_point": {"type": "Point", "coordinates": end},
              "route": {"type": "LineString", "coordinates": [[lon, lat], end]},
              "minutes": minutes, "distance_m": 1840.0, "gain_m": 60.0, "loss_m": 120.0,
              "on_trail_share": 0.8, "off_trail_m": 200.0, "max_grade_pct": 18.0,
              "off_trail_after_joining_m": 0.0,
              "steep_pitch_m": 12.0 if mode == "carry_out" else None,
              "road": {"name": "Skyline Blvd", "route_type": "ROAD", "highway": "secondary",
                       "surface": "asphalt", "access": None, "gate_on_road": False},
              "parking_lot": None,
              "gates_crossed": [{"geometry": {"type": "Point", "coordinates": end},
                                 "barrier": "gate", "access": None}],
              "warnings": []}
    four = {"kind": "4x4", "reachable": False,
            "reason": "no track or dirt road a 4x4 can reach is mapped in the staged area"}
    warnings = ["gates are read from OpenStreetMap"]
    if mode == "walk_out":
        warnings.append("this ground has no LANDFIRE vegetation")
    return {"exits": [two_wd, four], "mode": mode,
            "center": {"latitude": lat, "longitude": lon}, "resolution_m": 10.0,
            "assumptions": {}, "warnings": warnings, "staleness": {"current": True}}


class Stubs:
    """Staging and the three features, recording every call. `fail` names
    sections whose feature raises; `fail_modes` egress modes that raise;
    `fail_resolutions` grounds staging cannot fetch."""

    def __init__(self, monkeypatch, *, fail=(), fail_modes=(), fail_resolutions=()):
        self.fail, self.fail_modes = set(fail), set(fail_modes)
        self.staging = FakeStaging(fail_resolutions)
        self.calls = {"lz": [], "egress": [], "medical": []}
        self.services = FakeServices(
            [record("Stanford Hospital", *STANFORD, trauma="LEVEL I", helipad="Y",
                    address="300 PASTEUR DR", city="STANFORD"),
             record("Sequoia Hospital", *SEQUOIA, address="170 ALAMEDA DE LAS PULGAS",
                    city="REDWOOD CITY")],
            minutes={STANFORD: 40.0, SEQUOIA: 30.0})
        monkeypatch.setattr(staging_component, "prepare_area", self.staging)
        monkeypatch.setattr(lz_feature, "lz_candidates", self.lz)
        monkeypatch.setattr(egress_feature, "egress_routes", self.egress)
        monkeypatch.setattr(medical_feature, "nearest_medical_facilities", self.medical)

    def lz(self, ctx, **arguments):
        self.calls["lz"].append(arguments)
        ctx.staged(arguments["handle"])          # as the tool does: the handle must resolve
        if "lz" in self.fail:
            raise ValueError("no canopy panel: the ir fetch failed (HTTPStatusError 503)")
        return lz_payload(arguments)

    def egress(self, ctx, **arguments):
        self.calls["egress"].append(arguments)
        ctx.staged(arguments["handle"])
        if "egress" in self.fail or arguments["mode"] in self.fail_modes:
            raise ValueError(f"{arguments['mode']}: ground staged too fine")
        return egress_payload(arguments)

    def medical(self, ctx, **arguments):
        self.calls["medical"].append(arguments)
        if "medical" in self.fail:
            raise httpx.ConnectError("HIFLD could not be reached")
        rest = {key: value for key, value in arguments.items()
                if key not in ("latitude", "longitude")}
        return medical_impl(arguments["latitude"], arguments["longitude"], **rest,
                            query=self.services.query, route=self.services.route,
                            helipads=self.services.helipads, locate=self.services.locate)

    def nothing_called(self) -> bool:
        return not self.staging.calls and not any(self.calls.values())


@pytest.fixture
def ctx(tmp_path):
    return ServerContext.build(CaltopoConfig(cache_dir=tmp_path / "cache"))


def staged_dir(ctx, handle) -> Path:
    return Path(ctx.config.cache_dir) / "staged" / handle


def exported(path) -> list[dict]:
    """The features of an export file, as CalTopo would import them."""
    return read_export(path)["features"]


def with_lz(monkeypatch, change):
    """lz_candidates answering `lz_payload`, as `change` edits it in place."""
    def lz(ctx, **arguments):
        payload = lz_payload(arguments)
        change(payload)
        return payload
    monkeypatch.setattr(lz_feature, "lz_candidates", lz)


NEAREST_ANY = {"id": "lz_1", "grade": "minimal", "distance_m": 180.0, "bearing_deg": 90.0,
               "clear_diameter_m": 41.0}
NEAREST_PREFERRED = {"id": "lz_2", "grade": "preferred", "distance_m": 420.0,
                     "bearing_deg": 90.0, "clear_diameter_m": 42.0}


# --- Every section ------------------------------------------------------------


def test_every_section_answers_and_the_summary_reads_the_facts(ctx, monkeypatch):
    stubs = Stubs(monkeypatch)
    result = plan_medevac(ctx, *PATIENT)

    assert result["status"] == "ok"
    assert result["patient"] == {"latitude": PATIENT[0], "longitude": PATIENT[1]}
    assert {name: entry["status"] for name, entry in result["sections"].items()} == {
        "lz": "ok", "egress": "ok", "medical": "ok"}
    assert all(entry["error"] is None for entry in result["sections"].values())

    # Two grounds, each centred on the patient at the resolution its feature needs.
    centre = {"center_latitude": PATIENT[0], "center_longitude": PATIENT[1],
              "map_id": "", "public": True}
    assert stubs.staging.calls == [{**centre, "radius_m": 1000.0, "resolution_m": 2.0},
                                   {**centre, "radius_m": 3000.0, "resolution_m": 10.0}]
    handles = result["handles"]
    assert set(handles) == {"lz", "egress"} and handles["lz"] != handles["egress"]
    assert result["staging"]["lz"]["handle"] == handles["lz"]

    lz = result["sections"]["lz"]
    assert lz["ground"] == "lz" and lz["handle"] == handles["lz"]
    assert [zone["id"] for zone in lz["candidates"]] == ["lz_1", "lz_2", "lz_3"]
    first = lz["candidates"][0]
    assert "geometry" not in first and first["touchdown"]["type"] == "Point"
    assert first["road"]["on_road"] is True and first["chip_path"] == "/chips/lz_1.png"
    assert lz["candidates"][2]["powerline"] == "unknown"
    assert lz["grades"] == {"preferred": 1, "minimal": 1, "marginal": 1}
    assert lz["checks"]["trees"] == "not checked"
    assert lz["warnings"] == ["trees were not checked (no WMS token)"]
    assert lz["detail_call"] == {"tool": "lz_candidates", "arguments": {
        "handle": handles["lz"], "center_latitude": PATIENT[0],
        "center_longitude": PATIENT[1], "limit": 5, "whole_ground": True}}

    egress = result["sections"]["egress"]
    assert list(egress["modes"]) == ["carry_out", "walk_out"]
    carry = egress["modes"]["carry_out"]
    assert carry["detail_call"] == {"tool": "egress_routes", "arguments": {
        "handle": handles["egress"], "center_latitude": PATIENT[0],
        "center_longitude": PATIENT[1], "mode": "carry_out"}}
    assert carry["exits"][0] == {
        "kind": "2wd", "reachable": True, "minutes": 95.0, "distance_m": 1840.0,
        "gain_m": 60.0, "loss_m": 120.0,
        "exit_point": {"type": "Point", "coordinates": [PATIENT[1], PATIENT[0] + 0.012]},
        "road_name": "Skyline Blvd", "road_access": None, "gate_on_road": False,
        "parking_lot": None, "gates_crossed": 1, "off_trail_m": 200.0,
        "on_trail_share": 0.8, "off_trail_after_joining_m": 0.0, "steep_pitch_m": 12.0,
        "litter_start": None, "warnings": []}
    assert carry["exits"][1]["reachable"] is False and "4x4" in carry["exits"][1]["reason"]
    # One warning both modes gave, plain; one only walk-out gave, attributed.
    assert egress["warnings"] == ["gates are read from OpenStreetMap",
                                  "walk_out: this ground has no LANDFIRE vegetation"]

    medical = result["sections"]["medical"]
    assert [row["name"] for row in medical["categories"]["trauma_center"]] == [
        "Stanford Hospital"]
    assert [row["name"] for row in medical["categories"]["medical_center"]] == [
        "Sequoia Hospital", "Stanford Hospital"]           # by road time, the feature's order
    assert medical["categories"]["pediatric_trauma_center"] == []
    assert medical["table"].startswith("| Category |")
    assert medical["in_us"] is True
    assert medical["detail_call"] == {"tool": "nearest_medical_facilities", "arguments": {
        "latitude": PATIENT[0], "longitude": PATIENT[1], "per_category": 3}}

    summary = result["summary"]
    assert summary["lz"] == {
        "nearest_any": {"id": "lz_1", "grade": "minimal", "distance_m": 180.0,
                        "bearing_deg": 90.0, "clear_diameter_m": 41.0},
        "nearest_preferred": {"id": "lz_2", "grade": "preferred", "distance_m": 420.0,
                              "bearing_deg": 90.0, "clear_diameter_m": 42.0},
        "preferred_beyond_limit": 0, "preferred_detail_call": None}
    vehicle_end = {"road_name": "Skyline Blvd", "road_access": None, "gate_on_road": False,
                   "parking_lot": None}
    assert summary["egress"] == {
        "carry_out": {"kind": "2wd", "minutes": 95.0, "distance_m": 1840.0,
                      "off_trail_m": 200.0, **vehicle_end},
        "walk_out": {"kind": "2wd", "minutes": 31.0, "distance_m": 1840.0,
                     "off_trail_m": 200.0, **vehicle_end}}
    trauma = summary["medical"]["nearest_trauma_center"]
    assert trauma["name"] == "Stanford Hospital" and trauma["helipad"] == "Y"
    assert trauma["road_minutes"] == 40.0 and trauma["air_minutes"] is not None

    assert result["exports"] == [str(staged_dir(ctx, handles["lz"]) / "exports"
                                     / "medevac.geojson")]
    assert result["warnings"] == []
    assert "No verdict" in result["note"]


def test_the_summary_says_when_the_limit_left_the_preferred_zone_out(ctx, monkeypatch):
    """lz_candidates orders by distance first, so on real ground the nearest
    five are often all minimal or marginal: a bare null would read as "no
    preferred zone near", so the summary counts them and names the call
    that lists them."""
    stubs = Stubs(monkeypatch)
    result = plan_medevac(ctx, *PATIENT, lz_limit=1, include=["lz"],
                          overrides={"lz": {"max_slope_deg": 6.0}})
    lz = result["summary"]["lz"]
    assert lz["nearest_any"]["id"] == "lz_1"
    assert lz["nearest_preferred"] is None
    assert lz["preferred_beyond_limit"] == 1
    assert lz["preferred_detail_call"] == {"tool": "lz_candidates", "arguments": {
        "handle": result["handles"]["lz"], "center_latitude": PATIENT[0],
        "center_longitude": PATIENT[1], "limit": 0, "whole_ground": True,
        "max_slope_deg": 6.0}}
    assert result["sections"]["lz"]["truncated_to"] == 1
    # The call it names lists the preferred zone.
    listed = stubs.lz(ctx, **lz["preferred_detail_call"]["arguments"])
    assert "preferred" in [zone["grade"] for zone in listed["candidates"]]


def test_the_answer_carries_points_but_no_lines_or_polygons(ctx, monkeypatch):
    Stubs(monkeypatch)
    result = plan_medevac(ctx, *PATIENT)
    assert geometry_types_in(result) == {"Point"}
    json.dumps(result)                       # and it is JSON as it stands


def _no_preferred_anywhere(payload):
    payload["candidates"][1]["grade"] = "minimal"
    payload["summary"]["grades"] = {"preferred": 0, "minimal": 2, "marginal": 1}


def _no_zones_at_all(payload):
    payload["candidates"] = []
    payload["summary"].update(count=0, total_before_limit=0, truncated_to=None,
                              grades={"preferred": 0, "minimal": 0, "marginal": 0})


def _more_preferred_past_the_limit(payload):
    # The limit kept lz_1 (minimal) and lz_2 (preferred); two more preferred
    # zones lie past it.
    payload["summary"].update(total_before_limit=5,
                              grades={"preferred": 3, "minimal": 1, "marginal": 1})


@pytest.mark.parametrize("change, lz_limit, expected", [
    (_no_preferred_anywhere, 5,
     {"nearest_any": NEAREST_ANY, "nearest_preferred": None,
      "preferred_beyond_limit": 0, "preferred_detail_call": None}),
    (_no_zones_at_all, 5,
     {"nearest_any": None, "nearest_preferred": None,
      "preferred_beyond_limit": 0, "preferred_detail_call": None}),
    (_more_preferred_past_the_limit, 2,
     {"nearest_any": NEAREST_ANY, "nearest_preferred": NEAREST_PREFERRED,
      "preferred_beyond_limit": 2, "preferred_detail_call": None}),
], ids=["no-preferred-zone-on-the-ground", "no-zone-at-all",
        "preferred-inside-and-more-past-the-limit"])
def test_the_lz_summary_tells_none_on_the_ground_from_none_within_the_limit(
        ctx, monkeypatch, change, lz_limit, expected):
    """A planner tells "no preferred zone on the ground" (null, 0) from "none
    within the limit" (null, a count and a call) by these four fields alone;
    a preferred zone inside the limit is the nearest one anywhere, so there is
    nothing more to list even when more lie past it."""
    Stubs(monkeypatch)
    with_lz(monkeypatch, change)
    result = plan_medevac(ctx, *PATIENT, include=["lz"], lz_limit=lz_limit)
    assert result["status"] == "ok"
    assert result["sections"]["lz"]["status"] == "ok"
    assert result["summary"]["lz"] == expected


def test_no_trauma_center_found_is_an_answer_not_a_failed_lookup(ctx, monkeypatch):
    """`nearest_trauma_center` null inside an answered block is "none within
    the search radius"; the whole block null is "the lookup failed". An
    operator choosing a destination must be able to tell them apart."""
    stubs = Stubs(monkeypatch)
    stubs.services.records = [record("Sequoia Hospital", *SEQUOIA,
                                     address="170 ALAMEDA DE LAS PULGAS",
                                     city="REDWOOD CITY")]
    result = plan_medevac(ctx, *PATIENT)
    assert result["status"] == "ok"
    medical = result["sections"]["medical"]
    assert medical["status"] == "ok" and medical["error"] is None
    assert medical["categories"]["trauma_center"] == []
    assert [row["name"] for row in medical["categories"]["medical_center"]] == [
        "Sequoia Hospital"]
    assert (f"No trauma center was found within {WIDE_RADIUS_KM:.0f} km. The hospital list "
            "covers the 50 states, DC and US territories only.") in medical["warnings"]
    assert result["summary"]["medical"] == {"nearest_trauma_center": None}
    # The hospital that was found is still drawn, and the patient's ranking
    # line is true of it.
    features = exported(result["exports"][0])
    assert [f["properties"]["title"] for f in features
            if f["properties"].get("role") == "hospital"] == ["Sequoia Hospital"]
    assert "Nearest hospitals ranked by road time from here." in (
        features[0]["properties"]["description"])


def test_with_the_road_router_down_hospitals_are_ranked_by_straight_line(ctx, monkeypatch):
    """OSRM unreachable: no road minutes anywhere, each category in
    straight-line order - Sequoia (about 11 km) before Stanford (about
    13 km), although the road times the router would have given put
    Stanford first - and the section says why."""
    stubs = Stubs(monkeypatch)
    stubs.services.minutes = {STANFORD: 20.0, SEQUOIA: 30.0}
    stubs.services.route_error = httpx.ConnectError("OSRM unreachable")
    result = plan_medevac(ctx, *PATIENT)
    assert result["status"] == "ok"
    medical = result["sections"]["medical"]
    assert medical["status"] == "ok"
    assert medical["warnings"][0] == (
        "The road router could not be reached (ConnectError): road distance and time are "
        "missing, and each category is ranked by straight-line distance instead.")
    assert medical["road_start"] is None
    assert [row["name"] for row in medical["categories"]["medical_center"]] == [
        "Sequoia Hospital", "Stanford Hospital"]
    rows = [row for rows in medical["categories"].values() for row in rows]
    assert rows and all(row["road_minutes"] is None and row["road_distance_m"] is None
                        for row in rows)
    assert result["summary"]["medical"] == {"nearest_trauma_center": {
        "name": "Stanford Hospital", "trauma": "Level I", "helipad": "Y",
        "road_minutes": None, "air_minutes": 8.3, "ranked_by": "straight_line"}}
    # No road route, so no "road route starts" line on the patient.
    patient = exported(result["exports"][0])[0]
    assert "Road route starts" not in patient["properties"]["description"]


@pytest.mark.parametrize("services, in_us, first_warning", [
    ({"state": None}, False, "The point is not inside any US state, DC or territory"),
    ({"locate_error": httpx.ConnectError("TIGERweb could not be reached")}, None,
     "Whether the point is inside the US could not be checked (ConnectError)"),
], ids=["outside-the-us", "could-not-check"])
def test_the_medical_section_says_when_the_us_only_list_may_miss_a_hospital(
        ctx, monkeypatch, services, in_us, first_warning):
    """HIFLD is US-only: outside the US (`in_us` false) or when that could not
    be checked (null), the section says so first - a closer hospital across
    a border would be missing from every category."""
    stubs = Stubs(monkeypatch)
    for name, value in services.items():
        setattr(stubs.services, name, value)
    result = plan_medevac(ctx, *PATIENT, include=["medical", "lz"])
    medical = result["sections"]["medical"]
    assert medical["status"] == "ok"
    assert medical["in_us"] is in_us
    assert medical["warnings"][0].startswith(first_warning)
    assert result["summary"]["medical"]["nearest_trauma_center"]["name"] == "Stanford Hospital"


# --- Failures stay in their section --------------------------------------------


def test_one_section_failing_leaves_the_others_and_the_status_partial(ctx, monkeypatch):
    Stubs(monkeypatch, fail={"medical"})
    result = plan_medevac(ctx, *PATIENT)
    assert result["status"] == "partial"
    medical = result["sections"]["medical"]
    assert medical["status"] == "failed"
    assert medical["error"] == "ConnectError: HIFLD could not be reached"
    assert result["sections"]["lz"]["status"] == "ok"
    assert result["sections"]["egress"]["status"] == "ok"
    assert result["summary"]["medical"] is None and result["summary"]["lz"] is not None
    assert result["exports"]                                  # what was found is still drawn


def test_a_staging_failure_fails_only_the_sections_on_that_ground(ctx, monkeypatch):
    stubs = Stubs(monkeypatch, fail_resolutions={2.0})
    result = plan_medevac(ctx, *PATIENT)
    assert result["status"] == "partial"
    lz = result["sections"]["lz"]
    assert lz["status"] == "failed" and lz["ground"] == "lz"
    assert lz["error"].startswith("StagingFailed: the lz ground (1,000 m radius at 2 m) "
                                  "could not be staged")
    assert "RuntimeError: 3DEP could not be reached" in lz["error"]
    assert stubs.calls["lz"] == []                           # never reached the feature
    # No feature call was made, so none to repeat: the staging record has it.
    assert "detail_call" not in lz
    assert result["staging"]["lz"]["detail_call"]["tool"] == "prepare_area"
    assert result["sections"]["egress"]["status"] == "ok"
    assert result["sections"]["medical"]["status"] == "ok"
    assert result["staging"]["lz"]["status"] == "failed"
    assert result["handles"]["lz"] is None
    # The export goes beside the ground that was staged.
    egress_handle = result["handles"]["egress"]
    assert result["exports"] == [str(staged_dir(ctx, egress_handle) / "exports"
                                     / "medevac.geojson")]


def test_an_egress_ground_staging_failure_fails_only_egress(ctx, monkeypatch):
    """The 10 m exit ground cannot be fetched: egress fails naming that
    ground, with no modes and no feature call to repeat; the landing zones
    and hospitals stand, and are drawn beside the landing-zone ground."""
    stubs = Stubs(monkeypatch, fail_resolutions={10.0})
    result = plan_medevac(ctx, *PATIENT)
    assert result["status"] == "partial"
    assert result["sections"]["egress"] == {
        "status": "failed",
        "error": "StagingFailed: the egress ground (3,000 m radius at 10 m) could not be "
                 "staged, so this section did not run - RuntimeError: 3DEP could not be "
                 "reached",
        "warnings": [], "ground": "egress"}
    assert stubs.calls["egress"] == []
    assert result["summary"]["egress"] is None
    assert result["handles"] == {"lz": result["staging"]["lz"]["handle"], "egress": None}
    egress_staging = result["staging"]["egress"]
    assert egress_staging["status"] == "failed"
    assert egress_staging["detail_call"]["tool"] == "prepare_area"
    assert egress_staging["detail_call"]["arguments"]["resolution_m"] == 10.0
    assert result["sections"]["lz"]["status"] == "ok"
    assert result["sections"]["medical"]["status"] == "ok"
    assert result["summary"]["lz"]["nearest_preferred"]["id"] == "lz_2"
    assert result["exports"] == [str(staged_dir(ctx, result["handles"]["lz"]) / "exports"
                                     / "medevac.geojson")]
    features = exported(result["exports"][0])
    assert [f["geometry"]["type"] for f in features].count("LineString") == 0
    assert len(features) == 1 + 3 + 2                     # patient, zones, hospitals
    assert result["warnings"] == []


def test_both_grounds_failing_still_answers_medical_but_writes_no_file(ctx, monkeypatch):
    """Hospitals need no ground, so they still answer - but the export goes
    beside a staged ground and there is none: the answer says so."""
    Stubs(monkeypatch, fail_resolutions={2.0, 10.0})
    result = plan_medevac(ctx, *PATIENT)
    assert result["status"] == "partial"
    assert {name: entry["status"] for name, entry in result["sections"].items()} == {
        "lz": "failed", "egress": "failed", "medical": "ok"}
    assert result["handles"] == {"lz": None, "egress": None}
    assert result["summary"]["lz"] is None and result["summary"]["egress"] is None
    assert result["summary"]["medical"]["nearest_trauma_center"]["name"] == "Stanford Hospital"
    assert result["exports"] == []
    assert result["warnings"] == [
        "no ground was staged, so there is nowhere beside it to write the CalTopo export; "
        "each section's detail_call returns what it found"]


def test_every_section_failing_is_failed_and_draws_nothing(ctx, monkeypatch):
    Stubs(monkeypatch, fail={"lz", "egress", "medical"})
    result = plan_medevac(ctx, *PATIENT)
    assert result["status"] == "failed"
    assert all(entry["status"] == "failed" for entry in result["sections"].values())
    # Every mode raised: as null as a ground that could not be staged.
    assert result["summary"] == {"lz": None, "egress": None, "medical": None}
    assert result["exports"] == []
    assert result["warnings"] == ["nothing was found to draw, so no CalTopo export was "
                                  "written"]


def test_one_egress_mode_failing_keeps_the_other(ctx, monkeypatch):
    Stubs(monkeypatch, fail_modes={"carry_out"})
    result = plan_medevac(ctx, *PATIENT)
    egress = result["sections"]["egress"]
    assert egress["status"] == "failed"
    assert egress["error"] == "ValueError: carry_out: ground staged too fine"
    assert egress["modes"]["carry_out"]["status"] == "failed"
    assert egress["modes"]["walk_out"]["status"] == "ok"
    assert egress["modes"]["walk_out"]["exits"][0]["minutes"] == 31.0
    assert result["summary"]["egress"]["carry_out"] is None
    assert result["summary"]["egress"]["walk_out"]["minutes"] == 31.0
    assert result["status"] == "partial"


def test_egress_alone_with_one_mode_answering_is_partial_not_failed(ctx, monkeypatch):
    """The egress section fails when a mode fails, but a walk-out exit that
    came back is a way out: the answer must not read "failed"."""
    Stubs(monkeypatch, fail_modes={"carry_out"})
    result = plan_medevac(ctx, *PATIENT, include=["egress"])
    egress = result["sections"]["egress"]
    assert egress["status"] == "failed"
    assert egress["modes"]["walk_out"]["status"] == "ok"
    assert result["status"] == "partial"
    assert result["summary"]["egress"]["carry_out"] is None
    assert result["summary"]["egress"]["walk_out"]["minutes"] == 31.0
    # Carry-out raised and said nothing, so walk-out's warnings are its own.
    assert egress["warnings"] == ["walk_out: gates are read from OpenStreetMap",
                                  "walk_out: this ground has no LANDFIRE vegetation"]
    assert result["exports"]                                  # the walk-out route is drawn


def test_one_egress_mode_is_partial_even_when_every_other_section_failed(ctx, monkeypatch):
    Stubs(monkeypatch, fail={"lz", "medical"}, fail_modes={"carry_out"})
    result = plan_medevac(ctx, *PATIENT)
    assert {name: entry["status"] for name, entry in result["sections"].items()} == {
        "lz": "failed", "egress": "failed", "medical": "failed"}
    assert result["status"] == "partial"


def test_a_section_whose_feature_raised_keeps_its_detail_call(ctx, monkeypatch):
    """To retry it, or make it directly, the caller needs the exact call
    that raised - overrides included."""
    Stubs(monkeypatch, fail={"lz", "medical"})
    result = plan_medevac(ctx, *PATIENT, overrides={"lz": {"max_slope_deg": 6.0},
                                                    "medical": {"search_radius_km": 150.0}})
    lz = result["sections"]["lz"]
    assert lz["status"] == "failed" and lz["error"].startswith("ValueError: no canopy panel")
    assert lz["detail_call"] == {"tool": "lz_candidates", "arguments": {
        "handle": result["handles"]["lz"], "center_latitude": PATIENT[0],
        "center_longitude": PATIENT[1], "limit": 5, "whole_ground": True,
        "max_slope_deg": 6.0}}
    medical = result["sections"]["medical"]
    assert medical["error"] == "ConnectError: HIFLD could not be reached"
    assert medical["detail_call"] == {"tool": "nearest_medical_facilities", "arguments": {
        "latitude": PATIENT[0], "longitude": PATIENT[1], "per_category": 3,
        "search_radius_km": 150.0}}


def test_an_exit_at_the_patient_is_reported_but_not_drawn(ctx, monkeypatch):
    """A patient already on the exit has a one-cell path, which the feature
    returns as a Point route: a fact for the section, nothing to draw."""
    Stubs(monkeypatch)

    def on_the_road(ctx, **arguments):
        payload = egress_payload(arguments)
        here = {"type": "Point", "coordinates": [PATIENT[1], PATIENT[0]]}
        payload["exits"][0].update(route=here, exit_point=here, minutes=0.0,
                                   distance_m=0.0, gain_m=0.0, loss_m=0.0)
        return payload

    monkeypatch.setattr(egress_feature, "egress_routes", on_the_road)
    result = plan_medevac(ctx, *PATIENT, include=["egress"], egress_modes=["walk_out"])
    exit_ = result["sections"]["egress"]["modes"]["walk_out"]["exits"][0]
    assert exit_["reachable"] is True and exit_["distance_m"] == 0.0
    assert result["summary"]["egress"]["walk_out"] == {
        "kind": "2wd", "minutes": 0.0, "distance_m": 0.0, "off_trail_m": 200.0,
        "road_name": "Skyline Blvd",
        "road_access": None, "gate_on_road": False, "parking_lot": None}
    assert result["exports"] == []
    assert result["warnings"] == ["nothing was found to draw, so no CalTopo export was "
                                  "written"]


def test_an_exit_at_the_patient_says_its_vehicle_end_on_the_patient_s_marker(ctx,
                                                                             monkeypatch):
    """No line is drawn for an exit at the patient, so the patient's marker
    says what that line would have: a gated private road under the patient
    must not read as open on the map."""
    Stubs(monkeypatch)

    def on_a_gated_road(ctx, **arguments):
        payload = egress_payload(arguments)
        here = {"type": "Point", "coordinates": [PATIENT[1], PATIENT[0]]}
        payload["exits"][0].update(route=here, exit_point=here, minutes=0.0,
                                   distance_m=0.0, gain_m=0.0, loss_m=0.0, gates_crossed=[])
        payload["exits"][0]["road"].update(name="Old Haul Rd", access="private",
                                           gate_on_road=True)
        return payload

    monkeypatch.setattr(egress_feature, "egress_routes", on_a_gated_road)
    result = plan_medevac(ctx, *PATIENT, include=["lz", "egress"], egress_modes=["walk_out"])
    patient = exported(result["exports"][0])[0]["properties"]
    assert patient["title"] == "Patient"
    [note] = [row for row in patient["description"].splitlines()
              if row.startswith("Walk-out 2WD exit is here: ")]
    assert "Exit on Old Haul Rd." in note
    assert "Access on Old Haul Rd: private, as OpenStreetMap tags it." in note
    assert "Gate mapped somewhere on the stretch of Old Haul Rd the exit lies on" in note


def test_an_empty_egress_modes_is_fine_when_egress_is_left_out(ctx, monkeypatch):
    """"No modes" and "no egress" say the same thing: refusing it, and
    telling the caller to leave egress out, would ask for what they did."""
    Stubs(monkeypatch)
    result = plan_medevac(ctx, *PATIENT, include=["lz"], egress_modes=[])
    assert result["status"] == "ok"
    assert result["sections"]["egress"]["status"] == "skipped"


def test_a_mode_with_no_reachable_exit_says_so_in_the_summary(ctx, monkeypatch):
    Stubs(monkeypatch)
    real = egress_feature.egress_routes

    def unreachable(ctx, **arguments):
        payload = real(ctx, **arguments)
        payload["exits"] = [{"kind": kind, "reachable": False, "reason": "none mapped"}
                            for kind in ("2wd", "4x4")]
        return payload

    monkeypatch.setattr(egress_feature, "egress_routes", unreachable)
    result = plan_medevac(ctx, *PATIENT, include=["egress"], egress_modes=["walk_out"])
    assert result["summary"]["egress"] == {"walk_out": "none reachable"}


def _exit_in_a_lot(ctx, **arguments):
    """Current ground, gates known, the 2WD exit in a lot with no road beside
    it: `road` null, the lot naming the road it opens onto."""
    payload = egress_payload(arguments)
    payload["exits"][0].update(road=None, gates_crossed=[],
                               parking_lot={"id": "parking-w840", "off": "Skyline Boulevard"})
    payload["warnings"] = []
    return payload


def test_an_exit_in_a_lot_has_null_gate_on_road_with_no_gates_unknown_warning(ctx,
                                                                              monkeypatch):
    """On current ground a null `gate_on_road` also means "no road at the
    exit", not only "ground too old to say": the lot and its road are there,
    and no warning says gates are unknown."""
    Stubs(monkeypatch)
    monkeypatch.setattr(egress_feature, "egress_routes", _exit_in_a_lot)
    result = plan_medevac(ctx, *PATIENT, include=["egress"], egress_modes=["walk_out"])
    egress = result["sections"]["egress"]
    exit_ = egress["modes"]["walk_out"]["exits"][0]
    assert (exit_["road_name"], exit_["road_access"], exit_["gate_on_road"]) == (
        None, None, None)
    assert exit_["parking_lot"] == {"id": "parking-w840", "off": "Skyline Boulevard"}
    assert exit_["gates_crossed"] == 0
    assert egress["warnings"] == [] and result["warnings"] == []


def test_an_export_that_cannot_be_written_is_a_warning(ctx, monkeypatch):
    Stubs(monkeypatch)

    def full_disk(path, text):
        raise OSError("No space left on device")

    monkeypatch.setattr(_sections, "atomic_write_text", full_disk)
    result = plan_medevac(ctx, *PATIENT)
    assert result["status"] == "ok"
    assert result["exports"] == []
    assert result["warnings"] == ["the CalTopo export could not be written (OSError: No "
                                  "space left on device); every section's answer stands"]


def test_a_later_call_writing_no_file_names_the_earlier_one(ctx, monkeypatch):
    """The file's name is fixed: a later call on the same ground that writes
    none - here a full disk - leaves the earlier call's as it was, where it
    would pass for this answer's. The answer names it as an earlier call's;
    it is not removed, since the earlier answer gave that path."""
    Stubs(monkeypatch)
    first = plan_medevac(ctx, *PATIENT)
    (earlier,) = first["exports"]

    def full_disk(path, text):
        raise OSError("No space left on device")

    monkeypatch.setattr(_sections, "atomic_write_text", full_disk)
    second = plan_medevac(ctx, *PATIENT)
    assert second["handles"]["lz"] == first["handles"]["lz"]   # the same ground
    assert second["exports"] == [] and Path(earlier).is_file()
    assert second["warnings"] == [
        "the CalTopo export could not be written (OSError: No space left on device); every "
        "section's answer stands",
        f"the CalTopo export at {earlier} is an earlier call's, left as it was: this call "
        "wrote none, so it is not this answer's, and exports does not list it"]


def test_a_later_call_writing_its_file_names_no_earlier_one(ctx, monkeypatch):
    """The other side of the test above: a call that wrote its file names
    none as an earlier call's left in its place - that would call this
    answer's own file not its own."""
    Stubs(monkeypatch)
    first = plan_medevac(ctx, *PATIENT)
    second = plan_medevac(ctx, *PATIENT)
    assert second["handles"]["lz"] == first["handles"]["lz"]   # the same ground
    assert second["exports"]
    assert not any("left as it was" in warning for warning in second["warnings"])


# --- What the caller chooses --------------------------------------------------


def test_include_skips_sections_and_stages_nothing_for_them(ctx, monkeypatch):
    stubs = Stubs(monkeypatch)
    result = plan_medevac(ctx, *PATIENT, include=["medical"],
                          overrides={"lz": {"max_slope_deg": 6.0}})
    assert result["status"] == "ok"
    for name in ("lz", "egress"):
        assert result["sections"][name] == {"status": "skipped", "error": None,
                                             "warnings": []}
    assert stubs.staging.calls == [] and stubs.calls["lz"] == []
    assert result["staging"] == {} and result["handles"] == {}
    assert set(result["summary"]) == {"medical"}
    assert result["exports"] == []
    assert result["warnings"] == [
        "overrides['lz'] was not used: lz was left out of include",
        "no ground was staged, so there is nowhere beside it to write the CalTopo export; "
        "each section's detail_call returns what it found"]


def test_overrides_reach_every_call_of_their_section_and_its_detail_call(ctx, monkeypatch):
    stubs = Stubs(monkeypatch)
    result = plan_medevac(ctx, *PATIENT, lz_limit=2, medical_per_category=1,
                          egress_modes=["walk_out"],
                          overrides={"lz": {"max_slope_deg": 6.0},
                                     "egress": {"max_grade": 0.3},
                                     "medical": {"search_radius_km": 150.0}})
    assert stubs.calls["lz"][0]["max_slope_deg"] == 6.0 and stubs.calls["lz"][0]["limit"] == 2
    assert stubs.calls["egress"] == [{"handle": result["handles"]["egress"],
                                      "center_latitude": PATIENT[0],
                                      "center_longitude": PATIENT[1], "mode": "walk_out",
                                      "max_grade": 0.3}]
    assert stubs.calls["medical"] == [{"latitude": PATIENT[0], "longitude": PATIENT[1],
                                       "per_category": 1, "search_radius_km": 150.0}]
    sections = result["sections"]
    assert sections["lz"]["detail_call"]["arguments"]["max_slope_deg"] == 6.0
    assert sections["egress"]["modes"]["walk_out"]["detail_call"]["arguments"][
        "max_grade"] == 0.3
    assert sections["medical"]["detail_call"]["arguments"]["search_radius_km"] == 150.0


@pytest.mark.parametrize("overrides, message", [
    ({"parking": {"limit": 3}}, r"no section of plan_medevac: \['parking'\]"),
    ({"lz": {"max_slop_deg": 6}}, r"\['max_slop_deg'\], which lz_candidates does not take"),
    ({"egress": {"handle": "aoi-0123456789abcdef"}}, "cannot set 'handle'"),
    ({"egress": {"center_latitude": 1.0}}, "cannot set 'center_latitude'"),
    ({"lz": {"limit": 9}}, "cannot set 'limit': plan_medevac sets it from its own "
                           "argument lz_limit"),
    ({"egress": {"mode": "walk_out"}}, "argument egress_modes"),
    ({"medical": {"per_category": 5}}, "argument medical_per_category"),
    ({"medical": "fast"}, "must be a mapping"),
    ({"medical": []}, "must be a mapping"),
    ({"egress": {"max_grade": 0.3}}, r"\['max_grade'\], whose defaults differ by mode"),
])
def test_bad_overrides_are_refused_before_any_work(ctx, monkeypatch, overrides, message):
    stubs = Stubs(monkeypatch)
    with pytest.raises(ValueError, match=message):
        plan_medevac(ctx, *PATIENT, overrides=overrides)
    assert stubs.nothing_called()


@pytest.mark.parametrize("arguments, message", [
    ({"latitude": 95.0}, "latitude"),
    ({"longitude": float("nan")}, "longitude"),
    ({"include": ["lzz"]}, "no section of this scenario"),
    ({"include": []}, "empty"),
    ({"lz_radius_m": 5000.0}, "lz_radius_m must be above 0 and under 4,500 m"),
    ({"lz_radius_m": 4500.0}, "lz_radius_m must be above 0 and under 4,500 m"),
    ({"egress_radius_m": 4500}, "egress_radius_m must be above 0 and under 4,500 m"),
    ({"egress_radius_m": 0}, "egress_radius_m"),
    ({"egress_modes": ["drive_out"]}, r"egress_modes names \['drive_out'\]"),
    ({"egress_modes": []}, "egress_modes is empty"),
    ({"egress_modes": "walk_out"}, "must be a list"),
    ({"lz_limit": -1}, "lz_limit"),
    ({"medical_per_category": 0}, "medical_per_category"),
    ({"medical_per_category": 11}, "medical_per_category"),
])
def test_bad_arguments_are_refused_before_any_work(ctx, monkeypatch, arguments, message):
    stubs = Stubs(monkeypatch)
    call = {"latitude": PATIENT[0], "longitude": PATIENT[1], **arguments}
    with pytest.raises(ValueError, match=message):
        plan_medevac(ctx, **call)
    assert stubs.nothing_called()


#: Every number plan_medevac takes, each with `True`: Python counts a bool an
#: int, but True is nobody's latitude, radius or limit.
BOOL_ARGUMENTS = ["latitude", "longitude", "lz_radius_m", "egress_radius_m", "lz_limit",
                  "medical_per_category"]


@pytest.mark.parametrize("name", BOOL_ARGUMENTS)
def test_a_bool_for_a_number_is_refused_before_any_work(ctx, monkeypatch, name):
    stubs = Stubs(monkeypatch)
    call = {"latitude": PATIENT[0], "longitude": PATIENT[1], name: True}
    with pytest.raises(ValueError, match=rf"^{name} .*got True"):
        plan_medevac(ctx, **call)
    assert stubs.nothing_called()


# --- The export -----------------------------------------------------------------


def test_the_export_is_a_file_caltopo_imports(ctx, monkeypatch):
    Stubs(monkeypatch)
    result = plan_medevac(ctx, *PATIENT)
    collection = read_export(result["exports"][0])
    assert collection["type"] == "FeatureCollection"
    features = collection["features"]
    for feature in features:
        assert feature["type"] == "Feature"
        assert feature["geometry"]["type"] in ("Point", "LineString")
        assert feature["properties"]["class"] in ("Marker", "Shape")
        assert feature["properties"]["title"] and feature["properties"]["description"]

    patient = features[0]
    assert patient["properties"]["title"] == "Patient"
    assert patient["geometry"] == {"type": "Point", "coordinates": [PATIENT[1], PATIENT[0]]}
    assert patient["properties"]["marker-symbol"] == "point"
    # The medical tool's own point line, and no road-start note: the road is 10 m off.
    assert "Nearest hospitals ranked by road time from here." in (
        patient["properties"]["description"])
    assert "Road route starts" not in patient["properties"]["description"]

    zones = [f for f in features if f["properties"]["title"].startswith("lz_")]
    assert [f["properties"]["title"] for f in zones] == [
        "lz_1 (minimal)", "lz_2 (preferred)", "lz_3 (marginal)"]
    assert [f["properties"]["marker-color"] for f in zones] == [
        "#FFD700", "#00C853", "#FF8000"]
    assert "ON Skyline Blvd" in zones[0]["properties"]["description"]
    assert "The pilot decides." in zones[0]["properties"]["description"]
    assert "180 m at 90 deg from the patient." in zones[0]["properties"]["description"]
    # What lz_candidates did not check reaches the map, not only the JSON.
    assert all("Not checked here: trees." in zone["properties"]["description"]
               for zone in zones)
    # "unknown" is said, never read as "none near".
    assert "Power lines unknown here." in zones[2]["properties"]["description"]
    assert "Roads unknown here." in zones[2]["properties"]["description"]

    routes = [f for f in features if f["geometry"]["type"] == "LineString"]
    assert [f["properties"]["title"] for f in routes] == ["Carry-out 2WD 95 min",
                                                          "Walk-out 2WD 31 min"]
    assert all(f["properties"]["class"] == "Shape" and f["properties"]["stroke"]
               for f in routes)

    hospitals = [f for f in features if f["properties"].get("role") == "hospital"]
    assert {f["properties"]["title"] for f in hospitals} == {"Stanford Hospital",
                                                             "Sequoia Hospital"}
    assert all(f["properties"]["marker-symbol"] == HOSPITAL_SYMBOL
               and f["properties"]["marker-color"] == HOSPITAL_COLOR for f in hospitals)
    # One point for the patient: the medical tool's own origin dot is not repeated.
    assert len(features) == 1 + 3 + 2 + 2


def test_the_patient_marker_says_where_hospital_road_times_start(ctx, monkeypatch):
    """Over 200 m from a road the router knows, the medical tool's own point
    says the road time starts there. The scenario draws the patient in its
    place, so that line must move onto the patient, or a team reading the
    map takes every road minute as starting at the patient."""
    stubs = Stubs(monkeypatch)
    stubs.services.origin_snap = 640.0
    result = plan_medevac(ctx, *PATIENT)
    collection = read_export(result["exports"][0])
    text = collection["features"][0]["properties"]["description"]
    assert "Road route starts 640 m away; the walk or carry to it is not counted." in text
    assert "every distance, bearing and minute" not in text
    assert result["sections"]["medical"]["road_start"]["distance_from_point_m"] == 640


def test_each_exit_line_reads_its_minutes_road_and_gates_in_its_mode_s_colour(ctx,
                                                                             monkeypatch):
    """The map is what the field team reads: a carry-out drawn in the
    walk-out's blue, or a gate dropped from the line, would send a litter
    team the wrong way. Purple carry-out, blue walk-out, as documented."""
    Stubs(monkeypatch)
    result = plan_medevac(ctx, *PATIENT)
    routes = {f["properties"]["title"]: f for f in exported(result["exports"][0])
              if f["geometry"]["type"] == "LineString"}
    assert _sections.EGRESS_STROKE == {"carry_out": "#8000FF", "walk_out": "#0080FF"}
    disclaimer = ("Walked over the elevation model and OpenStreetMap, not a surveyed route; "
                  "read the minutes as a rank. The team leader decides.")
    for title, mode, label, minutes in (("Carry-out 2WD 95 min", "carry_out", "Carry-out", 95),
                                        ("Walk-out 2WD 31 min", "walk_out", "Walk-out", 31)):
        properties = routes[title]["properties"]
        assert properties["stroke"] == _sections.EGRESS_STROKE[mode]
        assert (properties["stroke-width"], properties["stroke-opacity"]) == (3, 0.9)
        assert properties["description"].splitlines() == [
            f"{label} to a 2WD exit: {minutes} min, 1,840 m, +60 m / -120 m.",
            "Exit on Skyline Blvd.",
            # Untagged access and the road's own gate fact are said, never
            # left to silence - and "no gate" only of the stretch looked at.
            "Access on Skyline Blvd: not tagged in the OpenStreetMap read for this plan, "
            "which does not make it public.",
            "1 gate on the way.",
            "No gate mapped on the stretch of Skyline Blvd the exit lies on.",
            "The road in beyond that stretch, and beyond the ground read for this plan, is "
            "not checked for gates.",
            # Any ground off a way is said on the line, however little.
            "200 m of this route is off any mapped trail, track or road.",
            disclaimer]
        assert routes[title]["geometry"] == {"type": "LineString", "coordinates": [
            [PATIENT[1], PATIENT[0]], [PATIENT[1], PATIENT[0] + 0.012]]}


def test_an_exit_line_counts_several_gates_and_carries_the_exit_s_warnings(ctx, monkeypatch):
    Stubs(monkeypatch)

    def gated(ctx, **arguments):
        payload = egress_payload(arguments)
        gate = payload["exits"][0]["gates_crossed"][0]
        payload["exits"][0].update(road=None, gates_crossed=[gate, gate],
                                   warnings=["the last 200 m is off any mapped way"])
        return payload

    monkeypatch.setattr(egress_feature, "egress_routes", gated)
    result = plan_medevac(ctx, *PATIENT, include=["egress"], egress_modes=["carry_out"])
    [line] = [f for f in exported(result["exports"][0])
              if f["geometry"]["type"] == "LineString"]
    text = line["properties"]["description"].splitlines()
    assert line["properties"]["stroke"] == "#8000FF"
    assert "2 gates on the way." in text
    assert "the last 200 m is off any mapped way" in text
    assert not any(row.startswith("Exit on") for row in text)    # no road to name
    assert text[-1].endswith("The team leader decides.")


def test_the_patient_and_landing_zones_are_drawn_in_their_documented_styles(ctx,
                                                                            monkeypatch):
    """The patient as the medical tool draws its point (a blue dot), every
    landing zone as a "point" coloured by grade."""
    Stubs(monkeypatch)
    result = plan_medevac(ctx, *PATIENT)
    features = exported(result["exports"][0])
    patient = features[0]["properties"]
    assert patient["title"] == "Patient"
    assert (patient["marker-symbol"], patient["marker-color"]) == (ORIGIN_SYMBOL, ORIGIN_COLOR)
    zones = [f["properties"] for f in features if f["properties"]["title"].startswith("lz_")]
    assert len(zones) == 3
    assert all(zone["marker-symbol"] == _sections.LZ_SYMBOL == "point" for zone in zones)
    assert all(zone["class"] == "Marker" for zone in zones)


def test_with_the_router_down_the_patient_marker_does_not_claim_road_time_ranking(
        ctx, monkeypatch):
    stubs = Stubs(monkeypatch)
    stubs.services.route_error = httpx.ConnectError("OSRM unreachable")
    result = plan_medevac(ctx, *PATIENT)
    trauma = result["summary"]["medical"]["nearest_trauma_center"]
    assert trauma["road_minutes"] is None
    text = exported(result["exports"][0])[0]["properties"]["description"]
    assert "ranked by road time" not in text
    # The degradation is said where an operator reads it: the map's point,
    # a top-level warning, or the summary block itself.
    told = " ".join([text, *result["warnings"], json.dumps(result["summary"]["medical"])])
    assert "straight-line" in told or "straight_line" in told


def test_with_no_hospital_drawn_the_patient_marker_carries_no_ranking_line(ctx,
                                                                         monkeypatch):
    """Outside the US nothing is found and no hospital is drawn: a line about
    how hospitals are ranked describes nothing on the map."""
    stubs = Stubs(monkeypatch)
    stubs.services.state, stubs.services.records = None, []
    result = plan_medevac(ctx, *PATIENT)
    features = exported(result["exports"][0])
    assert not [f for f in features if f["properties"].get("role") == "hospital"]
    assert "hospitals ranked" not in features[0]["properties"]["description"].lower()


@pytest.mark.xfail(strict=True, reason="PR #4 review B8: A later scenario call on the same "
                   "ground silently overwrites the export file (and drafts) an earlier answer "
                   "pointed to")
def test_a_later_call_on_the_same_ground_leaves_the_earlier_answer_s_file(ctx, monkeypatch):
    """The first answer's `exports` is what a coordinator imports, perhaps
    later: a second, narrower call on the same patient must not empty it of
    the routes and hospitals the first answer described."""
    Stubs(monkeypatch)
    first = plan_medevac(ctx, *PATIENT)
    titles = [f["properties"]["title"] for f in exported(first["exports"][0])]
    assert len(titles) == 8                  # patient, 3 zones, 2 routes, 2 hospitals
    second = plan_medevac(ctx, *PATIENT, include=["lz"], lz_limit=1)
    assert second["handles"]["lz"] == first["handles"]["lz"]   # the same ground
    assert second["exports"]
    assert [f["properties"]["title"] for f in exported(first["exports"][0])] == titles


@pytest.mark.parametrize("config", [
    lambda: CaltopoConfig(cache_dir=Path("cache")),
    CaltopoConfig.from_env,
], ids=["relative-cache-dir", "default-config-from-env"])
def test_every_export_path_is_absolute_wherever_the_server_runs(tmp_path, monkeypatch,
                                                                config):
    """The operator imports the path the answer gives; an MCP client starts
    the server in a directory the operator never sees, so a path relative to
    it leads nowhere."""
    (tmp_path / "server").mkdir()
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "server")
    # The default config, whatever the shell running the tests exports: no
    # cache dir, norms file or credential of the developer's own.
    for name in [name for name in os.environ if name.startswith("ZA_")]:
        monkeypatch.delenv(name)
    Stubs(monkeypatch)
    result = plan_medevac(ServerContext.build(config()), *PATIENT)
    assert result["exports"]
    assert all(os.path.isabs(path) for path in result["exports"])
    monkeypatch.chdir(tmp_path / "elsewhere")
    assert Path(result["exports"][0]).is_file()


# --- Through MCP ------------------------------------------------------------------


@pytest.mark.anyio
async def test_plan_medevac_is_registered_as_a_scenario_and_answers(monkeypatch, tmp_path):
    pytest.importorskip("fastmcp")
    from fastmcp import Client
    from fastmcp.exceptions import ToolError

    from sar_geo_mcp.server import build_server

    Stubs(monkeypatch)
    server = build_server(CaltopoConfig(cache_dir=tmp_path / "cache"))
    async with Client(server) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
        tool = tools["plan_medevac"]
        assert tool.meta["fastmcp"]["tags"] == ["scenario"]
        assert "NO VERDICT" in tool.description and "detail_call" in tool.description
        include = tool.inputSchema["properties"]["include"]["anyOf"][0]["items"]
        assert include["enum"] == ["lz", "egress", "medical"]

        answer = (await client.call_tool("plan_medevac", {
            "latitude": PATIENT[0], "longitude": PATIENT[1],
            "include": ["lz", "medical"]})).data
        assert answer["status"] == "ok"
        assert answer["sections"]["egress"]["status"] == "skipped"
        assert answer["summary"]["lz"]["nearest_preferred"]["id"] == "lz_2"

        with pytest.raises(ToolError):
            await client.call_tool("plan_medevac", {"latitude": PATIENT[0],
                                                    "longitude": PATIENT[1],
                                                    "include": ["parking"]})


@pytest.mark.anyio
@pytest.mark.parametrize("arguments, message", [
    ({"include": []}, "include is empty, so there would be nothing to do: leave it out for "
                      "every section, or name some of ['lz', 'egress', 'medical']"),
    ({"overrides": {"lz": {"limit": 3}}},
     "overrides['lz'] cannot set 'limit': plan_medevac sets it from its own argument "
     "lz_limit; pass that instead"),
    ({"lz_radius_m": 4500}, "lz_radius_m must be above 0 and under 4,500 m (prepare_area "
                            "stages at most 9,000 m across); got 4500.0"),
    ({"overrides": {"egress": {"max_grade": 0.3}}},
     "overrides['egress'] sets ['max_grade'], whose defaults differ by mode"),
], ids=["empty-include", "override-sets-limit", "radius-too-wide", "egress-override-two-modes"])
async def test_a_scenario_refusal_reaches_the_mcp_client_in_its_own_words(
        monkeypatch, tmp_path, arguments, message):
    """The refusals are written for the operator: through MCP they must
    arrive as written - not masked, not wrapped into something else - and
    still before any work."""
    pytest.importorskip("fastmcp")
    from fastmcp import Client
    from fastmcp.exceptions import ToolError

    from sar_geo_mcp.server import build_server

    stubs = Stubs(monkeypatch)
    server = build_server(CaltopoConfig(cache_dir=tmp_path / "cache"))
    async with Client(server) as client:
        with pytest.raises(ToolError, match=re.escape(message)):
            await client.call_tool("plan_medevac", {"latitude": PATIENT[0],
                                                    "longitude": PATIENT[1], **arguments})
    assert stubs.nothing_called()


@pytest.mark.anyio
@pytest.mark.xfail(strict=True, reason="PR #4 review B44: MCP wrappers turn JSON true into "
                   "1 / 1.0, defeating the scenarios' explicit bool refusals")
@pytest.mark.parametrize("name", BOOL_ARGUMENTS)
async def test_a_json_true_for_a_number_is_refused_through_mcp_too(monkeypatch, tmp_path,
                                                                   name):
    """`test_a_bool_for_a_number_is_refused_before_any_work`, as an agent
    calls the tool: `true` must not become a 1 m radius, 1 deg N or a limit
    of one zone."""
    pytest.importorskip("fastmcp")
    from fastmcp import Client

    from sar_geo_mcp.server import build_server

    stubs = Stubs(monkeypatch)
    server = build_server(CaltopoConfig(cache_dir=tmp_path / "cache"))
    async with Client(server) as client:
        result = await client.call_tool(
            "plan_medevac", {"latitude": PATIENT[0], "longitude": PATIENT[1], name: True},
            raise_on_error=False)
    assert result.is_error
    assert name in result.content[0].text
    assert stubs.nothing_called()


@pytest.mark.anyio
@pytest.mark.xfail(strict=True, reason="PR #4 review B40: egress gate_on_road null also means "
                   "'no road at the exit', but the docs define null as 'ground too old to "
                   "say'")
async def test_the_description_says_gate_on_road_is_null_for_an_exit_with_no_road(
        monkeypatch, tmp_path):
    """`test_an_exit_in_a_lot_has_null_gate_on_road_with_no_gates_unknown_warning`
    gives null on current ground; the description must not let a planner read
    that null as "unknown because the ground is old"."""
    pytest.importorskip("fastmcp")
    from fastmcp import Client

    from sar_geo_mcp.server import build_server

    server = build_server(CaltopoConfig(cache_dir=tmp_path / "cache"))
    async with Client(server) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
    description = " ".join(tools["plan_medevac"].description.split())
    start = description.index("`gate_on_road`")
    entry = description[start:description.index("`parking_lot`", start)]
    assert "too old to say" in entry
    assert re.search(r"\bno road\b|\blot\b", entry), entry


# --- Against the real features ----------------------------------------------------


def test_the_sections_read_what_the_real_features_return(ctx, monkeypatch):
    """Flat, open ground written to disk (`_stub_staging`) with a road 300 m
    north, and the real lz_candidates and egress_routes over it - no token,
    so no imagery is fetched - and the real medical logic over faked
    services. Holds the compact sections to the features' actual payloads,
    which the stubs above only imitate."""
    from test_planning_tools import _stub_staging, _trail_through

    road = _trail_through(PATIENT, offset_m=300, rid="road_1")
    road["properties"].update(route_type="ROAD", name="Skyline Blvd")
    # 100 m round the patient plus the 1,000 m buffer each side is 2,200 m
    # across; 1,000 px is 2.2 m a pixel, fine enough to grade a landing zone.
    _stub_staging(monkeypatch, routes=[road], width=1000, vegetation_panel=True)
    services = FakeServices([record("Stanford Hospital", *STANFORD, trauma="LEVEL I",
                                    helipad="Y")])
    monkeypatch.setattr(medical_feature, "nearest_medical_facilities",
                        lambda ctx, latitude, longitude, **rest: medical_impl(
                            latitude, longitude, **rest, query=services.query,
                            route=services.route, helipads=services.helipads,
                            locate=services.locate))

    # The real payloads, kept as the features returned them, so the compact
    # sections are held to the keys the features actually give - a section
    # built from `.get(key)` would quietly carry None for a renamed one.
    real = {}
    real_lz, real_egress = lz_feature.lz_candidates, egress_feature.egress_routes

    def keep_lz(ctx, **arguments):
        real["lz"] = copy.deepcopy(real_lz(ctx, **arguments))
        return real["lz"]

    def keep_egress(ctx, **arguments):
        payload = real_egress(ctx, **arguments)
        real.setdefault("egress", []).append(copy.deepcopy(payload))
        return payload

    monkeypatch.setattr(lz_feature, "lz_candidates", keep_lz)
    monkeypatch.setattr(egress_feature, "egress_routes", keep_egress)

    result = plan_medevac(ctx, *PATIENT, lz_radius_m=100, egress_radius_m=100)

    assert result["status"] == "ok", {name: entry["error"]
                                      for name, entry in result["sections"].items()}
    lz = result["sections"]["lz"]
    assert lz["candidates"] and lz["candidates"][0]["grade"] == "preferred"
    assert set(_sections.LZ_FIELDS) <= set(real["lz"]["candidates"][0])
    assert set(_sections.LZ_CHECKS) <= set(real["lz"]["assumptions"])
    assert {"grades", "total_before_limit", "truncated_to"} <= set(real["lz"]["summary"])
    exits = [exit_ for payload in real["egress"] for exit_ in payload["exits"]]
    reachable = [exit_ for exit_ in exits if exit_["reachable"]]
    assert reachable
    for exit_ in reachable:
        assert {"kind", "minutes", "distance_m", "gain_m", "loss_m", "exit_point", "route",
                "road", "parking_lot", "gates_crossed", "warnings"} <= set(exit_)
        assert {"name", "access", "gate_on_road"} <= set(exit_["road"])
    for exit_ in exits:
        if not exit_["reachable"]:
            assert {"kind", "reason"} <= set(exit_)
    assert lz["checks"]["structures"] == "not checked"
    for mode in ("carry_out", "walk_out"):
        exits = result["sections"]["egress"]["modes"][mode]["exits"]
        assert [exit_["kind"] for exit_ in exits] == ["2wd", "4x4"]
        assert exits[0]["reachable"] is True and exits[0]["road_name"] == "Skyline Blvd"
        assert exits[0]["exit_point"]["type"] == "Point"
        assert result["summary"]["egress"][mode]["kind"] == "2wd"
    assert result["summary"]["medical"]["nearest_trauma_center"]["name"] == "Stanford Hospital"
    assert geometry_types_in(result) == {"Point"}

    collection = read_export(result["exports"][0])
    kinds = [feature["geometry"]["type"] for feature in collection["features"]]
    assert kinds.count("LineString") == 2                 # a route per mode


# --- A trail that runs off the exit ground -------------------------------------


RECENTRE = {"latitude": PATIENT[0], "longitude": PATIENT[1] + 0.03}


def _beyond():
    return {"point": {"type": "Point", "coordinates": [PATIENT[1] + 0.045, PATIENT[0]]},
            "way_name": "Lake Trail", "minutes": 70.0, "distance_m": 4000.0,
            "bearing_deg": 90.0, "recentre": dict(RECENTRE),
            "problems": ["the 2WD exit leaves the trail for 900 m", "no 4x4 exit is reached"]}


def _trail_runs_off(handles, *, second_off_m=0.0):
    """egress_routes whose first ground's trail runs off it (its 2WD exit
    leaves the trail for 900 m), and whose re-centred ground - any other
    handle - reaches a road on the trail, 145 minutes out."""
    def egress(ctx, **arguments):
        handles.append(arguments["handle"])
        payload = egress_payload(arguments)
        two_wd = payload["exits"][0]
        if arguments["handle"] == handles[0]:
            two_wd["off_trail_after_joining_m"] = 900.0
            payload["trail_beyond"] = _beyond()
            payload["warnings"].append("Lake Trail leaves the staged ground ...")
        else:
            two_wd["off_trail_after_joining_m"] = second_off_m
            two_wd["minutes"] = 145.0
            payload["trail_beyond"] = None
        return payload
    return egress


def test_a_trail_off_the_ground_is_followed_on_ground_re_centred_along_it(ctx, monkeypatch):
    # Found in a live run: the trail ran off the 3,000 m ground and the
    # exits cut 1 km cross-country to a track; the trailhead lay past the edge.
    stubs = Stubs(monkeypatch)
    handles = []
    monkeypatch.setattr(egress_feature, "egress_routes", _trail_runs_off(handles))
    result = plan_medevac(ctx, *PATIENT, include=["egress"])
    # Staged once more, at the same radius and resolution, where egress said.
    again = [call for call in stubs.staging.calls
             if call["center_longitude"] == RECENTRE["longitude"]]
    assert len(again) == 1
    assert again[0]["radius_m"] == 3000.0 and again[0]["resolution_m"] == 10.0
    recentred = result["staging"]["egress_recentred"]
    assert recentred["status"] == "ok"
    assert result["handles"]["egress_recentred"] == recentred["handle"]
    for mode in ("carry_out", "walk_out"):
        entry = result["sections"]["egress"]["modes"][mode]
        assert entry["recentred"]["handle"] == recentred["handle"]
        assert entry["recentred"]["center"] == RECENTRE
        assert "Lake Trail ran off the first ground" in entry["recentred"]["because"]
        assert entry["detail_call"]["arguments"]["handle"] == recentred["handle"]
        assert entry["exits"][0]["minutes"] == 145.0
        assert entry["trail_beyond"] is None
    assert any(warning.startswith("the exits are from ground re-centred on")
               for warning in result["sections"]["egress"]["warnings"])
    # The first ground's off-the-trail lines are not drawn; the second's are.
    lines = [f for f in exported(result["exports"][0]) if f["geometry"]["type"] == "LineString"]
    assert lines and all("145 min" in line["properties"]["title"] for line in lines)


def test_re_centred_ground_no_better_keeps_the_first_answer(ctx, monkeypatch):
    Stubs(monkeypatch)
    handles = []
    monkeypatch.setattr(egress_feature, "egress_routes",
                        _trail_runs_off(handles, second_off_m=950.0))
    result = plan_medevac(ctx, *PATIENT, include=["egress"])
    walk = result["sections"]["egress"]["modes"]["walk_out"]
    assert walk["recentred"] is None
    assert walk["exits"][0]["minutes"] == 31.0
    assert walk["trail_beyond"]["way_name"] == "Lake Trail"
    assert walk["detail_call"]["arguments"]["handle"] == result["handles"]["egress"]
    assert any("kept no more to the trails" in warning
               for warning in result["sections"]["egress"]["warnings"])


def test_re_centred_ground_that_cannot_be_staged_keeps_the_first_answer(ctx, monkeypatch):
    stubs = Stubs(monkeypatch)
    first = stubs.staging

    def refuse_elsewhere(ctx, **arguments):
        if arguments["center_longitude"] != PATIENT[1]:
            first.calls.append(arguments)
            raise RuntimeError("3DEP could not be reached")
        return first(ctx, **arguments)
    monkeypatch.setattr(staging_component, "prepare_area", refuse_elsewhere)
    monkeypatch.setattr(egress_feature, "egress_routes", _trail_runs_off([]))
    result = plan_medevac(ctx, *PATIENT, include=["egress"])
    egress = result["sections"]["egress"]
    assert egress["status"] == "ok"
    assert egress["modes"]["walk_out"]["recentred"] is None
    assert egress["modes"]["walk_out"]["exits"][0]["minutes"] == 31.0
    assert result["staging"]["egress_recentred"]["status"] == "failed"
    assert any("could not be asked (StagingFailed" in warning for warning in egress["warnings"])
