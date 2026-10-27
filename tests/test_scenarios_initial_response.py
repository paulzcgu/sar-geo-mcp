"""plan_initial_response: hasty lines, parking, walking rings, landing zones,
hospitals and the CalTopo draft, in one call from a place last seen.

No network. Staging and the feature functions are stubbed with monkeypatch
where they live (`components.staging.prepare_area`,
`features.hasty.hasty_candidates`, ...), which is where the scenario reaches
them. The hasty stub saves a real candidate set into the staged directory,
so the export section runs the real `caltopo_export` over it and writes a
real file; the medical stub runs the real medical logic over faked HIFLD,
OSRM and Overpass answers. One test at the end runs the real hasty, parking
and rings features and the real staging path over flat ground written to
disk, to hold the compact sections to the payloads the features actually
return. The shared machinery has its own tests in
`test_scenarios_sections.py`; plan_medevac's are in
`test_scenarios_medevac.py`.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import httpx
import pytest
from test_medical import FakeServices, record
from test_scenarios_medevac import lz_payload
from test_scenarios_sections import FakeStaging, geometry_types_in

from sar_geo_mcp.components import candidates as candidates_component
from sar_geo_mcp.components import staging as staging_component
from sar_geo_mcp.config import CaltopoConfig
from sar_geo_mcp.context import ServerContext
from sar_geo_mcp.features import hasty as hasty_feature
from sar_geo_mcp.features import lz as lz_feature
from sar_geo_mcp.features import medical as medical_feature
from sar_geo_mcp.features import parking as parking_feature
from sar_geo_mcp.features import travel_rings as rings_feature
from sar_geo_mcp.planning.candidates import Candidate, CandidateSet, save_set, summaries
from sar_geo_mcp.planning.medical import (HOSPITAL_COLOR, ORIGIN_COLOR, RANKED_BY_ROAD,
                                         RANKED_STRAIGHT_ROUTER_DOWN)
from sar_geo_mcp.planning.medical import nearest_medical_facilities as medical_impl
from sar_geo_mcp.scenarios import _sections
from sar_geo_mcp.scenarios.initial_response import plan_initial_response
from exports import read_export

PLS = (37.4022, -122.3164)
STANFORD = (37.434, -122.175)
SEQUOIA = (37.48, -122.24)
SET_ID = "hasty-a1b2c3"
SECTIONS = ("hasty", "parking", "travel_rings", "lz", "medical", "export")


def _square(lon, lat, half):
    return [[lon - half, lat - half], [lon + half, lat - half], [lon + half, lat + half],
            [lon - half, lat + half], [lon - half, lat - half]]


class AreaStaging(FakeStaging):
    """FakeStaging, plus the `subject.json` an area handle has, so the real
    `caltopo_export` can draw the search area round the assignments."""

    def __call__(self, ctx, **arguments):
        payload = super().__call__(ctx, **arguments)
        lat, lon = arguments["center_latitude"], arguments["center_longitude"]
        radius = arguments["radius_m"]
        half = radius / 111_320
        subject = {"kind": "circle", "center": [lat, lon], "radius_m": radius,
                   "geometry": {"type": "Polygon", "coordinates": [_square(lon, lat, half)]},
                   "key": payload["handle"], "extent_m": 2 * radius}
        directory = Path(ctx.config.cache_dir) / "staged" / payload["handle"]
        (directory / "subject.json").write_text(json.dumps(subject), encoding="utf-8")
        return payload


def hasty_set(handle: str, limit: int) -> CandidateSet:
    """Three candidates the way hasty_candidates measures them: a trail, a
    forked road clipped at the edge with a norm flag, and a waterway
    corridor - each with its geometry, which the section must leave out."""
    lat, lon = PLS

    def facts(distance, minutes, bearing, **extra):
        return {"length_m": 2400.0, "distance_from_center_m": distance,
                "travel_minutes_from_center": minutes, "bearing_from_center_deg": bearing,
                "nearest_point": {"latitude": lat, "longitude": lon},
                "slope_mean_deg": 8.0, "gain_m": 120.0, "loss_m": 90.0,
                "ingress": {"junctions": 2, "area_crossings": 2},
                "effort": {"hours": 1.5, "assumptions": "see assumptions.effort"},
                "overlap_existing_pct": None, "overlap_labels": [],
                "overlap_reason": "no map was read, so overlap is unknown", **extra}

    candidates = [
        Candidate(id=f"{SET_ID}.1", kind="line", axis="trail", axis_ids=["route_0"],
                  sources=["trail"], name="Spring Ridge Trail",
                  geometry={"type": "LineString",
                            "coordinates": [[lon - 0.01, lat], [lon + 0.01, lat]]},
                  facts=facts(0.0, 0.0, None, branches=1, clipped_at_edge=False),
                  norm_flags=[], warnings=[]),
        Candidate(id=f"{SET_ID}.2", kind="line", axis="road", axis_ids=["road_1", "road_2"],
                  sources=["road"], name="Skyline Blvd",
                  geometry={"type": "MultiLineString", "coordinates": [
                      [[lon - 0.01, lat + 0.004], [lon, lat + 0.004]],
                      [[lon, lat + 0.004], [lon + 0.004, lat + 0.01]]]},
                  facts=facts(445.0, 6.2, 0.0, branches=2, clipped_at_edge=True),
                  norm_flags=[{"metric": "length_m", "value": 9200.0, "corpus_p10": 593.0,
                               "corpus_p50": 1800.0, "corpus_p90": 5200.0,
                               "band": "corpus", "note": "Line length 9200 is above the "
                               "corpus p10-p90 band (593 to 5200); unusual on these maps, "
                               "not necessarily wrong"}],
                  warnings=["Skyline Blvd forks into 2 pieces: not a single walk end to end"]),
        Candidate(id=f"{SET_ID}.3", kind="corridor", axis="waterway", axis_ids=["w_7"],
                  sources=["waterway"], name=None,
                  geometry={"type": "Polygon",
                            "coordinates": [_square(lon + 0.006, lat - 0.004, 0.0015)]},
                  facts=facts(610.0, 14.5, 135.0, width_m=30.0, area_acres=1.8),
                  norm_flags=[], warnings=[]),
    ]
    return CandidateSet(
        set_id=SET_ID, tool="hasty_candidates", params={"limit": limit}, handle=handle,
        created_at="2026-10-22T09:00:00+00:00", candidates=candidates,
        summary={"count": 3, "by_kind": {"line": 2, "corridor": 1},
                 "by_axis": {"trail": 1, "road": 1, "waterway": 1},
                 "dropped_beyond_max_distance": 0, "truncated_to": limit,
                 "total_before_limit": 41, "existing_assignments": 0},
        assumptions={"ordered_by": "distance_from_center_m ascending - a fact, not a ranking"},
        warnings=["no map was read, so overlap is unknown"], map_version=None)


def parking_payload(arguments):
    lat, lon = PLS

    def lot(index, name, distance, flags):
        point = {"latitude": lat + 0.001 * index, "longitude": lon}
        return {"id": f"parking-w{index}", "kind": "parking", "name": name, "point": point,
                "area_m2": 900.0, "surface": "gravel", "capacity": "12", "flags": flags,
                "distance_from_center_m": distance, "bearing_from_center_deg": 0.0,
                "nearest_way": {"name": "Skyline Blvd", "route_type": "ROAD",
                                "distance_m": 20.0},
                "geometry": {"type": "Polygon", "coordinates": [
                    _square(lon, point["latitude"], 0.0002)]},
                "warnings": []}

    lots = [lot(1, "Trailhead lot", 140.0, ["fee"]), lot(2, None, 380.0, ["access_private"])]
    limit = arguments.get("limit") or 0
    kept = lots[:limit] if limit else lots
    return {"candidates": kept,
            "summary": {"count": len(kept), "total_before_limit": 2,
                        "truncated_to": limit if limit and limit < 2 else None},
            "query_version": 5, "center": None, "warnings": [],
            "staleness": {"current": True}, "map_provenance": None}


def rings_payload(arguments):
    """What travel_time_rings returns: polygons, outlines and routes, all
    of which the section must leave out."""
    lat, lon = arguments["center_latitude"], arguments["center_longitude"]
    rings = []
    for index, minutes in enumerate(arguments["minutes"]):
        half = 0.004 * (index + 1)
        polygon = {"type": "Polygon", "coordinates": [_square(lon, lat, half)]}
        edge = minutes >= 60
        geometry = (polygon if not edge else {"type": "MultiPolygon", "coordinates": [
            polygon["coordinates"], [_square(lon + 0.05, lat, 0.001)]]})
        far = {"type": "Point", "coordinates": [lon + half, lat + half]}
        route = {"reach_m": 500.0 * (index + 1), "bearing_deg": 0.0, "minutes": minutes,
                 "trail_minutes": minutes / 2, "off_trail_minutes": minutes / 2,
                 "leaves_trail_at": {"type": "Point", "coordinates": [lon, lat + half / 2]},
                 "trail": {"name": "Spring Ridge Trail", "route_type": "TRAIL",
                           "highway": "path"},
                 "point": far,
                 "route": {"type": "LineString", "coordinates": [[lon, lat], far["coordinates"]]}}
        rings.append({
            "minutes": minutes, "geometry": geometry,
            "outline": {"type": "LineString", "coordinates": _square(lon, lat, half)},
            "parts": 2 if edge else 1, "area_acres": 120.0 * (index + 1) ** 2,
            "equivalent_radius_m": 618.0 * (index + 1), "max_reach_m": 900.0 * (index + 1),
            "max_reach_bearing_deg": 45.0, "farthest_point": far,
            "reach_by_direction_m": {direction: 500.0 * (index + 1) for direction in
                                     ("N", "NE", "E", "SE", "S", "SW", "W", "NW")},
            "routes_by_direction": {"N": route, "NE": None},
            "reaches_edge": edge})
    warnings = ["this ground has no LANDFIRE vegetation"]
    if any(ring["reaches_edge"] for ring in rings):
        warnings.append("the 60-minute ring reaches the edge of the staged ground")
    return {"rings": rings, "center": {"latitude": lat, "longitude": lon},
            "resolution_m": 10.0, "speed_factor": arguments.get("speed_factor", 1.0),
            "ground": {"vegetation": {"Tree": 0.6, "Shrub": 0.4}, "water": {"stream": 3},
                       "closed_share": 0.02},
            "assumptions": {}, "warnings": warnings, "staleness": {"current": True}}


class Stubs:
    """Staging and every feature, recording every call. `fail` names
    sections whose feature (or, for export, `caltopo_export`) raises;
    `fail_resolutions` grounds staging cannot fetch (None is the search
    ground's default raster)."""

    def __init__(self, monkeypatch, *, fail=(), fail_resolutions=()):
        self.fail = set(fail)
        self.staging = AreaStaging(fail_resolutions)
        self.calls = {name: [] for name in SECTIONS}
        self.services = FakeServices(
            [record("Stanford Hospital", *STANFORD, trauma="LEVEL I", helipad="Y",
                    address="300 PASTEUR DR", city="STANFORD"),
             record("Sequoia Hospital", *SEQUOIA, address="170 ALAMEDA DE LAS PULGAS",
                    city="REDWOOD CITY")],
            minutes={STANFORD: 40.0, SEQUOIA: 30.0})
        self.real_export = candidates_component.caltopo_export
        monkeypatch.setattr(staging_component, "prepare_area", self.staging)
        monkeypatch.setattr(hasty_feature, "hasty_candidates", self.hasty)
        monkeypatch.setattr(parking_feature, "parking_candidates", self.parking)
        monkeypatch.setattr(rings_feature, "travel_time_rings", self.rings)
        monkeypatch.setattr(lz_feature, "lz_candidates", self.lz)
        monkeypatch.setattr(medical_feature, "nearest_medical_facilities", self.medical)
        monkeypatch.setattr(candidates_component, "caltopo_export", self.export)

    def hasty(self, ctx, **arguments):
        self.calls["hasty"].append(arguments)
        resolved = ctx.staged(arguments["handle"])      # as the tool does
        if "hasty" in self.fail:
            raise ValueError("no line candidates: osm.json missing")
        cset = hasty_set(arguments["handle"], arguments.get("limit"))
        save_set(resolved.directory, cset)
        payload = cset.model_dump(exclude={"candidates", "manifest_stamp"})
        payload.update(candidates=summaries(cset), reused=False, staleness={"current": True},
                       map_provenance=None)
        return payload

    def parking(self, ctx, **arguments):
        self.calls["parking"].append(arguments)
        ctx.staged(arguments["handle"])
        if "parking" in self.fail:
            raise ValueError("osm.json predates parking")
        return parking_payload(arguments)

    def rings(self, ctx, **arguments):
        self.calls["travel_rings"].append(arguments)
        ctx.staged(arguments["handle"])
        if "travel_rings" in self.fail:
            raise ValueError("the location falls on a gap in the elevation model")
        return rings_payload(arguments)

    def lz(self, ctx, **arguments):
        self.calls["lz"].append(arguments)
        ctx.staged(arguments["handle"])
        if "lz" in self.fail:
            raise ValueError("no canopy panel")
        return lz_payload(arguments)

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

    def radii(self) -> dict:
        """{resolution: radius} of every ground staged."""
        return {call["resolution_m"]: call["radius_m"] for call in self.staging.calls}


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
    result = plan_initial_response(ctx, *PLS)

    assert result["status"] == "ok"
    assert statuses(result) == {name: "ok" for name in SECTIONS}
    assert all(entry["error"] is None for entry in result["sections"].values())
    assert result["place_last_seen"] == {"latitude": PLS[0], "longitude": PLS[1]}
    assert result["radius_m"] == 1500.0

    # Three grounds, each centred on the place last seen at the resolution
    # its features need; the rings ground the largest prepare_area stages,
    # since an hour at the model's top speed is 6,000 m.
    centre = {"center_latitude": PLS[0], "center_longitude": PLS[1], "map_id": "",
              "public": True}
    assert stubs.staging.calls == [{**centre, "radius_m": 1500.0, "resolution_m": None},
                                   {**centre, "radius_m": 4499.0, "resolution_m": 10.0},
                                   {**centre, "radius_m": 1000.0, "resolution_m": 2.0}]
    handles = result["handles"]
    assert list(handles) == ["search", "rings", "lz"] and len(set(handles.values())) == 3
    sections = result["sections"]
    assert {name: sections[name].get("ground") for name in SECTIONS} == {
        "hasty": "search", "parking": "search", "travel_rings": "rings", "lz": "lz",
        "medical": None, "export": None}

    hasty = sections["hasty"]
    assert hasty["set_id"] == SET_ID and hasty["handle"] == handles["search"]
    assert hasty["created_at"] == "2026-10-22T09:00:00+00:00" and hasty["reused"] is False
    assert [c["id"] for c in hasty["candidates"]] == [f"{SET_ID}.{i}" for i in (1, 2, 3)]
    trail, road, creek = hasty["candidates"]
    assert trail == {"id": f"{SET_ID}.1", "kind": "line", "axis": "trail",
                     "name": "Spring Ridge Trail", "length_m": 2400.0, "branches": 1,
                     "clipped_at_edge": False, "distance_from_center_m": 0.0,
                     "bearing_from_center_deg": None, "travel_minutes_from_center": 0.0,
                     "overlap_existing_pct": None, "overlap_labels": [],
                     "flags": [], "warnings": []}
    # Whether a map was read is the set's: said once, not on every candidate.
    assert hasty["overlap_reason"] == "no map was read, so overlap is unknown"
    assert not any("overlap_reason" in candidate for candidate in hasty["candidates"])
    assert road["flags"] == [hasty_set("h", 20).candidates[1].norm_flags[0]["note"]]
    assert road["branches"] == 2 and road["warnings"][0].startswith("Skyline Blvd forks")
    assert creek["width_m"] == 30.0 and creek["area_acres"] == 1.8
    assert "branches" not in creek
    assert hasty["by_axis"] == {"trail": 1, "road": 1, "waterway": 1}
    assert hasty["by_kind"] == {"line": 2, "corridor": 1}
    # The limit cut the list (41 found); max_distance_m was not given.
    assert hasty["total_before_limit"] == 41 and hasty["truncated_to"] == 20
    assert hasty["dropped_beyond_max_distance"] == 0
    # The default list is trails and roads, and the detail_call says so.
    assert hasty["detail_call"] == {"tool": "hasty_candidates", "arguments": {
        "handle": handles["search"], "limit": 20, "include": ["trail", "road"]}}

    parking = sections["parking"]
    assert [lot["id"] for lot in parking["candidates"]] == ["parking-w1", "parking-w2"]
    assert parking["candidates"][0]["flags"] == ["fee"]
    assert parking["detail_call"] == {"tool": "parking_candidates", "arguments": {
        "handle": handles["search"], "limit": 5}}

    rings = sections["travel_rings"]
    assert [ring["minutes"] for ring in rings["rings"]] == [15.0, 30.0, 60.0]
    assert set(rings["rings"][0]) == set(_sections.RING_FIELDS)
    assert rings["rings"][2]["reaches_edge"] is True and rings["rings"][2]["parts"] == 2
    assert rings["ground"] == "rings" and rings["ground_facts"]["closed_share"] == 0.02
    assert rings["speed_factor"] == 1.0 and rings["resolution_m"] == 10.0
    assert rings["detail_call"] == {"tool": "travel_time_rings", "arguments": {
        "handle": handles["rings"], "center_latitude": PLS[0], "center_longitude": PLS[1],
        "minutes": [15.0, 30.0, 60.0]}}

    lz = sections["lz"]
    assert [zone["id"] for zone in lz["candidates"]] == ["lz_1", "lz_2", "lz_3"]
    assert lz["detail_call"]["arguments"]["limit"] == 5

    medical = sections["medical"]
    assert [row["name"] for row in medical["categories"]["trauma_center"]] == [
        "Stanford Hospital"]
    assert medical["detail_call"]["arguments"] == {"latitude": PLS[0], "longitude": PLS[1],
                                                    "per_category": 3}

    export = sections["export"]
    exports_dir = staged_dir(ctx, handles["search"]) / "exports"
    assert export["path"] == str(exports_dir / f"{SET_ID}.geojson")
    assert export["assignments"] == 4                 # the forked road is two
    assert [entry["label"] for entry in export["labels"]] == [
        "U01 Spring Ridge Trail", "U02a Skyline Blvd", "U02b Skyline Blvd", "U03"]
    assert export["area_included"] is True
    assert export["detail_call"] == {"tool": "caltopo_export", "arguments": {
        "handle": handles["search"], "set_id": SET_ID, "include_area": True}}
    assert export["markers_path"] == str(exports_dir / "initial_response.geojson")
    # The place, two lots, three ring outlines, three zones, two hospitals.
    assert export["markers"] == {"Marker": 8, "Shape": 3}
    assert result["exports"] == [export["path"], export["markers_path"]]

    summary = result["summary"]
    assert set(summary) == {"hasty", "parking", "travel_rings", "lz", "medical"}
    assert summary["hasty"] == {"count": 3, "by_kind": {"line": 2, "corridor": 1},
                                "by_axis": {"trail": 1, "road": 1, "waterway": 1},
                                "total_before_limit": 41}
    assert summary["parking"] == {"count": 2, "total_before_limit": 2, "nearest": {
        "id": "parking-w1", "kind": "parking", "name": "Trailhead lot",
        "distance_from_center_m": 140.0, "bearing_from_center_deg": 0.0, "flags": ["fee"]}}
    assert summary["travel_rings"][0] == {"minutes": 15.0, "area_acres": 120.0,
                                          "equivalent_radius_m": 618.0, "max_reach_m": 900.0,
                                          "reaches_edge": False}
    assert summary["lz"]["nearest_any"]["id"] == "lz_1"
    assert summary["lz"]["nearest_preferred"]["id"] == "lz_2"
    assert summary["medical"]["nearest_trauma_center"]["name"] == "Stanford Hospital"
    assert result["warnings"] == []
    assert "No verdict" in result["note"]


def test_the_answer_carries_points_but_no_lines_or_polygons(ctx, monkeypatch):
    Stubs(monkeypatch)
    result = plan_initial_response(ctx, *PLS)
    assert geometry_types_in(result) == {"Point"}
    json.dumps(result)                       # and it is JSON as it stands


# --- What the hasty list leaves out ---------------------------------------------


def counted_hasty(*, map_version=None, existing_assignments=0):
    """A hasty_candidates stub that counts as the feature does: the stub
    set's candidates beyond `max_distance_m` dropped, then `limit` of the
    rest kept (0 for every one), the summary saying what each step dropped.
    `map_version` None is no map read: overlap unknown, as the feature
    says it; otherwise the map was read and covers none of them."""
    from sar_geo_mcp.planning.corridors import NO_MAP_OVERLAP_REASON

    read = map_version is not None

    def hasty(ctx, **arguments):
        resolved = ctx.staged(arguments["handle"])
        limit, reach = arguments.get("limit"), arguments.get("max_distance_m")
        cset = hasty_set(arguments["handle"], limit)
        near = [candidate for candidate in cset.candidates
                if reach is None or candidate.facts["distance_from_center_m"] <= reach]
        kept = near[:limit] if limit else near
        for candidate in kept:
            candidate.facts.update(overlap_existing_pct=0.0 if read else None,
                                   overlap_reason=None if read else NO_MAP_OVERLAP_REASON)
        by_kind, by_axis = {}, {}
        for candidate in kept:
            by_kind[candidate.kind] = by_kind.get(candidate.kind, 0) + 1
            by_axis[candidate.axis] = by_axis.get(candidate.axis, 0) + 1
        cset = cset.model_copy(update={
            "candidates": kept, "map_version": map_version,
            "warnings": [] if read else [NO_MAP_OVERLAP_REASON],
            "summary": {"count": len(kept), "by_kind": by_kind, "by_axis": by_axis,
                        "dropped_beyond_max_distance": len(cset.candidates) - len(near),
                        "truncated_to": limit if limit and len(near) > limit else None,
                        "total_before_limit": len(near),
                        "existing_assignments": existing_assignments}})
        save_set(resolved.directory, cset)
        payload = cset.model_dump(exclude={"candidates", "manifest_stamp"})
        payload.update(candidates=summaries(cset), reused=False, staleness={"current": True},
                       map_provenance=None)
        return payload

    return hasty


@pytest.mark.parametrize("hasty_limit, kept, truncated_to", [
    (1, [1], 1),            # the creek beyond 500 m, then the road past the limit
    (0, [1, 2], None),      # every one: only the creek dropped
])
def test_the_hasty_section_says_what_the_limit_and_max_distance_dropped(
        ctx, monkeypatch, hasty_limit, kept, truncated_to):
    """The documented counts that tell a planner the list is not every
    line reach the section as the feature gave them: what `max_distance_m`
    dropped, what the limit cut, and the assignments on the map read."""
    Stubs(monkeypatch)
    monkeypatch.setattr(hasty_feature, "hasty_candidates",
                        counted_hasty(map_version="2026-10-22T09:00:00+00:00",
                                      existing_assignments=2))
    result = plan_initial_response(ctx, *PLS, include=["hasty"], hasty_limit=hasty_limit,
                                   overrides={"hasty": {"max_distance_m": 500}})
    hasty = result["sections"]["hasty"]
    assert [candidate["id"] for candidate in hasty["candidates"]] == [
        f"{SET_ID}.{index}" for index in kept]
    assert hasty["dropped_beyond_max_distance"] == 1          # the creek, 610 m out
    assert hasty["total_before_limit"] == 2
    assert hasty["truncated_to"] == truncated_to
    assert hasty["existing_assignments"] == 2
    assert hasty["overlap_reason"] is None                    # the map was read
    assert [candidate["overlap_existing_pct"] for candidate in hasty["candidates"]] == [
        0.0] * len(kept)
    assert hasty["detail_call"]["arguments"] == {
        "handle": result["handles"]["search"], "limit": hasty_limit,
        "include": ["trail", "road"], "max_distance_m": 500}
    assert result["summary"]["hasty"] == {
        "count": len(kept), "by_kind": {"line": len(kept)},
        "by_axis": {"trail": 1, "road": 1} if len(kept) == 2 else {"trail": 1},
        "total_before_limit": 2}


def test_the_hasty_list_is_trails_and_roads_unless_the_caller_names_the_axes(ctx,
                                                                             monkeypatch):
    """A first operational period lists the walkable lines by default; the
    terrain corridors come back only when the caller asks - null for every
    axis, named in full because hasty_candidates reads null as trails and
    roads, or the axes named. The default is never the caller's, so it is no
    unused override."""
    stubs = Stubs(monkeypatch)
    default = plan_initial_response(ctx, *PLS, include=["hasty"])
    assert stubs.calls["hasty"][-1]["include"] == ["trail", "road"]
    assert default["sections"]["hasty"]["detail_call"]["arguments"]["include"] == [
        "trail", "road"]
    assert default["warnings"] == []
    every = plan_initial_response(ctx, *PLS, include=["hasty"],
                                  overrides={"hasty": {"include": None}})
    every_axis = ["trail", "road", "waterway"]
    assert stubs.calls["hasty"][-1]["include"] == every_axis
    assert every["sections"]["hasty"]["detail_call"]["arguments"]["include"] == every_axis
    named = plan_initial_response(ctx, *PLS, include=["hasty"],
                                  overrides={"hasty": {"include": ["trail", "waterway"]}})
    assert stubs.calls["hasty"][-1]["include"] == ["trail", "waterway"]
    assert named["warnings"] == []
    # Left out of include, no hasty call is made and nothing is said of it.
    calls = len(stubs.calls["hasty"])
    rings = plan_initial_response(ctx, *PLS, include=["travel_rings"])
    assert len(stubs.calls["hasty"]) == calls and rings["warnings"] == []


@pytest.mark.parametrize("include, warned", [
    (None, True),                               # the default trail-and-road list
    (["trail", "road"], True),
    (["trail", "waterway"], False),
    ("every", False),                           # null: every axis
])
def test_corridor_widths_with_no_corridor_axis_listed_are_said_unused(ctx, monkeypatch,
                                                                      include, warned):
    """Widths for corridors the list leaves out change nothing; the caller
    who gave them most likely meant the old every-axis list."""
    Stubs(monkeypatch)
    hasty = {"corridor_widths_m": {"waterway": 60}}
    if include == "every":
        hasty["include"] = None
    elif include is not None:
        hasty["include"] = include
    result = plan_initial_response(ctx, *PLS, include=["hasty"], overrides={"hasty": hasty})
    said = [w for w in result["warnings"] if "corridor_widths_m'] was not used" in w]
    assert bool(said) is warned
    if warned:
        assert "['road', 'trail']" in said[0] and "null for every axis" in said[0]


@pytest.mark.xfail(strict=True, reason="PR #4 review B43: hasty existing_assignments reads 0 "
                   "when no map was read, while the docs say it counts the map's assignments")
@pytest.mark.parametrize("routes", ["one-trail", "no-lines"])
def test_no_map_read_is_an_unknown_count_not_zero_assignments(ctx, monkeypatch, routes):
    """The default call reads no map (map_id ""), and the real feature then
    counts 0 existing assignments: the section must say unknown - null,
    with `overlap_reason` saying why - not "nothing assigned yet", and must
    say it with no candidates too, where no candidate carries the reason."""
    from test_planning_tools import _stub_staging, _trail_through

    _stub_staging(monkeypatch, routes=[_trail_through(PLS)] if routes == "one-trail" else [],
                  width=120)
    result = plan_initial_response(ctx, *PLS, radius_m=400, include=["hasty"])
    hasty = result["sections"]["hasty"]
    assert hasty["status"] == "ok"
    assert len(hasty["candidates"]) == (1 if routes == "one-trail" else 0)
    assert hasty["overlap_reason"] is not None
    assert hasty["existing_assignments"] is None


# --- Failures stay in their section --------------------------------------------


def test_one_section_failing_leaves_the_others_and_the_status_partial(ctx, monkeypatch):
    Stubs(monkeypatch, fail={"travel_rings"})
    result = plan_initial_response(ctx, *PLS)
    assert result["status"] == "partial"
    rings = result["sections"]["travel_rings"]
    assert rings["status"] == "failed"
    assert rings["error"] == "ValueError: the location falls on a gap in the elevation model"
    # The call that raised, to retry or make directly.
    assert rings["detail_call"]["tool"] == "travel_time_rings"
    assert {name for name, status in statuses(result).items() if status == "ok"} == {
        "hasty", "parking", "lz", "medical", "export"}
    assert result["summary"]["travel_rings"] is None
    assert result["summary"]["hasty"] is not None
    assert len(result["exports"]) == 2                 # what was found is still drawn


@pytest.mark.parametrize("resolution, failed", [
    (None, {"hasty", "parking"}),
    (10.0, {"travel_rings"}),
    (2.0, {"lz"}),
])
def test_a_staging_failure_fails_only_the_sections_on_that_ground(ctx, monkeypatch,
                                                                 resolution, failed):
    stubs = Stubs(monkeypatch, fail_resolutions={resolution})
    result = plan_initial_response(ctx, *PLS)
    assert result["status"] == "partial"
    assert {name for name, status in statuses(result).items() if status == "failed"} == failed
    ground = {None: "search", 10.0: "rings", 2.0: "lz"}[resolution]
    assert result["staging"][ground]["status"] == "failed"
    assert result["handles"][ground] is None
    assert result["sections"]["export"]["status"] == "ok"   # an export never fails
    for name in failed:
        entry = result["sections"][name]
        assert entry["error"].startswith(f"StagingFailed: the {ground} ground")
        assert "RuntimeError: 3DEP could not be reached" in entry["error"]
        assert "detail_call" not in entry                # no feature call was made
        assert stubs.calls[name] == []
    assert result["staging"][ground]["detail_call"]["tool"] == "prepare_area"


def test_a_search_ground_that_cannot_be_staged_still_leaves_the_markers(ctx, monkeypatch):
    """Without the search ground there is no hasty set to draft: the export
    says so as a warning - an export failure is never a failed section - and
    what the other sections found is still written, beside the rings
    ground. The hasty failure is counted once, on hasty."""
    stubs = Stubs(monkeypatch, fail_resolutions={None})
    result = plan_initial_response(ctx, *PLS)
    export = result["sections"]["export"]
    assert export["status"] == "ok" and export["error"] is None
    assert export["path"] is None
    assert export["warnings"][0] == ("the hasty section did not answer, so there is no "
                                     "candidate set to draw as draft assignments; its error "
                                     "says why")
    assert "detail_call" not in export and stubs.calls["export"] == []
    rings_dir = staged_dir(ctx, result["handles"]["rings"]) / "exports"
    assert export["markers_path"] == str(rings_dir / "initial_response.geojson")
    assert result["exports"] == [export["markers_path"]]
    assert result["warnings"] == []           # something was found, and it was written


def test_an_export_that_raises_is_a_warning_and_keeps_its_call(ctx, monkeypatch):
    """caltopo_export raising - a full disk - is a warning on the export
    section, never a failure: every planning section answered, so the
    status stays ok, and the call is kept to make again."""
    stubs = Stubs(monkeypatch, fail={"export"})
    result = plan_initial_response(ctx, *PLS, overrides={"export": {"label_prefix": "H"}})
    export = result["sections"]["export"]
    assert export["status"] == "ok" and export["error"] is None
    assert export["path"] is None and export["set_id"] == SET_ID
    assert export["warnings"][0] == (
        "the draft assignments could not be written (OSError: [Errno 28] No space left on "
        "device); every section's answer stands, and detail_call makes the export again")
    assert export["detail_call"] == {"tool": "caltopo_export", "arguments": {
        "handle": result["handles"]["search"], "set_id": SET_ID, "include_area": True,
        "label_prefix": "H"}}
    assert stubs.calls["export"] == [export["detail_call"]["arguments"]]
    assert result["status"] == "ok"
    # The markers are still drawn; no draft assignments file is claimed, and
    # the override was used - in the call that raised. What the top-level
    # warnings say is held by the B10 test below, not pinned empty here.
    assert result["exports"] == [export["markers_path"]]


def test_draft_assignments_not_written_are_said_at_the_top_level(ctx, monkeypatch):
    """The three hasty lines found are in no file - the markers file holds
    only the place, parking, rings, zones and hospitals - so the top level
    must say something found was not written, as plan_medevac's does and
    as the description promises; never that nothing was found."""
    Stubs(monkeypatch, fail={"export"})
    result = plan_initial_response(ctx, *PLS)
    assert result["summary"]["hasty"]["count"] == 3
    assert result["sections"]["export"]["path"] is None
    markers = read_export(result["exports"][0])
    assert "Assignment" not in {f["properties"]["class"] for f in markers["features"]}
    assert len(result["warnings"]) == 1
    warning = result["warnings"][0]
    assert "draft assignments" in warning and "written" in warning
    assert "nothing was found" not in warning


def test_every_section_failing_is_failed_and_draws_nothing(ctx, monkeypatch):
    Stubs(monkeypatch, fail={"hasty", "parking", "travel_rings", "lz", "medical"})
    result = plan_initial_response(ctx, *PLS)
    # The export finds nothing of its own, so it does not hold the status up.
    assert result["status"] == "failed"
    assert {name: status for name, status in statuses(result).items()} == {
        **{name: "failed" for name in SECTIONS[:-1]}, "export": "ok"}
    assert result["summary"] == {name: None for name in
                                 ("hasty", "parking", "travel_rings", "lz", "medical")}
    assert result["exports"] == []
    assert result["sections"]["export"]["markers_path"] is None
    assert result["warnings"] == ["nothing was found to draw, so no CalTopo export was "
                                  "written"]


def test_a_markers_file_that_cannot_be_written_is_a_warning(ctx, monkeypatch):
    Stubs(monkeypatch)

    def full_disk(path, text):
        raise OSError("No space left on device")

    monkeypatch.setattr(_sections, "atomic_write_text", full_disk)
    result = plan_initial_response(ctx, *PLS)
    export = result["sections"]["export"]
    assert result["status"] == "ok" and export["status"] == "ok"
    assert export["markers_path"] is None and export["markers"] == {}
    # Which file the warning names is B10's (below); why it failed is here.
    assert export["warnings"][-1].endswith("could not be written (OSError: No space left on "
                                           "device); every section's answer stands")
    assert result["exports"] == [export["path"]]       # the draft assignments still were


def test_a_markers_file_not_written_is_said_at_the_top_level(ctx, monkeypatch):
    """The landing zones, hospitals, rings and parking found are in no file
    when only the markers file fails: the top level must say so, and the
    export's own warning must name the markers file rather than "the
    CalTopo export", which beside the drafts' warning reads as both."""
    Stubs(monkeypatch)

    def full_disk(path, text):
        raise OSError("No space left on device")

    monkeypatch.setattr(_sections, "atomic_write_text", full_disk)
    result = plan_initial_response(ctx, *PLS)
    assert result["exports"] == [result["sections"]["export"]["path"]]
    assert len(result["warnings"]) == 1
    warning = result["warnings"][0]
    assert "written" in warning and "nothing was found" not in warning
    assert "initial_response.geojson" in warning or "marker" in warning
    section_warning = result["sections"]["export"]["warnings"][-1]
    assert "initial_response.geojson" in section_warning or "marker" in section_warning


def test_a_full_disk_says_nothing_was_written_not_that_nothing_was_found(ctx, monkeypatch):
    """Both files unwritable, every section answered: the status is ok, the
    export says why each file is missing, and the top level says something
    found was not written - never that the search found nothing."""
    from sar_geo_mcp import cache

    Stubs(monkeypatch)

    def full_disk(path, text):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(_sections, "atomic_write_text", full_disk)
    monkeypatch.setattr(cache, "atomic_write_text", full_disk)     # export_set's
    result = plan_initial_response(ctx, *PLS)
    assert result["status"] == "ok"
    assert set(statuses(result).values()) == {"ok"}
    export = result["sections"]["export"]
    assert export["path"] is None and export["markers_path"] is None
    assert len(export["warnings"]) == 2
    assert export["warnings"][0].startswith("the draft assignments could not be written "
                                            "(OSError: [Errno 28]")
    assert export["warnings"][1].startswith("the markers file initial_response.geojson could "
                                            "not be written (OSError: [Errno 28]")
    assert result["exports"] == []
    assert result["warnings"] == [
        "no CalTopo export was written, although the sections found things to draw: "
        "sections.export.warnings says why, and every section's answer stands"]


@pytest.mark.parametrize("disk_full, said", [
    (False, "the draft assignments of the hasty set were not written - only the markers "
            "file initial_response.geojson was: sections.export.error says why, and every "
            "section's answer stands"),
    (True, "no CalTopo export was written, although the sections found things to draw: "
           "sections.export.error and sections.export.warnings say why, and every "
           "section's answer stands"),
], ids=["markers-written", "markers-refused"])
def test_an_export_section_that_raised_is_pointed_to_by_its_error(ctx, monkeypatch,
                                                                  disk_full, said):
    """An export section that raised outside caltopo_export is failed and
    says why in its error, not its warnings (`guard`): the top level sends
    the planner there - and to the warnings too when the markers file was
    refused beside it - never to an empty list alone."""
    Stubs(monkeypatch)

    def bug(*args, **kwargs):
        raise RuntimeError("bug")

    def full_disk(path, text):
        raise OSError("No space left on device")

    monkeypatch.setattr("sar_geo_mcp.scenarios.initial_response.set_export_section", bug)
    if disk_full:
        monkeypatch.setattr(_sections, "atomic_write_text", full_disk)
    result = plan_initial_response(ctx, *PLS)
    export = result["sections"]["export"]
    assert export["status"] == "failed" and export["error"] == "RuntimeError: bug"
    assert bool(export["warnings"]) is disk_full
    assert (export["markers_path"] is None) is disk_full
    assert result["warnings"] == [said]


def test_no_ground_to_write_beside_is_not_nothing_found(ctx, monkeypatch):
    """Medical needs no ground: the hospitals are found, but there is no
    staged directory to write them beside."""
    Stubs(monkeypatch)
    result = plan_initial_response(ctx, *PLS, include=["medical", "export"])
    assert result["status"] == "ok"
    export = result["sections"]["export"]
    assert export["status"] == "ok" and export["markers_path"] is None
    assert export["warnings"][-1].startswith("no ground was staged, so there is nowhere")
    assert result["exports"] == []
    assert result["warnings"] == [
        "no CalTopo export was written, although the sections found things to draw: "
        "sections.export.warnings says why, and every section's answer stands"]


def test_every_section_asked_for_failing_is_failed_with_hasty_left_out(ctx, monkeypatch):
    """The export, asked for beside them, answered nothing of its own: it
    must not turn "failed" into "partial"."""
    Stubs(monkeypatch, fail={"parking", "travel_rings", "lz", "medical"})
    result = plan_initial_response(ctx, *PLS, include=["parking", "travel_rings", "lz",
                                                       "medical", "export"])
    assert result["status"] == "failed"
    assert result["sections"]["export"]["status"] == "ok"
    assert result["exports"] == []
    assert result["warnings"] == ["nothing was found to draw, so no CalTopo export was "
                                  "written"]


@pytest.mark.parametrize("include, fail, why", [
    (["travel_rings", "export"], (), "hasty was left out of include"),
    (None, ("hasty",), "the hasty section did not answer"),
])
def test_export_overrides_with_no_set_to_draw_are_said_unused(ctx, monkeypatch, include, fail,
                                                              why):
    stubs = Stubs(monkeypatch, fail=fail)
    result = plan_initial_response(ctx, *PLS, include=include,
                                   overrides={"export": {"label_prefix": "H"}})
    assert stubs.calls["export"] == []
    assert result["warnings"] == [f"overrides['export'] was not used: {why}, so there was no "
                                  "candidate set for caltopo_export to draw"]


# --- What the caller chooses --------------------------------------------------


def test_include_skips_sections_and_stages_nothing_for_them(ctx, monkeypatch):
    stubs = Stubs(monkeypatch)
    result = plan_initial_response(ctx, *PLS, include=["medical"],
                                   overrides={"hasty": {"min_length_m": 593}})
    assert result["status"] == "ok"
    for name in ("hasty", "parking", "travel_rings", "lz", "export"):
        assert result["sections"][name] == {"status": "skipped", "error": None,
                                             "warnings": []}
    assert stubs.staging.calls == [] and stubs.calls["hasty"] == []
    assert result["staging"] == {} and result["handles"] == {}
    assert set(result["summary"]) == {"medical"}
    assert result["exports"] == []
    assert result["warnings"] == ["overrides['hasty'] was not used: hasty was left out of "
                                  "include"]


def test_parking_alone_stages_only_the_search_ground(ctx, monkeypatch):
    stubs = Stubs(monkeypatch)
    result = plan_initial_response(ctx, *PLS, include=["parking"], parking_limit=1)
    assert stubs.radii() == {None: 1500.0}
    assert [lot["id"] for lot in result["sections"]["parking"]["candidates"]] == [
        "parking-w1"]
    assert result["summary"]["parking"]["total_before_limit"] == 2


def test_the_landing_zone_ground_is_the_one_plan_medevac_stages(ctx, monkeypatch):
    """The description sells it: the 1,000 m ground at 2 m is plan_medevac's
    default, so a medevac planned later from the place last seen stands on
    it - the same prepare_area call, so the same handle, and no second
    staging of a minute and a half."""
    from sar_geo_mcp.scenarios.medevac import plan_medevac

    stubs = Stubs(monkeypatch)
    initial = plan_initial_response(ctx, *PLS, include=["lz"])
    medevac = plan_medevac(ctx, *PLS, include=["lz"])
    lz_ground = {"center_latitude": PLS[0], "center_longitude": PLS[1], "radius_m": 1000.0,
                 "map_id": "", "public": True, "resolution_m": 2.0}
    assert stubs.staging.calls == [lz_ground, lz_ground]
    assert initial["handles"] == {"lz": medevac["handles"]["lz"]}
    assert initial["sections"]["lz"]["handle"] == medevac["sections"]["lz"]["handle"]


@pytest.mark.xfail(strict=True, reason="PR #4 review B41: 'a medevac planned later from this "
                   "point reuses' plan_initial_response's LZ ground fails whenever map_id is "
                   "given")
def test_a_map_id_reaches_only_the_search_ground(ctx, monkeypatch):
    """map_id is for the hasty lines' overlap with the map's assignments, so
    only the search ground takes it. Passed to the rings and landing-zone
    grounds too, it changes their handles: the plan_medevac reuse the
    description promises then fails whenever a map is named - the usual
    case - and those grounds read the live map for nothing."""
    from sar_geo_mcp.scenarios.medevac import plan_medevac

    stubs = Stubs(monkeypatch)
    initial = plan_initial_response(ctx, *PLS, map_id="ABC1234")
    assert {call["resolution_m"]: call["map_id"] for call in stubs.staging.calls} == {
        None: "ABC1234", 10.0: "", 2.0: ""}
    medevac = plan_medevac(ctx, *PLS, include=["lz"])
    assert medevac["handles"]["lz"] == initial["handles"]["lz"]


def test_with_hasty_left_out_the_export_draws_only_the_markers(ctx, monkeypatch):
    stubs = Stubs(monkeypatch)
    result = plan_initial_response(ctx, *PLS, include=["travel_rings", "export"])
    export = result["sections"]["export"]
    assert export["status"] == "ok" and export["path"] is None
    assert export["warnings"] == ["hasty was left out of include, so there is no candidate "
                                  "set to draw as draft assignments"]
    assert stubs.calls["export"] == []
    assert export["markers"] == {"Marker": 1, "Shape": 3}
    assert result["exports"] == [export["markers_path"]]
    assert result["status"] == "ok"


def _no_parking(ctx, **arguments):
    ctx.staged(arguments["handle"])
    return {"candidates": [], "summary": {"count": 0, "total_before_limit": 0,
                                          "truncated_to": None},
            "query_version": 5, "center": None, "warnings": [],
            "staleness": {"current": True}, "map_provenance": None}


@pytest.mark.parametrize("include, parking", [
    (["hasty", "export"], None),                      # no section beside hasty to draw
    (["hasty", "parking", "export"], _no_parking),    # parking asked, and finds none
])
def test_with_nothing_for_the_markers_file_only_the_drafts_are_written(ctx, monkeypatch,
                                                                       include, parking):
    """Parking, rings, landing zones and hospitals left out or finding
    nothing: the place last seen alone is not written, `markers_path` is
    null with no warning - nothing went wrong - and the drafts are the one
    file in the exports folder."""
    stubs = Stubs(monkeypatch)
    if parking is not None:
        monkeypatch.setattr(parking_feature, "parking_candidates", parking)
    result = plan_initial_response(ctx, *PLS, include=include)
    assert result["status"] == "ok"
    export = result["sections"]["export"]
    assert export["path"] == str(staged_dir(ctx, result["handles"]["search"]) / "exports"
                                 / f"{SET_ID}.geojson")
    assert export["markers_path"] is None and export["markers"] == {}
    assert export["warnings"] == []
    assert result["warnings"] == []
    assert result["exports"] == [export["path"]]
    exports_dir = Path(export["path"]).parent
    assert sorted(path.name for path in exports_dir.iterdir()) == [f"{SET_ID}.geojson"]
    assert len(stubs.calls["export"]) == 1


def test_with_export_left_out_nothing_is_written(ctx, monkeypatch):
    stubs = Stubs(monkeypatch)
    result = plan_initial_response(ctx, *PLS, include=list(SECTIONS[:-1]))
    assert result["status"] == "ok"
    assert result["sections"]["export"]["status"] == "skipped"
    assert result["exports"] == [] and stubs.calls["export"] == []
    for handle in result["handles"].values():
        assert not (staged_dir(ctx, handle) / "exports").exists()


def test_the_radius_takes_a_unit_and_is_converted_once(ctx, monkeypatch):
    stubs = Stubs(monkeypatch)
    result = plan_initial_response(ctx, *PLS, radius=1, radius_unit="mi", include=["hasty"])
    assert stubs.radii() == {None: 1609.344}
    assert result["radius_m"] == 1609.344
    plan_initial_response(ctx, *PLS, radius_m=800, include=["hasty"])
    assert stubs.staging.calls[-1]["radius_m"] == 800


@pytest.mark.parametrize("unit, metres", [("km", 2000.0), ("mi", 3218.688), ("yd", 1.8288)])
def test_a_radius_unit_beside_radius_m_is_never_silently_dropped(ctx, monkeypatch, unit,
                                                                 metres):
    """radius_m=2 with radius_unit "km" reads as two kilometres; today it
    stages a 2 m circle, says "ok" with no warning, and the hasty and
    parking answers, drawn from the planning buffer, still look plausible.
    Refused before any work, the unit applied, or a warning naming
    radius_unit as ignored - anything but silence."""
    stubs = Stubs(monkeypatch)
    try:
        result = plan_initial_response(ctx, *PLS, radius_m=2, radius_unit=unit,
                                       include=["hasty"])
    except ValueError as error:
        assert "radius" in str(error)
        assert stubs.nothing_called()
        return
    assert stubs.staging.calls[0]["radius_m"] == result["radius_m"]
    assert (result["radius_m"] == pytest.approx(metres)
            or any("radius_unit" in warning for warning in result["warnings"]))


@pytest.mark.parametrize("arguments, metres", [
    ({"radius": 0.5}, 0.5),                 # half a mile, with the unit left out
    ({"radius_m": 49.9}, 49.9),
])
def test_a_search_radius_under_50_m_is_answered_with_a_warning(ctx, monkeypatch, arguments,
                                                               metres):
    """A circle of half a metre keeps almost no hasty line, which reads as
    "no hasty lines here". Answered, as plan_urban_search answers it, with
    a warning giving the radius in metres and how half a mile is given."""
    stubs = Stubs(monkeypatch)
    result = plan_initial_response(ctx, *PLS, include=["hasty"], **arguments)
    assert result["status"] == "ok" and result["radius_m"] == metres
    assert stubs.staging.calls[0]["radius_m"] == metres
    assert len(result["warnings"]) == 1
    warning = result["warnings"][0]
    assert f"{metres:g} m" in warning and "radius_unit" in warning
    assert 'radius=0.5, radius_unit="mi"' in warning


@pytest.mark.parametrize("arguments", [{"radius_m": 50}, {"radius": 0.25, "radius_unit": "mi"}])
def test_a_search_radius_of_50_m_or_more_is_answered_without_a_warning(ctx, monkeypatch,
                                                                       arguments):
    Stubs(monkeypatch)
    result = plan_initial_response(ctx, *PLS, include=["hasty"], **arguments)
    assert result["status"] == "ok" and result["warnings"] == []


@pytest.mark.parametrize("include, hasty_said", [(["hasty"], True), (["lz", "medical"], False)])
def test_a_small_radius_warning_speaks_of_hasty_lines_only_when_they_were_asked_for(
        ctx, monkeypatch, include, hasty_said):
    """A hasty line is a candidate only where it crosses the search circle,
    at any radius: what a circle a metre across means is that an empty
    hasty section says nothing of the lines nearby. With no hasty section
    asked for, the warning says nothing of hasty lines at all."""
    Stubs(monkeypatch)
    result = plan_initial_response(ctx, *PLS, radius=0.5, include=include)
    (warning,) = result["warnings"]
    assert warning.startswith("the search radius is 0.5 m, a circle 1 m across rather than a "
                              "search area")
    assert ("an empty hasty section says nothing of the lines nearby" in warning) is hasty_said
    assert ("hasty" in warning) is hasty_said


@pytest.mark.xfail(strict=True, reason="PR #4 review B45: plan_initial_response refuses an empty "
                   "ring_minutes even when travel_rings is left out")
def test_an_empty_ring_minutes_is_fine_when_travel_rings_is_left_out(ctx, monkeypatch):
    """As plan_medevac takes egress_modes=[] with egress left out: no ring
    times and no rings say the same thing, and None is already accepted."""
    stubs = Stubs(monkeypatch)
    result = plan_initial_response(ctx, *PLS, include=["hasty"], ring_minutes=[])
    assert statuses(result) == {name: "ok" if name == "hasty" else "skipped"
                                for name in SECTIONS}
    assert stubs.calls["travel_rings"] == [] and "rings" not in result["staging"]


@pytest.mark.parametrize("minutes, overrides, radius", [
    ([15], None, 500.0),                    # 1,500 m at top speed, less the buffer
    ([30, 15], None, 2000.0),
    ([60], {"speed_factor": 0.5}, 2000.0),  # half pace: 3,000 m in the hour
    ([60, 120], None, 4499.0),              # past the largest prepare_area stages
    # Open ground at twice Tobler's off-path pace beats a trail: 3,000 m in
    # 15 minutes, so 2,000 m once the buffer is taken off.
    ([15], {"off_way_factor": 2.0}, 2000.0),
    # Grass at three times the default: 0.6 x 3 of Tobler's pace, 2,700 m.
    ([15], {"vegetation_factors": {"Herb": 3.0}}, 1750.0),
    ([1e307], None, 4499.0),                # an absurd time stages the cap, not a crash
    ([60], {"speed_factor": 1e308}, 4499.0),
])
def test_the_rings_ground_is_sized_for_the_longest_ring(ctx, monkeypatch, minutes, overrides,
                                                        radius):
    stubs = Stubs(monkeypatch)
    result = plan_initial_response(ctx, *PLS, include=["travel_rings"], ring_minutes=minutes,
                                   overrides={"travel_rings": overrides} if overrides else None)
    assert stubs.radii() == {10.0: radius}
    assert result["staging"]["rings"]["radius_m"] == radius
    # Ascending and once each, as the feature reads them.
    assert stubs.calls["travel_rings"][0]["minutes"] == sorted(float(m) for m in minutes)


def test_overrides_reach_every_call_of_their_section_and_its_detail_call(ctx, monkeypatch):
    stubs = Stubs(monkeypatch)
    result = plan_initial_response(
        ctx, *PLS, hasty_limit=0, lz_limit=2, medical_per_category=1,
        overrides={"hasty": {"min_length_m": 593, "corridor_widths_m": {"waterway": 60}},
                   "parking": {"max_distance_m": 800},
                   "travel_rings": {"speed_factor": 0.5, "cliff_slope_deg": 40},
                   "lz": {"max_slope_deg": 6.0},
                   "medical": {"search_radius_km": 150.0},
                   "export": {"label_prefix": "H", "include_area": False}})
    handles = result["handles"]
    hasty_call = stubs.calls["hasty"][0]
    assert hasty_call["limit"] == 0 and hasty_call["min_length_m"] == 593
    # The tool takes the model; the detail_call keeps the mapping, as JSON.
    assert isinstance(hasty_call["corridor_widths_m"], hasty_feature.CorridorWidths)
    assert hasty_call["corridor_widths_m"].as_overrides() == {"waterway": 60.0}
    assert result["sections"]["hasty"]["detail_call"]["arguments"] == {
        "handle": handles["search"], "limit": 0, "include": ["trail", "road"],
        "min_length_m": 593, "corridor_widths_m": {"waterway": 60.0}}
    assert stubs.calls["parking"] == [{"handle": handles["search"], "limit": 5,
                                       "max_distance_m": 800}]
    assert stubs.calls["travel_rings"][0]["speed_factor"] == 0.5
    assert stubs.calls["travel_rings"][0]["cliff_slope_deg"] == 40
    # The answer's own record of the pace and cell the rings were walked at.
    rings = result["sections"]["travel_rings"]
    assert rings["speed_factor"] == 0.5 and rings["resolution_m"] == 10.0
    assert rings["detail_call"]["arguments"]["speed_factor"] == 0.5
    assert stubs.calls["lz"][0]["max_slope_deg"] == 6.0 and stubs.calls["lz"][0]["limit"] == 2
    assert stubs.calls["medical"][0]["search_radius_km"] == 150.0
    assert stubs.calls["export"] == [{"handle": handles["search"], "set_id": SET_ID,
                                      "include_area": False, "label_prefix": "H"}]
    export = result["sections"]["export"]
    assert export["area_included"] is False
    assert export["labels"][0]["label"] == "H01 Spring Ridge Trail"
    json.dumps(result)


@pytest.mark.parametrize("overrides, message", [
    ({"egress": {"mode": "walk_out"}}, r"no section of plan_initial_response: \['egress'\]"),
    ({"hasty": {"max_distnace_m": 800}},
     r"\['max_distnace_m'\], which hasty_candidates does not take"),
    ({"hasty": {"handle": "aoi-0123456789abcdef"}}, "cannot set 'handle'"),
    ({"hasty": {"limit": 5}}, "cannot set 'limit': plan_initial_response sets it from its "
                              "own argument hasty_limit"),
    ({"parking": {"limit": 5}}, "argument parking_limit"),
    ({"travel_rings": {"minutes": [10]}}, "argument ring_minutes"),
    ({"travel_rings": {"center_latitude": 1.0}}, "cannot set 'center_latitude'"),
    ({"lz": {"limit": 9}}, "argument lz_limit"),
    ({"medical": {"per_category": 5}}, "argument medical_per_category"),
    ({"export": {"set_id": "hasty-000000"}}, "cannot set 'set_id': the scenario exports"),
    ({"export": {"candidate_ids": ["hasty-000000.1"]}}, "cannot set 'candidate_ids'"),
    ({"export": {"handle": "aoi-0123456789abcdef"}}, "cannot set 'handle'"),
    ({"export": {"label": "H"}}, r"caltopo_export does not take here; it accepts "
                                 r"\['label_prefix', 'resource_type', 'include_area', "
                                 r"'include_markers'\]"),
    ({"hasty": {"corridor_widths_m": {"waterway": -5}}}, "not a set of corridor widths"),
    ({"hasty": {"corridor_widths_m": {"drainage": 60}}}, "not a set of corridor widths"),
    ({"hasty": {"corridor_widths_m": {"gully": 30}}}, "not a set of corridor widths"),
    ({"hasty": {"corridor_widths_m": 60}}, "must be a mapping of corridor class"),
    ({"travel_rings": {"speed_factor": 0}}, r"\['speed_factor'\] must be a number above zero"),
    ({"travel_rings": {"speed_factor": "fast"}}, "must be a number above zero"),
    ({"travel_rings": {"off_way_factor": 0}}, r"\['travel_rings'\] cannot be walked: "
                                              r"off_way_factor \(0\) must be above zero"),
    ({"travel_rings": {"vegetation_factors": {"Moss": 2.0}}}, "not a LANDFIRE lifeform"),
    ({"travel_rings": {"cliff_slope_deg": 95}}, "cliff_slope_deg"),
    ({"export": {"label_prefix": "!!"}}, r"overrides\['export'\]: label_prefix \('!!'\) "
                                         "must be 1 to 6 letters or digits"),
    ({"parking": []}, "must be a mapping"),
])
def test_bad_overrides_are_refused_before_any_work(ctx, monkeypatch, overrides, message):
    stubs = Stubs(monkeypatch)
    with pytest.raises(ValueError, match=message):
        plan_initial_response(ctx, *PLS, overrides=overrides)
    assert stubs.nothing_called()


@pytest.mark.parametrize("arguments, message", [
    ({"latitude": 95.0}, "latitude"),
    ({"longitude": float("nan")}, "longitude"),
    ({"include": ["egress"]}, "no section of this scenario"),
    ({"include": []}, "empty"),
    ({"include": "hasty"}, "must be a list"),
    ({"include": ["export"]}, "include names only export, which draws what the other "
                              "sections find"),
    ({"radius_m": 5000.0}, "radius must be above 0 and under 4,500 m"),
    ({"radius_m": 4500.0}, "radius must be above 0 and under 4,500 m"),
    ({"radius": 3, "radius_unit": "mi"}, r"radius must be above 0 .* got 4828"),
    ({"radius_m": 0}, "radius must be above 0"),
    ({"radius": 1, "radius_m": 1000}, "give the radius once"),
    ({"radius": 1, "radius_unit": "furlong"}, "radius_unit 'furlong' is not one of"),
    ({"radius_unit": "furlong"}, "radius_unit 'furlong' is not one of"),
    ({"ring_minutes": []}, "ring_minutes: minutes is empty"),
    # Asked for, the rings need a time; a bad time is refused even when
    # the rings are left out - a wrong value is wrong either way.
    ({"ring_minutes": [], "include": ["travel_rings"]}, "ring_minutes"),
    ({"ring_minutes": [15, 0], "include": ["hasty"]}, "ring_minutes: a ring time"),
    ({"ring_minutes": [15, 0]}, "ring_minutes: a ring time"),
    ({"ring_minutes": list(range(1, 10))}, "ring_minutes: 9 ring times"),
    ({"ring_minutes": "15"}, "ring_minutes must be a list"),
    ({"ring_minutes": ["soon"]}, "ring_minutes: minutes must be a list of numbers"),
    ({"hasty_limit": -1}, "hasty_limit must be 0"),
    ({"hasty_limit": True}, "hasty_limit"),
    ({"parking_limit": 2.5}, "parking_limit"),
    ({"lz_limit": -1}, "lz_limit"),
    ({"medical_per_category": 0}, "medical_per_category"),
    ({"medical_per_category": 11}, "medical_per_category"),
])
def test_bad_arguments_are_refused_before_any_work(ctx, monkeypatch, arguments, message):
    stubs = Stubs(monkeypatch)
    call = {"latitude": PLS[0], "longitude": PLS[1], **arguments}
    with pytest.raises(ValueError, match=message):
        plan_initial_response(ctx, **call)
    assert stubs.nothing_called()


# --- The export -----------------------------------------------------------------


def test_the_export_files_are_ones_caltopo_imports(ctx, monkeypatch):
    Stubs(monkeypatch)
    result = plan_initial_response(ctx, *PLS)
    drafts_path, markers_path = (Path(path) for path in result["exports"])

    drafts = read_export(drafts_path)
    assert drafts["type"] == "FeatureCollection"
    assignments = [f for f in drafts["features"] if f["properties"]["class"] == "Assignment"]
    assert [f["geometry"]["type"] for f in assignments] == [
        "LineString", "LineString", "LineString", "Polygon"]
    assert all(f["properties"]["status"] == "DRAFT" for f in assignments)
    assert "check it on the ground before issuing" in assignments[0]["properties"]["description"]
    area = [f for f in drafts["features"] if f["properties"]["class"] == "Shape"]
    assert len(area) == 1 and area[0]["properties"]["title"].startswith("Search area, 1,500 m")
    # Four drafts (the forked road is two), the area, and caltopo_export's
    # own centre: what the section's `features` counts is the file's.
    assert result["sections"]["export"]["features"] == {"Assignment": 4, "Shape": 1,
                                                        "Marker": 1}
    counted = {}
    for feature in drafts["features"]:
        kind = feature["properties"]["class"]
        counted[kind] = counted.get(kind, 0) + 1
    assert counted == result["sections"]["export"]["features"]

    collection = read_export(markers_path)
    assert collection["type"] == "FeatureCollection"
    features = collection["features"]
    for feature in features:
        assert feature["type"] == "Feature"
        assert feature["geometry"]["type"] in ("Point", "LineString")
        assert feature["properties"]["class"] in ("Marker", "Shape")
        assert feature["properties"]["title"] and feature["properties"]["description"]

    place = features[0]
    assert place["properties"]["title"] == "Place last seen"
    assert place["geometry"] == {"type": "Point", "coordinates": [PLS[1], PLS[0]]}
    assert place["properties"]["marker-color"] == ORIGIN_COLOR
    # What every distance in the file is measured from, the hospitals' road
    # start, and that nothing here is decided - in that order.
    assert place["properties"]["description"].split("\n") == [
        "Where the subject was last seen. Hasty-line, parking and landing-zone distances and "
        "bearings, and every walking ring, are measured from here.",
        "Nearest hospitals ranked by road time from here.",
        "From plan_initial_response; nothing in this file is a decision."]

    lots = [f for f in features if f["properties"].get("marker-color") == "#808080"]
    assert [f["properties"]["title"] for f in lots] == ["Trailhead lot", "parking-w2"]
    # From the same point every other marker in the file names; a fee or
    # private lot says so on the map, so it never reads as open staging,
    # and what was not checked is said on every one.
    assert lots[0]["properties"]["description"].split("\n") == [
        "parking, 140 m from the place last seen.", "Flags: fee.",
        "From OpenStreetMap; gates, hours and the road in are not checked."]
    assert lots[1]["properties"]["description"].split("\n") == [
        "parking, 380 m from the place last seen.", "Flags: access_private.",
        "From OpenStreetMap; gates, hours and the road in are not checked."]

    rings = [f for f in features if f["geometry"]["type"] == "LineString"]
    assert [f["properties"]["title"] for f in rings] == ["15 min walk", "30 min walk",
                                                         "60 min walk"]
    # Light, middle and dark: spread over the ramp, not three near shades.
    strokes = _sections.RING_STROKES
    assert [f["properties"]["stroke"] for f in rings] == [strokes[0], strokes[4], strokes[7]]
    # The ring's numbers as a team reads them: hectares, the circle of the
    # same area, and the farthest reach with its bearing.
    assert [f["properties"]["description"].split("\n")[0] for f in rings] == [
        "Ground within 15 walking minutes of the place last seen at a steady pace: 120.0 acres, "
        "as much as a circle of 618 m radius; it reaches farthest 900 m at 45 deg.",
        "Ground within 30 walking minutes of the place last seen at a steady pace: 480.0 acres, "
        "as much as a circle of 1,236 m radius; it reaches farthest 1,800 m at 45 deg.",
        "Ground within 60 walking minutes of the place last seen at a steady pace: "
        "1,080.0 acres, as much as a circle of 1,854 m radius; it reaches farthest 2,700 m at "
        "45 deg."]
    assert "of the place last seen" in rings[0]["properties"]["description"]
    assert "not time since last seen" in rings[0]["properties"]["description"]
    assert "In 2 parts; only the largest is outlined here." in rings[2]["properties"][
        "description"]
    assert "Cut at the edge of the staged ground" in rings[2]["properties"]["description"]
    assert "Cut at the edge" not in rings[0]["properties"]["description"]
    # The stub's walk saw every source: nothing to own up to.
    assert "No OpenStreetMap" not in rings[0]["properties"]["description"]

    zones = [f for f in features if f["properties"]["title"].startswith("lz_")]
    assert len(zones) == 3
    assert "from the place last seen." in zones[0]["properties"]["description"]
    hospitals = [f for f in features if f["properties"].get("role") == "hospital"]
    assert {f["properties"]["marker-color"] for f in hospitals} == {HOSPITAL_COLOR}
    assert len(features) == 1 + 2 + 3 + 3 + 2


def test_a_ring_walked_without_a_source_says_so_on_the_map(ctx, monkeypatch):
    """Overpass down while the rings' ground was staged: no trail speeds
    anyone and no river stops anyone. The section has the walk's warning;
    the line a team reads in CalTopo must say it too."""
    from sar_geo_mcp.planning.walking import NO_EVT_WARNING, NO_OSM_WARNING

    Stubs(monkeypatch)

    def without_osm(ctx, **arguments):
        payload = rings_payload(arguments)
        payload["warnings"] = [NO_OSM_WARNING, NO_EVT_WARNING]
        return payload

    monkeypatch.setattr(rings_feature, "travel_time_rings", without_osm)
    result = plan_initial_response(ctx, *PLS, include=["travel_rings", "export"])
    assert result["sections"]["travel_rings"]["warnings"] == [NO_OSM_WARNING, NO_EVT_WARNING]
    features = read_export(result["exports"][0])["features"]
    rings = [f for f in features if f["geometry"]["type"] == "LineString"]
    assert len(rings) == 3
    for ring in rings:
        text = ring["properties"]["description"]
        assert _sections.RING_GROUND_NOTES[NO_OSM_WARNING] in text
        assert _sections.RING_GROUND_NOTES[NO_EVT_WARNING] in text


@pytest.mark.xfail(strict=True, reason="PR #4 review B32: summary.travel_rings flags "
                   "reaches_edge but not a walk with no OpenStreetMap or vegetation")
def test_the_summary_says_when_the_rings_were_walked_without_a_source(ctx, monkeypatch):
    """The summary is the facts to read first. A ring walked with no
    OpenStreetMap ignores trails and unbridged rivers, and one with no
    vegetation ignores brush: the summary's ring block, or a top-level
    warning, must say so next to the areas and reaches - not only the
    section's warnings and the CalTopo line."""
    from sar_geo_mcp.planning.walking import NO_EVT_WARNING, NO_OSM_WARNING

    Stubs(monkeypatch)

    def without_sources(ctx, **arguments):
        payload = rings_payload(arguments)
        payload["warnings"] = [NO_OSM_WARNING, NO_EVT_WARNING]
        return payload

    monkeypatch.setattr(rings_feature, "travel_time_rings", without_sources)
    result = plan_initial_response(ctx, *PLS, include=["travel_rings"])
    assert result["sections"]["travel_rings"]["warnings"] == [NO_OSM_WARNING, NO_EVT_WARNING]
    said = (json.dumps(result["summary"]["travel_rings"]) + json.dumps(result["warnings"])).lower()
    assert "openstreetmap" in said or "osm" in said
    assert "vegetation" in said or "landfire" in said


# --- A second call on the same ground -------------------------------------------


def _features(path) -> list[dict]:
    return read_export(path)["features"]


def _titles(path, kind=None) -> list[str]:
    return [feature["properties"]["title"] for feature in _features(path)
            if kind is None or feature["properties"]["class"] == kind]


def _two_calls_on_one_ground(ctx, monkeypatch):
    """A full call with its drafts labelled DIVA, then a narrower one on the
    same ground (hasty and one lot, the default labels): the two answers'
    export sections, the first's checked as it was written."""
    Stubs(monkeypatch)
    first = plan_initial_response(ctx, *PLS, overrides={"export": {"label_prefix": "DIVA"}})
    before = first["sections"]["export"]
    assert _titles(before["path"], "Assignment") == [
        "DIVA01 Spring Ridge Trail", "DIVA02a Skyline Blvd", "DIVA02b Skyline Blvd", "DIVA03"]
    assert len(_features(before["markers_path"])) == 11

    second = plan_initial_response(ctx, *PLS, include=["hasty", "parking", "export"],
                                   parking_limit=1)
    assert second["handles"]["search"] == first["handles"]["search"]
    return first, second


def test_a_later_call_on_the_same_ground_answers_with_files_holding_what_it_drew(
        ctx, monkeypatch):
    """The files a later call on the same ground names hold what that call
    drew - its labels, its one lot - never the earlier call's. Whether the
    earlier answer's files survive it is B8's (the test below), so nothing
    here pins the paths as shared or the folder's contents."""
    _first, second = _two_calls_on_one_ground(ctx, monkeypatch)
    after = second["sections"]["export"]
    assert _titles(after["path"], "Assignment") == [
        "U01 Spring Ridge Trail", "U02a Skyline Blvd", "U02b Skyline Blvd", "U03"]
    assert _titles(after["markers_path"]) == ["Place last seen", "Trailhead lot"]
    assert after["markers"] == {"Marker": 2}
    assert sorted(second["exports"]) == sorted([after["path"], after["markers_path"]])
    # It wrote its own markers file, so none is named as an earlier call's
    # left in its place: that would call this answer's own file not its own.
    assert not any("left as it was" in warning for warning in second["warnings"])


@pytest.mark.xfail(strict=True, reason="PR #4 review B8: A later scenario call on the same "
                   "ground silently overwrites the export files (drafts and markers) an earlier "
                   "answer pointed to")
def test_a_later_call_on_the_same_ground_leaves_the_earlier_answer_s_files(ctx, monkeypatch):
    """plan_medevac's B8 test, for both of this scenario's files: the first
    answer's drafts and markers are what a coordinator imports, perhaps
    later, and a second, narrower call must not swap them for its own. (If
    B8 is fixed by replacing the files and saying so instead, replace this
    test as the medevac one says.)"""
    first, _second = _two_calls_on_one_ground(ctx, monkeypatch)
    before = first["sections"]["export"]
    assert _titles(before["path"], "Assignment") == [
        "DIVA01 Spring Ridge Trail", "DIVA02a Skyline Blvd", "DIVA02b Skyline Blvd", "DIVA03"]
    assert len(_features(before["markers_path"])) == 11


@pytest.mark.parametrize("include, fail", [
    (["hasty", "export"], ()),                               # nothing else asked for
    (None, ("parking", "travel_rings", "lz", "medical")),    # everything else failed
], ids=["hasty-and-export", "markers-sources-failed"])
def test_a_later_call_drawing_no_markers_leaves_no_stale_markers_file(ctx, monkeypatch,
                                                                     include, fail):
    """A later call on the same ground that draws no markers file must not
    silently leave the earlier call's beside its fresh drafts: a planner
    who imports the folder gets the first call's landing zones, hospitals
    and rings - walked for other times, here - as if they were current.
    The old file is removed, or a warning names it."""
    stubs = Stubs(monkeypatch)
    first = plan_initial_response(ctx, *PLS, ring_minutes=[120, 240])
    stale = Path(first["sections"]["export"]["markers_path"])
    assert "120 min walk" in _titles(stale)

    stubs.fail.update(fail)
    second = plan_initial_response(ctx, *PLS, include=include)
    export = second["sections"]["export"]
    assert export["path"] is not None and export["markers_path"] is None
    assert Path(export["path"]).parent == stale.parent        # the same exports folder
    if any(stale.name in warning for warning in [*second["warnings"], *export["warnings"]]):
        return                                                # left, but named
    assert not stale.exists()
    assert sorted(path.name for path in stale.parent.iterdir()) == sorted(
        Path(path).name for path in second["exports"])


def test_a_later_call_with_export_left_out_names_no_earlier_file(ctx, monkeypatch):
    """As the description says: left out of `include`, the export writes
    nothing and names nothing already in the folder - the call makes no
    claim about any file, so only a call that asked for one is told an
    earlier call's is where its own would be."""
    Stubs(monkeypatch)
    first = plan_initial_response(ctx, *PLS)
    second = plan_initial_response(ctx, *PLS, include=["hasty", "parking"])
    assert second["handles"]["search"] == first["handles"]["search"]
    assert second["exports"] == []
    assert Path(first["sections"]["export"]["markers_path"]).is_file()
    assert not any("left as it was" in warning for warning in second["warnings"])


# --- Through MCP ------------------------------------------------------------------


@pytest.mark.anyio
async def test_plan_initial_response_is_registered_as_a_scenario_and_answers(monkeypatch,
                                                                             tmp_path):
    pytest.importorskip("fastmcp")
    from fastmcp import Client
    from fastmcp.exceptions import ToolError

    from sar_geo_mcp.server import build_server

    Stubs(monkeypatch)
    server = build_server(CaltopoConfig(cache_dir=tmp_path / "cache"))
    async with Client(server) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
        tool = tools["plan_initial_response"]
        assert tool.meta["fastmcp"]["tags"] == ["scenario"]
        assert "NO VERDICT" in tool.description and "detail_call" in tool.description
        properties = tool.inputSchema["properties"]
        assert properties["include"]["anyOf"][0]["items"]["enum"] == list(SECTIONS)
        assert properties["radius_unit"]["enum"] == ["m", "km", "ft", "yd", "mi"]
        assert properties["ring_minutes"]["default"] == [15.0, 30.0, 60.0]

        answer = (await client.call_tool("plan_initial_response", {
            "latitude": PLS[0], "longitude": PLS[1], "include": ["hasty", "parking"],
            "overrides": {"hasty": {"corridor_widths_m": {"waterway": 60}}}})).data
        assert answer["status"] == "ok"
        assert answer["sections"]["travel_rings"]["status"] == "skipped"
        assert answer["summary"]["hasty"]["count"] == 3
        assert answer["sections"]["hasty"]["detail_call"]["arguments"][
            "corridor_widths_m"] == {"waterway": 60.0}

        with pytest.raises(ToolError):
            await client.call_tool("plan_initial_response", {
                "latitude": PLS[0], "longitude": PLS[1], "include": ["egress"]})


# --- Against the real features ----------------------------------------------------


def test_the_sections_read_what_the_real_features_return(ctx, monkeypatch):
    """Flat ground written to disk (`_stub_staging`) with a named trail
    through the place last seen and a parking lot beside it, and the real
    staging path, hasty_candidates, parking_candidates, travel_time_rings
    and caltopo_export over it. Holds the compact sections to the features'
    actual payloads, which the stubs above only imitate."""
    from test_parking import _site, _square_3857
    from test_planning_tools import _stub_staging, _trail_through

    from sar_geo_mcp import staging as staging_module

    _stub_staging(monkeypatch, routes=[_trail_through(PLS)], width=120)
    fake_stage = staging_module.stage_terrain

    def with_a_lot(subject, map_id, out_dir, **kwargs):
        staged = fake_stage(subject, map_id, out_dir, **kwargs)
        osm_path = Path(out_dir) / "osm.json"
        osm = json.loads(osm_path.read_text(encoding="utf-8"))
        osm["parking"] = [_site("parking-w1", "parking",
                                _square_3857(PLS[1] + 0.002, PLS[0], 40),
                                name="Trailhead lot", surface="gravel")]
        osm_path.write_text(json.dumps(osm), encoding="utf-8")
        return staged

    monkeypatch.setattr(staging_module, "stage_terrain", with_a_lot)

    # The real payloads, kept as the features returned them, so the compact
    # sections are held to the keys the features actually give.
    real = {}
    for module, name in ((hasty_feature, "hasty_candidates"),
                         (parking_feature, "parking_candidates"),
                         (rings_feature, "travel_time_rings")):
        function = getattr(module, name)

        def keep(ctx, _function=function, _name=name, **arguments):
            real[_name] = copy.deepcopy(_function(ctx, **arguments))
            return real[_name]

        monkeypatch.setattr(module, name, keep)

    result = plan_initial_response(ctx, *PLS, radius_m=400, ring_minutes=[10, 20],
                                   include=["hasty", "parking", "travel_rings", "export"])

    assert result["status"] == "ok", {name: entry["error"]
                                      for name, entry in result["sections"].items()}
    hasty = result["sections"]["hasty"]
    lines = [c for c in real["hasty_candidates"]["candidates"] if c["kind"] == "line"]
    assert lines and lines[0]["name"] == "Long Trail route_0"
    line_facts = set(_sections.HASTY_FACTS) - {"width_m", "area_acres"}
    assert line_facts <= set(lines[0]["facts"])
    assert {"by_axis", "by_kind", "total_before_limit", "truncated_to",
            "dropped_beyond_max_distance", "existing_assignments"} <= set(
        real["hasty_candidates"]["summary"])
    assert {"set_id", "created_at", "reused"} <= set(real["hasty_candidates"])
    compact = next(c for c in hasty["candidates"] if c["id"] == lines[0]["id"])
    assert compact["travel_minutes_from_center"] == lines[0]["facts"][
        "travel_minutes_from_center"]
    assert result["summary"]["hasty"]["by_axis"]["trail"] >= 1

    assert [lot["name"] for lot in result["sections"]["parking"]["candidates"]] == [
        "Trailhead lot"]
    ring_keys = set(real["travel_time_rings"]["rings"][0])
    assert set(_sections.RING_FIELDS) <= ring_keys and "outline" in ring_keys
    assert {"resolution_m", "speed_factor", "ground"} <= set(real["travel_time_rings"])
    assert result["sections"]["travel_rings"]["ground_facts"] == real["travel_time_rings"][
        "ground"]
    assert [ring["minutes"] for ring in result["sections"]["travel_rings"]["rings"]] == [
        10.0, 20.0]
    # 20 minutes at top speed is 2,000 m; less the 1,000 m buffer.
    assert result["staging"]["rings"]["radius_m"] == 1000.0
    assert all(math.isfinite(ring["area_acres"])
               for ring in result["sections"]["travel_rings"]["rings"])
    assert geometry_types_in(result) == {"Point"}

    export = result["sections"]["export"]
    assert export["assignments"] == len(real["hasty_candidates"]["candidates"])
    drafts = read_export(export["path"])
    assert {f["properties"]["class"] for f in drafts["features"]} >= {"Assignment", "Shape"}
    markers = read_export(export["markers_path"])
    assert [f["geometry"]["type"] for f in markers["features"]].count("LineString") == 2


def test_hasty_truncated_to_says_the_limit_when_it_cut_the_list(ctx, monkeypatch):
    """Two trails and hasty_limit=1, through the real feature: one kept of
    two, and the section says the limit cut it."""
    from test_planning_tools import _stub_staging, _trail_through

    _stub_staging(monkeypatch, routes=[_trail_through(PLS),
                                       _trail_through(PLS, offset_m=150, rid="route_1")],
                  width=120)
    result = plan_initial_response(ctx, *PLS, radius_m=400, include=["hasty"], hasty_limit=1)
    hasty = result["sections"]["hasty"]
    assert [candidate["name"] for candidate in hasty["candidates"]] == ["Long Trail route_0"]
    assert hasty["total_before_limit"] == 2 and hasty["truncated_to"] == 1
    assert result["summary"]["hasty"]["count"] == 1
    assert result["summary"]["hasty"]["total_before_limit"] == 2


@pytest.mark.xfail(strict=True, reason="PR #4 review B31: hasty.truncated_to is the limit even "
                   "when nothing was dropped; parking's is null in the same answer")
def test_truncated_to_is_null_in_every_section_when_the_limit_cut_nothing(ctx, monkeypatch):
    """One trail and the default limit of 20, through the real hasty and
    parking features: nothing was cut, so `truncated_to` is null - in hasty
    as in parking, where the same field means the same thing."""
    from test_planning_tools import _stub_staging, _trail_through

    _stub_staging(monkeypatch, routes=[_trail_through(PLS)], width=120)
    result = plan_initial_response(ctx, *PLS, radius_m=400, include=["hasty", "parking"])
    hasty = result["sections"]["hasty"]
    assert len(hasty["candidates"]) == hasty["total_before_limit"] == 1
    assert hasty["truncated_to"] is None
    for name in ("hasty", "parking"):
        entry = result["sections"][name]
        cut = len(entry["candidates"]) < entry["total_before_limit"]
        assert (entry["truncated_to"] is not None) == cut, name


def test_hasty_minutes_say_their_pace_when_the_rings_walk_at_another(ctx, monkeypatch):
    """The real hasty and rings features over flat ground, one trail 300 m
    north. At half pace (speed_factor 0.5, the override the refusal text
    itself suggests) the 12-minute ring stops short of the trail, while the
    trail's travel_minutes_from_center - walked at the default pace - is
    under 12. The answer must say the hasty minutes are not at the rings'
    pace, naming travel_minutes_from_center."""
    from test_planning_tools import _stub_staging, _trail_through

    _stub_staging(monkeypatch, routes=[_trail_through(PLS, offset_m=300)], width=160)
    result = plan_initial_response(ctx, *PLS, radius_m=600, ring_minutes=[8, 12],
                                   include=["hasty", "travel_rings"],
                                   overrides={"travel_rings": {"speed_factor": 0.5}})
    assert result["status"] == "ok"
    trail = next(candidate for candidate in result["sections"]["hasty"]["candidates"]
                 if candidate["axis"] == "trail")
    rings = result["sections"]["travel_rings"]
    assert rings["speed_factor"] == 0.5
    twelve = next(ring for ring in rings["rings"] if ring["minutes"] == 12)
    # The contradiction, as the answer states it today.
    assert twelve["reach_by_direction_m"]["N"] < trail["distance_from_center_m"]
    assert trail["travel_minutes_from_center"] < 12
    said = [*result["warnings"], *result["sections"]["hasty"]["warnings"]]
    assert any("travel_minutes_from_center" in warning for warning in said)


def test_a_rings_pace_set_off_the_trails_is_named_beside_the_hasty_minutes(ctx, monkeypatch):
    """B14 for a pace that is not one factor on every step: the warning names
    each keyword and value, and says the order may differ rather than offer
    a division that would be wrong."""
    Stubs(monkeypatch)
    result = plan_initial_response(
        ctx, *PLS, include=["hasty", "travel_rings"],
        overrides={"travel_rings": {"off_way_factor": 0.3,
                                    "vegetation_factors": {"Shrub": 0.2}}})
    [warning] = [warning for warning in result["warnings"]
                 if "travel_minutes_from_center" in warning]
    assert '(off_way_factor 0.3, vegetation_factors {"Shrub": 0.2})' in warning
    assert "another order" in warning and "divided by" not in warning
    assert "walked at all" not in warning


@pytest.mark.parametrize("ring_overrides", [
    {"cliff_slope_deg": 20},
    {"cliff_slope_deg": 20, "speed_factor": 0.5},
], ids=["cliff-alone", "cliff-and-speed"])
def test_a_rings_cliff_limit_says_a_null_minute_may_be_reached(ctx, monkeypatch,
                                                                ring_overrides):
    """cliff_slope_deg is no pace: it moves which ground is closed, so a hasty
    line null on the default walk may be reached on the rings' walk, or the
    reverse. "Another order" alone would let a null read as ground the
    subject could not reach at their own cliff limit (B14, review)."""
    Stubs(monkeypatch)
    result = plan_initial_response(ctx, *PLS, include=["hasty", "travel_rings"],
                                   overrides={"travel_rings": ring_overrides})
    [warning] = [warning for warning in result["warnings"]
                 if "travel_minutes_from_center" in warning]
    assert "cliff_slope_deg 20" in warning and "another order" in warning
    assert "divided by" not in warning
    assert "changes which ground can be walked at all" in warning
    assert "null here may be reached on the rings' walk" in warning


@pytest.mark.parametrize("include, overrides, fail", [
    (["hasty", "travel_rings"], None, ()),
    (["hasty", "travel_rings"], {"travel_rings": {"speed_factor": 1}}, ()),
    (["hasty", "travel_rings"], {"travel_rings": {"off_way_factor": 0.6,
                                                  "cliff_slope_deg": 45}}, ()),
    (["hasty"], {"travel_rings": {"speed_factor": 0.5}}, ()),
    (["hasty", "travel_rings"], {"travel_rings": {"speed_factor": 0.5}}, ("hasty",)),
], ids=["no-override", "speed-factor-1", "default-values", "rings-left-out", "hasty-failed"])
def test_hasty_minutes_say_nothing_of_pace_when_none_differs(ctx, monkeypatch, include,
                                                              overrides, fail):
    """The other side of B14: rings walked on the default model, however
    that was asked for, walk the hasty minutes' own pace; rings left out
    leave their overrides to `unused_overrides`; a hasty section that did
    not answer has no minutes to warn of."""
    Stubs(monkeypatch, fail=fail)
    result = plan_initial_response(ctx, *PLS, include=include, overrides=overrides)
    assert not any("travel_minutes_from_center" in warning for warning in result["warnings"])


@pytest.mark.parametrize("route_error, ranking, ranked_by", [
    (None, RANKED_BY_ROAD, "road_time"),
    (httpx.ConnectError("OSRM unreachable"), RANKED_STRAIGHT_ROUTER_DOWN, "straight_line"),
], ids=["routed", "router-down"])
def test_the_place_last_seen_says_how_the_hospitals_were_ranked(ctx, monkeypatch,
                                                                route_error, ranking,
                                                                ranked_by):
    """B21 in this scenario's markers file and summary: with the road router
    down the place does not claim road-time ranking beside hospitals
    numbered "#1", and the nearest trauma center says how it is nearest."""
    stubs = Stubs(monkeypatch)
    stubs.services.route_error = route_error
    result = plan_initial_response(ctx, *PLS)
    markers_path = Path(result["sections"]["export"]["markers_path"])
    place = read_export(markers_path)["features"][0]
    assert place["properties"]["title"] == "Place last seen"
    lines = place["properties"]["description"].split("\n")
    assert [line for line in lines if "hospitals ranked" in line] == [ranking]
    assert result["summary"]["medical"]["nearest_trauma_center"]["ranked_by"] == ranked_by


def _stage_without_osm(monkeypatch):
    """The real staging path over flat ground, as during an Overpass outage:
    no osm.json on the staged ground."""
    from test_planning_tools import _stub_staging

    from sar_geo_mcp import staging as staging_module

    _stub_staging(monkeypatch, width=120)
    fake_stage = staging_module.stage_terrain

    def without_osm(subject, map_id, out_dir, **kwargs):
        staged = fake_stage(subject, map_id, out_dir, **kwargs)
        (Path(out_dir) / "osm.json").unlink()
        return staged

    monkeypatch.setattr(staging_module, "stage_terrain", without_osm)


def test_ground_with_osm_and_no_lots_or_lines_summarises_as_none(ctx, monkeypatch):
    """The other side of B19: OpenStreetMap was read and maps no lot and no
    trail or road here - a real "none", which the summary says as zero."""
    from test_planning_tools import _stub_staging

    _stub_staging(monkeypatch, routes=[], width=120)
    result = plan_initial_response(ctx, *PLS, radius_m=400, include=["hasty", "parking"])
    assert result["status"] == "ok"
    assert result["staging"]["search"]["osm_available"] is True
    assert result["summary"]["parking"] == {"count": 0, "total_before_limit": 0,
                                            "nearest": None}
    assert result["summary"]["hasty"] == {"count": 0, "by_kind": {}, "by_axis": {},
                                          "total_before_limit": 0}
    assert result["warnings"] == []


def test_parking_unknown_for_want_of_osm_is_not_summarised_as_none(ctx, monkeypatch):
    """Ground staged during an OpenStreetMap outage: parking is unknown, not
    absent, and the summary - the facts read first - must not be the
    "none found" block a ground with OpenStreetMap and no lots gives."""
    _stage_without_osm(monkeypatch)
    result = plan_initial_response(ctx, *PLS, radius_m=400, include=["parking"])
    assert result["staging"]["search"]["osm_available"] is False
    assert any("unknown rather than absent" in warning
               for warning in result["sections"]["parking"]["warnings"])
    assert result["summary"]["parking"] != {"count": 0, "total_before_limit": 0,
                                            "nearest": None}


def test_hasty_lines_unknown_for_want_of_osm_are_not_summarised_as_none(ctx, monkeypatch):
    """The same outage for hasty: no trail or road can be seen, and by_axis
    with no trail key must not read as "no trails here"."""
    _stage_without_osm(monkeypatch)
    result = plan_initial_response(ctx, *PLS, radius_m=400, include=["hasty"])
    assert result["staging"]["search"]["osm_available"] is False
    assert result["sections"]["hasty"]["status"] == "ok"
    assert result["summary"]["hasty"] != {"count": 0, "by_kind": {}, "by_axis": {},
                                          "total_before_limit": 0}


def test_what_the_outage_leaves_unknown_is_null_in_the_section_and_the_summary(ctx,
                                                                              monkeypatch):
    """The real hasty and parking features on ground staged with no
    osm.json: parking's counts are null, and so is each axis OpenStreetMap
    gives - trail, road, waterway - and the lines, in the section and the
    summary alike. Every axis the hasty list offers comes from OpenStreetMap,
    so nothing asked for could be counted: the count is null, not 0."""
    _stage_without_osm(monkeypatch)
    result = plan_initial_response(ctx, *PLS, radius_m=400, include=["hasty", "parking"],
                                   overrides={"hasty": {"include": None}})
    assert result["status"] == "ok"
    parking = result["sections"]["parking"]
    assert parking["candidates"] == [] and parking["total_before_limit"] is None
    assert result["summary"]["parking"] == {"count": None, "total_before_limit": None,
                                            "nearest": None}
    hasty = result["sections"]["hasty"]
    assert {axis: hasty["by_axis"][axis] for axis in ("trail", "road", "waterway")} == {
        "trail": None, "road": None, "waterway": None}
    assert hasty["by_kind"]["line"] is None
    summary = result["summary"]["hasty"]
    assert summary["by_axis"] == hasty["by_axis"] and summary["by_kind"] == hasty["by_kind"]
    assert summary["count"] is None and summary["total_before_limit"] is None


def test_a_hasty_axis_not_asked_for_is_not_made_unknown_by_the_outage(ctx, monkeypatch):
    """`include` names trail alone: road and waterway were never asked for,
    so the outage leaves only the trail unknown."""
    _stage_without_osm(monkeypatch)
    result = plan_initial_response(ctx, *PLS, radius_m=400, include=["hasty"],
                                   overrides={"hasty": {"include": ["trail"]}})
    hasty = result["sections"]["hasty"]
    assert hasty["status"] == "ok"
    assert hasty["by_axis"]["trail"] is None and hasty["by_kind"]["line"] is None
    assert not {"road", "waterway"} & set(hasty["by_axis"])
    assert result["summary"]["hasty"]["by_axis"] == hasty["by_axis"]


@pytest.mark.parametrize("include, kind", [(["trail", "road"], "line"),
                                           (["waterway"], "corridor")])
def test_hasty_asked_only_for_what_the_outage_hides_counts_nothing(ctx, monkeypatch, include,
                                                                   kind):
    """Every axis asked for comes from OpenStreetMap and the ground has
    none: no count could be made, so the summary's count and total are
    null - never the 0 a ground read and holding no trail gives - and so
    is the kind, corridors included when waterway was the only one."""
    _stage_without_osm(monkeypatch)
    result = plan_initial_response(ctx, *PLS, radius_m=400, include=["hasty"],
                                   overrides={"hasty": {"include": include}})
    hasty = result["sections"]["hasty"]
    assert hasty["status"] == "ok" and hasty["candidates"] == []
    assert hasty["total_before_limit"] is None
    assert result["summary"]["hasty"] == {"count": None, "by_kind": {kind: None},
                                          "by_axis": dict.fromkeys(include),
                                          "total_before_limit": None}
