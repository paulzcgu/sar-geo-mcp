"""debrief_assignment: the ground, the coverage and the terrain features of
one CalTopo assignment in one call.

No network. Staging and the three tool functions are stubbed with
monkeypatch where they live (`components.staging.prepare_terrain`,
`components.ground.terrain_metrics`, `features.coverage.coverage_for`,
`components.ground.identify_features`), which is where the scenario
reaches them; the staging stub writes the two files an assignment handle
is read back by. One test at the end runs the real terrain_metrics,
coverage_for and identify_features over a valley written to disk, to hold
the compact sections to the payloads the tools actually return. The shared
machinery has its own tests in `test_scenarios_sections.py`.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import get_args

import pytest
from pyproj import Transformer
from test_scenarios_sections import geometry_types_in

from sar_geo_mcp.components import ground as ground_component
from sar_geo_mcp.components import staging as staging_component
from sar_geo_mcp.config import CaltopoConfig
from sar_geo_mcp.context import ServerContext
from sar_geo_mcp.features import coverage as coverage_feature
from sar_geo_mcp.scenarios import _sections
from sar_geo_mcp.scenarios.debrief import (
    DEFAULT_FEATURE_KINDS,
    FeatureKind,
    debrief_assignment,
)
from sar_geo_mcp.terrain.features import REGISTRY

MAP_ID, LABEL = "MAP1234", "AB"
HANDLE = f"{MAP_ID}_{LABEL}"
TRACKS = ["101"]
#: The gap's centre, El Corte de Madera.
GAP_CENTRE = (37.4022, -122.3164)
_TO_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

CURRENT = {"current": True, "checked": True, "fingerprint": "f" * 32,
           "live_fingerprint": "f" * 32,
           "provenance": {"fetched_at": 1.0, "age_seconds": 0.0, "source": "public"}}


def _wgs84_square(lon, lat, half=0.001):
    return {"type": "Polygon", "coordinates": [[
        [lon - half, lat - half], [lon + half, lat - half], [lon + half, lat + half],
        [lon - half, lat + half], [lon - half, lat - half]]]}


def _mercator_square(lat, lon, half_m=40.0):
    """A gap outline as coverage_for gives it: in Web Mercator metres, the
    coverage grid's own coordinates."""
    x, y = _TO_3857.transform(lon, lat)
    return {"type": "Polygon", "coordinates": [[
        [x - half_m, y - half_m], [x + half_m, y - half_m], [x + half_m, y + half_m],
        [x - half_m, y + half_m], [x - half_m, y - half_m]]]}


def _line(lon, lat):
    return {"type": "LineString", "coordinates": [[lon, lat], [lon + 0.002, lat + 0.001]]}


LON, LAT = GAP_CENTRE[1], GAP_CENTRE[0]
DRAINAGE = {"id": "drainage_0", "length_m": 410.0, "full_length_m": 900.0,
            "contributing_area_m2": 250000.0, "depth_m": {"left": 6.0, "right": 4.5},
            "sidewall_slope_deg": {"left": 30.0, "right": 26.0}, "transect_count": 12,
            "any_transect_truncated": False,
            "method": "bank-top transect depth: perpendicular rays at 25 m intervals"}


def metrics_payload(staleness=None):
    """What terrain_metrics returns: the measured blocks, every shape with its
    outline, the routes as lines - which the section must leave out."""
    return {
        "schema_version": "0.4.0", "assignment_label": LABEL,
        "elevation": {"min_m": 300.0, "max_m": 520.0, "relief_m": 220.0},
        "slope": {"mean_deg": 18.0, "p90_deg": 31.0, "max_deg": 52.0,
                  "distribution": {"0_15": 0.4, "15_27": 0.35, "27_35": 0.15, "35_45": 0.08,
                                   "45_plus": 0.02}},
        "aspect": {"dominant": "NW", "distribution": {"NW": 0.4, "N": 0.6}},
        "vegetation": {"canopy_fraction": None, "density_class": None, "method": None,
                       "lifeform_fractions": {"Shrub": 0.6, "Tree": 0.4},
                       "dominant_lifeform": "Shrub", "top_types": [("Coastal Scrub", 0.6)],
                       "type_source": "LANDFIRE LF2024 EVT, 30 m"},
        "traverse": {"baseline_rate_kmh": 2.1, "vegetation_adjusted_rate_kmh": 1.4,
                     "effective_sweep_width_m": 9.6, "sweep_time_hours_per_pass": 7.5,
                     "method": "tobler1993 + lifeform/density sweep matrix",
                     "assumptions": {"sweep_width_m_by_lifeform": {"Shrub": 8.0}}},
        "hydrology": {"drainages": [DRAINAGE], "upstream_truncated": True,
                      "assumptions": {"min_contributing_area_m2": 10000.0}},
        "landforms": {"fractions": {"ridge": 0.1, "valley": 0.2, "bench": 0.05, "slope": 0.65},
                      "cliff_band_count": 1, "assumptions": {"cliff_slope_deg": 55.0}},
        "access": {"routes": [
            {"id": "route_0", "kind": "route", "geometry": _line(LON, LAT),
             "properties": {"route_type": "TRAIL", "name": "Steep Ravine Trail"}},
            {"id": "route_1", "kind": "route", "geometry": _line(LON, LAT + 0.001),
             "properties": {"route_type": "TRAIL", "name": "Steep Ravine Trail"}},
            {"id": "route_2", "kind": "route", "geometry": _line(LON + 0.001, LAT),
             "properties": {"route_type": "ROAD", "name": None}}],
            "ingress_count": 2, "source": "OpenStreetMap via Overpass"},
        "shapes": [
            {"id": "steep_35_0", "kind": "steep_band", "geometry": _wgs84_square(LON, LAT),
             "properties": {"area_m2": 9000.0}},
            {"id": "steep_45_0", "kind": "steep_band", "geometry": _wgs84_square(LON, LAT, 0.0005),
             "properties": {"area_m2": 2000.0}},
            {"id": "drainage_0", "kind": "drainage", "geometry": _line(LON, LAT),
             "properties": {"length_m": 410.0}},
            {"id": "cliff_0", "kind": "cliff_band", "geometry": _wgs84_square(LON, LAT, 0.0002),
             "properties": {"area_m2": 300.0}}],
        "data_quality": {"dem_resolution_m": 2.9, "latitude_deg": LAT, "area_m2": 520000.0,
                         "warnings": ["vegetation panel missing; density unknown"]},
        "staleness": dict(staleness or CURRENT),
    }


def coverage_payload(arguments, *, kind="area", staleness=None):
    """What coverage_for returns: the CoverageMetrics digest, gap outlines in
    Web Mercator included - which the section must leave out."""
    area = kind == "area"
    stretch = [list(_TO_3857.transform(LON, LAT)), list(_TO_3857.transform(LON + 0.001, LAT))]
    gap_geometry = (_mercator_square(LAT, LON) if area
                    else {"type": "LineString", "coordinates": stretch})
    return {
        "schema_version": "1.1.0", "assignment_label": LABEL, "assignment_kind": kind,
        "teams": list(arguments["track_names"]), "effective_sweep_width_m": 9.6,
        "covered_fraction": 0.62, "effort_coverage": 0.71 if area else None,
        "zone_area_m2": 520000.0 if area else None, "line_length_m": None if area else 1800.0,
        "tracks": [{"name": "101 AB", "length_in_zone_m": 38000.0, "length_total_m": 41000.0,
                    "in_zone_fraction": 0.93, "time_entered": "2026-10-21T16:02:11Z",
                    "time_exited": "2026-10-21T21:40:05Z"}],
        "gaps": [{"id": "gap_0", "area_m2": 41000.0 if area else None,
                  "length_m": None if area else 88.0, "geometry": gap_geometry,
                  "dominant_vegetation": None, "mean_slope_deg": None,
                  "unsearched_shapes": ([{"id": "steep_35_0", "kind": "steep_band",
                                          "fraction_in_gap": 0.36}] if area else [])},
                 {"id": "gap_1", "area_m2": 3200.0 if area else None,
                  "length_m": None if area else 30.0,
                  "geometry": (_mercator_square(LAT + 0.003, LON, 10.0) if area
                               else gap_geometry),
                  "dominant_vegetation": None, "mean_slope_deg": None,
                  "unsearched_shapes": []}],
        "closed_gaps": [{"id": "closed_0", "area_m2": 120.0 if area else None,
                         "length_m": None if area else 12.0, "max_lateral_m": None,
                         "geometry": _mercator_square(LAT, LON, 3.0)}],
        "warnings": ["closed 1 short GPS-noise sliver totalling 120 m2 as raster closing; "
                     "see closed_gaps"],
        "time_entered": "2026-10-21T16:02:11Z", "time_exited": "2026-10-21T21:40:05Z",
        "time_searching_hours": 5.63,
        "staleness": dict(staleness or CURRENT),
    }


SUMMARIES = {"steep_band": {"count": 2},
             "clearing": {"count": 0},
             "drainage": {"count": 1, "drainages": [DRAINAGE], "upstream_truncated": True},
             "cliff_band": {"count": 1},
             "route": {"count": 0}, "ingress": {"count": 0},
             "ridge": {"count": 0, "truncated": False},
             "cliff_pitch": {"count": 1, "tallest_vertical_extent_m": 14.0,
                             "total_area_m2": 80.0}}
SHAPE_IDS = {"steep_band": ["steep_35_0", "steep_45_0"], "drainage": ["drainage_0"],
             "cliff_band": ["cliff_0"], "cliff_pitch": ["pitch_0"]}
KIND_WARNINGS = {"drainage": ["a drainage enters through the DEM edge; contributing areas "
                              "are understated and channel ranking may be incomplete"],
                 "clearing": ["vegetation panel missing; clearings unknown"]}


def identify_payload(arguments, staleness=None):
    """What identify_features returns: per kind, every shape with its outline."""
    results = {}
    for kind in dict.fromkeys(arguments["kinds"]):
        ids = SHAPE_IDS.get(kind, [])
        results[kind] = {
            "kind": kind,
            "shapes": [{"id": shape_id, "kind": kind, "geometry": _wgs84_square(LON, LAT),
                        "properties": {}} for shape_id in ids],
            "summary": copy.deepcopy(SUMMARIES[kind]),
            "warnings": list(KIND_WARNINGS.get(kind, [])),
            "assumptions": {"min_area_m2": 500.0}}
    return {"results": results, "staleness": dict(staleness or CURRENT)}


class FakeTerrainStaging:
    """`prepare_terrain` without the network: records each call and writes
    the two files an assignment handle is read back by - the manifest and
    the subject, whose shape says whether the assignment is an area or a
    line (a Point is neither, as an unreadable subject is) - or raises what
    it is told to, as a map that cannot be read would. `saved_metrics`
    leaves a `terrain_metrics.json` beside the ground, as an earlier
    terrain_metrics call on the same staging would have."""

    def __init__(self, *, error=None, shape="Polygon", saved_metrics=False):
        self.calls, self.error, self.shape = [], error, shape
        self.saved_metrics = saved_metrics

    def __call__(self, ctx, **arguments):
        self.calls.append(arguments)
        if self.error is not None:
            raise self.error
        handle = f"{arguments['map_id']}_{arguments['label']}"
        directory = Path(ctx.config.cache_dir) / "staged" / handle
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "manifest.json").write_text(json.dumps(
            {"map_id": arguments["map_id"], "assignment": arguments["label"]}), encoding="utf-8")
        geometry = {"Polygon": _wgs84_square(LON, LAT, 0.004), "LineString": _line(LON, LAT),
                    "Point": {"type": "Point", "coordinates": [LON, LAT]}}[self.shape]
        (directory / "subject.json").write_text(json.dumps(
            {"label": arguments["label"], "geometry": geometry}), encoding="utf-8")
        if self.saved_metrics:
            (directory / "terrain_metrics.json").write_text("{}", encoding="utf-8")
        return {"handle": handle, "staged": True, "reason": "not staged before",
                "fingerprint": "f" * 32, "previous_fingerprint": None,
                "warnings": ["evt unavailable: HTTPStatusError"],
                "provenance": {"fetched_at": 1.0, "age_seconds": 0.0, "source": "public"}}


class Stubs:
    """Staging and the three tools, recording every call. `fail` names
    sections whose tool raises; `staleness` gives a section's tool a
    staleness block other than current; `kind` is coverage's assignment
    kind and the staged subject's shape, unless `shape` says otherwise;
    `saved_metrics` leaves metrics saved by an earlier call on the ground."""

    def __init__(self, monkeypatch, *, fail=(), staging_error=None, kind="area",
                 staleness=None, shape=None, saved_metrics=False):
        self.fail, self.kind = set(fail), kind
        self.staleness = staleness or {}
        self.staging = FakeTerrainStaging(
            error=staging_error, saved_metrics=saved_metrics,
            shape=shape or ("Polygon" if kind == "area" else "LineString"))
        self.calls = {"metrics": [], "coverage": [], "features": []}
        monkeypatch.setattr(staging_component, "prepare_terrain", self.staging)
        monkeypatch.setattr(ground_component, "terrain_metrics", self.metrics)
        monkeypatch.setattr(coverage_feature, "coverage_for", self.coverage)
        monkeypatch.setattr(ground_component, "identify_features", self.identify)

    def metrics(self, ctx, **arguments):
        self.calls["metrics"].append(arguments)
        if "metrics" in self.fail:
            raise OSError("dem.tif could not be read")
        return metrics_payload(self.staleness.get("metrics"))

    def coverage(self, ctx, **arguments):
        self.calls["coverage"].append(arguments)
        if "coverage" in self.fail:
            raise ValueError("no tracks on this map are named for team(s) 101.")
        return coverage_payload(arguments, kind=self.kind,
                                staleness=self.staleness.get("coverage"))

    def identify(self, ctx, **arguments):
        self.calls["features"].append(arguments)
        if "features" in self.fail:
            raise RuntimeError("flow routing ran out of memory")
        return identify_payload(arguments, self.staleness.get("features"))

    def nothing_called(self) -> bool:
        return not self.staging.calls and not any(self.calls.values())


@pytest.fixture
def ctx(tmp_path):
    return ServerContext.build(CaltopoConfig(cache_dir=tmp_path / "cache"))


def debrief(ctx, **arguments):
    return debrief_assignment(ctx, **{"map_id": MAP_ID, "label": LABEL,
                                      "track_names": TRACKS, **arguments})


# --- Every section ------------------------------------------------------------


def test_every_section_answers_and_the_summary_reads_the_facts(ctx, monkeypatch):
    stubs = Stubs(monkeypatch)
    result = debrief(ctx)

    assert result["status"] == "ok"
    assert result["assignment"] == {"map_id": MAP_ID, "label": LABEL}
    assert {name: entry["status"] for name, entry in result["sections"].items()} == {
        "metrics": "ok", "coverage": "ok", "features": "ok"}
    assert all(entry["error"] is None and entry["ground"] == "assignment"
               and entry["handle"] == HANDLE for entry in result["sections"].values())

    # One staging, of the assignment, and each tool once on its handle - the
    # metrics first, since they save the sweep width coverage reads.
    assert stubs.staging.calls == [{"map_id": MAP_ID, "label": LABEL, "public": True}]
    assert stubs.calls == {"metrics": [{"handle": HANDLE}],
                           "coverage": [{"handle": HANDLE, "track_names": ["101"]}],
                           "features": [{"handle": HANDLE,
                                         "kinds": ["steep_band", "drainage", "cliff_band"]}]}
    assert result["handles"] == {"assignment": HANDLE}
    record = result["staging"]["assignment"]
    assert record["status"] == "ok" and record["staged"] is True
    assert record["reason"] == "not staged before" and record["fingerprint"] == "f" * 32
    assert record["warnings"] == ["evt unavailable: HTTPStatusError"]
    assert record["detail_call"] == {"tool": "prepare_terrain", "arguments": {
        "map_id": MAP_ID, "label": LABEL, "public": True}}

    metrics = result["sections"]["metrics"]
    assert metrics["detail_call"] == {"tool": "terrain_metrics", "arguments": {"handle": HANDLE}}
    assert metrics["warnings"] == ["vegetation panel missing; density unknown"]
    assert metrics["area_m2"] == 520000.0 and metrics["dem_resolution_m"] == 2.9
    assert metrics["elevation"]["relief_m"] == 220.0 and metrics["slope"]["p90_deg"] == 31.0
    assert metrics["landform"] == {"fractions": {"ridge": 0.1, "valley": 0.2, "bench": 0.05,
                                                 "slope": 0.65}, "cliff_band_count": 1}
    drainage = metrics["hydrology"]["drainages"][0]
    assert "method" not in drainage and drainage["depth_m"] == {"left": 6.0, "right": 4.5}
    assert metrics["hydrology"]["drainage_count"] == 1
    assert metrics["hydrology"]["upstream_truncated"] is True
    assert "assumptions" not in metrics["traverse"]
    assert metrics["traverse"]["effective_sweep_width_m"] == 9.6
    assert metrics["access"] == {"routes": 3, "routes_by_type": {"TRAIL": 2, "ROAD": 1},
                                 "route_names": ["Steep Ravine Trail"], "ingress_count": 2,
                                 "source": "OpenStreetMap via Overpass"}
    assert metrics["shapes"] == {"steep_band": 2, "drainage": 1, "cliff_band": 1}
    assert "staleness" not in metrics

    coverage = result["sections"]["coverage"]
    assert coverage["detail_call"] == {"tool": "coverage_for", "arguments": {
        "handle": HANDLE, "track_names": ["101"]}}
    assert coverage["assignment_kind"] == "area" and coverage["teams"] == ["101"]
    assert coverage["covered_fraction"] == 0.62 and coverage["effort_coverage"] == 0.71
    assert coverage["tracks"][0]["name"] == "101 AB"
    assert coverage["gap_count"] == 2 and coverage["gap_area_m2"] == 44200.0
    assert coverage["gap_length_m"] is None
    assert [gap["id"] for gap in coverage["gaps"]] == ["gap_0", "gap_1"]
    gap = coverage["gaps"][0]
    assert "geometry" not in gap
    assert gap["unsearched_shapes"] == [{"id": "steep_35_0", "kind": "steep_band",
                                         "fraction_in_gap": 0.36}]
    # The Mercator outline comes back as a WGS84 point inside it.
    assert gap["point"]["type"] == "Point"
    assert gap["point"]["coordinates"] == pytest.approx([LON, LAT], abs=1e-5)
    assert coverage["closed_gaps"] == {"count": 1, "area_m2": 120.0, "length_m": None}
    assert coverage["warnings"] == ["closed 1 short GPS-noise sliver totalling 120 m2 as "
                                    "raster closing; see closed_gaps"]

    features = result["sections"]["features"]
    assert features["detail_call"] == {"tool": "identify_features", "arguments": {
        "handle": HANDLE, "kinds": ["steep_band", "drainage", "cliff_band"]}}
    assert list(features["kinds"]) == ["steep_band", "drainage", "cliff_band"]
    assert features["kinds"]["steep_band"] == {"count": 2, "ids": ["steep_35_0", "steep_45_0"],
                                               "summary": {}, "warnings": []}
    # The drainages' measurements are the metrics'; the single fact stays.
    assert features["kinds"]["drainage"]["summary"] == {"upstream_truncated": True}
    assert features["kinds"]["drainage"]["warnings"] == KIND_WARNINGS["drainage"]
    # A warning one kind gave says which.
    assert features["warnings"] == [f"drainage: {KIND_WARNINGS['drainage'][0]}"]

    summary = result["summary"]
    assert summary["metrics"] == {
        "area_m2": 520000.0, "relief_m": 220.0, "slope_mean_deg": 18.0, "slope_p90_deg": 31.0,
        "dominant_lifeform": "Shrub", "density_class": None, "effective_sweep_width_m": 9.6,
        "sweep_time_hours_per_pass": 7.5, "drainages": 1, "cliff_bands": 1,
        "ingress_points": 2}
    assert summary["coverage"] == {
        "assignment_kind": "area", "covered_fraction": 0.62, "effort_coverage": 0.71,
        "tracks": 1, "gaps": 2, "gap_area_m2": 44200.0, "gap_length_m": None,
        "largest_gap": {"id": "gap_0", "area_m2": 41000.0, "length_m": None,
                        "point": gap["point"]},
        "time_searching_hours": 5.63}
    assert summary["features"] == {"steep_band": 2, "drainage": 1, "cliff_band": 1}

    # The boundary check, once, at the top: the latest section's.
    assert result["staleness"] == CURRENT
    assert result["exports"] == []
    assert result["warnings"] == []
    assert "No verdict" in result["note"]


def test_the_answer_carries_points_but_no_lines_or_polygons(ctx, monkeypatch):
    Stubs(monkeypatch)
    result = debrief(ctx, kinds=list(REGISTRY))
    assert geometry_types_in(result) == {"Point"}
    json.dumps(result)                       # and it is JSON as it stands


def test_a_line_assignment_reports_lengths_not_areas(ctx, monkeypatch):
    Stubs(monkeypatch, kind="line")
    result = debrief(ctx)
    coverage = result["sections"]["coverage"]
    assert coverage["assignment_kind"] == "line"
    assert coverage["gap_area_m2"] is None and coverage["gap_length_m"] == 118.0
    assert coverage["closed_gaps"] == {"count": 1, "area_m2": None, "length_m": 12.0}
    point = coverage["gaps"][0]["point"]
    assert point["type"] == "Point"
    assert LON <= point["coordinates"][0] <= LON + 0.001
    assert point["coordinates"][1] == pytest.approx(LAT, abs=1e-6)
    assert result["summary"]["coverage"]["largest_gap"]["length_m"] == 88.0


def test_no_gap_totals_zero_for_the_assignment_s_own_measure(ctx, monkeypatch):
    """Tracks that left no gap: 0 m2 unsearched is the answer, not a null a
    reader could take for "not measured"; the length an area lacks stays null."""
    Stubs(monkeypatch, saved_metrics=True)

    def swept(ctx, **arguments):
        payload = coverage_payload(arguments)
        payload.update(gaps=[], closed_gaps=[], covered_fraction=1.0)
        return payload

    monkeypatch.setattr(coverage_feature, "coverage_for", swept)
    result = debrief(ctx, include=["coverage"])
    coverage = result["sections"]["coverage"]
    assert coverage["gaps"] == [] and coverage["gap_count"] == 0
    assert coverage["gap_area_m2"] == 0.0 and coverage["gap_length_m"] is None
    assert coverage["closed_gaps"] == {"count": 0, "area_m2": 0.0, "length_m": None}
    assert result["summary"]["coverage"]["largest_gap"] is None


# --- Failures stay in their section --------------------------------------------


def test_one_section_failing_leaves_the_others_and_the_status_partial(ctx, monkeypatch):
    Stubs(monkeypatch, fail={"features"})
    result = debrief(ctx)
    assert result["status"] == "partial"
    features = result["sections"]["features"]
    assert features["status"] == "failed"
    assert features["error"] == "RuntimeError: flow routing ran out of memory"
    # The call that raised, to make again.
    assert features["detail_call"] == {"tool": "identify_features", "arguments": {
        "handle": HANDLE, "kinds": ["steep_band", "drainage", "cliff_band"]}}
    assert result["sections"]["metrics"]["status"] == "ok"
    assert result["sections"]["coverage"]["status"] == "ok"
    assert result["summary"]["features"] is None and result["summary"]["coverage"] is not None
    # The latest check that answered.
    assert result["staleness"] == CURRENT


def test_coverage_failing_leaves_the_ground_and_its_features(ctx, monkeypatch):
    Stubs(monkeypatch, fail={"coverage"})
    result = debrief(ctx)
    assert result["status"] == "partial"
    coverage = result["sections"]["coverage"]
    assert coverage["error"].startswith("ValueError: no tracks on this map are named")
    assert coverage["detail_call"]["tool"] == "coverage_for"
    assert result["sections"]["features"]["status"] == "ok"


def test_an_assignment_that_cannot_be_staged_fails_every_section(ctx, monkeypatch):
    stubs = Stubs(monkeypatch, staging_error=LookupError(
        "assignment 'AB' is ambiguous (prefix match); it could be: AB1, AB2"))
    result = debrief(ctx)
    assert result["status"] == "failed"
    for entry in result["sections"].values():
        assert entry["status"] == "failed"
        assert entry["error"] == (
            "StagingFailed: the assignment ground (assignment 'AB' on map MAP1234) could not "
            "be staged, so this section did not run - LookupError: assignment 'AB' is "
            "ambiguous (prefix match); it could be: AB1, AB2")
        # No call was made, so none to repeat: the staging record has it.
        assert "detail_call" not in entry
    assert stubs.calls == {"metrics": [], "coverage": [], "features": []}
    record = result["staging"]["assignment"]
    assert record["status"] == "failed" and record["handle"] is None
    assert record["detail_call"]["tool"] == "prepare_terrain"
    assert result["handles"] == {"assignment": None}
    assert result["staleness"] is None
    assert result["summary"] == {"metrics": None, "coverage": None, "features": None}


def test_failed_metrics_fail_an_area_s_coverage_naming_why(ctx, monkeypatch):
    """An area's sweep width is the metrics': without them this call's
    coverage would be computed from a width it did not measure, or not at
    all, in words that point at the wrong tool."""
    stubs = Stubs(monkeypatch, fail={"metrics"})
    result = debrief(ctx)
    assert result["status"] == "partial"
    assert result["sections"]["metrics"]["error"] == "OSError: dem.tif could not be read"
    coverage = result["sections"]["coverage"]
    assert coverage["status"] == "failed"
    assert coverage["error"] == (
        "DependencyFailed: coverage of an area assignment takes its sweep width from "
        "terrain_metrics, which the metrics section runs first, and it failed, so this "
        "section did not run - OSError: dem.tif could not be read")
    assert "detail_call" not in coverage
    assert stubs.calls["coverage"] == []
    assert result["sections"]["features"]["status"] == "ok"


@pytest.mark.parametrize("saved, width", [
    (False, " and no metrics are saved on this ground, so this line's corridor is "
            "coverage_for's default width, as its own warning says"),
    (True, ", so this line's corridor width is the one terrain_metrics saved on this ground "
           "at an earlier call"),
])
def test_failed_metrics_still_let_a_line_s_coverage_run(ctx, monkeypatch, saved, width):
    """A line's corridor width has a fallback, so its coverage still runs -
    saying the metrics failed, and which width it used instead."""
    stubs = Stubs(monkeypatch, fail={"metrics"}, kind="line", saved_metrics=saved)
    result = debrief(ctx)
    coverage = result["sections"]["coverage"]
    assert coverage["status"] == "ok" and len(stubs.calls["coverage"]) == 1
    assert coverage["warnings"][-1] == "the metrics section failed" + width
    assert result["status"] == "partial"


@pytest.mark.xfail(strict=True, reason="PR #4 review B37: If the metrics section fails, "
                                       "coverage of an area is refused even when valid saved "
                                       "metrics exist; leaving metrics out computes it")
def test_failed_metrics_on_an_area_with_metrics_saved_use_the_saved_width(ctx, monkeypatch):
    """An earlier call saved this ground's metrics (a re-staging would have
    discarded them, so they describe it as staged now): a metrics section
    that fails this time leaves the area's sweep width where coverage_for
    reads it, as it does for a line - and as leaving metrics out does."""
    stubs = Stubs(monkeypatch, fail={"metrics"}, saved_metrics=True)
    result = debrief(ctx)
    coverage = result["sections"]["coverage"]
    assert coverage["status"] == "ok", coverage["error"]
    assert len(stubs.calls["coverage"]) == 1
    note = coverage["warnings"][-1]
    assert "metrics section failed" in note and "saved" in note
    assert result["status"] == "partial"
    left_out = debrief(ctx, include=["coverage"])
    assert left_out["sections"]["coverage"]["covered_fraction"] == coverage["covered_fraction"]


def test_failed_metrics_and_an_unreadable_kind_do_not_guess_at_coverage(ctx, monkeypatch):
    """Whether the assignment is an area or a line cannot be read off the
    staged subject (a Point, as an unreadable one): coverage might need the
    width that failed, so it does not run - it is not guessed to be a line."""
    stubs = Stubs(monkeypatch, fail={"metrics"}, shape="Point")
    result = debrief(ctx)
    coverage = result["sections"]["coverage"]
    assert coverage["status"] == "failed"
    assert coverage["error"] == (
        "DependencyFailed: coverage of this assignment (whether it is an area or a line could "
        "not be read from the staged ground) takes its sweep width from terrain_metrics, which "
        "the metrics section runs first, and it failed, so this section did not run - "
        "OSError: dem.tif could not be read")
    assert "detail_call" not in coverage
    assert stubs.calls["coverage"] == []
    assert result["sections"]["features"]["status"] == "ok"


def test_every_section_failing_is_failed(ctx, monkeypatch):
    Stubs(monkeypatch, fail={"metrics", "coverage", "features"})
    result = debrief(ctx)
    assert result["status"] == "failed"
    assert result["sections"]["coverage"]["error"].startswith("DependencyFailed")
    assert result["staleness"] is None


# --- The boundary check ---------------------------------------------------------


MOVED = {"current": False, "checked": True, "fingerprint": "a" * 32,
         "live_fingerprint": "b" * 32,
         "reason": "the assignment boundary has changed since this was staged"}
UNCHECKED = {"current": None, "checked": False,
             "reason": "could not revalidate (ConnectError); the boundary may have "
                       "changed since this was staged"}


@pytest.mark.parametrize("moved_in", [{"features"}, {"coverage", "features"},
                                      {"metrics", "coverage", "features"}])
def test_a_boundary_that_moved_puts_every_section_on_the_old_polygon(ctx, monkeypatch,
                                                                     moved_in):
    """All three are computed from the one staged ground, so a move the
    latest check found is said of every section, not only the ones whose
    own check saw it - the metrics computed seconds before describe the
    same old polygon."""
    Stubs(monkeypatch, staleness={name: MOVED for name in moved_in})
    result = debrief(ctx)
    assert result["staleness"] == MOVED
    assert result["warnings"] == [
        "the assignment boundary has changed since it was staged: every section here "
        "(metrics, coverage, features) was computed on the staged ground, so its numbers "
        "describe the OLD polygon - call again, and prepare_terrain re-stages the ground for "
        "the boundary as it now stands, before acting on them"]
    assert all("staleness" not in entry for entry in result["sections"].values())


def test_a_boundary_found_current_at_the_latest_check_needs_no_warning(ctx, monkeypatch):
    """The metrics' check could not reach the map, but the features' check,
    later, found the staged ground still the assignment: that holds for the
    metrics too, and the top-level block says so - no contradicting warning."""
    Stubs(monkeypatch, staleness={"metrics": UNCHECKED})
    result = debrief(ctx)
    assert result["staleness"] == CURRENT and result["warnings"] == []


def test_a_boundary_that_could_not_be_checked_is_said(ctx, monkeypatch):
    Stubs(monkeypatch, staleness={"metrics": UNCHECKED, "features": UNCHECKED},
          fail={"coverage"})
    result = debrief(ctx)
    assert result["staleness"] == UNCHECKED
    assert result["warnings"] == [
        "the assignment boundary could not be checked against the live map at the latest "
        "check (could not revalidate (ConnectError); the boundary may have changed since "
        "this was staged): every section here (metrics, features) describes the ground as "
        "it was staged, which may not be the assignment as it now stands"]


def test_an_unchecked_latest_check_says_an_earlier_one_found_it_as_staged(ctx, monkeypatch):
    Stubs(monkeypatch, staleness={"features": UNCHECKED})
    result = debrief(ctx)
    assert result["staleness"] == UNCHECKED
    assert result["warnings"] == [
        "the assignment boundary could not be checked against the live map at the latest "
        "check (could not revalidate (ConnectError); the boundary may have changed since "
        "this was staged); an earlier check in this call found it as staged: every section "
        "here (metrics, coverage, features) describes the ground as it was staged, which may "
        "not be the assignment as it now stands"]


def test_a_move_an_earlier_check_found_is_not_lost(ctx, monkeypatch):
    """Found moved, then not checkable: nothing since showed it back, so the
    move is said. Found moved, then as staged again: it may be being edited."""
    Stubs(monkeypatch, staleness={"metrics": MOVED, "features": UNCHECKED})
    result = debrief(ctx)
    assert result["warnings"][0].startswith("the assignment boundary has changed since it "
                                            "was staged: every section here (metrics, "
                                            "coverage, features)")
    Stubs(monkeypatch, staleness={"metrics": MOVED})
    result = debrief(ctx)
    assert result["staleness"] == CURRENT
    assert result["warnings"] == [
        "a check during this call found the assignment boundary changed since it was staged, "
        "and the latest check found it as staged again, so it may be being edited: every "
        "section here (metrics, coverage, features) describes the boundary as it was staged "
        "- call again before acting on them"]


# --- What the caller chooses --------------------------------------------------


def test_include_skips_sections_and_their_arguments(ctx, monkeypatch):
    """With coverage left out an empty track list means nothing, and with
    features left out so does an empty kind list."""
    stubs = Stubs(monkeypatch)
    result = debrief(ctx, include=["metrics"], track_names=[], kinds=[])
    assert result["status"] == "ok"
    for name in ("coverage", "features"):
        assert result["sections"][name] == {"status": "skipped", "error": None, "warnings": []}
    assert stubs.calls["coverage"] == [] and stubs.calls["features"] == []
    assert set(result["summary"]) == {"metrics"}
    assert result["sections"]["metrics"]["ground"] == "assignment"


def test_coverage_without_metrics_runs_when_an_earlier_call_saved_them(ctx, monkeypatch):
    """Left out of include, the metrics are not run, and coverage_for is
    called to read the ones an earlier call saved on this ground (the real
    read is in test_the_sections_read_what_the_real_tools_return)."""
    stubs = Stubs(monkeypatch, saved_metrics=True)
    result = debrief(ctx, include=["coverage"])
    coverage = result["sections"]["coverage"]
    assert coverage["status"] == "ok" and result["status"] == "ok"
    assert coverage["warnings"] == coverage_payload({"track_names": TRACKS})["warnings"]
    assert stubs.calls["metrics"] == [] and len(stubs.calls["coverage"]) == 1
    assert result["sections"]["metrics"]["status"] == "skipped"


def test_coverage_of_an_area_with_no_metrics_saved_or_asked_for_says_to_ask(ctx, monkeypatch):
    """Nothing saved on this ground (every first call: a staging discards
    them): the area has no sweep width, and the error names the scenario's
    own remedy - not coverage_for's local path and command line."""
    stubs = Stubs(monkeypatch)
    result = debrief(ctx, include=["coverage", "features"])
    coverage = result["sections"]["coverage"]
    assert coverage["status"] == "failed"
    assert coverage["error"] == (
        "DependencyFailed: coverage of an area assignment takes its sweep width from "
        "terrain_metrics, and none is saved on this ground (staging discards it whenever the "
        "ground changes), so this section did not run - add \"metrics\" to include")
    assert "detail_call" not in coverage
    assert stubs.calls["coverage"] == [] and stubs.calls["metrics"] == []
    assert result["sections"]["features"]["status"] == "ok"
    assert result["status"] == "partial"


def test_coverage_of_a_line_with_no_metrics_saved_or_asked_for_runs_saying_so(ctx, monkeypatch):
    stubs = Stubs(monkeypatch, kind="line")
    result = debrief(ctx, include=["coverage"])
    coverage = result["sections"]["coverage"]
    assert coverage["status"] == "ok" and len(stubs.calls["coverage"]) == 1
    assert coverage["warnings"][-1] == (
        "metrics was left out of include and no metrics are saved on this ground, so this "
        "line's corridor is coverage_for's default width, as its own warning says - add "
        "\"metrics\" to include for the width this ground's vegetation gives")


def test_a_line_whose_metrics_an_earlier_call_saved_carries_no_scenario_note(ctx, monkeypatch):
    """Metrics left out but saved on this ground: coverage_for reads the
    corridor width they saved, so the section says nothing of its own - a
    "left out" note would tell the operator the 10 m default was used."""
    stubs = Stubs(monkeypatch, kind="line", saved_metrics=True)
    result = debrief(ctx, include=["coverage"])
    coverage = result["sections"]["coverage"]
    assert coverage["status"] == "ok" and result["status"] == "ok"
    assert stubs.calls["metrics"] == [] and len(stubs.calls["coverage"]) == 1
    assert coverage["warnings"] == coverage_payload({"track_names": TRACKS},
                                                    kind="line")["warnings"]
    assert not any("left out" in warning for warning in coverage["warnings"])


def test_kinds_reach_identify_features_once_each_in_order(ctx, monkeypatch):
    stubs = Stubs(monkeypatch)
    result = debrief(ctx, kinds=["cliff_pitch", "ridge", "cliff_pitch"], public=False)
    assert stubs.staging.calls[0]["public"] is False
    assert stubs.calls["features"] == [{"handle": HANDLE, "kinds": ["cliff_pitch", "ridge"]}]
    pitch = result["sections"]["features"]["kinds"]["cliff_pitch"]
    assert pitch == {"count": 1, "ids": ["pitch_0"],
                     "summary": {"tallest_vertical_extent_m": 14.0, "total_area_m2": 80.0},
                     "warnings": []}
    assert result["summary"]["features"] == {"cliff_pitch": 1, "ridge": 0}


def test_the_kind_enum_is_the_registry():
    """The schema's enum is written out, so a new detector must be added to
    it: this fails until it is."""
    assert list(get_args(FeatureKind)) == list(REGISTRY)
    assert set(DEFAULT_FEATURE_KINDS) <= set(REGISTRY)


@pytest.mark.parametrize("overrides, message", [
    ({"lz": {"limit": 3}}, r"no section of debrief_assignment: \['lz'\]"),
    ({"metrics": {"resolution_m": 2.0}},
     r"overrides\['metrics'\] names \['resolution_m'\], but terrain_metrics takes no keyword "
     r"debrief_assignment does not set itself, so there is nothing to override"),
    ({"coverage": {"corridor_width_m": 20}},
     r"\['corridor_width_m'\], but coverage_for takes no keyword"),
    ({"metrics": {"handle": "OTHER_X"}}, "cannot set 'handle': the scenario stages the "
                                         "assignment itself with prepare_terrain"),
    ({"coverage": {"track_names": ["102"]}}, "cannot set 'track_names': debrief_assignment "
                                             "sets it from its own argument track_names"),
    ({"features": {"kinds": ["ridge"]}}, "cannot set 'kinds': debrief_assignment sets it "
                                         "from its own argument kinds"),
    ({"features": "all"}, "must be a mapping"),
    (["metrics"], "overrides must be a mapping"),
])
def test_bad_overrides_are_refused_before_any_work(ctx, monkeypatch, overrides, message):
    stubs = Stubs(monkeypatch)
    with pytest.raises(ValueError, match=message):
        debrief(ctx, overrides=overrides)
    assert stubs.nothing_called()


def test_empty_overrides_are_no_overrides(ctx, monkeypatch):
    Stubs(monkeypatch)
    result = debrief(ctx, overrides={"metrics": {}, "coverage": None})
    assert result["status"] == "ok" and result["warnings"] == []


@pytest.mark.parametrize("arguments, message", [
    ({"map_id": ""}, "map_id must be the CalTopo map id"),
    ({"map_id": None}, "map_id"),
    ({"label": "  "}, "label must be the assignment as CalTopo labels it"),
    ({"include": ["gaps"]}, "no section of this scenario"),
    ({"include": []}, "empty"),
    ({"track_names": "101"}, r"track_names must be a list of track names, such as \['101'\]"),
    ({"track_names": []}, "track_names is empty, but coverage is asked for"),
    ({"track_names": ["101", ""]}, "track_names must be names"),
    ({"track_names": [101]}, "track_names must be names"),
    ({"kinds": ["drainages"]}, "unknown feature kind"),
    ({"kinds": "drainage"}, "kinds must be a list of feature kinds"),
    ({"kinds": []}, "kinds must name at least one feature kind"),
    ({"kinds": [None]}, "kinds must be feature kind names"),
    # Checked whether or not the section runs: a wrong value is wrong either way.
    ({"kinds": ["drainages"], "include": ["metrics"]}, "unknown feature kind"),
    ({"track_names": "101", "include": ["metrics"]}, "track_names must be a list"),
])
def test_bad_arguments_are_refused_before_any_work(ctx, monkeypatch, arguments, message):
    stubs = Stubs(monkeypatch)
    with pytest.raises(ValueError, match=message):
        debrief(ctx, **arguments)
    assert stubs.nothing_called()


# --- Ground already staged ----------------------------------------------------


@pytest.mark.xfail(strict=True, reason="PR #4 review B42: debrief staging record's warnings is "
                                       "always empty when the ground is reused (the common "
                                       "case)")
def test_a_reused_assignment_ground_still_reports_what_it_lacks(monkeypatch, tmp_path):
    """The real prepare_terrain, with the map read and the download faked as
    test_handles fakes them: the ground staged without vegetation and roads
    still lacks them when the next debrief finds it current, and the staging
    record says so - except an OpenStreetMap outage a later osm.json
    retired, as prepare_area's reuse reports them (`manifest_warnings`)."""
    from test_handles import SQUARE as QA103, FakeAssignment, FakeReader

    from sar_geo_mcp.staging import osm_outage_warning, write_manifest
    from sar_geo_mcp.terrain.linear import OSM_QUERY_VERSION

    evt = "evt unavailable: HTTPStatusError"
    outage = osm_outage_warning("OverpassUnavailable: 504")
    reader = FakeReader(tmp_path / "cache", [FakeAssignment(QA103)])
    ctx = ServerContext(config=reader.config, reader=reader)
    monkeypatch.setattr("sar_geo_mcp.handles.find_assignment",
                        lambda features, label: reader.assignment[0])
    stagings = []

    def stage_terrain(assignment, map_id, out_dir, *, cache_dir, buffer_m, fingerprint,
                      access="public", subject_kind="assignment", **kwargs):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        write_manifest(out_dir, map_id=map_id, assignment_label=assignment.label,
                       bbox=(0, 0, 1, 1), dem_bbox=(0, 0, 1, 1), dem_resolution_m=1.0,
                       latitude_deg=37.4, panels={}, warnings=[evt, outage],
                       assignment_fingerprint=fingerprint, access=access)
        stagings.append(Path(out_dir))
        return {"warnings": [evt, outage]}

    monkeypatch.setattr("sar_geo_mcp.staging.stage_terrain", stage_terrain)
    monkeypatch.setattr(ground_component, "identify_features",
                        lambda ctx, **arguments: identify_payload(arguments))

    first = debrief(ctx, label="QA103", include=["features"])
    assert first["staging"]["assignment"]["warnings"] == [evt, outage]
    second = debrief(ctx, label="QA103", include=["features"])
    assert len(stagings) == 1, "the second call reused the ground"
    record = second["staging"]["assignment"]
    assert record["staged"] is False and record["reason"] == "already current"
    assert record["warnings"] == [evt, outage]

    # Roads fetched since: the outage is no longer so, the missing vegetation is.
    directory = stagings[0]
    (directory / "osm.json").write_text(json.dumps({"routes": []}), encoding="utf-8")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    manifest["osm_refreshed_versions"] = [OSM_QUERY_VERSION]
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    third = debrief(ctx, label="QA103", include=["features"])
    assert third["staging"]["assignment"]["warnings"] == [evt]


# --- Through MCP ------------------------------------------------------------------


@pytest.mark.anyio
async def test_debrief_assignment_is_registered_as_a_scenario_and_answers(monkeypatch, tmp_path):
    pytest.importorskip("fastmcp")
    from fastmcp import Client
    from fastmcp.exceptions import ToolError

    from sar_geo_mcp.server import build_server

    Stubs(monkeypatch)
    server = build_server(CaltopoConfig(cache_dir=tmp_path / "cache"))
    async with Client(server) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
        tool = tools["debrief_assignment"]
        assert tool.meta["fastmcp"]["tags"] == ["scenario"]
        assert "NO VERDICT" in tool.description and "detail_call" in tool.description
        properties = tool.inputSchema["properties"]
        assert properties["include"]["anyOf"][0]["items"]["enum"] == [
            "metrics", "coverage", "features"]
        assert properties["kinds"]["items"]["enum"] == list(REGISTRY)
        assert properties["kinds"]["default"] == ["steep_band", "drainage", "cliff_band"]
        assert tool.inputSchema["required"] == ["map_id", "label", "track_names"]

        answer = (await client.call_tool("debrief_assignment", {
            "map_id": MAP_ID, "label": LABEL, "track_names": TRACKS,
            "include": ["metrics", "coverage"]})).data
        assert answer["status"] == "ok"
        assert answer["sections"]["features"]["status"] == "skipped"
        assert answer["summary"]["coverage"]["gaps"] == 2

        with pytest.raises(ToolError):
            await client.call_tool("debrief_assignment", {
                "map_id": MAP_ID, "label": LABEL, "track_names": TRACKS,
                "kinds": ["drainages"]})


@pytest.mark.anyio
@pytest.mark.xfail(strict=True, reason="PR #4 review B38: debrief docstring can be read as "
                                       "saying get_assignment's outline is in Web Mercator; "
                                       "it is WGS84")
async def test_the_description_gives_get_assignment_s_outline_as_wgs84(tmp_path):
    """The description sends a debriefer to get_assignment for the outline:
    no clause naming it may put it in Web Mercator, which a client would
    'reproject' to the Gulf of Guinea - its outline is CalTopo's WGS84."""
    pytest.importorskip("fastmcp")
    from fastmcp import Client

    from sar_geo_mcp.server import build_server

    server = build_server(CaltopoConfig(cache_dir=tmp_path / "cache"))
    async with Client(server) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
    clauses = re.split(r"(?<=[.;])\s+", " ".join(tools["debrief_assignment"].description.split()))
    about = [clause for clause in clauses if "get_assignment" in clause]
    assert about, "the description names get_assignment for the outline"
    assert not [clause for clause in about if "Web Mercator" in clause or "3857" in clause]
    assert any("WGS84" in clause for clause in about)
    assert "WGS84" in tools["get_assignment"].description


@pytest.mark.anyio
async def test_get_assignment_s_outline_is_caltopo_s_longitude_and_latitude(monkeypatch,
                                                                           tmp_path):
    """What the description above must say: the outline get_assignment
    returns is the map's own WGS84 geometry, vertex for vertex - degrees,
    not Mercator metres."""
    from test_server import SQUARE as QA103
    from test_server import call, stub_caltopo

    from sar_geo_mcp.server import build_server

    stub_caltopo(monkeypatch)
    server = build_server(CaltopoConfig(cache_dir=tmp_path / "cache"))
    result = await call(server, "get_assignment", map_id="M1", label="QA103")
    assert result["geometry"]["coordinates"] == QA103["coordinates"]
    lon, lat = result["geometry"]["coordinates"][0][0][:2]
    assert (lon, lat) == (-122.32, 37.40)


# --- Against the real tools -------------------------------------------------------


def team_track(title, start, end):
    """A GPS track on the valley's map, from `start` to `end` (Mercator
    offsets from the raster origin), ten minutes walking it."""
    from test_boundary import m

    return {"properties": {"title": title, "class": "Shape"},
            "geometry": {"type": "LineString",
                         "coordinates": [m(*start) + [100.0, 0], m(*end) + [100.0, 600_000]]}}


def real_valley(monkeypatch, tmp_path, tracks, *, subject=None, check=None):
    """The V-shaped valley `test_server._staged_valley` writes to disk, with
    `tracks` on its map, staged for this scenario: `prepare_terrain` stubbed
    to find it already current, and the on-disk read the scenario makes -
    the subject's shape, whether metrics are saved - pointed at it (through
    `handles.resolve`; `_staged_valley` stubs only the revalidating path the
    tools take). `subject`, a WGS84 geometry, replaces the whole-raster
    square; `check`, the tools' boundary check, replaces "current" - with
    no map state, as when the check could not be made. Returns the staged
    directory."""
    from test_server import _staged_valley

    from sar_geo_mcp.handles import Resolved

    data_dir = _staged_valley(monkeypatch, tmp_path, features=tracks)
    if subject is not None:
        record = json.loads((data_dir / "assignment.json").read_text(encoding="utf-8"))
        record["geometry"] = subject
        (data_dir / "assignment.json").write_text(json.dumps(record), encoding="utf-8")
    if check is not None:
        monkeypatch.setattr("sar_geo_mcp.context.resolve",
                            lambda handle, reader, *, revalidate=True: Resolved(
                                data_dir, dict(check)))
    monkeypatch.setattr("sar_geo_mcp.handles.resolve",
                        lambda handle, reader, *, revalidate=True: Resolved(
                            data_dir, {"current": None, "checked": False,
                                       "reason": "revalidation skipped by caller"}))
    monkeypatch.setattr(staging_component, "prepare_terrain",
                        lambda ctx, **arguments: {"handle": "M1_AB", "staged": False,
                                                  "reason": "already current",
                                                  "fingerprint": "f" * 32})
    return data_dir


def keep_real_answers(monkeypatch) -> dict:
    """The three real tools, each answer kept by section as the tool gave
    it, to hold a compact section to the payload it was built from."""
    real = {}

    def keep(name, function):
        def kept(ctx, **arguments):
            real[name] = copy.deepcopy(function(ctx, **arguments))
            return real[name]
        return kept

    monkeypatch.setattr(ground_component, "terrain_metrics",
                        keep("metrics", ground_component.terrain_metrics))
    monkeypatch.setattr(coverage_feature, "coverage_for",
                        keep("coverage", coverage_feature.coverage_for))
    monkeypatch.setattr(ground_component, "identify_features",
                        keep("features", ground_component.identify_features))
    return real


def test_the_sections_read_what_the_real_tools_return(ctx, monkeypatch, tmp_path):
    """A V-shaped valley written to disk behind a stubbed boundary check,
    with one team track across it, and the real terrain_metrics,
    coverage_for and identify_features over it. Holds the compact sections
    to the keys the tools actually give - a section built from `.get(key)`
    would quietly carry None for a renamed one - and the gap point to the
    coordinates coverage_for really gives its gap outlines in."""
    from test_boundary import m

    real_valley(monkeypatch, tmp_path, [team_track("101 AB", (50, 200), (350, 200))])
    real = keep_real_answers(monkeypatch)

    # Coverage alone, before any metrics were saved: an area has no sweep
    # width, and the section says to ask for the metrics - without the local
    # path and command line coverage_for's own refusal would name.
    alone = debrief(ctx, map_id="M1", include=["coverage"])
    assert alone["sections"]["coverage"]["status"] == "failed"
    assert alone["sections"]["coverage"]["error"].startswith("DependencyFailed: ")
    assert alone["sections"]["coverage"]["error"].endswith('add "metrics" to include')
    assert "coverage" not in real

    result = debrief(ctx, map_id="M1", kinds=list(REGISTRY))
    assert result["status"] == "ok", {name: entry["error"]
                                      for name, entry in result["sections"].items()}

    metrics = result["sections"]["metrics"]
    payload = real["metrics"]
    assert set(_sections.TRAVERSE_FIELDS) <= set(payload["traverse"])
    assert payload["hydrology"]["drainages"], "the valley has a drainage"
    assert set(_sections.DRAINAGE_FIELDS) <= set(payload["hydrology"]["drainages"][0])
    assert metrics["hydrology"]["drainage_count"] == len(payload["hydrology"]["drainages"])
    assert metrics["shapes"] == {
        kind: sum(1 for shape in payload["shapes"] if shape["kind"] == kind)
        for kind in dict.fromkeys(shape["kind"] for shape in payload["shapes"])}
    assert metrics["area_m2"] == payload["data_quality"]["area_m2"]
    assert result["summary"]["metrics"]["relief_m"] == payload["elevation"]["relief_m"]

    coverage = result["sections"]["coverage"]
    payload = real["coverage"]
    assert set(_sections.COVERAGE_FIELDS) <= set(payload)
    assert payload["gaps"], "one track across the valley leaves gaps either side"
    assert set(_sections.GAP_FIELDS) <= set(payload["gaps"][0])
    assert coverage["assignment_kind"] == "area" and 0 < coverage["covered_fraction"] < 1
    assert coverage["gap_count"] == len(payload["gaps"])
    # Every gap point lies on the staged assignment, in WGS84 - which it
    # would not if the outline it came from were not Web Mercator.
    west, north = m(0, 0)
    east, south = m(400, 400)
    for gap in coverage["gaps"]:
        lon, lat = gap["point"]["coordinates"]
        assert west <= lon <= east and south <= lat <= north

    features = result["sections"]["features"]
    payload = real["features"]
    assert list(features["kinds"]) == list(REGISTRY)
    for kind, entry in features["kinds"].items():
        # The valley has no vegetation panel, so its clearings are unknown:
        # null, never the 0 the detector's empty envelope carries.
        unknown = kind == "clearing"
        assert entry["count"] == (None if unknown else len(payload["results"][kind]["shapes"]))
        assert entry["ids"] == [shape["id"] for shape in payload["results"][kind]["shapes"]]
    # A drainage the metrics measured is one the features name.
    assert {d["id"] for d in metrics["hydrology"]["drainages"]} <= set(
        features["kinds"]["drainage"]["ids"])

    assert result["staleness"]["current"] is True
    assert geometry_types_in(result) == {"Point"}
    json.dumps(result)

    # Coverage alone again, now that the full call saved the metrics: the
    # real coverage_for reads the sweep width they saved, and nothing else
    # is computed.
    del real["metrics"], real["coverage"]
    again = debrief(ctx, map_id="M1", include=["coverage"])
    assert again["status"] == "ok", again["sections"]["coverage"]["error"]
    assert "metrics" not in real and "coverage" in real
    assert again["sections"]["coverage"]["effective_sweep_width_m"] == (
        metrics["traverse"]["effective_sweep_width_m"])
    assert again["sections"]["coverage"]["covered_fraction"] == coverage["covered_fraction"]
    assert again["sections"]["coverage"]["gap_count"] == coverage["gap_count"]


def valley_line():
    """A line assignment across the valley, west to east along its middle."""
    from test_boundary import m

    return {"type": "LineString", "coordinates": [m(50, 200), m(350, 200)]}


def test_a_real_line_assignment_s_gap_point_lies_on_the_line(ctx, monkeypatch, tmp_path):
    """The valley as a LINE assignment, its team's track along the west
    third of it: every line field read off what the real coverage_for
    gives, and the gap point - where a team is sent back to - on the line
    itself, in WGS84, in the stretch nobody walked."""
    from shapely.geometry import LineString, Point
    from test_boundary import m

    line = valley_line()
    data_dir = real_valley(monkeypatch, tmp_path, [team_track("101 AB", (50, 200), (150, 200))],
                           subject=line)
    real = keep_real_answers(monkeypatch)
    result = debrief(ctx, map_id="M1")
    assert result["status"] == "ok", {name: entry["error"]
                                      for name, entry in result["sections"].items()}

    coverage = result["sections"]["coverage"]
    payload = real["coverage"]
    assert coverage["assignment_kind"] == payload["assignment_kind"] == "line"
    assert coverage["effort_coverage"] is None and coverage["zone_area_m2"] is None
    assert coverage["line_length_m"] == payload["line_length_m"] > 0
    assert 0 < coverage["covered_fraction"] < 1
    # The corridor is the width the metrics measured, which they saved first.
    assert coverage["effective_sweep_width_m"] == (
        result["sections"]["metrics"]["traverse"]["effective_sweep_width_m"])
    assert payload["gaps"], "a track along the west third leaves the east of the line"
    assert coverage["gap_count"] == len(payload["gaps"])
    assert coverage["gap_length_m"] == pytest.approx(sum(gap["length_m"]
                                                         for gap in payload["gaps"]))
    assert coverage["gap_area_m2"] is None
    assert coverage["closed_gaps"]["area_m2"] is None
    axis = LineString([_TO_3857.transform(*vertex) for vertex in line["coordinates"]])
    for gap in coverage["gaps"]:
        assert gap["area_m2"] is None and gap["length_m"] > 0
        lon, lat = gap["point"]["coordinates"]
        # On the line: within the six decimal places it is rounded to.
        assert axis.distance(Point(_TO_3857.transform(lon, lat))) < 0.5
        assert lon > m(150, 200)[0], "the gap is east of where the track stopped"
    largest = result["summary"]["coverage"]["largest_gap"]
    assert largest["length_m"] == payload["gaps"][0]["length_m"]
    assert largest["area_m2"] is None
    # The section's warnings are coverage_for's own, and no local path.
    assert coverage["warnings"] == payload["warnings"]
    assert str(data_dir) not in json.dumps(result)


def test_coverage_fails_when_the_boundary_could_not_be_checked(ctx, monkeypatch, tmp_path):
    """The map could not be reached after staging (UNCHECKED): the ground is
    on disk, so the metrics and features answer, but the tracks are read off
    the map, so the real coverage_for refuses rather than compute around a
    boundary nobody checked - and the section keeps the call to retry."""
    real_valley(monkeypatch, tmp_path, [team_track("101 AB", (50, 200), (350, 200))],
                check=UNCHECKED)
    result = debrief(ctx, map_id="M1")
    assert result["status"] == "partial"
    coverage = result["sections"]["coverage"]
    assert coverage["status"] == "failed"
    assert coverage["error"] == ("RuntimeError: cannot read the map's tracks: "
                                 + UNCHECKED["reason"])
    assert coverage["detail_call"] == {"tool": "coverage_for", "arguments": {
        "handle": "M1_AB", "track_names": ["101"]}}
    assert result["sections"]["metrics"]["status"] == "ok"
    assert result["sections"]["features"]["status"] == "ok"
    assert result["summary"]["coverage"] is None
    assert result["staleness"] == UNCHECKED
    assert result["warnings"] == [
        "the assignment boundary could not be checked against the live map at the latest "
        f"check ({UNCHECKED['reason']}): every section here (metrics, features) describes "
        "the ground as it was staged, which may not be the assignment as it now stands"]


def test_ground_with_no_openstreetmap_has_unknown_ways_in_not_none(ctx, monkeypatch, tmp_path):
    """No osm.json on the ground: the metrics' access is null and the
    summary's ingress points null - never 0, which would tell a planner the
    assignment has no way in by trail when no road was ever fetched - and
    the missing input is a warning."""
    from sar_geo_mcp.terrain.features.routes import MISSING_OSM_WARNING

    data_dir = real_valley(monkeypatch, tmp_path, [team_track("101 AB", (50, 200), (350, 200))])
    (data_dir / "osm.json").unlink()
    result = debrief(ctx, map_id="M1", include=["metrics"])
    metrics = result["sections"]["metrics"]
    assert metrics["status"] == "ok"
    assert metrics["access"] is None
    assert result["summary"]["metrics"]["ingress_points"] is None
    assert MISSING_OSM_WARNING in metrics["warnings"]


def test_the_access_counts_read_the_routes_staging_writes(ctx, monkeypatch, tmp_path):
    """Routes as the real `routes_for` writes them into osm.json, from an
    Overpass answer: a trail and a road across the valley, each crossing the
    boundary twice, and a creek. The section counts them by the type the
    staging gave them - a renamed key would count every one "unknown" - and
    the ways in are the trail's and the road's crossings, not the creek's."""
    from test_boundary import ORIGIN_X, ORIGIN_Y, m

    from sar_geo_mcp.terrain.linear import OSM_QUERY_VERSION, routes_for

    def node(x, y):
        lon, lat = m(x, y)
        return {"lat": lat, "lon": lon}

    overpass = {"elements": [
        {"type": "way", "id": 1, "tags": {"highway": "path", "name": "Valley Trail"},
         "geometry": [node(-50, 120), node(450, 120)]},
        {"type": "way", "id": 2, "tags": {"highway": "service"},
         "geometry": [node(250, -50), node(250, 450)]},
        {"type": "way", "id": 3, "tags": {"waterway": "stream", "name": "Valley Creek"},
         "geometry": [node(-50, 300), node(450, 300)]}]}
    routes = routes_for((ORIGIN_X - 100, ORIGIN_Y - 500, ORIGIN_X + 500, ORIGIN_Y + 100),
                        cache_dir=tmp_path / "osm-tiles", get_json=lambda query: overpass)
    data_dir = real_valley(monkeypatch, tmp_path, [team_track("101 AB", (50, 200), (350, 200))])
    (data_dir / "osm.json").write_text(json.dumps(
        {"routes": routes, "query_version": OSM_QUERY_VERSION}), encoding="utf-8")
    real = keep_real_answers(monkeypatch)
    result = debrief(ctx, map_id="M1", include=["metrics", "features"],
                     kinds=["route", "ingress"])
    assert result["status"] == "ok"

    access = result["sections"]["metrics"]["access"]
    by_type = {}
    for route in real["metrics"]["access"]["routes"]:
        kind = route["properties"]["route_type"]
        by_type[kind] = by_type.get(kind, 0) + 1
    assert access["routes_by_type"] == by_type == {"TRAIL": 1, "ROAD": 1, "WATERWAY": 1}
    assert access["routes"] == 3
    assert access["route_names"] == ["Valley Trail", "Valley Creek"]
    assert access["ingress_count"] == real["metrics"]["access"]["ingress_count"] == 4
    assert result["summary"]["metrics"]["ingress_points"] == 4
    assert result["sections"]["features"]["kinds"]["ingress"]["count"] == 4
    assert result["sections"]["features"]["kinds"]["route"]["count"] == 3


# --- Which tracks a team name selects -----------------------------------------------


@pytest.mark.parametrize("titles, team, used", [
    (["UAS-WINGTRA FLIGHT 3"], "3", ["UAS-WINGTRA FLIGHT 3"]),
    (["50% (1.2mi) - Hiker"], "50", ["50% (1.2mi) - Hiker"]),
    (["3 AB", "UAS-WINGTRA FLIGHT 3"], "3", ["3 AB"]),
], ids=["drone sortie alone", "probability ring alone", "drone sortie beside the team's own"])
def test_a_weakly_named_line_is_used_only_when_nothing_is_plainly_named(
        ctx, monkeypatch, tmp_path, titles, team, used):
    """The real selection reaches the answer: a line whose name only loosely
    reads as the team's is the fallback when it is all there is, and set
    aside beside a plainly named one. (That either is said: B2, below.)"""
    real_valley(monkeypatch, tmp_path, [team_track(title, (50, 150 + 150 * index),
                                                   (350, 150 + 150 * index))
                                        for index, title in enumerate(titles)])
    result = debrief(ctx, map_id="M1", track_names=[team])
    coverage = result["sections"]["coverage"]
    assert coverage["status"] == "ok", coverage["error"]
    assert [track["name"] for track in coverage["tracks"]] == used
    assert result["summary"]["coverage"]["tracks"] == len(used)
    assert 0 < coverage["covered_fraction"] < 1


def test_a_plainly_named_track_carries_no_warning_about_its_name(ctx, monkeypatch, tmp_path):
    """The control for B2: the team's own track, plainly named, is counted
    with nothing to check."""
    real_valley(monkeypatch, tmp_path, [team_track("3 AB", (50, 200), (350, 200))])
    result = debrief(ctx, map_id="M1", track_names=["3"])
    coverage = result["sections"]["coverage"]
    assert [track["name"] for track in coverage["tracks"]] == ["3 AB"]
    assert coverage["warnings"] == [] and result["warnings"] == []


@pytest.mark.parametrize("title, team", [("UAS-WINGTRA FLIGHT 3", "3"),
                                         ("50% (1.2mi) - Hiker", "50")])
def test_coverage_from_a_weakly_named_line_names_it(ctx, monkeypatch, tmp_path, title, team):
    """Nothing plainly named for the team: the coverage is computed from a
    line whose name reads as something else - a drone sortie, a probability
    ring - and the answer must say which, for the operator to check."""
    real_valley(monkeypatch, tmp_path, [team_track(title, (50, 200), (350, 200))])
    result = debrief(ctx, map_id="M1", track_names=[team])
    coverage = result["sections"]["coverage"]
    assert [track["name"] for track in coverage["tracks"]] == [title]
    assert any(title in warning for warning in coverage["warnings"]), coverage["warnings"]


def test_a_weakly_named_line_set_aside_is_named(ctx, monkeypatch, tmp_path):
    """Beside the team's own track, a drone sortie's is set aside - which
    is how coverage ends up computed from part of a team's work if "FLIGHT 3"
    was theirs, so the answer must name it."""
    real_valley(monkeypatch, tmp_path, [team_track("3 AB", (50, 150), (350, 150)),
                                        team_track("UAS-WINGTRA FLIGHT 3", (50, 300),
                                                   (350, 300))])
    result = debrief(ctx, map_id="M1", track_names=["3"])
    coverage = result["sections"]["coverage"]
    assert [track["name"] for track in coverage["tracks"]] == ["3 AB"]
    assert any("UAS-WINGTRA FLIGHT 3" in warning for warning in coverage["warnings"])


def test_a_line_assignment_s_fallback_to_a_weak_name_is_said_once(ctx, monkeypatch, tmp_path):
    """A line selects its tracks twice (the corridor, and the wider band
    the gap closing reads): the fallback is still one fact, said once."""
    real_valley(monkeypatch, tmp_path, [team_track("UAS-WINGTRA FLIGHT 3", (50, 200),
                                                   (350, 200))], subject=valley_line())
    result = debrief(ctx, map_id="M1", track_names=["3"])
    coverage = result["sections"]["coverage"]
    assert coverage["assignment_kind"] == "line"
    assert len([warning for warning in coverage["warnings"]
                if "UAS-WINGTRA FLIGHT 3" in warning]) == 1, coverage["warnings"]


def letter_only_selection(ctx, track_names):
    """The tracks a debrief counted for `track_names`, and its coverage
    warnings: none when the name is refused before any work."""
    try:
        result = debrief(ctx, map_id="M1", track_names=track_names)
    except ValueError:
        return [], []
    coverage = result["sections"]["coverage"]
    return [track["name"] for track in coverage.get("tracks") or []], coverage["warnings"]


def test_the_assignment_label_as_a_track_name_does_not_count_a_drawn_trail(ctx, monkeypatch,
                                                                           tmp_path):
    """track_names=['AB'] - the assignment label, an easy slip - and the
    map's only line is a drawn trail whose name has "ab" inside a word: it
    is no team's track, so it is not counted silently."""
    title = "Slab Creek trail (drawn)"
    real_valley(monkeypatch, tmp_path, [team_track(title, (50, 200), (350, 200))])
    selected, warnings = letter_only_selection(ctx, ["AB"])
    assert title not in selected or any(title in warning for warning in warnings)


def test_a_team_letter_does_not_select_every_title_with_that_letter(ctx, monkeypatch,
                                                                    tmp_path):
    titles = ["101 BT", "Cabin loop recon", "Helispot alpha"]
    real_valley(monkeypatch, tmp_path, [team_track(title, (50, 150 + 100 * index),
                                                   (350, 150 + 100 * index))
                                        for index, title in enumerate(titles)])
    selected, _ = letter_only_selection(ctx, ["A"])
    assert "Cabin loop recon" not in selected and "Helispot alpha" not in selected


# --- A count whose input is missing is unknown, never 0 ----------------------------


def test_a_kind_whose_input_is_missing_is_counted_unknown():
    """A detector with no input answers an empty envelope and a warning:
    that kind's count is unknown, not a measured 0."""
    from sar_geo_mcp.terrain.features.clearings import MISSING_PANEL_WARNING, ClearingDetector
    from sar_geo_mcp.terrain.features.ingress import IngressDetector
    from sar_geo_mcp.terrain.features.routes import MISSING_OSM_WARNING

    section = {"kinds": {
        "clearing": _sections.compact_kind(
            ClearingDetector().envelope(warnings=[MISSING_PANEL_WARNING]).model_dump()),
        "ingress": _sections.compact_kind(
            IngressDetector().envelope(warnings=[MISSING_OSM_WARNING]).model_dump())}}
    assert _sections.features_facts(section) == {"clearing": None, "ingress": None}


def test_the_summary_s_feature_counts_agree_with_the_metrics_on_unknown_ground(
        ctx, monkeypatch, tmp_path):
    """No osm.json and no vegetation panel: the summary's clearings, routes
    and ways in are null, as the metrics' ingress points are in the same
    answer - while a kind whose input is there keeps its count."""
    data_dir = real_valley(monkeypatch, tmp_path, [team_track("101 AB", (50, 200), (350, 200))])
    (data_dir / "osm.json").unlink()
    result = debrief(ctx, map_id="M1",
                     kinds=["steep_band", "drainage", "clearing", "route", "ingress"])
    assert result["status"] == "ok"
    assert result["summary"]["metrics"]["ingress_points"] is None
    counts = result["summary"]["features"]
    assert isinstance(counts["drainage"], int) and counts["drainage"] >= 1
    assert {kind: counts[kind] for kind in ("clearing", "route", "ingress")} == {
        "clearing": None, "route": None, "ingress": None}


# --- Known defects in what the answer says ----------------------------------------


@pytest.mark.parametrize("metrics", ["left out", "failed"])
def test_a_line_on_the_default_corridor_names_no_local_path_or_command(ctx, monkeypatch,
                                                                       tmp_path, metrics):
    """A line with no width saved runs on coverage_for's 10 m default: the
    section says so, but not in coverage_for's words for the command line -
    the server's own cache path and a debrief-app `--data` command no caller of a
    scenario can run."""
    data_dir = real_valley(monkeypatch, tmp_path,
                           [team_track("101 AB", (50, 200), (150, 200))], subject=valley_line())
    if metrics == "failed":
        def unreadable(ctx, **arguments):
            raise OSError("dem.tif could not be read")

        monkeypatch.setattr(ground_component, "terrain_metrics", unreadable)
        result = debrief(ctx, map_id="M1", include=["metrics", "coverage"])
    else:
        result = debrief(ctx, map_id="M1", include=["coverage"])
    coverage = result["sections"]["coverage"]
    assert coverage["status"] == "ok", coverage["error"]
    assert any("10 m" in warning for warning in coverage["warnings"]), coverage["warnings"]
    answer = json.dumps(result)
    assert [said for said in (str(data_dir), "terrain --data", "--data") if said in answer] == []


def test_tracks_that_miss_the_assignment_are_refused_without_asking_for_a_directory(
        ctx, monkeypatch, tmp_path):
    """The team's only track is kilometres off the assignment: the refusal
    names the team, and asks nothing about a data directory the caller
    never chose."""
    real_valley(monkeypatch, tmp_path, [team_track("101 AB", (5000, 5000), (5300, 5000))])
    result = debrief(ctx, map_id="M1")
    coverage = result["sections"]["coverage"]
    assert coverage["status"] == "failed"
    assert coverage["error"].startswith("TrackSelectionError: ") and "101" in coverage["error"]
    assert "directory" not in coverage["error"]
