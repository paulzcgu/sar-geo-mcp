import json

import pytest

from sar_geo_mcp.coverage.metrics import CoverageMetrics, compute_coverage


def test_metrics_round_trip():
    metrics = CoverageMetrics(
        assignment_label="TG 1211, 1215", teams=["1211"],
        effective_sweep_width_m=10.6, covered_fraction=0.995,
        effort_coverage=15.74, zone_area_m2=20500.0,
        tracks=[{"name": "1211.2 QQ",
                 "length_in_zone_m": 2857.0,
                 "length_total_m": 3563.0, "in_zone_fraction": 0.80}],
        gaps=[], warnings=[],
    )

    restored = CoverageMetrics.model_validate_json(metrics.model_dump_json())

    assert restored.covered_fraction == pytest.approx(0.995)
    assert restored.tracks[0].name == "1211.2 QQ"
    assert not hasattr(restored.tracks[0], "geometry")
    assert restored.schema_version == "1.1.0"


def test_covered_fraction_is_bounded():
    with pytest.raises(Exception):
        CoverageMetrics(assignment_label="A", teams=["1"],
                        effective_sweep_width_m=10.0, covered_fraction=1.5,
                        effort_coverage=1.0, zone_area_m2=100.0,
                        tracks=[], gaps=[], warnings=[])


# --- time bounds (Phase 8b, commit 1) --------------------------------------


def test_track_summary_time_fields_default_to_none_and_round_trip():
    from sar_geo_mcp.coverage.metrics import TrackSummary

    bare = TrackSummary(name="x", length_in_zone_m=1.0, length_total_m=1.0,
                        in_zone_fraction=1.0)
    assert bare.time_entered is None
    assert bare.time_exited is None

    timed = TrackSummary(name="1211.2 QQ", length_in_zone_m=100.0,
                         length_total_m=100.0, in_zone_fraction=1.0,
                         time_entered="2025-06-14T16:33:00Z",
                         time_exited="2025-06-14T20:44:00Z")
    restored = TrackSummary.model_validate_json(timed.model_dump_json())
    assert restored.time_entered == "2025-06-14T16:33:00Z"
    assert restored.time_exited == "2025-06-14T20:44:00Z"


def test_compute_coverage_reports_time_entered_exited_and_searching_hours(tmp_path):
    # Two tracks recorded well inside the zone, with real epoch-ms
    # timestamps spanning 2025-06-14T16:33:00Z .. 20:44:00Z between them (no
    # single track spans the whole window) - CoverageMetrics must report the
    # union: earliest entered, latest exited, across every track.
    _write_terrain_fixture(tmp_path)

    assignment_geometry = {
        "type": "Polygon",
        "coordinates": [[
            [-122.000, 37.700], [-121.999, 37.700],
            [-121.999, 37.701], [-122.000, 37.701], [-122.000, 37.700],
        ]],
    }
    (tmp_path / "assignment.json").write_text(
        json.dumps({"geometry": assignment_geometry}), encoding="utf-8"
    )

    early = {
        "properties": {"title": "1.1 AA", "class": "Shape"},
        "geometry": {"type": "LineString", "coordinates": [
            [-121.9998, 37.7002, 100.0, 1749918780000],   # 16:33:00Z
            [-121.9995, 37.7004, 102.0, 1749924000000],   # 18:00:00Z
        ]},
    }
    late = {
        "properties": {"title": "1.2 BB", "class": "Shape"},
        "geometry": {"type": "LineString", "coordinates": [
            [-121.9993, 37.7006, 103.0, 1749924000000],   # 18:00:00Z
            [-121.9992, 37.7008, 105.0, 1749933840000],   # 20:44:00Z
        ]},
    }

    result = compute_coverage(tmp_path, ["1"], features=[early, late])
    metrics = result.metrics

    assert metrics.time_entered == "2025-06-14T16:33:00Z"
    assert metrics.time_exited == "2025-06-14T20:44:00Z"
    assert metrics.time_searching_hours == pytest.approx(4 + 11 / 60, abs=0.01)
    by_name = {t.name: t for t in metrics.tracks}
    assert by_name["1.1 AA"].time_entered == "2025-06-14T16:33:00Z"
    assert by_name["1.2 BB"].time_exited == "2025-06-14T20:44:00Z"


def test_compute_coverage_time_fields_null_without_any_timestamp_ordinate(tmp_path):
    # Tracks without timestamps -> None, never a guess.
    _write_terrain_fixture(tmp_path)

    assignment_geometry = {
        "type": "Polygon",
        "coordinates": [[
            [-122.000, 37.700], [-121.999, 37.700],
            [-121.999, 37.701], [-122.000, 37.701], [-122.000, 37.700],
        ]],
    }
    (tmp_path / "assignment.json").write_text(
        json.dumps({"geometry": assignment_geometry}), encoding="utf-8"
    )

    no_timestamps = {
        "properties": {"title": "1.1 AA", "class": "Shape"},
        "geometry": {"type": "LineString", "coordinates": [
            [-121.9998, 37.7002],
            [-121.9992, 37.7008],
        ]},
    }

    result = compute_coverage(tmp_path, ["1"], features=[no_timestamps])
    metrics = result.metrics

    assert metrics.time_entered is None
    assert metrics.time_exited is None
    assert metrics.time_searching_hours is None
    assert metrics.tracks[0].time_entered is None
    assert metrics.tracks[0].time_exited is None


def test_compute_coverage_needs_terrain_metrics_for_an_area_assignment(tmp_path):
    # Area coverage is Koopman math over the vegetation-adjusted sweep width;
    # there is no fallback for a missing ESW, unlike the line-assignment
    # corridor default (see test_compute_coverage_line_falls_back_without_terrain).
    assignment_geometry = {
        "type": "Polygon",
        "coordinates": [[
            [-122.000, 37.700], [-121.999, 37.700],
            [-121.999, 37.701], [-122.000, 37.701], [-122.000, 37.700],
        ]],
    }
    (tmp_path / "assignment.json").write_text(
        json.dumps({"geometry": assignment_geometry}), encoding="utf-8"
    )

    with pytest.raises(FileNotFoundError, match="Call the `terrain_metrics` tool on this handle first"):
        compute_coverage(tmp_path, ["1211"], features=[])


def test_compute_coverage_of_a_line_assignment_produces_a_line_document(tmp_path):
    # The training map's 19 assignment features are all LineString, so an
    # operator will point this at one - it must produce a full corridor
    # coverage document, not the ValueError this used to raise (amended
    # 2026-08-17, operator decision: line assignments are in scope).
    _write_terrain_fixture(tmp_path, sweep_width_m=10.0)

    line_geometry = {
        "type": "LineString",
        "coordinates": [[-122.000, 37.700], [-121.999, 37.701]],
    }
    (tmp_path / "assignment.json").write_text(
        json.dumps({"geometry": line_geometry}), encoding="utf-8"
    )

    # Hugs the line for its whole length, well inside the 5 m corridor
    # half-width (ESW 10 m).
    feature = {
        "properties": {"title": "1.1 AA", "class": "Shape"},
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [-122.0000, 37.70003, 100.0, 0],
                [-121.9990, 37.70103, 105.0, 1000],
            ],
        },
    }

    result = compute_coverage(tmp_path, ["1"], features=[feature])
    metrics = result.metrics

    assert metrics.assignment_kind == "line"
    # A line document carries no area and no Koopman effort number - both
    # must be null, not a coherent-looking 0.0.
    assert metrics.zone_area_m2 is None
    assert metrics.effort_coverage is None
    assert metrics.line_length_m is not None and metrics.line_length_m > 0
    assert metrics.covered_fraction == pytest.approx(1.0, abs=0.01)
    assert metrics.gaps == []
    # Regression (Phase 9i): a fully-covered line has nothing to close -
    # closing must never invent a closed_gaps entry where there was no gap.
    assert metrics.closed_gaps == []
    for gap in metrics.gaps:
        assert gap.length_m is not None
        assert gap.area_m2 is None


def test_compute_coverage_of_an_area_assignment_still_reports_area_kind(tmp_path):
    _write_terrain_fixture(tmp_path)

    assignment_geometry = {
        "type": "Polygon",
        "coordinates": [[
            [-122.000, 37.700], [-121.999, 37.700],
            [-121.999, 37.701], [-122.000, 37.701], [-122.000, 37.700],
        ]],
    }
    (tmp_path / "assignment.json").write_text(
        json.dumps({"geometry": assignment_geometry}), encoding="utf-8"
    )

    feature = {
        "properties": {"title": "1.1 AA", "class": "Shape"},
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [-121.9998, 37.7002, 100.0, 0],
                [-121.9992, 37.7008, 105.0, 1000],
            ],
        },
    }

    result = compute_coverage(tmp_path, ["1"], features=[feature])
    metrics = result.metrics

    assert metrics.assignment_kind == "area"
    assert metrics.line_length_m is None
    assert metrics.zone_area_m2 is not None and metrics.zone_area_m2 > 0
    assert metrics.effort_coverage is not None


def test_compute_coverage_line_falls_back_without_terrain(tmp_path):
    # The terrain step is polygon math and may never have been run for a
    # line assignment. Coverage must still work, using the 10 m default
    # corridor (5 m half-width) and saying so - unless corridor_width_m was
    # given (see test_compute_coverage_line_corridor_width_overrides_terrain).
    line_geometry = {
        "type": "LineString",
        "coordinates": [[-122.000, 37.700], [-121.999, 37.701]],
    }
    (tmp_path / "assignment.json").write_text(
        json.dumps({"geometry": line_geometry}), encoding="utf-8"
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"map_id": "M1", "assignment": "LINE1", "latitude_deg": 37.7}),
        encoding="utf-8",
    )

    feature = {
        "properties": {"title": "1.1 AA", "class": "Shape"},
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [-122.0000, 37.70003, 100.0, 0],
                [-121.9990, 37.70103, 105.0, 1000],
            ],
        },
    }

    result = compute_coverage(tmp_path, ["1"], features=[feature])
    metrics = result.metrics

    assert metrics.assignment_kind == "line"
    assert metrics.assignment_label == "LINE1"
    assert metrics.effective_sweep_width_m == 10.0
    assert any("default" in w and "10 m" in w for w in metrics.warnings)


def test_a_line_s_name_warning_comes_before_its_default_corridor(tmp_path):
    """As on an area (PR #4 review B2): whether these are the team's lines
    decides whether any number below - the default corridor's included -
    is the team's coverage, so that is said first."""
    line_geometry = {
        "type": "LineString",
        "coordinates": [[-122.000, 37.700], [-121.999, 37.701]],
    }
    (tmp_path / "assignment.json").write_text(
        json.dumps({"geometry": line_geometry}), encoding="utf-8"
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"map_id": "M1", "assignment": "LINE1", "latitude_deg": 37.7}),
        encoding="utf-8",
    )
    feature = {
        "properties": {"title": "UAS-WINGTRA FLIGHT 1", "class": "Shape"},
        "geometry": {"type": "LineString",
                     "coordinates": [[-122.0000, 37.70003, 100.0, 0],
                                     [-121.9990, 37.70103, 105.0, 1000]]},
    }

    warnings = compute_coverage(tmp_path, ["1"], features=[feature]).metrics.warnings

    assert warnings[0].startswith("no track is plainly named for team 1, so this coverage "
                                  "counts UAS-WINGTRA FLIGHT 1")
    assert "default" in warnings[1] and "10 m" in warnings[1]


def test_compute_coverage_line_corridor_width_overrides_terrain(tmp_path):
    # corridor_width_m is an explicit operator choice and wins even when
    # terrain (and therefore a vegetation-adjusted ESW) is available.
    _write_terrain_fixture(tmp_path, sweep_width_m=10.0)

    line_geometry = {
        "type": "LineString",
        "coordinates": [[-122.000, 37.700], [-121.999, 37.701]],
    }
    (tmp_path / "assignment.json").write_text(
        json.dumps({"geometry": line_geometry}), encoding="utf-8"
    )

    feature = {
        "properties": {"title": "1.1 AA", "class": "Shape"},
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [-122.0000, 37.70003, 100.0, 0],
                [-121.9990, 37.70103, 105.0, 1000],
            ],
        },
    }

    result = compute_coverage(tmp_path, ["1"], features=[feature], corridor_width_m=30.0)

    assert result.metrics.effective_sweep_width_m == 30.0
    assert not any("default" in w for w in result.metrics.warnings)


def _write_terrain_fixture(data_dir, latitude_deg=37.7, sweep_width_m=10.0):
    from sar_geo_mcp.terrain.metrics import (
        AspectMetrics, DataQuality, ElevationMetrics, SlopeMetrics,
        TerrainMetrics, TraverseMetrics, VegetationMetrics,
    )

    terrain = TerrainMetrics(
        assignment_label="T1",
        elevation=ElevationMetrics(min_m=100.0, max_m=200.0, relief_m=100.0),
        slope=SlopeMetrics(mean_deg=10.0, p90_deg=20.0, max_deg=30.0, distribution={}),
        aspect=AspectMetrics(dominant="N", distribution={}),
        vegetation=VegetationMetrics(),
        traverse=TraverseMetrics(
            baseline_rate_kmh=3.0, vegetation_adjusted_rate_kmh=2.0,
            effective_sweep_width_m=sweep_width_m,
            sweep_time_hours_per_pass=1.0, method="test", assumptions={},
        ),
        shapes=[],
        data_quality=DataQuality(dem_resolution_m=2.5, latitude_deg=latitude_deg,
                                 area_m2=10000.0, warnings=[]),
    )
    (data_dir / "terrain_metrics.json").write_text(
        terrain.model_dump_json(), encoding="utf-8"
    )


def test_compute_coverage_projects_wgs84_features_into_the_zone(tmp_path):
    # CalTopo features are WGS84 lon/lat; the zone compute_coverage builds
    # is Web Mercator. Without reprojecting the features first, a real
    # track's coordinates can never spatially intersect the zone at all.
    _write_terrain_fixture(tmp_path)

    assignment_geometry = {
        "type": "Polygon",
        "coordinates": [[
            [-122.000, 37.700], [-121.999, 37.700],
            [-121.999, 37.701], [-122.000, 37.701], [-122.000, 37.700],
        ]],
    }
    (tmp_path / "assignment.json").write_text(
        json.dumps({"geometry": assignment_geometry}), encoding="utf-8"
    )

    feature = {
        "properties": {"title": "1.1 AA", "class": "Shape"},
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [-121.9998, 37.7002, 100.0, 0],
                [-121.9992, 37.7008, 105.0, 1000],
            ],
        },
    }

    result = compute_coverage(tmp_path, ["1"], features=[feature])
    metrics = result.metrics

    assert metrics.tracks[0].name == "1.1 AA"
    assert metrics.covered_fraction > 0.0
    # Track geometry lives in the sidecar, not the digest. It must be
    # projected: Web Mercator x is on the order of millions of metres, not
    # WGS84 degrees (~-122).
    assert len(result.track_geojson["features"]) == 1
    sidecar_feature = result.track_geojson["features"][0]
    assert sidecar_feature["properties"]["name"] == "1.1 AA"
    x = sidecar_feature["geometry"]["coordinates"][0][0]
    assert abs(x) > 1000


def test_compute_coverage_includes_a_mostly_outside_track_with_an_informational_note(tmp_path):
    # No transit classification (operator decision, spec SS5.2 amended
    # 2026-08-17): a track that mostly misses the assignment is still
    # included and contributes its real (if small) coverage. It is
    # surfaced as an informational note for the debrief agent, not
    # excluded and not an error.
    _write_terrain_fixture(tmp_path)

    assignment_geometry = {
        "type": "Polygon",
        "coordinates": [[
            [-122.000, 37.700], [-121.999, 37.700],
            [-121.999, 37.701], [-122.000, 37.701], [-122.000, 37.700],
        ]],
    }
    (tmp_path / "assignment.json").write_text(
        json.dumps({"geometry": assignment_geometry}), encoding="utf-8"
    )

    # Only grazes the zone corner - well under the 10% informational
    # threshold (MOSTLY_OUTSIDE_FRACTION).
    feature = {
        "properties": {"title": "1.1 AA", "class": "Shape"},
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [-121.9998, 37.7002, 100.0, 0],
                [-119.0000, 40.0000, 105.0, 1000],
            ],
        },
    }

    result = compute_coverage(tmp_path, ["1"], features=[feature])

    assert [t.name for t in result.metrics.tracks] == ["1.1 AA"]
    assert result.metrics.tracks[0].in_zone_fraction <= 0.10
    assert any(
        "1.1 AA" in w and "transit recording?" in w
        for w in result.metrics.warnings
    )


def _dense_track_feature(title: str, base_lon: float, n_points: int = 2000):
    # A real GPS sweep track: a couple thousand points, zig-zagging so it
    # covers ground rather than running in one straight line - the same
    # shape that made TrackSummary.geometry 1,460x the rest of the digest.
    coordinates = []
    for i in range(n_points):
        lon = base_lon + (i % 50) * 0.000004
        lat = 37.7000 + (i / n_points) * 0.0009
        coordinates.append([lon, lat, 100.0 + i * 0.01, 1_700_000_000_000 + i * 1000])
    return {
        "properties": {"title": title, "class": "Shape"},
        "geometry": {"type": "LineString", "coordinates": coordinates},
    }


def test_serialised_metrics_stay_small_for_a_realistic_multi_track_case(tmp_path):
    # This is the digest re-billed every debrief turn (see coverage.metrics
    # module docstring). Seven dense tracks is what the real TQ1211-1215 run
    # looks like; before track geometry moved to the sidecar this file was
    # 5.6 MB. It must stay small regardless of how many points the raw GPS
    # tracks have.
    _write_terrain_fixture(tmp_path)

    assignment_geometry = {
        "type": "Polygon",
        "coordinates": [[
            [-122.000, 37.700], [-121.995, 37.700],
            [-121.995, 37.705], [-122.000, 37.705], [-122.000, 37.700],
        ]],
    }
    (tmp_path / "assignment.json").write_text(
        json.dumps({"geometry": assignment_geometry}), encoding="utf-8"
    )

    features = [_dense_track_feature(f"1.{i} AA", -121.9990 + i * 0.0001)
                for i in range(7)]

    result = compute_coverage(tmp_path, ["1"], features=features)

    assert len(result.metrics.tracks) == 7
    serialised = result.metrics.model_dump_json()
    assert len(serialised.encode("utf-8")) < 20_000, (
        f"digest is {len(serialised.encode('utf-8'))} bytes; track geometry "
        "must not leak back into CoverageMetrics"
    )


def _shape(shape_id: str, geometry) -> "MetricShape":
    from shapely.geometry import mapping

    from sar_geo_mcp.terrain.metrics import MetricShape

    return MetricShape(id=shape_id, kind="steep", properties={},
                       geometry=mapping(geometry))


def _terrain_with_shapes(shapes) -> "TerrainMetrics":
    from sar_geo_mcp.terrain.metrics import (
        AspectMetrics, DataQuality, ElevationMetrics, SlopeMetrics,
        TerrainMetrics, TraverseMetrics, VegetationMetrics,
    )

    return TerrainMetrics(
        assignment_label="T1",
        elevation=ElevationMetrics(min_m=100.0, max_m=200.0, relief_m=100.0),
        slope=SlopeMetrics(mean_deg=10.0, p90_deg=20.0, max_deg=30.0, distribution={}),
        aspect=AspectMetrics(dominant="N", distribution={}),
        vegetation=VegetationMetrics(),
        traverse=TraverseMetrics(
            baseline_rate_kmh=3.0, vegetation_adjusted_rate_kmh=2.0,
            effective_sweep_width_m=10.0,
            sweep_time_hours_per_pass=1.0, method="test", assumptions={},
        ),
        shapes=shapes,
        data_quality=DataQuality(dem_resolution_m=2.5, latitude_deg=37.7,
                                 area_m2=10000.0, warnings=[]),
    )


def test_attribute_gaps_reports_the_fraction_of_a_shape_thats_unsearched():
    from shapely.geometry import Polygon, mapping

    from sar_geo_mcp.coverage.metrics import GapSummary, _attribute_gaps

    # A 10x10 shape (area 100); the gap covers exactly its right half
    # (x in [5, 10]), area 50 -> half the shape's own extent is unsearched.
    half_in_shape = _shape("steep_35_0", Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]))
    terrain = _terrain_with_shapes([half_in_shape])
    gap = GapSummary(id="gap_0", area_m2=50.0,
                     geometry=mapping(Polygon([(5, 0), (15, 0), (15, 10), (5, 10)])))

    _attribute_gaps([gap], terrain)

    assert len(gap.unsearched_shapes) == 1
    assert gap.unsearched_shapes[0].id == "steep_35_0"
    assert gap.unsearched_shapes[0].fraction_in_gap == pytest.approx(0.5, abs=0.01)


def test_attribute_gaps_excludes_a_shape_that_only_touches_the_gap_edge():
    from shapely.geometry import Polygon, mapping

    from sar_geo_mcp.coverage.metrics import GapSummary, _attribute_gaps

    # Shares only the edge x=10 with the gap - zero-area intersection, not
    # a meaningful overlap. These gap polygons have thousands of stair-step
    # vertices, so an adjacency hit like this is likely on real data.
    touching_shape = _shape("steep_35_1", Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]))
    terrain = _terrain_with_shapes([touching_shape])
    gap = GapSummary(id="gap_0", area_m2=100.0,
                     geometry=mapping(Polygon([(10, 0), (20, 0), (20, 10), (10, 10)])))

    _attribute_gaps([gap], terrain)

    assert gap.unsearched_shapes == []


def test_attribute_gaps_sorts_by_fraction_descending():
    from shapely.geometry import Polygon, mapping

    from sar_geo_mcp.coverage.metrics import GapSummary, _attribute_gaps

    mostly_in = _shape("steep_a", Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]))
    partly_in = _shape("steep_b", Polygon([(20, 0), (60, 0), (60, 10), (20, 10)]))
    terrain = _terrain_with_shapes([partly_in, mostly_in])
    # Gap covers all of steep_a (area 100, fraction 1.0) and a quarter of
    # steep_b (area 400; overlap is x in [20,30], area 100 -> fraction 0.25).
    gap = GapSummary(id="gap_0", area_m2=200.0,
                     geometry=mapping(Polygon([(0, 0), (30, 0), (30, 10), (0, 10)])))

    _attribute_gaps([gap], terrain)

    assert [o.id for o in gap.unsearched_shapes] == ["steep_a", "steep_b"]
    assert gap.unsearched_shapes[0].fraction_in_gap == pytest.approx(1.0)
    assert gap.unsearched_shapes[1].fraction_in_gap == pytest.approx(0.25)


# --- is_stale (Phase 9d, operator bug report: a coverage_metrics.json that
# predates the Phase 8b timestamp feature was consumed silently, leaving
# SAR-110 fields 11-13 blank) ------------------------------------------------


def _metrics(schema_version="1.1.0", time_entered="2025-06-14T16:33:00Z",
            time_exited="2025-06-14T17:53:00Z", tracks=None):
    from sar_geo_mcp.coverage.metrics import CoverageMetrics

    return CoverageMetrics(
        schema_version=schema_version, assignment_label="QA103",
        assignment_kind="line", teams=["103"], effective_sweep_width_m=10.0,
        covered_fraction=0.4, effort_coverage=None, zone_area_m2=None,
        tracks=tracks if tracks is not None else [
            {"name": "103XN", "length_in_zone_m": 100.0, "length_total_m": 200.0,
             "in_zone_fraction": 0.5},
        ],
        gaps=[], warnings=[], time_entered=time_entered, time_exited=time_exited,
    )


def test_current_schema_with_real_times_is_not_stale():
    from sar_geo_mcp.coverage.metrics import is_stale

    assert is_stale(_metrics()) is False


def test_pre_timestamp_schema_version_is_stale_even_with_no_tracks():
    from sar_geo_mcp.coverage.metrics import is_stale

    metrics = _metrics(schema_version="1.0.0", time_entered=None, time_exited=None,
                       tracks=[])
    assert is_stale(metrics) is True


def test_pre_timestamp_schema_version_is_stale_even_if_times_happen_to_be_set():
    # A defensive case: schema_version alone is the direct signal - it does
    # not matter whether this particular file happens to carry real times.
    from sar_geo_mcp.coverage.metrics import is_stale

    assert is_stale(_metrics(schema_version="1.0.0")) is True


def test_current_schema_with_null_times_and_real_tracks_is_stale():
    # Defence in depth: even a current-schema document is flagged if it
    # somehow carries no time window despite having selected tracks - the
    # QA103 bug (nulls silently reaching the SAR-110 form) either way.
    from sar_geo_mcp.coverage.metrics import is_stale

    metrics = _metrics(time_entered=None, time_exited=None)
    assert is_stale(metrics) is True


def test_current_schema_with_null_times_and_no_tracks_is_not_stale():
    # No tracks at all means there is nothing to have timestamped - a
    # legitimately empty run, not a stale file.
    from sar_geo_mcp.coverage.metrics import is_stale

    metrics = _metrics(time_entered=None, time_exited=None, tracks=[])
    assert is_stale(metrics) is False


def test_malformed_schema_version_string_is_treated_as_stale_not_a_crash():
    from sar_geo_mcp.coverage.metrics import is_stale

    metrics = _metrics(schema_version="not-a-version")
    assert is_stale(metrics) is True  # never crashes; sorts as ancient


# --- closed_gaps (Phase 9i, GPS-noise gap closing) --------------------------


def test_closed_gap_round_trips_and_defaults_to_empty():
    from sar_geo_mcp.coverage.metrics import ClosedGap

    bare_metrics = _metrics()
    assert bare_metrics.closed_gaps == []

    closed = ClosedGap(id="closed_gap_0", length_m=32.0, max_lateral_m=7.0,
                       geometry={"type": "LineString", "coordinates": [[0, 0], [1, 0]]})
    restored = ClosedGap.model_validate_json(closed.model_dump_json())
    assert restored.length_m == pytest.approx(32.0)
    assert restored.max_lateral_m == pytest.approx(7.0)
    assert restored.area_m2 is None  # a line closed-gap carries no area - honest null


def test_closed_gaps_warning_summarises_line_closures():
    from sar_geo_mcp.coverage.metrics import ClosedGap, _closed_gaps_warning

    closed = [
        ClosedGap(id="closed_gap_0", length_m=26.0, max_lateral_m=5.2,
                  geometry={"type": "LineString", "coordinates": [[0, 0], [1, 0]]}),
        ClosedGap(id="closed_gap_1", length_m=26.0, max_lateral_m=15.2,
                  geometry={"type": "LineString", "coordinates": [[0, 0], [1, 0]]}),
    ]

    warnings = _closed_gaps_warning(closed)

    assert len(warnings) == 1
    assert "closed 2" in warnings[0]
    assert "52 m" in warnings[0]          # 26 + 26 total length
    assert "15" in warnings[0]            # worst (max) lateral distance, 15.2 -> 15
    assert "closed_gaps" in warnings[0]


def test_closed_gaps_warning_is_empty_when_nothing_closed():
    from sar_geo_mcp.coverage.metrics import _closed_gaps_warning

    assert _closed_gaps_warning([]) == []


def test_closed_area_warning_summarises_area_closures():
    from sar_geo_mcp.coverage.metrics import ClosedGap, _closed_area_warning

    closed = [
        ClosedGap(id="closed_area_0", area_m2=80.0,
                  geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}),
    ]

    warnings = _closed_area_warning(closed)

    assert len(warnings) == 1
    assert "closed 1" in warnings[0]
    assert "80 m" in warnings[0]
    assert "closed_gaps" in warnings[0]


def test_closed_area_warning_is_empty_when_nothing_closed():
    from sar_geo_mcp.coverage.metrics import _closed_area_warning

    assert _closed_area_warning([]) == []


def test_compute_coverage_of_a_line_assignment_closes_gps_noise_gaps(tmp_path):
    # End-to-end: a real GPS-noise gap (short, interior, laterally
    # consistent - the QB102 pattern) must disappear from `gaps`, appear in
    # `closed_gaps` with the right fields, raise covered_fraction, and be
    # named in `warnings` - not just at the corridor.py unit level.
    import pyproj
    from shapely.geometry import LineString
    from shapely.ops import transform as shapely_transform

    _write_terrain_fixture(tmp_path, latitude_deg=37.7, sweep_width_m=10.0)

    to_wgs84 = pyproj.Transformer.from_crs(3857, 4326, always_xy=True).transform
    from sar_geo_mcp.geo import ground_meters_per_mercator_unit
    factor = ground_meters_per_mercator_unit(37.7)

    # Build the route and a track that hugs it, detouring 7 ground-metres
    # off for a 44 ground-metre interior stretch (closes per corridor.py's
    # own tests) - constructed in a local Mercator-like frame around a real
    # basemap point, then projected to WGS84 exactly via pyproj so the
    # ground distances land where intended after compute_coverage's own
    # WGS84->Mercator reprojection.
    base_x, base_y = pyproj.Transformer.from_crs(4326, 3857, always_xy=True).transform(
        -122.000, 37.700
    )

    def pt(x_ground, y_ground):
        return to_wgs84(base_x + x_ground / factor, base_y + y_ground / factor)

    route_ground = [(0, 0), (500, 0)]
    route_wgs84 = [pt(x, y) for x, y in route_ground]
    (tmp_path / "assignment.json").write_text(
        json.dumps({"geometry": {"type": "LineString", "coordinates": route_wgs84}}),
        encoding="utf-8",
    )

    track_ground = [(0, 0.5), (237, 0.5), (237, 7.0), (281, 7.0), (281, 0.5), (500, 0.5)]
    track_wgs84 = [[*pt(x, y), 100.0, i * 1000] for i, (x, y) in enumerate(track_ground)]
    feature = {
        "properties": {"title": "1.1 AA", "class": "Shape"},
        "geometry": {"type": "LineString", "coordinates": track_wgs84},
    }

    result = compute_coverage(tmp_path, ["1"], features=[feature])
    metrics = result.metrics

    assert metrics.gaps == []
    assert len(metrics.closed_gaps) == 1
    closed = metrics.closed_gaps[0]
    assert 25.0 < closed.length_m < 50.0
    assert closed.max_lateral_m == pytest.approx(7.0, abs=1.0)
    assert closed.area_m2 is None
    assert metrics.covered_fraction == pytest.approx(1.0, abs=0.02)
    assert any("closed 1" in w and "closed_gaps" in w for w in metrics.warnings)


# --- regression: fully-covered / zero-coverage behave unchanged (Phase 9i) --


def test_compute_coverage_zero_coverage_line_has_no_closed_gaps(tmp_path):
    # A track that only grazes the very start of the route: almost the
    # whole line is one uncovered run touching the route's OWN end, so it
    # is never interior and must never close, regardless of length/lateral
    # - this is exactly the "genuinely unsearched tail" case closing exists
    # to protect (QB102's real gap_0, 3952 m, behaves the same way).
    _write_terrain_fixture(tmp_path, latitude_deg=37.7, sweep_width_m=10.0)

    line_geometry = {
        "type": "LineString",
        "coordinates": [[-122.000, 37.700], [-121.990, 37.700]],
    }
    (tmp_path / "assignment.json").write_text(
        json.dumps({"geometry": line_geometry}), encoding="utf-8"
    )
    # A short stub right at the route's start only.
    feature = {
        "properties": {"title": "1.1 AA", "class": "Shape"},
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [-122.0000, 37.70000, 100.0, 0],
                [-121.9995, 37.70000, 100.0, 500],
            ],
        },
    }

    result = compute_coverage(tmp_path, ["1"], features=[feature])
    metrics = result.metrics

    assert metrics.covered_fraction < 0.2
    assert len(metrics.gaps) == 1
    assert metrics.closed_gaps == []
    assert not any("closed" in w and "closed_gaps" in w for w in metrics.warnings)


def test_compute_coverage_fully_covered_area_has_no_closed_gaps(tmp_path):
    # Regression: a swath that already covers the entire zone has no
    # slivers for close_gaps to find - closing must not invent one, and
    # covered_fraction/gaps must come back exactly as before this feature.
    _write_terrain_fixture(tmp_path, sweep_width_m=100.0)

    assignment_geometry = {
        "type": "Polygon",
        "coordinates": [[
            [-122.0000, 37.7000], [-121.9995, 37.7000],
            [-121.9995, 37.7005], [-122.0000, 37.7005], [-122.0000, 37.7000],
        ]],
    }
    (tmp_path / "assignment.json").write_text(
        json.dumps({"geometry": assignment_geometry}), encoding="utf-8"
    )
    # A wide-ESW track straight across the middle of a small zone -
    # comfortably sweeps the whole thing.
    feature = {
        "properties": {"title": "1.1 AA", "class": "Shape"},
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [-122.0002, 37.70025, 100.0, 0],
                [-121.9993, 37.70025, 100.0, 1000],
            ],
        },
    }

    result = compute_coverage(tmp_path, ["1"], features=[feature])
    metrics = result.metrics

    assert metrics.covered_fraction == pytest.approx(1.0, abs=0.01)
    assert metrics.gaps == []
    assert metrics.closed_gaps == []


def test_compute_coverage_near_zero_coverage_area_has_no_closed_gaps(tmp_path):
    # A track that only grazes the zone's corner - almost the entire zone
    # is one big uncovered gap, far larger than anything a sweep-width-scale
    # closing disc could bridge.
    _write_terrain_fixture(tmp_path, sweep_width_m=10.0)

    assignment_geometry = {
        "type": "Polygon",
        "coordinates": [[
            [-122.000, 37.700], [-121.999, 37.700],
            [-121.999, 37.701], [-122.000, 37.701], [-122.000, 37.700],
        ]],
    }
    (tmp_path / "assignment.json").write_text(
        json.dumps({"geometry": assignment_geometry}), encoding="utf-8"
    )
    feature = {
        "properties": {"title": "1.1 AA", "class": "Shape"},
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [-121.9998, 37.7002, 100.0, 0],
                [-119.0000, 40.0000, 105.0, 1000],
            ],
        },
    }

    result = compute_coverage(tmp_path, ["1"], features=[feature])
    metrics = result.metrics

    assert metrics.covered_fraction < 0.15
    assert len(metrics.gaps) == 1
    assert metrics.closed_gaps == []


# --- a gap overlap says WHAT it is, not just which id -----------------------
#
# The feature registry owns what kinds exist; `steep_35_0` is a steep band
# and `cliff_2` a cliff band only because of an id convention, and an id
# format is not an interface. `_attribute_gaps` has the kind in hand while
# it builds these, so it carries it.

def test_a_gap_overlap_carries_the_registry_kind():
    from shapely.geometry import Polygon, mapping

    from sar_geo_mcp.coverage.metrics import GapSummary, _attribute_gaps
    from sar_geo_mcp.terrain.metrics import MetricShape

    inside = MetricShape(id="drainage_0", kind="drainage", properties={},
                         geometry=mapping(Polygon([(0, 0), (10, 0),
                                                   (10, 10), (0, 10)])))
    terrain = _terrain_with_shapes([inside])
    gap = GapSummary(id="gap_0", area_m2=100.0,
                     geometry=mapping(Polygon([(0, 0), (10, 0),
                                               (10, 10), (0, 10)])))

    _attribute_gaps([gap], terrain)

    assert gap.unsearched_shapes[0].kind == "drainage"


def test_an_older_coverage_document_still_loads_without_a_kind():
    """Real coverage_metrics.json files on disk predate the field. An empty
    kind means "written by an older version", never "no kind"."""
    from sar_geo_mcp.coverage.metrics import ShapeOverlap

    overlap = ShapeOverlap(id="steep_35_0", fraction_in_gap=0.5)

    assert overlap.kind == ""
