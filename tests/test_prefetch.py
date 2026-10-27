"""Prefetch (F5).

Two properties carry the weight here. First, that nothing downloads until
somebody says so - the plan is an estimate and must cost nothing. Second,
that a marker somebody dropped on a hospital two counties away does not
drag that whole corridor into the download, and that when it is excluded
the caller is TOLD, because a silently narrowed prefetch is how a laptop
reaches a trailhead half provisioned.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sar_geo_mcp.prefetch import (
    ORPHAN_LIMIT_M,
    REQUEST_PIXELS,
    buffered_bbox,
    cluster_features,
    partition,
    plan_prefetch,
    run_prefetch,
)
from sar_geo_mcp.tiles import level_for, tiles_covering

# El Corte, in WGS84 - the ground this project was built against.
WEST, SOUTH = -122.3164, 37.4022


def square(west: float, south: float, size_deg: float = 0.01) -> dict:
    return {"type": "Polygon", "coordinates": [[
        [west, south], [west + size_deg, south],
        [west + size_deg, south + size_deg], [west, south + size_deg],
        [west, south],
    ]]}


def feature(feature_class: str, title: str, west: float, south: float,
            size_deg: float = 0.01) -> dict:
    return {"id": title, "geometry": square(west, south, size_deg),
            "properties": {"class": feature_class, "title": title}}


ASSIGNMENT = feature("Assignment", "QA103", WEST, SOUTH)
NEIGHBOUR = feature("Assignment", "AD104", WEST + 0.012, SOUTH)
NEARBY_MARKER = feature("Marker", "trailhead", WEST - 0.005, SOUTH, 0.0005)
#: Two degrees of longitude away - roughly 175 km. Somebody's hospital pin.
FAR_MARKER = feature("Marker", "hospital", WEST - 2.0, SOUTH, 0.0005)


# ---------------------------------------------------------------------------
# Working out the ground.
# ---------------------------------------------------------------------------

def test_an_assignment_is_buffered_more_widely_than_a_marker():
    """Searchers leave the boundary and coverage needs the ground around
    it; a marker only needs enough context to be readable."""
    assignment_box, _, _ = buffered_bbox(ASSIGNMENT)
    marker_box, _, _ = buffered_bbox(NEARBY_MARKER)

    assignment_margin = assignment_box[2] - assignment_box[0]
    marker_margin = marker_box[2] - marker_box[0]
    assert assignment_margin > marker_margin


def test_a_feature_without_usable_geometry_is_skipped_not_fatal():
    """Plenty of CalTopo objects carry no geometry. That is not an error."""
    assert buffered_bbox({"properties": {"class": "Marker"}}) is None
    assert buffered_bbox({"geometry": {}, "properties": {}}) is None


def test_features_on_the_same_ground_become_one_cluster():
    entries = [buffered_bbox(f) for f in (ASSIGNMENT, NEIGHBOUR, NEARBY_MARKER)]
    clusters = cluster_features([e for e in entries if e])

    assert len(clusters) == 1
    assert clusters[0].has_assignment
    assert set(clusters[0].labels) == {"QA103", "AD104", "trailhead"}


def test_clustering_merges_chains_not_just_pairs():
    """Two extents can become neighbours only once a third joins them. A
    single pass would leave that chain as separate clusters and prefetch
    the middle of a search area twice."""
    entries = [buffered_bbox(feature("Shape", f"leg{i}", WEST + i * 0.012, SOUTH))
               for i in range(4)]
    clusters = cluster_features([e for e in entries if e])
    assert len(clusters) == 1, "a chain of adjacent extents is one piece of country"


def test_a_far_marker_forms_its_own_cluster_and_is_skipped_with_a_reason():
    """The headline guard: one pin must not cost a region."""
    entries = [buffered_bbox(f) for f in (ASSIGNMENT, FAR_MARKER)]
    keep, skipped = partition(cluster_features([e for e in entries if e]))

    assert [c.labels for c in keep] == [["QA103"]]
    assert len(skipped) == 1
    cluster, reason = skipped[0]
    assert cluster.labels == ["hospital"]
    assert "km" in reason, "a skip must say why, not just happen"


def test_a_nearby_marker_is_kept_even_though_it_has_no_assignment():
    """A trailhead beside the search is worth having offline."""
    entries = [buffered_bbox(f) for f in (ASSIGNMENT, NEARBY_MARKER)]
    keep, skipped = partition(cluster_features([e for e in entries if e]))

    assert skipped == []
    assert len(keep) == 1


def test_a_map_with_no_assignments_at_all_keeps_everything():
    """Prefetching around markers and lines is a deliberate act, not an
    accident to be second-guessed."""
    entries = [buffered_bbox(f) for f in (NEARBY_MARKER, FAR_MARKER)]
    keep, skipped = partition(cluster_features([e for e in entries if e]))

    assert skipped == []
    assert len(keep) == 2


# ---------------------------------------------------------------------------
# Planning costs nothing.
# ---------------------------------------------------------------------------

def test_planning_downloads_nothing(tmp_path):
    """The estimate must be free, or nobody will ask for it first."""
    plan = plan_prefetch("M1", [ASSIGNMENT, NEIGHBOUR], cache_dir=tmp_path)

    assert plan.missing_tiles > 0
    assert list(tmp_path.rglob("*.tif")) == [], "planning wrote a tile"
    assert list(tmp_path.rglob("*.png")) == []


def test_the_summary_reports_size_clusters_and_skips_before_any_fetch(tmp_path):
    plan = plan_prefetch("M1", [ASSIGNMENT, FAR_MARKER], cache_dir=tmp_path)
    summary = plan.summary()

    assert summary["estimated_mb"] > 0
    assert summary["tiles_missing"] > 0
    assert len(summary["clusters"]) == 1
    assert len(summary["skipped"]) == 1
    assert "hospital" in summary["skipped"][0]["labels"]


def test_the_prefetch_level_matches_what_a_real_request_will_ask_for(tmp_path):
    """Prefetching at a level nobody requests is an expensive way to miss.

    The pipeline renders each assignment at REQUEST_PIXELS across its own
    buffered extent, so that is the level whose tiles must land in the
    cache.
    """
    box, _, _ = buffered_bbox(ASSIGNMENT)
    expected = {t for t in tiles_covering(box, level_for(box, REQUEST_PIXELS,
                                                         REQUEST_PIXELS))}

    plan = plan_prefetch("M1", [ASSIGNMENT], cache_dir=tmp_path)
    dem = next(layer for layer in plan.layers if layer.layer == "dem")
    assert set(dem.missing) == expected


def test_already_cached_tiles_are_counted_not_refetched(tmp_path):
    plan = plan_prefetch("M1", [ASSIGNMENT], cache_dir=tmp_path)
    first = plan.missing_tiles

    for tile in plan.layers[0].missing[:2]:
        path = tmp_path / tile.key("dem", "tif")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"cached")

    again = plan_prefetch("M1", [ASSIGNMENT], cache_dir=tmp_path)
    assert again.missing_tiles == first - 2
    assert again.cached_tiles == 2


def test_panels_are_planned_only_when_layer_sets_are_supplied(tmp_path):
    """Elevation and landcover need no credentials; panels need a token.
    A caller without one still gets a useful prefetch."""
    bare = plan_prefetch("M1", [ASSIGNMENT], cache_dir=tmp_path)
    assert {layer.layer for layer in bare.layers} == {"dem"}

    with_panels = plan_prefetch("M1", [ASSIGNMENT], cache_dir=tmp_path,
                                panel_sets={"slope": ["sf"]})
    assert "panel" in {layer.layer for layer in with_panels.layers}


# ---------------------------------------------------------------------------
# Fetching, under a budget.
# ---------------------------------------------------------------------------

def test_the_budget_stops_the_fetch_and_says_so(tmp_path, monkeypatch):
    """A caller who thinks they are provisioned and is not has a worse
    problem than one who knows they ran short."""
    written = []

    def fake_tile(tile, cache_dir, get_bytes=None):
        path = Path(cache_dir) / tile.key("dem", "tif")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * 1000)
        written.append(tile)
        return path

    monkeypatch.setattr("sar_geo_mcp.sources.dem.fetch_dem_tile", fake_tile)

    result = run_prefetch("M1", [ASSIGNMENT, NEIGHBOUR], cache_dir=tmp_path,
                          budget_bytes=2500)

    assert result["stopped_early"] is True
    assert result["bytes_written"] <= 3000
    assert len(written) == 3, "stopped as soon as the budget was reached"


def test_a_failing_tile_does_not_abandon_the_rest(tmp_path, monkeypatch):
    """A partly provisioned laptop still beats an unprovisioned one."""
    calls = []

    def flaky(tile, cache_dir, get_bytes=None):
        calls.append(tile)
        if len(calls) == 1:
            raise OSError("connection reset")
        path = Path(cache_dir) / tile.key("dem", "tif")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"y" * 10)
        return path

    monkeypatch.setattr("sar_geo_mcp.sources.dem.fetch_dem_tile", flaky)

    result = run_prefetch("M1", [ASSIGNMENT], cache_dir=tmp_path,
                          budget_bytes=10_000_000)

    assert result["failure_count"] == 1
    assert result["tiles_fetched"] == len(calls) - 1
    assert "OSError" in result["failures"][0]


def test_a_failure_report_never_carries_the_url(tmp_path, monkeypatch):
    """The WMS base URL carries the token in its path. It must not reach a
    result, a log, or a filename."""
    def boom(base_url, layers, tile, cache_dir, get_bytes=None):
        raise RuntimeError(f"failed fetching {base_url}")

    monkeypatch.setattr("sar_geo_mcp.caltopo.wms.fetch_panel_tile", boom)
    monkeypatch.setattr(
        "sar_geo_mcp.sources.dem.fetch_dem_tile",
        lambda tile, cache_dir, get_bytes=None: _stub_tile(cache_dir, tile))

    secret = "https://caltopo.com/api/SUPERSECRETTOKEN/wms"
    result = run_prefetch("M1", [ASSIGNMENT], cache_dir=tmp_path,
                          budget_bytes=10_000_000,
                          panel_sets={"slope": ["sf"]}, base_url=secret)

    blob = repr(result)
    assert "SUPERSECRETTOKEN" not in blob
    assert result["failure_count"] > 0, "the panel fetches did fail"


def _stub_tile(cache_dir, tile):
    path = Path(cache_dir) / tile.key("dem", "tif")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"z" * 10)
    return path


def test_nothing_is_refetched_on_a_second_run(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sar_geo_mcp.sources.dem.fetch_dem_tile",
        lambda tile, cache_dir, get_bytes=None: _stub_tile(cache_dir, tile))

    first = run_prefetch("M1", [ASSIGNMENT], cache_dir=tmp_path,
                         budget_bytes=10_000_000)
    second = run_prefetch("M1", [ASSIGNMENT], cache_dir=tmp_path,
                          budget_bytes=10_000_000)

    assert first["tiles_fetched"] > 0
    assert second["tiles_fetched"] == 0
    assert second["bytes_written"] == 0


def test_elevation_is_fetched_before_imagery(tmp_path, monkeypatch):
    """Elevation is the measurement substrate. A budget that runs out
    should run out on pictures, not on the thing every metric needs."""
    order = []

    monkeypatch.setattr(
        "sar_geo_mcp.sources.dem.fetch_dem_tile",
        lambda tile, cache_dir, get_bytes=None: (
            order.append("dem"), _stub_tile(cache_dir, tile))[1])

    def panel(base_url, layers, tile, cache_dir, get_bytes=None):
        order.append("panel")
        path = Path(cache_dir) / tile.key(f"wms-{'-'.join(layers)}", "png")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"p" * 10)
        return path

    monkeypatch.setattr("sar_geo_mcp.caltopo.wms.fetch_panel_tile", panel)

    run_prefetch("M1", [ASSIGNMENT], cache_dir=tmp_path, budget_bytes=10_000_000,
                 panel_sets={"slope": ["sf"]}, base_url="https://example/wms")

    assert order, "nothing ran"
    assert order[0] == "dem"
    assert order.index("panel") > order.count("dem") - 1

# ---------------------------------------------------------------------------
# Only assignments drive terrain. Worth 950 MB on a real map.
# ---------------------------------------------------------------------------

def test_markers_do_not_drag_their_own_zoom_level_into_the_download(tmp_path):
    """A 50 m marker has a tiny extent, so `level_for` picks a very deep
    zoom for it - and nothing in this project ever renders a DEM for a
    marker. Enumerating tiles per feature on a real 85-feature map came to
    266 tiles and an estimated 1.1 GB; restricting terrain to assignments
    brought the same map to 35 tiles and 147 MB.
    """
    with_marker = plan_prefetch("M1", [ASSIGNMENT, NEARBY_MARKER],
                                cache_dir=tmp_path)
    without = plan_prefetch("M1", [ASSIGNMENT], cache_dir=tmp_path)

    assert with_marker.missing_tiles == without.missing_tiles, (
        "the marker pulled its own tiles in"
    )
    # It still counts as mapped ground for reporting purposes.
    assert "trailhead" in with_marker.clusters[0].labels


def test_a_marker_only_map_still_prefetches_something(tmp_path):
    """Falling back is the same choice `partition` makes: a caller
    prefetching around markers is doing it on purpose."""
    plan = plan_prefetch("M1", [NEARBY_MARKER], cache_dir=tmp_path)
    assert plan.missing_tiles > 0


def test_the_prefetch_level_is_capped(tmp_path):
    """A pathologically small assignment would otherwise request tiles
    finer than anything downstream reads, and that mistake is measured in
    gigabytes."""
    from sar_geo_mcp.prefetch import MAX_PREFETCH_LEVEL, prefetch_level

    tiny = feature("Assignment", "AD999", WEST, SOUTH, 0.00002)
    box, _, _ = buffered_bbox(tiny)
    assert prefetch_level(box) <= MAX_PREFETCH_LEVEL

    plan = plan_prefetch("M1", [tiny], cache_dir=tmp_path)
    assert all(tile.level <= MAX_PREFETCH_LEVEL
               for layer in plan.layers for tile in layer.missing)


def test_many_assignments_on_one_hillside_share_tiles(tmp_path):
    """19 assignments on the real training map needed 35 DEM tiles, not
    19 private sets."""
    assignments = [feature("Assignment", f"AD{i}", WEST + i * 0.004, SOUTH)
                   for i in range(6)]
    plan = plan_prefetch("M1", assignments, cache_dir=tmp_path)

    alone = plan_prefetch("M1", [assignments[0]], cache_dir=tmp_path)
    assert plan.missing_tiles < alone.missing_tiles * 6
