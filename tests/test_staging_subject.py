"""Staging generalised from an assignment to any subject with geometry and a
label. The debrief app reads `assignment.json`; a planner-chosen area has no
assignment, so `subject.json` is what every reader falls back to."""

from __future__ import annotations

import json

import numpy as np
from rasterio.transform import from_origin
from test_grid import make_wgs84_square, write_fetch_dir

from sar_geo_mcp.sources import osm as osm_source
from sar_geo_mcp.staging import stage_terrain, write_manifest
from sar_geo_mcp.terrain.grid import load_assignment_grid

SQUARE = make_wgs84_square(-122.3398, 37.3388, -122.3392, 37.3396)
TRANSFORM = from_origin(-13618826.5, 4486605.0, 10.0, 10.0)


class Area:
    """Anything with geometry, label and model_dump_json is a subject."""

    def __init__(self, geometry):
        self.geometry = geometry
        self.label = "aoi-0123456789abcdef"

    def model_dump_json(self, indent=None):
        return json.dumps({"kind": "polygon", "geometry": self.geometry,
                           "label": self.label}, indent=indent)


def stub_fetchers(monkeypatch, tmp_path):
    """Elevation, vegetation and roads come from public services; the tests
    replace the three fetchers with writers of a tiny DEM and nothing else."""
    import rasterio

    def fake_fetch_dem(bbox, width_px, height_px, *, cache_dir):
        path = tmp_path / "fake_dem.tif"
        with rasterio.open(path, "w", driver="GTiff", height=4, width=4, count=1,
                           dtype="float32", crs="EPSG:3857", transform=TRANSFORM) as dst:
            dst.write(np.zeros((4, 4), dtype="float32"), 1)
        return path

    monkeypatch.setattr("sar_geo_mcp.staging.fetch_dem", fake_fetch_dem)
    monkeypatch.setattr("sar_geo_mcp.staging.write_evt_clip",
                        lambda bbox, out, cache_dir: (_ for _ in ()).throw(RuntimeError("no evt")))
    monkeypatch.setattr("sar_geo_mcp.terrain.linear.routes_for",
                        lambda bbox, cache_dir: [])
    # BOTH halves of the OSM read, or neither. `stage_terrain` calls
    # `routes_for` and then `parking_for`, and its comment there says
    # parking costs no extra Overpass traffic because it re-reads the tiles
    # routes has just written. That is true in production and false here:
    # stubbing routes means no tile is ever written, so an unstubbed
    # `parking_for` goes to the real network. These tests then passed only
    # while Overpass was up, and on 2026-09-22 - with Overpass unreachable,
    # which is exactly when someone is working on Overpass handling - they
    # spent the whole 90 s budget and failed on a missing osm.json.
    monkeypatch.setattr("sar_geo_mcp.terrain.linear.parking_for",
                        lambda bbox, cache_dir: [])
    # And gates (query version 5), for the same reason: a third read of the
    # tiles no stub ever wrote.
    monkeypatch.setattr("sar_geo_mcp.terrain.linear.gates_for",
                        lambda bbox, cache_dir: [])


def test_stage_terrain_writes_subject_json_for_an_area_and_no_assignment_json(tmp_path, monkeypatch):
    stub_fetchers(monkeypatch, tmp_path)
    out = tmp_path / "staged"
    stage_terrain(Area(SQUARE), "", out, cache_dir=tmp_path / "cache",
                  fingerprint="abc", access="public", subject_kind="area")
    assert (out / "subject.json").is_file()
    assert not (out / "assignment.json").exists()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["subject_kind"] == "area"
    assert manifest["assignment"] == "aoi-0123456789abcdef"
    assert manifest["assignment_fingerprint"] == "abc"
    assert manifest["map_id"] == ""


def test_staged_osm_json_records_the_query_version_it_was_fetched_with(tmp_path, monkeypatch):
    """A directory staged by an older query is missing barriers, water edges
    and some road types; the version is what lets a reader notice."""
    from sar_geo_mcp.sources.osm import OSM_QUERY_VERSION

    stub_fetchers(monkeypatch, tmp_path)
    out = tmp_path / "staged"
    stage_terrain(Area(SQUARE), "", out, cache_dir=tmp_path / "cache",
                  fingerprint="abc", access="public", subject_kind="area")
    payload = json.loads((out / "osm.json").read_text(encoding="utf-8"))
    assert payload["routes"] == []
    assert payload["query_version"] == OSM_QUERY_VERSION


def test_staged_osm_json_carries_parking_beside_the_routes(monkeypatch, tmp_path):
    """The staged file is what `parking_candidates` reads; a version 3
    staging that wrote only routes would answer "no lots here" for ever."""
    from sar_geo_mcp.terrain import linear

    stub_fetchers(monkeypatch, tmp_path)
    monkeypatch.setattr(linear, "routes_for", lambda *a, **k: [{"id": "route_0"}])
    monkeypatch.setattr(linear, "parking_for", lambda *a, **k: [{"id": "parking-w10"}])
    monkeypatch.setattr(linear, "gates_for", lambda *a, **k: [])

    out = tmp_path / "staged"
    stage_terrain(Area(SQUARE), "", out, cache_dir=tmp_path / "cache",
                  fingerprint="abc", access="public", subject_kind="area")
    payload = json.loads((out / "osm.json").read_text(encoding="utf-8"))

    assert payload["query_version"] == osm_source.OSM_QUERY_VERSION
    assert [site["id"] for site in payload["parking"]] == ["parking-w10"]
    assert [route["id"] for route in payload["routes"]] == ["route_0"]


def test_a_first_staging_writes_gates_into_osm_json(monkeypatch, tmp_path):
    """Query version 5: the gates egress says are on a road or a route come
    out of the same tiles, in the same file, as the routes and parking."""
    from sar_geo_mcp.terrain import linear

    stub_fetchers(monkeypatch, tmp_path)
    monkeypatch.setattr(linear, "gates_for", lambda *a, **k: [{"id": "gate-n1"}])

    out = tmp_path / "staged"
    stage_terrain(Area(SQUARE), "", out, cache_dir=tmp_path / "cache",
                  fingerprint="abc", access="public", subject_kind="area")
    payload = json.loads((out / "osm.json").read_text(encoding="utf-8"))

    assert payload["query_version"] == osm_source.OSM_QUERY_VERSION == 5
    assert payload["gates"] == [{"id": "gate-n1"}]


def test_stage_terrain_still_writes_assignment_json_for_an_assignment(tmp_path, monkeypatch):
    """The debrief app reads assignment.json out of this directory; it must
    keep appearing for assignment subjects."""
    stub_fetchers(monkeypatch, tmp_path)
    subject = Area(SQUARE)
    subject.label = "QA103"
    out = tmp_path / "staged"
    stage_terrain(subject, "MAP1", out, cache_dir=tmp_path / "cache", fingerprint="f")
    assert (out / "assignment.json").is_file()
    assert (out / "subject.json").is_file()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["subject_kind"] == "assignment"


def test_write_manifest_defaults_subject_kind_to_assignment(tmp_path):
    write_manifest(tmp_path, map_id="M", assignment_label="A", bbox=(0, 0, 1, 1),
                   dem_bbox=(0, 0, 1, 1), dem_resolution_m=1.0, latitude_deg=37.0,
                   panels={}, warnings=[])
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["subject_kind"] == "assignment"


def test_grid_reads_subject_json_when_present(tmp_path):
    data_dir = write_fetch_dir(tmp_path, np.zeros((40, 40)), TRANSFORM, SQUARE)
    with_assignment_only = load_assignment_grid(data_dir).mask.sum()
    other = make_wgs84_square(-122.3396, 37.3390, -122.3394, 37.3392)
    (data_dir / "subject.json").write_text(json.dumps({"geometry": other}), encoding="utf-8")
    with_subject = load_assignment_grid(data_dir).mask.sum()
    assert with_subject < with_assignment_only


def test_grid_falls_back_to_assignment_json(tmp_path):
    data_dir = write_fetch_dir(tmp_path, np.zeros((40, 40)), TRANSFORM, SQUARE)
    assert not (data_dir / "subject.json").exists()
    assert load_assignment_grid(data_dir).mask.any()


def test_read_subject_prefers_subject_json(tmp_path):
    from sar_geo_mcp.terrain.grid import read_subject

    data_dir = write_fetch_dir(tmp_path, np.zeros((4, 4)), TRANSFORM, SQUARE)
    assert read_subject(data_dir)["letter"] == "T"
    (data_dir / "subject.json").write_text(json.dumps({"kind": "area"}), encoding="utf-8")
    assert read_subject(data_dir) == {"kind": "area"}


def test_an_unreachable_overpass_warns_with_its_reason_not_just_a_class_name(
        tmp_path, monkeypatch):
    """`osm unavailable: HTTPStatusError` tells an operator nothing. The
    warning now carries the sentence the exception was written to say."""
    from sar_geo_mcp.sources.osm import OverpassUnavailable

    stub_fetchers(monkeypatch, tmp_path)

    def refuse(bbox, cache_dir):
        raise OverpassUnavailable(
            "OpenStreetMap could not be reached: every Overpass endpoint refused "
            "or timed out (last: 429 Too Many Requests from overpass-api.de). "
            "Trails and roads will be missing from this area until it succeeds.",
            status=429, endpoint="https://overpass-api.de/api/interpreter")

    monkeypatch.setattr("sar_geo_mcp.terrain.linear.routes_for", refuse)

    out = tmp_path / "staged"
    result = stage_terrain(Area(SQUARE), "", out, cache_dir=tmp_path / "cache",
                           fingerprint="abc", access="public", subject_kind="area")

    warning = next(w for w in result["warnings"] if w.startswith("osm unavailable"))
    assert "OverpassUnavailable" in warning
    assert "429 Too Many Requests" in warning
    assert "Trails and roads will be missing" in warning
    assert not (out / "osm.json").exists()


# --- no warning carries a local path or a URL ------------------------------
#
# The app's Safety page shows these warnings to planners. An OSError's
# message is the local cache path it failed on, and an httpx error's is the
# URL it asked; `net_errors.safe_reason` is how the rest of the hub says why
# without saying where. OverpassUnavailable is the one exception whose
# message is kept: the hub writes it for a planner, from a host name and a
# status, and "429 Too Many Requests from overpass-api.de" is the reason
# the test above exists.

def _osm_warning(warnings):
    from sar_geo_mcp.staging import is_osm_outage_warning

    return next(w for w in warnings if is_osm_outage_warning(w))


def test_an_osm_failure_on_a_local_path_does_not_put_the_path_in_the_warning(
        tmp_path, monkeypatch):
    from sar_geo_mcp.staging import osm_outage_warning

    stub_fetchers(monkeypatch, tmp_path)
    part = tmp_path / "cache" / "osm" / "tile_z14_2631_6339.json.part"

    def fail(bbox, cache_dir):
        raise PermissionError(13, "Permission denied", str(part))

    monkeypatch.setattr("sar_geo_mcp.terrain.linear.parking_for", fail)

    out = tmp_path / "staged"
    result = stage_terrain(Area(SQUARE), "", out, cache_dir=tmp_path / "cache",
                           fingerprint="abc", access="public", subject_kind="area")

    warning = _osm_warning(result["warnings"])
    assert warning == osm_outage_warning("PermissionError")
    assert str(tmp_path) not in warning and "tile_z14" not in warning
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert warning in manifest["warnings"]
    assert not any(str(tmp_path) in w for w in manifest["warnings"])


def test_an_osm_failure_on_a_url_does_not_put_the_url_in_the_warning(tmp_path, monkeypatch):
    import httpx

    from sar_geo_mcp.staging import osm_outage_warning

    stub_fetchers(monkeypatch, tmp_path)
    url = "https://overpass.agency.example/api/interpreter?key=s3cret"
    request = httpx.Request("POST", url)

    def reject(bbox, cache_dir):
        raise httpx.HTTPStatusError(f"Client error '400 Bad Request' for url '{url}'",
                                    request=request, response=httpx.Response(400, request=request))

    monkeypatch.setattr("sar_geo_mcp.terrain.linear.routes_for", reject)

    result = stage_terrain(Area(SQUARE), "", tmp_path / "staged", cache_dir=tmp_path / "cache",
                           fingerprint="abc", access="public", subject_kind="area")

    warning = _osm_warning(result["warnings"])
    assert warning == osm_outage_warning("HTTPStatusError 400")
    assert "s3cret" not in warning and "agency.example" not in warning


# --- a re-stage during an outage: "osm.json exists" must mean THIS staging ----
#
# `stage_terrain` writes osm.json only on success. Before 0.17.1 it never
# removed an old one, so a boundary that moved and was re-staged while
# Overpass was down kept the OLD boundary's clipped file - and once
# `manifest_warnings` trusted "the file exists", the outage was hidden too.

MOVED = make_wgs84_square(-122.3396, 37.3390, -122.3394, 37.3392)


def _stub_ridge_dem(monkeypatch, tmp_path):
    """A 40 x 40 ridge under both SQUARE and MOVED, so the metrics bundle
    can be computed over what `stage_terrain` actually wrote."""
    import rasterio

    def fake_fetch_dem(bbox, width_px, height_px, *, cache_dir):
        path = tmp_path / "fake_ridge_dem.tif"
        col = np.abs(np.arange(40) - 20).astype("float32")
        elevation = (20 - col)[None, :].repeat(40, axis=0) * 2.0
        with rasterio.open(path, "w", driver="GTiff", height=40, width=40, count=1,
                           dtype="float32", crs="EPSG:3857", transform=TRANSFORM,
                           nodata=-999999.0) as dst:
            dst.write(elevation, 1)
        return path

    monkeypatch.setattr("sar_geo_mcp.staging.fetch_dem", fake_fetch_dem)


def _overpass_down(monkeypatch):
    from sar_geo_mcp.sources.osm import OverpassUnavailable

    def refuse(bbox, cache_dir):
        raise OverpassUnavailable("every Overpass endpoint refused", status=429)

    monkeypatch.setattr("sar_geo_mcp.terrain.linear.routes_for", refuse)


def test_a_moved_boundary_restaged_in_an_outage_drops_the_old_boundarys_osm_json(
        tmp_path, monkeypatch):
    from sar_geo_mcp.staging import is_osm_outage_warning, manifest_warnings
    from sar_geo_mcp.terrain.metrics import compute_terrain_metrics

    stub_fetchers(monkeypatch, tmp_path)
    _stub_ridge_dem(monkeypatch, tmp_path)
    out = tmp_path / "staged"
    subject = Area(SQUARE)
    stage_terrain(subject, "MAP1", out, cache_dir=tmp_path / "cache", fingerprint="old")
    assert (out / "osm.json").is_file()

    _overpass_down(monkeypatch)
    subject.geometry = MOVED
    result = stage_terrain(subject, "MAP1", out, cache_dir=tmp_path / "cache",
                           fingerprint="moved")

    assert not (out / "osm.json").exists()
    assert any(is_osm_outage_warning(w) for w in result["warnings"])
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert any(is_osm_outage_warning(w) for w in manifest_warnings(out, manifest))
    metrics = compute_terrain_metrics(out)
    assert any(is_osm_outage_warning(w) for w in metrics.data_quality.warnings)
    assert metrics.access is None


def test_an_outage_with_no_previous_manifest_drops_a_stray_osm_json(tmp_path, monkeypatch):
    """No manifest, no way to say what ground the file was fetched for - so
    it cannot be said to belong to this staging."""
    from sar_geo_mcp.staging import is_osm_outage_warning
    from sar_geo_mcp.sources.osm import OSM_QUERY_VERSION

    stub_fetchers(monkeypatch, tmp_path)
    _overpass_down(monkeypatch)
    out = tmp_path / "staged"
    out.mkdir()
    # The current version, so the only thing wrong with the file is that no
    # manifest says what ground it was fetched for.
    (out / "osm.json").write_text(json.dumps({"routes": [{"id": "stray"}],
                                              "query_version": OSM_QUERY_VERSION}),
                                  encoding="utf-8")

    result = stage_terrain(Area(SQUARE), "", out, cache_dir=tmp_path / "cache",
                           fingerprint="abc", subject_kind="area")

    assert not (out / "osm.json").exists()
    assert any(is_osm_outage_warning(w) for w in result["warnings"])


def test_the_same_ground_restaged_in_an_outage_keeps_its_osm_json_and_says_so(
        tmp_path, monkeypatch):
    from sar_geo_mcp.staging import is_osm_outage_warning, manifest_warnings
    from sar_geo_mcp.sources.osm import OSM_QUERY_VERSION

    stub_fetchers(monkeypatch, tmp_path)
    monkeypatch.setattr("sar_geo_mcp.terrain.linear.routes_for",
                        lambda bbox, cache_dir: [{"id": "route_0"}])
    out = tmp_path / "staged"
    stage_terrain(Area(SQUARE), "", out, cache_dir=tmp_path / "cache",
                  fingerprint="abc", subject_kind="area")
    before = (out / "osm.json").read_bytes()

    _overpass_down(monkeypatch)
    result = stage_terrain(Area(SQUARE), "", out, cache_dir=tmp_path / "cache",
                           fingerprint="abc", subject_kind="area")

    assert (out / "osm.json").read_bytes() == before
    kept = ("the OSM refetch failed (OverpassUnavailable: every Overpass endpoint "
            "refused); the osm.json staged earlier for this same ground is kept "
            f"(query version {OSM_QUERY_VERSION})")
    assert kept in result["warnings"]
    assert not any(is_osm_outage_warning(w) for w in result["warnings"])
    # Kept on record, returned in THIS staging's response only: on reuse
    # the file it names is simply there, and saying so again is noise.
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert kept in manifest["warnings"]
    assert kept not in manifest_warnings(out, manifest)


def test_the_kept_warning_names_the_version_of_the_file_kept(tmp_path, monkeypatch):
    stub_fetchers(monkeypatch, tmp_path)
    out = tmp_path / "staged"
    stage_terrain(Area(SQUARE), "", out, cache_dir=tmp_path / "cache",
                  fingerprint="abc", subject_kind="area")
    (out / "osm.json").write_text(json.dumps({"routes": [], "query_version": 3}),
                                  encoding="utf-8")

    _overpass_down(monkeypatch)
    result = stage_terrain(Area(SQUARE), "", out, cache_dir=tmp_path / "cache",
                           fingerprint="abc", subject_kind="area")

    assert any(w.endswith("is kept (query version 3)") for w in result["warnings"])


def test_the_kept_warning_carries_no_path(tmp_path, monkeypatch):
    stub_fetchers(monkeypatch, tmp_path)
    out = tmp_path / "staged"
    stage_terrain(Area(SQUARE), "", out, cache_dir=tmp_path / "cache",
                  fingerprint="abc", subject_kind="area")

    def fail(bbox, cache_dir):
        raise PermissionError(13, "Permission denied", str(tmp_path / "cache" / "t.part"))

    monkeypatch.setattr("sar_geo_mcp.terrain.linear.routes_for", fail)
    result = stage_terrain(Area(SQUARE), "", out, cache_dir=tmp_path / "cache",
                           fingerprint="abc", subject_kind="area")

    kept = next(w for w in result["warnings"] if w.startswith("the OSM refetch failed"))
    assert "(PermissionError)" in kept and str(tmp_path) not in kept


def test_an_unreadable_osm_json_on_the_same_ground_is_not_kept(tmp_path, monkeypatch):
    from sar_geo_mcp.staging import is_osm_outage_warning

    stub_fetchers(monkeypatch, tmp_path)
    out = tmp_path / "staged"
    stage_terrain(Area(SQUARE), "", out, cache_dir=tmp_path / "cache",
                  fingerprint="abc", subject_kind="area")
    (out / "osm.json").write_text("{not json", encoding="utf-8")

    _overpass_down(monkeypatch)
    result = stage_terrain(Area(SQUARE), "", out, cache_dir=tmp_path / "cache",
                           fingerprint="abc", subject_kind="area")

    assert not (out / "osm.json").exists()
    assert any(is_osm_outage_warning(w) for w in result["warnings"])


# --- a file that cannot be removed says so without saying where -----------------

def _unlink_refused(monkeypatch):
    """Refuse to remove osm.json the way Windows does when another process
    holds it open: a PermissionError whose message is the full local path."""
    from pathlib import Path

    real_unlink = Path.unlink

    def refuse(self, missing_ok=False):
        if self.name == "osm.json":
            raise PermissionError(13, "The process cannot access the file", str(self))
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", refuse)


def test_other_grounds_osm_json_that_cannot_be_removed_fails_without_a_path(
        tmp_path, monkeypatch):
    import pytest

    stub_fetchers(monkeypatch, tmp_path)
    out = tmp_path / "staged"
    subject = Area(SQUARE)
    stage_terrain(subject, "MAP1", out, cache_dir=tmp_path / "cache", fingerprint="old")

    _overpass_down(monkeypatch)
    _unlink_refused(monkeypatch)
    subject.geometry = MOVED
    with pytest.raises(RuntimeError) as caught:
        stage_terrain(subject, "MAP1", out, cache_dir=tmp_path / "cache", fingerprint="moved")

    assert str(caught.value) == ("osm.json staged for other ground could not be removed "
                                 "(PermissionError); stage again")
    assert str(tmp_path) not in str(caught.value)
    # Nothing chained: a traceback printer walks __context__ and __cause__,
    # and the PermissionError there would carry the path all the same.
    assert caught.value.__context__ is None and caught.value.__cause__ is None


# --- a gates-only failure is an OSM failure like any other (query version 5) ------

def _gates_down(monkeypatch):
    from sar_geo_mcp.sources.osm import OverpassUnavailable

    def refuse(bbox, cache_dir):
        raise OverpassUnavailable("every Overpass endpoint refused", status=429)

    monkeypatch.setattr("sar_geo_mcp.terrain.linear.gates_for", refuse)


def test_a_gates_failure_on_the_same_ground_keeps_the_file_and_says_so(tmp_path, monkeypatch):
    """Routes and parking answered, gates did not: nothing is written, so the
    file staged earlier for this ground is kept, exactly as for a routes
    failure - never rewritten without its gates, which would read as "no
    gate here"."""
    from sar_geo_mcp.staging import is_osm_kept_warning, is_osm_outage_warning

    stub_fetchers(monkeypatch, tmp_path)
    out = tmp_path / "staged"
    stage_terrain(Area(SQUARE), "", out, cache_dir=tmp_path / "cache",
                  fingerprint="abc", subject_kind="area")
    before = (out / "osm.json").read_bytes()

    _gates_down(monkeypatch)
    result = stage_terrain(Area(SQUARE), "", out, cache_dir=tmp_path / "cache",
                           fingerprint="abc", subject_kind="area")

    assert (out / "osm.json").read_bytes() == before
    assert any(is_osm_kept_warning(w) for w in result["warnings"])
    assert not any(is_osm_outage_warning(w) for w in result["warnings"])


def test_a_gates_failure_on_other_ground_removes_the_file_and_says_so(tmp_path, monkeypatch):
    from sar_geo_mcp.staging import is_osm_kept_warning, is_osm_outage_warning

    stub_fetchers(monkeypatch, tmp_path)
    _stub_ridge_dem(monkeypatch, tmp_path)
    out = tmp_path / "staged"
    subject = Area(SQUARE)
    stage_terrain(subject, "MAP1", out, cache_dir=tmp_path / "cache", fingerprint="old")
    assert (out / "osm.json").is_file()

    _gates_down(monkeypatch)
    subject.geometry = MOVED
    result = stage_terrain(subject, "MAP1", out, cache_dir=tmp_path / "cache",
                           fingerprint="moved")

    assert not (out / "osm.json").exists()
    assert any(is_osm_outage_warning(w) for w in result["warnings"])
    assert not any(is_osm_kept_warning(w) for w in result["warnings"])


def test_a_gates_failure_on_a_local_path_does_not_put_the_path_in_the_warning(
        tmp_path, monkeypatch):
    from sar_geo_mcp.staging import osm_outage_warning

    stub_fetchers(monkeypatch, tmp_path)
    part = tmp_path / "cache" / "osm_tile_v5_c2631_r6339.json.part"

    def fail(bbox, cache_dir):
        raise PermissionError(13, "Permission denied", str(part))

    monkeypatch.setattr("sar_geo_mcp.terrain.linear.gates_for", fail)
    out = tmp_path / "staged"
    result = stage_terrain(Area(SQUARE), "", out, cache_dir=tmp_path / "cache",
                           fingerprint="abc", access="public", subject_kind="area")

    warning = _osm_warning(result["warnings"])
    assert warning == osm_outage_warning("PermissionError")
    assert str(tmp_path) not in warning and "osm_tile" not in warning
    assert not (out / "osm.json").exists()


def test_restaging_removes_terrain_metrics_saved_for_the_old_ground(tmp_path, monkeypatch):
    """A moved assignment is re-staged into the SAME directory. Metrics the
    `terrain_metrics` tool saved for the old polygon must not survive it, or
    `slope_stats` and `coverage_for` would answer for ground that has gone."""
    stub_fetchers(monkeypatch, tmp_path)
    out = tmp_path / "staged"
    stage_terrain(Area(SQUARE), "", out, cache_dir=tmp_path / "cache",
                  fingerprint="abc", access="public", subject_kind="area")
    (out / "terrain_metrics.json").write_text("{}", encoding="utf-8")

    stage_terrain(Area(SQUARE), "", out, cache_dir=tmp_path / "cache",
                  fingerprint="def", access="public", subject_kind="area")

    assert not (out / "terrain_metrics.json").exists()
