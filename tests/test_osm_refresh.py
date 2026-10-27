"""Topping up a staged directory's `osm.json` when the query has moved on.

0.11.0 bumped `OSM_QUERY_VERSION` to fetch parking. Every directory staged
before that holds a version-2 `osm.json` with no parking in it, and
`parking_candidates` correctly, but unhelpfully, answers "no lots can be
read from this" forever. `refresh_osm` is the fix: re-run `routes_for` and
`parking_for` over the directory's OWN bbox and rewrite just that file,
leaving the DEM and EVT - unaffected by an Overpass query change - alone.
"""

from __future__ import annotations

import json

import pytest

from sar_geo_mcp.staging import (
    OSM_UNAVAILABLE,
    is_osm_outage_warning,
    manifest_warnings,
    osm_outage_warning,
    refresh_osm,
    write_manifest,
)
from sar_geo_mcp.terrain import linear
from sar_geo_mcp.terrain.linear import OSM_QUERY_VERSION, OverpassUnavailable

BBOX = (0.0, 0.0, 100.0, 100.0)


def _staged_dir(tmp_path, *, osm=None, bbox=BBOX, manifest_extra=None):
    out = tmp_path / "staged"
    out.mkdir()
    write_manifest(out, map_id="M", assignment_label="aoi-x", bbox=bbox,
                   dem_bbox=bbox, dem_resolution_m=10.0, latitude_deg=37.4,
                   panels={}, warnings=[], subject_kind="area")
    if manifest_extra:
        manifest_path = out / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(manifest_extra)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    if osm is not None:
        (out / "osm.json").write_text(json.dumps(osm), encoding="utf-8")
    return out


def _refuse_overpass(monkeypatch, message="every Overpass endpoint refused"):
    def refuse(bbox, cache_dir):
        raise OverpassUnavailable(message, status=429, endpoint="https://overpass-api.de")

    monkeypatch.setattr(linear, "routes_for", refuse)


def _network_forbidden(monkeypatch):
    """Fail loudly if any fetcher is called - proves the cheap path is cheap."""

    def boom(*args, **kwargs):
        raise AssertionError("Overpass must not be reached on the cheap path")

    monkeypatch.setattr(linear, "routes_for", boom)
    monkeypatch.setattr(linear, "parking_for", boom)
    monkeypatch.setattr(linear, "gates_for", boom)


# --- the version check itself -------------------------------------------------

def test_a_version_2_directory_is_refreshed_to_current_and_gains_parking(tmp_path, monkeypatch):
    out = _staged_dir(tmp_path, osm={"routes": [{"id": "old"}], "query_version": 2})
    monkeypatch.setattr(linear, "routes_for", lambda bbox, cache_dir: [{"id": "route_0"}])
    monkeypatch.setattr(linear, "parking_for", lambda bbox, cache_dir: [{"id": "parking-w10"}])
    monkeypatch.setattr(linear, "gates_for", lambda bbox, cache_dir: [])

    warnings = refresh_osm(out, tmp_path / "cache")

    payload = json.loads((out / "osm.json").read_text(encoding="utf-8"))
    assert payload["query_version"] == OSM_QUERY_VERSION
    assert payload["parking"] == [{"id": "parking-w10"}]
    assert payload["routes"] == [{"id": "route_0"}]
    assert len(warnings) == 1
    assert "topped it up" in warnings[0] and "query version 2" in warnings[0]


def test_a_current_directory_is_not_refetched_and_costs_no_network_call(tmp_path, monkeypatch):
    """This is the path every reuse of every staged directory runs through,
    so it must cost a file read, never an Overpass round trip. If a future
    change makes the version check eager, this test fails on the
    AssertionError raised by the forbidden fetchers, not on a wrong answer."""
    out = _staged_dir(tmp_path, osm={"routes": [], "parking": [], "query_version": OSM_QUERY_VERSION})
    _network_forbidden(monkeypatch)

    before = (out / "osm.json").read_bytes()
    warnings = refresh_osm(out, tmp_path / "cache")

    assert warnings == []
    assert (out / "osm.json").read_bytes() == before


# --- failure leaves the old file intact ----------------------------------------

def test_an_overpass_failure_leaves_the_old_file_byte_identical_and_warns(tmp_path, monkeypatch):
    out = _staged_dir(tmp_path, osm={"routes": [{"id": "old"}], "query_version": 2})
    before = (out / "osm.json").read_bytes()
    _refuse_overpass(monkeypatch, "429 Too Many Requests")

    warnings = refresh_osm(out, tmp_path / "cache")

    assert (out / "osm.json").read_bytes() == before
    assert len(warnings) == 1
    assert "429 Too Many Requests" in warnings[0]
    assert "parking stays unreadable" in warnings[0]


def test_no_part_file_survives_a_failed_refresh(tmp_path, monkeypatch):
    out = _staged_dir(tmp_path, osm={"routes": [], "query_version": 2})
    _refuse_overpass(monkeypatch)
    refresh_osm(out, tmp_path / "cache")
    assert list(out.glob("*.part")) == []


# --- a directory with no osm.json at all ---------------------------------------

def test_a_directory_with_no_osm_json_is_fetched(tmp_path, monkeypatch):
    out = _staged_dir(tmp_path)  # never had osm.json - an earlier Overpass fetch failed
    assert not (out / "osm.json").exists()
    monkeypatch.setattr(linear, "routes_for", lambda bbox, cache_dir: [])
    monkeypatch.setattr(linear, "parking_for", lambda bbox, cache_dir: [{"id": "p1"}])
    monkeypatch.setattr(linear, "gates_for", lambda bbox, cache_dir: [])

    warnings = refresh_osm(out, tmp_path / "cache")

    payload = json.loads((out / "osm.json").read_text(encoding="utf-8"))
    assert payload["query_version"] == OSM_QUERY_VERSION
    assert payload["parking"] == [{"id": "p1"}]
    assert len(warnings) == 1 and "was missing" in warnings[0]


def test_a_directory_with_no_osm_json_that_fails_again_stays_missing(tmp_path, monkeypatch):
    out = _staged_dir(tmp_path)
    _refuse_overpass(monkeypatch)

    warnings = refresh_osm(out, tmp_path / "cache")

    assert not (out / "osm.json").exists()
    assert len(warnings) == 1 and "still missing" in warnings[0]


# --- the bbox must come from the manifest, never be guessed --------------------

def test_no_manifest_at_all_leaves_the_file_alone_and_says_so(tmp_path, monkeypatch):
    out = tmp_path / "staged"
    out.mkdir()
    (out / "osm.json").write_text(json.dumps({"routes": [], "query_version": 2}), encoding="utf-8")
    _network_forbidden(monkeypatch)

    warnings = refresh_osm(out, tmp_path / "cache")

    assert "no manifest.json" in warnings[0]
    assert json.loads((out / "osm.json").read_text(encoding="utf-8"))["query_version"] == 2


def test_a_manifest_without_a_bbox_leaves_the_file_alone_and_says_so(tmp_path, monkeypatch):
    out = tmp_path / "staged"
    out.mkdir()
    (out / "manifest.json").write_text(json.dumps({"map_id": "M"}), encoding="utf-8")
    (out / "osm.json").write_text(json.dumps({"routes": [], "query_version": 2}), encoding="utf-8")
    _network_forbidden(monkeypatch)

    warnings = refresh_osm(out, tmp_path / "cache")

    assert "no bbox_3857" in warnings[0]
    assert json.loads((out / "osm.json").read_text(encoding="utf-8"))["query_version"] == 2


# --- candidate sets computed from the old file must go stale -------------------

def test_a_candidate_set_saved_before_the_refresh_reports_stale_after_it(tmp_path, monkeypatch):
    from sar_geo_mcp.planning.candidates import CandidateSet, save_set, set_staleness

    out = _staged_dir(tmp_path, osm={"routes": [], "query_version": 2})
    cset = CandidateSet(set_id="hasty-abc123", tool="hasty", params={}, handle="aoi-x",
                        created_at="now", candidates=[], summary={}, assumptions={},
                        warnings=[])
    save_set(out, cset)
    loaded = CandidateSet.model_validate_json(
        (out / "candidates" / "hasty-abc123.json").read_text(encoding="utf-8"))
    assert set_staleness(out, loaded) is None  # sanity: current right after saving

    monkeypatch.setattr(linear, "routes_for", lambda bbox, cache_dir: [])
    monkeypatch.setattr(linear, "parking_for", lambda bbox, cache_dir: [])
    monkeypatch.setattr(linear, "gates_for", lambda bbox, cache_dir: [])
    refresh_osm(out, tmp_path / "cache")

    staleness = set_staleness(out, loaded)
    assert staleness is not None and "re-staged" in staleness


def test_a_failed_refresh_does_not_disturb_the_manifest_stamp(tmp_path, monkeypatch):
    """A refresh that changed nothing must not falsely mark a current
    candidate set stale."""
    from sar_geo_mcp.planning.candidates import CandidateSet, save_set, set_staleness

    out = _staged_dir(tmp_path, osm={"routes": [], "query_version": 2})
    cset = CandidateSet(set_id="hasty-abc123", tool="hasty", params={}, handle="aoi-x",
                        created_at="now", candidates=[], summary={}, assumptions={},
                        warnings=[])
    save_set(out, cset)
    loaded = CandidateSet.model_validate_json(
        (out / "candidates" / "hasty-abc123.json").read_text(encoding="utf-8"))

    _refuse_overpass(monkeypatch)
    refresh_osm(out, tmp_path / "cache")

    assert set_staleness(out, loaded) is None



# --- version 3: parking is there, only distribution lines are missing ---------

def test_a_version_3_top_up_says_distribution_lines_not_parking(tmp_path, monkeypatch):
    """Version 3 already fetched parking; what 4 adds is `power=minor_line`.
    Saying "parking can now be read" would tell a planner parking was
    missing when it never was."""
    out = _staged_dir(tmp_path, osm={"routes": [], "parking": [{"id": "p"}],
                                     "query_version": 3})
    monkeypatch.setattr(linear, "routes_for", lambda bbox, cache_dir: [])
    monkeypatch.setattr(linear, "parking_for", lambda bbox, cache_dir: [{"id": "p"}])
    monkeypatch.setattr(linear, "gates_for", lambda bbox, cache_dir: [])

    warnings = refresh_osm(out, tmp_path / "cache")

    assert len(warnings) == 1
    assert "query version 3" in warnings[0] and "topped it up" in warnings[0]
    assert "minor_line" in warnings[0]
    assert "parking" not in warnings[0]


def test_a_version_3_top_up_during_an_outage_leaves_parking_readable(tmp_path, monkeypatch):
    out = _staged_dir(tmp_path, osm={"routes": [], "parking": [{"id": "p"}],
                                     "query_version": 3})
    _refuse_overpass(monkeypatch, "429 Too Many Requests")

    warnings = refresh_osm(out, tmp_path / "cache")

    assert len(warnings) == 1
    assert "query version 3" in warnings[0] and "minor_line" in warnings[0]
    assert "parking stays unreadable" not in warnings[0]
    assert "parking is unaffected" in warnings[0]


def test_a_version_3_directory_without_a_manifest_does_not_blame_parking(tmp_path):
    out = tmp_path / "staged"
    out.mkdir()
    (out / "osm.json").write_text(json.dumps({"routes": [], "query_version": 3}),
                                  encoding="utf-8")
    warnings = refresh_osm(out, tmp_path / "cache")
    assert len(warnings) == 1 and "parking stays unreadable" not in warnings[0]
    assert "minor_line" in warnings[0] and "parking is unaffected" in warnings[0]


def test_a_failed_version_3_top_up_puts_until_on_the_lines_not_on_parking(tmp_path, monkeypatch):
    """"parking is unaffected until it succeeds" reads as if parking will be
    affected once it does. The clause that waits is the lines'."""
    out = _staged_dir(tmp_path, osm={"routes": [], "parking": [{"id": "p"}],
                                     "query_version": 3})
    _refuse_overpass(monkeypatch, "429 Too Many Requests")

    warnings = refresh_osm(out, tmp_path / "cache")

    # Since 0.18.0 a version-3 file also lacks road facts, named last, so the
    # waiting clause is theirs; parking still never waits.
    assert warnings[0].endswith("stay unknown on this ground until it succeeds")
    assert "parking is unaffected until" not in warnings[0]


def test_an_osm_json_without_a_version_is_called_version_1_as_the_other_tools_call_it(
        tmp_path, monkeypatch):
    """The first query wrote no version. Landing zones and boundaries call
    such a file version 1; a top-up calling it version 0 would show one
    directory with two numbers."""
    out = _staged_dir(tmp_path, osm={"routes": [{"id": "old"}]})
    _refuse_overpass(monkeypatch)

    warnings = refresh_osm(out, tmp_path / "cache")

    assert "query version 1" in warnings[0] and "version 0" not in warnings[0]


# --- a stale outage warning retires once the file it complained of is whole ---
#
# `stage_terrain` records "osm unavailable: ..." in the manifest when the
# first Overpass fetch dies, and `prepare_area` hands the manifest's warnings
# back on every reuse. Before 0.17.1 nothing ever took that warning out, so
# ground staged during an outage told every planner, on every later call,
# that trails and roads were missing - long after a top-up had fetched them.

OTHER = "evt unavailable: OSError"


def _manifest(out):
    return json.loads((out / "manifest.json").read_text(encoding="utf-8"))


def test_the_outage_warning_is_built_from_its_constant():
    assert osm_outage_warning("OSError") == f"{OSM_UNAVAILABLE}: OSError"


def test_outage_warnings_are_recognised_by_their_constant_in_every_form_written():
    # This release's form, and both forms earlier releases left on disk.
    assert is_osm_outage_warning(osm_outage_warning("OverpassUnavailable: every mirror refused"))
    assert is_osm_outage_warning("osm unavailable: HTTPStatusError")
    assert is_osm_outage_warning("osm unavailable: ConnectTimeout: overpass-api.de")
    # Anything else is some other fact and stays.
    assert not is_osm_outage_warning(OTHER)
    assert not is_osm_outage_warning("osm.json is still missing: fetching it failed again "
                                     "(OSError); routes and parking stay unreadable")
    assert not is_osm_outage_warning("the map said osm unavailable: at noon")
    assert not is_osm_outage_warning(None)


def test_a_successful_top_up_takes_the_outage_warning_out_of_the_manifest(tmp_path, monkeypatch):
    out = _staged_dir(tmp_path, manifest_extra={"warnings": [
        osm_outage_warning("OverpassUnavailable: every Overpass endpoint refused"), OTHER]})
    monkeypatch.setattr(linear, "routes_for", lambda bbox, cache_dir: [])
    monkeypatch.setattr(linear, "parking_for", lambda bbox, cache_dir: [])
    monkeypatch.setattr(linear, "gates_for", lambda bbox, cache_dir: [])

    refresh_osm(out, tmp_path / "cache")

    manifest = _manifest(out)
    assert manifest["warnings"] == [OTHER]
    # Moved, not deleted: see the size tests below for why.
    assert [entry["warning"] for entry in manifest["osm_retired_warnings"]] == [
        osm_outage_warning("OverpassUnavailable: every Overpass endpoint refused")]
    assert manifest["osm_refreshed_versions"] == [OSM_QUERY_VERSION]


def test_a_successful_top_up_also_retires_a_kept_file_warning(tmp_path, monkeypatch):
    """"the osm.json staged earlier for this same ground is kept (query
    version 3)" stops being about the file on disk once it is replaced."""
    from sar_geo_mcp.staging import osm_kept_warning

    kept = osm_kept_warning("OverpassUnavailable: every Overpass endpoint refused", 3)
    out = _staged_dir(tmp_path, osm={"routes": [], "parking": [], "query_version": 3},
                      manifest_extra={"warnings": [kept, OTHER]})
    monkeypatch.setattr(linear, "routes_for", lambda bbox, cache_dir: [])
    monkeypatch.setattr(linear, "parking_for", lambda bbox, cache_dir: [])
    monkeypatch.setattr(linear, "gates_for", lambda bbox, cache_dir: [])

    refresh_osm(out, tmp_path / "cache")

    manifest = _manifest(out)
    assert manifest["warnings"] == [OTHER]
    assert [entry["warning"] for entry in manifest["osm_retired_warnings"]] == [kept]


def test_manifest_warnings_never_return_the_retired_list(tmp_path):
    outage = osm_outage_warning("ConnectTimeout")
    out = _staged_dir(tmp_path, manifest_extra={
        "warnings": [OTHER],
        # A record of a past top-up; the number is history, not the current version.
        "osm_retired_warnings": [{"warning": outage, "retired_by_query_version": 4}]})
    assert manifest_warnings(out, _manifest(out)) == [OTHER]


@pytest.mark.parametrize("already_retired", [0, 1, 2])
@pytest.mark.parametrize("others", [0, 1])
def test_a_top_up_that_retires_warnings_still_makes_the_manifest_strictly_longer(
        tmp_path, monkeypatch, already_retired, others):
    """The staleness stamp is mtime AND size, and `refresh_osm` promises a
    strictly longer manifest (see CANDIDATE SETS in its docstring). Deleting
    the outage warning while appending one version number could land on
    exactly the old size - review found it with 15- and 18-character
    reasons - so retired warnings are moved to a list that only grows. This
    walks every reason length a real warning could plausibly have, over a
    manifest `refresh_osm` itself wrote (indent=2), with and without other
    warnings and earlier retirements."""
    monkeypatch.setattr(linear, "routes_for", lambda bbox, cache_dir: [])
    monkeypatch.setattr(linear, "parking_for", lambda bbox, cache_dir: [])
    monkeypatch.setattr(linear, "gates_for", lambda bbox, cache_dir: [])
    for length in range(0, 121):
        out = tmp_path / f"r{length}"
        out.mkdir()
        manifest = {
            "map_id": "M", "bbox_3857": list(BBOX), "panels": {},
            "warnings": [osm_outage_warning("x" * length)] + [OTHER] * others,
            "osm_refreshed_versions": [3],
        }
        if already_retired:
            manifest["osm_retired_warnings"] = [
                {"warning": osm_outage_warning("y" * n), "retired_by_query_version": 3}
                for n in range(already_retired)]
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (out / "osm.json").write_text(json.dumps({"routes": [], "query_version": 3}),
                                      encoding="utf-8")
        before = (out / "manifest.json").stat().st_size

        refresh_osm(out, tmp_path / "cache")

        after = (out / "manifest.json").stat().st_size
        assert after > before, f"reason of {length} characters: {before} -> {after} bytes"


def test_a_failed_top_up_leaves_the_outage_warning_where_it_was(tmp_path, monkeypatch):
    outage = osm_outage_warning("OverpassUnavailable: every Overpass endpoint refused")
    out = _staged_dir(tmp_path, manifest_extra={"warnings": [outage, OTHER]})
    _refuse_overpass(monkeypatch)

    refresh_osm(out, tmp_path / "cache")

    assert _manifest(out)["warnings"] == [outage, OTHER]


def test_manifest_warnings_drop_the_outage_once_osm_json_is_current(tmp_path):
    """Ground healed by 0.17.0's top-up has a current `osm.json` and a
    manifest that still carries the outage - 0.17.0 never took it out, and
    a current file is never refreshed again. Reading has to catch it."""
    outage = osm_outage_warning("ConnectTimeout")
    # `osm_refreshed_versions` is what that top-up wrote - every top-up by
    # every hub has, since `refresh_osm` was added with it.
    out = _staged_dir(tmp_path, osm={"routes": [], "parking": [],
                                     "query_version": OSM_QUERY_VERSION},
                      manifest_extra={"warnings": [outage, OTHER],
                                      "osm_refreshed_versions": [OSM_QUERY_VERSION]})
    assert manifest_warnings(out, _manifest(out)) == [OTHER]


def test_manifest_warnings_drop_the_outage_while_osm_json_is_behind(tmp_path):
    """"osm unavailable" describes the FIRST fetch. Once any osm.json exists
    it is false, whatever version wrote it; the version lag is
    `refresh_osm`'s to report, in its own warnings."""
    outage = osm_outage_warning("ConnectTimeout")
    out = _staged_dir(tmp_path, osm={"routes": [], "parking": [], "query_version": 3},
                      manifest_extra={"warnings": [outage, OTHER],
                                      "osm_refreshed_versions": [3]})
    assert manifest_warnings(out, _manifest(out)) == [OTHER]


def test_manifest_warnings_keep_the_outage_beside_a_file_no_top_up_wrote(tmp_path):
    """An older hub re-staging a moved assignment boundary during an outage
    left the OLD boundary's osm.json beside the new outage warning. No
    `osm_refreshed_versions` means no top-up wrote that file after the
    outage, so it is a leftover and the outage is still the truth."""
    outage = osm_outage_warning("ConnectTimeout")
    out = _staged_dir(tmp_path, osm={"routes": [{"id": "old-boundary"}],
                                     "query_version": OSM_QUERY_VERSION},
                      manifest_extra={"warnings": [outage, OTHER],
                                      "subject_kind": "assignment"})
    assert manifest_warnings(out, _manifest(out)) == [outage, OTHER]


def test_manifest_warnings_drop_the_outage_once_a_top_up_wrote_the_file(tmp_path):
    outage = osm_outage_warning("ConnectTimeout")
    out = _staged_dir(tmp_path, osm={"routes": [], "query_version": OSM_QUERY_VERSION},
                      manifest_extra={"warnings": [outage, OTHER],
                                      "osm_refreshed_versions": [OSM_QUERY_VERSION]})
    assert manifest_warnings(out, _manifest(out)) == [OTHER]


def test_manifest_warnings_drop_the_kept_file_note_whenever_osm_json_exists(tmp_path):
    """The kept-file warning is a record of one staging, returned in that
    staging's response; on reuse the file it names is simply there."""
    from sar_geo_mcp.staging import osm_kept_warning

    kept = osm_kept_warning("OverpassUnavailable: every Overpass endpoint refused",
                            OSM_QUERY_VERSION)
    out = _staged_dir(tmp_path, osm={"routes": [], "query_version": OSM_QUERY_VERSION},
                      manifest_extra={"warnings": [kept, OTHER]})
    assert manifest_warnings(out, _manifest(out)) == [OTHER]
    assert _manifest(out)["warnings"] == [kept, OTHER]  # still on record


def test_manifest_warnings_keep_the_outage_while_osm_json_is_missing(tmp_path):
    outage = osm_outage_warning("ConnectTimeout")
    out = _staged_dir(tmp_path, manifest_extra={"warnings": [outage, OTHER]})
    assert manifest_warnings(out, _manifest(out)) == [outage, OTHER]


def test_an_areas_osm_json_retires_the_outage_without_a_top_up_record(tmp_path):
    """An area's bbox never changes, so its file is its own ground."""
    outage = osm_outage_warning("ConnectTimeout")
    out = _staged_dir(tmp_path, osm={"routes": [], "query_version": OSM_QUERY_VERSION},
                      manifest_extra={"warnings": [outage, OTHER], "subject_kind": "area"})
    assert manifest_warnings(out, _manifest(out)) == [OTHER]


def test_an_unparseable_file_a_top_up_wrote_still_retires_the_outage(tmp_path):
    """Whether the file exists and a top-up wrote it, not whether it parses:
    parsing is `refresh_osm`'s question, which refetches or removes an
    unreadable file before `prepare_area` reads these warnings."""
    outage = osm_outage_warning("ConnectTimeout")
    out = _staged_dir(tmp_path, manifest_extra={"warnings": [outage],
                                                "osm_refreshed_versions": [3]})
    (out / "osm.json").write_text("{not json", encoding="utf-8")
    assert manifest_warnings(out, _manifest(out)) == []


# --- a failed top-up says why, never where -------------------------------------

def _raise_on_a_local_path(monkeypatch, tmp_path):
    part = tmp_path / "cache" / "osm" / "tile_z14_2631_6339.json.part"

    def fail(bbox, cache_dir):
        raise PermissionError(13, "Permission denied", str(part))

    monkeypatch.setattr(linear, "routes_for", lambda bbox, cache_dir: [])
    monkeypatch.setattr(linear, "parking_for", fail)


def test_a_failed_top_up_on_a_local_path_does_not_put_the_path_in_the_warning(
        tmp_path, monkeypatch):
    out = _staged_dir(tmp_path, osm={"routes": [], "query_version": 2})
    _raise_on_a_local_path(monkeypatch, tmp_path)

    warnings = refresh_osm(out, tmp_path / "cache")

    assert len(warnings) == 1
    assert "(PermissionError)" in warnings[0]
    assert str(tmp_path) not in warnings[0] and "tile_z14" not in warnings[0]


def test_a_failed_first_fetch_on_a_local_path_does_not_put_the_path_in_the_warning(
        tmp_path, monkeypatch):
    out = _staged_dir(tmp_path)
    _raise_on_a_local_path(monkeypatch, tmp_path)

    warnings = refresh_osm(out, tmp_path / "cache")

    assert len(warnings) == 1 and "still missing" in warnings[0]
    assert "(PermissionError)" in warnings[0]
    assert str(tmp_path) not in warnings[0] and "tile_z14" not in warnings[0]


def test_a_failed_top_up_on_a_url_names_the_status_not_the_url(tmp_path, monkeypatch):
    import httpx

    out = _staged_dir(tmp_path, osm={"routes": [], "query_version": 2})
    url = "https://overpass.agency.example/api/interpreter?key=s3cret"
    request = httpx.Request("POST", url)

    def reject(bbox, cache_dir):
        raise httpx.HTTPStatusError(f"Client error '400 Bad Request' for url '{url}'",
                                    request=request, response=httpx.Response(400, request=request))

    monkeypatch.setattr(linear, "routes_for", reject)

    warnings = refresh_osm(out, tmp_path / "cache")

    assert "(HTTPStatusError 400)" in warnings[0]
    assert "s3cret" not in warnings[0] and "agency.example" not in warnings[0]


# --- an unreadable osm.json is fetched again, not a crash -----------------------

def test_an_unreadable_osm_json_is_fetched_again(tmp_path, monkeypatch):
    out = _staged_dir(tmp_path)
    (out / "osm.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(linear, "routes_for", lambda bbox, cache_dir: [])
    monkeypatch.setattr(linear, "parking_for", lambda bbox, cache_dir: [{"id": "p1"}])
    monkeypatch.setattr(linear, "gates_for", lambda bbox, cache_dir: [])

    warnings = refresh_osm(out, tmp_path / "cache")

    payload = json.loads((out / "osm.json").read_text(encoding="utf-8"))
    assert payload["query_version"] == OSM_QUERY_VERSION
    assert payload["parking"] == [{"id": "p1"}]
    assert len(warnings) == 1 and "could not be read" in warnings[0]
    assert "fetched it again" in warnings[0]


def test_an_unreadable_osm_json_that_cannot_be_fetched_again_is_removed(tmp_path, monkeypatch):
    """Left in place, a file nothing can read would make `osm_available`
    say yes and hide the outage from `manifest_warnings`."""
    outage = osm_outage_warning("ConnectTimeout")
    out = _staged_dir(tmp_path, manifest_extra={"warnings": [outage]})
    (out / "osm.json").write_text("{not json", encoding="utf-8")
    _refuse_overpass(monkeypatch)

    warnings = refresh_osm(out, tmp_path / "cache")

    assert not (out / "osm.json").exists()
    assert len(warnings) == 1 and "could not be read" in warnings[0]
    assert "until it succeeds" in warnings[0]
    assert manifest_warnings(out, _manifest(out)) == [outage]


# --- an unreadable file never survives a refresh that cannot replace it ---------

def test_no_manifest_and_an_unreadable_osm_json_removes_it(tmp_path, monkeypatch):
    out = tmp_path / "staged"
    out.mkdir()
    (out / "osm.json").write_text("{not json", encoding="utf-8")
    _network_forbidden(monkeypatch)

    warnings = refresh_osm(out, tmp_path / "cache")

    assert not (out / "osm.json").exists()
    assert len(warnings) == 1 and "no manifest.json" in warnings[0]
    assert "removed" in warnings[0]


def test_no_bbox_and_an_unreadable_osm_json_removes_it(tmp_path, monkeypatch):
    out = tmp_path / "staged"
    out.mkdir()
    (out / "manifest.json").write_text(json.dumps({"map_id": "M"}), encoding="utf-8")
    (out / "osm.json").write_text("{not json", encoding="utf-8")
    _network_forbidden(monkeypatch)

    warnings = refresh_osm(out, tmp_path / "cache")

    assert not (out / "osm.json").exists()
    assert len(warnings) == 1 and "no bbox_3857" in warnings[0]
    assert "removed" in warnings[0]


def test_a_readable_but_behind_file_without_a_bbox_is_still_left_alone(tmp_path, monkeypatch):
    out = tmp_path / "staged"
    out.mkdir()
    (out / "manifest.json").write_text(json.dumps({"map_id": "M"}), encoding="utf-8")
    (out / "osm.json").write_text(json.dumps({"routes": [], "query_version": 2}),
                                  encoding="utf-8")
    _network_forbidden(monkeypatch)

    refresh_osm(out, tmp_path / "cache")

    assert json.loads((out / "osm.json").read_text(encoding="utf-8"))["query_version"] == 2


@pytest.mark.parametrize("layout", ["refetch fails", "no manifest", "no bbox"])
def test_an_unreadable_osm_json_that_cannot_be_removed_fails_without_a_path(
        tmp_path, monkeypatch, layout):
    from test_staging_subject import _unlink_refused

    if layout == "no manifest":
        out = tmp_path / "staged"
        out.mkdir()
    elif layout == "no bbox":
        out = tmp_path / "staged"
        out.mkdir()
        (out / "manifest.json").write_text(json.dumps({"map_id": "M"}), encoding="utf-8")
    else:
        out = _staged_dir(tmp_path)
    (out / "osm.json").write_text("{not json", encoding="utf-8")
    _refuse_overpass(monkeypatch)
    _unlink_refused(monkeypatch)

    with pytest.raises(RuntimeError) as caught:
        refresh_osm(out, tmp_path / "cache")

    assert str(caught.value) == ("an unreadable osm.json could not be removed "
                                 "(PermissionError); stage again")
    assert str(tmp_path) not in str(caught.value)
    assert caught.value.__context__ is None and caught.value.__cause__ is None


# --- version 4: road class, access and gates -------------------------------------

def test_a_version_4_top_up_names_road_class_access_and_gates(tmp_path, monkeypatch):
    out = _staged_dir(tmp_path, osm={"routes": [], "parking": [], "query_version": 4})
    monkeypatch.setattr(linear, "routes_for", lambda bbox, cache_dir: [])
    monkeypatch.setattr(linear, "parking_for", lambda bbox, cache_dir: [])
    monkeypatch.setattr(linear, "gates_for", lambda bbox, cache_dir: [{"id": "gate-n1"}])
    warnings = refresh_osm(out, tmp_path / "cache")
    payload = json.loads((out / "osm.json").read_text(encoding="utf-8"))
    assert payload["query_version"] == 5 and payload["gates"] == [{"id": "gate-n1"}]
    assert "query version 4" in warnings[0] and "gates" in warnings[0]
    assert "parking" not in warnings[0] and "minor_line" not in warnings[0]


def test_a_failed_version_4_top_up_says_road_facts_stay_unknown(tmp_path, monkeypatch):
    out = _staged_dir(tmp_path, osm={"routes": [], "parking": [], "query_version": 4})
    _refuse_overpass(monkeypatch)
    warnings = refresh_osm(out, tmp_path / "cache")
    assert "road class, access and gates stay unknown" in warnings[0]
    assert warnings[0].endswith("until it succeeds")


def test_a_gates_failure_writes_nothing_and_keeps_the_old_file(tmp_path, monkeypatch):
    """Routes, parking and gates are written together or not at all: a file
    with routes but no gates would read as version 5 with "no gates here"."""
    out = _staged_dir(tmp_path, osm={"routes": [], "parking": [], "query_version": 4})
    before = (out / "osm.json").read_bytes()
    monkeypatch.setattr(linear, "routes_for", lambda bbox, cache_dir: [])
    monkeypatch.setattr(linear, "parking_for", lambda bbox, cache_dir: [])

    def refuse(bbox, cache_dir):
        raise OverpassUnavailable("every Overpass endpoint refused", status=429)

    monkeypatch.setattr(linear, "gates_for", refuse)
    warnings = refresh_osm(out, tmp_path / "cache")
    assert (out / "osm.json").read_bytes() == before
    assert "still at query version 4" in warnings[0]


# --- what an old file lacks is cumulative ------------------------------------------

def test_a_version_2_file_lacks_parking_distribution_lines_and_road_facts():
    """Every group it lacks, parking first: a version-2 file was staged
    before parking (3), distribution lines (4) and road facts (5)."""
    from sar_geo_mcp.staging import _what_an_old_osm_lacks

    assert _what_an_old_osm_lacks(2) == (
        "before parking, distribution power lines (power=minor_line) or road class, "
        "access and gates were asked for",
        "parking stays unreadable, distribution power lines (power=minor_line) stay "
        "unchecked, and road class, access and gates stay unknown on this ground",
        "parking can now be read, distribution power lines (power=minor_line) are "
        "checked, and road class, access and gates can now be read here")


def test_a_version_3_file_lacks_distribution_lines_and_road_facts_not_parking():
    from sar_geo_mcp.staging import _what_an_old_osm_lacks

    assert _what_an_old_osm_lacks(3) == (
        "before distribution power lines (power=minor_line) or road class, access "
        "and gates were asked for",
        "parking is unaffected; distribution power lines (power=minor_line) stay "
        "unchecked, and road class, access and gates stay unknown on this ground",
        "distribution power lines (power=minor_line) are checked, and road class, "
        "access and gates can now be read here")


def test_a_version_4_file_lacks_only_road_facts():
    from sar_geo_mcp.staging import _what_an_old_osm_lacks

    assert _what_an_old_osm_lacks(4) == (
        "before road class, access and gates were asked for",
        "parking and power lines are unaffected; road class, access and gates stay "
        "unknown on this ground",
        "road class, access and gates can now be read here")


def test_a_failed_version_2_top_up_names_every_missing_group(tmp_path, monkeypatch):
    out = _staged_dir(tmp_path, osm={"routes": [{"id": "old"}], "query_version": 2})
    _refuse_overpass(monkeypatch)
    warning = refresh_osm(out, tmp_path / "cache")[0]
    assert "parking stays unreadable" in warning and "minor_line" in warning
    assert "road class, access and gates stay unknown" in warning
    assert warning.endswith("on this ground until it succeeds")


def test_a_gates_failure_with_a_path_puts_no_path_in_the_top_up_warning(tmp_path, monkeypatch):
    out = _staged_dir(tmp_path, osm={"routes": [], "parking": [], "query_version": 4})
    monkeypatch.setattr(linear, "routes_for", lambda bbox, cache_dir: [])
    monkeypatch.setattr(linear, "parking_for", lambda bbox, cache_dir: [])
    part = tmp_path / "cache" / "osm_tile_v5_c2631_r6339.json.part"

    def fail(bbox, cache_dir):
        raise PermissionError(13, "Permission denied", str(part))

    monkeypatch.setattr(linear, "gates_for", fail)
    warning = refresh_osm(out, tmp_path / "cache")[0]
    assert "(PermissionError)" in warning
    assert str(tmp_path) not in warning and "osm_tile" not in warning


# --- a missing or unreadable file names everything osm.json carries (0.18.0) -------

EVERYTHING = "routes, parking, road class, access and gates"


def _answers(monkeypatch):
    monkeypatch.setattr(linear, "routes_for", lambda bbox, cache_dir: [])
    monkeypatch.setattr(linear, "parking_for", lambda bbox, cache_dir: [])
    monkeypatch.setattr(linear, "gates_for", lambda bbox, cache_dir: [])


def test_a_missing_file_fetched_names_everything_it_now_carries(tmp_path, monkeypatch):
    out = _staged_dir(tmp_path)
    _answers(monkeypatch)
    assert refresh_osm(out, tmp_path / "cache") == [
        "osm.json was missing from this ground (an earlier Overpass fetch had failed); "
        f"fetched it now over 1 Overpass tile so {EVERYTHING} can be read here"]


def test_a_missing_file_that_fails_again_names_everything_and_ends_on_until(
        tmp_path, monkeypatch):
    out = _staged_dir(tmp_path)
    _refuse_overpass(monkeypatch)
    assert refresh_osm(out, tmp_path / "cache") == [
        "osm.json is still missing: fetching it failed again (OverpassUnavailable: "
        f"every Overpass endpoint refused); {EVERYTHING} stay unreadable from this "
        "ground until it succeeds"]


def test_an_unreadable_file_fetched_again_names_everything(tmp_path, monkeypatch):
    out = _staged_dir(tmp_path)
    (out / "osm.json").write_text("{trunc", encoding="utf-8")
    _answers(monkeypatch)
    assert refresh_osm(out, tmp_path / "cache") == [
        "osm.json could not be read (it is not valid JSON); fetched it again over "
        f"1 Overpass tile so {EVERYTHING} can be read here"]


def test_an_unreadable_file_that_fails_again_names_everything_and_ends_on_until(
        tmp_path, monkeypatch):
    out = _staged_dir(tmp_path)
    (out / "osm.json").write_text("{trunc", encoding="utf-8")
    _refuse_overpass(monkeypatch)
    assert refresh_osm(out, tmp_path / "cache") == [
        "osm.json could not be read (it is not valid JSON) and fetching it again "
        "failed (OverpassUnavailable: every Overpass endpoint refused), so it was "
        f"removed; {EVERYTHING} stay unreadable from this ground until it succeeds"]


def test_an_unreadable_file_with_no_manifest_names_everything(tmp_path):
    out = tmp_path / "staged"
    out.mkdir()
    (out / "osm.json").write_text("{trunc", encoding="utf-8")
    assert refresh_osm(out, tmp_path / "cache") == [
        "osm.json could not be read (it is not valid JSON) but this directory has no "
        "manifest.json to read a bbox from, so it was removed; "
        f"{EVERYTHING} stay unreadable from this ground"]


# --- saved terrain metrics follow the osm.json they were measured from --------

def test_a_successful_top_up_removes_terrain_metrics_measured_from_the_old_file(
        tmp_path, monkeypatch):
    """Terrain metrics read osm.json (routes and access), so metrics saved
    before a top-up describe roads the directory no longer holds."""
    out = _staged_dir(tmp_path, osm={"routes": [], "query_version": 2})
    (out / "terrain_metrics.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(linear, "routes_for", lambda bbox, cache_dir: [])
    monkeypatch.setattr(linear, "parking_for", lambda bbox, cache_dir: [])
    monkeypatch.setattr(linear, "gates_for", lambda bbox, cache_dir: [])

    refresh_osm(out, tmp_path / "cache")

    assert not (out / "terrain_metrics.json").exists()


def test_a_failed_or_needless_top_up_keeps_terrain_metrics(tmp_path, monkeypatch):
    failed = _staged_dir(tmp_path, osm={"routes": [], "query_version": 2})
    (failed / "terrain_metrics.json").write_text("{}", encoding="utf-8")
    _refuse_overpass(monkeypatch)
    refresh_osm(failed, tmp_path / "cache")
    assert (failed / "terrain_metrics.json").exists()

    (tmp_path / "second").mkdir()
    current = _staged_dir(tmp_path / "second",
                          osm={"routes": [], "parking": [], "query_version": OSM_QUERY_VERSION})
    (current / "terrain_metrics.json").write_text("{}", encoding="utf-8")
    _network_forbidden(monkeypatch)
    refresh_osm(current, tmp_path / "cache")
    assert (current / "terrain_metrics.json").exists()
