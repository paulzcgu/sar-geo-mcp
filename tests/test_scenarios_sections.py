"""The machinery every scenario shares (`scenarios/_sections.py`): section
status, the refusals made before any work, staging each ground once, error
text that never carries a credential, and the section helpers. The
scenarios' own tests are in `test_scenarios_medevac.py`,
`test_scenarios_initial_response.py`, `test_scenarios_urban_search.py` and
`test_scenarios_debrief.py`.
No network: staging and every feature are stubbed."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from pathlib import Path
from typing import get_args, get_type_hints

import pytest

from sar_geo_mcp.components import candidates as candidates_component
from sar_geo_mcp.components import staging as staging_component
from sar_geo_mcp.config import CaltopoConfig
from sar_geo_mcp.context import ServerContext
from sar_geo_mcp.features import hasty as hasty_feature
from sar_geo_mcp.features import parking as parking_feature
from sar_geo_mcp.planning import parking as parking_domain
from sar_geo_mcp.scenarios import MODULES
from sar_geo_mcp.scenarios._sections import (
    FAILED,
    OK,
    SKIPPED,
    Ground,
    Outcome,
    StagingFailed,
    call_feature,
    check_egress_overrides,
    check_hasty_overrides,
    check_limit,
    check_overrides,
    check_per_category,
    check_point,
    check_radius,
    compact_exit,
    error_text,
    ground_handle,
    guard,
    lz_marker,
    merged_warnings,
    overall_status,
    parking_section,
    requested,
    rings_ground,
    route_line,
    section,
    stage_grounds,
)
from sar_geo_mcp.scenarios import _sections

POINT = (37.4022, -122.3164)
GEOMETRY_TYPES = {"Polygon", "MultiPolygon", "LineString", "MultiLineString"}
#: The staging components as they are, before any test stands a stub in:
#: a scenario run with caching disabled must reach the real refusal.
REAL_PREPARE_AREA = staging_component.prepare_area
REAL_PREPARE_TERRAIN = staging_component.prepare_terrain
#: A credential of each kind the server holds, none containing another, and
#: an error message carrying all three - as a CalTopo GetMap URL carries the
#: WMS token in its path.
SECRETS = {"wms_token": "WMSs3cr3tT0k3n", "mapbox_token": "pk.m4pb0xS3cr3t",
           "google_maps_key": "AIzaG00gl3S3cr3t"}
LEAK = ("GET https://caltopo.com/api/{wms_token}/wms failed; "
        "https://api.mapbox.com/v4/x.png?access_token={mapbox_token}; "
        "https://maps.googleapis.com/maps/api/staticmap?key={google_maps_key}"
        ).format(**SECRETS)
#: LEAK as every error a scenario writes down must give it.
LEAK_REDACTED = ("RuntimeError: GET https://caltopo.com/api/<redacted>/wms failed; "
                 "https://api.mapbox.com/v4/x.png?access_token=<redacted>; "
                 "https://maps.googleapis.com/maps/api/staticmap?key=<redacted>")
#: Each scenario's own tests, by the scenario function's name: the module
#: whose `Stubs` stand in for its staging and features, and the name there
#: of the point it is called at (None: an assignment, not a point).
SCENARIO_TESTS = {"plan_medevac": ("test_scenarios_medevac", "PATIENT"),
                  "plan_initial_response": ("test_scenarios_initial_response", "PLS"),
                  "plan_urban_search": ("test_scenarios_urban_search", "SAN_CARLOS"),
                  "debrief_assignment": ("test_scenarios_debrief", None)}


class FakeStaging:
    """`prepare_area` without the network: records each call, writes the
    one file `ctx.staged` needs to resolve an area handle, and refuses the
    resolutions it is told to, as an unreachable 3DEP would."""

    def __init__(self, fail_resolutions=()):
        self.calls = []
        self.fail = set(fail_resolutions)

    def __call__(self, ctx, **arguments):
        self.calls.append(arguments)
        if arguments.get("resolution_m") in self.fail:
            raise RuntimeError("3DEP could not be reached")
        digest = hashlib.sha256(json.dumps(arguments, sort_keys=True).encode()).hexdigest()
        handle = f"aoi-{digest[:16]}"
        directory = Path(ctx.config.cache_dir) / "staged" / handle
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "manifest.json").write_text(json.dumps({"map_id": ""}), encoding="utf-8")
        radius = arguments["radius_m"]
        return {"handle": handle, "staged": True, "reason": "not staged before",
                "extent_m": 2 * radius, "dem_resolution_m": arguments.get("resolution_m"),
                "warnings": ["evt unavailable"], "map_provenance": None,
                "osm_available": True,
                "area": {"kind": "circle", "radius_m": radius,
                         "center": {"latitude": arguments["center_latitude"],
                                    "longitude": arguments["center_longitude"]},
                         "geometry": {"type": "Polygon", "coordinates": [[
                             [0, 0], [0, 1], [1, 1], [0, 0]]]}}}


class LeakyStaging(FakeStaging):
    """FakeStaging whose refusal carries every configured credential, as an
    error from below the component may (LEAK)."""

    def __call__(self, ctx, **arguments):
        if arguments.get("resolution_m") in self.fail:
            self.calls.append(arguments)
            raise RuntimeError(LEAK)
        return super().__call__(ctx, **arguments)


@pytest.fixture
def ctx(tmp_path):
    return ServerContext.build(CaltopoConfig(cache_dir=tmp_path / "cache"))


@pytest.fixture
def secret_ctx(tmp_path):
    """A context holding all three credentials (SECRETS)."""
    return ServerContext.build(CaltopoConfig(cache_dir=tmp_path / "cache", **SECRETS))


def leaks(payload) -> list[str]:
    """Every configured credential found anywhere in a payload."""
    text = json.dumps(payload)
    return [secret for secret in SECRETS.values() if secret in text]


def raise_leak(*args, **kwargs):
    raise RuntimeError(LEAK)


def scenario_under_stubs(name: str, monkeypatch, **options):
    """(the scenario function `name`, the leading arguments its own tests
    call it with, those tests' `Stubs` - staging and every feature stood in
    for, each call recorded). Imported here, not at the top: those modules
    import this one for FakeStaging."""
    module_name, point = SCENARIO_TESTS[name]
    tests = importlib.import_module(module_name)
    function = getattr(scenario_module(name), name)
    arguments = (getattr(tests, point) if point is not None
                 else (tests.MAP_ID, tests.LABEL, tests.TRACKS))
    return function, tuple(arguments), tests.Stubs(monkeypatch, **options)


def scenario_module(name: str):
    """The `scenarios` module defining the scenario function `name`."""
    return next(module for module in MODULES if hasattr(module, name))


def geometry_types_in(value) -> set[str]:
    """Every GeoJSON geometry type anywhere in a payload."""
    found = set()
    if isinstance(value, dict):
        if isinstance(value.get("type"), str) and "coordinates" in value:
            found.add(value["type"])
        for item in value.values():
            found |= geometry_types_in(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            found |= geometry_types_in(item)
    return found


# --- Status ------------------------------------------------------------------


@pytest.mark.parametrize("statuses, expected", [
    ([OK, OK], OK),
    ([OK, SKIPPED], OK),
    ([FAILED, FAILED], FAILED),
    ([FAILED, SKIPPED], FAILED),
    ([OK, FAILED], "partial"),
    ([OK, FAILED, SKIPPED], "partial"),
])
def test_overall_status_counts_only_the_sections_asked_for(statuses, expected):
    sections = {str(i): section(status) for i, status in enumerate(statuses)}
    assert overall_status(sections) == expected


@pytest.mark.parametrize("export, others, expected", [
    (OK, [FAILED, FAILED], FAILED),          # an export that drew nothing holds nothing up
    (FAILED, [OK, OK], OK),                  # nor does one that crashed take anything down
    (OK, [OK, FAILED], "partial"),
    (OK, [SKIPPED, OK], OK),
])
def test_the_export_section_never_counts_in_the_overall_status(export, others, expected):
    """It finds nothing of its own, and its every problem is a warning: an
    unwritable file must not read as a finding lost."""
    sections = {str(i): section(status) for i, status in enumerate(others)}
    assert overall_status({**sections, "export": section(export)}) == expected


@pytest.mark.xfail(strict=True, reason="PR #4 review B7: overall_status answers 'ok' when no "
                                       "section counts (all([]) is True)")
@pytest.mark.parametrize("sections", [
    {},
    {"lz": section(SKIPPED)},
    {"lz": section(SKIPPED), "egress": section(SKIPPED)},
    {"export": section(OK)},
    {"hasty": section(SKIPPED), "export": section(OK)},
], ids=["nothing", "one-skipped", "all-skipped", "export-alone", "skipped-and-export"])
def test_overall_status_never_says_ok_when_no_section_counts(sections):
    """"ok" tells the reader the plan holds answers: from a call that ran
    nothing it would read as a medevac or first response laid out. Refusing
    to answer at all is as good as any status but "ok"."""
    try:
        status = overall_status(sections)
    except (AssertionError, ValueError):
        return
    assert status != OK


def test_a_failed_section_that_still_holds_an_answer_makes_the_status_partial():
    """An egress whose carry-out raised but whose walk-out answered has a
    way out to report: "failed" would tell the reader to throw it away."""
    egress = section(FAILED, error="ValueError: x",
                     modes={"carry_out": {"status": FAILED}, "walk_out": {"status": OK}})
    nothing = section(FAILED, error="ValueError: x",
                      modes={"carry_out": {"status": FAILED}, "walk_out": {"status": FAILED}})
    assert overall_status({"egress": egress}) == "partial"
    assert overall_status({"egress": egress, "lz": section(FAILED)}) == "partial"
    assert overall_status({"egress": nothing, "lz": section(FAILED)}) == FAILED
    assert overall_status({"egress": egress, "lz": section(SKIPPED)}) == "partial"
    # An urban grouping that raised after the blocks were drawn: the blocks
    # are saved and still answer.
    urban = section(FAILED, error="ValueError: x",
                    blocks={"set_id": "populated-b10c55", "count": 9})
    assert overall_status({"urban": urban}) == "partial"
    assert overall_status({"urban": section(FAILED, error="ValueError: x")}) == FAILED


def test_the_summary_reads_answered_sections_and_egress_in_part():
    """One `summarise` for every scenario: a block per section asked for, in
    the facts' order, null when the section did not answer - and an egress
    with one mode back still says that mode's quickest exit, with null for
    the mode that failed, but only when it is named as read in part."""
    exit_ = {"kind": "2wd", "reachable": True, "minutes": 41.0, "distance_m": 2300.0,
             "off_trail_m": 120.0, "road_name": "Old Haul Rd", "road_access": "private",
             "gate_on_road": True,
             "parking_lot": None}
    egress = section(FAILED, error="ValueError: x", modes={
        "carry_out": {"status": FAILED, "exits": []},
        "walk_out": {"status": OK, "exits": [exit_, {"kind": "4x4", "reachable": False}]}})
    sections = {"lz": section(OK, n=2), "egress": egress, "medical": section(FAILED),
                "export": section(OK)}
    facts = {"medical": lambda s: "medical", "egress": _sections.egress_facts,
             "lz": lambda s: s["n"]}
    asked = ("lz", "egress", "medical", "export")

    summary = _sections.summarise(sections, asked, facts, in_part=("egress",))
    assert list(summary) == ["medical", "egress", "lz"]
    # The quickest exit keeps its vehicle end: read first, a gated private
    # road that said only its minutes would read as one an ambulance can use.
    assert summary == {"medical": None, "lz": 2, "egress": {
        "carry_out": None,
        "walk_out": {"kind": "2wd", "minutes": 41.0, "distance_m": 2300.0,
                     "off_trail_m": 120.0, "road_name": "Old Haul Rd", "road_access": "private",
                     "gate_on_road": True, "parking_lot": None}}}
    # Not named as read in part, a failed section is null like any other;
    # left out of the sections asked for, it has no block at all.
    assert _sections.summarise(sections, asked, facts)["egress"] is None
    assert "lz" not in _sections.summarise(sections, ("egress",), facts)
    # A mode that answered with no reachable exit says so, not null.
    stuck = section(OK, modes={"walk_out": {"status": OK, "exits": [
        {"kind": "2wd", "reachable": False}, {"kind": "4x4", "reachable": False}]}})
    assert _sections.egress_facts(stuck) == {"walk_out": "none reachable"}


def test_overrides_for_a_section_left_out_are_said_unused_in_section_order():
    overrides = {"lz": {"max_slope_deg": 6.0}, "egress": {}, "medical": {"search_radius_km": 9}}
    assert _sections.unused_overrides(overrides, ("lz", "egress")) == [
        "overrides['medical'] was not used: medical was left out of include"]
    assert _sections.unused_overrides(overrides, ("egress",)) == [
        "overrides['lz'] was not used: lz was left out of include",
        "overrides['medical'] was not used: medical was left out of include"]
    assert _sections.unused_overrides(overrides, ("lz", "egress", "medical")) == []


def test_a_section_always_has_status_error_and_warnings():
    assert section(OK, count=3) == {"status": OK, "error": None, "warnings": [], "count": 3}


def test_guard_records_an_exception_and_lets_an_interrupt_through(ctx):
    def broken():
        raise ValueError("no canopy panel")

    failed = guard(ctx, broken)
    assert failed.section == {"status": FAILED, "error": "ValueError: no canopy panel",
                              "warnings": []}
    assert failed.features == []

    def interrupted():
        raise KeyboardInterrupt

    # BaseException is the operator's: Ctrl-C must still stop the call.
    with pytest.raises(KeyboardInterrupt):
        guard(ctx, interrupted)


def test_a_feature_call_that_raised_leaves_its_detail_call_on_the_failed_section(ctx):
    def no_canopy(ctx, **arguments):
        raise ValueError("no canopy panel")

    arguments = {"handle": "aoi-0123456789abcdef", "limit": 5}
    failed = guard(ctx, lambda: Outcome(section(
        OK, answer=call_feature(ctx, "lz_candidates", no_canopy, arguments))))
    assert failed.section == {
        "status": FAILED, "error": "ValueError: no canopy panel", "warnings": [],
        "detail_call": {"tool": "lz_candidates", "arguments": arguments}}

    def interrupted(ctx, **arguments):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        guard(ctx, lambda: call_feature(ctx, "lz_candidates", interrupted, {}))


def test_error_text_never_carries_a_configured_credential(tmp_path):
    ctx = ServerContext.build(CaltopoConfig(cache_dir=tmp_path, wms_token="SEKRITTOKEN",
                                            mapbox_token="pk.MAPBOXKEY"))
    error = RuntimeError("GET https://caltopo.com/api/SEKRITTOKEN/wms?x=1 failed; "
                         "also https://api.mapbox.com/?access_token=pk.MAPBOXKEY")
    text = error_text(ctx, error)
    assert "SEKRITTOKEN" not in text and "MAPBOXKEY" not in text
    assert text.startswith("RuntimeError: GET https://caltopo.com/api/<redacted>/wms")


def test_error_text_blanks_the_google_maps_key_too(secret_ctx):
    text = error_text(secret_ctx, RuntimeError(LEAK))
    assert leaks(text) == [] and text == LEAK_REDACTED


@pytest.mark.xfail(strict=True, reason="PR #4 review B28: error_text redacts credentials in a "
                                       "fixed order, so one containing another leaks its "
                                       "remainder")
@pytest.mark.parametrize("credentials, secret", [
    ({"wms_token": "AB12", "google_maps_key": "AB12cdEF34"}, "AB12cdEF34"),
    ({"wms_token": "AB12", "mapbox_token": "pk.AB12xyz"}, "pk.AB12xyz"),
    ({"mapbox_token": "pk.ABC", "google_maps_key": "pk.ABCdefGHI"}, "pk.ABCdefGHI"),
], ids=["wms-in-google", "wms-in-mapbox", "mapbox-in-google"])
def test_a_credential_containing_another_is_redacted_whole(tmp_path, credentials, secret):
    ctx = ServerContext.build(CaltopoConfig(cache_dir=tmp_path, **credentials))
    text = error_text(ctx, RuntimeError(f"https://example/?key={secret} failed"))
    assert text == "RuntimeError: https://example/?key=<redacted> failed"


def test_guard_redacts_on_both_of_its_paths(secret_ctx):
    """A feature call that raised (FeatureFailed, which keeps its call) and
    anything else a section raises: each error is "<Type>: <message>" with
    every credential blanked."""
    failed = guard(secret_ctx, lambda: Outcome(section(
        OK, answer=call_feature(secret_ctx, "lz_candidates", raise_leak, {"limit": 5}))))
    assert failed.section["error"] == LEAK_REDACTED
    assert failed.section["detail_call"] == {"tool": "lz_candidates",
                                             "arguments": {"limit": 5}}
    other = guard(secret_ctx, raise_leak)
    assert other.section["error"] == LEAK_REDACTED
    for outcome in (failed, other):
        assert leaks(outcome.section) == [] and "<redacted>" in outcome.section["error"]


def test_a_staging_record_and_an_export_warning_never_carry_a_credential(secret_ctx,
                                                                        monkeypatch):
    """The other places an error is written down: a ground or an assignment
    that could not be staged, draft assignments or an export file that could
    not be written."""
    monkeypatch.setattr(staging_component, "prepare_area", LeakyStaging({2.0}))
    staging = stage_grounds(secret_ctx, {"lz": Ground(1000, 2.0), "egress": Ground(3000, 10.0)},
                            latitude=POINT[0], longitude=POINT[1])
    monkeypatch.setattr(staging_component, "prepare_terrain", raise_leak)
    staging["assignment"] = _sections.stage_assignment(secret_ctx, map_id="MAP1234", label="AB")
    monkeypatch.setattr(candidates_component, "caltopo_export", raise_leak)
    drafts = _sections.set_export_section(
        secret_ctx, "hasty", section(OK, handle=staging["egress"]["handle"],
                                     set_id="hasty-a1b2c3"), overrides={})
    monkeypatch.setattr(_sections, "atomic_write_text", raise_leak)
    paths, problems = _sections.write_export(secret_ctx, staging, ("lz", "egress"),
                                             "medevac.geojson", [])

    assert staging["egress"]["status"] == OK and paths == []
    assert staging["lz"]["error"] == staging["assignment"]["error"] == LEAK_REDACTED
    assert drafts.section["warnings"] == [
        f"the draft assignments could not be written ({LEAK_REDACTED}); every section's "
        "answer stands, and detail_call makes the export again"]
    assert problems == [f"the CalTopo export could not be written ({LEAK_REDACTED}); every "
                        "section's answer stands"]
    assert leaks([staging, drafts.section, problems]) == []
    with pytest.raises(StagingFailed) as unstaged:
        ground_handle(staging, "lz")
    assert leaks(str(unstaged.value)) == [] and LEAK_REDACTED in str(unstaged.value)


# --- Refusals ----------------------------------------------------------------


@pytest.mark.parametrize("latitude, longitude", [
    (91, 0), (-91, 0), (0, 181), (0, -181), (float("nan"), 0), (float("-inf"), 0),
    (0, float("inf")), ("37.4", -122.3), (True, 0), (False, 0), (None, 0), (0, None)])
def test_bad_coordinates_are_refused(latitude, longitude):
    with pytest.raises(ValueError, match="latitude|longitude"):
        check_point(latitude, longitude)


@pytest.mark.parametrize("latitude, longitude", [
    (90, 0), (-90, 0), (0, 180), (0, -180), (90.0, 180.0), (-90.0, -180.0), (0, 0)])
def test_coordinates_on_the_limits_are_accepted(latitude, longitude):
    check_point(latitude, longitude)


@pytest.mark.parametrize("radius", [0, -5, 4500, 4500.1, float("inf"), True, "1000"])
def test_a_radius_out_of_bounds_is_refused_first(radius):
    with pytest.raises(ValueError, match="under 4,500 m"):
        check_radius("lz_radius_m", radius, latitude=POINT[0], longitude=POINT[1])


@pytest.mark.parametrize("latitude", [0.0, 37.4022, 64.0, 84.0])
def test_check_radius_refuses_exactly_what_prepare_area_refuses(latitude):
    """prepare_area measures its own circle polygon, which puts a 4,500 m
    circle a hair over 9,000 m across - and a 4,499.9 m one over it near the
    poles. Whatever check_radius lets through, build_area (the step
    prepare_area refuses with) must accept, and the other way round."""
    from sar_geo_mcp.planning.area import build_area

    for radius in (4499.0, 4499.9, 4499.99, 4499.9999):
        try:
            build_area(center_latitude=latitude, center_longitude=-122.3, radius_m=radius)
        except ValueError:
            with pytest.raises(ValueError, match="prepare_area would refuse it - the area "
                                                 "is 9000 m across"):
                check_radius("lz_radius_m", radius, latitude=latitude, longitude=-122.3)
        else:
            check_radius("lz_radius_m", radius, latitude=latitude, longitude=-122.3)
    # 4,499 m stages everywhere a scenario's point can be, short of the pole.
    check_radius("lz_radius_m", 4499.0, latitude=latitude, longitude=-122.3,
                 resolution_m=2.0)


def test_check_radius_refuses_a_ground_too_many_pixels_at_its_resolution():
    """3DEP serves at most 8,000 pixels a side: 4,000 m round the point plus
    the buffer at 1 m is 10,000, which prepare_area refuses before fetching."""
    check_radius("lz_radius_m", 4000, latitude=POINT[0], longitude=POINT[1],
                 resolution_m=2.0)
    with pytest.raises(ValueError, match="lz_radius_m of 4,000 m cannot be staged here.*"
                                         "8000 per axis"):
        check_radius("lz_radius_m", 4000, latitude=POINT[0], longitude=POINT[1],
                     resolution_m=1.0)


@pytest.mark.xfail(strict=True, reason="PR #4 review B26: near the antimeridian or a pole a "
                                       "ground is refused with 'use a smaller radius' advice "
                                       "that cannot work")
@pytest.mark.parametrize("latitude, longitude", [(0.0, 180.0), (0.0, -180.0), (90.0, 0.0),
                                                 (-90.0, 0.0)])
def test_a_ground_refused_for_where_it_lies_names_the_place_not_the_radius(latitude,
                                                                           longitude):
    """No 10 m circle stages at (0, 180) or on a pole: the cause is the
    location, and advice to shrink the radius sends the caller round in
    circles."""
    with pytest.raises(ValueError) as refused:
        check_radius("lz_radius_m", 10, latitude=latitude, longitude=longitude)
    message = str(refused.value)
    assert "smaller radius" not in message
    assert "antimeridian" in message or "pole" in message


@pytest.mark.xfail(strict=True, reason="PR #4 review B26: a scenario refuses a section that "
                                       "stages nothing over another section's ground near the "
                                       "antimeridian or a pole")
@pytest.mark.parametrize("latitude, longitude", [(0.0, 180.0), (90.0, 0.0)])
def test_a_section_that_stages_nothing_answers_at_the_antimeridian_and_the_pole(
        ctx, monkeypatch, latitude, longitude):
    """The medical tool answers at any point; plan_medevac with medical
    alone stages nothing, so no ground it leaves out may refuse it."""
    plan_medevac, _patient, stubs = scenario_under_stubs("plan_medevac", monkeypatch)
    result = plan_medevac(ctx, latitude, longitude, include=["medical"])
    assert result["status"] == OK and result["sections"]["medical"]["status"] == OK
    assert len(stubs.calls["medical"]) == 1 and stubs.staging.calls == []


def test_the_rings_cap_stages_up_to_88_degrees_either_side():
    """Where the walking rings' cap does stage (B27 pins where it does not)."""
    for latitude in (88.0, -88.0, 60.0, 0.0):
        check_radius("rings", _sections.RINGS_MAX_RADIUS_M, latitude=latitude, longitude=0.0,
                     resolution_m=_sections.WALKING_RESOLUTION_M)


@pytest.mark.xfail(strict=True, reason="PR #4 review B27: above about 88.08 deg the rings' "
                                       "4,499 m cap is refused over a radius the caller never "
                                       "named, with 'use a smaller radius'")
def test_near_the_pole_the_rings_are_not_refused_over_a_radius_the_caller_never_named(
        ctx, monkeypatch):
    """Either the rings run on the ground that does stage there, or the
    refusal names what the caller controls (ring_minutes) - never "a smaller
    radius", which plan_initial_response has no argument for."""
    plan, _place, stubs = scenario_under_stubs("plan_initial_response", monkeypatch)
    try:
        result = plan(ctx, 88.58, 0.0, include=["travel_rings"])
    except ValueError as error:
        assert "smaller radius" not in str(error)
        assert "ring_minutes" in str(error)
        assert stubs.nothing_called()
        return
    assert result["sections"]["travel_rings"]["status"] == OK
    (radius,) = stubs.radii().values()
    check_radius("rings", radius, latitude=88.58, longitude=0.0,
                 resolution_m=_sections.WALKING_RESOLUTION_M)


def test_near_the_pole_the_rings_never_refuse_a_call_that_leaves_them_out(ctx, monkeypatch):
    plan, _place, stubs = scenario_under_stubs("plan_initial_response", monkeypatch)
    result = plan(ctx, 88.58, 0.0, include=["hasty"])
    assert result["status"] == OK and result["sections"]["hasty"]["status"] == OK
    assert stubs.radii() == {None: 1500.0}


def test_include_is_checked_and_kept_in_the_scenario_s_order():
    names = ("lz", "egress", "medical")
    assert requested(None, names) == names
    assert requested(["medical", "lz", "lz"], names) == ("lz", "medical")
    with pytest.raises(ValueError, match=r"\['lzz'\].*its sections are"):
        requested(["lzz"], names)
    with pytest.raises(ValueError, match="empty"):
        requested([], names)
    with pytest.raises(ValueError, match="list"):
        requested("lz", names)
    # The export draws what the others find: alone it would draw nothing.
    with_export = ("lz", "medical", "export")
    assert requested(["export", "lz"], with_export) == ("lz", "export")
    with pytest.raises(ValueError, match=r"include names only export.*\['lz', 'medical'\]"):
        requested(["export", "export"], with_export)


@pytest.mark.xfail(strict=True, reason="PR #4 review B7: requested() uses up a one-shot include "
                                       "iterable, so every section is skipped")
@pytest.mark.parametrize("include, expected", [
    (lambda: iter(["medical", "lz"]), ("lz", "medical")),
    (lambda: (name for name in ["egress"]), ("egress",)),
    (lambda: map(str, ["lz", "lz"]), ("lz",)),
], ids=["iter", "generator", "map"])
def test_include_given_as_a_one_shot_iterable_is_read_once(include, expected):
    """A direct caller (1c makes every tool a plain function) may pass any
    iterable: it is the same request as the list."""
    assert requested(include(), ("lz", "egress", "medical")) == expected


@pytest.mark.xfail(strict=True, reason="PR #4 review B7: requested() uses up a one-shot include "
                                       "iterable, so its refusals are skipped")
@pytest.mark.parametrize("include, names, message", [
    (lambda: iter([]), ("lz", "egress", "medical"), "include is empty"),
    (lambda: iter(["export"]), ("hasty", "parking", "export"), "include names only export"),
], ids=["empty", "export-alone"])
def test_a_one_shot_include_is_refused_as_its_list_would_be(include, names, message):
    with pytest.raises(ValueError, match=message):
        requested(include(), names)


@pytest.mark.xfail(strict=True, reason="PR #4 review B7: a one-shot include runs nothing and "
                                       "answers 'ok'")
def test_a_scenario_given_a_one_shot_include_answers_as_for_the_list(ctx, monkeypatch):
    plan_medevac, patient, stubs = scenario_under_stubs("plan_medevac", monkeypatch)
    result = plan_medevac(ctx, *patient, include=iter(["lz", "medical"]))
    assert {name: entry["status"] for name, entry in result["sections"].items()} == {
        "lz": OK, "egress": SKIPPED, "medical": OK}
    assert result["status"] == OK and set(result["summary"]) == {"lz", "medical"}
    assert len(stubs.staging.calls) == 1 and len(stubs.calls["lz"]) == 1


@pytest.mark.xfail(strict=True, reason="PR #4 review B29: malformed override keys or include "
                                       "values raise TypeError instead of the promised "
                                       "ValueError")
@pytest.mark.parametrize("call, message", [
    (lambda: check_overrides("plan_medevac", {1: {}, "zz": {}}, ("lz", "egress", "medical"), {}),
     "names no section"),
    (lambda: check_overrides("plan_medevac", {"lz": {1: 2, "zz": 3}},
                             ("lz", "egress", "medical"), {}),
     "does not take"),
    (lambda: check_overrides("plan_x", {"urban": {1: {}, "zz": {}}}, ("urban",), {}),
     "no tool it calls"),
    (lambda: check_overrides("plan_x", {"urban": {"populated_areas": {1: 2, "zz": 3}}},
                             ("urban",), {}),
     "does not take"),
    (lambda: requested(5, ("lz", "egress", "medical")), "include must be a list"),
], ids=["section-keys", "keyword-keys", "tool-keys", "urban-keyword-keys", "include-int"])
def test_malformed_keys_and_include_are_refused_with_a_value_error(call, message):
    with pytest.raises(ValueError, match=message):
        call()


@pytest.mark.xfail(strict=True, reason="PR #4 review B29: _check_parts raises a bare "
                                       "StopIteration when scenario_sets names every keyword "
                                       "the last urban tool takes")
def test_an_urban_override_check_needs_no_free_keyword_for_its_example():
    """The shape example a refusal gives is built from a keyword the last
    tool takes and nobody sets: with none left, the check itself must still
    answer - "no overrides" as none, a bad shape as a ValueError."""
    sets = {"urban": {"max_distance_m": "radius", "marker_color": "colour"}}
    assert check_overrides("plan_x", {"urban": {}}, ("urban",), sets) == {"urban": {}}
    assert check_overrides("plan_x", None, ("urban",), sets) == {"urban": {}}
    with pytest.raises(ValueError, match="no tool it calls"):
        check_overrides("plan_x", {"urban": {"bridge_m": 10}}, ("urban",), sets)


def test_overrides_are_checked_against_the_feature_tool_and_the_scenario():
    sections = ("lz", "egress", "medical")
    sets = {"lz": {"limit": "lz_limit"}}
    assert check_overrides("plan_x", None, sections, sets) == {
        "lz": {}, "egress": {}, "medical": {}}
    assert check_overrides("plan_x", {"lz": {"max_slope_deg": 6}}, sections, sets)["lz"] == {
        "max_slope_deg": 6}

    with pytest.raises(ValueError, match=r"no section of plan_x: \['parking'\]"):
        check_overrides("plan_x", {"parking": {}}, sections, sets)
    with pytest.raises(ValueError) as unknown:
        check_overrides("plan_x", {"lz": {"max_slop_deg": 6}}, sections, sets)
    assert "max_slop_deg" in str(unknown.value) and "lz_candidates" in str(unknown.value)
    assert "max_slope_deg" in str(unknown.value)          # names what it does take
    assert "'limit'" not in str(unknown.value).split("accepts")[1]
    with pytest.raises(ValueError, match="cannot set 'handle': the scenario stages"):
        check_overrides("plan_x", {"lz": {"handle": "aoi-0"}}, sections, sets)
    with pytest.raises(ValueError, match="cannot set 'latitude': the scenario's own"):
        check_overrides("plan_x", {"medical": {"latitude": 1}}, sections, sets)
    with pytest.raises(ValueError, match="cannot set 'limit': plan_x sets it from its own "
                                         "argument lz_limit"):
        check_overrides("plan_x", {"lz": {"limit": 1}}, sections, sets)
    with pytest.raises(ValueError, match="mapping"):
        check_overrides("plan_x", {"lz": 6}, sections, sets)
    with pytest.raises(ValueError, match="mapping"):
        check_overrides("plan_x", [("lz", {})], sections, sets)
    # None is "no overrides"; a falsy value of any other kind is a mistake.
    assert check_overrides("plan_x", {"lz": None}, sections, sets)["lz"] == {}
    for falsy in ([], 0, "", False):
        with pytest.raises(ValueError, match=r"overrides\['lz'\] must be a mapping"):
            check_overrides("plan_x", {"lz": falsy}, sections, sets)


@pytest.mark.parametrize("name, keyword", [
    ("lz", "limit"), ("egress", "mode"), ("medical", "per_category"), ("parking", "limit"),
    ("hasty", "limit"), ("travel_rings", "minutes")])
def test_a_keyword_a_helper_always_sets_is_refused_without_the_scenario_naming_it(
        name, keyword):
    """The helper spreads the overrides after its own arguments, so an
    override of one of them would silently win - an egress `mode` would run
    walk-out twice, one labelled carry-out. A later scenario that forgets
    its `scenario_sets` entry must still be safe."""
    with pytest.raises(ValueError, match=f"cannot set '{keyword}': the scenario sets it "
                                         "from one of its own arguments"):
        check_overrides("plan_next", {name: {keyword: 3}}, (name,), {})


@pytest.mark.parametrize("keyword", ["handle", "set_id", "candidate_ids"])
def test_the_export_s_handle_and_set_are_the_scenario_s(keyword):
    """The export draws the set the scenario's own section made, on its
    ground: another set is a direct caltopo_export call."""
    with pytest.raises(ValueError, match=f"cannot set '{keyword}': the scenario"):
        check_overrides("plan_next", {"export": {keyword: "x"}}, ("export",), {})
    # What the drawing looks like is the caller's to change.
    assert check_overrides("plan_next", {"export": {"include_area": False}}, ("export",),
                           {}) == {"export": {"include_area": False}}


@pytest.mark.parametrize("name, given, converted", [
    pytest.param("lz", {"max_slope_deg": "6"}, 6.0, id="lz-slope-text"),
    pytest.param("medical", {"search_radius_km": "200"}, 200.0, id="medical-radius-text"),
    pytest.param("lz", {"max_slope_deg": None}, None, id="lz-slope-null"),
    pytest.param("export", {"resource_type": 5}, None, id="export-type-number"),
    pytest.param("export", {"include_area": "false"}, False, id="export-area-false"),
    pytest.param("export", {"include_markers": "no"}, False, id="export-markers-no"),
    pytest.param("medical", {"check_osm_helipads": "false"}, False, id="medical-helipads-false"),
])
def test_an_override_value_is_checked_against_the_feature_s_annotation(name, given, converted):
    """Over MCP the feature tool's schema converts "6" to 6.0 and "false" to
    False, and refuses a null slope or a numeric resource type; an override
    reaches the plain function with no schema at all. So `check_overrides`
    must refuse the value - naming the keyword - or hand back what the tool
    would have read, never the raw text: "false" is truthy."""
    (keyword, value), = given.items()
    try:
        checked = check_overrides("plan_x", {name: given}, (name,), {})
    except ValueError as error:
        assert keyword in str(error)
        return
    assert converted is not None, f"{keyword}={value!r} was let through unchecked"
    assert checked[name][keyword] == converted
    assert type(checked[name][keyword]) is type(converted)


#: Override values a feature tool would refuse or convert over MCP, by
#: scenario: (scenario, overrides, (the stub's call list, keyword, the value
#: the feature must receive when converted rather than refused) or None
#: when only a refusal will do).
BAD_OVERRIDE_VALUES = [
    pytest.param("plan_medevac", {"lz": {"max_slope_deg": "6"}}, ("lz", "max_slope_deg", 6.0),
                 id="medevac-lz-slope-text"),
    pytest.param("plan_medevac", {"lz": {"max_slope_deg": None}}, None, id="medevac-lz-slope-null"),
    pytest.param("plan_medevac", {"medical": {"search_radius_km": "200"}},
                 ("medical", "search_radius_km", 200.0), id="medevac-medical-radius-text"),
    pytest.param("plan_medevac", {"medical": {"search_radius_km": 1000}}, None,
                 id="medevac-medical-radius-range"),
    pytest.param("plan_medevac", {"medical": {"check_osm_helipads": "false"}},
                 ("medical", "check_osm_helipads", False), id="medevac-medical-helipads-false"),
    pytest.param("plan_initial_response", {"hasty": {"include": ["trails"]}}, None,
                 id="initial-hasty-include-enum"),
    pytest.param("plan_initial_response", {"hasty": {"include": "trail"}}, None,
                 id="initial-hasty-include-string"),
    pytest.param("plan_initial_response", {"hasty": {"max_distance_m": "far"}}, None,
                 id="initial-hasty-distance-text"),
    pytest.param("plan_initial_response", {"hasty": {"min_length_m": 0}}, None,
                 id="initial-hasty-length-zero"),
    pytest.param("plan_initial_response", {"lz": {"max_slope_deg": "6"}},
                 ("lz", "max_slope_deg", 6.0), id="initial-lz-slope-text"),
    pytest.param("plan_initial_response", {"export": {"resource_type": 5}}, None,
                 id="initial-export-type-number"),
    pytest.param("plan_initial_response", {"export": {"include_area": "false"}},
                 ("export", "include_area", False), id="initial-export-area-false"),
    pytest.param("plan_initial_response", {"export": {"include_markers": "no"}},
                 ("export", "include_markers", False), id="initial-export-markers-no"),
    pytest.param("plan_urban_search", {"medical": {"search_radius_km": 1000}}, None,
                 id="urban-medical-radius-range"),
    pytest.param("plan_urban_search", {"medical": {"check_osm_helipads": "false"}},
                 ("medical", "check_osm_helipads", False), id="urban-medical-helipads-false"),
    pytest.param("plan_urban_search", {"export": {"include_area": "false"}},
                 ("export", "include_area", False), id="urban-export-area-false"),
    # The schema takes "nan" as a float; NaN passes every range check, so
    # the section would answer "ok" with nothing found. Refused instead.
    pytest.param("plan_medevac", {"lz": {"max_slope_deg": "nan"}}, None,
                 id="medevac-lz-slope-nan"),
    pytest.param("plan_initial_response", {"hasty": {"max_distance_m": "nan"}}, None,
                 id="initial-hasty-distance-nan"),
    # A pace is read by rings_ground, in its own words, but text the schema
    # reads as a number reaches it as that number.
    pytest.param("plan_initial_response", {"travel_rings": {"speed_factor": "0.5"}},
                 ("travel_rings", "speed_factor", 0.5), id="initial-rings-pace-text"),
]


@pytest.mark.parametrize("scenario, overrides, converted", BAD_OVERRIDE_VALUES)
def test_a_bad_override_value_is_refused_before_any_work_or_read_as_the_tool_reads_it(
        ctx, monkeypatch, scenario, overrides, converted):
    """A CALLER'S MISTAKE IS NOT A SECTION FAILURE: a value the feature
    would refuse is refused before anything is staged, not after minutes of
    staging as a failed section in Python's words. A value the feature tool
    converts over MCP reaches the feature converted, so the scenario and the
    tool never answer one request two ways."""
    plan, point, stubs = scenario_under_stubs(scenario, monkeypatch)
    ((_section, given),) = overrides.items()
    ((keyword, _value),) = given.items()
    try:
        result = plan(ctx, *point, overrides=overrides)
    except ValueError as error:
        assert keyword in str(error) and stubs.nothing_called()
        return
    assert converted is not None, (
        f"{overrides} ran instead of being refused: status {result['status']}")
    calls, _keyword, value = converted
    received = stubs.calls[calls][0][keyword]
    assert received == value and type(received) is type(value)
    if keyword == "include_area":
        export = result["sections"]["export"]
        assert export["area_included"] is False
        assert export["detail_call"]["arguments"]["include_area"] is False


@pytest.mark.anyio
@pytest.mark.parametrize("overrides, area_included", [
    pytest.param({"export": {"include_area": "false"}}, False, id="area-false"),
    pytest.param({"hasty": {"include": ["trial"]}}, None, id="hasty-include-enum"),
])
async def test_over_mcp_a_bad_override_value_is_refused_or_read_as_the_feature_tool_reads_it(
        monkeypatch, tmp_path, overrides, area_included):
    """The scenario's `overrides` schema is an object of objects, so nothing
    is converted or refused on the way in: the scenario must do what the
    feature tool's own schema does for the same JSON."""
    from fastmcp import Client
    from fastmcp.exceptions import ToolError

    from sar_geo_mcp.server import build_server

    _plan, place, stubs = scenario_under_stubs("plan_initial_response", monkeypatch)
    server = build_server(CaltopoConfig(cache_dir=tmp_path / "cache"))
    async with Client(server) as client:
        try:
            answer = (await client.call_tool("plan_initial_response", {
                "latitude": place[0], "longitude": place[1],
                "overrides": overrides})).data
        except ToolError as error:
            ((_section, given),) = overrides.items()
            assert all(keyword in str(error) for keyword in given) and stubs.nothing_called()
            return
    assert area_included is not None, f"{overrides} ran: status {answer['status']}"
    assert answer["sections"]["export"]["area_included"] is area_included


@pytest.mark.parametrize("name, given, message", [
    ("lz", {"max_slope_deg": 8}, r"max_slope_deg \(8\) is above marginal_slope_deg \(7.0\)"),
    ("lz", {"marginal_slope_deg": 4.0}, r"max_slope_deg \(5.0\) is above marginal_slope_deg"),
    ("lz", {"powerline_notice_m": 0}, r"powerline_notice_m \(0\) must be above zero"),
    ("egress", {"max_grade": -0.1}, r"max_grade \(-0.1\) is below zero"),
    ("egress", {"off_way_factor": 0}, r"off_way_factor \(0.0\) must be above zero"),
    ("medical", {"helicopter_speed_kmh": 1000}, "helicopter_speed_kmh must be between 50"),
    ("hasty", {"min_length_m": -5}, "min_length_m must be greater than 0"),
])
def test_an_override_a_feature_s_own_checks_refuse_is_refused_before_any_work(name, given,
                                                                            message):
    """The ranges no schema holds - a slope limit above the marginal one, a
    negative grade - are the feature's own checks, which it makes only once
    the ground is staged: run here over the tool's own defaults, in its own
    words, so the caller hears them before minutes of staging."""
    with pytest.raises(ValueError, match=rf"overrides\[{name!r}\]: {message}"):
        check_overrides("plan_x", {name: given}, (name,), {})
    # Against the defaults, not in isolation: 6 deg is inside the marginal 7.
    assert check_overrides("plan_x", {"lz": {"max_slope_deg": 6}}, ("lz",), {}) == {
        "lz": {"max_slope_deg": 6}}


def test_a_value_the_tool_converts_reaches_it_as_the_tool_reads_it():
    """A whole number where a float goes is the same number, kept as written
    so a refusal quotes it and the detail_call reads as the caller wrote it;
    anything the schema changes is passed on changed."""
    checked = check_overrides("plan_x", {
        "hasty": {"include": ("trail", "road"), "min_length_m": 593},
        "travel_rings": {"vegetation_factors": {"Herb": 2, "Shrub": "1.5"}},
        "export": {"include_markers": "0"}}, ("hasty", "travel_rings", "export"), {})
    assert checked["hasty"] == {"include": ["trail", "road"], "min_length_m": 593}
    assert type(checked["hasty"]["min_length_m"]) is int
    assert checked["travel_rings"] == {"vegetation_factors": {"Herb": 2, "Shrub": 1.5}}
    assert checked["export"] == {"include_markers": False}


@pytest.mark.parametrize("name, given, place", [
    ("lz", {"max_slope_deg": "nan"}, "max_slope_deg"),
    ("lz", {"max_slope_deg": "NaN", "marginal_slope_deg": "nan"}, "max_slope_deg"),
    ("lz", {"powerline_clearance_m": float("inf")}, "powerline_clearance_m"),
    ("hasty", {"max_distance_m": "nan"}, "max_distance_m"),
    ("parking", {"max_distance_m": "inf"}, "max_distance_m"),
    ("travel_rings", {"vegetation_factors": {"Herb": "nan"}}, r"vegetation_factors\['Herb'\]"),
])
def test_a_number_that_is_not_finite_is_refused_though_the_schema_takes_it(name, given, place):
    """The tool's schema reads "nan" as a float, and NaN compares false with
    every limit: a slope limit of NaN passes each range check and grades no
    ground, so the section would answer "ok" with no landing zone - silence
    that reads as "none here". Refused before any work instead."""
    with pytest.raises(ValueError, match=rf"overrides\[{name!r}\]: {place} must be a finite "
                                         "number; got .*no range check can be trusted"):
        check_overrides("plan_x", {name: given}, (name,), {})


def test_a_pace_given_as_text_is_read_as_the_tool_reads_it_before_its_own_check():
    """`speed_factor` is checked by rings_ground in its own words, but "0.5"
    is a pace as the tool's schema reads it: refused as not a number, the
    refusal would ask for the very value it was given. Text the schema does
    not read, and anything that is not text, stay as given for that check,
    which is stricter than the schema at times - a flag is no pace."""
    rings = ("travel_rings",)
    checked = check_overrides("plan_x", {"travel_rings": {"speed_factor": "0.5"}}, rings, {})
    assert checked == {"travel_rings": {"speed_factor": 0.5}}
    assert rings_ground([60.0], checked["travel_rings"]) == Ground(2000.0, 10.0)
    for kept in ("fast", "nan", True, 0):
        overrides = check_overrides("plan_x", {"travel_rings": {"speed_factor": kept}}, rings,
                                    {})["travel_rings"]
        assert overrides["speed_factor"] == kept and type(overrides["speed_factor"]) is type(kept)
        with pytest.raises(ValueError, match=r"\['speed_factor'\] must be a number above zero"):
            rings_ground([15.0], overrides)


def test_drafts_in_a_type_the_corpus_has_no_median_for_are_said(ctx, monkeypatch):
    """The drafts carry the resource type as given, so a typo - DOGS for DOG
    - would reach CalTopo unremarked: the export section says so, naming
    the types the corpus knows. A known type, or none given, says nothing."""
    monkeypatch.setattr(candidates_component, "caltopo_export", lambda ctx, **arguments: {
        "path": "/x.geojson", "set_id": arguments["set_id"], "assignments": 1,
        "features": {"Assignment": 1}, "labels": [], "area_included": True, "warnings": []})
    source = section(OK, handle="aoi-0123456789abcdef", set_id="urban-c0ffee")

    def warnings(**settings):
        return _sections.set_export_section(ctx, "urban", source, overrides={},
                                            settings=settings).section["warnings"]

    (said,) = warnings(resource_type="DOGS")
    assert said.startswith("the draft assignments carry resourceType 'DOGS' as given")
    assert "GROUND" in said and "DOG_TRAIL" in said
    assert warnings(resource_type="dog") == warnings() == []


@pytest.mark.parametrize("widths", [{"waterway": 60},
                                    hasty_feature.CorridorWidths(waterway=60)],
                         ids=["mapping", "model"])
def test_corridor_widths_reach_the_tool_as_its_model_and_the_detail_call_as_json(
        ctx, monkeypatch, widths):
    """A direct caller may hand `check_hasty_overrides` the tool's own
    CorridorWidths: like JSON's mapping it reaches hasty_candidates as the
    model, and its detail_call as the mapping, which JSON can carry."""
    checked = check_hasty_overrides({"corridor_widths_m": widths, "min_length_m": 593})
    assert checked == {"corridor_widths_m": {"waterway": 60.0}, "min_length_m": 593}
    received = []

    def fake_hasty(ctx, **arguments):
        received.append(arguments)
        return {"set_id": "hasty-a1b2c3", "candidates": [], "summary": {}, "warnings": []}

    monkeypatch.setattr(hasty_feature, "hasty_candidates", fake_hasty)
    outcome = _sections.hasty_section(ctx, "aoi-0123456789abcdef", limit=5, overrides=checked)
    (call,) = received
    assert isinstance(call["corridor_widths_m"], hasty_feature.CorridorWidths)
    assert call["corridor_widths_m"].as_overrides() == {"waterway": 60.0}
    assert outcome.section["detail_call"] == {"tool": "hasty_candidates", "arguments": {
        "handle": "aoi-0123456789abcdef", "limit": 5, "min_length_m": 593,
        "corridor_widths_m": {"waterway": 60.0}}}
    json.dumps(outcome.section)


def test_the_rings_ground_fits_the_longest_ring_and_no_more_than_can_be_staged():
    assert rings_ground([15.0], {}) == Ground(500.0, 10.0)
    assert rings_ground([15.0, 45.0], {}) == Ground(3500.0, 10.0)
    assert rings_ground([60.0], {"speed_factor": 0.5}) == Ground(2000.0, 10.0)
    assert rings_ground([60.0], {}) == Ground(4499.0, 10.0)
    # The cap is a radius prepare_area stages, at 10 m.
    check_radius("rings", 4499.0, latitude=POINT[0], longitude=POINT[1], resolution_m=10.0)
    for speed in (0, -1, float("nan"), "fast", True):
        with pytest.raises(ValueError, match="speed_factor"):
            rings_ground([15.0], {"speed_factor": speed})
    # Sized for the quickest ground the overrides give: open ground at twice
    # Tobler's off-path pace covers 6,000 m in 30 minutes, past the cap.
    assert rings_ground([15.0], {"off_way_factor": 2.0}) == Ground(2000.0, 10.0)
    assert rings_ground([30.0], {"off_way_factor": 2.0}) == Ground(4499.0, 10.0)
    # The rest of the walking model's overrides change no size.
    assert rings_ground([15.0], {"cliff_slope_deg": 40, "creek_crossing_minutes": 5}) == \
        Ground(500.0, 10.0)
    # An absurd time or pace stages the cap: what the feature itself takes.
    assert rings_ground([1e307], {}) == Ground(4499.0, 10.0)
    assert rings_ground([60.0], {"speed_factor": 1e308}) == Ground(4499.0, 10.0)
    for bad, match in (({"off_way_factor": -1}, "off_way_factor"),
                       ({"vegetation_factors": {"Moss": 2}}, "not a LANDFIRE lifeform"),
                       ({"vegetation_factors": [("Herb", 2)]}, "cannot be walked"),
                       ({"cliff_slope_deg": 90}, "cliff_slope_deg")):
        with pytest.raises(ValueError, match=match):
            rings_ground([15.0], bad)


def test_ring_shades_spread_over_the_ramp_shortest_lightest():
    shade = _sections._ring_shade
    assert [shade(i, 1) for i in range(1)] == [0]
    assert [shade(i, 3) for i in range(3)] == [0, 4, 7]
    assert [shade(i, 8) for i in range(8)] == list(range(8))
    # None of them the red of an assignment or a hospital, nor an LZ grade's.
    from sar_geo_mcp.planning.export import ASSIGNMENT_STROKE
    from sar_geo_mcp.planning.medical import HOSPITAL_COLOR
    taken = {ASSIGNMENT_STROKE, HOSPITAL_COLOR, *_sections.LZ_COLOR.values(),
             *_sections.EGRESS_STROKE.values(), _sections.PARKING_COLOR}
    assert not taken & set(_sections.RING_STROKES)
    for colour in _sections.RING_STROKES:        # cyan: no red in it at all
        assert colour.startswith("#00")


def test_an_export_label_prefix_is_refused_in_the_exporter_s_words():
    _sections.check_export_overrides({"label_prefix": "DIV-A", "include_area": False})
    _sections.check_export_overrides({})
    for bad in ("!!", "", "TOOLONGX", None, 5):
        with pytest.raises(ValueError, match=r"overrides\['export'\]: label_prefix .* must "
                                             "be 1 to 6 letters or digits"):
            _sections.check_export_overrides({"label_prefix": bad})


@pytest.mark.parametrize("status, warning", [
    (SKIPPED, "hasty was left out of include, so there is no candidate set to draw as "
              "draft assignments"),
    (FAILED, "the hasty section did not answer, so there is no candidate set to draw as "
             "draft assignments; its error says why"),
])
def test_a_set_export_with_no_set_is_a_warning_not_a_failure(ctx, status, warning):
    outcome = _sections.set_export_section(ctx, "hasty", section(status), overrides={})
    entry = outcome.section
    assert (entry["status"], entry["error"], entry["warnings"]) == (OK, None, [warning])
    assert entry["path"] is None and entry["set_id"] is None


#: The fields an export section carries on every path, drafted or not.
EXPORT_FIELDS = {"status", "error", "warnings", "path", "set_id", "assignments", "features",
                 "labels", "area_included", "detail_call"}


def _drafted(ctx, **arguments):
    """caltopo_export's payload for a set drafted as three assignments."""
    return {"path": "/cache/staged/aoi-0123456789abcdef/exports/hasty-a1b2c3.geojson",
            "set_id": arguments["set_id"], "assignments": 3,
            "features": {"Assignment": 3, "Shape": 1, "Marker": 1},
            "labels": [{"id": "hasty-a1b2c3.1", "label": "H01 Spring Ridge Trail"}],
            "area_included": True, "warnings": []}


REVIEW_B15 = pytest.mark.xfail(strict=True, reason="PR #4 review B15: an 'ok' export section "
                                                   "has no assignments/labels/features/"
                                                   "area_included when nothing was drafted")


@pytest.mark.parametrize("path", [
    "drafted",
    pytest.param("skipped", marks=REVIEW_B15),
    pytest.param("failed", marks=REVIEW_B15),
    pytest.param("raised", marks=REVIEW_B15),
])
def test_an_export_section_has_the_same_fields_on_every_path(ctx, monkeypatch, path):
    """A client that checks `status == "ok"` and reads `assignments` or
    `labels` must not get a KeyError because nothing was drafted: what is
    not there is null, 0 or empty, and the call is kept when one was made."""
    source = {"drafted": section(OK, handle="aoi-0123456789abcdef", set_id="hasty-a1b2c3"),
              "skipped": section(SKIPPED), "failed": section(FAILED, error="ValueError: x"),
              "raised": section(OK, handle="aoi-0123456789abcdef", set_id="hasty-a1b2c3")}[path]

    def full_disk(ctx, **arguments):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(candidates_component, "caltopo_export",
                        full_disk if path == "raised" else _drafted)
    entry = _sections.set_export_section(ctx, "hasty", source, overrides={}).section
    assert entry["status"] == OK and entry["error"] is None
    assert set(entry) == EXPORT_FIELDS
    if path == "drafted":
        assert entry["assignments"] == 3 and entry["area_included"] is True
        return
    assert entry["path"] is None and entry["assignments"] == 0
    assert entry["labels"] == [] and entry["features"] == {} and entry["area_included"] is None
    if path == "raised":
        assert entry["detail_call"]["tool"] == "caltopo_export"
    else:
        assert entry["detail_call"] is None


@REVIEW_B15
@pytest.mark.parametrize("scenario, partial", [
    ("plan_initial_response", {"include": ["travel_rings", "export"]}),
    ("plan_initial_response", {"fail": {"hasty"}}),
    ("plan_initial_response", {"fail": {"export"}}),
    ("plan_urban_search", {"include": ["parking", "export"]}),
    ("plan_urban_search", {"fail": {"grouping"}}),
], ids=["initial-hasty-left-out", "initial-hasty-failed", "initial-export-raised",
        "urban-left-out", "urban-failed"])
def test_a_scenario_s_export_section_has_the_fields_of_a_full_answer(ctx, monkeypatch,
                                                                     scenario, partial):
    """The same fields on a scenario's export section whether its set was
    left out, failed, or could not be drafted, as when it was drafted."""
    plan, point, _stubs = scenario_under_stubs(scenario, monkeypatch)
    full = plan(ctx, *point)["sections"]["export"]
    scenario_under_stubs(scenario, monkeypatch, fail=partial.get("fail", ()))
    export = plan(ctx, *point, include=partial.get("include"))["sections"]["export"]
    assert export["status"] == OK
    assert set(export) == set(full)
    assert export["assignments"] == 0 and export["labels"] == []


def test_a_set_export_that_raises_is_a_warning_that_keeps_its_call(ctx, monkeypatch):
    from sar_geo_mcp.components import candidates as candidates_component

    def full_disk(ctx, **arguments):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(candidates_component, "caltopo_export", full_disk)
    source = section(OK, handle="aoi-0123456789abcdef", set_id="hasty-a1b2c3")
    outcome = _sections.set_export_section(ctx, "hasty", source,
                                           overrides={"label_prefix": "H"})
    entry = outcome.section
    assert entry["status"] == OK and entry["error"] is None and entry["path"] is None
    assert entry["warnings"] == [
        "the draft assignments could not be written (OSError: [Errno 28] No space left on "
        "device); every section's answer stands, and detail_call makes the export again"]
    assert entry["detail_call"] == {"tool": "caltopo_export", "arguments": {
        "handle": "aoi-0123456789abcdef", "set_id": "hasty-a1b2c3", "include_area": True,
        "label_prefix": "H"}}


def test_a_set_export_s_settings_win_over_its_overrides(ctx, monkeypatch):
    """What the scenario sets from its own arguments is held by the helper,
    as every keyword a helper sets is: a scenario whose scenario_sets forgot
    one still drafts in its own resource type, never the override's."""
    from sar_geo_mcp.components import candidates as candidates_component

    made = []

    def record(ctx, **arguments):
        made.append(arguments)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(candidates_component, "caltopo_export", record)
    source = section(OK, handle="aoi-0123456789abcdef", set_id="urban-c0ffee")
    outcome = _sections.set_export_section(
        ctx, "urban", source,
        overrides={"resource_type": "GROUND", "include_markers": True, "label_prefix": "U",
                   "include_area": False},
        settings={"resource_type": "DOG", "include_markers": False})
    assert made == [{"handle": "aoi-0123456789abcdef", "set_id": "urban-c0ffee",
                     "include_area": False, "label_prefix": "U", "resource_type": "DOG",
                     "include_markers": False}]
    assert outcome.section["detail_call"]["arguments"] == made[0]


def test_counts_are_refused_unless_whole_and_in_range():
    check_limit("hasty_limit", 0, "candidate")
    for bad in (-1, True, 2.5, "3"):
        with pytest.raises(ValueError, match=r"hasty_limit must be 0 \(every candidate\)"):
            check_limit("hasty_limit", bad, "candidate")
    check_per_category(10)
    for bad in (0, 11, True, 2.0):
        with pytest.raises(ValueError, match="medical_per_category must be between 1 and 10"):
            check_per_category(bad)


def test_overrides_are_checked_against_the_real_tool_not_a_stub(monkeypatch):
    """A test standing a stub in for the feature must not change what an
    override may name: the keywords are read once, from the tool."""
    monkeypatch.setattr(parking_feature, "parking_candidates", lambda ctx, **kw: kw)
    checked = check_overrides("plan_x", {"parking": {"max_distance_m": 800}},
                              ("parking",), {})
    assert checked == {"parking": {"max_distance_m": 800}}
    with pytest.raises(ValueError, match="does not take"):
        check_overrides("plan_x", {"parking": {"anything": 1}}, ("parking",), {})


def test_a_section_calling_two_features_takes_its_overrides_by_tool():
    """The urban section calls populated_areas and urban_assignments, and
    both take max_distance_m: a flat mapping could not say which call it is
    for, so each tool gets its own, checked against that tool."""
    assert check_overrides("plan_x", None, ("urban",), {}) == {"urban": {}}
    # A tool given nothing (or None) is left out, so "no overrides" reads
    # the same as for any section.
    assert check_overrides("plan_x", {"urban": {"populated_areas": {},
                                                "urban_assignments": None}},
                           ("urban",), {}) == {"urban": {}}
    given = {"populated_areas": {"min_parcels": 2, "max_distance_m": 900},
             "urban_assignments": {"max_distance_m": 900}}
    assert check_overrides("plan_x", {"urban": given}, ("urban",), {}) == {"urban": given}
    with pytest.raises(ValueError, match=r"overrides\['urban'\] names \['max_distance_m'\], "
                                         r"which is no tool it calls: the urban section calls "
                                         r"\['populated_areas', 'urban_assignments'\]"):
        check_overrides("plan_x", {"urban": {"max_distance_m": 900}}, ("urban",), {})
    with pytest.raises(ValueError, match=r"overrides\['urban'\]\['urban_assignments'\] names "
                                         r"\['bridge_m'\], which urban_assignments does not "
                                         r"take here; it accepts \['max_distance_m', "
                                         r"'marker_color'\]"):
        check_overrides("plan_x", {"urban": {"urban_assignments": {"bridge_m": 10}}},
                        ("urban",), {})
    with pytest.raises(ValueError, match=r"overrides\['urban'\] must be a mapping by tool"):
        check_overrides("plan_x", {"urban": 5}, ("urban",), {})
    for falsy in ([], 0, ""):
        with pytest.raises(ValueError, match=r"overrides\['urban'\]\['populated_areas'\] "
                                             "must be a mapping"):
            check_overrides("plan_x", {"urban": {"populated_areas": falsy}}, ("urban",), {})
    # What a scenario sets from its own arguments is refused in every tool.
    with pytest.raises(ValueError, match=r"overrides\['urban'\]\['populated_areas'\] cannot "
                                         "set 'building_markers': plan_x sets it from its own "
                                         "argument markers"):
        check_overrides("plan_x", {"urban": {"populated_areas": {"building_markers": True}}},
                        ("urban",), {"urban": {"building_markers": "markers"}})


@pytest.mark.parametrize("tool, keyword", [
    ("populated_areas", "handle"), ("populated_areas", "building_markers"),
    ("urban_assignments", "handle"), ("urban_assignments", "target_ha"),
    ("urban_assignments", "resource_type"), ("urban_assignments", "building_markers")])
def test_a_keyword_the_urban_helper_always_sets_is_refused_without_the_scenario_naming_it(
        tool, keyword):
    with pytest.raises(ValueError, match=f"cannot set '{keyword}': the scenario"):
        check_overrides("plan_next", {"urban": {tool: {keyword: 3}}}, ("urban",), {})


@pytest.mark.parametrize("latitude", [0.0, 37.5072, 64.0])
def test_check_urban_radius_refuses_exactly_what_populated_areas_refuses(tmp_path, latitude):
    """The circle prepare_area stages, read by the real populated_areas with
    a token and a fetcher that must never be reached: whatever it refuses
    for width before fetching, check_urban_radius refuses, and the other
    way round."""
    from sar_geo_mcp.planning.area import build_area
    from sar_geo_mcp.planning.populated import populated_areas

    def fetch(*args, **kwargs):
        raise AssertionError("fetched")

    refused_by = {}
    for radius in (3800.0, 3870.0, 3890.0, 3896.0, 3900.0, 4000.0):
        directory = tmp_path / f"{latitude}-{radius}"
        directory.mkdir()
        area = build_area(center_latitude=latitude, center_longitude=-122.3, radius_m=radius)
        (directory / "subject.json").write_text(area.model_dump_json(), encoding="utf-8")
        try:
            populated_areas(directory, wms_base_url="https://x/wms", cache_dir=tmp_path,
                            fetch=fetch)
        except ValueError as error:
            refused_by[radius] = "Stage a smaller area" in str(error)
        except AssertionError:
            refused_by[radius] = False
        if refused_by[radius]:
            with pytest.raises(ValueError, match="too wide to read lot lines over.*populated_"
                                                 "areas would refuse it - the staged area is"):
                _sections.check_urban_radius("radius", radius, latitude=latitude,
                                             longitude=-122.3)
        else:
            _sections.check_urban_radius("radius", radius, latitude=latitude, longitude=-122.3)
    assert refused_by[3800.0] is False and refused_by[4000.0] is True


def test_urban_override_values_are_refused_in_the_domain_s_words():
    check = _sections.check_urban_overrides
    check({}, max_distance_m=800.0, building_markers=False)
    check({"populated_areas": {"pixel_m": 2.0, "min_parcels": 1, "limit": 0},
           "urban_assignments": {"max_distance_m": None}},
          max_distance_m=800.0, building_markers=False)
    with pytest.raises(ValueError, match=r"\['populated_areas'\]: pixel_m \(2.5\)"):
        check({"populated_areas": {"pixel_m": 2.5}}, max_distance_m=800.0,
              building_markers=False)
    with pytest.raises(ValueError, match=r"\['populated_areas'\]: pixel_m, max_parcel_area_m2, "
                                         r".* and limit are numbers"):
        check({"populated_areas": {"bridge_m": "wide"}}, max_distance_m=800.0,
              building_markers=False)
    with pytest.raises(ValueError, match=r"\['urban_assignments'\]: max_distance_m \(0\)"):
        check({"urban_assignments": {"max_distance_m": 0}}, max_distance_m=800.0,
              building_markers=False)
    # A colour is read, and refused, only when markers are drawn.
    check({"populated_areas": {"marker_color": "red"}}, max_distance_m=800.0,
          building_markers=False)
    with pytest.raises(ValueError, match=r"\['populated_areas'\]: marker_color \('red'\)"):
        check({"populated_areas": {"marker_color": "red"}}, max_distance_m=800.0,
              building_markers=True)
    check({"populated_areas": {"marker_color": "ff0000"}}, max_distance_m=800.0,
          building_markers=True)


def test_the_grouping_s_block_rules_are_the_domain_function_s_defaults():
    """urban_assignments draws its blocks through planning.populated with
    every block rule but max_distance_m at that function's defaults; the
    populated_areas tool's own defaults are the same rules, so a call that
    overrides nothing draws the blocks the assignments group."""
    assert _sections.DRAWN_BLOCK_RULES == _sections.GROUPED_BLOCK_RULES
    assert set(_sections.GROUPED_BLOCK_RULES) == set(_sections.BLOCK_RULES)


@pytest.mark.parametrize("value", [None, 0.5, 10, 25.0])
def test_a_target_size_is_none_or_above_zero(value):
    _sections.check_target_ha(value)


@pytest.mark.parametrize("value", [0, -1.0, float("nan"), float("inf"), True, "10"])
def test_a_target_size_that_is_not_one_is_refused(value):
    with pytest.raises(ValueError, match="target_ha must be a number of hectares above zero"):
        _sections.check_target_ha(value)


# --- Staging -----------------------------------------------------------------


def test_each_distinct_ground_is_staged_once_and_shared(ctx, monkeypatch):
    fake = FakeStaging()
    monkeypatch.setattr(staging_component, "prepare_area", fake)
    staging = stage_grounds(ctx, {"lz": Ground(1000, 2.0), "rings": Ground(3000, 10.0),
                                  "egress": Ground(3000, 10.0)},
                            latitude=POINT[0], longitude=POINT[1], map_id="M1",
                            public=False)
    assert len(fake.calls) == 2
    assert fake.calls[0] == {"center_latitude": POINT[0], "center_longitude": POINT[1],
                             "radius_m": 1000, "map_id": "M1", "public": False,
                             "resolution_m": 2.0}
    assert staging["rings"]["handle"] == staging["egress"]["handle"] != staging["lz"]["handle"]
    lz = staging["lz"]
    assert lz["status"] == OK and lz["error"] is None and lz["warnings"] == ["evt unavailable"]
    assert lz["staged"] is True and lz["osm_available"] is True
    assert lz["detail_call"] == {"tool": "prepare_area", "arguments": fake.calls[0]}
    # The area's outline stays with prepare_area.
    assert geometry_types_in(staging) == set()
    assert ground_handle(staging, "lz") == lz["handle"]


def test_a_ground_that_cannot_be_staged_fails_only_what_stands_on_it(ctx, monkeypatch):
    monkeypatch.setattr(staging_component, "prepare_area", FakeStaging(fail_resolutions={2.0}))
    staging = stage_grounds(ctx, {"lz": Ground(1000, 2.0), "egress": Ground(3000, 10.0)},
                            latitude=POINT[0], longitude=POINT[1])
    assert staging["lz"]["status"] == FAILED and staging["lz"]["handle"] is None
    assert staging["lz"]["error"] == "RuntimeError: 3DEP could not be reached"
    assert staging["egress"]["status"] == OK
    with pytest.raises(StagingFailed, match=r"the lz ground \(1,000 m radius at 2 m\).*"
                                            r"RuntimeError: 3DEP could not be reached"):
        ground_handle(staging, "lz")
    failed = guard(ctx, lambda: Outcome(section(OK, handle=ground_handle(staging, "lz"))))
    assert failed.section["status"] == FAILED
    assert failed.section["error"].startswith("StagingFailed: the lz ground")


def test_a_shared_ground_that_cannot_be_staged_fails_every_name_on_it_once(secret_ctx,
                                                                         monkeypatch):
    """Two names on one ground share one staging, failed as well as staged:
    it is tried once, and each name has its own record saying the same
    redacted reason - never one name read as staged."""
    fake = LeakyStaging({10.0})
    monkeypatch.setattr(staging_component, "prepare_area", fake)
    staging = stage_grounds(secret_ctx, {"rings": Ground(3000, 10.0),
                                         "egress": Ground(3000, 10.0)},
                            latitude=POINT[0], longitude=POINT[1])
    assert len(fake.calls) == 1
    assert staging["rings"] == staging["egress"] and staging["rings"] is not staging["egress"]
    for name in ("rings", "egress"):
        record = staging[name]
        assert record["status"] == FAILED and record["handle"] is None
        assert record["error"] == LEAK_REDACTED
        assert "<redacted>" in record["error"] and leaks(record) == []
        with pytest.raises(StagingFailed, match=rf"the {name} ground \(3,000 m radius at 10 m\) "
                                                r"could not be staged.*RuntimeError: GET"):
            ground_handle(staging, name)


@pytest.mark.xfail(strict=True, reason="PR #4 review B30: a failed staging record lacks "
                                       "dem_resolution_m, staged and osm_available")
def test_a_ground_that_cannot_be_staged_has_the_fields_of_one_staged(ctx, monkeypatch):
    """The description promises every record its fields: what staging never
    produced is null, as `handle` already is."""
    monkeypatch.setattr(staging_component, "prepare_area", FakeStaging(fail_resolutions={10.0}))
    staging = stage_grounds(ctx, {"lz": Ground(1000, 2.0), "egress": Ground(3000, 10.0)},
                            latitude=POINT[0], longitude=POINT[1])
    assert staging["lz"]["status"] == OK and staging["egress"]["status"] == FAILED
    assert set(staging["egress"]) == set(staging["lz"])
    for key in ("dem_resolution_m", "staged", "osm_available"):
        assert staging["egress"][key] is None


@pytest.mark.xfail(strict=True, reason="PR #4 review B30: a failed assignment staging record "
                                       "lacks staged, reason and fingerprint")
def test_an_assignment_that_cannot_be_staged_has_the_fields_of_one_staged(ctx, monkeypatch):
    def staged(ctx, **arguments):
        return {"handle": f"{arguments['map_id']}_{arguments['label']}", "staged": False,
                "reason": "already current", "fingerprint": "f" * 32, "warnings": []}

    monkeypatch.setattr(staging_component, "prepare_terrain", staged)
    ok = _sections.stage_assignment(ctx, map_id="MAP1234", label="AB")
    monkeypatch.setattr(staging_component, "prepare_terrain", raise_leak)
    failed = _sections.stage_assignment(ctx, map_id="MAP1234", label="ZZ")
    assert ok["status"] == OK and failed["status"] == FAILED
    assert set(failed) == set(ok)
    for key in ("staged", "reason", "fingerprint"):
        assert failed[key] is None


# --- Landing zones and egress --------------------------------------------------


def _zone(**changes):
    zone = {"id": "lz_1", "grade": "minimal",
            "touchdown": {"type": "Point", "coordinates": [POINT[1], POINT[0]]},
            "clear_diameter_m": 35.0, "slope_mean_deg": 2.0, "slope_max_deg": 4.0,
            "canopy_fraction": 0.0, "distance_m": 230.0, "bearing_deg": 90.0,
            "powerline": None, "road": None, "parking_lot": None, "warnings": []}
    return {**zone, **changes}


def test_a_landing_zone_marker_says_what_was_not_checked():
    """The map is what the field team sees: a marker silent about trees
    reads as clear of them, and "no power line" on ground where only
    transmission lines were asked for overclaims."""
    checks = {"structures": "not checked", "trees": "not checked",
              "water_sources_checked": [], "powerline_kinds_checked": ["line"],
              "road_class": "unknown"}
    text = lz_marker(_zone(), checks, point_name="the patient")["properties"]["description"]
    assert "230 m at 90 deg from the patient." in text
    assert "Not checked here: trees, structures, open water." in text
    assert "No mapped transmission line near; distribution lines were not checked" in text
    assert "No mapped power line near" not in text

    every = {"structures": "checked", "trees": "checked",
             "water_sources_checked": ["landfire_open_water"],
             "powerline_kinds_checked": ["line", "minor_line"], "road_class": "checked"}
    text = lz_marker(_zone(), every)["properties"]["description"]
    assert "Not checked" not in text
    assert "No mapped power line near" in text and "from the point." in text


def test_egress_overrides_with_a_per_mode_default_need_one_mode():
    """A max_grade meant to tighten the litter's 40 % would put a grade
    limit on a walk-out that has none."""
    with pytest.raises(ValueError, match=r"sets \['max_grade'\], whose defaults differ by "
                                         r"mode \(carry_out: .*max_grade 0.4; walk_out: "
                                         r".*max_grade None\)"):
        check_egress_overrides(("carry_out", "walk_out"), {"max_grade": 0.3})
    check_egress_overrides(("carry_out",), {"max_grade": 0.3, "off_way_factor": 0.1})
    check_egress_overrides(("carry_out", "walk_out"), {})


def test_a_warning_from_some_modes_is_attributed_even_when_another_failed():
    both = {"carry_out": ["gates"], "walk_out": ["gates", "no LANDFIRE"]}
    assert merged_warnings(both, ("carry_out", "walk_out")) == [
        "gates", "walk_out: no LANDFIRE"]
    # carry_out raised and said nothing: walk-out's warnings are its own.
    assert merged_warnings({"walk_out": ["gates", "no LANDFIRE"]},
                           ("carry_out", "walk_out")) == [
        "walk_out: gates", "walk_out: no LANDFIRE"]
    assert merged_warnings({"walk_out": ["gates"]}, ("walk_out",)) == ["gates"]


def test_a_compact_exit_keeps_the_vehicle_end_s_facts():
    exit_ = {"kind": "2wd", "reachable": True, "minutes": 12.0, "distance_m": 900.0,
             "gain_m": 10.0, "loss_m": 40.0,
             "exit_point": {"type": "Point", "coordinates": [0.0, 0.0]},
             "route": {"type": "LineString", "coordinates": [[0, 0], [0, 1]]},
             "road": None, "parking_lot": {"id": "way/9", "off": "Kings Mountain Rd"},
             "gates_crossed": [], "warnings": []}
    compact = compact_exit(exit_)
    assert compact["road_name"] is None and compact["gate_on_road"] is None
    assert compact["parking_lot"] == {"id": "way/9", "off": "Kings Mountain Rd"}
    assert "route" not in compact
    gated = compact_exit({**exit_, "road": {"name": "Old Haul Rd", "access": "private",
                                            "gate_on_road": True}})
    assert (gated["road_name"], gated["road_access"], gated["gate_on_road"]) == (
        "Old Haul Rd", "private", True)


#: A reachable exit as egress_routes gives one, with nothing at the vehicle
#: end to say.
EXIT = {"kind": "2wd", "reachable": True, "minutes": 31.0, "distance_m": 1840.0,
        "gain_m": 60.0, "loss_m": 120.0,
        "exit_point": {"type": "Point", "coordinates": [0.0, 1.0]},
        "route": {"type": "LineString", "coordinates": [[0.0, 0.0], [0.0, 1.0]]},
        "road": None, "parking_lot": None, "gates_crossed": [], "warnings": []}


def test_an_exit_line_says_its_minutes_road_and_gates_crossed():
    """What the CalTopo line says today, held while B1 adds the rest."""
    line = route_line("carry_out", {**EXIT, "road": {"name": "Skyline Blvd", "access": None,
                                                     "gate_on_road": False},
                                    "gates_crossed": [{"barrier": "gate"}] * 2})
    text = line["properties"]["description"].splitlines()
    assert text[0] == "Carry-out to a 2WD exit: 31 min, 1,840 m, +60 m / -120 m."
    assert "Exit on Skyline Blvd." in text and "2 gates on the way." in text
    assert line["properties"]["title"] == "Carry-out 2WD 31 min"
    assert line["geometry"] == EXIT["route"]


@pytest.mark.parametrize("changes, said", [
    ({"road": {"name": "Old Haul Rd", "access": None, "gate_on_road": True}}, ("gate",)),
    ({"road": {"name": "Old Haul Rd", "access": "private", "gate_on_road": False}},
     ("private",)),
    ({"road": {"name": "Old Road", "access": None, "gate_on_road": None}}, ("gate", "unknown")),
    ({"parking_lot": {"id": "way/9", "off": "Kings Mountain Rd"}}, ("lot", "kings mountain rd")),
    ({"parking_lot": "unknown"}, ("lot", "unknown")),
], ids=["gate-on-road", "private-road", "gates-unknown", "lot-and-its-road", "lots-unknown"])
def test_an_exit_line_says_what_the_vehicle_end_holds(changes, said):
    """The line is what a field team reads in CalTopo: silent about a gate
    or a private road it reads as open, and an exit in a lot with no road
    beside it must still say where the vehicle meets the team. Gates that
    are unknown are said, never left to read as none."""
    text = route_line("carry_out", {**EXIT, **changes})["properties"]["description"]
    assert any(all(word in line.lower() for word in said) for line in text.splitlines()), text
    assert "0 gates" not in text


def test_an_exit_line_on_ground_with_gates_unknown_says_so(ctx, monkeypatch, tmp_path):
    """The real egress_routes over staged ground whose OpenStreetMap has no
    gate list: the payload warns that gates are unknown, and the line drawn
    for CalTopo - all the field team sees - must say so too."""
    import numpy as np
    import test_egress

    from sar_geo_mcp.features import egress as egress_feature
    from sar_geo_mcp.planning.egress import egress_routes as walk
    from sar_geo_mcp.planning.egress_roads import GATES_UNKNOWN_WARNING

    ground = tmp_path / "ground"
    ground.mkdir()
    directory = test_egress.stage(ground, np.zeros((21, 60)),
                                  [test_egress.north_south("ROAD", 40, 21, name="New Road")],
                                  gates=None)
    monkeypatch.setattr(egress_feature, "egress_routes",
                        lambda ctx, handle, **arguments: walk(directory, **arguments))
    longitude, latitude = test_egress.TO4326.transform(*test_egress.xy(10, 10))
    outcome = _sections.egress_section(ctx, "aoi-0123456789abcdef", latitude=latitude,
                                       longitude=longitude, modes=("carry_out",), overrides={})
    assert outcome.section["status"] == OK
    assert GATES_UNKNOWN_WARNING in outcome.section["warnings"]
    assert outcome.features
    for line in outcome.features:
        text = line["properties"]["description"]
        assert "Exit on New Road." in text
        assert any("gate" in said.lower() and "unknown" in said.lower()
                   for said in text.splitlines()), text


def test_an_exit_line_says_gates_even_when_none_are_mapped():
    """Gates read and none on the road or the way: said, so the line never
    leaves a reader to guess whether gates were looked for at all - but
    said of the stretch of road looked at, the road in beyond it unchecked.
    Untagged access is said too: it does not make a road public."""
    text = route_line("carry_out", {**EXIT, "road": {"name": "Skyline Blvd", "access": None,
                                                     "gate_on_road": False}}
                      )["properties"]["description"].splitlines()
    assert text[1:5] == [
        "Exit on Skyline Blvd.",
        "Access on Skyline Blvd: not tagged in the OpenStreetMap read for this plan, which "
        "does not make it public.",
        "No gate mapped on the way out or on the stretch of Skyline Blvd the exit lies on.",
        "The road in beyond that stretch, and beyond the ground read for this plan, is not "
        "checked for gates."]


def test_no_gate_on_the_exit_s_stretch_is_not_said_of_the_whole_road(ctx, monkeypatch,
                                                                      tmp_path):
    """OpenStreetMap maps a named road as several ways, and `gate_on_road`
    is the one way the exit lies on: with a gate mapped on Skyline Blvd's
    other way, the line must not say Skyline Blvd has none."""
    import numpy as np
    import test_egress

    from sar_geo_mcp.features import egress as egress_feature
    from sar_geo_mcp.planning.egress import egress_routes as walk

    ground = tmp_path / "ground"
    ground.mkdir()
    near = test_egress.way("ROAD", (0, 40), (10, 40), highway="residential",
                           name="Skyline Blvd")
    far = test_egress.way("ROAD", (10, 40), (20, 40), highway="residential",
                          name="Skyline Blvd")
    far["id"] = "route_Skyline Blvd_2"
    directory = test_egress.stage(ground, np.zeros((21, 60)), [near, far],
                                  gates=[test_egress.gate(18, 40)])
    monkeypatch.setattr(egress_feature, "egress_routes",
                        lambda ctx, handle, **arguments: walk(directory, **arguments))
    longitude, latitude = test_egress.TO4326.transform(*test_egress.xy(3, 10))
    outcome = _sections.egress_section(ctx, "aoi-0123456789abcdef", latitude=latitude,
                                       longitude=longitude, modes=("carry_out",), overrides={})
    exit_ = outcome.section["modes"]["carry_out"]["exits"][0]
    assert (exit_["road_name"], exit_["gate_on_road"]) == ("Skyline Blvd", False)
    [line] = outcome.features
    text = line["properties"]["description"].splitlines()
    assert not any(row.startswith("No gate mapped on Skyline Blvd") for row in text), text
    assert ("No gate mapped on the way out or on the stretch of Skyline Blvd the exit lies "
            "on.") in text
    assert any(row.startswith("The road in beyond that stretch") for row in text), text


def test_an_unnamed_exit_road_is_named_by_its_kind():
    """Many tracks carry no name; the line still says what the vehicle
    drives to, rather than nothing."""
    track = {"name": None, "route_type": "TRACK", "highway": "track", "access": None,
             "gate_on_road": True}
    text = route_line("carry_out", {**EXIT, "kind": "4x4", "road": track}
                      )["properties"]["description"].splitlines()
    assert "Exit on an unnamed track (OpenStreetMap highway=track)." in text
    assert any(row.startswith("Gate mapped somewhere on the stretch of the unnamed track")
               for row in text), text


def test_a_lot_exit_drawn_without_the_answer_s_warnings_says_gates_are_unknown():
    """Without the answer's warnings a lot exit with no road beside it
    cannot tell whether gates were read: unknown, never "no gate"."""
    in_a_lot = {**EXIT, "parking_lot": {"id": "way/9", "off": "Kings Mountain Rd"}}
    text = route_line("walk_out", in_a_lot)["properties"]["description"].splitlines()
    assert not any(row.startswith("No gate mapped") for row in text), text
    assert ("Gates unknown on this ground, never none: the way out and the road into the "
            "lot may be gated.") in text


def test_an_exit_at_the_point_puts_its_vehicle_end_in_the_point_notes(ctx, monkeypatch):
    """An exit at the point's own cell comes back with a Point route: no
    line is drawn, so what the line would say goes on the scenario's point."""
    from sar_geo_mcp.features import egress as egress_feature

    here = {"type": "Point", "coordinates": [-122.3, 37.4]}
    at_the_point = {**EXIT, "minutes": 0.0, "distance_m": 0.0, "route": here,
                    "exit_point": here,
                    "road": {"name": "Old Haul Rd", "access": "private", "gate_on_road": True}}
    monkeypatch.setattr(egress_feature, "egress_routes", lambda ctx, **arguments: {
        "exits": [at_the_point, {"kind": "4x4", "reachable": False, "reason": "x"}],
        "warnings": []})
    outcome = _sections.egress_section(ctx, "aoi-0123456789abcdef", latitude=37.4,
                                       longitude=-122.3, modes=("carry_out",), overrides={})
    assert outcome.features == []
    assert outcome.point_notes == [
        "Carry-out 2WD exit is here: Exit on Old Haul Rd. Access on Old Haul Rd: private, as "
        "OpenStreetMap tags it. Gate mapped somewhere on the stretch of Old Haul Rd the exit "
        "lies on: a vehicle may need the combination or the ranger to reach the exit. The "
        "road in beyond that stretch, and beyond the ground read for this plan, is not "
        "checked for gates."]


def test_an_exit_in_a_lot_hears_what_the_ground_cannot_tell_from_the_answer():
    """A lot exit with no road beside it carries nothing that says gates
    are unknown: the answer's own warnings do, and the line says them. On
    ground that read gates, the road into the lot was still never looked
    at, and the line says that instead of nothing."""
    from sar_geo_mcp.planning.egress_roads import (GATES_UNKNOWN_WARNING,
                                                   ROAD_FACTS_UNKNOWN_WARNING)

    in_a_lot = {**EXIT, "parking_lot": {"id": "way/9", "off": "Kings Mountain Rd"}}

    def said(warnings):
        return route_line("walk_out", in_a_lot, ground_warnings=warnings
                          )["properties"]["description"].splitlines()

    assert "Exit in parking lot way/9, off Kings Mountain Rd." in said([])
    assert said([])[2:4] == ["No gate mapped on the way out.",
                             "Gates on the road into the lot are not checked."]
    assert any(line.startswith("Gates unknown on this ground")
               and "the road into the lot may be gated" in line
               for line in said([GATES_UNKNOWN_WARNING])), said([GATES_UNKNOWN_WARNING])
    old = said(["walk_out is slow", ROAD_FACTS_UNKNOWN_WARNING.format(version=4)])
    assert any(line.startswith("Road class, access and gates unknown on this ground")
               and "may be private, gated or a motorway" in line for line in old), old
    for text in (said([GATES_UNKNOWN_WARNING]), old):
        assert not any(line.startswith("No gate mapped") for line in text)


# --- Parking -----------------------------------------------------------------


def test_parking_section_is_compact_and_says_how_to_get_the_outlines(ctx, monkeypatch):
    received = []

    def fake_parking(ctx, **arguments):
        received.append(arguments)
        return {"candidates": [{
                    "id": "parking-w1", "kind": "parking", "name": "Trailhead lot",
                    "point": {"latitude": POINT[0], "longitude": POINT[1]},
                    "area_m2": 900.0, "surface": "gravel", "capacity": "12",
                    "flags": ["fee"], "distance_from_center_m": 140.0,
                    "bearing_from_center_deg": 90.0, "nearest_way": None,
                    "geometry": {"type": "Polygon",
                                 "coordinates": [[[0, 0], [0, 1], [1, 1], [0, 0]]]},
                    "warnings": []}],
                "summary": {"count": 1, "total_before_limit": 1, "truncated_to": None},
                "query_version": 5, "center": None, "warnings": ["w"],
                "staleness": {"current": True}}

    monkeypatch.setattr(parking_feature, "parking_candidates", fake_parking)
    outcome = parking_section(ctx, "aoi-0123456789abcdef", limit=10,
                              overrides={"max_distance_m": 800},
                              point_name="the place last seen")
    assert received == [{"handle": "aoi-0123456789abcdef", "limit": 10,
                         "max_distance_m": 800}]
    entry = outcome.section
    assert entry["status"] == OK and entry["warnings"] == ["w"]
    assert entry["detail_call"] == {"tool": "parking_candidates", "arguments": received[0]}
    assert entry["candidates"][0]["name"] == "Trailhead lot"
    assert geometry_types_in(entry) == set()
    marker = outcome.features[0]
    assert marker["geometry"] == {"type": "Point", "coordinates": [POINT[1], POINT[0]]}
    assert marker["properties"]["class"] == "Marker"
    assert "fee" in marker["properties"]["description"]
    assert "140 m from the place last seen." in marker["properties"]["description"]


# --- A count whose input is missing is unknown, never 0 ----------------------------


@pytest.mark.parametrize("version, warnings, known", [
    (5, [], True),
    (3, [], True),
    # No osm.json: the feature's own warning (and it reads the file as version 1).
    (1, [parking_domain.MISSING_OSM_WARNING], False),
    (None, [parking_domain.MISSING_OSM_WARNING], False),
    # Fetched before parking was asked for.
    (2, [parking_domain.STALE_QUERY_WARNING.format(version=2)], False),
], ids=["current", "parking-version", "no-osm", "no-osm-no-version", "stale"])
def test_parking_with_no_lots_to_read_is_unknown_not_none(ctx, monkeypatch, version, warnings,
                                                          known):
    """No lot found is "none here" only where lots could be read: on ground
    with no osm.json, or one older than parking, the section's total and
    the summary's counts are null, the warning saying why beside them."""
    def no_lots(ctx, **arguments):
        payload = {"candidates": [],
                   "summary": {"count": 0, "total_before_limit": 0, "truncated_to": None},
                   "center": None, "warnings": list(warnings), "staleness": {"current": True}}
        return payload if version is None else {**payload, "query_version": version}

    monkeypatch.setattr(parking_feature, "parking_candidates", no_lots)
    entry = parking_section(ctx, "aoi-0123456789abcdef", limit=5, overrides={}).section
    assert entry["status"] == OK and entry["warnings"] == warnings
    assert entry["candidates"] == [] and entry["truncated_to"] is None
    assert entry["total_before_limit"] == (0 if known else None)
    assert _sections.parking_facts(entry) == {"count": 0 if known else None,
                                              "total_before_limit": 0 if known else None,
                                              "nearest": None}


@pytest.mark.parametrize("include, by_axis, by_kind, blind", [
    (["trail"], {"trail": None}, {"line": None}, True),
    (["trail", "road"], {"trail": None, "road": None}, {"line": None}, True),
    (["waterway"], {"waterway": None}, {"corridor": None}, True),
    (["road", "waterway"], {"road": None, "waterway": None}, {"line": None, "corridor": None},
     True),
], ids=["trail", "lines", "waterway", "road-waterway"])
def test_hasty_with_nothing_asked_for_readable_counts_nothing(ctx, monkeypatch, include, by_axis,
                                                              by_kind, blind):
    """No osm.json, nothing kept. A kind none of whose axes asked for could
    be read is null, and when no axis asked for could be, so are the total
    and the summary's count: the feature's 0 there counted nothing. Every
    axis the hasty list offers is read from OpenStreetMap, so with none of
    it nothing asked for was counted."""
    from sar_geo_mcp.planning.boundary import MISSING_OSM_WARNING

    def no_osm(ctx, **arguments):
        return {"set_id": "hasty-a1b2c3", "candidates": [], "params": {"include": include},
                "summary": {"count": 0, "by_kind": {}, "by_axis": {}, "total_before_limit": 0},
                "warnings": [MISSING_OSM_WARNING]}

    monkeypatch.setattr(hasty_feature, "hasty_candidates", no_osm)
    entry = _sections.hasty_section(ctx, "aoi-0123456789abcdef", limit=5,
                                    overrides={"include": include}).section
    assert entry["by_axis"] == by_axis and entry["by_kind"] == by_kind
    assert entry["total_before_limit"] == (None if blind else 0)
    assert _sections.hasty_facts(entry) == {"count": None if blind else 0, "by_kind": by_kind,
                                            "by_axis": by_axis,
                                            "total_before_limit": None if blind else 0}


def test_hasty_counts_on_ground_with_openstreetmap_are_the_feature_s(ctx, monkeypatch):
    """With osm.json read, an axis with nothing kept is absent - a real
    "none" - and no null is added."""
    def with_osm(ctx, **arguments):
        return {"set_id": "hasty-a1b2c3", "candidates": [], "params": {"include": ["trail"]},
                "summary": {"count": 0, "by_kind": {}, "by_axis": {}, "total_before_limit": 0},
                "warnings": ["no hasty candidates in this area with these settings"]}

    monkeypatch.setattr(hasty_feature, "hasty_candidates", with_osm)
    entry = _sections.hasty_section(ctx, "aoi-0123456789abcdef", limit=5, overrides={}).section
    assert entry["by_axis"] == {} and entry["by_kind"] == {}


# --- One CalTopo assignment ------------------------------------------------------


def test_an_assignment_that_cannot_be_staged_is_a_record_naming_it(ctx, monkeypatch):
    """`stage_assignment` never raises: the record fails, and every section
    standing on it fails with the assignment named, not a radius it has not
    got."""
    def unreadable(ctx, **arguments):
        raise LookupError("no assignment matches 'ZZ'")

    monkeypatch.setattr(staging_component, "prepare_terrain", unreadable)
    staging = {"assignment": _sections.stage_assignment(ctx, map_id="MAP1234", label="ZZ")}
    record = staging["assignment"]
    assert record["status"] == FAILED and record["handle"] is None
    assert record["error"] == "LookupError: no assignment matches 'ZZ'"
    assert record["detail_call"] == {"tool": "prepare_terrain", "arguments": {
        "map_id": "MAP1234", "label": "ZZ", "public": True}}
    with pytest.raises(StagingFailed, match=r"the assignment ground \(assignment 'ZZ' on map "
                                            r"MAP1234\) could not be staged"):
        ground_handle(staging, "assignment")


def test_staleness_warnings_speak_for_every_section_on_the_one_ground():
    """The sections share one staged ground, so the latest check speaks for
    all of them - as the scenario's top-level `staleness` does - and a move
    an earlier check found is not lost."""
    current = {"current": True, "checked": True}
    moved = {"current": False, "checked": True}
    unchecked = {"current": None, "checked": False, "reason": "could not revalidate (Timeout)"}
    warnings = _sections.staleness_warnings
    assert warnings({}) == []
    assert warnings({"metrics": current, "features": current}) == []
    assert warnings({"metrics": unchecked, "features": current}) == []
    for checks in ({"metrics": current, "features": moved},
                   {"metrics": moved, "coverage": unchecked}):
        (said,) = warnings(checks)
        assert f"every section here ({', '.join(checks)}) was computed on the staged ground, " \
               "so its numbers describe the OLD polygon" in said
    (said,) = warnings({"metrics": unchecked, "features": unchecked})
    assert "at the latest check (could not revalidate (Timeout)): every section here " \
           "(metrics, features) describes the ground as it was staged" in said
    assert "an earlier check" not in said
    (said,) = warnings({"metrics": current, "features": unchecked})
    assert "; an earlier check in this call found it as staged:" in said
    (said,) = warnings({"metrics": moved, "features": current})
    assert said.startswith("a check during this call found the assignment boundary changed")
    assert "OLD polygon" not in said


def test_a_gap_point_is_projected_back_from_web_mercator():
    """coverage_for gives gap outlines in the coverage grid's own Web
    Mercator metres; the point the section gives is WGS84, inside the gap."""
    from pyproj import Transformer

    x, y = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform(
        POINT[1], POINT[0])
    # An L-shaped gap, whose centroid lies outside it.
    ring = [[x, y], [x + 100, y], [x + 100, y + 10], [x + 10, y + 10], [x + 10, y + 100],
            [x, y + 100], [x, y]]
    point = _sections.gap_point({"type": "Polygon", "coordinates": [ring]})
    assert point["type"] == "Point"
    lon, lat = point["coordinates"]
    assert POINT[1] <= lon <= POINT[1] + 0.001 and POINT[0] <= lat <= POINT[0] + 0.001
    back_x, back_y = Transformer.from_crs("EPSG:4326", "EPSG:3857",
                                          always_xy=True).transform(lon, lat)
    from shapely.geometry import Point, Polygon
    assert Polygon(ring).buffer(0.5).contains(Point(back_x, back_y))
    assert _sections.gap_point(None) is None


def test_feature_kinds_and_track_names_are_refused_in_the_tools_words():
    assert _sections.check_feature_kinds(["ridge", "drainage", "ridge"], needed=True) == [
        "ridge", "drainage"]
    assert _sections.check_feature_kinds([], needed=False) == []
    with pytest.raises(ValueError, match="unknown feature kind.*Available: steep_band"):
        _sections.check_feature_kinds(["ridges"], needed=False)
    assert _sections.check_track_names(("101", "102"), needed=True) == ["101", "102"]
    assert _sections.check_track_names([], needed=False) == []
    with pytest.raises(ValueError, match="track_names must be a list"):
        _sections.check_track_names(None, needed=False)


def test_an_assignment_s_kind_is_read_from_disk_or_is_unknown(ctx):
    directory = Path(ctx.config.cache_dir) / "staged" / "MAP1234_AB"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text("{}", encoding="utf-8")
    (directory / "subject.json").write_text(json.dumps({"geometry": {
        "type": "LineString", "coordinates": [[0, 0], [1, 1]]}}), encoding="utf-8")
    assert _sections.assignment_kind(ctx, "MAP1234_AB") == "line"
    assert _sections.assignment_kind(ctx, "MAP1234_CD") is None        # nothing staged
    assert _sections.assignment_kind(ctx, "nonsense") is None


@pytest.mark.parametrize("kinds, got", [(None, "NoneType"), (3, "int"),
                                        ({"ridge": True}, "dict")])
def test_feature_kinds_that_are_not_a_list_are_refused_naming_what_came(ctx, monkeypatch,
                                                                       kinds, got):
    """Refused whether or not the features are asked for, as every argument
    is, and before the map is read."""
    for needed in (True, False):
        with pytest.raises(ValueError, match=f"kinds must be a list of feature kinds; got {got}"):
            _sections.check_feature_kinds(kinds, needed=needed)
    debrief, assignment, stubs = scenario_under_stubs("debrief_assignment", monkeypatch)
    with pytest.raises(ValueError, match=f"kinds must be a list of feature kinds; got {got}"):
        debrief(ctx, *assignment, kinds=kinds, include=["metrics"])
    assert stubs.nothing_called()


def test_an_unreadable_subject_is_an_unknown_kind_and_coverage_does_not_guess(ctx,
                                                                              monkeypatch):
    """A staged subject that cannot be parsed reads as a kind unknown - not
    an error - so with the metrics failed, coverage refuses to guess a sweep
    width and never calls coverage_for."""
    from sar_geo_mcp.features import coverage as coverage_feature

    directory = Path(ctx.config.cache_dir) / "staged" / "MAP1234_AB"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text("{}", encoding="utf-8")
    (directory / "subject.json").write_text("{not json", encoding="utf-8")
    assert _sections.assignment_kind(ctx, "MAP1234_AB") is None
    called = []
    monkeypatch.setattr(coverage_feature, "coverage_for",
                        lambda ctx, **arguments: called.append(arguments))
    metrics = section(FAILED, error="OSError: dem.tif could not be read")
    failed = guard(ctx, lambda: _sections.coverage_section(
        ctx, "MAP1234_AB", track_names=["101"], overrides={}, metrics=metrics))
    assert failed.section["status"] == FAILED and called == []
    assert failed.section["error"].startswith(
        "DependencyFailed: coverage of this assignment (whether it is an area or a line could "
        "not be read from the staged ground) takes its sweep width from terrain_metrics")
    assert failed.section["error"].endswith("OSError: dem.tif could not be read")
    assert "detail_call" not in failed.section


@pytest.mark.parametrize("geometry", [{"type": "Polygon", "coordinates": []},
                                      {"type": "GeometryCollection", "geometries": []},
                                      {}, None], ids=["empty-polygon", "empty-collection",
                                                      "empty-mapping", "none"])
def test_a_gap_with_no_outline_has_no_point(geometry):
    assert _sections.gap_point(geometry) is None


# --- Every scenario ------------------------------------------------------------


@pytest.mark.parametrize("module", MODULES, ids=lambda module: module.__name__.rsplit(".")[-1])
def test_every_scenario_names_its_sections_facts_and_settings_alike(module):
    """What `_sections` reads every scenario by: its sections each a known
    section with one helper, the `include` enum those very sections, facts
    only for sections it has, and each keyword it sets one its feature
    takes - else the refusal naming its argument could never fire."""
    sections = module.SECTIONS
    assert len(set(sections)) == len(sections)
    assert set(sections) <= set(_sections.SECTION_TOOLS) | set(_sections.SECTION_PARTS)
    assert set(module.FACTS) <= set(sections) and _sections.EXPORT_SECTION not in module.FACTS
    assert list(module.FACTS) == [name for name in sections if name in module.FACTS]
    function = next(value for name, value in vars(module).items()
                    if name in SCENARIO_TESTS and callable(value))
    include = get_type_hints(function)["include"]
    (listed, _none) = get_args(include)
    assert get_args(get_args(listed)[0]) == sections
    assert set(module.SCENARIO_SETS) <= set(sections)
    parameters = inspect.signature(function).parameters
    for name, sets in module.SCENARIO_SETS.items():
        if name in _sections.SECTION_PARTS:
            takes = {keyword for tool in _sections.SECTION_PARTS[name]
                     for keyword in _sections.FEATURE_KEYWORDS[tool]}
        else:
            takes = set(_sections.FEATURE_KEYWORDS[_sections.SECTION_TOOLS[name][0]])
        assert set(sets) <= takes, name
        assert set(sets.values()) <= set(parameters), name


@pytest.mark.parametrize("name", list(SCENARIO_TESTS))
def test_every_scenario_answers_in_one_envelope(ctx, monkeypatch, name):
    """MCP clients read the four scenarios' answers generically: the same
    top-level fields, a section for every section the scenario has, each
    with status, error and warnings, and a staging record and handle for
    every ground."""
    plan, arguments, _stubs = scenario_under_stubs(name, monkeypatch)
    result = plan(ctx, *arguments)
    assert {"status", "summary", "sections", "staging", "handles", "exports", "warnings",
            "note"} <= set(result)
    assert result["status"] == OK and isinstance(result["note"], str) and result["note"]
    assert list(result["sections"]) == list(scenario_module(name).SECTIONS)
    for entry in result["sections"].values():
        assert {"status", "error", "warnings"} <= set(entry)
        assert entry["status"] == OK and entry["error"] is None
        assert isinstance(entry["warnings"], list)
    assert set(result["handles"]) == set(result["staging"])
    for ground, record in result["staging"].items():
        assert {"status", "error", "warnings", "handle", "detail_call"} <= set(record)
        assert record["status"] == OK and result["handles"][ground] == record["handle"]
    assert set(result["summary"]) == set(scenario_module(name).FACTS)
    assert isinstance(result["exports"], list) and isinstance(result["warnings"], list)
    json.dumps(result)


def test_every_scenario_taking_a_radius_unit_warns_of_a_radius_a_lot_or_two_across(
        ctx, monkeypatch):
    """`radius` is read in `radius_unit`, metres unless given, so half a
    mile asked as `radius=0.5` is half a metre: a hasty or parking search of
    a lot or two that reads as a search of the area. Every scenario taking
    the unit says so, as plan_urban_search does."""
    warned = {}
    for name in SCENARIO_TESTS:
        if "radius_unit" not in inspect.signature(getattr(scenario_module(name),
                                                          name)).parameters:
            continue
        plan, arguments, _stubs = scenario_under_stubs(name, monkeypatch)
        result = plan(ctx, *arguments, radius=0.5)
        assert result["radius_m"] == 0.5
        warned[name] = [warning for warning in result["warnings"]
                        if "0.5 m" in warning and "radius_unit" in warning]
    assert set(warned) == {"plan_initial_response", "plan_urban_search"}
    assert all(len(said) == 1 for said in warned.values()), warned


#: Per scenario with caching disabled: the sections that stand on staged
#: ground, and where the "nowhere to write" warning is said.
UNCACHED = {
    "plan_medevac": (("lz", "egress"), "warnings"),
    "plan_initial_response": (("hasty", "parking", "travel_rings", "lz"), "export"),
    "plan_urban_search": (("urban", "parking"), "export"),
}


@pytest.mark.parametrize("name", list(UNCACHED))
def test_with_caching_disabled_a_point_scenario_answers_what_needs_no_ground(monkeypatch,
                                                                             name):
    """CaltopoConfig documents cache_dir=None as the test setting: every
    ground is refused by staging itself, each section standing on one fails
    saying so, the hospitals still answer, and nothing raises."""
    plan, point, stubs = scenario_under_stubs(name, monkeypatch)
    monkeypatch.setattr(staging_component, "prepare_area", REAL_PREPARE_AREA)
    result = plan(ServerContext.build(CaltopoConfig(cache_dir=None)), *point)
    grounds, said_in = UNCACHED[name]
    refusal = "ValueError: staging needs a cache directory; none is configured"
    assert result["status"] == "partial"
    for record in result["staging"].values():
        assert (record["status"], record["error"], record["handle"]) == (FAILED, refusal, None)
    for section_name in grounds:
        entry = result["sections"][section_name]
        assert entry["status"] == FAILED and entry["error"].startswith("StagingFailed: the ")
        assert entry["error"].endswith(refusal)
    assert result["sections"]["medical"]["status"] == OK and len(stubs.calls["medical"]) == 1
    assert result["summary"]["medical"]["nearest_trauma_center"]["name"] == "Stanford Hospital"
    assert result["exports"] == []
    nowhere = ("no ground was staged, so there is nowhere beside it to write the CalTopo "
               "export; each section's detail_call returns what it found")
    if said_in == "warnings":
        assert result["warnings"] == [nowhere]
    else:
        export = result["sections"]["export"]
        assert export["status"] == OK and nowhere in export["warnings"]
        assert export["path"] is None and export["markers_path"] is None
        assert any(warning.startswith("no CalTopo export was written")
                   for warning in result["warnings"])


def test_with_caching_disabled_a_debrief_fails_every_section_without_raising(monkeypatch):
    debrief, assignment, stubs = scenario_under_stubs("debrief_assignment", monkeypatch)
    monkeypatch.setattr(staging_component, "prepare_terrain", REAL_PREPARE_TERRAIN)
    result = debrief(ServerContext.build(CaltopoConfig(cache_dir=None)), *assignment)
    assert result["status"] == FAILED and result["handles"] == {"assignment": None}
    assert result["staging"]["assignment"]["error"] == (
        "ValueError: staging needs a cache directory; none is configured")
    for entry in result["sections"].values():
        assert entry["status"] == FAILED
        assert "staging needs a cache directory" in entry["error"]
    assert not any(stubs.calls.values())
    assert result["summary"] == {"metrics": None, "coverage": None, "features": None}


def test_plan_medevac_never_carries_a_credential_from_a_failure(secret_ctx, monkeypatch):
    """Every way plan_medevac writes an error down - a ground that could not
    be staged, a section standing on it, a feature that raised, one egress
    mode that raised, an export file that could not be written - with an
    error carrying all three credentials."""
    from sar_geo_mcp.features import egress as egress_feature
    from sar_geo_mcp.features import medical as medical_feature

    plan_medevac, patient, stubs = scenario_under_stubs("plan_medevac", monkeypatch)
    monkeypatch.setattr(staging_component, "prepare_area", LeakyStaging({2.0}))

    def egress(ctx, **arguments):
        if arguments["mode"] == "carry_out":
            raise_leak()
        return stubs.egress(ctx, **arguments)

    monkeypatch.setattr(egress_feature, "egress_routes", egress)
    monkeypatch.setattr(medical_feature, "nearest_medical_facilities", raise_leak)
    monkeypatch.setattr(_sections, "atomic_write_text", raise_leak)
    result = plan_medevac(secret_ctx, *patient)

    assert result["status"] == "partial" and result["exports"] == []
    sections = result["sections"]
    said = [result["staging"]["lz"]["error"], sections["lz"]["error"],
            sections["egress"]["error"], sections["egress"]["modes"]["carry_out"]["error"],
            sections["medical"]["error"], *result["warnings"]]
    assert len(said) == 6 and sections["egress"]["modes"]["walk_out"]["status"] == OK
    for text in said:
        assert "<redacted>" in text, text
    assert leaks(result) == []


def test_plan_initial_response_never_carries_a_credential_from_a_failure(secret_ctx,
                                                                        monkeypatch):
    """The rings ground, the parking feature, the draft assignments and the
    markers file each failing with all three credentials in the error."""
    plan, place, _stubs = scenario_under_stubs("plan_initial_response", monkeypatch)
    monkeypatch.setattr(staging_component, "prepare_area", LeakyStaging({10.0}))
    monkeypatch.setattr(parking_feature, "parking_candidates", raise_leak)
    monkeypatch.setattr(candidates_component, "caltopo_export", raise_leak)
    monkeypatch.setattr(_sections, "atomic_write_text", raise_leak)
    result = plan(secret_ctx, *place)

    sections = result["sections"]
    assert (sections["hasty"]["status"], sections["lz"]["status"]) == (OK, OK)
    said = [result["staging"]["rings"]["error"], sections["travel_rings"]["error"],
            sections["parking"]["error"], *sections["export"]["warnings"]]
    assert len(said) == 5
    for text in said:
        assert "<redacted>" in text, text
    assert leaks(result) == []


@pytest.mark.parametrize("failing", ["staging", "metrics"])
def test_debrief_assignment_never_carries_a_credential_from_a_failure(secret_ctx, monkeypatch,
                                                                     failing):
    """An assignment that could not be staged fails every section with its
    reason; metrics that raised fail an area's coverage with theirs."""
    from sar_geo_mcp.components import ground as ground_component

    debrief, assignment, _stubs = scenario_under_stubs(
        "debrief_assignment", monkeypatch,
        staging_error=RuntimeError(LEAK) if failing == "staging" else None)
    if failing == "metrics":
        monkeypatch.setattr(ground_component, "terrain_metrics", raise_leak)
    result = debrief(secret_ctx, *assignment)

    assert result["status"] == (FAILED if failing == "staging" else "partial")
    said = [result["sections"]["metrics"]["error"], result["sections"]["coverage"]["error"]]
    if failing == "staging":
        said.append(result["staging"]["assignment"]["error"])
    for text in said:
        assert "<redacted>" in text, text
    assert leaks(result) == []
