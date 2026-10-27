"""Every tool is a plain function first (AGENTS.md, rule 1).

A scenario calls feature and component functions directly, with the
`ServerContext`, never through MCP. These tests call several of them that
way and check each payload against the same call made through fastmcp's
in-memory client, on a server built around the very same context - so the
map reader and the staged ground are shared, and a difference can only come
from the wrapper. A scenario then sees exactly what an MCP caller sees.

`tests/test_tool_registration.py` checks the pattern's shape for every tool
(the function exists, takes `ctx` first and then the wrapper's parameters,
and the wrapper only calls it, passing each argument on); these check the
answers for a spread of them: stateless components, the components that
read a map, and features on staged ground. Two more check what differs
between the paths by design: a direct payload must not hand out the
server's own norms, and an MCP caller's bad argument is refused in the
tool's own name, not the wrapper's.

The last group calls the plain functions whose bodies only hand on: to a
domain function (`nearest_medical_facilities`, `shaded_relief`,
`slope_shading`, `unmarked_paths`, `site_facts`), with the resolver stubbed
where there is one. The registration test sees the wrapper hand each
argument to the plain function; these see the plain function hand each to
the domain function under its own name, the server's panel endpoint and
cache with them, and the resolver's status come back as `staleness`. A
swapped pair there (helicopter speed and air overhead, track width and
known tolerance) answers wrongly with no error, and no other test calls
these bodies: the scenario tests stub the plain functions themselves.
"""

from __future__ import annotations

import copy
import functools
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("fastmcp")

from fastmcp import Client, FastMCP  # noqa: E402
from fastmcp.exceptions import ToolError  # noqa: E402
from pydantic_core import to_jsonable_python  # noqa: E402
from test_medical import FakeServices  # noqa: E402
from test_medical import record as hifld_record  # noqa: E402
from test_parking import _site, _square_3857  # noqa: E402
from test_planning_tools import PLS, _square, _stub_staging, _trail_through  # noqa: E402

from sar_geo_mcp.caltopo.reader import MapReader  # noqa: E402
from sar_geo_mcp.components import (  # noqa: E402
    caltopo,
    candidates,
    coords,
    geometry,
    ground,
    norms,
    staging,
)
from sar_geo_mcp.config import CaltopoConfig  # noqa: E402
from sar_geo_mcp.context import ServerContext  # noqa: E402
from sar_geo_mcp.features import (  # noqa: E402
    boundary,
    egress,
    hasty,
    parking,
    relief,
    unmarked_paths,
)
from sar_geo_mcp.features import medical as medical_feature  # noqa: E402
from sar_geo_mcp.handles import Resolved  # noqa: E402
from sar_geo_mcp.planning import medical as medical_domain  # noqa: E402
from sar_geo_mcp.planning import site as site_domain  # noqa: E402
from sar_geo_mcp.planning.norms import load_norms  # noqa: E402
from sar_geo_mcp.server import LAYERS, SERVER_NAME  # noqa: E402


@pytest.fixture
def served(monkeypatch, tmp_path):
    """A server and the one context its tools close over.

    Built the way `build_server` builds it, but keeping hold of the context,
    which `build_server` does not hand back: a direct call has to use the
    same context to be comparing like with like."""
    monkeypatch.setattr(
        "sar_geo_mcp.caltopo.reader.session_module.fetch_public_map",
        lambda map_id, host, since=0: {"result": {
            "state": {"features": [
                {"id": "a1", "properties": {"class": "Assignment", "letter": "AA",
                                            "number": "", "title": "AA"},
                 "geometry": _square(PLS, 300)},
                {"id": "m1", "properties": {"class": "Marker", "title": "CP",
                                            "marker-symbol": "cp"},
                 "geometry": {"type": "Point", "coordinates": [PLS[1], PLS[0], 0, 0]}}]},
            "timestamp": 1_787_989_336_147}},
    )
    ctx = ServerContext.build(CaltopoConfig(cache_dir=tmp_path / "cache"))
    return serve(ctx), ctx


def serve(ctx):
    """A server whose tools close over `ctx`, built as `build_server` builds one."""
    mcp = FastMCP(SERVER_NAME)
    for layer in LAYERS:
        layer.register(mcp, ctx)
    return mcp


async def through_mcp(mcp, name, **arguments):
    async with Client(mcp) as client:
        result = await client.call_tool(name, arguments)
    return result.data


def as_json(payload):
    """A direct call's payload as JSON reads it: an MCP caller gets lists for
    tuples and plain dicts for models, so that is what is compared."""
    return to_jsonable_python(payload)


@pytest.mark.anyio
async def test_stateless_components_answer_the_same_without_mcp(served):
    mcp, ctx = served
    calls = [
        (coords.parse_coordinate, {"text": "37 24.132 N 122 18.984 W",
                                   "near_latitude": PLS[0], "near_longitude": PLS[1]}),
        (coords.parse_coordinate, {"text": "the second creek crossing"}),
        (coords.convert_coordinate, {"latitude": PLS[0], "longitude": PLS[1]}),
        (coords.distance_bearing, {"from_latitude": PLS[0], "from_longitude": PLS[1],
                                   "to_latitude": 37.41, "to_longitude": -122.30}),
        (geometry.geometry_bounds, {"geometry": _square(PLS, 300)}),
        (geometry.point_in_geometry, {"latitude": PLS[0], "longitude": PLS[1],
                                      "geometry": _square(PLS, 300)}),
        (geometry.buffer_geometry, {"geometry": {"type": "LineString", "coordinates": [
            [PLS[1] - 0.01, PLS[0]], [PLS[1] + 0.01, PLS[0]]]}, "distance_m": 15}),
        (norms.assignment_norms, {"resource_type": "GROUND"}),
    ]
    for function, arguments in calls:
        direct = function(ctx, **arguments)
        assert as_json(direct) == await through_mcp(mcp, function.__name__, **arguments), (
            function.__name__)


@pytest.mark.anyio
async def test_map_reading_components_answer_the_same_without_mcp(served):
    """The provenance block carries the read's age, which moves between two
    calls a moment apart, so it is compared by its keys; everything else
    must match exactly."""
    mcp, ctx = served
    calls = [
        (caltopo.list_assignments, {"map_id": "MAP1"}),
        (caltopo.get_assignment, {"map_id": "MAP1", "label": "aa"}),
        (caltopo.list_map_features, {"map_id": "MAP1", "feature_class": "Marker"}),
    ]
    for function, arguments in calls:
        direct = as_json(function(ctx, **arguments))
        via = await through_mcp(mcp, function.__name__, **arguments)
        assert set(direct.pop("provenance")) == set(via.pop("provenance"))
        assert direct == via, function.__name__
    # Not vacuous: the stub's map came back through both paths.
    assert caltopo.list_assignments(ctx, "MAP1")["assignments"][0]["label"] == "AA"


@pytest.mark.anyio
async def test_features_on_staged_ground_answer_the_same_without_mcp(served, monkeypatch):
    mcp, ctx = served
    _stub_staging(monkeypatch, routes=[_trail_through(PLS)])
    area = {"center_latitude": PLS[0], "center_longitude": PLS[1], "radius_m": 400}
    handle = staging.prepare_area(ctx, **area)["handle"]
    # The same request through MCP names the same ground, already staged.
    assert (await through_mcp(mcp, "prepare_area", **area))["handle"] == handle

    square = _square(PLS, 100)
    direct = boundary.describe_boundary(ctx, handle, boundary.BoundaryTarget(geometry=square))
    via = await through_mcp(mcp, "describe_boundary", handle=handle,
                            target={"geometry": square})
    assert as_json(direct) == via

    direct = egress.egress_routes(ctx, handle, PLS[0], PLS[1], mode="carry_out")
    via = await through_mcp(mcp, "egress_routes", handle=handle, center_latitude=PLS[0],
                            center_longitude=PLS[1], mode="carry_out")
    assert as_json(direct) == via
    assert [exit["kind"] for exit in via["exits"]] == ["2wd", "4x4"]


@pytest.mark.anyio
async def test_parking_answers_the_same_without_mcp(served, monkeypatch):
    """The staging stub writes roads and trails but no parking, which would
    leave only the envelope to compare. A lot is added to the staged OSM
    here, so a candidate row goes through both paths too. On its own ground
    rather than the egress test's, because a lot is also a way out and
    would change that test's exits."""
    mcp, ctx = served
    _stub_staging(monkeypatch, routes=[_trail_through(PLS)])
    handle = staging.prepare_area(ctx, center_latitude=PLS[0], center_longitude=PLS[1],
                                  radius_m=400)["handle"]
    osm_path = ctx.staged(handle).directory / "osm.json"
    osm = json.loads(osm_path.read_text(encoding="utf-8"))
    osm["parking"] = [_site("parking-w1", "parking",
                            _square_3857(PLS[1] + 0.001, PLS[0], 40),
                            name="Trailhead lot", surface="gravel", capacity="12")]
    osm_path.write_text(json.dumps(osm), encoding="utf-8")

    direct = as_json(parking.parking_candidates(ctx, handle))
    assert direct == await through_mcp(mcp, "parking_candidates", handle=handle)
    assert [c["name"] for c in direct["candidates"]] == ["Trailhead lot"]


def test_a_direct_payload_does_not_hand_out_the_server_s_norms(served):
    """Over MCP a payload is serialised, so a caller cannot reach the
    server's own objects. A direct caller could: `assignment_norms` once
    returned a shallow copy, and a scenario changing a nested value in it
    changed the norms hasty, boundary and urban read on every later call."""
    _mcp, ctx = served
    before = copy.deepcopy(ctx.norms)
    payload = norms.assignment_norms(ctx, "GROUND")
    nested = [value for value in payload.values() if isinstance(value, dict)]
    assert nested, "the norms should have nested sections to share"
    for value in nested:
        value["changed by a scenario"] = True
    assert ctx.norms == before


@pytest.mark.anyio
async def test_a_bad_argument_is_refused_in_the_tool_s_own_name(served):
    """FastMCP validates arguments with pydantic, which names the function in
    the error. The wrapper is `_<tool>_tool` in the source, so unless it is
    renamed (`context.mcp_tool`) a caller reads "call[_hasty_candidates_tool]"
    and may try to call a tool that does not exist. Every tool, through the
    real server: an argument no tool takes is refused before any body runs."""
    mcp, _ctx = served
    wrong = {}
    async with Client(mcp) as client:
        names = [tool.name for tool in await client.list_tools()]
        for name in names:
            with pytest.raises(ToolError) as refused:
                await client.call_tool(name, {"not_a_parameter": 1})
            message = str(refused.value)
            if f"call[{name}]" not in message or "_tool]" in message:
                wrong[name] = message.splitlines()[0]
    assert names and wrong == {}, wrong


@pytest.mark.anyio
async def test_a_set_made_without_mcp_is_the_one_mcp_reads_back(served, monkeypatch):
    """A candidate set persists in the staged directory, so a scenario that
    generates it directly and a planner who asks for it over MCP hold the
    same set and the same ids: the second call reads the first one's back."""
    mcp, ctx = served
    _stub_staging(monkeypatch, routes=[_trail_through(PLS)])
    handle = staging.prepare_area(ctx, center_latitude=PLS[0], center_longitude=PLS[1],
                                  radius_m=400)["handle"]

    direct = as_json(hasty.hasty_candidates(ctx, handle, include=["trail"]))
    via = await through_mcp(mcp, "hasty_candidates", handle=handle, include=["trail"])
    assert direct.pop("reused") is False and via.pop("reused") is True
    assert direct == via
    assert direct["candidates"], "the stub's trail should give a candidate"

    candidate_id = direct["candidates"][0]["id"]
    direct = candidates.get_candidate(ctx, handle, candidate_id)
    via = await through_mcp(mcp, "get_candidate", handle=handle, candidate_id=candidate_id)
    assert as_json(direct) == via
    assert via["geometry"]["type"] == "LineString"


# --------------------------------------------------------------------------
# Plain functions that hand on to a domain function


#: An area handle; the stubbed resolver answers for any.
HANDLE = "aoi-0123456789ab"
#: What the resolver says of ground whose map could not be read.
UNREADABLE = ("could not read map MAP1 (ConnectError); anything compared "
              "against the map is unavailable")


@pytest.fixture
def staged_ground(monkeypatch, tmp_path):
    """The resolver behind `ctx.staged`, stubbed: it records what it was
    asked and answers with a directory and a status of its own - a map it
    could not read, with the warning that says so - so a payload's
    `staleness` and warnings can be traced to it. The status is the one
    `handles.resolve` gives an area whose map read failed."""
    ground_ = SimpleNamespace(
        directory=tmp_path / "staged" / HANDLE,
        status={"current": True, "checked": True, "reason": "area input is immutable",
                "map": "unreadable", "map_reason": UNREADABLE, "warnings": [UNREADABLE]},
        asked=[])

    def resolve(handle, reader):
        ground_.asked.append((handle, reader))
        return Resolved(ground_.directory, copy.deepcopy(ground_.status))

    monkeypatch.setattr("sar_geo_mcp.context.resolve", resolve)
    return ground_


def recorder(monkeypatch, module, name, answer):
    """`module.name`, the domain function a plain function calls, replaced
    by one that records each call and answers a copy of `answer`."""
    calls = []

    def record(*args, **kwargs):
        calls.append((args, kwargs))
        return copy.deepcopy(answer)

    monkeypatch.setattr(module, name, record)
    return calls


#: A value for each of the medical tool's keywords, none the default and no
#: two alike, so one dropped or handed to another parameter shows.
MEDICAL_ARGUMENTS = {"per_category": 2, "search_radius_km": 45.0,
                     "helicopter_speed_kmh": 130.0, "air_overhead_minutes": 12.0,
                     "check_osm_helipads": False}


@pytest.mark.anyio
async def test_medical_hands_each_argument_to_its_own_parameter(served, monkeypatch):
    """The plain function's body is one call to the domain function, which
    nothing else runs: the scenarios' tests stub the plain function. Speed
    and overhead swapped there would put a helicopter at 12 km/h with 130
    minutes on the ground, and no test failed."""
    mcp, ctx = served
    answer = {"rows": [{"name": "Middle", "air_minutes": 14.6}], "warnings": ["as found"]}
    calls = recorder(monkeypatch, medical_feature, "nearest_medical_facilities_impl", answer)

    direct = medical_feature.nearest_medical_facilities(ctx, PLS[0], PLS[1],
                                                        **MEDICAL_ARGUMENTS)
    via = await through_mcp(mcp, "nearest_medical_facilities", latitude=PLS[0],
                            longitude=PLS[1], **MEDICAL_ARGUMENTS)
    assert direct == answer and via == answer
    assert calls == [((PLS[0], PLS[1]), MEDICAL_ARGUMENTS)] * 2


def test_medical_answers_what_the_domain_function_answers_for_the_same_arguments(
        served, monkeypatch):
    """End to end with HIFLD, OSRM, Overpass and the Census lookup faked:
    the plain function's payload is the domain function's for the same
    arguments, so the air time is the caller's speed and overhead and not
    the defaults (240 km/h, 5 min), the search radius is the one asked of
    HIFLD, `per_category` cuts the list, and helipads are not asked of
    OpenStreetMap when the caller said not to."""
    _mcp, ctx = served
    point = (37.42, -122.30)

    def hospitals():
        return FakeServices([
            hifld_record("NEAR", 37.43, -122.30, helipad="N", address="1 A"),     # ~1.1 km
            hifld_record("MIDDLE", 37.47, -122.30, helipad="Y", address="2 B"),   # ~5.6 km
            hifld_record("FAR", 37.52, -122.30, helipad="Y", address="3 C"),      # ~11.1 km
        ])

    used, expected = hospitals(), hospitals()
    monkeypatch.setattr(medical_feature, "nearest_medical_facilities_impl", functools.partial(
        medical_domain.nearest_medical_facilities, query=used.query, route=used.route,
        helipads=used.helipads, locate=used.locate))

    payload = medical_feature.nearest_medical_facilities(ctx, *point, **MEDICAL_ARGUMENTS)
    assert payload == medical_domain.nearest_medical_facilities(
        *point, query=expected.query, route=expected.route, helipads=expected.helipads,
        locate=expected.locate, **MEDICAL_ARGUMENTS)

    rows = [row for row in payload["rows"] if row["category"] == "medical_center"]
    assert [row["name"] for row in rows] == ["Near", "Middle"]      # two, not the default three
    middle = rows[1]
    assert middle["air_minutes"] == pytest.approx(
        middle["straight_line_m"] / 1000 / 130.0 * 60 + 12.0, abs=0.05)
    assert middle["air_minutes"] > 12.0                          # the overhead alone is 12
    assert used.queries[0][0] == 45.0
    assert used.pad_points == []


@pytest.mark.anyio
async def test_shaded_relief_hands_on_its_arguments_and_the_server_s_panel_endpoint(
        monkeypatch, tmp_path, staged_ground):
    """The CalTopo images need the server's WMS endpoint, which no caller
    passes: dropped, every CalTopo figure would come back null. Track width
    and known tolerance are both metres, so a swap is silent too."""
    ctx = ServerContext.build(CaltopoConfig(cache_dir=tmp_path / "cache",
                                            wms_token="test-wms-token"))
    mcp = serve(ctx)
    endpoint = ctx.config.wms_base_url()
    assert endpoint and "test-wms-token" in endpoint
    answer = {"lineaments": [{"length_m": 140.0}], "images": None, "warnings": ["raster"]}
    calls = recorder(monkeypatch, relief, "shaded_relief_impl", answer)
    arguments = {"layer": "r5", "returns": "image", "track_width_m": 6.5,
                 "known_tolerance_m": 22.0, "limit": 7}

    direct = relief.shaded_relief(ctx, HANDLE, **arguments)
    via = await through_mcp(mcp, "shaded_relief", handle=HANDLE, **arguments)
    expected = {**answer, "staleness": staged_ground.status}
    assert direct == expected and via == as_json(expected)
    assert calls == [((staged_ground.directory,),
                      {"wms_base_url": endpoint, "cache_dir": tmp_path / "cache",
                       **arguments})] * 2
    assert staged_ground.asked == [(HANDLE, ctx.reader)] * 2


@pytest.mark.anyio
async def test_slope_shading_hands_on_its_geometry_and_the_server_s_panel_endpoint(
        monkeypatch, tmp_path, staged_ground):
    """A geometry dropped on the way would shade the whole staged ground
    instead of the assignment the caller drew."""
    ctx = ServerContext.build(CaltopoConfig(cache_dir=tmp_path / "cache",
                                            wms_token="test-wms-token"))
    mcp = serve(ctx)
    answer = {"bands": [{"band": "30-45", "share": 0.4}], "images": None, "warnings": []}
    calls = recorder(monkeypatch, relief, "slope_shading_impl", answer)
    arguments = {"geometry": _square(PLS, 150), "layer": "sl", "returns": "both"}

    direct = relief.slope_shading(ctx, HANDLE, **arguments)
    via = await through_mcp(mcp, "slope_shading", handle=HANDLE, **arguments)
    expected = {**answer, "staleness": staged_ground.status}
    assert direct == expected and via == as_json(expected)
    assert calls == [((staged_ground.directory,),
                      {"wms_base_url": ctx.config.wms_base_url(),
                       "cache_dir": tmp_path / "cache", **arguments})] * 2
    assert calls[0][1]["wms_base_url"] is not None
    assert staged_ground.asked == [(HANDLE, ctx.reader)] * 2


@pytest.mark.anyio
async def test_unmarked_paths_hands_on_every_argument_and_adds_the_map_s_warning(
        served, monkeypatch, staged_ground):
    """Nine keywords, two of them metres side by side; and the resolver's
    warning - the map could not be read - joins the feature's own, after
    them, so a planner reading the candidate roads reads the caveat too."""
    mcp, ctx = served
    answer = {"candidates": [{"id": "unmarked-1", "sources": ["blm"]}],
              "warnings": ["from the feature"]}
    calls = recorder(monkeypatch, unmarked_paths, "unmarked_paths_impl", answer)
    arguments = {"sources": ["blm", "usgs_trails"], "known_tolerance_m": 12.5,
                 "track_width_m": 6.0, "min_length_m": 80.0, "limit": 5, "refresh": True,
                 "write_geojson": False, "geojson_limit": 3, "bridges": False}

    direct = unmarked_paths.unmarked_paths(ctx, HANDLE, **arguments)
    via = await through_mcp(mcp, "unmarked_paths", handle=HANDLE, **arguments)
    expected = {**answer, "warnings": ["from the feature", UNREADABLE],
                "staleness": staged_ground.status}
    assert direct == expected and via == as_json(expected)
    assert calls == [((staged_ground.directory,), arguments)] * 2

    # Ground whose status carries no warning adds none.
    del staged_ground.status["warnings"]
    quiet = unmarked_paths.unmarked_paths(ctx, HANDLE, **arguments)
    assert quiet["warnings"] == ["from the feature"]
    assert quiet["staleness"] == staged_ground.status


@pytest.mark.anyio
async def test_site_facts_hands_on_its_band_and_carries_the_staleness(
        served, monkeypatch, staged_ground):
    """A rope-rescue or helispot answer on ground whose map could not be
    read must say so: the resolver's status rides along as `staleness`.
    The band is the planner's, not the default 35 m."""
    mcp, ctx = served
    answer = {"drop_m": 18.0, "span_m": 42.0, "warnings": []}
    calls = recorder(monkeypatch, site_domain, "site_facts", answer)
    face = _square(PLS, 30)

    direct = ground.site_facts(ctx, HANDLE, face, band_m=20.0)
    via = await through_mcp(mcp, "site_facts", handle=HANDLE, geometry=face, band_m=20.0)
    expected = {**answer, "staleness": staged_ground.status}
    assert direct == expected and via == as_json(expected)
    assert calls == [((staged_ground.directory, face), {"band_m": 20.0})] * 2
    assert staged_ground.asked == [(HANDLE, ctx.reader)] * 2


@pytest.mark.xfail(strict=True, reason=(
    "PR #4 review B24: ServerContext.norms defaults to {}: a context built without "
    ".build() returns empty norms or fails later with a bare KeyError"))
def test_a_context_built_without_build_reads_the_corpus_its_config_names(
        monkeypatch, tmp_path):
    """AGENTS.md has a script call the plain functions with a
    `ServerContext`. Built directly rather than by `ServerContext.build`,
    the context's norms were `{}`: `assignment_norms` answered nothing, with
    a resource type it raised KeyError('polygon_area_ha'), and
    `describe_boundary` dropped its area flag and said it had banded against
    "the whole corpus", which was never loaded. Loading the corpus the
    config names is right, and so is refusing the context (norms made a
    required field); a silently empty one is not."""
    config = CaltopoConfig(cache_dir=tmp_path / "cache")
    try:
        bare = ServerContext(config=config, reader=MapReader(config))
    except TypeError:
        return                    # refused: norms are a required field
    assert bare.norms and bare.norms == load_norms(config.norms_path)
    assert norms.assignment_norms(bare, "GROUND")["target_area_acres"]["p50"] == 24.7

    _stub_staging(monkeypatch, routes=[_trail_through(PLS)])
    built = ServerContext.build(config)
    handle = staging.prepare_area(built, center_latitude=PLS[0], center_longitude=PLS[1],
                                  radius_m=400)["handle"]
    target = boundary.BoundaryTarget(geometry=_square(PLS, 20))       # 0.4 acres
    expected = boundary.describe_boundary(built, handle, target, resource_type="GROUND")
    assert [flag["metric"] for flag in expected["norm_flags"]] == ["area_acres"]
    assert boundary.describe_boundary(bare, handle, target, resource_type="GROUND") == expected
