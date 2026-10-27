"""The planning tools through the in-memory client: registration, schemas,
JSON-shaped answers. The computation is tested where it lives."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sar_geo_mcp.geo import HA_PER_ACRE
from PIL import Image

pytest.importorskip("fastmcp")

from fastmcp import Client  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402
from test_grid import write_fetch_dir  # noqa: E402

from sar_geo_mcp.config import CaltopoConfig  # noqa: E402
from sar_geo_mcp.planning.lz import (  # noqa: E402
    NO_TOKEN_CHIP_WARNING,
    STRUCTURES_NOT_CHECKED,
    TREES_NOT_CHECKED,
    WATER_EVT_MISSING_WARNING,
)
from sar_geo_mcp.server import build_server  # noqa: E402
from sar_geo_mcp.sources.osm import OSM_QUERY_VERSION  # noqa: E402

PLS = (37.4022, -122.3164)


@pytest.fixture
def server(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sar_geo_mcp.caltopo.reader.session_module.fetch_public_map",
        lambda map_id, host, since=0: {"result": {
            "state": {"features": [
                {"id": "a1", "properties": {"class": "Assignment", "letter": "AA", "number": ""},
                 "geometry": _square(PLS, 300)}]},
            "timestamp": 1_787_989_336_147}},
    )
    return build_server(CaltopoConfig(cache_dir=tmp_path / "cache"))


async def call(server, name, **arguments):
    async with Client(server) as client:
        result = await client.call_tool(name, arguments)
    return result.data


@pytest.mark.anyio
async def test_the_three_planning_tools_are_registered_with_descriptions(server):
    async with Client(server) as client:
        tools = {t.name: t for t in await client.list_tools()}
    for name in ("prepare_area", "assignment_norms", "describe_boundary"):
        assert name in tools and len(tools[name].description or "") > 80, name
    # resource_type changes which band the area flag is read against, so the
    # docstring has to say what passing it does.
    description = tools["describe_boundary"].description
    assert "resource_type" in description and "DOG_TRAIL" in description


@pytest.mark.anyio
async def test_assignment_norms_returns_the_file_and_a_resource_slice(server):
    everything = await call(server, "assignment_norms")
    assert everything["provenance"]["maps"] == 65
    assert "polygon_area_acres" in everything and "polygon_area_ha" not in everything
    ground = await call(server, "assignment_norms", resource_type="GROUND")
    assert ground["resource_type"] == "GROUND"
    assert ground["target_area_acres"]["p50"] == 24.7          # 10 ha
    assert ground["provenance"]["maps"] == 65


@pytest.mark.anyio
async def test_prepare_area_stages_and_describe_boundary_reads_the_handle(server, monkeypatch, tmp_path):
    _stub_staging(monkeypatch)
    prepared = await call(server, "prepare_area", center_latitude=PLS[0],
                          center_longitude=PLS[1], radius_m=400, map_id="MAP1")
    assert prepared["handle"].startswith("aoi-") and prepared["staged"] is True
    assert prepared["map_provenance"] is not None

    described = await call(server, "describe_boundary", handle=prepared["handle"],
                           target={"geometry": _square(PLS, 100)})
    assert described["shape"] in ("block", "corridor")
    assert "composition_pct" in described["boundary"]
    assert described["staleness"]["current"] is True
    assert described["staleness"]["reason"] == "area input is immutable"
    assert "map_provenance" in described["staleness"]
    # Lifted to the top level so the map context's age is beside the answer,
    # not buried in the staleness block.
    assert described["map_provenance"] == described["staleness"]["map_provenance"]
    assert isinstance(described["map_provenance"], dict)


@pytest.mark.anyio
async def test_describe_boundary_reports_map_provenance_for_an_assignment_handle(
        server, monkeypatch):
    """An assignment handle's status carries the map read under `provenance`;
    an area's under `map_provenance`. Both surface the same way."""
    _stub_staging(monkeypatch)
    prepared = await call(server, "prepare_terrain", map_id="MAP1", label="AA")
    described = await call(server, "describe_boundary", handle=prepared["handle"],
                           target={"assignment": "AA"})
    provenance = described["map_provenance"]
    assert isinstance(provenance, dict)
    assert set(provenance) == set(described["staleness"]["provenance"])


@pytest.mark.anyio
async def test_describe_boundary_map_provenance_is_none_without_a_map(server, monkeypatch):
    _stub_staging(monkeypatch)
    prepared = await call(server, "prepare_area", geometry=_square(PLS, 400))
    described = await call(server, "describe_boundary", handle=prepared["handle"],
                           target={"geometry": _square(PLS, 100)})
    assert described["staleness"]["map"] == "none"
    assert described["map_provenance"] is None


@pytest.mark.anyio
async def test_describe_boundary_by_assignment_label_excludes_itself_from_neighbours(server, monkeypatch):
    _stub_staging(monkeypatch)
    prepared = await call(server, "prepare_area", center_latitude=PLS[0],
                          center_longitude=PLS[1], radius_m=400, map_id="MAP1")
    described = await call(server, "describe_boundary", handle=prepared["handle"],
                           target={"assignment": "AA"})
    assert described["boundary"]["shared_pct"] == 0.0
    assert described["boundary"]["shared_with"] == []


@pytest.mark.anyio
async def test_describe_boundary_target_must_have_exactly_one_key(server, monkeypatch):
    _stub_staging(monkeypatch)
    prepared = await call(server, "prepare_area", geometry=_square(PLS, 400))
    with pytest.raises(Exception, match="exactly one"):
        await call(server, "describe_boundary", handle=prepared["handle"], target={})
    with pytest.raises(Exception, match="no candidate set"):
        await call(server, "describe_boundary", handle=prepared["handle"],
                   target={"candidate": "hasty-000000.1"})
    with pytest.raises(Exception, match="no map"):
        await call(server, "describe_boundary", handle=prepared["handle"],
                   target={"assignment": "AA"})


@pytest.mark.anyio
async def test_describe_boundary_target_schema_names_its_fields(server):
    async with Client(server) as client:
        tools = {t.name: t for t in await client.list_tools()}
    schema = tools["describe_boundary"].inputSchema
    target = schema["properties"]["target"]
    # fastmcp may inline the model or reference it under $defs; resolve either.
    if "$ref" in target:
        target = schema["$defs"][target["$ref"].split("/")[-1]]
    assert set(target["properties"]) == {"assignment", "geometry", "candidate"}


@pytest.mark.anyio
async def test_describe_boundary_refuses_two_keys(server, monkeypatch):
    _stub_staging(monkeypatch)
    prepared = await call(server, "prepare_area", geometry=_square(PLS, 400))
    with pytest.raises(Exception, match="exactly one"):
        await call(server, "describe_boundary", handle=prepared["handle"],
                   target={"assignment": "AA", "geometry": _square(PLS, 50)})


@pytest.mark.anyio
async def test_prepare_area_refuses_the_extent_limit(server):
    with pytest.raises(Exception, match="9000"):
        await call(server, "prepare_area", center_latitude=PLS[0], center_longitude=PLS[1],
                   radius_m=6000)


@pytest.mark.anyio
async def test_prepare_area_takes_a_radius_in_miles_and_returns_the_range(server, monkeypatch):
    async with Client(server) as client:
        tools = {t.name: t for t in await client.list_tools()}
    unit = tools["prepare_area"].inputSchema["properties"]["radius_unit"]
    assert set(unit["enum"]) == {"m", "km", "ft", "yd", "mi"}

    _stub_staging(monkeypatch)
    in_miles = await call(server, "prepare_area", center_latitude=PLS[0],
                          center_longitude=PLS[1], radius=0.5, radius_unit="mi")
    in_metres = await call(server, "prepare_area", center_latitude=PLS[0],
                           center_longitude=PLS[1], radius_m=804.672)
    assert in_miles["handle"] == in_metres["handle"]
    assert in_miles["area"]["radius_m"] == pytest.approx(804.672)
    assert in_miles["area"]["geometry"]["type"] == "Polygon"


@pytest.mark.anyio
async def test_hasty_candidates_returns_summaries_without_geometry(server, monkeypatch):
    _stub_staging(monkeypatch, routes=[_trail_through(PLS)])
    prepared = await call(server, "prepare_area", center_latitude=PLS[0],
                          center_longitude=PLS[1], radius_m=400, map_id="MAP1")
    result = await call(server, "hasty_candidates", handle=prepared["handle"])
    assert result["set_id"].startswith("hasty-")
    assert result["summary"]["count"] >= 1
    first = result["candidates"][0]
    assert "geometry" not in first and first["id"].startswith(result["set_id"] + ".")
    assert first["facts"]["length_m"] > 0
    assert result["staleness"]["current"] is True
    assert result["map_provenance"] is not None
    assert result["assumptions"]["ordered_by"].startswith("distance_from_center_m")


@pytest.mark.anyio
async def test_get_candidate_returns_the_geometry_and_describe_boundary_takes_the_id(server, monkeypatch):
    _stub_staging(monkeypatch, routes=[_trail_through(PLS)])
    prepared = await call(server, "prepare_area", center_latitude=PLS[0],
                          center_longitude=PLS[1], radius_m=400)
    result = await call(server, "hasty_candidates", handle=prepared["handle"], include=["trail"])
    cid = result["candidates"][0]["id"]
    full = await call(server, "get_candidate", handle=prepared["handle"], candidate_id=cid)
    assert full["id"] == cid and full["geometry"]["type"] == "LineString"
    described = await call(server, "describe_boundary", handle=prepared["handle"],
                           target={"candidate": cid})
    assert described["shape"] == "line"
    assert described["boundary"]["composition_pct"]["trail"] > 90


@pytest.mark.anyio
async def test_get_candidate_refuses_unknown_and_malformed_ids(server, monkeypatch):
    _stub_staging(monkeypatch, routes=[_trail_through(PLS)])
    prepared = await call(server, "prepare_area", center_latitude=PLS[0],
                          center_longitude=PLS[1], radius_m=400)
    await call(server, "hasty_candidates", handle=prepared["handle"])
    with pytest.raises(Exception, match="not a candidate id"):
        await call(server, "get_candidate", handle=prepared["handle"], candidate_id="../x")
    with pytest.raises(Exception, match="no candidate set"):
        await call(server, "get_candidate", handle=prepared["handle"], candidate_id="hasty-000000.1")


@pytest.mark.anyio
async def test_hasty_candidates_same_params_same_set(server, monkeypatch):
    _stub_staging(monkeypatch, routes=[_trail_through(PLS)])
    prepared = await call(server, "prepare_area", center_latitude=PLS[0],
                          center_longitude=PLS[1], radius_m=400)
    a = await call(server, "hasty_candidates", handle=prepared["handle"], include=["trail", "road"])
    b = await call(server, "hasty_candidates", handle=prepared["handle"], include=["road", "trail"])
    assert a["set_id"] == b["set_id"]
    assert [c["id"] for c in a["candidates"]] == [c["id"] for c in b["candidates"]]


@pytest.mark.anyio
async def test_parking_candidates_is_registered_and_answers_from_the_staged_ground(
        server, monkeypatch, tmp_path):
    _stub_staging(monkeypatch)
    async with Client(server) as client:
        tools = {t.name: t for t in await client.list_tools()}
    assert "parking_candidates" in tools
    description = tools["parking_candidates"].description or ""
    # The docstring has to say what a hardstand is, or a caller reads an
    # unverified row as a lot.
    assert "hardstand" in description and len(description) > 80

    area = await call(server, "prepare_area", center_latitude=PLS[0],
                      center_longitude=PLS[1], radius_m=400)
    payload = await call(server, "parking_candidates", handle=area["handle"])

    assert set(payload) >= {"candidates", "summary", "warnings", "staleness",
                            "query_version", "center"}
    assert isinstance(payload["candidates"], list)


@pytest.mark.anyio
async def test_parking_candidates_refuses_an_unknown_handle(server):
    with pytest.raises(Exception):
        await call(server, "parking_candidates", handle="aoi-0000000000000000")


@pytest.mark.anyio
async def test_the_same_call_twice_reads_the_set_back_instead_of_measuring_again(
        server, monkeypatch, tmp_path):
    """The set id is a hash of the parameters and the ground behind a handle
    does not change, so the second call is the same measurement: reading it
    back keeps the candidate ids a planner is holding pointing at the same
    lines, and costs seconds rather than minutes on a real area. A set that
    predates a re-stage is measured again."""
    _stub_staging(monkeypatch, routes=[_trail_through(PLS)])
    prepared = await call(server, "prepare_area", center_latitude=PLS[0],
                          center_longitude=PLS[1], radius_m=400)

    first = await call(server, "hasty_candidates", handle=prepared["handle"], include=["trail"])
    second = await call(server, "hasty_candidates", handle=prepared["handle"], include=["trail"])

    assert first["reused"] is False and second["reused"] is True
    assert second["set_id"] == first["set_id"]
    assert second["created_at"] == first["created_at"]        # the facts are dated
    assert second["candidates"] == first["candidates"]

    manifest_path = tmp_path / "cache" / "staged" / prepared["handle"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["restaged"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    third = await call(server, "hasty_candidates", handle=prepared["handle"], include=["trail"])
    assert third["reused"] is False
    assert third["set_id"] == first["set_id"]


@pytest.mark.anyio
async def test_hasty_candidates_reuses_only_while_the_map_version_matches(
        server, monkeypatch, tmp_path):
    """The overlap facts in a set were measured against the map as it stood
    at generation. If an assignment is drawn on the map after that, a set
    reused on parameters alone would silently describe stale overlap - so a
    set is reused only while the map's own version (its CalTopo change
    token) still matches the one it was generated against."""
    _stub_staging(monkeypatch, routes=[_trail_through(PLS)])
    prepared = await call(server, "prepare_area", center_latitude=PLS[0],
                          center_longitude=PLS[1], radius_m=400, map_id="MAP1")

    first = await call(server, "hasty_candidates", handle=prepared["handle"], include=["trail"])
    assert first["reused"] is False
    assert first["map_version"] is not None

    second = await call(server, "hasty_candidates", handle=prepared["handle"], include=["trail"])
    assert second["reused"] is True
    assert second["map_version"] == first["map_version"]
    assert second["set_id"] == first["set_id"]

    # The map changes: a new assignment is drawn, and CalTopo's change token
    # moves. The reader forces revalidation on every resolve (max_age_seconds
    # is 0 for an area's map read), so the next call picks this up itself.
    monkeypatch.setattr(
        "sar_geo_mcp.caltopo.reader.session_module.fetch_public_map",
        lambda map_id, host, since=0: {"result": {
            "state": {"features": [
                {"id": "a1", "properties": {"class": "Assignment", "letter": "AA", "number": ""},
                 "geometry": _square(PLS, 300)},
                {"id": "a2", "properties": {"class": "Assignment", "letter": "AB", "number": ""},
                 "geometry": _square(PLS, 60)}]},
            "timestamp": 1_787_989_999_999}},
    )

    third = await call(server, "hasty_candidates", handle=prepared["handle"], include=["trail"])
    assert third["reused"] is False
    assert third["set_id"] == first["set_id"]           # same parameters, same id
    assert third["map_version"] != first["map_version"]
    assert "AB" in third["candidates"][0]["facts"]["overlap_labels"]


@pytest.mark.anyio
async def test_hasty_candidates_without_a_map_reuses_regardless_of_map_version(
        server, monkeypatch):
    """No map means overlap was never measured against one, so there is
    nothing for a later map read to invalidate: a set staged without a
    map_id reuses across calls, and `map_version` is None both times."""
    _stub_staging(monkeypatch, routes=[_trail_through(PLS)])
    prepared = await call(server, "prepare_area", center_latitude=PLS[0],
                          center_longitude=PLS[1], radius_m=400)

    first = await call(server, "hasty_candidates", handle=prepared["handle"], include=["trail"])
    second = await call(server, "hasty_candidates", handle=prepared["handle"], include=["trail"])

    assert first["map_version"] is None and second["map_version"] is None
    assert first["reused"] is False and second["reused"] is True


@pytest.mark.anyio
async def test_get_candidate_dates_the_facts_it_returns(server, monkeypatch):
    """A set is measured once and persists; the map it was measured against is
    read fresh every call. Which is which has to be on the payload, or an
    overlap of zero reads as "nobody is on this line" when it means "nobody
    was, when this set was made"."""
    _stub_staging(monkeypatch, routes=[_trail_through(PLS)])
    prepared = await call(server, "prepare_area", center_latitude=PLS[0],
                          center_longitude=PLS[1], radius_m=400, map_id="MAP1")
    result = await call(server, "hasty_candidates", handle=prepared["handle"], include=["trail"])

    full = await call(server, "get_candidate", handle=prepared["handle"],
                      candidate_id=result["candidates"][0]["id"])

    assert full["set_id"] == result["set_id"]
    assert full["created_at"] == result["created_at"]
    assert "created_at" in full["facts_note"] and "overlap" in full["facts_note"]
    assert full["staleness"]["map_provenance"] is not None


@pytest.mark.anyio
async def test_hasty_candidates_reports_travel_minutes_rounded_with_its_cost_model(
        server, monkeypatch):
    """Travel time is the fact a planner bands candidates on, so it has to be
    on the payload at the precision the measurement supports, with the cost
    model stated once beside `effort`."""
    _stub_staging(monkeypatch, routes=[_trail_through(PLS)])
    prepared = await call(server, "prepare_area", center_latitude=PLS[0],
                          center_longitude=PLS[1], radius_m=400)

    result = await call(server, "hasty_candidates", handle=prepared["handle"], include=["trail"])

    minutes = result["candidates"][0]["facts"]["travel_minutes_from_center"]
    assert minutes is not None and minutes == round(minutes, 1)
    travel = result["assumptions"]["travel"]
    assert "movement" in travel and "dem_void" in travel and "not_a_route" in travel
    assert "effort" in result["assumptions"]


@pytest.mark.anyio
async def test_hasty_candidates_min_length_m_is_its_own_set(server, monkeypatch):
    """A caller asking for assignment-sized runs gets its own set id, not a
    silently different answer under the default set's name."""
    _stub_staging(monkeypatch, routes=[_trail_through(PLS)])
    prepared = await call(server, "prepare_area", center_latitude=PLS[0],
                          center_longitude=PLS[1], radius_m=400)

    default = await call(server, "hasty_candidates", handle=prepared["handle"], include=["trail"])
    asked = await call(server, "hasty_candidates", handle=prepared["handle"],
                       include=["trail"], min_length_m=593.0)

    assert default["set_id"] != asked["set_id"]
    assert asked["params"]["min_length_m"] == 593.0
    assert asked["assumptions"]["min_candidate_m"] == 593.0
    async with Client(server) as client:
        tools = {t.name: t for t in await client.list_tools()}
    assert "593" in (tools["hasty_candidates"].description or "")


@pytest.mark.anyio
async def test_hasty_candidates_unknown_include_value_raises(server, monkeypatch):
    _stub_staging(monkeypatch, routes=[_trail_through(PLS)])
    prepared = await call(server, "prepare_area", center_latitude=PLS[0],
                          center_longitude=PLS[1], radius_m=400)
    with pytest.raises(Exception, match="include"):
        await call(server, "hasty_candidates", handle=prepared["handle"], include=["fence"])


@pytest.mark.anyio
async def test_get_candidate_warns_when_the_set_was_computed_before_a_restage(server, monkeypatch, tmp_path):
    _stub_staging(monkeypatch, routes=[_trail_through(PLS)])
    prepared = await call(server, "prepare_area", center_latitude=PLS[0],
                          center_longitude=PLS[1], radius_m=400)
    result = await call(server, "hasty_candidates", handle=prepared["handle"], include=["trail"])
    cid = result["candidates"][0]["id"]

    manifest_path = tmp_path / "cache" / "staged" / prepared["handle"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["restaged"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    full = await call(server, "get_candidate", handle=prepared["handle"], candidate_id=cid)
    assert full["staleness"]["current"] is True
    assert len(full["set_warnings"]) == 1 and "re-staged" in full["set_warnings"][0]
    # The same candidate described carries the same warning: whichever tool a
    # planner reaches the stale geometry through, they are told.
    described = await call(server, "describe_boundary", handle=prepared["handle"],
                           target={"candidate": cid})
    assert any("re-staged" in w for w in described["warnings"])


@pytest.mark.anyio
async def test_get_candidate_reports_no_set_warnings_when_the_set_is_current(server, monkeypatch):
    _stub_staging(monkeypatch, routes=[_trail_through(PLS)])
    prepared = await call(server, "prepare_area", center_latitude=PLS[0],
                          center_longitude=PLS[1], radius_m=400)
    result = await call(server, "hasty_candidates", handle=prepared["handle"], include=["trail"])
    full = await call(server, "get_candidate", handle=prepared["handle"],
                      candidate_id=result["candidates"][0]["id"])
    assert full["set_warnings"] == []


@pytest.mark.anyio
async def test_hasty_candidates_without_a_map_says_overlap_is_unknown(server, monkeypatch):
    """No map means overlap was never measured. 0.0 would read as "measured,
    and nobody is on this line", which is the opposite of what is known."""
    _stub_staging(monkeypatch, routes=[_trail_through(PLS)])
    prepared = await call(server, "prepare_area", center_latitude=PLS[0],
                          center_longitude=PLS[1], radius_m=400)

    result = await call(server, "hasty_candidates", handle=prepared["handle"], include=["trail"])

    first = result["candidates"][0]
    assert first["facts"]["overlap_existing_pct"] is None
    assert "overlap was not computed" in first["facts"]["overlap_reason"]
    assert any("overlap was not computed" in w for w in result["warnings"])
    assert "manifest_stamp" not in result


@pytest.mark.anyio
async def test_hasty_candidates_with_a_map_measures_overlap(server, monkeypatch):
    _stub_staging(monkeypatch, routes=[_trail_through(PLS)])
    prepared = await call(server, "prepare_area", center_latitude=PLS[0],
                          center_longitude=PLS[1], radius_m=400, map_id="MAP1")

    result = await call(server, "hasty_candidates", handle=prepared["handle"], include=["trail"])

    first = result["candidates"][0]
    assert first["facts"]["overlap_existing_pct"] > 0     # the AA square straddles the trail
    assert first["facts"]["overlap_reason"] is None
    assert not any("overlap was not computed" in w for w in result["warnings"])


@pytest.mark.anyio
async def test_hasty_candidates_carries_an_unreadable_maps_reason_into_its_warnings(
        server, monkeypatch):
    """The resolver knows WHY the map is missing; a caller reading `warnings`
    should not have to go looking in the staleness block for the half of the
    story that explains the other half."""
    _stub_staging(monkeypatch, routes=[_trail_through(PLS)])
    prepared = await call(server, "prepare_area", center_latitude=PLS[0],
                          center_longitude=PLS[1], radius_m=400, map_id="MAP1")
    monkeypatch.setattr("sar_geo_mcp.caltopo.reader.MapReader.read", _refuses)

    result = await call(server, "hasty_candidates", handle=prepared["handle"], include=["trail"])

    assert result["staleness"]["map"] == "unreadable"
    assert any("could not read map MAP1" in w for w in result["warnings"])
    assert result["candidates"][0]["facts"]["overlap_existing_pct"] is None


@pytest.mark.anyio
async def test_hasty_candidates_limit_keeps_the_first_candidates(server, monkeypatch):
    _stub_staging(monkeypatch, routes=[_trail_through(PLS), _trail_through(PLS, offset_m=200,
                                                                          rid="route_1")])
    prepared = await call(server, "prepare_area", center_latitude=PLS[0],
                          center_longitude=PLS[1], radius_m=400)

    everything = await call(server, "hasty_candidates", handle=prepared["handle"], include=["trail"])
    limited = await call(server, "hasty_candidates", handle=prepared["handle"],
                         include=["trail"], limit=1)

    assert everything["summary"]["count"] == 2
    assert len(limited["candidates"]) == 1
    assert limited["summary"]["truncated_to"] == 1
    assert limited["summary"]["total_before_limit"] == 2
    assert limited["set_id"] != everything["set_id"]
    assert any("only the first 1 of 2" in w for w in limited["warnings"])


@pytest.mark.anyio
async def test_hasty_candidates_takes_corridor_widths_as_named_fields(server, monkeypatch):
    _stub_staging(monkeypatch)
    prepared = await call(server, "prepare_area", center_latitude=PLS[0],
                          center_longitude=PLS[1], radius_m=400)

    result = await call(server, "hasty_candidates", handle=prepared["handle"],
                        corridor_widths_m={"waterway": 60})

    widths = result["assumptions"]["corridor_widths_m"]
    assert widths == {"waterway": 60.0}
    with pytest.raises(Exception, match="waterway"):
        await call(server, "hasty_candidates", handle=prepared["handle"],
                   corridor_widths_m={"waterway": 0})


@pytest.mark.anyio
async def test_hasty_candidates_schema_names_the_widths_and_the_include_values(server):
    """A model reads the schema, not the docstring: the five kinds are an enum
    and the three widths are named fields, so a sixth kind or a misspelt class
    is refused before any ground is read."""
    async with Client(server) as client:
        tools = {t.name: t for t in await client.list_tools()}
    schema = tools["hasty_candidates"].inputSchema

    def resolve(node):
        # fastmcp may inline a model or reference it under $defs; resolve either.
        while "$ref" in node:
            node = schema["$defs"][node["$ref"].split("/")[-1]]
        return node

    widths = resolve(_one_of_object(schema["properties"]["corridor_widths_m"], schema))
    assert set(widths["properties"]) == {"waterway"}

    include = _one_of_object(schema["properties"]["include"], schema)
    item = resolve(include["items"])
    assert set(item["enum"]) == {"trail", "road", "waterway"}


@pytest.mark.anyio
async def test_hasty_candidates_works_on_an_assignment_handle(server, monkeypatch):
    """`prepare_terrain` stages an assignment, not an area: the subject has no
    centre of its own, so the centroid stands in and the set says so."""
    _stub_staging(monkeypatch, routes=[_trail_through(PLS)])
    prepared = await call(server, "prepare_terrain", map_id="MAP1", label="AA")

    result = await call(server, "hasty_candidates", handle=prepared["handle"], include=["trail"])

    assert result["assumptions"]["center_note"] == "the polygon's centroid"
    assert result["summary"]["count"] >= 1
    cid = result["candidates"][0]["id"]
    full = await call(server, "get_candidate", handle=prepared["handle"], candidate_id=cid)
    assert full["id"] == cid and full["geometry"]["type"] == "LineString"


@pytest.mark.anyio
async def test_populated_areas_saves_blocks_a_planner_can_fetch_by_id(monkeypatch, tmp_path):
    """Through the adapter: summaries without geometry, the set read back on
    the second call without fetching again, and a block's polygon from
    `get_candidate`."""
    from test_populated import elm_street, fake_fetch, town

    parcels, roofs = town()
    calls = []
    _stub_staging(monkeypatch, routes=[elm_street(PLS)])
    monkeypatch.setattr("sar_geo_mcp.caltopo.wms.fetch_panel",
                        fake_fetch(parcels, roofs, calls=calls, centre=PLS))
    server = build_server(CaltopoConfig(cache_dir=tmp_path / "cache", wms_token="tok"))
    area = await call(server, "prepare_area", center_latitude=PLS[0],
                      center_longitude=PLS[1], radius_m=170)

    first = await call(server, "populated_areas", handle=area["handle"])
    assert first["reused"] is False
    assert first["set_id"].startswith("populated-")
    assert [c["name"] for c in first["candidates"]] == ["Elm Street",
                                                         "Elm Street - large parcel"]
    assert all("geometry" not in c for c in first["candidates"])
    assert {c["kind"] for c in first["candidates"]} == {"block"}

    fetched = len(calls)
    second = await call(server, "populated_areas", handle=area["handle"])
    assert second["reused"] is True and len(calls) == fetched
    assert second["candidates"] == first["candidates"]

    full = await call(server, "get_candidate", handle=area["handle"],
                      candidate_id=first["candidates"][0]["id"])
    assert full["geometry"]["type"] == "Polygon"
    assert full["facts"]["parcels"] == 12
    assert "markers" not in full                     # off unless asked for

    marked = await call(server, "populated_areas", handle=area["handle"],
                        building_markers=True)
    assert marked["set_id"] != first["set_id"]
    assert marked["summary"]["markers"] == 16
    assert all("markers" not in c for c in marked["candidates"])
    block = await call(server, "get_candidate", handle=area["handle"],
                       candidate_id=marked["candidates"][0]["id"])
    assert len(block["markers"]) == 12
    assert {m["properties"]["marker-symbol"] for m in block["markers"]} == {"hut"}
    assert {m["properties"]["marker-color"] for m in block["markers"]} == {"#FFFFFF"}
    assert block["geometry"] == full["geometry"]


@pytest.mark.anyio
async def test_urban_assignments_then_caltopo_export_end_to_end(monkeypatch, tmp_path):
    """Stage in miles, group the town into assignments, export them for
    CalTopo, and read the file back."""
    from test_populated import elm_street, fake_fetch, town

    parcels, roofs = town()
    _stub_staging(monkeypatch, routes=[elm_street(PLS)])
    monkeypatch.setattr("sar_geo_mcp.caltopo.wms.fetch_panel",
                        fake_fetch(parcels, roofs, centre=PLS))
    server = build_server(CaltopoConfig(cache_dir=tmp_path / "cache", wms_token="tok"))
    area = await call(server, "prepare_area", center_latitude=PLS[0],
                      center_longitude=PLS[1], radius=0.1, radius_unit="mi")

    grouped = await call(server, "urban_assignments", handle=area["handle"], target_acres=2.0 / HA_PER_ACRE,
                         building_markers=True)
    assert grouped["set_id"].startswith("urban-") and grouped["reused"] is False
    (summary,) = grouped["candidates"]
    assert summary["kind"] == "assignment" and summary["name"] == "Elm Street"
    assert "geometry" not in summary and "markers" not in summary
    again = await call(server, "urban_assignments", handle=area["handle"], target_acres=2.0 / HA_PER_ACRE,
                       building_markers=True)
    assert again["reused"] is True

    exported = await call(server, "caltopo_export", handle=area["handle"],
                          set_id=grouped["set_id"])
    assert exported["assignments"] == 1 and exported["area_included"] is True
    assert exported["features"] == {"Assignment": 1, "Marker": 17, "Shape": 1}
    assert exported["labels"][0]["label"] == "U01 Elm Street"
    written = json.loads(Path(exported["path"]).read_text(encoding="utf-8"))
    assignment = next(f for f in written["features"] if f["properties"]["class"] == "Assignment")
    assert assignment["properties"]["status"] == "DRAFT"

    with pytest.raises(Exception, match="exactly one"):
        await call(server, "caltopo_export", handle=area["handle"])


@pytest.mark.anyio
async def test_populated_areas_without_a_token_is_refused(server, monkeypatch):
    _stub_staging(monkeypatch)
    area = await call(server, "prepare_area", center_latitude=PLS[0],
                      center_longitude=PLS[1], radius_m=170)
    with pytest.raises(Exception, match="WMS token"):
        await call(server, "populated_areas", handle=area["handle"])


@pytest.mark.anyio
@pytest.mark.parametrize("tool, staging, arguments", [
    ("populated_areas", {"radius_m": 170}, {}),
    ("urban_assignments", {"radius": 0.1, "radius_unit": "mi"}, {"target_acres": 2.0 / HA_PER_ACRE}),
])
async def test_the_urban_tools_carry_the_handles_staleness_and_warnings(
        server, monkeypatch, tmp_path, tool, staging, arguments):
    """Rule 1's envelope, which plan_urban_search reads directly:
    `staleness` is the resolver's status as it stands at this call, the
    resolver's warnings join the set's own in `warnings` - once, and never
    saved into the set, so a set computed while the map was unreadable does
    not go on saying so once it is readable - and the set's bookkeeping
    (`map_version`, `manifest_stamp`) stays out of the answer."""
    from test_populated import elm_street, fake_fetch, town

    from sar_geo_mcp.caltopo.reader import MapReader

    parcels, roofs = town()
    _stub_staging(monkeypatch, routes=[elm_street(PLS)])
    monkeypatch.setattr("sar_geo_mcp.caltopo.wms.fetch_panel",
                        fake_fetch(parcels, roofs, centre=PLS))
    real_read = MapReader.read
    offline = False

    def read(self, *args, **kwargs):
        if offline:
            raise RuntimeError("no network")
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(MapReader, "read", read)
    # The `server` fixture's public-map stub stays in force; this one has
    # the WMS token the parcel and structures layers need.
    server = build_server(CaltopoConfig(cache_dir=tmp_path / "cache", wms_token="tok"))
    area = await call(server, "prepare_area", center_latitude=PLS[0],
                      center_longitude=PLS[1], map_id="MAP1", **staging)
    # Ground with no OSM, so the set has a warning of its own to keep.
    (tmp_path / "cache" / "staged" / area["handle"] / "osm.json").unlink()

    offline = True
    computed = await call(server, tool, handle=area["handle"], **arguments)
    reread = await call(server, tool, handle=area["handle"], **arguments)
    offline = False
    online = await call(server, tool, handle=area["handle"], **arguments)

    reason = ("could not read map MAP1 (RuntimeError); anything compared "
              "against the map is unavailable")
    assert computed["reused"] is False and computed["candidates"]
    assert computed["staleness"] == {
        "current": True, "checked": True, "reason": "area input is immutable",
        "map": "unreadable", "map_reason": reason, "warnings": [reason]}
    assert computed["warnings"][-1] == reason
    assert computed["warnings"].count(reason) == 1
    own = computed["warnings"][:-1]
    # The set's own warnings first, the resolver's after them.
    assert own and own[0].startswith("osm.json is missing from the staged ground")

    assert reread["reused"] is True
    assert reread["warnings"] == [*own, reason]
    assert reread["staleness"] == computed["staleness"]

    assert online["reused"] is True
    assert online["warnings"] == own
    assert online["staleness"]["map"] == "read"
    assert online["staleness"]["map_provenance"] is not None
    assert "warnings" not in online["staleness"]

    for payload in (computed, reread, online):
        assert "map_version" not in payload
        assert "manifest_stamp" not in payload


def _refuses(*args, **kwargs):
    raise RuntimeError("no network")


def _one_of_object(node, schema):
    """An optional field's schema is `anyOf: [<the type>, {"type": "null"}]`;
    return the half that is not null (or the node itself when it is plain)."""
    for key in ("anyOf", "oneOf"):
        if key in node:
            for option in node[key]:
                if option.get("type") != "null":
                    return option
    return node


def _trail_through(center, offset_m=0, rid="route_0"):
    """A trail crossing the area west to east through its centre, in the
    Web Mercator GeoJSON osm.json carries. `offset_m` moves it north, for a
    second trail farther from the centre than the first."""
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    lat, lon = center
    x, y = t.transform(lon, lat)
    y += offset_m / 0.7952        # ground metres per Mercator unit at this latitude
    return {"id": rid, "kind": "route",
            "geometry": {"type": "LineString", "coordinates": [[x - 1500, y], [x + 1500, y]]},
            "properties": {"route_type": "TRAIL", "name": f"Long Trail {rid}",
                           "osm_id": 1, "surface": None}}


@pytest.mark.anyio
async def test_norms_file_can_be_replaced_per_server(tmp_path):
    path = tmp_path / "norms.json"
    path.write_text(json.dumps({"provenance": {"corpus": "elsewhere", "maps": 1},
                                "polygon_area_ha": {"p10": 1, "p25": 2, "p50": 3, "p75": 4, "p90": 5}}),
                    encoding="utf-8")
    server = build_server(CaltopoConfig(cache_dir=tmp_path / "cache", norms_path=path))
    norms = await call(server, "assignment_norms")
    assert norms["provenance"]["corpus"] == "elsewhere"


# --- helpers ---------------------------------------------------------------------------

def _square(center, half_m):
    import math
    lat, lon = center
    dlat = half_m / 111_320
    dlon = half_m / (111_320 * math.cos(math.radians(lat)))
    return {"type": "Polygon", "coordinates": [[
        [lon - dlon, lat - dlat], [lon + dlon, lat - dlat],
        [lon + dlon, lat + dlat], [lon - dlon, lat + dlat], [lon - dlon, lat - dlat]]]}


def _stub_staging(monkeypatch, routes=(), width: int = 64, vegetation_panel: bool = False):
    """Real staging downloads a DEM; the tests write a flat one over the
    area's own extent instead, through the same stage_terrain seam.
    `vegetation_panel` writes panels/vegetation.png as bare ground, the
    debrief-app path lz_candidates reads a canopy panel from when the
    staged directory already carries one."""
    from pathlib import Path

    from sar_geo_mcp.staging import compute_aoi, write_manifest

    def fake_stage(subject, map_id, out_dir, *, cache_dir, buffer_m, fingerprint,
                   access="public", subject_kind="assignment", **kwargs):
        out_dir = Path(out_dir)
        bbox = compute_aoi(subject.geometry, buffer_m=buffer_m)
        pixel = (bbox[2] - bbox[0]) / width
        height = max(1, int(round((bbox[3] - bbox[1]) / pixel)))
        transform = from_origin(bbox[0], bbox[3], pixel, pixel)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_fetch_dir(out_dir, np.zeros((height, width)), transform, subject.geometry,
                        latitude=37.4022)
        if vegetation_panel:
            panels = out_dir / "panels"
            panels.mkdir(parents=True, exist_ok=True)
            pixels = np.zeros((height, width, 3), dtype="uint8")
            pixels[:] = (180, 170, 150)   # bare-ground CIR colour
            Image.fromarray(pixels, "RGB").save(panels / "vegetation.png")
        (out_dir / "subject.json").write_text(subject.model_dump_json(), encoding="utf-8")
        (out_dir / "osm.json").write_text(
            json.dumps({"routes": list(routes), "query_version": OSM_QUERY_VERSION}),
            encoding="utf-8")
        write_manifest(out_dir, map_id=map_id, assignment_label=subject.label, bbox=bbox,
                       dem_bbox=bbox, dem_resolution_m=pixel * 0.7952, latitude_deg=37.4022,
                       panels={}, warnings=[], assignment_fingerprint=fingerprint,
                       access=access, subject_kind=subject_kind)
        return {"warnings": [], "dem_resolution_m": pixel * 0.7952}

    monkeypatch.setattr("sar_geo_mcp.staging.stage_terrain", fake_stage)


@pytest.mark.anyio
async def test_lz_candidates_says_what_road_and_parking_lot_mean(server):
    """Both fields are three-state, and `on_road` follows a ruling that
    reads wider than a caller would guess, so the docstring has to say so."""
    async with Client(server) as client:
        tools = {t.name: t for t in await client.list_tools()}
    description = tools["lz_candidates"].description
    assert "`road`" in description and "`parking_lot`" in description
    assert "Within the whole clearing" in description
    assert "`powerline`, `road`, `parking_lot`, `chips`" in description


@pytest.mark.anyio
async def test_lz_candidates_is_registered_and_refuses_coarse_ground(server, monkeypatch):
    """The stub stages 64 px over the whole buffered area - tens of metres a
    pixel - so the honest answer through the server is the refusal, with
    the pixel size named. That is the end-to-end path worth pinning: the
    handle resolves, the directory is read, the guard speaks."""
    async with Client(server) as client:
        tools = {t.name: t for t in await client.list_tools()}
    description = tools["lz_candidates"].description
    assert "preferred" in description and "marginal" in description
    assert "pilot" in description
    assert "unknown" in description and "null" in description

    from fastmcp.exceptions import ToolError

    _stub_staging(monkeypatch)
    prepared = await call(server, "prepare_area", center_latitude=PLS[0],
                          center_longitude=PLS[1], radius_m=400)
    with pytest.raises(ToolError, match="coarser than"):
        await call(server, "lz_candidates", handle=prepared["handle"],
                   center_latitude=PLS[0], center_longitude=PLS[1])


@pytest.mark.anyio
async def test_lz_candidates_grades_fine_ground_through_the_server(server, monkeypatch):
    """A flat, open, finely staged area: one preferred landing zone comes
    back through the server with the wrapper's own additions - the
    handle's staleness, and the structures and chip warnings, since the
    fixture config has no WMS token so neither layer is fetched and no
    network is touched. The canopy panel is read from disk, which is the
    debrief-app path."""
    # radius 100 m plus the 1,000 m planning buffer each side is 2,200 m
    # across; 1,100 px is 2 m per pixel, under the tool's 3 m limit.
    _stub_staging(monkeypatch, width=1100, vegetation_panel=True)
    prepared = await call(server, "prepare_area", center_latitude=PLS[0],
                          center_longitude=PLS[1], radius_m=100)
    result = await call(server, "lz_candidates", handle=prepared["handle"],
                        center_latitude=PLS[0], center_longitude=PLS[1])
    assert result["summary"]["count"] == 1
    only = result["candidates"][0]
    assert only["grade"] == "preferred"
    assert only["chip_path"] is None
    assert only["powerline"] is None          # the stub writes osm.json with no barriers
    # no token, so neither the structures layer nor a chip could be had
    # the stub stages osm.json but no evt.tif, so only shorelines closed water
    assert result["warnings"] == [f"{STRUCTURES_NOT_CHECKED} (no WMS token)",
                                  WATER_EVT_MISSING_WARNING,
                                  TREES_NOT_CHECKED.format(reason="no WMS token"),
                                  NO_TOKEN_CHIP_WARNING]
    assert result["staleness"]["current"] is True
    assert result["assumptions"]["grading"].endswith("the pilot decides")


@pytest.mark.anyio
async def test_lz_candidates_lets_the_caller_set_both_clearances(server):
    """The app adjusts both margins, so both are in the schema with the
    defaults the tool grades at."""
    async with Client(server) as client:
        tools = {t.name: t for t in await client.list_tools()}
    properties = tools["lz_candidates"].inputSchema["properties"]
    assert properties["structure_clearance_m"]["default"] == 30.0
    assert properties["powerline_clearance_m"]["default"] == 60.0


@pytest.mark.anyio
async def test_lz_candidates_refuses_a_negative_structure_clearance(server, monkeypatch):
    """Passed through, not dropped: the tool's own guard answers, naming
    the parameter. Fine ground, so the coarse-raster guard does not speak
    first."""
    from fastmcp.exceptions import ToolError

    _stub_staging(monkeypatch, width=1100, vegetation_panel=True)
    prepared = await call(server, "prepare_area", center_latitude=PLS[0],
                          center_longitude=PLS[1], radius_m=100)
    with pytest.raises(ToolError, match="structure_clearance_m"):
        await call(server, "lz_candidates", handle=prepared["handle"],
                   center_latitude=PLS[0], center_longitude=PLS[1],
                   structure_clearance_m=-1)


@pytest.mark.anyio
async def test_lz_candidates_passes_the_configured_static_keys_and_never_returns_them(
        monkeypatch, tmp_path):
    """The keys ride from the configuration to the tool's chip sources.
    That neither comes back in the envelope is test_lz's to prove: the fake
    here returns an empty one."""
    monkeypatch.setattr(
        "sar_geo_mcp.caltopo.reader.session_module.fetch_public_map",
        lambda map_id, host, since=0: {"result": {
            "state": {"features": [
                {"id": "a1", "properties": {"class": "Assignment", "letter": "AA", "number": ""},
                 "geometry": _square(PLS, 300)}]},
            "timestamp": 1_787_989_336_147}},
    )
    seen = {}

    def fake_lz(directory, **kwargs):
        seen.update(kwargs)
        return {"candidates": [], "warnings": []}

    monkeypatch.setattr("sar_geo_mcp.features.lz.lz_candidates_impl", fake_lz)
    server = build_server(CaltopoConfig(cache_dir=tmp_path / "cache",
                                        mapbox_token="SECRET-MAPBOX",
                                        google_maps_key="SECRET-GOOGLE"))
    _stub_staging(monkeypatch, width=1100, vegetation_panel=True)
    prepared = await call(server, "prepare_area", center_latitude=PLS[0],
                          center_longitude=PLS[1], radius_m=100)
    await call(server, "lz_candidates", handle=prepared["handle"],
               center_latitude=PLS[0], center_longitude=PLS[1])
    assert seen["mapbox_token"] == "SECRET-MAPBOX"
    assert seen["google_maps_key"] == "SECRET-GOOGLE"


def test_the_static_keys_are_read_from_the_environment_and_kept_out_of_repr(monkeypatch):
    monkeypatch.setenv("ZA_MAPBOX_TOKEN", "SECRET-MAPBOX")
    monkeypatch.setenv("ZA_GOOGLE_MAPS_KEY", "SECRET-GOOGLE")
    config = CaltopoConfig.from_env()
    assert config.mapbox_token == "SECRET-MAPBOX"
    assert config.google_maps_key == "SECRET-GOOGLE"
    assert "SECRET" not in repr(config)


@pytest.mark.anyio
async def test_egress_routes_is_registered_with_a_full_description(server):
    async with Client(server) as client:
        tools = {t.name: t for t in await client.list_tools()}
    description = tools["egress_routes"].description
    for words in ("resolution_m=10.0", "walk_out", "carry_out", "2WD", "4x4",
                  "BOTH KINDS ALWAYS COME BACK", "gate", "version 4", "unknown",
                  "rank", "team leader", "too fine", "on its line"):
        assert words in description, words
    properties = tools["egress_routes"].inputSchema["properties"]
    assert properties["mode"]["default"] == "walk_out"
    for name in ("on_way_factor", "off_way_factor", "max_grade"):
        assert properties[name]["default"] is None


@pytest.mark.anyio
async def test_egress_routes_passes_its_settings_through_and_merges_staleness(
        server, monkeypatch):
    seen = {}

    def fake_egress(directory, **kwargs):
        seen.update(kwargs)
        return {"exits": [], "warnings": ["from the tool"]}

    monkeypatch.setattr("sar_geo_mcp.features.egress.egress_routes_impl", fake_egress)
    _stub_staging(monkeypatch)
    prepared = await call(server, "prepare_area", center_latitude=PLS[0],
                          center_longitude=PLS[1], radius_m=400)
    result = await call(server, "egress_routes", handle=prepared["handle"],
                        center_latitude=PLS[0], center_longitude=PLS[1], mode="carry_out",
                        on_way_factor=0.3, off_way_factor=0.1, max_grade=0.35,
                        leave_way_factor=0.2)
    assert seen == {"center_latitude": PLS[0], "center_longitude": PLS[1],
                    "mode": "carry_out", "on_way_factor": 0.3, "off_way_factor": 0.1,
                    "max_grade": 0.35, "leave_way_factor": 0.2}
    assert result["warnings"] == ["from the tool", *result["staleness"].get("warnings", [])]
    assert result["staleness"]["current"] is True


@pytest.mark.anyio
async def test_egress_routes_answers_from_the_staged_ground(server, monkeypatch):
    """The stub stages flat ground and an osm.json with no ways and no gate
    list: both exits come back, each saying none is mapped, and gates are
    said to be unknown, not absent."""
    _stub_staging(monkeypatch)
    prepared = await call(server, "prepare_area", center_latitude=PLS[0],
                          center_longitude=PLS[1], radius_m=400)
    result = await call(server, "egress_routes", handle=prepared["handle"],
                        center_latitude=PLS[0], center_longitude=PLS[1])
    assert [e["kind"] for e in result["exits"]] == ["2wd", "4x4"]
    assert all(e["reachable"] is False for e in result["exits"])
    assert result["exits"][0]["reason"].startswith("no road or parking lot")
    assert any("gates are unknown" in w for w in result["warnings"])
