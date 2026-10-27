"""A candidate set as GeoJSON CalTopo imports as draft assignments."""

from __future__ import annotations

import json
import re

import pytest
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box, mapping, shape

from sar_geo_mcp.components.candidates import caltopo_export
from sar_geo_mcp.config import CaltopoConfig
from sar_geo_mcp.context import ServerContext
from sar_geo_mcp.planning.candidates import Candidate, CandidateSet, save_set
from sar_geo_mcp.planning.export import caltopo_collection, export_set

LON, LAT = -122.2409, 37.3569
D = 0.001                               # about 90 m east-west, 110 m north-south here


def _square(dx=0.0, dy=0.0, size=D):
    return box(LON + dx, LAT + dy, LON + dx + size, LAT + dy + size)


def _candidate(index, geometry, *, name="Elm Street", facts=None, markers=None,
               set_id="urban-abc123"):
    return {"id": f"{set_id}.{index}", "kind": "assignment", "axis": "urban",
            "axis_ids": [], "sources": [], "name": name, "geometry": mapping(geometry),
            "facts": facts or {"streets": ["Elm Street"], "parcels": 12, "buildings": 10,
                               "area_ha": 1.2, "distance_from_center_m": 150.0,
                               "bearing_from_center_deg": 90.0},
            "norm_flags": [], "warnings": [], "markers": markers}


def _assignments(collection):
    return [f for f in collection["features"] if f["properties"]["class"] == "Assignment"]


def test_each_candidate_is_a_draft_caltopo_assignment_labelled_in_order():
    collection, labels, warnings = caltopo_collection(
        [_candidate(1, _square()), _candidate(2, _square(dx=0.002), name="Oak Lane")],
        resource_type="GROUND", set_id="urban-abc123")
    assert warnings == []
    first, second = _assignments(collection)
    assert first["properties"]["letter"] == "U01 Elm Street"
    assert second["properties"]["letter"] == "U02 Oak Lane"
    for feature in (first, second):
        props = feature["properties"]
        assert props["class"] == "Assignment" and props["status"] == "DRAFT"
        assert props["resourceType"] == "GROUND" and props["title"] == props["letter"]
        assert "urban-abc123" in props["description"] and "12 lot(s)" in props["description"]
    assert labels == [{"label": "U01 Elm Street", "candidate_id": "urban-abc123.1"},
                      {"label": "U02 Oak Lane", "candidate_id": "urban-abc123.2"}]


def test_a_candidate_through_the_centre_is_described_without_a_bearing():
    """hasty_candidates gives a line through the centre - a place last seen
    at a trailhead - a null bearing, since it has no direction from there.
    Formatting the null raised, and the whole set failed to export."""
    line = LineString([(LON, LAT), (LON + D, LAT + D)])
    collection, _labels, _ = caltopo_collection([_candidate(
        1, line, name="Spring Ridge Trail",
        facts={"length_m": 150.0, "distance_from_center_m": 0.0,
               "bearing_from_center_deg": None})])
    (feature,) = _assignments(collection)
    assert "150 m long. 0 m from the centre." in feature["properties"]["description"]


def test_a_multipolygon_becomes_one_assignment_per_part():
    parts = MultiPolygon([_square(size=2 * D), _square(dx=0.005)])
    collection, labels, _ = caltopo_collection([_candidate(1, parts)])
    assert [label["label"] for label in labels] == ["U01a Elm Street", "U01b Elm Street"]
    assert all(f["geometry"]["type"] == "Polygon" for f in _assignments(collection))
    big, small = (shape(f["geometry"]) for f in _assignments(collection))
    assert big.area > small.area                     # largest part is "a"


def test_a_hole_is_cut_open_into_one_ring_covering_the_same_ground():
    outer = _square(size=4 * D)
    hole = _square(dx=D, dy=D, size=D)
    donut = Polygon(outer.exterior, [hole.exterior])
    collection, _labels, warnings = caltopo_collection([_candidate(1, donut)])
    (feature,) = _assignments(collection)
    coordinates = feature["geometry"]["coordinates"]
    assert len(coordinates) == 1                     # one ring, CalTopo keeps it
    assert warnings == []


def test_a_line_candidate_becomes_a_line_assignment():
    line = LineString([(LON, LAT), (LON + D, LAT + D)])
    collection, labels, _ = caltopo_collection(
        [_candidate(1, line, name="Spring Ridge Trail", facts={"length_m": 150.0})])
    (feature,) = _assignments(collection)
    assert feature["geometry"]["type"] == "LineString"
    assert "150 m long" in feature["properties"]["description"]


def test_a_point_is_left_out_with_a_warning():
    collection, labels, warnings = caltopo_collection([_candidate(1, Point(LON, LAT))])
    assert _assignments(collection) == [] and labels == []
    assert "Point" in warnings[0]


def test_markers_go_in_only_when_asked():
    marker = {"type": "Feature", "geometry": {"type": "Point", "coordinates": [LON, LAT]},
              "properties": {"class": "Marker", "marker-symbol": "hut",
                             "marker-color": "#FFFFFF"}}
    with_markers, _, _ = caltopo_collection([_candidate(1, _square(), markers=[marker])])
    without, _, _ = caltopo_collection([_candidate(1, _square(), markers=[marker])],
                                       include_markers=False)
    assert sum(f["properties"]["class"] == "Marker" for f in with_markers["features"]) == 1
    assert not any(f["properties"]["class"] == "Marker" for f in without["features"])


def test_the_search_area_goes_in_as_a_shape_with_its_centre():
    area = {"kind": "circle", "center": [LAT, LON], "radius_m": 804.672,
            "geometry": mapping(Point(LON, LAT).buffer(0.008))}
    collection, _, _ = caltopo_collection([_candidate(1, _square())], area=area)
    shapes = [f for f in collection["features"] if f["properties"]["class"] == "Shape"]
    centres = [f for f in collection["features"] if f["properties"]["class"] == "Marker"]
    assert len(shapes) == 1 and shapes[0]["geometry"]["type"] == "LineString"
    assert "0.50 mi" in shapes[0]["properties"]["title"]
    assert centres[0]["geometry"]["coordinates"] == [LON, LAT]


def test_every_geometry_is_a_valid_geojson_type():
    """shapely maps a ring as "LinearRing", which no GeoJSON reader takes."""
    area = {"kind": "circle", "center": [LAT, LON], "radius_m": 500.0,
            "geometry": mapping(Point(LON, LAT).buffer(0.005))}
    donut = Polygon(_square(size=4 * D).exterior, [_square(dx=D, dy=D).exterior])
    collection, _, _ = caltopo_collection(
        [_candidate(1, donut), _candidate(2, LineString([(LON, LAT), (LON + D, LAT)]))],
        area=area)
    valid = {"Point", "LineString", "Polygon", "MultiPoint", "MultiLineString",
             "MultiPolygon"}
    assert {f["geometry"]["type"] for f in collection["features"]} <= valid
    json.dumps(collection)                            # plain JSON throughout


def test_labels_widen_past_99():
    candidates = [_candidate(i, _square(dx=i * 0.002)) for i in range(1, 101)]
    _, labels, _ = caltopo_collection(candidates)
    assert labels[0]["label"].startswith("U001 ") and labels[-1]["label"].startswith("U100 ")


# --------------------------------------------------------------------------
# export_set: the file beside the staged ground


def _staged_set(tmp_path, directory=None):
    directory = directory or tmp_path / "aoi"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "subject.json").write_text(json.dumps({
        "kind": "circle", "center": [LAT, LON], "radius_m": 500.0, "key": "aoi-x",
        "extent_m": 1000.0, "geometry": mapping(Point(LON, LAT).buffer(0.005))}),
        encoding="utf-8")
    candidates = [Candidate(**{k: v for k, v in _candidate(i, _square(dx=i * 0.002)).items()})
                  for i in (1, 2, 3)]
    cset = CandidateSet(set_id="urban-abc123", tool="urban_assignments", params={},
                        handle="aoi-x", created_at="2026-10-20T00:00:00+00:00",
                        candidates=candidates, summary={}, assumptions={}, warnings=[])
    save_set(directory, cset)
    return directory


def test_export_set_writes_the_file_beside_the_staged_ground(tmp_path):
    directory = _staged_set(tmp_path)
    result = export_set(directory, "urban-abc123")
    path = directory / "exports" / "urban-abc123.geojson"
    assert result["path"] == str(path) and path.is_file()
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["type"] == "FeatureCollection"
    assert result["assignments"] == 3 and result["area_included"] is True
    assert result["features"] == {"Assignment": 3, "Shape": 1, "Marker": 1}


def test_export_set_takes_chosen_candidates_in_the_order_given(tmp_path):
    directory = _staged_set(tmp_path)
    result = export_set(directory, "urban-abc123",
                        candidate_ids=["urban-abc123.3", "urban-abc123.1"], include_area=False)
    assert [label["candidate_id"] for label in result["labels"]] == [
        "urban-abc123.3", "urban-abc123.1"]
    assert result["features"] == {"Assignment": 2}


@pytest.mark.parametrize("prefix", ["", "TOOLONGPREFIX", "U 1", "U/"])
def test_a_bad_label_prefix_is_refused(tmp_path, prefix):
    directory = _staged_set(tmp_path)
    with pytest.raises(ValueError, match="label_prefix"):
        export_set(directory, "urban-abc123", label_prefix=prefix)


def test_an_unknown_set_is_refused(tmp_path):
    directory = _staged_set(tmp_path)
    with pytest.raises(KeyError):
        export_set(directory, "urban-ffffff")


# --------------------------------------------------------------------------
# caltopo_export: the component, on an area handle


HANDLE = "aoi-0123456789abcdef"


def _staged_area(tmp_path):
    """`_staged_set`'s ground as prepare_area leaves an area handle - with a
    manifest and no map, so it resolves with no network - and a server over
    its cache. A second set, urban-def456 (two Oak Lane assignments), is
    saved beside urban-abc123."""
    cache = tmp_path / "cache"
    directory = cache / "staged" / HANDLE
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps({"map_id": ""}), encoding="utf-8")
    _staged_set(tmp_path, directory)
    other = [Candidate(**_candidate(i, _square(dy=i * 0.002), name="Oak Lane",
                                    set_id="urban-def456")) for i in (1, 2)]
    save_set(directory, CandidateSet(set_id="urban-def456", tool="urban_assignments",
                                     params={}, handle=HANDLE,
                                     created_at="2026-10-20T00:00:00+00:00",
                                     candidates=other, summary={}, assumptions={},
                                     warnings=[]))
    return ServerContext.build(CaltopoConfig(cache_dir=cache)), directory


def test_caltopo_export_takes_the_set_from_the_candidate_ids(tmp_path):
    ctx, directory = _staged_area(tmp_path)
    result = caltopo_export(ctx, HANDLE, candidate_ids=["urban-def456.2"], include_area=False)
    assert result["set_id"] == "urban-def456"
    assert result["path"] == str(directory / "exports" / "urban-def456.geojson")
    assert result["labels"] == [{"label": "U01 Oak Lane", "candidate_id": "urban-def456.2"}]
    assert not (directory / "exports" / "urban-abc123.geojson").exists()


def test_caltopo_export_refuses_candidate_ids_from_two_sets_and_writes_nothing(tmp_path):
    """A candidate is read from its set by index alone: without this
    refusal urban-def456.1 would be drafted as urban-abc123's first
    candidate, Elm Street, under urban-abc123's labels."""
    ctx, directory = _staged_area(tmp_path)
    for ids in (["urban-abc123.1", "urban-def456.1"], ["urban-def456.2", "urban-abc123.3"]):
        with pytest.raises(ValueError, match=r"^candidate_ids must all come from one set; "
                                             r"export each set on its own$"):
            caltopo_export(ctx, HANDLE, candidate_ids=ids)
    assert not (directory / "exports").exists()


@pytest.mark.parametrize("given", [
    {"set_id": "urban-abc123", "candidate_ids": ["urban-abc123.1"]},
    {},
    {"candidate_ids": []},
])
def test_caltopo_export_takes_exactly_one_of_set_id_or_candidate_ids(tmp_path, given):
    ctx, directory = _staged_area(tmp_path)
    with pytest.raises(ValueError, match="give exactly one of set_id or candidate_ids"):
        caltopo_export(ctx, HANDLE, **given)
    assert not (directory / "exports").exists()


@pytest.mark.xfail(strict=True, reason="PR #4 review B46: caltopo_export drafts a repeated "
                                       "candidate id twice, under two labels")
@pytest.mark.parametrize("ids, repeated", [
    (["urban-abc123.1", "urban-abc123.1"], "urban-abc123.1"),
    (["urban-abc123.3", "urban-abc123.2", "urban-abc123.3"], "urban-abc123.3"),
])
def test_caltopo_export_refuses_a_repeated_candidate_id(tmp_path, ids, repeated):
    """One lot of ground drafted twice - U01 Elm Street and U02 Elm Street,
    the same polygon, both DRAFT - can be issued to two teams. A repeat is
    refused, naming it, before any file is written."""
    ctx, directory = _staged_area(tmp_path)
    with pytest.raises(ValueError, match=re.escape(repeated)):
        caltopo_export(ctx, HANDLE, candidate_ids=ids)
    assert not (directory / "exports").exists()


# --- One folder per file ------------------------------------------------------

def test_every_feature_goes_in_one_new_folder():
    from sar_geo_mcp.planning.export import in_one_folder

    features = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [0, 0]},
                 "properties": {"class": "Marker", "title": "PLS"}}]
    grouped = in_one_folder(features, "SAR Geo Medevac 2026-01-01 10:00")
    folder, member = grouped
    assert folder["properties"] == {"class": "Folder", "visible": True, "labelVisible": True,
                                    "title": "SAR Geo Medevac 2026-01-01 10:00"}
    assert folder["geometry"] is None and folder["id"]
    assert member["properties"]["folderId"] == folder["id"]
    assert "folderId" not in features[0]["properties"], "the caller's features are not changed"
    assert in_one_folder(features, "x")[0]["id"] != folder["id"], "each file gets its own folder"


@pytest.mark.parametrize("properties", [
    {"class": "Folder", "title": "Existing"},
    {"class": "Marker", "title": "Filed", "folderId": "abc"},
])
def test_a_folder_inside_a_folder_is_refused(properties):
    """CalTopo has no folder inside a folder: a file that would need one is
    an error to look at, not something to flatten quietly."""
    from sar_geo_mcp.planning.export import in_one_folder

    with pytest.raises(ValueError, match="no folder inside a folder"):
        in_one_folder([{"type": "Feature", "geometry": None, "properties": properties}], "x")


def test_a_folder_title_says_what_and_when():
    import datetime as dt

    from sar_geo_mcp.planning.export import folder_title

    assert folder_title("Urban search", dt.datetime(2026, 1, 2, 3, 4)) == \
        "SAR Geo Urban search 2026-01-02 03:04"
