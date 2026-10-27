"""Hasty candidates: every trail and road as a line, every drainage, ridge and
waterway as a thin corridor, measured and ordered by distance from the centre."""

from __future__ import annotations

import json

import numpy as np
import pytest
from test_boundary import GROUND_PER_UNIT, ORIGIN_X, ORIGIN_Y, m, polygon, route, staged, valley_dem

from sar_geo_mcp.planning.boundary import load_ground
from sar_geo_mcp.planning.candidates import Candidate, CandidateSet
from sar_geo_mcp.planning.corridors import ALL_INCLUDE, DEFAULT_INCLUDE, hasty_candidates
from sar_geo_mcp.planning.norms import load_norms
from sar_geo_mcp.geo import M2_PER_ACRE


def area_subject(data_dir, center_units=(200, 200), radius_units=150):
    """Write an area subject.json (a circle in Mercator terms) and return it as
    the dict the generator takes."""
    lon, lat = m(*center_units)
    from shapely.geometry import Point, mapping
    circle = mapping(Point(ORIGIN_X + center_units[0], ORIGIN_Y - center_units[1]).buffer(radius_units))
    # Back to WGS84 for the subject file.
    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely.ops import transform
    to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    geometry = mapping(transform(to_wgs84.transform, shape(circle)))
    subject = {"kind": "circle", "center": [lat, lon], "radius_m": radius_units * GROUND_PER_UNIT,
               "geometry": json.loads(json.dumps(geometry)), "map_id": None,
               "key": "aoi-0123456789abcdef", "extent_m": 2 * radius_units * GROUND_PER_UNIT}
    (data_dir / "subject.json").write_text(json.dumps(subject), encoding="utf-8")
    return subject


def u(x_units, y_units):
    """Mercator offsets from the raster origin -> a Web Mercator (x, y)."""
    return (ORIGIN_X + x_units, ORIGIN_Y - y_units)


def write_subject(data_dir, geometry_3857, center_units=None, kind="polygon"):
    """Write a subject.json for an arbitrary Web Mercator geometry.

    The centre defaults to the geometry's own representative point, which is
    inside it even for a donut or a MultiPolygon.
    """
    from pyproj import Transformer
    from shapely.geometry import mapping
    from shapely.ops import transform

    to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    geometry = json.loads(json.dumps(mapping(transform(to_wgs84.transform, geometry_3857))))
    if center_units is None:
        point = geometry_3857.representative_point()
        lon, lat = to_wgs84.transform(point.x, point.y)
    else:
        lon, lat = m(*center_units)
    subject = {"kind": kind, "center": [lat, lon], "geometry": geometry, "map_id": None,
               "key": "aoi-0123456789abcdef", "extent_m": 400 * GROUND_PER_UNIT}
    (data_dir / "subject.json").write_text(json.dumps(subject), encoding="utf-8")
    return subject


def generate(data_dir, **kwargs):
    subject = json.loads((data_dir / "subject.json").read_text(encoding="utf-8"))
    return hasty_candidates(load_ground(data_dir), subject, norms=load_norms(), **kwargs)


# --- lines ------------------------------------------------------------------------

def test_named_ways_of_one_class_merge_into_one_run(tmp_path):
    west = route("route_0", "TRAIL", [(0, 200), (200, 200)], name="Ridge Trail")
    east = route("route_1", "TRAIL", [(200, 200), (400, 200)], name="Ridge Trail")
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[west, east])
    area_subject(data_dir)
    cset = generate(data_dir, include=["trail"])
    lines = [c for c in cset.candidates if c.kind == "line"]
    assert len(lines) == 1
    trail = lines[0]
    assert trail.name == "Ridge Trail" and trail.axis == "trail"
    assert set(trail.axis_ids) == {"route_0", "route_1"}
    # The whole way, to the edge of the staged ground - not the 300 units the
    # circle used to leave of it.
    assert trail.facts["length_m"] == pytest.approx(400 * GROUND_PER_UNIT, rel=0.05)
    assert trail.geometry["type"] == "LineString"


def test_two_crossing_unnamed_ways_are_two_runs_not_four_stubs(tmp_path):
    """The corpus rule: a hasty line runs a trail END TO END, never split at a
    junction. Noding the two ways against each other before merging - what a
    `unary_union` does - cuts each in half at the crossing and reports four
    stubs where a planner sees two trails."""
    east_west = route("route_0", "TRAIL", [(0, 200), (400, 200)])
    north_south = route("route_1", "TRAIL", [(300, 100), (300, 300)])
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[east_west, north_south])
    area_subject(data_dir)                      # circle, centre (200, 200), radius 150

    cset = generate(data_dir, include=["trail"])

    lines = [c for c in cset.candidates if c.kind == "line"]
    assert len(lines) == 2
    # Both run to the edge of the staged ground: the E-W way spans the raster
    # (400 units) and the N-S way is its own 100 to 300 span (200 units).
    lengths = sorted(c.facts["length_m"] for c in lines)
    assert lengths[1] == pytest.approx(400 * GROUND_PER_UNIT, rel=0.1)
    assert lengths[0] == pytest.approx(200 * GROUND_PER_UNIT, rel=0.1)


def test_each_crossing_run_is_attributed_to_its_own_way(tmp_path):
    """Provenance survives the merge: a way that merely crosses a run is not
    one of the ways the run is made of."""
    east_west = route("route_0", "TRAIL", [(0, 200), (400, 200)])
    north_south = route("route_1", "TRAIL", [(300, 100), (300, 300)])
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[east_west, north_south])
    area_subject(data_dir)

    cset = generate(data_dir, include=["trail"])

    by_length = sorted(cset.candidates, key=lambda c: -c.facts["length_m"])
    assert by_length[0].axis_ids == ["route_0"]     # the 300-unit E-W way
    assert by_length[1].axis_ids == ["route_1"]     # the 200-unit N-S way


def test_two_named_ways_meeting_at_a_point_stay_separate(tmp_path):
    """Alpha and Beta touch, but a run is one NAMED way end to end; the merge
    is per name, so touching never fuses two names into one candidate."""
    alpha = route("route_0", "TRAIL", [(0, 200), (200, 200)], name="Alpha")
    beta = route("route_1", "TRAIL", [(200, 100), (200, 300)], name="Beta")
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[alpha, beta])
    area_subject(data_dir)

    cset = generate(data_dir, include=["trail"])

    assert sorted(c.name for c in cset.candidates) == ["Alpha", "Beta"]
    assert {c.name: c.axis_ids for c in cset.candidates} == {"Alpha": ["route_0"],
                                                             "Beta": ["route_1"]}


def test_a_gap_bridged_between_two_ways_is_flagged_on_the_candidate(tmp_path):
    """Two stretches of one named trail with a 5-unit (4 m) gap are one run,
    and the candidate says the gap was joined rather than hiding it."""
    west = route("route_0", "TRAIL", [(0, 200), (200, 200)], name="Ridge Trail")
    east = route("route_1", "TRAIL", [(205, 200), (400, 200)], name="Ridge Trail")
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[west, east])
    area_subject(data_dir)

    cset = generate(data_dir, include=["trail"])

    assert len(cset.candidates) == 1
    assert "joined a gap of 4 m between ways" in cset.candidates[0].warnings


def test_unnamed_ways_of_different_classes_stay_separate(tmp_path):
    trail = route("route_0", "TRAIL", [(0, 200), (200, 200)])
    road = route("route_1", "ROAD", [(200, 200), (400, 200)])
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[trail, road])
    area_subject(data_dir)
    cset = generate(data_dir)
    kinds = sorted((c.axis, c.kind) for c in cset.candidates)
    assert kinds == [("road", "line"), ("trail", "line")]


def test_include_filters_and_road_covers_tracks(tmp_path):
    track = route("route_0", "TRACK", [(0, 200), (400, 200)])
    creek = route("route_1", "WATERWAY", [(200, 0), (200, 400)])
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[track, creek])
    area_subject(data_dir)
    roads_only = generate(data_dir, include=["road"])
    assert [c.axis for c in roads_only.candidates] == ["road"]
    creeks_only = generate(data_dir, include=["waterway"])
    assert [c.axis for c in creeks_only.candidates] == ["waterway"]
    assert creeks_only.candidates[0].kind == "corridor"
    with pytest.raises(ValueError, match="include"):
        generate(data_dir, include=["fence"])



def test_an_edge_warning_for_an_axis_not_asked_for_is_dropped(tmp_path):
    """The ground warns when a drainage or ridgeline runs off the raster; that
    speaks of those axes' ranking alone, and the hasty list holds neither, so
    every set drops both and keeps every other ground warning."""
    import dataclasses

    from sar_geo_mcp.planning.boundary import EDGE_WARNINGS

    track = route("route_0", "TRACK", [(0, 200), (400, 200)])
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[track])
    subject = area_subject(data_dir)
    other = "evt unavailable: X"
    ground = dataclasses.replace(load_ground(data_dir), warnings=[
        EDGE_WARNINGS["drainage"], EDGE_WARNINGS["ridge"], other])

    def warnings_for(include):
        return hasty_candidates(ground, subject, norms=load_norms(), include=include).warnings

    for include in (None, ["trail", "road"], list(ALL_INCLUDE)):
        warnings = warnings_for(include)
        assert other in warnings
        assert not set(EDGE_WARNINGS.values()) & set(warnings)

def creek(name="Test Creek"):
    """A mapped creek down the valley floor, the whole height of the raster."""
    return route("route_c", "WATERWAY", [(205, 0), (205, 400)], name=name)


# --- a named way is one candidate, drawn to its ends -------------------------------

def y_trail(name="Anniversary Trail"):
    """Three ways of one name meeting at (200, 150): a Y with 90-unit arms.

    `linemerge` chains lines only through a node where exactly TWO ends meet,
    so a three-way junction chops this into three runs - which is the defect.
    Every arm is well inside the test circle, so nothing here is about clipping.
    """
    return [route("route_0", "TRAIL", [(200, 150), (200, 60)], name=name),
            route("route_1", "TRAIL", [(200, 150), (110, 150)], name=name),
            route("route_2", "TRAIL", [(200, 150), (290, 150)], name=name)]


def test_a_y_shaped_named_trail_is_one_candidate_of_all_three_branches(tmp_path):
    """The operator's ruling: a named way inside the staged area is ONE
    candidate, drawn to its ends. A fork in it is acceptable - that is what a
    planner would draw by hand - so the three arms are one assignment whose
    length is their sum, not three candidates of a third of it each."""
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=y_trail())
    area_subject(data_dir)

    cset = generate(data_dir, include=["trail"])

    lines = [c for c in cset.candidates if c.kind == "line"]
    assert len(lines) == 1
    trail = lines[0]
    assert trail.name == "Anniversary Trail"
    assert sorted(trail.axis_ids) == ["route_0", "route_1", "route_2"]
    assert trail.facts["branches"] == 3
    assert trail.facts["length_m"] == pytest.approx(270 * GROUND_PER_UNIT, rel=0.02)
    assert trail.geometry["type"] == "MultiLineString"
    assert len(trail.geometry["coordinates"]) == 3


def test_a_forked_candidates_facts_are_all_measured(tmp_path):
    """Everything downstream of the axis - the elevation profile, the effort
    rate, the walking time from the centre - has to walk a multi-part axis.
    A None here would mean a planner is handed a fork with no numbers on it."""
    data_dir = staged(tmp_path, valley_dem(), routes=y_trail())
    area_subject(data_dir)

    facts = generate(data_dir, include=["trail"]).candidates[0].facts

    assert facts["branches"] == 3
    assert facts["slope_mean_deg"] is not None and facts["slope_mean_deg"] >= 0
    assert facts["gain_m"] is not None and facts["loss_m"] is not None
    assert facts["effort"]["hours_one_way"] > 0
    assert facts["travel_minutes_from_center"] is not None
    assert facts["travel_minutes_from_center"] >= 0
    assert facts["distance_from_center_m"] == pytest.approx(50 * GROUND_PER_UNIT, rel=0.05)


def test_the_length_floor_judges_the_whole_trail_not_one_branch(tmp_path):
    """The point of the fix. Every arm of the Y is 72 m; the trail is 215 m.
    A floor between the two used to return NOTHING - a trail a planner would
    absolutely assign, invisible because it was measured a fragment at a time."""
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=y_trail())
    area_subject(data_dir)

    cset = generate(data_dir, include=["trail"], min_length_m=100.0)

    assert cset.assumptions["min_candidate_m"] == 100.0
    assert len(cset.candidates) == 1
    assert cset.candidates[0].facts["length_m"] == pytest.approx(270 * GROUND_PER_UNIT, rel=0.02)
    # Each arm on its own is under the floor, which is what made it disappear.
    assert 90 * GROUND_PER_UNIT < 100.0


def test_a_closed_loop_with_a_spur_is_one_candidate(tmp_path):
    """A loop meets itself, and the spur meets it at a T. Neither junction is
    a two-end node, so `linemerge` leaves them as separate runs; they are one
    named trail and one assignment."""
    loop = route("route_0", "TRAIL",
                 [(150, 150), (250, 150), (250, 250), (150, 250), (150, 150)],
                 name="Loop Trail")
    spur = route("route_1", "TRAIL", [(200, 250), (200, 320)], name="Loop Trail")
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[loop, spur])
    area_subject(data_dir)

    cset = generate(data_dir, include=["trail"])

    assert len(cset.candidates) == 1
    candidate = cset.candidates[0]
    assert candidate.facts["length_m"] == pytest.approx(470 * GROUND_PER_UNIT, rel=0.02)
    assert candidate.facts["branches"] == 2
    assert sorted(candidate.axis_ids) == ["route_0", "route_1"]


def test_two_ways_sharing_a_name_but_not_touching_stay_two_candidates(tmp_path):
    """One name is not one candidate: the rule is one candidate per CONNECTED
    component of a name. Two stretches of `Skyline Trail` 130 units apart are
    two assignments, and uniting them would invent a line nobody can walk."""
    north = route("route_0", "TRAIL", [(60, 120), (340, 120)], name="Skyline Trail")
    south = route("route_1", "TRAIL", [(60, 280), (340, 280)], name="Skyline Trail")
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[north, south])
    area_subject(data_dir)

    cset = generate(data_dir, include=["trail"])

    assert len(cset.candidates) == 2
    assert [c.name for c in cset.candidates] == ["Skyline Trail", "Skyline Trail"]
    assert sorted(c.axis_ids for c in cset.candidates) == [["route_0"], ["route_1"]]
    for candidate in cset.candidates:
        assert candidate.facts["branches"] == 1
        assert candidate.geometry["type"] == "LineString"


def test_unnamed_ways_that_touch_at_a_t_stay_two_candidates(tmp_path):
    """Unnamed ways keep the old merge: only where one way's END meets
    another's. There is no name to justify uniting two unrelated tracks, so a
    spur off an unnamed track is its own candidate."""
    track = route("route_0", "TRAIL", [(60, 200), (340, 200)])
    spur = route("route_1", "TRAIL", [(200, 200), (200, 320)])
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[track, spur])
    area_subject(data_dir)

    cset = generate(data_dir, include=["trail"])

    assert len(cset.candidates) == 2
    assert all(c.name is None for c in cset.candidates)
    assert sorted(c.axis_ids for c in cset.candidates) == [["route_0"], ["route_1"]]
    for candidate in cset.candidates:
        assert candidate.facts["branches"] == 1


def test_the_branch_warning_names_what_a_fork_means_for_a_team(tmp_path):
    """A planner must not be handed a fork without being told: an assignment
    in two branches is not one walk end to end."""
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=y_trail())
    area_subject(data_dir)
    forked = generate(data_dir, include=["trail"]).candidates[0]

    assert forked.facts["branches"] == 3
    assert any("not a single walk end to end" in w for w in forked.warnings)
    assert any("3 branches" in w for w in forked.warnings)


def test_a_simple_line_carries_one_branch_and_no_warning(tmp_path):
    straight = route("route_0", "TRAIL", [(60, 200), (340, 200)], name="Straight Trail")
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[straight])
    area_subject(data_dir)

    candidate = generate(data_dir, include=["trail"]).candidates[0]

    assert candidate.facts["branches"] == 1
    assert candidate.geometry["type"] == "LineString"
    assert not any("branches" in w for w in candidate.warnings)


def test_a_corridor_axis_carries_no_branch_count(tmp_path):
    """A waterway axis is one axis, and `branches` is a fact about a LINE
    candidate."""
    data_dir = staged(tmp_path, valley_dem(), routes=[creek()])
    area_subject(data_dir)

    corridor = generate(data_dir, include=["waterway"]).candidates[0]

    assert corridor.kind == "corridor"
    assert "branches" not in corridor.facts


# --- corridors ----------------------------------------------------------------------

def test_a_waterway_becomes_a_corridor_of_the_norms_width(tmp_path):
    data_dir = staged(tmp_path, valley_dem(), routes=[creek()])
    area_subject(data_dir)
    cset = generate(data_dir, include=["waterway"])
    assert cset.candidates, cset.warnings
    corridor = cset.candidates[0]
    assert corridor.kind == "corridor" and corridor.axis == "waterway"
    assert corridor.name == "Test Creek" and corridor.sources == ["waterway"]
    assert corridor.axis_ids == ["route_c"]
    assert corridor.facts["width_m"] == 30.0
    assert corridor.geometry["type"] == "Polygon"
    assert corridor.facts["area_acres"] == pytest.approx(corridor.facts["length_m"] * 30.0 / M2_PER_ACRE, rel=0.15)


def test_drainages_and_ridges_are_not_offered(tmp_path):
    """The elevation model's drainages and ridges came back as hundreds of
    short, unstable pieces; the hasty list no longer offers them, and naming
    one is refused rather than answered with nothing."""
    data_dir = staged(tmp_path, valley_dem())
    area_subject(data_dir)
    assert ALL_INCLUDE == ("trail", "road", "waterway")
    for axis in ("drainage", "ridge"):
        with pytest.raises(ValueError, match="unknown include"):
            generate(data_dir, include=[axis])
        with pytest.raises(ValueError, match="unknown corridor width"):
            generate(data_dir, corridor_widths_m={axis: 60})
    # The valley holds a drainage, and no candidate is built on it.
    assert not generate(data_dir, include=list(ALL_INCLUDE)).candidates


def test_corridors_are_listed_only_when_include_names_them(tmp_path):
    """The default list is trails and roads; a waterway corridor comes back
    when `include` names it."""
    data_dir = staged(tmp_path, valley_dem(), routes=[creek()])
    area_subject(data_dir)
    default = generate(data_dir)
    assert default.params["include"] == ["road", "trail"]
    assert not default.candidates
    every = generate(data_dir, include=list(ALL_INCLUDE))
    assert [c.axis for c in every.candidates] == ["waterway"]
    assert default.set_id != every.set_id


def test_corridor_widths_can_be_overridden(tmp_path):
    data_dir = staged(tmp_path, valley_dem(), routes=[creek()])
    area_subject(data_dir)
    cset = generate(data_dir, include=["waterway"], corridor_widths_m={"waterway": 60})
    assert cset.candidates[0].facts["width_m"] == 60.0


def test_a_creek_too_short_to_send_a_team_down_is_not_a_corridor(tmp_path):
    """A 60 m piece of creek inside the area is not a line a team can be
    assigned. Lines keep the 30 m minimum - a mapped trail is a mapped trail -
    but a corridor needs 100 m of axis."""
    from sar_geo_mcp.planning.corridors import MIN_CORRIDOR_AXIS_M

    stub = route("route_s", "WATERWAY", [(150, 200), (225, 200)], name="Stub Creek")  # 60 m
    data_dir = staged(tmp_path, valley_dem(), routes=[stub, creek()])
    area_subject(data_dir)

    cset = generate(data_dir, include=["waterway"])

    assert MIN_CORRIDOR_AXIS_M == 100.0
    assert [c.name for c in cset.candidates] == ["Test Creek"]
    assert cset.assumptions["min_corridor_axis_m"] == 100.0
    assert cset.assumptions["min_candidate_m"] == 30.0


def test_a_short_mapped_trail_is_still_a_line_candidate(tmp_path):
    """The corridor minimum is a corridor's; a 40 m mapped trail is a mapped
    trail, and the corpus's hasty lines are what people actually walk."""
    trail = route("route_0", "TRAIL", [(200, 175), (200, 225)])   # 50 units, 40 m
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[trail])
    area_subject(data_dir)

    cset = generate(data_dir, include=["trail"])

    assert [round(c.facts["length_m"]) for c in cset.candidates] == [40]


def test_a_staged_evt_reaches_the_corridor_effort(tmp_path, monkeypatch):
    """`hasty_candidates` passes the staged evt.tif down to `candidate_effort`,
    so the lifeform source on a corridor is the raster's, not the default."""
    monkeypatch.setattr("sar_geo_mcp.planning.effort.composition",
                        lambda path, geom: {"fractions": {"Shrub": 1.0}, "dominant_lifeform": "Shrub",
                                            "top_types": [], "source": "LANDFIRE test"})
    data_dir = staged(tmp_path, valley_dem(), routes=[creek()])
    (data_dir / "evt.tif").write_bytes(b"")     # exists; composition is stubbed above
    area_subject(data_dir)

    cset = generate(data_dir, include=["waterway"])

    corridor = cset.candidates[0]
    assert corridor.facts["effort"]["assumptions"]["lifeform_source"] == "LANDFIRE test"


# --- a line runs to the edge of the MEASURED GROUND --------------------------------------
#
# The staged raster is the area plus the planning buffer, and it is the ground
# where slope, climb, effort and travel time were actually measured. A team is
# handed a trail, not the arc where a circle happened to cut it, so a LINE
# candidate is offered to the edge of that raster. Corridors are unchanged:
# they are traced FROM the raster and have no meaning past the subject.

def test_a_line_leaving_the_area_is_offered_to_the_edge_of_the_staged_ground(tmp_path):
    """Measured on a real 1,500 m staging: Hamms Gulch Trail is 4,470 m and
    only 3,491 m of it fell inside the circle, so a planner was handed 979 m
    less trail than the trail. Here the trail runs from 20 to 380 units - well
    outside the circle at both ends, well inside the raster - and all 360
    units are the candidate where the circle used to cut it to 300."""
    trail = route("route_0", "TRAIL", [(20, 200), (380, 200)], name="Ridge Trail")
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[trail])
    area_subject(data_dir)                       # circle, centre (200, 200), radius 150

    candidate = generate(data_dir, include=["trail"]).candidates[0]

    assert candidate.facts["length_m"] == pytest.approx(360 * GROUND_PER_UNIT, rel=0.02)
    assert candidate.facts["clipped_at_edge"] is False
    assert candidate.facts["branches"] == 1
    assert not any("terrain data stops" in w for w in candidate.warnings)


def test_a_line_running_past_the_staged_raster_is_cut_there_and_says_so(tmp_path):
    """The buffer is generous, not infinite. A way that continues past the
    measured ground is cut at its edge, and the candidate says what that means:
    the assignment stops where the terrain data stops, not where the trail
    does."""
    trail = route("route_0", "TRAIL", [(100, 200), (600, 200)], name="Long Trail")
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[trail])
    area_subject(data_dir)

    candidate = generate(data_dir, include=["trail"]).candidates[0]

    assert candidate.facts["length_m"] == pytest.approx(300 * GROUND_PER_UNIT, rel=0.02)
    assert candidate.facts["clipped_at_edge"] is True
    assert any("stops where the terrain data stops" in w for w in candidate.warnings), \
        candidate.warnings
    assert any("159 m of it lies beyond the raster" in w for w in candidate.warnings)


def test_a_way_that_ENDS_on_the_raster_edge_is_clipped_at_edge_too(tmp_path):
    """Measured, and the reason this is not a length comparison: the staged
    osm.json is cut to the staging box before a candidate is built from it
    (`terrain.linear.routes_for`), so on a real staging Skyline Boulevard -
    tens of kilometres of road - arrives as a way that stops dead on the
    boundary with nothing left to clip. A team cannot be told whether such a
    way ends there or runs on, and it makes no difference to them: the
    assignment stops where the data does."""
    trail = route("route_0", "TRAIL", [(200, 200), (400, 200)], name="Boundary Trail")
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[trail])   # raster is 0..400
    area_subject(data_dir)

    candidate = generate(data_dir, include=["trail"]).candidates[0]

    assert candidate.facts["length_m"] == pytest.approx(200 * GROUND_PER_UNIT, rel=0.02)
    assert candidate.facts["clipped_at_edge"] is True
    assert any("cannot be seen from here" in w for w in candidate.warnings), \
        candidate.warnings


def test_a_line_wholly_inside_the_area_is_unchanged_and_not_clipped(tmp_path):
    trail = route("route_0", "TRAIL", [(200, 175), (200, 225)])   # 50 units, 40 m
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[trail])
    area_subject(data_dir)

    candidate = generate(data_dir, include=["trail"]).candidates[0]

    assert round(candidate.facts["length_m"]) == 40
    assert candidate.facts["clipped_at_edge"] is False
    assert not any("terrain data stops" in w for w in candidate.warnings)


def test_a_way_that_never_enters_the_area_is_not_a_candidate(tmp_path):
    """The staged ground says how far a candidate may RUN; it is the subject
    area that says whether there is a candidate at all. A trail 30 units north
    of the circle is on measured ground and is still nobody's assignment."""
    north = route("route_0", "TRAIL", [(0, 20), (400, 20)], name="North Rim")
    inside = route("route_1", "TRAIL", [(0, 200), (400, 200)], name="Ridge Trail")
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[north, inside])
    area_subject(data_dir)

    cset = generate(data_dir, include=["trail"])

    assert [c.name for c in cset.candidates] == ["Ridge Trail"]


def test_a_corridor_still_stops_at_the_subject_area(tmp_path):
    """A waterway corridor has no meaning past the subject, so it is clipped
    to the area - and `clipped_at_edge` is a LINE candidate's fact."""
    data_dir = staged(tmp_path, valley_dem(), routes=[creek()])
    area_subject(data_dir)

    corridor = generate(data_dir, include=["waterway"]).candidates[0]

    assert corridor.facts["length_m"] == pytest.approx(300 * GROUND_PER_UNIT, rel=0.1)
    assert 400 * GROUND_PER_UNIT > corridor.facts["length_m"] + 20
    assert "clipped_at_edge" not in corridor.facts


# --- the subject's own shape -------------------------------------------------------------

def test_a_multipolygon_subject_reports_candidates_in_each_part(tmp_path):
    """A staged subject can be a MultiPolygon (two lobes of one search area);
    reading its ring must not assume a single exterior."""
    from shapely.geometry import MultiPolygon, Polygon

    west_square = Polygon([u(20, 180), u(160, 180), u(160, 220), u(20, 220)])
    east_square = Polygon([u(240, 180), u(380, 180), u(380, 220), u(240, 220)])
    west = route("route_0", "TRAIL", [(0, 200), (200, 200)], name="West")
    east = route("route_1", "TRAIL", [(200, 200), (400, 200)], name="East")
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[west, east])
    write_subject(data_dir, MultiPolygon([west_square, east_square]))

    cset = generate(data_dir, include=["trail"])

    assert sorted(c.name for c in cset.candidates) == ["East", "West"]
    for candidate in cset.candidates:
        # Each way runs its own 200 units on the staged ground; what the lobes
        # decide is that each way is a candidate, and where a team gets on.
        assert candidate.facts["length_m"] == pytest.approx(200 * GROUND_PER_UNIT, rel=0.05)
        assert candidate.facts["ingress"]["area_crossings"] == 2


def test_a_hole_in_the_area_counts_towards_the_crossings(tmp_path):
    """`area_crossings` is where a team gets ONTO the candidate, so it counts
    every meeting with the subject's ring: the axis crosses the outer ring
    twice and the edge of the hole twice. Read against `area.exterior` the
    hole edge is invisible and the count is 2, which hides two of the ways a
    team gets on."""
    from shapely.geometry import Polygon

    outer = Polygon([u(50, 50), u(350, 50), u(350, 350), u(50, 350)])
    hole = Polygon([u(150, 150), u(190, 150), u(190, 250), u(150, 250)])
    trail = route("route_0", "TRAIL", [(0, 200), (400, 200)])
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[trail])
    write_subject(data_dir, outer.difference(hole))

    cset = generate(data_dir, include=["trail"])

    assert cset.candidates[0].facts["ingress"]["area_crossings"] == 4


def test_a_run_leaving_and_re_entering_the_staged_ground_keeps_every_piece(tmp_path):
    """A trail that loops north out of the staged raster and comes back is cut
    into two 250-unit pieces. BOTH are the candidate - a way is one candidate,
    drawn to its ends - so `length_m` is their sum, `branches` is 2, and the
    candidate says both that it is in two branches and why."""
    trail = route("route_0", "TRAIL", [(100, 250), (100, -60), (300, -60), (300, 250)])
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[trail])
    area_subject(data_dir)                       # circle, centre (200, 200), radius 150

    cset = generate(data_dir, include=["trail"])

    assert len(cset.candidates) == 1
    candidate = cset.candidates[0]
    assert candidate.facts["length_m"] == pytest.approx(500 * GROUND_PER_UNIT, rel=0.05)
    assert candidate.facts["branches"] == 2
    assert candidate.facts["clipped_at_edge"] is True
    assert candidate.geometry["type"] == "MultiLineString"
    assert any("leaves and re-enters the staged ground" in w for w in candidate.warnings)
    assert any("not a single walk end to end" in w for w in candidate.warnings)


def test_a_line_subject_says_its_area_is_the_implicit_corridor(tmp_path):
    """A line subject has no area of its own; `project_geometry_to_3857`
    buffers it to a 50 m corridor either side, and the set says so rather
    than reporting an area nobody asked for."""
    from shapely.geometry import LineString

    from sar_geo_mcp.terrain.grid import CORRIDOR_HALF_WIDTH_M

    trail = route("route_0", "TRAIL", [(0, 200), (400, 200)])
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[trail])
    write_subject(data_dir, LineString([u(100, 200), u(300, 200)]), kind="line")

    cset = generate(data_dir, include=["trail"])

    assert CORRIDOR_HALF_WIDTH_M == 50.0
    assert cset.assumptions["area_note"] == (
        "the subject is a line; the area is the 50 m corridor either side "
        "(terrain.grid.CORRIDOR_HALF_WIDTH_M)")


def test_a_point_subject_is_refused(tmp_path):
    from shapely.geometry import Point

    data_dir = staged(tmp_path, np.zeros((40, 40)))
    write_subject(data_dir, Point(u(200, 200)), kind="point")

    with pytest.raises(ValueError, match="a point subject has no area"):
        generate(data_dir)


def test_a_center_given_as_a_mapping_is_read_the_same_as_a_pair(tmp_path):
    trail = route("route_0", "TRAIL", [(0, 210), (400, 210)])
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[trail])
    subject = area_subject(data_dir)
    latitude, longitude = subject["center"]

    from_pair = generate(data_dir, include=["trail"])
    subject["center"] = {"latitude": latitude, "longitude": longitude}
    from_mapping = hasty_candidates(load_ground(data_dir), subject, norms=load_norms(),
                                    include=["trail"])

    assert from_mapping.assumptions["center"] == from_pair.assumptions["center"]
    assert from_mapping.candidates[0].facts["distance_from_center_m"] == pytest.approx(
        from_pair.candidates[0].facts["distance_from_center_m"])


# --- facts and order --------------------------------------------------------------------

def test_candidates_are_ordered_by_distance_from_the_centre(tmp_path):
    near = route("route_0", "TRAIL", [(0, 210), (400, 210)], name="Near")
    far = route("route_1", "ROAD", [(0, 80), (400, 80)], name="Far")
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[far, near])
    area_subject(data_dir)
    cset = generate(data_dir)
    assert [c.name for c in cset.candidates] == ["Near", "Far"]
    assert cset.candidates[0].facts["distance_from_center_m"] < cset.candidates[1].facts["distance_from_center_m"]
    assert cset.candidates[1].facts["distance_from_center_m"] == pytest.approx(120 * GROUND_PER_UNIT, rel=0.05)
    assert 0 <= cset.candidates[1].facts["bearing_from_center_deg"] < 360
    assert set(cset.candidates[1].facts["nearest_point"]) == {"latitude", "longitude"}


def test_max_distance_drops_far_candidates(tmp_path):
    near = route("route_0", "TRAIL", [(0, 210), (400, 210)])
    far = route("route_1", "ROAD", [(0, 80), (400, 80)])
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[far, near])
    area_subject(data_dir)
    cset = generate(data_dir, max_distance_m=50)
    assert [c.axis for c in cset.candidates] == ["trail"]
    assert cset.summary["dropped_beyond_max_distance"] == 1


def test_facts_carry_slope_gain_loss_ingress_and_effort(tmp_path):
    trail = route("route_0", "TRAIL", [(0, 200), (400, 200)])
    spur = route("route_1", "ROAD", [(300, 100), (300, 300)])
    data_dir = staged(tmp_path, valley_dem(), routes=[trail, spur])
    area_subject(data_dir)
    cset = generate(data_dir, include=["trail"])
    trail_c = cset.candidates[0]
    facts = trail_c.facts
    assert facts["slope_mean_deg"] > 10.0                 # across the valley sidewalls
    assert facts["gain_m"] > 0 and facts["loss_m"] > 0     # down into the valley and up again
    assert facts["ingress"]["junctions"] == 1              # the spur road
    assert facts["ingress"]["area_crossings"] == 2         # enters and leaves the circle
    assert facts["effort"]["hours_one_way"] > 0


def test_gain_and_loss_never_bridge_a_dem_void(tmp_path):
    """Two benches 100 m apart in height with a void between them. Dropping
    the NaN elevations BEFORE differencing joins the benches and reports a
    100 m climb nobody can measure; differencing first leaves the deltas that
    span the void as NaN, and those are what get dropped."""
    dem = np.zeros((40, 40))
    dem[:, 25:] = 100.0
    dem[:, 20:25] = -999999.0            # the void between the benches
    trail = route("route_0", "TRAIL", [(0, 200), (400, 200)])
    data_dir = staged(tmp_path, dem, routes=[trail])
    area_subject(data_dir)

    facts = generate(data_dir, include=["trail"]).candidates[0].facts

    assert facts["gain_m"] == 0.0
    assert facts["loss_m"] == 0.0


def test_an_axis_over_a_void_reports_no_slope_and_no_climb_rather_than_zero(tmp_path):
    """Every sample is in the void: nothing was measured. 0.0 would read as
    flat ground with no ascent - a fact about the terrain nobody established -
    and `nanmean` over an all-NaN sample is a RuntimeWarning and a NaN, which
    is not JSON either."""
    import warnings as warnings_module

    dem = np.zeros((40, 40))
    dem[15:25, :] = -999999.0            # a void band across the whole line
    trail = route("route_0", "TRAIL", [(0, 200), (400, 200)])
    data_dir = staged(tmp_path, dem, routes=[trail])
    area_subject(data_dir)

    with warnings_module.catch_warnings():
        warnings_module.simplefilter("error", RuntimeWarning)
        facts = generate(data_dir, include=["trail"]).candidates[0].facts

    assert facts["slope_mean_deg"] is None
    assert facts["gain_m"] is None and facts["loss_m"] is None
    # The line is still a line: its length and its effort are measured.
    assert facts["length_m"] > 0 and facts["effort"]["hours_one_way"] > 0


def test_overlap_with_existing_assignments_is_reported(tmp_path):
    trail = route("route_0", "TRAIL", [(0, 200), (400, 200)])
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[trail])
    area_subject(data_dir)
    existing = polygon((50, 150), (250, 150), (250, 250), (50, 250))
    features = [{"id": "a1", "properties": {"class": "Assignment", "letter": "AB", "number": "104"},
                 "geometry": existing}]
    cset = generate(data_dir, map_features=features)
    facts = cset.candidates[0].facts
    # 200 of the candidate's 400 units lie under the existing assignment; the
    # candidate is the whole way on staged ground, so that is half of it.
    assert facts["overlap_existing_pct"] == pytest.approx(200 / 400 * 100, abs=8)
    assert facts["overlap_labels"] == ["AB104"]
    assert facts["overlap_reason"] is None      # measured, so no reason is owed
    assert cset.summary["existing_assignments"] == 1


def test_without_a_map_overlap_is_unknown_and_says_why(tmp_path):
    """0.0 would be indistinguishable from a real zero - "measured, and no
    existing assignment touches this line" - which is exactly the fact a
    planner deciding whether a line is already somebody's job needs. So it is
    None, with the reason on the candidate and once on the set."""
    trail = route("route_0", "TRAIL", [(0, 200), (400, 200)])
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[trail])
    area_subject(data_dir)

    cset = generate(data_dir, include=["trail"], map_features=None)

    facts = cset.candidates[0].facts
    assert facts["overlap_existing_pct"] is None
    assert facts["overlap_labels"] == []
    assert facts["overlap_reason"] == (
        "no map was staged with this area, or the map could not be read, so "
        "overlap was not computed")
    assert cset.warnings.count(facts["overlap_reason"]) == 1


def test_a_map_with_no_assignments_is_a_measured_zero_overlap(tmp_path):
    """The other side of the same coin: the map WAS read and carries nothing
    this line touches. That is a measurement, so it is 0.0 and owes no reason."""
    trail = route("route_0", "TRAIL", [(0, 200), (400, 200)])
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[trail])
    area_subject(data_dir)

    cset = generate(data_dir, include=["trail"], map_features=[])

    facts = cset.candidates[0].facts
    assert facts["overlap_existing_pct"] == 0.0
    assert facts["overlap_reason"] is None
    assert not any("overlap was not computed" in w for w in cset.warnings)


def test_an_existing_assignment_that_will_not_project_is_named_in_a_warning(tmp_path):
    """A malformed assignment geometry is a fact about the map, not something
    to swallow: the overlap figures are computed WITHOUT it, so the set has to
    say which labels are missing from them."""
    trail = route("route_0", "TRAIL", [(0, 200), (400, 200)])
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[trail])
    area_subject(data_dir)
    features = [{"id": "a1", "properties": {"class": "Assignment", "letter": "AB", "number": "104"},
                 "geometry": {"type": "Polygon", "coordinates": []}}]

    cset = generate(data_dir, map_features=features)

    assert any("AB104" in w for w in cset.warnings), cset.warnings
    assert cset.summary["existing_assignments"] == 0


def test_a_candidate_through_the_centre_has_no_bearing(tmp_path):
    """Zero distance means there is no direction to give; 0 degrees would read
    as due north."""
    trail = route("route_0", "TRAIL", [(0, 200), (400, 200)])
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[trail])
    area_subject(data_dir)

    facts = generate(data_dir, include=["trail"]).candidates[0].facts

    assert facts["distance_from_center_m"] == pytest.approx(0.0, abs=1e-6)
    assert facts["bearing_from_center_deg"] is None


def test_norm_flags_on_length_for_lines_and_width_for_corridors(tmp_path):
    trail = route("route_0", "TRAIL", [(0, 200), (400, 200)])
    data_dir = staged(tmp_path, valley_dem(), routes=[trail, creek()])
    area_subject(data_dir)
    cset = generate(data_dir, include=["trail", "waterway"], corridor_widths_m={"waterway": 200})
    line = next(c for c in cset.candidates if c.kind == "line")
    corridor = next(c for c in cset.candidates if c.kind == "corridor")
    assert [f["metric"] for f in line.norm_flags] == ["length_m"]       # 240 m < p10 593
    assert [f["metric"] for f in corridor.norm_flags] == ["width_m"]    # 200 m > p90 118


# --- the set ------------------------------------------------------------------------------

def test_the_set_is_deterministic_and_carries_ids_and_assumptions(tmp_path):
    trail = route("route_0", "TRAIL", [(0, 200), (400, 200)])
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[trail])
    subject = area_subject(data_dir)
    first = generate(data_dir)
    second = generate(data_dir)
    assert isinstance(first, CandidateSet)
    assert first.set_id == second.set_id and first.set_id.startswith("hasty-")
    assert first.model_dump(exclude={"created_at"}) == second.model_dump(exclude={"created_at"})
    assert first.candidates[0].id == f"{first.set_id}.1"
    latitude, longitude = subject["center"]      # the subject's own centre, [lat, lon]
    assert first.assumptions["center"] == {"latitude": pytest.approx(latitude),
                                           "longitude": pytest.approx(longitude)}
    assert first.assumptions["merge_touch_m"] == 5.0
    assert first.params["include"] == sorted(DEFAULT_INCLUDE)
    assert first.summary["count"] == 1 and first.summary["by_kind"] == {"line": 1}


def test_no_osm_means_no_lines_and_says_so(tmp_path):
    data_dir = staged(tmp_path, np.zeros((40, 40)))
    (data_dir / "osm.json").unlink()
    area_subject(data_dir)
    cset = generate(data_dir)
    assert cset.candidates == []
    assert any("no line candidates" in w for w in cset.warnings)


# --- what the payload costs -----------------------------------------------------------------

def test_the_effort_assumptions_that_never_vary_sit_once_on_the_set(tmp_path):
    """The traverse table is a few hundred bytes and identical for every
    candidate. It belongs beside the set; a candidate keeps only what its own
    measurement chose."""
    trail = route("route_0", "TRAIL", [(0, 200), (400, 200)])
    data_dir = staged(tmp_path, valley_dem(), routes=[trail, creek()])
    area_subject(data_dir)

    cset = generate(data_dir, include=["trail", "waterway"])

    shared = cset.assumptions["effort"]
    assert shared["density_class"] == "moderate"
    assert "no CIR panel" in shared["density_note"]
    assert shared["traverse"]["speed_factor_by_lifeform"]["Shrub"] == 0.4
    assert shared["movement"]["off_way_factor"] == 0.6
    assert shared["movement"]["source"] == "sar_geo_mcp.terrain.movement"

    line = next(c for c in cset.candidates if c.kind == "line")
    corridor = next(c for c in cset.candidates if c.kind == "corridor")
    # A line's assumptions are ALL shared, so it points at the set's copy
    # rather than carrying `{}`, which reads as "this one assumed nothing".
    assert line.facts["effort"]["assumptions"] == "see the set's assumptions.effort"
    assert set(corridor.facts["effort"]["assumptions"]) == {
        "mask_note", "lifeform_source", "lifeform_fractions"}


def test_facts_are_rounded_to_what_the_measurement_supports(tmp_path):
    """Full float precision is bytes in every payload and a claim about the
    ground nobody made: a 10 m DEM does not know a line's length to the
    picometre."""
    trail = route("route_0", "TRAIL", [(0, 210), (400, 210)])
    data_dir = staged(tmp_path, valley_dem(), routes=[trail, creek()])
    area_subject(data_dir)

    facts = generate(data_dir, include=["trail"]).candidates[0].facts

    def places(value):
        text = repr(float(value))
        return len(text.split(".")[1]) if "." in text and "e" not in text else 0

    assert places(facts["length_m"]) <= 1
    assert places(facts["distance_from_center_m"]) <= 1
    assert places(facts["slope_mean_deg"]) <= 1
    assert places(facts["nearest_point"]["latitude"]) <= 6
    assert places(facts["nearest_point"]["longitude"]) <= 6
    assert places(facts["effort"]["hours_one_way"]) <= 3
    corridor = generate(data_dir, include=["waterway"]).candidates[0].facts
    assert places(corridor["area_acres"]) <= 2


def test_limit_keeps_the_first_candidates_and_says_how_many_there_were(tmp_path):
    near = route("route_0", "TRAIL", [(0, 205), (400, 205)], name="Near")
    middle = route("route_1", "TRAIL", [(0, 160), (400, 160)], name="Middle")
    far = route("route_2", "TRAIL", [(0, 90), (400, 90)], name="Far")
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[near, middle, far])
    area_subject(data_dir)

    everything = generate(data_dir, include=["trail"], limit=0)   # 0 is every candidate
    limited = generate(data_dir, include=["trail"], limit=2)

    assert [c.name for c in everything.candidates] == ["Near", "Middle", "Far"]
    assert [c.name for c in limited.candidates] == ["Near", "Middle"]
    assert limited.summary["truncated_to"] == 2
    assert limited.summary["total_before_limit"] == 3
    assert limited.summary["count"] == 2
    assert any("only the first 2 of 3" in w for w in limited.warnings)
    # A different set of candidates is a different set.
    assert limited.params["limit"] == 2 and limited.set_id != everything.set_id
    assert everything.summary["truncated_to"] is None
    assert everything.summary["total_before_limit"] == 3
    assert not any("only the first" in w for w in everything.warnings)


def test_a_bare_call_returns_the_fifty_nearest_and_says_there_were_more(tmp_path):
    """Measured before this default existed: a bare call on a 1.5 km area came
    back with 1,369 candidates and about 350k tokens. The list is ordered by
    distance, so the default is the nearest fifty and the summary says how many
    there were; a caller who wants all of them passes 0."""
    routes = []
    for index in range(60):
        row, column = divmod(index, 5)
        y = 500 + row * 100
        x = 500 + column * 200
        routes.append(route(f"route_{index}", "TRAIL", [(x, y), (x + 150, y)],
                            name=f"Trail {index:02d}"))
    data_dir = staged(tmp_path, np.zeros((200, 200)), routes=routes)
    area_subject(data_dir, center_units=(1000, 1000), radius_units=800)

    default = generate(data_dir, include=["trail"])
    everything = generate(data_dir, include=["trail"], limit=0)

    assert default.summary["count"] == 50
    assert default.summary["total_before_limit"] == 60
    assert default.summary["truncated_to"] == 50
    assert any("only the first 50 of 60" in w for w in default.warnings)
    assert everything.summary["count"] == 60
    assert everything.summary["truncated_to"] is None
    assert not any("only the first" in w for w in everything.warnings)
    # The nearest fifty of the same ordering, not a different set of lines.
    assert [c.name for c in default.candidates] == [c.name for c in everything.candidates[:50]]
    # A different set of candidates is a different set.
    assert default.set_id != everything.set_id


def test_a_limit_no_shorter_than_the_list_truncates_nothing(tmp_path):
    trail = route("route_0", "TRAIL", [(0, 200), (400, 200)])
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[trail])
    area_subject(data_dir)

    cset = generate(data_dir, include=["trail"], limit=5)

    assert cset.summary["count"] == 1 and cset.summary["total_before_limit"] == 1
    assert not any("only the first" in w for w in cset.warnings)


def test_an_unknown_corridor_width_class_is_refused_by_name(tmp_path):
    """A typo silently ignored hands the planner the corpus width and lets
    them believe it is the one they asked for."""
    data_dir = staged(tmp_path, valley_dem())
    area_subject(data_dir)

    with pytest.raises(ValueError, match="drainges"):
        generate(data_dir, corridor_widths_m={"drainges": 300})


def test_a_corridor_width_of_zero_is_refused(tmp_path):
    data_dir = staged(tmp_path, valley_dem())
    area_subject(data_dir)

    with pytest.raises(ValueError, match="greater than 0"):
        generate(data_dir, corridor_widths_m={"waterway": 0})
    with pytest.raises(ValueError, match="greater than 0"):
        generate(data_dir, corridor_widths_m={"waterway": -30})


def test_one_summary_row_stays_small_enough_to_list_a_hundred(tmp_path):
    """The tool returns every candidate's summary in ONE payload, so the row
    is the unit that matters: a hundred rows is the realistic call on a real
    area, and a row that carried the traverse table would make that payload
    unreadable on its own.

    Measured on the widest facts a line has - the no-map case, which carries
    the overlap reason as well as everything else - and on a full-length trail
    rather than a stub, so no norm flag is in the count. A flag is a real
    addition to a row (about 250 characters, from the norms table), but it is
    the exception, not what every row costs.

    The budget was 900 characters until `clipped_at_edge` was added; the fact
    costs 26 of them and buys a consumer the difference between "this trail is
    1.4 km" and "this trail is 1.4 km of ground we measured, and it carries
    on". The trail here ends inside the raster, so the row carries the fact
    and not the warning that goes with it - a candidate cut by the edge pays
    another 200 characters, like a norm flag, and like a norm flag it is the
    exception.
    """
    from sar_geo_mcp.planning.candidates import summaries

    trail = route("route_0", "TRAIL", [(100, 1000), (1900, 1000)], name="Purisima Creek Trail")
    data_dir = staged(tmp_path, np.zeros((200, 200)), routes=[trail])
    area_subject(data_dir, center_units=(1000, 1000), radius_units=800)

    cset = generate(data_dir, include=["trail"])
    row = json.dumps(summaries(cset)[0])

    assert cset.candidates[0].facts["length_m"] > 1000      # an ordinary hasty line
    assert cset.candidates[0].facts["overlap_reason"]       # the widest facts, not the narrowest
    assert len(row) < 950, f"{len(row)} characters:\n{row}"
