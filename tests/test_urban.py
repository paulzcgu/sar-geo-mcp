"""Urban assignments: populated blocks merged into groups of a target size,
never across an arterial, pairing across water only two at a time."""

from __future__ import annotations

import json

import pytest
from shapely.geometry import LineString, Point, Polygon, box, mapping, shape
from shapely.ops import unary_union

from sar_geo_mcp.planning.urban import (
    NO_ARTERIALS_WARNING,
    group_blocks,
    street_links,
    urban_assignments,
    urban_params,
    urban_set,
)

# Ground metres per map unit of 1: the squares below are drawn in metres.
SCALE = 1.0


def _row(count, *, size=100.0, gap=20.0, y=0.0):
    """`count` size-metre squares (1 ha at 100 m) in a row, `gap` apart."""
    return [box(i * (size + gap), y, i * (size + gap) + size, y + size) for i in range(count)]


def _members(groups):
    return sorted(sorted(group["members"]) for group in groups)


def _ha(geometry):
    return geometry.area / 10_000


def test_neighbours_merge_towards_the_target_and_no_further():
    shapes = _row(6)                                  # six 1 ha blocks, a street apart
    groups = group_blocks(shapes, target_ha=3.0, scale=SCALE)
    assert sorted(m for g in groups for m in g["members"]) == list(range(6))
    for group in groups:
        member_ha = sum(_ha(shapes[m]) for m in group["members"])
        assert member_ha <= 3.0 * 1.4 + 1e-9          # never past the ceiling
        assert member_ha >= 3.0 * 0.6 - 1e-9          # every group reached the settled size
        assert group["geometry"].geom_type == "Polygon"
    # Pairs of 2 ha pass the settled share (1.8 ha) and stop: two pairs
    # would make 4 ha, no nearer the 3 ha target.
    assert _members(groups) == [[0, 1], [2, 3], [4, 5]]


def _two_streets_and_a_cul_de_sac():
    """Two parallel east-west streets, their lot rows back to back along a
    fence at y=0 (each block is a street both sides, so the fence is its
    edge), and a cul-de-sac running north off the upper street.

    Block 0: Upper Street, y 0..60, its centreline at y=30.
    Block 1: Lower Street, y -60..0, its centreline at y=-30.
    Block 2: Court, x 0..60, y 70..200 (a 10 m gap: the closing will span
    it), its centreline at x=30 from the upper street's centreline north."""
    shapes = [box(0, 0, 300, 60), box(0, -60, 300, 0), box(0, 70, 60, 200)]
    names = ["Upper Street", "Lower Street", "Court"]
    streets = {"Upper Street": [LineString([(-20, 30), (320, 30)])],
               "Lower Street": [LineString([(-20, -30), (320, -30)])],
               "Court": [LineString([(30, 30), (30, 190)])]}
    return shapes, names, streets


def test_blocks_backing_onto_each_other_are_not_linked_but_a_side_street_is():
    shapes, names, streets = _two_streets_and_a_cul_de_sac()
    links = street_links(shapes, names, streets, scale=SCALE)
    assert frozenset((0, 2)) in links                 # the court opens onto Upper Street
    assert frozenset((0, 1)) not in links             # a fence between them, no street


def test_a_street_s_two_sides_are_never_split_between_back_to_back_teams():
    """The lot rows either side of a fence go to different teams, each with
    its own street both sides: grouping them would hand each street's two
    sides to two teams (seen on a hand-drawn plan)."""
    shapes, names, streets = _two_streets_and_a_cul_de_sac()
    links = street_links(shapes, names, streets, scale=SCALE)
    groups = group_blocks(shapes, target_ha=3.0, scale=SCALE, street_links=links)
    assert _members(groups) == [[0, 2], [1]]


def test_a_tiny_group_with_no_street_to_grow_along_joins_the_block_behind_it():
    shapes, names, streets = _two_streets_and_a_cul_de_sac()
    links = street_links(shapes, names, streets, scale=SCALE)
    # Lower Street alone is 1.8 ha: under 30 % of a 10 ha target, it may
    # join what it backs onto.
    groups = group_blocks(shapes, target_ha=10.0, scale=SCALE, street_links=links)
    assert _members(groups) == [[0, 1, 2]]


def test_two_streets_that_meet_behind_a_third_streets_houses_stay_apart():
    """Their streets join, but the ground between them is another block's:
    merged, the team's outline would come in two pieces (seen on a
    hand-drawn plan: two streets with a third street's houses between)."""
    shapes = [box(0, 0, 100, 100), box(0, 110, 100, 210), box(0, 101, 100, 109)]
    groups = group_blocks(shapes, target_ha=3.0, scale=SCALE,
                          street_links={frozenset((0, 1))})
    assert not any({0, 1} <= set(group["members"]) for group in groups)
    for group in groups:
        assert group["geometry"].geom_type == "Polygon"


def test_two_runs_of_one_street_are_linked_and_unnamed_blocks_link_by_adjacency():
    shapes = _row(3)
    links = street_links(shapes, ["Elm", "Elm", None], {"Elm": []}, scale=SCALE)
    assert links == {frozenset((0, 1)), frozenset((1, 2))}


def test_the_street_between_merged_blocks_is_inside_the_outline():
    shapes = _row(2)
    (group,) = group_blocks(shapes, target_ha=2.0, scale=SCALE)
    assert group["geometry"].contains(box(100.0, 10.0, 120.0, 90.0))   # the 20 m street


def test_an_arterial_is_never_crossed():
    shapes = _row(2)
    boulevard = LineString([(110.0, -50.0), (110.0, 150.0)])           # down the street
    groups = group_blocks(shapes, target_ha=2.0, scale=SCALE, arterials=boulevard)
    assert _members(groups) == [[0], [1]]


def test_small_groups_pair_across_water_two_at_most():
    # Three isles 100 m apart across water: too far to be street neighbours.
    shapes = _row(3, gap=100.0)
    groups = group_blocks(shapes, target_ha=5.0, scale=SCALE)
    sizes = sorted(len(g["members"]) for g in groups)
    assert sizes == [1, 2]                            # one pair, never a chain of three
    pair = next(g for g in groups if len(g["members"]) == 2)
    assert pair["water_links"] == 1
    assert pair["geometry"].geom_type == "Polygon"    # drawn as one by the strip


def test_far_apart_blocks_never_meet():
    shapes = _row(2, gap=400.0)
    assert _members(group_blocks(shapes, target_ha=5.0, scale=SCALE)) == [[0], [1]]


def test_groups_share_no_ground_and_hold_every_block_whole():
    shapes = _row(4) + _row(4, y=120.0)               # two rows a street apart
    groups = group_blocks(shapes, target_ha=2.5, scale=SCALE)
    outlines = [g["geometry"] for g in groups]
    for i, a in enumerate(outlines):
        for b in outlines[i + 1:]:
            assert a.intersection(b).area < 0.01
    for group in groups:
        for member in group["members"]:
            assert group["geometry"].contains(shapes[member].buffer(-0.01))


def test_a_block_hemmed_inside_a_larger_group_joins_it_rather_than_vanishing():
    """A 20 m block in a 26 m courtyard of a 4 ha block, 3 m from it all
    round: too big a merge for a 2 ha target, so they are two groups, but
    the large one's closing spans the courtyard and reaches all the small
    one's ground first. Found live on ground staged while Overpass was
    down, where lots grouped by adjacency alone made one 132 ha block round
    a small one: the small group's outline came back empty and
    urban_assignments raised "max() iterable argument is empty"."""
    from shapely.geometry import Polygon

    big = Polygon(box(0, 0, 200, 200).exterior.coords,
                  [box(87, 87, 113, 113).exterior.coords])
    small = box(90, 90, 110, 110)
    groups = group_blocks([big, small], target_ha=2.0, scale=SCALE)
    (group,) = groups
    assert group["members"] == [0, 1]
    assert group["geometry"].geom_type == "Polygon" and not group["geometry"].is_empty
    assert group["geometry"].contains(small.buffer(-0.01))
    assert _ha(group["geometry"]) == pytest.approx(4.0)


def _ring(outer, inner):
    """`outer` with `inner` cut out: a block round a courtyard."""
    return Polygon(outer.exterior.coords, [inner.exterior.coords])


def _share(geometry, block):
    """How much of `block`'s ground lies inside `geometry`."""
    return geometry.intersection(block).area / block.area


def _drawn_where_counted(shapes, groups):
    """Every block is counted by exactly one group, most of its ground lies
    inside that group's outline, and no other group's outline holds more
    than a sliver of it: a team sent to a group searches the houses its
    counts hold, and only those."""
    assert sorted(m for g in groups for m in g["members"]) == list(range(len(shapes)))
    for group in groups:
        for m in group["members"]:
            assert _share(group["geometry"], shapes[m]) > 0.5, (group["members"], m)
            for other in groups:
                if other is not group:
                    assert _share(other["geometry"], shapes[m]) < 0.05, (
                        f"block {m}, counted in {group['members']}, lies in "
                        f"{other['members']}'s outline")


#: Hemmed two deep, each 3 m inside the next: a 4 ha block round a 26 m
#: courtyard, a 20 m block round a 10 m courtyard in it, and a 6 m block in
#: that.
TWO_DEEP = [_ring(box(0, 0, 200, 200), box(87, 87, 113, 113)),
            _ring(box(90, 90, 110, 110), box(95, 95, 105, 105)),
            box(97, 97, 103, 103)]
#: A 4 ha block with a slot 20 m wide and 50 m deep open to the east; the
#: block standing in it, 3 m from the walls, is `_slotted`.
SLOTTED = Polygon([(0, 0), (200, 0), (200, 90), (150, 90), (150, 110), (200, 110),
                   (200, 200), (0, 200)])
#: Its mirror image to the east, the two slots meeting at x = 200.
SLOTTED_EAST = Polygon([(200, 0), (400, 0), (400, 200), (200, 200), (200, 110), (250, 110),
                        (250, 90), (200, 90)])


def _slotted(proud_m):
    """The block in `SLOTTED`'s slot, its east end `proud_m` past the mouth."""
    return box(155, 93, 200 + proud_m, 107)


@pytest.mark.parametrize("target_ha", [
    pytest.param(0.01, id="three groups of their own, each hemmed"),
    pytest.param(2.0, id="the inner two grouped, then hemmed together"),
])
def test_blocks_hemmed_two_deep_all_join_the_outermost_group(target_ha):
    """With a target under the middle block's size no two may merge, so each
    inner block is a group of its own whose outline the outer group's
    closing has already reached; at 2 ha the inner two merge first and are
    hemmed as one. Either way all three are counted, in the one outline
    drawn round them."""
    groups = group_blocks(TWO_DEEP, target_ha=target_ha, scale=SCALE)
    (group,) = groups
    assert group["members"] == [0, 1, 2]
    assert _ha(group["geometry"]) == pytest.approx(4.0)
    _drawn_where_counted(TWO_DEEP, groups)


def test_a_courtyard_too_wide_to_close_stays_open_round_the_group_inside_it():
    """A 100 m courtyard is wider than the closing spans, and another
    group's block stands in it, so the outer group keeps it as a hole and
    the inner group - a block round its own small courtyard, and the block
    in that - is drawn in it, not hemmed."""
    shapes = [_ring(box(0, 0, 200, 200), box(50, 50, 150, 150)),
              _ring(box(60, 60, 140, 140), box(87, 87, 113, 113)),
              box(90, 90, 110, 110)]
    groups = group_blocks(shapes, target_ha=2.0, scale=SCALE)
    assert [g["members"] for g in groups] == [[0], [1, 2]]
    outer, inner = (g["geometry"] for g in groups)
    assert len(outer.interiors) == 1 and _ha(outer) == pytest.approx(3.0)
    assert inner.equals(box(60, 60, 140, 140))            # its own courtyard filled
    assert outer.intersection(inner).area < 0.01
    _drawn_where_counted(shapes, groups)


@pytest.mark.parametrize("proud_m", [
    pytest.param(0.0, id="flush with the mouth"),
    pytest.param(0.01, id="1 cm proud"),
    pytest.param(0.5, id="50 cm proud"),
    pytest.param(2.0, id="2 m proud"),
])
def test_a_block_in_a_slot_of_a_larger_group_is_drawn_where_it_is_counted(proud_m):
    """The slot's closing reaches the small block, too big a merge for a
    2 ha target. Flush with the slot's mouth, every bit of its ground is
    already claimed and it joins the large group. A few centimetres more,
    and the strip outside the mouth became its whole outline: a 2 x 14 m
    assignment carrying all its houses, while 96 % of its ground and every
    house was drawn inside the large group's outline and left out of its
    counts. Moving a block by centimetres must not flip that."""
    shapes = [SLOTTED, _slotted(proud_m)]
    _drawn_where_counted(shapes, group_blocks(shapes, target_ha=2.0, scale=SCALE))


@pytest.mark.parametrize("hemmed", [
    pytest.param([box(152, 92, 198, 108), box(205, 93, 245, 107)],
                 id="one block in each slot, grouped together"),
    pytest.param([box(155, 93, 245, 107)], id="one block across both slots"),
])
def test_blocks_hemmed_between_two_groups_are_counted_where_they_are_drawn(hemmed):
    """Two 4 ha blocks side by side, their slots meeting. What stands in the
    slots is hemmed by both outlines, and went whole to the one holding
    most of it: a block wholly inside the east group's outline was counted
    in the west group, and the east team's counts left out houses its
    polygon holds."""
    shapes = [SLOTTED, SLOTTED_EAST, *hemmed]
    _drawn_where_counted(shapes, group_blocks(shapes, target_ha=2.0, scale=SCALE))


#: A 6 ha block round two 26 m courtyards, a 20 m block in each: the two
#: small blocks are 80 m apart, too far to be street neighbours, so they pair
#: across water before the large block's closing hems them both.
TWO_COURTYARDS = [Polygon(box(0, 0, 300, 200).exterior.coords,
                          [box(87, 87, 113, 113).exterior.coords,
                           box(187, 87, 213, 113).exterior.coords]),
                  box(90, 90, 110, 110), box(190, 90, 210, 110)]


def test_a_hemmed_water_pair_joined_by_the_hosts_own_ground_takes_no_strip():
    """Both blocks of the pair stand inside the large block's outline, which
    joins them by its own ground: no strip is drawn, no water link counted,
    so no team is warned of water that is not there."""
    (group,) = group_blocks(TWO_COURTYARDS, target_ha=2.0, scale=SCALE)
    assert group["members"] == [0, 1, 2] and group["hemmed"] == [1, 2]
    assert group["water_links"] == 0
    assert group["geometry"].equals(box(0, 0, 300, 200))
    _drawn_where_counted(TWO_COURTYARDS, [group])


def test_a_hemmed_water_pair_brings_the_strip_its_isle_needs():
    """A block in the slot, paired across water with an isle 80 m east: the
    isle is cut off from the large block's outline but for the strip, so the
    strip comes with it and the host counts the water link."""
    shapes = [SLOTTED, _slotted(0.0), box(280, 95, 300, 105)]
    (group,) = group_blocks(shapes, target_ha=2.0, scale=SCALE)
    assert group["members"] == [0, 1, 2] and group["water_links"] == 1
    assert group["geometry"].geom_type == "Polygon"
    _drawn_where_counted(shapes, [group])


def test_a_block_under_a_square_metre_still_has_an_outline():
    """Every block lies inside its own group's outline, however small: an
    empty outline drops the block from the drawing and made
    urban_assignments raise."""
    shapes = [box(0, 0, 100, 100), box(300, 300, 300.6, 300.6)]
    groups = group_blocks(shapes, target_ha=2.0, scale=SCALE)
    assert _members(groups) == [[0], [1]]
    for group in groups:
        assert not group["geometry"].is_empty
    _drawn_where_counted(shapes, groups)


def test_a_target_that_is_not_above_zero_is_refused():
    with pytest.raises(ValueError, match="target_ha"):
        group_blocks(_row(2), target_ha=0, scale=SCALE)


# --------------------------------------------------------------------------
# urban_assignments over a staged town (the synthetic one test_populated draws)


def _town_dir(tmp_path, *, query_version=None):
    from test_populated import _staged

    directory = _staged(tmp_path)
    if query_version is not None:
        payload = json.loads((directory / "osm.json").read_text(encoding="utf-8"))
        payload["query_version"] = query_version
        (directory / "osm.json").write_text(json.dumps(payload), encoding="utf-8")
    return directory


def _run(tmp_path, **kwargs):
    from test_populated import WMS, fake_fetch, town

    directory = _town_dir(tmp_path, query_version=kwargs.pop("query_version", None))
    parcels, roofs = town()
    return urban_assignments(directory, wms_base_url=WMS, cache_dir=tmp_path / "cache",
                             fetch=fake_fetch(parcels, roofs), **kwargs)


def test_the_default_target_is_the_corpus_median_for_the_resource():
    from sar_geo_mcp.planning.norms import load_norms, target_area_range_ha

    norms = load_norms()
    assert target_area_range_ha("GROUND", norms)["p50"] == 10.0


def test_urban_assignments_merge_the_towns_blocks(tmp_path):
    result = _run(tmp_path, target_ha=2.0)
    assert result["assumptions"]["grouping"]["target_ha"] == 2.0
    assert result["assumptions"]["grouping"]["target_source"] == "given"
    (assignment,) = result["assignments"]              # the street and the apartments
    facts = assignment["facts"]
    assert facts["blocks"] == 2 and facts["parcels"] == 13 and facts["buildings"] == 16
    assert set(facts["streets"]) == {"Elm Street"}
    assert assignment["name"] == "Elm Street"
    assert result["summary"]["count"] == 1 and result["summary"]["parcels"] == 13


def test_urban_assignments_default_to_the_ground_norm(tmp_path):
    result = _run(tmp_path)
    grouping = result["assumptions"]["grouping"]
    assert grouping["target_ha"] == 10.0
    assert grouping["target_source"] == "corpus median for GROUND"


def test_arterials_unknown_on_old_ground_is_said(tmp_path):
    result = _run(tmp_path, target_ha=2.0, query_version=4)
    assert NO_ARTERIALS_WARNING.format(version=4) in result["warnings"]
    assert result["assumptions"]["grouping"]["arterials"] == "unknown"


def test_markers_travel_with_the_group(tmp_path):
    result = _run(tmp_path, target_ha=2.0, building_markers=True)
    (assignment,) = result["assignments"]
    assert len(assignment["markers"]) == 16 == assignment["facts"]["markers"]
    assert result["summary"]["markers"] == 16


def test_the_urban_set_holds_assignment_candidates(tmp_path):
    from test_populated import WMS, fake_fetch, town

    directory = _town_dir(tmp_path)
    parcels, roofs = town()
    params = urban_params(target_ha=2.0, resource_type="ground", max_distance_m=None)
    assert params == {"method": 3, "target_ha": 2.0, "resource_type": "GROUND",
                      "max_distance_m": None}
    cset = urban_set(directory, params, handle="aoi-test", norms=None, wms_base_url=WMS,
                     cache_dir=tmp_path / "cache", fetch=fake_fetch(parcels, roofs))
    assert cset.set_id.startswith("urban-")
    (candidate,) = cset.candidates
    assert candidate.kind == "assignment" and candidate.axis == "urban"
    assert candidate.geometry["type"] == "Polygon"
    assert candidate.facts["block_names"] == ["Elm Street", "Elm Street - large parcel"]


def test_a_target_left_to_the_norms_puts_them_in_the_set_id(tmp_path):
    """The norms choose a target left out, so their digest is hashed in and
    another norms file is another set (PR #4 review B23); a target given
    reads no norms and hashes none. A set is never grouped under other
    norms than its id names."""
    from sar_geo_mcp.planning.candidates import make_set_id
    from sar_geo_mcp.planning.norms import load_norms

    agency = load_norms()
    agency["area_ha_by_resource"]["medians"]["GROUND"] = 1.0

    def set_id(target_ha, norms):
        return make_set_id("urban", urban_params(target_ha=target_ha, resource_type="GROUND",
                                                 max_distance_m=None, norms=norms))

    assert set_id(None, None) == set_id(None, load_norms())
    assert set_id(None, agency) != set_id(None, None)
    assert set_id(2.0, agency) == set_id(2.0, None)
    with pytest.raises(ValueError, match="made under norms"):
        urban_set(tmp_path, urban_params(target_ha=None, resource_type="GROUND",
                                         max_distance_m=None, norms=agency),
                  handle="aoi-test", norms=None, wms_base_url=None, cache_dir=None)


# --------------------------------------------------------------------------
# urban_assignments over hemmed blocks (populated_areas faked)


def _wgs84(geometry):
    """A shape drawn in EPSG:3857 metres near 0°N 0°E, where a map unit is a
    ground metre, in WGS84."""
    import numpy
    import shapely
    from pyproj import Transformer

    to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform
    return shapely.transform(geometry, lambda xy: numpy.column_stack(to_wgs84(xy[:, 0],
                                                                            xy[:, 1])))


def _fake_blocks(monkeypatch, layout):
    """`populated_areas` faked to answer the blocks of `layout`, each
    `(street, shape, parcels, buildings)`, with a house marker per building
    a metre inside the block's edge. Returns the blocks."""
    blocks = []
    for street, block, parcels, buildings in layout:
        edge = block.buffer(-1.0).exterior
        houses = [edge.interpolate(i / buildings, normalized=True) for i in range(buildings)]
        blocks.append({
            "name": street, "geometry": mapping(_wgs84(block)),
            "facts": {"street": street, "parcels": parcels, "buildings": buildings},
            "markers": [{"type": "Feature", "geometry": mapping(_wgs84(house)),
                         "properties": {"class": "Marker", "title": f"{street} {i}"}}
                        for i, house in enumerate(houses)]})
    centre = _wgs84(Point(200, 100))
    answer = {"blocks": blocks, "center": {"latitude": centre.y, "longitude": centre.x},
              "warnings": [], "summary": {"parcels_left_out": 0}, "assumptions": {}}
    monkeypatch.setattr("sar_geo_mcp.planning.populated.populated_areas",
                        lambda directory, **kwargs: answer)
    return blocks


TOWN_TWO_DEEP = [("Oak Street", TWO_DEEP[0], 60, 55), ("Elm Court", TWO_DEEP[1], 8, 7),
                 ("Ash Lane", TWO_DEEP[2], 3, 2)]


def _town_slotted(proud_m):
    return [("Oak Street", SLOTTED, 60, 55), ("Elm Court", _slotted(proud_m), 5, 5)]


TOWN_TWO_SLOTS = [("A Street", SLOTTED, 40, 40), ("B Street", SLOTTED_EAST, 40, 40),
                  ("C1 Court", box(152, 92, 198, 108), 6, 6),
                  ("C2 Court", box(205, 93, 245, 107), 5, 5)]


@pytest.mark.parametrize("layout", [
    pytest.param(TOWN_TWO_DEEP, id="hemmed two deep"),
    pytest.param(_town_slotted(0.0), id="in a slot, flush"),
    pytest.param(_town_slotted(2.0), id="in a slot, 2 m proud"),
    pytest.param(TOWN_TWO_SLOTS, id="between two groups"),
])
def test_every_house_is_counted_once_in_the_assignment_drawn_round_it(
        monkeypatch, tmp_path, layout):
    """The counts a team is handed are the houses inside the polygon it is
    sent to: every block is in one assignment, the summary adds up to the
    blocks, no assignment is a sliver of no area carrying houses, and every
    house marker lies inside its own assignment's polygon and no other's."""
    blocks = _fake_blocks(monkeypatch, layout)
    result = urban_assignments(tmp_path, target_ha=2.0, building_markers=True)
    assignments = result["assignments"]

    assert sorted(name for a in assignments for name in a["blocks"]) == sorted(
        block["name"] for block in blocks)
    summary = result["summary"]
    assert summary["blocks"] == len(blocks)
    assert summary["parcels"] == sum(block["facts"]["parcels"] for block in blocks)
    assert summary["buildings"] == sum(block["facts"]["buildings"] for block in blocks)
    assert summary["markers"] == sum(len(block["markers"]) for block in blocks)
    for assignment in assignments:
        polygon = shape(assignment["geometry"])
        assert assignment["facts"]["area_ha"] > 0, assignment["name"]
        outside = [marker["properties"]["title"] for marker in assignment["markers"]
                   if not polygon.contains(shape(marker["geometry"]))]
        assert outside == [], f"{assignment['name']} counts houses outside it"
        foreign = [marker["properties"]["title"] for other in assignments
                   if other is not assignment for marker in other["markers"]
                   if polygon.contains(shape(marker["geometry"]))]
        assert foreign == [], f"{assignment['name']} holds houses counted elsewhere"


def test_blocks_hemmed_two_deep_are_counted_in_the_assignment_drawn_round_them(
        monkeypatch, tmp_path):
    """Before the fix the two inner blocks' outline came back empty and
    urban_assignments raised; now the outer block's assignment holds them:
    their streets, blocks, parcels, buildings and markers in its facts, and
    no assignment of their own."""
    _fake_blocks(monkeypatch, TOWN_TWO_DEEP)
    result = urban_assignments(tmp_path, target_ha=2.0, building_markers=True)
    (assignment,) = result["assignments"]
    assert assignment["name"] == "Oak Street +2"
    assert assignment["blocks"] == ["Oak Street", "Elm Court", "Ash Lane"]
    facts = assignment["facts"]
    assert facts["streets"] == ["Oak Street", "Elm Court", "Ash Lane"]
    assert (facts["blocks"], facts["parcels"], facts["buildings"], facts["markers"]) == (
        3, 71, 64, 64)
    assert len(assignment["markers"]) == 64
    assert facts["area_ha"] == pytest.approx(4.0, abs=0.01) and facts["parts"] == 1
    assert result["summary"]["count"] == 1
    assert result["summary"]["area_ha"]["min"] == facts["area_ha"]


def test_a_hemmed_water_pair_inside_the_host_warns_of_no_water(monkeypatch, tmp_path):
    """Two courts paired across water, both standing in the large block's
    courtyards: one block of ground, so the assignment counts no water link
    and carries no warning of water to drive round."""
    _fake_blocks(monkeypatch, [("Oak Street", TWO_COURTYARDS[0], 60, 55),
                               ("Elm Court", TWO_COURTYARDS[1], 4, 4),
                               ("Ash Court", TWO_COURTYARDS[2], 4, 4)])
    result = urban_assignments(tmp_path, target_ha=2.0)
    (assignment,) = result["assignments"]
    assert assignment["facts"]["water_links"] == 0
    # Oak Street alone is past the 2.8 ha ceiling: the courts did not carry it.
    assert assignment["warnings"] == []


#: A 2.65 ha block with a 24 m slot 100 m deep, open to the east.
SLOTTED_SMALL = Polygon([(0, 0), (175, 0), (175, 68), (75, 68), (75, 92), (175, 92),
                         (175, 165), (0, 165)])


def test_a_hemmed_join_past_the_ceiling_is_said(monkeypatch, tmp_path):
    """Merging stops at 140 % of the target; a block standing in the slot
    joins anyway, to be counted where it is drawn. When that is what takes
    the assignment past the ceiling, its warnings name the block."""
    _fake_blocks(monkeypatch, [("Oak Street", SLOTTED_SMALL, 40, 40),
                               ("Elm Court", box(78, 71, 177, 89), 5, 5)])
    result = urban_assignments(tmp_path, target_ha=2.0)
    (assignment,) = result["assignments"]
    assert assignment["blocks"] == ["Oak Street", "Elm Court"]
    (warning,) = assignment["warnings"]
    assert warning.startswith("Elm Court, most of whose ground lies inside this assignment's")
    assert "2.83 ha, past the 2.80 ha (140 % of the target)" in warning
