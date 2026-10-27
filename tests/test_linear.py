import json

import pytest
from shapely.geometry import box

from sar_geo_mcp.terrain.linear import (
    LINEAR_ASSUMPTIONS,
    OSM_TILE_M,
    build_overpass_query,
    classify_site,
    classify_way,
    gates_for,
    ingress_candidates,
    is_site_shaped,
    parking_for,
    routes_for,
    site_flags,
    tiles_covering_osm,
)

# A small AOI in public parkland.
BBOX = (-13606608.2, 4490700.1, -13605941.3, 4491434.5)
NEIGHBOUR_BBOX = (-13606208.2, 4491100.1, -13605541.3, 4491834.5)


def fake_overpass_payload() -> dict:
    """Two ways crossing the AOI, mirroring what Overpass really returns here."""
    return {
        "elements": [
            {
                "type": "way", "id": 1,
                "tags": {"highway": "path", "name": "Betsy Crowder Trail"},
                "geometry": [
                    {"lat": 37.3690, "lon": -122.2300},
                    {"lat": 37.3720, "lon": -122.2250},
                    {"lat": 37.3745, "lon": -122.2243},
                ],
            },
            {
                "type": "way", "id": 2,
                "tags": {"highway": "service"},
                "geometry": [
                    {"lat": 37.3700, "lon": -122.2295},
                    {"lat": 37.3702, "lon": -122.2280},
                ],
            },
        ]
    }


def test_classify_way_separates_trails_from_roads():
    assert classify_way({"highway": "path"}) == "TRAIL"
    assert classify_way({"highway": "footway"}) == "TRAIL"
    assert classify_way({"highway": "track"}) == "TRACK"
    assert classify_way({"highway": "service"}) == "ROAD"
    assert classify_way({"highway": "secondary"}) == "ROAD"
    assert classify_way({"waterway": "stream"}) == "WATERWAY"
    assert classify_way({"highway": "nonsense"}) == "ROAD_OTHER"


def test_overpass_query_covers_the_tile_and_asks_for_geometry():
    query = build_overpass_query(0, 0)

    assert "out geom" in query
    assert "highway" in query
    assert "waterway" in query


def test_neighbouring_assignments_share_osm_tiles():
    assert tiles_covering_osm(BBOX) == tiles_covering_osm(NEIGHBOUR_BBOX)


def test_second_assignment_makes_no_new_request(tmp_path):
    calls = []

    def fake_get(query: str) -> dict:
        calls.append(query)
        return fake_overpass_payload()

    routes_for(BBOX, cache_dir=tmp_path, get_json=fake_get)
    routes_for(NEIGHBOUR_BBOX, cache_dir=tmp_path, get_json=fake_get)

    assert len(calls) == 1


def test_routes_are_clipped_and_named(tmp_path):
    def fake_get(query: str) -> dict:
        return fake_overpass_payload()

    routes = routes_for(BBOX, cache_dir=tmp_path, get_json=fake_get)

    names = [r["properties"]["name"] for r in routes]
    assert "Betsy Crowder Trail" in names
    trail = next(r for r in routes if r["properties"]["name"] == "Betsy Crowder Trail")
    assert trail["kind"] == "route"
    assert trail["properties"]["route_type"] == "TRAIL"
    assert trail["id"].startswith("route_")
    assert trail["geometry"]["type"] in ("LineString", "MultiLineString")


def test_route_records_keep_the_power_tag(tmp_path):
    """A wire and a fence are both BARRIER; only the tag tells them apart,
    so the tag travels with the record. Absent on a fence is `None`, not a
    missing key: a missing key is how a record staged BEFORE this field
    is recognised."""
    payload = {"elements": [
        {"type": "way", "id": 1,
         "tags": {"power": "line", "name": "Jefferson-Martin"},
         "geometry": [{"lat": 37.3690, "lon": -122.2300},
                      {"lat": 37.3745, "lon": -122.2243}]},
        {"type": "way", "id": 2,
         "tags": {"barrier": "fence"},
         "geometry": [{"lat": 37.3700, "lon": -122.2300},
                      {"lat": 37.3740, "lon": -122.2250}]},
    ]}
    routes = routes_for(BBOX, cache_dir=tmp_path, get_json=lambda query: payload)

    by_id = {route["properties"]["osm_id"]: route for route in routes}
    assert by_id[1]["properties"]["route_type"] == "BARRIER"
    assert by_id[1]["properties"]["power"] == "line"
    assert by_id[2]["properties"]["route_type"] == "BARRIER"
    assert "power" in by_id[2]["properties"]
    assert by_id[2]["properties"]["power"] is None


def test_cached_payload_is_reused_from_disk(tmp_path):
    calls = []

    def fake_get(query: str) -> dict:
        calls.append(query)
        return fake_overpass_payload()

    routes_for(BBOX, cache_dir=tmp_path, get_json=fake_get)

    def exploding_get(query: str) -> dict:
        raise AssertionError("must not hit the network")

    again = routes_for(BBOX, cache_dir=tmp_path, get_json=exploding_get)

    assert len(calls) == 1
    assert len(again) >= 1


def test_ingress_candidates_are_where_routes_cross_the_boundary():
    assignment = box(0, 0, 100, 100)
    route = {
        "id": "route_0", "kind": "route",
        "geometry": {"type": "LineString",
                     "coordinates": [[-50.0, 50.0], [150.0, 50.0]]},
        "properties": {"route_type": "TRAIL", "name": "Test Trail"},
    }

    candidates = ingress_candidates([route], assignment)

    assert len(candidates) == 2  # enters and exits
    assert candidates[0]["kind"] == "ingress"
    assert candidates[0]["geometry"]["type"] == "Point"
    assert candidates[0]["properties"]["route_id"] == "route_0"


def test_route_fully_outside_produces_no_ingress():
    assignment = box(0, 0, 100, 100)
    route = {
        "id": "route_0", "kind": "route",
        "geometry": {"type": "LineString",
                     "coordinates": [[500.0, 500.0], [600.0, 600.0]]},
        "properties": {"route_type": "TRAIL", "name": "Far Away"},
    }

    assert ingress_candidates([route], assignment) == []


def test_assumptions_are_exported():
    assert "osm_tile_m" in LINEAR_ASSUMPTIONS
    assert OSM_TILE_M > 0


# --- the classes the anatomy report matched boundaries to ----------------------

def test_classify_way_covers_every_anatomy_class():
    assert classify_way({"highway": "cycleway"}) == "TRAIL"
    assert classify_way({"highway": "tertiary_link"}) == "ROAD"
    assert classify_way({"highway": "pedestrian"}) == "ROAD"
    assert classify_way({"highway": "raceway"}) == "ROAD_OTHER"
    assert classify_way({"waterway": "stream"}) == "WATERWAY"
    assert classify_way({"power": "line"}) == "BARRIER"
    assert classify_way({"railway": "rail"}) == "BARRIER"
    assert classify_way({"barrier": "fence"}) == "BARRIER"
    assert classify_way({"barrier": "wall"}) == "BARRIER"
    assert classify_way({"man_made": "pipeline"}) == "BARRIER"
    assert classify_way({"natural": "water"}) == "WATER_EDGE"
    assert classify_way({"natural": "coastline"}) == "WATER_EDGE"
    assert classify_way({"landuse": "reservoir"}) == "WATER_EDGE"
    assert classify_way({"barrier": "hedge"}) == "OTHER"
    assert classify_way({}) == "OTHER"


def test_both_classifiers_tolerate_a_null_tags_dict():
    # Overpass omits `tags` rather than sending null, so this is defensive
    # rather than urgent - but the two sibling classifiers in this module
    # should not disagree about it: `classify_site` already guards with
    # `tags = tags or {}`, and `classify_way` should answer its own
    # "nothing here" rather than raising on a `None` tags dict.
    assert classify_way(None) == "OTHER"
    assert classify_site(None) == ""


def test_classify_site_offers_parking_regardless_of_access_or_deck():
    """2026-09-09 operator ruling: nothing is refused for access or for
    being a deck any more - that is the search manager's phone call, not
    this function's guess. `classify_site` returns `"parking"` for every
    site-shaped parking element; `""` now means only "not a parking or
    hardstand element at all". `site_flags` is where access and deck
    restrictions show up, checked separately below."""
    assert classify_site({"amenity": "parking"}) == "parking"
    assert classify_site({"amenity": "parking", "access": "yes"}) == "parking"
    assert classify_site({"amenity": "parking", "access": "permissive"}) == "parking"
    assert classify_site({"amenity": "parking", "access": "private"}) == "parking"
    assert classify_site({"amenity": "parking", "access": "customers"}) == "parking"
    # A deck is a phone call, not a wall: still offered, flagged
    # `covered_deck` (see test_site_flags_covered_deck_for_a_flagged_parking_subtag).
    assert classify_site({"amenity": "parking", "parking": "underground"}) == "parking"
    assert classify_site({"amenity": "parking", "parking": "multi-storey"}) == "parking"
    assert classify_site({"amenity": "parking", "parking": "surface"}) == "parking"
    # A stacked value ("private, and customers may also come") still offers,
    # and site_flags names both parts - not an either/or a naive
    # split-and-take-the-first-part reading would miss.
    assert classify_site({"amenity": "parking", "access": "private;customers"}) == "parking"
    # A space after the semicolon must not hide the flagged part from
    # ACCESS_FLAGGED: `_tag` only strips the ends of the whole string, so
    # the second token here keeps its leading space unless the split itself
    # strips each part before returning it. This pair used to be refused;
    # it is now offered and flagged both `permissive` and `access_private`.
    assert classify_site({"amenity": "parking", "access": "permissive; private"}) == "parking"
    assert site_flags({"amenity": "parking", "access": "permissive; private"}) == [
        "permissive", "access_private",
    ]


def test_classify_site_offers_public_hardstand_and_nothing_else():
    assert classify_site({"amenity": "school"}) == "hardstand"
    assert classify_site({"amenity": "place_of_worship"}) == "hardstand"
    assert classify_site({"leisure": "park"}) == "hardstand"
    assert classify_site({"landuse": "recreation_ground"}) == "hardstand"
    assert classify_site({"tourism": "picnic_site"}) == "hardstand"
    assert classify_site({"highway": "rest_area"}) == "hardstand"
    # A private school is offered too, flagged access_private - a search
    # manager's phone call to the school, not this function's guess.
    assert classify_site({"amenity": "school", "access": "private"}) == "hardstand"
    # Everything the route query is really for stays out of this one.
    assert classify_site({"highway": "path"}) == ""
    assert classify_site({"waterway": "stream"}) == ""
    assert classify_site({}) == ""


def test_classify_site_wins_parking_over_hardstand():
    # The headline claim of the docstring: a lot inside a school is a lot.
    # Reordering the amenity-parking check after the hardstand loop would
    # not fail any other test in this file.
    assert classify_site({"amenity": "parking", "leisure": "park"}) == "parking"


def test_classify_site_normalises_tag_case():
    # Case-folding on the values this module actually branches on: the
    # amenity that opens the parking path, and the access value that closes
    # it. (Swapping `_tag` for a raw `tags.get` on the HARDSTAND_TAGS loop
    # or the `parking` sub-tag is a separate mutation - see the two asserts
    # below, which pin those specifically.)
    assert classify_site({"amenity": "Parking"}) == "parking"
    assert classify_site({"amenity": "parking", "access": "Private"}) == "parking"
    # Case-folding must reach ACCESS_FLAGGED too, or `Private` would slip
    # past it unflagged.
    assert site_flags({"amenity": "parking", "access": "Private"}) == ["access_private"]
    # `_tag(tags, key) in values` must be used for the HARDSTAND_TAGS loop,
    # not a raw `tags.get(key) in values`: OSM tag casing is not guaranteed,
    # and `Park` is exactly what a lazy contributor writes.
    assert classify_site({"leisure": "Park"}) == "hardstand"
    # Same for the `parking` sub-tag against PARKING_FLAGGED: a raw
    # `tags.get("parking")` would let `Underground` slip past `covered_deck`
    # unflagged.
    assert classify_site({"amenity": "parking", "parking": "Underground"}) == "parking"
    assert site_flags({"amenity": "parking", "parking": "Underground"}) == [
        "access_unknown", "covered_deck",
    ]


def test_site_flags_say_what_a_planner_must_know_before_driving_there():
    assert site_flags({"amenity": "parking", "access": "yes"}) == []
    assert site_flags({"amenity": "parking"}) == ["access_unknown"]
    assert site_flags({"amenity": "parking", "access": "permissive"}) == ["permissive"]
    assert site_flags({"amenity": "parking", "access": "yes", "fee": "yes"}) == ["fee"]
    # A value nobody here parsed is not offered for free: it must still
    # produce access_unknown, in order, alongside a fee flag.
    assert site_flags({"amenity": "parking", "fee": "yes"}) == ["fee", "access_unknown"]


def test_site_flags_treats_an_unparsed_access_value_as_more_unknown_than_absent():
    # `employees` is a real value on real lots - it is not `yes`, and it is
    # not in ACCESS_REFUSED either, so it must still warn the planner.
    assert site_flags({"amenity": "parking", "access": "employees"}) == ["access_unknown"]
    assert site_flags({"amenity": "parking", "access": "public"}) == []


def test_site_flags_fee_recognises_more_than_fee_equals_yes():
    assert site_flags({"amenity": "parking", "access": "yes", "fee": "no"}) == []
    assert site_flags({"amenity": "parking", "access": "yes", "fee": "interval"}) == ["fee"]
    assert site_flags({"amenity": "parking", "access": "yes", "charge": "2 USD"}) == ["fee"]


def test_site_flags_fee_zero_and_none_are_also_free():
    assert site_flags({"amenity": "parking", "access": "yes", "fee": "0"}) == []
    assert site_flags({"amenity": "parking", "access": "yes", "fee": "none"}) == []
    assert site_flags({"amenity": "parking", "access": "yes", "charge": "0"}) == []
    assert site_flags({"amenity": "parking", "access": "yes", "charge": "none"}) == []


def test_site_flags_reads_parsed_parts_not_the_raw_string():
    # A trailing separator or a stacked value that is otherwise all
    # recognised parts must not read as unknown just because the whole
    # raw string does not equal "yes" or "public" verbatim.
    assert site_flags({"amenity": "parking", "access": "yes;"}) == []
    assert site_flags({"amenity": "parking", "access": "public;yes"}) == []
    # But a stacked value with a genuinely unparsed sibling is still
    # unknown, the same as `permissive;fee` below.
    assert site_flags({"amenity": "parking", "access": "public;foot"}) == [
        "access_unknown"
    ]


def test_site_flags_permissive_does_not_swallow_an_unparsed_sibling():
    # `permissive;fee` is two claims stacked: one this module understands
    # (permissive) and one it does not (fee, meaning something else here -
    # an unparsed access value, not the `fee` tag). By the module's own
    # argument for `access_unknown` - an unparsed value is MORE unknown
    # than an absent tag - the unparsed sibling must still raise the flag
    # even though `permissive` is also present.
    assert site_flags({"amenity": "parking", "access": "permissive;fee"}) == [
        "permissive",
        "access_unknown",
    ]


def test_site_flags_names_each_flagged_access_value():
    # 2026-09-09 operator ruling: each ACCESS_FLAGGED value gets its own
    # named flag rather than a shared "refused" bucket - a lot tagged
    # `access=private` and one tagged `access=customers` are different
    # phone calls for the planner. A recognised-but-flagged value is not
    # ALSO access_unknown: this module did understand it.
    #
    # `residents` used to be here too (`access_residents`). As of the SAME
    # date's residential-parking ruling, it is EXCLUDED rather than flagged
    # - see test_is_site_shaped_excludes_residential_access below, and
    # ACCESS_EXCLUDED's comment for why a resident's parking space gets a
    # different answer than a locked commercial gate.
    assert site_flags({"amenity": "parking", "access": "private"}) == ["access_private"]
    assert site_flags({"amenity": "parking", "access": "no"}) == ["access_no"]
    assert site_flags({"amenity": "parking", "access": "permit"}) == ["access_permit"]
    assert site_flags({"amenity": "parking", "access": "customers"}) == ["access_customers"]


def test_site_flags_stacked_access_names_every_flagged_part_in_order():
    assert site_flags({"amenity": "parking", "access": "private;customers"}) == [
        "access_private", "access_customers",
    ]
    # The whitespace bug this pair was originally written to guard against
    # is still worth guarding, even though the pair is offered now: a
    # leading space on the second token must not hide it from
    # ACCESS_FLAGGED.
    assert site_flags({"amenity": "parking", "access": "permissive; private"}) == [
        "permissive", "access_private",
    ]


def test_site_flags_covered_deck_for_a_flagged_parking_subtag():
    # A deck is not refused any more - `covered_deck` says what it is, and
    # the planner decides whether that rules a site out.
    assert site_flags({"amenity": "parking", "parking": "underground"}) == [
        "access_unknown", "covered_deck",
    ]
    assert site_flags(
        {"amenity": "parking", "parking": "multi-storey", "access": "yes"}
    ) == ["covered_deck"]
    assert site_flags({"amenity": "parking", "parking": "surface"}) == ["access_unknown"]


def test_covered_deck_still_flags_all_three_structural_parking_values():
    """Pinned separately from the residential-parking exclusions below, so
    the split between PARKING_FLAGGED (a parking structure - a commercial
    building, a phone call, same as ACCESS_FLAGGED) and PARKING_EXCLUDED (a
    residential garage or an on-street space - not this module's to offer at
    all) cannot drift back into one bucket. `rooftop` in particular is not
    exercised anywhere else in this file."""
    for value in ("underground", "multi-storey", "rooftop"):
        tags = {"amenity": "parking", "parking": value, "access": "yes"}
        assert is_site_shaped(tags) is True, value
        assert classify_site(tags) == "parking", value
        assert site_flags(tags) == ["covered_deck"], value


def test_is_site_shaped_excludes_residential_access():
    """2026-09-09 operator ruling: a resident's parking space is somebody's
    home, not a phone call - unlike ACCESS_FLAGGED (private/no/permit/
    customers), `access=residents` is excluded from `is_site_shaped`
    outright, not merely flagged. `classify_site` and `parking_for` both
    build on `is_site_shaped`, so excluding it there is enough to keep it
    out of both."""
    assert is_site_shaped({"amenity": "parking", "access": "residents"}) is False
    assert classify_site({"amenity": "parking", "access": "residents"}) == ""
    # Case and whitespace fold the same way `_tag`/`_access_parts` fold them
    # everywhere else in this module.
    assert is_site_shaped({"amenity": "parking", "access": "Residents"}) is False
    assert is_site_shaped(
        {"amenity": "parking", "access": "permissive; residents"}
    ) is False
    # A stacked value excludes if ANY part is residential, even alongside a
    # part (permissive) that would otherwise offer the lot on its own.
    assert classify_site({"amenity": "parking", "access": "permissive;residents"}) == ""


def test_site_flags_no_longer_names_access_residents():
    # `residents` is out of ACCESS_FLAGGED entirely now, so even called
    # directly (bypassing the `classify_site` exclusion) `site_flags` can no
    # longer produce the old flag.
    assert "access_residents" not in site_flags({"amenity": "parking", "access": "residents"})


def test_is_site_shaped_excludes_residential_parking_subtags():
    """2026-09-09 operator ruling: `garage_boxes`, `carports`, `sheds` are
    private residential garages, excluded for the same "somebody's home"
    reason as access=residents above. `carports` and `sheds` used to be in
    PARKING_FLAGGED, producing `covered_deck`; this pins that they no longer
    do, alongside the newly-excluded `garage_boxes`."""
    for value in ("garage_boxes", "carports", "sheds"):
        tags = {"amenity": "parking", "parking": value}
        assert is_site_shaped(tags) is False, value
        assert classify_site(tags) == "", value
    # Case-folding reaches the `parking` sub-tag here too.
    assert is_site_shaped({"amenity": "parking", "parking": "Carports"}) is False


def test_is_site_shaped_excludes_on_street_parking():
    """`lane` and `street_side` are on-street parking spaces along a road -
    not a lot a team can stage in at all. There was never a flag for these;
    they are simply not site-shaped."""
    for value in ("lane", "street_side"):
        tags = {"amenity": "parking", "parking": value}
        assert is_site_shaped(tags) is False, value
        assert classify_site(tags) == "", value


@pytest.mark.parametrize("tags", [
    {"amenity": "parking", "access": "residents"},
    {"amenity": "parking", "parking": "garage_boxes"},
    {"amenity": "parking", "parking": "carports"},
    {"amenity": "parking", "parking": "sheds"},
    {"amenity": "parking", "parking": "lane"},
    {"amenity": "parking", "parking": "street_side"},
], ids=[
    "access_residents", "garage_boxes", "carports", "sheds", "lane", "street_side",
])
def test_parking_for_emits_no_row_for_an_excluded_value(tmp_path, tags):
    payload = {"elements": [{
        "type": "way", "id": 900, "tags": tags,
        "geometry": [
            {"lat": 37.3700, "lon": -122.2290}, {"lat": 37.3700, "lon": -122.2280},
            {"lat": 37.3710, "lon": -122.2280}, {"lat": 37.3710, "lon": -122.2290},
            {"lat": 37.3700, "lon": -122.2290},
        ],
    }]}
    sites = parking_for(BBOX, cache_dir=tmp_path, get_json=lambda query: payload)
    assert sites == []


def test_the_overpass_query_asks_for_the_new_classes():
    query = build_overpass_query(-3403, 1122)
    for filter_ in ('way["highway"]', 'way["waterway"]', 'way["natural"~"^(coastline|water)$"]',
                    'way["power"="line"]', 'way["railway"]', 'way["barrier"~"^(fence|wall)$"]',
                    'way["man_made"="pipeline"]', 'way["landuse"="reservoir"]'):
        assert filter_ in query, filter_


def test_the_query_asks_for_parking_and_public_hardstand():
    query = build_overpass_query(0, 0)

    # A lot is mapped as an area and, in places, as a bare node.
    assert 'node["amenity"="parking"]' in query
    assert 'way["amenity"="parking"]' in query
    assert "community_centre|place_of_worship|school" in query
    assert "park|pitch|sports_centre" in query
    # landuse and tourism are single-value HARDSTAND_TAGS entries: no
    # alternation needed, so the helper emits a plain equality for each.
    assert 'way["landuse"="recreation_ground"]' in query
    assert 'way["tourism"="picnic_site"]' in query
    # The routes are still the point of this query.
    assert 'way["highway"]' in query
    # Version 5: gates, as nodes, for the egress road facts.
    assert 'node["barrier"~"^(gate|lift_gate|swing_gate)$"]' in query
    # A 400 is never retried (see OVERPASS_RETRY_STATUS): a malformed query
    # is not a slow answer, it is no answer, on the first endpoint, mid-incident.
    assert query.count("(") == query.count(")")
    assert query.endswith(");out geom;")


def test_the_query_version_moved_so_old_tiles_are_not_read_as_complete():
    from sar_geo_mcp.terrain.linear import OSM_QUERY_VERSION

    # A version 4 tile holds no gates and its routes carry no road class,
    # access or tracktype. Read as if it did, it says "no gate on this
    # road", which is a different claim entirely - see
    # test_version_5_is_the_road_tags_and_gates_version.
    assert OSM_QUERY_VERSION == 5


def test_tile_files_carry_the_query_version_so_old_caches_refetch(tmp_path):
    from sar_geo_mcp.terrain.linear import OSM_QUERY_VERSION, fetch_osm_tile

    calls = []
    payload = fetch_osm_tile(-3403, 1122, tmp_path, get_json=lambda q: calls.append(q) or {"elements": []})
    assert payload == {"elements": []}
    assert (tmp_path / f"osm_tile_v{OSM_QUERY_VERSION}_c-3403_r1122.json").is_file()
    # A v1 file on disk is not read.
    (tmp_path / "osm_tile_c-3403_r1122.json").write_text('{"elements": [1]}', encoding="utf-8")
    fetch_osm_tile(-3403, 1122, tmp_path, get_json=lambda q: calls.append(q) or {"elements": []})
    assert len(calls) == 1, "the v2 file was cached; the v1 file was ignored"


def test_a_closed_water_way_contributes_its_ring_as_a_water_edge(tmp_path):
    ring = [{"lat": 37.3700, "lon": -122.2295}, {"lat": 37.3700, "lon": -122.2280},
            {"lat": 37.3710, "lon": -122.2280}, {"lat": 37.3710, "lon": -122.2295},
            {"lat": 37.3700, "lon": -122.2295}]
    payload = {"elements": [{"type": "way", "id": 9, "tags": {"natural": "water"},
                             "geometry": ring}]}
    routes = routes_for(BBOX, cache_dir=tmp_path, get_json=lambda q: payload)
    assert routes and routes[0]["properties"]["route_type"] == "WATER_EDGE"
    assert routes[0]["geometry"]["type"] in ("LineString", "MultiLineString")


def test_a_park_polygon_is_not_a_route(tmp_path):
    payload = fake_overpass_payload()
    payload["elements"].append({
        "type": "way", "id": 77, "tags": {"leisure": "park", "name": "Windy Hill"},
        "geometry": [
            {"lat": 37.3700, "lon": -122.2290}, {"lat": 37.3700, "lon": -122.2280},
            {"lat": 37.3710, "lon": -122.2280}, {"lat": 37.3710, "lon": -122.2290},
            {"lat": 37.3700, "lon": -122.2290},
        ],
    })

    routes = routes_for(BBOX, cache_dir=tmp_path, get_json=lambda query: payload)

    assert [r["properties"]["name"] for r in routes] == ["Betsy Crowder Trail", None]
    assert all(r["properties"]["route_type"] != "OTHER" for r in routes)


def test_a_service_area_is_both_a_site_and_a_route(tmp_path):
    """`highway=services` is a motorway service area: hardstand a team could
    park on AND a way they can drive in on. The guard in `routes_for` tests
    BOTH classifiers for exactly this element - dropping on `classify_site`
    alone would take it out of the routes, where it belongs."""
    payload = {"elements": [{
        "type": "way", "id": 88, "tags": {"highway": "services", "name": "Woodside services"},
        "geometry": [
            {"lat": 37.3700, "lon": -122.2295},
            {"lat": 37.3702, "lon": -122.2280},
        ],
    }]}

    routes = routes_for(BBOX, cache_dir=tmp_path, get_json=lambda query: payload)

    assert [r["properties"]["name"] for r in routes] == ["Woodside services"]
    assert routes[0]["properties"]["route_type"] == "ROAD_OTHER"


def test_a_refused_site_is_not_a_route(tmp_path):
    """A private lot and a private-access park are OFFERED by
    `classify_site` now (2026-09-09 operator ruling) - but `routes_for`'s
    guard asks `is_site_shaped`, the structural question, not
    `classify_site`'s offer question, and access never enters that question.
    Both are exactly as site-shaped as their public counterparts, so both
    must still vanish from the routes here: this is the gap that used to let
    a gated private estate reach the planner as a route shape, and the two
    questions staying separate is what keeps it from reopening."""
    payload = fake_overpass_payload()
    payload["elements"].extend([
        {
            "type": "way", "id": 78,
            "tags": {"amenity": "parking", "access": "private", "name": "Private Lot"},
            "geometry": [
                {"lat": 37.3700, "lon": -122.2290}, {"lat": 37.3700, "lon": -122.2280},
                {"lat": 37.3710, "lon": -122.2280}, {"lat": 37.3710, "lon": -122.2290},
                {"lat": 37.3700, "lon": -122.2290},
            ],
        },
        {
            "type": "way", "id": 79,
            "tags": {"leisure": "park", "access": "private", "name": "Gated Park"},
            "geometry": [
                {"lat": 37.3701, "lon": -122.2291}, {"lat": 37.3701, "lon": -122.2281},
                {"lat": 37.3711, "lon": -122.2281}, {"lat": 37.3711, "lon": -122.2291},
                {"lat": 37.3701, "lon": -122.2291},
            ],
        },
    ])

    routes = routes_for(BBOX, cache_dir=tmp_path, get_json=lambda query: payload)

    names = [r["properties"]["name"] for r in routes]
    assert "Private Lot" not in names
    assert "Gated Park" not in names
    assert all(r["properties"]["route_type"] != "OTHER" for r in routes)


def test_a_residential_lot_is_dropped_from_parking_but_falls_through_to_an_other_route(tmp_path):
    """2026-09-09 operator ruling: `access=residents` is EXCLUDED from
    `is_site_shaped`, not merely flagged like `access=private` above - so
    `routes_for`'s `is_site_shaped` guard no longer trips for it either.
    That is a deliberate, examined consequence, not an oversight: an
    excluded residential lot is no longer "site-shaped" at all, so it is
    just an ordinary untagged-for-our-purposes closed way again, and
    `routes_for` treats it exactly as it treated any such way before
    version 3 ever taught this module to recognise parking - it comes back
    as a class-OTHER route, the same as the park in
    `test_a_park_polygon_is_not_a_route` would if that park were not
    site-shaped."""
    payload = {"elements": [{
        "type": "way", "id": 555,
        "tags": {"amenity": "parking", "access": "residents", "name": "Apartment Lot"},
        "geometry": [
            {"lat": 37.3700, "lon": -122.2290}, {"lat": 37.3700, "lon": -122.2280},
            {"lat": 37.3710, "lon": -122.2280}, {"lat": 37.3710, "lon": -122.2290},
            {"lat": 37.3700, "lon": -122.2290},
        ],
    }]}

    routes = routes_for(BBOX, cache_dir=tmp_path, get_json=lambda query: payload)
    sites = parking_for(BBOX, cache_dir=tmp_path, get_json=lambda query: payload)

    assert sites == []
    assert [r["properties"]["name"] for r in routes] == ["Apartment Lot"]
    assert routes[0]["properties"]["route_type"] == "OTHER"


def test_a_parking_node_yields_no_route(tmp_path):
    """Nodes are skipped by `routes_for`'s `element.get("type") != "way"`
    check - correct, since a route needs a line, not a point - but nothing
    pinned that behaviour until now, and a later change is about to read
    these same nodes for a different purpose."""
    payload = {"elements": [{
        "type": "node", "id": 55, "tags": {"amenity": "parking"},
        "lat": 37.3701, "lon": -122.2291,
    }]}

    routes = routes_for(BBOX, cache_dir=tmp_path, get_json=lambda query: payload)

    assert routes == []


# --- parking_for: the shapes a planner can actually stage on -----------------

def parking_payload() -> dict:
    """A mapped lot, a bare node, a park, and a private lot - the four
    shapes/tags that decide what flags a row carries. All four earn a row
    now (2026-09-09 operator ruling); the private lot used to be the one
    case that was refused outright."""
    return {
        "elements": [
            {"type": "way", "id": 10,
             "tags": {"amenity": "parking", "name": "Windy Hill lot",
                      "surface": "asphalt", "capacity": "40", "fee": "yes"},
             "geometry": [
                 {"lat": 37.3700, "lon": -122.2290}, {"lat": 37.3700, "lon": -122.2285},
                 {"lat": 37.3705, "lon": -122.2285}, {"lat": 37.3705, "lon": -122.2290},
                 {"lat": 37.3700, "lon": -122.2290},
             ]},
            {"type": "node", "id": 11, "lat": 37.3710, "lon": -122.2270,
             "tags": {"amenity": "parking", "access": "yes"}},
            {"type": "way", "id": 12, "tags": {"leisure": "park", "name": "The Green"},
             "geometry": [
                 {"lat": 37.3690, "lon": -122.2300}, {"lat": 37.3690, "lon": -122.2295},
                 {"lat": 37.3695, "lon": -122.2295}, {"lat": 37.3695, "lon": -122.2300},
                 {"lat": 37.3690, "lon": -122.2300},
             ]},
            {"type": "way", "id": 13, "tags": {"amenity": "parking", "access": "private"},
             "geometry": [
                 {"lat": 37.3700, "lon": -122.2260}, {"lat": 37.3700, "lon": -122.2255},
                 {"lat": 37.3705, "lon": -122.2255}, {"lat": 37.3705, "lon": -122.2260},
                 {"lat": 37.3700, "lon": -122.2260},
             ]},
        ]
    }


def test_parking_for_reads_lots_nodes_and_hardstand_from_the_same_tiles(tmp_path):
    sites = parking_for(BBOX, cache_dir=tmp_path, get_json=lambda query: parking_payload())

    by_id = {site["id"]: site for site in sites}
    assert set(by_id) == {
        "parking-w10", "parking-n11", "hardstand-w12", "parking-w13",
    }

    lot = by_id["parking-w10"]
    assert lot["kind"] == "parking"
    assert lot["geometry"]["type"] == "Polygon"
    assert lot["properties"]["name"] == "Windy Hill lot"
    assert lot["properties"]["surface"] == "asphalt"
    assert lot["properties"]["capacity"] == "40"
    assert lot["properties"]["flags"] == ["fee", "access_unknown"]

    node = by_id["parking-n11"]
    assert node["geometry"]["type"] == "Point"
    assert "node_only" in node["properties"]["flags"]

    assert by_id["hardstand-w12"]["properties"]["flags"] == ["access_unknown",
                                                             "parking_unverified"]

    # id 13 is the private lot: 2026-09-09 operator ruling is that this is
    # OFFERED, not hidden - a private lot may still be the right staging
    # area after a phone call to the owner. It carries its own row, flagged
    # access_private, not the access_unknown catch-all.
    private_lot = by_id["parking-w13"]
    assert private_lot["kind"] == "parking"
    assert private_lot["properties"]["flags"] == ["access_private"]


def test_parking_for_shares_the_tile_cache_with_the_routes(tmp_path):
    calls = []

    def fake_get(query: str) -> dict:
        calls.append(query)
        return parking_payload()

    routes_for(BBOX, cache_dir=tmp_path, get_json=fake_get)
    parking_for(BBOX, cache_dir=tmp_path, get_json=fake_get)

    # One fetch, two readings of it: parking must cost no extra Overpass
    # traffic at all.
    assert len(calls) == 1


def test_parking_for_excludes_a_lot_entirely_outside_the_bbox(tmp_path):
    """The `intersects(clip)` check runs on the metric shape, after the
    WGS84-to-3857 transform. A lot several kilometres north of the AOI must
    not sneak through as "touches the box" just because it was fetched in
    the same tile batch."""

    payload = {"elements": [{
        "type": "way", "id": 200, "tags": {"amenity": "parking", "name": "Far Lot"},
        "geometry": [
            {"lat": 37.4000, "lon": -122.2400}, {"lat": 37.4000, "lon": -122.2395},
            {"lat": 37.4005, "lon": -122.2395}, {"lat": 37.4005, "lon": -122.2400},
            {"lat": 37.4000, "lon": -122.2400},
        ],
    }]}

    sites = parking_for(BBOX, cache_dir=tmp_path, get_json=lambda query: payload)

    assert sites == []


def test_parking_for_dedupes_a_lot_that_spans_two_tiles(tmp_path):
    """A way near a tile boundary is returned by Overpass in every tile it
    touches. `routes_for` already dedupes on this; `parking_for` must too,
    or a lot straddling a boundary would offer a planner the same lot
    twice under two different rows."""

    calls = []

    def fake_get(query: str) -> dict:
        calls.append(query)
        return {"elements": [{
            "type": "way", "id": 20, "tags": {"amenity": "parking", "name": "Border Lot"},
            "geometry": [
                {"lat": 0.0001, "lon": 0.0100}, {"lat": 0.0001, "lon": 0.0200},
                {"lat": 0.0005, "lon": 0.0200}, {"lat": 0.0005, "lon": 0.0100},
                {"lat": 0.0001, "lon": 0.0100},
            ],
        }]}

    bbox = (100.0, 0.0, 4100.0, 100.0)  # spans exactly two OSM tiles
    assert len(tiles_covering_osm(bbox)) == 2

    sites = parking_for(bbox, cache_dir=tmp_path, get_json=fake_get)

    assert len(calls) == 2  # both tiles really were fetched
    assert [site["id"] for site in sites] == ["parking-w20"]


def test_parking_for_keeps_the_complete_copy_when_the_first_tile_is_broken(tmp_path):
    """A lot spanning a tile boundary can arrive broken in one tile's
    response and complete in the other's - a half-digitised way, or geometry
    Overpass clipped oddly to that tile's query box. `parking_for` marks an
    element `seen` only once it has actually earned a row (open ring and
    missing-coordinate node both `continue` before that line), so the broken
    first copy must not block the complete second copy from being returned.

    This is the case `test_parking_for_dedupes_a_lot_that_spans_two_tiles`
    cannot catch: that test uses two IDENTICAL complete copies, which pass
    under either placement of `seen.add`. Here the "broken" copy is an OPEN
    ring - `coordinates[0] != coordinates[-1]` - so it is dropped by the
    shape check rather than turned into a row, and only the closed ring
    earns one.

    Which tile serves the broken copy is fixed by the tile's OWN bbox
    inside the query string, not by call order: keying on `len(calls)`
    instead would mean that if tile iteration order ever changed, the
    complete copy could arrive first and this test would pass without
    exercising the `seen.add` placement it exists to pin."""

    from sar_geo_mcp.terrain.linear import _tile_bounds_wgs84

    closed_ring = [
        {"lat": 0.0001, "lon": 0.0100}, {"lat": 0.0001, "lon": 0.0200},
        {"lat": 0.0005, "lon": 0.0200}, {"lat": 0.0005, "lon": 0.0100},
        {"lat": 0.0001, "lon": 0.0100},
    ]
    open_ring = closed_ring[:-1]  # same lot, missing its closing point

    bbox = (100.0, 0.0, 4100.0, 100.0)  # spans exactly two OSM tiles
    tiles = tiles_covering_osm(bbox)
    assert len(tiles) == 2

    broken_col, broken_row = min(tiles)
    south, west, north, east = _tile_bounds_wgs84(broken_col, broken_row)
    broken_area = f"{south:.6f},{west:.6f},{north:.6f},{east:.6f}"

    calls = []

    def fake_get(query: str) -> dict:
        calls.append(query)
        geometry = open_ring if broken_area in query else closed_ring
        return {"elements": [{
            "type": "way", "id": 20, "tags": {"amenity": "parking", "name": "Border Lot"},
            "geometry": geometry,
        }]}

    sites = parking_for(bbox, cache_dir=tmp_path, get_json=fake_get)

    assert len(calls) == 2  # both tiles really were fetched
    assert [site["id"] for site in sites] == ["parking-w20"]
    assert sites[0]["geometry"]["type"] == "Polygon"


def test_parking_for_repairs_a_self_intersecting_lot_without_crashing(tmp_path):
    """A badly-digitised lot (a bowtie) is invalid shapely geometry.
    `buffer(0)` repairs it, but can turn one polygon into a `MultiPolygon` -
    this pins that the result still comes out as one row, with geometry that
    survives a `mapping()`/`json.dumps()` round trip for a later task to
    read."""

    lon0, lat0, step = -122.2290, 37.3700, 0.0005
    bowtie = [
        (lon0, lat0), (lon0 + step, lat0 + step), (lon0 + 2 * step, lat0),
        (lon0 + 2 * step, lat0 + 2 * step), (lon0 + step, lat0 + step),
        (lon0, lat0 + 2 * step), (lon0, lat0),
    ]
    payload = {"elements": [{
        "type": "way", "id": 99, "tags": {"amenity": "parking", "name": "Bowtie Lot"},
        "geometry": [{"lat": lat, "lon": lon} for lon, lat in bowtie],
    }]}

    sites = parking_for(BBOX, cache_dir=tmp_path, get_json=lambda query: payload)

    assert len(sites) == 1
    geometry = sites[0]["geometry"]
    assert geometry["type"] in ("Polygon", "MultiPolygon")
    json.dumps(geometry)  # must serialise cleanly either way

    # The type check above passes even on an UNREPAIRED bowtie - it is still
    # a Polygon and still serialises. What actually pins the `buffer(0)`
    # repair is validity: a self-intersecting bowtie is invalid OGC geometry,
    # repaired or not, so rebuild it and check that property directly.
    from shapely.geometry import shape as shapely_shape

    assert shapely_shape(geometry).is_valid


def test_parking_for_keeps_both_lobes_of_an_oppositely_wound_bowtie(tmp_path):
    """The bowtie above has its two lobes wind the SAME way, so `buffer(0)`
    and `make_valid` repair it identically and the test above cannot tell
    them apart. Here the two lobes wind OPPOSITE ways - the case
    `buffer(0)`'s winding-rule repair gets wrong: it keeps only one lobe,
    silently reporting a lot at half its real size. `make_valid` repairs by
    geometry and keeps both.

    Both repairs are computed independently here, in the same WGS84-degree
    space `parking_for` repairs in, then projected to 3857 the same way it
    does, so the comparison is exact rather than a coincidence of
    tolerance."""
    from shapely.geometry import Polygon
    from shapely.ops import transform as shapely_transform, unary_union
    from shapely.validation import make_valid
    from pyproj import Transformer

    lon0, lat0, step = -122.2290, 37.3700, 0.0005
    bowtie = [
        (lon0, lat0), (lon0 + 2 * step, lat0 + 2 * step), (lon0 + 2 * step, lat0),
        (lon0, lat0 + 2 * step), (lon0, lat0),
    ]
    raw = Polygon(bowtie)
    assert not raw.is_valid

    payload = {"elements": [{
        "type": "way", "id": 100, "tags": {"amenity": "parking", "name": "Opposite Bowtie"},
        "geometry": [{"lat": lat, "lon": lon} for lon, lat in bowtie],
    }]}

    sites = parking_for(BBOX, cache_dir=tmp_path, get_json=lambda query: payload)
    assert len(sites) == 1

    from shapely.geometry import shape as shapely_shape

    returned_area = shapely_shape(sites[0]["geometry"]).area

    to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    def area_3857(geom) -> float:
        return shapely_transform(to_3857.transform, geom).area

    buffer0_area = area_3857(raw.buffer(0))
    made = make_valid(raw)
    make_valid_area = area_3857(unary_union(
        [part for part in getattr(made, "geoms", [made])
         if part.geom_type in ("Polygon", "MultiPolygon")]))

    assert make_valid_area == pytest.approx(2 * buffer0_area, rel=1e-9)
    assert returned_area == pytest.approx(make_valid_area, rel=1e-9)
    assert returned_area != pytest.approx(buffer0_area, rel=1e-2)


def test_parking_for_ids_do_not_collide_between_a_node_and_a_way(tmp_path):
    """OSM ids are unique only PER TYPE: node 12 and way 12 are unrelated
    real-world elements, and both can legitimately carry `amenity=parking`.
    The `n`/`w` prefix in `f"{kind}-{osm_type[0]}{osm_id}"` is what is
    supposed to keep their rows apart on a map where the id becomes a
    marker id. This is a characterisation test - it passes against the
    current code as written; it exists to catch a future change to the id
    format that reintroduces the collision, not to prove a fix."""

    payload = {"elements": [
        {"type": "node", "id": 12, "lat": 37.3710, "lon": -122.2270,
         "tags": {"amenity": "parking", "access": "yes"}},
        {"type": "way", "id": 12, "tags": {"amenity": "parking", "access": "yes"},
         "geometry": [
             {"lat": 37.3700, "lon": -122.2290}, {"lat": 37.3700, "lon": -122.2285},
             {"lat": 37.3705, "lon": -122.2285}, {"lat": 37.3705, "lon": -122.2290},
             {"lat": 37.3700, "lon": -122.2290},
         ]},
    ]}

    sites = parking_for(BBOX, cache_dir=tmp_path, get_json=lambda query: payload)

    by_id = {site["id"]: site for site in sites}
    assert set(by_id) == {"parking-n12", "parking-w12"}
    assert by_id["parking-n12"]["geometry"]["type"] == "Point"
    assert by_id["parking-w12"]["geometry"]["type"] == "Polygon"


def test_a_closed_service_area_appears_in_both_routes_and_parking(tmp_path):
    """`highway=services` is `is_site_shaped` per Task 3's guard AND a
    `classify_site` hardstand: a motorway rest area is ground a search can
    both drive in on and park at. `routes_for` and `parking_for` read the
    same cached tile independently, and neither one's presence should
    exclude the other."""
    payload = {"elements": [{
        "type": "way", "id": 88, "tags": {"highway": "services", "name": "Woodside services"},
        "geometry": [
            {"lat": 37.3700, "lon": -122.2295}, {"lat": 37.3700, "lon": -122.2285},
            {"lat": 37.3705, "lon": -122.2285}, {"lat": 37.3705, "lon": -122.2295},
            {"lat": 37.3700, "lon": -122.2295},
        ],
    }]}

    routes = routes_for(BBOX, cache_dir=tmp_path, get_json=lambda query: payload)

    sites = parking_for(BBOX, cache_dir=tmp_path, get_json=lambda query: payload)

    assert routes and routes[0]["properties"]["route_type"] == "ROAD_OTHER"
    assert sites and sites[0]["id"] == "hardstand-w88"
    assert sites[0]["properties"]["flags"] == ["access_unknown", "parking_unverified"]


def test_ingress_counts_only_walkable_classes():
    from shapely.geometry import box

    square = box(0, 0, 100, 100)
    trail = {"id": "route_0", "kind": "route",
             "geometry": {"type": "LineString", "coordinates": [[-10, 50], [110, 50]]},
             "properties": {"route_type": "TRAIL", "name": None}}
    creek = {"id": "route_1", "kind": "route",
             "geometry": {"type": "LineString", "coordinates": [[50, -10], [50, 110]]},
             "properties": {"route_type": "WATERWAY", "name": None}}
    fence = {"id": "route_2", "kind": "route",
             "geometry": {"type": "LineString", "coordinates": [[-10, 20], [110, 20]]},
             "properties": {"route_type": "BARRIER", "name": None}}
    points = ingress_candidates([trail, creek, fence], square)
    assert len(points) == 2
    assert {p["properties"]["route_id"] for p in points} == {"route_0"}


# --- Overpass resilience: mirrors, retries, and an honest failure -------------
#
# Everything here drives the real `_http_post` through an httpx.MockTransport,
# so the retry policy is exercised without a socket and without a real sleep.

import httpx

from sar_geo_mcp.terrain import linear as linear_module
from sar_geo_mcp.terrain.linear import (
    OVERPASS_ATTEMPTS_PER_ENDPOINT,
    OVERPASS_BUDGET_S,
    OVERPASS_URL,
    OVERPASS_URLS,
    OverpassUnavailable,
    _http_post,
    overpass_endpoints,
)

OK_PAYLOAD = {"elements": [{"type": "way", "id": 7, "tags": {"highway": "path"},
                            "geometry": [{"lat": 37.37, "lon": -122.23},
                                         {"lat": 37.372, "lon": -122.225}]}]}


def _record_sleeps():
    waits: list[float] = []
    return waits, waits.append


def _factory(handler):
    """A client_factory that answers every request from `handler`."""
    def build() -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(handler))
    return build


def test_the_first_endpoint_is_still_overpass_api_de():
    assert OVERPASS_URLS[0] == "https://overpass-api.de/api/interpreter"
    assert OVERPASS_URL == OVERPASS_URLS[0]
    # Two machines, not three names - see
    # test_the_mirror_list_holds_no_two_names_for_one_machine below for why
    # overpass.kumi.systems is gone: it was a CNAME of overpass.private.coffee,
    # not a second server.
    assert len(OVERPASS_URLS) >= 2


def test_the_mirror_list_holds_no_two_names_for_one_machine():
    """overpass.kumi.systems is a CNAME of overpass.private.coffee - both
    resolve to flanders.servers.private.coffee, 193.219.97.30, checked
    2026-09-20. Listing both made the list read three deep while it was two
    deep, and sent up to six hanging attempts to one dead host inside a
    single 90 s budget."""
    hosts = [httpx.URL(url).host for url in OVERPASS_URLS]
    assert "overpass.kumi.systems" not in hosts
    assert len(hosts) == len(set(hosts))


def test_the_endpoint_list_is_overridable_from_the_environment(monkeypatch):
    monkeypatch.setenv("ZA_OVERPASS_URLS",
                       "https://osm.agency.example/api/interpreter , https://b.example/i ")
    assert overpass_endpoints() == ("https://osm.agency.example/api/interpreter",
                                    "https://b.example/i")
    monkeypatch.delenv("ZA_OVERPASS_URLS")
    assert overpass_endpoints() == OVERPASS_URLS


def test_the_endpoint_list_drops_exact_repeats_from_the_environment(monkeypatch):
    """An agency pasting a list twice gets one of each. Aliased HOSTS are
    not detectable here without DNS - see the module comment."""
    monkeypatch.setenv("ZA_OVERPASS_URLS",
                       "https://a.example/i, https://a.example/i, https://b.example/i")
    assert overpass_endpoints() == ("https://a.example/i", "https://b.example/i")


def test_assumptions_record_the_budget_and_the_endpoints():
    assert LINEAR_ASSUMPTIONS["overpass_budget_s"] == OVERPASS_BUDGET_S
    # Hosts only, never the URL (0.17.1): a configured one may carry a key.
    assert LINEAR_ASSUMPTIONS["overpass_endpoints"][0] == httpx.URL(OVERPASS_URLS[0]).host
    assert LINEAR_ASSUMPTIONS["overpass_attempts_per_endpoint"] >= 2


def test_assumptions_do_not_promise_three_attempts_to_a_hung_host():
    """Provenance is read by someone deciding how much "no trails here"
    is worth. A bare "3 attempts per endpoint" overstates what a HUNG
    endpoint actually got - since a timeout abandons the mirror after one
    attempt, three is the ceiling for a retryable status only. The field
    has to say which, or it reads as three tries against a host that was
    asked exactly once."""
    recorded = LINEAR_ASSUMPTIONS["overpass_attempts_per_endpoint"]
    assert recorded == OVERPASS_ATTEMPTS_PER_ENDPOINT
    note = LINEAR_ASSUMPTIONS["overpass_attempts_note"]
    assert "timeout" in note.lower()
    # Not just the word - the note must say the hung host gets ONE attempt,
    # which is the number the bare field hides.
    assert "one attempt" in note.lower()


def test_a_429_on_the_first_mirror_falls_through_to_the_second():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        if request.url.host == "overpass-api.de":
            return httpx.Response(429)
        return httpx.Response(200, json=OK_PAYLOAD)

    waits, sleep = _record_sleeps()
    payload = _http_post("[out:json];", client_factory=_factory(handler), sleep=sleep)

    assert payload == OK_PAYLOAD
    assert seen[0] == "overpass-api.de"
    assert seen[-1] != "overpass-api.de", "it moved on to a mirror"
    assert seen.count("overpass-api.de") == OVERPASS_ATTEMPTS_PER_ENDPOINT
    assert all(w >= 0 for w in waits)


def test_retry_after_is_honoured_without_actually_sleeping():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "1"})
        return httpx.Response(200, json=OK_PAYLOAD)

    waits, sleep = _record_sleeps()
    payload = _http_post("[out:json];", client_factory=_factory(handler), sleep=sleep)

    assert payload == OK_PAYLOAD
    assert waits == [1.0], "the server's own Retry-After, not a backoff guess"


def test_an_absurd_retry_after_is_capped():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, headers={"Retry-After": "86400"})

    waits, sleep = _record_sleeps()
    with pytest.raises(OverpassUnavailable):
        _http_post("[out:json];", client_factory=_factory(handler), sleep=sleep)

    assert waits and max(waits) <= 30.0


def test_every_endpoint_failing_raises_with_the_last_status():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        return httpx.Response(429)

    waits, sleep = _record_sleeps()
    with pytest.raises(OverpassUnavailable) as caught:
        _http_post("[out:json];", client_factory=_factory(handler), sleep=sleep)

    message = str(caught.value)
    assert "429" in message and "Too Many Requests" in message
    assert "OpenStreetMap" in message
    assert "Trails and roads will be missing" in message
    assert caught.value.status == 429
    assert caught.value.endpoint == overpass_endpoints()[-1]
    assert set(seen) == {httpx.URL(u).host for u in overpass_endpoints()}


def test_a_timeout_is_reported_by_name_and_not_as_a_status():
    """A hung or unroutable host is named by its exception rather than by a
    status code, because there is no status - the server never answered.

    It is NOT retried on the same endpoint: since 2026-09-21 a timeout
    abandons that mirror after one attempt and moves to the next, which is
    why this test no longer carries "is_retried" in its name. What a
    timeout still gets is an honest report; `status` stays None so a caller
    can tell "nobody answered" from "the server said no"."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("no route", request=request)

    waits, sleep = _record_sleeps()
    with pytest.raises(OverpassUnavailable) as caught:
        _http_post("[out:json];", client_factory=_factory(handler), sleep=sleep)

    assert "ConnectTimeout" in str(caught.value)
    assert caught.value.status is None


def test_a_400_is_not_retried_because_the_query_is_our_bug():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        return httpx.Response(400, text="parse error")

    waits, sleep = _record_sleeps()
    with pytest.raises(httpx.HTTPStatusError):
        _http_post("[out:json];broken", client_factory=_factory(handler), sleep=sleep)

    assert len(seen) == 1, "a malformed query is not an availability problem"
    assert waits == []


def test_the_wall_clock_budget_stops_the_loop():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        return httpx.Response(503)

    clock = {"now": 0.0}

    def monotonic() -> float:
        clock["now"] += 50.0
        return clock["now"]

    waits, sleep = _record_sleeps()
    with pytest.raises(OverpassUnavailable) as caught:
        _http_post("[out:json];", client_factory=_factory(handler), sleep=sleep,
                   monotonic=monotonic)

    assert "budget" in str(caught.value)
    assert len(seen) < len(overpass_endpoints()) * OVERPASS_ATTEMPTS_PER_ENDPOINT


def test_a_hanging_first_mirror_still_leaves_the_second_a_turn():
    """The failure this came from: a host that completes TLS and then never
    answers. A timeout is not "busy, try again" - it is "this host is not
    answering" - so it does not get the same three-attempt treatment a 429
    gets (see test_a_429_on_the_first_mirror_is_retried_not_abandoned): one
    hang moves straight to the next mirror, so the survivor is handed a real
    share of the budget rather than a scrap thinned out by two more wasted
    attempts against a host that was never going to answer. Real tile
    fetches against overpass-api.de, measured 2026-09-21, ran up to ~66 s -
    a scrap under ~11 s (three shrinking attempts on a dead first mirror)
    cannot complete one; the whole-remaining-budget share this test pins
    can."""
    seen: list[str] = []
    given_timeouts: dict[str, float] = {}
    clock = {"now": 0.0}

    def monotonic() -> float:
        return clock["now"]

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        given_timeouts[request.url.host] = request.extensions["timeout"]["read"]
        if request.url.host == "overpass-api.de":
            # A hang, not a fast failure: the clock only moves because the
            # read timeout THIS request was actually given finally expired -
            # exactly like flanders.servers.private.coffee on 2026-09-21,
            # which completed TLS in 0.37 s and then never answered.
            clock["now"] += request.extensions["timeout"]["read"]
            raise httpx.ReadTimeout("no answer", request=request)
        return httpx.Response(200, json=OK_PAYLOAD)

    waits, sleep = _record_sleeps()
    payload = _http_post("[out:json];", client_factory=_factory(handler), sleep=sleep,
                          monotonic=monotonic)

    assert payload == OK_PAYLOAD
    # Reached once, not three times: a hang is abandoned after a single
    # attempt rather than retried like a busy status would be.
    assert seen.count("overpass-api.de") == 1
    second_host = httpx.URL(overpass_endpoints()[1]).host
    assert second_host in seen
    # Not just reached - given a timeout large enough to be a real attempt.
    # With the first mirror abandoned after one hang instead of three, the
    # second is the only endpoint left, so it gets the WHOLE remaining
    # budget rather than a share thinned out by wasted attempts - well over
    # the 2-mirror-default floor of 30 s this asserts, and over the ~11 s a
    # naive per-attempt-only share would have left it.
    assert given_timeouts[second_host] >= 30.0


def test_a_429_on_the_first_mirror_is_retried_not_abandoned():
    """A status means "busy, try again" - unlike a timeout (see
    test_a_hanging_first_mirror_still_leaves_the_second_a_turn, which moves
    to the next mirror after a single attempt), a busy host keeps its full
    attempt-and-backoff allowance before the next mirror is tried. This
    pins that the fix for a hanging host does not regress into abandoning a
    merely-busy one on its first 429 too."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        if request.url.host == "overpass-api.de":
            return httpx.Response(429)
        return httpx.Response(200, json=OK_PAYLOAD)

    waits, sleep = _record_sleeps()
    payload = _http_post("[out:json];", client_factory=_factory(handler), sleep=sleep)

    assert payload == OK_PAYLOAD
    assert seen.count("overpass-api.de") == OVERPASS_ATTEMPTS_PER_ENDPOINT
    # One backoff sleep between each of the three attempts on the busy
    # host - two waits, not the zero a hang now produces.
    assert len(waits) == OVERPASS_ATTEMPTS_PER_ENDPOINT - 1


def test_the_query_asks_for_distribution_lines_and_says_so_in_its_version():
    from sar_geo_mcp.terrain.linear import (OSM_QUERY_VERSION, OSM_QUERY_VERSION_MINOR_LINES,
                                            build_overpass_query)

    # Distribution lines arrived at 4; the query has moved on since (5, for
    # road tags and gates) and still asks for them.
    assert OSM_QUERY_VERSION == 5 and OSM_QUERY_VERSION_MINOR_LINES == 4
    assert 'way["power"="minor_line"]' in build_overpass_query(0, 0)


def test_a_distribution_line_is_a_barrier_with_its_power_tag(tmp_path):
    payload = {"elements": [
        {"type": "way", "id": 7, "tags": {"power": "minor_line"},
         "geometry": [{"lat": 37.3690, "lon": -122.2300},
                      {"lat": 37.3745, "lon": -122.2243}]}]}
    routes = routes_for(BBOX, cache_dir=tmp_path, get_json=lambda query: payload)
    assert routes[0]["properties"]["route_type"] == "BARRIER"
    assert routes[0]["properties"]["power"] == "minor_line"


def test_a_successful_fetch_still_writes_the_cache_tile(tmp_path):
    from sar_geo_mcp.terrain.linear import OSM_QUERY_VERSION, fetch_osm_tile

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=OK_PAYLOAD)

    waits, sleep = _record_sleeps()

    def get_json(query: str) -> dict:
        return _http_post(query, client_factory=_factory(handler), sleep=sleep)

    payload = fetch_osm_tile(-3403, 1122, tmp_path, get_json=get_json)

    destination = tmp_path / f"osm_tile_v{OSM_QUERY_VERSION}_c-3403_r1122.json"
    assert payload == OK_PAYLOAD
    assert json.loads(destination.read_text(encoding="utf-8")) == OK_PAYLOAD
    assert not list(tmp_path.glob("*.part"))



def test_an_endpoint_configured_with_credentials_never_names_them(monkeypatch):
    """`staging` keeps OverpassUnavailable's message in its warning, because
    the hub writes that message from a host name and a status. An agency
    that points `ZA_OVERPASS_URLS` at a mirror behind basic auth must not
    see its password on a planner's screen: the host is named, the
    userinfo in front of it is not."""
    monkeypatch.setenv("ZA_OVERPASS_URLS",
                       "https://planner:hunter2@osm.agency.example/api/interpreter")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429)

    waits, sleep = _record_sleeps()
    with pytest.raises(OverpassUnavailable) as caught:
        _http_post("[out:json];", client_factory=_factory(handler), sleep=sleep)

    message = str(caught.value)
    assert "osm.agency.example" in message
    assert "hunter2" not in message and "planner:" not in message


# --- a host is named, and nothing else in the URL ever is -----------------------

@pytest.mark.parametrize("url, expected", [
    ("https://h?apikey=SECRET", "h"),
    ("https://h#SECRET", "h"),
    ("https://h/api/interpreter?key=SECRET#SECRET", "h"),
    ("https://planner:hunter2@osm.agency.example/api/interpreter", "osm.agency.example"),
    ("https://osm.example:8080/api/interpreter", "osm.example:8080"),
    ("https://[2001:db8::1]:8443/api/interpreter", "[2001:db8::1]:8443"),
    ("https://[2001:db8::1]/api/interpreter", "[2001:db8::1]"),
])
def test_an_endpoint_is_named_by_its_host_and_port_only(url, expected):
    assert linear_module._host(url) == expected


@pytest.mark.parametrize("url", [
    "https://planner:hun/ter2@osm.agency.example/api/interpreter",
    "https://planner/x:hunter2@osm.agency.example/api/interpreter",
    "https://planner:hunter2@osm.agency.example:notaport/api",
    # A '?' or '#' in the userinfo ends the authority just as early, and
    # leaves the '@' in the query or the fragment rather than the path.
    "https://admin:1234#rest@osm.agency.example/api",
    "https://user:12?pw@host/api",
    "https://TOKEN#x@host/api",
])
def test_an_endpoint_whose_userinfo_cannot_be_told_from_its_host_names_nothing(url):
    """A '/', '?' or '#' in an unencoded password ends the authority early,
    so a parser reads part of the userinfo as the host. Rather than guess,
    name no host."""
    host = linear_module._host(url)
    assert host == linear_module.UNREADABLE_ENDPOINT
    for secret in ("planner", "hun", "ter2", "hunter2", "notaport", "admin", "1234",
                   "rest", "user", "12", "pw", "token", "TOKEN"):
        assert secret.lower() not in host.lower()


SECRET_ENDPOINTS = ("https://planner:hunter2@osm.agency.example/api/interpreter,"
                    "https://mirror.example?apikey=SECRET")


def test_the_reported_endpoints_are_hosts(monkeypatch):
    from sar_geo_mcp.terrain.linear import reported_endpoints

    monkeypatch.setenv("ZA_OVERPASS_URLS", SECRET_ENDPOINTS)
    assert reported_endpoints() == ["osm.agency.example", "mirror.example"]


def test_no_configured_credential_or_query_reaches_a_routes_result(monkeypatch, tmp_path):
    """`LINEAR_ASSUMPTIONS` is every routes result's `assumptions`, and
    `_http_post` rewrites its endpoint list on every fetch."""
    from test_features import add_osm, bare_dir

    from sar_geo_mcp.terrain.features.context import TerrainContext
    from sar_geo_mcp.terrain.features.routes import RouteDetector

    monkeypatch.setenv("ZA_OVERPASS_URLS", SECRET_ENDPOINTS)
    monkeypatch.setitem(LINEAR_ASSUMPTIONS, "overpass_endpoints",
                        LINEAR_ASSUMPTIONS["overpass_endpoints"])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=OK_PAYLOAD)

    waits, sleep = _record_sleeps()
    _http_post("[out:json];", client_factory=_factory(handler), sleep=sleep)

    data_dir = bare_dir(tmp_path)
    add_osm(data_dir)
    result = RouteDetector().detect(TerrainContext(data_dir))

    dumped = result.model_dump_json()
    assert "osm.agency.example" in dumped
    for secret in ("hunter2", "planner", "apikey", "SECRET"):
        assert secret not in dumped


# --- query version 5: road class, access, tracktype and gates ---------------------


def _bbox_around(lat: float, lon: float, half_m: float = 500.0):
    """A Web Mercator box `half_m` units either side of a WGS84 point."""
    from pyproj import Transformer

    x, y = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform(lon, lat)
    return (x - half_m, y - half_m, x + half_m, y + half_m)


def test_version_5_is_the_road_tags_and_gates_version():
    from sar_geo_mcp.terrain.linear import OSM_QUERY_VERSION, OSM_QUERY_VERSION_ROUTE_TAGS
    assert OSM_QUERY_VERSION == 5 and OSM_QUERY_VERSION_ROUTE_TAGS == 5


def test_the_query_asks_for_gate_nodes():
    query = build_overpass_query(2631, 6339)
    assert 'node["barrier"~"^(gate|lift_gate|swing_gate)$"]' in query
    assert query.endswith(");out geom;") and query.count("(") == query.count(")")


def test_a_route_record_always_carries_highway_access_and_tracktype(tmp_path):
    payload = {"elements": [
        {"type": "way", "id": 7, "tags": {"highway": "track", "access": "Private",
                                          "tracktype": "grade3", "name": "Ridge Fire Road"},
         "geometry": [{"lat": 37.3561, "lon": -122.2419}, {"lat": 37.3571, "lon": -122.2409}]},
        {"type": "way", "id": 8, "tags": {"highway": "path"},
         "geometry": [{"lat": 37.3561, "lon": -122.2419}, {"lat": 37.3565, "lon": -122.2415}]}]}
    routes = routes_for(_bbox_around(37.3566, -122.2414), tmp_path, get_json=lambda q: payload)
    by_osm = {r["properties"]["osm_id"]: r["properties"] for r in routes}
    assert by_osm[7]["highway"] == "track" and by_osm[7]["access"] == "private"
    assert by_osm[7]["tracktype"] == "grade3"
    assert set(by_osm[8]) >= {"highway", "access", "tracktype"}   # always written
    assert by_osm[8]["access"] is None and by_osm[8]["tracktype"] is None


def test_gates_are_read_from_nodes(tmp_path):
    bbox = _bbox_around(37.3566, -122.2414)
    # A leftover version-4 tile for every tile the box covers, holding a gate
    # inside the box. It must be ignored: a version-4 tile never asked for
    # gates, so one found there would be luck, not an answer.
    for col, row in tiles_covering_osm(bbox):
        (tmp_path / f"osm_tile_v4_c{col}_r{row}.json").write_text(json.dumps({"elements": [
            {"type": "node", "id": 95, "lat": 37.3566, "lon": -122.2414,
             "tags": {"barrier": "gate"}}]}), encoding="utf-8")
    payload = {"elements": [
        {"type": "node", "id": 91, "lat": 37.3566, "lon": -122.2414,
         "tags": {"barrier": "gate", "access": "private"}},
        {"type": "node", "id": 92, "lat": 37.3567, "lon": -122.2413,
         "tags": {"barrier": "lift_gate"}},
        {"type": "node", "id": 93, "lat": 37.3568, "lon": -122.2412,
         "tags": {"barrier": "bollard"}},
        # A gate about 5 km north: in the tile's answer, outside the box.
        {"type": "node", "id": 96, "lat": 37.4016, "lon": -122.2414,
         "tags": {"barrier": "gate"}}]}
    calls = []
    gates = gates_for(bbox, tmp_path, get_json=lambda q: calls.append(q) or payload)
    assert calls, "the version-4 tiles were read instead of fetching at version 5"
    assert [(g["properties"]["osm_id"], g["properties"]["barrier"], g["properties"]["access"])
            for g in gates] == [(91, "gate", "private"), (92, "lift_gate", None)]
    assert gates[0]["id"] == "gate-n91" and gates[0]["kind"] == "gate"
    assert gates[0]["geometry"]["type"] == "Point"
    x, y = gates[0]["geometry"]["coordinates"]
    assert abs(x) > 1e6 and abs(y) > 1e6                  # EPSG:3857 metres, not degrees
    assert bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


# --- classify_way reads tags the way the stored fields do (0.18.0) -----------------

@pytest.mark.parametrize("tags, expected", [
    ({"highway": "Track "}, "TRACK"),
    ({"highway": "PATH"}, "TRAIL"),
    ({"highway": " Secondary"}, "ROAD"),
    ({"highway": "Busway"}, "ROAD_OTHER"),
    ({"waterway": " Stream "}, "WATERWAY"),
    ({"power": "Minor_Line"}, "BARRIER"),
    ({"railway": "Rail "}, "BARRIER"),
    ({"barrier": "Fence"}, "BARRIER"),
    ({"man_made": "Pipeline "}, "BARRIER"),
    ({"natural": "Water"}, "WATER_EDGE"),
    ({"landuse": "RESERVOIR"}, "WATER_EDGE"),
    ({"highway": "   "}, "OTHER"),
], ids=["track", "path", "secondary", "busway", "stream", "minor-line", "rail", "fence",
        "pipeline", "water", "reservoir", "blank-highway"])
def test_classify_way_normalises_every_tag_it_reads(tags, expected):
    """Through `_tag`, as the stored `highway`, `access` and `tracktype` are:
    a way stored as highway "track" is never classified as anything but a
    TRACK, and a blank highway is no highway, as it is stored (null)."""
    assert classify_way(tags) == expected


def test_a_route_record_stores_power_normalised_like_its_class(tmp_path):
    """`classify_way` now reads "Line" as a power line, so the stored `power`
    that `planning.lz` compares against "line" must read the same, or a wire
    would be classed a barrier and then missed as one."""
    payload = {"elements": [
        {"type": "way", "id": 7, "tags": {"power": "Line "},
         "geometry": [{"lat": 37.3690, "lon": -122.2300},
                      {"lat": 37.3745, "lon": -122.2243}]},
        {"type": "way", "id": 8, "tags": {"barrier": "fence"},
         "geometry": [{"lat": 37.3690, "lon": -122.2300},
                      {"lat": 37.3745, "lon": -122.2243}]}]}
    routes = routes_for(BBOX, cache_dir=tmp_path, get_json=lambda query: payload)
    by_osm = {r["properties"]["osm_id"]: r["properties"] for r in routes}
    assert by_osm[7]["route_type"] == "BARRIER" and by_osm[7]["power"] == "line"
    assert by_osm[8]["power"] is None
