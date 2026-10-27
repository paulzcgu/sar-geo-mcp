"""nearest_medical_facilities: HIFLD read right, ranked by road, honest when a
service is missing. Every network call is faked; the live check is marked."""

import httpx
import pytest

from sar_geo_mcp.planning import medical
from sar_geo_mcp.planning.medical import (
    HELIPAD_MATCH_M, Facility, facility_from_record, merge_campuses,
    nearest_medical_facilities, parse_trauma)
from sar_geo_mcp.sources import osrm as osrm_source
from sar_geo_mcp.sources import tigerweb as tigerweb_source
from sar_geo_mcp.sources.hifld import query_hospitals
from sar_geo_mcp.sources.osm import helipads_near
from sar_geo_mcp.sources.osrm import osrm_table
from sar_geo_mcp.sources.tigerweb import point_state

POINT = (37.42, -122.30)


def record(name, lat, lon, *, trauma="NOT AVAILABLE", helipad="N", kind="GENERAL ACUTE CARE",
           phone="(650) 555-0100", address="1 MAIN ST", city="TOWN", hid=None):
    return {"attributes": {"ID": hid or name, "NAME": name, "ADDRESS": address, "CITY": city,
                           "STATE": "CA", "ZIP": "94000", "TELEPHONE": phone, "TYPE": kind,
                           "STATUS": "OPEN", "TRAUMA": trauma, "HELIPAD": helipad,
                           "VAL_DATE": 1_700_000_000_000},
            "geometry": {"x": lon, "y": lat}}


class FakeServices:
    """HIFLD, OSRM, Overpass and the Census state lookup in one, with call records."""

    def __init__(self, records, *, minutes=None, wide=None, helipads=(), origin_snap=10.0,
                 route_error=None, pad_error=None, state="CA", locate_error=None):
        self.records, self.wide = records, wide or []
        self.minutes = minutes or {}
        self.pads, self.origin_snap = list(helipads), origin_snap
        self.route_error, self.pad_error = route_error, pad_error
        self.queries, self.routed, self.pad_points = [], [], []
        self.state, self.locate_error, self.located = state, locate_error, []

    def query(self, lat, lon, radius_km, where):
        self.queries.append((radius_km, where))
        return self.records if radius_km < medical.WIDE_RADIUS_KM else self.wide

    def route(self, origin, destinations):
        self.routed.append(destinations)
        if self.route_error:
            raise self.route_error
        legs = []
        for lat, lon in destinations:
            minutes = self.minutes.get((lat, lon), 30.0)
            legs.append({"distance_m": None if minutes is None else minutes * 1000.0,
                         "duration_s": None if minutes is None else minutes * 60.0,
                         "snap_m": 5.0})
        return {"origin_snap_m": self.origin_snap,
                "origin_snap": {"latitude": origin[0], "longitude": origin[1]}, "legs": legs}

    def helipads(self, points, radius_m):
        self.pad_points.append(points)
        if self.pad_error:
            raise self.pad_error
        return self.pads

    def locate(self, lat, lon):
        self.located.append((lat, lon))
        if self.locate_error:
            raise self.locate_error
        return self.state


def run(services, **kw):
    return nearest_medical_facilities(*POINT, query=services.query, route=services.route,
                                      helipads=services.helipads, locate=services.locate,
                                      **kw)


# --- Reading the TRAUMA field --------------------------------------------------


@pytest.mark.parametrize("raw, adult, ped_level, pediatric, state", [
    ("NOT AVAILABLE", None, None, False, None),
    (None, None, None, False, None),
    ("LEVEL I", 1, None, False, None),
    ("LEVEL IV", 4, None, False, None),
    ("LEVEL V", 5, None, False, None),
    ("LEVEL II ADULT", 2, None, False, None),
    ("LEVEL I ADULT, LEVEL II PEDIATRIC", 1, 2, True, None),
    ("LEVEL I PEDIATRIC", None, 1, True, None),
    ("LEVEL II / PEDIATRIC", 2, None, True, None),
    ("PEDIATRIC", None, None, True, None),
    ("Level I Trauma Center", 1, None, False, None),
    ("LEVEL II, LEVEL III PEDIATRIC, LEVEL II REHAB", 2, 3, True, None),
    ("LEVEL I PEDIATRIC REHAB", None, None, False, None),
    ("CTH", None, None, False, "CTH"),
])
def test_the_trauma_field_is_read_into_levels(raw, adult, ped_level, pediatric, state):
    trauma = parse_trauma(raw)
    assert (trauma.adult_level, trauma.pediatric_level, trauma.pediatric,
            trauma.state_designation) == (adult, ped_level, pediatric, state)


def test_trauma_labels_read_as_a_person_would_say_them():
    assert parse_trauma("LEVEL I ADULT, LEVEL II PEDIATRIC").label() == "Level I, Level II Pediatric"
    assert parse_trauma("TRF").label() == "State-designated (TRF)"
    assert parse_trauma("NOT AVAILABLE").label() == "None"


# --- Records into facilities ---------------------------------------------------


def test_a_record_reads_its_phone_address_and_helipad():
    facility = facility_from_record(record("UCSF MEDICAL CENTER", 37.7, -122.4,
                                           helipad="Y", address="505 PARNASSUS AVE",
                                           city="SAN FRANCISCO"))
    assert facility.name == "UCSF Medical Center"
    assert facility.address == "505 Parnassus Ave, San Francisco, CA 94000"
    assert facility.helipad == "Y" and facility.phone == "(650) 555-0100"
    assert facility.validated == "2023-11-14"


def test_not_available_is_unknown_not_no():
    facility = facility_from_record(record("A", 37.5, -122.3, helipad="NOT AVAILABLE",
                                           phone="NOT AVAILABLE"))
    assert facility.helipad == "unknown" and facility.phone is None


def test_a_record_without_a_position_is_skipped():
    bad = record("A", 37.5, -122.3)
    bad["geometry"] = {}
    bad["attributes"]["LATITUDE"] = None
    assert facility_from_record(bad) is None


def test_one_campus_listed_twice_is_one_entry_with_the_best_of_both():
    a = facility_from_record(record("STANFORD HEALTH CARE", 37.4346, -122.1749,
                                    trauma="LEVEL I", helipad="N", address="300 PASTEUR DR"))
    b = facility_from_record(record("STANFORD HEALTH CARE", 37.4343, -122.1771,
                                    helipad="Y", phone="NOT AVAILABLE", address="500 PASTEUR DR"))
    c = facility_from_record(record("STANFORD HOSPITAL TRANSPLANT", 37.4325, -122.1753,
                                    address="300 PASTEUR DR"))
    merged = merge_campuses([b, c, a])
    assert len(merged) == 1
    campus = merged[0]
    assert campus.trauma.adult_level == 1 and campus.helipad == "Y"
    assert campus.phone == "(650) 555-0100"
    assert campus.also_listed_as == ["Stanford Hospital Transplant"]


def test_same_name_far_apart_is_two_hospitals():
    a = facility_from_record(record("KAISER", 37.40, -122.30, address="1 A ST"))
    b = facility_from_record(record("KAISER", 37.60, -122.30, address="2 B ST"))
    assert len(merge_campuses([a, b])) == 2


# --- Choosing and ranking ------------------------------------------------------


def test_each_category_is_ranked_by_road_time_not_straight_line():
    near_slow = record("NEAR", 37.43, -122.30, trauma="LEVEL II", address="1 A")
    far_fast = record("FAR", 37.50, -122.30, trauma="LEVEL I", address="2 B")
    services = FakeServices([near_slow, far_fast],
                            minutes={(37.43, -122.30): 50.0, (37.50, -122.30): 20.0})
    result = run(services)
    trauma = [r for r in result["rows"] if r["category"] == "trauma_center"]
    assert [r["name"] for r in trauma] == ["Far", "Near"]
    assert trauma[0]["road_minutes"] == 20.0 and trauma[0]["road_distance_m"] == 20000
    assert len(services.routed) == 1          # one router request for every category


def test_psychiatric_and_long_term_hospitals_are_left_out():
    services = FakeServices([
        record("PSYCH", 37.43, -122.30, kind="PSYCHIATRIC", address="1 A"),
        record("LTC", 37.43, -122.31, kind="LONG TERM CARE", address="2 B"),
        record("ACUTE", 37.45, -122.30, address="3 C"),
    ])
    names = {r["name"] for r in run(services)["rows"]}
    assert names == {"Acute"}


def test_a_military_hospital_counts_only_with_a_trauma_designation():
    services = FakeServices([
        record("VA CLINIC", 37.43, -122.30, kind="MILITARY", address="1 A"),
        record("ARMY MEDICAL CENTER", 37.44, -122.30, kind="MILITARY", trauma="LEVEL I",
               address="2 B"),
    ], wide=[])
    names = {r["name"] for r in run(services)["rows"]}
    assert names == {"Army Medical Center"}


def test_a_childrens_hospital_is_never_an_adult_trauma_center():
    services = FakeServices([
        record("LUCILE SALTER PACKARD CHILDREN'S HOSPITAL", 37.436, -122.175, kind="CHILDREN",
               trauma="LEVEL I ADULT, LEVEL I PEDIATRIC", address="725 WELCH RD"),
        record("STANFORD HEALTH CARE", 37.4346, -122.1749, trauma="LEVEL I",
               address="300 PASTEUR DR"),
    ])
    rows = run(services)["rows"]
    adult = [r["name"] for r in rows if r["category"] == "trauma_center"]
    child = [r for r in rows if r["category"] == "pediatric_trauma_center"]
    assert adult == ["Stanford Health Care"]
    assert [r["name"] for r in child] == ["Lucile Salter Packard Children's Hospital"]
    assert child[0]["trauma"]["label"] == "Level I Pediatric"
    assert "adult designation" in child[0]["notes"][0]
    assert child[0]["trauma"]["raw"] == "LEVEL I ADULT, LEVEL I PEDIATRIC"


def test_a_missing_category_is_searched_again_wider():
    services = FakeServices(
        [record("LOCAL", 37.43, -122.30, trauma="LEVEL I", address="1 A")],
        wide=[record("FAR CHILDREN'S", 39.0, -122.30, trauma="LEVEL I PEDIATRIC",
                     kind="CHILDREN", address="9 Z")])
    result = run(services)
    assert services.queries[1] == (medical.WIDE_RADIUS_KM,
                                   "STATUS = 'OPEN' AND TRAUMA <> 'NOT AVAILABLE'")
    assert result["searched_radius_km"]["pediatric_trauma_center"] == medical.WIDE_RADIUS_KM
    assert result["searched_radius_km"]["trauma_center"] == 100.0
    assert any(r["category"] == "pediatric_trauma_center" and r["name"] == "Far Children's"
               for r in result["rows"])


def test_nothing_anywhere_is_a_warning_per_category():
    result = run(FakeServices([]))
    assert result["rows"] == []
    assert sum("was found within 400 km" in w for w in result["warnings"]) == 3


def test_records_beyond_the_radius_are_trimmed_by_ground_distance():
    services = FakeServices([record("EDGE", 38.40, -122.30, address="1 A")])   # ~109 km
    result = run(services)
    assert not any(r["name"] == "Edge" and r["category"] == "medical_center"
                   for r in result["rows"])


# --- Road ----------------------------------------------------------------------


def test_router_failure_degrades_to_straight_line_with_a_warning():
    services = FakeServices([record("A", 37.43, -122.30, address="1 A"),
                             record("B", 37.50, -122.30, address="2 B")],
                            route_error=httpx.ConnectError("down"))
    result = run(services)
    medical_rows = [r for r in result["rows"] if r["category"] == "medical_center"]
    assert [r["name"] for r in medical_rows] == ["A", "B"]
    assert medical_rows[0]["road_minutes"] is None
    assert any("road router could not be reached (ConnectError)" in w
               for w in result["warnings"])


def test_an_unroutable_hospital_goes_after_the_routed_ones():
    services = FakeServices([record("ISLAND", 37.43, -122.30, address="1 A"),
                             record("MAINLAND", 37.50, -122.30, address="2 B")],
                            minutes={(37.43, -122.30): None})
    result = run(services)
    names = [r["name"] for r in result["rows"] if r["category"] == "medical_center"]
    assert names == ["Mainland", "Island"]
    assert any("No road route was found to Island" in w for w in result["warnings"])


def test_a_point_far_from_any_road_says_the_road_starts_elsewhere():
    services = FakeServices([record("A", 37.43, -122.30)], origin_snap=1840.0)
    result = run(services)
    assert result["road_start"]["distance_from_point_m"] == 1840
    assert any("1,840 m from the nearest road" in w for w in result["warnings"])


def test_osrm_table_asks_lon_lat_and_reads_the_answer():
    asked = {}

    class Response:
        def json(self):
            return {"code": "Ok", "distances": [[0, 18158.3, None]],
                    "durations": [[0, 1681.9, None]],
                    "sources": [{"distance": 685.2, "location": [-122.305, 37.4247]}],
                    "destinations": [{}, {"distance": 91.6}, {"distance": 17.5}]}

    def get(url, params):
        asked["url"], asked["params"] = url, params
        return Response()

    table = osrm_table((37.42, -122.30), [(37.4344, -122.1756), (37.7557, -122.4058)], get=get)
    assert asked["url"].endswith(
        "/table/v1/driving/-122.300000,37.420000;-122.175600,37.434400;-122.405800,37.755700")
    assert asked["params"] == {"sources": "0", "annotations": "distance,duration"}
    assert table["origin_snap_m"] == 685.2
    assert table["origin_snap"] == {"latitude": 37.4247, "longitude": -122.305}
    assert table["legs"][0] == {"distance_m": 18158.3, "duration_s": 1681.9, "snap_m": 91.6}
    assert table["legs"][1]["duration_s"] is None


def test_osrm_url_override(monkeypatch):
    monkeypatch.setenv(osrm_source.OSRM_URL_ENV, "http://router.local:5000/")
    assert osrm_source.osrm_url() == "http://router.local:5000"


# --- Helipads and air ----------------------------------------------------------


def test_air_time_only_where_there_is_a_helipad():
    services = FakeServices([
        record("PAD", 37.52, -122.30, helipad="Y", address="1 A"),          # ~11.1 km
        record("NOPAD", 37.43, -122.30, helipad="N", address="2 B"),
    ])
    rows = {r["name"]: r for r in run(services)["rows"] if r["category"] == "medical_center"}
    expected = rows["Pad"]["straight_line_m"] / 1000 / 240 * 60 + 5
    assert rows["Pad"]["air_minutes"] == pytest.approx(expected, abs=0.05)
    assert rows["Nopad"]["air_minutes"] is None and rows["Nopad"]["faster"] is None
    assert rows["Pad"]["faster"] == "air"


def test_osm_helipad_turns_an_hifld_no_into_a_yes_with_a_warning():
    services = FakeServices([record("A", 37.43, -122.30, helipad="N", address="1 A")],
                            helipads=[{"latitude": 37.4305, "longitude": -122.3, "name": "A Pad"}])
    result = run(services)
    row = result["rows"][0]
    assert row["helipad"] == "Y" and row["helipad_source"] == "HIFLD + OpenStreetMap"
    assert row["helipad_osm_name"] == "A Pad"
    assert any("HIFLD lists no helipad but OpenStreetMap" in w for w in result["warnings"])


def test_osm_silence_does_not_overrule_an_hifld_yes():
    services = FakeServices([record("A", 37.43, -122.30, helipad="Y", address="1 A")])
    assert run(services)["rows"][0]["helipad"] == "Y"


def test_a_helipad_too_far_away_is_not_the_hospitals():
    services = FakeServices([record("A", 37.43, -122.30, helipad="NOT AVAILABLE")],
                            helipads=[{"latitude": 37.44, "longitude": -122.3, "name": "x"}])
    result = run(services)
    assert result["rows"][0]["helipad"] == "unknown"
    assert any("Helipad 'unknown'" in w for w in result["warnings"])


def test_overpass_failure_leaves_hifld_answer_with_a_warning():
    services = FakeServices([record("A", 37.43, -122.30, helipad="Y")],
                            pad_error=RuntimeError("busy"))
    result = run(services)
    assert result["rows"][0]["helipad"] == "Y"
    assert any("helipad is HIFLD's answer alone" in w for w in result["warnings"])


def test_helipad_check_can_be_turned_off():
    services = FakeServices([record("A", 37.43, -122.30)])
    run(services, check_osm_helipads=False)
    assert services.pad_points == []


def test_osm_helipad_query_reads_ways_by_their_centre():
    asked = []

    def post(query):
        asked.append(query)
        return {"elements": [
            {"type": "way", "center": {"lat": 37.43, "lon": -122.17}, "tags": {"name": "P"}},
            {"type": "node", "lat": 37.5, "lon": -122.2, "tags": {}},
            {"type": "relation", "tags": {}},
        ]}

    pads = helipads_near([(37.43, -122.17)], HELIPAD_MATCH_M, post=post)
    assert 'nwr(around:300,37.430000,-122.170000)[aeroway~"^(helipad|heliport)$"];' in asked[0]
    assert pads == [{"latitude": 37.43, "longitude": -122.17, "name": "P"},
                    {"latitude": 37.5, "longitude": -122.2, "name": None}]


# --- The hospital query -------------------------------------------------------


def test_the_hospital_query_pages_and_widens_for_mercator():
    pages = [{"features": [record("A", 37.4, -122.3)], "exceededTransferLimit": True},
             {"features": [record("B", 37.4, -122.3)]}]
    asked = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def json(self):
            return self.payload

    def get(url, params):
        asked.append(params)
        return Response(pages[len(asked) - 1])

    features = query_hospitals(37.42, -122.30, 100.0, get=get)
    assert [f["attributes"]["NAME"] for f in features] == ["A", "B"]
    assert [p["resultOffset"] for p in asked] == ["0", "2000"]
    assert float(asked[0]["distance"]) == pytest.approx(100 / 0.7936, rel=1e-3)
    assert asked[0]["geometry"] == "-122.3,37.42" and asked[0]["outSR"] == "4326"


def test_a_refused_hospital_query_is_an_error():
    class Response:
        def json(self):
            return {"error": {"code": 400, "message": "Invalid URL"}}

    with pytest.raises(RuntimeError, match="refused the query: 400"):
        query_hospitals(37.42, -122.30, 50.0, get=lambda url, params: Response())


# --- The table and the markers ------------------------------------------------


def test_the_table_has_every_column_asked_for():
    services = FakeServices([record("A", 37.52, -122.30, trauma="LEVEL II", helipad="Y",
                                    address="1 MAIN ST", city="TOWN")],
                            minutes={(37.52, -122.30): 75.0})
    result = run(services)
    lines = result["table"].splitlines()
    assert lines[0] == ("| Category | # | Hospital | Trauma level | Helipad | Road distance "
                        "| Road time | Air time | Phone | Address |")
    assert ("| Trauma center | 1 | A | Level II | Y | 75.0 km / 46.6 mi | 1 h 15 min | 8 min "
            "| (650) 555-0100 | 1 Main St, Town, CA 94000 |") in lines


def test_markers_draw_each_hospital_once_with_all_its_categories():
    services = FakeServices([record("A", 37.43, -122.30, trauma="LEVEL I PEDIATRIC, LEVEL I")])
    markers = run(services)["markers"]
    assert markers["type"] == "FeatureCollection"
    roles = [f["properties"]["role"] for f in markers["features"]]
    assert roles == ["origin", "hospital"]
    assert markers["features"][1]["properties"]["categories"] == [
        "medical_center", "trauma_center", "pediatric_trauma_center"]
    assert markers["features"][0]["geometry"]["coordinates"] == [POINT[1], POINT[0]]


def test_markers_import_into_caltopo_as_markers_carrying_the_table_row():
    # CalTopo keeps a marker's title, description, symbol and colour and
    # nothing else: a plain GeoJSON point arrived as a bare black dot with
    # an empty description (found in a live CalTopo import, 2026-10-21).
    services = FakeServices(
        [record("A", 37.52, -122.30, trauma="LEVEL II", helipad="Y",
                phone="(650) 555-0199", address="1 MAIN ST", city="TOWN")],
        minutes={(37.52, -122.30): 75.0}, origin_snap=640.0)
    origin, hospital = run(services)["markers"]["features"]
    for feature in (origin, hospital):
        assert feature["properties"]["class"] == "Marker"
        assert feature["properties"]["marker-size"] == 1
    assert hospital["properties"]["marker-symbol"] == "firstaid"
    assert hospital["properties"]["marker-color"] == "#FF0000"
    assert (origin["properties"]["marker-symbol"],
            origin["properties"]["marker-color"]) == ("point", "#0000FF")
    assert hospital["properties"]["title"] == "A"
    assert hospital["properties"]["description"].splitlines() == [
        "Medical center #1, Trauma center #1",
        "Trauma: Level II",
        "Helipad: Y",
        "Road: 75.0 km / 46.6 mi, 1 h 15 min",
        "Air: 8 min",
        "Phone: (650) 555-0199",
        "Address: 1 Main St, Town, CA 94000",
        "HIFLD validated 2023-11-14. Confirm before transport.",
    ]
    assert "640 m away" in origin["properties"]["description"]


def test_the_point_marker_names_the_road_start_only_when_it_is_far():
    near = run(FakeServices([record("A", 37.43, -122.30)], origin_snap=40.0))
    assert "Road route" not in near["markers"]["features"][0]["properties"]["description"]


@pytest.mark.parametrize("services, said", [
    ({}, medical.RANKED_BY_ROAD),
    ({"minutes": {(37.44, -122.30): None}}, medical.RANKED_BY_ROAD_SOME_UNROUTED),
    ({"route_error": httpx.ConnectError("OSRM unreachable")},
     medical.RANKED_STRAIGHT_ROUTER_DOWN),
    ({"minutes": {(37.43, -122.30): None, (37.44, -122.30): None}},
     medical.RANKED_STRAIGHT_NO_ROUTE),
], ids=["all-routed", "one-unrouted", "router-down", "none-routed"])
def test_the_point_marker_says_how_the_hospitals_drawn_were_ranked(services, said):
    """The point is read beside each hospital's "#1": it says road time only
    when every hospital drawn has one, and straight-line distance - and why -
    when none has (PR #4 review B21). The road starts 10 m off, so the
    ranking is all it says."""
    answer = run(FakeServices([record("A", 37.43, -122.30, address="1 A"),
                               record("B", 37.44, -122.30, address="2 B")], **services))
    assert len(answer["rows"]) == 2
    assert answer["markers"]["features"][0]["properties"]["description"] == said


def test_with_no_hospital_drawn_the_point_marker_says_nothing_of_ranking():
    """Outside the US nothing is found: a ranking line would describe no
    marker on the map."""
    answer = run(FakeServices([], state=None))
    [point] = answer["markers"]["features"]
    assert point["properties"]["description"] == ""


def test_a_marker_without_a_helipad_says_so_instead_of_an_air_time():
    services = FakeServices([record("A", 37.43, -122.30, helipad="N", phone="NOT AVAILABLE")])
    hospital = run(services)["markers"]["features"][1]
    lines = hospital["properties"]["description"].splitlines()
    assert "Air: — (no helipad known)" in lines and "Phone: —" in lines


# --- Inside the US at all ----------------------------------------------------------


def test_a_point_inside_the_us_says_its_state_and_adds_no_warning():
    services = FakeServices([record("A", 37.43, -122.30)])
    result = run(services)
    assert services.located == [POINT]
    assert result["point"]["in_us"] is True and result["point"]["us_state"] == "CA"
    assert not any(warning.startswith(("The point is not inside", "Whether the point is"))
                   for warning in result["warnings"])


def test_a_point_outside_the_us_is_warned_first_and_still_answered():
    # Downtown Vancouver was told its nearest trauma center was Bellingham,
    # 87 km away, with Vancouver General 1 km off and no word of it.
    services = FakeServices([record("A", 37.43, -122.30, trauma="LEVEL II")], state=None)
    result = run(services)
    assert result["point"]["in_us"] is False and result["point"]["us_state"] is None
    assert result["warnings"][0].startswith("The point is not inside any US state")
    assert "another country" in result["warnings"][0]
    assert result["rows"]


def test_a_failed_state_lookup_is_a_warning_not_an_error():
    services = FakeServices([record("A", 37.43, -122.30)],
                            locate_error=httpx.ConnectError("down"))
    result = run(services)
    assert result["point"]["in_us"] is None
    assert result["warnings"][0].startswith("Whether the point is inside the US could not")
    assert result["rows"]


def test_point_state_reads_the_census_answer():
    seen = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def json(self):
            return self.payload

    def answer(payload):
        def get(url, params):
            seen.append((url, params))
            return Response(payload)
        return get

    assert point_state(37.36, -122.24, get=answer(
        {"features": [{"attributes": {"STUSAB": "CA"}}]})) == "CA"
    assert point_state(49.28, -123.12, get=answer({"features": []})) is None
    url, params = seen[0]
    assert url.endswith("/MapServer/0/query")
    assert params["geometry"] == "-122.24,37.36" and params["inSR"] == "4326"
    with pytest.raises(RuntimeError, match="refused the query: 400"):
        point_state(0, 0, get=answer({"error": {"code": 400}}))


def test_inputs_are_checked_before_any_network_call():
    services = FakeServices([])
    for kwargs in ({"per_category": 0}, {"search_radius_km": 500},
                   {"helicopter_speed_kmh": 10}, {"air_overhead_minutes": -1}):
        with pytest.raises(ValueError):
            run(services, **kwargs)
    with pytest.raises(ValueError):
        nearest_medical_facilities(95, 0, query=services.query)
    assert services.queries == []


def test_assumptions_name_hosts_not_urls(monkeypatch):
    monkeypatch.setenv(osrm_source.OSRM_URL_ENV, "https://user:secret@router.local/osrm?key=abc")
    monkeypatch.setenv(tigerweb_source.US_STATES_URL_ENV, "https://u:pw@states.local/q?token=xyz")
    result = run(FakeServices([]))
    assert result["assumptions"]["road"]["host"] == "router.local"
    assert result["assumptions"]["coverage"]["host"] == "states.local"
    assert "pw@" not in str(result) and "xyz" not in str(result)
    assert "secret" not in str(result) and "abc" not in str(result)


# --- Live ------------------------------------------------------------------------


@pytest.mark.live
def test_live_bay_area_point():
    result = nearest_medical_facilities(*POINT)
    trauma = [r for r in result["rows"] if r["category"] == "trauma_center"]
    pediatric = [r for r in result["rows"] if r["category"] == "pediatric_trauma_center"]
    assert trauma and pediatric
    assert trauma[0]["road_minutes"] is not None
    assert any(r["trauma"]["adult_level"] == 1 for r in trauma)
    assert all("Children" not in r["name"] for r in trauma)
    assert result["point"]["in_us"] is True and result["point"]["us_state"] == "CA"
    assert all(f["properties"]["class"] == "Marker" for f in result["markers"]["features"])


@pytest.mark.live
def test_live_census_knows_the_border():
    assert point_state(48.99, -122.75) == "WA"      # Blaine, on the border
    assert point_state(49.28, -123.12) is None      # Vancouver
    assert point_state(42.30, -83.03) is None       # Windsor, south of Detroit
    assert point_state(13.47, 144.75) == "GU"
