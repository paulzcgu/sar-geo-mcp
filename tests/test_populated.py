"""Populated blocks off synthetic CalTopo panels: lots read from drawn lot
lines, built-up from the structures layer, grouped by the street they front.

The town is drawn in ground metres around CENTRE and rendered by a fake
panel fetcher at whatever bbox and size the tool asks for, so no network is
touched:

    y (m)
    115 +------------------+   one 80 x 70 m apartment lot (5,600 m2, over
        |  [] []    [] []  |   an acre) with four 20 m roofs, each inside a
     45 +------------------+   parcel drawn tight round it (unit parcels)
     40 +--+--+--+--+--+--+
        |[]|[]|[]|[]|[]|[]|    six 20 x 30 m lots, a 10 m roof on each
     10 +--+--+--+--+--+--+
      0 ===== Elm Street =====  (20 m right-of-way, centreline in OSM)
    -10 +--+--+--+--+--+--+
        |[]|[]|[]|[]|[]|[]|    six more
    -40 +--+--+--+--+--+--+
    -45 +------------------+
        |        []        |   a 100 x 100 m farm lot, one 10 m roof
   -145 +------------------+
        -60               60   x (m)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from sar_geo_mcp.planning.populated import (
    NO_OSM_WARNING,
    NO_PARCELS_WARNING,
    populated_areas,
    populated_params,
    populated_set,
)
from sar_geo_mcp.sources.osm import OSM_QUERY_VERSION

CENTRE = (36.5212, -121.8105)
SCALE = math.cos(math.radians(CENTRE[0]))
TOKEN = "SECRET-TOKEN-123"
WMS = f"https://caltopo.example/api/{TOKEN}/wms"


def _merc(x_m: float, y_m: float, centre=CENTRE) -> tuple[float, float]:
    """Ground metres east and north of `centre` as Web Mercator."""
    from pyproj import Transformer

    x0, y0 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform(
        centre[1], centre[0])
    scale = math.cos(math.radians(centre[0]))
    return x0 + x_m / scale, y0 + y_m / scale


def _rect(x0, y0, x1, y1):
    return (x0, y0, x1, y1)


def town(*, lots=True, apartment=True, farm=True):
    """(lot rectangles, roof rectangles) in ground metres."""
    parcels, roofs = [], []
    if lots:
        for i in range(6):
            x0 = -60 + 20 * i
            for y0, y1 in ((10, 40), (-40, -10)):
                parcels.append(_rect(x0, y0, x0 + 20, y1))
                cy = (y0 + y1) / 2
                roofs.append(_rect(x0 + 5, cy - 5, x0 + 15, cy + 5))
    if apartment:
        parcels.append(_rect(-40, 45, 40, 115))
        for cx, cy in ((-20, 65), (20, 65), (-20, 95), (20, 95)):
            unit = _rect(cx - 10, cy - 10, cx + 10, cy + 10)
            parcels.append(unit)          # the unit parcel, tight round the roof
            roofs.append(unit)
    if farm:
        parcels.append(_rect(-50, -145, 50, -45))
        roofs.append(_rect(-5, -100, 5, -90))
    return parcels, roofs


def fake_fetch(parcels, roofs, *, calls=None, blank_parcels=False, centre=CENTRE):
    from PIL import Image, ImageDraw

    def fetch(base_url, layers, bbox, width, height, cache_dir):
        if calls is not None:
            calls.append(layers)
        west, south, east, north = bbox

        def px(x_m, y_m):
            x, y = _merc(x_m, y_m, centre)
            return ((x - west) / (east - west) * width, (north - y) / (north - south) * height)

        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        if layers == ["parcel"] and not blank_parcels:
            for x0, y0, x1, y1 in parcels:
                (a, b), (c, d) = px(x0, y1), px(x1, y0)
                draw.rectangle([round(a), round(b), round(c), round(d)],
                               outline=(160, 32, 240, 255), width=1)
        elif layers == ["structures"]:
            for x0, y0, x1, y1 in roofs:
                (a, b), (c, d) = px(x0, y1), px(x1, y0)
                draw.rectangle([round(a) + 1, round(b) + 1, round(c) - 1, round(d) - 1],
                               fill=(250, 200, 120, 100))
        path = Path(cache_dir)
        path.mkdir(parents=True, exist_ok=True)
        out = path / f"{'-'.join(layers)}.png"
        image.save(out)
        return out

    return fetch


def elm_street(centre=CENTRE) -> dict:
    """Elm Street through the middle of the town, as osm.json carries it."""
    return {"id": "way/1", "kind": "route",
            "geometry": {"type": "LineString",
                         "coordinates": [list(_merc(-150, 0, centre)),
                                         list(_merc(150, 0, centre))]},
            "properties": {"name": "Elm Street", "route_type": "ROAD"}}


def _staged(tmp_path, *, radius_m=170.0, osm=True):
    directory = tmp_path / "staged"
    directory.mkdir()
    lat, lon = CENTRE
    ring = []
    for k in range(65):
        angle = 2 * math.pi * k / 64
        ring.append([lon + radius_m * math.cos(angle) / (111_320 * SCALE),
                     lat + radius_m * math.sin(angle) / 110_574])
    (directory / "subject.json").write_text(json.dumps(
        {"label": "aoi", "geometry": {"type": "Polygon", "coordinates": [ring]}}),
        encoding="utf-8")
    if osm:
        street = elm_street()
        (directory / "osm.json").write_text(json.dumps(
            {"routes": [street], "parking": [], "query_version": OSM_QUERY_VERSION}),
            encoding="utf-8")
    return directory


def _run(tmp_path, *, layout=None, osm=True, **kwargs):
    parcels, roofs = layout if layout is not None else town()
    directory = _staged(tmp_path, osm=osm)
    return populated_areas(directory, wms_base_url=WMS, cache_dir=tmp_path / "cache",
                           fetch=fake_fetch(parcels, roofs), **kwargs)


def _shape(block):
    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely.ops import transform

    to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    return transform(to_3857.transform, shape(block["geometry"]))


def _point(x_m, y_m):
    from shapely.geometry import Point

    return Point(*_merc(x_m, y_m))


def test_a_street_is_one_block_across_both_sides_and_the_street(tmp_path):
    result = _run(tmp_path)
    streets = [b for b in result["blocks"] if b["kind"] == "street"]
    assert len(streets) == 1, result["blocks"]
    block = streets[0]
    assert block["name"] == "Elm Street"
    assert block["facts"]["street"] == "Elm Street"
    assert block["facts"]["parcels"] == 12
    assert block["facts"]["buildings"] == 12
    outline = _shape(block)
    # Both rows of houses and the street between them.
    for x in (-50, -10, 30, 50):
        assert outline.contains(_point(x, 25)), x
        assert outline.contains(_point(x, -25)), x
        assert outline.contains(_point(x, 0)), x
    point = block["facts"]["point"]
    from shapely.geometry import Point
    from shapely.geometry import shape as shapely_shape
    assert shapely_shape(block["geometry"]).contains(
        Point(point["longitude"], point["latitude"]))


def test_a_block_the_ring_reaches_is_kept_whole_however_far_its_middle_is(tmp_path):
    """The ring is extended to the blocks it touches, never cut through
    them: the apartment lot starts 45 m north of the centre and its middle
    is 80 m off, so a 50 m ring keeps all of it."""
    result = _run(tmp_path, max_distance_m=50.0)
    (apartment,) = [b for b in result["blocks"] if b["kind"] == "large_parcel"]
    assert apartment["facts"]["distance_from_center_m"] > 50.0
    assert _shape(apartment).contains(_point(0, 110))          # its far end too
    (tmp_path / "narrow").mkdir()
    assert not [b for b in _run(tmp_path / "narrow", max_distance_m=40.0)["blocks"]
                if b["kind"] == "large_parcel"]


def test_an_apartment_lot_is_its_own_block_with_its_unit_parcels_folded_in(tmp_path):
    result = _run(tmp_path)
    large = [b for b in result["blocks"] if b["kind"] == "large_parcel"]
    assert len(large) == 1, result["blocks"]
    block = large[0]
    assert block["name"] == "Elm Street - large parcel"
    assert block["facts"]["parcels"] == 1
    assert block["facts"]["buildings"] == 4
    assert _shape(block).contains(_point(0, 80))
    # No micro-block for a unit parcel.
    assert all(b["facts"]["median_parcel_m2"] > 300 for b in result["blocks"])
    # No house in two blocks.
    street = next(b for b in result["blocks"] if b["kind"] == "street")
    overlap = _shape(street).intersection(_shape(block)).area * SCALE * SCALE
    assert overlap < 0.01, overlap                   # no shared ground at all (0.20.0)


def test_a_farmhouse_on_a_big_lot_is_not_populated(tmp_path):
    result = _run(tmp_path)
    farmhouse = _point(0, -95)
    assert not any(_shape(b).contains(farmhouse) for b in result["blocks"])
    assert result["summary"]["count"] == 2


def test_rural_ground_says_so_rather_than_returning_an_empty_list(tmp_path):
    result = _run(tmp_path, layout=town(lots=False, apartment=False))
    assert result["blocks"] == []
    assert any("rural" in warning for warning in result["warnings"])


def test_no_lot_lines_is_a_warning(tmp_path):
    parcels, roofs = town()
    directory = _staged(tmp_path)
    result = populated_areas(directory, wms_base_url=WMS, cache_dir=tmp_path / "cache",
                             fetch=fake_fetch(parcels, roofs, blank_parcels=True))
    assert result["blocks"] == []
    assert NO_PARCELS_WARNING in result["warnings"]


def test_without_osm_lots_are_grouped_by_adjacency_with_no_street(tmp_path):
    result = _run(tmp_path, osm=False)
    assert NO_OSM_WARNING in result["warnings"]
    streets = [b for b in result["blocks"] if b["kind"] == "street"]
    assert len(streets) == 1
    assert streets[0]["name"] is None
    assert streets[0]["facts"]["parcels"] == 12


def test_a_run_shorter_than_min_parcels_joins_the_block_it_touches(tmp_path):
    """Too few houses for a block of their own are still searched: with the
    block whose lots they touch (circled on a hand-drawn plan)."""
    result = _run(tmp_path, min_parcels=13)
    (block,) = result["blocks"]
    assert block["kind"] == "large_parcel"
    assert block["facts"]["parcels"] == 13
    assert result["summary"]["parcels_left_out"] == 0
    assert any("too few for a block of their own" in w for w in block["warnings"])
    assert _shape(block).contains(_point(-50, -25))           # the far row too


def test_a_run_shorter_than_min_parcels_touching_nothing_is_counted_not_hidden(tmp_path):
    result = _run(tmp_path, layout=town(apartment=False), min_parcels=13)
    assert result["blocks"] == []
    assert result["summary"]["parcels_left_out"] == 12
    assert any("12 built-up lot(s)" in warning for warning in result["warnings"])


def test_a_roofed_lot_on_the_street_is_never_folded_into_the_street(tmp_path):
    """A small lot that is nearly all roof reads as a condo unit's parcel and
    is folded into its neighbour - but a downtown lot whose only neighbour
    is the street must stay a lot. Folded into the right-of-way it was
    searched by no one (seen in a live run)."""
    parcels, roofs = town(apartment=False, farm=False)
    parcels.append(_rect(74, 12, 90, 32))          # 320 m2, 14 m past the row
    roofs.append(_rect(74, 12, 90, 32))            # all roof: a footprint parcel
    result = _run(tmp_path, layout=(parcels, roofs))
    (block,) = result["blocks"]
    assert block["facts"]["parcels"] == 13
    assert _shape(block).contains(_point(82, 22))


def test_a_vacant_lot_on_the_street_is_searched_with_it(tmp_path):
    """A lot with no roof between the houses of its street is part of the
    street's block, not a notch in it: a child hides on a vacant lot too."""
    parcels, roofs = town(apartment=False, farm=False)
    roofs = [r for r in roofs if not (r[0] == -15 and r[1] > 0)]   # the lot at x -20..0
    result = _run(tmp_path, layout=(parcels, roofs))
    (block,) = result["blocks"]
    assert block["facts"]["parcels"] == 11
    assert block["facts"]["vacant_lots"] == 1
    assert _shape(block).contains(_point(-10, 25))


def test_blocks_are_ordered_by_distance_and_limit_says_what_it_dropped(tmp_path):
    result = _run(tmp_path, limit=1)
    assert result["summary"]["count"] == 1
    assert result["summary"]["total_before_limit"] == 2
    assert result["summary"]["truncated_to"] == 1
    assert result["blocks"][0]["kind"] == "street"     # it holds the centre


def test_no_token_is_refused(tmp_path):
    directory = _staged(tmp_path)
    with pytest.raises(ValueError, match="WMS token"):
        populated_areas(directory, wms_base_url=None, cache_dir=tmp_path / "cache")


def test_a_failed_fetch_is_refused_without_the_url_or_token(tmp_path):
    directory = _staged(tmp_path)

    def failing(base_url, layers, bbox, width, height, cache_dir):
        raise RuntimeError(f"GET {base_url} failed")

    with pytest.raises(ValueError) as caught:
        populated_areas(directory, wms_base_url=WMS, cache_dir=tmp_path / "cache",
                        fetch=failing)
    assert TOKEN not in str(caught.value)
    assert caught.value.__context__ is None and caught.value.__cause__ is None


def test_an_area_too_big_to_read_lot_lines_over_is_refused(tmp_path):
    directory = _staged(tmp_path, radius_m=6000.0)
    with pytest.raises(ValueError, match="smaller area"):
        populated_areas(directory, wms_base_url=WMS, cache_dir=tmp_path / "cache",
                        fetch=fake_fetch([], []))


def test_the_panel_pixel_is_said_before_anything_is_fetched(tmp_path):
    """`panel_pixel_m` is the width rule on its own, for a scenario to ask
    before it stages: 1 m up to 4 km across, coarser to 2 m, refused past."""
    from sar_geo_mcp.planning.populated import panel_pixel_m
    from sar_geo_mcp.terrain.grid import read_subject

    def subject(radius_m):
        (tmp_path / str(radius_m)).mkdir()
        return read_subject(_staged(tmp_path / str(radius_m), radius_m=radius_m))["geometry"]

    town_circle = subject(170.0)
    assert panel_pixel_m(town_circle) == 1.0
    assert panel_pixel_m(town_circle, 0.5) == 0.5
    assert 1.4 < panel_pixel_m(subject(3000.0)) < 1.6
    with pytest.raises(ValueError, match="smaller area"):
        panel_pixel_m(subject(6000.0))


def test_the_parameter_checks_are_public_and_the_tool_s_own():
    from sar_geo_mcp.planning.populated import check_params

    rules = {"pixel_m": 1.0, "max_parcel_area_m2": 4047.0, "min_building_m2": 30.0,
             "large_parcel_min_cover": 0.1, "bridge_m": 12.0, "min_parcels": 3,
             "max_distance_m": None, "limit": 0}
    check_params(**rules)
    with pytest.raises(ValueError, match=r"min_parcels \(0\) must be at least 1"):
        check_params(**{**rules, "min_parcels": 0})


@pytest.mark.parametrize("bad", [
    {"pixel_m": 3.0}, {"pixel_m": 0}, {"max_parcel_area_m2": 0}, {"min_building_m2": 0},
    {"large_parcel_min_cover": 0}, {"large_parcel_min_cover": 1.5}, {"bridge_m": -1},
    {"min_parcels": 0}, {"max_distance_m": 0}, {"limit": -1},
])
def test_parameters_out_of_range_are_refused(tmp_path, bad):
    directory = _staged(tmp_path)
    with pytest.raises(ValueError):
        populated_areas(directory, wms_base_url=WMS, cache_dir=tmp_path / "cache",
                        fetch=fake_fetch([], []), **bad)


def test_the_set_carries_block_candidates_with_stable_ids(tmp_path):
    parcels, roofs = town()
    directory = _staged(tmp_path)
    params = populated_params(pixel_m=1.0, max_parcel_area_m2=4047.0, min_building_m2=30.0,
                              large_parcel_min_cover=0.10, bridge_m=12.0, min_parcels=3,
                              max_distance_m=None, limit=0)
    first = populated_set(directory, params, handle="aoi-test", wms_base_url=WMS,
                          cache_dir=tmp_path / "cache", fetch=fake_fetch(parcels, roofs))
    again = populated_set(directory, params, handle="aoi-test", wms_base_url=WMS,
                          cache_dir=tmp_path / "cache", fetch=fake_fetch(parcels, roofs))
    assert first.set_id == again.set_id
    assert first.set_id.startswith("populated-")
    assert [c.id for c in first.candidates] == [f"{first.set_id}.1", f"{first.set_id}.2"]
    assert {c.kind for c in first.candidates} == {"block"}
    assert {c.axis for c in first.candidates} == {"street", "large_parcel"}
    assert first.candidates[0].geometry["type"] in ("Polygon", "MultiPolygon")


def test_parcel_lines_are_resampled_as_a_mask():
    from sar_geo_mcp.caltopo.wms import LAYERS, MAX_ALPHA, resample_filter_for

    assert "parcel" in LAYERS
    assert resample_filter_for(["parcel"]) == MAX_ALPHA


# --------------------------------------------------------------------------
# building markers


def _roof_boxes(roofs):
    from shapely.geometry import box

    return [box(*_merc(x0, y0), *_merc(x1, y1)) for x0, y0, x1, y1 in roofs]


def test_markers_are_off_unless_asked_for(tmp_path):
    result = _run(tmp_path)
    for block in result["blocks"]:
        assert "markers" not in block
        assert "markers" not in block["facts"]
    assert "markers" not in result["summary"]
    assert "markers" not in result["assumptions"]


def test_each_building_gets_one_white_house_marker_on_its_roof(tmp_path):
    from pyproj import Transformer
    from shapely.geometry import Point

    _parcels, roofs = town()
    result = _run(tmp_path, building_markers=True)
    street = next(b for b in result["blocks"] if b["kind"] == "street")
    large = next(b for b in result["blocks"] if b["kind"] == "large_parcel")
    assert len(street["markers"]) == 12 and street["facts"]["markers"] == 12
    assert len(large["markers"]) == 4 and large["facts"]["markers"] == 4
    assert result["summary"]["markers"] == 16
    assert result["assumptions"]["markers"]["symbol"] == "hut"

    boxes = _roof_boxes(roofs)
    to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    marked = []
    for block in (street, large):
        for marker in block["markers"]:
            assert marker["type"] == "Feature"
            assert marker["geometry"]["type"] == "Point"
            properties = marker["properties"]
            assert properties["class"] == "Marker"
            assert properties["marker-symbol"] == "hut"
            assert properties["marker-color"] == "#FFFFFF"
            point = Point(*to_3857.transform(*marker["geometry"]["coordinates"]))
            on = [i for i, roof in enumerate(boxes) if roof.contains(point)]
            assert len(on) == 1, marker
            marked.extend(on)
    # Every roof in a block is marked, and none twice; the farmhouse is in no
    # block, so it is not marked at all.
    farm = len(boxes) - 1
    assert sorted(marked) == [i for i in range(len(boxes)) if i != farm]


def test_the_block_outline_is_the_same_with_or_without_markers(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    plain = _run(tmp_path / "a")
    marked = _run(tmp_path / "b", building_markers=True)
    assert ([b["geometry"] for b in plain["blocks"]]
            == [b["geometry"] for b in marked["blocks"]])


@pytest.mark.parametrize("given, written", [("ff0000", "#FF0000"), ("#00aa33", "#00AA33")])
def test_a_marker_colour_is_written_the_way_caltopo_writes_it(tmp_path, given, written):
    result = _run(tmp_path, building_markers=True, marker_color=given)
    colours = {m["properties"]["marker-color"]
               for b in result["blocks"] for m in b["markers"]}
    assert colours == {written}


@pytest.mark.parametrize("bad", ["white", "#fff", "#GGGGGG", ""])
def test_a_colour_caltopo_cannot_read_is_refused(tmp_path, bad):
    with pytest.raises(ValueError, match="marker_color"):
        _run(tmp_path, building_markers=True, marker_color=bad)


def test_marker_parameters_do_not_change_the_set_id_of_a_call_without_markers():
    base = dict(pixel_m=1.0, max_parcel_area_m2=4047.0, min_building_m2=30.0,
                large_parcel_min_cover=0.10, bridge_m=12.0, min_parcels=3,
                max_distance_m=None, limit=0)
    assert populated_params(**base) == populated_params(**base, marker_color="#123456")
    assert "building_markers" not in populated_params(**base)
    marked = populated_params(**base, building_markers=True, marker_color="ff0000")
    assert marked["marker_color"] == "#FF0000"
    assert populated_params(**base) != marked


def test_markers_travel_with_the_candidate_and_stay_out_of_summaries(tmp_path):
    from sar_geo_mcp.planning.candidates import summaries

    parcels, roofs = town()
    directory = _staged(tmp_path)
    params = populated_params(pixel_m=1.0, max_parcel_area_m2=4047.0, min_building_m2=30.0,
                              large_parcel_min_cover=0.10, bridge_m=12.0, min_parcels=3,
                              max_distance_m=None, limit=0, building_markers=True)
    cset = populated_set(directory, params, handle="aoi-test", wms_base_url=WMS,
                         cache_dir=tmp_path / "cache", fetch=fake_fetch(parcels, roofs))
    assert [len(c.markers) for c in cset.candidates] == [12, 4]
    for summary in summaries(cset):
        assert "markers" not in summary and "geometry" not in summary
        assert summary["facts"]["markers"] in (12, 4)


# --------------------------------------------------------------------------
# one piece per block, no shared ground (0.20.0)


def _raw(members, mask):
    import numpy as np

    return {"kind": "street", "street_index": 1, "members": members,
            "window": (0, mask.shape[0], 0, mask.shape[1]), "mask": np.asarray(mask)}


def test_a_run_cut_by_another_blocks_lot_becomes_two_blocks():
    """Lots 1 and 3 front one street, lot 2 between them another: the run
    falls apart around lot 2, and each piece is a block with its own lot."""
    import numpy as np
    from affine import Affine
    from scipy import ndimage

    from sar_geo_mcp.planning.populated import _one_piece_each

    regions = np.zeros((10, 30), dtype=np.int32)
    regions[:, 0:10], regions[:, 10:20], regions[:, 20:30] = 1, 2, 3
    lot_mask = np.array([False, True, True, True])
    run = _raw([1, 3], np.ones((10, 30), dtype=bool))          # closed over lot 2
    middle = _raw([2], regions == 2)
    pieces = _one_piece_each([run, middle], regions=regions, lot_mask=lot_mask,
                             block_owner=np.zeros(4, dtype=np.int32),
                             transform=Affine.identity(), centre=(15.0, 5.0))
    assert sorted(p["members"] for p in pieces) == [[1], [2], [3]]
    assert [p["split"] for p in pieces if p["members"] != [2]] == [True, True]
    total = np.zeros((10, 30), dtype=int)
    for piece in pieces:
        total += piece["mask"]
        assert ndimage.label(piece["mask"])[1] == 1          # one piece each
    assert total.max() == 1                                   # no cell in two


def test_ground_two_blocks_both_reach_goes_to_the_nearer():
    """Both closings reach the street between lot 1 and lot 2; the block
    nearer the centre keeps it and the other gives it up."""
    import numpy as np
    from affine import Affine

    from sar_geo_mcp.planning.populated import _one_piece_each

    regions = np.zeros((10, 25), dtype=np.int32)
    regions[:, 0:10], regions[:, 10:15], regions[:, 15:25] = 1, 4, 2   # 4: the street
    lot_mask = np.array([False, True, True, False, False])
    west = np.zeros((10, 25), dtype=bool)
    west[:, 0:15] = True
    east = np.zeros((10, 25), dtype=bool)
    east[:, 10:25] = True
    pieces = _one_piece_each([_raw([1], west), _raw([2], east)], regions=regions,
                             lot_mask=lot_mask, block_owner=np.zeros(5, dtype=np.int32),
                             transform=Affine.identity(), centre=(20.0, 5.0))  # nearer east
    by_member = {tuple(p["members"]): p["mask"] for p in pieces}
    assert by_member[(2,)][:, 10:15].all()                    # east kept the street
    assert not by_member[(1,)][:, 10:15].any()
    assert not (by_member[(1,)] & by_member[(2,)]).any()


def test_every_block_is_one_polygon_with_no_shared_ground(tmp_path):
    result = _run(tmp_path)
    outlines = [_shape(b) for b in result["blocks"]]
    assert all(o.geom_type == "Polygon" for o in outlines)
    for i, a in enumerate(outlines):
        for b in outlines[i + 1:]:
            assert a.intersection(b).area * SCALE * SCALE < 0.01


def test_a_street_running_past_the_first_read_is_read_out_to_its_end(tmp_path):
    """Elm Street's lots run 500 m east, past the 200 m margin first read
    round a 170 m circle: the panel is read wider until the block the
    circle keeps no longer runs into its edge, so the street is drawn to
    its end rather than cut where the panel stopped."""
    parcels, roofs = [], []
    for i in range(28):
        x0 = -60 + 20 * i
        for y0, y1 in ((10, 40), (-40, -10)):
            parcels.append(_rect(x0, y0, x0 + 20, y1))
            cy = (y0 + y1) / 2
            roofs.append(_rect(x0 + 5, cy - 5, x0 + 15, cy + 5))
    directory = _staged(tmp_path)
    street = elm_street()
    street["geometry"]["coordinates"] = [list(_merc(-150, 0)), list(_merc(600, 0))]
    (directory / "osm.json").write_text(json.dumps(
        {"routes": [street], "parking": [], "query_version": OSM_QUERY_VERSION}),
        encoding="utf-8")
    calls = []
    result = populated_areas(directory, wms_base_url=WMS, cache_dir=tmp_path / "cache",
                             fetch=fake_fetch(parcels, roofs, calls=calls))
    assert result["assumptions"]["margin_m"] > 200.0
    (block,) = [b for b in result["blocks"] if b["name"] == "Elm Street"]
    assert block["facts"]["clipped_at_edge"] is False
    assert block["facts"]["parcels"] == 56
    assert _shape(block).covers(_point(490, 25))
    assert len(calls) > 2                              # read again, wider


def test_a_block_within_the_first_read_is_read_once(tmp_path):
    calls = []
    directory = _staged(tmp_path)
    parcels, roofs = town()
    result = populated_areas(directory, wms_base_url=WMS, cache_dir=tmp_path / "cache",
                             fetch=fake_fetch(parcels, roofs, calls=calls))
    assert result["assumptions"]["margin_m"] == 200.0
    assert len(calls) == 2                             # parcels and structures, once


def _street(name, x0, y0, x1, y1, way="way/9"):
    return {"id": way, "kind": "route",
            "geometry": {"type": "LineString",
                         "coordinates": [list(_merc(x0, y0)), list(_merc(x1, y1))]},
            "properties": {"name": name, "route_type": "ROAD"}}


def _lots_along(x_starts, *, width=20):
    """Lots both sides of Elm Street (y 10..40 and -40..-10), a roof on each."""
    parcels, roofs = [], []
    for x0 in x_starts:
        for y0, y1 in ((10, 40), (-40, -10)):
            parcels.append(_rect(x0, y0, x0 + width, y1))
            cy = (y0 + y1) / 2
            roofs.append(_rect(x0 + 5, cy - 5, x0 + width - 5, cy + 5))
    return parcels, roofs


def _staged_with(tmp_path, routes, *, radius_m):
    directory = _staged(tmp_path, radius_m=radius_m)
    (directory / "osm.json").write_text(json.dumps(
        {"routes": routes, "parking": [], "query_version": OSM_QUERY_VERSION}),
        encoding="utf-8")
    return directory


def test_a_cross_street_ends_a_run_and_the_run_past_it_is_its_own_block(tmp_path):
    """Oak Street crosses Elm Street at x=200, a 20 m right-of-way the
    closing would span. Elm's run is cut there: the circle keeps the run up
    to Oak, and the run past it - which the circle does not reach - is no
    part of this answer."""
    parcels, roofs = _lots_along([-60 + 20 * i for i in range(13)]          # -60..200
                                 + [210 + 20 * i for i in range(8)])        # 210..370
    parcels = [lot for lot in parcels if not (lot[0] < 210 and lot[2] > 190)]
    roofs = [roof for roof in roofs if not (roof[0] < 210 and roof[2] > 190)]
    routes = [_street("Elm Street", -150, 0, 450, 0, "way/1"),
              _street("Oak Street", 200, -150, 200, 150, "way/2")]
    directory = _staged_with(tmp_path, routes, radius_m=170.0)
    result = populated_areas(directory, wms_base_url=WMS, cache_dir=tmp_path / "cache",
                             fetch=fake_fetch(parcels, roofs))
    (block,) = [b for b in result["blocks"] if b["name"] == "Elm Street"]
    outline = _shape(block)
    assert outline.covers(_point(170, 25))
    assert not outline.intersects(_point(260, 25))
    assert block["facts"]["parcels"] == 24                 # 12 a side, -60..180


def test_a_run_is_finished_out_to_its_cross_street(tmp_path):
    """The lots at the end of Elm's run face Oak Street, the cross street,
    and are filed under it; Oak's block lies outside the circle. They are
    searched with Elm's run, so its edge is the cross street rather than a
    back fence short of it - and Oak's far side stays out."""
    elm, elm_roofs = _lots_along([-60 + 20 * i for i in range(9)])           # -60..120
    corner = [_rect(120, 10, 140, 40), _rect(120, -40, 140, -10)]            # face Oak
    corner_roofs = [_rect(124, 15, 136, 35), _rect(124, -35, 136, -15)]
    far = [_rect(160, y, 190, y + 20) for y in (-60, -40, -20, 0, 20, 40)]   # Oak's far side
    far_roofs = [_rect(165, y + 5, 185, y + 15) for y in (-60, -40, -20, 0, 20, 40)]
    routes = [_street("Elm Street", -150, 0, 150, 0, "way/1"),
              _street("Oak Street", 150, -150, 150, 150, "way/2")]
    directory = _staged_with(tmp_path, routes, radius_m=60.0)
    result = populated_areas(directory, wms_base_url=WMS, cache_dir=tmp_path / "cache",
                             fetch=fake_fetch(elm + corner + far,
                                              elm_roofs + corner_roofs + far_roofs))
    (block,) = [b for b in result["blocks"] if b["name"] == "Elm Street"]
    outline = _shape(block)
    assert outline.covers(_point(130, 25)) and outline.covers(_point(130, -25))
    assert not outline.intersects(_point(175, 30))
    assert block["facts"]["parcels"] == 20
