"""`lz_candidates`: where a helicopter can put down, graded the way the
CalTopo First Response safety course grades a landing zone.

Every raster here is synthetic and one metre per pixel unless a test says
otherwise, so a pixel is a metre and the numbers in the assertions are
metres."""

from __future__ import annotations

import io
import json
import math
import re
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point, shape as shapely_shape

from test_grid import make_wgs84_square, write_fetch_dir

from sar_geo_mcp.caltopo.wms import LAYERS
from sar_geo_mcp.planning.lz import (
    CHIP_FAILED_WARNING,
    CHIP_M,
    CHIP_PX,
    FOOTPRINT_ANGLES_DEG,
    MINOR_LINES_NOT_CHECKED,
    MISSING_OSM_WARNING,
    NO_CANOPY_ERROR,
    NO_TOKEN_CHIP_WARNING,
    POWER_NOT_FETCHED_WARNING,
    POWER_UNLABELLED_WARNING,
    SOURCE_REFUSED_WARNING,
    STRUCTURES_COVERED_MOST,
    STRUCTURES_NOT_CHECKED,
    STRUCTURES_NO_TRANSPARENCY,
    TREES_NOT_CHECKED,
    WATER_EVT_MISSING_WARNING,
    WATER_NOT_CHECKED_WARNING,
    WATER_SOURCE_EVT,
    WATER_SOURCE_OSM,
    _closed_near_lines,
    _closed_near_structures,
    largest_inscribed_circle,
    lz_candidates,
    rectangle_fits_at,
    rectangle_footprint,
)
from sar_geo_mcp.sources.osm import OSM_QUERY_VERSION

TO4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform
ORIGIN = (-13618826.5, 4486605.0)
# One metre per pixel needs the Mercator pixel scaled by 1/cos(lat) at the
# origin's latitude; `write_fetch_dir` stamps the latitude the grid reads.
LATITUDE = 37.34
_UNITS_PER_M = 1.0 / math.cos(math.radians(LATITUDE))

MAGENTA = (200, 60, 120)   # vegetation in CIR: high NIR (R), low red (G)
TAN = (180, 170, 150)      # bare ground


def test_rectangle_footprint_is_the_rectangle_at_angle_zero():
    footprint = rectangle_footprint(length_px=6, width_px=4, angle_deg=0.0)
    rows, cols = footprint.shape
    assert rows == cols and rows % 2 == 1
    centre = rows // 2
    # Six along the columns, four along the rows, at angle zero.
    assert footprint[centre, centre - 2:centre + 3].all()
    assert footprint[centre - 1:centre + 2, centre].all()
    assert not footprint[centre - 3, centre]
    # |along| <= 3 is seven cells, |across| <= 2 is five: 35 exactly.
    assert int(footprint.sum()) == 35


@pytest.mark.parametrize("length_px,width_px", [(6, 4), (12, 4), (40, 30)])
def test_rectangle_footprint_at_ninety_swaps_the_axes(length_px, width_px):
    zero = rectangle_footprint(length_px=length_px, width_px=width_px, angle_deg=0.0)
    ninety = rectangle_footprint(length_px=length_px, width_px=width_px, angle_deg=90.0)
    assert np.array_equal(ninety, zero.T)


def test_largest_inscribed_circle_of_a_square_is_at_its_centre():
    mask = np.zeros((41, 41), dtype=bool)
    mask[10:31, 10:31] = True           # a 21 px square
    hit = largest_inscribed_circle(mask, pixel_m=1.0)
    assert hit is not None
    row, col, radius_m = hit
    assert (row, col) == (20, 20)
    assert radius_m == pytest.approx(11.0, abs=1.0)


def test_largest_inscribed_circle_treats_the_raster_edge_as_an_edge():
    """Open ground running off the raster is not infinitely open."""
    mask = np.ones((21, 21), dtype=bool)
    row, col, radius_m = largest_inscribed_circle(mask, pixel_m=1.0)
    assert (row, col) == (10, 10)
    assert radius_m == pytest.approx(11.0, abs=1.0)


def test_largest_inscribed_circle_of_nothing_is_none():
    assert largest_inscribed_circle(np.zeros((5, 5), dtype=bool), pixel_m=1.0) is None


def test_rectangle_fits_only_at_the_angle_that_matches_the_ground():
    """A 50 by 70 opening takes a 45 by 60 footprint only with the long side
    along the long axis - at 90 degrees, not 0."""
    mask = np.zeros((120, 120), dtype=bool)
    mask[20:90, 30:80] = True             # 70 rows tall, 50 cols wide
    assert rectangle_fits_at(mask, length_px=60, width_px=45,
                             angles_deg=(0.0,)) is None
    assert rectangle_fits_at(mask, length_px=60, width_px=45,
                             angles_deg=(0.0, 90.0)) == 90.0


def test_rectangle_does_not_fit_a_gap_narrower_than_its_width():
    mask = np.zeros((120, 120), dtype=bool)
    mask[:, 40:70] = True                 # a 30 px wide strip
    assert rectangle_fits_at(mask, length_px=60, width_px=45,
                             angles_deg=(0.0, 30.0, 60.0, 90.0, 120.0, 150.0)) is None


def test_rectangle_fits_a_strip_drawn_along_thirty_degrees_only_at_thirty():
    """Pins the rotation and its handedness: a strip whose long axis runs
    30 degrees anticlockwise from east on a north-up map (rows increase
    southward, so it climbs to the north-east) takes the footprint at 30
    and at no axis-aligned angle."""
    rr, cc = np.mgrid[:120, :120]
    u, v = cc - 60, rr - 60
    a = math.radians(-30.0)          # negated: rows go south
    along = u * math.cos(a) + v * math.sin(a)
    across = -u * math.sin(a) + v * math.cos(a)
    mask = (np.abs(along) <= 40) & (np.abs(across) <= 12)
    assert rectangle_fits_at(mask, length_px=60, width_px=20,
                             angles_deg=(0.0, 90.0)) is None
    assert rectangle_fits_at(mask, length_px=60, width_px=20,
                             angles_deg=(0.0, 30.0)) == 30.0
    # The strip climbs north-east: the cell 30 columns east of centre that
    # lies on the axis is north of centre (a smaller row), not south.
    on_axis_rows = rr[(np.abs(across) <= 0.5) & (cc == 90)]
    # Built from the compass bearing, not the module's formula: a flipped sign in `rectangle_footprint` fails here and nowhere else.
    assert on_axis_rows.size and on_axis_rows.max() < 60


def stage(tmp_path, elevation, vegetated, pixel_m=1.0):
    """A staged directory: `elevation` in metres on a `pixel_m` grid,
    `vegetated` a bool mask painted into the CIR panel. The subject is the
    raster minus a two-pixel rim, exactly as `test_site_facts.stage` insets
    it."""
    rows, cols = elevation.shape
    pixel = pixel_m * _UNITS_PER_M
    ox, oy = ORIGIN
    transform = from_origin(ox, oy, pixel, pixel)
    inset = 2 * pixel
    west, north = TO4326(ox + inset, oy - inset)
    east, south = TO4326(ox + cols * pixel - inset, oy - rows * pixel + inset)
    write_fetch_dir(tmp_path, elevation, transform,
                    make_wgs84_square(west, south, east, north), latitude=LATITUDE)
    pixels = np.zeros((rows, cols, 3), dtype="uint8")
    pixels[:] = TAN
    pixels[vegetated] = MAGENTA
    (tmp_path / "panels").mkdir()
    Image.fromarray(pixels, "RGB").save(tmp_path / "panels" / "vegetation.png")
    return tmp_path


def centre_wgs84(rows, cols):
    ox, oy = ORIGIN
    lon, lat = TO4326(ox + cols / 2 * _UNITS_PER_M, oy - rows / 2 * _UNITS_PER_M)
    return lat, lon


def run(directory, rows=200, cols=200, **overrides):
    lat, lon = centre_wgs84(rows, cols)
    return lz_candidates(directory, center_latitude=lat, center_longitude=lon,
                         **overrides)


def plane(rows, cols, slope_deg, axis="col"):
    """Elevation rising at `slope_deg` along columns (or rows), 1 m pixels."""
    rise = math.tan(math.radians(slope_deg))
    index = np.arange(cols) if axis == "col" else np.arange(rows)[:, None]
    return np.broadcast_to(index * rise, (rows, cols)).astype("float64").copy()


def three_openings():
    """30 m at the centre, 70 m to the north-east, 30 m to the south-west:
    the ground the ordering and cap tests share."""
    vegetated = np.ones((300, 300), dtype=bool)
    vegetated[135:165, 135:165] = False       # centre, 30 m
    vegetated[15:85, 215:285] = False         # north-east, 70 m, ~140 m out
    vegetated[250:280, 20:50] = False         # south-west, 30 m, ~160 m out
    return vegetated


def test_a_flat_open_field_is_one_preferred_candidate_at_its_centre(tmp_path):
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    result = run(directory)
    assert len(result["candidates"]) == 1
    only = result["candidates"][0]
    assert only["grade"] == "preferred"
    assert only["preferred_fit"]["fits"] is True
    assert only["preferred_fit"]["angle_deg"] in (0.0, 30.0, 60.0, 90.0, 120.0, 150.0)
    lat, lon = centre_wgs84(200, 200)
    assert only["touchdown"]["coordinates"][1] == pytest.approx(lat, abs=3e-5)
    assert only["touchdown"]["coordinates"][0] == pytest.approx(lon, abs=3e-5)
    # 196 m of open ground inside the rim: a ~98 m radius, so ~196 m across.
    assert only["clear_diameter_m"] == pytest.approx(196.0, abs=6.0)
    assert only["geometry"]["type"] in ("Polygon", "MultiPolygon")


def test_a_gap_narrower_than_the_circle_is_nothing(tmp_path):
    vegetated = np.ones((200, 200), dtype=bool)
    vegetated[:, 90:110] = False           # a 20 m wide open strip
    directory = stage(tmp_path, np.zeros((200, 200)), vegetated)
    assert run(directory)["candidates"] == []


def test_a_gap_that_takes_the_circle_but_not_the_rectangle_is_minimal(tmp_path):
    vegetated = np.ones((200, 200), dtype=bool)
    vegetated[:, 85:115] = False           # 30 m wide
    directory = stage(tmp_path, np.zeros((200, 200)), vegetated)
    candidates = run(directory)["candidates"]
    assert [c["grade"] for c in candidates] == ["minimal"]
    assert candidates[0]["preferred_fit"]["fits"] is False
    assert candidates[0]["clear_diameter_m"] == pytest.approx(30.0, abs=3.0)


def test_a_fifty_by_seventy_opening_is_preferred_with_the_long_side_along_it(tmp_path):
    vegetated = np.ones((200, 200), dtype=bool)
    vegetated[60:130, 75:125] = False      # 70 rows tall, 50 cols wide
    directory = stage(tmp_path, np.zeros((200, 200)), vegetated)
    candidates = run(directory)["candidates"]
    assert [c["grade"] for c in candidates] == ["preferred"]
    assert candidates[0]["preferred_fit"]["angle_deg"] == 90.0


def test_a_six_degree_open_field_is_marginal(tmp_path):
    directory = stage(tmp_path, plane(200, 200, 6.0), np.zeros((200, 200), dtype=bool))
    candidates = run(directory)["candidates"]
    assert [c["grade"] for c in candidates] == ["marginal"]


def test_an_eight_degree_open_field_is_nothing(tmp_path):
    directory = stage(tmp_path, plane(200, 200, 8.0), np.zeros((200, 200), dtype=bool))
    assert run(directory)["candidates"] == []


def test_a_flat_pocket_in_a_six_degree_field_is_minimal_and_the_field_marginal(tmp_path):
    """The pocket is its own patch: its rim is a step, steeper than 7, so the
    7-degree mask splits there. The pocket grades from the 5-degree mask
    (minimal: 30 m across, no rectangle) and the field around it from the
    7-degree one (marginal).

    The pocket is sunk 3 m below the plane at its centre so the step runs
    all the way round. Level with its own centre it would meet the plane
    with NO step at the two points where the plane's contour crosses the
    disc, the 7-degree mask would stay joined through them, and the pocket
    and field would be one patch and one candidate."""
    elevation = plane(200, 200, 6.0)
    rr, cc = np.ogrid[:200, :200]
    pocket = (rr - 100) ** 2 + (cc - 100) ** 2 <= 18 ** 2
    elevation[pocket] = float(elevation[100, 100]) - 3.0
    directory = stage(tmp_path, elevation, np.zeros((200, 200), dtype=bool))
    candidates = run(directory)["candidates"]
    grades = sorted(c["grade"] for c in candidates)
    assert grades == ["marginal", "minimal"]
    pocket_candidate = next(c for c in candidates if c["grade"] == "minimal")
    # 36 m across, less the two-cell steep rim the 3x3 gradient paints.
    assert pocket_candidate["clear_diameter_m"] == pytest.approx(32.0, abs=5.0)


def test_a_patch_a_quarter_vegetated_is_dropped_and_a_sixth_is_kept(tmp_path):
    rr, cc = np.ogrid[:200, :200]
    quarter = ((rr % 2) == 0) & ((cc % 2) == 0)            # 25 %
    directory = stage(tmp_path, np.zeros((200, 200)), quarter)
    assert run(directory)["candidates"] == []

    sixth = ((rr % 3) == 0) & ((cc % 2) == 0)              # 16.7 %
    (tmp_path / "b").mkdir()
    directory = stage(tmp_path / "b", np.zeros((200, 200)), sixth)
    candidates = run(directory)["candidates"]
    assert len(candidates) == 1
    assert candidates[0]["canopy_fraction"] == pytest.approx(0.167, abs=0.03)


def test_two_pockets_in_one_field_are_graded_each_on_its_own_ground(tmp_path):
    """Pocket A is a 52 m flat disc, pocket B a 48 by 64 m flat rectangle,
    both sunk 6 m into the same 6-degree field. B is preferred with the
    rectangle along its long side, A is minimal, the field around them is
    marginal, and each polygon holds its own touchdown and not the other's.
    Before the per-pocket rule, one entry could carry B's fit at A's
    touchdown.

    Six metres, because the field drops 2.7 m across A's radius: sunk
    only 3 m, A's downhill rim would be a 0.27 m step that a 3x3 gradient
    reads as flat, A would join the field's 7-degree patch, and the field
    would rightly not be offered."""
    elevation = plane(300, 300, 6.0)
    rr, cc = np.ogrid[:300, :300]
    a = (rr - 150) ** 2 + (cc - 80) ** 2 <= 26 ** 2
    b = (np.abs(rr - 150) <= 32) & (np.abs(cc - 210) <= 24)   # 64 rows by 48 cols
    elevation[a] = float(elevation[150, 80]) - 6.0
    elevation[b] = float(elevation[150, 210]) - 6.0
    directory = stage(tmp_path, elevation, np.zeros((300, 300), dtype=bool))
    candidates = run(directory, rows=300, cols=300)["candidates"]
    by_grade = {c["grade"]: c for c in candidates}
    assert sorted(by_grade) == ["marginal", "minimal", "preferred"]
    assert by_grade["preferred"]["preferred_fit"]["angle_deg"] == 90.0
    point_a = Point(by_grade["minimal"]["touchdown"]["coordinates"])
    point_b = Point(by_grade["preferred"]["touchdown"]["coordinates"])
    polygon_a = shapely_shape(by_grade["minimal"]["geometry"])
    polygon_b = shapely_shape(by_grade["preferred"]["geometry"])
    assert polygon_a.contains(point_a) and not polygon_a.contains(point_b)
    assert polygon_b.contains(point_b) and not polygon_b.contains(point_a)


def test_a_level_pocket_is_graded_from_its_own_polygon_and_its_field_is_not_offered(tmp_path):
    """Level with its centre, the pocket's rim is no step where the field's
    contour crosses it, so pocket and field are ONE 7-degree patch. The
    pocket is still its own 5-degree pocket: minimal, with a polygon the
    size of the pocket and not the field. The field is not offered as
    marginal, because its patch holds a surviving pocket."""
    elevation = plane(200, 200, 6.0)
    rr, cc = np.ogrid[:200, :200]
    pocket = (rr - 100) ** 2 + (cc - 100) ** 2 <= 18 ** 2
    elevation[pocket] = float(elevation[100, 100])
    directory = stage(tmp_path, elevation, np.zeros((200, 200), dtype=bool))
    candidates = run(directory)["candidates"]
    assert [c["grade"] for c in candidates] == ["minimal"]
    minx, _miny, maxx, _maxy = shapely_shape(candidates[0]["geometry"]).bounds
    assert (maxx - minx) * 111_320 * math.cos(math.radians(LATITUDE)) < 60.0


def test_a_vegetated_pocket_does_not_take_the_clean_field_down_with_it(tmp_path):
    """The level pocket is a quarter vegetated and is dropped for canopy;
    the 6-degree field around it is then offered as marginal, its own
    canopy being under one percent. Before the canopy check moved into
    grading, the dropped pocket silenced the whole patch."""
    elevation = plane(200, 200, 6.0)
    rr, cc = np.ogrid[:200, :200]
    pocket = (rr - 100) ** 2 + (cc - 100) ** 2 <= 18 ** 2
    elevation[pocket] = float(elevation[100, 100])
    vegetated = pocket & ((rr % 2) == 0) & ((cc % 2) == 0)
    directory = stage(tmp_path, elevation, vegetated)
    candidates = run(directory)["candidates"]
    assert [c["grade"] for c in candidates] == ["marginal"]
    assert candidates[0]["canopy_fraction"] < 0.02


def test_a_dem_hole_is_closed_ground_not_flat_ground(tmp_path):
    """`np.gradient` leaves a NaN cell's own slope finite while its
    neighbours go NaN, so without the finite guard a void would grade as
    flat. Everything but a 20 m strip is void: nothing fits."""
    elevation = np.full((200, 200), np.nan)
    elevation[:, 90:110] = 0.0
    directory = stage(tmp_path, elevation, np.zeros((200, 200), dtype=bool))
    assert run(directory)["candidates"] == []


def test_a_hundred_openings_on_a_large_raster_are_a_hundred_candidates(tmp_path):
    """A 600 m field of trees with a 10 by 10 grid of 40 m openings. Every
    opening is its own pocket; the count pins the per-pocket rule and the
    windowed work keeps this under a few seconds where the full-raster
    version took minutes."""
    vegetated = np.ones((600, 600), dtype=bool)
    for r in range(10):
        for c in range(10):
            vegetated[20 + r * 58: 60 + r * 58, 20 + c * 58: 60 + c * 58] = False
    directory = stage(tmp_path, np.zeros((600, 600)), vegetated)
    candidates = run(directory, rows=600, cols=600, limit=0)["candidates"]
    assert len(candidates) == 100
    assert {c["grade"] for c in candidates} == {"minimal"}


def test_facts_are_measured_on_the_footprint(tmp_path):
    """A 3-degree plane: slope reads 3, elevation reads the plane at the
    touchdown, aspect is the downhill direction (west for a plane rising
    along the columns, since aspect is the bearing of steepest DESCENT)."""
    directory = stage(tmp_path, plane(200, 200, 3.0), np.zeros((200, 200), dtype=bool))
    only = run(directory)["candidates"][0]
    assert only["slope_mean_deg"] == pytest.approx(3.0, abs=0.3)
    assert only["slope_max_deg"] == pytest.approx(3.0, abs=0.3)
    assert only["elevation_m"] == pytest.approx(100 * math.tan(math.radians(3.0)), abs=1.0)
    assert only["aspect"] == "W"
    assert only["distance_m"] == pytest.approx(0.0, abs=3.0)
    # Under the planner's own feet: a bearing to that cell is not a fact.
    assert only["bearing_deg"] is None


def test_nearest_first_ids_in_that_order_and_the_cap_counts_grades_before_it(tmp_path):
    """Three openings: 30 m at the centre (minimal), 70 m to the north-east
    (preferred), 30 m to the south-west (minimal). Nearest first; a limit
    of 2 keeps the first two and the summary still counts all three."""
    directory = stage(tmp_path, np.zeros((300, 300)), three_openings())
    result = run(directory, rows=300, cols=300, limit=2)

    candidates = result["candidates"]
    assert [c["id"] for c in candidates] == ["lz_1", "lz_2"]
    assert [c["grade"] for c in candidates] == ["minimal", "preferred"]
    assert candidates[0]["distance_m"] < candidates[1]["distance_m"]
    assert 30.0 <= candidates[1]["bearing_deg"] <= 60.0   # north-east
    assert result["summary"] == {
        "count": 2, "total_before_limit": 3, "truncated_to": 2,
        "grades": {"preferred": 1, "minimal": 2, "marginal": 0},
    }


def test_limit_zero_returns_every_candidate(tmp_path):
    directory = stage(tmp_path, np.zeros((300, 300)), three_openings())
    result = run(directory, rows=300, cols=300, limit=0)
    assert result["summary"]["count"] == 3
    assert result["summary"]["truncated_to"] is None


def test_the_envelope_names_every_threshold_and_says_the_pilot_decides(tmp_path):
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    result = run(directory, max_slope_deg=4.0, marginal_slope_deg=6.0,
                 max_canopy=0.1, min_diameter_m=20.0,
                 preferred_width_m=40.0, preferred_length_m=50.0)
    assumptions = result["assumptions"]
    assert assumptions["max_slope_deg"] == 4.0
    assert assumptions["marginal_slope_deg"] == 6.0
    assert assumptions["max_canopy"] == 0.1
    assert assumptions["min_diameter_m"] == 20.0
    assert assumptions["preferred_footprint_m"] == [40.0, 50.0]
    assert assumptions["footprint_angles_deg"] == list(FOOTPRINT_ANGLES_DEG)
    assert "anticlockwise from east" in assumptions["angle_frame"]
    assert "pilot decides" in assumptions["grading"]
    assert "CIR" in assumptions["canopy_method"]
    assert result["center"] == {"latitude": pytest.approx(centre_wgs84(200, 200)[0]),
                                "longitude": pytest.approx(centre_wgs84(200, 200)[1])}
    assert result["resolution_m"] == pytest.approx(1.0, abs=0.05)
    assert "aspect_rule" in assumptions and "facts_frame" in assumptions
    assert assumptions["canopy_threshold"] == 0.15  # a literal on purpose: a silent change to the threshold must fail here
    # Working state never leaks.
    assert not any(key.startswith("_") for key in result["candidates"][0])


def test_slope_limits_out_of_order_are_refused(tmp_path):
    """A marginal limit below the preferred one would let a 5 to 7 degree
    pocket grade minimal and leave its 7-degree label at zero."""
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    with pytest.raises(ValueError, match="marginal_slope_deg"):
        run(directory, max_slope_deg=7.0, marginal_slope_deg=5.0)


def test_a_negative_limit_is_refused(tmp_path):
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    with pytest.raises(ValueError, match="negative"):
        run(directory, limit=-1)


def test_a_flat_field_has_no_aspect_and_a_mixed_disc_has_none_either(tmp_path):
    """Steepest descent on a flat field is the sign of DEM noise; a disc
    that falls away east on one half and west on the other has no single
    aspect. Both are null, never a confident compass point."""
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    assert run(directory)["candidates"][0]["aspect"] is None

    rr, cc = np.ogrid[:200, :200]
    ridge = -np.abs(cc - 100) * math.tan(math.radians(3.0))    # a 3-degree roof ridge along rows
    (tmp_path / "b").mkdir()
    directory = stage(tmp_path / "b", np.broadcast_to(ridge, (200, 200)).copy(),
                      np.zeros((200, 200), dtype=bool))
    candidates = run(directory)["candidates"]
    assert candidates and candidates[0]["aspect"] is None


def test_at_equal_distance_the_better_grade_then_the_wider_opening_comes_first(tmp_path):
    """Mirrored east and west of centre at the same distance: preferred
    beats minimal, and between two minimals the wider one comes first.
    That is the tie-break that keeps lz_n meaning the same thing to a
    second planner.

    The tie holds by symmetry, not tuning: the touchdown of an even-sized
    opening is its upper-left centre cell, so cols 30:100 and 221:251 put
    the touchdowns at 64 and 235, mirror images about the centre meridian,
    and geodesic distance is symmetric across it. If the tie rule in
    `largest_inscribed_circle` ever changes, this assertion fails rather
    than going quiet."""
    vegetated = np.ones((300, 300), dtype=bool)
    vegetated[115:185, 30:100] = False        # west, 70 m: preferred
    vegetated[135:165, 221:251] = False       # east, 30 m: minimal, same distance
    directory = stage(tmp_path, np.zeros((300, 300)), vegetated)
    candidates = run(directory, rows=300, cols=300)["candidates"]
    assert [c["grade"] for c in candidates] == ["preferred", "minimal"]
    assert candidates[0]["id"] == "lz_1"
    assert candidates[0]["distance_m"] == candidates[1]["distance_m"]

    vegetated = np.ones((300, 300), dtype=bool)
    vegetated[130:170, 45:85] = False         # west, 40 m: minimal
    vegetated[135:165, 221:251] = False       # east, 30 m: minimal, same distance
    (tmp_path / "b").mkdir()
    directory = stage(tmp_path / "b", np.zeros((300, 300)), vegetated)
    candidates = run(directory, rows=300, cols=300)["candidates"]
    assert [round(c["clear_diameter_m"]) for c in candidates] == [39, 29]
    assert candidates[0]["distance_m"] == candidates[1]["distance_m"]


def stage_without_panel(tmp_path, elevation):
    directory = stage(tmp_path, elevation, np.zeros(elevation.shape, dtype=bool))
    (directory / "panels" / "vegetation.png").unlink()
    (directory / "panels").rmdir()
    return directory


def test_without_a_panel_or_a_token_it_refuses_with_the_reason(tmp_path):
    directory = stage_without_panel(tmp_path, np.zeros((200, 200)))
    with pytest.raises(ValueError, match="canopy"):
        run(directory)


def test_without_a_panel_it_fetches_the_ir_layer_at_the_raster_and_keeps_it(tmp_path):
    """The fetched pixels drive the grade - not just a file appearing - and
    the fetch is georeferenced to the raster it is filling in for: a
    MAGENTA (vegetated) panel everywhere except a 30 m open strip at cols
    85:115, the inverse of `test_a_gap_that_takes_the_circle_but_not_the_
    rectangle_is_minimal`, grades minimal at 30 m, and the bbox the fake
    was called with is the raster's own."""
    directory = stage_without_panel(tmp_path, np.zeros((200, 200)))
    (tmp_path / "cache").mkdir()
    calls = []

    def fake_fetch(base_url, layers, bbox, width, height, cache_dir):
        calls.append((layers, bbox, width, height))
        if layers == ["structures"]:    # no building: a transparent panel
            return _structures_fetch()(base_url, layers, bbox, width, height, cache_dir)
        pixels = np.zeros((height, width, 3), dtype="uint8")
        pixels[:] = MAGENTA
        pixels[:, 85:115] = TAN            # 30 m wide open strip
        path = Path(cache_dir) / "ir.png"
        Image.fromarray(pixels, "RGB").save(path)
        return path

    result = run(directory, wms_base_url="https://caltopo.example/wms",
                 cache_dir=tmp_path / "cache", fetch=fake_fetch)
    assert (directory / "panels" / "vegetation.png").is_file()
    assert result["candidates"][0]["grade"] == "minimal"
    assert result["candidates"][0]["clear_diameter_m"] == pytest.approx(30.0, abs=3.0)

    ox, oy = ORIGIN
    u = _UNITS_PER_M
    assert calls[0][0] == ["ir"]
    assert calls[0][1] == pytest.approx((ox, oy - 200 * u, ox + 200 * u, oy))
    assert calls[0][2:] == (200, 200)


def test_a_failing_panel_fetch_is_refused_as_no_canopy(tmp_path):
    directory = stage_without_panel(tmp_path, np.zeros((200, 200)))
    (tmp_path / "cache").mkdir()

    def fake_fetch(base_url, layers, bbox, width, height, cache_dir):
        raise _token_bearing_401()

    with pytest.raises(ValueError, match="canopy") as refused:
        run(directory, wms_base_url="https://caltopo.example/wms",
            cache_dir=tmp_path / "cache", fetch=fake_fetch)
    assert "401" in str(refused.value)
    assert "SECRET-TOKEN" not in str(refused.value)
    # nor in the chain, which a logged traceback would print: neither the
    # explicit cause nor the implicit context holds the token-bearing error
    assert refused.value.__cause__ is None
    assert refused.value.__context__ is None


def _token_bearing_401():
    """What httpx raises for a refused GetMap: its message carries the full
    URL, and a CalTopo WMS URL carries the account token in its path."""
    request = httpx.Request("GET", "https://caltopo.example/SECRET-TOKEN/wms?LAYERS=ir")
    response = httpx.Response(401, request=request)
    return httpx.HTTPStatusError(
        f"Client error '401 Unauthorized' for url '{request.url}'",
        request=request, response=response)


def test_a_coarse_raster_is_refused_with_the_number_named(tmp_path):
    """A 4 m grid: 50 px over 200 m. The message names both the pixel and
    the limit, and the guard runs BEFORE the panel is looked for."""
    directory = stage(tmp_path, np.zeros((50, 50)), np.zeros((50, 50), dtype=bool),
                      pixel_m=4.0)
    (directory / "panels" / "vegetation.png").unlink()
    with pytest.raises(ValueError) as refused:
        run(directory, rows=50, cols=50)
    assert "4.0 m per pixel" in str(refused.value)
    assert "3.0 m" in str(refused.value)


def _line_3857(x0, y0, x1, y1):
    return {"type": "LineString", "coordinates": [[x0, y0], [x1, y1]]}


def write_osm(directory, routes):
    (directory / "osm.json").write_text(json.dumps(
        {"routes": routes, "parking": [], "query_version": OSM_QUERY_VERSION}),
        encoding="utf-8")


def _north_south_line_at_col(col, rows=200, props=None):
    """A route record as `stage_terrain` writes one, running the full
    height of the raster `col` metres east of its left edge. `power` is
    written on every record since 0.16.0; a test that wants an older
    record deletes the key."""
    ox, oy = ORIGIN
    x = ox + col * _UNITS_PER_M
    return {"id": "route_0", "kind": "route",
            "geometry": _line_3857(x, oy, x, oy - rows * _UNITS_PER_M),
            "properties": {"route_type": "BARRIER", "name": "Jefferson-Martin",
                           "osm_id": 1, "surface": None, "power": "line",
                           **(props or {})}}


def test_a_power_line_forty_metres_off_is_reported_at_forty(tmp_path):
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    write_osm(directory, [_north_south_line_at_col(140)])   # touchdown ~ col 100
    # No clearance: this is the distance report, and 60 m would close the ground it measures.
    result = run(directory, powerline_clearance_m=0.0)
    only = result["candidates"][0]
    assert only["powerline"]["distance_m"] == pytest.approx(40.0, abs=2.0)
    assert only["powerline"]["name"] == "Jefferson-Martin"
    assert MISSING_OSM_WARNING not in result["warnings"]


def test_a_power_line_past_the_notice_distance_is_null(tmp_path):
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    write_osm(directory, [_north_south_line_at_col(190)])   # ~90 m off
    only = run(directory, powerline_notice_m=50.0)["candidates"][0]
    assert only["powerline"] is None


def test_a_fence_is_not_a_power_line(tmp_path):
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    write_osm(directory, [_north_south_line_at_col(110, props={"power": None})])
    only = run(directory)["candidates"][0]
    assert only["powerline"] is None


def test_missing_osm_reports_unknown_and_warns_once(tmp_path):
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    assert not (directory / "osm.json").exists()
    result = run(directory)
    assert result["candidates"][0]["powerline"] == "unknown"
    assert result["warnings"].count(MISSING_OSM_WARNING) == 1


def test_a_barrier_staged_before_the_power_field_reports_unknown(tmp_path):
    """A record with no `power` key at all predates 0.16.0: it might be a
    wire. Unknown, and a warning naming the re-stage - never null."""
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    old = _north_south_line_at_col(110)
    del old["properties"]["power"]
    write_osm(directory, [old])
    result = run(directory)
    assert result["candidates"][0]["powerline"] == "unknown"
    assert result["warnings"].count(POWER_UNLABELLED_WARNING) == 1
    assert "force=True" in POWER_UNLABELLED_WARNING


def test_version_4_ground_checked_both_kinds_of_line(tmp_path):
    """Version 4 is the version that first asked for `power=minor_line`.
    The query has moved on (5, road tags and gates), but a version-4 file
    still checked both kinds of line: comparing against the current version
    rather than the minor-lines one would call it "distribution lines never
    asked for", which is untrue."""
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    (directory / "osm.json").write_text(json.dumps(
        {"routes": [_north_south_line_at_col(140)], "parking": [], "query_version": 4}),
        encoding="utf-8")
    result = run(directory)
    assert result["assumptions"]["powerline_kinds_checked"] == ["line", "minor_line"]
    assert MINOR_LINES_NOT_CHECKED not in result["warnings"]


def test_old_ground_still_closes_the_transmission_lines_it_knows(tmp_path):
    """A version-3 osm.json was queried for `power=line` only, never
    `power=minor_line` - but one staged by 0.16.0 or later carries every
    barrier's `power` key (the key arrived in 0.16.0, while the query was
    at version 3), so a `line` way on it is known, not unknown: it still
    closes ground and is still reported, and the warning says only that
    distribution lines were never asked for."""
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    line = _north_south_line_at_col(100)
    (directory / "osm.json").write_text(json.dumps(
        {"routes": [line], "parking": [], "query_version": 3}), encoding="utf-8")
    result = run(directory, limit=0)
    assert result["warnings"].count(MINOR_LINES_NOT_CHECKED) == 1
    assert result["assumptions"]["powerline_kinds_checked"] == ["line"]
    assert result["candidates"]
    for candidate in result["candidates"]:
        assert candidate["powerline"]["distance_m"] > 60.0


def test_the_nearest_of_two_power_lines_is_the_one_reported(tmp_path):
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    far = _north_south_line_at_col(20, props={"name": "far"})
    near = _north_south_line_at_col(130, props={"name": "near"})
    near["id"] = "route_1"
    write_osm(directory, [far, near])
    # No clearance: the two lines' 60 m margins would close the whole field.
    only = run(directory, powerline_clearance_m=0.0)["candidates"][0]
    assert only["powerline"]["name"] == "near"
    assert only["powerline"]["distance_m"] == pytest.approx(30.0, abs=2.0)


def test_a_zero_notice_distance_is_refused(tmp_path):
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    with pytest.raises(ValueError, match="powerline_notice_m"):
        run(directory, powerline_notice_m=0.0)


def test_osm_with_no_barriers_is_null_and_no_warning(tmp_path):
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    write_osm(directory, [])
    result = run(directory)
    assert result["candidates"][0]["powerline"] is None
    assert MISSING_OSM_WARNING not in result["warnings"]
    assert POWER_UNLABELLED_WARNING not in result["warnings"]


def test_ground_near_a_power_line_is_closed_before_fitting(tmp_path):
    """A 200 m open field with a line down its middle: nothing within 60 m of
    the line may be a touchdown, and what is left either side is too narrow
    for the preferred 45 by 60 m rectangle - at 45 m clearance the same
    strips ARE preferred, which is why this is run at 30 m below to confirm
    it would fail there."""
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    line = _north_south_line_at_col(100)
    write_osm(directory, [line])
    result = run(directory, limit=0)
    assert result["assumptions"]["powerline_clearance_m"] == 60.0
    assert result["candidates"]
    assert all(c["grade"] != "preferred" for c in result["candidates"])
    for candidate in result["candidates"]:
        assert candidate["powerline"]["distance_m"] > 60.0

    # The candidates' own patch outlines keep their distance too, not just
    # the reported touchdown-to-line figure: build the line in WGS84 from
    # the same endpoints the helper burned into the raster, and check every
    # candidate polygon against it.
    (x0, y0), (x1, y1) = line["geometry"]["coordinates"]
    line_wgs84 = LineString([TO4326(x0, y0), TO4326(x1, y1)])
    metres_per_degree_lon = 111_320 * math.cos(math.radians(LATITUDE))
    for candidate in result["candidates"]:
        polygon = shapely_shape(candidate["geometry"])
        distance_m = polygon.distance(line_wgs84) * metres_per_degree_lon
        assert distance_m >= 60.0 - 2.0   # a cell and a half of rounding


def test_a_distribution_line_closes_ground_and_is_named_as_one(tmp_path):
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    write_osm(directory, [_north_south_line_at_col(130, props={"power": "minor_line"})])
    candidates = run(directory, limit=0)["candidates"]
    assert candidates and all(c["powerline"]["power"] == "minor_line" for c in candidates)
    assert all(c["powerline"]["distance_m"] > 60.0 for c in candidates)


def test_a_zero_clearance_closes_nothing_and_a_negative_one_is_refused(tmp_path):
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    write_osm(directory, [_north_south_line_at_col(140)])
    assert run(directory, powerline_clearance_m=0.0)["candidates"][0]["grade"] == "preferred"
    with pytest.raises(ValueError, match="powerline_clearance_m"):
        run(directory, powerline_clearance_m=-1.0)


def test_unknown_power_lines_close_nothing(tmp_path):
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    only = run(directory)["candidates"][0]
    assert only["powerline"] == "unknown" and only["grade"] == "preferred"


#: A footprint pixel as CalTopo draws it: orange at alpha 100, inside the
#: 64-127 band measured on a real panel - never opaque.
FOOTPRINT_RGBA = (255, 171, 0, 100)


def _structures_fetch(block=None, calls=None):
    """Answers the `structures` layer with a transparent panel holding one
    orange block at `block` = (row0, row1, col0, col1) in panel pixels,
    drawn semi-transparent as CalTopo draws it, and any other layer as a
    bare-ground panel."""
    def fetch(base_url, layers, bbox, width, height, cache_dir):
        if calls is not None:
            calls.append(layers)
        if layers == ["structures"]:
            pixels = np.zeros((height, width, 4), dtype="uint8")
            if block:
                r0, r1, c0, c1 = block
                pixels[r0:r1, c0:c1] = FOOTPRINT_RGBA
            image = Image.fromarray(pixels, "RGBA")
        else:
            image = Image.fromarray(np.full((height, width, 3), 90, dtype="uint8"), "RGB")
        path = Path(cache_dir) / f"{'-'.join(layers)}.png"
        image.save(path)
        return path
    return fetch


def test_a_structure_closes_the_ground_around_it(tmp_path):
    """A house in the middle of a 200 m field: no touchdown within 30 m of it."""
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    (tmp_path / "cache").mkdir()
    result = run(directory, wms_base_url="https://caltopo.example/wms",
                 cache_dir=tmp_path / "cache", fetch=_structures_fetch((90, 110, 90, 110)),
                 limit=0)
    assert result["assumptions"]["structure_clearance_m"] == 30.0
    assert result["assumptions"]["structures"] == "checked"
    assert (directory / "panels" / "structures.png").is_file()
    assert result["candidates"]
    for candidate in result["candidates"]:
        assert _metres_from_box(candidate, 90, 110, 90, 110) > 30.0 - 1.0  # a cell of rounding


def _metres_from_box(candidate, r0, r1, c0, c1):
    """The touchdown's distance from the raster box rows r0:r1, cols c0:c1,
    in metres (the stage helper's pixel is one metre); 0 inside it."""
    ox, oy = ORIGIN
    lon, lat = candidate["touchdown"]["coordinates"]
    x, y = Transformer.from_crs(4326, 3857, always_xy=True).transform(lon, lat)
    col, row = (x - ox) / _UNITS_PER_M, (oy - y) / _UNITS_PER_M
    dx = max(c0 - col, 0, col - c1)
    dy = max(r0 - row, 0, row - r1)
    return (dx * dx + dy * dy) ** 0.5


# --- open water --------------------------------------------------------------

EVT_OPEN_WATER = 7292   # LANDFIRE "Open Water", lifeform "Water"
EVT_GRASS = 7145        # any non-water code


def _box_wgs84(r0, r1, c0, c1):
    """Raster box rows r0:r1, cols c0:c1 (metres on the stage grid) as a
    WGS84 polygon, to compare with a candidate's patch geometry."""
    from shapely.geometry import box

    ox, oy = ORIGIN
    west, north = TO4326(ox + c0 * _UNITS_PER_M, oy - r0 * _UNITS_PER_M)
    east, south = TO4326(ox + c1 * _UNITS_PER_M, oy - r1 * _UNITS_PER_M)
    return box(west, south, east, north)


def _lake_ring(r0, r1, c0, c1, closed=True):
    """A WATER_EDGE record as `stage_terrain` writes one: a lake's shoreline
    around the raster box, a closed ring unless `closed` is False."""
    ox, oy = ORIGIN
    corners = [(c0, r0), (c1, r0), (c1, r1), (c0, r1)]
    coords = [[ox + c * _UNITS_PER_M, oy - r * _UNITS_PER_M] for c, r in corners]
    if closed:
        coords.append(coords[0])
    return {"id": "route_lake", "kind": "route",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {"route_type": "WATER_EDGE", "name": "Upper Blue Lake",
                           "osm_id": 38802567, "surface": None, "power": None}}


def write_evt(directory, codes):
    """An evt.tif of 30 m pixels from the stage grid's origin."""
    import rasterio

    codes = np.asarray(codes, dtype="int16")
    pixel = 30.0 * _UNITS_PER_M
    with rasterio.open(directory / "evt.tif", "w", driver="GTiff", height=codes.shape[0],
                       width=codes.shape[1], count=1, dtype="int16", crs="EPSG:3857",
                       transform=from_origin(ORIGIN[0], ORIGIN[1], pixel, pixel)) as dst:
        dst.write(codes, 1)


def _overlap_m2(candidate, lake):
    """Square metres of a candidate's patch inside `lake` (WGS84), near
    enough at this latitude for an assertion of none."""
    patch = shapely_shape(candidate["geometry"])
    degrees = patch.intersection(lake).area
    return degrees * (111_320.0 ** 2) * math.cos(math.radians(LATITUDE))


def test_a_lake_mapped_as_a_closed_shoreline_is_never_a_landing_zone(tmp_path):
    """The bug found in a live run: a lake is flat and reads as bare on
    the canopy proxy, so it was graded preferred and offered first. An
    80 m lake in a 200 m flat field: the shore is still graded, the water
    never is."""
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    write_osm(directory, [_lake_ring(60, 140, 60, 140)])
    result = run(directory, limit=0)
    lake = _box_wgs84(61, 139, 61, 139)
    assert result["candidates"], "the dry ground around the lake is still a zone"
    for candidate in result["candidates"]:
        assert _metres_from_box(candidate, 60, 140, 60, 140) > 0.0
        assert _overlap_m2(candidate, lake) < 1.0
    assert result["assumptions"]["water_sources_checked"] == [WATER_SOURCE_OSM]
    assert result["warnings"].count(WATER_EVT_MISSING_WARNING) == 1


def test_without_the_lake_the_same_field_is_one_zone_over_the_water(tmp_path):
    """The control for the test above: the same field with no shoreline is
    one preferred zone centred where the lake would be - which is the
    grade the lake got before water was closed."""
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    write_osm(directory, [])
    only = run(directory)["candidates"][0]
    assert only["grade"] == "preferred"
    assert _metres_from_box(only, 60, 140, 60, 140) == 0.0


def test_landfire_open_water_closes_a_lake_osm_never_mapped(tmp_path):
    """No shoreline in OpenStreetMap, but LANDFIRE has the 90 m of open
    water at 60-150 m: nothing is offered on it."""
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    write_osm(directory, [])
    codes = np.full((7, 7), EVT_GRASS)
    codes[2:5, 2:5] = EVT_OPEN_WATER
    write_evt(directory, codes)
    result = run(directory, limit=0)
    lake = _box_wgs84(61, 149, 61, 149)
    assert result["candidates"]
    for candidate in result["candidates"]:
        assert _overlap_m2(candidate, lake) < 1.0
    assert result["assumptions"]["water_sources_checked"] == [WATER_SOURCE_EVT,
                                                              WATER_SOURCE_OSM]
    assert WATER_EVT_MISSING_WARNING not in result["warnings"]
    assert WATER_NOT_CHECKED_WARNING not in result["warnings"]


def test_an_open_shoreline_closes_its_line_but_fills_nothing(tmp_path):
    """A lake drawn as several ways: one way is not a ring and has no
    inside, so only the line itself is closed. LANDFIRE is the source for
    that lake's water."""
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    write_osm(directory, [_lake_ring(60, 140, 60, 140, closed=False)])
    result = run(directory, limit=0)
    inside = _box_wgs84(62, 138, 62, 138)
    assert any(_overlap_m2(candidate, inside) > 1000.0 for candidate in result["candidates"])


def test_with_no_evt_and_no_osm_water_is_not_checked_and_said(tmp_path):
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    result = run(directory)
    assert result["candidates"][0]["grade"] == "preferred"
    assert result["assumptions"]["water_sources_checked"] == []
    assert result["warnings"].count(WATER_NOT_CHECKED_WARNING) == 1
    assert WATER_EVT_MISSING_WARNING not in result["warnings"]


# --- trees from true-colour imagery ----------------------------------------

def _pine_fetch(pine_rc=None, calls=None):
    """A fake panel fetcher: the canopy (`ir`) layer bare, no structures, and
    the NAIP true-colour (`n3`) layer bare sand with, when `pine_rc` is
    given, one pine at that raster (row, col) - a 4 m crown and its shadow
    to the north-west, which the CIR proxy cannot see at all."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_trees import ground, paint_pine

    def fetch(base_url, layers, bbox, width, height, cache_dir):
        if calls is not None:
            calls.append({"layers": layers, "width": width, "height": height})
        if layers == ["structures"]:
            return _structures_fetch()(base_url, layers, bbox, width, height, cache_dir)
        if layers == ["n3"]:
            pixels = ground(height, width)
            if pine_rc is not None:
                west, _south, _east, north = bbox
                col0 = round((west - ORIGIN[0]) / _UNITS_PER_M)
                row0 = round((ORIGIN[1] - north) / _UNITS_PER_M)
                paint_pine(pixels, pine_rc[0] - row0, pine_rc[1] - col0, radius=4)
        else:
            pixels = np.zeros((height, width, 3), dtype="uint8")
            pixels[:] = TAN
        path = Path(cache_dir) / f"{'-'.join(layers)}-{width}x{height}.png"
        Image.fromarray(pixels, "RGB").save(path)
        return path
    return fetch


def test_a_lone_pine_the_canopy_proxy_misses_is_kept_out_of_every_clear_circle(tmp_path):
    """The bug found in a live run: pines inside a 50 m circle
    that the CIR proxy read as 3 % canopy and its speck rule fitted over.
    One pine in the middle of a 200 m field: the field is still graded, but
    no clear circle holds the pine."""
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    (tmp_path / "cache").mkdir()
    result = run(directory, wms_base_url="https://caltopo.example/wms",
                 cache_dir=tmp_path / "cache", fetch=_pine_fetch((100, 100)), limit=0)
    assert result["assumptions"]["trees"] == "checked"
    assert "n3" in result["assumptions"]["trees_method"]
    assert result["candidates"]
    for candidate in result["candidates"]:
        clear_radius = candidate["clear_diameter_m"] / 2.0
        # the crown's core cells, surely closed; a cell of rounding allowed
        assert _metres_from_box(candidate, 98, 102, 98, 102) >= clear_radius - 1.0
    assert not any(w.startswith("trees were not checked") for w in result["warnings"])


def test_the_same_field_without_the_pine_is_one_zone_over_where_it_stood(tmp_path):
    """The control: the imagery alone decides, so with no pine the field is
    the one preferred zone it always was, centred on the pine's ground."""
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    (tmp_path / "cache").mkdir()
    calls = []
    only = run(directory, wms_base_url="https://caltopo.example/wms",
               cache_dir=tmp_path / "cache", fetch=_pine_fetch(None, calls))["candidates"][0]
    assert only["grade"] == "preferred"
    assert _metres_from_box(only, 96, 104, 96, 104) < 10.0
    # one tree panel over the whole circle, at the grid's own metre
    trees = [c for c in calls if c["layers"] == ["n3"] and c["width"] != CHIP_PX]
    assert len(trees) == 1


def test_without_a_token_trees_are_not_checked_and_it_says_so(tmp_path):
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    result = run(directory)
    assert result["candidates"][0]["grade"] == "preferred"
    assert result["assumptions"]["trees"] == "not checked"
    assert result["warnings"].count(TREES_NOT_CHECKED.format(reason="no WMS token")) == 1


def test_a_failed_tree_fetch_is_a_warning_naming_the_type_not_the_url(tmp_path):
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    (tmp_path / "cache").mkdir()
    inner = _pine_fetch()

    def fetch(base_url, layers, bbox, width, height, cache_dir):
        if layers == ["n3"] and width != CHIP_PX:
            raise RuntimeError(f"GET {base_url}?token=secret failed")
        return inner(base_url, layers, bbox, width, height, cache_dir)

    result = run(directory, wms_base_url="https://caltopo.example/wms",
                 cache_dir=tmp_path / "cache", fetch=fetch)
    assert result["candidates"]
    assert result["assumptions"]["trees"] == "not checked"
    assert TREES_NOT_CHECKED.format(reason="RuntimeError") in result["warnings"]
    assert not any("secret" in w for w in result["warnings"])


def write_structures(directory, pixels, mode="RGBA"):
    """A structures panel staged on disk, as the debrief app would leave it."""
    Image.fromarray(pixels, mode).save(directory / "panels" / "structures.png")


def test_an_outlined_footprint_closes_its_inside(tmp_path):
    """A 120 m building drawn as a 2 m outline only: its inside would be a
    fine 52 m clearing if holes were not filled. No touchdown inside it or
    within 30 m of it, and the open ground east of it still grades."""
    directory = stage(tmp_path, np.zeros((300, 300)), np.zeros((300, 300), dtype=bool))
    pixels = np.zeros((300, 300, 4), dtype="uint8")
    pixels[40:160, 40:160] = FOOTPRINT_RGBA
    pixels[42:158, 42:158] = 0
    write_structures(directory, pixels)
    result = run(directory, rows=300, cols=300, limit=0)
    assert result["assumptions"]["structures"] == "checked"
    assert result["candidates"]
    for candidate in result["candidates"]:
        assert _metres_from_box(candidate, 40, 160, 40, 160) > 30.0 - 1.0


def test_a_thin_outline_on_a_finer_panel_survives_the_shrink(tmp_path):
    """The same outlined building on a panel four times finer than the grid,
    its outline two panel pixels (half a metre) wide: shrunk by nearest
    neighbour the outline falls between samples and the building vanishes;
    shrunk by a box filter it is a fainter cell (alpha 50) that still
    counts. Every edge sits on panel pixels 4k and 4k+1, which Pillow's
    nearest-neighbour shrink (sampling 4k+2) steps over on all four sides."""
    directory = stage(tmp_path, np.zeros((300, 300)), np.zeros((300, 300), dtype=bool))
    pixels = np.zeros((1200, 1200, 4), dtype="uint8")
    pixels[160:642, 160:642] = FOOTPRINT_RGBA
    pixels[162:640, 162:640] = 0
    write_structures(directory, pixels)
    result = run(directory, rows=300, cols=300, limit=0)
    assert result["candidates"]
    for candidate in result["candidates"]:
        assert _metres_from_box(candidate, 40, 161, 40, 161) > 30.0 - 1.0


def test_a_faint_halo_is_not_a_building(tmp_path):
    """Alpha below `STRUCTURE_ALPHA_MIN` is resampling halo, not a
    footprint: a block drawn at alpha 20 closes nothing."""
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    pixels = np.zeros((200, 200, 4), dtype="uint8")
    pixels[90:110, 90:110] = (255, 171, 0, 20)
    write_structures(directory, pixels)
    result = run(directory)
    assert result["assumptions"]["structures"] == "checked"
    only = result["candidates"][0]
    assert only["grade"] == "preferred" and _metres_from_box(only, 90, 110, 90, 110) == 0.0


def test_a_structures_panel_with_no_transparency_is_not_checked(tmp_path):
    """An RGB panel has no alpha: converted, it would read as a building in
    every cell and close every field while saying "checked"."""
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    write_structures(directory, np.full((200, 200, 3), 90, dtype="uint8"), mode="RGB")
    result = run(directory)
    assert result["assumptions"]["structures"] == "not checked"
    assert f"{STRUCTURES_NOT_CHECKED} ({STRUCTURES_NO_TRANSPARENCY})" in result["warnings"]
    assert result["candidates"][0]["grade"] == "preferred"


def test_a_structures_layer_covering_most_of_the_ground_closes_nothing(tmp_path):
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    pixels = np.zeros((200, 200, 4), dtype="uint8")
    pixels[:] = FOOTPRINT_RGBA
    write_structures(directory, pixels)
    result = run(directory)
    assert result["assumptions"]["structures"] == "not checked"
    assert f"{STRUCTURES_NOT_CHECKED} ({STRUCTURES_COVERED_MOST})" in result["warnings"]
    assert result["candidates"][0]["grade"] == "preferred"


def test_without_a_structures_layer_grading_runs_and_says_so(tmp_path):
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    result = run(directory)
    assert result["candidates"][0]["grade"] == "preferred"
    assert result["assumptions"]["structures"] == "not checked"
    assert f"{STRUCTURES_NOT_CHECKED} (no WMS token)" in result["warnings"]


def test_a_failed_structures_fetch_is_a_warning_not_a_refusal(tmp_path):
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    (tmp_path / "cache").mkdir()

    def fetch(base_url, layers, bbox, width, height, cache_dir):
        if layers == ["structures"]:
            raise RuntimeError("502")
        return _structures_fetch()(base_url, layers, bbox, width, height, cache_dir)

    result = run(directory, wms_base_url="https://caltopo.example/wms",
                 cache_dir=tmp_path / "cache", fetch=fetch)
    assert result["candidates"] and result["assumptions"]["structures"] == "not checked"
    assert f"{STRUCTURES_NOT_CHECKED} (RuntimeError)" in result["warnings"]
    assert not any("502" in w for w in result["warnings"])


def _imagery_fetch(calls):
    """A fake panel fetcher that records every call and paints a flat grey
    panel for either chip layer, `imagery` or `n3` (or a bare-ground CIR
    panel if the canopy layer is asked for), so the outline is the only
    thing on a chip that is not grey.
    The structures layer is answered fully transparent - no building - since
    an opaque RGB panel would read as a structure in every cell and close
    the whole field."""
    def fetch(base_url, layers, bbox, width, height, cache_dir):
        calls.append({"layers": layers, "bbox": bbox, "width": width, "height": height})
        if layers == ["structures"]:
            return _structures_fetch()(base_url, layers, bbox, width, height, cache_dir)
        if layers == ["ir"]:
            pixels = np.zeros((height, width, 3), dtype="uint8")
            pixels[:] = TAN
        else:
            pixels = np.full((height, width, 3), 90, dtype="uint8")
        path = Path(cache_dir) / f"{'-'.join(layers)}.png"
        Image.fromarray(pixels, "RGB").save(path)
        return path
    return fetch


def test_a_chip_is_cut_around_the_touchdown_with_the_outline_drawn(tmp_path):
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    (tmp_path / "cache").mkdir()
    calls = []
    result = run(directory, wms_base_url="https://caltopo.example/wms",
                 cache_dir=tmp_path / "cache", fetch=_imagery_fetch(calls))
    only = result["candidates"][0]
    chip = [c for c in calls if c["layers"] == ["imagery"]]
    assert len(chip) == 1 and chip[0]["width"] == CHIP_PX and chip[0]["height"] == CHIP_PX
    # the NAIP chip is the same square, asked for once as well
    # n3 is also the tree layer, fetched once over the whole circle: a chip is CHIP_PX square
    naip = [c for c in calls if c["layers"] == ["n3"] and c["width"] == CHIP_PX]
    assert len(naip) == 1 and naip[0]["bbox"] == chip[0]["bbox"]
    west, south, east, north = chip[0]["bbox"]
    assert (east - west) * math.cos(math.radians(LATITUDE)) == pytest.approx(CHIP_M, rel=0.02)
    assert (north - south) * math.cos(math.radians(LATITUDE)) == pytest.approx(CHIP_M, rel=0.02)
    # Centred on the touchdown: the raster centre, in Mercator.
    ox, oy = ORIGIN
    assert (west + east) / 2 == pytest.approx(ox + 100 * _UNITS_PER_M, abs=1.5 * _UNITS_PER_M)
    assert (south + north) / 2 == pytest.approx(oy - 100 * _UNITS_PER_M, abs=1.5 * _UNITS_PER_M)
    path = Path(only["chip_path"])
    assert path.is_file() and path.parent == directory / "lz"
    assert re.fullmatch(r"[0-9a-f]{8}-caltopo-imagery\.png", path.name)
    with Image.open(path) as image:
        assert image.size == (CHIP_PX, CHIP_PX)
        pixels = np.asarray(image.convert("RGB"))
    # The outline is a colour the imagery does not carry; somewhere on the
    # chip it has been drawn.
    assert (pixels != 90).any()
    assert result["assumptions"]["chip"] == {
        "size_m": CHIP_M, "px": CHIP_PX, "sources": ["caltopo-imagery", "caltopo-naip"]}
    assert NO_TOKEN_CHIP_WARNING not in result["warnings"]


def test_without_a_token_the_chip_is_null_and_said_once(tmp_path):
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    result = run(directory)
    assert result["candidates"][0]["chip_path"] is None
    assert result["candidates"][0]["chips"] == []
    assert result["assumptions"]["chip"]["sources"] == []
    assert result["warnings"].count(NO_TOKEN_CHIP_WARNING) == 1


def test_a_failed_chip_is_left_out_and_names_the_candidate_and_source(tmp_path):
    """Both CalTopo chip layers fail: no chips, chip_path null, and one
    warning per source naming the candidate and the source."""
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    (tmp_path / "cache").mkdir()

    def fetch(base_url, layers, bbox, width, height, cache_dir):
        if layers in (["imagery"], ["n3"]):
            raise RuntimeError("502 from the tile server")
        if layers == ["structures"]:    # no building: a transparent panel
            return _structures_fetch()(base_url, layers, bbox, width, height, cache_dir)
        raise AssertionError("only the chip and structures layers should be fetched here")

    result = run(directory, wms_base_url="https://caltopo.example/wms",
                 cache_dir=tmp_path / "cache", fetch=fetch)
    only = result["candidates"][0]
    assert only["chip_path"] is None and only["chips"] == []
    for source in ("caltopo-imagery", "caltopo-naip"):
        message = CHIP_FAILED_WARNING.format(candidate="lz_1", source=source,
                                             reason="RuntimeError")
        assert message in result["warnings"]
        assert message in only["warnings"]


def test_when_the_first_source_fails_chip_path_is_the_next_one(tmp_path):
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    (tmp_path / "cache").mkdir()
    grey = _imagery_fetch([])

    def fetch(base_url, layers, bbox, width, height, cache_dir):
        if layers == ["imagery"]:
            raise RuntimeError("502 from the tile server")
        return grey(base_url, layers, bbox, width, height, cache_dir)

    only = run(directory, wms_base_url="https://caltopo.example/wms",
               cache_dir=tmp_path / "cache", fetch=fetch)["candidates"][0]
    assert [c["source"] for c in only["chips"]] == ["caltopo-naip"]
    assert only["chip_path"] == only["chips"][0]["path"]


def test_a_refused_chip_names_the_status_and_never_the_token(tmp_path):
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    (tmp_path / "cache").mkdir()

    def fetch(base_url, layers, bbox, width, height, cache_dir):
        if layers == ["imagery"]:
            raise _token_bearing_401()
        return _structures_fetch()(base_url, layers, bbox, width, height, cache_dir)

    result = run(directory, wms_base_url="https://caltopo.example/wms",
                 cache_dir=tmp_path / "cache", fetch=fetch)
    message = CHIP_FAILED_WARNING.format(candidate="lz_1", source="caltopo-imagery",
                                         reason="HTTPStatusError 401")
    assert message in result["warnings"]
    everything = json.dumps(result)
    assert "401" in everything and "SECRET-TOKEN" not in everything


def test_chips_are_cut_only_for_the_candidates_returned(tmp_path):
    """Three openings, a limit of 2: two chips, not three. A chip is a
    network fetch, and the capped-off candidate is not shown to anyone."""
    directory = stage(tmp_path, np.zeros((300, 300)), three_openings())
    (tmp_path / "cache").mkdir()
    calls = []
    result = run(directory, rows=300, cols=300, limit=2,
                 wms_base_url="https://caltopo.example/wms",
                 cache_dir=tmp_path / "cache", fetch=_imagery_fetch(calls))
    assert len([c for c in calls if c["layers"] == ["imagery"]]) == 2
    assert len([c for c in calls if c["layers"] == ["n3"] and c["width"] == CHIP_PX]) == 2
    names = [Path(c["chip_path"]).name for c in result["candidates"]]
    assert all(re.fullmatch(r"[0-9a-f]{8}-caltopo-imagery\.png", name) for name in names)
    assert names[0] != names[1]


def test_the_outline_lands_where_the_ground_is(tmp_path):
    """An opening with a tail to the SOUTH: outline pixels appear in the
    lower half of the chip beside the tail and not in the mirrored band
    above. A flipped row axis, a swapped row/col or the wrong origin
    corner all fail here and nowhere else."""
    vegetated = np.ones((200, 200), dtype=bool)
    vegetated[70:130, 70:130] = False       # a 60 m block, centred
    vegetated[130:170, 95:105] = False      # a 10 m wide tail running 40 m south
    directory = stage(tmp_path, np.zeros((200, 200)), vegetated)
    (tmp_path / "cache").mkdir()
    calls = []
    result = run(directory, wms_base_url="https://caltopo.example/wms",
                 cache_dir=tmp_path / "cache", fetch=_imagery_fetch(calls))
    only = result["candidates"][0]
    with Image.open(only["chip_path"]) as image:
        pixels = np.asarray(image.convert("RGB"))
    drawn = (pixels != 90).any(axis=2)
    # The chip is 200 m at 0.5 m/px, centred on the touchdown (about the
    # block's centre, row 100). The tail spans rows 130-170 of the raster,
    # i.e. chip rows ~260-340, columns ~190-210. The band below starts at
    # row 270, past the block's own bottom edge at ~260, on purpose: only
    # the tail can satisfy it, not the block. Above the block, rows 30-70
    # of the raster are chip rows ~60-140 with nothing to draw except the
    # block's top edge at ~140.
    assert drawn[270:340, 180:220].any()
    assert not drawn[60:130, 180:220].any()


def test_each_source_is_its_own_outlined_chip_in_a_fixed_order(tmp_path):
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    (tmp_path / "cache").mkdir()
    calls = []
    result = run(directory, wms_base_url="https://caltopo.example/wms",
                 cache_dir=tmp_path / "cache", fetch=_imagery_fetch(calls))
    chips = result["candidates"][0]["chips"]
    assert [c["source"] for c in chips] == ["caltopo-imagery", "caltopo-naip"]
    # the labels are CalTopo's own layer titles, as its WMS names them
    assert [c["label"] for c in chips] == ["CalTopo Imagery", "CalTopo NAIP 2019 to 2021"]
    assert [c["label"] for c in chips] == [f"CalTopo {LAYERS['imagery']}",
                                           f"CalTopo {LAYERS['n3']}"]
    assert [c["attribution"] for c in chips] == ["CalTopo", "CalTopo, USDA NAIP"]
    assert all(Path(c["path"]).is_file() for c in chips)
    assert [Path(c["path"]).name.split("-", 1)[1] for c in chips] == [
        "caltopo-imagery.png", "caltopo-naip.png"]
    for chip in chips:
        with Image.open(chip["path"]) as image:
            assert (np.asarray(image.convert("RGB")) != 90).any()   # outlined
    assert result["candidates"][0]["chip_path"] == chips[0]["path"]
    assert ["n3"] in [call["layers"] for call in calls]


def _grey_png(size=CHIP_PX):
    buffer = io.BytesIO()
    Image.fromarray(np.full((size, size, 3), 80, dtype="uint8"), "RGB").save(buffer, "PNG")
    return buffer.getvalue()


def _key_bearing_403():
    """What httpx would raise for a refused static image: the key rides in
    the query string of the URL the error carries."""
    request = httpx.Request("GET", "https://maps.googleapis.com/maps/api/staticmap"
                                   "?zoom=18&key=SECRET-KEY")
    response = httpx.Response(403, request=request)
    return httpx.HTTPStatusError(
        f"Client error '403 Forbidden' for url '{request.url}'",
        request=request, response=response)


def test_a_keyed_source_joins_the_chips_and_a_failed_one_is_left_out(tmp_path):
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    (tmp_path / "cache").mkdir()
    asked = []

    def static(url):
        asked.append(url)
        if "mapbox" in url:
            return _grey_png()
        raise _key_bearing_403()

    result = run(directory, wms_base_url="https://caltopo.example/wms",
                 cache_dir=tmp_path / "cache", fetch=_imagery_fetch([]),
                 mapbox_token="tok", google_maps_key="k", get_static=static)
    only = result["candidates"][0]
    sources = [c["source"] for c in only["chips"]]
    assert sources == ["caltopo-imagery", "caltopo-naip", "mapbox-satellite"]
    mapbox = only["chips"][2]
    assert mapbox["label"] == "Mapbox Satellite"
    assert mapbox["attribution"] == "\u00a9 Mapbox \u00a9 OpenStreetMap \u00a9 Maxar"
    # held in memory, never written raw: only the outlined chip is saved
    assert not list((tmp_path / "cache").glob("mapbox-satellite*"))
    assert re.fullmatch(r"[0-9a-f]{8}-mapbox-satellite\.png", Path(mapbox["path"]).name)
    with Image.open(mapbox["path"]) as image:
        assert image.size == (CHIP_PX, CHIP_PX)
        assert (np.asarray(image.convert("RGB")) != 80).any()     # outlined
    assert any("google-satellite" in w and "403" in w for w in result["warnings"])
    assert any("google-satellite" in w for w in only["warnings"])
    assert result["assumptions"]["chip"]["sources"] == [
        "caltopo-imagery", "caltopo-naip", "mapbox-satellite", "google-satellite"]
    assert len(asked) == 2


def test_no_static_key_ever_reaches_the_result(tmp_path):
    """A 403 from Google whose URL carries the key: the warning says 403
    and the source, and neither key is anywhere in the envelope."""
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    (tmp_path / "cache").mkdir()

    def static(url):
        if "mapbox" in url:
            return _grey_png()
        raise _key_bearing_403()

    result = run(directory, wms_base_url="https://caltopo.example/wms",
                 cache_dir=tmp_path / "cache", fetch=_imagery_fetch([]),
                 mapbox_token="SECRET-MAPBOX", google_maps_key="SECRET-KEY",
                 get_static=static)
    message = SOURCE_REFUSED_WARNING.format(source="google-satellite",
                                            reason="HTTPStatusError 403")
    assert message in result["warnings"]
    everything = json.dumps(result)
    assert "SECRET-KEY" not in everything and "SECRET-MAPBOX" not in everything


def test_static_sources_alone_still_cut_chips(tmp_path):
    """No CalTopo token and no cache directory: a Mapbox key alone still
    gives each candidate a chip, and the no-source warning is not raised."""
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    result = run(directory, mapbox_token="tok", get_static=lambda url: _grey_png())
    only = result["candidates"][0]
    assert [c["source"] for c in only["chips"]] == ["mapbox-satellite"]
    assert only["chip_path"] == only["chips"][0]["path"]
    assert Path(only["chip_path"]).is_file()
    assert NO_TOKEN_CHIP_WARNING not in result["warnings"]


def test_a_refusing_static_source_is_asked_once_and_said_once(tmp_path):
    """Google answers 403 every time: asked for the first of three
    candidates and never again, one warning in the envelope, and each
    candidate saying why it has no Google chip. Mapbox goes on."""
    directory = stage(tmp_path, np.zeros((300, 300)), three_openings())
    asked = []

    def static(url):
        asked.append("google" if "google" in url else "mapbox")
        if "google" in url:
            raise _key_bearing_403()
        return _grey_png()

    result = run(directory, rows=300, cols=300, limit=0, mapbox_token="tok",
                 google_maps_key="k", get_static=static)
    candidates = result["candidates"]
    assert len(candidates) == 3
    assert asked.count("google") == 1 and asked.count("mapbox") == 3
    message = SOURCE_REFUSED_WARNING.format(source="google-satellite",
                                            reason="HTTPStatusError 403")
    assert result["warnings"].count(message) == 1
    assert not any("google-satellite chip for" in w for w in result["warnings"])
    for candidate in candidates:
        assert [c["source"] for c in candidate["chips"]] == ["mapbox-satellite"]
        assert message in candidate["warnings"]


def test_an_unreachable_static_source_is_asked_once(tmp_path):
    directory = stage(tmp_path, np.zeros((300, 300)), three_openings())
    asked = []

    def static(url):
        asked.append(url)
        raise httpx.ConnectError("the static image request failed")

    result = run(directory, rows=300, cols=300, limit=0, mapbox_token="tok",
                 get_static=static)
    assert len(result["candidates"]) == 3 and len(asked) == 1
    assert result["warnings"].count(SOURCE_REFUSED_WARNING.format(
        source="mapbox-satellite", reason="ConnectError")) == 1


def test_a_static_server_error_is_retried_for_the_next_candidate(tmp_path):
    """A 500 is not a refusal: the next candidate asks again."""
    directory = stage(tmp_path, np.zeros((300, 300)), three_openings())
    asked = []

    def static(url):
        asked.append(url)
        request = httpx.Request("GET", "https://api.mapbox.com/x")
        raise httpx.HTTPStatusError("500", request=request,
                                    response=httpx.Response(500, request=request))

    result = run(directory, rows=300, cols=300, limit=0, mapbox_token="tok",
                 get_static=static)
    assert len(asked) == 3
    assert sum("mapbox-satellite chip for" in w for w in result["warnings"]) == 3


def test_a_static_answer_not_the_size_asked_for_is_refused(tmp_path):
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    result = run(directory, mapbox_token="tok", get_static=lambda url: _grey_png(size=300))
    only = result["candidates"][0]
    assert only["chips"] == [] and only["chip_path"] is None
    assert CHIP_FAILED_WARNING.format(candidate="lz_1", source="mapbox-satellite",
                                      reason="ValueError") in result["warnings"]


def _tail_ground(tmp_path):
    """A 60 m block with a 10 m tail running 40 m south, as
    `test_the_outline_lands_where_the_ground_is` draws it."""
    vegetated = np.ones((200, 200), dtype=bool)
    vegetated[70:130, 70:130] = False
    vegetated[130:170, 95:105] = False
    return stage(tmp_path, np.zeros((200, 200)), vegetated)


@pytest.mark.parametrize("provider", ["mapbox", "google"])
def test_the_outline_lands_where_the_ground_is_on_a_static_chip(tmp_path, provider):
    """The block's north edge (raster row 70) and the tail's south end (row
    170) are drawn at the chip rows the requested zoom puts them at, worked
    out here from the URL's own centre and zoom. A box scaled as if every
    source were exactly 200 m - wrong for Google's whole zoom by about 5 %,
    some ten rows at the tail - fails, as does a flipped or shifted one."""
    directory = _tail_ground(tmp_path)
    urls = []

    def static(url):
        urls.append(url)
        return _grey_png()

    keys = {"mapbox_token": "tok"} if provider == "mapbox" else {"google_maps_key": "k"}
    only = run(directory, get_static=static, **keys)["candidates"][0]
    assert len(only["chips"]) == 1
    url = urls[0]
    if provider == "mapbox":
        lon, lat, zoom = (float(v) for v in url.split("/static/")[1].split(",")[:3])
        tile_px = 512
    else:
        lat, lon = (float(v) for v in
                    url.split("center=")[1].split("&")[0].replace("%2C", ",").split(","))
        zoom = int(url.split("zoom=")[1].split("&")[0])
        tile_px = 256
    units_per_px = 2 * math.pi * 6378137.0 / (tile_px * 2 ** zoom)
    _cx, cy = Transformer.from_crs(4326, 3857, always_xy=True).transform(lon, lat)
    ox, oy = ORIGIN
    north_row = CHIP_PX / 2 + (cy - (oy - 70 * _UNITS_PER_M)) / units_per_px
    south_row = CHIP_PX / 2 + (cy - (oy - 170 * _UNITS_PER_M)) / units_per_px

    with Image.open(only["chip_path"]) as image:
        drawn = (np.asarray(image.convert("RGB")) != 80).any(axis=2)
    rows = np.nonzero(drawn[:, 150:250].any(axis=1))[0]
    assert rows.min() == pytest.approx(north_row, abs=2.0)
    assert rows.max() == pytest.approx(south_row, abs=2.0)


def _grid(cols=10, rows=1):
    """A 1 m grid of `rows` by `cols` cells, for the margin boundary tests."""
    from types import SimpleNamespace

    return SimpleNamespace(elevation=np.zeros((rows, cols)), pixel_size_m=1.0,
                           transform=from_origin(0.0, float(rows), 1.0, 1.0))


def test_a_one_metre_structure_clearance_closes_two_cells_on(tmp_path):
    """1 m cells, a 1 m clearance, a building filling cell 0: its edge may
    be at cell 0's east side and the ground at cell 2's west side, 1 m
    apart, so cell 2 is closed. Cell 3 is at least 2 m off, and open."""
    pixels = np.zeros((1, 10, 4), dtype="uint8")
    pixels[0, 0] = FOOTPRINT_RGBA
    panel = tmp_path / "structures.png"
    Image.fromarray(pixels, "RGBA").save(panel)
    closed, why = _closed_near_structures(_grid(), panel, 1.0)
    assert why is None
    assert closed[0, :3].all() and not closed[0, 3:].any()


def test_a_one_metre_line_clearance_closes_two_cells_on():
    """The same boundary for a wire: burned into column 0 of a 1 m grid, a
    1 m clearance closes column 2 and leaves column 3."""
    line = {"geometry": {"type": "LineString",
                         "coordinates": [[0.5, 0.0], [0.5, 10.0]]}}
    closed = _closed_near_lines(_grid(rows=10), [line], 1.0)
    assert closed[:, :3].all() and not closed[:, 3:].any()



@pytest.mark.parametrize("osm", [
    {"routes": [], "parking": [], "query_version": 1},
    {"routes": [], "parking": []},                        # no version: version 1
], ids=["version-1", "no-version"])
def test_ground_staged_before_barriers_were_fetched_has_unknown_power_lines(tmp_path, osm):
    """A version-1 osm.json fetched no barrier at all, so no power line was
    ever asked for: unknown, never "transmission lines checked"."""
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    (directory / "osm.json").write_text(json.dumps(osm), encoding="utf-8")
    result = run(directory)
    assert result["candidates"][0]["powerline"] == "unknown"
    assert result["assumptions"]["powerline_kinds_checked"] == []
    assert result["warnings"].count(POWER_NOT_FETCHED_WARNING.format(version=1)) == 1
    assert MINOR_LINES_NOT_CHECKED not in result["warnings"]


def test_with_a_token_but_no_cache_directory_the_structures_reason_says_so(tmp_path):
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    result = run(directory, wms_base_url="https://caltopo.example/wms")
    assert f"{STRUCTURES_NOT_CHECKED} (no cache directory)" in result["warnings"]
    assert not any("no WMS token" in w for w in result["warnings"])


@pytest.mark.parametrize("content", [b"not a png at all",
                                     b"\x89PNG\r\n\x1a\n" + b"\x00" * 40],
                         ids=["not-an-image", "truncated-png"])
def test_an_unreadable_structures_panel_is_not_checked_and_is_removed(tmp_path, content):
    """A panel Pillow cannot read is a warning, not a failed call - and it is
    removed, so the next call fetches it afresh instead of failing on the
    same file forever."""
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    panel = directory / "panels" / "structures.png"
    panel.write_bytes(content)
    result = run(directory)
    assert result["candidates"][0]["grade"] == "preferred"
    assert result["assumptions"]["structures"] == "not checked"
    ours = [w for w in result["warnings"] if w.startswith(STRUCTURES_NOT_CHECKED)]
    assert len(ours) == 1 and "removed" in ours[0], result["warnings"]
    # One bracket, never one inside another (the app fixed nesting in 1e535e7),
    # and no promise the next call cannot keep without a token.
    assert ours[0].count("(") == 1, ours[0]
    assert "the next call fetches it again" not in ours[0]
    assert not panel.exists()


def test_a_one_pixel_outline_shrunk_onto_the_grid_still_closes_ground(tmp_path):
    """A staged panel 2.5 times finer than the grid - as a debrief-app
    panel over coarser staging can be - holding a 1 px outline at alpha 64
    at every phase against the grid. A box shrink averages each line over
    up to three panel pixels and leaves some at 21, under
    `STRUCTURE_ALPHA_MIN`; the max-pooled shrink keeps every one at 64."""
    directory = stage(tmp_path, np.zeros((300, 300)), np.zeros((300, 300), dtype=bool))
    size = 750                              # 300 * 2.5
    pixels = np.zeros((size, size, 4), dtype="uint8")
    columns = list(range(20, 730, 23))       # 23 is prime to 2.5: every phase
    for col in columns:
        pixels[:, col] = (255, 171, 0, 64)
    write_structures(directory, pixels)
    from types import SimpleNamespace
    grid = SimpleNamespace(elevation=np.zeros((300, 300)), pixel_size_m=1.0)
    closed, why = _closed_near_structures(grid, directory / "panels" / "structures.png", 0.0)
    assert why is None
    for col in columns:
        at = col * 300 / size
        assert closed[:, int(at)].all() or closed[:, min(299, int(at) + 1)].all(), col



# --- the road fact (query version 5) ----------------------------------------------

SKYLINE = {"route_type": "ROAD", "name": "Skyline Boulevard", "power": None,
           "highway": "primary", "surface": "asphalt", "access": None, "tracktype": None}


def _road_at_col(col, rows=200, **props):
    """A road record as a version-5 `stage_terrain` writes one, north-south
    at `col` metres east of the raster's left edge."""
    return _north_south_line_at_col(col, rows=rows, props={**SKYLINE, **props})


def _lot_3857(col0, row0, col1, row1):
    """A parking polygon over pixel columns col0..col1 and rows row0..row1."""
    ox, oy = ORIGIN
    xs = (ox + col0 * _UNITS_PER_M, ox + col1 * _UNITS_PER_M)
    ys = (oy - row0 * _UNITS_PER_M, oy - row1 * _UNITS_PER_M)
    return {"type": "Polygon", "coordinates": [[[xs[0], ys[0]], [xs[1], ys[0]],
                                                [xs[1], ys[1]], [xs[0], ys[1]],
                                                [xs[0], ys[0]]]]}


def _write_osm_payload(directory, routes, parking=(), query_version=OSM_QUERY_VERSION):
    (directory / "osm.json").write_text(json.dumps(
        {"routes": list(routes), "parking": list(parking), "gates": [],
         "query_version": query_version}), encoding="utf-8")


def _small_opening(tmp_path):
    """A 30 m opening centred on column and row 100 in forest: a clear
    circle of about 30 m, so `on_road` (within the circle's radius) is a
    real question. On the open 200 m field the circle is 195 m across and
    a road 40 m off lies inside it."""
    vegetated = np.ones((200, 200), dtype=bool)
    vegetated[85:115, 85:115] = False
    return stage(tmp_path, np.zeros((200, 200)), vegetated)


def test_a_road_through_the_touchdown_is_named_and_on_road(tmp_path):
    directory = _small_opening(tmp_path)
    write_osm(directory, [_road_at_col(100)])
    result = run(directory)
    only = result["candidates"][0]
    road = only["road"]
    assert 0.0 <= road.pop("distance_m") <= 1.0
    assert road == {"name": "Skyline Boulevard", "route_type": "ROAD",
                    "highway": "primary", "surface": "asphalt", "on_road": True}
    assert result["assumptions"]["road_class"] == "checked"


def test_a_road_forty_metres_off_is_beside_not_on(tmp_path):
    directory = _small_opening(tmp_path)
    write_osm(directory, [_road_at_col(140)])
    only = run(directory)["candidates"][0]
    assert only["clear_diameter_m"] < 80.0
    assert only["road"]["distance_m"] == pytest.approx(40.0, abs=1.5)
    assert only["road"]["on_road"] is False


def test_on_road_means_within_the_clear_circle(tmp_path):
    """On the open field the clear circle is 195 m across, so a road 40 m
    from the touchdown runs through it: on the landing zone, not beside it."""
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    write_osm(directory, [_road_at_col(140)])
    only = run(directory)["candidates"][0]
    assert only["clear_diameter_m"] / 2 > 40.0
    assert only["road"]["on_road"] is True


def test_a_track_and_a_road_other_are_roads_too_and_the_nearest_wins(tmp_path):
    directory = _small_opening(tmp_path)
    write_osm(directory, [
        _road_at_col(160),
        _road_at_col(130, route_type="TRACK", name="Ridge Fire Road", highway="track",
                     surface="dirt")])
    road = run(directory)["candidates"][0]["road"]
    assert road["name"] == "Ridge Fire Road" and road["route_type"] == "TRACK"
    assert road["surface"] == "dirt" and road["distance_m"] == pytest.approx(30.0, abs=1.5)
    write_osm(directory, [_road_at_col(120, route_type="ROAD_OTHER", name="Busway",
                                       highway="busway")])
    assert run(directory)["candidates"][0]["road"]["route_type"] == "ROAD_OTHER"


def test_no_road_within_the_notice_distance_is_null(tmp_path):
    directory = _small_opening(tmp_path)
    write_osm(directory, [_road_at_col(300)])                 # ~200 m off
    assert run(directory)["candidates"][0]["road"] is None
    write_osm(directory, [_road_at_col(140)])                 # ~40 m off
    assert run(directory, powerline_notice_m=30.0)["candidates"][0]["road"] is None


@pytest.mark.parametrize("props", [
    {"route_type": "TRAIL", "highway": "path"},
    {"route_type": "BARRIER", "highway": None, "power": "line"},
    {"route_type": "WATERWAY", "highway": None},
    {"highway": "proposed"},
    {"highway": "construction"},
    {"route_type": "ROAD_OTHER", "highway": "via_ferrata"},
    {"route_type": "ROAD_OTHER", "highway": "bus_guideway"},
], ids=["trail", "power-line", "creek", "proposed", "construction", "via-ferrata",
        "bus-guideway"])
def test_only_a_built_road_or_track_is_a_road(tmp_path, props):
    """Egress's rule (`egress.NOT_BUILT`, `egress.NOT_WALKED`): a road not on
    the ground yet or any more is no road, and a cabled climb or a concrete
    bus track is not one a planner means."""
    directory = _small_opening(tmp_path)
    write_osm(directory, [_road_at_col(100, **props)])
    assert run(directory, powerline_clearance_m=0.0)["candidates"][0]["road"] is None


def test_without_osm_json_the_road_is_unknown(tmp_path):
    directory = _small_opening(tmp_path)
    result = run(directory)
    only = result["candidates"][0]
    assert only["road"] == "unknown" and only["powerline"] == "unknown"
    assert only["parking_lot"] == "unknown"
    assert result["assumptions"]["road_class"] == "unknown"


def test_a_touchdown_inside_a_parking_lot_says_the_lot_and_its_road(tmp_path):
    directory = _small_opening(tmp_path)
    lot = {"id": "parking-w123", "kind": "parking", "geometry": _lot_3857(80, 80, 120, 120),
           "properties": {"name": None, "osm_id": 123, "osm_type": "way"}}
    _write_osm_payload(directory, [_road_at_col(125)], parking=[lot])
    only = run(directory)["candidates"][0]
    assert only["parking_lot"] == {"id": "parking-w123", "off": "Skyline Boulevard"}


def test_a_lot_the_touchdown_is_not_in_and_public_hardstand_are_not_lots(tmp_path):
    directory = _small_opening(tmp_path)
    away = {"id": "parking-w1", "kind": "parking", "geometry": _lot_3857(150, 150, 190, 190),
            "properties": {}}
    school = {"id": "hardstand-w2", "kind": "hardstand",
              "geometry": _lot_3857(80, 80, 120, 120), "properties": {}}
    _write_osm_payload(directory, [_road_at_col(125)], parking=[away, school])
    assert run(directory)["candidates"][0]["parking_lot"] is None


def test_a_lot_with_no_named_way_near_says_off_nothing(tmp_path):
    directory = _small_opening(tmp_path)
    lot = {"id": "parking-w123", "kind": "parking", "geometry": _lot_3857(80, 80, 120, 120),
           "properties": {}}
    _write_osm_payload(directory, [], parking=[lot])
    assert run(directory)["candidates"][0]["parking_lot"] == {"id": "parking-w123", "off": None}


def test_version_4_ground_has_roads_but_no_road_class(tmp_path):
    directory = _small_opening(tmp_path)
    old = _road_at_col(100)
    for key in ("highway", "access", "tracktype"):
        del old["properties"][key]
    _write_osm_payload(directory, [old], query_version=4)
    result = run(directory)
    road = result["candidates"][0]["road"]
    assert road["name"] == "Skyline Boulevard" and road["highway"] is None
    assert road["on_road"] is True
    assert result["assumptions"]["road_class"] == "unknown"


def test_a_road_closes_and_regrades_nothing(tmp_path):
    """Said, never closed: the same candidates, grades and fits with a road
    through the middle opening and another beside the north-east one."""
    directory = stage(tmp_path, np.zeros((300, 300)), three_openings())
    without = run(directory, rows=300, cols=300, limit=0)
    _write_osm_payload(directory, [_road_at_col(150, rows=300), _road_at_col(250, rows=300)])
    with_roads = run(directory, rows=300, cols=300, limit=0)

    def shape(result):
        return [(c["grade"], c["touchdown"], c["clear_diameter_m"], c["preferred_fit"])
                for c in result["candidates"]]

    assert shape(with_roads) == shape(without)
    assert with_roads["summary"] == without["summary"]
    assert any(c["road"] and c["road"]["on_road"] for c in with_roads["candidates"])


def test_a_road_anywhere_in_the_clearing_is_on_it_by_the_operators_ruling(tmp_path):
    """Operator's ruling, 2026-09-23, "Within the whole clearing": `on_road`
    is `distance_m <= clear_diameter_m / 2`. A road 40 m from the touchdown
    of a 195 m field runs through the clearing the helicopter would use, so
    the zone is on that road - traffic control, not just a neighbour."""
    directory = stage(tmp_path, np.zeros((200, 200)), np.zeros((200, 200), dtype=bool))
    write_osm(directory, [_road_at_col(140)])
    only = run(directory)["candidates"][0]
    assert only["clear_diameter_m"] == pytest.approx(195.0, abs=1.0)
    assert only["road"]["distance_m"] == pytest.approx(40.0, abs=1.5)
    assert only["road"]["on_road"] is True


def test_a_motorway_is_a_road_a_touchdown_can_be_on(tmp_path):
    """Egress walks nobody out along a motorway; a landing zone beside one
    is exactly the zone that needs traffic control, so it is named."""
    directory = _small_opening(tmp_path)
    write_osm(directory, [_road_at_col(100, name="Junipero Serra Freeway",
                                       highway="motorway", surface=None)])
    road = run(directory)["candidates"][0]["road"]
    assert road["name"] == "Junipero Serra Freeway" and road["highway"] == "motorway"
    assert road["on_road"] is True


@pytest.mark.parametrize("version", [1, 2, None], ids=["version-1", "version-2", "no-version"])
def test_ground_staged_before_parking_was_fetched_has_an_unknown_lot(tmp_path, version):
    """Parking arrived with query version 3: on older ground a touchdown in
    a lot cannot be told from one in no lot, so the lot is unknown, never
    null. The road is still read - roads were fetched by every version."""
    directory = _small_opening(tmp_path)
    payload = {"routes": [_road_at_col(100)]}
    if version is not None:
        payload["query_version"] = version
    (directory / "osm.json").write_text(json.dumps(payload), encoding="utf-8")
    only = run(directory, powerline_clearance_m=0.0)["candidates"][0]
    assert only["parking_lot"] == "unknown"
    assert only["road"]["name"] == "Skyline Boulevard"


def test_version_3_ground_with_no_lot_under_the_touchdown_says_none(tmp_path):
    directory = _small_opening(tmp_path)
    _write_osm_payload(directory, [_road_at_col(100)], query_version=3)
    assert run(directory)["candidates"][0]["parking_lot"] is None


def test_the_missing_osm_warning_keeps_its_words_and_adds_roads_and_lots():
    """Appended to, never reworded: a reader may match the first sentence."""
    assert MISSING_OSM_WARNING.startswith(
        "osm.json is missing from the staged ground, so power lines near these "
        "landing zones are unknown rather than absent")
    added = MISSING_OSM_WARNING.split("unknown rather than absent", 1)[1]
    assert "road" in added and "lot" in added


# --- where it looks: the planner's circle, or the whole staged ground --------


def stage_small_subject(tmp_path, elevation, vegetated, *, inset_px):
    """`stage`, but the subject is the raster less `inset_px` on every side:
    a small circle's worth in the middle, and the rest the planning buffer."""
    stage(tmp_path, elevation, vegetated)
    rows, cols = elevation.shape
    ox, oy = ORIGIN
    inset = inset_px * _UNITS_PER_M
    west, north = TO4326(ox + inset, oy - inset)
    east, south = TO4326(ox + cols * _UNITS_PER_M - inset, oy - rows * _UNITS_PER_M + inset)
    transform = from_origin(ox, oy, _UNITS_PER_M, _UNITS_PER_M)
    write_fetch_dir(tmp_path, elevation, transform,
                    make_wgs84_square(west, south, east, north), latitude=LATITUDE)
    return tmp_path


def _field_in_the_buffer():
    """Vegetation everywhere but an 80 m field in the north-west corner,
    wholly outside a subject inset 90 m from every side of 200 m."""
    vegetated = np.ones((200, 200), dtype=bool)
    vegetated[5:85, 5:85] = False
    return vegetated


def test_a_zone_past_the_circle_is_found_only_when_the_whole_ground_is_asked(tmp_path):
    # Found in a live run: a marginal zone on the exit trail 1,290 m out
    # was staged (the buffer reached 2,000 m) but never graded, the circle
    # being 1,000 m.
    directory = stage_small_subject(tmp_path, np.zeros((200, 200)), _field_in_the_buffer(),
                                    inset_px=90)
    inside = run(directory, limit=0)
    assert inside["candidates"] == []
    assert inside["assumptions"]["searched"].startswith("inside the planner's circle")
    whole = run(directory, limit=0, whole_ground=True)
    assert [c["grade"] for c in whole["candidates"]] == ["preferred"]
    assert _metres_from_box(whole["candidates"][0], 5, 85, 5, 85) == 0
    assert whole["assumptions"]["searched"].startswith("the whole staged ground")


def test_trees_in_the_buffer_are_checked_when_the_whole_ground_is_graded(tmp_path):
    # Without the tree check reaching the buffer, a pine in the middle of the
    # buffer's field would be fitted over: the CIR proxy cannot see it.
    directory = stage_small_subject(tmp_path, np.zeros((200, 200)), _field_in_the_buffer(),
                                    inset_px=90)
    (tmp_path / "cache").mkdir()
    calls = []
    result = run(directory, limit=0, whole_ground=True,
                 wms_base_url="https://caltopo.example/wms", cache_dir=tmp_path / "cache",
                 fetch=_pine_fetch((45, 45), calls))
    assert result["assumptions"]["trees"] == "checked"
    for candidate in result["candidates"]:
        assert _metres_from_box(candidate, 43, 47, 43, 47) >= candidate["clear_diameter_m"] / 2 - 1
    # The tree panel was fetched over the whole raster, not the subject's box.
    trees = [c for c in calls if c["layers"] == ["n3"] and c["width"] != CHIP_PX]
    assert trees and trees[0]["width"] >= 200 and trees[0]["height"] >= 200
