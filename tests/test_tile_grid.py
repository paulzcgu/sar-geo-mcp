"""The shared tile grid (F3).

Named `test_tile_grid` to keep it clear of `test_tiles.py`, which covers
`geo.plan_tiles` - the request-splitter that keeps a single WMS call under
the server's pixel ceiling. Different concern: that one subdivides one
request, this one is the fixed grid that lets separate requests share.
"""

from __future__ import annotations

import math

import pytest

from sar_geo_mcp.tiles import (
    COARSER_TOLERANCE,
    MAX_LEVEL,
    TILE_PX,
    Tile,
    WORLD_M,
    level_for,
    mosaic_bbox,
    pixel_size_m,
    pixel_window,
    tile_size_m,
    tiles_covering,
)

# El Corte, the ground this project was built against, in EPSG:3857.
BBOX = (-13617151.3, 4494360.9, -13615236.6, 4496264.4)


# ---------------------------------------------------------------------------
# Levels.
# ---------------------------------------------------------------------------

def test_levels_follow_the_standard_slippy_map_pyramid():
    """Familiar numbers beat clever ones: a level here means what it means
    in every other mapping tool, so a cached tile is debuggable."""
    assert pixel_size_m(0) == pytest.approx(156543.03, abs=0.01)
    assert pixel_size_m(16) == pytest.approx(2.3886, abs=1e-4)
    for level in range(1, 20):
        assert pixel_size_m(level) == pytest.approx(pixel_size_m(level - 1) / 2)


def test_a_tile_is_the_wms_ceiling_wide():
    """1024 is exactly CalTopo's GetMap cap, so one tile is one request and
    no tile is ever stitched from sub-requests."""
    assert TILE_PX == 1024
    assert tile_size_m(16) == pytest.approx(pixel_size_m(16) * 1024)


def test_the_chosen_level_is_never_meaningfully_coarser_than_asked():
    """Handing back a softer grid than requested would quietly reduce the
    resolution of a measurement."""
    for width in (256, 512, 1024, 1289, 2048):
        level = level_for(BBOX, width, width)
        requested = (BBOX[2] - BBOX[0]) / width
        assert pixel_size_m(level) <= requested * COARSER_TOLERANCE


def test_a_finer_request_selects_a_finer_level():
    coarse = level_for(BBOX, 256, 256)
    fine = level_for(BBOX, 2048, 2048)
    assert fine > coarse


def test_a_request_sitting_on_a_level_does_not_jump_to_the_finer_one():
    """The tolerance exists for exactly this case - an exact-fit request
    should not fetch at double resolution and halve it."""
    level = 16
    size = pixel_size_m(level)
    aligned = (0.0, 0.0, size * 512, size * 512)
    assert level_for(aligned, 512, 512) == level


def test_levels_are_clamped_rather_than_raising():
    """A caller asked for a picture of some ground; refusing over a
    resolution technicality helps nobody."""
    absurdly_fine = level_for((0.0, 0.0, 1.0, 1.0), 100_000, 100_000)
    assert absurdly_fine == MAX_LEVEL

    whole_world = level_for((-2e7, -2e7, 2e7, 2e7), 1, 1)
    assert whole_world >= 0


@pytest.mark.parametrize("width,height", [(0, 10), (10, 0), (-5, 10)])
def test_a_degenerate_request_is_refused(width, height):
    with pytest.raises(ValueError):
        level_for(BBOX, width, height)


# ---------------------------------------------------------------------------
# Coverage and geometry.
# ---------------------------------------------------------------------------

def test_tiles_cover_the_whole_bbox():
    level = level_for(BBOX, 1024, 1024)
    covered = mosaic_bbox(tiles_covering(BBOX, level))

    assert covered[0] <= BBOX[0] and covered[1] <= BBOX[1]
    assert covered[2] >= BBOX[2] and covered[3] >= BBOX[3]


def test_tiles_are_ordered_north_to_south_so_a_canvas_can_be_filled_in_order():
    """Image row 0 is maximum northing. Getting this backwards would flip
    every mosaic vertically - and a vertically flipped DEM still looks like
    terrain, which is what makes it worth asserting."""
    tiles = tiles_covering(BBOX, level_for(BBOX, 1024, 1024))
    rows = [tile.row for tile in tiles]
    assert rows == sorted(rows, reverse=True)


def test_adjacent_tiles_abut_exactly():
    """A gap or an overlap of even one unit would show as a seam."""
    left = Tile(16, 5, 3)
    right = Tile(16, 6, 3)
    above = Tile(16, 5, 4)

    assert left.bbox[2] == pytest.approx(right.bbox[0])
    assert left.bbox[3] == pytest.approx(above.bbox[1])


def test_the_grid_is_fixed_so_overlapping_requests_share_tiles():
    """The entire point of F3. Two assignments over the same ground must
    resolve to overlapping tile sets, not to two private downloads."""
    level = level_for(BBOX, 1024, 1024)
    span = BBOX[2] - BBOX[0]
    nudged = (BBOX[0] + span / 10, BBOX[1] + span / 10,
              BBOX[2] + span / 10, BBOX[3] + span / 10)

    first = set(tiles_covering(BBOX, level))
    second = set(tiles_covering(nudged, level))
    assert first & second, "overlapping ground produced disjoint tile sets"


def test_western_and_southern_hemispheres_use_negative_indices():
    """The grid origin is the Mercator origin, not the world corner. Every
    coordinate in this project is in the western hemisphere, so negative
    columns are the normal case, not an edge case."""
    tiles = tiles_covering(BBOX, 16)
    assert all(tile.col < 0 for tile in tiles)
    assert all(tile.row > 0 for tile in tiles), "northern hemisphere"


def test_a_tile_key_is_layer_first_so_one_layer_can_be_dropped():
    assert Tile(16, -5, 3).key("dem", "tif") == "tiles/dem/16/-5/3.tif"


# ---------------------------------------------------------------------------
# Windowing.
# ---------------------------------------------------------------------------

def test_the_window_of_a_bbox_inside_its_own_mosaic_is_the_right_size():
    level = level_for(BBOX, 1024, 1024)
    tiles = tiles_covering(BBOX, level)
    left, top, right, bottom = pixel_window(mosaic_bbox(tiles), BBOX, level)

    expected_width = (BBOX[2] - BBOX[0]) / pixel_size_m(level)
    assert right - left == pytest.approx(expected_width, abs=1)
    assert 0 <= left and 0 <= top
    assert right <= len({t.col for t in tiles}) * TILE_PX
    assert bottom <= len({t.row for t in tiles}) * TILE_PX


def test_a_window_is_never_empty():
    """A sliver bbox rounding to zero pixels would produce an unreadable
    raster rather than a small one."""
    level = 16
    size = pixel_size_m(level)
    outer = (0.0, 0.0, size * 100, size * 100)
    sliver = (0.0, 0.0, size / 1000, size / 1000)
    left, top, right, bottom = pixel_window(outer, sliver, level)
    assert right > left and bottom > top


def test_a_full_tile_window_starts_at_the_origin():
    tile = Tile(16, -5, 3)
    assert pixel_window(tile.bbox, tile.bbox, 16) == (0, 0, TILE_PX, TILE_PX)


# ---------------------------------------------------------------------------
# Slippy addressing.
# ---------------------------------------------------------------------------

def _slippy_bbox(level: int, x: int, y: int):
    """What a slippy-map client with 1024 px tiles means by z/x/y: the same
    pixel size as our level, counted from the world's north-west corner."""
    size = tile_size_m(level)
    min_x = -WORLD_M / 2 + x * size
    max_y = WORLD_M / 2 - y * size
    return (min_x, max_y - size, min_x + size, max_y)


@pytest.mark.parametrize("level,x,y", [
    (3, 0, 0), (3, 1, 0), (12, 655, 400), (17, 21000, 30000),
    (20, 0, 262143), (20, 262143, 262143), (20, 262143, 0),
])
def test_from_slippy_addresses_the_same_ground_as_the_client(level, x, y):
    tile = Tile.from_slippy(level, x, y)
    assert tile.level == level
    assert tile.bbox == pytest.approx(_slippy_bbox(level, x, y), rel=1e-9)


def test_the_slippy_tile_south_east_of_the_origin_is_hub_0_minus_1():
    # The slippy tile just south-east of the Mercator origin is hub (0, -1).
    half = 2 ** (12 - 3)
    assert Tile.from_slippy(12, half, half) == Tile(12, 0, -1)


@pytest.mark.parametrize("level,x,y", [
    (2, 0, 0), (12, -1, 0), (12, 0, 1024), (12, 1024, 0), (21, 0, 0),
])
def test_from_slippy_refuses_addresses_off_the_grid(level, x, y):
    with pytest.raises(ValueError):
        Tile.from_slippy(level, x, y)
