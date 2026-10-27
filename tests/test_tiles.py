import pytest

from sar_geo_mcp.geo import Tile, plan_tiles

BBOX = (0.0, 0.0, 2000.0, 1000.0)


def test_single_tile_when_within_limit():
    tiles = plan_tiles(BBOX, width_px=1024, height_px=512, max_px=1024)

    assert len(tiles) == 1
    assert tiles[0].bbox == pytest.approx(BBOX)
    assert (tiles[0].x_off, tiles[0].y_off) == (0, 0)


def test_splits_width_beyond_limit():
    tiles = plan_tiles(BBOX, width_px=2048, height_px=512, max_px=1024)

    assert len(tiles) == 2
    assert [t.width_px for t in tiles] == [1024, 1024]
    assert [t.x_off for t in tiles] == [0, 1024]


def test_remainder_tile_is_narrower():
    tiles = plan_tiles(BBOX, width_px=1500, height_px=600, max_px=1024)

    assert [t.width_px for t in tiles] == [1024, 476]
    assert [t.x_off for t in tiles] == [0, 1024]


def test_tile_bboxes_tile_the_input_exactly():
    tiles = plan_tiles(BBOX, width_px=2048, height_px=512, max_px=1024)

    assert tiles[0].bbox[0] == pytest.approx(0.0)
    assert tiles[0].bbox[2] == pytest.approx(1000.0)
    assert tiles[1].bbox[0] == pytest.approx(1000.0)
    assert tiles[1].bbox[2] == pytest.approx(2000.0)


def test_row_zero_is_the_top_of_the_image():
    tiles = plan_tiles(BBOX, width_px=512, height_px=2048, max_px=1024)

    top_row = [t for t in tiles if t.row == 0][0]
    bottom_row = [t for t in tiles if t.row == 1][0]

    # Row 0 is drawn at the top of the image, which is maximum northing.
    assert top_row.y_off == 0
    assert top_row.bbox[3] == pytest.approx(1000.0)
    assert bottom_row.bbox[3] == pytest.approx(500.0)
    assert bottom_row.bbox[1] == pytest.approx(0.0)


def test_grid_covers_both_axes():
    tiles = plan_tiles(BBOX, width_px=2048, height_px=2048, max_px=1024)

    assert len(tiles) == 4
    assert {(t.col, t.row) for t in tiles} == {(0, 0), (1, 0), (0, 1), (1, 1)}


def test_rejects_nonpositive_size():
    with pytest.raises(ValueError, match="positive"):
        plan_tiles(BBOX, width_px=0, height_px=100)


def test_tile_is_hashable_for_use_as_a_key():
    tile = Tile(col=0, row=0, bbox=BBOX, width_px=10, height_px=10, x_off=0, y_off=0)

    assert hash(tile) is not None
