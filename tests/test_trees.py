"""`terrain.trees`: trees from true-colour NAIP - shadows, and the green
crowns beside them - as a building block any tool can read.

Synthetic images at one metre per pixel: bare tan ground, a pine painted as
a dark green crown with a grey shadow thrown to the north-west, and a wet
meadow painted as the same dark green with no shadow."""

from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image
from rasterio.transform import from_origin

from sar_geo_mcp.terrain.trees import (
    TREE_LAYER,
    TREES_METHOD,
    detect_trees,
    trees_on_grid,
)

GROUND = (180, 170, 150)   # sunlit sand and granite
CROWN = (50, 80, 45)       # a conifer crown on NAIP
MEADOW = (60, 80, 55)      # a wet meadow: as dark and green as a crown
SHADOW = (42, 44, 43)      # a cast shadow on sand: dark and neutral


def ground(rows=200, cols=200):
    pixels = np.zeros((rows, cols, 3), dtype="uint8")
    pixels[:] = GROUND
    return pixels


def paint_pine(pixels, row, col, radius=3, shadow_len=8):
    """A crown of `radius` m at (row, col) and its shadow running north-west,
    painted as the ground it falls on at 30 % brightness."""
    rr, cc = np.mgrid[:pixels.shape[0], :pixels.shape[1]]
    for step in range(1, shadow_len + 1):
        sr, sc = row - step, col - step
        disc = (rr - sr) ** 2 + (cc - sc) ** 2 <= (radius - 1) ** 2
        pixels[disc] = (pixels[disc] * 0.3).astype("uint8")
    pixels[(rr - row) ** 2 + (cc - col) ** 2 <= radius ** 2] = CROWN
    return pixels


def test_bare_ground_has_no_trees():
    found = detect_trees(ground(), 1.0)
    assert not found.trees.any()


def test_a_pine_is_its_crown_and_its_shadow():
    pixels = paint_pine(ground(), 100, 100)
    found = detect_trees(pixels, 1.0)
    assert found.crown[100, 100] and found.trees[100, 100]
    assert found.shadow[95, 95] and found.trees[95, 95]
    assert not found.trees[100, 140]     # open ground 40 m east
    assert not found.trees[160, 100]


def test_a_wet_meadow_is_not_a_tree_though_it_is_as_green_as_one():
    pixels = ground()
    pixels[40:160, 40:160] = MEADOW
    found = detect_trees(pixels, 1.0)
    assert not found.trees.any()


def test_a_pine_in_a_meadow_is_found_and_the_meadow_is_not():
    pixels = ground()
    pixels[40:160, 40:160] = MEADOW
    paint_pine(pixels, 100, 100)
    found = detect_trees(pixels, 1.0)
    assert found.trees[100, 100]
    # green meadow beyond the shadow's reach of the pine stays open
    assert not found.trees[150, 150] and not found.trees[50, 150]


def test_a_single_pixel_is_noise_not_a_tree():
    pixels = ground()
    pixels[100, 100] = SHADOW
    assert not detect_trees(pixels, 1.0).trees.any()


def test_terrain_shade_is_compared_with_itself():
    """The north half of the image is a slope in shade, uniformly darker: it
    is ground, not one long shadow."""
    pixels = ground()
    pixels[:100] = (np.asarray(GROUND) * 0.45).astype("uint8")
    assert detect_trees(pixels, 1.0).trees[20:80, 20:180].mean() < 0.01


class _Grid:
    """The slice of `AssignmentGrid` that `trees_on_grid` reads."""
    def __init__(self, shape, pixel_m=2.0, mask=None):
        self.elevation = np.zeros(shape)
        self.pixel_size_m = pixel_m
        self.transform = from_origin(1000.0, 5000.0, pixel_m, pixel_m)
        self.mask = np.ones(shape, dtype=bool) if mask is None else mask


def _fetch_of(image, calls):
    def fetch(base_url, layers, bbox, width, height, cache_dir):
        calls.append({"layers": layers, "bbox": bbox, "width": width, "height": height})
        path = cache_dir / "n3.png"
        Image.fromarray(image[:height, :width]).save(path)
        return path
    return fetch


def test_on_a_two_metre_grid_imagery_is_asked_at_a_metre_and_pooled_per_cell(tmp_path):
    pixels = paint_pine(ground(), 100, 100)
    calls = []
    grid = _Grid((100, 100))
    share = trees_on_grid(grid, wms_base_url="https://caltopo.example/wms",
                          cache_dir=tmp_path, fetch=_fetch_of(pixels, calls))
    assert calls[0]["layers"] == [TREE_LAYER]
    assert (calls[0]["width"], calls[0]["height"]) == (200, 200)
    assert share.shape == (100, 100)
    assert share[50, 50] == pytest.approx(1.0)       # the crown's centre cell
    assert share[50, 80] == 0.0
    assert 0.0 <= share.min() and share.max() <= 1.0


def test_only_the_bounding_box_of_within_is_fetched_and_the_rest_is_zero(tmp_path):
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:60, 30:80] = True
    calls = []
    grid = _Grid((100, 100), mask=mask)
    pixels = np.zeros((80, 100, 3), dtype="uint8")
    pixels[:] = SHADOW                                  # everything reads as tree
    share = trees_on_grid(grid, wms_base_url="https://caltopo.example/wms",
                          cache_dir=tmp_path, fetch=_fetch_of(pixels, calls))
    assert (calls[0]["width"], calls[0]["height"]) == (100, 80)
    west, south, east, north = calls[0]["bbox"]
    assert (west, north) == (1000.0 + 30 * 2.0, 5000.0 - 20 * 2.0)
    assert share[~mask].max() == 0.0


def test_without_a_token_or_a_cache_it_raises_with_the_reason(tmp_path):
    grid = _Grid((10, 10))
    with pytest.raises(ValueError, match="no WMS token"):
        trees_on_grid(grid, wms_base_url=None, cache_dir=tmp_path)
    with pytest.raises(ValueError, match="no cache directory"):
        trees_on_grid(grid, wms_base_url="https://caltopo.example/wms", cache_dir=None)


def test_a_panel_of_the_wrong_size_is_refused(tmp_path):
    grid = _Grid((50, 50))
    with pytest.raises(ValueError, match="not 100 x 100"):
        trees_on_grid(grid, wms_base_url="https://caltopo.example/wms", cache_dir=tmp_path,
                      fetch=_fetch_of(ground(60, 60), []))


def test_the_method_names_its_source_and_thresholds():
    assert "n3" in TREES_METHOD and "shadow" in TREES_METHOD and "crown" in TREES_METHOD
