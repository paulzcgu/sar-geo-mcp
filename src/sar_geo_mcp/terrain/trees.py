"""Trees from true-colour NAIP imagery: the shared building block.

Any tool that needs to know where trees stand reads them here: a landing zone
keeps clear of them, and a clearing, a staging site or a search segment's
canopy can be measured the same way. Two layers of API:

- `detect_trees(rgb, m_per_px)`: pure, on any true-colour image, returning
  crown, shadow and their union pixel by pixel. No network, no grid.
- `trees_on_grid(grid, ...)`: fetches CalTopo's NAIP true colour over a
  staged grid (or part of it) and returns, per grid cell, the share of its
  ground that is tree or tree shadow. A caller that needs "any tree here"
  tests `> 0`; one that needs canopy cover averages it over its cells.

WHY NOT THE CIR CANOPY PROXY
----------------------------
`terrain.vegetation` asks how red a pixel is on CalTopo's NAIP false-colour
render. Sierra conifers render dark maroon there and their shadows grey, so a
field of scattered pines near Upper Blue Lake read as 3 % canopy where the
imagery shows about 18 % crown. The proxy is right about green, leafy ground;
it is wrong about dark conifers, which are most of the West's trees.

WHAT A TREE LOOKS LIKE FROM ABOVE
---------------------------------
Colour alone cannot tell a pine from a wet meadow: on NAIP both are dark green
(a meadow measured L 60, ExG 0.22, much like a crown). What a meadow never has
is a cast shadow. So:

- a SHADOW pixel is well under the local ground brightness and not green,
  or so dark that its colour does not matter (a shadow on grass stays a
  little green) - what a tall thing throws on the ground beside it;
- a CROWN pixel is green and within `CROWN_SHADOW_REACH_M` of a shadow.

Local ground brightness is the median over `REF_WINDOW_M`, so a slope in
terrain shade is compared with itself, not with the sunlit ridge. A dark
neutral surface - a burn scar, dark rock, a small asphalt lot in bright
ground - can read as shadow. A caller using trees as obstacles is wrong in
the safe direction there; one measuring canopy should prefer `crown`.

The thresholds were set on one live area (Sierra, NAIP 2019-2021) and are
named constants for the next one to move.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np

#: CalTopo's NAIP true-colour layer (`caltopo.wms.LAYERS["n3"]`).
TREE_LAYER = "n3"

#: Ground metres per imagery pixel to aim for. NAIP is 0.6 m; at 1 m a 4 m
#: crown is four pixels and a 3 km circle is a 3,000 px panel.
TREE_PANEL_M = 1.0

#: The window local ground brightness is the median over.
REF_WINDOW_M = 60.0

#: A shadow is darker than this share of local ground brightness...
SHADOW_RATIO = 0.55
#: ...and not green: excess green (2G-R-B)/(R+G+B) under this...
SHADOW_MAX_EXG = 0.08
#: ...unless it is darker than this share, whatever its colour: a pine's
#: shadow on a meadow keeps the grass's green tint, but no meadow is this
#: dark against its own surroundings.
DEEP_SHADOW_RATIO = 0.35

#: A crown is at least this green...
CROWN_MIN_EXG = 0.04
#: ...and within this many metres of a shadow. A crown's far edge is a
#: crown's width from the shadow it throws, and a big pine is 5-6 m across.
CROWN_SHADOW_REACH_M = 6.0

TREES_METHOD = ("NAIP true colour (CalTopo n3): shadow = under "
                f"{SHADOW_RATIO:g} of the {REF_WINDOW_M:g} m local median brightness "
                f"and excess green under {SHADOW_MAX_EXG:g}, or under "
                f"{DEEP_SHADOW_RATIO:g} of it at any colour; crown = excess green over "
                f"{CROWN_MIN_EXG:g} within {CROWN_SHADOW_REACH_M:g} m of a shadow")

__all__ = ["CROWN_MIN_EXG", "CROWN_SHADOW_REACH_M", "DEEP_SHADOW_RATIO", "REF_WINDOW_M", "SHADOW_MAX_EXG",
           "SHADOW_RATIO", "TREES_METHOD", "TREE_LAYER", "TREE_PANEL_M", "TreePixels",
           "detect_trees", "trees_on_grid"]


class TreePixels(NamedTuple):
    """Per-pixel masks over one image: `trees` is `crown | shadow` with
    one-pixel noise opened away; `crown` and `shadow` are before that."""
    crown: np.ndarray
    shadow: np.ndarray
    trees: np.ndarray


def _local_reference(luminance: np.ndarray, m_per_px: float) -> np.ndarray:
    """Median brightness over `REF_WINDOW_M`, cheaply: block means over
    about 8 m, a median over those, and back up to full size."""
    from scipy import ndimage

    block = max(1, int(round(8.0 / m_per_px)))
    rows, cols = luminance.shape
    pad_r, pad_c = (-rows) % block, (-cols) % block
    padded = np.pad(luminance, ((0, pad_r), (0, pad_c)), mode="edge")
    coarse = padded.reshape(padded.shape[0] // block, block,
                            padded.shape[1] // block, block).mean(axis=(1, 3))
    window = max(1, int(round(REF_WINDOW_M / (block * m_per_px))))
    coarse = ndimage.median_filter(coarse, size=window, mode="nearest")
    return np.repeat(np.repeat(coarse, block, axis=0), block, axis=1)[:rows, :cols]


def detect_trees(rgb: np.ndarray, m_per_px: float) -> TreePixels:
    """Crowns, shadows and trees in a true-colour image of `m_per_px`
    ground metres per pixel (see the module docstring)."""
    from scipy import ndimage

    pixels = np.asarray(rgb, dtype="float64")[..., :3]
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    total = red + green + blue
    luminance = total / 3.0
    exg = (2.0 * green - red - blue) / (total + 1e-9)
    reference = _local_reference(luminance, m_per_px)

    shadow = (((luminance < SHADOW_RATIO * reference) & (exg < SHADOW_MAX_EXG))
              | (luminance < DEEP_SHADOW_RATIO * reference))
    if shadow.any():
        near_shadow = (ndimage.distance_transform_edt(~shadow) * m_per_px
                       <= CROWN_SHADOW_REACH_M)
    else:
        near_shadow = np.zeros(shadow.shape, dtype=bool)
    crown = (exg > CROWN_MIN_EXG) & near_shadow
    # One pixel is noise, not a tree: a 2 x 2 opening at about a metre.
    trees = ndimage.binary_opening(crown | shadow, structure=np.ones((2, 2), dtype=bool))
    return TreePixels(crown=crown, shadow=shadow, trees=trees)


def trees_on_grid(grid, *, wms_base_url: str | None, cache_dir: Path | None,
                  fetch=None, within: np.ndarray | None = None) -> np.ndarray:
    """Per grid cell, the share of its ground (0 to 1) that is tree or tree
    shadow, over the cells of `within` (default: the grid's own mask - the
    planner's circle or the assignment); 0 elsewhere.

    The imagery is fetched over the bounding box of `within` at an exact
    multiple of the grid, about `TREE_PANEL_M` per pixel, so every grid
    cell is a whole block of imagery pixels and its share is that block's
    mean. `fetch` is `caltopo.wms.fetch_panel` unless injected; the panel
    and its tiles are cached there, so a second caller over the same
    ground pays only the classification.

    Raises when the imagery cannot be had (no token, no cache directory, a
    failed fetch, a panel of the wrong size). Whether that refuses the
    caller or only warns is the caller's call."""
    from PIL import Image

    if fetch is None:
        from ..caltopo.wms import fetch_panel as fetch
    pixel_m = float(grid.pixel_size_m)
    shape = grid.elevation.shape
    share = np.zeros(shape, dtype="float32")
    region = grid.mask if within is None else within
    rows = np.flatnonzero(region.any(axis=1))
    cols = np.flatnonzero(region.any(axis=0))
    if rows.size == 0:
        return share
    if not wms_base_url:
        raise ValueError("no WMS token")
    if cache_dir is None:
        raise ValueError("no cache directory")
    r0, r1, c0, c1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1

    factor = max(1, int(round(pixel_m / TREE_PANEL_M)))
    transform = grid.transform
    west = transform.c + c0 * transform.a
    east = transform.c + c1 * transform.a
    north = transform.f + r0 * transform.e
    south = transform.f + r1 * transform.e
    width, height = (c1 - c0) * factor, (r1 - r0) * factor
    source = fetch(wms_base_url, [TREE_LAYER], (west, south, east, north),
                   width, height, Path(cache_dir))
    with Image.open(source) as image:
        rgb = np.asarray(image.convert("RGB"))
    if rgb.shape[:2] != (height, width):
        raise ValueError(f"the {TREE_LAYER} panel came back {rgb.shape[1]} x "
                         f"{rgb.shape[0]}, not {width} x {height}")
    trees = detect_trees(rgb, pixel_m / factor).trees
    blocks = trees.reshape(r1 - r0, factor, c1 - c0, factor).mean(axis=(1, 3))
    share[r0:r1, c0:c1] = np.where(region[r0:r1, c0:c1], blocks, 0.0)
    return share
