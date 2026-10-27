"""Re-derive the `sf` legend from a tile CalTopo actually rendered.

The legend in `caltopo/shading.py` ships UNVERIFIED. This replaces guesswork
with measurement: fetch one panel over ground whose slope the staged DEM
already knows, then report which colours appear and what slope each covers.

    ZA_WMS_TOKEN=... ./.venv/Scripts/python.exe scripts/derive_sf_palette.py \
        --directory cache/aoi-<handle>

Ground with real relief matters: a flat area renders one colour and teaches
nothing. Pick somewhere with everything from bench to cliff in frame.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from sar_geo_mcp.caltopo.wms import fetch_panel
from sar_geo_mcp.config import CaltopoConfig
from sar_geo_mcp.planning.rasters import georeference
from sar_geo_mcp.terrain.grid import load_assignment_grid
from sar_geo_mcp.terrain.slope import compute_slope_aspect

#: Colours covering less of the panel than this are antialiasing, not bands.
MIN_SHARE = 0.001


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path,
                        help="a staged directory holding dem.tif and manifest.json")
    parser.add_argument("--layer", default="sf")
    parser.add_argument("--cache", default=Path("cache"), type=Path)
    arguments = parser.parse_args()

    config = CaltopoConfig.from_env()
    base_url = config.wms_base_url()
    if not base_url:
        print("ZA_WMS_TOKEN is not set; nothing to fetch.")
        return 1

    grid = load_assignment_grid(arguments.directory)
    block = georeference(grid.transform, grid.elevation.shape, grid.latitude_deg)
    # No `resample`: `fetch_panel` takes the filter from the layer, and `sf`
    # is declared categorical (`wms.CATEGORICAL_LAYERS`), so the panel arrives
    # nearest-neighbour. That matters here more than anywhere - this script
    # derives the legend FROM these pixels, and an interpolating resample
    # would feed it blended colours that are not bands at all. Measured, at
    # the usual 1.675x downsample: LANCZOS left 3.3% to 19.2% of pixels
    # matching no legend colour, against 0.0% at nearest-neighbour.
    panel = fetch_panel(base_url, [arguments.layer], tuple(block["bbox_3857"]),
                        block["width_px"], block["height_px"], arguments.cache)

    with Image.open(panel) as image:
        rgba = np.asarray(image.convert("RGBA"))

    slope = compute_slope_aspect(grid.elevation, grid.pixel_size_m).slope_deg
    opaque = rgba[:, :, 3] > 0
    flat = [tuple(int(channel) for channel in pixel)
            for pixel in rgba[:, :, :3][opaque]]
    counts = Counter(flat)
    total = max(1, len(flat))

    transparent = slope[~opaque]
    transparent = transparent[np.isfinite(transparent)]
    print(f"panel: {panel}")
    print(f"opaque {opaque.mean():.1%} of {rgba.shape[0]}x{rgba.shape[1]} px\n")
    if transparent.size:
        print(f"TRANSPARENT covers slope {np.nanmin(transparent):.1f} to "
              f"{np.nanmax(transparent):.1f} deg "
              f"(p99 {np.nanpercentile(transparent, 99):.1f}) "
              "- the legend's lowest threshold is at or above that p99\n")
    else:
        print("TRANSPARENT: no finite slope underneath it (nodata coincides "
              "with every transparent pixel, or the panel is fully opaque) "
              "- nothing to say about the legend's lowest threshold\n")

    print(f"{'colour':>18}  {'share':>7}  {'slope min':>9}  {'p50':>6}  {'max':>6}")
    for colour, count in counts.most_common():
        share = count / total
        if share < MIN_SHARE:
            continue
        selected = opaque & np.all(rgba[:, :, :3] == np.array(colour), axis=2)
        values = slope[selected]
        values = values[np.isfinite(values)]
        if not values.size:
            continue
        print(f"{str(colour):>18}  {share:>6.1%}  {values.min():>9.1f}  "
              f"{np.percentile(values, 50):>6.1f}  {values.max():>6.1f}")

    print("\nEach row is one legend band. Copy the colours and the slope "
          "ranges into SF_LEGEND in src/sar_geo_mcp/caltopo/shading.py, then "
          "replace LEGEND_PROVENANCE with the date, the ground and the "
          "resolution this was run at.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
