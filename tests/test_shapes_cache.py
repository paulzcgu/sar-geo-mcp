"""The shape questions cache one grid per staged directory.

`prepare_terrain` re-stages a moved assignment into the SAME directory, so a
cache keyed by the path alone answers questions about the new boundary with
the old raster. `stage_terrain` writes manifest.json last, which is why its
stat is what the key carries.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from rasterio.transform import from_origin
from test_grid import make_wgs84_square, write_fetch_dir

from sar_geo_mcp.terrain import shapes

TRANSFORM = from_origin(-13618826.5, 4486605.0, 10.0, 10.0)
GEOMETRY = make_wgs84_square(-122.3400, 37.3380, -122.3380, 37.3400)


@pytest.fixture(autouse=True)
def _fresh_grid_cache():
    shapes._grid_for.cache_clear()
    yield
    shapes._grid_for.cache_clear()


def test_a_re_staged_directory_is_read_again_rather_than_served_stale(tmp_path):
    write_fetch_dir(tmp_path, np.zeros((40, 40)), TRANSFORM, GEOMETRY)
    first = shapes._grid_for(str(tmp_path))
    assert float(np.nanmax(first.elevation)) == pytest.approx(0.0)

    # A re-stage: a new dem.tif in the same directory, manifest.json last.
    write_fetch_dir(tmp_path, np.full((40, 40), 100.0), TRANSFORM, GEOMETRY)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["warnings"] = ["re-staged"]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    second = shapes._grid_for(str(tmp_path))
    assert float(np.nanmax(second.elevation)) == pytest.approx(100.0)


def test_repeated_questions_on_one_staging_read_the_grid_once(tmp_path):
    write_fetch_dir(tmp_path, np.zeros((40, 40)), TRANSFORM, GEOMETRY)
    first = shapes._grid_for(str(tmp_path))
    second = shapes._grid_for(str(tmp_path))
    assert first is second
    assert shapes._grid_for.cache_info().hits == 1
