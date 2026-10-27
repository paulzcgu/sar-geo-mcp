"""`cliff_pitch`: steep ground measured by the drop you would rig on.

WHY THIS EXISTS
---------------
`cliff_band` answers "what stops a ground team" and its 200 m2 floor is right
for that. It is the wrong instrument for "what can we rig on", and not by a
little: a vertical face has almost no planimetric footprint, so an area floor
discards precisely the steepest ground it is meant to find. A 20 m drop 15 m
wide covers 210 m2 at 55 degrees but only 53 m2 at 80.

Run over El Corte de Madera Creek OSP at 1 m, `cliff_band` returned ZERO
patches over 65 degrees. The same raster at a 40 m2 floor, ranked by vertical
extent rather than area, returned eleven - the tallest a 27 m face at a 70
degree mean. This detector is that second measurement.

NOTE ON RESOLUTION: the 10 m fixture most feature tests use cannot express a
steep-but-short face at all. At 10 m pixels, clearing 55 degrees needs a rise
of 14 m across one pixel, so every qualifying face is automatically tall. The
vertical-extent floor only binds on metre-scale ground, which is why the
threshold test below builds its own 1 m raster.
"""

import numpy as np
import pytest
from pyproj import Transformer
from rasterio.transform import from_origin

from test_grid import LATITUDE, make_wgs84_square, write_fetch_dir

from sar_geo_mcp.terrain.features import REGISTRY, identify, list_kinds

TO4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform

# The square other feature tests use, over a 40x40 grid of 10 m pixels.
SQUARE = make_wgs84_square(-122.3398, 37.3388, -122.3392, 37.3396)
TRANSFORM = from_origin(-13618826.5, 4486605.0, 10.0, 10.0)


def step_dem(rows=40, cols=40, *, rise=40.0, at=5):
    """Flat bench, one abrupt `rise`-metre step, flat bench again.

    The step spans a single pixel column, which is the shape a real cliff has
    on a metre-scale DEM and the shape an area floor cannot see.
    """
    profile = np.zeros(cols)
    profile[at:] = rise
    return profile[None, :].repeat(rows, axis=0)


def ramp_dem(rows=40, cols=40, step=16.0):
    """Uniform east-ascending ramp, ~63.6 degrees everywhere."""
    return (np.arange(cols) * step)[None, :].repeat(rows, axis=0)


def stage(tmp_path, elevation, transform=TRANSFORM, geometry=SQUARE):
    write_fetch_dir(tmp_path, elevation, transform, geometry)
    return tmp_path


def fine_raster(tmp_path, elevation, pixel_m=1.0):
    """Stage `elevation` at metre-scale pixels, with a mask that fits it.

    The geometry is derived from the transform rather than hard-coded so the
    two cannot drift apart.
    """
    ox, oy = -13618826.5, 4486605.0
    transform = from_origin(ox, oy, pixel_m, pixel_m)
    rows, cols = elevation.shape
    inset = 3 * pixel_m
    west, north = TO4326(ox + inset, oy - inset)
    east, south = TO4326(ox + cols * pixel_m - inset, oy - rows * pixel_m + inset)
    geometry = make_wgs84_square(west, south, east, north)
    write_fetch_dir(tmp_path, elevation, transform, geometry)
    return tmp_path


def test_registered_and_self_describing():
    assert "cliff_pitch" in REGISTRY
    entry = next(k for k in list_kinds() if k["kind"] == "cliff_pitch")
    assert "min_vertical_extent_m" in entry["assumptions"]
    assert entry["assumptions"]["min_area_m2"] < 200.0, (
        "the whole point is a floor below cliff_band's")


def test_finds_a_face_that_cliff_band_discards(tmp_path):
    """The regression this detector exists for, at the resolution it happened.

    A 25 m drop across one metre-scale pixel column: about 34 m2 of
    planimetric footprint, so `cliff_band`'s 200 m2 floor discards it
    entirely, while the drop is nearly three rope-lengths of station.
    """
    elevation = np.zeros((60, 60))
    elevation[:, 30:] = 25.0
    directory = fine_raster(tmp_path, elevation)
    results = identify(directory, ["cliff_band", "cliff_pitch"])

    assert results["cliff_band"].summary["count"] == 0, (
        "a one-pixel-wide face is under the 200 m2 floor - that is the bug")
    assert results["cliff_pitch"].summary["count"] >= 1
    pitch = results["cliff_pitch"].shapes[0].properties
    assert pitch["area_m2"] < 200.0, "confirming it is the area floor doing the discarding"
    assert pitch["vertical_extent_m"] > 15.0


def test_shapes_carry_vertical_extent_and_span(tmp_path):
    directory = stage(tmp_path, step_dem(rise=40.0))
    shape = identify(directory, ["cliff_pitch"])["cliff_pitch"].shapes[0]
    props = shape.properties
    assert props["vertical_extent_m"] == pytest.approx(40.0, abs=8.0)
    assert props["span_m"] > 50.0, "the step runs the height of the mask"
    assert props["slope_mean_deg"] > 55.0
    assert props["slope_max_deg"] >= props["slope_mean_deg"]
    assert "area_m2" in props


def test_ranked_by_vertical_extent_and_id_follows_the_rank(tmp_path):
    """Two faces, the taller one narrower. Area alone would invert this."""
    elevation = np.zeros((40, 40))
    elevation[:, 6:] = 20.0
    elevation[0:9, 20:] = 90.0
    elevation[9:, 20:] = 20.0
    directory = stage(tmp_path, elevation)
    shapes = identify(directory, ["cliff_pitch"])["cliff_pitch"].shapes
    extents = [s.properties["vertical_extent_m"] for s in shapes]
    assert extents == sorted(extents, reverse=True), "tallest first"
    assert [s.id for s in shapes] == [f"pitch_{i}" for i in range(len(shapes))]


def test_a_short_step_is_not_a_pitch(tmp_path):
    """Steep but only ~6 m of drop: an edge-transition drill, not a station.

    Reporting it as a pitch would bury the real ones. Needs metre-scale
    pixels to express at all - see the module docstring.
    """
    elevation = np.zeros((60, 60))
    elevation[:, 30:] = 6.0
    directory = fine_raster(tmp_path, elevation)
    result = identify(directory, ["cliff_pitch"])["cliff_pitch"]
    assert result.summary["count"] == 0, (
        f"6 m of drop should not be a pitch; got {result.summary}")


def test_a_tall_step_on_the_same_fine_raster_is_a_pitch(tmp_path):
    """The control for the test above: same geometry, more drop."""
    elevation = np.zeros((60, 60))
    elevation[:, 30:] = 25.0
    directory = fine_raster(tmp_path, elevation)
    result = identify(directory, ["cliff_pitch"])["cliff_pitch"]
    assert result.summary["count"] >= 1
    assert result.summary["tallest_vertical_extent_m"] > 15.0


def test_summary_carries_the_tallest_and_the_count(tmp_path):
    directory = stage(tmp_path, step_dem(rise=40.0))
    summary = identify(directory, ["cliff_pitch"])["cliff_pitch"].summary
    assert summary["count"] >= 1
    assert summary["tallest_vertical_extent_m"] == pytest.approx(40.0, abs=8.0)


def test_flat_ground_is_an_empty_answer_not_a_warning(tmp_path):
    directory = stage(tmp_path, np.zeros((40, 40)))
    result = identify(directory, ["cliff_pitch"])["cliff_pitch"]
    assert result.summary["count"] == 0
    assert result.warnings == []


def test_a_uniform_steep_ramp_reports_one_pitch(tmp_path):
    directory = stage(tmp_path, ramp_dem())
    shapes = identify(directory, ["cliff_pitch"])["cliff_pitch"].shapes
    assert len(shapes) == 1
    assert shapes[0].properties["vertical_extent_m"] > 30.0


def test_cliff_band_shapes_gain_vertical_extent_too(tmp_path):
    """Pure addition: the existing detector keeps its floor and its ids, and
    answers the vertical question when it does fire."""
    directory = stage(tmp_path, ramp_dem())
    shapes = identify(directory, ["cliff_band"])["cliff_band"].shapes
    assert shapes, "the ramp is over 200 m2 and over 55 degrees"
    assert shapes[0].id.startswith("cliff_")
    assert "vertical_extent_m" in shapes[0].properties
    assert "span_m" in shapes[0].properties
