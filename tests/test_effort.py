"""Effort per candidate, from the traverse maths that already exists."""

from __future__ import annotations

import numpy as np
import pytest
from rasterio.features import rasterize
from shapely.geometry import LineString, box
from test_boundary import ORIGIN_X, ORIGIN_Y, GROUND_PER_UNIT, staged, valley_dem

from sar_geo_mcp.planning.boundary import load_ground
from sar_geo_mcp.planning.effort import _window, candidate_effort
from sar_geo_mcp.terrain.traverse import tobler_rate_kmh, traverse_metrics


def test_a_line_takes_length_over_the_vegetation_adjusted_rate(tmp_path):
    ground = load_ground(staged(tmp_path, np.zeros((40, 40))))
    axis = LineString([(ORIGIN_X + 50, ORIGIN_Y - 200), (ORIGIN_X + 350, ORIGIN_Y - 200)])
    effort = candidate_effort(ground, axis, "line", evt_path=None)
    length_km = 300 * GROUND_PER_UNIT / 1000
    # Flat ground: Tobler at 0 slope is 6 * exp(-0.175) = 5.04 km/h; a line is walked on
    # a mapped way, factor 1.0, and vegetation does not slow it (terrain.movement).
    assert effort["rate_kmh"] == pytest.approx(5.04, rel=0.02)
    assert effort["hours_one_way"] == pytest.approx(length_km / 5.04, rel=0.02)
    assert effort["method"].startswith("tobler")
    assert effort["assumptions"]["density_class"] == "moderate"
    assert "no CIR panel" in effort["assumptions"]["density_note"]


def test_one_sampler_serves_the_line_rate_and_the_elevation_profile(tmp_path):
    """`sample_along` is the single place an axis becomes raster indices, so
    a line's slopes and a candidate's elevation profile are read at exactly
    the same points rather than by two copies of the same arithmetic."""
    from sar_geo_mcp.planning.effort import sample_along

    ground = load_ground(staged(tmp_path, valley_dem()))
    axis = LineString([(ORIGIN_X + 50, ORIGIN_Y - 200), (ORIGIN_X + 350, ORIGIN_Y - 200)])

    rows, cols = sample_along(ground, axis)

    assert rows.shape == cols.shape and rows.size > 1
    assert rows.max() < ground.slope_deg.shape[0] and cols.max() < ground.slope_deg.shape[1]
    effort = candidate_effort(ground, axis, "line", evt_path=None)
    expected = float(tobler_rate_kmh(ground.slope_deg[rows, cols]).mean())
    assert effort["baseline_rate_kmh"] == pytest.approx(expected, rel=1e-12)


def test_a_steep_line_is_slower(tmp_path):
    ground = load_ground(staged(tmp_path, valley_dem()))
    flat_axis = LineString([(ORIGIN_X + 205, ORIGIN_Y - 50), (ORIGIN_X + 205, ORIGIN_Y - 350)])
    wall_axis = LineString([(ORIGIN_X + 50, ORIGIN_Y - 200), (ORIGIN_X + 350, ORIGIN_Y - 200)])
    flat = candidate_effort(ground, flat_axis, "line", evt_path=None)
    wall = candidate_effort(ground, wall_axis, "line", evt_path=None)
    assert wall["rate_kmh"] < flat["rate_kmh"]


def test_a_corridor_reports_sweep_time_per_pass(tmp_path):
    ground = load_ground(staged(tmp_path, np.zeros((40, 40))))
    ribbon = box(ORIGIN_X + 50, ORIGIN_Y - 215, ORIGIN_X + 350, ORIGIN_Y - 185)
    effort = candidate_effort(ground, ribbon, "corridor", evt_path=None)
    assert effort["effective_sweep_width_m"] > 0
    assert effort["sweep_time_hours_per_pass"] > 0
    assert effort["area_m2"] == pytest.approx(300 * 30 * GROUND_PER_UNIT ** 2, rel=0.02)
    assert effort["assumptions"]["lifeform_source"] == "none; sweep width default"


def test_a_corridor_reads_lifeforms_from_evt_when_present(tmp_path, monkeypatch):
    ground = load_ground(staged(tmp_path, np.zeros((40, 40))))
    monkeypatch.setattr("sar_geo_mcp.planning.effort.composition",
                        lambda path, geom: {"fractions": {"Shrub": 1.0}, "dominant_lifeform": "Shrub",
                                            "top_types": [], "source": "LANDFIRE test"})
    (tmp_path / "evt.tif").write_bytes(b"")   # exists; composition is stubbed above
    ribbon = box(ORIGIN_X + 50, ORIGIN_Y - 215, ORIGIN_X + 350, ORIGIN_Y - 185)
    effort = candidate_effort(ground, ribbon, "corridor", evt_path=tmp_path / "evt.tif")
    assert effort["effective_sweep_width_m"] == pytest.approx(8.0, rel=0.01)   # Shrub, moderate
    assert effort["assumptions"]["lifeform_source"] == "LANDFIRE test"


def test_a_missing_evt_file_means_the_default_sweep_width(tmp_path):
    """corridors.py passes the staged path unconditionally; an unstaged EVT
    (a warning at staging time) must fall back, not crash."""
    ground = load_ground(staged(tmp_path, np.zeros((40, 40))))
    ribbon = box(ORIGIN_X + 50, ORIGIN_Y - 215, ORIGIN_X + 350, ORIGIN_Y - 185)
    effort = candidate_effort(ground, ribbon, "corridor", evt_path=tmp_path / "missing.tif")
    assert effort["assumptions"]["lifeform_source"] == "none; sweep width default"


def test_a_corridor_with_a_dem_void_matches_traverse_metrics_directly(tmp_path):
    """The corridor path must go through terrain.traverse.traverse_metrics,
    not reimplement its arithmetic: a DEM void inside the ribbon is the case
    where a hand-rolled "drop the NaN pixels" mean diverges from
    traverse_metrics' "NaN maps to flat" via tobler_rate_kmh's nan_to_num."""
    dem = np.zeros((40, 40))
    dem[19:21, 10:20] = -999999.0   # a DEM void inside the ribbon
    ground = load_ground(staged(tmp_path, dem))
    ribbon = box(ORIGIN_X + 50, ORIGIN_Y - 215, ORIGIN_X + 350, ORIGIN_Y - 185)

    effort = candidate_effort(ground, ribbon, "corridor", evt_path=None)

    # Build the mask exactly as candidate_effort does (all_touched=False;
    # the 30-unit ribbon is wider than a pixel so this is non-empty).
    mask = rasterize([(ribbon, 1)], out_shape=ground.slope_deg.shape,
                     transform=ground.grid.transform, fill=0, all_touched=False).astype(bool)
    assert mask.any()
    area_m2 = ribbon.area * ground.scale ** 2
    expected = traverse_metrics(slope_deg=ground.slope_deg, mask=mask, area_m2=area_m2,
                                density_class="moderate", lifeform_fractions={})

    assert effort["sweep_time_hours_per_pass"] == pytest.approx(
        expected["sweep_time_hours_per_pass"], rel=1e-9)
    assert effort["baseline_rate_kmh"] == pytest.approx(expected["baseline_rate_kmh"], rel=1e-9)


def _full_raster_corridor(ground, ribbon):
    """What the corridor branch computed before it worked in a window: the
    mask rasterized over the WHOLE slope array."""
    mask = rasterize([(ribbon, 1)], out_shape=ground.slope_deg.shape,
                     transform=ground.grid.transform, fill=0, all_touched=False).astype(bool)
    assert mask.any()
    return traverse_metrics(slope_deg=ground.slope_deg, mask=mask,
                            area_m2=ribbon.area * ground.scale ** 2,
                            density_class="moderate", lifeform_fractions={})


@pytest.mark.parametrize("ribbon_units, where", [
    ((50, 215, 350, 185), "away from the raster edge"),
    ((0, 400, 400, 370), "hard against the raster edge"),
])
def test_the_window_a_corridor_is_measured_in_gives_the_full_rasters_answer(
        tmp_path, ribbon_units, where):
    """A corridor is a 30 m ribbon on a raster kilometres across; rasterizing
    it at full size once per candidate is the whole raster's work a hundred
    times for the same few hundred pixels. The window has to hold every pixel
    the corridor covers - including where it runs off the edge, which is where
    a padded window would otherwise be clipped short."""
    ground = load_ground(staged(tmp_path, valley_dem()))
    x0, y0, x1, y1 = ribbon_units
    ribbon = box(ORIGIN_X + x0, ORIGIN_Y - y0, ORIGIN_X + x1, ORIGIN_Y - y1)

    effort = candidate_effort(ground, ribbon, "corridor", evt_path=None)

    row0, row1, col0, col1 = _window(ground, ribbon)
    assert (row1 - row0) * (col1 - col0) < ground.slope_deg.size
    expected = _full_raster_corridor(ground, ribbon)
    assert effort["baseline_rate_kmh"] == pytest.approx(expected["baseline_rate_kmh"], rel=1e-12)
    assert effort["sweep_time_hours_per_pass"] == pytest.approx(
        expected["sweep_time_hours_per_pass"], rel=1e-12)
    assert effort["baseline_rate_kmh"] > 0, where


def test_a_corridor_wider_than_a_pixel_gets_the_centre_inside_note(tmp_path):
    ground = load_ground(staged(tmp_path, np.zeros((40, 40))))
    ribbon = box(ORIGIN_X + 50, ORIGIN_Y - 215, ORIGIN_X + 350, ORIGIN_Y - 185)   # 30 units tall
    effort = candidate_effort(ground, ribbon, "corridor", evt_path=None)
    assert effort["assumptions"]["mask_note"] == \
        "statistics use pixels whose centre lies inside the corridor"


def test_a_corridor_thinner_than_a_pixel_falls_back_to_touched_pixels(tmp_path):
    ground = load_ground(staged(tmp_path, np.zeros((40, 40))))
    ribbon = box(ORIGIN_X + 100, ORIGIN_Y - 198, ORIGIN_X + 300, ORIGIN_Y - 196)   # 2 units tall
    effort = candidate_effort(ground, ribbon, "corridor", evt_path=None)
    assert effort["assumptions"]["mask_note"] == \
        "the corridor is thinner than a pixel; statistics use touched pixels"
    assert effort["sweep_time_hours_per_pass"] is not None


def test_a_line_entirely_over_a_dem_void_still_returns_a_flat_rate_hour(tmp_path):
    """Every sampled slope is NaN; tobler_rate_kmh maps NaN to flat rather
    than the line producing an empty array and a None/zero result."""
    dem = np.zeros((40, 40))
    dem[15:25, :] = -999999.0   # a DEM void band spanning the whole line's row range
    ground = load_ground(staged(tmp_path, dem))
    axis = LineString([(ORIGIN_X + 50, ORIGIN_Y - 200), (ORIGIN_X + 350, ORIGIN_Y - 200)])
    effort = candidate_effort(ground, axis, "line", evt_path=None)
    length_km = 300 * GROUND_PER_UNIT / 1000
    assert effort["rate_kmh"] == pytest.approx(5.04, rel=0.02)
    assert isinstance(effort["hours_one_way"], float)
    assert effort["hours_one_way"] == pytest.approx(length_km / 5.04, rel=0.02)


def test_the_shared_assumptions_can_be_left_out_for_a_caller_that_states_them_once(tmp_path):
    """A set generator measures a hundred candidates against one density class
    and one traverse table; repeating them per candidate is the same few
    hundred bytes a hundred times. What is left is what this candidate's own
    measurement chose."""
    ground = load_ground(staged(tmp_path, np.zeros((40, 40))))
    axis = LineString([(ORIGIN_X + 50, ORIGIN_Y - 200), (ORIGIN_X + 350, ORIGIN_Y - 200)])
    ribbon = box(ORIGIN_X + 50, ORIGIN_Y - 215, ORIGIN_X + 350, ORIGIN_Y - 185)

    line = candidate_effort(ground, axis, "line", evt_path=None,
                            include_shared_assumptions=False)
    corridor = candidate_effort(ground, ribbon, "corridor", evt_path=None,
                                include_shared_assumptions=False)

    assert line["assumptions"] == {}
    assert set(corridor["assumptions"]) == {"mask_note", "lifeform_source", "lifeform_fractions"}
    # Only the assumptions are dropped; every measurement is still there.
    full = candidate_effort(ground, ribbon, "corridor", evt_path=None)
    assert corridor["sweep_time_hours_per_pass"] == full["sweep_time_hours_per_pass"]
    assert set(full["assumptions"]) > set(corridor["assumptions"])


def test_a_degenerate_corridor_returns_none_without_raising(tmp_path):
    """A corridor with no pixel anywhere near it is empty even under
    all_touched=True; the corridor path must fall through to a None sweep
    time, not raise."""
    ground = load_ground(staged(tmp_path, np.zeros((40, 40))))
    off_grid = box(ORIGIN_X - 10_000, ORIGIN_Y - 10_000, ORIGIN_X - 9_999, ORIGIN_Y - 9_999)
    effort = candidate_effort(ground, off_grid, "corridor", evt_path=None)
    assert effort["sweep_time_hours_per_pass"] is None


# --- a multi-part axis ---------------------------------------------------------------

def multi_axis(*parts_units):
    """A MultiLineString from (x, y) unit pairs, one list per part."""
    from shapely.geometry import MultiLineString

    return MultiLineString([[(ORIGIN_X + x, ORIGIN_Y - y) for x, y in part]
                            for part in parts_units])


def test_every_part_of_a_multi_part_axis_is_sampled(tmp_path):
    """A named way that forks is one candidate whose axis is a
    MultiLineString. Reading it as one concatenated line samples the parts by
    their share of the total, so a short branch can pick up no sample at all
    and the fork is measured as though it were not there."""
    from sar_geo_mcp.planning.effort import sample_along

    ground = load_ground(staged(tmp_path, np.zeros((40, 40))))
    axis = multi_axis([(50, 60), (350, 60)], [(198, 340), (202, 340)])

    rows, cols = sample_along(ground, axis)

    assert rows.size == cols.size and rows.size > 2
    # Row 6 is the long northern branch, row 34 the 4-unit southern stub.
    assert 6 in set(rows.tolist()) and 34 in set(rows.tolist())


def test_a_multi_part_line_takes_its_whole_length_over_the_rate(tmp_path):
    """`length_m` is the axis's TOTAL, both branches, and the one-way time is
    that length over the rate - the same arithmetic a simple line gets."""
    ground = load_ground(staged(tmp_path, np.zeros((40, 40))))
    axis = multi_axis([(50, 100), (350, 100)], [(50, 300), (350, 300)])

    effort = candidate_effort(ground, axis, "line", evt_path=None)

    length_km = 600 * GROUND_PER_UNIT / 1000
    assert effort["length_m"] == pytest.approx(600 * GROUND_PER_UNIT, rel=1e-3)
    assert effort["rate_kmh"] == pytest.approx(5.04, rel=0.02)
    assert effort["hours_one_way"] == pytest.approx(length_km / 5.04, rel=0.02)


def test_a_multi_part_axiss_slope_is_measured_over_all_of_its_parts(tmp_path):
    """A team walks every branch, so the rate is over every branch: the two
    together sit between the flat branch alone and the steep one alone."""
    ground = load_ground(staged(tmp_path, valley_dem()))
    flat = [(205, 50), (205, 350)]          # along the valley floor
    wall = [(50, 200), (350, 200)]          # across both sidewalls
    both = candidate_effort(ground, multi_axis(flat, wall), "line", evt_path=None)
    flat_only = candidate_effort(ground, multi_axis(flat), "line", evt_path=None)
    wall_only = candidate_effort(ground, multi_axis(wall), "line", evt_path=None)

    assert wall_only["rate_kmh"] < both["rate_kmh"] < flat_only["rate_kmh"]
    assert both["hours_one_way"] > 0
