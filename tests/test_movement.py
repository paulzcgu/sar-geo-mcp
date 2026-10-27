"""The shared movement model: one set of walking numbers for every tool."""

import math

import numpy as np
import pytest

from sar_geo_mcp.terrain import movement
from sar_geo_mcp.terrain.movement import DEFAULT, MovementModel

FLAT_KMH = 6.0 * math.exp(-0.175)


# --- one source ------------------------------------------------------------------


def test_every_tool_reads_the_same_tobler():
    from sar_geo_mcp.planning import egress_graph
    from sar_geo_mcp.terrain import traverse

    assert traverse.tobler_rate_kmh is movement.tobler_rate_kmh
    assert egress_graph.signed_tobler_kmh is movement.signed_tobler_kmh


def test_walk_out_walks_as_everyone_else_walks():
    from sar_geo_mcp.planning.egress_graph import MODES

    assert MODES["walk_out"]["on_way_factor"] == DEFAULT.on_way_factor
    assert MODES["walk_out"]["off_way_factor"] == DEFAULT.off_way_factor


def test_the_steep_band_that_stops_a_team_is_the_slope_the_walk_closes():
    from sar_geo_mcp.terrain.features.steep_bands import SteepBandDetector

    assert max(SteepBandDetector.assumptions["thresholds_deg"]) == DEFAULT.impassable_slope_deg


def test_the_traverse_table_states_the_models_factors():
    from sar_geo_mcp.terrain.traverse import ASSUMPTIONS

    assert ASSUMPTIONS["speed_factor_by_lifeform"] == dict(movement.VEGETATION_FACTORS)
    assert ASSUMPTIONS["off_way_factor"] == movement.OFF_WAY_FACTOR
    assert "speed_factor_by_density" not in ASSUMPTIONS


def test_slope_has_one_gradient():
    from sar_geo_mcp.terrain.slope import compute_slope_aspect, slope_gradient

    rng = np.random.default_rng(7)
    z = rng.uniform(0, 30, (12, 9))
    degrees = compute_slope_aspect(z, 10.0).slope_deg
    assert np.allclose(np.tan(np.radians(degrees)), slope_gradient(z, 10.0), rtol=1e-5)


# --- the numbers agree across tools ---------------------------------------------


def test_a_corridor_in_brush_is_swept_at_the_rings_off_trail_speed():
    from sar_geo_mcp.terrain.traverse import traverse_metrics

    result = traverse_metrics(slope_deg=np.zeros((4, 4)), mask=np.ones((4, 4), dtype=bool),
                              area_m2=10_000.0, density_class="moderate",
                              lifeform_fractions={"Shrub": 1.0})
    assert result["vegetation_adjusted_rate_kmh"] == pytest.approx(FLAT_KMH * 0.6 * 0.4)


def test_density_narrows_sweep_width_but_never_changes_speed():
    from sar_geo_mcp.terrain.traverse import traverse_metrics

    kwargs = dict(slope_deg=np.zeros((4, 4)), mask=np.ones((4, 4), dtype=bool),
                  area_m2=10_000.0, lifeform_fractions={"Tree": 1.0})
    dense = traverse_metrics(density_class="dense", **kwargs)
    open_ = traverse_metrics(density_class="open", **kwargs)
    assert dense["vegetation_adjusted_rate_kmh"] == open_["vegetation_adjusted_rate_kmh"]
    assert dense["effective_sweep_width_m"] < open_["effective_sweep_width_m"]


# --- the model ---------------------------------------------------------------------


def test_the_mean_vegetation_factor_weights_by_area_and_leaves_water_out():
    assert DEFAULT.mean_vegetation_factor({"Shrub": 0.5, "Herb": 0.5}) == pytest.approx(0.65)
    assert DEFAULT.mean_vegetation_factor({"Shrub": 0.5, "Water": 0.5}) == pytest.approx(0.4)
    assert DEFAULT.mean_vegetation_factor({}) == 1.0
    assert DEFAULT.mean_vegetation_factor({"Water": 1.0}) == 1.0
    assert DEFAULT.vegetation_factor("Unknown") == 1.0


def test_on_a_way_vegetation_never_applies():
    assert float(DEFAULT.on_way_kmh(0.0)) == pytest.approx(FLAT_KMH)
    assert float(DEFAULT.off_way_kmh(0.0, {"Shrub": 1.0})) == pytest.approx(FLAT_KMH * 0.24)


def test_an_override_is_a_new_model_and_the_default_is_untouched():
    slow = DEFAULT.with_overrides(speed_factor=0.5, vegetation_factors={"Shrub": 0.2})
    assert slow.speed_factor == 0.5 and slow.vegetation_factor("Shrub") == 0.2
    assert DEFAULT.speed_factor == 1.0 and DEFAULT.vegetation_factor("Shrub") == 0.4
    with pytest.raises(TypeError):
        slow.vegetation_factors["Shrub"] = 0.9               # read-only


def test_the_assumptions_carry_every_number():
    stated = MovementModel().assumptions()
    for key in ("tobler", "speed_factor", "on_way_factor", "off_way_factor",
                "vegetation_factors", "impassable_slope_deg", "creek_crossing_minutes"):
        assert key in stated
    assert stated["source"] == "sar_geo_mcp.terrain.movement"
