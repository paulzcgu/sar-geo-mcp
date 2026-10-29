import math

import numpy as np
import pytest

from sar_geo_mcp.terrain.traverse import (
    ASSUMPTIONS,
    effective_sweep_width_m,
    tobler_rate_kmh,
    traverse_metrics,
)


def test_tobler_flat_ground_is_about_five_kmh():
    rate = tobler_rate_kmh(np.zeros((4, 4)))

    assert float(rate.mean()) == pytest.approx(6.0 * math.exp(-3.5 * 0.05), rel=1e-6)


def test_tobler_steep_ground_is_slow():
    rate = tobler_rate_kmh(np.full((4, 4), 40.0))

    assert float(rate.mean()) < 0.5


def test_sweep_width_grass_wider_than_chaparral():
    grass = effective_sweep_width_m({"Herb": 1.0}, "open")
    chaparral = effective_sweep_width_m({"Shrub": 1.0}, "moderate")

    assert grass > chaparral * 2


def test_sweep_width_blends_lifeform_fractions():
    esw = effective_sweep_width_m({"Tree": 0.5, "Herb": 0.5}, "moderate")
    tree = ASSUMPTIONS["sweep_width_m_by_lifeform"]["Tree"]
    herb = ASSUMPTIONS["sweep_width_m_by_lifeform"]["Herb"]

    assert esw == pytest.approx((0.5 * tree + 0.5 * herb), rel=1e-6)


def test_sweep_width_density_modifier_narrows_in_dense_cover():
    moderate = effective_sweep_width_m({"Tree": 1.0}, "moderate")
    dense = effective_sweep_width_m({"Tree": 1.0}, "dense")

    assert dense < moderate


def test_sweep_width_unknown_lifeform_uses_default():
    esw = effective_sweep_width_m({"Martian": 1.0}, "moderate")

    assert esw == pytest.approx(ASSUMPTIONS["sweep_width_m_default"], rel=1e-6)


def test_traverse_metrics_assembles_and_records_assumptions():
    slope_deg = np.full((10, 10), 10.0)
    mask = np.ones((10, 10), dtype=bool)

    result = traverse_metrics(
        slope_deg=slope_deg, mask=mask, area_m2=100_000.0,
        density_class="dense", lifeform_fractions={"Tree": 1.0},
    )

    assert 0 < result["baseline_rate_kmh"] < 6
    assert result["vegetation_adjusted_rate_kmh"] < result["baseline_rate_kmh"]
    assert result["sweep_time_hours_per_pass"] > 0
    assert result["effective_sweep_width_m"] > 0
    assert result["assumptions"] == ASSUMPTIONS
    assert "tobler" in result["method"]


def test_traverse_metrics_sweep_time_scales_with_area():
    kwargs = dict(
        slope_deg=np.full((10, 10), 10.0), mask=np.ones((10, 10), dtype=bool),
        density_class="open", lifeform_fractions={"Herb": 1.0},
    )

    small = traverse_metrics(area_m2=50_000.0, **kwargs)
    large = traverse_metrics(area_m2=100_000.0, **kwargs)

    assert large["sweep_time_hours_per_pass"] == pytest.approx(
        2 * small["sweep_time_hours_per_pass"], rel=1e-6
    )


def test_the_sweep_width_basis_says_when_the_width_is_the_default():
    """A width off known lifeforms says whether density was measured; a
    width with no counted lifeform - none known, or only excluded ones such
    as water - is the flat default, and says so."""
    from sar_geo_mcp.terrain.traverse import (BASIS_DEFAULT, BASIS_DENSITY_ASSUMED,
                                              BASIS_MEASURED, sweep_width_basis)

    assert sweep_width_basis({"Tree": 1.0}, density_measured=True) == BASIS_MEASURED
    assert sweep_width_basis({"Tree": 1.0}, density_measured=False) == BASIS_DENSITY_ASSUMED
    assert sweep_width_basis({}, density_measured=True) == BASIS_DEFAULT
    assert sweep_width_basis({"Water": 1.0}, density_measured=False) == BASIS_DEFAULT
