"""Traverse rate and sweep time: the computed basis for time divergences.

Every constant here is an ASSUMPTION, not a measurement. They are recorded in
the output verbatim so a planner can see and challenge them; a later CLI flag
may override them per-search. Do not tune silently.

How fast someone WALKS is not decided here: Tobler, the off-way factor and
the vegetation factors are `terrain.movement`'s, the one model every tool
shares. What stays here is DETECTION - effective sweep width by lifeform and
density - which no walking tool needs.
"""

import numpy as np

from . import movement
from .movement import tobler_rate_kmh  # noqa: F401  (re-exported; callers import it here)

ASSUMPTIONS = {
    "tobler": movement.TOBLER_FORMULA,
    # A sweep is walked off any mapped way: Tobler's off-path factor times the
    # LANDFIRE lifeform factor, area-weighted. Density does not change speed.
    "off_way_factor": movement.OFF_WAY_FACTOR,
    "speed_factor_by_lifeform": dict(movement.VEGETATION_FACTORS),
    "movement_model": "sar_geo_mcp.terrain.movement",
    # Effective sweep width (metres) by LANDFIRE lifeform: how far to the side a
    # searcher reliably detects. Type matters more than density: chaparral is
    # the narrow case, grass the wide one, timber in between.
    "sweep_width_m_by_lifeform": {
        "Tree": 15.0,
        "Shrub": 8.0,
        "Herb": 30.0,
        "Sparse": 25.0,
        "Barren": 30.0,
        "Developed": 20.0,
        "Agriculture": 25.0,
    },
    "sweep_width_m_default": 15.0,
    # Sweep width modifier by density class (dense understory narrows detection).
    "sweep_width_factor_by_density": {"open": 1.2, "moderate": 1.0, "dense": 0.7},
    "excluded_lifeforms": ["Water", "Snow-Ice", "Fill-NoData"],
}


#: How a sweep width was reached, as `sweep_width_basis` names it: off the
#: ground's lifeforms and its measured density; off its lifeforms, density
#: assumed moderate (ground with no vegetation panel); or the flat default,
#: no lifeform being known - a number read off no vegetation at all.
BASIS_MEASURED = "lifeform_and_density"
BASIS_DENSITY_ASSUMED = "lifeform_moderate_density_assumed"
BASIS_DEFAULT = "default_width"


def sweep_width_basis(lifeform_fractions: dict[str, float], *,
                      density_measured: bool) -> str:
    """Which of the three bases `effective_sweep_width_m` stands on for
    these lifeforms: the default when none of them counts (all excluded,
    or none known)."""
    excluded = set(ASSUMPTIONS["excluded_lifeforms"])
    if not any(fraction > 0 for lifeform, fraction in lifeform_fractions.items()
               if lifeform not in excluded):
        return BASIS_DEFAULT
    return BASIS_MEASURED if density_measured else BASIS_DENSITY_ASSUMED


def default_width_warning(width_m: float) -> str:
    """What a reader must hear beside a sweep width that is the default."""
    return (f"vegetation type unknown, so the sweep width is the flat {width_m:g} m default, "
            "not one read off this ground's vegetation: the sweep time and any coverage "
            "computed from it rest on that default")


def effective_sweep_width_m(lifeform_fractions: dict[str, float],
                            density: str) -> float:
    widths = ASSUMPTIONS["sweep_width_m_by_lifeform"]
    excluded = set(ASSUMPTIONS["excluded_lifeforms"])
    default = ASSUMPTIONS["sweep_width_m_default"]

    total = 0.0
    weight = 0.0
    for lifeform, fraction in lifeform_fractions.items():
        if lifeform in excluded:
            continue
        total += fraction * widths.get(lifeform, default)
        weight += fraction
    base = (total / weight) if weight > 0 else default

    return base * ASSUMPTIONS["sweep_width_factor_by_density"][density]


def traverse_metrics(slope_deg: np.ndarray, mask: np.ndarray, area_m2: float,
                     density_class: str, lifeform_fractions: dict[str, float], *,
                     density_measured: bool = True) -> dict:
    """The walking rate, sweep width and one pass's hours over a footprint.
    `density_measured` is False when `density_class` is an assumption
    standing in for a density no panel measured; `sweep_width_basis` then
    says so."""
    rates = tobler_rate_kmh(slope_deg)[mask]
    baseline = float(rates.mean()) if rates.size else 0.0

    # Off-way walking over this footprint's lifeforms, by the shared model.
    adjusted = (baseline * movement.DEFAULT.off_way_factor
                * movement.DEFAULT.mean_vegetation_factor(lifeform_fractions))

    esw = effective_sweep_width_m(lifeform_fractions, density_class)
    # One pass: cover area at (sweep width x adjusted speed).
    sweep_time_hours = (
        area_m2 / (esw * adjusted * 1000.0) if adjusted > 0 and esw > 0 else None
    )

    return {
        "baseline_rate_kmh": baseline,
        "vegetation_adjusted_rate_kmh": adjusted,
        "effective_sweep_width_m": esw,
        "sweep_width_basis": sweep_width_basis(lifeform_fractions,
                                               density_measured=density_measured),
        "sweep_time_hours_per_pass": sweep_time_hours,
        "method": ("tobler1993 off-way x LANDFIRE lifeform speed factor (terrain.movement) "
                   "+ lifeform/density sweep matrix"),
        "assumptions": ASSUMPTIONS,
    }
