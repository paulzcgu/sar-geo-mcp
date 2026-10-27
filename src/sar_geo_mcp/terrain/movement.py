"""How fast a person walks over staged ground: the one place every number is.

Every tool that turns ground into walking time reads it from here - hasty
candidates' effort and travel minutes, the travel-time rings, the debrief
traverse metrics, and egress's walk-out mode - so two numbers in two tools
cannot disagree about how fast someone crosses the same brush.

THE MODEL
---------
- Tobler's hiking function sets the speed from the grade:
  6 * exp(-3.5 * |grade + 0.05|) km/h, fastest on a gentle downhill. A
  directed walk reads the SIGNED grade of each step; a team walking a line
  both ways reads the unsigned slope.
- On a mapped trail, track or road the speed is Tobler's (factor 1.0). Off
  one it is Tobler's own off-path factor, 0.6, times a VEGETATION factor for
  the LANDFIRE lifeform there. A trail cuts through the brush beside it, so
  vegetation never slows a walker on one.
- Vegetation is read by TYPE (LANDFIRE lifeform), the one vegetation layer
  every staged ground has. Canopy density from a CIR panel, where one
  exists, is reported but does not change speed: the lifeform factor already
  carries a typical density, and counting thick brush twice would be a guess
  on a guess (operator's ruling, 2026-10-20). Density still narrows sweep
  width, which is detection, not movement - see `traverse`.
- Open water is closed, not slow. Ground steeper than IMPASSABLE_SLOPE_DEG
  off a way is closed. A creek costs a fixed number of minutes to get into.

Every number is an ASSUMPTION, not a measurement: starting values, not
calibrated against tracks. They are returned verbatim beside every answer
built on them (`MovementModel.assumptions`), and a caller may override any
of them per call through `MovementModel.with_overrides`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from types import MappingProxyType

import numpy as np

#: Tobler 1993: v = MAX * exp(-DECAY * |grade + OFFSET|) km/h.
TOBLER_MAX_KMH = 6.0
TOBLER_DECAY = 3.5
TOBLER_OFFSET = 0.05
TOBLER_FORMULA = "v_kmh = 6 * exp(-3.5 * |grade + 0.05|), grade = tan(slope)"

#: Ground factors on Tobler's speed: on a mapped way, and off one (Tobler's
#: own off-path factor).
ON_WAY_FACTOR = 1.0
OFF_WAY_FACTOR = 0.6

#: Off-way speed factor by LANDFIRE lifeform, multiplying OFF_WAY_FACTOR.
#: Brush (chaparral, scrub) is the slow case, grass and sparse ground the
#: quick one, timber between; rock and scree are slow underfoot; fields have
#: fences. A lifeform not listed (no data, unknown) is 1.0: an absence of
#: data slows no one.
VEGETATION_FACTORS = MappingProxyType({
    "Tree": 0.8, "Shrub": 0.4, "Herb": 0.9, "Sparse": 0.9, "Barren": 0.7,
    "Agriculture": 0.8, "Developed": 1.0, "Snow-Ice": 0.5,
})
#: Lifeforms that are closed, not slow.
CLOSED_LIFEFORMS = frozenset({"Water"})

#: Ground off a way steeper than this is never entered. A 10 m elevation
#: model smooths a cliff, and 45 degrees is where `steep_band` says a ground
#: team stops; that detector reads this constant.
IMPASSABLE_SLOPE_DEG = 45.0
#: Minutes to get into a creek, at a speed factor of 1.0.
CREEK_CROSSING_MINUTES = 5.0


def signed_tobler_kmh(grade) -> np.ndarray:
    """Tobler's speed on a SIGNED grade (rise over run; negative is downhill)."""
    grade = np.asarray(grade, dtype="float64")
    return TOBLER_MAX_KMH * np.exp(-TOBLER_DECAY * np.abs(grade + TOBLER_OFFSET))


def tobler_rate_kmh(slope_deg) -> np.ndarray:
    """Tobler's speed on an unsigned slope in degrees - a line walked both
    ways. NaN reads as flat: right when averaging a rate along an axis, and
    the reason a least-cost walk masks voids before reading a rate."""
    slope_rad = np.radians(np.nan_to_num(np.asarray(slope_deg, dtype="float64"), nan=0.0))
    return signed_tobler_kmh(np.tan(slope_rad))


@dataclass(frozen=True)
class MovementModel:
    """The numbers one walk is costed with. `DEFAULT` is what every tool uses
    unless a caller overrides a value."""

    speed_factor: float = 1.0
    on_way_factor: float = ON_WAY_FACTOR
    off_way_factor: float = OFF_WAY_FACTOR
    vegetation_factors: MappingProxyType = field(default=VEGETATION_FACTORS)
    creek_crossing_minutes: float = CREEK_CROSSING_MINUTES
    impassable_slope_deg: float = IMPASSABLE_SLOPE_DEG

    def with_overrides(self, *, speed_factor: float | None = None,
                       off_way_factor: float | None = None,
                       creek_crossing_minutes: float | None = None,
                       impassable_slope_deg: float | None = None,
                       vegetation_factors: dict[str, float] | None = None) -> "MovementModel":
        """This model with the caller's values, validated. None keeps a value."""
        changes: dict = {}
        for name, value in (("speed_factor", speed_factor),
                            ("off_way_factor", off_way_factor),
                            ("creek_crossing_minutes", creek_crossing_minutes),
                            ("impassable_slope_deg", impassable_slope_deg)):
            if value is not None:
                changes[name] = float(value)
        for name in ("speed_factor", "off_way_factor"):
            value = changes.get(name, getattr(self, name))
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} ({value:g}) must be above zero; it scales a "
                                 "walking speed, and at zero nobody ever arrives")
        crossing = changes.get("creek_crossing_minutes", self.creek_crossing_minutes)
        if not (math.isfinite(crossing) and crossing >= 0):
            raise ValueError(f"creek_crossing_minutes ({crossing:g}) must be zero or more")
        steepest = changes.get("impassable_slope_deg", self.impassable_slope_deg)
        if not 0 < steepest < 90:
            raise ValueError(f"cliff_slope_deg ({steepest:g}) must be between 0 and 90 "
                             "degrees")
        if vegetation_factors:
            merged = dict(self.vegetation_factors)
            for lifeform, factor in vegetation_factors.items():
                if lifeform in CLOSED_LIFEFORMS:
                    raise ValueError("open water is closed, not slowed, and takes no "
                                     "vegetation factor")
                if lifeform not in VEGETATION_FACTORS:
                    raise ValueError(f"{lifeform!r} is not a LANDFIRE lifeform; "
                                     f"vegetation_factors takes {', '.join(VEGETATION_FACTORS)}")
                factor = float(factor)
                if not (math.isfinite(factor) and factor > 0):
                    raise ValueError(f"the vegetation factor for {lifeform} ({factor:g}) must "
                                     "be above zero; ground nobody can enter is a closure, "
                                     "not a factor")
                merged[lifeform] = factor
            changes["vegetation_factors"] = MappingProxyType(merged)
        return replace(self, **changes)

    def vegetation_factor(self, lifeform: str) -> float:
        """The off-way factor for one lifeform: 1.0 for one not listed."""
        return float(self.vegetation_factors.get(lifeform, 1.0))

    def mean_vegetation_factor(self, lifeform_fractions: dict[str, float]) -> float:
        """The area-weighted vegetation factor over a footprint's lifeforms.
        Closed lifeforms (water) are left out - nobody walks across them, so
        they say nothing about the pace on the rest - and an empty or
        all-water footprint is 1.0."""
        total = weight = 0.0
        for lifeform, fraction in (lifeform_fractions or {}).items():
            if lifeform in CLOSED_LIFEFORMS or not fraction:
                continue
            total += float(fraction) * self.vegetation_factor(lifeform)
            weight += float(fraction)
        return total / weight if weight > 0 else 1.0

    def off_way_kmh(self, slope_deg, lifeform_fractions: dict[str, float]) -> np.ndarray:
        """Off-way speed on an unsigned slope, over a footprint's lifeforms."""
        return (tobler_rate_kmh(slope_deg) * self.speed_factor * self.off_way_factor
                * self.mean_vegetation_factor(lifeform_fractions))

    def on_way_kmh(self, slope_deg) -> np.ndarray:
        """On-way speed on an unsigned slope: vegetation does not apply."""
        return tobler_rate_kmh(slope_deg) * self.speed_factor * self.on_way_factor

    def assumptions(self) -> dict:
        """Every number, verbatim, for the answer built on it."""
        return {
            "tobler": TOBLER_FORMULA,
            "speed_factor": self.speed_factor,
            "on_way_factor": self.on_way_factor,
            "off_way_factor": self.off_way_factor,
            "vegetation_factors": dict(self.vegetation_factors),
            "vegetation": ("LANDFIRE lifeform; its factor multiplies the off-way factor, and "
                           "a way's own cells ignore it. A lifeform not listed is 1.0. Canopy "
                           "density, where measured, does not change speed"),
            "closed_lifeforms": sorted(CLOSED_LIFEFORMS),
            "impassable_slope_deg": self.impassable_slope_deg,
            "creek_crossing_minutes": self.creek_crossing_minutes,
            "source": "sar_geo_mcp.terrain.movement",
        }


DEFAULT = MovementModel()
