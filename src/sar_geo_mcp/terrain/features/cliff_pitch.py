"""Steep ground measured by the drop you would rig on, not the area it covers.

WHY THIS IS A SECOND DETECTOR AND NOT A CHANGED DEFAULT
-------------------------------------------------------
`cliff_band` answers a real question - "what stops a ground team" - and its
200 m2 floor is correct for it: a 3 m outcrop does not stop anybody, and
reporting it would bury the bands that do. That contract stays.

This detector answers a different question: "what can a rope team work on?"
For that, an area floor is not merely conservative, it is inverted. A vertical
face has almost no planimetric footprint - a 20 m drop 15 m wide covers
210 m2 at 55 degrees, 109 m2 at 70, and 53 m2 at 80 - so the steeper the
ground, the more likely an area threshold discards it. Measured over El Corte
de Madera Creek OSP at 1 m pixels, `cliff_band` returned ZERO patches above
65 degrees. Re-run at a 40 m2 floor and ranked by vertical extent, the same
raster returned eleven, the tallest a 27 m face at a 70 degree mean, with a
trail crossing 8 m above its lip.

THE TWO THRESHOLDS THAT MATTER
------------------------------
`min_vertical_extent_m` is the one doing the work. 8 m is about where a drop
stops being an edge-transition drill and starts being a station you can lower
a litter down, and it replaces area as the way short noise is rejected. Note
it only binds on metre-scale ground: at 10 m pixels, clearing 55 degrees
already requires 14 m of rise across one pixel, so every qualifying face is
tall by construction.

`min_area_m2` is kept only to drop single-pixel speckle, which is why it sits
at 40 rather than 200.

Shapes come back TALLEST FIRST and their ids follow that order, so
`pitch_0` is the best pitch on the ground. That is a deliberate departure
from the other detectors, whose ids follow raster scan order: a rank is the
useful thing to cite here, and nothing downstream depends on these ids yet.

WHAT THIS STILL CANNOT TELL YOU
-------------------------------
A DEM smooths anything narrower than a few pixels, so an undercut or overhung
face reads SHALLOWER than it is, never steeper - a conservative error, but an
error. Bare-earth returns see neither boulders nor downed timber nor brush.
And a 70 degree mean over 27 m of drop is not a plumb wall. Every pitch this
finds is a place to go and look with a clinometer, not a place to plan a
session around.
"""

import numpy as np

from .base import FeatureDetector, FeatureResult, polygonize
from .context import TerrainContext


class CliffPitchDetector(FeatureDetector):
    kind = "cliff_pitch"
    description = ("Steep ground ranked by the vertical drop you could rig on - "
                   "the rope-work question that an area threshold inverts. "
                   "Tallest first.")
    assumptions = {
        "slope_deg": 55.0,
        "min_vertical_extent_m": 8.0,
        "min_area_m2": 40.0,
        "note": ("area is kept only to drop single-pixel speckle; vertical "
                 "extent is the real filter. A vertical face has almost no "
                 "planimetric footprint, so an area floor discards the "
                 "steepest ground it is meant to find"),
    }

    def detect(self, ctx: TerrainContext) -> FeatureResult:
        grid = ctx.grid
        slope_deg = np.nan_to_num(ctx.slope_aspect.slope_deg, nan=0.0)
        shapes = polygonize(
            (slope_deg >= self.assumptions["slope_deg"]) & grid.mask,
            grid.transform, grid.pixel_size_m,
            kind=self.kind, id_prefix="pitch",
            min_area_m2=self.assumptions["min_area_m2"],
            elevation=grid.elevation, slope_deg=slope_deg,
            min_vertical_extent_m=self.assumptions["min_vertical_extent_m"],
            rank_by_vertical_extent=True,
            extra_properties={"threshold_deg": self.assumptions["slope_deg"]},
        )
        summary = {"count": len(shapes)}
        if shapes:
            summary["tallest_vertical_extent_m"] = shapes[0].properties["vertical_extent_m"]
            summary["total_area_m2"] = sum(s.properties["area_m2"] for s in shapes)
        return self.envelope(shapes, summary=summary)
