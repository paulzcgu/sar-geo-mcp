"""Bridges across the gaps where a road meets another and the bench fades.

WHY
---
The bench detector (`lineaments.py`) sees a road as a strip FLATTER THAN
ITS SURROUNDINGS. Where roads meet, the ground opens into a landing or a
junction that is flat on every side, so the signal fades exactly there and
every line stops short. Measured on real ground (Mendocino County, 900 m
round a test point, 2 m pixels, 2026-10-21): of 1,108 line ends,
6 touched another line; at two junctions a planner circled on the map,
every line stopped 15-45 m short of where the roads plainly met.

Plain SLOPE does not fade there. A road is level along its whole length,
junctions included, and on a slope map it reads as one continuous dark
stroke. So a gap is bridged along the least-cost path over slope, which
follows the road across the junction instead of cutting a chord through
whatever is between.

WHAT IS ACCEPTED
----------------
A line end is bridged to the nearest other line (or mapped way) within
MAX_GAP_M when the flat path between them:
  - is at most MAX_DETOUR times the straight gap, so it goes across rather
    than round; and
  - is road-gentle INSIDE: 80% of it at or under P80_MAX_DEG, nothing over
    MAX_DEG. The first and last TRIM_M are left out of that test, because a
    traced line's end sits on the road's cut bank or fill, not its tread:
    on the same ground the end pixels read 17-26 deg while the junction
    between them read 5-10, and judging the whole path bridged none of the
    junctions that were circled.
With those rules, 110 of 575 line ends within 40 m of another line were
bridged, median 18 m; both circled junctions were closed. Measured against
NAIP imagery over the circled area, the bridges ran on ground with a
median NDVI of 0.02 - bare dirt - against 0.30 for the area as a whole.

A bridge is never a detection. It is reported apart from the lines, marked
as a bridge, so nothing reads it as a bench that was seen.

All of this is pixel-space geometry on arrays in hand: no files, no network.
"""

from __future__ import annotations

import numpy as np

#: The longest gap bridged. Junction gaps measured 15-45 m; past this a
#: "gap" is more often two separate benches on one hillside.
MAX_GAP_M = 40.0

#: Path length over straight gap. A path that has to go much further round
#: than across is not crossing a junction.
MAX_DETOUR = 1.6

#: Left out of the slope test at each end - see the module docstring.
TRIM_M = 6.0

#: 80th percentile of slope along the path's interior, and its maximum.
#: A road's tread is near level; a junction landing on 2 m pixels read 5-10.
P80_MAX_DEG = 12.0
MAX_DEG = 18.0

#: Where the router may not go at all: no road crosses this.
BLOCKED_DEG = 25.0

#: Ends closer than this already touch; nothing to bridge.
TOUCHING_M = 2.0

BRIDGE_ASSUMPTIONS = {
    "method": ("least-cost path over slope between a line end and the nearest "
               "other line or mapped way, cost 1 + (slope / 6 deg)^2, blocked "
               "over 25 deg"),
    "max_gap_m": MAX_GAP_M,
    "max_detour": MAX_DETOUR,
    "interior_slope": (f"80% at or under {P80_MAX_DEG:.0f} deg, none over "
                       f"{MAX_DEG:.0f}, leaving out {TRIM_M:.0f} m at each end"),
    "is_not": "a detection: a bridge joins two lines across flat ground, it was not seen",
}


def _cost(slope_deg: np.ndarray) -> np.ndarray:
    slope = np.nan_to_num(slope_deg, nan=90.0)
    cost = 1.0 + (slope / 6.0) ** 2
    cost[slope > BLOCKED_DEG] = 1e4
    return cost


def junction_bridges(lines, targets, slope_deg: np.ndarray, pixel_size_m: float, *,
                     max_gap_m: float = MAX_GAP_M) -> list[dict]:
    """Bridges from the ends of `lines` to the nearest of `targets`.

    `lines` and `targets` are shapely LineStrings in raster pixel space
    (col, row); `targets` should hold `lines` themselves first, then any
    mapped ways, so a returned `to` index below len(lines) is a line.
    Returns dicts: `from` (line index), `end` ("start" or "end"), `to`
    (target index), `path` [(col, row), ...], `length_m`, `steepest_deg`
    and `p80_deg`.
    """
    from shapely import STRtree
    from shapely.geometry import LineString, Point
    from skimage.graph import route_through_array

    if not lines:
        return []
    cost = _cost(slope_deg)
    slope = np.nan_to_num(slope_deg, nan=90.0)
    rows, cols = slope.shape
    tree = STRtree(list(targets))
    reach = max_gap_m / pixel_size_m
    trim = max(1, int(round(TRIM_M / pixel_size_m)))

    bridges: list[dict] = []
    seen: list[tuple] = []
    for index, line in enumerate(lines):
        for label, end in (("start", Point(line.coords[0])), ("end", Point(line.coords[-1]))):
            best = None
            for target in tree.query(end, predicate="dwithin", distance=reach):
                target = int(target)
                if target == index:
                    continue
                gap = targets[target].distance(end)
                if gap * pixel_size_m >= TOUCHING_M and (best is None or gap < best[0]):
                    best = (gap, target)
            if best is None:
                continue
            gap, target = best
            meet = targets[target].interpolate(targets[target].project(end))
            r0, c0 = int(round(end.y)), int(round(end.x))
            r1, c1 = int(round(meet.y)), int(round(meet.x))
            if not (0 <= r0 < rows and 0 <= c0 < cols and 0 <= r1 < rows and 0 <= c1 < cols):
                continue
            pad = int(reach) + 2
            top, left = max(min(r0, r1) - pad, 0), max(min(c0, c1) - pad, 0)
            window = cost[top:min(max(r0, r1) + pad, rows), left:min(max(c0, c1) + pad, cols)]
            path, _ = route_through_array(window, (r0 - top, c0 - left), (r1 - top, c1 - left),
                                          fully_connected=True, geometric=True)
            path = [(float(c + left), float(r + top)) for r, c in path]
            if len(path) < 2:
                continue
            length_m = LineString(path).length * pixel_size_m
            if length_m > MAX_DETOUR * max(gap * pixel_size_m, 2 * TOUCHING_M):
                continue
            inner = [slope[int(r), int(c)] for c, r in path[trim:-trim]] or \
                    [slope[int(r), int(c)] for c, r in path]
            p80 = float(np.percentile(inner, 80))
            steepest = float(max(inner))
            if p80 > P80_MAX_DEG or steepest > MAX_DEG:
                continue
            # Two ends that bridge to each other find the same path twice.
            key = (round(path[0][0]), round(path[0][1]), round(path[-1][0]), round(path[-1][1]))
            if any(abs(key[0] - other[2]) <= 2 and abs(key[1] - other[3]) <= 2
                   and abs(key[2] - other[0]) <= 2 and abs(key[3] - other[1]) <= 2
                   for other in seen):
                continue
            seen.append(key)
            bridges.append({"from": index, "end": label, "to": target, "path": path,
                            "length_m": round(length_m, 1),
                            "steepest_deg": round(steepest, 1), "p80_deg": round(p80, 1)})
    return bridges
