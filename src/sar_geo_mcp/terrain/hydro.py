"""Drainage extraction and ravine depth measurement.

Channels are thresholded on contributing AREA, not cell count, so the network
does not change shape when the DEM resolution does.

Ravine depth is BANK-TOP TRANSECT DEPTH: at intervals along a channel, walk
perpendicular up each side to the first major slope break and report the climb.
That is the number a searcher experiences — "you will climb 30 m at 40 degrees
to get out of this" — rather than the channel-to-divide relief, which
overstates crossing effort.
"""

import math

import numpy as np

HYDRO_ASSUMPTIONS = {
    # A channel must drain at least this much ground to count as a drainage.
    "min_contributing_area_m2": 10000.0,  # 1 hectare
    "min_channel_length_m": 40.0,
    # Bank-top search: walk outward until the slope falls below this, having
    # first exceeded it. The break is where the sidewall gives way to the bench
    # or ridge above.
    "bank_top_slope_deg": 15.0,
    "max_transect_m": 150.0,
    "transect_interval_m": 25.0,
    # Slope is smoothed over this many samples before looking for the break, so
    # a single noisy lidar pixel cannot end the walk early.
    "slope_smoothing_samples": 3,
}


def channel_mask(accumulation: np.ndarray, pixel_size_m: float,
                 min_area_m2: float) -> np.ndarray:
    """Cells draining at least `min_area_m2` of ground."""
    cell_area_m2 = pixel_size_m**2
    min_cells = min_area_m2 / cell_area_m2
    return accumulation >= min_cells


def upstream_is_truncated(accumulation: np.ndarray, pixel_size_m: float,
                          min_area_m2: float) -> bool:
    """True when a channel enters through the DEM edge.

    Flow accumulation over a clipped window cannot see upstream area outside
    it, so any channel touching the border means contributing areas — and
    therefore channel ranking — are understated. Callers surface this rather
    than presenting the numbers as complete.
    """
    channels = channel_mask(accumulation, pixel_size_m, min_area_m2)
    if not channels.any():
        return False
    return bool(
        channels[0, :].any() or channels[-1, :].any()
        or channels[:, 0].any() or channels[:, -1].any()
    )


def extract_channels(accumulation: np.ndarray, network: list, transform,
                     pixel_size_m: float,
                     min_area_m2: float = HYDRO_ASSUMPTIONS["min_contributing_area_m2"],
                     min_length_m: float = HYDRO_ASSUMPTIONS["min_channel_length_m"],
                     clip_to=None, *, kind: str = "drainage",
                     id_prefix: str = "drainage") -> list[dict]:
    """Rank traced channel centrelines by contributing area.

    `network` comes from the pysheds backend, which follows D8 flow paths.
    Thresholding accumulation and vectorizing the resulting mask is NOT
    equivalent: it yields a wide blob whose traced outline is many times the
    channel's real length.

    `clip_to` restricts channels to a geometry — normally the assignment. The
    DEM deliberately covers a context buffer so flow routing can see upstream
    area, but a drainage in the buffer that never touches the assignment is not
    this assignment's problem. Without clipping, a large neighbouring valley
    outranks the drainage actually inside the zone. Contributing area is still
    measured along the FULL traced path, because a watershed does not stop at
    an assignment boundary.

    `kind` and `id_prefix` exist because the ridge detector runs this same
    tracing over the negated surface: a ridge crest is where the inverted
    hill's water would collect. Ids are `<id_prefix>_<rank>`.
    """
    from rasterio.transform import rowcol
    from shapely.geometry import LineString, mapping

    if not network:
        return []

    mercator_per_ground = abs(transform.a) / pixel_size_m
    cell_area_m2 = pixel_size_m**2
    min_cells = min_area_m2 / cell_area_m2

    candidates = []
    for coordinates in network:
        if len(coordinates) < 2:
            continue

        line = LineString(coordinates)

        rows, cols = rowcol(transform, [c[0] for c in coordinates],
                            [c[1] for c in coordinates])
        rows = np.clip(np.asarray(rows), 0, accumulation.shape[0] - 1)
        cols = np.clip(np.asarray(cols), 0, accumulation.shape[1] - 1)
        peak_cells = float(accumulation[rows, cols].max())
        if peak_cells < min_cells:
            continue

        full_length_m = line.length / mercator_per_ground

        measured = line
        if clip_to is not None:
            measured = line.intersection(clip_to)
            if measured.is_empty:
                continue
            if measured.geom_type == "MultiLineString":
                # Keep the longest run inside the geometry; a channel that
                # weaves in and out is reported by its principal segment.
                measured = max(measured.geoms, key=lambda part: part.length)
            if measured.geom_type != "LineString":
                continue

        length_m = measured.length / mercator_per_ground
        if length_m < min_length_m:
            continue

        candidates.append(
            {
                "geometry": mapping(measured),
                "length_m": length_m,
                "full_length_m": full_length_m,
                "contributing_area_m2": peak_cells * cell_area_m2,
            }
        )

    candidates.sort(key=lambda item: -item["contributing_area_m2"])

    return [
        {
            "id": f"{id_prefix}_{index}",
            "kind": kind,
            "geometry": candidate["geometry"],
            "properties": {
                "length_m": candidate["length_m"],
                "full_length_m": candidate["full_length_m"],
                "contributing_area_m2": candidate["contributing_area_m2"],
            },
        }
        for index, candidate in enumerate(candidates)
    ]


def bank_top_along_ray(elevations: np.ndarray, sample_spacing_m: float,
                       break_slope_deg: float, smoothing: int) -> dict:
    """Walk outward from a channel and find the first major slope break.

    `elevations[0]` is the channel bottom. The bank top is the first sample
    where the smoothed local slope falls below `break_slope_deg` AFTER it has
    exceeded it — i.e. where the sidewall gives way to a bench or ridge. If it
    never breaks within the ray, the result is flagged `truncated` and reports
    the far end, so a deep ravine is never silently reported as shallow.
    """
    elevations = np.asarray(elevations, dtype="float64")
    if elevations.size < 2:
        return {"distance_m": 0.0, "climb_m": 0.0, "slope_deg": 0.0,
                "truncated": False}

    rise = np.diff(elevations)
    local_slope_deg = np.degrees(np.arctan(np.abs(rise) / sample_spacing_m))

    if smoothing > 1:
        kernel = np.ones(smoothing) / smoothing
        local_slope_deg = np.convolve(local_slope_deg, kernel, mode="same")

    steep = local_slope_deg >= break_slope_deg
    if not steep.any():
        # No wall at all — the channel sits in flat ground. That is a complete
        # answer, not a truncated one: there is nothing to climb out of.
        return {"distance_m": 0.0, "climb_m": 0.0, "slope_deg": 0.0,
                "truncated": False}

    first_steep = int(np.argmax(steep))
    after = np.where(~steep[first_steep:])[0]
    break_index = first_steep + int(after[0]) if after.size else None

    # Truncated means the wall was still climbing when the ray ran out, so the
    # reported depth is a floor rather than the real bank top.
    truncated = break_index is None
    index = (elevations.size - 1) if truncated else break_index

    distance_m = index * sample_spacing_m
    climb_m = float(elevations[index] - elevations[0])
    slope_deg = (
        math.degrees(math.atan(abs(climb_m) / distance_m)) if distance_m > 0 else 0.0
    )

    return {
        "distance_m": distance_m,
        "climb_m": climb_m,
        "slope_deg": slope_deg,
        "truncated": bool(truncated),
    }


def _sample_ray(elevation: np.ndarray, transform, start_xy, direction,
                max_distance_m: float, pixel_size_m: float) -> np.ndarray:
    """Nearest-neighbour sampling along a ray, one sample per pixel."""
    inverse = ~transform
    mercator_per_ground = abs(transform.a) / pixel_size_m
    steps = int(max_distance_m / pixel_size_m)

    samples = []
    for step in range(steps + 1):
        offset = step * pixel_size_m * mercator_per_ground
        x = start_xy[0] + direction[0] * offset
        y = start_xy[1] + direction[1] * offset
        col, row = inverse * (x, y)
        row_index, col_index = int(row), int(col)
        if not (0 <= row_index < elevation.shape[0]
                and 0 <= col_index < elevation.shape[1]):
            break
        value = elevation[row_index, col_index]
        if not np.isfinite(value):
            break
        samples.append(float(value))

    return np.asarray(samples)


def measure_channel_depth(channel: dict, elevation: np.ndarray, transform,
                          pixel_size_m: float) -> dict:
    """Bank-top depth for one drainage, measured on perpendicular transects.

    Left and right are reported separately because sidewalls are routinely
    asymmetric, and a team crossing needs to know which side is the problem.
    """
    from shapely.geometry import shape as shapely_shape

    line = shapely_shape(channel["geometry"])
    interval = HYDRO_ASSUMPTIONS["transect_interval_m"]
    max_transect = HYDRO_ASSUMPTIONS["max_transect_m"]
    break_slope = HYDRO_ASSUMPTIONS["bank_top_slope_deg"]
    smoothing = HYDRO_ASSUMPTIONS["slope_smoothing_samples"]

    mercator_per_ground = abs(transform.a) / pixel_size_m
    length_ground_m = line.length / mercator_per_ground
    positions = np.arange(interval, max(length_ground_m - interval, interval),
                          interval)

    left_results, right_results = [], []
    for position in positions:
        fraction = (position / length_ground_m) if length_ground_m > 0 else 0.5
        point = line.interpolate(min(max(fraction, 0.0), 1.0), normalized=True)
        ahead = line.interpolate(min(fraction + 0.02, 1.0), normalized=True)

        dx, dy = ahead.x - point.x, ahead.y - point.y
        norm = math.hypot(dx, dy)
        if norm == 0:
            continue
        # Perpendicular unit vectors, left and right of travel direction.
        left_dir = (-dy / norm, dx / norm)
        right_dir = (dy / norm, -dx / norm)

        for direction, results in ((left_dir, left_results),
                                   (right_dir, right_results)):
            samples = _sample_ray(elevation, transform, (point.x, point.y),
                                  direction, max_transect, pixel_size_m)
            if samples.size >= 2:
                results.append(
                    bank_top_along_ray(samples, pixel_size_m, break_slope, smoothing)
                )

    def summarise(results: list[dict], key: str) -> float | None:
        values = [r[key] for r in results]
        return float(np.median(values)) if values else None

    return {
        "depth_m": {
            "left": summarise(left_results, "climb_m"),
            "right": summarise(right_results, "climb_m"),
        },
        "sidewall_slope_deg": {
            "left": summarise(left_results, "slope_deg"),
            "right": summarise(right_results, "slope_deg"),
        },
        "transect_count": len(left_results) + len(right_results),
        "any_transect_truncated": any(
            r["truncated"] for r in left_results + right_results
        ),
        "method": (
            "bank-top transect depth: perpendicular rays at "
            f"{interval:g} m intervals, walk to the first slope break below "
            f"{break_slope:g} deg, capped at {max_transect:g} m; median of transects"
        ),
    }
