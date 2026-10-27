"""Hillshade, and the reason there is more than one sun.

A hillshade lit from one azimuth is nearly blind to features running
PARALLEL to the light, which is most of what a contouring road does. The
last two tests here are the evidence for that claim and for the fix, and
they are why `multi_azimuth_hillshade` exists rather than a single call
with a good default.
"""

import numpy as np
import pytest

from sar_geo_mcp.terrain.hillshade import (
    DEFAULT_AZIMUTHS,
    DEFAULT_SUN_ALTITUDE_DEG,
    hillshade,
    multi_azimuth_hillshade,
    to_grey_bytes,
)


def flat(rows=40, cols=40):
    return np.zeros((rows, cols), dtype="float32")


def north_south_bench(rows=80, cols=80, *, at=40, width=4, depth=1.5):
    """A cut bench running north-south: a trench `depth` deep, `width` wide,
    carved into otherwise LEVEL ground. The trench walls are the only relief
    in the grid, and they face due east and due west.

    This used to be carved into a plane falling 40 units over 80 columns
    (~26.5 degrees). That regional tilt does not cancel out under
    north/south lighting - both trench walls sit on the same tilted
    background, so the "along" hillshade retained real contrast (0.174)
    that had nothing to do with the trench, leaving
    test_one_sun_is_blind_to_a_bench_running_along_the_light with under 2%
    relative margin against its 0.25 cutoff (0.1744 / (0.25 * 0.7082) =
    0.9850). Levelling the background removes that confound: measured
    along=0.100, across=0.920, ratio 0.1087 - the margin the trench itself
    actually earns, not a residual borrowed from the fixture's slope.
    """
    ground = np.zeros((rows, cols), dtype="float32")
    ground[:, at:at + width] -= depth
    return ground


def contrast(image):
    """Spread across the bench, measured along one row."""
    return float(image[40].max() - image[40].min())


def test_flat_ground_is_uniform_at_the_suns_altitude():
    shade = hillshade(flat(), pixel_size_m=1.0, azimuth_deg=315.0, altitude_deg=30.0)
    # cos(zenith) with zenith = 90 - 30; a flat surface returns sin(altitude).
    assert np.allclose(shade, 0.5, atol=1e-6)


def test_a_slope_facing_the_sun_is_brighter_than_one_facing_away():
    # Elevation must DECREASE eastward (descend east) to be a genuinely
    # east-facing slope: aspect is the steepest-DESCENT bearing (slope.py,
    # pinned by test_aspect_east_descending_plane_reads_east). Rising
    # eastward, as `linspace(0.0, 40.0, ...)` would give, is west-facing
    # and made this assertion fail outright (0.0 vs 0.969) before the fix -
    # not a close call, a straight inversion.
    east_facing = np.tile(np.linspace(40.0, 0.0, 40), (40, 1)).astype("float32")
    lit_from_east = hillshade(east_facing, 1.0, azimuth_deg=90.0, altitude_deg=30.0)
    lit_from_west = hillshade(east_facing, 1.0, azimuth_deg=270.0, altitude_deg=30.0)
    assert lit_from_east.mean() > lit_from_west.mean()


def test_values_stay_in_range():
    shade = hillshade(north_south_bench(), 1.0, azimuth_deg=315.0)
    assert shade.min() >= 0.0 and shade.max() <= 1.0
    # Range alone would pass for an all-zeros (or all-mid-grey) return; pin
    # down that the trench actually produces varying illumination. Measured
    # spread across the bench row is ~0.735 (0.033 to 0.767).
    assert shade[40].max() - shade[40].min() > 0.1


def test_one_sun_is_blind_to_a_bench_running_along_the_light():
    """The whole argument for multi-azimuth, as a test."""
    bench = north_south_bench()
    along = hillshade(bench, 1.0, azimuth_deg=0.0)    # light from the north
    across = hillshade(bench, 1.0, azimuth_deg=90.0)  # light from the east
    # Measured on the level fixture: along=0.09999996, across=0.91962,
    # ratio 0.1087 against the 0.25 cutoff (was 0.2463 on the old tilted
    # fixture - under 2% relative margin, mostly a fixture artefact).
    assert contrast(along) < 0.25 * contrast(across)


def test_the_composite_keeps_the_contrast_the_worst_sun_loses():
    bench = north_south_bench()
    along = hillshade(bench, 1.0, azimuth_deg=0.0)
    composite = multi_azimuth_hillshade(bench, 1.0)
    # Measured on the level fixture: composite=0.5, along=0.09999996,
    # ratio 5.00x against the 2.0x cutoff (was 4.80x on the old fixture).
    assert contrast(composite) > 2.0 * contrast(along)


def test_defaults_are_four_azimuths_and_a_low_sun():
    assert len(DEFAULT_AZIMUTHS) == 4
    assert sorted(set(a % 180 for a in DEFAULT_AZIMUTHS)) == [0.0, 45.0, 90.0, 135.0]
    assert DEFAULT_SUN_ALTITUDE_DEG == 30.0


def test_nan_elevation_does_not_propagate_everywhere():
    ground = north_south_bench()
    ground[10, 10] = np.nan
    shade = multi_azimuth_hillshade(ground, 1.0)
    assert np.isfinite(shade).mean() > 0.95


def test_nan_pixel_that_is_never_finite_stays_nan():
    """Pins the running-best refactor of `multi_azimuth_hillshade` against
    silently turning a universally-missing pixel into mid-grey.

    A single NaN elevation sample at (10, 10) makes `compute_slope_aspect`
    produce NaN slope/aspect at its four axis-aligned neighbours - and
    since slope/aspect are computed once and shared across azimuths, those
    neighbours are NaN under every one of the four default azimuths, not
    just some (checked directly below for (9, 10)). The combination logic
    scores a NaN shade as zero deviation from mid-grey purely so a real
    value from another azimuth wins the contest - it must never fall back
    to fabricating 0.5 as the *output* when there is no real value from any
    azimuth to prefer instead. That would report a missing measurement as
    a value, which is the thing this codebase's null-not-zero rule forbids.
    """
    ground = north_south_bench()
    ground[10, 10] = np.nan
    for azimuth in DEFAULT_AZIMUTHS:
        assert np.isnan(hillshade(ground, 1.0, azimuth_deg=azimuth)[9, 10])

    shade = multi_azimuth_hillshade(ground, 1.0)
    assert np.isnan(shade[9, 10])


def test_isolated_nan_does_not_blank_its_neighbours():
    """The flip side of the pin above: a single missing elevation sample
    must not spread past the gradient stencil that actually touches it.
    (10, 10) itself, and pixels two steps away from it, sit outside that
    reach and must come out finite.
    """
    ground = north_south_bench()
    ground[10, 10] = np.nan
    shade = multi_azimuth_hillshade(ground, 1.0)
    assert np.isfinite(shade[10, 10])
    assert np.isfinite(shade[8, 10])
    assert np.isfinite(shade[12, 10])
    assert np.isfinite(shade[10, 8])
    assert np.isfinite(shade[10, 12])


def test_no_azimuths_is_refused():
    with pytest.raises(ValueError):
        multi_azimuth_hillshade(flat(), 1.0, azimuths=())


def test_to_grey_bytes_maps_the_documented_points():
    # NaN renders mid-grey rather than black - the point of the function,
    # per its docstring - and is pinned here alongside the plain 0/0.5/1
    # endpoints so a future change can't quietly turn NaN into 0.
    shade = np.array([np.nan, 0.0, 0.5, 1.0], dtype="float64")
    grey = to_grey_bytes(shade)
    assert grey.dtype == np.uint8
    # NaN and 0.5 both land on 127 (0.5 * 255 = 127.5, truncated by the
    # uint8 cast) - which is exactly the point: NaN reads as the same
    # mid-grey as a real flat-ground measurement, not as black (0).
    assert grey.tolist() == [127, 0, 127, 255]
