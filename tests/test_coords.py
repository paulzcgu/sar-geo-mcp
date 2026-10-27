"""Reading coordinates a searcher said out loud.

The governing rule these tests defend: a wrong coordinate is worse than no
coordinate. "No location recorded" makes a planner go and ask; a plausible
but transposed pair sends a team to the wrong drainage. Every ambiguous case
below must resolve to None, not to a best guess.
"""

from __future__ import annotations

import pytest

from sar_geo_mcp.coords import is_near, parse_angle, parse_pair

# Roughly the El Corte assignments this system was built against.
NEAR = (37.42, -122.31)


# ---------------------------------------------------------------------------
# Single angles.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("37.4244", 37.4244),
    ("-122.3151", -122.3151),
    ("+37.4244", 37.4244),
    ("37.4244 N", 37.4244),
    ("122.3151 W", -122.3151),
    ("37.4244N", 37.4244),
])
def test_decimal_degrees(text, expected):
    assert parse_angle(text) == pytest.approx(expected)


@pytest.mark.parametrize("text,expected", [
    # What almost every SAR GPS is actually set to display.
    ("37 25.464", 37.4244),
    ("37° 25.464'", 37.4244),
    ("122 18.906 W", -122.3151),
    ("-122 18.906", -122.3151),
])
def test_degrees_decimal_minutes(text, expected):
    assert parse_angle(text) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("text,expected", [
    ("37 25 27.84", 37.4244),
    ("37°25'27.84\"", 37.4244),
    ("122 18 54.36 W", -122.3151),
])
def test_degrees_minutes_seconds(text, expected):
    assert parse_angle(text) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("text,expected", [
    # The letter before the number: how many GPS units display a point.
    ("N37.4244", 37.4244),
    ("W122.3151", -122.3151),
    ("w 122.3151", -122.3151),
    ("N37 25.464", 37.4244),
    ("N37° 25.464'", 37.4244),
    ("W 122° 18.906'", -122.3151),
    ("N37°25'27.84\"", 37.4244),
    ("S33.8688", -33.8688),
    ("E151.2093", 151.2093),
])
def test_a_hemisphere_letter_before_the_number(text, expected):
    assert parse_angle(text) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("text", [
    "N37.4244N",                     # a letter on both sides
    "N37.4244S",                     # and contradicting each other
    "W 122 18.906 W",
    "N",                             # a letter and no number
    "N37 61.5",                      # minutes >= 60, letter or not
])
def test_a_letter_on_both_sides_or_on_nothing_is_refused(text):
    assert parse_angle(text) is None


@pytest.mark.parametrize("text", [
    "",
    "the second creek crossing",
    "north of the saddle",
    "10S 0561234 4141234",          # UTM - deliberately unparsed, see module docs
    "37 61.5",                       # minutes >= 60
    "37 25 61.0",                    # seconds >= 60
    "about 37.4",                    # prose around a number
])
def test_anything_not_a_coordinate_returns_none(text):
    assert parse_angle(text) is None


def test_utm_is_refused_rather_than_misread():
    """A mis-stated UTM zone lands kilometres away with no outward sign of
    error, so it is recorded verbatim as words and never as a position."""
    assert parse_angle("10S 0561234 4141234") is None
    assert parse_pair("10S 0561234 4141234") is None


# ---------------------------------------------------------------------------
# Pairs.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "37.4244, -122.3151",
    "37.4244,-122.3151",
    "37.4244 / -122.3151",
    "37 25.464, 122 18.906 W",
])
def test_a_well_formed_pair_parses(text):
    latitude, longitude = parse_pair(text, near=NEAR)
    assert latitude == pytest.approx(37.4244, abs=1e-3)
    assert longitude == pytest.approx(-122.3151, abs=1e-3)


@pytest.mark.parametrize("text", [
    # Letters first, as a handheld displays it, with no comma between.
    "N37 25.464 W122 18.906",
    "N37°25.464' W122°18.906'",
    "N 37° 25.464' W 122° 18.906'",
    "N37°25'27.84\" W122°18'54.36\"",
    "N37.4244 W122.3151",
    "N37.4244, W122.3151",
    "W122.3151 N37.4244",            # longitude first, letters still say which
])
def test_a_pair_with_the_letters_first_parses(text):
    for near in (NEAR, None):
        latitude, longitude = parse_pair(text, near=near)
        assert latitude == pytest.approx(37.4244, abs=1e-3)
        assert longitude == pytest.approx(-122.3151, abs=1e-3)


@pytest.mark.parametrize("text", [
    "N37.4244N W122.3151",           # a letter on both sides of one value
    "N37.4244 122.3151",             # one letter, no comma: where does it split?
    "N37 W122 N38",                  # three values
])
def test_a_doubtful_letters_first_pair_is_refused(text):
    assert parse_pair(text, near=NEAR) is None


def test_a_transposed_pair_is_corrected_when_it_is_unambiguous():
    """Longitude first is how most GPS units display it and how nobody
    writes it down. Latitude cannot exceed 90, so this one is decidable
    without any reference point at all."""
    latitude, longitude = parse_pair("-122.3151, 37.4244")
    assert latitude == pytest.approx(37.4244)
    assert longitude == pytest.approx(-122.3151)


def test_a_missing_western_sign_is_only_inferred_against_a_reference():
    """A bare positive 122 in a California search is almost certainly a
    dropped minus - but "almost certainly" is not certainly, so the guess is
    only made when there is something to compare against."""
    with_reference = parse_pair("37.4244, 122.3151", near=NEAR)
    assert with_reference[1] == pytest.approx(-122.3151)

    without_reference = parse_pair("37.4244, 122.3151")
    assert without_reference[1] == pytest.approx(122.3151), (
        "the sign must not be invented from the number alone"
    )


def test_an_ambiguous_transposition_is_broken_only_by_the_reference():
    """Both values under 90, so nothing about the numbers themselves says
    which is which."""
    latitude, longitude = parse_pair("-77.0, 38.9", near=(38.9, -77.0))
    assert (latitude, longitude) == pytest.approx((38.9, -77.0))


@pytest.mark.parametrize("text", [
    "37.4244",                       # one value, not a pair
    "somewhere near the creek",
    "95.0, 200.0",                   # impossible as either
    "37.4244, 37.4244, 12.0",        # three values
])
def test_an_unreadable_pair_returns_none(text):
    assert parse_pair(text, near=NEAR) is None


def test_two_impossible_latitudes_are_refused_not_swapped():
    """Swapping only helps when exactly one value is impossible as a
    latitude. Two means this is not a lat/lon pair at all."""
    assert parse_pair("122.0, 130.0") is None


# ---------------------------------------------------------------------------
# Distance flag.
# ---------------------------------------------------------------------------

def test_a_point_in_the_assignment_is_not_flagged():
    assert is_near(37.4244, -122.3151, NEAR) is True


def test_a_point_far_away_is_flagged():
    """Not rejected - a team can legitimately point at a trailhead well
    outside their zone - but a transposed coordinate looks exactly like
    this, and the reader should see the doubt."""
    assert is_near(45.0, -122.31, NEAR) is False


def test_no_reference_means_nothing_is_flagged():
    assert is_near(0.0, 0.0, None) is True


def test_a_legitimate_far_point_is_kept_as_stated_not_corrected():
    """A team can name a trailhead or road end well outside their own zone.
    Quietly snapping that to something nearby would be inventing a location -
    the whole failure this module exists to avoid. Far-but-stated survives,
    and is flagged by is_near instead."""
    latitude, longitude = parse_pair("45.0, -122.31", near=NEAR)
    assert (latitude, longitude) == pytest.approx((45.0, -122.31))
    assert is_near(latitude, longitude, NEAR) is False


def test_a_sign_is_never_flipped_towards_a_reference_it_already_agrees_with():
    """The only flip on offer is the dropped western minus. A value already
    on the reference's side of zero is left alone."""
    latitude, longitude = parse_pair("37.60, -122.10", near=NEAR)
    assert latitude == pytest.approx(37.60)
    assert longitude == pytest.approx(-122.10)


def test_a_nearby_pair_is_never_rearranged():
    """The bias is towards what the searcher said: an alternative has to be
    an order of magnitude closer before it wins, so a point already in the
    assignment is returned untouched."""
    for text in ("37.43, -122.30", "37.41, -122.32"):
        latitude, longitude = parse_pair(text, near=NEAR)
        assert is_near(latitude, longitude, NEAR)
        assert latitude == pytest.approx(float(text.split(",")[0]))


# ---------------------------------------------------------------------------
# UTM and MGRS/USNG.
#
# These are what CalTopo shows by default for a California incident and what
# most teams read off a unit. An earlier version refused them outright on the
# grounds that a mis-stated zone lands kilometres away - a real risk, but
# refusing the format people actually use is a gap, not safety. The zone is
# derived from the assignment rather than trusted, and every decoding is
# checked against where it should have landed.
# ---------------------------------------------------------------------------

from sar_geo_mcp.coords import (  # noqa: E402
    parse_location,
    parse_mgrs,
    parse_utm,
)

# The same point in four notations. 10S is the Bay Area: zone 10, band S
# (32-40N). All four must land within a few metres of each other.
UTM_FORMS = [
    "10S 561234 4141234",
    "10 561234 4141234",
    "10S 561234mE 4141234mN",
    "10S 561234,4141234",
]
MGRS_FORMS = [
    "10S EG 61234 41234",
    "10SEG6123441234",
]


@pytest.mark.parametrize("text", UTM_FORMS)
def test_utm_parses_in_every_shape_a_team_writes_it(text):
    latitude, longitude = parse_utm(text, near=NEAR)
    assert latitude == pytest.approx(37.4159, abs=1e-3)
    assert longitude == pytest.approx(-122.3080, abs=1e-3)


@pytest.mark.parametrize("text", MGRS_FORMS)
def test_mgrs_parses_and_agrees_with_the_same_point_in_utm(text):
    """MGRS is UTM with the leading digits replaced by 100km square letters,
    so the two notations of one point must resolve identically."""
    from_mgrs = parse_mgrs(text, near=NEAR)
    from_utm = parse_utm("10S 561234 4141234", near=NEAR)
    assert from_mgrs == pytest.approx(from_utm, abs=1e-6)


def test_mgrs_precision_follows_the_digit_count():
    """A grid reference names a square, and fewer digits means a bigger one.
    A coarse reference must still land inside the fine one's square, not be
    offset to its south-west corner."""
    fine = parse_mgrs("10S EG 61234 41234", near=NEAR)
    coarse = parse_mgrs("10S EG 61 41", near=NEAR)
    assert coarse is not None
    # 2-digit precision is a 1km square, so within roughly 0.01 degrees.
    assert abs(coarse[0] - fine[0]) < 0.02
    assert abs(coarse[1] - fine[1]) < 0.02


def test_the_zone_letter_is_read_as_a_band_and_as_a_hemisphere():
    """`10S` is the latitude BAND (32-40N, northern) in a real grid
    reference, but plenty of people write it meaning "southern hemisphere".
    Both readings are tried and the assignment decides."""
    latitude, _ = parse_utm("10S 561234 4141234", near=NEAR)
    assert latitude > 0, "the band reading should win next to a northern assignment"


def test_a_grid_reference_that_decodes_far_away_is_refused_not_flagged():
    """Unlike a lat/lon - where a distant point may be a real trailhead - a
    grid reference landing far off is a decode error: wrong 100km square,
    mistyped zone, dropped digit. There is no honest way to keep it."""
    assert parse_utm("10S 561234 4141234", near=(45.0, -100.0)) is None
    assert parse_mgrs("10S EG 61234 41234", near=(45.0, -100.0)) is None


def test_a_grid_reference_still_parses_with_no_reference_at_all():
    """Full MGRS carries its own band letter, so it is globally
    unambiguous even with nothing to compare against."""
    assert parse_mgrs("10S EG 61234 41234") is not None
    assert parse_utm("10S 561234 4141234") is not None


@pytest.mark.parametrize("text", [
    "10S EG 6123 412",        # odd digit count - not a grid reference
    "99S EG 61234 41234",     # zone out of range
    "10I EG 61234 41234",     # I is never a band letter
    "10S IG 61234 41234",     # I is never a square letter
    "the second creek crossing",
])
def test_a_malformed_grid_reference_returns_none(text):
    assert parse_mgrs(text, near=NEAR) is None


# ---------------------------------------------------------------------------
# The one entry point.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", UTM_FORMS + MGRS_FORMS + [
    "37.4159, -122.3080",
    "37 24.95 N, 122 18.48 W",
    "N37 24.95 W122 18.48",
    "N 37° 24.95' W 122° 18.48'",
])
def test_parse_location_reads_every_supported_format(text):
    point = parse_location(text, near=NEAR)
    assert point is not None, f"{text!r} should have parsed"
    assert point[0] == pytest.approx(37.4159, abs=2e-3)
    assert point[1] == pytest.approx(-122.3080, abs=2e-3)


def test_parse_location_returns_none_for_a_description():
    """Which is a complete answer, not a failure - the searcher's words are
    kept and no position is invented."""
    assert parse_location("the second creek crossing", near=NEAR) is None
    assert parse_location("", near=NEAR) is None


# ---------------------------------------------------------------------------
# Formatting - the inverse of the parsers.
#
# Tested by round trip rather than against hand-worked examples: formatting
# and parsing share the same grid tables, so a value that formats and parses
# back to itself proves both halves agree. A hand-typed expected string only
# proves I typed it the same way twice.
# ---------------------------------------------------------------------------

from sar_geo_mcp.coords import (  # noqa: E402
    band_for,
    format_ddm,
    format_mgrs,
    format_utm,
    utm_zone_for,
)


@pytest.mark.parametrize("latitude,longitude", [
    (37.4159, -122.3080),      # El Corte
    (37.0, -122.0),
    (38.9, -77.0),             # a different zone entirely
    (-33.87, 151.21),          # southern hemisphere, eastern longitude
])
def test_every_format_parses_back_to_where_it_started(latitude, longitude):
    near = (latitude, longitude)
    for formatted in (format_utm(latitude, longitude),
                      format_mgrs(latitude, longitude),
                      format_ddm(latitude, longitude)):
        assert formatted is not None
        back = parse_location(formatted, near=near)
        assert back is not None, f"{formatted!r} did not parse back"
        assert back[0] == pytest.approx(latitude, abs=1e-3)
        assert back[1] == pytest.approx(longitude, abs=1e-3)


def test_the_bay_area_formats_as_zone_10_band_s():
    """The value CalTopo shows for this ground, so a wrong table would be
    obvious to anyone who has used the map."""
    assert utm_zone_for(-122.31) == 10
    assert band_for(37.42) == "S"
    assert format_utm(37.4159, -122.3080).startswith("10S ")
    assert format_mgrs(37.4159, -122.3080).startswith("10S EG ")


@pytest.mark.parametrize("digits,expected_len", [(5, 5), (4, 4), (3, 3), (2, 2), (1, 1)])
def test_mgrs_precision_follows_the_requested_digits(digits, expected_len):
    formatted = format_mgrs(37.4159, -122.3080, digits=digits)
    easting, northing = formatted.split()[-2:]
    assert len(easting) == len(northing) == expected_len


def test_a_coarse_grid_reference_names_the_square_the_point_is_in():
    """Truncated, not rounded: rounding up would name the neighbouring
    square, which is a kilometre away at 2-digit precision."""
    fine = parse_location(format_mgrs(37.4159, -122.3080, digits=5), near=NEAR)
    coarse = parse_location(format_mgrs(37.4159, -122.3080, digits=2), near=NEAR)
    assert abs(coarse[0] - fine[0]) < 0.02
    assert abs(coarse[1] - fine[1]) < 0.02


def test_the_polar_regions_have_no_utm_or_mgrs():
    """The grids do not apply above 84N or below 80S - null is the honest
    answer, not a number computed off the end of the table."""
    assert band_for(85.0) is None
    assert format_utm(85.0, 0.0) is None
    assert format_mgrs(-81.0, 0.0) is None


# ---------------------------------------------------------------------------
# Degrees, minutes and seconds out.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("latitude,longitude,expected", [
    (37.35687, -122.24093, "37°21'24.7\"N 122°14'27.3\"W"),
    (-33.8568, 151.2153, "33°51'24.5\"S 151°12'55.1\"E"),
    (0.0, 0.0, "0°00'00.0\"N 0°00'00.0\"E"),
    (5.5, -0.5, "5°30'00.0\"N 0°30'00.0\"W"),
])
def test_dms_writes_degrees_minutes_and_tenths_of_a_second(latitude, longitude, expected):
    from sar_geo_mcp.coords import format_dms

    assert format_dms(latitude, longitude) == expected


def test_dms_carries_a_rounded_sixty_seconds_into_the_minute():
    """59.95 seconds rounds to the next minute, never to "60.0"."""
    from sar_geo_mcp.coords import format_dms

    text = format_dms(37.0 + 59 / 60 + 59.96 / 3600, -(122.0 + 59 / 60 + 59.97 / 3600))
    assert text == "38°00'00.0\"N 123°00'00.0\"W"
    assert "60.0" not in text


@pytest.mark.parametrize("latitude,longitude", [
    (37.35687, -122.24093), (-33.8568, 151.2153), (64.1466, -21.9426),
])
def test_dms_parses_back_to_within_a_tenth_of_a_second(latitude, longitude):
    from sar_geo_mcp.coords import format_dms, parse_location

    back = parse_location(format_dms(latitude, longitude))
    assert back is not None
    assert back[0] == pytest.approx(latitude, abs=0.05 / 3600 + 1e-9)
    assert back[1] == pytest.approx(longitude, abs=0.05 / 3600 + 1e-9)
