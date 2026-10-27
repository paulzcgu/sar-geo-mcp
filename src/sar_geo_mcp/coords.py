"""Reading a coordinate a searcher said out loud, and refusing to guess.

WHY THIS IS CAREFUL
-------------------
A wrong coordinate on a filed SAR-110 is worse than no coordinate. "No
location recorded" sends a planner to ask someone; a plausible-looking but
transposed pair sends a team to the wrong drainage. So every function here
fails to `None` rather than to a best guess, and the caller records what the
searcher actually said either way.

Three things go wrong in practice and each is handled explicitly:

  Transposition. Someone reads "122.31, 37.42" - longitude first, the way
  most GPS units display it, and the way nobody writes it down. Latitude
  beyond +-90 is impossible, so a swapped pair is detectable whenever the
  longitude is over 90 degrees, which it is everywhere in California. When
  the swap is unambiguous this corrects it and says so; when it is not, it
  refuses.

  Sign. Western longitudes are negative. A searcher reading "122 18.9 west"
  off a unit means -122.315. A bare positive 122 in a US search is
  overwhelmingly a missing minus rather than a point in Xinjiang, but
  "overwhelmingly" is not "certainly", so the sign is only inferred when the
  caller supplies a reference point to compare against (the assignment's own
  centre) - never from the number alone.

  Format. Decimal degrees, degrees-decimal-minutes (what almost every SAR
  GPS is actually set to), degrees-minutes-seconds, UTM, and MGRS/USNG all
  appear, because all of them are things a CalTopo user or a handheld will
  show. `parse_location` tries each; anything it cannot read returns None
  and is recorded as words.

An earlier version of this module refused UTM outright, reasoning that a
mis-stated zone lands kilometres away with no outward sign of error. The
risk was real but the conclusion was wrong: refusing the format teams
actually use is not safety, it is just a gap. The answer is the same
reference point that resolves a transposed lat/lon - the zone is DERIVED
rather than trusted, and every grid decoding is checked against where it
should have landed. See `_pick_grid_candidate`, which refuses a far result
outright rather than flagging it: a grid reference far from the assignment
is not a distant landmark, it is a decode error.
"""

from __future__ import annotations

import math
import re

# Decimal degrees: "37.4244", "-122.3151"
_DECIMAL = r"[-+]?\d{1,3}(?:\.\d+)?"

# Degrees + decimal minutes: "37 25.464", "37° 25.464'"
_DDM = r"([-+]?\d{1,3})\s*[°d\s]\s*(\d{1,2}(?:\.\d+)?)\s*['m]?"

# Degrees minutes seconds: "37 25 27.8", "37°25'27.8\""
_DMS = r"([-+]?\d{1,3})\s*[°d\s]\s*(\d{1,2})\s*['m\s]\s*(\d{1,2}(?:\.\d+)?)\s*[\"s]?"

_HEMI = r"([NSEWnsew])"

# Longitude is never beyond this, latitude never beyond 90 - the two facts
# that make transposition detectable at all.
MAX_LATITUDE = 90.0
MAX_LONGITUDE = 180.0

# How far from the assignment a point may sit before it is flagged. Generous
# on purpose: a team can legitimately point at a trailhead well outside their
# zone, so this marks doubt rather than rejecting anything.
NEAR_ASSIGNMENT_DEGREES = 0.5


def _apply_hemisphere(value: float, hemisphere: str | None) -> float:
    if not hemisphere:
        return value
    if hemisphere.upper() in ("S", "W"):
        return -abs(value)
    return abs(value)


def parse_angle(text: str) -> float | None:
    """One angle in decimal degrees, or None when it is not a coordinate.

    Accepts decimal degrees, degrees-decimal-minutes, and DMS, each with an
    optional N/S/E/W letter after the number or before it - "W122°00.092'"
    is how many GPS units display a point. A letter on both sides is
    refused rather than one of them believed. Returns None rather than a
    guess for anything else - including UTM, MGRS, and free text.
    """
    if not text:
        return None
    cleaned = text.strip()

    prefix = re.match(rf"{_HEMI}\s*(?=[-+]?\d)", cleaned)
    if prefix:
        rest = cleaned[prefix.end():]
        if re.search(rf"\d\s*['\"ms]?\s*{_HEMI}$", rest):
            return None
        value = parse_angle(rest)
        return None if value is None else _apply_hemisphere(value, prefix.group(1))

    match = re.fullmatch(rf"{_DMS}\s*{_HEMI}?", cleaned)
    if match:
        degrees, minutes, seconds, hemisphere = match.groups()
        if float(minutes) >= 60 or float(seconds) >= 60:
            return None
        sign = -1.0 if degrees.startswith("-") else 1.0
        value = abs(float(degrees)) + float(minutes) / 60 + float(seconds) / 3600
        return _apply_hemisphere(sign * value, hemisphere)

    match = re.fullmatch(rf"{_DDM}\s*{_HEMI}?", cleaned)
    if match:
        degrees, minutes, hemisphere = match.groups()
        if float(minutes) >= 60:
            return None
        sign = -1.0 if degrees.startswith("-") else 1.0
        value = abs(float(degrees)) + float(minutes) / 60
        return _apply_hemisphere(sign * value, hemisphere)

    match = re.fullmatch(rf"({_DECIMAL})\s*{_HEMI}?", cleaned)
    if match:
        value, hemisphere = match.groups()
        return _apply_hemisphere(float(value), hemisphere)

    return None


def parse_pair(text: str, *, near: tuple[float, float] | None = None
               ) -> tuple[float, float] | None:
    """A latitude/longitude pair from one string, or None.

    `near` is a (latitude, longitude) the point is expected to be somewhere
    around - the assignment's own centre. It is used for exactly two things,
    both of which are guesses this function otherwise refuses to make:
    inferring a missing western sign, and deciding a transposed pair. Without
    it, both stay unresolved and the pair is rejected rather than filed
    wrong.
    """
    if not text:
        return None
    parts = [p for p in re.split(r"\s*[,;/]\s*|\s{2,}", text.strip()) if p]
    if len(parts) != 2:
        # One more shape worth trying: "37 25.464 N 122 18.906 W" with no
        # separator at all. Split on the hemisphere letters instead.
        # Letters first, "N37 30.000 W122 15.000", splits before the second
        # letter instead - tried first, since "N 37 ..." would otherwise
        # split off the bare "N".
        if re.match(r"[NSEWnsew]\s*[-+]?\d", text.strip()):
            split = re.split(r"\s+(?=[NSEWnsew]\s*[-+]?\d)", text.strip())
        else:
            split = re.split(r"(?<=[NnSs])\s+(?=\d|[-+])", text.strip())
        if len(split) == 2:
            parts = split
        else:
            return None

    first, second = (parse_angle(parts[0]), parse_angle(parts[1]))
    if first is None or second is None:
        return None

    latitude, longitude = first, second

    # Transposition: latitude cannot exceed 90, so a first value beyond that
    # is unambiguously the longitude.
    if abs(latitude) > MAX_LATITUDE:
        if abs(longitude) > MAX_LATITUDE:
            return None            # both impossible as latitude - not a pair
        latitude, longitude = longitude, latitude

    if abs(latitude) > MAX_LATITUDE or abs(longitude) > MAX_LONGITUDE:
        return None

    if near is None:
        return latitude, longitude

    # Order and sign can BOTH be wrong at once, so they are resolved together
    # rather than in sequence. Fixing signs first is actively harmful: on a
    # transposed pair it makes each value self-consistent with the wrong slot
    # and destroys the evidence that they were swapped.
    return _closest_reading(latitude, longitude, near)


def _sign_variants(value: float, reference: float) -> list[float]:
    """`value`, plus its negation when the sign looks dropped.

    Only ever offers the flip that would move a value onto the reference's
    own side of zero - the dropped minus on a western longitude, and nothing
    else. A negation is never offered on a number that already agrees in
    sign with where the assignment is.
    """
    if (value > 0 > reference) or (value < 0 < reference):
        return [value, -value]
    return [value]


def _closest_reading(latitude: float, longitude: float,
                     near: tuple[float, float]) -> tuple[float, float]:
    """Pick between "as stated" and its transposed/sign-flipped readings.

    Deliberately biased towards what the searcher actually said: an
    alternative has to be an order of magnitude closer to the assignment
    before it wins. A team can legitimately name somewhere well outside their
    own zone - a trailhead, a road end - and quietly "correcting" that into a
    nearby point would be inventing a location, which is the one thing this
    module exists to avoid. Far-but-stated survives, and is flagged instead
    (see PointOfInterest.outside_assignment).
    """
    as_stated = (latitude, longitude)
    distance_as_stated = math.dist(as_stated, near)

    candidates: set[tuple[float, float]] = set()
    for lat, lon in ((latitude, longitude), (longitude, latitude)):
        for lat_variant in _sign_variants(lat, near[0]):
            for lon_variant in _sign_variants(lon, near[1]):
                if abs(lat_variant) <= MAX_LATITUDE and abs(lon_variant) <= MAX_LONGITUDE:
                    candidates.add((lat_variant, lon_variant))
    candidates.discard(as_stated)
    if not candidates:
        return as_stated

    best = min(candidates, key=lambda point: math.dist(point, near))
    if math.dist(best, near) * 10 < distance_as_stated:
        return best
    return as_stated


def is_near(latitude: float, longitude: float,
            near: tuple[float, float] | None,
            tolerance: float = NEAR_ASSIGNMENT_DEGREES) -> bool:
    """Whether a point sits close enough to the assignment to be unremarkable.

    Used only to set `PointOfInterest.outside_assignment`, which is a note to
    the reader, not a rejection - see that field's own comment.
    """
    if near is None:
        return True
    return (abs(latitude - near[0]) <= tolerance
            and abs(longitude - near[1]) <= tolerance)


# ---------------------------------------------------------------------------
# UTM and MGRS/USNG.
#
# These are what CalTopo shows by default for a California incident and what
# most teams read off a unit, so refusing them - as an earlier version of
# this module did - refuses the format people actually use.
#
# The objection that made it look unsafe (a mis-stated zone lands kilometres
# away with no outward sign) is answered by the same reference point that
# resolves a transposed lat/lon: we know where the assignment is, so a zone
# can be DERIVED rather than trusted, and any conversion checked against
# where it should have landed. A decoding that disagrees with the reference
# is refused rather than filed.
# ---------------------------------------------------------------------------

# Latitude bands, 8 degrees each from -80. I and O are omitted throughout
# MGRS because they read as 1 and 0.
_BANDS = "CDEFGHJKLMNPQRSTUVWX"
_BAND_HEIGHT = 8.0
_BAND_BASE = -80.0

# MGRS 100km square letters. Columns run in three sets by zone; rows in two,
# offset by five for even zones. Both drop I and O.
_COLUMN_SETS = ("ABCDEFGH", "JKLMNPQR", "STUVWXYZ")
_ROW_LETTERS = "ABCDEFGHJKLMNPQRSTUV"
_ROW_CYCLE_M = 2_000_000
_SQUARE_M = 100_000

# How far a decoded grid reference may land from the assignment before it is
# refused outright. Unlike a lat/lon - where a far point is kept and flagged,
# because a team can legitimately name a distant trailhead - a grid reference
# landing this far off is not a distant landmark, it is a decode error: a
# wrong 100km square, a mistyped zone, a dropped digit. Half a degree is far
# wider than any assignment and far tighter than those failures.
GRID_SANITY_DEGREES = 0.5

_UTM_RE = re.compile(
    r"""^(?P<zone>[0-9]{1,2})\s*(?P<band>[C-HJ-NP-X])?\s*[,\s]\s*
        (?P<easting>\d{6,7})\s*(?:m?E)?\s*[,\s]\s*
        (?P<northing>\d{6,8})\s*(?:m?N)?$""",
    re.IGNORECASE | re.VERBOSE,
)

_MGRS_RE = re.compile(
    r"""^(?P<zone>[0-9]{1,2})\s*(?P<band>[C-HJ-NP-X])\s*
        (?P<square>[A-HJ-NP-Z]{2})\s*
        (?P<digits>[\d\s]{2,12})$""",
    re.IGNORECASE | re.VERBOSE,
)


def _band_latitude(band: str) -> float | None:
    """The middle latitude of a band letter - enough to pin a UTM northing
    cycle without pretending to more precision than a letter carries."""
    index = _BANDS.find(band.upper())
    if index < 0:
        return None
    return _BAND_BASE + index * _BAND_HEIGHT + _BAND_HEIGHT / 2


def _utm_to_wgs84(zone: int, northern: bool,
                  easting: float, northing: float) -> tuple[float, float] | None:
    """One UTM position as (latitude, longitude), or None if it is not one."""
    if not 1 <= zone <= 60:
        return None
    epsg = (32600 if northern else 32700) + zone
    try:
        from pyproj import Transformer

        transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326",
                                           always_xy=True)
        longitude, latitude = transformer.transform(easting, northing)
    except Exception:
        return None
    if not (math.isfinite(latitude) and math.isfinite(longitude)):
        return None
    if abs(latitude) > MAX_LATITUDE or abs(longitude) > MAX_LONGITUDE:
        return None
    return latitude, longitude


def _pick_grid_candidate(candidates, near):
    """Choose between decodings of one grid reference, or refuse.

    A grid reference that lands far from the assignment is not a distant
    landmark the way a lat/lon can be - it is a decode error. So unlike
    `parse_pair`, this REFUSES a far result rather than keeping and flagging
    it. The searcher's words survive either way; only the false precision is
    dropped.
    """
    if not candidates:
        return None
    if near is None:
        return candidates[0]
    best = min(candidates, key=lambda point: math.dist(point, near))
    if not is_near(best[0], best[1], near, GRID_SANITY_DEGREES):
        return None
    return best


def parse_utm(text: str, *, near: tuple[float, float] | None = None):
    """A UTM grid reference, or None.

    `10S 561234 4141234`, `10 561234 4141234`, `10S 561234mE 4141234mN`.

    The letter after the zone is genuinely ambiguous: in `10S` it is the
    latitude BAND (32-40N, which is northern), but plenty of people write
    `10N`/`10S` meaning the hemisphere. Both readings are tried and the one
    nearest the assignment wins; with no reference the band reading is
    preferred, since that is what a grid reference actually means.
    """
    match = _UTM_RE.match(text.strip())
    if not match:
        return None
    zone = int(match.group("zone"))
    band = (match.group("band") or "").upper()
    easting = float(match.group("easting"))
    northing = float(match.group("northing"))

    candidates = []
    if band:
        band_latitude = _band_latitude(band)
        if band_latitude is not None:
            point = _utm_to_wgs84(zone, band_latitude >= 0, easting, northing)
            if point:
                candidates.append(point)
        # The hemisphere-letter reading, for N and S only - no other letter
        # is ever used that way.
        if band in ("N", "S"):
            point = _utm_to_wgs84(zone, band == "N", easting, northing)
            if point and point not in candidates:
                candidates.append(point)
    else:
        # No letter at all, so the hemisphere comes from the reference.
        northern = near is None or near[0] >= 0
        point = _utm_to_wgs84(zone, northern, easting, northing)
        if point:
            candidates.append(point)

    return _pick_grid_candidate(candidates, near)


def parse_mgrs(text: str, *, near: tuple[float, float] | None = None):
    """An MGRS / USNG grid reference, or None.

    `10S EG 61234 41234`, `10SEG6123441234`, `10S EG 6123 4123`. Precision
    follows the digit count exactly as the standard defines it: ten digits
    is one metre, eight is ten metres, and so on.
    """
    match = _MGRS_RE.match(text.strip())
    if not match:
        return None
    zone = int(match.group("zone"))
    band = match.group("band").upper()
    square = match.group("square").upper()
    digits = re.sub(r"\s+", "", match.group("digits"))
    if not 1 <= zone <= 60 or not digits or len(digits) % 2:
        return None

    half = len(digits) // 2
    if half > 5:
        return None
    # A grid reference names a SQUARE, and its size is set by how many digits
    # were given. Scaling by the missing digits keeps a coarse reference from
    # reading as a precise one offset to the south-west.
    precision = 10 ** (5 - half)
    easting_digits = int(digits[:half]) * precision
    northing_digits = int(digits[half:]) * precision

    column_set = _COLUMN_SETS[(zone - 1) % 3]
    column_index = column_set.find(square[0])
    if column_index < 0:
        return None
    easting = (column_index + 1) * _SQUARE_M + easting_digits

    row_offset = 0 if zone % 2 else 5     # even zones start five letters in
    row_index = _ROW_LETTERS.find(square[1])
    if row_index < 0:
        return None
    row_northing = ((row_index - row_offset) % len(_ROW_LETTERS)) * _SQUARE_M

    band_latitude = _band_latitude(band)
    if band_latitude is None:
        return None
    northern = band_latitude >= 0

    # Row letters repeat every 2,000,000 m, so the band letter says which
    # cycle this is. Try each and keep the ones landing inside that band.
    candidates = []
    for cycle in range(0, 11):
        northing = row_northing + cycle * _ROW_CYCLE_M + northing_digits
        point = _utm_to_wgs84(zone, northern, easting, northing)
        if point is None:
            continue
        if abs(point[0] - band_latitude) <= _BAND_HEIGHT:
            candidates.append(point)

    return _pick_grid_candidate(candidates, near)


def parse_location(text: str, *, near: tuple[float, float] | None = None
                   ) -> tuple[float, float] | None:
    """Any coordinate format a searcher might give, or None.

    The single entry point callers should use. Grid formats are tried before
    lat/lon because they are the more distinctive shapes - "10S EG 61234
    41234" is unmistakably MGRS, whereas a bare pair of numbers could be
    several things - and because getting the order wrong would let a partial
    match on the looser format win over an exact one on the tighter.

    None means "this is not a coordinate", which for this system is a
    complete answer: the searcher's own words are kept either way, and no
    position is invented.
    """
    if not text or not text.strip():
        return None
    for parse in (parse_mgrs, parse_utm):
        point = parse(text, near=near)
        if point is not None:
            return point
    return parse_pair(text, near=near)


# ---------------------------------------------------------------------------
# Formatting - the inverse of the parsers above.
#
# A planner reading a lat/lon off a map often has to hand a team the same
# point in the format their unit is set to. These reuse the exact tables the
# parsers use, so a value that formats here parses back to itself; the tests
# assert that round trip rather than trusting hand-worked examples.
# ---------------------------------------------------------------------------


def utm_zone_for(longitude: float) -> int:
    """The UTM zone a longitude falls in.

    The Norway/Svalbard exceptions (zones 32 and 31X-37X are widened or
    split) are deliberately NOT implemented: this is a California SAR tool,
    the exceptions apply above 56N, and a half-remembered special case is
    worse than a documented absence. Anything up there should be checked
    against a real grid.
    """
    return int((longitude + 180) / 6) % 60 + 1


def band_for(latitude: float) -> str | None:
    """The MGRS latitude band letter, or None outside the banded range."""
    if not -80 <= latitude < 84:
        return None
    index = min(int((latitude - _BAND_BASE) / _BAND_HEIGHT), len(_BANDS) - 1)
    return _BANDS[index]


def _to_utm(latitude: float, longitude: float):
    """(zone, band, easting, northing), or None outside the banded range."""
    band = band_for(latitude)
    if band is None:
        return None
    zone = utm_zone_for(longitude)
    epsg = (32600 if latitude >= 0 else 32700) + zone
    try:
        from pyproj import Transformer

        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}",
                                           always_xy=True)
        easting, northing = transformer.transform(longitude, latitude)
    except Exception:
        return None
    if not (math.isfinite(easting) and math.isfinite(northing)):
        return None
    return zone, band, easting, northing


def format_utm(latitude: float, longitude: float) -> str | None:
    """`10S 561234 4141234` - metre precision, which is past what any
    handheld resolves, so no false precision is implied by more digits."""
    utm = _to_utm(latitude, longitude)
    if utm is None:
        return None
    zone, band, easting, northing = utm
    return f"{zone}{band} {round(easting)} {round(northing)}"


def format_mgrs(latitude: float, longitude: float, digits: int = 5) -> str | None:
    """`10S EG 61234 41234` - USNG, as CalTopo labels it.

    `digits` is per axis: 5 is one metre, 4 is ten, down to 1 for a 10km
    square. Truncated rather than rounded, because a grid reference names
    the square a point is IN - rounding up would name the neighbouring one.
    """
    if not 1 <= digits <= 5:
        return None
    utm = _to_utm(latitude, longitude)
    if utm is None:
        return None
    zone, band, easting, northing = utm

    column_set = _COLUMN_SETS[(zone - 1) % 3]
    column_index = int(easting // _SQUARE_M) - 1
    if not 0 <= column_index < len(column_set):
        return None
    column = column_set[column_index]

    row_offset = 0 if zone % 2 else 5
    row_index = (int(northing // _SQUARE_M) + row_offset) % len(_ROW_LETTERS)
    row = _ROW_LETTERS[row_index]

    scale = 10 ** (5 - digits)
    easting_digits = int((easting % _SQUARE_M) // scale)
    northing_digits = int((northing % _SQUARE_M) // scale)
    return (f"{zone}{band} {column}{row} "
            f"{easting_digits:0{digits}d} {northing_digits:0{digits}d}")


def format_ddm(latitude: float, longitude: float) -> str:
    """`37 25.464 N, 122 18.906 W` - what most SAR handhelds display."""

    def one(value: float, positive: str, negative: str) -> str:
        hemisphere = positive if value >= 0 else negative
        value = abs(value)
        degrees = int(value)
        minutes = (value - degrees) * 60
        return f"{degrees} {minutes:.3f} {hemisphere}"

    return f"{one(latitude, 'N', 'S')}, {one(longitude, 'E', 'W')}"


def format_dms(latitude: float, longitude: float) -> str:
    """`37°25'27.8"N 122°18'54.4"W` - degrees, minutes and seconds, as a
    paper map margin, a dispatch log or a phone's compass app writes them.

    A tenth of a second is about 3 m, the same order as `format_ddm`'s
    thousandth of a minute (1.85 m). Rounding is done on the whole value in
    tenths of a second, so 59.95" carries into the next minute rather than
    printing as 60.0". `parse_location` reads this string back.
    """

    def one(value: float, positive: str, negative: str) -> str:
        hemisphere = positive if value >= 0 else negative
        tenths = round(abs(value) * 36000)
        degrees, tenths = divmod(tenths, 36000)
        minutes, tenths = divmod(tenths, 600)
        return f"{degrees}°{minutes:02d}'{tenths / 10:04.1f}\"{hemisphere}"

    return f"{one(latitude, 'N', 'S')} {one(longitude, 'E', 'W')}"
