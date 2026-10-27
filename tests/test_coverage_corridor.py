"""Corridor coverage: line-assignment math (see coverage.corridor module docstring).

37.36 deg N is a real assignment latitude (QC1202, reused from
test_coverage_mercator.py) - cos(37.36 deg) ~= 0.7949, so a geometry whose
Web Mercator measurement is 1000 units must report ~795 ground units, not
1000. This project has shipped the Web Mercator units bug twice before, so
every module in this package that measures ground distance carries a test
that would fail if the ground_factor conversion were silently dropped.
"""

import pytest
from shapely.geometry import LineString

from sar_geo_mcp.coverage.corridor import compute_corridor_coverage
from sar_geo_mcp.geo import ground_meters_per_mercator_unit

LATITUDE_DEG = 37.36


def test_middle_of_the_line_covered_by_a_parallel_track():
    # A 1000 m straight assignment line; a track runs parallel 3 m away for
    # the middle 600 m only. At corridor width 10 (half-width 5), that 3 m
    # offset is well inside the corridor, so the middle ~60% is covered and
    # the two flanking stretches are not.
    line = LineString([(0, 0), (1000, 0)])
    track = LineString([(200, 3), (800, 3)])

    coverage = compute_corridor_coverage(
        line, [track], half_width_ground_m=5.0, ground_factor=1.0
    )

    assert coverage.covered_fraction == pytest.approx(0.6, abs=0.03)
    assert coverage.total_length_m == pytest.approx(1000.0)
    assert len(coverage.gaps) == 2
    # Largest first - both flanking gaps are the same length here, but the
    # ordering contract must hold regardless.
    assert coverage.gaps[0].length_m >= coverage.gaps[1].length_m
    for gap in coverage.gaps:
        assert gap.length_m == pytest.approx(200.0, abs=15.0)
        assert gap.geometry["type"] == "LineString"


def test_narrower_corridor_puts_the_same_track_entirely_outside():
    # Same geometry, but corridor width 4 (half-width 2): the track's 3 m
    # offset now exceeds the half-width, so nothing is covered at all.
    line = LineString([(0, 0), (1000, 0)])
    track = LineString([(200, 3), (800, 3)])

    coverage = compute_corridor_coverage(
        line, [track], half_width_ground_m=2.0, ground_factor=1.0
    )

    assert coverage.covered_fraction == 0.0
    assert len(coverage.gaps) == 1
    assert coverage.gaps[0].length_m == pytest.approx(1000.0, abs=1.0)


def test_a_short_uncovered_stub_is_dropped_but_still_counts_against_coverage():
    # The track itself has a 10 m physical break (x in [95, 105]). At
    # corridor half-width 2, only a few metres around the break's midpoint
    # actually fall outside the corridor - well under MIN_GAP_LENGTH_M (25 m).
    # It must not be reported as a gap (GPS-wobble scale noise), but it must
    # still shave a little off covered_fraction rather than being invented
    # away as covered.
    line = LineString([(0, 0), (200, 0)])
    tracks = [LineString([(0, 0), (95, 0)]), LineString([(105, 0), (200, 0)])]

    coverage = compute_corridor_coverage(
        line, tracks, half_width_ground_m=2.0, ground_factor=1.0
    )

    assert coverage.gaps == []
    assert coverage.covered_fraction < 1.0
    assert coverage.covered_fraction > 0.9


def test_no_tracks_leaves_the_whole_line_as_one_gap():
    line = LineString([(0, 0), (100, 0)])

    coverage = compute_corridor_coverage(
        line, [], half_width_ground_m=5.0, ground_factor=1.0
    )

    assert coverage.covered_fraction == 0.0
    assert len(coverage.gaps) == 1
    assert coverage.gaps[0].length_m == pytest.approx(100.0, abs=1.0)


def test_ground_factor_mutation_pin_at_a_real_latitude():
    factor = ground_meters_per_mercator_unit(LATITUDE_DEG)
    assert factor == pytest.approx(0.7949, abs=0.0005)

    # --- distance threshold: a track offset by 4.5 ground metres must be
    # judged against a 5 m ground half-width using GROUND distance, not raw
    # Mercator distance. In Mercator units that offset is ~5.66 units - if
    # the ground_factor conversion were dropped when comparing against the
    # half-width, 5.66 > 5.0 would (wrongly) mark the whole line uncovered.
    half_width_ground_m = 5.0
    ground_offset_m = 4.5
    mercator_offset = ground_offset_m / factor
    assert mercator_offset > half_width_ground_m  # confirms the raw value would flip the verdict

    length_m = 200.0
    length_mercator = length_m / factor
    line = LineString([(0, 0), (length_mercator, 0)])
    track = LineString([(0, mercator_offset), (length_mercator, mercator_offset)])

    coverage = compute_corridor_coverage(
        line, [track], half_width_ground_m=half_width_ground_m, ground_factor=factor
    )

    assert coverage.covered_fraction == pytest.approx(1.0)
    assert coverage.gaps == []

    # --- total_length_m: Mercator length x ground_factor must equal the
    # ground length exactly (by construction, length_mercator = length_m /
    # factor). Mutant: `total_length_mercator * ground_factor` with the
    # multiplication dropped would report length_mercator (~251.6), not 200.
    assert coverage.total_length_m == pytest.approx(200.0, rel=1e-9)
    assert coverage.total_length_m != pytest.approx(length_mercator, abs=1.0)


# --- GPS-noise gap closing (Phase 9i) --------------------------------------
#
# Measured on QB102 (6856 m route, corridor half-width 5.3 m): four of five
# reported gaps were GPS wobble (a track that never left the corridor by
# more than ~15 m, over stretches of 26-40 m with covered route on both
# sides) and one (gap_0, 3952 m) was genuinely never entered. All the
# fixtures below use half_width_ground_m=5.0 (sweep_width=10.0) so the
# GAP_CLOSE_MAX_LENGTH_FACTOR=5.0 ceiling is 50 m and the
# GAP_CLOSE_MAX_LATERAL_FACTOR=3.0 ceiling is 15 m - the same proportions as
# the real data.
#
# Every fixture below builds a gap from a "hugging" track (offset 0.5 m,
# well inside the corridor) that stops short, an offset "detour" track
# across the gap, then hugging resumes. A detour offset has to clear the
# half-width (5 m) or nothing is uncovered at all; a hugging track's own
# endpoint also stays within half-width of route positions near it, which
# eats a few (~5, sqrt(half_width^2 - 0.5^2)) metres of nominal gap length
# off EACH side once corner-rounding is accounted for, and 2 m sample
# quantisation can shift that by a further couple of metres depending on
# where the endpoint's x falls on the sampling grid. Every detour below is
# built wide (44 m) so the realised gap lands comfortably clear of both
# MIN_GAP_LENGTH_M (25 m, or it would not even reach the closing test) and
# GAP_CLOSE_MAX_LENGTH_FACTOR's 50 m ceiling regardless of that jitter -
# these tests care about which side of the closing rule a gap falls on, not
# about reproducing QB102's exact metres.

HALF_WIDTH = 5.0  # sweep_width 10.0 -> length ceiling 50 m, lateral ceiling 15 m
DETOUR_WIDTH = 44.0  # realises as a ~30-38 m gap after corner-rounding - see above


def _hugging_track(x0: float, x1: float, y: float = 0.5) -> LineString:
    """A track segment that hugs the route (y=0) well inside the corridor."""
    return LineString([(x0, y), (x1, y)])


def _detour_track(x0: float, x1: float, y: float) -> LineString:
    """A track segment offset `y` ground metres from the route."""
    return LineString([(x0, y), (x1, y)])


def test_interior_gap_with_track_7m_off_is_closed():
    # A ~30-38 m interior gap where the track detours 7 m off the route
    # (past the 5 m half-width, so genuinely uncovered) instead of
    # vanishing - GPS jitter while still walking the line, not a real miss.
    x1 = 237 + DETOUR_WIDTH
    line = LineString([(0, 0), (500, 0)])
    tracks = [
        _hugging_track(0, 237),
        _detour_track(237, x1, y=7.0),
        _hugging_track(x1, 500),
    ]

    coverage = compute_corridor_coverage(line, tracks, half_width_ground_m=HALF_WIDTH)

    assert coverage.gaps == []
    assert len(coverage.closed_gaps) == 1
    closed = coverage.closed_gaps[0]
    assert 25.0 < closed.length_m < 50.0
    assert closed.max_lateral_m == pytest.approx(7.0, abs=0.5)
    assert closed.geometry["type"] == "LineString"
    # The closed stretch was actually walked - covered_fraction must credit
    # it, not just quietly drop it from `gaps`.
    assert coverage.covered_fraction == pytest.approx(1.0, abs=0.02)


def test_interior_gap_with_track_20m_off_is_kept():
    # Same position, but the track wanders 20 m off - past the 15 m lateral
    # ceiling. This is a real detour (around an obstacle, say), not sensor
    # noise, so it must stay a reported gap.
    x1 = 237 + DETOUR_WIDTH
    line = LineString([(0, 0), (500, 0)])
    tracks = [
        _hugging_track(0, 237),
        _detour_track(237, x1, y=20.0),
        _hugging_track(x1, 500),
    ]

    coverage = compute_corridor_coverage(line, tracks, half_width_ground_m=HALF_WIDTH)

    assert coverage.closed_gaps == []
    assert len(coverage.gaps) == 1
    assert 25.0 < coverage.gaps[0].length_m < 50.0


def test_gap_touching_the_route_end_is_kept_even_when_short_and_close():
    # Same short length and same 7 m lateral offset as the closed case
    # above, but the gap runs right up to the end of the route - nothing
    # covered beyond it. This is exactly what protects a genuinely
    # unsearched tail: an end gap is never closed, no matter how it scores
    # on length or lateral distance.
    x1 = 237 + DETOUR_WIDTH
    line = LineString([(0, 0), (x1, 0)])
    tracks = [
        _hugging_track(0, 237),
        _detour_track(237, x1, y=7.0),
    ]

    coverage = compute_corridor_coverage(line, tracks, half_width_ground_m=HALF_WIDTH)

    assert coverage.closed_gaps == []
    assert len(coverage.gaps) == 1
    assert coverage.gaps[0].length_m > 25.0


def test_long_interior_gap_with_small_lateral_offset_is_kept():
    # ~250 m interior gap, only 7 m off - passes "interior" and "lateral"
    # but is far too long (ceiling 50 m) to be GPS noise: a real 250 m
    # stretch cannot be explained away as a few metres of sensor drift.
    d = 260.0
    x1 = 250 + d
    line = LineString([(0, 0), (x1 + 250, 0)])
    tracks = [
        _hugging_track(0, 250),
        _detour_track(250, x1, y=7.0),
        _hugging_track(x1, x1 + 250),
    ]

    coverage = compute_corridor_coverage(line, tracks, half_width_ground_m=HALF_WIDTH)

    assert coverage.closed_gaps == []
    assert len(coverage.gaps) == 1
    assert coverage.gaps[0].length_m > 100.0


def test_closed_gap_fields_and_ids_do_not_collide_with_open_gap_ids():
    # Two independent closeable gaps plus one that stays open - closed_gaps
    # and gaps are disjoint id spaces built from disjoint candidate sets,
    # not filtered after the fact from a single shared id sequence.
    d = DETOUR_WIDTH
    line = LineString([(0, 0), (1200, 0)])
    tracks = [
        _hugging_track(0, 237),
        _detour_track(237, 237 + d, y=7.0),           # closes
        _hugging_track(237 + d, 600),
        _detour_track(600, 600 + d, y=20.0),           # stays open (too far off)
        _hugging_track(600 + d, 700),
        _detour_track(700, 700 + d, y=7.0),           # closes
        _hugging_track(700 + d, 1200),
    ]

    coverage = compute_corridor_coverage(line, tracks, half_width_ground_m=HALF_WIDTH)

    assert len(coverage.gaps) == 1
    assert len(coverage.closed_gaps) == 2
    assert 25.0 < coverage.gaps[0].length_m < 50.0
    for closed in coverage.closed_gaps:
        assert 25.0 < closed.length_m < 50.0
        assert closed.max_lateral_m == pytest.approx(7.0, abs=0.5)


def test_lateral_track_geometries_rescues_the_closing_test_from_a_tight_clip():
    # A real caller (metrics._compute_line_coverage) clips its track set
    # tight to the corridor (exactly half_width_ground_m) before this
    # function ever sees it, for its own reasons (see that module - it
    # needs `length_in_zone_m` to mean "length inside the corridor"). That
    # clip removes the very detour points the lateral test needs: the
    # nearest SURVIVING clipped point for a mid-gap sample ends up being
    # tens of metres away along-route (right at the clip boundary near the
    # gap's edge), not a few metres off across it - inflating max_lateral
    # well past the ceiling for a gap that would otherwise close, and this
    # is exactly what happened on real QB102 data before the fix (only 2 of
    # 4 real GPS-noise gaps closed). `lateral_track_geometries` lets a
    # caller hand over a more generously clipped (or full) track set for
    # the lateral measurement ONLY, without changing what counts as
    # covered/uncovered.
    line = LineString([(0, 0), (500, 0)])
    track = LineString([
        (0, 0.5), (237, 0.5), (237, 7.0), (281, 7.0), (281, 0.5), (500, 0.5),
    ])
    corridor_zone = line.buffer(HALF_WIDTH)
    tightly_clipped = track.intersection(corridor_zone)

    narrow_only = compute_corridor_coverage(
        line, [tightly_clipped], half_width_ground_m=HALF_WIDTH
    )
    assert narrow_only.closed_gaps == []
    assert len(narrow_only.gaps) == 1  # stays open: the clip starved the lateral test

    with_lateral = compute_corridor_coverage(
        line, [tightly_clipped], half_width_ground_m=HALF_WIDTH,
        lateral_track_geometries=[track],
    )
    assert with_lateral.gaps == []
    assert len(with_lateral.closed_gaps) == 1
    assert with_lateral.closed_gaps[0].max_lateral_m == pytest.approx(7.0, abs=0.5)
