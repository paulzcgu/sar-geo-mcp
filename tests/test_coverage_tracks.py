import pytest

from sar_geo_mcp.coverage.tracks import normalise_label, matches_team


def test_normalise_strips_separators_and_case():
    assert normalise_label("1215.2 xq") == "12152xq"
    assert normalise_label("1215-2ZZ") == "12152zz"
    assert normalise_label("1215ZZ") == "1215zz"
    assert normalise_label("  1211.2 QQ  ") == "12112qq"


@pytest.mark.parametrize("label", [
    "102_XY", "102.NG", "102 ng", "xy 102",
])
def test_matches_team_accepts_every_operator_named_variant(label):
    # Teams provide only their number; searchers write it with any
    # separator and either order (spec SS5.1 amended 2026-08-17).
    assert matches_team(label, "102") is True


def test_matches_team_rejects_a_different_team():
    assert matches_team("1215.1 CB", "1211") is False


@pytest.mark.parametrize("label", ["QX106", "106DS"])
def test_matches_team_matches_the_team_number_in_either_order(label):
    # Real label on the training map: initials first, team number after.
    assert matches_team(label, "106") is True


@pytest.mark.parametrize("label", ["1024", "3102", "21021"])
def test_matches_team_rejects_a_longer_digit_run(label):
    # The team id must appear as a MAXIMAL digit run: "1024" and "3102"
    # continue the digit run beyond "102" - a different team, not team 102
    # with a stray extra digit.
    assert matches_team(label, "102") is False


def test_matches_team_matches_a_team_number_mid_label_too():
    # "QA1211" is an assignment label that contains 1211 as a standalone
    # digit run - matches_team on its own can't (and per spec shouldn't)
    # tell that apart from a real track; select_tracks's class whitelist is
    # what excludes assignment labels before they ever reach matches_team
    # (see test_select_excludes_assignment_class_features_even_when_the_label_matches).
    assert matches_team("QA1211", "1211") is True


from shapely.geometry import Polygon, mapping

from sar_geo_mcp.coverage.tracks import (
    SelectedTrack,
    TrackSelectionError,
    line_from_feature,
    select_tracks,
)


def feature(title, coords, cls="Shape"):
    return {"properties": {"title": title, "class": cls},
            "geometry": {"type": "LineString", "coordinates": coords}}


def square(x0, y0, size=10.0):
    return Polygon([(x0, y0), (x0 + size, y0), (x0 + size, y0 + size),
                    (x0, y0 + size)])


def test_line_from_feature_strips_elevation_and_timestamp():
    # Real CalTopo tracks are [lon, lat, elevation_m, timestamp_ms].
    line = line_from_feature(feature("t", [[0, 0, 100, 1786811471289],
                                           [1, 1, 110, 1786811481289]]))

    assert list(line.coords) == [(0.0, 0.0), (1.0, 1.0)]


def test_line_from_feature_returns_none_for_a_degenerate_line():
    assert line_from_feature(feature("t", [[0, 0, 1, 2]])) is None


def test_select_keeps_only_tracks_intersecting_the_assignment():
    zone = square(0, 0)
    inside = feature("1211.1 AA", [[1, 1, 0, 0], [9, 9, 0, 0]])
    elsewhere = feature("1211.2 AA", [[50, 50, 0, 0], [59, 59, 0, 0]])

    selected = select_tracks([inside, elsewhere], zone, ["1211"])

    assert [t.name for t in selected] == ["1211.1 AA"]


def test_select_keeps_a_track_named_initials_first():
    # Real observed label on the training map: "QX106" for team 106. Prefix
    # matching silently dropped this sortie; substring matching keeps it.
    zone = square(0, 0)
    inside = feature("QX106", [[1, 1, 0, 0], [9, 9, 0, 0]])

    selected = select_tracks([inside], zone, ["106"])

    assert [t.name for t in selected] == ["QX106"]


def test_select_excludes_assignment_class_features_even_when_the_label_matches():
    # "QA1211" is an assignment label that happens to contain "1211" - as a
    # bare string it would now match team 1211 under substring matching.
    # What actually excludes it is select_tracks's class whitelist: an
    # assignment feature is class "Assignment", which is not in
    # TRACK_CLASSES, so it never reaches matches_team at all.
    zone = square(0, 0)
    assignment_labelled_like_a_track = feature(
        "QA1211", [[1, 1, 0, 0], [9, 9, 0, 0]], cls="Assignment"
    )

    with pytest.raises(TrackSelectionError, match="1211"):
        select_tracks([assignment_labelled_like_a_track], zone, ["1211"])


def test_select_keeps_several_tracks_sharing_one_name():
    # Same searcher, two sorties, same label - both are real coverage.
    zone = square(0, 0)
    features = [feature("1211.1 AA", [[1, 1, 0, 0], [4, 4, 0, 0]]),
                feature("1211.1 AA", [[5, 5, 0, 0], [9, 9, 0, 0]])]

    assert len(select_tracks(features, zone, ["1211"])) == 2


def test_select_collapses_the_same_sortie_recorded_in_two_containers():
    # Real case on the training map: "105XB" appears as class Shape and class
    # AppTrack, identical geometry (hausdorff 0.0, length ratio 1.0) - the
    # same recorded sortie, exported twice. Counted as two tracks,
    # effort_coverage (which sums length) would double for it.
    zone = square(0, 0, size=50.0)
    same_points = [[1, 1, 0, 0], [10, 15, 0, 0], [25, 5, 0, 0]]
    shape_export = feature("105XB", same_points, cls="Shape")
    app_track_export = feature("105XB", same_points, cls="AppTrack")

    selected = select_tracks([shape_export, app_track_export], zone, ["105"])

    assert len(selected) == 1
    assert selected[0].name == "105XB"


def test_select_keeps_a_same_named_track_of_very_different_length():
    # Real case on the training map: "105XJ" also appears as Shape and AppTrack,
    # but 135 vs 2988 vertices, a 72x length ratio - genuinely different
    # things (not the same sortie re-exported), must not collapse.
    zone = square(0, 0, size=200.0)
    short = feature("105XJ", [[1, 1, 0, 0], [2, 2, 0, 0]])
    long = feature("105XJ", [[1, 1, 0, 0], [150, 1, 0, 0]])

    selected = select_tracks([short, long], zone, ["105"])

    assert len(selected) == 2


def test_select_keeps_two_real_sorties_of_similar_length_but_different_route():
    # Real case on the training map: "103XA" legitimately appears twice - a
    # morning and an evening sortie. A length-ratio check alone could not
    # tell that apart from a duplicate export of the same sortie, since two
    # real sorties can happen to run similar distances; the Hausdorff
    # distance (large here - the routes don't overlap) is what keeps them
    # separate.
    zone = square(0, 0, size=50.0)
    morning = feature("103XA", [[1, 1, 0, 0], [1, 21, 0, 0]])   # north, length 20
    evening = feature("103XA", [[1, 1, 0, 0], [21, 1, 0, 0]])   # east, length 20

    selected = select_tracks([morning, evening], zone, ["103"])

    assert len(selected) == 2


def test_select_records_clipped_and_total_length():
    zone = square(0, 0)
    # Runs from inside the zone out well past its edge.
    crossing = feature("1211.1 AA", [[5, 5, 0, 0], [5, 105, 0, 0]])

    track = select_tracks([crossing], zone, ["1211"])[0]

    assert track.length_in_zone_m == pytest.approx(5.0, abs=0.1)
    assert track.length_total_m == pytest.approx(100.0, abs=0.1)


def test_select_includes_a_track_that_only_grazes_the_zone():
    # Observed on real data: 8 m of a 3250 m track fell inside the zone. No
    # transit classification (operator decision, spec SS5.2 amended
    # 2026-08-17): the graze is included and contributes its real, if tiny,
    # length - nothing is excluded on geometry alone.
    zone = square(0, 0)
    graze = feature("1211.1 AA", [[9.5, 5, 0, 0], [400, 5, 0, 0]])

    selected = select_tracks([graze], zone, ["1211"])

    assert [t.name for t in selected] == ["1211.1 AA"]
    track = selected[0]
    assert track.length_in_zone_m == pytest.approx(0.5, abs=0.01)
    assert track.in_zone_fraction < 0.10


def test_select_clips_to_the_zone_buffered_by_clip_margin():
    # A searcher walking just outside the boundary genuinely sweeps into it
    # (operator decision, spec SS5.2): clip_margin extends the geometry a
    # track is clipped/tested against, but length_in_zone_m still measures
    # against the UNBUFFERED zone (it feeds the Koopman effort number over
    # the unbuffered zone_area_m2).
    zone = square(0, 0)  # 0..10 in both axes
    just_outside = feature("1211.1 AA", [[10.5, 5, 0, 0], [20, 5, 0, 0]])

    selected = select_tracks([just_outside], zone, ["1211"], clip_margin=1.0)

    assert len(selected) == 1
    track = selected[0]
    # Clipped against the buffered zone (right edge at x=11 along this flat
    # edge): the segment from x=10.5 to x=11 survives.
    assert track.geometry.length == pytest.approx(0.5, abs=0.01)
    # But it never touches the unbuffered zone (which ends at x=10).
    assert track.length_in_zone_m == 0.0


def test_select_without_clip_margin_excludes_a_track_just_outside_the_zone():
    zone = square(0, 0)
    just_outside = feature("1211.1 AA", [[10.5, 5, 0, 0], [20, 5, 0, 0]])

    with pytest.raises(TrackSelectionError, match="none intersect"):
        select_tracks([just_outside], zone, ["1211"])


def test_select_errors_when_the_team_has_no_tracks_on_the_map():
    with pytest.raises(TrackSelectionError, match="9999"):
        select_tracks([feature("1211.1 AA", [[1, 1, 0, 0], [2, 2, 0, 0]])],
                      square(0, 0), ["9999"])


def test_select_errors_differently_when_tracks_exist_but_miss_the_zone():
    far = feature("1211.1 AA", [[50, 50, 0, 0], [59, 59, 0, 0]])

    with pytest.raises(TrackSelectionError, match="none intersect"):
        select_tracks([far], square(0, 0), ["1211"])


# --- time bounds (Phase 8b, commit 1): epoch-ms timestamps of the IN-ZONE
# points, not the whole recorded track ------------------------------------


def test_select_records_time_entered_and_exited_from_in_zone_points_only():
    zone = square(0, 0)  # 0..10 in both axes
    # Enters the zone at t=1000, leaves it at t=3000 (last in-zone point),
    # then keeps recording outside the zone until t=9000 - that tail must
    # not affect the reported bounds.
    track = feature("1211.1 AA", [
        [5, 1, 0, 1000],   # inside, entry
        [5, 5, 0, 2000],   # inside
        [5, 9, 0, 3000],   # inside, last before leaving
        [5, 20, 0, 9000],  # outside
    ])

    selected = select_tracks([track], zone, ["1211"])[0]

    assert selected.time_entered_ms == 1000
    assert selected.time_exited_ms == 3000


def test_select_time_bounds_are_none_without_a_timestamp_ordinate():
    zone = square(0, 0)
    # Only [x, y] - no elevation, no timestamp.
    track = feature("1211.1 AA", [[1, 1], [9, 9]])

    selected = select_tracks([track], zone, ["1211"])[0]

    assert selected.time_entered_ms is None
    assert selected.time_exited_ms is None


def test_select_time_bounds_use_the_buffered_zone_when_clip_margin_is_given():
    zone = square(0, 0)  # 0..10
    # Entirely outside the raw zone (x=10.5) but inside a 1 m buffer.
    just_outside = feature("1211.1 AA", [[10.5, 5, 0, 4000], [20, 5, 0, 5000]])

    selected = select_tracks([just_outside], zone, ["1211"], clip_margin=1.0)[0]

    # The buffered point (x=10.5, within the 1 m margin) counts; the point
    # further out (x=20) does not.
    assert selected.time_entered_ms == 4000
    assert selected.time_exited_ms == 4000


def test_select_time_window_across_two_tracks_takes_the_widest_span():
    zone = square(0, 0)
    early_track = feature("1211.1 AA", [[1, 1, 0, 1000], [2, 2, 0, 1500]])
    late_track = feature("1211.2 BB", [[3, 3, 0, 8000], [4, 4, 0, 8500]])

    selected = select_tracks([early_track, late_track], zone, ["1211"])
    by_name = {t.name: t for t in selected}

    assert by_name["1211.1 AA"].time_entered_ms == 1000
    assert by_name["1211.2 BB"].time_exited_ms == 8500


# --- what the names the tracks were chosen by leave to check (PR #4 review
# B2, B18): said in `warnings`, which reaches the caller - the log does not


def test_a_fallback_to_a_weakly_named_line_is_said_naming_it():
    zone = square(0, 0)
    flight = feature("UAS-WINGTRA FLIGHT 3", [[1, 1, 0, 0], [9, 9, 0, 0]])
    warnings = []

    selected = select_tracks([flight], zone, ["3"], warnings=warnings)

    assert [t.name for t in selected] == ["UAS-WINGTRA FLIGHT 3"]
    assert len(warnings) == 1
    assert warnings[0].startswith("no track is plainly named for team 3")
    assert "UAS-WINGTRA FLIGHT 3" in warnings[0]


def test_a_set_aside_line_is_named_only_when_it_reaches_the_assignment():
    """A ring across the map would not have counted either way: naming it
    would bury the line whose setting aside changes this coverage."""
    zone = square(0, 0)
    features = [feature("3 AB", [[1, 1, 0, 0], [9, 9, 0, 0]]),
                feature("UAS-WINGTRA FLIGHT 3", [[1, 9, 0, 0], [9, 1, 0, 0]]),
                feature("75% (2.3mi) - ring 3", [[50, 50, 0, 0], [59, 59, 0, 0]])]
    warnings = []

    selected = select_tracks(features, zone, ["3"], warnings=warnings)

    assert [t.name for t in selected] == ["3 AB"]
    assert len(warnings) == 1
    assert warnings[0].startswith("set aside UAS-WINGTRA FLIGHT 3, which reaches")
    assert "ring" not in warnings[0]


@pytest.mark.parametrize("title, word", [("PG&E Distribution 104", "Distribution"),
                                         ("OSM path 104", "OSM")])
def test_the_reason_a_line_reads_as_something_else_is_one_its_title_shows(title, word):
    """Not only drone flights and rings are demoted: the reason given must
    be one a planner can find in a power line's or a map import's title."""
    warnings = []
    select_tracks([feature(title, [[1, 1, 0, 0], [9, 9, 0, 0]])], square(0, 0), ["104"],
                  warnings=warnings)
    assert len(warnings) == 1 and title in warnings[0] and word in warnings[0].split(title)[1]


def test_with_several_teams_a_warning_does_not_say_the_team():
    """"the team's own" beside teams 3, 4 leaves the planner guessing which."""
    warnings = []
    select_tracks([feature("UAS-WINGTRA FLIGHT 3", [[1, 1, 0, 0], [9, 9, 0, 0]])],
                  square(0, 0), ["3", "4"], warnings=warnings)
    assert len(warnings) == 1
    assert "check it is one of those teams' own" in warnings[0]
    assert "the team's" not in warnings[0]


def test_a_plainly_named_track_leaves_nothing_to_check():
    warnings = []
    select_tracks([feature("3 AB", [[1, 1, 0, 0], [9, 9, 0, 0]])], square(0, 0), ["3"],
                  warnings=warnings)
    assert warnings == []


def test_none_intersecting_names_a_set_aside_line_that_does():
    """The plainly named track is elsewhere; the only line reaching the
    assignment was set aside - "none intersect" alone would hide it."""
    features = [feature("3 AB", [[50, 50, 0, 0], [59, 59, 0, 0]]),
                feature("UAS-WINGTRA FLIGHT 3", [[1, 1, 0, 0], [9, 9, 0, 0]])]

    with pytest.raises(TrackSelectionError, match="none intersect") as raised:
        select_tracks(features, square(0, 0), ["3"])

    assert "UAS-WINGTRA FLIGHT 3" in str(raised.value)


def test_a_name_with_no_number_says_what_it_chose_by_a_word_alone():
    """The assignment label as a track name finds every team's line on it;
    a line a team number also chose needs no second look on its account."""
    zone = square(0, 0)
    features = [feature("101 AB", [[1, 1, 0, 0], [9, 9, 0, 0]]),
                feature("102 AB", [[1, 9, 0, 0], [9, 1, 0, 0]]),
                feature("Slab Creek trail", [[1, 5, 0, 0], [9, 5, 0, 0]])]
    warnings = []

    selected = select_tracks(features, zone, ["101", "AB"], warnings=warnings)

    assert [t.name for t in selected] == ["101 AB", "102 AB"]
    assert len(warnings) == 1
    assert warnings[0].startswith("'AB' has no team number in it, so it selected 102 AB ")
