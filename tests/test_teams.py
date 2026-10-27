"""Deciding whether a track's name belongs to a team.

The generous rule - team digits anywhere in the title, spatial filter as
the real gate - holds on a map whose lines are all searcher tracks. On a
real agency map the Shape layer also carries drone sorties, flight logs
and probability rings, every one of which can cross an assignment and none
of which the spatial filter can tell from somebody walking.

Titles here are the shapes observed on that map, with the surnames
replaced - a searcher's name is not needed to test a matcher.
"""

from __future__ import annotations

import pytest

from sar_geo_mcp.coverage.teams import (
    STRONG,
    WEAK,
    match_strength,
    matches_team,
    normalise_team,
    partition,
)


# ---------------------------------------------------------------------------
# What a team id can be typed as.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("typed", ["103", "Team 103", "T103", "103A", " 103 "])
def test_a_team_id_is_recognised_however_it_is_typed(typed):
    assert normalise_team(typed) == "103"


# ---------------------------------------------------------------------------
# Plainly a team.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("title,team", [
    ("DE-104-Doe", "104"),          # agency convention: zone-team-name
    ("GY-104A-Christy", "104"),         # with a sortie letter
    ("103XN", "103"),                   # the convention this package knew
    ("102_XY", "102"),
    ("xy 102", "102"),                  # written backwards
    ("1211.1 QZ", "1211"),              # team and sortie, dot-separated
    ("T104", "104"),
    ("Team 104", "104"),
])
def test_a_searcher_track_matches_strongly(title, team):
    assert match_strength(title, team) == STRONG


def test_the_digit_run_rule_is_unchanged():
    """`1024` is a different team, not team 102 with a trailing digit."""
    assert match_strength("1024 NG", "102") is None
    assert match_strength("1024 NG", "1024") == STRONG


def test_a_dotted_sortie_number_still_matches_its_team():
    """Stripping separators before the digit-run test merges the team id
    with its sortie number and silently breaks this - which happened once
    already, on a live map."""
    assert match_strength("1211.1 QZ", "1211") == STRONG


# ---------------------------------------------------------------------------
# Digits that are not a team.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("title,team", [
    ("UAS-WINGTRA FLIGHT 3", "3"),                    # a drone sortie
    ("Jun-15th-2025-10-30AM-Flight-Airdata", "15"),     # a flight log
    ("Jun-15th-2025-10-30AM-Flight-Airdata", "2025"),
    ("50% (1.2mi) - Hiker", "50"),                      # a probability ring
    ("50% (1.2mi) - Hiker", "1"),                       # inside the distance
])
def test_a_number_that_reads_as_something_else_matches_weakly(title, team):
    """Weak, not absent. A team COULD be called FLIGHT 3, so the match is
    demoted and reported rather than discarded."""
    assert match_strength(title, team) == WEAK


def test_a_title_with_no_team_digits_does_not_match():
    assert match_strength("Power Distribution Line OSM Way", "102") is None
    assert match_strength("Approx social trail to ridge", "102") is None


# ---------------------------------------------------------------------------
# Using the verdict.
# ---------------------------------------------------------------------------

def test_partition_separates_the_plain_from_the_doubtful():
    titles = ["DE-104-Doe", "GY-104A-Smith", "UAS-WINGTRA FLIGHT 104",
              "Power Distribution Line"]
    strong, weak = partition(titles, ["104"])

    assert strong == ["DE-104-Doe", "GY-104A-Smith"]
    assert weak == ["UAS-WINGTRA FLIGHT 104"]


def test_a_track_matching_any_requested_team_strongly_counts_as_strong():
    strong, weak = partition(["DE-104-Doe"], ["999", "104"])
    assert strong and not weak


def test_the_yes_no_form_still_answers_for_both_strengths():
    """`matches_team` is kept for callers that only need membership, but it
    cannot distinguish a searcher from a drone - which is why the tracks
    module uses `match_strength` instead."""
    assert matches_team("DE-104-Doe", "104")
    assert matches_team("UAS-WINGTRA FLIGHT 104", "104")
    assert not matches_team("Power Distribution Line", "104")


# ---------------------------------------------------------------------------
# A name with no number in it (PR #4 review B18).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("title,team", [
    ("Slab Creek trail (drawn)", "AB"),     # the assignment label, passed by mistake
    ("Cabin loop recon", "A"),
    ("Helispot alpha", "A"),
    ("Table Mountain spur", "ab"),
])
def test_a_name_with_no_number_never_matches_inside_another_word(title, team):
    """The letter counterpart of the digit-run rule: as a bare substring
    the label `AB` was a strong match on a drawn trail, and `A` on nearly
    every title on the map."""
    assert match_strength(title, team) is None


@pytest.mark.parametrize("title", ["101 AB", "AB-3", "ab_101", "Seg AB"])
def test_a_name_with_no_number_matches_as_a_word_of_its_own(title):
    assert match_strength(title, "AB") == STRONG


def test_a_word_beside_a_flight_still_matches_weakly():
    """The title still argues against itself, number or not."""
    assert match_strength("Alpha drone sortie", "Alpha") == WEAK
