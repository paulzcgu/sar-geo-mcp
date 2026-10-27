"""Deciding whether a track's name belongs to a team, and how sure we are.

WHY STRENGTH, NOT A YES/NO
--------------------------
Matching a team id anywhere in a title is generous on purpose: searchers
write `102_XY`, `xy 102` and `1211.1 QZ`, and the spatial filter was
treated as the real gate. That reasoning holds on a map whose lines are all
searcher tracks. It does not hold on a real agency map, whose Shape layer
also carries:

    UAS-WINGTRA FLIGHT 3              a drone sortie   -> "team 3"
    Jun-15th-2025-10-30AM-Flight-Airdata a flight log    -> "team 15", "2025"
    50% (1.2mi) - Hiker                  a probability ring -> "team 50"

Every one of those can cross an assignment, and the spatial filter cannot
tell a drone line from somebody walking. Counting one as team coverage
reports ground as searched that nobody swept - the same class of error the
staleness rule exists to prevent, arriving by a different door.

So a match now carries evidence, and callers prefer the strong kind:

    STRONG  the digits match and nothing in the title argues against them
            `DE-104-Doe`, `104 XY`, `1211.1 QZ`, `T104`
    WEAK    they match, but a unit or ordinal follows them, or the title
            carries a word like FLIGHT or AIRDATA
            `50% (1.2mi) - Hiker`, `UAS-WINGTRA FLIGHT 3`

A NAME WITH NO NUMBER IN IT
---------------------------
Teams are numbers, but a caller can pass a word - a team's call sign, a
searcher's surname, or, an easy slip, the assignment's own label `AB`.
Matched as a bare substring, `AB` was a strong match on "Slab Creek trail
(drawn)" and `A` on nearly every title on the map, so a drawn trail was
counted as the team's coverage with nothing said (PR #4 review B18). A
word now matches only as a whole word - the letter counterpart of the
digit-run rule - so `AB` finds `101 AB` and `AB-3` and never `Slab`. That
still selects every team's line on assignment AB, which is why the tracks
module names, in a warning, every line such a name alone selected - a line
a team number passed beside it also chose needs no second look.

WHAT THIS DOES NOT DO
---------------------
It does not decide that a track is a drone. It cannot: a team could
legitimately be called `FLIGHT 3`. It reports how the name matched and
lets the caller act - which in practice means using strong matches when
any exist, and telling somebody when only weak ones do.
"""

from __future__ import annotations

import re

STRONG = "strong"
WEAK = "weak"

#: Words that argue the digits beside them are not a team. Every entry was
#: observed on a real agency map. Kept long and distinctive on purpose: a
#: list that grows by imagination starts excluding real teams, and short
#: words like "m" would match half the alphabet soup on any map.
_NOT_A_TEAM_WORDS = (
    "flight", "airdata", "wingtra", "drone", "uav", "aircraft",
    "january", "february", "march", "april", "june", "july",
    "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "aug", "sep", "sept", "oct", "nov", "dec",
    "osm", "distribution",
)

#: What can follow a number and prove it was a measurement, an ordinal or a
#: clock time rather than a team: `50%`, `1.2mi`, `15th`, `10:30AM`.
_UNIT_SUFFIX = re.compile(r"^\s*(%|mi\b|km\b|ft\b|st\b|nd\b|rd\b|th\b|am\b|pm\b)",
                          re.IGNORECASE)


def normalise_team(team: str) -> str:
    """The digits a team id comes down to.

    Operators type `103`, `Team 103`, `T103` and `103A` for the same team.
    The letters are dropped for matching; the caller keeps whatever they
    typed for display. A name with no digits at all is kept whole, trimmed
    and lower-cased, and `match_strength` matches it as a word of its own.
    """
    digits = re.findall(r"\d+", team)
    return digits[0] if digits else team.strip().lower()


def has_team_number(team: str) -> bool:
    """Whether `team` carries a number - what a team id is. A name without
    one selects tracks by a word alone, which the caller must be told: an
    assignment's label can name every team's lines on it."""
    return re.search(r"\d", team) is not None


#: One letter of any alphabet: what a name with no number must not run
#: into on either side (see match_strength). Digits and separators are
#: boundaries, as letters are for a team number.
_LETTER = r"[^\W\d_]"


#: A decimal number carrying a unit: `1.2mi`, `0.75 km`. Digits anywhere
#: inside one are part of a measurement. Detected as a whole SPAN, not by
#: what follows the digits, because `1.2mi` and `1211.1 QZ` have the same
#: shape up to the unit - and the second is a real team-and-sortie label
#: that must keep matching.
_DECIMAL_MEASUREMENT = re.compile(r"\d+\.\d+\s*(mi|km|ft|m)\b", re.IGNORECASE)


def _adverse_context(title: str, match_start: int, match_end: int) -> bool:
    """Does anything in the title argue these digits are not a team id?

    Three signals, all cheap and all drawn from real titles: a unit or
    ordinal immediately after the number, the number sitting inside a
    decimal measurement, and a distinctive non-team word anywhere in the
    title. Any one is enough to demote a match rather than discard it - demoting keeps a genuinely odd team name usable while
    still telling somebody it looked wrong.
    """
    if _UNIT_SUFFIX.match(title[match_end:]):
        return True
    for span in _DECIMAL_MEASUREMENT.finditer(title):
        if span.start() <= match_start and match_end <= span.end():
            return True
    lowered = title.lower()
    return any(word in lowered for word in _NOT_A_TEAM_WORDS)


def match_strength(title: str, team: str) -> str | None:
    """How convincingly `title` names `team`. None when it does not.

    The digit-run rule is unchanged and still decides whether there is a
    match at all: `1024` is a different team, not team 102 with a trailing
    digit. Case-folded but NOT separator-stripped - stripping merges a team
    id with its sortie number and breaks real labels like `1211.1 QZ`.

    What is new is the verdict. A match in a title that also says "FLIGHT"
    or carries a percent sign is reported WEAK, so a caller can prefer the
    tracks that plainly belong to the team and say what it set aside.

    A name with no number in it matches only as a whole word, never as
    letters inside another one: `AB` matches `101 AB` and `AB-3`, not
    "Slab Creek trail", and `A` matches `Team A`, not "Cabin loop recon".
    """
    wanted = normalise_team(team)
    if not wanted:
        return None

    if has_team_number(team):
        pattern = rf"(?<!\d){re.escape(wanted)}(?!\d)"
    else:
        # The letter counterpart of the digit-run rule. As a bare substring
        # a word matched inside any other - the assignment label `AB` was a
        # strong match on "Slab Creek trail (drawn)", and a drawn trail was
        # counted as the team's coverage. Nothing in such a title argues
        # against a match, so the checks below never demoted it either.
        pattern = rf"(?<!{_LETTER}){re.escape(wanted)}(?!{_LETTER})"
    found = re.search(pattern, title.lower())
    if not found:
        return None

    return (WEAK if _adverse_context(title, found.start(), found.end())
            else STRONG)


def matches_team(title: str, team: str) -> bool:
    """Back-compatible yes/no. Prefer `match_strength` where the answer
    matters - this cannot distinguish a searcher from a drone."""
    return match_strength(title, team) is not None


def partition(titles: list[str], teams: list[str]
              ) -> tuple[list[str], list[str]]:
    """Split candidate titles into strongly and weakly matched.

    Returned separately so a caller can use the strong ones and still say
    what it set aside. A silently narrowed track list is how coverage ends
    up computed from half a team's work.
    """
    strong: list[str] = []
    weak: list[str] = []
    for title in titles:
        best = None
        for team in teams:
            strength = match_strength(title, team)
            if strength == STRONG:
                best = STRONG
                break
            if strength == WEAK:
                best = WEAK
        if best == STRONG:
            strong.append(title)
        elif best == WEAK:
            weak.append(title)
    return strong, weak
