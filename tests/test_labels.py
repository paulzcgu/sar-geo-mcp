"""Finding an assignment when the label is not the shape you expected.

Every awkward case here was read off a real agency map (174 assignments),
not invented. The conventions this package grew up on are the minority
there: 87 of those labels carry a bracketed team number, 3 carry nothing
at all, and one names an area worked by five teams.
"""

from __future__ import annotations

import pytest

from sar_geo_mcp.caltopo.labels import (
    bracketed_numbers,
    core_of,
    label_of,
    match,
    normalise,
    variants,
)
from sar_geo_mcp.caltopo.assignment import find_assignment

SQUARE = {"type": "Polygon", "coordinates": [[
    [-122.32, 37.40], [-122.30, 37.40], [-122.30, 37.42],
    [-122.32, 37.42], [-122.32, 37.40]]]}


_next_id = iter(range(1000))


def assignment(letter: str = "", number: str = "") -> dict:
    return {"id": f"f{next(_next_id)}", "geometry": SQUARE, "properties": {
        "class": "Assignment", "letter": letter, "number": number}}


# ---------------------------------------------------------------------------
# Reading a label.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("letter,number,expected", [
    ("QA", "103", "QA103"),          # the convention this package grew up on
    ("BT (201)", "", "BT (201)"),    # everything in one field, team bracketed
    ("DK", "", "DK"),
    ("Area A (203, 219, 220)", "", "Area A (203, 219, 220)"),
    ("", "", ""),                    # 3 of 174 on the real map
])
def test_the_label_is_reported_as_the_map_shows_it(letter, number, expected):
    """Verbatim, decorations and all. Somebody scanning CalTopo for what
    this tool just named has to see the same string."""
    assert label_of(assignment(letter, number)) == expected


@pytest.mark.parametrize("label,expected", [
    ("BT (201)", "BT"),
    ("CA (308) *", "CA"),
    ("Area A (203, 219, 220, 221, 222)", "Area A"),
    ("DK", "DK"),
])
def test_the_core_drops_brackets_and_status_marks(label, expected):
    """What is in brackets says who was sent; the core says where. Teams
    change between operational periods, the ground does not."""
    assert core_of(label) == expected


def test_bracketed_numbers_are_recovered():
    assert bracketed_numbers("Area A (203, 219, 220)") == ["203", "219", "220"]
    assert bracketed_numbers("DK") == []


def test_normalise_folds_the_ways_people_write_the_same_thing():
    assert normalise("BT (201)") == normalise("bt-201") == normalise("BT201")


# ---------------------------------------------------------------------------
# Matching.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query", ["BT", "bt", "BT (201)", "bt-201", "BT201",
                                   " bt ", "Bt(201)"])
def test_every_reasonable_spelling_finds_the_assignment(query):
    """Three people will write the same assignment three ways, and none of
    them is wrong."""
    found = find_assignment([assignment("BT (201)")], query)
    assert found.label == "BT (201)"


def test_the_old_convention_is_untouched():
    """The map this package was built against must keep working."""
    features = [assignment("QA", "103"), assignment("QB", "102")]
    assert find_assignment(features, "QA103").label == "QA103"
    assert find_assignment(features, "QA").label == "QA103"
    assert find_assignment(features, "qa103").label == "QA103"


def test_an_exact_label_beats_something_that_merely_starts_the_same():
    """`AB` must find `AB`, not complain about `ABC`. Tiering exists for
    exactly this."""
    features = [assignment("AB"), assignment("ABC"), assignment("ABD")]
    assert find_assignment(features, "AB").label == "AB"


def test_a_named_area_is_reachable_by_its_name():
    features = [assignment("Area A (203, 219, 220, 221, 222)")]
    assert find_assignment(features, "Area A").label.startswith("Area A")


# ---------------------------------------------------------------------------
# Refusing to guess.
# ---------------------------------------------------------------------------

def test_an_ambiguous_query_lists_what_it_matched_and_refuses():
    """Picking one would send a team to the wrong ground. Being asked to
    type a fuller name is a far smaller cost."""
    features = [assignment("BA"), assignment("BB"), assignment("BC")]
    with pytest.raises(LookupError) as raised:
        find_assignment(features, "B")

    message = str(raised.value)
    assert "ambiguous" in message
    for label in ("BA", "BB", "BC"):
        assert label in message


def test_a_missing_assignment_names_what_is_available():
    features = [assignment("BT (201)"), assignment("DK")]
    with pytest.raises(LookupError) as raised:
        find_assignment(features, "ZZ")

    message = str(raised.value)
    assert "BT (201)" in message and "DK" in message


def test_unlabelled_assignments_are_reported_not_ignored(caplog):
    """Three assignments on the real map carry no label. They cannot be
    looked up by name, and somebody hunting for one deserves to know that
    rather than concluding the map is wrong."""
    features = [assignment("DK"), assignment(""), assignment("")]
    with pytest.raises(LookupError) as raised:
        find_assignment(features, "ZZ")

    assert "no label" in str(raised.value)
    assert "2 assignment" in str(raised.value)


def big_map(count: int = 100) -> list[dict]:
    """`count` labelled assignments, AA100 to DV199 for the default 100,
    and two with no label at all."""
    labelled = [assignment(f"{chr(65 + i // 26)}{chr(65 + i % 26)}", str(100 + i))
                for i in range(count)]
    return labelled + [assignment(""), assignment("")]


def available_names(message: str) -> list[str]:
    """The names a not-found message lists, up to any cut."""
    listed = message.split("Available: ", 1)[1].split(" ... (", 1)[0]
    return listed.split(", ")


def test_a_long_available_list_is_cut_short_and_says_how_many_there_are():
    """On a big map the not-found message is the operator's only hint about
    which labels exist. It stays readable - about 400 characters of names,
    not every label on the map - says how many labels there really are, and
    keeps the note about the unlabelled ones after the cut."""
    with pytest.raises(LookupError) as raised:
        find_assignment(big_map(), "ZZ")

    message = str(raised.value)
    assert message.startswith("assignment 'ZZ' not found. Available: AA100, AB101, ")
    assert " ... (100 total)." in message
    assert "DV199" not in message                     # the last label, cut
    listed = message.split("Available: ", 1)[1].split(" ... (", 1)[0]
    assert 300 < len(listed) <= 400
    assert message.endswith(" 2 assignment(s) on this map carry no label at all "
                            "and cannot be looked up by name.")
    assert len(message) < 600


def test_a_short_available_list_is_not_cut():
    """Under the limit every label is listed and no total is tacked on."""
    features = big_map(20)
    labels = sorted(label_of(f) for f in features if label_of(f))
    with pytest.raises(LookupError) as raised:
        find_assignment(features, "ZZ")

    message = str(raised.value)
    assert f"Available: {', '.join(labels)}. 2 assignment(s)" in message
    assert " ... " not in message and "total)" not in message


@pytest.mark.xfail(strict=True, reason=(
    "PR #4 review NEW: find_assignment's 400-character cut of the Available "
    "list falls mid-label, so the list ends in a fragment that is not a label "
    "(here 'C', from CF157)"))
def test_a_cut_available_list_names_only_whole_labels():
    """Every name listed must be a label the map has: an operator reading
    "..., CE156, C ... (100 total)" takes C for an assignment."""
    features = big_map()
    labels = {label_of(f) for f in features if label_of(f)}
    with pytest.raises(LookupError) as raised:
        find_assignment(features, "ZZ")

    listed = available_names(str(raised.value))
    assert set(listed) <= labels, f"not labels: {sorted(set(listed) - labels)}"


def test_an_empty_query_finds_nothing_rather_than_everything():
    assert match([assignment("DK")], "")[0] == []
    assert match([assignment("DK")], "   ")[0] == []


def test_an_unlabelled_assignment_has_no_variants():
    """It must not match the empty string, or it would answer to anything."""
    assert variants(assignment("")) == set()


# --- a label is not always unique -----------------------------------------
#
# Measured on 255 real CalTopo maps (2026-09-04): 51 labels across 21 of the
# 96 assignment-carrying maps name MORE THAN ONE distinct assignment - same
# letter/number, same operational period, different polygons. One map calls
# four different assignments "330". Every label tier correctly refuses to
# choose, which left callers with "use the full label" when the full label
# was what they had typed. The id tier is the way out.

def identified(feature_id, letter, number=""):
    """Like this module's `assignment` helper, but carrying a CalTopo id."""
    return {"id": feature_id,
            "properties": {"class": "Assignment", "letter": letter,
                           "number": number},
            "geometry": {"type": "Polygon", "coordinates": []}}


def test_a_feature_id_names_exactly_one_assignment():
    from sar_geo_mcp.caltopo.labels import match

    features = [identified("aaa-111", "DR"), identified("bbb-222", "DR")]

    matched, how = match(features, "bbb-222")

    assert how == "id"
    assert [f["id"] for f in matched] == ["bbb-222"]


def test_the_id_tier_outranks_every_label_tier():
    """An id is unambiguous by construction; a label may not be."""
    from sar_geo_mcp.caltopo.labels import match

    features = [identified("DR", "AB"), identified("other", "DR")]

    # "DR" is BOTH a real label here and one feature's id. The id wins.
    matched, how = match(features, "DR")
    assert how == "id"
    assert matched[0]["properties"]["letter"] == "AB"


def test_duplicate_labels_still_refuse_to_guess():
    from sar_geo_mcp.caltopo.labels import match

    features = [identified("aaa-111", "DR"), identified("bbb-222", "DR")]

    matched, how = match(features, "DR")

    assert how == "exact"
    assert len(matched) == 2, "picking one would send a team to the wrong ground"


def test_a_repeated_id_never_widens_the_id_tier():
    """A malformed feature list must not turn the one unambiguous handle
    into an ambiguous answer."""
    from sar_geo_mcp.caltopo.labels import match

    features = [identified("same", "DR"), identified("same", "DR")]

    matched, how = match(features, "same")

    assert how == "id"
    assert len(matched) == 1
