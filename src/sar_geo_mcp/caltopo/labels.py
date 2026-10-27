"""Finding an assignment when the label is not the shape you expected.

WHY
---
This package grew up against one agency's convention, where CalTopo's
`letter` and `number` fields hold exactly what their names suggest and
concatenate to a tidy `QA103`. Another agency's real map, read while
testing, looks like this instead - everything in `letter`, `number` empty:

    BT (201)                            letter, then the TEAM in brackets
    CA (308) *                          plus a status mark
    Area A (203, 219, 220, 221, 222)    a named area worked by five teams
    DK                                  letters alone
    ''                                  nothing at all, on three of them

Under the old rule, `find_assignment("BT")` failed on `BT (201)` - the
label was compared whole, so a bracketed team number made an assignment
unreachable by the name everyone calls it. Half that map was unusable.

THE RULE
--------
Build every string a person might reasonably use for an assignment, match
against all of them, and when more than one assignment answers to what was
typed, say so and list them rather than picking. Guessing which assignment
somebody meant sends a team to the wrong ground, which is the one outcome
worth failing to avoid.

Matching is deliberately tiered - exact before core before prefix - so a
label that IS an exact match is never beaten by something that merely
starts the same way. `AB` must find `AB` even when `ABC` exists.
"""

from __future__ import annotations

import re

#: A bracketed group: team numbers, dates, whatever an agency puts there.
_BRACKETED = re.compile(r"[(\[{][^)\]}]*[)\]}]")

#: Trailing status marks. Seen in the wild: `CA (308) *` where the asterisk
#: appears to flag a state the map's own legend explains.
_DECORATION = re.compile(r"[*#!+~]+")

#: Everything that is not a letter or a digit. Removed for comparison only -
#: never for display, because the label a human reads should stay theirs.
_NOISE = re.compile(r"[^a-z0-9]+")


def normalise(text: str) -> str:
    """Case-folded, with punctuation and spacing removed.

    `BT (201)`, `bt-201` and `BT201` all normalise together, which is the
    point: three people will write the same assignment three ways.
    """
    return _NOISE.sub("", text.lower())


def core_of(text: str) -> str:
    """The label with brackets and status marks stripped.

    `CA (308) *` -> `CA`, `Area A (203, 219)` -> `Area A`. This is the part
    that identifies the ground; what is in brackets identifies who was sent
    to it, which changes between operational periods while the ground does
    not.
    """
    without = _BRACKETED.sub(" ", text)
    without = _DECORATION.sub(" ", without)
    return " ".join(without.split())


def bracketed_numbers(text: str) -> list[str]:
    """Numbers found inside brackets - usually the teams assigned.

    Returned so a caller can offer `BT (201)` to somebody who typed `BT201`,
    and so team lookup has somewhere to start when the tracks themselves are
    named by a convention nobody wrote down.
    """
    numbers: list[str] = []
    for group in _BRACKETED.findall(text):
        numbers.extend(re.findall(r"\d+", group))
    return numbers


def label_of(feature: dict) -> str:
    """The label as the map shows it - decorations and all.

    Kept verbatim for display and error messages. A person scanning CalTopo
    for what this tool just named has to see the same string.
    """
    properties = feature.get("properties") or {}
    letter = str(properties.get("letter") or "").strip()
    number = str(properties.get("number") or "").strip()
    if letter and number:
        # The tidy convention: two fields that concatenate.
        return f"{letter}{number}" if letter[-1].isalpha() else f"{letter} {number}"
    return letter or number


def variants(feature: dict) -> set[str]:
    """Every normalised string that should find this assignment.

    Includes the bracket-stripped core and the core rejoined to its
    bracketed numbers, so `BT (201)` answers to `BT`, `BT 201` and `BT201`
    without anybody having to know which form the map used.
    """
    label = label_of(feature)
    if not label:
        return set()

    core = core_of(label)
    forms = {label, core}
    for number in bracketed_numbers(label):
        forms.add(f"{core}{number}")

    properties = feature.get("properties") or {}
    letter = str(properties.get("letter") or "").strip()
    number = str(properties.get("number") or "").strip()
    if letter:
        forms.update({letter, core_of(letter)})
    if letter and number:
        forms.add(f"{letter}{number}")

    return {normalise(form) for form in forms if normalise(form)}


def match(features: list[dict], query: str) -> tuple[list[dict], str]:
    """Assignments answering to `query`, and how they were matched.

    Tiers, strongest first. Each tier is only consulted when the one above
    it found nothing, so an exact label is never beaten by a prefix - `AB`
    finds `AB` even on a map that also has `ABC`.

    The top tier is the feature's own CalTopo id, because a LABEL IS NOT
    ALWAYS UNIQUE. Measured across 255 real maps (2026-09-04): 51 labels on
    21 of 96 assignment-carrying maps name more than one distinct
    assignment - same `letter`/`number`, same operational period, different
    polygons - and one map calls four different assignments `330`. Every
    label tier below correctly refuses to choose between them, which left
    the caller with an error telling them to "use the full label" when the
    full label was exactly what they had typed. An id is the only handle
    that can always name one of them, so it is offered as a query.

    Returns every match rather than one, so the caller can refuse an
    ambiguous answer instead of receiving an arbitrary one.
    """
    identified = [f for f in features if str(f.get("id") or "") == query]
    if identified:
        # An id names exactly one feature by construction. Sliced anyway:
        # a malformed feature list that repeats an id must not silently
        # widen this tier into an ambiguous answer.
        return identified[:1], "id"

    wanted = normalise(query)
    if not wanted:
        return [], "empty"

    exact = [f for f in features if wanted in variants(f)]
    if exact:
        return exact, "exact"

    core = [f for f in features if normalise(core_of(label_of(f))) == wanted]
    if core:
        return core, "core"

    prefix = [f for f in features
              if any(v.startswith(wanted) for v in variants(f))]
    if prefix:
        return prefix, "prefix"

    return [], "none"
