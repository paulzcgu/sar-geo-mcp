"""Candidate sets: what a generating tool returns and where it persists.

IDS
---
A set id is the tool name plus six hex characters of a hash over the
parameters: `hasty-3f9a2c`. The same parameters on the same handle produce
the same id and identical content, so a planner re-running a tool gets the
same candidates back rather than a fresh numbering. A candidate id is the
set id, a dot, and a one-based index: `hasty-3f9a2c.4`.

SUMMARIES FIRST
---------------
A generating tool returns candidates without geometry; `get_candidate`
returns one with it. A long planning conversation therefore never holds
thousands of coordinates in context - the rule `list_map_features` already
follows. Sets persist as `candidates/<set_id>.json` inside the staged
directory, so ids stay valid for the life of the handle.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from pydantic import BaseModel

from ..cache import atomic_write_text

CANDIDATES_DIRNAME = "candidates"
_SET_ID = re.compile(r"^[a-z]+-[0-9a-f]{6}$")
_CANDIDATE_ID = re.compile(r"^([a-z]+-[0-9a-f]{6})\.([1-9][0-9]*)$")


class Candidate(BaseModel):
    id: str
    kind: str                       # line | corridor | block  (cell in a later phase)
    axis: str | None                # trail | road | waterway (hasty; sets saved before
                                    # 0.37.0 may hold drainage | ridge) |
                                    # street | large_parcel (a block) | None
    axis_ids: list[str]             # the Ground line ids the candidate was built from
    sources: list[str]              # where it came from, e.g. ["waterway"]
    name: str | None
    geometry: dict                  # GeoJSON, WGS84
    facts: dict
    norm_flags: list[dict]
    warnings: list[str]
    #: CalTopo Marker features drawn with the candidate - a block's house
    #: markers, when `populated_areas` was asked for them - or None. Left out
    #: of summaries with the geometry: a block can hold hundreds.
    markers: list[dict] | None = None


class CandidateSet(BaseModel):
    set_id: str
    tool: str
    params: dict
    handle: str
    created_at: str
    candidates: list[Candidate]
    summary: dict
    assumptions: dict
    warnings: list[str]
    manifest_stamp: list | None = None  # [st_mtime_ns, st_size] of manifest.json at save time
    #: The map state (CalTopo change token, or a read timestamp when there is
    #: no token) the set's overlap facts were measured against; None when the
    #: handle has no map. A set is reused only while this still matches the
    #: map as read now - see `features._sets.set_map_version` - because the facts were
    #: measured once, against one map state, and an assignment drawn since
    #: would make a reused set silently out of date.
    map_version: str | None = None


def make_set_id(tool: str, params: dict) -> str:
    """Strict canonicalisation: no `default=str`. The id contract - same
    params, same id - must not depend on how an arbitrary object happens to
    stringify, so a non-JSON-native value raises TypeError rather than being
    silently coerced.

    The id is over the PARAMETERS only, so it is unique within one staged
    directory - one handle, one piece of ground - and nowhere else: the same
    parameters on two areas give the same id, and the two sets live in two
    directories. Sets are only ever loaded from the directory the handle
    resolves to, so the two never meet.
    """
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return f"{tool}-{hashlib.sha256(canonical.encode()).hexdigest()[:6]}"


def candidate_id(set_id: str, index: int) -> str:
    if index < 1:
        raise ValueError("candidate indexes are one-based")
    return f"{set_id}.{index}"


def parse_candidate_id(candidate: str) -> tuple[str, int]:
    match = _CANDIDATE_ID.match(candidate or "")
    if not match:
        raise ValueError(
            f"not a candidate id: {candidate!r}. Use an id from hasty_candidates, "
            "such as hasty-3f9a2c.4")
    return match.group(1), int(match.group(2))


def _sets_dir(directory: Path) -> Path:
    return Path(directory) / CANDIDATES_DIRNAME


def save_set(directory: Path, cset: CandidateSet) -> Path:
    from .boundary import _manifest_stamp  # imported inside to avoid any import cycle

    if not _SET_ID.match(cset.set_id):
        raise ValueError(f"not a set id: {cset.set_id!r}")
    stamp = _manifest_stamp(Path(directory))
    cset = cset.model_copy(update={"manifest_stamp": list(stamp) if stamp is not None else None})
    folder = _sets_dir(directory)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{cset.set_id}.json"
    # Two `hasty_candidates` calls on one handle save the same set id, so
    # each writer gets its own temp name. See `cache.atomic`.
    atomic_write_text(path, cset.model_dump_json(indent=1))
    return path


def load_set(directory: Path, set_id: str) -> CandidateSet:
    if not _SET_ID.match(set_id or ""):
        raise ValueError(f"not a set id: {set_id!r}")
    path = _sets_dir(directory) / f"{set_id}.json"
    if not path.is_file():
        raise KeyError(f"no candidate set {set_id!r} in this staged directory; "
                       "run the generating tool again")
    return CandidateSet.model_validate_json(path.read_text(encoding="utf-8"))


def set_staleness(directory: Path, cset: CandidateSet) -> str | None:
    """None when the set matches the directory's current staging; otherwise
    why it may be stale, for the caller to surface as a warning."""
    from .boundary import _manifest_stamp

    current = _manifest_stamp(Path(directory))
    if cset.manifest_stamp is None or current is None:
        return "this candidate set does not record the staging it was computed from"
    if list(cset.manifest_stamp) != list(current):
        return ("this candidate set was computed before the directory was re-staged; "
                "run the generating tool again")
    return None


def find_candidate_in(cset: CandidateSet, candidate: str) -> Candidate:
    """The candidate an id names, inside a set the caller already holds.

    The one place an index becomes a record, so a caller that loaded the set
    for another reason too - its staleness, say - reads the candidate off
    that same read and still refuses a bad index in the same words as
    `find_candidate`.
    """
    set_id, index = parse_candidate_id(candidate)
    if index > len(cset.candidates):
        raise KeyError(f"no candidate {candidate!r}: set {set_id} has "
                       f"{len(cset.candidates)} candidates")
    return cset.candidates[index - 1]


def find_candidate(directory: Path, candidate: str) -> Candidate:
    set_id, _index = parse_candidate_id(candidate)
    return find_candidate_in(load_set(directory, set_id), candidate)


def summaries(cset: CandidateSet) -> list[dict]:
    """Every candidate without its geometry or markers. `markers` is left
    out entirely, not sent as null, so a summary reads as it always has."""
    return [c.model_dump(exclude={"geometry", "markers"}) for c in cset.candidates]
