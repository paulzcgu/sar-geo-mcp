"""Helpers the candidate-set features share: deciding whether a set already
on disk can be read back instead of measured again. Not tools."""

from __future__ import annotations

from ..planning.candidates import CandidateSet, load_set, set_staleness


def set_map_version(resolved) -> str | None:
    """A string identifying the exact map state a candidate set's overlap
    facts were measured against: CalTopo's own change token, or (for a
    watched, authenticated map, which carries no token) the read's
    timestamp. None when the handle has no map.

    Not read from `resolved.status`'s provenance dict: every read here
    forces revalidation (`max_age_seconds=0`), so `fetched_at` moves on
    every call even when nothing on the map changed - see the
    cache-revalidated path in `caltopo/state.py:fetch_map_state`. The token
    is the one thing on `MapState` that stays put across an unchanged read.
    """
    state = resolved.state
    if state is None:
        return None
    token = getattr(state, "token", None)
    return str(token) if token is not None else str(state.fetched_at)


def reusable_set(directory, set_id: str, map_version: str | None) -> CandidateSet | None:
    """A set of this id already in the staged directory whose staging still
    matches and whose map version still matches, or None.

    The id is a hash of the parameters and the ground behind a handle does not
    change, so re-running the same call is the same measurement: reading it
    back is seconds rather than minutes on a real area, and the candidate ids
    a planner is already holding stay pointing at the same lines. A set that
    predates a re-stage is not reused - it would describe ground that has
    since been replaced - and neither is one measured against an earlier map
    state: an assignment drawn since would leave its overlap facts silently
    out of date. Both `None` is a match - a handle with no map never had
    overlap to go stale.
    """
    try:
        cset = load_set(directory, set_id)
    except KeyError:
        return None
    if cset.map_version != map_version:
        return None
    return cset if set_staleness(directory, cset) is None else None
