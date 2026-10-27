"""Cached CalTopo map state, with an age that is always reported.

THE PROBLEM THIS SOLVES
-----------------------
Map state is the one thing in this system that changes during an operation.
Assignments get redrawn, markers get dropped, tracks get uploaded. Caching
it wrongly is not a performance bug: coverage computed against a boundary
that has since moved reports ground as searched that nobody walked.

Before this, there was no cache at all - correct, but `list_assignments`
then `get_assignment` then `list_map_features` was three full downloads of
the same map, 2.75 MB each on a real incident.

THE PROTOCOL - CHANGE DETECTION, NOT MERGING
--------------------------------------------
CalTopo's `since/{token}` endpoint returns only what changed, and it is
tiny: measured against a real map, the full fetch was 2 752 019 bytes and
the unchanged incremental reply was 84. That is a 30 000x saving on the
common case, and the common case - nothing changed in the last minute - is
almost every call.

So the incremental reply is used as a CHANGE DETECTOR and never as a merge
source:

    empty reply  -> nothing changed -> serve the cached state, roll the token
    any content  -> something changed -> full refetch, replace wholesale

The reason is deletions. The full reply carries an `ids` map listing
membership by feature class; the incremental reply carries `ids: null`. So
an incremental payload cannot tell us what was REMOVED, and a merge built
on it would keep a deleted assignment forever. Whether a deletion even
produces incremental content is not something this code can verify without
mutating somebody's map, so it is designed for the case where it does not:
`FULL_REFRESH_SECONDS` forces a periodic full fetch that reconciles
membership regardless.

Detecting rather than merging also removes an entire class of partial-merge
bug, which for a 30 000x saving already banked is a trade worth taking.

OFFLINE
-------
A command post often has no connectivity. A refresh failure with a cached
copy in hand returns that copy MARKED - `stale=True`, with an age and a
reason - rather than raising. We do not refuse to work offline, and we
never let an old answer present itself as a current one.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..cache import CacheIndex, atomic_write_text

STATE_DIRNAME = "caltopo-state"

#: Serve a cached map without touching the network for this long. Short,
#: because the cheap call is the change check, not this window.
DEFAULT_MAX_AGE_SECONDS = 300.0

#: Force a full fetch this often no matter what the change detector says,
#: so membership - and therefore deletions - is reconciled. See the module
#: docstring for why the incremental reply cannot be trusted for this.
FULL_REFRESH_SECONDS = 900.0


@dataclass(frozen=True)
class MapState:
    """A map's features plus everything needed to judge how much to trust them."""

    map_id: str
    features: list[dict]
    token: int | None
    fetched_at: float
    #: When membership was last established by a full fetch. Deletions are
    #: only visible as of this moment, not as of `fetched_at`.
    membership_at: float
    stale: bool = False
    reason: str = ""
    source: str = "network"

    @property
    def age_seconds(self) -> float:
        return max(0.0, time.time() - self.fetched_at)

    def provenance(self) -> dict:
        """The freshness block every tool payload carries.

        A planner reasoning off forty-minute-old boundaries has to be able
        to see that they are forty minutes old. This is the same principle
        as recording a searcher's claim beside the computed value: the
        caller is told what they are holding.
        """
        block = {
            "fetched_at": self.fetched_at,
            "age_seconds": round(self.age_seconds, 1),
            "source": self.source,
            "membership_age_seconds": round(
                max(0.0, time.time() - self.membership_at), 1),
        }
        if self.stale:
            block["stale"] = True
            block["reason"] = self.reason
        return block


def _state_path(cache_dir: Path, map_id: str) -> Path:
    # map_id comes from a caller and lands in a filename, so it is
    # restricted to what CalTopo actually issues rather than trusted.
    safe = "".join(c for c in map_id if c.isalnum() or c in "-_")
    if not safe:
        raise ValueError(f"unusable map id: {map_id!r}")
    return Path(cache_dir) / STATE_DIRNAME / f"{safe}.json"


def read_cached(cache_dir: Path, map_id: str) -> MapState | None:
    path = _state_path(Path(cache_dir), map_id)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    try:
        return MapState(
            map_id=map_id,
            features=payload["features"],
            token=payload.get("token"),
            fetched_at=float(payload["fetched_at"]),
            membership_at=float(payload.get("membership_at",
                                            payload["fetched_at"])),
            source="cache",
        )
    except (KeyError, TypeError, ValueError):
        return None


def write_cached(cache_dir: Path, state: MapState) -> None:
    path = _state_path(Path(cache_dir), state.map_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Two threads cache the same map at the start of a session - the web
    # thread for the PLS echo, the staging thread behind it - so the write
    # goes through a temp name unique to this writer. See `cache.atomic`.
    atomic_write_text(path, json.dumps({
        "features": state.features,
        "token": state.token,
        "fetched_at": state.fetched_at,
        "membership_at": state.membership_at,
    }))
    CacheIndex(cache_dir).record(path, layer="caltopo-state", tier="mutable")


def _token_of(payload: dict) -> int | None:
    """CalTopo's change token: `result.timestamp`, milliseconds since epoch."""
    result = payload.get("result", payload)
    token = result.get("timestamp") if isinstance(result, dict) else None
    return token if isinstance(token, int) else None


def _changed(payload: dict) -> bool:
    """Did the incremental reply carry anything at all?

    Any content means something moved and we stop trusting the cached copy.
    Deliberately not "did anything RELEVANT move" - deciding that from a
    partial payload is how merge bugs start.
    """
    from .session import extract_features

    return bool(extract_features(payload))


def fetch_map_state(
    map_id: str,
    host: str = "caltopo.com",
    cache_dir: Path | str | None = None,
    max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS,
    fetch=None,
    cached: MapState | None = None,
) -> MapState:
    """Current map state, from cache when it is fresh enough to trust.

    `max_age_seconds=0` forces revalidation and is what every computation
    uses: a listing may be slightly stale provided it says so, but a
    computed result may not be. `fetch(map_id, host, since)` is injectable
    so the tests never touch the network.

    `cached` is the prior to revalidate against when the caller keeps it in
    memory itself - a long-running server watching one map all day, which
    has no use for a disk cache and no wish to rewrite 2.75 MB of JSON on
    every 84-byte probe. It takes precedence over anything under
    `cache_dir`; with neither, the first call is a full fetch.

    Raises only when there is nothing to return - no cached copy AND no
    network. With a cached copy in hand a network failure comes back marked
    stale, never as an exception.
    """
    from .session import extract_features, fetch_public_map

    fetch = fetch or (lambda mid, h, since: fetch_public_map(mid, h, since=since))
    if cached is None and cache_dir:
        cached = read_cached(Path(cache_dir), map_id)
    now = time.time()

    if cached is not None and cached.age_seconds < max_age_seconds:
        return cached

    needs_membership = (
        cached is None
        or cached.token is None
        or (now - cached.membership_at) >= FULL_REFRESH_SECONDS
    )

    try:
        if not needs_membership:
            # The cheap path: ~84 bytes to learn whether anything moved.
            probe = fetch(map_id, host, cached.token)
            if not _changed(probe):
                refreshed = MapState(
                    map_id=map_id,
                    features=cached.features,
                    token=_token_of(probe) or cached.token,
                    fetched_at=now,
                    membership_at=cached.membership_at,
                    source="cache-revalidated",
                )
                if cache_dir:
                    write_cached(cache_dir, refreshed)
                return refreshed

        payload = fetch(map_id, host, 0)
        state = MapState(
            map_id=map_id,
            features=extract_features(payload),
            token=_token_of(payload),
            fetched_at=now,
            membership_at=now,
            source="network",
        )
        if cache_dir:
            write_cached(cache_dir, state)
        return state

    except Exception as error:
        if cached is None:
            raise
        # Offline with a copy in hand: hand it over, labelled. Refusing to
        # answer at a command post is worse than answering with an age
        # attached - provided the age is attached.
        return MapState(
            map_id=cached.map_id,
            features=cached.features,
            token=cached.token,
            fetched_at=cached.fetched_at,
            membership_at=cached.membership_at,
            stale=True,
            reason=f"refresh failed ({type(error).__name__}); serving cached copy",
            source="cache",
        )
