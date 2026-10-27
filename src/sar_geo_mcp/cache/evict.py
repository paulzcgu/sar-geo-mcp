"""Keeping the cache under a ceiling, on a laptop that has to keep working.

WHY ONLY THE STATIC TIER
------------------------
Static ground - elevation, vegetation, roads, basemap - is what grows
without bound: it is keyed by geography, never expires, and every new
incident adds more. Map state is kilobytes and expires by age, so evicting
it by size would save nothing and could throw away the only copy of a map
a laptop currently has no signal to refetch.

WHY LEAST-RECENTLY-USED
-----------------------
The access pattern is one incident at a time. Ground from last month's
search is exactly what should go first, and ground being asked about right
now is exactly what should not. `last_used` comes from F1's index, which
is why eviction had to wait for it.

WHAT IS NEVER EVICTED
---------------------
Anything a staged assignment depends on. Deleting a tile out from under a
handle would turn the next question about that assignment into a silent
refetch at best, and at worst an error in the middle of a debrief. Staged
directories are small; the tiles they rest on are not, and those are the
ones held.

The cache does not work out which tiles those are. What a staged directory
is, and which tiles its extent maps onto, is SAR knowledge and lives in
`staging.tiles_in_use`; this module is plumbing (AGENTS.md). So
eviction takes the keys to hold as `protected`, and the `evict_cache`
component wires the two together. `protected` has no default on purpose: a
caller that forgot it would quietly evict staged ground, so every caller
has to say what it holds, even when that is nothing.

A FILE THE INDEX HAS NEVER SEEN IS NOT TOUCHED
-----------------------------------------------
It has no `last_used`, so there is no honest basis for ranking it. Deleting
something because we know nothing about it is the wrong instinct - it would
mean an upgrade that predates the index quietly wiping a laptop's whole
prefetch.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .index import CacheIndex

#: Default ceiling for the static tier. Generous, because the failure this
#: prevents - a full disk in the field - is worse than the disk space, and
#: because a prefetched incident can legitimately run to a few hundred
#: megabytes.
DEFAULT_CEILING_BYTES = 5_000_000_000


@dataclass(frozen=True)
class EvictionPlan:
    """What would be removed, before anything is."""

    evict: list[str]
    bytes_freed: int
    bytes_held: int
    protected: int
    unknown_skipped: int

    def summary(self) -> dict:
        return {
            "would_evict": len(self.evict),
            "mb_freed": round(self.bytes_freed / 1e6, 1),
            "mb_remaining": round(self.bytes_held / 1e6, 1),
            # Named for what `evict_cache` passes as `protected`. Callers
            # read this key, so it outlived the cache deciding it itself.
            "protected_by_staged_assignments": self.protected,
            "unindexed_files_left_alone": self.unknown_skipped,
            "examples": self.evict[:10],
        }


def plan_eviction(cache_dir: Path,
                  ceiling_bytes: int = DEFAULT_CEILING_BYTES, *,
                  protected: Iterable[str]) -> EvictionPlan:
    """Decide what to drop. Removes nothing.

    `protected` is the cache keys that stay whatever the ceiling - the
    tiles staged ground rests on (`staging.tiles_in_use`). Pass an empty
    collection to protect nothing.
    """
    cache_dir = Path(cache_dir)
    index = CacheIndex(cache_dir)
    entries = [entry for entry in index.entries() if entry.tier == "static"]
    total = sum(entry.bytes for entry in entries)

    held = set(protected)
    held_count = sum(1 for entry in entries if entry.key in held)

    # Oldest use first - ground from last month's search goes before ground
    # somebody is asking about now.
    candidates = [entry for entry in entries if entry.key not in held]
    candidates.sort(key=lambda entry: entry.last_used)

    evict: list[str] = []
    freed = 0
    for entry in candidates:
        if total - freed <= ceiling_bytes:
            break
        evict.append(entry.key)
        freed += entry.bytes

    unknown = sum(1 for path in cache_dir.rglob("*")
                  if path.is_file() and index.entry(path) is None)

    return EvictionPlan(evict=evict, bytes_freed=freed,
                        bytes_held=total - freed, protected=held_count,
                        unknown_skipped=unknown)


def evict(cache_dir: Path, ceiling_bytes: int = DEFAULT_CEILING_BYTES,
          execute: bool = False, *, protected: Iterable[str]) -> dict:
    """Bring the static tier under `ceiling_bytes`, never touching `protected`.

    `execute` defaults to False, so the first call says what would go.
    Deleting cached ground on a laptop that may be about to lose
    connectivity is not something to do on a guess.
    """
    cache_dir = Path(cache_dir)
    plan = plan_eviction(cache_dir, ceiling_bytes, protected=protected)
    summary = plan.summary()
    summary["ceiling_mb"] = round(ceiling_bytes / 1e6, 1)

    if not execute:
        summary["executed"] = False
        return summary

    index = CacheIndex(cache_dir)
    removed = 0
    for key in plan.evict:
        path = cache_dir / key
        try:
            path.unlink()
            index.forget(path)
            removed += 1
        except OSError:
            # A file already gone, or locked by another process. Neither is
            # worth failing an eviction over.
            continue
    summary["executed"] = True
    summary["removed"] = removed
    return summary
