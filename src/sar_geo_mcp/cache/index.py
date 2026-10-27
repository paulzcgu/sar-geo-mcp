"""An index of what the cache holds, how big it is, and how old.

WHY THIS EXISTS
---------------
Before this, nothing recorded when a cached file was fetched, so nothing
could decide whether it was stale, report its age to a caller, or evict the
least useful thing when a field laptop filled up. The files carried no
history at all.

WHY IT IS ALLOWED TO FAIL
-------------------------
The index is metadata, not truth - the truth is the file on disk. A corrupt
or unwritable index must never turn a working download into a failed one,
so every public function here swallows database errors and carries on. The
cost of a lost row is a slightly wrong size report; the cost of a raised
exception would be a failed fetch in the field.

TIERS
-----
`static` is ground that does not move - elevation, vegetation, roads,
basemap imagery. Keyed by geography, never expires by time. `mutable` is
CalTopo map state, which changes during an operation and expires by age.
They are counted separately because they are evicted by different rules:
static by size, mutable by time.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

INDEX_FILENAME = "index.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    key            TEXT PRIMARY KEY,
    layer          TEXT NOT NULL,
    tier           TEXT NOT NULL,
    bytes          INTEGER NOT NULL,
    source_version TEXT,
    fetched_at     REAL NOT NULL,
    last_used      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS entries_by_use ON entries (tier, last_used);
"""


@dataclass(frozen=True)
class CacheEntry:
    key: str
    layer: str
    tier: str
    bytes: int
    source_version: str | None
    fetched_at: float
    last_used: float

    @property
    def age_seconds(self) -> float:
        return max(0.0, time.time() - self.fetched_at)


class CacheIndex:
    """Metadata for one cache directory.

    Connections are opened per operation rather than held. The debrief UI
    runs prepare jobs in a threadpool beside the server, and a long-lived
    connection shared across threads is a locking problem waiting to
    happen; these writes are small and infrequent enough that the open cost
    does not matter.
    """

    def __init__(self, cache_dir: Path | str):
        self.cache_dir = Path(cache_dir)
        self.path = self.cache_dir / INDEX_FILENAME

    # -- plumbing ---------------------------------------------------------

    @contextmanager
    def _connect(self):
        """A connection, or None when the index is unusable.

        Yielding None rather than raising is what keeps a broken index from
        breaking a download - see the module docstring.
        """
        connection = None
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, timeout=5.0)
            connection.executescript(_SCHEMA)
            yield connection
            connection.commit()
        except (sqlite3.Error, OSError):
            yield None
        finally:
            if connection is not None:
                try:
                    connection.close()
                except sqlite3.Error:
                    pass

    def _key_for(self, path: Path | str) -> str:
        """A cache-relative key, so the index survives the directory moving.

        A field bundle is unzipped to a different absolute path on every
        laptop; storing absolute paths would orphan every row on the first
        machine that was not the developer's.
        """
        path = Path(path)
        try:
            return path.relative_to(self.cache_dir).as_posix()
        except ValueError:
            return path.name

    # -- recording --------------------------------------------------------

    def record(self, path: Path | str, layer: str, tier: str = "static",
               source_version: str | None = None) -> None:
        """Note that a file was fetched and written."""
        path = Path(path)
        try:
            size = path.stat().st_size
        except OSError:
            return
        now = time.time()
        with self._connect() as connection:
            if connection is None:
                return
            try:
                connection.execute(
                    "INSERT INTO entries "
                    "(key, layer, tier, bytes, source_version, fetched_at, last_used) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET "
                    "layer=excluded.layer, tier=excluded.tier, "
                    "bytes=excluded.bytes, source_version=excluded.source_version, "
                    "fetched_at=excluded.fetched_at, last_used=excluded.last_used",
                    (self._key_for(path), layer, tier, size, source_version, now, now),
                )
            except sqlite3.Error:
                return

    def touch(self, path: Path | str) -> None:
        """Note a cache hit. Feeds LRU eviction; never creates a row.

        A hit on a file the index has never seen - written before the index
        existed, or by an older version - stays unrecorded rather than
        being invented with a fabricated `fetched_at`. A made-up age is
        worse than a missing one, because something downstream will believe
        it.
        """
        with self._connect() as connection:
            if connection is None:
                return
            try:
                connection.execute(
                    "UPDATE entries SET last_used = ? WHERE key = ?",
                    (time.time(), self._key_for(path)),
                )
            except sqlite3.Error:
                return

    def forget(self, path: Path | str) -> None:
        with self._connect() as connection:
            if connection is None:
                return
            try:
                connection.execute("DELETE FROM entries WHERE key = ?",
                                   (self._key_for(path),))
            except sqlite3.Error:
                return

    # -- reading ----------------------------------------------------------

    _COLUMNS = ("SELECT key, layer, tier, bytes, source_version, fetched_at, "
                "last_used FROM entries")

    def entry(self, path: Path | str) -> CacheEntry | None:
        with self._connect() as connection:
            if connection is None:
                return None
            try:
                row = connection.execute(
                    f"{self._COLUMNS} WHERE key = ?", (self._key_for(path),)
                ).fetchone()
            except sqlite3.Error:
                return None
        return CacheEntry(*row) if row else None

    def entries(self) -> list[CacheEntry]:
        with self._connect() as connection:
            if connection is None:
                return []
            try:
                rows = connection.execute(
                    f"{self._COLUMNS} ORDER BY last_used DESC").fetchall()
            except sqlite3.Error:
                return []
        return [CacheEntry(*row) for row in rows]

    def summary(self) -> dict:
        """Totals per tier and per layer - what `cache_status` will report."""
        entries = self.entries()
        by_layer: dict[str, dict] = {}
        by_tier: dict[str, dict] = {}
        for entry in entries:
            for bucket, name in ((by_layer, entry.layer), (by_tier, entry.tier)):
                slot = bucket.setdefault(name, {"files": 0, "bytes": 0})
                slot["files"] += 1
                slot["bytes"] += entry.bytes
        return {
            "files": len(entries),
            "bytes": sum(entry.bytes for entry in entries),
            "by_tier": by_tier,
            "by_layer": by_layer,
        }


# -- convenience ----------------------------------------------------------
#
# Two lines at each fetch site is the whole integration cost. Keeping it
# that small is deliberate: bookkeeping that is awkward to call gets left
# out of the next fetcher somebody writes.

def note_write(cache_dir: Path | str, path: Path | str, layer: str,
               tier: str = "static", source_version: str | None = None) -> None:
    CacheIndex(cache_dir).record(path, layer, tier, source_version)


def note_hit(cache_dir: Path | str, path: Path | str) -> None:
    CacheIndex(cache_dir).touch(path)
