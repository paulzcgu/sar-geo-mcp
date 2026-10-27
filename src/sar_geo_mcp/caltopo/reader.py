"""One reader for CalTopo map state; the held session behind private maps.

WHY A HELD SESSION
------------------
`caltopo_python.CaltopoSession.__init__` starts a daemon `requestThread`
before it reads a single credential, and that thread waits on an event
for the life of the process. Its target is a bound method, so it keeps
the session alive, and the session keeps the whole map and a
`requests.Session`. Nothing ends it - not `closeMap()`, not `sync=False`,
not dropping the object. Measured with the network stubbed: twenty
sessions opened and dropped left twenty threads and 88 MB behind
(2026-09-07). So a session is opened ONCE per host, account and map and
held for the life of the process, with the library's own sync loop
keeping it current.

WHY SNAPSHOTS
-------------
The library merges into `mapData` in place from its sync thread, and
`getFeatures()` hands back that dict by reference. Its `syncCallback`
fires BEFORE the merge, so it is no use as a consistent point. The
snapshot is taken on the read side instead, with the library's `syncing`
flag (set at the start of every merge, cleared in a `finally`) as a
sequence lock: pause the loop, wait out any merge in progress, copy,
and confirm nothing moved during the copy. Tools only ever see the copy.

WHAT A 401 DOES
---------------
The library answers a 401 by closing the map: map id None, sync off,
cache emptied, and from then on `getFeatures()` returns an empty list
with only a log line. That must never reach a caller as "empty map".
The last snapshot is served, marked stale with the status, and the map is
reopened on the same session with backoff - no new session, so nothing
leaks.
"""

from __future__ import annotations

import copy
import dataclasses
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from . import session as session_module
from .state import (
    DEFAULT_MAX_AGE_SECONDS,
    MapState,
    fetch_map_state,
    read_cached,
    write_cached,
)

#: How often a held session polls `/since/`. The library defaults to 10 s,
#: which is more than a tool server needs; every answer carries its age.
WATCH_INTERVAL_SECONDS = 30.0

#: Per-request timeout for a held session, including the initial full
#: fetch. Matches the public path (`fetch_public_map`).
WATCH_TIMEOUT_SECONDS = 60

#: After the library closes a map on a 401: first retry after this long,
#: doubling on each refusal, capped.
REOPEN_BACKOFF_SECONDS = 30.0
REOPEN_BACKOFF_CAP_SECONDS = 300.0

#: A map closed for this long is more likely a rotated key than a blip,
#: and credentials are only read when a session is constructed.
ROTATED_KEY_HINT_SECONDS = 3600.0

#: Attempts to take a consistent copy before giving up. A merge takes
#: milliseconds; twenty torn copies in a row means something else is wrong.
SNAPSHOT_ATTEMPTS = 20


def _iso(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%H:%M:%SZ")


class WatchedMap:
    """One held, syncing session for one map, read through snapshots.

    `session` is a `caltopo_python.CaltopoSession` opened with `sync=True`
    and this object's `on_*` methods as callbacks. The callbacks run in
    the library's sync thread and only assign attributes; they never take
    `lock`, so they cannot deadlock against a read in progress.
    """

    def __init__(self, map_id: str, session, *, interval: float,
                 clock=time.time):
        self.map_id = map_id
        self.session = session
        self.interval = interval
        self.clock = clock
        self.lock = threading.Lock()
        self.snapshot: MapState | None = None
        self.snapshot_ts_local: int = -1
        #: Feature-change callbacks fired since the disk copy was written.
        self.dirty = False
        self.written_ts_local: int = -1
        self.disconnected_at: float | None = None
        self.closed_at: float | None = None
        self.closed_reason = ""
        self.backoff = REOPEN_BACKOFF_SECONDS
        self.next_reopen_at = 0.0

    # -- callbacks, run in the library's sync thread -----------------------

    def on_changed(self, *_args) -> None:
        self.dirty = True

    def on_disconnected(self) -> None:
        self.disconnected_at = self.clock()

    def on_reconnected(self) -> None:
        self.disconnected_at = None

    def on_map_closed(self, bad_response=None) -> None:
        status = getattr(bad_response, "status_code", None)
        self.closed_at = self.clock()
        self.closed_reason = (f"CalTopo closed the map (HTTP {status})"
                              if status else "CalTopo closed the map")
        self.next_reopen_at = self.closed_at + self.backoff

    # -- reading ----------------------------------------------------------

    def read(self) -> MapState:
        """The latest consistent snapshot, marked stale if the session is
        disconnected or the map was closed underneath it."""
        with self.lock:
            self._maybe_reopen()
            self._refresh_snapshot()
            if self.snapshot is None:
                raise RuntimeError(
                    f"map {self.map_id}: nothing to serve - "
                    f"{self.closed_reason or 'no sync has completed yet'}")
            return self._marked(self.snapshot)

    def take_dirty(self) -> bool:
        """True once per snapshot that should reach the disk cache: the
        first one after a start, and any after a feature changed."""
        with self.lock:
            if self.snapshot is None or self.snapshot_ts_local == self.written_ts_local:
                return False
            if self.written_ts_local != -1 and not self.dirty:
                return False
            self.dirty = False
            self.written_ts_local = self.snapshot_ts_local
            return True

    def _refresh_snapshot(self) -> None:
        session = self.session
        if getattr(session, "mapID", None) is None:
            return  # closed: keep serving the last snapshot
        session._pause()
        try:
            for _attempt in range(SNAPSHOT_ATTEMPTS):
                while session.syncing:
                    time.sleep(0.01)
                ts_local = int(session.lastSuccessfulSyncTSLocal)
                if ts_local <= 0 or ts_local == self.snapshot_ts_local:
                    return
                features = copy.deepcopy(session.mapData["state"]["features"])
                if (not session.syncing
                        and int(session.lastSuccessfulSyncTSLocal) == ts_local):
                    break
            else:
                raise RuntimeError(
                    f"map {self.map_id}: could not take a consistent snapshot "
                    f"in {SNAPSHOT_ATTEMPTS} attempts")
        finally:
            session._resume()
        fetched_at = ts_local / 1000.0
        self.snapshot = MapState(
            map_id=self.map_id, features=features, token=None,
            fetched_at=fetched_at, membership_at=fetched_at, source="watched",
        )
        self.snapshot_ts_local = ts_local

    def _marked(self, snapshot: MapState) -> MapState:
        now = self.clock()
        if self.closed_at is not None:
            reason = (f"{self.closed_reason} at {_iso(self.closed_at)}; serving "
                      f"the last snapshot; reopen in "
                      f"{max(0, round(self.next_reopen_at - now))} s")
            if now - self.closed_at > ROTATED_KEY_HINT_SECONDS:
                reason += ("; closed for over an hour - if the key was rotated, "
                           "restart the server")
        elif self.disconnected_at is not None:
            reason = (f"watched session disconnected since "
                      f"{_iso(self.disconnected_at)}; serving the last snapshot")
        else:
            return snapshot
        return dataclasses.replace(snapshot, stale=True, reason=reason)

    def _maybe_reopen(self) -> None:
        if self.closed_at is None or self.clock() < self.next_reopen_at:
            return
        session = self.session
        # closeMap() turned sync off, and openMap only restarts the loop
        # when it is on.
        session.sync = True
        try:
            opened = bool(session.openMap(self.map_id))
        except Exception:
            opened = False
        if opened and getattr(session, "mapID", None):
            self.closed_at = None
            self.closed_reason = ""
            self.backoff = REOPEN_BACKOFF_SECONDS
            return
        # Still refused (a 401 inside openMap closes it again and fires
        # on_map_closed, which reset closed_at). Back off further.
        self.backoff = min(self.backoff * 2, REOPEN_BACKOFF_CAP_SECONDS)
        self.next_reopen_at = self.clock() + self.backoff


class MapReader:
    """The one thing that reads CalTopo for a process.

    `read(map_id, public=...)` is the whole interface. Public maps go
    through `fetch_map_state`'s token protocol with the last state held
    in memory as the prior, so a revalidation costs one 84-byte probe and
    no disk I/O. Private maps go through one `WatchedMap` per host,
    account and map, created on first use and held for the life of the
    process.
    """

    def __init__(self, config, *, sync_interval: float = WATCH_INTERVAL_SECONDS,
                 clock=time.time):
        self.config = config
        self.sync_interval = sync_interval
        self.clock = clock
        self._lock = threading.Lock()
        self._priors: dict[str, MapState] = {}
        self._watched: dict[tuple[str, str, str], WatchedMap] = {}

    @property
    def cache_dir(self) -> Path | None:
        return self.config.cache_dir

    def read(self, map_id: str, *, public: bool,
             max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS) -> MapState:
        """Current map state by whichever path the caller is entitled to.

        `max_age_seconds` is the listing window for a public map, exactly
        as in `fetch_map_state`. For a watched map the check is continuous
        rather than per call: the snapshot is never older than one poll
        interval while the session is connected, and its age is reported
        rather than forced to zero. Older than that means disconnected,
        which the snapshot already says.
        """
        if public:
            return self._read_public(map_id, max_age_seconds)
        return self._read_watched(map_id)

    def close(self) -> None:
        """Stop every held session's polling. Tests and long-lived
        consumers call this; the server's daemon threads die with it."""
        with self._lock:
            watched = list(self._watched.values())
            self._watched.clear()
        for item in watched:
            try:
                item.session.closeMap()
            except Exception:
                pass

    # -- public ------------------------------------------------------------

    def _read_public(self, map_id: str, max_age_seconds: float) -> MapState:
        cache_dir = self.config.cache_dir
        prior = self._priors.get(map_id)
        if prior is None and cache_dir:
            prior = read_cached(Path(cache_dir), map_id)

        def fetch(mid, host, since):
            return session_module.fetch_public_map(mid, host, since=since)

        state = fetch_map_state(
            map_id, self.config.caltopo_host, cache_dir=None,
            max_age_seconds=max_age_seconds, fetch=fetch, cached=prior,
        )
        if state is prior:
            # Served from the window without a fetch. Say "cache" as the
            # disk path always has, whichever memory it came from.
            state = dataclasses.replace(state, source="cache")
        self._priors[map_id] = state
        if cache_dir and state.source == "network":
            write_cached(cache_dir, state)
        return state

    # -- watched -----------------------------------------------------------

    def _read_watched(self, map_id: str) -> MapState:
        key = (self.config.caltopo_host, self.config.caltopo_account or "", map_id)
        watched = self._watched.get(key)
        if watched is None:
            with self._lock:
                watched = self._watched.get(key)
                if watched is None:
                    watched = self._open_watched(map_id)
                    self._watched[key] = watched
        state = watched.read()
        if self.config.cache_dir and watched.take_dirty():
            write_cached(self.config.cache_dir, state)
        return state

    def _open_watched(self, map_id: str) -> WatchedMap:
        watched = WatchedMap(map_id, None, interval=self.sync_interval,
                             clock=self.clock)
        watched.session = session_module.open_map(
            map_id, self.config,
            sync=True,
            sync_interval=int(self.sync_interval),
            sync_timeout=WATCH_TIMEOUT_SECONDS,
            disconnectedCallback=watched.on_disconnected,
            reconnectedCallback=watched.on_reconnected,
            mapClosedCallback=watched.on_map_closed,
            propertyUpdateCallback=watched.on_changed,
            geometryUpdateCallback=watched.on_changed,
            newFeatureCallback=watched.on_changed,
            deletedFeatureCallback=watched.on_changed,
        )
        return watched
