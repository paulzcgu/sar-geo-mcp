"""One reader for CalTopo map state, and the held session behind it.

Every session here is a fake shaped like `caltopo_python.CaltopoSession`
- the attributes the reader depends on are pinned against the real class
in `test_the_library_still_has_what_the_reader_depends_on`.
"""

from __future__ import annotations

import copy

import pytest

from sar_geo_mcp.caltopo.reader import MapReader, WatchedMap
from sar_geo_mcp.caltopo.state import read_cached
from sar_geo_mcp.config import CaltopoConfig

FEATURES = [
    {"id": "a1", "properties": {"class": "Assignment", "letter": "QA",
                                "number": "103"},
     "geometry": {"type": "Point", "coordinates": [-122.31, 37.41]}},
]


class FakeSession:
    """Just enough of `CaltopoSession` for the snapshot protocol."""

    def __init__(self, features=FEATURES, ts_local=1_000_000):
        self.mapID = "M1"
        self.sync = True
        self.syncing = False
        self.lastSuccessfulSyncTSLocal = ts_local
        self.mapData = {"ids": {}, "state": {"features": copy.deepcopy(features)}}
        self.paused = False
        self.opened = []

    def _pause(self):
        self.paused = True

    def _resume(self):
        self.paused = False

    def openMap(self, map_id):
        self.opened.append(map_id)
        return True

    def closeMap(self):
        self.mapID = None
        self.sync = False


class Clock:
    def __init__(self, now=1_700_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


class Response:
    def __init__(self, status_code):
        self.status_code = status_code


# ---------------------------------------------------------------------------
# The snapshot protocol.
# ---------------------------------------------------------------------------

def test_a_read_is_a_copy_of_the_libraries_features_not_the_list_itself():
    """The library merges into `mapData` in place from its own thread, so a
    tool must never hold the live list."""
    session = FakeSession()
    watched = WatchedMap("M1", session, interval=30, clock=Clock())

    state = watched.read()

    assert state.features == FEATURES
    assert state.features is not session.mapData["state"]["features"]
    assert state.source == "watched"
    assert state.fetched_at == pytest.approx(1_000.0)
    assert not session.paused, "sync must be resumed after the copy"


def test_the_copy_is_taken_once_per_poll_not_once_per_read():
    session = FakeSession()
    watched = WatchedMap("M1", session, interval=30, clock=Clock())

    first = watched.read()
    session.mapData["state"]["features"].append({"id": "late"})
    second = watched.read()

    assert second.features is first.features, "same poll, same snapshot"
    session.lastSuccessfulSyncTSLocal += 30_000
    third = watched.read()
    assert [f["id"] for f in third.features] == ["a1", "late"]


def test_a_poll_that_lands_mid_copy_makes_the_reader_copy_again(monkeypatch):
    """The library's `syncing` flag is the sequence lock: if the timestamp
    moved while we copied, a merge slipped in and the copy is torn."""
    session = FakeSession()
    calls = {"n": 0}
    original_deepcopy = copy.deepcopy

    def racing_deepcopy(value, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            session.lastSuccessfulSyncTSLocal += 1  # a merge completed underneath
        return original_deepcopy(value, *args, **kwargs)

    monkeypatch.setattr(copy, "deepcopy", racing_deepcopy)
    watched = WatchedMap("M1", session, interval=30, clock=Clock())
    state = watched.read()

    assert calls["n"] == 2, "the torn copy must be thrown away"
    assert state.fetched_at == pytest.approx(session.lastSuccessfulSyncTSLocal / 1000)


def test_nothing_synced_yet_raises_rather_than_serving_an_empty_map():
    """`lastSuccessfulSyncTSLocal` is 0 until the first poll succeeds. An
    empty feature list at that moment is not "an empty map"."""
    session = FakeSession(ts_local=0)
    watched = WatchedMap("M1", session, interval=30, clock=Clock())

    with pytest.raises(RuntimeError, match="no sync has completed"):
        watched.read()


# ---------------------------------------------------------------------------
# Disconnected and closed maps are served marked, and reopened.
# ---------------------------------------------------------------------------

def test_a_disconnected_session_serves_the_snapshot_marked_stale():
    session = FakeSession()
    clock = Clock()
    watched = WatchedMap("M1", session, interval=30, clock=clock)
    watched.read()

    watched.on_disconnected()
    state = watched.read()
    assert state.stale is True
    assert "disconnected since" in state.reason

    watched.on_reconnected()
    assert watched.read().stale is False


def test_a_closed_map_keeps_serving_the_last_snapshot_marked_with_the_status():
    session = FakeSession()
    clock = Clock()
    watched = WatchedMap("M1", session, interval=30, clock=clock)
    before = watched.read()

    session.closeMap()                       # what the library does on a 401
    watched.on_map_closed(Response(401))
    state = watched.read()

    assert state.features == before.features
    assert state.stale is True
    assert "HTTP 401" in state.reason
    assert session.opened == [], "no reopen before the backoff has elapsed"


def test_a_closed_map_is_reopened_after_the_backoff_on_the_same_session():
    session = FakeSession()
    clock = Clock()
    watched = WatchedMap("M1", session, interval=30, clock=clock)
    watched.read()
    session.closeMap()
    watched.on_map_closed(Response(401))

    clock.now += 31
    session.mapID = None                     # openMap will restore it

    def reopen(map_id):
        session.opened.append(map_id)
        session.mapID = map_id
        session.lastSuccessfulSyncTSLocal += 5_000
        return True

    session.openMap = reopen
    state = watched.read()

    assert session.opened == ["M1"]
    assert session.sync is True, "closeMap turned sync off; openMap needs it on"
    assert state.stale is False
    assert state.fetched_at == pytest.approx(1_005.0)


def test_a_refused_reopen_backs_off_twice_as_long_up_to_the_cap():
    session = FakeSession()
    clock = Clock()
    watched = WatchedMap("M1", session, interval=30, clock=clock)
    watched.read()
    session.closeMap()
    watched.on_map_closed(Response(401))

    def refuse(map_id):
        session.opened.append(map_id)
        watched.on_map_closed(Response(401))   # the library closes it again
        return False

    session.openMap = refuse

    waits = []
    for _ in range(6):
        clock.now = watched.next_reopen_at
        watched.read()
        waits.append(round(watched.next_reopen_at - clock.now))

    assert len(session.opened) == 6
    assert waits == [60, 120, 240, 300, 300, 300]
    assert watched.read().stale is True


def test_a_map_closed_for_over_an_hour_points_at_a_rotated_key():
    session = FakeSession()
    clock = Clock()
    watched = WatchedMap("M1", session, interval=30, clock=clock)
    watched.read()
    session.closeMap()
    watched.on_map_closed(Response(401))
    session.openMap = lambda map_id: False

    clock.now += 3601
    assert "restart the server" in watched.read().reason


# ---------------------------------------------------------------------------
# MapReader - the public path with an in-memory prior.
# ---------------------------------------------------------------------------

def full_reply(features=FEATURES, token=1_787_989_336_147):
    return {"status": "ok", "result": {
        "ids": {"Assignment": ["a1"]},
        "state": {"type": "FeatureCollection", "features": features},
        "timestamp": token,
    }}


def empty_reply(token=1_787_989_349_855):
    return {"status": "ok", "result": {
        "state": {"type": "FeatureCollection", "features": []},
        "timestamp": token,
    }}


class Recorder:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[int] = []

    def __call__(self, map_id, host, since=0):
        self.calls.append(since)
        if not self.replies:
            raise AssertionError(f"unexpected fetch (since={since})")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def public(monkeypatch, tmp_path):
    """A reader whose public fetches are recorded and never networked."""
    def make(*replies):
        recorder = Recorder(*replies)
        monkeypatch.setattr("sar_geo_mcp.caltopo.reader.session_module.fetch_public_map",
                            recorder)
        return MapReader(CaltopoConfig(cache_dir=tmp_path / "cache")), recorder
    return make


def test_a_public_read_inside_the_window_is_served_from_memory(public):
    reader, recorder = public(full_reply())

    first = reader.read("M1", public=True)
    second = reader.read("M1", public=True)

    assert recorder.calls == [0]
    assert second.features is first.features
    assert second.source == "cache"


def test_a_public_revalidation_probes_from_the_prior_and_does_not_rewrite_the_disk(
        public, tmp_path):
    """The 2.75 MB file was rewritten on every computation just to roll a
    token. The prior lives in memory now; the disk copy is for restarts."""
    reader, recorder = public(full_reply(), empty_reply())
    reader.read("M1", public=True)
    path = tmp_path / "cache" / "caltopo-state" / "M1.json"
    written_at = path.stat().st_mtime_ns

    state = reader.read("M1", public=True, max_age_seconds=0)

    assert recorder.calls == [0, 1_787_989_336_147]
    assert state.source == "cache-revalidated"
    assert path.stat().st_mtime_ns == written_at, "cheap path must not rewrite"


def test_a_public_full_refetch_does_reach_the_disk(public, tmp_path):
    moved = dict(FEATURES[0], geometry={"type": "Point", "coordinates": [0, 0]})
    reader, recorder = public(
        full_reply(),
        {"result": {"state": {"features": [moved]}, "timestamp": 2}},
        full_reply([moved], token=3),
    )
    reader.read("M1", public=True)
    state = reader.read("M1", public=True, max_age_seconds=0)

    assert state.source == "network"
    assert read_cached(tmp_path / "cache", "M1").token == 3


def test_a_fresh_reader_starts_from_the_disk_copy(public, tmp_path):
    reader, _ = public(full_reply())
    reader.read("M1", public=True)

    again, recorder = public(empty_reply())
    state = again.read("M1", public=True, max_age_seconds=0)

    assert recorder.calls == [1_787_989_336_147], "probe from the disk copy's token"
    assert len(state.features) == 1


def test_a_public_read_offline_with_a_prior_is_served_marked(public):
    reader, _ = public(full_reply())
    reader.read("M1", public=True)
    reader_offline, _ = public(OSError("no signal"))
    reader_offline._priors = reader._priors

    state = reader_offline.read("M1", public=True, max_age_seconds=0)
    assert state.stale is True
    assert "OSError" in state.reason


# ---------------------------------------------------------------------------
# MapReader - one held session per map, proven against the real class.
# ---------------------------------------------------------------------------

@pytest.fixture
def credentials(tmp_path):
    ini = tmp_path / "cts.ini"
    ini.write_text("[you@example.com]\nid=x\nkey=y\naccountId=z\n", encoding="utf-8")
    return CaltopoConfig(caltopo_config_path=ini, caltopo_account="you@example.com",
                         cache_dir=tmp_path / "cache")


def test_a_watched_map_is_opened_once_and_held(credentials, monkeypatch):
    opened = []

    class Session(FakeSession):
        def __init__(self, host, map_id, **kwargs):
            super().__init__()
            self.mapID = map_id
            self.kwargs = kwargs
            opened.append(self)

    monkeypatch.setattr("sar_geo_mcp.caltopo.reader.session_module.CaltopoSession", Session)
    reader = MapReader(credentials, sync_interval=30)

    for _ in range(5):
        state = reader.read("M1", public=False)

    assert len(opened) == 1
    assert state.features == FEATURES
    session = opened[0]
    assert session.kwargs["sync"] is True
    assert session.kwargs["syncTimeout"] == 60
    assert session.kwargs["syncInterval"] == 30
    for name in ("disconnectedCallback", "reconnectedCallback", "mapClosedCallback",
                 "propertyUpdateCallback", "geometryUpdateCallback",
                 "newFeatureCallback", "deletedFeatureCallback"):
        assert callable(session.kwargs[name]), name


def test_the_disk_copy_is_written_for_the_first_snapshot_and_after_a_change(
        credentials, monkeypatch, tmp_path):
    class Session(FakeSession):
        def __init__(self, host, map_id, **kwargs):
            super().__init__()
            self.mapID = map_id
            self.on_changed = kwargs["newFeatureCallback"]

    monkeypatch.setattr("sar_geo_mcp.caltopo.reader.session_module.CaltopoSession", Session)
    reader = MapReader(credentials)
    path = tmp_path / "cache" / "caltopo-state" / "M1.json"

    reader.read("M1", public=False)
    assert path.exists(), "a restart needs an offline copy"
    first = path.stat().st_mtime_ns

    session = next(iter(reader._watched.values())).session
    session.lastSuccessfulSyncTSLocal += 30_000       # a quiet poll
    reader.read("M1", public=False)
    assert path.stat().st_mtime_ns == first, "nothing changed, nothing written"

    session.on_changed({"id": "new"})                  # the library saw a change
    session.mapData["state"]["features"].append({"id": "new"})
    session.lastSuccessfulSyncTSLocal += 30_000
    reader.read("M1", public=False)
    assert [f["id"] for f in read_cached(tmp_path / "cache", "M1").features] == ["a1", "new"]


def test_holding_the_real_session_class_leaks_no_thread_per_read(credentials, monkeypatch):
    """The regression the whole module exists for, against the real class.

    `_setupSession` and `openMap` are patched so nothing touches the
    network; `__init__` still starts the library's requestThread, which is
    the thread that used to leak once per read. Twenty reads must leave
    the thread count where one open left it.
    """
    import gc
    import threading

    from caltopo_python import CaltopoSession

    def fake_setup(self):
        return True

    def fake_open(self, mapID="", **kwargs):
        self.mapID = mapID
        self.lastSuccessfulSyncTSLocal = 1_000_000
        self.mapData = {"ids": {}, "state": {"features": copy.deepcopy(FEATURES)}}
        return True

    monkeypatch.setattr(CaltopoSession, "_setupSession", fake_setup)
    monkeypatch.setattr(CaltopoSession, "openMap", fake_open)
    monkeypatch.setattr(CaltopoSession, "_start", lambda self: None)

    reader = MapReader(credentials)
    reader.read("M1", public=False)
    gc.collect()
    threads_after_one_open = threading.active_count()

    for _ in range(20):
        reader.read("M1", public=False)
    gc.collect()

    assert threading.active_count() == threads_after_one_open
    alive = [o for o in gc.get_objects() if isinstance(o, CaltopoSession)]
    assert len(alive) == 1


def test_the_library_still_has_what_the_reader_depends_on():
    """The snapshot protocol reads attributes the library does not
    advertise. A rename in `caltopo_python` must fail here, not in the
    field as a torn copy or a silently empty map."""
    import inspect

    from caltopo_python import CaltopoSession

    for name in ("_pause", "_resume", "openMap", "closeMap", "getFeatures"):
        assert callable(getattr(CaltopoSession, name)), name

    source = inspect.getsource(CaltopoSession)
    for attribute in ("self.syncing", "self.lastSuccessfulSyncTSLocal",
                      "self.disconnectedFlag", "self.mapID", "self.sync=",
                      "self.requestThread"):
        assert attribute in source, attribute

    parameters = inspect.signature(CaltopoSession.__init__).parameters
    for keyword in ("sync", "syncInterval", "syncTimeout",
                    "disconnectedCallback", "reconnectedCallback",
                    "mapClosedCallback", "propertyUpdateCallback",
                    "geometryUpdateCallback", "newFeatureCallback",
                    "deletedFeatureCallback"):
        assert keyword in parameters, keyword
