"""Cached CalTopo map state (F2).

Every fetch is injected. The behaviour under test is the protocol - when
we go to the network, what we do with the reply, and what we tell the
caller about the age of what they got - not CalTopo's uptime.

The numbers driving the design came from a real map: a full fetch was
2 752 019 bytes for 85 features; the unchanged incremental reply was 84.
"""

from __future__ import annotations

import time

import pytest

from sar_geo_mcp.caltopo.state import (
    DEFAULT_MAX_AGE_SECONDS,
    FULL_REFRESH_SECONDS,
    MapState,
    fetch_map_state,
    read_cached,
    write_cached,
)

FEATURES = [
    {"id": "a1", "properties": {"class": "Assignment", "letter": "QA",
                                "number": "103"},
     "geometry": {"type": "Point", "coordinates": [-122.31, 37.41]}},
    {"id": "m1", "properties": {"class": "Marker", "title": "CP"},
     "geometry": {"type": "Point", "coordinates": [-122.30, 37.40]}},
]


def full_reply(features=FEATURES, token=1_787_989_336_147):
    """What `since/0` returns: features, a token, and an `ids` membership map."""
    return {"status": "ok", "result": {
        "ids": {"Assignment": ["a1"], "Marker": ["m1"]},
        "state": {"type": "FeatureCollection", "features": features},
        "timestamp": token,
    }}


def empty_reply(token=1_787_989_349_855):
    """What `since/{token}` returns when nothing moved. Note `ids` is absent -
    which is exactly why this reply cannot be merged, only detected on."""
    return {"status": "ok", "result": {
        "state": {"type": "FeatureCollection", "features": []},
        "timestamp": token,
    }}


class Recorder:
    """A fetch stub that remembers what it was asked for."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[int] = []

    def __call__(self, map_id, host, since):
        self.calls.append(since)
        if not self.replies:
            raise AssertionError(f"unexpected fetch (since={since})")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


# ---------------------------------------------------------------------------
# The cheap path.
# ---------------------------------------------------------------------------

def test_first_fetch_is_a_full_fetch_and_gets_cached(tmp_path):
    fetch = Recorder(full_reply())
    state = fetch_map_state("M1", cache_dir=tmp_path, fetch=fetch)

    assert fetch.calls == [0], "no token yet, so it must ask for everything"
    assert len(state.features) == 2
    assert state.token == 1_787_989_336_147
    assert state.source == "network"
    assert read_cached(tmp_path, "M1") is not None


def test_a_fresh_cache_is_served_without_touching_the_network(tmp_path):
    fetch_map_state("M1", cache_dir=tmp_path, fetch=Recorder(full_reply()))

    served = fetch_map_state("M1", cache_dir=tmp_path, fetch=Recorder())
    assert served.source == "cache"
    assert len(served.features) == 2


def test_past_the_window_it_probes_with_the_token_not_a_full_fetch(tmp_path):
    """The 30 000x saving: an unchanged map costs 84 bytes, not 2.75 MB."""
    fetch_map_state("M1", cache_dir=tmp_path, fetch=Recorder(full_reply()))

    probe = Recorder(empty_reply())
    state = fetch_map_state("M1", cache_dir=tmp_path, max_age_seconds=0,
                            fetch=probe)

    assert probe.calls == [1_787_989_336_147], "must ask since the known token"
    assert state.source == "cache-revalidated"
    assert len(state.features) == 2


def test_the_rolled_token_is_used_on_the_next_probe(tmp_path):
    """Each probe returns a newer token; failing to store it would make every
    later probe re-report the same window of changes."""
    fetch_map_state("M1", cache_dir=tmp_path, fetch=Recorder(full_reply()))
    fetch_map_state("M1", cache_dir=tmp_path, max_age_seconds=0,
                    fetch=Recorder(empty_reply(token=999)))

    probe = Recorder(empty_reply())
    fetch_map_state("M1", cache_dir=tmp_path, max_age_seconds=0, fetch=probe)
    assert probe.calls == [999]


# ---------------------------------------------------------------------------
# Change detection, not merging.
# ---------------------------------------------------------------------------

def test_any_incremental_content_triggers_a_full_refetch(tmp_path):
    """The incremental reply is a signal, never a source.

    It carries no `ids`, so it cannot say what was DELETED; merging from it
    would keep a removed assignment on the map forever.
    """
    fetch_map_state("M1", cache_dir=tmp_path, fetch=Recorder(full_reply()))

    moved = dict(FEATURES[0], geometry={"type": "Point",
                                        "coordinates": [-122.0, 37.0]})
    fetch = Recorder(
        {"result": {"state": {"features": [moved]}, "timestamp": 2}},  # probe
        full_reply([moved], token=3),                                   # refetch
    )
    state = fetch_map_state("M1", cache_dir=tmp_path, max_age_seconds=0,
                            fetch=fetch)

    assert fetch.calls == [1_787_989_336_147, 0], "probe, then full refetch"
    assert state.source == "network"
    assert len(state.features) == 1, "replaced wholesale, not merged"


def test_a_deletion_is_seen_because_the_refetch_replaces_wholesale(tmp_path):
    """The failure this protocol exists to prevent: a deleted assignment
    lingering in the cache and being searched against."""
    fetch_map_state("M1", cache_dir=tmp_path, fetch=Recorder(full_reply()))

    fetch = Recorder(
        {"result": {"state": {"features": [FEATURES[1]]}, "timestamp": 2}},
        full_reply([FEATURES[1]], token=3),
    )
    state = fetch_map_state("M1", cache_dir=tmp_path, max_age_seconds=0,
                            fetch=fetch)

    assert [f["id"] for f in state.features] == ["m1"], "a1 must be gone"


def test_membership_is_reconciled_by_a_forced_full_fetch(tmp_path):
    """A deletion that produces an EMPTY incremental reply would be invisible
    to the probe. We cannot verify CalTopo's behaviour there without mutating
    somebody's map, so the design assumes the worst and reconciles on a
    timer."""
    fetch_map_state("M1", cache_dir=tmp_path, fetch=Recorder(full_reply()))

    stale = read_cached(tmp_path, "M1")
    write_cached(tmp_path, MapState(
        map_id="M1", features=stale.features, token=stale.token,
        fetched_at=stale.fetched_at,
        membership_at=time.time() - FULL_REFRESH_SECONDS - 1,
    ))

    fetch = Recorder(full_reply(token=5))
    fetch_map_state("M1", cache_dir=tmp_path, max_age_seconds=0, fetch=fetch)
    assert fetch.calls == [0], "stale membership must force a full fetch"


# ---------------------------------------------------------------------------
# What the caller is told.
# ---------------------------------------------------------------------------

def test_provenance_always_reports_age_and_source(tmp_path):
    state = fetch_map_state("M1", cache_dir=tmp_path, fetch=Recorder(full_reply()))
    block = state.provenance()

    assert block["age_seconds"] < 5
    assert block["source"] == "network"
    assert "membership_age_seconds" in block
    assert "stale" not in block


def test_offline_serves_the_cached_copy_marked_rather_than_failing(tmp_path):
    """A command post often has no connectivity. Refusing to answer is worse
    than answering with an age attached - provided the age IS attached."""
    fetch_map_state("M1", cache_dir=tmp_path, fetch=Recorder(full_reply()))

    offline = Recorder(OSError("network is unreachable"))
    state = fetch_map_state("M1", cache_dir=tmp_path, max_age_seconds=0,
                            fetch=offline)

    assert state.stale is True
    assert len(state.features) == 2, "the cached answer still gets served"
    block = state.provenance()
    assert block["stale"] is True
    assert "OSError" in block["reason"]


def test_offline_with_nothing_cached_raises_rather_than_inventing(tmp_path):
    """No copy and no network is a real failure and must be reported as one."""
    with pytest.raises(OSError):
        fetch_map_state("M1", cache_dir=tmp_path,
                        fetch=Recorder(OSError("unreachable")))


def test_a_stale_marked_state_never_claims_to_be_fresh(tmp_path):
    fetch_map_state("M1", cache_dir=tmp_path, fetch=Recorder(full_reply()))
    state = fetch_map_state("M1", cache_dir=tmp_path, max_age_seconds=0,
                            fetch=Recorder(OSError("down")))
    assert state.source == "cache"
    assert state.provenance()["source"] != "network"


# ---------------------------------------------------------------------------
# Robustness.
# ---------------------------------------------------------------------------

def test_a_corrupt_cache_file_is_ignored_not_fatal(tmp_path):
    fetch_map_state("M1", cache_dir=tmp_path, fetch=Recorder(full_reply()))
    (tmp_path / "caltopo-state" / "M1.json").write_text("{ not json",
                                                        encoding="utf-8")

    assert read_cached(tmp_path, "M1") is None
    state = fetch_map_state("M1", cache_dir=tmp_path, fetch=Recorder(full_reply()))
    assert len(state.features) == 2


def test_a_map_id_cannot_escape_the_cache_directory(tmp_path):
    """`map_id` reaches a filename from a tool argument, so it is filtered
    rather than trusted.

    The property is containment, not rejection: separators and dots are
    stripped, so a traversal attempt collapses to a harmless name inside
    the cache directory.
    """
    from sar_geo_mcp.caltopo.state import _state_path

    path = _state_path(tmp_path, "../../etc/passwd")
    assert tmp_path.resolve() in path.resolve().parents
    assert ".." not in path.parts


def test_a_map_id_with_nothing_usable_in_it_is_refused(tmp_path):
    """Filtering to an empty name would collide every such map on one file."""
    with pytest.raises(ValueError):
        read_cached(tmp_path, "../..")


def test_working_without_a_cache_dir_at_all_still_fetches(tmp_path):
    state = fetch_map_state("M1", cache_dir=None, fetch=Recorder(full_reply()))
    assert len(state.features) == 2
    assert state.source == "network"


def test_the_default_window_is_short_enough_to_be_defensible():
    """Five minutes is already a long time on a moving incident; anything
    longer would need a much better argument than convenience."""
    assert DEFAULT_MAX_AGE_SECONDS <= 300


# ---------------------------------------------------------------------------
# An in-memory prior, for a server that watches one map all day.
# ---------------------------------------------------------------------------

def test_an_in_memory_prior_is_revalidated_without_any_disk_cache():
    """A long-running server keeps the last state itself: the probe still
    costs 84 bytes, and nothing is written anywhere."""
    prior = fetch_map_state("M1", fetch=Recorder(full_reply()))

    probe = Recorder(empty_reply())
    state = fetch_map_state("M1", max_age_seconds=0, fetch=probe, cached=prior)

    assert probe.calls == [1_787_989_336_147], "must ask since the prior's token"
    assert state.source == "cache-revalidated"
    assert state.features is prior.features
    assert state.token == 1_787_989_349_855


def test_an_in_memory_prior_wins_over_the_disk_cache(tmp_path):
    fetch_map_state("M1", cache_dir=tmp_path, fetch=Recorder(full_reply(token=1)))
    prior = fetch_map_state("M1", fetch=Recorder(full_reply(token=2)))

    probe = Recorder(empty_reply())
    fetch_map_state("M1", cache_dir=tmp_path, max_age_seconds=0, fetch=probe,
                    cached=prior)

    assert probe.calls == [2]


def test_an_in_memory_prior_is_served_marked_when_the_probe_fails():
    prior = fetch_map_state("M1", fetch=Recorder(full_reply()))

    state = fetch_map_state("M1", max_age_seconds=0, cached=prior,
                            fetch=Recorder(ConnectionError("no signal")))

    assert state.stale is True
    assert state.features is prior.features
    assert "ConnectionError" in state.reason
