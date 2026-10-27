"""Staged assignments and the binding rule (E4/F4).

The rule under test:

    Any derived product records a fingerprint of the mutable input it came
    from, and revalidates before returning a result.

This is the one caching mistake in the system with an operational
consequence. A staged directory built from a boundary that has since been
redrawn will happily produce coverage for the OLD polygon - reporting
ground as searched that nobody walked. So a listing may be slightly stale
provided it says so, and a computation may not be stale at all.
"""

from __future__ import annotations

import json

import pytest

from sar_geo_mcp.config import CaltopoConfig
from sar_geo_mcp.handles import (
    TerrainHandle,
    fingerprint_assignment,
    prepare_terrain,
    resolve,
    staged_fingerprint,
)


class FakeAssignment:
    def __init__(self, geometry, letter="QA", number="103", title="QA103"):
        self.geometry = geometry
        self.letter = letter
        self.number = number
        self.title = title
        self.label = f"{letter}{number}"

    def model_dump_json(self, indent=None):
        return json.dumps({"label": self.label, "geometry": self.geometry})


SQUARE = {"type": "Polygon", "coordinates": [[
    [-122.32, 37.40], [-122.30, 37.40], [-122.30, 37.42],
    [-122.32, 37.42], [-122.32, 37.40]]]}

MOVED = {"type": "Polygon", "coordinates": [[
    [-122.31, 37.41], [-122.29, 37.41], [-122.29, 37.43],
    [-122.31, 37.43], [-122.31, 37.41]]]}


# ---------------------------------------------------------------------------
# The fingerprint covers what a computation depends on, and nothing else.
# ---------------------------------------------------------------------------

def test_a_moved_boundary_changes_the_fingerprint():
    assert (fingerprint_assignment(FakeAssignment(SQUARE))
            != fingerprint_assignment(FakeAssignment(MOVED)))


def test_a_retitled_assignment_does_not_change_the_fingerprint():
    """A cosmetic edit must not force a re-stage. Re-staging that is not
    needed trains people to ignore the signal."""
    original = FakeAssignment(SQUARE, title="Ridge north")
    renamed = FakeAssignment(SQUARE, title="Ridge north (revised)")
    assert fingerprint_assignment(original) == fingerprint_assignment(renamed)


def test_float_noise_from_a_json_round_trip_is_not_a_moved_boundary():
    """CalTopo round-trips geometry through JSON. Without rounding, every
    refetch would look like an edit and nothing would ever stay valid."""
    noisy = json.loads(json.dumps(SQUARE))
    noisy["coordinates"][0][0][0] -= 1e-12

    assert (fingerprint_assignment(FakeAssignment(SQUARE))
            == fingerprint_assignment(FakeAssignment(noisy)))


def test_renumbering_an_assignment_does_change_the_fingerprint():
    """The label identifies which ground a result is about, so it is part
    of what a computation depends on."""
    assert (fingerprint_assignment(FakeAssignment(SQUARE, number="103"))
            != fingerprint_assignment(FakeAssignment(SQUARE, number="104")))


# ---------------------------------------------------------------------------
# Handles are ours, not paths a caller composes.
# ---------------------------------------------------------------------------

def test_a_handle_round_trips():
    handle = TerrainHandle("MAP1234", "QA103")
    assert TerrainHandle.parse(handle.key) == handle


def test_a_made_up_handle_is_refused_rather_than_guessed_at():
    """Guessing what a malformed handle meant would be guessing at which
    ground to compute."""
    with pytest.raises(ValueError, match="prepare_terrain"):
        TerrainHandle.parse("nonsense")


def test_a_handle_cannot_escape_the_cache_directory(tmp_path):
    directory = TerrainHandle("../../etc", "passwd").directory(tmp_path)
    assert tmp_path.resolve() in directory.resolve().parents
    assert ".." not in directory.parts


# ---------------------------------------------------------------------------
# Staging, and skipping it.
# ---------------------------------------------------------------------------

class FakeState:
    def __init__(self, features=()):
        self.features = list(features)

    def provenance(self):
        return {"source": "test"}


class FakeReader:
    """Records every read; the assignment it 'finds' is whatever the test
    put in `assignment[0]`."""

    def __init__(self, cache_dir, assignment):
        self.config = CaltopoConfig(cache_dir=cache_dir)
        self.cache_dir = cache_dir
        self.assignment = assignment
        self.reads: list[tuple[str, bool, float]] = []
        self.offline = False

    def read(self, map_id, *, public, max_age_seconds=300.0):
        self.reads.append((map_id, public, max_age_seconds))
        if self.offline:
            raise OSError("network is unreachable")
        return FakeState()


@pytest.fixture
def reader(tmp_path):
    return FakeReader(tmp_path / "cache", [FakeAssignment(SQUARE)])


def stub_everything(monkeypatch, reader, staged: list):
    """Stub the assignment lookup and the staging, so no network is touched."""
    monkeypatch.setattr("sar_geo_mcp.handles.find_assignment",
                        lambda features, label: reader.assignment[0])

    def fake_stage(a, map_id, out_dir, *, cache_dir, buffer_m, fingerprint,
                   access="public", subject_kind="assignment", **kwargs):
        from pathlib import Path

        from sar_geo_mcp.staging import write_manifest

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        write_manifest(out_dir, map_id=map_id, assignment_label=a.label,
                       bbox=(0, 0, 1, 1), dem_bbox=(0, 0, 1, 1),
                       dem_resolution_m=1.0, latitude_deg=37.4, panels={},
                       warnings=[], assignment_fingerprint=fingerprint,
                       access=access)
        staged.append(fingerprint)
        return {"warnings": []}

    monkeypatch.setattr("sar_geo_mcp.staging.stage_terrain", fake_stage)


def test_first_call_stages_and_returns_a_handle(monkeypatch, reader):
    staged = []
    stub_everything(monkeypatch, reader, staged)

    result = prepare_terrain("M1", "QA103", reader)
    assert result["staged"] is True
    assert result["handle"] == "M1_QA103"
    assert len(staged) == 1


def test_an_unchanged_assignment_is_not_re_staged(monkeypatch, reader):
    """The reason a planner can ask a dozen questions for the cost of one
    stage."""
    staged = []
    stub_everything(monkeypatch, reader, staged)

    prepare_terrain("M1", "QA103", reader)
    again = prepare_terrain("M1", "QA103", reader)

    assert again["staged"] is False
    assert again["reason"] == "already current"
    assert len(staged) == 1, "the ground was downloaded twice"


def test_a_moved_boundary_forces_a_re_stage(monkeypatch, reader):
    """The whole point of the fingerprint."""
    staged = []
    stub_everything(monkeypatch, reader, staged)

    prepare_terrain("M1", "QA103", reader)
    reader.assignment[0] = FakeAssignment(MOVED)
    again = prepare_terrain("M1", "QA103", reader)

    assert again["staged"] is True
    assert "changed" in again["reason"]
    assert len(staged) == 2


def test_force_re_stages_an_unchanged_assignment(monkeypatch, reader):
    staged = []
    stub_everything(monkeypatch, reader, staged)

    prepare_terrain("M1", "QA103", reader)
    prepare_terrain("M1", "QA103", reader, force=True)
    assert len(staged) == 2


def test_staging_without_a_cache_directory_is_refused(monkeypatch):
    with pytest.raises(ValueError, match="cache directory"):
        prepare_terrain("M1", "QA103", FakeReader(None, [FakeAssignment(SQUARE)]))


# ---------------------------------------------------------------------------
# Revalidation before a computation - the rule itself.
# ---------------------------------------------------------------------------

def test_resolve_reports_current_when_the_boundary_has_not_moved(monkeypatch, reader):
    stub_everything(monkeypatch, reader, [])
    prepare_terrain("M1", "QA103", reader)

    resolved = resolve("M1_QA103", reader)
    assert resolved.status["current"] is True
    assert resolved.status["checked"] is True


def test_resolve_reports_stale_when_the_boundary_has_moved(monkeypatch, reader):
    """The failure this whole phase exists to prevent: a number computed
    against a polygon that is no longer the assignment."""
    stub_everything(monkeypatch, reader, [])
    prepare_terrain("M1", "QA103", reader)

    reader.assignment[0] = FakeAssignment(MOVED)
    resolved = resolve("M1_QA103", reader)

    assert resolved.status["current"] is False
    assert resolved.status["checked"] is True
    assert "prepare_terrain again" in resolved.status["reason"]


def test_an_unverifiable_boundary_is_not_reported_as_verified(monkeypatch, reader):
    """Offline. "Could not check" and "checked and fine" are different
    facts, and only one of them is safe to plan against - so `current` is
    None, never True."""
    stub_everything(monkeypatch, reader, [])
    prepare_terrain("M1", "QA103", reader)

    reader.offline = True
    resolved = resolve("M1_QA103", reader)

    assert resolved.status["current"] is None
    assert resolved.status["checked"] is False
    assert "may have changed" in resolved.status["reason"]


def test_revalidation_forces_a_check_regardless_of_the_listing_window(monkeypatch,
                                                                     reader):
    """A listing may be slightly stale provided it says so. A computation
    may not be stale at all, so it must not be served from the listing
    cache."""
    stub_everything(monkeypatch, reader, [])
    prepare_terrain("M1", "QA103", reader)
    reader.reads.clear()

    resolve("M1_QA103", reader)
    assert [r[2] for r in reader.reads] == [0], \
        "a computation revalidated against a cached listing"


def test_resolving_a_handle_that_was_never_staged_says_so(reader):
    with pytest.raises(FileNotFoundError, match="prepare_terrain"):
        resolve("M1_QA103", reader)


def test_skipping_revalidation_says_it_was_skipped(monkeypatch, reader):
    stub_everything(monkeypatch, reader, [])
    prepare_terrain("M1", "QA103", reader)

    resolved = resolve("M1_QA103", reader, revalidate=False)
    assert resolved.status["current"] is None
    assert resolved.status["checked"] is False


def test_the_fingerprint_is_recorded_in_the_manifest(monkeypatch, reader):
    """It has to survive a restart: a fingerprint held only in memory would
    make every new process re-stage everything."""
    stub_everything(monkeypatch, reader, [])
    prepare_terrain("M1", "QA103", reader)

    directory = TerrainHandle("M1", "QA103").directory(reader.cache_dir)
    assert staged_fingerprint(directory) == fingerprint_assignment(
        FakeAssignment(SQUARE))


# ---------------------------------------------------------------------------
# A handle remembers which path it was read by.
# ---------------------------------------------------------------------------

def test_the_manifest_records_the_access_path(tmp_path):
    from sar_geo_mcp.handles import staged_access
    from sar_geo_mcp.staging import write_manifest

    write_manifest(tmp_path, map_id="M1", assignment_label="QA103",
                   bbox=(0, 0, 1, 1), dem_bbox=(0, 0, 1, 1), dem_resolution_m=1.0,
                   latitude_deg=37.4, panels={}, warnings=[],
                   access="authenticated")
    assert staged_access(tmp_path) == "authenticated"


def test_a_manifest_from_before_access_was_recorded_reads_as_public(tmp_path):
    """Every handle staged before this key existed was revalidated as
    public, so that is what it still means."""
    from sar_geo_mcp.handles import staged_access
    from sar_geo_mcp.staging import write_manifest

    write_manifest(tmp_path, map_id="M1", assignment_label="QA103",
                   bbox=(0, 0, 1, 1), dem_bbox=(0, 0, 1, 1), dem_resolution_m=1.0,
                   latitude_deg=37.4, panels={}, warnings=[])
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["access"] == "public"

    del manifest["access"]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert staged_access(tmp_path) == "public"


def test_an_authenticated_stage_is_revalidated_through_the_same_path(monkeypatch, reader):
    """The `public` flag used to be accepted here and ignored: a private
    map's handle was revalidated through the public endpoint, failed, and
    reported "could not check" forever."""
    stub_everything(monkeypatch, reader, [])
    prepare_terrain("M1", "QA103", reader, public=False)
    assert reader.reads[-1] == ("M1", False, 0)

    reader.reads.clear()
    resolved = resolve("M1_QA103", reader)
    assert reader.reads == [("M1", False, 0)]
    assert resolved.status["current"] is True


def test_a_current_assignment_prepared_by_the_other_path_is_read_by_it(monkeypatch,
                                                                       reader):
    """A map whose owner stopped sharing it publicly, prepared again with
    public=False: the ground is current, so nothing is staged - and every
    later revalidation reads the map the way the call that just worked did,
    not through the public endpoint it was first staged by (PR #4 review
    B49). The manifest is not rewritten, so nothing computed from the
    ground reads as made before a re-stage."""
    from sar_geo_mcp.handles import staged_access

    staged = []
    stub_everything(monkeypatch, reader, staged)
    prepare_terrain("M1", "QA103", reader, public=True)
    directory = TerrainHandle("M1", "QA103").directory(reader.cache_dir)
    manifest = (directory / "manifest.json").read_bytes()

    again = prepare_terrain("M1", "QA103", reader, public=False)

    assert (again["staged"], again["reason"]) == (False, "already current")
    assert len(staged) == 1
    assert (directory / "manifest.json").read_bytes() == manifest
    assert staged_access(directory) == "authenticated"
    reader.reads.clear()
    resolve("M1_QA103", reader)
    assert reader.reads == [("M1", False, 0)]


def test_a_re_stage_reads_the_map_by_its_own_call_s_path(monkeypatch, reader):
    """A path recorded for the old ground does not outlive it: the manifest
    a re-stage writes holds the path its call named."""
    from sar_geo_mcp.handles import ACCESS_FILENAME, staged_access

    stub_everything(monkeypatch, reader, [])
    prepare_terrain("M1", "QA103", reader, public=True)
    prepare_terrain("M1", "QA103", reader, public=False)
    directory = TerrainHandle("M1", "QA103").directory(reader.cache_dir)
    assert (directory / ACCESS_FILENAME).is_file()

    prepare_terrain("M1", "QA103", reader, public=True, force=True)

    assert not (directory / ACCESS_FILENAME).exists()
    assert staged_access(directory) == "public"


def test_resolve_hands_back_the_state_it_revalidated_against(monkeypatch, reader):
    """`coverage_for` needs the map's tracks, and they are in the very
    state the boundary was just checked against - a second read would be
    a second probe for the same answer."""
    stub_everything(monkeypatch, reader, [])
    prepare_terrain("M1", "QA103", reader)

    resolved = resolve("M1_QA103", reader)
    assert resolved.state is not None
    assert resolved.state.provenance() == {"source": "test"}

    reader.offline = True
    resolved = resolve("M1_QA103", reader)
    assert resolved.state is None
    assert resolved.status["checked"] is False
