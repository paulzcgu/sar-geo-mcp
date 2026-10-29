"""Staged assignments, addressed by handle, revalidated before they are used.

WHAT A HANDLE IS FOR (E4)
-------------------------
Terrain computation takes a staged directory, which made it a poor tool: as
a one-shot call it would download a DEM every time - tens of megabytes and
tens of seconds - and as a path argument it would make an LLM responsible
for filesystem layout. A handle fixes both. `prepare_terrain` stages the
ground once and returns an opaque key; every later question names the key.

The handle is issued by this module and is never a path a caller composes,
so the directory layout stays this package's business. Area handles
(`aoi-...`, from `planning.area`) are the exception to revalidation: their
input is immutable, so their status is a constant, and only the map
context they were staged with is re-read.

THE BINDING RULE (F4)
---------------------
> Any derived product records a fingerprint of the mutable input it came
> from, and revalidates before returning a result.

CalTopo assignments move during an operation. A staged directory built from
a boundary that has since been redrawn will happily produce coverage for the
OLD polygon - reporting ground as searched that nobody walked. That is the
one caching mistake in this system with an operational consequence, so it
gets the strictest handling:

  a listing may be slightly stale provided it SAYS so;
  a computation may not be stale at all.

So every computation here forces revalidation - `max_age_seconds=0` - no
matter how recently a listing was served.

WHAT THE FINGERPRINT COVERS
---------------------------
The geometry, and only the properties a computation actually depends on. A
retitled assignment must not invalidate a DEM clip; a moved boundary must.
Fingerprinting the whole feature would make every cosmetic edit a full
re-stage, and re-staging that is not needed trains people to ignore the
signal.

The manifest also records which CalTopo path the assignment was read by,
so the revalidation is made through the same one (see `staged_access`). A
later call that finds the ground current and has just read the map by the
other path makes that path the one every later read takes
(`record_access`), without a re-stage.

WHAT SURVIVES A CHANGE
----------------------
The tiles. Geography did not move - only the polygon over it did - so
re-staging after a boundary edit costs compute, not bandwidth. That is the
payoff of putting the static layers on a shared grid, and the reason these
two phases were designed together.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .cache import atomic_write_text
from .caltopo.assignment import find_assignment

STAGED_DIRNAME = "staged"

#: Area handles (planning.area) begin with this; assignment handles begin
#: with a map id, which is alphanumeric, so the two cannot collide.
AREA_PREFIX = "aoi-"
_AREA_HANDLE = re.compile(r"^aoi-[0-9a-f]{16}$")


def is_area_handle(handle: str) -> bool:
    return handle.startswith(AREA_PREFIX)

#: The properties a computation depends on. Everything else about an
#: assignment - title, description, status, who edited it - can change
#: without invalidating a single number computed from the ground.
FINGERPRINTED_PROPERTIES = ("letter", "number")


def fingerprint_assignment(assignment) -> str:
    """A stable hash of what a computation actually depends on.

    Coordinates are rounded before hashing. CalTopo round-trips geometry
    through JSON, and a boundary that came back with one extra digit of
    float noise is not a boundary that moved; without the rounding every
    refetch would look like an edit and nothing would ever stay valid.
    """
    geometry = assignment.geometry or {}

    def rounded(value):
        if isinstance(value, float):
            return round(value, 7)   # ~1 cm; far finer than any edit
        if isinstance(value, list):
            return [rounded(item) for item in value]
        return value

    payload = {
        "type": geometry.get("type"),
        "coordinates": rounded(geometry.get("coordinates")),
        "properties": {name: getattr(assignment, name, None)
                       for name in FINGERPRINTED_PROPERTIES},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]


@dataclass(frozen=True)
class TerrainHandle:
    map_id: str
    label: str

    @property
    def key(self) -> str:
        safe = "".join(c for c in f"{self.map_id}_{self.label}"
                       if c.isalnum() or c in "-_")
        if not safe:
            raise ValueError(f"unusable handle: {self.map_id}/{self.label}")
        return safe

    @classmethod
    def parse(cls, handle: str) -> "TerrainHandle":
        """Read back a handle this module issued.

        Anything unrecognisable is refused rather than coerced: a handle is
        our own token, so a malformed one means a caller invented it, and
        guessing what they meant would be guessing at which ground to
        compute.
        """
        if "_" not in handle:
            raise ValueError(
                f"not a terrain handle: {handle!r}. Call prepare_terrain first "
                "and use the handle it returns.")
        map_id, _, label = handle.partition("_")
        return cls(map_id=map_id, label=label)

    def directory(self, cache_dir: Path) -> Path:
        return Path(cache_dir) / STAGED_DIRNAME / self.key


def _reuse_warnings(directory: Path) -> list[str]:
    """The staged manifest's warnings as they stand now, for ground found
    current and reused; none when the manifest cannot be read (the
    fingerprint just read from it, so that is a race, not a state)."""
    from .staging import manifest_warnings

    try:
        manifest = json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return manifest_warnings(directory, manifest)


def staged_fingerprint(directory: Path) -> str | None:
    try:
        manifest = json.loads((Path(directory) / "manifest.json")
                              .read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = manifest.get("assignment_fingerprint")
    return value if isinstance(value, str) else None


#: Beside a staged directory's manifest: the CalTopo path a later call over
#: the same staged ground read the map by, when it is not the one the
#: manifest records. See `record_access`.
ACCESS_FILENAME = "access.json"

_ACCESS_PATHS = ("public", "authenticated")


def staged_access(directory: Path) -> str:
    """`public` or `authenticated`: the path the staged ground's map is read by.

    That is the path a later prepare call over this ground last read the map
    by, which `record_access` wrote beside the manifest, else the one the
    manifest records from staging. A manifest without the key predates it,
    and every such directory was revalidated as public, so that is what an
    absent key means.
    """
    directory = Path(directory)
    return (_recorded_access(directory / ACCESS_FILENAME)
            or _recorded_access(directory / "manifest.json") or "public")


def _recorded_access(path: Path) -> str | None:
    """The `access` a JSON file records, or None when it records none."""
    try:
        recorded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = recorded.get("access") if isinstance(recorded, dict) else None
    return value if value in _ACCESS_PATHS else None


def record_access(directory: Path, public: bool) -> bool:
    """Make `public` the path every later read of this staged ground's map
    takes, without staging it again. True when that changed the path.

    For a prepare call answered from ground already staged. The ground does
    not depend on how the map is read, but every later `resolve` reads the
    map by `staged_access`, so a call that names the other path - a private
    map's call corrected from public=True to public=False - would otherwise
    have its map read the way the first call said: through the public
    endpoint, failing, every section that compares against the map left
    without it while the corrected call's own read had worked (PR #4 review
    B49).

    Only once the call's own read of the map by that path has worked: a
    later call leaving `public` at its default true on a private map would
    otherwise move a handle off the one path that reads it.

    Written to its own file, never into the manifest: candidate sets and
    the loaded ground are held current by the manifest's stamp, its mtime
    and size (`boundary._manifest_stamp`), so rewriting it would report
    every set on the handle as made before a re-stage that never happened.
    Staging writes a new manifest and removes this file
    (`staging.write_manifest`): the manifest then records the latest path.
    """
    access = "public" if public else "authenticated"
    if staged_access(directory) == access:
        return False
    atomic_write_text(Path(directory) / ACCESS_FILENAME, json.dumps({"access": access}))
    return True


def prepare_terrain(map_id: str, label: str, reader, *,
                    public: bool = True,
                    buffer_m: float | None = None,
                    force: bool = False) -> dict:
    """Stage the ground for one assignment and return a handle for it.

    `reader` is the server's `MapReader`; `public` says which of its paths
    the map is read by, and the choice is recorded with the ground - in the
    manifest when it is staged, by `record_access` when it is found current
    - so every later revalidation uses the same one.

    Re-staging is skipped when the directory already exists AND its
    fingerprint still matches the live assignment - which is the common
    case, and the reason a planner can ask a dozen questions about an
    assignment for the cost of one stage. The path this call read the map
    by is still recorded (`record_access`): it has just worked, and a map
    whose owner stopped sharing it publicly is otherwise revalidated
    through the public endpoint for ever, "could not check" on every
    question, however often it is prepared again with public=False.

    The assignment is fetched with forced revalidation, so "already staged"
    is never asserted against a boundary nobody checked.
    """
    from .staging import DEFAULT_BUFFER_M, stage_terrain

    if reader.cache_dir is None:
        raise ValueError("staging needs a cache directory; none is configured")

    handle = TerrainHandle(map_id=map_id, label=label)
    directory = handle.directory(reader.cache_dir)

    state = reader.read(map_id, public=public, max_age_seconds=0)
    assignment = find_assignment(state.features, label)
    current = fingerprint_assignment(assignment)
    existing = staged_fingerprint(directory)

    if not force and existing is not None and existing == current:
        record_access(directory, public)
        # What the ground lacks is still so on reuse - vegetation that could
        # not be fetched is still missing - and prepare_area hands it back
        # the same way: the manifest's warnings as they stand now, an outage
        # a later osm.json retired left out (`staging.manifest_warnings`).
        return {"handle": handle.key, "staged": False, "reason": "already current",
                "fingerprint": current,
                "warnings": _reuse_warnings(directory),
                "provenance": state.provenance()}

    result = stage_terrain(
        assignment, map_id, directory, cache_dir=reader.cache_dir,
        buffer_m=DEFAULT_BUFFER_M if buffer_m is None else buffer_m,
        fingerprint=current,
        access="public" if public else "authenticated")
    return {
        "handle": handle.key,
        "staged": True,
        "reason": ("boundary changed since it was staged" if existing
                   else "not staged before"),
        "fingerprint": current,
        "previous_fingerprint": existing,
        "warnings": result["warnings"],
        "provenance": state.provenance(),
    }


@dataclass(frozen=True)
class Resolved:
    """A staged directory, its revalidation status, and the map state the
    check was made against (None when it could not be made)."""

    directory: Path
    status: dict
    state: object | None = None


def _resolve_area(handle: str, reader) -> Resolved:
    """An area's input is immutable, so its status is a constant and costs no
    network. Map context, when the area was staged with a map id, is read
    fresh so anything compared against the map carries provenance."""
    if not _AREA_HANDLE.match(handle):
        raise ValueError(f"not an area handle: {handle!r}. Use the handle prepare_area returned.")
    directory = Path(reader.cache_dir) / STAGED_DIRNAME / handle
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"nothing staged for handle {handle!r}. Call prepare_area first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status: dict = {"current": True, "checked": True, "reason": "area input is immutable"}
    state = None
    map_id = manifest.get("map_id") or ""
    if not map_id:
        status["map"] = "none"
        status["map_reason"] = (
            "this area was staged without a map_id, so there are no "
            "assignments or tracks to compare against")
    else:
        try:
            state = reader.read(map_id, public=staged_access(directory) == "public",
                                max_age_seconds=0)
            status["map"] = "read"
            status["map_provenance"] = state.provenance()
        except Exception as error:
            reason = (f"could not read map {map_id} ({type(error).__name__}); anything "
                      "compared against the map is unavailable")
            status["map"] = "unreadable"
            status["map_reason"] = reason
            status["warnings"] = [reason]
    return Resolved(directory, status, state)


def resolve(handle: str, reader, *, revalidate: bool = True) -> Resolved:
    """The staged directory for a handle, checked against the live assignment.

    `status` carries `current` - False when the boundary has moved since
    staging - alongside the provenance of the check. The map is read by
    the path recorded for the ground (`staged_access`): the one it was
    staged by, or one a later prepare call has since read the map by -
    never one the caller has to remember.

    `revalidate=False` exists only for the offline case, and says so in the
    status rather than pretending the check passed.
    """
    if reader.cache_dir is None:
        raise ValueError("no cache directory is configured, so nothing can be staged")
    if is_area_handle(handle):
        return _resolve_area(handle, reader)

    parsed = TerrainHandle.parse(handle)
    directory = parsed.directory(reader.cache_dir)
    if not (directory / "manifest.json").exists():
        raise FileNotFoundError(
            f"nothing staged for handle {handle!r}. Call prepare_terrain first.")

    staged = staged_fingerprint(directory)
    if not revalidate:
        return Resolved(directory, {"current": None, "checked": False,
                                    "reason": "revalidation skipped by caller"})

    public = staged_access(directory) == "public"
    try:
        state = reader.read(parsed.map_id, public=public, max_age_seconds=0)
        assignment = find_assignment(state.features, parsed.label)
    except Exception as error:
        # Offline. Say the check could not be made rather than implying it
        # passed - an unchecked boundary and a checked one are different
        # facts, and only one of them is safe to plan against.
        return Resolved(directory, {
            "current": None, "checked": False,
            "reason": f"could not revalidate ({type(error).__name__}); "
                      "the boundary may have changed since this was staged",
        })

    live = fingerprint_assignment(assignment)
    return Resolved(directory, {
        "current": live == staged,
        "checked": True,
        "fingerprint": staged,
        "live_fingerprint": live,
        "provenance": state.provenance(),
        **({} if live == staged else {
            "reason": "the assignment boundary has changed since this was "
                      "staged; call prepare_terrain again before trusting "
                      "any number computed from it"}),
    }, state)
