"""Components: reading and exporting candidate sets a feature generated."""

from __future__ import annotations

from typing import Any

from ..context import ServerContext, mcp_tool
from ..planning.candidates import (
    Candidate,
    CandidateSet,
    find_candidate_in,
    load_set,
    parse_candidate_id,
    set_staleness,
)
from ..planning.export import export_set

#: What a candidate's facts are AS OF. A set is measured once and persists;
#: the map it was measured against is read fresh on every call. Saying so is
#: the difference between a planner reading `overlap_existing_pct` as "nobody
#: is on this line" and as "nobody was on it when this set was made".
CANDIDATE_FACTS_NOTE = (
    "facts, including overlap with existing assignments, were measured when the "
    "set was generated (created_at) against the map version then current "
    "(map_version); the set is regenerated, not reused, once the map changes; "
    "staleness.map_provenance is the map as read now")


def candidate_and_staleness(directory, candidate_id: str) -> tuple[CandidateSet, Candidate, str | None]:
    """The set an id belongs to, the candidate it names, and a warning when
    the set was computed before the directory was last re-staged (None when
    the set still matches). `find_candidate` loads the set fresh each time;
    this loads it once so the candidate, its set and its staleness all come
    from the same read, and reads the candidate out of it with
    `find_candidate_in` so a bad index is refused in the one wording."""
    set_id, _index = parse_candidate_id(candidate_id)
    cset = load_set(directory, set_id)
    return cset, find_candidate_in(cset, candidate_id), set_staleness(directory, cset)


def caltopo_export(ctx: ServerContext, handle: str, set_id: str | None = None,
                   candidate_ids: list[str] | None = None,
                   label_prefix: str = "U", resource_type: str = "GROUND",
                   include_area: bool = True,
                   include_markers: bool = True) -> dict[str, Any]:
    """The `caltopo_export` tool as a plain function: `ctx`, then the
    tool's arguments; the tool's payload back. What each argument and field
    means is in the tool's description, in `register` below."""
    resolved = ctx.staged(handle)
    if (set_id is None) == (not candidate_ids):
        raise ValueError("give exactly one of set_id or candidate_ids")
    if candidate_ids:
        sets = {parse_candidate_id(cid)[0] for cid in candidate_ids}
        if len(sets) != 1:
            raise ValueError("candidate_ids must all come from one set; export each set "
                             "on its own")
        set_id = sets.pop()
    payload = export_set(resolved.directory, set_id, candidate_ids=candidate_ids or None,
                         label_prefix=label_prefix, resource_type=resource_type,
                         include_area=include_area, include_markers=include_markers)
    payload["warnings"] = [*payload["warnings"], *resolved.status.get("warnings", [])]
    return payload


def get_candidate(ctx: ServerContext, handle: str, candidate_id: str) -> dict[str, Any]:
    """The `get_candidate` tool as a plain function: `ctx`, then the tool's
    arguments; the tool's payload back. What each argument and field means
    is in the tool's description, in `register` below."""
    resolved = ctx.staged(handle)
    cset, candidate, set_warning = candidate_and_staleness(resolved.directory, candidate_id)
    payload = candidate.model_dump()
    # Only a block asked for with markers has them; every other candidate
    # reads as it always has, with no `markers` key at all.
    if payload.get("markers") is None:
        payload.pop("markers", None)
    payload["set_id"] = cset.set_id
    payload["created_at"] = cset.created_at
    payload["facts_note"] = CANDIDATE_FACTS_NOTE
    payload["staleness"] = resolved.status
    payload["set_warnings"] = [set_warning] if set_warning else []
    return payload


def register(mcp, ctx) -> None:

    @mcp_tool(mcp, name="caltopo_export", tags={"component"})
    def _caltopo_export_tool(handle: str, set_id: str | None = None,
                             candidate_ids: list[str] | None = None,
                             label_prefix: str = "U", resource_type: str = "GROUND",
                             include_area: bool = True,
                             include_markers: bool = True) -> dict[str, Any]:
        """Write a candidate set as a GeoJSON file CalTopo imports as DRAFT
        assignments. Nothing is sent to CalTopo: the file is written beside
        the staged ground and a planner imports it (CalTopo: Import, choose
        the file), checks every assignment and issues it.

        Give `set_id` (every candidate of a set, in order - for example the
        `urban-xxxxxx` set from `urban_assignments`) or `candidate_ids`
        (those candidates, in that order, all from one set). Each becomes a
        CalTopo Assignment - `status` DRAFT, `resourceType`, and a label
        `letter` of `label_prefix` plus a number and the name, such as
        `U03 Main Street +2` - with a description holding its streets,
        lots, buildings, area and where it came from. CalTopo holds one
        ring per polygon and one polygon per assignment, so a multi-part
        candidate becomes one assignment per part (`U05a`, `U05b`) and a
        hole is cut open along a zero-width bridge. Lines become line
        assignments. With `include_markers`, house markers a set carries go
        in as CalTopo Markers; with `include_area`, an area handle's search
        area goes in as a Shape (the range circle) with a Marker on its
        centre.

        Everything in the file sits in one CalTopo folder, `folder` (titled
        "SAR Geo <set id> <date time>"), so the import lands as one group the
        planner can move or delete together.

        Returns `path` (the .geojson written), `folder`, `assignments` (how
        many were drawn), `features` (a count by CalTopo class), `labels` (each
        label with the candidate it came from), `area_included` and
        `warnings` - including when the set predates a re-stage.
        """
        return caltopo_export(ctx, handle=handle, set_id=set_id,
                              candidate_ids=candidate_ids, label_prefix=label_prefix,
                              resource_type=resource_type, include_area=include_area,
                              include_markers=include_markers)

    @mcp_tool(mcp, name="get_candidate", tags={"component"})
    def _get_candidate_tool(handle: str, candidate_id: str) -> dict[str, Any]:
        """One candidate in full, geometry included, by the id a generating
        tool returned (for example `hasty-3f9a2c.4`). Sets persist inside the
        staged directory, so an id stays valid for the life of the handle;
        an id from a set that was never generated here is refused.

        `set_warnings` reports the set's staleness, which has three
        outcomes: empty when the set still matches the directory's current
        staging; one warning when the set was computed before the directory
        was re-staged, so the geometry may no longer match the ground; and
        one warning when the set recorded no staging at all and the check
        could not be made. An empty list is the only one of the three that
        means the geometry is known to be current.

        `created_at` and `set_id` date the facts: they were measured when the
        set was generated, so a candidate's overlap with existing assignments
        is as the map stood then, while `staleness.map_provenance` is the map
        as read now. `facts_note` says the same thing beside the numbers.
        """
        return get_candidate(ctx, handle=handle, candidate_id=candidate_id)
