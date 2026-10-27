"""Feature: the anatomy of one assignment or drawn shape."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, model_validator

from ..caltopo.assignment import find_assignment
from ..components.candidates import candidate_and_staleness
from ..context import ServerContext, mcp_tool
from ..planning.boundary import describe_boundary as describe_boundary_impl

class BoundaryTarget(BaseModel):
    """Exactly one of these. A one-key object on the wire, so the docstring's
    examples stay true; typed here so a model sees the fields in the schema."""

    assignment: str | None = None      # an assignment label on the staged map
    geometry: dict | None = None       # a GeoJSON Polygon or LineString, WGS84
    candidate: str | None = None       # a candidate id

    @model_validator(mode="after")
    def _exactly_one(self):
        given = [name for name in ("assignment", "geometry", "candidate")
                 if getattr(self, name) is not None]
        if len(given) != 1:
            raise ValueError('target must have exactly one key: "assignment", "geometry" or "candidate"')
        return self


def describe_boundary(ctx: ServerContext, handle: str, target: BoundaryTarget,
                      resource_type: str = "") -> dict[str, Any]:
    """The `describe_boundary` tool as a plain function: `ctx`, then the
    tool's arguments; the tool's payload back. What each argument and field
    means is in the tool's description, in `register` below."""
    resolved = ctx.staged(handle)
    map_features = resolved.state.features if resolved.state is not None else None

    set_warning = None
    if target.candidate is not None:
        _cset, candidate, set_warning = candidate_and_staleness(
            resolved.directory, target.candidate)
        geometry, exclude = candidate.geometry, None
    elif target.assignment is not None:
        if map_features is None:
            raise ValueError("describing an assignment needs the map: this handle has "
                             "no map, or the map could not be read (see staleness)")
        assignment = find_assignment(map_features, target.assignment)
        geometry, exclude = assignment.geometry, assignment.label
    else:
        geometry, exclude = target.geometry, None

    anatomy = describe_boundary_impl(resolved.directory, geometry, norms=ctx.norms,
                                     resource_type=resource_type,
                                     map_features=map_features, exclude_label=exclude)
    payload = anatomy.model_dump()
    if set_warning:
        payload["warnings"] = [*payload.get("warnings", []), set_warning]
    payload["staleness"] = resolved.status
    # The shared edges are read off the map, so the map's age belongs
    # beside the answer rather than inside the staleness block. An area
    # handle records it under `map_provenance` and an assignment handle
    # under `provenance` (there it is also the revalidation's own read);
    # a handle with no readable map has neither.
    payload["map_provenance"] = (resolved.status.get("map_provenance")
                                 or resolved.status.get("provenance"))
    return payload


def register(mcp, ctx) -> None:

    @mcp_tool(mcp, name="describe_boundary", tags={"feature"})
    def _describe_boundary_tool(handle: str, target: BoundaryTarget,
                                resource_type: str = "") -> dict[str, Any]:
        """The anatomy of one assignment or drawn shape on staged ground:
        what its edges follow, how it was drawn, what it holds.

        `target` has exactly one key: `{"assignment": "QA103"}` (an
        assignment on the map the handle was staged with),
        `{"geometry": <GeoJSON Polygon or LineString, WGS84>}`, or
        `{"candidate": "hasty-3f9a2c.4"}` (an id from `hasty_candidates`).

        Walks the boundary every 10 m and names the nearest mapped feature
        within tolerance - trail, road, waterway, elevation-model drainage
        or ridge, fence or power line - as the anatomy report did for 4,528
        real boundaries. Reports the composition, the on-feature share, the
        straight drawn edges, the edges shared with neighbouring assignments,
        the runs an agent can cite by feature id, the interior's linear
        metres, slope and landform, the corridor/block/line classification
        with its axis, and norm flags against the corpus. It measures and
        flags; it never scores. A target outside the staged extent is refused.

        `resource_type` (GROUND, DOG_TRAIL, UAV, ...) bands the area flag
        against that resource's corpus target range instead of the whole
        corpus; blank uses the whole corpus. `assumptions.area_band` says
        which band was applied.

        `map_provenance` says how old the map the neighbours were read from
        is, or is null when the handle has no map.
        """
        return describe_boundary(ctx, handle=handle, target=target,
                                 resource_type=resource_type)
