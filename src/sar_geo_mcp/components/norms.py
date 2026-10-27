"""Components: the assignment corpus norms."""

from __future__ import annotations

import copy
from typing import Any

from ..context import ServerContext, mcp_tool
from ..planning.norms import target_area_range_ha


def assignment_norms(ctx: ServerContext, resource_type: str = "") -> dict[str, Any]:
    """The `assignment_norms` tool as a plain function: `ctx`, then the
    tool's arguments; the tool's payload back. What each argument and field
    means is in the tool's description, in `register` below."""
    # "filtered" in the spec means annotated: the whole file comes back
    # (about 800 tokens) plus the resource's target range, so a reader
    # keeps the provenance beside the number.
    # A deep copy, not `dict(...)`: these are the server's own norms, which
    # hasty, boundary and urban read on every call. Over MCP the payload is
    # serialised, so nothing is shared; a scenario calling this directly
    # gets the objects themselves, and changing one nested value would
    # change every later answer in this server process.
    payload = copy.deepcopy(ctx.norms)
    if resource_type:
        payload["resource_type"] = resource_type.strip().upper()
        payload["target_area_ha"] = target_area_range_ha(resource_type, ctx.norms)
    return payload


def register(mcp, ctx) -> None:

    @mcp_tool(mcp, name="assignment_norms", tags={"component"})
    def _assignment_norms_tool(resource_type: str = "") -> dict[str, Any]:
        """What real assignments look like: sizes, corridor widths, how much
        of a boundary follows a mapped feature, with the corpus's provenance.

        From 65 real searches and 4,673 assignments. Use it to tell a human
        where a default came from, and to read a norm flag on a candidate
        or a boundary. `resource_type` (GROUND, GROUND_1, DOG_HRD, UAV, ...)
        adds that resource's target area range, derived from its corpus
        median by the stated ratio rule. The server can be started with
        another agency's file (`--norms`).
        """
        return assignment_norms(ctx, resource_type=resource_type)
