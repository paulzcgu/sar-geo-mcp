"""Feature: coverage and gaps from GPS tracks."""

from __future__ import annotations

from typing import Any

from ..context import ServerContext, mcp_tool


def coverage_for(ctx: ServerContext, handle: str, track_names: list[str]) -> dict[str, Any]:
    """The `coverage_for` tool as a plain function: `ctx`, then the tool's
    arguments; the tool's payload back. What each argument and field means
    is in the tool's description, in `register` below."""
    from ..coverage.metrics import compute_coverage

    resolved = ctx.staged(handle)
    if resolved.state is None:
        detail = resolved.status.get("map_reason") or resolved.status.get("reason")
        raise RuntimeError(f"cannot read the map's tracks: {detail}")
    result = compute_coverage(resolved.directory, track_names,
                              features=resolved.state.features)
    # The digest only. `track_geojson` is the dense track geometry kept
    # for a renderer, and is exactly what must stay out of a model's
    # context (see `CoverageResult`).
    payload = result.metrics.model_dump(mode="json")
    payload["staleness"] = resolved.status
    return payload


def register(mcp, ctx) -> None:

    @mcp_tool(mcp, name="coverage_for", tags={"feature"})
    def _coverage_for_tool(handle: str, track_names: list[str]) -> dict[str, Any]:
        """Coverage and gaps for a staged assignment, from named GPS tracks.

        The tracks are read from the same map the assignment came from -
        the very map state the boundary was just revalidated against, so
        this costs one read, not two. Carries the same `staleness` block as
        the other computations, and for the same reason: coverage computed
        against a boundary that has since moved reports ground as searched
        that nobody walked. When the boundary could not be checked the
        tracks could not be read either, and that is reported rather than
        computed around. An area assignment needs `terrain_metrics` called
        on the handle first: the sweep width comes from it.
        `sweep_width_basis` says how that width was reached - the metrics'
        own basis, "corridor_width_m" or "default_corridor" for a line -
        and when it is "default_width", read off no vegetation at all, a
        warning says every number here rests on it.
        """
        return coverage_for(ctx, handle=handle, track_names=track_names)
