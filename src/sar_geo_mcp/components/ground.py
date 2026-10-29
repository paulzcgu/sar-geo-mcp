"""Components: single measurements on staged ground."""

from __future__ import annotations

from typing import Any

from ..context import ServerContext, mcp_tool


def terrain_metrics(ctx: ServerContext, handle: str) -> dict[str, Any]:
    """The `terrain_metrics` tool as a plain function: `ctx`, then the
    tool's arguments; the tool's payload back. What each argument and field
    means is in the tool's description, in `register` below."""
    from ..cache import atomic_write_text
    from ..terrain.metrics import compute_terrain_metrics
    from ..terrain.shapes import METRICS_FILENAME

    resolved = ctx.staged(handle)
    metrics = compute_terrain_metrics(resolved.directory)
    atomic_write_text(resolved.directory / METRICS_FILENAME,
                      metrics.model_dump_json(indent=2))
    payload = metrics.model_dump()
    payload["staleness"] = resolved.status
    return payload


def site_facts(ctx: ServerContext, handle: str, geometry: dict,
               band_m: float = 35.0) -> dict[str, Any]:
    """The `site_facts` tool as a plain function: `ctx`, then the tool's
    arguments; the tool's payload back. What each argument and field means
    is in the tool's description, in `register` below."""
    from ..planning.site import site_facts as site_facts_impl

    resolved = ctx.staged(handle)
    return {**site_facts_impl(resolved.directory, geometry, band_m=band_m),
            "staleness": resolved.status}


def list_feature_kinds(ctx: ServerContext) -> dict[str, Any]:
    """The `list_feature_kinds` tool as a plain function: `ctx` in, the
    tool's payload back. What each field means is in the tool's
    description, in `register` below."""
    from ..terrain.features import list_kinds

    return {"kinds": list_kinds()}


def identify_features(ctx: ServerContext, handle: str, kinds: list[str]) -> dict[str, Any]:
    """The `identify_features` tool as a plain function: `ctx`, then the
    tool's arguments; the tool's payload back. What each argument and field
    means is in the tool's description, in `register` below."""
    from ..terrain.features import identify, validate_kinds

    # Before handle resolution: a hallucinated kind is corrected even
    # on a bad handle, and an invalid request never costs network I/O.
    validate_kinds(kinds)
    resolved = ctx.staged(handle)
    results = identify(resolved.directory, kinds)
    return {
        "results": {kind: result.model_dump()
                    for kind, result in results.items()},
        "staleness": resolved.status,
    }


def slope_stats(ctx: ServerContext, handle: str, shape_id: str) -> dict[str, Any]:
    """The `slope_stats` tool as a plain function: `ctx`, then the tool's
    arguments; the tool's payload back. What each argument and field means
    is in the tool's description, in `register` below."""
    from ..terrain.shapes import shape_slope_statistics

    resolved = ctx.staged(handle)
    return {**shape_slope_statistics(resolved.directory, shape_id),
            "staleness": resolved.status}


def elevation_profile(ctx: ServerContext, handle: str, drainage_id: str,
                      samples: int = 50) -> dict[str, Any]:
    """The `elevation_profile` tool as a plain function: `ctx`, then the
    tool's arguments; the tool's payload back. What each argument and field
    means is in the tool's description, in `register` below."""
    from ..terrain.shapes import drainage_elevation_profile

    resolved = ctx.staged(handle)
    return {**drainage_elevation_profile(resolved.directory, drainage_id, samples),
            "staleness": resolved.status}


def register(mcp, ctx) -> None:

    @mcp_tool(mcp, name="terrain_metrics", tags={"component"})
    def _terrain_metrics_tool(handle: str) -> dict[str, Any]:
        """Everything measured for a staged assignment: slope, landform,
        hydrology, vegetation, access.

        The answer carries a `staleness` block. When `current` is false the
        assignment boundary has moved since this ground was staged, and the
        numbers describe the OLD polygon - call `prepare_terrain` again
        before acting on them. When `checked` is false the boundary could
        not be verified at all, which is a different fact from verified-ok
        and is reported as such.

        The metrics are also saved beside the staged ground, and
        `slope_stats`, `elevation_profile` and `coverage_for` read them
        from there, so call this first on a handle.

        `traverse.sweep_width_basis` says how the sweep width was reached:
        "lifeform_and_density", "lifeform_moderate_density_assumed" (no
        vegetation panel measured density) or "default_width" - no
        vegetation type known, so the width is a flat default read off no
        vegetation at all, and a warning says so.
        """
        return terrain_metrics(ctx, handle=handle)

    @mcp_tool(mcp, name="site_facts", tags={"component"})
    def _site_facts_tool(handle: str, geometry: dict,
                         band_m: float = 35.0) -> dict[str, Any]:
        """Is this a place a team can occupy and work? Measurements, not a verdict.

        Every other tool here answers a version of "sweep this ground looking
        for a person". This one answers the other question: can a team stand
        here, get to it, and work it. That is the rope-rescue question, and
        equally the helispot, landing zone, command post and staging area.

        `geometry` is a GeoJSON Polygon or MultiPolygon in WGS84 falling on the
        staged raster - a `cliff_pitch` or `steep_band` shape, a drawn polygon,
        a buffered line. Four things come back:

        - `face` - vertical extent, span along the contour, slope quartiles,
          aspect. Vertical extent is the number a rigger reads.
        - `bench_above` - the band within `band_m` of the high edge and level
          with it: the ground the operation actually stands on. A site with a
          good face and no bench is a different problem from the reverse, so
          this is never folded into one number.
        - `landing_below` - the same at the low edge, where a litter arrives.
        - `vegetation` - lifeform fractions on all THREE footprints separately,
          because a clean face above a scrub bench is not a usable site. Brush
          disqualifies ground however good the slope: it cannot be moved
          through with a litter and it hides the lip.

        A missing vegetation panel is a warning and a null, never a zero - a
        shrub fraction of 0.0 that really means "not measured" is the most
        dangerous answer this could give.

        There is deliberately no feasibility verdict. The thresholds that turn
        a shrub fraction or a bench slope into a decision belong to the agency
        running the operation; this reports what the ground is.
        """
        return site_facts(ctx, handle=handle, geometry=geometry, band_m=band_m)

    @mcp_tool(mcp, name="list_feature_kinds", tags={"component"})
    def _list_feature_kinds_tool() -> dict[str, Any]:
        """The terrain feature kinds this server can identify.

        One entry per kind `identify_features` accepts: the kind name, one
        line on what it finds, and the thresholds the detector assumes.
        Generated from the detector registry, so it cannot go stale.
        """
        return list_feature_kinds(ctx)

    @mcp_tool(mcp, name="identify_features", tags={"component"})
    def _identify_features_tool(handle: str, kinds: list[str]) -> dict[str, Any]:
        """Identify selected terrain features on a staged assignment.

        `kinds` names which features to look for - `list_feature_kinds`
        enumerates them. Only what is asked for is computed: asking for
        `drainage` alone never reads vegetation or roads. Each kind comes
        back as the same envelope - shapes with citable ids, a summary,
        warnings, and the assumptions used. A missing input degrades into
        a warning, never an error; an unknown kind, or an empty list, is
        an error naming the kinds that exist. Carries the same `staleness`
        block as every other computation, for the same reason.
        """
        return identify_features(ctx, handle=handle, kinds=kinds)

    @mcp_tool(mcp, name="slope_stats", tags={"component"})
    def _slope_stats_tool(handle: str, shape_id: str) -> dict[str, Any]:
        """Slope statistics inside one named shape of a staged assignment.

        `shape_id` is the registry id that appears in that assignment's
        terrain metrics - the answer to "quantify that sidewall"; call
        `terrain_metrics` on the handle first. A shape
        covering no valid elevation returns nulls with `pixels: 0`, not
        zeros: a zero mean slope reads as flat ground.
        """
        return slope_stats(ctx, handle=handle, shape_id=shape_id)

    @mcp_tool(mcp, name="elevation_profile", tags={"component"})
    def _elevation_profile_tool(handle: str, drainage_id: str,
                                samples: int = 50) -> dict[str, Any]:
        """Elevation along a drainage centreline - how steep is the channel.

        Distances come back in true ground metres, not the Web Mercator
        units the grid is stored in, which at California latitudes differ
        by about a quarter. `drainage_id` comes from `terrain_metrics`,
        which has to be called on the handle first.
        """
        return elevation_profile(ctx, handle=handle, drainage_id=drainage_id,
                                 samples=samples)
