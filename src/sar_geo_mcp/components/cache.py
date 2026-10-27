"""Components: the local cache, and filling it before connectivity is lost."""

from __future__ import annotations

from typing import Any

from ..context import ServerContext, mcp_tool
from ..prefetch import plan_prefetch, run_prefetch

#: The panel sets worth having offline, mirroring what the debrief
#: pipeline renders. Duplicated deliberately rather than imported: the app
#: depends on this package, never the other way round.
PANEL_SETS: dict[str, list[str]] = {
    "topo_slope": ["mbt", "sf"],
    "slope": ["sf"],
    "imagery": ["imagery"],
    "vegetation": ["ir"],
    "contours": ["mbt", "cm5"],
    "roads": ["mbt", "mbr", "mvum"],
}


def prefetch_map(ctx: ServerContext, map_id: str, public: bool = True,
                 execute: bool = False, budget_mb: float = 500.0,
                 include_panels: bool = True) -> dict[str, Any]:
    """The `prefetch_map` tool as a plain function: `ctx`, then the tool's
    arguments; the tool's payload back. What each argument and field means
    is in the tool's description, in `register` below."""
    state = ctx.state(map_id, public)
    base_url = ctx.config.wms_base_url() if include_panels else None
    panel_sets = PANEL_SETS if base_url else None

    if not execute:
        plan = plan_prefetch(map_id, state.features, ctx.config.cache_dir,
                             panel_sets=panel_sets)
        summary = plan.summary()
        summary["executed"] = False
        summary["panels_included"] = bool(base_url)
        if include_panels and not base_url:
            summary["note"] = ("no WMS token configured, so basemap "
                               "panels are not included; elevation "
                               "needs no credentials")
        return summary

    result = run_prefetch(map_id, state.features, ctx.config.cache_dir,
                          budget_bytes=int(budget_mb * 1e6),
                          panel_sets=panel_sets, base_url=base_url)
    result["executed"] = True
    return result


def cache_status(ctx: ServerContext) -> dict[str, Any]:
    """The `cache_status` tool as a plain function: `ctx` in, the tool's
    payload back. What each field means is in the tool's description, in
    `register` below."""
    from ..cache import CacheIndex

    if not ctx.config.cache_dir:
        return {"cache_dir": None, "note": "caching is disabled"}
    summary = CacheIndex(ctx.config.cache_dir).summary()
    summary["cache_dir"] = str(ctx.config.cache_dir)
    summary["mb"] = round(summary["bytes"] / 1e6, 1)
    return summary


def evict_cache(ctx: ServerContext, ceiling_mb: float = 5000.0,
                execute: bool = False) -> dict[str, Any]:
    """The `evict_cache` tool as a plain function: `ctx`, then the tool's
    arguments; the tool's payload back. What each argument and field means
    is in the tool's description, in `register` below."""
    from ..cache.evict import evict
    from ..staging import tiles_in_use

    if not ctx.config.cache_dir:
        return {"cache_dir": None, "note": "caching is disabled"}
    # The cache cannot tell which of its tiles staged ground rests on -
    # that is staging's knowledge, not plumbing's - so it is asked here
    # and the answer handed to the cache.
    return evict(ctx.config.cache_dir, int(ceiling_mb * 1e6), execute=execute,
                 protected=tiles_in_use(ctx.config.cache_dir))


def register(mcp, ctx) -> None:

    @mcp_tool(mcp, name="prefetch_map", tags={"component"})
    def _prefetch_map_tool(map_id: str, public: bool = True, execute: bool = False,
                           budget_mb: float = 500.0,
                           include_panels: bool = True) -> dict[str, Any]:
        """Cache a map's terrain ahead of losing connectivity.

        A command post is often somewhere with no signal, and finding that
        out at the trailhead is too late. This walks the map, works out
        which ground its features sit on, and pulls down the tiles.

        `execute` defaults to FALSE: the first call is an estimate and
        downloads nothing. Read `estimated_mb` and the `skipped` list, then
        call again with `execute=True`. Nothing here ever silently pulls a
        gigabyte down a tethered phone.

        Clusters of features far from any assignment - somebody's hospital
        pin, a staging area two counties away - are excluded and listed in
        `skipped` with the reason, so a narrowed prefetch is always visible
        rather than silent.
        """
        return prefetch_map(ctx, map_id=map_id, public=public, execute=execute,
                            budget_mb=budget_mb, include_panels=include_panels)

    @mcp_tool(mcp, name="cache_status", tags={"component"})
    def _cache_status_tool() -> dict[str, Any]:
        """What this machine has cached: size, age, and split by layer.

        Static ground (elevation, vegetation, roads, basemap) never expires
        - it is keyed by geography and shared between assignments. Map
        state expires by age, because it changes during an operation.
        """
        return cache_status(ctx)

    @mcp_tool(mcp, name="evict_cache", tags={"component"})
    def _evict_cache_tool(ceiling_mb: float = 5000.0,
                          execute: bool = False) -> dict[str, Any]:
        """Bring the cache back under a size ceiling, oldest use first.

        Reports by default and deletes only with `execute=True`: removing
        cached ground from a laptop that may be about to lose connectivity
        is not something to do on a guess.

        Only static ground is considered. Map state is kilobytes and
        expires by age, and on a machine with no signal it may be the only
        copy of a map there is. Tiles that staged assignments still rest on
        are never removed, and neither is any file the index has never seen
        - there is no honest basis for ranking something we know nothing
        about.
        """
        return evict_cache(ctx, ceiling_mb=ceiling_mb, execute=execute)
