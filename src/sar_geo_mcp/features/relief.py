"""Features: what the ground looks like, from CalTopo's rasters and our own."""

from __future__ import annotations

from typing import Any, Literal

from ..context import ServerContext, mcp_tool
from ..planning.rasters import shaded_relief as shaded_relief_impl
from ..planning.rasters import slope_shading as slope_shading_impl


def shaded_relief(ctx: ServerContext, handle: str, layer: str = "r3",
                  returns: Literal["data", "image", "both"] = "both",
                  track_width_m: float = 4.0,
                  known_tolerance_m: float = 15.0,
                  limit: int | None = None) -> dict[str, Any]:
    """The `shaded_relief` tool as a plain function: `ctx`, then the tool's
    arguments; the tool's payload back. What each argument and field means
    is in the tool's description, in `register` below."""
    resolved = ctx.staged(handle)
    return {**shaded_relief_impl(resolved.directory,
                                 wms_base_url=ctx.config.wms_base_url(),
                                 cache_dir=ctx.config.cache_dir,
                                 layer=layer, returns=returns,
                                 track_width_m=track_width_m,
                                 known_tolerance_m=known_tolerance_m,
                                 limit=limit),
            "staleness": resolved.status}


def slope_shading(ctx: ServerContext, handle: str, geometry: dict | None = None,
                  layer: str = "sf",
                  returns: Literal["data", "image", "both"] = "data") -> dict[str, Any]:
    """The `slope_shading` tool as a plain function: `ctx`, then the tool's
    arguments; the tool's payload back. What each argument and field means
    is in the tool's description, in `register` below."""
    resolved = ctx.staged(handle)
    return {**slope_shading_impl(resolved.directory,
                                 wms_base_url=ctx.config.wms_base_url(),
                                 cache_dir=ctx.config.cache_dir,
                                 geometry=geometry, layer=layer,
                                 returns=returns),
            "staleness": resolved.status}


def register(mcp, ctx) -> None:

    @mcp_tool(mcp, name="shaded_relief", tags={"feature"})
    def _shaded_relief_tool(handle: str, layer: str = "r3",
                            returns: Literal["data", "image", "both"] = "both",
                            track_width_m: float = 4.0,
                            known_tolerance_m: float = 15.0,
                            limit: int | None = None) -> dict[str, Any]:
        """Terrain you can see, and linear features OpenStreetMap has not got.

        Two things come back, and they answer differently.

        The IMAGES are for looking at: CalTopo's shaded relief and a local
        multi-azimuth hillshade, each clean and again with the staged OSM
        ways painted over it. Anything linear that is not painted is a
        candidate. The clean member of each pair is unmarked on purpose - a
        coordinate grid sits on top of exactly the faint grades you are
        looking for - so look at the clean one and locate on the annotated
        one. `georeference` makes pixel to coordinate arithmetic you can
        check.

        The LINEAMENTS are measured, and come from the elevation rather than
        from a picture of it: a cut road is a run of ground locally flatter
        than the hillside around it. Anything within `known_tolerance_m` of
        a mapped way is dropped, so what is left is what OSM has not got.

        Nothing here is a road. A fault scarp, a pipeline cut, a cattle
        terrace, an old railbed and a logging grade are indistinguishable by
        this method, and the false-positive rate on real ground has not been
        measured. Treat every lineament as somewhere to look, not somewhere
        to send a team.

        READ THE DENSITY BEFORE THE LIST. `lineament_summary.per_km2` beside
        `noise_floor_per_km2` is what says whether the list means anything:
        undisturbed rough ground reports about 60 per km2 by this method, so
        a run at 61 per km2 has found the noise floor, not sixty tracks.
        `limit` keeps the longest few (default 50, `limit=0` for every one),
        and `total_before_limit` says how many there were.

        `lineaments` is NULL under returns="image" - nothing was looked for.
        Null means not looked for; an empty list means looked for and none
        found, and the two are not the same claim about the ground.

        This tool needs NO CalTopo credential. Without a token only the two
        rendered relief images are absent, with a warning; the hillshade and
        the overlay come from ground already staged.
        """
        return shaded_relief(ctx, handle=handle, layer=layer, returns=returns,
                             track_width_m=track_width_m,
                             known_tolerance_m=known_tolerance_m, limit=limit)

    @mcp_tool(mcp, name="slope_shading", tags={"feature"})
    def _slope_shading_tool(handle: str, geometry: dict | None = None,
                            layer: str = "sf",
                            returns: Literal["data", "image", "both"] = "data") -> dict[str, Any]:
        """CalTopo's slope shading beside this package's own, and where they differ.

        The DEM under every other measurement here is a dynamic mosaic: ask
        for 1 m over ground flown at 10 m and it returns a smooth
        interpolation, with nothing in the response saying so. The cheap way
        to notice is a second, independently rendered slope source over the
        same ground, and that is what this is.

        `steeper_in_caltopo` is the deliverable: polygons where CalTopo reads
        at least one band steeper than the DEM does, ready to hand to
        `cliff_pitch` or `site_facts`. They are ranked by how badly the two
        disagree and then by vertical extent, never by area - a vertical face
        is narrow in plan, and ranking by square metres hides exactly the
        ground worth finding.

        `geometry` is an optional WGS84 Polygon restricting every fraction
        and patch to one footprint. Omitted, the whole staged extent is read.

        `steeper_in_caltopo` polygons are WGS84, like every GeoJSON this
        package emits, so a patch goes straight into either tool unchanged.

        TWO LIMITS WORTH KNOWING. The legend is UNVERIFIED until
        `scripts/derive_sf_palette.py` has been run against real ground -
        `legend.provenance` says which, and a high `caltopo.unmatched_fraction`
        usually means it is wrong rather than the ground odd (an
        interpolating resample can raise it too, which is why `sf` is fetched
        nearest-neighbour). And this tool has no fallback: with no WMS token
        every CalTopo figure is null - `steeper_in_caltopo` included, which
        is null for "not compared" rather than an empty list for "compared,
        and nothing reads steeper".

        No verdict. Which disagreement matters is the agency's call.
        """
        return slope_shading(ctx, handle=handle, geometry=geometry, layer=layer,
                             returns=returns)
