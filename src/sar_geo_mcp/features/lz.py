"""Feature: where a helicopter can land."""

from __future__ import annotations

from typing import Any

from ..context import ServerContext, mcp_tool
from ..planning.lz import lz_candidates as lz_candidates_impl


def lz_candidates(ctx: ServerContext, handle: str, center_latitude: float, center_longitude: float,
                  limit: int = 10,
                  max_slope_deg: float = 5.0, marginal_slope_deg: float = 7.0,
                  max_canopy: float = 0.20, min_diameter_m: float = 25.0,
                  preferred_width_m: float = 45.0, preferred_length_m: float = 60.0,
                  powerline_notice_m: float = 150.0,
                  powerline_clearance_m: float = 60.0,
                  structure_clearance_m: float = 30.0,
                  whole_ground: bool = False) -> dict[str, Any]:
    """The `lz_candidates` tool as a plain function: `ctx`, then the tool's
    arguments; the tool's payload back. What each argument and field means
    is in the tool's description, in `register` below."""
    resolved = ctx.staged(handle)
    payload = lz_candidates_impl(
        resolved.directory, center_latitude=center_latitude,
        center_longitude=center_longitude, limit=limit,
        max_slope_deg=max_slope_deg, marginal_slope_deg=marginal_slope_deg,
        max_canopy=max_canopy, min_diameter_m=min_diameter_m,
        preferred_width_m=preferred_width_m, preferred_length_m=preferred_length_m,
        powerline_notice_m=powerline_notice_m,
        powerline_clearance_m=powerline_clearance_m,
        structure_clearance_m=structure_clearance_m, whole_ground=whole_ground,
        wms_base_url=ctx.config.wms_base_url(), cache_dir=ctx.config.cache_dir,
        mapbox_token=ctx.config.mapbox_token, google_maps_key=ctx.config.google_maps_key)
    payload["warnings"] = [*payload.get("warnings", []),
                           *resolved.status.get("warnings", [])]
    payload["staleness"] = resolved.status
    # No map_provenance, unlike parking_candidates: a landing zone is
    # graded off rasters and OpenStreetMap and reads nothing from the
    # CalTopo map, so nothing here dates from a map read.
    return payload


def register(mcp, ctx) -> None:

    @mcp_tool(mcp, name="lz_candidates", tags={"feature"})
    def _lz_candidates_tool(handle: str, center_latitude: float, center_longitude: float,
                            limit: int = 10,
                            max_slope_deg: float = 5.0, marginal_slope_deg: float = 7.0,
                            max_canopy: float = 0.20, min_diameter_m: float = 25.0,
                            preferred_width_m: float = 45.0, preferred_length_m: float = 60.0,
                            powerline_notice_m: float = 150.0,
                            powerline_clearance_m: float = 60.0,
                            structure_clearance_m: float = 30.0,
                            whole_ground: bool = False) -> dict[str, Any]:
        """Where a helicopter can put down near a staged area: open, flat
        patches graded the way the CalTopo First Response safety course
        grades a landing zone, nearest the given point first.

        Call `prepare_area` first - WITH `resolution_m=2.0`: at the default
        raster (3.9 m per pixel; the limit is 3 m) a 25 m circle is a
        handful of cells and this tool refuses to grade it, naming the
        pixel size. Pass the point the planner asked about as the centre,
        so distance and bearing are from what they typed.

        THREE GRADES, AND WHAT EACH IS. `preferred`: a `min_diameter_m`
        circle AND a `preferred_width_m` by `preferred_length_m` rectangle
        fit inside open ground at 0 to `max_slope_deg`. `minimal`: only the
        circle fits at that slope. `marginal`: the circle fits only once the
        slope limit is relaxed to `marginal_slope_deg` - offered, never
        preferred, and only for a patch holding no better pocket. Open means
        not vegetated on the NAIP false-colour proxy once one- and two-cell
        specks are ignored; a candidate whose footprint is more than
        `max_canopy` vegetated is dropped. The course's own numbers are 0 to
        5 degrees (7 if desperate), 0 to 20 % canopy, 75 ft across at the
        least and 150 by 200 ft for a medium aircraft.

        Each candidate carries a `touchdown` Point at the centre of the
        largest circle the patch holds (the point farthest from every edge),
        `clear_diameter_m`, `preferred_fit` (whether the rectangle fits and
        at what angle, anticlockwise from east on a north-up map), slope
        mean and max over that circle, its canopy fraction, elevation,
        aspect (null on ground too flat or too mixed to have one), geodesic
        distance and bearing from the centre (bearing null underfoot), the
        patch polygon in WGS84, `powerline`, `road`, `parking_lot`, `chips` and
        `chip_path`.

        FOUR CLOSURES RUN BEFORE ANY FOOTPRINT IS FITTED, so a clearing split
        by a wire or a house is graded on what is left of it. Ground within
        `powerline_clearance_m` of a mapped power line of either kind - a
        transmission line (`power=line`) or a distribution line
        (`power=minor_line`) - is closed; lines that are unknown close
        nothing. Ground within `structure_clearance_m` of a building on
        CalTopo's structures layer is closed too; when that layer cannot be
        had it is a warning and `assumptions.structures` is "not checked",
        never a refusal, since the layer is incomplete even when present.
        `structures: "checked"` means CalTopo's mapped footprints were kept
        clear of, not that there are no buildings - an unmapped one is for
        the chip. A power-line clearance of 0 closes nothing; a structure
        clearance of 0 still closes the footprint itself; a negative
        clearance is refused.

        TREES ARE CLOSED FROM TRUE-COLOUR NAIP (CalTopo `n3`, through the
        shared `terrain.trees` block): a cast shadow, and a green crown within
        6 m of one, close the cell they are in - no speck forgiveness, so one
        pine keeps a clear circle off it. The CIR canopy proxy above misses
        dark conifers (a scatter of pines read as 3 % there), which is why this
        runs as well. A wet meadow is green but throws no shadow and stays
        open. Without a WMS token or when the imagery cannot be had, a warning
        says trees were not checked and `assumptions.trees` is "not checked";
        `assumptions.trees_method` states the rule.

        OPEN WATER IS CLOSED TOO, since a lake is flat and treeless and would
        otherwise grade as the best zone on the map: LANDFIRE open water from
        the staged `evt.tif` (30 m, so it may close a little shore) and every
        OpenStreetMap lake, reservoir or coast shoreline, a closed ring filled.
        `assumptions.water_sources_checked` lists which of
        "landfire_open_water" and "osm_shorelines" were read; with neither, a
        warning says water was not checked and the chips are the only check.

        `powerline` is ONE OF THREE THINGS and they are never collapsed: the
        nearest staged power line of either kind within `powerline_notice_m`
        of the touchdown as `{distance_m, name, power}`, `power` being `line`
        or `minor_line`; null when none is that close; the string "unknown"
        when OpenStreetMap could not be read for this ground, was staged
        before any barrier was fetched (query version 1), or was staged
        before wires were labelled. Read null as "no mapped wire that close",
        never as "no wire". `assumptions.powerline_kinds_checked` says which
        kinds that answer covers: `["line", "minor_line"]` on current
        ground; `["line"]` on ground staged at query version 3, before
        distribution lines were asked for (a warning names the re-stage);
        `[]` when power lines are unknown.

        `road` is the nearest mapped road, track or other vehicle way (OSM
        ROAD, TRACK or ROAD_OTHER) within `powerline_notice_m` of the
        touchdown, as `{name, route_type, highway, surface, distance_m,
        on_road}`. `on_road` is true when the way comes within half
        `clear_diameter_m` of the touchdown - anywhere inside the clearing,
        however wide (the operator's ruling of 2026-09-23,
        "Within the whole clearing"), so a road 40 m off in a 195 m field is
        ON that zone and it needs traffic control; false means beside it. A way not built
        (proposed, under construction, abandoned) or not a road anyone means
        (a via ferrata, a bus guideway) is left out, as egress leaves it out;
        a motorway is kept. `road` is null when none is that close, and the
        string "unknown" when there is no OpenStreetMap for this ground. On
        ground staged before query version 5, `highway` is null - the road
        class was never asked for - and `assumptions.road_class` is "unknown"
        ("checked" from version 5); an unbuilt road cannot be told from a
        built one there, so it is named like any other. Never guess a class
        from the name. `parking_lot` is `{id, off}` when the touchdown lies
        inside a mapped parking lot's outline - not public hardstand, and not
        a lot mapped only as a point - where `off` names the road it opens
        onto, the same name `parking_candidates` gives it (null when no named
        way is near); null when the touchdown is in no lot; and the string
        "unknown" when there is no OpenStreetMap for this ground or it was
        staged before parking was fetched (query version 3). Read null as "in
        no mapped lot", never as "in no lot". Both are SAID, never used: a
        road or a lot closes, flags and regrades nothing. With no osm.json the
        warnings say power lines, roads and lots are all unknown.

        `chips` is a list of 200 m imagery squares around the touchdown
        with the patch outlined, for a human or a vision model to look at
        for the boulders, talus, wires and trees an elevation model cannot
        see: one `{source, label, path, attribution}` per source, in a
        fixed order - CalTopo imagery and CalTopo NAIP when the server has
        a WMS token, then Mapbox satellite and Google satellite, each only
        when the server is configured with that provider's key. Two sources
        are two looks at the same ground from different dates and cameras.
        ANY PAGE SHOWING A CHIP MUST SHOW ITS `attribution` BESIDE IT: the
        providers' terms require the credit, and the Mapbox chip carries
        only its logo on the image. A source that fails is left out with a
        warning naming the candidate and the source; a Mapbox or Google
        source that refuses (401 or 403) or cannot be reached is not asked
        again for the rest of the call, and one warning says so. A failed
        chip never fails the call. `chip_path` is the first chip's path,
        null when there is none. `assumptions.chip.sources` lists the
        sources ASKED, a refused one included; which chips exist is each
        candidate's `chips` list. Google's terms restrict how its imagery
        may be used, and a deployment must satisfy itself on them before
        configuring that key.

        The canopy panel is fetched through the same WMS path when the
        staged ground lacks it; with no panel and no token, or when that
        fetch fails, the call is refused outright, because a landing zone
        graded on slope alone is the false confidence the course warns
        against.

        WHERE IT LOOKS. Inside the planner's circle (or the assignment) by
        default. `whole_ground` grades every cell staged instead, the
        1,000 m planning buffer included - for a patient, whose nearest
        landing zone may lie just past the circle; the trees, canopy and
        structures checks then cover the buffer too, at the cost of more
        imagery. `assumptions.searched` says which.

        `limit` 0 returns every candidate; `summary.grades` counts each
        grade BEFORE the limit. NO VERDICT: this is a filter on elevation
        and imagery rasters, not a site survey. Be conservative, and the
        pilot decides.
        """
        return lz_candidates(ctx, handle=handle, center_latitude=center_latitude,
                             center_longitude=center_longitude, limit=limit,
                             max_slope_deg=max_slope_deg,
                             marginal_slope_deg=marginal_slope_deg, max_canopy=max_canopy,
                             min_diameter_m=min_diameter_m,
                             preferred_width_m=preferred_width_m,
                             preferred_length_m=preferred_length_m,
                             powerline_notice_m=powerline_notice_m,
                             powerline_clearance_m=powerline_clearance_m,
                             structure_clearance_m=structure_clearance_m,
                             whole_ground=whole_ground)
