"""Feature: roads and trails on the ground the map does not show."""

from __future__ import annotations

from typing import Any, Literal

from ..context import ServerContext, mcp_tool
from ..planning.unmarked_paths import unmarked_paths as unmarked_paths_impl

#: The sources `unmarked_paths` reads, as an enum in the schema so a model
#: sees the allowed values. Kept equal to `unmarked_paths.ALL_SOURCES` by
#: tests/test_tool_registration.py.
UnmarkedSource = Literal["usfs_roads", "usfs_trails", "blm", "usgs_trails",
                         "usgs_roads", "terrain", "imagery"]


def unmarked_paths(ctx: ServerContext, handle: str,
                   sources: list[UnmarkedSource] | None = None,
                   known_tolerance_m: float | None = None,
                   track_width_m: float = 4.0, min_length_m: float = 50.0,
                   limit: int | None = None, refresh: bool = False,
                   write_geojson: bool = True,
                   geojson_limit: int = 0, bridges: bool = True) -> dict[str, Any]:
    """The `unmarked_paths` tool as a plain function: `ctx`, then the
    tool's arguments; the tool's payload back. What each argument and field
    means is in the tool's description, in `register` below."""
    resolved = ctx.staged(handle)
    payload = unmarked_paths_impl(resolved.directory, sources=sources,
                                  known_tolerance_m=known_tolerance_m,
                                  track_width_m=track_width_m,
                                  min_length_m=min_length_m, limit=limit,
                                  refresh=refresh, write_geojson=write_geojson,
                                  geojson_limit=geojson_limit, bridges=bridges)
    payload["warnings"] = [*payload["warnings"], *resolved.status.get("warnings", [])]
    return {**payload, "staleness": resolved.status}


def register(mcp, ctx) -> None:

    @mcp_tool(mcp, name="unmarked_paths", tags={"feature"})
    def _unmarked_paths_tool(handle: str, sources: list[UnmarkedSource] | None = None,
                             known_tolerance_m: float | None = None,
                             track_width_m: float = 4.0, min_length_m: float = 50.0,
                             limit: int | None = None, refresh: bool = False,
                             write_geojson: bool = True,
                             geojson_limit: int = 0, bridges: bool = True) -> dict[str, Any]:
        """Roads and trails the land agencies have on record that the map
        does not show, ranked by how many independent sources saw each.

        "Unmarked" means not in the staged OpenStreetMap ways - what nearly
        every map app, CalTopo's base layers included, draws. The sources
        (`sources`; by default every agency source plus `imagery`, and NOT
        `terrain`):

        - `usfs_roads` - the Forest Service road inventory, open AND closed.
          A closed road (maintenance level 1) is often still a walkable bench
          and often missing from OSM: the most useful thing this finds.
        - `usfs_trails` - Forest Service system trails.
        - `blm` - BLM routes (GTLF).
        - `usgs_trails`, `usgs_roads` - the National Map's trails, 4WD and
          closed roads. Much of this is other agencies' data republished, so
          it is counted under its originator, never as a second sighting.
        - `imagery` - not a line source but a check on every line the others
          found: USGS NAIP aerial imagery, read for a corridor barer (lower
          NDVI) than the ground 8 m either side, the way a dirt road reads -
          does an old road still show on the ground? Reported as
          `imagery.bare` on each line and NEVER counted as a sighting: a dry
          creek bed or a bare ridge passes it as readily as a road, so it
          changes no tier and no order. Its tiles are cached with the staged
          ground; the first call fetches them (about 15-60 s). `imagery.bare`
          null means NOT LOOKED AT.
        - `terrain` - EXPERIMENTAL, read only when named. Benches in the DEM
          (the same measurement as `shaded_relief`'s lineaments): ground
          flatter than the slope around it, which a ridge crest or a channel
          bottom is as much as a cut road. On four public-land areas 82-99%
          of its lines sat on a crest or in a channel. Every call that reads
          it carries a warning saying so; treat a `terrain_only` line as a
          place to look, never as a path.

        `bridges` (default on) joins a line's end to the nearest other line
        or mapped way within 40 m along FLAT ground, where a line fades out
        at a junction or landing. Bridges come back in their own
        `bridges` list, and green in the CalTopo file - a bridge is a join,
        never a detection.

        Pieces from different sources along the same ground become one
        candidate. Each says `tier` - `corroborated` (two or more independent
        sources), `agency_record` (one agency inventory) or `terrain_only`
        (only when `terrain` was named) -
        with `independent_sources`, every `sightings` entry and how much of
        the line it covers, `length_m`, `nearest_mapped`, and
        `close_to_mapped` when it may be a mapped way drawn elsewhere.
        `points` gives start, middle and end to navigate to.

        A failed source is `status: failed` in `sources` and a warning - NOT
        LOOKED FOR, never "none here". Agency answers are cached with the
        staged ground; `refresh=True` asks again.

        With `write_geojson` (default), the candidates are written to
        `<staged>/exports/unmarked-paths.geojson`, which CalTopo imports
        (Import, choose the file) as line Shapes coloured by tier with the
        evidence in each description. Nothing is sent to CalTopo. The file
        holds EVERY candidate by default (`geojson_limit=0`), not just the
        `limit` returned here: a road found in short pieces ranks low, and
        a planner looking at the map should still see it.

        Nothing here is a verified route. An agency line is where a road was
        surveyed, perhaps decades ago; a bench can be a ridge, a channel or
        a fault scarp. Game
        trails are not in any of these sources and are mostly too narrow for
        the DEM - see `assumptions.game_trails`. Look at `shaded_relief`'s
        images to judge a candidate by eye.
        """
        return unmarked_paths(ctx, handle=handle, sources=sources,
                              known_tolerance_m=known_tolerance_m,
                              track_width_m=track_width_m, min_length_m=min_length_m,
                              limit=limit, refresh=refresh, write_geojson=write_geojson,
                              geojson_limit=geojson_limit, bridges=bridges)
