"""Unmarked roads and trails on staged ground, from every source at once.

The sources are the land agencies' records (`sources/agency.py`), checked
against NAIP imagery (`imagery.py`), and - experimental, only when named -
the elevation (`terrain/lineaments.py`). "Unmarked" is measured against the
staged OpenStreetMap ways - what almost every map app, CalTopo's base layers
among them, draws. The fusion and its rules are in `terrain/unmarked.py`;
this module holds the I/O: the staged directory, the agency fetches, the
WGS84 emission and the CalTopo file.

NOT A WRITE TO CALTOPO
----------------------
Like `caltopo_export`, this writes a GeoJSON file beside the staged ground
and nothing else (README, *What is not here*). A planner imports it into
CalTopo (Import, choose the file), where each candidate arrives as a line
Shape, coloured by how many independent sources saw it, with its evidence
in the description. Nothing reaches a map teams navigate by without a
person in between.

GAME TRAILS
-----------
No source here maps them. No agency inventories them, and a game trail's
30-50 cm tread is under one pixel of even 1 m LiDAR, so the bench method
sees one only where it is cut deep across a steep slope. A caller can try
`track_width_m=1.0` on 1 m ground, and should expect the noise floor to
rise with it. The response says so in `assumptions.game_trails`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..geo import ground_meters_per_mercator_unit, to_wgs84_geojson
from ..terrain.unmarked import (
    AGENCY_TOLERANCE_M,
    CLOSE_TO_MAPPED_M,
    MIN_LENGTH_M,
    SAMPLE_STEP_M,
    TIERS,
    cross_support,
    lineages_of,
    median_distance,
    merge_same_paths,
    tier_of,
    unmapped_parts,
)
from ..terrain.bridges import BRIDGE_ASSUMPTIONS
from ..sources.agency import AGENCY_SOURCES, SOURCES, load_agency_ways
from .imagery import IMAGERY_ASSUMPTIONS

#: "imagery" is not a line source: it checks every candidate the others
#: found for a bare corridor in NAIP (see `imagery.py`) and reports it as
#: `imagery.bare`. Since 0.42.0 it never counts as a sighting: a dry creek
#: bed or a bare ridge passes it as readily as a road.
ALL_SOURCES = (*AGENCY_SOURCES, "terrain", "imagery")

#: What a call reads when it names no sources: the agency inventories and the
#: imagery check on their lines. "terrain" is experimental and read only when
#: named: on four public-land areas (6.3 m ground, 2026-10-27) 82-99% of its
#: benches sat on a crest or in a channel, where the bench test fires as
#: readily as on a cut road.
DEFAULT_SOURCES = (*AGENCY_SOURCES, "imagery")

TERRAIN_EXPERIMENTAL = (
    "terrain is experimental: its benches are ground flatter than the slope "
    "around it, which a ridge crest or a channel bottom is as much as a cut "
    "road. On four public-land areas 82-99% of them sat on a crest or in a "
    "channel. Read terrain_only lines as a place to look, never as a path")

EXPORTS_DIRNAME = "exports"
EXPORT_FILENAME = "unmarked-paths.geojson"

DEFAULT_LIMIT = 50

#: Line colour in the CalTopo file, by tier. Chosen to read against both
#: CalTopo's topo and its imagery base, and apart from the red of a drafted
#: assignment.
TIER_STROKE = {"corroborated": "#FF00FF", "agency_record": "#FF8000",
               "terrain_only": "#FFD700"}
TIER_TITLE = {"corroborated": "Unmarked?", "agency_record": "Unmarked?",
              "terrain_only": "Bench?"}

#: A bridge across a junction gap - drawn apart, so it is never read as a
#: line that was seen.
BRIDGE_STROKE = "#00C853"

#: OSM route classes a person travels along: an agency road counts as
#: mapped only when one of THESE runs with it, so a Forest Service road
#: under a mapped power line is still unmarked. Terrain benches are
#: compared against every staged way instead, as `shaded_relief` does - a
#: pipeline cut or a stream terrace is a mapped feature, not a lost road.
MAPPED_PATH_CLASSES = frozenset({"TRAIL", "TRACK", "ROAD", "ROAD_OTHER"})

UNMARKED_ASSUMPTIONS = {
    "unmarked_means": ("not within the tolerance of a way in the staged "
                       "OpenStreetMap data, for at least min_length_m; not 'absent "
                       "from every map'"),
    "agency_tolerance_m": AGENCY_TOLERANCE_M,
    "min_length_m": MIN_LENGTH_M,
    "same_path_fraction": ("a piece within the tolerance of a candidate along at "
                           "least half its length is the same path"),
    "independence": ("agreement is counted in lineages (whose survey it is), not "
                     "layers: USGS trails republish Forest Service lines and count "
                     "once with them"),
    "tiers": {"corroborated": "two or more independent lineages saw it",
              "agency_record": "one land agency's inventory only",
              "terrain_only": ("a bench in the elevation only (experimental "
                               "terrain source) - read the noise floor before "
                               "the list")},
    "default_sources": ("the agency inventories, with imagery as a check on "
                        "their lines; terrain only when named, as experimental"),
    "imagery_role": ("a check reported as imagery.bare on each line, never a "
                     "sighting: it does not change a tier or the order"),
    "close_to_mapped": (f"median distance under {CLOSE_TO_MAPPED_M:.0f} m from a "
                        "mapped way: may be that way, drawn elsewhere by another "
                        "survey, rather than a second path"),
    "agency_records_are_stale": ("an inventory line is where a road was surveyed, "
                                 "not proof it is passable now; a closed road may "
                                 "be overgrown or washed out"),
    "game_trails": ("no source here maps them, and a 30-50 cm tread is below one "
                    "pixel of 1 m LiDAR; track_width_m=1.0 on 1 m ground may find a "
                    "deeply cut one, with a higher noise floor"),
    "status": "unverified - every candidate is a place to look, not a route",
}


def _check_sources(sources: list[str] | None) -> list[str]:
    chosen = list(DEFAULT_SOURCES) if not sources else list(dict.fromkeys(sources))
    unknown = [source for source in chosen if source not in ALL_SOURCES]
    if unknown:
        raise ValueError(f"unknown source(s) {unknown}; choose from {list(ALL_SOURCES)}")
    return chosen


def _checked_limit(limit: int | None) -> int | None:
    if limit is None:
        return DEFAULT_LIMIT
    if limit < 0:
        raise ValueError(f"limit must be 0 (every one) or more, got {limit}")
    return None if limit == 0 else limit


def _to_3857(geometry_wgs84: dict):
    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely.ops import transform

    to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    return transform(to_3857.transform, shape(geometry_wgs84))


def _end_points(line, transform_back) -> dict:
    start = transform_back(*line.coords[0])
    end = transform_back(*line.coords[-1])
    middle = transform_back(*line.interpolate(0.5, normalized=True).coords[0])
    return {"start": {"latitude": round(start[1], 6), "longitude": round(start[0], 6)},
            "middle": {"latitude": round(middle[1], 6), "longitude": round(middle[0], 6)},
            "end": {"latitude": round(end[1], 6), "longitude": round(end[0], 6)}}


def _title(candidate: dict) -> str:
    words = [TIER_TITLE[candidate["tier"]]]
    if candidate["kind"] == "road":
        words.append("Rd" if not candidate.get("ref") else f"Rd {candidate['ref']}")
    elif candidate["kind"] == "trail":
        words.append("Trail" if not candidate.get("ref") else f"Trail {candidate['ref']}")
    if candidate.get("name"):
        words.append(candidate["name"].title())
    if candidate.get("status"):
        words.append(f"({candidate['status']})")
    return " ".join(words)


def _description(candidate: dict) -> str:
    parts = [f"{candidate['length_m']:.0f} m not in OpenStreetMap. "
             f"Seen by {len(candidate['independent_sources'])} independent source(s): "
             f"{', '.join(candidate['independent_sources'])}."]
    for sighting in candidate["sightings"]:
        label = sighting.get("layer") or sighting["source"]
        bits = [label]
        if sighting.get("ref") or sighting.get("name"):
            bits.append(" ".join(str(value) for value in (sighting.get("ref"),
                                                          sighting.get("name")) if value))
        if sighting.get("status"):
            bits.append(sighting["status"])
        parts.append(f"- {', '.join(bits)} (along {sighting['coverage'] * 100:.0f}%).")
    imagery = candidate.get("imagery") or {}
    if imagery.get("bare") is not None:
        parts.append(f"NAIP imagery: NDVI {imagery['along_ndvi']:.2f} along it, "
                     f"{imagery['flank_ndvi']:.2f} beside it"
                     + (" - a bare corridor." if imagery["bare"] else "."))
    nearest = candidate.get("nearest_mapped")
    if nearest:
        parts.append(f"Nearest mapped way: {nearest.get('name') or 'unnamed'}, "
                     f"{nearest['distance_m']:.0f} m.")
    if candidate.get("close_to_mapped"):
        parts.append("Runs close to a mapped way for most of its length: may be that "
                     "way drawn elsewhere.")
    parts.append(f"Draft from sar-geo-mcp unmarked_paths, {candidate['id']}; "
                 "unverified - check it on the ground before relying on it.")
    return " ".join(parts)


def _bridge_description(bridge: dict) -> str:
    target = bridge["to"] or f"mapped way {bridge.get('to_mapped') or '(unnamed)'}"
    parts = [f"{bridge['length_m']:.0f} m across flat ground from {bridge['from']} "
             f"to {target}, where a line fades out at a junction or landing. "
             f"Steepest {bridge['steepest_deg']:.0f} deg inside."]
    imagery = bridge.get("imagery") or {}
    if imagery.get("bare") is True:
        parts.append("Bare ground on NAIP imagery along it.")
    parts.append("A bridge is a join, not a detection - unverified.")
    return " ".join(parts)


def caltopo_lines(candidates: list[dict], bridges: list[dict] | None = None) -> dict:
    """Candidates (and bridges) as a FeatureCollection CalTopo imports as
    line Shapes."""
    features = []
    for bridge in bridges or []:
        features.append({
            "type": "Feature", "geometry": bridge["geometry"],
            "properties": {"class": "Shape", "title": f"Bridge? {bridge['id']}",
                           "description": _bridge_description(bridge),
                           "stroke": BRIDGE_STROKE, "stroke-width": 2,
                           "stroke-opacity": 0.9},
        })
    for candidate in candidates:
        features.append({
            "type": "Feature", "geometry": candidate["geometry"],
            "properties": {"class": "Shape", "title": _title(candidate),
                           "description": _description(candidate),
                           "stroke": TIER_STROKE[candidate["tier"]],
                           "stroke-width": 3, "stroke-opacity": 0.9},
        })
    return {"type": "FeatureCollection", "features": features}


def unmarked_paths(directory: Path, *, sources: list[str] | None = None,
                   known_tolerance_m: float | None = None,
                   track_width_m: float = 4.0, min_length_m: float = MIN_LENGTH_M,
                   limit: int | None = None, refresh: bool = False,
                   write_geojson: bool = True, geojson_limit: int | None = 0,
                   bridges: bool = True, get=None, imagery_get=None) -> dict[str, Any]:
    """Unmarked road and trail candidates on the staged ground, ranked by how
    many independent sources saw each one. See the module docstring.

    `sources` None reads DEFAULT_SOURCES: the agency inventories and the
    imagery check. "terrain" is read only when named, with a warning that
    it is experimental.

    `limit` bounds the list returned (default DEFAULT_LIMIT, 0 for every
    one); `geojson_limit` bounds the CalTopo file separately, and its
    default 0 writes EVERY candidate. They differ on purpose: the reply is
    read by an agent, the file is looked at on a map. With the file capped
    at the reply's 50, on real ground (Mendocino County, 2026-10-21) a
    planner importing it saw 50 of 562 candidates, and a road found in
    pieces ranked #127, #134, #271 and #382 never reached the map."""
    from pyproj import Transformer
    from shapely.geometry import box, shape
    from shapely.ops import unary_union

    from ..cache import atomic_write_text
    from ..terrain.grid import load_assignment_grid
    from .export import folder_title, in_one_folder
    from ..terrain.lineaments import (
        DEFAULT_TOLERANCE_M,
        LINEAMENT_ASSUMPTIONS,
        find_lineaments,
        subtract_known_ways,
    )
    from .rasters import (
        MIN_DENSITY_SAMPLE,
        NOISE_FLOOR_PER_KM2,
        _load_routes,
        georeference,
    )

    chosen = _check_sources(sources)
    limit = _checked_limit(limit)
    if geojson_limit is not None and geojson_limit < 0:
        raise ValueError(f"geojson_limit must be 0 (every one) or more, got {geojson_limit}")
    file_limit = geojson_limit or None
    if known_tolerance_m is not None and known_tolerance_m <= 0:
        raise ValueError(f"known_tolerance_m must be positive, got {known_tolerance_m}")
    if min_length_m <= 0:
        raise ValueError(f"min_length_m must be positive, got {min_length_m}")

    directory = Path(directory)
    grid = load_assignment_grid(directory)
    block = georeference(grid.transform, grid.elevation.shape, grid.latitude_deg)
    scale = ground_meters_per_mercator_unit(grid.latitude_deg)     # ground m per unit
    to_units = 1.0 / scale
    raster_box = box(*block["bbox_3857"])
    to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform
    warnings: list[str] = []

    agency_tolerance = known_tolerance_m or AGENCY_TOLERANCE_M
    terrain_tolerance = known_tolerance_m or DEFAULT_TOLERANCE_M
    fuse_tolerance = max(agency_tolerance, terrain_tolerance)

    routes, route_warnings = _load_routes(directory)
    osm_available = not route_warnings
    if route_warnings:
        warnings.append("osm.json missing: nothing is known to be mapped, so every "
                        "line is offered as unmarked")
    path_geoms = [shape(route["geometry"]) for route in routes
                  if route.get("geometry") and (route.get("properties") or {}).get(
                      "route_type") in MAPPED_PATH_CLASSES]
    mapped_paths = unary_union(path_geoms) if path_geoms else None
    mapped_buffer = (mapped_paths.buffer(agency_tolerance * to_units)
                     if mapped_paths is not None else None)
    named_paths = [(shape(route["geometry"]), route) for route in routes
                   if route.get("geometry") and (route.get("properties") or {}).get(
                       "route_type") in MAPPED_PATH_CLASSES]

    pieces: list[dict] = []
    per_source: dict[str, dict] = {}

    agency = [source for source in chosen if source in SOURCES]
    answers = (load_agency_ways(directory, block["bbox_wgs84"], agency,
                                refresh=refresh, get=get) if agency else {})
    for source in agency:
        answer = answers[source]
        if answer["status"] != "ok":
            per_source[source] = {"status": "failed", "reason": answer.get("reason"),
                                  "features": None, "km_on_ground": None,
                                  "km_unmarked": None, "pieces": None}
            warnings.append(f"{source} could not be read ({answer.get('reason')}); its "
                            "lines are NOT LOOKED FOR, not absent")
            continue
        km_on_ground = km_unmarked = 0.0
        count = 0
        for record in answer["features"]:
            geom = _to_3857(record["geometry"]).intersection(raster_box)
            if geom.is_empty:
                continue
            count += 1
            km_on_ground += geom.length * scale / 1000.0
            for part in unmapped_parts(geom, mapped_buffer, min_length_m * to_units):
                km_unmarked += part.length * scale / 1000.0
                pieces.append({key: record.get(key) for key in
                               ("source", "lineage", "layer", "kind", "name", "ref",
                                "status", "detail")} | {"geom": part})
        per_source[source] = {"status": "ok", "reason": None, "features": count,
                              "km_on_ground": round(km_on_ground, 3),
                              "km_unmarked": round(km_unmarked, 3),
                              "pieces": sum(1 for piece in pieces
                                            if piece["source"] == source),
                              "fetched_at": answer.get("fetched_at")}

    rows, cols = grid.elevation.shape
    ground_km2 = rows * cols * grid.pixel_size_m ** 2 / 1e6
    terrain_summary = None
    if "terrain" in chosen:
        warnings.append(TERRAIN_EXPERIMENTAL)
        found = find_lineaments(grid.elevation, grid.pixel_size_m,
                                track_width_m=track_width_m,
                                min_length_m=min_length_m)
        residue, subtract_warnings = subtract_known_ways(
            found, routes, tolerance_m=terrain_tolerance, transform=grid.transform,
            latitude_deg=grid.latitude_deg, min_length_m=min_length_m)
        warnings.extend(warning for warning in subtract_warnings
                        if warning not in warnings and osm_available)
        for lineament in residue:
            pieces.append({"source": "terrain", "lineage": "terrain",
                           "layer": "DEM bench (lineament)", "kind": "bench",
                           "name": None, "ref": None, "status": None,
                           "detail": {key: lineament.get(key) for key in
                                      ("length_m", "sinuosity", "mean_width_m",
                                       "end_to_end_bearing_deg", "traced_by",
                                       "clipped_from_m", "mapped_share")
                                      if key in lineament},
                           "geom": shape(lineament["geometry"])})
        per_km2 = len(residue) / ground_km2 if ground_km2 else None
        terrain_summary = {"lineaments": len(residue),
                           "per_km2": round(per_km2, 1) if per_km2 is not None else None,
                           "noise_floor_per_km2": NOISE_FLOOR_PER_KM2,
                           "ground_km2": round(ground_km2, 4)}
        per_source["terrain"] = {"status": "ok", "reason": None, "features": len(found),
                                 "pieces": len(residue), "per_km2": terrain_summary["per_km2"],
                                 "noise_floor_per_km2": NOISE_FLOOR_PER_KM2}
        if (per_km2 is not None and per_km2 >= NOISE_FLOOR_PER_KM2
                and len(residue) >= MIN_DENSITY_SAMPLE):
            warnings.append(
                f"terrain found {len(residue)} benches over {ground_km2:.2f} km2 "
                f"({per_km2:.0f} per km2), at or above the ~{NOISE_FLOOR_PER_KM2:.0f} per "
                "km2 the method reports on undisturbed rough ground: read terrain_only "
                "candidates as a noise floor, and trust a bench only where another "
                "source agrees")

    candidates = merge_same_paths(pieces, fuse_tolerance * to_units)
    cross_support(candidates, fuse_tolerance * to_units)

    sampler = None
    if "imagery" in chosen:
        sampler = _imagery(directory, block, grid.latitude_deg, candidates, per_source,
                           warnings, refresh=refresh, get=imagery_get)

    for candidate in candidates:
        candidate["tier"] = tier_of(candidate)
        candidate["independent_sources"] = lineages_of(candidate)
        median_units = median_distance(candidate["geom"], mapped_paths,
                                       SAMPLE_STEP_M * to_units)
        candidate["median_m"] = None if median_units is None else median_units * scale
        candidate["close"] = (candidate["median_m"] is not None
                              and candidate["median_m"] < CLOSE_TO_MAPPED_M)
    # Within a tier, a line hugging a mapped way goes last: on real ground
    # (Tahoe NF, 2026-10-21) the terrain-only benches at 0 m from a mapped
    # road were that road's own cut and fill, not a second path.
    tier_rank = {tier: index for index, tier in enumerate(TIERS)}
    candidates.sort(key=lambda item: (tier_rank[item["tier"]],
                                      -len(item["independent_sources"]),
                                      item["close"], -item["geom"].length))

    for rank, candidate in enumerate(candidates):
        candidate["id"] = f"unmarked-{rank}"
    bridge_records = (_bridges(grid, candidates, path_geoms, routes, sampler, to_wgs84)
                      if bridges and candidates else [])

    total_before_limit = len(candidates)
    by_tier = {tier: sum(1 for item in candidates if item["tier"] == tier) for tier in TIERS}
    if limit is not None and total_before_limit > limit:
        in_file = (min(file_limit, total_before_limit) if file_limit else total_before_limit)
        warnings.append(f"only the first {limit} of {total_before_limit} candidates are "
                        "listed (limit, ranked by tier then length); limit=0 returns "
                        "every one" + (f". The CalTopo file holds {in_file}"
                                       if write_geojson else ""))
    # Describe as many as either the reply or the file needs.
    wanted = [bound for bound in (limit, file_limit if write_geojson else limit)]
    needed = None if None in wanted else max(wanted)
    if needed is not None:
        candidates = candidates[:needed]

    described = []
    for index, candidate in enumerate(candidates):
        line = candidate["geom"]
        nearest = None
        if named_paths:
            gap, route = min(((geometry.distance(line), route)
                              for geometry, route in named_paths), key=lambda pair: pair[0])
            nearest = {"name": (route.get("properties") or {}).get("name"),
                       "distance_m": round(gap * scale, 1)}
        median_m = candidate["median_m"]
        described.append({
            "id": candidate["id"],
            "bridged_to": [bridge["id"] for bridge in bridge_records
                           if candidate["id"] in (bridge["from"], bridge["to"])],
            "imagery": candidate.get("imagery"),
            "tier": candidate["tier"],
            "kind": candidate["kind"],
            "name": candidate.get("name"),
            "ref": candidate.get("ref"),
            "status": candidate.get("status"),
            "length_m": round(line.length * scale, 1),
            "independent_sources": candidate["independent_sources"],
            "sightings": candidate["sightings"],
            "detail": candidate.get("detail"),
            "nearest_mapped": nearest,
            "median_distance_to_mapped_m": None if median_m is None else round(median_m, 1),
            "close_to_mapped": candidate["close"],
            "points": _end_points(line, to_wgs84),
            "geometry": to_wgs84_geojson(line),
        })

    out = described[:limit] if limit is not None else described

    def touching(listed):
        ids = {item["id"] for item in listed}
        return [bridge for bridge in bridge_records
                if bridge["from"] in ids or bridge["to"] in ids]

    export = None
    if write_geojson:
        in_file = described[:file_limit] if file_limit else described
        file_bridges = touching(in_file)
        path = directory / EXPORTS_DIRNAME / EXPORT_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = caltopo_lines(in_file, file_bridges)
        lines["features"] = in_one_folder(lines["features"], folder_title("Unmarked paths"))
        atomic_write_text(path, json.dumps(lines, indent=1))
        export = {"path": str(path), "features": len(in_file), "bridges": len(file_bridges),
                  "how_to_import": ("CalTopo: Import, choose this file. Each candidate "
                                    "arrives as a line Shape - magenta corroborated, "
                                    "orange one agency record, gold terrain only, "
                                    "green a bridge across a junction - with its "
                                    "evidence in the description")}

    return {
        "candidates": out,
        "bridges": touching(out),
        "summary": {"count": len(out), "total_before_limit": total_before_limit,
                    "bridges": len(bridge_records),
                    "by_tier": by_tier, "truncated_to": limit,
                    "osm_available": osm_available, "terrain": terrain_summary},
        "sources": per_source,
        "export": export,
        "georeference": block,
        "assumptions": {**UNMARKED_ASSUMPTIONS,
                        "terrain_method": LINEAMENT_ASSUMPTIONS["method"],
                        "terrain_false_positives": LINEAMENT_ASSUMPTIONS["false_positives"],
                        "terrain_tolerance_m": terrain_tolerance,
                        "bridges": BRIDGE_ASSUMPTIONS,
                        "imagery": IMAGERY_ASSUMPTIONS,
                        "agency_tolerance_used_m": agency_tolerance},
        "warnings": warnings,
    }


def _imagery(directory: Path, block: dict, latitude_deg: float, candidates: list[dict],
             per_source: dict, warnings: list[str], *, refresh: bool, get):
    """Fetch NAIP over the box and test every candidate for a bare corridor,
    recorded as `candidate["imagery"]` - a check, never a sighting. Returns
    the sampler, or None when no tile could be fetched."""
    from .imagery import NdviSampler, bare_corridor, load_naip

    answer = load_naip(directory, block["bbox_3857"], latitude_deg, refresh=refresh, get=get)
    summary = {key: answer[key] for key in ("status", "tiles", "missing", "pixel_m", "reason")}
    if answer["status"] == "failed":
        summary.update({"checked": None, "bare": None, "not_looked_at": None})
        per_source["imagery"] = summary
        warnings.append(f"imagery could not be read ({answer['reason']}); no line was "
                        "checked against it - NOT LOOKED AT, not 'no bare corridor'")
        return None
    if answer["status"] == "partial":
        warnings.append(f"imagery: {len(answer['missing'])} of {answer['tiles']} tiles "
                        f"could not be read ({answer['reason']}); lines there carry "
                        "imagery.bare = null, NOT LOOKED AT")

    sampler = NdviSampler(answer["paths"])
    bare = unknown = 0
    for candidate in candidates:
        result = bare_corridor(candidate["geom"], sampler, latitude_deg)
        candidate["imagery"] = result
        if result["bare"] is None:
            unknown += 1
        elif result["bare"]:
            bare += 1
    summary.update({"checked": len(candidates) - unknown, "bare": bare,
                    "not_looked_at": unknown})
    per_source["imagery"] = summary
    return sampler


def _bridges(grid, candidates: list[dict], path_geoms, routes, sampler, to_wgs84) -> list[dict]:
    """Junction bridges between candidate ends and other candidates or
    mapped paths, along flat ground - see `terrain/bridges.py`."""
    from shapely.geometry import LineString

    from ..geo import to_wgs84_geojson
    from ..terrain.bridges import junction_bridges
    from ..terrain.slope import compute_slope_aspect
    from ..terrain.unmarked import line_parts
    from .imagery import bare_corridor

    inverse = ~grid.transform

    def to_pixels(line):
        return LineString([inverse * point for point in line.coords])

    lines = [to_pixels(candidate["geom"]) for candidate in candidates]
    mapped, mapped_names = [], []
    for geometry, route in zip(path_geoms, [route for route in routes if route.get("geometry")
                                            and (route.get("properties") or {}).get(
                                                "route_type") in MAPPED_PATH_CLASSES]):
        for part in line_parts(geometry):
            mapped.append(to_pixels(part))
            mapped_names.append((route.get("properties") or {}).get("name"))
    slope = compute_slope_aspect(grid.elevation, grid.pixel_size_m).slope_deg
    found = junction_bridges(lines, lines + mapped, slope, grid.pixel_size_m)

    records = []
    for index, bridge in enumerate(found):
        line_3857 = LineString([grid.transform * point for point in bridge["path"]])
        to_line = bridge["to"] < len(candidates)
        records.append({
            "id": f"bridge-{index}",
            "from": candidates[bridge["from"]]["id"], "end": bridge["end"],
            "to": candidates[bridge["to"]]["id"] if to_line else None,
            "to_mapped": None if to_line else mapped_names[bridge["to"] - len(candidates)],
            "length_m": bridge["length_m"], "steepest_deg": bridge["steepest_deg"],
            "p80_deg": bridge["p80_deg"],
            "imagery": (bare_corridor(line_3857, sampler, grid.latitude_deg)
                        if sampler is not None else None),
            "geometry": to_wgs84_geojson(line_3857),
        })
    return records
