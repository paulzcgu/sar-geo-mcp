"""Building a staged fetch directory - the format this package already reads.

WHY THIS MOVED HERE
-------------------
`terrain/grid.py` and `terrain/metrics.py` have always read `manifest.json`,
`dem.tif`, `evt.tif`, `osm.json`, and `assignment.json` out of a directory.
This package therefore defined that format while something else produced it,
which is the wrong way round: a format nobody here writes is a format that
drifts, and it meant no consumer of this package could stage ground for
itself.

What stays with the debrief app is the basemap panels. They are pictures for
a human report and for vision analysis, they need a WMS token, and nothing
computed here reads them. The debrief app and the planning tools both call
`stage_terrain`; the app then adds its panels to the same directory.

NO CREDENTIALS
--------------
Everything staged here - elevation, vegetation, roads - comes from public
services. That is deliberate: a planner should be able to stage ground for
an assignment without anyone issuing them a key.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .cache import atomic_write_text
from .sources.dem import fetch_dem, read_dem_grid
from .sources.landfire import write_evt_clip
from .terrain.shapes import METRICS_FILENAME
from .geo import (
    Bbox,
    bbox_center_latitude,
    buffer_bbox,
    geometry_bounds_wgs84,
    ground_meters_per_mercator_unit,
    to_web_mercator,
)
from .tiles import level_for, tiles_covering

#: Longest edge of a staged raster, in pixels. Everything downstream was
#: tuned against this size, so it stays the default: a caller who does not ask
#: for a resolution gets the raster every existing consumer assumes.
STAGE_PIXELS = 1024

#: Ceiling on the longest edge, set by the 3DEP exportImage service rather
#: than by us (`sources.dem.MAX_DEM_PIXELS`). `fetch_dem` tiles at 1,024 and
#: mosaics, so the practical cost of a larger request is bandwidth and time,
#: not a different code path.
MAX_STAGE_PIXELS = 8000

#: Ground metres of margin around an assignment. Searchers leave the
#: boundary and coverage needs the ground beside it, so the staged extent is
#: deliberately wider than the polygon.
DEFAULT_BUFFER_M = 200.0

#: What a staged directory's manifest says when its first OpenStreetMap
#: fetch failed: `f"{OSM_UNAVAILABLE}: {reason}"`, built only by
#: `osm_outage_warning` and recognised only by `is_osm_outage_warning`.
#: Every release since the first has written exactly this prefix, so the
#: manifests already on disk are recognised by it too.
OSM_UNAVAILABLE = "osm unavailable"


def osm_outage_warning(reason: str) -> str:
    """The manifest warning `stage_terrain` records when Overpass fails."""
    return f"{OSM_UNAVAILABLE}: {reason}"


def is_osm_outage_warning(warning) -> bool:
    """True for a warning `osm_outage_warning` built - by this release or an
    earlier one. Anchored on the constant, never a substring search: a
    warning that merely mentions the words is some other fact."""
    return isinstance(warning, str) and warning.startswith(f"{OSM_UNAVAILABLE}: ")


#: Recorded INSTEAD of the outage warning when a re-stage of the same ground
#: could not refetch OpenStreetMap and the `osm.json` staged earlier for that
#: ground is kept. Deliberately not matched by `is_osm_outage_warning`: the
#: file exists and was fetched for this box, so nothing about it is missing.
OSM_REFETCH_FAILED = "the OSM refetch failed"


def osm_kept_warning(reason: str, version: int) -> str:
    """What `stage_terrain` records when it keeps the same ground's file."""
    return (f"{OSM_REFETCH_FAILED} ({reason}); the osm.json staged earlier for "
            f"this same ground is kept (query version {version})")


def is_osm_kept_warning(warning) -> bool:
    """True for a warning `osm_kept_warning` built."""
    return isinstance(warning, str) and warning.startswith(f"{OSM_REFETCH_FAILED} (")


def _retires_on_top_up(warning) -> bool:
    """A manifest warning a successful `refresh_osm` retires: the outage,
    or the note that an earlier same-ground file was kept. Both speak of
    the `osm.json` the top-up has just replaced."""
    return is_osm_outage_warning(warning) or is_osm_kept_warning(warning)


def _staged_bbox(out_dir: Path) -> tuple[float, ...] | None:
    """The `bbox_3857` the directory's manifest says it was staged over, or
    None when there is no manifest or it cannot say."""
    try:
        manifest = json.loads((Path(out_dir) / "manifest.json").read_text(encoding="utf-8"))
        bbox = tuple(float(value) for value in manifest["bbox_3857"])
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None
    return bbox if len(bbox) == 4 else None


def _same_ground(previous: tuple[float, ...] | None, bbox) -> bool:
    """Whether two staged boxes are the same ground. A millimetre of Web
    Mercator slack only: the box is computed deterministically from the
    subject and the buffer, and round-trips through JSON exactly."""
    if previous is None:
        return False
    return all(math.isclose(a, float(b), rel_tol=0.0, abs_tol=1e-3)
               for a, b in zip(previous, bbox))


def _readable_osm_version(osm_path: Path) -> int | None:
    """The query version of a readable `osm.json`, or None when the file is
    missing or cannot be parsed. A file without a version is version 1."""
    try:
        payload = json.loads(Path(osm_path).read_text(encoding="utf-8"))
        return int(payload.get("query_version") or 1)
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def discard_terrain_metrics(out_dir: Path) -> None:
    """Remove a saved `terrain_metrics.json` once the ground under it changes.

    The `terrain_metrics` tool saves its answer beside the staged ground,
    and `slope_stats`, `elevation_profile` and `coverage_for` read it from
    there. A re-stage writes a moved assignment into the SAME directory, and
    a top-up rewrites the `osm.json` the metrics read, so a file left behind
    would describe the old ground with no warning. Removing it makes those
    tools ask for `terrain_metrics` again instead.
    """
    (Path(out_dir) / METRICS_FILENAME).unlink(missing_ok=True)


def _remove_osm(osm_path: Path, what: str) -> None:
    """Remove `osm_path`, or raise a RuntimeError that names why and never
    where: "osm.json {what} could not be removed (PermissionError); stage
    again". An OSError's message is the full local path, and a tool error
    reaches the agent as text.

    The RuntimeError is raised OUTSIDE the `except` block, so it carries no
    `__context__` - raised inside, even `from None` would keep the
    PermissionError, path and all, one traceback printer away. Callers
    call this outside their own `except` blocks for the same reason.
    """
    from .net_errors import safe_reason

    failed = None
    try:
        Path(osm_path).unlink(missing_ok=True)
    except OSError as error:
        failed = safe_reason(error)
    if failed is not None:
        raise RuntimeError(f"{what} could not be removed ({failed}); stage again")


def _osm_failure_reason(error: BaseException) -> str:
    """Why an OpenStreetMap fetch failed, fit for a warning a planner reads.

    `net_errors.safe_reason` - the type, and an HTTP error's status - for
    everything but one exception: an `OSError`'s message is the local cache
    path it failed on and an httpx error's is the URL it asked, and neither
    belongs on the app's Safety page.

    The one exception is `OverpassUnavailable`, whose message the hub
    writes itself for exactly this reader, from constant phrases, the last
    status or exception type, and a host name (`linear._host`, which drops
    any credentials in front of it). Reducing it to its class name would
    undo the 2026-09-21 ruling that "osm unavailable: HTTPStatusError" told
    an operator nothing, where "429 Too Many Requests from overpass-api.de"
    tells them every mirror throttled them.
    """
    from .net_errors import safe_reason
    from .terrain.linear import OverpassUnavailable

    if isinstance(error, OverpassUnavailable):
        return f"{type(error).__name__}: {error}"
    return safe_reason(error)


def manifest_warnings(out_dir: Path, manifest: dict) -> list[str]:
    """The manifest's warnings as they stand now, for a caller that hands
    them back on reuse.

    WHY NOT JUST `manifest["warnings"]`
    -----------------------------------
    Two warnings describe one moment and stop being news after it.

    The OUTAGE warning describes the first fetch, written once at staging.
    Once an `osm.json` exists that was written AFTER it, the warning is
    false, whatever query version wrote the file: a file that is merely
    behind is `refresh_osm`'s to report, in its own warnings. "Written
    after" is read from the manifest: `osm_refreshed_versions` is appended
    by every successful top-up, by every hub since `refresh_osm` existed.
    An outage beside a file with no such record is kept in an ASSIGNMENT
    directory - that file is a leftover, such as an older hub's re-stage of
    a moved boundary leaving the old boundary's file in place (this hub
    removes it; see the end of `stage_terrain`'s OSM block). An AREA
    directory has one bbox for life, so its file is its own ground whoever
    wrote it - an older hub's forced re-stage in an outage, or a staging
    that crashed between osm.json and the manifest - and the outage is
    dropped there as it would be after a top-up. Before 0.17.1 nothing
    took the outage out: `refresh_osm` now does when it succeeds, but
    ground topped up under 0.17.0 or earlier is caught only here.

    The KEPT-FILE warning (`osm_kept_warning`) is returned in the response
    of the staging whose refetch failed, and stays in the manifest as a
    record; on reuse the file it names is simply there, so it is dropped
    whenever `osm.json` exists.

    Every other warning is returned as written.
    """
    warnings = list(manifest.get("warnings", []))
    if not (Path(out_dir) / "osm.json").is_file():
        return warnings
    # An area's key hashes its geometry and buffer, so its directory has one
    # bbox for life and any file in it is its own ground, whoever wrote it.
    # Only an assignment's boundary can move under a leftover file.
    written_after_outage = (bool(manifest.get("osm_refreshed_versions"))
                            or manifest.get("subject_kind") == "area")
    return [w for w in warnings
            if not is_osm_kept_warning(w)
            and not (is_osm_outage_warning(w) and written_after_outage)]


def compute_aoi(geometry: dict, buffer_m: float) -> Bbox:
    """Project the assignment and expand it by a true ground distance."""
    bounds_wgs84 = geometry_bounds_wgs84(geometry)
    latitude = bbox_center_latitude(bounds_wgs84)
    bbox_3857 = to_web_mercator(bounds_wgs84)
    if buffer_m == 0.0:
        return bbox_3857
    return buffer_bbox(bbox_3857, ground_meters=buffer_m, latitude_deg=latitude)


def pixels_for_resolution(extent_m: float, resolution_m: float) -> int:
    """Longest raster edge needed to measure `extent_m` at `resolution_m`.

    WHY A CALLER WOULD ASK
    ----------------------
    The 1,024-pixel default plus the 1,000 m planning buffer caps a staged
    area at roughly 2.9 ground metres per pixel however small the area is. At
    that resolution a 25 m rock face across a couple of metres of ground is
    invisible: two site studies over El Corte de Madera and Purisima both
    found NO cliffs at 6-7 m per pixel and found real ones at 1 m. Anything
    asking a question about a small, steep feature needs this lever.

    REFUSE, DO NOT DEGRADE
    ----------------------
    A request the service cannot serve is an error naming the limit, not a
    coarser raster delivered quietly under the resolution that was asked for.
    Everything downstream is a measurement, and a measurement taken at a
    resolution the caller did not choose is worse than no measurement.
    """
    if resolution_m <= 0:
        raise ValueError(f"resolution_m must be positive; got {resolution_m}")
    if extent_m <= 0:
        raise ValueError(f"extent_m must be positive; got {extent_m}")

    pixels = math.ceil(extent_m / resolution_m)
    if pixels > MAX_STAGE_PIXELS:
        finest = extent_m / MAX_STAGE_PIXELS
        raise ValueError(
            f"{resolution_m:g} m per pixel over {extent_m:.0f} m needs "
            f"{pixels} pixels; 3DEP exportImage caps at {MAX_STAGE_PIXELS} per "
            f"axis, so the finest this extent can be staged at is "
            f"{finest:.2f} m. Ask for a coarser resolution or a smaller area.")
    return max(1, pixels)


def panel_dimensions(bbox_3857: Bbox, max_px: int = STAGE_PIXELS) -> tuple[int, int]:
    """Pixel dimensions matching the AOI's aspect, so pixels stay square.

    Square pixels keep panels undistorted and stop the 3DEP service widening
    the requested bbox to square it itself, which would leave the DEM
    covering a different extent than the panels.
    """
    width = bbox_3857[2] - bbox_3857[0]
    height = bbox_3857[3] - bbox_3857[1]
    if width <= 0 or height <= 0:
        raise ValueError(f"degenerate bbox: {bbox_3857}")

    if width >= height:
        return max_px, max(1, round(max_px * height / width))
    return max(1, round(max_px * width / height)), max_px


def write_manifest(
    out_dir: Path,
    map_id: str,
    assignment_label: str,
    bbox: Bbox,
    dem_bbox: Bbox,
    dem_resolution_m: float,
    latitude_deg: float,
    panels: dict[str, str],
    warnings: list[str],
    assignment_fingerprint: str | None = None,
    access: str = "public",
    subject_kind: str = "assignment",
    buffer_m: float | None = None,
    dem_size_px: tuple[int, int] | None = None,
) -> Path:
    """Record what was fetched. Credentials are deliberately absent.

    `bbox_3857` is the panel extent and `dem_bbox_3857` the DEM's own extent;
    they are recorded separately because the 3DEP service may return a wider
    area than requested. `dem_resolution_m` is true ground metres, converted
    from the raster's Web Mercator units using `latitude_deg`.

    `assignment_fingerprint` is what makes staleness detectable: it is the
    hash of the assignment geometry this directory was built from, so a
    boundary that moved afterwards can be noticed rather than silently
    computed against.

    `access` says which CalTopo path the assignment was read by - `public`
    or `authenticated` - so that every later revalidation of this
    directory uses the same one without the caller repeating it. A
    private map revalidated through the public endpoint fails quietly,
    and "could not check" forever is the outcome this key prevents. A
    later call that reuses the ground and reads the map by the other path
    records it beside the manifest (`handles.record_access`); writing a
    manifest removes that record, since the manifest then holds the latest
    path.

    `subject_kind` is `assignment` (the default, and every directory that
    predates the key) or `area` (a planner-chosen point or polygon staged
    by `planning.area.prepare_area`). An area's `assignment_fingerprint` is
    the hash of its own request: the input cannot move, so the fingerprint
    is the input - and for an area that hash covers the buffer too, so a
    directory staged with a different margin is a different handle.

    `buffer_m` is the margin this raster was actually staged with, recorded
    so a directory can say how much ground it holds. It is written only when
    given; a directory that predates the key does not claim a buffer it may
    not have been built with.

    `dem_size_px` is the raster size `fetch_dem` was asked for, recorded as
    `dem_width_px` and `dem_height_px`. With the extent, that size is what
    chose the pyramid level the elevation tiles came from
    (`tiles.level_for`), so it is what `tiles_in_use` needs to hold those
    tiles against eviction: ground staged at a chosen resolution can rest
    on another level than its extent at the default 1,024 px. Written only
    when given, like `buffer_m`. A caller that rewrites the manifest - the
    debrief app, adding its panels - should pass it on; a manifest without
    it has the size worked back from `dem_resolution_m` instead, which
    can leave a level held that nothing rests on.
    """
    manifest = {
        "map_id": map_id,
        "assignment": assignment_label,
        "bbox_3857": list(bbox),
        "dem_bbox_3857": list(dem_bbox),
        "dem_resolution_m": dem_resolution_m,
        "latitude_deg": latitude_deg,
        "panels": panels,
        "warnings": warnings,
        "access": access,
        "subject_kind": subject_kind,
    }
    if assignment_fingerprint is not None:
        manifest["assignment_fingerprint"] = assignment_fingerprint
    if buffer_m is not None:
        manifest["buffer_m"] = float(buffer_m)
    if dem_size_px is not None:
        manifest["dem_width_px"], manifest["dem_height_px"] = (
            int(dem_size_px[0]), int(dem_size_px[1]))
    path = Path(out_dir) / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # The manifest just written records the path its map is read by. A path
    # a later call over the ground it replaces recorded beside it
    # (`handles.record_access`) would outvote that; it is the old ground's.
    # Imported lazily, for the reason `tiles_in_use` gives.
    from .handles import ACCESS_FILENAME

    (Path(out_dir) / ACCESS_FILENAME).unlink(missing_ok=True)
    return path


def tiles_in_use(cache_dir: Path) -> set[str]:
    """Cache keys of the elevation tiles staged directories still rest on.

    Cache eviction must never take these (`cache.evict`), but the cache
    cannot work them out for itself: it is plumbing with no SAR knowledge
    (AGENTS.md). Where staged directories live is `handles`'
    business; what their manifests say and which tiles an extent maps onto
    is this module's. So the `evict_cache` component asks here and hands
    the keys over. Until 0.27.1 the cache did this itself, importing this
    module upwards.

    Derived from each staged manifest's own extent and raster size rather
    than from a stored tile list: a tile list would go stale the moment a
    level or a buffer changed, and it would go stale silently.

    AT THE LEVEL STAGING FETCHED
    ----------------------------
    `fetch_dem` takes its tiles from the level the extent AND the raster
    size choose. Ground staged at a chosen resolution - `prepare_area`'s
    `resolution_m`, which plan_medevac and plan_initial_response pass on
    every call for their landing-zone (2 m), egress and rings (10 m) ground
    - often rests on another level than its extent takes at the default
    1,024 px. Which ground does turns on its radius and latitude; the 2 m
    landing-zone ground at its default 1,000 m does at nearly every
    latitude. Until 0.31.9 this held the default level for every
    directory, so `evict_cache` could delete every tile under such ground
    while reporting none protected, and a later re-stage of it on a laptop
    with no signal then failed (PR #4 review B3). Which level, or levels,
    each directory is held at is `_staged_dem_levels`.
    """
    # Lazily: `handles` owns the staged layout, and it imports this module
    # lazily too, so neither has to load the other just to be imported.
    from .handles import STAGED_DIRNAME

    held: set[str] = set()
    staged_root = Path(cache_dir) / STAGED_DIRNAME
    if not staged_root.exists():
        return held

    for manifest_path in staged_root.glob("*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            bbox = tuple(manifest["bbox_3857"])
        except (OSError, ValueError, KeyError, TypeError):
            # An unreadable manifest protects nothing. That is the safe
            # direction only because the alternative - protecting
            # everything - would disable eviction entirely.
            continue
        try:
            for level in _staged_dem_levels(manifest, bbox):
                for tile in tiles_covering(bbox, level):
                    held.add(tile.key("dem", "tif"))
        except ValueError:
            continue
    return held


def _staged_dem_levels(manifest: dict, bbox) -> list[int]:
    """The pyramid level(s) a staged directory's elevation tiles were
    fetched at, from what its manifest can tell, best first:

    1. `dem_width_px` and `dem_height_px`, written by `stage_terrain` since
       0.31.9: the very size `fetch_dem` was asked for, so exactly the
       level it fetched.
    2. A manifest from before then, or one rewritten without them: the
       size worked back from `dem_resolution_m` and `latitude_deg`, which
       every manifest `write_manifest` wrote carries
       (`_sizes_from_resolution`). It can leave one level too many held -
       a few tiles eviction could have taken - never one too few, which is
       the failure this exists to prevent.
    3. Neither credible - a hand-written manifest, or one with no size
       staging could have asked for and a resolution or latitude no
       staging could have written: the default 1,024 px raster, the one
       every directory staged without a resolution has, and what every
       directory was held at until 0.31.9.
    """
    sizes = (_recorded_size(manifest) or _sizes_from_resolution(manifest, bbox)
             or [panel_dimensions(bbox)])
    levels = [level_for(bbox, width, height) for width, height in sizes]
    return list(range(min(levels), max(levels) + 1))


def _staging_could_ask_for(edge) -> bool:
    """A raster edge some staging could have asked for. Nothing here stages
    more than MAX_STAGE_PIXELS on an edge - `prepare_area` sizes a raster
    through `pixels_for_resolution`, which refuses more, and everything
    else stages at STAGE_PIXELS - so a larger number is a damaged manifest,
    and believing it could turn one eviction into millions of tile keys at
    the deepest level. JSON's `true` is an int to Python, so booleans are
    refused by name."""
    return (isinstance(edge, int) and not isinstance(edge, bool)
            and 1 <= edge <= MAX_STAGE_PIXELS)


def _finite_number(value) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        # A JSON integer past a float's range: no staging wrote it, and
        # letting the error out would stop `evict_cache` for the whole
        # cache over one damaged manifest.
        return False


def _recorded_size(manifest: dict) -> list[tuple[int, int]] | None:
    """The raster size the manifest records, or None when it records none
    that staging could have asked for."""
    size = (manifest.get("dem_width_px"), manifest.get("dem_height_px"))
    if all(_staging_could_ask_for(edge) for edge in size):
        return [size]
    return None


def _sizes_from_resolution(manifest: dict, bbox) -> list[tuple[int, int]] | None:
    """Raster sizes bracketing the one the elevation was staged at, worked
    back from `dem_resolution_m` and `latitude_deg`; None when the manifest
    cannot say.

    `dem_resolution_m` is the raster's pixel WIDTH in ground metres
    (`read_dem_grid` reads the transform's x step), and the raster spans
    `bbox_3857` exactly (`fetch_dem` writes it over the requested extent).
    So the width in pixels follows exactly. When the extent is at least as
    wide as tall, the width was the longest edge `panel_dimensions` was
    given and the height follows from it exactly too.

    When it is taller than wide, the height was the longest edge and the
    width was rounded from it, so a few heights round to the same width and
    the manifest cannot say which. Rounding the extent over the pixel
    width on each axis - the obvious inversion - lands on another level
    than staging fetched for about one tall extent in 1,500, measured over
    random extents and sizes. So both ends of the run are returned
    instead: every height that rounds to this width lies between them, a
    taller raster never takes a coarser level, and every level from one
    end's to the other's is held. At the sizes staging asks for, the two
    ends differ by a level for about one tall extent in 600.
    """
    resolution_m = manifest.get("dem_resolution_m")
    latitude_deg = manifest.get("latitude_deg")
    if not (_finite_number(resolution_m) and _finite_number(latitude_deg)):
        return None
    scale = ground_meters_per_mercator_unit(latitude_deg)
    if resolution_m <= 0 or scale <= 0:
        return None
    pixel = resolution_m / scale
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    if width <= 0 or height <= 0:
        raise ValueError(f"degenerate bbox: {bbox}")
    columns = width / pixel
    # A sound manifest lands within a hair of a whole number of pixels from
    # one to MAX_STAGE_PIXELS. Anything else is damage - a pixel wider than
    # the extent, or a count no staging asked for - and is checked before
    # rounding, because a damaged pixel size can make this infinite and
    # `round` raises on that rather than answering.
    if not (math.isfinite(columns) and 0.5 <= columns < MAX_STAGE_PIXELS + 0.5):
        return None
    columns = max(1, round(columns))

    if width >= height:
        return [panel_dimensions(bbox, max_px=columns)]
    # Every height that rounds to `columns` lies between these two: each is
    # taken up to a pixel beyond where rounding could put that end of the
    # run, which can add a level but never miss one. A one-pixel width
    # comes from any height of one pixel upwards, so its run starts at one.
    # The longer end is cut at what staging could have asked for.
    if columns == 1:
        shortest = 1
    else:
        shortest = math.floor((columns - 0.5) * height / width)
        if shortest > MAX_STAGE_PIXELS:
            return None
    longest = min(MAX_STAGE_PIXELS, math.ceil((columns + 0.5) * height / width))
    return [(columns, shortest), (columns, longest)]


def stage_terrain(subject, map_id: str, out_dir: Path, *,
                  cache_dir: Path,
                  buffer_m: float = DEFAULT_BUFFER_M,
                  fingerprint: str | None = None,
                  access: str = "public",
                  subject_kind: str = "assignment",
                  stage_pixels: int = STAGE_PIXELS) -> dict:
    """Write subject.json, dem.tif, evt.tif, osm.json and manifest.json.

    `subject` is anything with `geometry` (GeoJSON, WGS84), `label` and
    `model_dump_json()`: a CalTopo `Assignment`, or a planner's
    `AreaOfInterest`. `subject.json` is always written and is what
    `terrain.grid` reads; `assignment.json` is written as well for
    assignment subjects, because the debrief app reads it by that name.

    Everything here comes from a public service; no credentials are read.
    Returns `{"bbox": ..., "warnings": [...]}` - the caller (the debrief app)
    adds its basemap panels to the same directory and rewrites the manifest
    with them.

    Elevation is NOT wrapped in a try: it is the substrate every measurement
    rests on, so its absence is a hard failure rather than a degraded run.
    Vegetation and roads are optional and downgrade to a warning.
    """
    import httpx

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Read before anything is overwritten: if the OSM fetch fails, this is
    # how an `osm.json` already in the directory is judged (see below).
    previous_bbox = _staged_bbox(out_dir)
    discard_terrain_metrics(out_dir)

    (out_dir / "subject.json").write_text(
        subject.model_dump_json(indent=2), encoding="utf-8")
    if subject_kind == "assignment":
        (out_dir / "assignment.json").write_text(
            subject.model_dump_json(indent=2), encoding="utf-8")

    bounds_wgs84 = geometry_bounds_wgs84(subject.geometry)
    latitude = bbox_center_latitude(bounds_wgs84)
    bbox = compute_aoi(subject.geometry, buffer_m=buffer_m)
    # `stage_pixels` defaults to STAGE_PIXELS, so a caller that does not ask
    # for a resolution gets exactly the raster it always got.
    width_px, height_px = panel_dimensions(bbox, max_px=stage_pixels)
    warnings: list[str] = []

    dem_path = fetch_dem(bbox, width_px, height_px, cache_dir=cache_dir)
    (out_dir / "dem.tif").write_bytes(dem_path.read_bytes())
    dem_resolution_units, dem_bbox = read_dem_grid(dem_path)
    dem_resolution_m = dem_resolution_units * ground_meters_per_mercator_unit(latitude)

    try:
        write_evt_clip(bbox, out_dir / "evt.tif", cache_dir=cache_dir)
    except (RuntimeError, OSError, httpx.HTTPError) as error:
        warnings.append(f"evt unavailable: {type(error).__name__}")

    osm_failure = None
    try:
        from .terrain import linear

        routes = linear.routes_for(bbox, cache_dir=cache_dir)
        # The same cached tiles, read a second time for the ground a search
        # parks on. No extra Overpass traffic: `parking_for` re-reads the
        # files `routes_for` has just written.
        parking = linear.parking_for(bbox, cache_dir=cache_dir)
        # And a third time for gates (query version 5), the same way.
        gates = linear.gates_for(bbox, cache_dir=cache_dir)
        # The query version travels with the data: an Overpass query that
        # gained barriers, water edges, parking and then gates leaves every
        # directory staged before it quietly short of features, and a reader
        # that cannot see which query wrote the file cannot tell "no lots
        # here" from "lots were never asked for".
        (out_dir / "osm.json").write_text(
            json.dumps({"routes": routes, "parking": parking, "gates": gates,
                        "query_version": linear.OSM_QUERY_VERSION}),
            encoding="utf-8")
    except (RuntimeError, OSError, httpx.HTTPError, ValueError) as error:
        # The reason, not just the class name, when the hub wrote it: "osm
        # unavailable: HTTPStatusError" left an operator with no way to know
        # that every Overpass mirror had throttled them, and therefore no
        # way to know that the candidate list was missing every trail, road
        # and parking lot. But only the hub's own sentence - never another
        # exception's message, which is a cache path or a URL (0.17.1; see
        # `_osm_failure_reason`).
        #
        # Routes, parking and gates are written together or not at all: if
        # `parking_for` or `gates_for` raises after `routes_for` succeeded,
        # nothing is written and the whole fetch downgrades to one warning -
        # the same all-or-nothing shape a routes-only failure always had,
        # rather than a directory that lost its trails because a car park or
        # gate query failed.
        #
        # WHICH warning depends on the file already here. "osm.json exists"
        # is what `osm_available` and `manifest_warnings` trust, so it must
        # mean "fetched for THIS staging's ground". A file staged earlier
        # over the same box still is, and is kept, with a warning saying so
        # rather than claiming OSM is missing. A file for another box - a
        # boundary that moved - or one no manifest can place, is not: it is
        # removed, and the outage is recorded. Decided after this block, so
        # a removal that fails is not chained to this error (`_remove_osm`).
        osm_failure = _osm_failure_reason(error)

    if osm_failure is not None:
        osm_path = out_dir / "osm.json"
        kept_version = (_readable_osm_version(osm_path)
                        if _same_ground(previous_bbox, bbox) else None)
        if kept_version is not None:
            warnings.append(osm_kept_warning(osm_failure, kept_version))
        else:
            _remove_osm(osm_path, "osm.json staged for other ground")
            warnings.append(osm_outage_warning(osm_failure))

    write_manifest(
        out_dir, map_id=map_id, assignment_label=subject.label, bbox=bbox,
        dem_bbox=dem_bbox, dem_resolution_m=dem_resolution_m,
        latitude_deg=latitude, panels={}, warnings=warnings,
        assignment_fingerprint=fingerprint, access=access,
        subject_kind=subject_kind, buffer_m=buffer_m,
        # The size `fetch_dem` was asked for, so `tiles_in_use` holds the
        # tiles at the level it fetched them at, not the default size's.
        dem_size_px=(width_px, height_px),
    )
    return {"bbox": bbox, "dem_bbox": dem_bbox,
            "dem_resolution_m": dem_resolution_m, "latitude_deg": latitude,
            "width_px": width_px, "height_px": height_px,
            "warnings": warnings}


#: What an `osm.json` carries, for the warnings about one that is missing or
#: unreadable: all of it is unreadable then, not only the routes and parking
#: the file held before query version 5.
_OSM_CARRIES = "routes, parking, road class, access and gates"


def _what_an_old_osm_lacks(version: int) -> tuple[str, str, str]:
    """What an `osm.json` at query `version` is missing, worded by the
    version actually found rather than "older than current", as three
    phrases: (what it was staged before, what stays missing while it is not
    topped up, what topping it up gains).

    CUMULATIVE, parking first: every group the file predates is named, in
    the order the query gained them - parking (3), distribution power lines
    (4), road class, access and gates (5) - so a version-2 file names all
    three and a version-4 file only the last. A group the file already
    holds is said to be unaffected, never named as missing: "parking stays
    unreadable" on a version-3 file would send a planner looking for a
    problem that is not there. The caller may append "until it succeeds",
    so the missing groups come last and the waiting clause is theirs."""
    from .terrain import linear

    # (first version, noun, while missing, once gained, when held)
    groups = (
        (linear.OSM_QUERY_VERSION_PARKING, "parking",
         "parking stays unreadable", "parking can now be read", "parking"),
        (linear.OSM_QUERY_VERSION_MINOR_LINES,
         "distribution power lines (power=minor_line)",
         "distribution power lines (power=minor_line) stay unchecked",
         "distribution power lines (power=minor_line) are checked", "power lines"),
        (linear.OSM_QUERY_VERSION_ROUTE_TAGS, "road class, access and gates",
         "road class, access and gates stay unknown",
         "road class, access and gates can now be read", None),
    )
    lacking = [group for group in groups if version < group[0]]
    if not lacking:
        return ("before the current query",
                "what the current query adds stays missing from this ground",
                "what the current query adds can now be read here")
    held = [group[4] for group in groups if version >= group[0] and group[4]]

    def listed(parts: list[str], last: str) -> str:
        # "a", "a<last>b", "a, b<last>c". A part may carry its own "and"
        # (road class, access and gates), so the clauses join on ", and ".
        if len(parts) == 1:
            return parts[0]
        return ", ".join(parts[:-1]) + last + parts[-1]

    staged_before = f"before {listed([g[1] for g in lacking], ' or ')} were asked for"
    missing = listed([g[2] for g in lacking], ", and ")
    gained = listed([g[3] for g in lacking], ", and ")
    if held:
        verb = "is" if len(held) == 1 else "are"
        missing = f"{' and '.join(held)} {verb} unaffected; {missing}"
    return staged_before, f"{missing} on this ground", f"{gained} here"


def refresh_osm(out_dir: Path, cache_dir: Path) -> list[str]:
    """Top up `osm.json` in an ALREADY-STAGED directory when it is behind
    `terrain.linear.OSM_QUERY_VERSION`, or missing outright.

    WHY THIS LIVES HERE, NOT IN `planning.area`
    --------------------------------------------
    `stage_terrain` above is the one place that already knows the shape of
    `osm.json` and how to build it (`routes_for`, `parking_for` and
    `gates_for`, tagged with the version that wrote them). Refreshing it is
    the same write with a narrower scope - skip the DEM and EVT, which no
    Overpass query change touches - so it belongs beside that code, not
    duplicated in `planning.area`, which only needs to know THAT a directory
    may need topping up, not HOW `osm.json` is built.

    WHY THIS MUST STAY CHEAP ON THE NORMAL PATH
    --------------------------------------------
    `prepare_area` calls this on every reuse of every staged directory, so
    the common case - a directory already at the current version - has to
    cost one file read, not one Overpass round trip. The version check
    below runs before anything that could touch the network.

    THE BBOX COMES FROM THE DIRECTORY, NOT THE CALLER
    --------------------------------------------------
    `manifest.json` already records the exact box this directory was
    staged over (`bbox_3857`); recomputing it from the caller's request
    would risk landing OSM data over a slightly different box than the DEM
    and EVT already sitting beside it. If the manifest cannot say what
    ground was staged, refreshing would have to guess, so it does not: the
    file is left exactly as it was and a warning says why.

    A DIRECTORY WITH NO `osm.json` AT ALL
    --------------------------------------
    An old staging where the very first Overpass fetch failed left no file
    at all - `stage_terrain` writes routes, parking and gates together or
    not at all. That is treated the same as "behind the current version": there
    is nothing to lose by trying again, and a planner should not have to
    know the difference between "stale" and "never arrived" to get it
    fixed. Failing again leaves the directory exactly as it was - still
    missing the file - and warns. An `osm.json` that cannot be parsed is
    treated the same way (0.17.1): fetched again, and removed if that
    fails, so a file nothing can read never passes for one that exists.

    NEVER A HALF-WRITTEN FILE
    --------------------------
    `routes_for`, `parking_for` and `gates_for` are all called, and all
    must succeed, before anything is written; the write itself goes through
    `atomic_write_text` (temp file, then rename). An Overpass failure at
    any point therefore leaves the existing `osm.json` byte-for-byte as it
    was.

    CANDIDATE SETS MEASURED FROM THE OLD FILE
    -------------------------------------------
    A successful refresh also rewrites `manifest.json`, appending to an
    `osm_refreshed_versions` list, because `candidates.set_staleness` and
    `boundary.ground_for` both key their staleness off that file's stamp
    (`boundary._manifest_stamp`: its mtime AND size - a cheap proxy, not a
    content hash). `stage_terrain` already relies on the same mechanism -
    it writes the manifest LAST, so a newer manifest is the signal that
    everything beside it is new - but a stamp that changed only by mtime is
    one that could fail to change at all: two writes close enough together
    can land on a filesystem whose clock has not ticked between them, and a
    proxy that ignores content would not notice. Appending an element to a
    list makes the manifest strictly LONGER every single time this runs,
    which is a guarantee about the file's size, not a hope about its clock
    - so the stamp changing is a fact, not a race. (A single scalar field
    holding just the version number was tried and rejected here: `3` and
    `4` are the same number of bytes, so a same-length bump could still
    land on an unlucky, un-ticked mtime.) A candidate set saved before this
    call must not go on looking current once the routes underneath it have
    changed; this is how it starts reporting stale.

    The same rewrite retires the manifest's OSM outage warning
    (`is_osm_outage_warning`) and any kept-file warning
    (`is_osm_kept_warning`): the file they spoke of is now whole and
    current, and `prepare_area` returns the manifest's warnings on every
    reuse. They are MOVED to `osm_retired_warnings`, a list that only
    grows, as objects longer than the strings they replace - never deleted,
    which could leave the file its old size and break the promise above.
    `manifest_warnings` never returns that list. Every other warning is
    kept. A failed refresh rewrites nothing -
    the stamp must not move - so where an older file exists the outage is
    dropped at read time instead (`manifest_warnings`), and where no file
    exists the outage is still true and still said.

    Returns the warnings to fold into the caller's response - empty when
    the file was already current, so a caller can tell "topped up" from
    "nothing to do" without inspecting anything else.
    """
    import httpx

    from .terrain import linear

    out_dir = Path(out_dir)
    osm_path = out_dir / "osm.json"
    manifest_path = out_dir / "manifest.json"

    # The first query wrote no version: a file without one is version 1, as
    # the landing-zone and boundary tools call it. 0 when there is no file -
    # or when there is one nothing can parse (a write cut short, a disk
    # error): that is treated as missing and fetched again, rather than
    # failing every `prepare_area` on this ground with a JSONDecodeError.
    version = 0
    unreadable = False
    if osm_path.is_file():
        try:
            payload = json.loads(osm_path.read_text(encoding="utf-8"))
            version = int(payload.get("query_version") or 1)
        except (ValueError, TypeError, AttributeError):
            unreadable = True
    had_file = version > 0

    if had_file and version >= linear.OSM_QUERY_VERSION:
        return []
    staged_before, still_missing, gained = _what_an_old_osm_lacks(version)
    state = ("could not be read (it is not valid JSON)" if unreadable
             else f"is behind (query version {version})")

    # With nothing to refresh against, a readable file is left as staged;
    # an unreadable one is removed, so `osm_available` never says yes over
    # a file nothing can read.
    if unreadable:
        left = f"so it was removed; {_OSM_CARRIES} stay unreadable from this ground"
    else:
        left = f"so it was left as staged; {still_missing}"
    if not manifest_path.is_file():
        if unreadable:
            _remove_osm(osm_path, "an unreadable osm.json")
        return [f"osm.json {state} but this directory has "
                f"no manifest.json to read a bbox from, {left}"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bbox = manifest.get("bbox_3857")
    if bbox is None:
        if unreadable:
            _remove_osm(osm_path, "an unreadable osm.json")
        return [f"osm.json {state} but manifest.json has "
                f"no bbox_3857 to refresh it against, {left}"]
    bbox = tuple(bbox)
    tiles = len(linear.tiles_covering_osm(bbox))
    fetches = f"{tiles} Overpass tile{'s' if tiles != 1 else ''}"

    reason = None
    try:
        routes = linear.routes_for(bbox, cache_dir=cache_dir)
        parking = linear.parking_for(bbox, cache_dir=cache_dir)
        gates = linear.gates_for(bbox, cache_dir=cache_dir)
    except (RuntimeError, OSError, httpx.HTTPError, ValueError) as error:
        # Why, never where: see `_osm_failure_reason`. Handled after this
        # block, so a removal that fails is not chained to this error.
        reason = _osm_failure_reason(error)
    if reason is not None:
        if unreadable:
            # Not left in place: a file nothing can read would still make
            # `osm_available` say yes and `manifest_warnings` hide an outage.
            _remove_osm(osm_path, "an unreadable osm.json")
            return ["osm.json could not be read (it is not valid JSON) and fetching "
                    f"it again failed ({reason}), so it was removed; {_OSM_CARRIES} "
                    "stay unreadable from this ground until it succeeds"]
        if had_file:
            return [f"osm.json is still at query version {version}: refreshing it to "
                    f"{linear.OSM_QUERY_VERSION} failed ({reason}); "
                    f"{still_missing} until it succeeds"]
        return [f"osm.json is still missing: fetching it failed again "
                f"({reason}); {_OSM_CARRIES} stay "
                "unreadable from this ground until it succeeds"]

    atomic_write_text(osm_path, json.dumps(
        {"routes": routes, "parking": parking, "gates": gates,
         "query_version": linear.OSM_QUERY_VERSION}))
    discard_terrain_metrics(out_dir)
    # See CANDIDATE SETS above: appending, not overwriting, so the file is
    # guaranteed strictly longer and the staleness stamp is guaranteed to
    # change - not merely likely to, on a clock that happened to tick.
    manifest.setdefault("osm_refreshed_versions", []).append(linear.OSM_QUERY_VERSION)
    # The file an outage or kept-file warning spoke of has just been
    # replaced by a whole, current one, so those warnings are retired
    # rather than repeated to every later caller. MOVED to a list that only
    # grows, never deleted: deleting a warning while appending one version
    # number can leave the file exactly as long as it was (review found it
    # with 15- and 18-character reasons), and the stamp above is promised
    # to change. Each retired entry is an object that is longer than the
    # string it replaces, so the move alone always adds bytes. Anything
    # else in `warnings` - the debrief app's included - is kept as written.
    retired = [w for w in manifest.get("warnings", []) if _retires_on_top_up(w)]
    if retired:
        manifest["warnings"] = [w for w in manifest["warnings"]
                                if not _retires_on_top_up(w)]
        manifest.setdefault("osm_retired_warnings", []).extend(
            {"warning": w, "retired_by_query_version": linear.OSM_QUERY_VERSION}
            for w in retired)
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2))

    if had_file:
        return [f"osm.json was staged at query version {version}, {staged_before}; "
                f"topped it up to {linear.OSM_QUERY_VERSION} over {fetches} so "
                f"{gained}"]
    if unreadable:
        return [f"osm.json could not be read (it is not valid JSON); fetched it again "
                f"over {fetches} so {_OSM_CARRIES} can be read here"]
    return [f"osm.json was missing from this ground (an earlier Overpass fetch had "
            f"failed); fetched it now over {fetches} so {_OSM_CARRIES} can be "
            "read here"]
