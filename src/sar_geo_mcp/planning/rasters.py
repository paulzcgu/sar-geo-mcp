"""Orchestration for the two raster tools: staged ground in, envelopes out.

WHAT LIVES HERE AND WHAT DOES NOT
---------------------------------
Everything with I/O. The math is in `terrain/hillshade.py`,
`terrain/lineaments.py` and `caltopo/shading.py`, each tested without a
directory, a network or a token.

THE ROUND TRIP
--------------
A feature seen in a picture is worthless unless it can be turned into
something a team navigates to. Every raster written here is north-up
EPSG:3857 at one pixel size, and the georeference block carries both
bounding boxes, the pixel dimensions, true ground metres per pixel and all
four corners - so pixel to coordinate is arithmetic a caller can check
rather than trust.

WHICH CRS LEAVES THIS MODULE
----------------------------
Measurement happens in EPSG:3857, because that is the projection the staged
raster, the staged OSM ways and CalTopo's WMS all share. NOTHING LEAVES IN
IT. Every `geometry` in either envelope is reprojected to WGS84 here, at the
emission boundary, with `geo.to_wgs84_geojson` - the same helper
`corridors.py` and `parking.py` emit through.

That reprojection sits in THIS module and not in `terrain/lineaments.py`,
deliberately. `find_lineaments` returns pixel coordinates and knows nothing
about projections; `subtract_known_ways` projects to 3857 only because that
is the frame the staged OSM ways it subtracts are already in - a measuring
frame, not an output format. Orchestration holds the output contract, the
way it already holds `georeference`'s `bbox_wgs84` and `corners_wgs84`, so
the terrain module stays pure math and gains no opinion about what a caller
reads.

`transform * (x, y)` raises a PendingDeprecationWarning in this version of
affine (favouring `transform @ (x, y)`); `terrain/hydro.py` and
`terrain/lineaments.py` already use the `*` form throughout, and this
module matches them rather than being the one place that quietly
diverges. A repo-wide switch to `@` is a separate, single-purpose task.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pyproj import Transformer

from ..geo import ground_meters_per_mercator_unit, to_wgs84_geojson
from ..net_errors import safe_reason

_TO_WGS84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

#: Colour a known OSM way is painted in. Chosen to sit outside both the grey
#: hillshade ramp and the slope legend, so an overlay is never mistaken for
#: the data under it.
WAY_RGB = (0, 200, 255)
WAY_WIDTH_PX = 2

#: Grid spacing in ground metres, and the colour it is drawn in.
GRID_SPACING_M = 500.0
GRID_RGB = (255, 255, 255)


def pixel_to_wgs84(transform, column: float, row: float) -> tuple[float, float]:
    """(column, row) on the raster to (longitude, latitude).

    Column and row are pixel EDGES, not centres: (0, 0) is the north-west
    corner of the north-west pixel, which is what the bbox reports.
    """
    x, y = transform * (column, row)
    return _TO_WGS84.transform(x, y)


def georeference(transform, shape: tuple[int, int], latitude_deg: float) -> dict:
    """Everything a caller needs to locate a pixel, stated rather than implied."""
    rows, cols = shape
    west, north = pixel_to_wgs84(transform, 0, 0)
    east, south = pixel_to_wgs84(transform, cols, rows)

    return {
        "crs": "EPSG:3857",
        "north_up": True,
        "width_px": cols,
        "height_px": rows,
        "ground_m_per_px": round(
            abs(transform.a) * ground_meters_per_mercator_unit(latitude_deg), 3),
        "bbox_wgs84": [west, south, east, north],
        "bbox_3857": [transform.c, transform.f - rows * abs(transform.e),
                      transform.c + cols * abs(transform.a), transform.f],
        "corners_wgs84": {
            "north_west": list(pixel_to_wgs84(transform, 0, 0)),
            "north_east": list(pixel_to_wgs84(transform, cols, 0)),
            "south_east": list(pixel_to_wgs84(transform, cols, rows)),
            "south_west": list(pixel_to_wgs84(transform, 0, rows)),
        },
    }


def render_grey(shade: np.ndarray, destination: Path) -> Path:
    """A 0..1 illumination array as an 8-bit greyscale PNG."""
    from PIL import Image

    from ..terrain.hillshade import to_grey_bytes

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_grey_bytes(shade), mode="L").save(destination, format="PNG")
    return destination


def _to_pixels(geometry: dict, transform):
    """A 3857 geometry's coordinates as (column, row) tuples."""
    inverse = ~transform
    lines = []
    kind = geometry.get("type")
    if kind == "LineString":
        parts = [geometry["coordinates"]]
    elif kind == "MultiLineString":
        parts = geometry["coordinates"]
    else:
        return lines
    for part in parts:
        lines.append([inverse * (x, y) for x, y in part])
    return lines


def draw_ways(image, routes: list[dict], transform, *, grid: bool):
    """Paint staged OSM ways, and optionally a labelled coordinate grid.

    The grid goes on the OVERLAID image only. A graticule sits on top of
    exactly the faint grades the clean image exists to reveal, so the pair is
    clean for seeing and annotated for locating.

    Never mutates `image`: `Image.convert("RGB")` returns a fresh copy even
    when the source is already RGB (Pillow's own `convert()` falls through
    to `self.copy()` whenever the target mode matches), and this is checked
    directly by `test_draw_ways_does_not_mutate_its_input` rather than
    trusted - Task 6 draws two overlays from one base image, and an
    aliasing regression here would silently paint the "clean" one too.
    """
    from PIL import ImageDraw

    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)

    for route in routes:
        for line in _to_pixels(route.get("geometry", {}), transform):
            if len(line) >= 2:
                draw.line(line, fill=WAY_RGB, width=WAY_WIDTH_PX)

    if grid:
        spacing_px = max(16, int(GRID_SPACING_M / abs(transform.a)))
        width, height = canvas.size
        for x in range(0, width, spacing_px):
            draw.line([(x, 0), (x, height)], fill=GRID_RGB, width=1)
            lon, _ = pixel_to_wgs84(transform, x, 0)
            draw.text((x + 2, 2), f"{lon:.4f}", fill=GRID_RGB)
        for y in range(0, height, spacing_px):
            draw.line([(0, y), (width, y)], fill=GRID_RGB, width=1)
            _, lat = pixel_to_wgs84(transform, 0, y)
            draw.text((2, y + 2), f"{lat:.4f}", fill=GRID_RGB)

    return canvas


RELIEF_LAYERS = ("r", "r2", "r3")

NO_TOKEN_WARNING = (
    "no CalTopo WMS token configured, so the rendered relief layers are "
    "absent; the local hillshade and the overlay do not need one and are "
    "unaffected"
)

#: Said when `returns=\"image\"` means the lineament search never ran.
#:
#: `lineaments: []` was the old answer, and an empty list of unmapped tracks
#: reads as "this ground has no unmapped tracks" - the single most dangerous
#: thing this tool could say, because that is precisely the claim it exists
#: to refuse to make. The value is null and this says why. It also corrects
#: the no-token warning, which used to state by name that "the lineaments...
#: are unaffected" - true in `data` and `both`, false here, and the two
#: warnings appeared together.
NOT_REQUESTED_LINEAMENTS_WARNING = (
    "returns=\"image\", so no lineament search was run: `lineaments` is null, "
    "meaning NOT LOOKED FOR, not none found. Ask for returns=\"data\" or "
    "\"both\" to search this ground"
)

#: How many lineaments a caller who names no limit gets, longest first.
#: `limit=0` is every one - `corridors.DEFAULT_LIMIT`'s convention exactly.
#:
#: An unbounded list was the defect. Measured on correlated-roughness
#: hillsides at 1 m pixels (the module's own documented worst realistic case,
#: 51 deg plane, 5 px correlation length, 3.0 m amplitude): 1000^2 -> 53
#: lineaments, 2000^2 -> 245, 4000^2 -> 971. The staging ceiling is 8000 px,
#: so a full area can return several thousand LineStrings - sorted by length,
#: so the longest NOISE sits at the top of the list. `lineaments.py` says in
#: its own docstring that the orchestration layer "should still not hand this
#: function a whole tile without bounding the component count first", and
#: this tool did exactly that, with no count, no density and no warning.
DEFAULT_LINEAMENT_LIMIT = 50

#: The figure that turns a count into a judgement. Those same measurements
#: come to 53, 61 and 61 per km2 across a 16x span of area - undisturbed
#: ground with nothing cut into it reporting a near-constant density, which
#: matches `lineaments.py`'s own measured false-positive rate of about 60 per
#: km2. So a caller reading "971 unmapped tracks" and a caller reading "61
#: per km2, and undisturbed rough ground reads about 60" learn opposite
#: things from the same run, and only the second one is true.
NOISE_FLOOR_PER_KM2 = 60.0

#: How many lineaments must be in hand before the density above is quoted
#: back as a warning. `lineaments.py` measured its false-positive rate as 10
#: lineaments per 0.16 km2; below about that many, a density is one or two
#: features divided by a small patch and says nothing about a noise floor -
#: a single real bench on a 95 m square reads 110 per km2. The FIGURE is
#: always reported; only the accusation waits for a population.
MIN_DENSITY_SAMPLE = 10


def _load_routes(directory: Path) -> tuple[list[dict], list[str]]:
    """Staged OSM ways, whose geometry is already EPSG:3857."""
    import json

    path = Path(directory) / "osm.json"
    if not path.exists():
        return [], ["osm.json missing; nothing is known to be already mapped, "
                    "so every lineament is offered unfiltered"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("routes", [])), []


def shaded_relief(directory: Path, *, wms_base_url: str | None,
                  cache_dir: Path | None, layer: str = "r3",
                  returns: str = "both", track_width_m: float = 4.0,
                  known_tolerance_m: float = 15.0,
                  limit: int | None = None) -> dict:
    """Paired rasters for the eye, and DEM-derived lineaments for the numbers.

    Needs no credential: only the two CalTopo images do, and their absence is
    a warning rather than a failure.

    `limit` keeps the first lineaments after the ordering (longest first) and
    says in `lineament_summary` how many there were; None is
    `DEFAULT_LINEAMENT_LIMIT` and 0 is every one. `hasty_candidates` and
    `parking_candidates` both take `limit` on the same terms, and
    `corridors.py` records `total_before_limit` and warns when it truncates -
    this matches them rather than inventing a third convention.

    NULL IS NOT ZERO, in both directions here. Under `returns="image"` no
    search is run, so `lineaments` is None with a warning saying so: an empty
    list would read as "this ground has no unmapped tracks", which is the one
    claim this tool must never make.
    """
    from PIL import Image
    from shapely.geometry import shape as shapely_shape

    from ..terrain.grid import load_assignment_grid
    from ..terrain.hillshade import (
        DEFAULT_AZIMUTHS,
        DEFAULT_SUN_ALTITUDE_DEG,
        HILLSHADE_ASSUMPTIONS,
        multi_azimuth_hillshade,
    )
    from ..terrain.lineaments import (
        LINEAMENT_ASSUMPTIONS,
        find_lineaments,
        subtract_known_ways,
    )

    if layer not in RELIEF_LAYERS:
        raise ValueError(f"layer must be one of {RELIEF_LAYERS}, got {layer!r}")
    if returns not in ("data", "image", "both"):
        raise ValueError(f'returns must be "data", "image" or "both", got {returns!r}')

    directory = Path(directory)
    grid = load_assignment_grid(directory)
    warnings: list[str] = []

    routes, route_warnings = _load_routes(directory)
    warnings.extend(route_warnings)

    # Ground metres per pixel is TRUE ground, not Mercator units, so the
    # density below is per real square kilometre - the same correction
    # `georeference.ground_m_per_px` reports.
    rows, cols = grid.elevation.shape
    ground_km2 = (rows * cols * grid.pixel_size_m**2) / 1e6

    lineaments: list[dict] | None = None
    summary = {"count": None, "total_before_limit": None, "truncated_to": None,
               "per_km2": None, "ground_km2": round(ground_km2, 4),
               "noise_floor_per_km2": NOISE_FLOOR_PER_KM2}
    if returns in ("data", "both"):
        limit = _checked_lineament_limit(limit)
        found = find_lineaments(grid.elevation, grid.pixel_size_m,
                                track_width_m=track_width_m)
        lineaments, subtract_warnings = subtract_known_ways(
            found, routes, tolerance_m=known_tolerance_m,
            transform=grid.transform, latitude_deg=grid.latitude_deg)
        warnings.extend(subtract_warnings)

        total_before_limit = len(lineaments)
        per_km2 = total_before_limit / ground_km2 if ground_km2 else None
        summary.update({"total_before_limit": total_before_limit,
                        "truncated_to": limit,
                        "per_km2": round(per_km2, 1) if per_km2 is not None else None})
        if limit is not None and total_before_limit > limit:
            lineaments = lineaments[:limit]
            warnings.append(
                f"only the longest {limit} of {total_before_limit} lineaments are "
                "listed (limit); the rest are shorter. limit=0 returns every one")
        summary["count"] = len(lineaments)
        if (per_km2 is not None and per_km2 >= NOISE_FLOOR_PER_KM2
                and total_before_limit >= MIN_DENSITY_SAMPLE):
            warnings.append(
                f"{total_before_limit} lineaments over {ground_km2:.2f} km2 is "
                f"{per_km2:.0f} per km2, at or above the ~{NOISE_FLOOR_PER_KM2:.0f} "
                "per km2 this method reports on UNDISTURBED ground with correlated "
                "roughness (see assumptions.false_positives). Read this list as a "
                "noise floor, not as a set of tracks")
        for index, lineament in enumerate(lineaments):
            lineament["id"] = f"lineament-{index}"
            # `subtract_known_ways` measures against the staged OSM ways, which
            # are EPSG:3857, and hands back its residue in that same frame. The
            # envelope is GeoJSON, so it goes out in WGS84 - see this module's
            # docstring for why the reprojection lives here rather than there.
            lineament["geometry"] = to_wgs84_geojson(
                shapely_shape(lineament["geometry"]))
    else:
        warnings.append(NOT_REQUESTED_LINEAMENTS_WARNING)

    images = {"relief": None, "relief_osm": None,
              "hillshade": None, "hillshade_osm": None}

    if returns in ("image", "both"):
        out = directory / "rasters"
        shade = multi_azimuth_hillshade(grid.elevation, grid.pixel_size_m)
        images["hillshade"] = str(render_grey(shade, out / "hillshade.png"))

        with Image.open(images["hillshade"]) as base:
            overlaid = draw_ways(base, routes, grid.transform, grid=True)
        overlaid_path = out / "hillshade_osm.png"
        overlaid.save(overlaid_path, format="PNG")
        images["hillshade_osm"] = str(overlaid_path)

        if wms_base_url and cache_dir:
            from ..caltopo.wms import fetch_panel

            block = georeference(grid.transform, grid.elevation.shape,
                                 grid.latitude_deg)
            try:
                panel = fetch_panel(wms_base_url, [layer], tuple(block["bbox_3857"]),
                                    block["width_px"], block["height_px"],
                                    Path(cache_dir))
            except Exception as error:  # noqa: BLE001 - a missing image, not a refusal
                # Never the message: `fetch_panel`'s error and httpx's
                # `HTTPStatusError` both carry the full GetMap URL, whose
                # path holds the account token (see `net_errors.safe_reason`).
                warnings.append(
                    f"CalTopo {layer} panel could not be fetched ({safe_reason(error)})")
            else:
                images["relief"] = str(panel)
                with Image.open(panel) as base:
                    relief_overlaid = draw_ways(base, routes, grid.transform, grid=True)
                relief_path = out / f"relief_{layer}_osm.png"
                relief_overlaid.save(relief_path, format="PNG")
                images["relief_osm"] = str(relief_path)
        else:
            warnings.append(NO_TOKEN_WARNING)

    return {
        "layer": layer,
        "georeference": georeference(grid.transform, grid.elevation.shape,
                                     grid.latitude_deg),
        "images": images,
        "azimuths_deg": list(DEFAULT_AZIMUTHS),
        "sun_altitude_deg": DEFAULT_SUN_ALTITUDE_DEG,
        "lineaments": lineaments,
        "lineament_summary": summary,
        "assumptions": {**HILLSHADE_ASSUMPTIONS, **LINEAMENT_ASSUMPTIONS,
                        "lineament_density": (
                            "lineament_summary.per_km2 is total_before_limit over "
                            "the WHOLE staged raster in true ground km2, not over "
                            "the subject area; compare it against "
                            "noise_floor_per_km2"),
                        "lineaments_null_means": (
                            "null is NOT LOOKED FOR (returns=\"image\"); an empty "
                            "list is looked for and none found")},
        "warnings": warnings,
    }


def _checked_lineament_limit(limit: int | None) -> int | None:
    """`DEFAULT_LINEAMENT_LIMIT` for None, None (unlimited) for 0, else the number.

    `corridors._checked_limit`'s rule, in its wording: an unbounded default is
    the failure this exists to prevent, and a caller who wants all of them
    says so with 0.
    """
    if limit is None:
        return DEFAULT_LINEAMENT_LIMIT
    limit = int(limit)
    if limit < 0:
        raise ValueError("limit must be a positive count, 0 for every lineament, "
                         f"or None for the default of {DEFAULT_LINEAMENT_LIMIT}; "
                         f"got {limit}")
    return limit or None


#: Connected pixels below which a disagreement patch is noise. Deliberately a
#: NOISE FLOOR and not an area filter: a vertical face is narrow in plan, and
#: an area floor is what made cliff detection return zero cliffs where six
#: existed. Ranking is by band delta then vertical extent, never by area.
MIN_PATCH_PIXELS = 4

#: Above this, something systematic is wrong rather than the render odd.
#:
#: The warning used to name the legend as the sole suspect, and that was not
#: honest: an interpolating resample in the fetch raises this figure on a
#: perfectly correct legend, because a blended pixel is exactly the colour
#: the legend refuses. Measured on a synthetic `sf` render with a correct
#: legend at the normal 1.675x downsample, LANCZOS gave unmatched fractions
#: of 0.033 to 0.192 depending on how much band boundary the ground holds -
#: straddling this threshold - against 0.000 at nearest-neighbour.
#: `caltopo/wms.py:CATEGORICAL_LAYERS` now forces nearest-neighbour for
#: `sf`, so that cause is closed; the warning still names it, because a
#: caller who overrode the filter, or a future layer added without being
#: declared categorical, would see it again and deserves the real suspect
#: list rather than a confident accusation of the legend.
UNMATCHED_WARNING_FRACTION = 0.10

NO_TOKEN_SLOPE_WARNING = (
    "no CalTopo WMS token configured; slope_shading has no fallback and "
    "every CalTopo figure is null, not zero"
)

#: Colour the disagreement patches are outlined in. Distinct from every
#: legend colour so an outline is never read as a band.
PATCH_OUTLINE_RGB = (0, 255, 128)

#: What the reported `dem_band` says when the patch's DEM slope reads below
#: the legend's lowest edge entirely - the case that matters most (CalTopo
#: sees a band, the DEM sees nothing steep at all). `int(max(0, index))`
#: would silently misreport this as the legend's shallowest band; that is
#: worse than the truth, which is that the DEM has no band to offer here.
BELOW_LEGEND_LABEL = "below_legend"


def _dem_band_label(index: int, labels: list[str]):
    """A DEM band index to the same label CalTopo bands are named with.

    `BELOW_LEGEND` becomes an explicit label rather than being clamped into
    band 0 - clamping would claim the DEM read the shallowest legend band
    when it actually read shallower than the legend covers at all, which is
    exactly the disagreement this tool exists to surface honestly.
    """
    from ..caltopo.shading import BELOW_LEGEND, UNKNOWN

    if index == BELOW_LEGEND:
        return BELOW_LEGEND_LABEL
    if index == UNKNOWN:
        return None
    return labels[index]


def _iter_polygon_rings(geometry: dict):
    """Rings of a GeoJSON Polygon or MultiPolygon, uniformly.

    `rasterio.features.shapes` polygonizes with 4-connectivity by default;
    an 8-connected component that touches itself only at a corner (a
    diagonal chain of steep pixels) comes back as MORE THAN ONE polygon for
    that one component. Merging them with `shapely.ops.unary_union` before
    this function sees them can yield a MultiPolygon, so both geometry
    types are walked here rather than assuming Polygon and silently
    dropping the rest of the ground.
    """
    kind = geometry.get("type")
    if kind == "Polygon":
        yield from geometry["coordinates"]
    elif kind == "MultiPolygon":
        for polygon in geometry["coordinates"]:
            yield from polygon


def _outline_patches(panel_path: Path, outlines_3857: list[dict], transform,
                     destination: Path, *, colour: tuple[int, int, int] = PATCH_OUTLINE_RGB
                     ) -> Path:
    """A panel with polygon outlines drawn on it in `colour`, saved atomically.

    For `slope_shading`: so the disagreement can be LOOKED at as well as
    counted: a scatter of small patches along one gully reads differently
    from one solid face, and no fraction in the envelope carries that.

    Takes the patches' EPSG:3857 geometries, NOT the WGS84 ones the envelope
    carries: the raster's affine transform maps Mercator units to pixels, so
    inverting it on degrees would place every outline at the top-left corner
    of the image. The patch record that goes to the caller is reprojected
    (see this module's docstring); the drawing keeps the measuring frame.

    Saved through `atomic_output`, the way `fetch_panel` saves its own
    Pillow image: a crash mid-save must not leave a torn PNG at
    `destination` for a later reader to open.
    """
    from PIL import Image, ImageDraw

    from ..cache import atomic_output

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    inverse = ~transform

    with Image.open(panel_path) as source:
        canvas = source.convert("RGB")
    draw = ImageDraw.Draw(canvas)

    for geometry in outlines_3857:
        if not geometry:
            continue
        for ring in _iter_polygon_rings(geometry):
            pixels = [inverse * (x, y) for x, y in ring]
            if len(pixels) >= 2:
                draw.line(pixels, fill=colour, width=2)

    with atomic_output(destination) as temporary:
        canvas.save(temporary, format="PNG")
    return destination


def _dem_summary(slope_deg: np.ndarray, footprint: np.ndarray,
                 warnings: list[str]) -> dict | None:
    """Mean/p90/max DEM slope over the footprint, or None with a warning.

    `np.nanmax`/`np.nanpercentile` on a zero-size selection raise
    `ValueError` rather than returning NaN (unlike `np.nanmean`, which only
    warns) - an empty footprint (a supplied `geometry` that selects no
    pixels) or an all-nodata DEM must not crash this tool. A missing
    measurement here is null, never zero and never a traceback.
    """
    values = slope_deg[footprint]
    values = values[np.isfinite(values)]
    if values.size == 0:
        warnings.append(
            "no DEM pixel with a finite slope inside the footprint; dem "
            "summary statistics are null, not zero")
        return None
    return {
        "mean_deg": float(values.mean()),
        "p90_deg": float(np.percentile(values, 90)),
        "max_deg": float(values.max()),
    }


def slope_shading(directory: Path, *, wms_base_url: str | None,
                  cache_dir: Path | None, geometry: dict | None = None,
                  layer: str = "sf", returns: str = "data",
                  fetch=None) -> dict:
    """CalTopo's slope bands beside this package's own, and where they differ.

    `fetch` is the panel fetcher, injected so the tests need no network. It
    defaults to `caltopo.wms.fetch_panel`.
    """
    # Refused here as `shaded_relief` refuses it, not only by the MCP
    # schema: a scenario calls the feature function directly, and an
    # unknown value would otherwise read as "data", with `images` null.
    if returns not in ("data", "image", "both"):
        raise ValueError(f'returns must be "data", "image" or "both", got {returns!r}')

    import numpy as np
    from PIL import Image
    from rasterio.features import shapes as raster_shapes
    from scipy import ndimage
    from shapely.geometry import mapping as shapely_mapping, shape as shapely_shape
    from shapely.ops import unary_union

    from ..caltopo.shading import (
        BELOW_LEGEND,
        LEGEND_PROVENANCE,
        UNKNOWN,
        band_edges_deg,
        band_labels,
        bin_degrees_to_bands,
        decode_slope_bands,
    )
    from ..terrain.grid import load_assignment_grid, project_geometry_to_3857
    from ..terrain.slope import compute_slope_aspect

    directory = Path(directory)
    grid = load_assignment_grid(directory)
    block = georeference(grid.transform, grid.elevation.shape, grid.latitude_deg)
    warnings: list[str] = []

    legend = {"edges_deg": band_edges_deg(), "labels": band_labels(),
              "provenance": LEGEND_PROVENANCE}
    empty = {
        "layer": layer,
        "raster": block,
        "caltopo": {"bands": None, "unmatched_fraction": None,
                    "below_legend_fraction": None},
        "dem": None,
        "agreement_fraction": None,
        # None, not []: nothing was compared, so nothing can be said about
        # what disagrees. An empty list is "compared, and the two sources
        # agree everywhere" - which on a cliff band is the opposite answer.
        "steeper_in_caltopo": None,
        "legend": legend,
        "assumptions": {"comparison": "band for band on a pixel-aligned grid",
                        "ranking": "band delta then vertical extent, never area",
                        "steeper_in_caltopo_null_means": (
                            "null is NOT COMPARED; an empty list is compared "
                            "and nothing read steeper")},
        "warnings": warnings,
        "images": {"panel": None, "disagreement": None},
    }

    if not wms_base_url or cache_dir is None:
        warnings.append(NO_TOKEN_SLOPE_WARNING)
        return empty

    if fetch is None:
        from ..caltopo.wms import fetch_panel as fetch

    try:
        panel_path = fetch(wms_base_url, [layer], tuple(block["bbox_3857"]),
                           block["width_px"], block["height_px"], Path(cache_dir))
    except Exception as error:  # noqa: BLE001 - a missing panel, not a refusal
        # Never the message: `fetch_panel`'s error and httpx's
        # `HTTPStatusError` both carry the full GetMap URL, whose path holds
        # the account token (see `net_errors.safe_reason`). `empty` is the
        # same "not compared" shape a missing token returns.
        warnings.append(
            f"CalTopo {layer} panel could not be fetched ({safe_reason(error)})")
        return empty

    with Image.open(panel_path) as image:
        rgba = np.asarray(image.convert("RGBA"))

    caltopo_bands = decode_slope_bands(rgba)
    measured = compute_slope_aspect(grid.elevation, grid.pixel_size_m)
    dem_bands = bin_degrees_to_bands(measured.slope_deg.astype("float64"))

    footprint = np.ones(caltopo_bands.shape, dtype=bool)
    if geometry is not None:
        from rasterio.features import rasterize
        footprint = rasterize([(project_geometry_to_3857(geometry), 1)],
                              out_shape=caltopo_bands.shape,
                              transform=grid.transform, fill=0,
                              all_touched=True).astype(bool)

    total = int(footprint.sum())
    unmatched = int((footprint & (caltopo_bands == UNKNOWN)).sum())
    below = int((footprint & (caltopo_bands == BELOW_LEGEND)).sum())
    unmatched_fraction = unmatched / total if total else None

    if unmatched_fraction and unmatched_fraction > UNMATCHED_WARNING_FRACTION:
        warnings.append(
            f"{unmatched_fraction:.0%} of the panel is unmatched (decoded to "
            f"no legend band). Two causes look identical here: the legend may "
            f"be wrong - it is {legend['provenance']} - or the panel may have "
            f"been resampled with an interpolating filter, which blends across "
            f"band boundaries into colours no legend contains (measured: up to "
            f"19% unmatched at LANCZOS on a CORRECT legend, 0% at "
            f"nearest-neighbour). `sf` is fetched nearest-neighbour by "
            f"default, so check the legend first")

    # Comparable pixels only: a palette that fails to decode half the raster
    # must not inflate agreement by shrinking what it is measured over.
    comparable = footprint & (caltopo_bands >= 0) & (dem_bands != UNKNOWN)
    comparable_count = int(comparable.sum())

    labels = band_labels()
    if comparable_count:
        caltopo_fractions = {
            label: float((comparable & (caltopo_bands == index)).sum()) / comparable_count
            for index, label in enumerate(labels)}
        dem_fractions = {
            label: float((comparable & (dem_bands == index)).sum()) / comparable_count
            for index, label in enumerate(labels)}
        # The DEM's own below-legend share, over the SAME denominator as the
        # dem band fractions beside it. Without it those fractions silently
        # sum to less than 1 with nothing saying where the rest went.
        # Measured on a gentle-DEM fixture: `caltopo` bands summed to 1.0
        # with a below_legend_fraction of 0.3125, while `dem` bands summed to
        # 0.0 and the missing 100% was unreported - and "the DEM reads
        # shallower than CalTopo" is the entire thesis of this tool, so its
        # largest instance was the one number left out.
        #
        # `comparable` requires `caltopo_bands >= 0`, so CalTopo's own
        # below-legend pixels are outside it by construction and its
        # fractions sum to 1 there; that is why `caltopo.below_legend_fraction`
        # is over the footprint and this one is not. Both denominators are
        # stated in `assumptions.denominator`.
        dem_below_fraction = (
            float((comparable & (dem_bands == BELOW_LEGEND)).sum()) / comparable_count)
        agreement = float((comparable & (caltopo_bands == dem_bands)).sum()) / comparable_count
    else:
        caltopo_fractions = dem_fractions = None
        dem_below_fraction = None
        agreement = None
        warnings.append("no pixel decoded to a band with a DEM slope beside it")

    dem_summary = _dem_summary(measured.slope_deg, footprint, warnings)

    steeper = comparable & (caltopo_bands > dem_bands)
    # None when there was nothing to compare - no pixel decoded to a band with
    # a DEM slope beside it - and a list only once a comparison actually
    # happened. `[]` then means "compared, and nothing reads steeper", which
    # is a measurement; `[]` for "never compared" is a claim nobody made.
    patches: list[dict] | None = [] if comparable_count else None
    #: The same patches in the raster's own EPSG:3857 frame, for `_outline_patches`
    #: to invert the affine transform on. Kept beside the WGS84 records rather
    #: than recovered from them, so the picture and the payload cannot drift.
    outlines_3857: list[dict] = []
    if steeper.any():
        # 8-connectivity, not the ndimage default of 4: a vertical face
        # rasterizes as a diagonal chain as often as an axis-aligned one,
        # and 4-connectivity would fracture that chain into pieces each
        # likely under MIN_PATCH_PIXELS - the same area-floor blindness
        # this tool exists to avoid, arriving through connectivity instead
        # of an explicit area filter.
        structure = np.ones((3, 3), dtype=int)
        components, count = ndimage.label(steeper, structure=structure)
        for label in range(1, count + 1):
            selected = components == label
            if int(selected.sum()) < MIN_PATCH_PIXELS:
                continue
            elevations = grid.elevation[selected]
            finite = elevations[np.isfinite(elevations)]

            # `raster_shapes` polygonizes with 4-connectivity regardless of
            # how the component above was labelled, so one 8-connected,
            # diagonally-touching component can come back as more than one
            # polygon here. Union them rather than keeping only the first -
            # taking geometries[0] would silently drop real ground.
            polygons = [
                shapely_shape(geom) for geom, value in raster_shapes(
                    selected.astype("uint8"), mask=selected,
                    transform=grid.transform)
                if value == 1]
            merged = unary_union(polygons) if polygons else None

            outlines_3857.append(
                shapely_mapping(merged) if merged is not None else None)
            patches.append({
                "id": f"steeper-{label}",
                # WGS84, not the 3857 the comparison was measured in: this is
                # the deliverable the docstring hands to `cliff_pitch` and
                # `site_facts`, and both read GeoJSON as WGS84.
                "geometry": to_wgs84_geojson(merged) if merged is not None else None,
                "area_m2": round(float(selected.sum()) * grid.pixel_size_m**2, 1),
                "caltopo_band": labels[int(caltopo_bands[selected].max())],
                "dem_band": _dem_band_label(int(dem_bands[selected].min()), labels),
                "band_delta": int((caltopo_bands[selected] - dem_bands[selected]).max()),
                "vertical_extent_m": (round(float(finite.max() - finite.min()), 1)
                                      if finite.size else None),
            })
        patches.sort(key=lambda patch: (patch["band_delta"],
                                        patch["vertical_extent_m"] or 0.0),
                     reverse=True)

    images = {"panel": None, "disagreement": None}
    if returns in ("image", "both"):
        images["panel"] = str(panel_path)
        images["disagreement"] = str(
            _outline_patches(panel_path, outlines_3857, grid.transform,
                             directory / "rasters" / "slope_disagreement.png"))

    dem_block = {"bands": dem_fractions,
                "below_legend_fraction": dem_below_fraction,
                "mean_deg": None, "p90_deg": None, "max_deg": None}
    if dem_summary is not None:
        dem_block.update(dem_summary)

    return {
        "layer": layer,
        "raster": block,
        "caltopo": {"bands": caltopo_fractions,
                    "unmatched_fraction": unmatched_fraction,
                    "below_legend_fraction": below / total if total else None},
        "dem": dem_block,
        "agreement_fraction": agreement,
        "steeper_in_caltopo": patches,
        "legend": legend,
        "assumptions": {"comparison": "band for band on a pixel-aligned grid",
                        "ranking": "band delta then vertical extent, never area",
                        "steeper_in_caltopo_null_means": (
                            "null is NOT COMPARED; an empty list is compared "
                            "and nothing read steeper"),
                        "denominator": (
                            "every band fraction, agreement_fraction and "
                            "dem.below_legend_fraction are over COMPARABLE "
                            "pixels - inside the footprint, decoded to a known "
                            "CalTopo band, and carrying a DEM slope. "
                            "caltopo.unmatched_fraction and "
                            "caltopo.below_legend_fraction are over the whole "
                            "FOOTPRINT, because a pixel that is unmatched or "
                            "below CalTopo's legend is by definition not "
                            "comparable and would have no denominator "
                            "otherwise. So caltopo.bands sums to 1 and "
                            "dem.bands sums to 1 minus "
                            "dem.below_legend_fraction")},
        "warnings": warnings,
        "images": images,
    }
