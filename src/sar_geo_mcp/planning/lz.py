"""Where a helicopter can put down: open, flat ground on a staged area,
graded the way the CalTopo First Response safety course grades a landing
zone, and measured rather than judged.

THE COURSE'S METHOD, AUTOMATED UP TO THE POINT WHERE A HUMAN LOOKS
------------------------------------------------------------------
Slope 0 to 5 degrees (7 in extremis), canopy 0 to 20 %, a clear footprint
of 75 ft across at the least and 150 by 200 ft for a medium aircraft, then
imagery over the top to find the boulders, wires and trees an elevation
model cannot see. This module does the first three on rasters the hub has
already staged, and cuts the picture for the fourth. It returns no verdict:
the course's own last line is that the pilot decides, and every envelope
says so in its assumptions.

WHAT A GRADE IS
---------------
`preferred`  a 25 m circle AND a 45 by 60 m rectangle fit inside the open
             ground at 0 to 5 degrees.
`minimal`    only the circle fits at 0 to 5 degrees.
`marginal`   the circle fits only once the slope limit is relaxed to 7.
A patch that fits nothing is not a candidate. The numbers are parameters
and every one of them is reported back.

HOW A FOOTPRINT IS FITTED
-------------------------
On a binary mask of open, flat cells. The circle is fitted with a distance
transform: the cell farthest from any closed cell is the touchdown point,
and its distance is the clear radius - the pole of inaccessibility, the
same idea `parking.py` uses for a marker inside an L-shaped lot. The
rectangle is fitted by morphological erosion with a rotated footprint at
six angles; a surviving cell at any angle means it fits. Both fits round
toward caution: the circle's radius is the centre-to-centre distance less
half a pixel, and the rectangle demands up to a cell more than its stated
length, so neither ever claims more clear ground than there is.

WHY A SPECK DOES NOT CLOSE GROUND, AND STILL COUNTS
----------------------------------------------------
The course says 0 to 20 % canopy "to account for mapping inaccuracies". A
cell is either vegetated or not on the CIR proxy, and a clearing on real
imagery is peppered with single vegetated cells - a shrub, a shadow, a
rendering artefact. If every one of those closed ground, the circle would
never fit anywhere real. So the fitting mask is the vegetated mask after a
3 by 3 morphological opening: a one-cell speck and a two-cell-wide sliver
are not obstacles for FITTING. They are not forgotten either: the canopy
fraction reported per candidate is the share of raw vegetated cells inside
its footprint, and a candidate over the limit is dropped. (A moving-window
canopy fraction was tried first and rejected: it shrinks every clearing at
its hard edge by nearly half the window, so a 30 m strip read as 16 m.)
The cost is that at 2 m staging a tree crown under three cells across is
fitted over; it still counts in the fraction. Since 0.25.0 that tree is
caught by the true-colour tree check (`terrain.trees`), which closes every
cell holding a crown or its shadow with no speck rule at all.

WHAT LEAVES THIS MODULE
-----------------------
Every geometry in the envelope is WGS84, emitted through
`geo.to_wgs84_geojson` like every other polygon this package returns;
measurement is in EPSG:3857 on the staged raster's own grid.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from ..caltopo.wms import LAYERS
from ..net_errors import safe_reason
from ..sources.static_imagery import GOOGLE_ATTRIBUTION, MAPBOX_ATTRIBUTION
from ..terrain.trees import TREES_METHOD, trees_on_grid

#: Coarsest raster this tool will grade. At 3.9 m per pixel (the default
#: 1,024 px staging of a 1,000 m area) a 25 m circle is six cells across and
#: an erosion by three is arithmetic noise; the app stages LZ ground at 2 m.
MAX_PIXEL_M = 3.0

#: The angles a rectangle footprint is tried at. Symmetric under 180, so
#: six steps of 30 cover every orientation to within 15 degrees.
FOOTPRINT_ANGLES_DEG: tuple[float, ...] = (0.0, 30.0, 60.0, 90.0, 120.0, 150.0)

#: The three grades, best first. A string and not an enum so it survives
#: JSON unchanged; the tool's docstring is where a caller reads what each
#: means.
GRADES: tuple[str, ...] = ("preferred", "minimal", "marginal")

#: Side of the square a vegetated patch must fill to close ground for
#: fitting. Three cells: a one-cell speck or a two-cell sliver is fitted
#: over, and counted in the canopy fraction regardless.
SPECK_CELLS = 3

TOO_COARSE_ERROR = (
    "this ground is staged at {pixel:.1f} m per pixel, coarser than the {limit:.1f} m "
    "this tool will grade a landing zone at - a 25 m circle is a handful of cells "
    "there and the fit is noise. Stage the area again with resolution_m=2.0")

NO_CANOPY_ERROR = (
    "no canopy panel: this ground has no panels/vegetation.png and no WMS base "
    "URL or cache directory was given to fetch the NAIP false-colour layer with. A "
    "landing zone graded on slope alone is exactly the false confidence the course "
    "warns about, so nothing is graded. Provide a WMS token or stage through the "
    "debrief app, which writes the panel")

#: The OSM classes a landing zone's `road` fact can name (`terrain.linear.
#: classify_way`): a road, a track, or another highway a vehicle uses.
ROAD_CLASSES = frozenset({"ROAD", "TRACK", "ROAD_OTHER"})

# The fourth same-named constant in this package, on purpose: each names what
# ITS caller loses, and this one loses wires - and, since 0.18.0, the road and
# lot facts. The first sentence is kept word for word and the second appended,
# so a reader matching the first still matches.
MISSING_OSM_WARNING = (
    "osm.json is missing from the staged ground, so power lines near these "
    "landing zones are unknown rather than absent. The road each zone touches "
    "or sits beside, and whether it lies in a parking lot, are unknown too")

POWER_UNLABELLED_WARNING = (
    "the staged ground was written before barrier records carried the power "
    "tag, so a wire cannot be told from a fence here: power lines are unknown. "
    "Re-stage this area with force=True to label them")

POWER_NOT_FETCHED_WARNING = (
    "this ground was staged with OpenStreetMap query version {version}, before "
    "barriers - power lines among them - were fetched at all, so power lines are "
    "unknown rather than absent. Re-stage this area with force=True to fetch them")

MINOR_LINES_NOT_CHECKED = (
    "this ground was staged before distribution lines (power=minor_line) were "
    "asked for, so only transmission lines were checked; re-stage it with "
    "force=True to check both")

TREES_NOT_CHECKED = (
    "trees were not checked on true-colour imagery ({reason}), so a single tree "
    "or a scatter of them may stand inside a clear circle: the CIR canopy proxy "
    "alone misses dark conifers. Check every candidate's chip for trees")

#: The two sources open water is closed from (`_closed_water`), as
#: `assumptions.water_sources_checked` names them.
WATER_SOURCE_EVT = "landfire_open_water"
WATER_SOURCE_OSM = "osm_shorelines"

WATER_NOT_CHECKED_WARNING = (
    "open water was not checked: there is no readable evt.tif and no osm.json for "
    "this ground, so a lake reads as flat open ground and may be offered as a "
    "landing zone. Check every candidate's chip for water")

WATER_EVT_MISSING_WARNING = (
    "LANDFIRE open water could not be read for this ground (evt.tif is missing or "
    "unreadable), so only mapped OpenStreetMap shorelines closed water; a lake "
    "mapped as several ways, or not mapped, may still be offered. Check the chips")

#: The CalTopo layer the canopy proxy is read from; what the debrief app
#: renders into `panels/vegetation.png` and what `terrain/vegetation.py` says
#: it decodes.
CANOPY_LAYER = "ir"

CANOPY_METHOD = "CIR proxy (R-G)/(R+G) on rendered NAIP false-colour IR, per cell"

#: CalTopo's building-footprint layer, read into `panels/structures.png`.
STRUCTURES_LAYER = "structures"

STRUCTURES_NOT_CHECKED = ("structures were not checked: CalTopo's structures layer "
                          "could not be had for this ground")

#: The least alpha a structures-panel pixel needs to count as a building.
#: Measured on a real CalTopo panel over a house at Windy Hill: CalTopo
#: draws footprints at alpha 64-127, never opaque, so a threshold of 128
#: would see no building at all; the faint 1-31 band is resampling halo,
#: which would widen every building by up to three cells.
STRUCTURE_ALPHA_MIN = 32

#: A structures panel reading as building over more than this share of the
#: raster is not a structures layer - an error image, a basemap, a renamed
#: layer - and closing on it would close every field while saying
#: "checked".
STRUCTURE_MAX_COVER = 0.5

STRUCTURES_NO_TRANSPARENCY = "the panel has no transparency"
STRUCTURES_COVERED_MOST = ("the layer covered most of the ground, which is not a "
                           "structures layer")

#: The chip: what the imagery read looks at. 200 m square at 0.5 m per
#: pixel is the whole preferred footprint with room to see what surrounds
#: it, at a resolution NAIP actually has. At mid-latitudes a 200 m, 400 px
#: request resolves to tile level 18 and a 1.05x downsample; 512 px would
#: cross into level 19 upsampling. One chip per source, each file named
#: `lz/<key>-<source>.png`, the key the first 8 hex of sha256 over the
#: WGS84 touchdown's f"{lon:.6f},{lat:.6f}". (Before 0.17.0 there was one
#: source and the file was `lz/<key>.png`; such a file is left where it is
#: and not reused - a chip is cheap to cut again.)
CHIP_LAYER = "imagery"
CHIP_M = 200.0
CHIP_PX = 400

#: The CalTopo chip sources, in the order they are cut and listed:
#: (source, label, WMS layer, attribution). The label is "CalTopo " and the
#: layer's own title in `caltopo.wms.LAYERS`, so it says what CalTopo calls
#: it. Cut when there is a WMS base URL and a cache directory.
CHIP_SOURCES: tuple[tuple[str, str, str, str], ...] = tuple(
    (source, f"CalTopo {LAYERS[layer]}", layer, attribution)
    for source, layer, attribution in (("caltopo-imagery", CHIP_LAYER, "CalTopo"),
                                       ("caltopo-naip", "n3", "CalTopo, USDA NAIP")))

#: The keyed static-image sources (`sources/static_imagery.py`), after the CalTopo
#: ones and in this order: (source, label, attribution). Each is cut only
#: when the server is configured with its key.
STATIC_SOURCES: tuple[tuple[str, str, str], ...] = (
    ("mapbox-satellite", "Mapbox Satellite", MAPBOX_ATTRIBUTION),
    ("google-satellite", "Google Satellite", GOOGLE_ATTRIBUTION))

#: The static sources whose refusal ends them for the rest of a call: a
#: 401 or 403 is a key that will not work for the next candidate either,
#: and a transport error is a host that is not there. A CalTopo WMS failure
#: is not on this list - it is usually one tile - and stays per candidate.
REFUSED_STATUSES = (401, 403)

NO_TOKEN_CHIP_WARNING = (
    "no imagery source: no WMS base URL or cache directory, and no Mapbox or Google "
    "key, so no imagery chips were cut: chips is empty and chip_path is null on every "
    "candidate, and the imagery read cannot be made")

CHIP_FAILED_WARNING = ("the {source} chip for {candidate} could not be made ({reason}); "
                       "it is left out of that candidate's chips")

SOURCE_REFUSED_WARNING = ("{source} chips were not cut for the remaining landing zones "
                          "after it refused ({reason})")

#: Aspect is sampled only from footprint cells at least this steep, and is
#: reported only when that sample covers at least ASPECT_MIN_SHARE of the
#: disc and its circular mean is at least ASPECT_MIN_RESULTANT strong. On
#: a flat field the bearing of steepest descent is the sign of DEM noise,
#: and a disc that faces east on one side and west on the other has no
#: aspect at all: both reported as a confident compass point on a landing
#: zone would be a wrong answer, so both are null.
ASPECT_MIN_SLOPE_DEG = 1.0
ASPECT_MIN_SHARE = 0.25
ASPECT_MIN_RESULTANT = 0.5

__all__ = ["ASPECT_MIN_RESULTANT", "ASPECT_MIN_SHARE", "ASPECT_MIN_SLOPE_DEG",
           "CANOPY_LAYER", "CANOPY_METHOD", "CHIP_FAILED_WARNING", "CHIP_LAYER",
           "CHIP_M", "CHIP_PX", "CHIP_SOURCES", "FOOTPRINT_ANGLES_DEG", "GRADES", "MAX_PIXEL_M",
           "MINOR_LINES_NOT_CHECKED", "MISSING_OSM_WARNING", "NO_CANOPY_ERROR",
           "NO_TOKEN_CHIP_WARNING", "POWER_NOT_FETCHED_WARNING",
           "POWER_UNLABELLED_WARNING", "REFUSED_STATUSES",
           "SOURCE_REFUSED_WARNING", "SPECK_CELLS",
           "STRUCTURES_COVERED_MOST", "STRUCTURES_LAYER", "STRUCTURES_NOT_CHECKED",
           "STATIC_SOURCES", "STRUCTURES_NO_TRANSPARENCY", "STRUCTURE_ALPHA_MIN",
           "STRUCTURE_MAX_COVER", "TOO_COARSE_ERROR", "TREES_NOT_CHECKED",
           "WATER_EVT_MISSING_WARNING",
           "WATER_NOT_CHECKED_WARNING", "WATER_SOURCE_EVT", "WATER_SOURCE_OSM",
           "largest_inscribed_circle", "lz_candidates", "rectangle_fits_at",
           "rectangle_footprint"]


def rectangle_footprint(*, length_px: float, width_px: float,
                        angle_deg: float) -> np.ndarray:
    """A boolean structuring element: `length_px` by `width_px`, its long
    axis turned `angle_deg` anticlockwise from east on a north-up map (rows
    increase southward, hence the negated angle), centred in an odd square
    big enough to hold it at any angle. Integer offsets against a
    half-width bound mean the element is one cell longer than `length_px`
    when that is an integer: the rectangle demands a cell more ground than
    asked, never less."""
    size = int(math.ceil(math.hypot(length_px, width_px))) | 1
    half = size // 2
    offsets = np.arange(size) - half
    v, u = np.meshgrid(offsets, offsets, indexing="ij")   # v down rows, u along cols
    a = -math.radians(angle_deg)
    along = u * math.cos(a) + v * math.sin(a)
    across = -u * math.sin(a) + v * math.cos(a)
    eps = 1e-9   # cos(90 deg) is 6e-17, not 0, and a boundary cell must not tip on that
    return ((np.abs(along) <= length_px / 2.0 + eps)
            & (np.abs(across) <= width_px / 2.0 + eps))


def largest_inscribed_circle(mask: np.ndarray, *, pixel_m: float
                             ) -> tuple[int, int, float] | None:
    """(row, col, radius_m) of the largest circle inside `mask`, or None
    when the mask is empty. The raster edge counts as closed ground: the
    mask is padded with one closed cell all round before the transform, so
    open ground running off the raster is not read as infinitely open.
    The transform measures centre to centre, which is half a pixel
    generous, so half a pixel is taken off: the radius reported is exact
    for an odd-width patch and half a pixel short for an even one, never
    long. Ties in the transform resolve to the first cell in row-major
    order, so an even-sized plateau's touchdown is its upper-left centre
    cell."""
    from scipy import ndimage

    if not mask.any():
        return None
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    distance = ndimage.distance_transform_edt(padded)[1:-1, 1:-1]
    row, col = np.unravel_index(int(np.argmax(distance)), distance.shape)
    return int(row), int(col), (float(distance[row, col]) - 0.5) * float(pixel_m)


def rectangle_fits_at(mask: np.ndarray, *, length_px: float, width_px: float,
                      angles_deg: tuple[float, ...]) -> float | None:
    """The first angle in `angles_deg` at which a `length_px` by `width_px`
    rectangle fits inside `mask`, or None. Erosion by the rotated footprint
    with the outside of the raster closed: any surviving cell is a centre
    the whole footprint fits around. Erodes every cell of `mask`; a caller
    with one patch in a large raster passes a window around it."""
    from scipy import ndimage

    if not mask.any():
        return None
    for angle in angles_deg:
        footprint = rectangle_footprint(length_px=length_px, width_px=width_px,
                                        angle_deg=angle)
        if ndimage.binary_erosion(mask, structure=footprint, border_value=0).any():
            return float(angle)
    return None


def _aspect_of(slope_deg: np.ndarray, aspect_deg: np.ndarray) -> str | None:
    """The footprint's compass aspect, or None when the ground is too flat
    or too mixed to have one (see the constants above)."""
    from .site import _cardinal  # package-internal helper shared by planning modules

    steep = (np.isfinite(slope_deg) & np.isfinite(aspect_deg)
            & (slope_deg >= ASPECT_MIN_SLOPE_DEG))
    if steep.sum() < ASPECT_MIN_SHARE * max(1, slope_deg.size):
        return None
    radians = np.radians(aspect_deg[steep])
    resultant = math.hypot(float(np.sin(radians).mean()), float(np.cos(radians).mean()))
    if resultant < ASPECT_MIN_RESULTANT:
        return None
    return _cardinal(aspect_deg[steep])


def _footprint_window(shape: tuple[int, int], row: int, col: int, radius_px: float
                      ) -> tuple[slice, slice, np.ndarray]:
    """Row slice, column slice and the local boolean disc of the circle at
    (row, col); the slices are clipped to the raster so a touchdown near
    its edge indexes nothing outside it."""
    reach = int(math.ceil(radius_px))
    rs = slice(max(0, row - reach), min(shape[0], row + reach + 1))
    cs = slice(max(0, col - reach), min(shape[1], col + reach + 1))
    rr, cc = np.ogrid[rs.start:rs.stop, cs.start:cs.stop]
    return rs, cs, (rr - row) ** 2 + (cc - col) ** 2 <= radius_px ** 2


def _patch_polygon_3857(piece: np.ndarray, transform):
    """The True cells of the boolean window `piece` as a shapely polygon in
    the raster's own CRS; `transform` maps the window's own (col, row) to
    that CRS, so the caller composes the raster transform with the window's
    offset."""
    from rasterio.features import shapes as rasterio_shapes
    from shapely.geometry import shape as shapely_shape
    from shapely.ops import unary_union

    polygons = [shapely_shape(geom) for geom, value in
                rasterio_shapes(piece.astype("uint8"), mask=piece, transform=transform)
                if value == 1]
    return unary_union(polygons)


def _grade_patches(*, open_ok: np.ndarray, flat_preferred: np.ndarray,
                   flat_marginal: np.ndarray,
                   vegetated: np.ndarray, max_canopy: float,
                   pixel_m: float, min_diameter_m: float,
                   preferred_width_m: float, preferred_length_m: float,
                   transform) -> list[dict[str, Any]]:
    """Every connected pocket of open ground at 0 to 5 degrees is graded on
    its own: the circle fitting inside it is `minimal`, the rectangle as
    well is `preferred`. A connected patch of open ground at 0 to 7
    degrees is offered as `marginal` only when no surviving pocket lies in
    it. The canopy check is part of grading, so a pocket dropped for
    canopy does not exist for the purpose of that fallback.

    WHY EACH POCKET ON ITS OWN GROUND. A candidate's grade, touchdown,
    polygon and fit must all describe the same ground. The earlier rule of
    one candidate per 7-degree patch could not guarantee that once a patch
    held two pockets: the touchdown came from the largest circle across
    both, the rectangle fit from either, and the polygon from whichever
    held the touchdown, so one entry could carry a preferred fit that
    existed only in the pocket it did not point at.

    WHY CANOPY IS GRADING. A pocket that is dropped for canopy must not
    take a clean field down with it: with the drop after grading, the
    vegetated pocket claimed its 7-degree patch and the marginal fallback
    for the open field around it was never reached.

    WHY EVERY FIT IS WINDOWED. The mask is labelled once for each slope
    limit and every fit, canopy count and vectorisation runs on the
    patch's bounding box, never the whole raster; a thousand-pixel raster
    with two thousand patches is seconds rather than minutes. The
    windowed circle is exact, not approximate: nothing of a connected
    component lies outside its bounding box, so the box edge is closed
    ground with respect to it, which is what the full raster said too.
    The erosion is not padded for the same reason, `border_value=0`
    closing everything past the box edge."""
    from affine import Affine
    from scipy import ndimage

    radius_m = min_diameter_m / 2.0
    min_cells = min_diameter_m / pixel_m
    length_px = preferred_length_m / pixel_m
    width_px = preferred_width_m / pixel_m
    labels_5, _n5 = ndimage.label(open_ok & flat_preferred)
    labels_7, _n7 = ndimage.label(open_ok & flat_marginal)

    def wide_enough(sl: tuple[slice, slice]) -> bool:
        return (sl[0].stop - sl[0].start >= min_cells
                and sl[1].stop - sl[1].start >= min_cells)

    def fit_circle(labels, label, sl):
        """(piece, row, col, clear_radius_m, canopy, footprint) for the
        largest circle in the patch, or None when the circle does not fit
        or the ground under it is over the canopy limit."""
        piece = labels[sl] == label
        hit = largest_inscribed_circle(piece, pixel_m=pixel_m)
        if hit is None or hit[2] < radius_m:
            return None
        row, col = hit[0] + sl[0].start, hit[1] + sl[1].start
        clear_radius_m = hit[2]
        footprint = _footprint_window(labels.shape, row, col, clear_radius_m / pixel_m)
        rs, cs, disc = footprint
        canopy = float(vegetated[rs, cs][disc].mean())
        if canopy > max_canopy:
            return None
        return piece, row, col, clear_radius_m, canopy, footprint

    def polygon(piece, sl):
        offset = transform * Affine.translation(sl[1].start, sl[0].start)
        return _patch_polygon_3857(piece, offset)

    entries: list[dict[str, Any]] = []
    claimed_7: set[int] = set()

    for label, sl in enumerate(ndimage.find_objects(labels_5), start=1):
        if sl is None or not wide_enough(sl):
            continue
        fitted = fit_circle(labels_5, label, sl)
        if fitted is None:
            continue
        piece, row, col, clear_radius_m, canopy, footprint = fitted
        fit_angle = None
        # The rectangle needs its short side to fit as a circle first;
        # skipping the erosion when it cannot is most of the saving.
        if clear_radius_m >= min(preferred_width_m, preferred_length_m) / 2.0:
            fit_angle = rectangle_fits_at(piece, length_px=length_px, width_px=width_px,
                                          angles_deg=FOOTPRINT_ANGLES_DEG)
        claimed_7.add(int(labels_7[row, col]))
        entries.append({
            "grade": "preferred" if fit_angle is not None else "minimal",
            "row": row, "col": col,
            "clear_radius_m": clear_radius_m,
            "preferred_fit": {"fits": fit_angle is not None, "angle_deg": fit_angle},
            "polygon_3857": polygon(piece, sl),
            "canopy": canopy,
            "footprint": footprint,
        })

    for label, sl in enumerate(ndimage.find_objects(labels_7), start=1):
        if sl is None or label in claimed_7 or not wide_enough(sl):
            continue
        fitted = fit_circle(labels_7, label, sl)
        if fitted is None:
            continue
        piece, row, col, clear_radius_m, canopy, footprint = fitted
        entries.append({
            "grade": "marginal",
            "row": row, "col": col,
            "clear_radius_m": clear_radius_m,
            "preferred_fit": {"fits": False, "angle_deg": None},
            "polygon_3857": polygon(piece, sl),
            "canopy": canopy,
            "footprint": footprint,
        })
    return entries


def _canopy_panel(directory: Path, grid, *, wms_base_url: str | None,
                  cache_dir: Path | None, fetch) -> Path:
    """`panels/vegetation.png`, fetched at the raster's own georeference when
    the staged directory lacks it. Written where `TerrainContext` reads it,
    so the clearing detector and this tool see one file. Area staging
    writes no panels at all - the debrief app adds them - which is why a
    tool on an area handle has to be able to fetch its own.

    Written atomically, because a torn panel that `is_file()` trusts would
    break every later call on this handle, and nothing repairs it.

    `manifest.json`'s `panels` map is deliberately not updated: nothing in
    the hub reads it, and the app rewrites the manifest when it stages."""
    from ..cache import atomic_write_bytes

    panel = directory / "panels" / "vegetation.png"
    if panel.is_file():
        return panel
    if not wms_base_url or cache_dir is None:
        raise ValueError(NO_CANOPY_ERROR)
    from .rasters import georeference

    block = georeference(grid.transform, grid.elevation.shape, grid.latitude_deg)
    reason = None
    try:
        source = fetch(wms_base_url, [CANOPY_LAYER], tuple(block["bbox_3857"]),
                       block["width_px"], block["height_px"], Path(cache_dir))
    except Exception as error:  # noqa: BLE001 - getting the panel is this tool's one refusal
        reason = safe_reason(error)
    if reason is not None:
        # Raised OUTSIDE the except block: the failed fetch's own message
        # carries the GetMap URL, token and all, and raised inside it the
        # refusal would hold that error as `__context__` even with `from
        # None` - one introspecting handler away from a log.
        raise ValueError(f"no canopy panel: the {CANOPY_LAYER} fetch failed "
                         f"({reason}); nothing is graded")
    panel.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(panel, Path(source).read_bytes())
    return panel


def _structures_panel(directory: Path, grid, *, wms_base_url: str | None,
                      cache_dir: Path | None, fetch) -> tuple[Path | None, str | None]:
    """`panels/structures.png`, fetched at the raster's georeference when
    the staged directory lacks it, or (None, why) when it cannot be had.
    Not a refusal, unlike the canopy panel: the layer is incomplete even
    when present, so grading without it loses a filter, not the grade.

    `fetch_panel` resamples this layer with `wms.MAX_ALPHA`, not LANCZOS:
    a 1 px outline must survive the shrink onto the grid (see there)."""
    panel = directory / "panels" / "structures.png"
    if panel.is_file():
        return panel, None
    if not wms_base_url:
        return None, f"{STRUCTURES_NOT_CHECKED} (no WMS token)"
    if cache_dir is None:
        return None, f"{STRUCTURES_NOT_CHECKED} (no cache directory)"
    from ..cache import atomic_write_bytes
    from .rasters import georeference

    block = georeference(grid.transform, grid.elevation.shape, grid.latitude_deg)
    try:
        source = fetch(wms_base_url, [STRUCTURES_LAYER], tuple(block["bbox_3857"]),
                       block["width_px"], block["height_px"], Path(cache_dir))
        panel.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(panel, Path(source).read_bytes())
    except Exception as error:  # noqa: BLE001 - a missing filter, not a refusal
        return None, f"{STRUCTURES_NOT_CHECKED} ({safe_reason(error)})"
    return panel, None


def _closed_near_structures(grid, panel: Path, clearance_m: float
                            ) -> tuple[np.ndarray | None, str | None]:
    """(cells within `clearance_m` of a building on the structures panel,
    None), or (None, why) when the panel cannot be read as a structures
    layer. The layer is transparent everywhere but a footprint, drawn
    semi-transparent (see `STRUCTURE_ALPHA_MIN`), so alpha is the whole
    test. A clearance of 0 still closes the footprint itself.

    A footprint drawn as an outline only would leave its inside open, so
    holes are filled. A panel finer than the grid is shrunk by taking the
    maximum alpha over each cell's footprint (`wms.max_pool`, the same rule
    the fetch uses): nearest neighbour would step over a thin outline, and
    an average would fade it under `STRUCTURE_ALPHA_MIN` wherever it
    straddles two cells.

    A panel Pillow cannot read raises here; the caller turns that into
    "not checked".

    The margin is never short, the same rule `_closed_near_lines` keeps:
    a cell is closed when its centre is within the clearance plus a FULL
    cell diagonal of a building cell's centre - `(clearance_m + sqrt(2) *
    pixel_m) / pixel_m` in cells. A full diagonal and not half, because
    the distance transform measures centre to centre while the building's
    edge may lie anywhere in its cell and the touchdown ground anywhere in
    its own: each end can be up to half a diagonal nearer than its centre,
    so the two nearest points can be a whole diagonal closer than the
    centres are."""
    from PIL import Image
    from scipy import ndimage

    shape = grid.elevation.shape
    with Image.open(panel) as image:
        has_alpha = (image.mode in ("RGBA", "LA", "PA")
                     or "transparency" in image.info)
        if not has_alpha:
            # Converted, it would read as opaque everywhere and close every
            # field while saying "checked".
            return None, STRUCTURES_NO_TRANSPARENCY
        alpha = image.convert("RGBA").getchannel("A")
    size = (shape[1], shape[0])
    if alpha.size != size:
        shrinking = alpha.size[0] > size[0] or alpha.size[1] > size[1]
        if shrinking:
            from ..caltopo.wms import max_pool

            alpha = max_pool(np.asarray(alpha), size[0], size[1])
        else:
            alpha = alpha.resize(size, Image.NEAREST)
    structure = np.asarray(alpha) >= STRUCTURE_ALPHA_MIN
    if not structure.any():
        return np.zeros(shape, dtype=bool), None
    structure = ndimage.binary_fill_holes(structure)
    if structure.mean() > STRUCTURE_MAX_COVER:
        return None, STRUCTURES_COVERED_MOST
    if clearance_m <= 0:
        return structure, None
    # Compared in cells: no ground-metre copy of a raster-sized array. A
    # full cell diagonal added, so the margin is never short (see above).
    pixel_m = float(grid.pixel_size_m)
    reach_px = (clearance_m + math.sqrt(2) * pixel_m) / pixel_m
    return ndimage.distance_transform_edt(~structure) <= reach_px, None


class Masks(NamedTuple):
    """The rasters every grade is read from (see `_masks`). Callers read
    slope off `slope_aspect.slope_deg` rather than a separate key. Callers
    may narrow `open_ok` with further closures - power lines, structures -
    before grading."""
    vegetated: np.ndarray
    slope_aspect: Any
    flat_preferred: np.ndarray
    flat_marginal: np.ndarray
    open_ok: np.ndarray


def _masks(grid, panel: Path, *, max_slope_deg: float, marginal_slope_deg: float,
           region: np.ndarray | None = None) -> Masks:
    """The rasters every grade is read from, built once: the vegetated
    cells off the CIR proxy, the two slope masks, and `open_ok` - inside
    the planner's circle, measured, and not closed by vegetation once
    specks are opened away (see the module docstring)."""
    from scipy import ndimage

    from ..terrain.slope import compute_slope_aspect
    from ..terrain.vegetation import VEGETATION_PROXY_THRESHOLD, cir_proxy

    proxy = cir_proxy(panel, grid.elevation.shape)
    vegetated = proxy > VEGETATION_PROXY_THRESHOLD

    finite = np.isfinite(grid.elevation)
    slope_aspect = compute_slope_aspect(grid.elevation, grid.pixel_size_m)
    slope = slope_aspect.slope_deg
    flat_preferred = finite & (slope <= max_slope_deg)
    flat_marginal = finite & (slope <= marginal_slope_deg)

    # Specks do not close ground for fitting; every vegetated cell still
    # counts in the fraction below (see the module docstring).
    closed = ndimage.binary_opening(vegetated, structure=np.ones((SPECK_CELLS, SPECK_CELLS),
                                                                 dtype=bool))
    open_ok = (grid.mask if region is None else region) & finite & ~closed

    return Masks(vegetated=vegetated, slope_aspect=slope_aspect,
                flat_preferred=flat_preferred, flat_marginal=flat_marginal,
                open_ok=open_ok)


def _osm_payload(directory: Path) -> dict | None:
    """The staged osm.json, read once for the power lines, the road fact and
    the parking lots, or None when there is none."""
    import json

    path = directory / "osm.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _power_lines(payload: dict | None) -> tuple[list[dict] | None, str | None, list[str]]:
    """From the staged osm.json `payload` (None when there is no file), the
    staged `power=line` and `power=minor_line` ways as (list,
    warning, kinds_checked). `None` for the list means unknown - no
    osm.json, or a barrier record with no `power` key, which predates the
    field and might be a wire; `kinds_checked` is `[]` then. Three states
    leave this function and the caller keeps all three: a list (possibly
    empty) is "mapped wires, here they are", None is "could not tell".

    Ground staged before `OSM_QUERY_VERSION_BARRIERS` (2) - version 1, or
    a file with no `query_version`, which is version 1 - fetched no barrier
    of any kind, so no power line was ever asked for: unknown, with a
    warning naming the version, never "transmission lines checked".

    Ground staged before `OSM_QUERY_VERSION_MINOR_LINES` (4) was queried
    for `power=line` only, never `power=minor_line` - but its barriers already carry the
    `power` key, version 3 having added that. So this is not "could not
    tell": the known transmission lines are kept and returned as usual,
    `kinds_checked` is `["line"]`, and a warning says distribution lines
    were never asked for and ground was staged, and possibly closed,
    without them."""
    from ..sources.osm import OSM_QUERY_VERSION_BARRIERS, OSM_QUERY_VERSION_MINOR_LINES

    if payload is None:
        return None, MISSING_OSM_WARNING, []
    # Absent means version 1, the same reading `boundary.class_lines` gives
    # it: the key arrived with version 2, so a file without it predates it.
    version = int(payload.get("query_version", 1))
    if version < OSM_QUERY_VERSION_BARRIERS:
        return None, POWER_NOT_FETCHED_WARNING.format(version=version), []
    routes = payload.get("routes") or []
    barriers = [route for route in routes
                if (route.get("properties") or {}).get("route_type") == "BARRIER"]
    if any("power" not in (route.get("properties") or {}) for route in barriers):
        return None, POWER_UNLABELLED_WARNING, []
    # The minor-lines version, not the current one: a later query (5, road
    # tags and gates) added nothing a power line reads, so a version-4 file
    # checked both kinds.
    if version < OSM_QUERY_VERSION_MINOR_LINES:
        lines = [route for route in barriers if route["properties"].get("power") == "line"]
        return lines, MINOR_LINES_NOT_CHECKED, ["line"]
    lines = [route for route in barriers
            if route["properties"].get("power") in ("line", "minor_line")]
    return lines, None, ["line", "minor_line"]


def _closed_near_lines(grid, lines: list[dict], clearance_m: float) -> np.ndarray:
    """Cells within `clearance_m` of any of `lines` (EPSG:3857 GeoJSON), as a
    boolean mask on the grid. The lines are burned in, and a Euclidean
    distance transform measures every cell's distance from them, so the
    margin is round at a line's end and never short: a cell is closed when
    any part of it may lie within the clearance of any part of a burned
    cell. That takes the clearance plus a FULL cell diagonal, compared
    centre to centre: the wire may run anywhere through its burned cell
    and the ground may be anywhere in its own, each up to half a diagonal
    from its cell's centre, so the two nearest points can be a whole
    diagonal closer than the centres are."""
    from rasterio.features import rasterize
    from scipy import ndimage
    from shapely.geometry import shape as shapely_shape

    shape = grid.elevation.shape
    if not lines or clearance_m <= 0:
        return np.zeros(shape, dtype=bool)
    # `lines` is already clipped to the staged area (see terrain.linear's
    # route fetch), so a line beyond this raster is never seen here - and
    # the staged buffer (1,000 m for a planning area) is far wider than any
    # clearance this is called with.
    burned = rasterize([(shapely_shape(line["geometry"]), 1) for line in lines],
                       out_shape=shape, transform=grid.transform, fill=0,
                       all_touched=True).astype(bool)
    if not burned.any():
        return np.zeros(shape, dtype=bool)
    pixel_m = float(grid.pixel_size_m)
    distance_m = ndimage.distance_transform_edt(~burned) * pixel_m
    # Centre to centre, so a full cell diagonal is added (see above).
    return distance_m <= clearance_m + math.sqrt(2) * pixel_m


def _closed_water(directory: Path, grid, osm: dict | None
                  ) -> tuple[np.ndarray, list[str], list[str]]:
    """Open water as a mask on the grid, with the sources that saw it and
    any warnings, as (closed, sources_checked, warnings).

    A lake is flat and treeless, so slope and the canopy proxy alone read
    it as the best landing zone on the map. Two sources close it, and
    either alone is enough to close a cell:

    - LANDFIRE open water in `evt.tif` (lifeform "Water"). It is 30 m, so
      it can spill up to a pixel onto the shore; that closes some beach,
      which is the safe direction to be wrong in.
    - OpenStreetMap shorelines (`WATER_EDGE`: natural=water, coastline,
      landuse=reservoir). Every one is burned in as a line, and a closed
      ring is filled. A ring is filled whatever it bounds: an island mapped
      as a closed coastline is closed too, which loses ground and never
      offers water. A lake drawn as several open ways is left to LANDFIRE.

    With neither source the mask is empty, `sources_checked` is [] and a
    warning says water was not checked."""
    from rasterio.errors import RasterioError
    from rasterio.features import rasterize
    from shapely.geometry import Polygon
    from shapely.geometry import shape as shapely_shape

    from ..sources.landfire import lifeform_of
    from ..terrain.evt import codes_at_cells
    from ..terrain.movement import CLOSED_LIFEFORMS

    shape = grid.elevation.shape
    closed = np.zeros(shape, dtype=bool)
    sources: list[str] = []
    warnings: list[str] = []

    evt_path = Path(directory) / "evt.tif"
    evt_read = False
    if evt_path.is_file():
        rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
        try:
            codes = codes_at_cells(evt_path, grid.transform, rows.ravel(), cols.ravel())
        except (OSError, RasterioError):
            codes = None
        if codes is not None:
            unique, inverse = np.unique(codes, return_inverse=True)
            water_of = np.array([lifeform_of(int(code)) in CLOSED_LIFEFORMS
                                 for code in unique], dtype=bool)
            closed |= water_of[inverse].reshape(shape)
            sources.append(WATER_SOURCE_EVT)
            evt_read = True

    if osm is not None:
        shapes = []
        for route in osm.get("routes") or []:
            if (route.get("properties") or {}).get("route_type") != "WATER_EDGE":
                continue
            line = shapely_shape(route["geometry"])
            shapes.append((line, 1))
            coords = list(line.coords) if line.geom_type == "LineString" else []
            if len(coords) >= 4 and coords[0] == coords[-1]:
                ring = Polygon(coords)
                if ring.is_valid and not ring.is_empty:
                    shapes.append((ring, 1))
        if shapes:
            closed |= rasterize(shapes, out_shape=shape, transform=grid.transform,
                                fill=0, all_touched=True).astype(bool)
        sources.append(WATER_SOURCE_OSM)

    if not sources:
        warnings.append(WATER_NOT_CHECKED_WARNING)
    elif not evt_read:
        warnings.append(WATER_EVT_MISSING_WARNING)
    return closed, sources, warnings


def _nearest_power_line(x: float, y: float, lines: list[dict], tree,
                        *, scale: float, notice_m: float) -> dict | None:
    """The nearest of `lines` to the touchdown (x, y) in EPSG:3857 within
    `notice_m` ground metres, as {distance_m, name, power}, or None when
    nothing mapped is that close - which is NOT "no wire", and the tool's
    docstring says so."""
    from shapely.geometry import Point

    if tree is None:
        return None
    indices, distances = tree.query_nearest(
        Point(x, y), max_distance=notice_m / scale, return_distance=True,
        all_matches=False)
    if len(indices) == 0:
        return None
    route = lines[int(indices[0])]
    return {"distance_m": round(float(distances[0]) * scale, 1),
            "name": (route.get("properties") or {}).get("name"),
            "power": (route.get("properties") or {}).get("power")}


def _roads(payload: dict) -> list[dict]:
    """The staged ways a landing zone's `road` fact can name: every ROAD,
    TRACK and ROAD_OTHER, except a way egress also ignores - one not built
    (`egress_roads.NOT_BUILT`: proposed, under construction, abandoned...)
    and one nobody means by a road (`egress_roads.NOT_WALKED`: a via
    ferrata, a bus guideway). A motorway IS kept: a touchdown beside one
    needs traffic control more than any. On ground staged before query
    version 5 no record carries `highway`, so nothing is left out for it
    there."""
    from .egress_roads import NOT_BUILT, NOT_WALKED

    roads = []
    for route in payload.get("routes") or []:
        properties = route.get("properties") or {}
        if properties.get("route_type") not in ROAD_CLASSES:
            continue
        highway = str(properties.get("highway") or "").strip().lower()
        if highway in NOT_BUILT | NOT_WALKED:
            continue
        roads.append(route)
    return roads


def _nearest_road(x: float, y: float, roads: list[dict], tree, *, scale: float,
                  notice_m: float, clear_radius_m: float) -> dict | None:
    """The nearest of `roads` to the touchdown (x, y) in EPSG:3857 within
    `notice_m` ground metres, as {name, route_type, highway, surface,
    distance_m, on_road}, or None when none is that close. `on_road` is the
    way coming within `clear_radius_m` of the touchdown: through the clear
    circle, so on the landing zone, not beside it (the operator's ruling of
    2026-09-23, "Within the whole clearing": however wide the clearing). Said, never closing or
    regrading anything. Exact ties go to the name, then the class, so the
    same ground reads the same way twice."""
    from shapely.geometry import Point

    if tree is None:
        return None
    indices, distances = tree.query_nearest(
        Point(x, y), max_distance=notice_m / scale, return_distance=True,
        all_matches=True)
    if len(indices) == 0:
        return None
    ranked = sorted(
        (float(distance) * scale,
         str((roads[int(index)].get("properties") or {}).get("name") or ""),
         str((roads[int(index)].get("properties") or {}).get("route_type") or ""),
         int(index))
        for index, distance in zip(indices, distances))
    distance_m, _name, _class, index = ranked[0]
    properties = roads[index].get("properties") or {}
    distance_m = round(distance_m, 1)
    return {"name": properties.get("name"),
            "route_type": properties.get("route_type"),
            # Absent before query version 5: unknown, and said as null.
            "highway": properties.get("highway"),
            "surface": properties.get("surface"),
            "distance_m": distance_m,
            "on_road": distance_m <= clear_radius_m}


def _parking_lots(payload: dict) -> list[tuple[str, Any]]:
    """The staged parking lots with an outline, as (id, EPSG:3857 shape).
    A lot mapped only as a node has no inside to stand in, and public
    hardstand - a school yard, a park - is not a parking lot."""
    from shapely.geometry import shape as shapely_shape

    from ..terrain.linear import SITE_PARKING

    lots = []
    for site in payload.get("parking") or []:
        if site.get("kind") != SITE_PARKING or not site.get("geometry"):
            continue
        geometry = shapely_shape(site["geometry"])
        if geometry.geom_type in ("Polygon", "MultiPolygon") and not geometry.is_empty:
            lots.append((site.get("id"), geometry))
    return lots


def _parking_lot_at(x: float, y: float, lots: list[tuple[str, Any]], named_ways: list[dict],
                    ways_tree, *, scale: float) -> dict | None:
    """{id, off} for the first lot whose outline covers the touchdown, where
    `off` is the named way the staging-area tab names the same lot after
    (`parking._nearest_named_way`), null when none is near; None when the
    touchdown is in no lot."""
    from shapely.geometry import Point

    from .parking import _nearest_named_way

    touchdown = Point(x, y)
    for lot_id, geometry in lots:
        if geometry.covers(touchdown):
            nearest = _nearest_named_way(geometry, named_ways, ways_tree, scale)
            return {"id": lot_id, "off": nearest["name"] if nearest else None}
    return None


def _chip_key(entry: dict) -> str:
    """The first 8 hex of sha256 over the WGS84 touchdown.

    Keyed on the touchdown's own ground, not on the candidate's rank: a
    later call on the same staged ground - a different centre, a different
    cap - assigns ranks afresh, and a name keyed on rank alone (lz_1.png)
    would have a new lz_1 silently overwrite an earlier run's. Rank still
    lives in the envelope's `id`, so nothing is lost by leaving it out of
    the filename."""
    import hashlib

    longitude, latitude = entry["touchdown"]["coordinates"]
    return hashlib.sha256(f"{longitude:.6f},{latitude:.6f}".encode()).hexdigest()[:8]


def _chip_picture(source: str, *, entry: dict, scale: float,
                  wms_base_url: str | None, cache_dir: Path | None, fetch,
                  get_static, mapbox_token: str, google_maps_key: str):
    """(picture, bbox_3857): `source`'s image around the touchdown, not yet
    outlined, and the exact Web Mercator box it covers.

    A CalTopo source is fetched through the WMS path at a box of `CHIP_M`
    ground metres, and the picture is the panel's path. A static source is
    asked through `sources/static_imagery.py`, which computes the box its answer
    covers - Google's whole zoom makes it only near `CHIP_M` - and the
    picture is its bytes, held in memory and never written: nothing would
    read them back, Mapbox's satellite style answers in JPEG, and Google's
    terms restrict storing its imagery. Only the outlined chip is saved.
    An answer that is not `CHIP_PX` square is refused: the outline is drawn
    at `CHIP_PX` over the box, and would miss the ground."""
    import io

    from PIL import Image

    from ..sources.static_imagery import google_request, mapbox_request

    x, y = entry["_x"], entry["_y"]
    layer = {name: wms for name, _label, wms, _credit in CHIP_SOURCES}.get(source)
    if layer is not None:
        half = CHIP_M / 2.0 / scale
        bbox = (x - half, y - half, x + half, y + half)
        return Path(fetch(wms_base_url, [layer], bbox, CHIP_PX, CHIP_PX,
                          Path(cache_dir))), bbox
    latitude = entry["touchdown"]["coordinates"][1]
    if source == "mapbox-satellite":
        url, bbox = mapbox_request(x, y, latitude=latitude, size_m=CHIP_M, px=CHIP_PX,
                                   token=mapbox_token)
    else:
        url, bbox = google_request(x, y, latitude=latitude, size_m=CHIP_M, px=CHIP_PX,
                                   key=google_maps_key)
    picture = io.BytesIO(get_static(url))
    with Image.open(picture) as image:
        if image.size != (CHIP_PX, CHIP_PX):
            raise ValueError("the answer was not the size asked for")
    picture.seek(0)
    return picture, bbox


def _cut_chip(source: str, *, directory: Path, entry: dict, scale: float,
              wms_base_url: str | None, cache_dir: Path | None, fetch, get_static,
              mapbox_token: str, google_maps_key: str) -> Path:
    """`source`'s picture around the touchdown with the patch outline drawn
    on it in `WAY_RGB` (cyan) - natural-colour imagery is full of green, not
    cyan - saved as a PNG at `<directory>/lz/<key>-<source>.png`. Drawn in
    the measuring frame (3857) with the picture's own box, the way
    `rasters._outline_patches` documents - inverting the chip's own affine
    on degrees would put every outline in one corner."""
    from affine import Affine
    from shapely.geometry import mapping

    # package-internal helpers shared by planning modules
    from .rasters import WAY_RGB, _outline_patches

    picture, (west, south, east, north) = _chip_picture(
        source, entry=entry, scale=scale, wms_base_url=wms_base_url,
        cache_dir=cache_dir, fetch=fetch, get_static=get_static,
        mapbox_token=mapbox_token, google_maps_key=google_maps_key)
    unit_x = (east - west) / CHIP_PX
    unit_y = (north - south) / CHIP_PX
    transform = Affine(unit_x, 0.0, west, 0.0, -unit_y, north)
    destination = directory / "lz" / f"{_chip_key(entry)}-{source}.png"
    return _outline_patches(picture, [mapping(entry["_polygon_3857"])],
                            transform, destination, colour=WAY_RGB)


def _source_refused(source: str, error: BaseException) -> bool:
    """Whether `error` from `source` ends that source for the rest of the
    call: only a static source, and only a 401 or 403 (a key that will not
    work for the next candidate either) or a transport error (a host that
    is not there). `get_static` re-raises both as the same httpx types,
    stripped of the URL, so the test is on the type."""
    import httpx

    if source not in {name for name, _label, _credit in STATIC_SOURCES}:
        return False
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in REFUSED_STATUSES
    return isinstance(error, httpx.TransportError)


def check_params(*, max_slope_deg: float = 5.0, marginal_slope_deg: float = 7.0,
                 limit: int | None = 10, powerline_notice_m: float = 150.0,
                 powerline_clearance_m: float = 60.0,
                 structure_clearance_m: float = 30.0) -> None:
    """The parameter checks `lz_candidates` makes once it knows the ground
    is fine enough to grade, before it reads a raster: a ValueError naming
    the first one out of range. Public, and needing no ground, so a scenario
    refuses a bad override in these words before it stages any - not as a
    failed section after minutes of staging."""
    if max_slope_deg > marginal_slope_deg:
        raise ValueError(
            f"max_slope_deg ({max_slope_deg}) is above marginal_slope_deg "
            f"({marginal_slope_deg}); the marginal limit is the relaxed one and must "
            "not be tighter than the preferred limit")
    if limit is not None and int(limit) < 0:
        raise ValueError(f"limit ({limit}) is negative; 0 means every candidate")
    if powerline_notice_m <= 0:
        raise ValueError(f"powerline_notice_m ({powerline_notice_m}) must be above "
                         "zero; a wire cannot be noticed from no distance")
    if powerline_clearance_m < 0:
        raise ValueError(f"powerline_clearance_m ({powerline_clearance_m}) is below "
                         "zero; a clearance is a distance kept from a wire, and 0 "
                         "keeps none")
    if structure_clearance_m < 0:
        raise ValueError(f"structure_clearance_m ({structure_clearance_m}) is below "
                         "zero; a clearance is a distance kept from a building, and 0 "
                         "keeps none")


def lz_candidates(directory, *, center_latitude: float, center_longitude: float,
                  limit: int = 10,
                  max_slope_deg: float = 5.0, marginal_slope_deg: float = 7.0,
                  max_canopy: float = 0.20, min_diameter_m: float = 25.0,
                  preferred_width_m: float = 45.0, preferred_length_m: float = 60.0,
                  powerline_notice_m: float = 150.0,
                  powerline_clearance_m: float = 60.0,
                  structure_clearance_m: float = 30.0,
                  whole_ground: bool = False,
                  wms_base_url: str | None = None, cache_dir: Path | None = None,
                  fetch=None, mapbox_token: str = "", google_maps_key: str = "",
                  get_static=None) -> dict[str, Any]:
    """Landing zone candidates on the ground staged under `directory`.

    See the module docstring for the method. `fetch` is the panel fetcher,
    injected so the tests need no network; it defaults to
    `caltopo.wms.fetch_panel`. `get_static` is the static-image getter for
    the keyed sources, injected likewise; it defaults to
    `sources.static_imagery.get_static`. Raises `ValueError` when the raster is too
    coarse to grade, the two slope limits are out of order, `limit` is
    negative, `powerline_notice_m` is not above zero,
    `powerline_clearance_m` or `structure_clearance_m` is below zero, or no
    canopy panel can be had.

    Only cells inside the grid's mask (the planner's circle, or the
    assignment) are graded, unless `whole_ground`: then every staged cell
    is, the planning buffer included, and the true-colour tree check is
    fetched over the whole raster too, so the buffer is held to the same
    checks as the circle.

    Four closures run before any footprint is fitted. Ground within
    `powerline_clearance_m` of any mapped power line - a transmission
    `line` or a distribution `minor_line` - is closed; unknown lines close
    nothing. Ground within `structure_clearance_m` of any footprint on
    CalTopo's `structures` layer (`panels/structures.png`, fetched through
    the WMS path when absent) is closed too. A structures layer that
    cannot be had, or that does not read as one, is a warning and
    `assumptions.structures` "not checked", never a refusal: the layer is
    incomplete even when present, so its absence loses a filter, not the
    grade. `structures: "checked"` means CalTopo's mapped footprints were
    kept clear of, not that there are no buildings - an unmapped one is
    for the chip. A structure clearance of 0 still closes the footprint
    itself; a power-line clearance of 0 closes nothing.

    Trees are closed from true-colour NAIP (`terrain.trees.trees_on_grid`): cast
    shadows and the green crowns beside them, cell for cell with no speck
    forgiveness. Without a WMS token or when the fetch fails it is a
    warning and `assumptions.trees` "not checked", never a refusal.

    Open water is closed too (`_closed_water`): LANDFIRE open water from
    `evt.tif` and OpenStreetMap shorelines, a closed ring filled. A lake is
    flat and reads as unvegetated, so without this it grades as the best
    zone on the map. `assumptions.water_sources_checked` names the sources
    that could be read; with neither, a warning says water was not checked.

    `powerline` per candidate is a distance, name and `power` (`line` or
    `minor_line`), `null` for nothing mapped within `powerline_notice_m`
    of the touchdown, or `"unknown"` when OpenStreetMap could not be read
    or was staged before wires were labelled; `null` is never "no wire".
    The distance is measured from the touchdown; the patch's edge may be as
    close as `powerline_clearance_m`. Two lines at exactly the same
    distance report the first in tree order.

    `road` per candidate is the nearest mapped ROAD, TRACK or ROAD_OTHER
    within `powerline_notice_m` of the touchdown, as {name, route_type,
    highway, surface, distance_m, on_road}; `on_road` is the way coming
    within half `clear_diameter_m` of the touchdown, through the clear
    circle. A way not built (proposed, under construction...) or not a road
    anyone means (a via ferrata, a bus guideway) is left out, as egress
    leaves it out; a motorway is kept. `null` is nothing mapped that close;
    `"unknown"` is no osm.json. `highway` is null on ground staged before
    query version 5, and `assumptions.road_class` says "unknown" there,
    "checked" from version 5. `parking_lot` is {id, off} when the touchdown
    lies inside a mapped parking lot's outline (not public hardstand, not a
    lot mapped only as a point), `off` naming the road it opens onto as
    `parking_candidates` does, else null; `"unknown"` when there is no
    osm.json or it was staged before parking was fetched (query version 3).
    Both are said, never closing, flagging or regrading ground. On ground
    staged before version 5 no road carries `highway`, so an unbuilt road
    cannot be told from a built one there and is named like any other.

    `chips` is a list of 200 m imagery squares around the touchdown with
    the patch outlined, for a human or a vision model to look at, one
    `{source, label, path, attribution}` per source that could be had, in
    a fixed order: CalTopo imagery and CalTopo NAIP when there is a WMS
    base URL and a cache directory, then Mapbox when `mapbox_token` is
    given, then Google when `google_maps_key` is. `attribution` is the
    credit a page showing that chip must show beside it. A source that
    fails is left out with a warning naming the candidate, the source and
    the failure's type and status - never its message, which can carry a
    URL and the key in it. A static source that answers 401 or 403, or
    cannot be reached, is not asked again this call: one warning says so
    in the envelope, and every candidate after it carries it too. A
    CalTopo failure is usually one tile and stays per candidate. A failed
    chip never fails the call. `chip_path` is the first chip's path, null
    when there is none. Each file is named by the touchdown and the source,
    not by rank.
    """
    from pyproj import Geod, Transformer
    from shapely import STRtree
    from shapely.geometry import shape as shapely_shape

    from ..geo import to_wgs84_geojson
    from ..terrain.grid import load_assignment_grid
    from ..sources.osm import OSM_QUERY_VERSION_PARKING, OSM_QUERY_VERSION_ROUTE_TAGS
    from ..terrain.vegetation import VEGETATION_PROXY_THRESHOLD
    from .parking import _named_ways, _nearest_way_index

    directory = Path(directory)
    if fetch is None:
        from ..caltopo.wms import fetch_panel as fetch
    if get_static is None:
        from ..sources.static_imagery import get_static
    grid = load_assignment_grid(directory)
    pixel_m = float(grid.pixel_size_m)
    if pixel_m > MAX_PIXEL_M:
        raise ValueError(TOO_COARSE_ERROR.format(pixel=pixel_m, limit=MAX_PIXEL_M))
    check_params(max_slope_deg=max_slope_deg, marginal_slope_deg=marginal_slope_deg,
                 limit=limit, powerline_notice_m=powerline_notice_m,
                 powerline_clearance_m=powerline_clearance_m,
                 structure_clearance_m=structure_clearance_m)
    warnings: list[str] = []

    # ground metres per Mercator unit; read by the power-line distance and
    # the imagery chip
    scale = pixel_m / abs(grid.transform.a)
    osm = _osm_payload(directory)
    lines, osm_warning, powerline_kinds_checked = _power_lines(osm)
    if osm_warning:
        warnings.append(osm_warning)
    tree = STRtree([shapely_shape(line["geometry"]) for line in lines]) if lines else None
    # The road fact and the lot: said, never closing or regrading ground.
    # Roads were fetched by every query version, so only a missing file
    # makes them unknown; road CLASS (`highway`) arrived with version 5.
    roads = _roads(osm) if osm is not None else None
    road_tree = STRtree([shapely_shape(road["geometry"]) for road in roads]) if roads else None
    # Parking arrived with query version 3: without the file, or on ground
    # staged before it, a touchdown in a lot cannot be told from one in no
    # lot, so the lot is unknown, never null.
    lots_known = (osm is not None
                  and int(osm.get("query_version", 1)) >= OSM_QUERY_VERSION_PARKING)
    lots = _parking_lots(osm) if lots_known else []
    named_ways = _named_ways(osm.get("routes") or []) if lots else []
    ways_tree = _nearest_way_index(named_ways) if lots else None
    road_class = ("checked" if osm is not None
                  and int(osm.get("query_version", 1)) >= OSM_QUERY_VERSION_ROUTE_TAGS
                  else "unknown")

    panel = _canopy_panel(directory, grid, wms_base_url=wms_base_url,
                          cache_dir=cache_dir, fetch=fetch)
    # The planner's circle (or the assignment), or - `whole_ground` - every
    # cell staged, its planning buffer included: a medevac takes the nearest
    # place a helicopter can land, wherever the circle happened to end.
    region = np.ones(grid.mask.shape, dtype=bool) if whole_ground else grid.mask
    masks = _masks(grid, panel, region=region, max_slope_deg=max_slope_deg,
                   marginal_slope_deg=marginal_slope_deg)
    if lines is not None:
        # Closed before fitting, not filtered after: a clearing split by a
        # wire is two smaller clearings, and each is graded on what is left
        # of it.
        masks = masks._replace(open_ok=masks.open_ok & ~_closed_near_lines(
            grid, lines, powerline_clearance_m))
    structures_panel, structures_why = _structures_panel(
        directory, grid, wms_base_url=wms_base_url, cache_dir=cache_dir, fetch=fetch)
    structures_closed = None
    if structures_panel is not None:
        unreadable = None
        try:
            structures_closed, why = _closed_near_structures(grid, structures_panel,
                                                             structure_clearance_m)
        except (OSError, ValueError, SyntaxError) as error:
            # Pillow raises all three on a file it cannot read: an
            # UnidentifiedImageError (an OSError), a truncated image, a
            # broken chunk. Not a failed call: a lost filter, like any
            # structures layer that cannot be had.
            unreadable = safe_reason(error)
        if unreadable is not None:
            structures_closed = None
            # Removed, so the next call fetches it afresh rather than failing
            # on the same file for the life of the directory. Nothing else
            # can read it either.
            structures_panel.unlink(missing_ok=True)
            # No inner brackets (the reason is already inside one), and no
            # promise of a refetch: without a token the next call cannot.
            why = (f"the panel could not be read, {unreadable}, and was removed "
                   "so a later call can fetch it afresh")
        if why:
            structures_why = f"{STRUCTURES_NOT_CHECKED} ({why})"
    if structures_closed is not None:
        masks = masks._replace(open_ok=masks.open_ok & ~structures_closed)
    else:
        warnings.append(structures_why)
    # Water is flat and reads as unvegetated, so without this a lake is
    # the best-graded zone on the map. Closed, not filtered: a meadow on a
    # lakeshore is graded on its dry ground.
    water_closed, water_sources, water_warnings = _closed_water(directory, grid, osm)
    masks = masks._replace(open_ok=masks.open_ok & ~water_closed)
    warnings.extend(water_warnings)
    # Trees from true-colour imagery (`terrain.trees`): the CIR proxy misses
    # dark conifers, and its speck rule fits over every single tree. Any
    # tree or tree shadow in a cell closes it, with no speck forgiveness:
    # one pine in a clear circle is one obstacle.
    trees_checked = False
    trees_why = ("no WMS token" if not wms_base_url
                 else "no cache directory" if cache_dir is None else None)
    if trees_why is None:
        try:
            tree_share = trees_on_grid(grid, wms_base_url=wms_base_url,
                                       cache_dir=cache_dir, fetch=fetch, within=region)
        except Exception as error:  # noqa: BLE001 - a lost filter, never a lost grade
            # `safe_reason`, never the message: a GetMap URL carries the token.
            trees_why = safe_reason(error)
        else:
            masks = masks._replace(open_ok=masks.open_ok & ~(tree_share > 0))
            trees_checked = True
    if trees_why is not None:
        warnings.append(TREES_NOT_CHECKED.format(reason=trees_why))

    graded = _grade_patches(open_ok=masks.open_ok, flat_preferred=masks.flat_preferred,
                            flat_marginal=masks.flat_marginal, vegetated=masks.vegetated,
                            max_canopy=max_canopy,
                            pixel_m=pixel_m, min_diameter_m=min_diameter_m,
                            preferred_width_m=preferred_width_m,
                            preferred_length_m=preferred_length_m,
                            transform=grid.transform)

    to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    geod = Geod(ellps="WGS84")

    entries: list[dict[str, Any]] = []
    for item in graded:
        row, col = item["row"], item["col"]
        rows_, cols_, disc = item["footprint"]
        x, y = grid.transform * (col + 0.5, row + 0.5)
        longitude, latitude = to_wgs84.transform(x, y)
        bearing, _back, distance_m = geod.inv(center_longitude, center_latitude,
                                              longitude, latitude)
        # a bearing to the cell you are standing on is not a fact
        bearing_deg = None if distance_m < pixel_m else round(float(bearing % 360.0), 1)
        footprint_slope = masks.slope_aspect.slope_deg[rows_, cols_][disc]
        footprint_aspect = masks.slope_aspect.aspect_deg[rows_, cols_][disc]
        slopes = footprint_slope[np.isfinite(footprint_slope)]
        if lines is None:
            powerline = "unknown"
        else:
            powerline = _nearest_power_line(x, y, lines, tree, scale=scale,
                                            notice_m=powerline_notice_m)
        clear_diameter_m = round(2.0 * item["clear_radius_m"], 1)
        if roads is None:
            road = "unknown"
        else:
            # Against the diameter as reported, so a reader halving it gets
            # the same answer.
            road = _nearest_road(x, y, roads, road_tree, scale=scale,
                                 notice_m=powerline_notice_m,
                                 clear_radius_m=clear_diameter_m / 2.0)
        entries.append({
            "grade": item["grade"],
            "touchdown": {"type": "Point", "coordinates": [longitude, latitude]},
            "clear_diameter_m": clear_diameter_m,
            "preferred_fit": item["preferred_fit"],
            "slope_mean_deg": round(float(slopes.mean()), 1) if slopes.size else None,
            "slope_max_deg": round(float(slopes.max()), 1) if slopes.size else None,
            "canopy_fraction": round(item["canopy"], 3),
            "elevation_m": round(float(grid.elevation[row, col]), 1),
            "aspect": _aspect_of(footprint_slope, footprint_aspect),
            "distance_m": round(float(distance_m), 1),
            "bearing_deg": bearing_deg,
            "powerline": powerline,
            "road": road,
            "parking_lot": (_parking_lot_at(x, y, lots, named_ways, ways_tree, scale=scale)
                            if lots_known else "unknown"),
            "geometry": to_wgs84_geojson(item["polygon_3857"]),
            "_x": x, "_y": y, "_polygon_3857": item["polygon_3857"],
            "warnings": [],
        })

    # Distance, then grade (best first), then clear diameter: two openings
    # the same distance out keep one order between runs, so a row number
    # means the same thing to a second planner.
    entries.sort(key=lambda entry: (entry["distance_m"], GRADES.index(entry["grade"]),
                                    -entry["clear_diameter_m"]))
    for index, entry in enumerate(entries, start=1):
        entry["id"] = f"lz_{index}"

    grades = {grade: sum(1 for entry in entries if entry["grade"] == grade)
              for grade in GRADES}
    total = len(entries)
    truncated = None
    if limit and int(limit) > 0 and total > int(limit):
        entries = entries[:int(limit)]
        truncated = int(limit)

    # (source, label, attribution), in the fixed order they are cut
    chip_sources: list[tuple[str, str, str]] = []
    if wms_base_url and cache_dir is not None:
        chip_sources += [(source, label, credit)
                         for source, label, _layer, credit in CHIP_SOURCES]
    for static, credential in zip(STATIC_SOURCES, (mapbox_token, google_maps_key)):
        if credential:
            chip_sources.append(static)
    if not chip_sources:
        warnings.append(NO_TOKEN_CHIP_WARNING)
    # A static source that refused - a bad key, or no host - is not asked
    # again this call: source -> the warning that says so, which every
    # later candidate carries too.
    refused: dict[str, str] = {}
    # Cut AFTER the cap, for the candidates a planner will actually see: a
    # chip is a network fetch, and one for a candidate nobody is shown is
    # traffic for nothing.
    for entry in entries:
        entry["chips"] = []
        for source, label, credit in chip_sources:
            if source in refused:
                entry["warnings"].append(refused[source])
                continue
            try:
                path = _cut_chip(source, directory=directory, entry=entry, scale=scale,
                                 wms_base_url=wms_base_url, cache_dir=cache_dir,
                                 fetch=fetch, get_static=get_static,
                                 mapbox_token=mapbox_token,
                                 google_maps_key=google_maps_key)
            except Exception as error:  # noqa: BLE001 - one chip must not sink the set
                # `safe_reason`, never the message: a static source's URL
                # carries its key in the query string.
                if _source_refused(source, error):
                    message = SOURCE_REFUSED_WARNING.format(source=source,
                                                            reason=safe_reason(error))
                    refused[source] = message
                else:
                    message = CHIP_FAILED_WARNING.format(
                        candidate=entry["id"], source=source, reason=safe_reason(error))
                warnings.append(message)
                # on the candidate too: a reader of one row must see why a chip is missing
                entry["warnings"].append(message)
                continue
            entry["chips"].append({"source": source, "label": label, "path": str(path),
                                   "attribution": credit})
        entry["chip_path"] = entry["chips"][0]["path"] if entry["chips"] else None

    for entry in entries:
        for key in [key for key in entry if key.startswith("_")]:
            del entry[key]

    return {
        "candidates": entries,
        "summary": {"count": len(entries), "total_before_limit": total,
                    "truncated_to": truncated, "grades": grades},
        "center": {"latitude": center_latitude, "longitude": center_longitude},
        "resolution_m": round(pixel_m, 3),
        "assumptions": {
            "max_slope_deg": max_slope_deg,
            "marginal_slope_deg": marginal_slope_deg,
            "max_canopy": max_canopy,
            "speck_cells": SPECK_CELLS,
            "canopy_method": CANOPY_METHOD,
            "min_diameter_m": min_diameter_m,
            "preferred_footprint_m": [preferred_width_m, preferred_length_m],
            "footprint_angles_deg": list(FOOTPRINT_ANGLES_DEG),
            "angle_frame": ("preferred_fit.angle_deg is the long axis of the fitted "
                            "rectangle, in degrees anticlockwise from east on a "
                            "north-up map, 0 to 180"),
            "aspect_rule": {"min_slope_deg": ASPECT_MIN_SLOPE_DEG,
                            "min_share": ASPECT_MIN_SHARE,
                            "min_resultant": ASPECT_MIN_RESULTANT,
                            "otherwise": "null: too flat or too mixed to have an aspect"},
            "facts_frame": ("slope, canopy and aspect are measured over the clear "
                            "circle at the touchdown, not the whole patch polygon"),
            "canopy_threshold": VEGETATION_PROXY_THRESHOLD,
            "canopy_layer": CANOPY_LAYER,
            "chip": {"size_m": CHIP_M, "px": CHIP_PX,
                     "sources": [source for source, _label, _credit in chip_sources]},
            "powerline_notice_m": powerline_notice_m,
            "powerline_clearance_m": powerline_clearance_m,
            "powerline_kinds_checked": powerline_kinds_checked,
            "road_class": road_class,
            "structure_clearance_m": structure_clearance_m,
            "structures": "checked" if structures_closed is not None else "not checked",
            "water_sources_checked": water_sources,
            "trees": "checked" if trees_checked else "not checked",
            "trees_method": TREES_METHOD,
            "searched": ("the whole staged ground, its planning buffer included"
                         if whole_ground else "inside the planner's circle or assignment"),
            "grading": ("a filter on elevation and imagery rasters, not a site "
                        "survey; be conservative, and the pilot decides"),
        },
        "warnings": warnings,
    }
