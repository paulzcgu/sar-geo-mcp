"""Load a fetch output directory into arrays ready for terrain math.

Everything downstream computes over (array, mask) pairs produced here and
never re-reads files. The mask is the assignment geometry rasterized onto the
DEM grid; statistics are computed inside it, context stays outside it.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyproj import Transformer
from shapely.geometry import shape as shapely_shape
from shapely.ops import transform as shapely_transform

from ..geo import ground_meters_per_mercator_unit

# Line assignments (roads, drainages) are searched as a corridor. Ground metres.
CORRIDOR_HALF_WIDTH_M = 50.0

_WGS84_TO_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


@dataclass(frozen=True)
class AssignmentGrid:
    """The DEM and assignment mask on a shared pixel grid."""

    elevation: np.ndarray  # float32, NaN where nodata
    transform: object  # affine.Affine
    pixel_size_m: float  # ground metres per pixel
    latitude_deg: float
    mask: np.ndarray  # bool, True inside the assignment (or corridor)
    geometry_3857: object  # shapely geometry, Web Mercator
    area_m2: float  # ground area of the assignment


def project_geometry_to_3857(geometry_wgs84: dict):
    """Project GeoJSON to Web Mercator; buffer lines into search corridors.

    The corridor half-width is a ground distance, so it is inflated to
    Mercator units at the geometry's own latitude before buffering.
    """
    geom = shapely_shape(geometry_wgs84)
    projected = shapely_transform(_WGS84_TO_3857.transform, geom)

    if projected.geom_type in ("LineString", "MultiLineString"):
        latitude = geom.centroid.y
        scale = 1.0 / math.cos(math.radians(latitude))
        projected = projected.buffer(CORRIDOR_HALF_WIDTH_M * scale)

    return projected


#: How to get a staged directory back. Every error below names the FILE and
#: this, never the directory: a staged directory's path is a local path, and
#: these errors reach tool callers verbatim.
_RESTAGE = ("stage it again - prepare_area or prepare_terrain - "
            "since this step only reads staged ground and never downloads")


def _staged_file_error(name: str, error: BaseException) -> OSError:
    """A path-free error for a staged file that is missing or unreadable:
    the file's name and, for an unreadable one, the failure's TYPE
    (`net_errors.safe_reason`) - never the message, which names the path."""
    from ..net_errors import safe_reason

    if isinstance(error, FileNotFoundError):
        return FileNotFoundError(f"{name} is missing from the staged ground; {_RESTAGE}")
    return OSError(f"{name} could not be read from the staged ground "
                   f"({safe_reason(error)}); {_RESTAGE}")


def _read_staged_json(data_dir: Path, name: str) -> dict:
    """A staged JSON file, or a path-free error. The error is raised AFTER
    the except block, so it carries neither a cause nor a context that
    could hold the original, path-bearing one."""
    try:
        return json.loads((data_dir / name).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        failure = _staged_file_error(name, error)
    raise failure


def read_subject(data_dir: Path) -> dict:
    """The staged subject: subject.json, or assignment.json for directories
    staged before subject.json existed."""
    data_dir = Path(data_dir)
    name = "subject.json"
    if not (data_dir / name).exists():
        name = "assignment.json"
        if not (data_dir / name).exists():
            raise FileNotFoundError(f"subject.json is missing from the staged ground; "
                                    f"{_RESTAGE}")
    return _read_staged_json(data_dir, name)


def load_assignment_grid(data_dir: Path) -> AssignmentGrid:
    """Read dem.tif, manifest.json, and subject.json (or assignment.json) into
    an AssignmentGrid. A missing or unreadable file is refused naming the
    file, never its path (see `_staged_file_error`)."""
    import rasterio
    from rasterio.errors import RasterioError
    from rasterio.features import rasterize

    data_dir = Path(data_dir)
    dem_path = data_dir / "dem.tif"
    if not dem_path.exists():
        raise FileNotFoundError(f"dem.tif is missing from the staged ground; {_RESTAGE}")

    manifest = _read_staged_json(data_dir, "manifest.json")
    assignment = read_subject(data_dir)
    latitude = float(manifest["latitude_deg"])

    try:
        with rasterio.open(dem_path) as dataset:
            elevation = dataset.read(1).astype("float32")
            transform = dataset.transform
            nodata = dataset.nodata
    except (OSError, RasterioError) as error:
        failure = _staged_file_error("dem.tif", error)
    else:
        failure = None
    if failure is not None:
        raise failure

    if nodata is not None:
        elevation[elevation == nodata] = np.nan
    # 3DEP occasionally uses very large negative fills without declaring nodata.
    elevation[elevation < -10000] = np.nan

    ground_per_unit = ground_meters_per_mercator_unit(latitude)
    pixel_size_m = abs(transform.a) * ground_per_unit

    geometry_3857 = project_geometry_to_3857(assignment["geometry"])
    mask = rasterize(
        [(geometry_3857, 1)],
        out_shape=elevation.shape,
        transform=transform,
        fill=0,
        all_touched=True,
    ).astype(bool)

    area_m2 = geometry_3857.area * ground_per_unit**2

    return AssignmentGrid(
        elevation=elevation,
        transform=transform,
        pixel_size_m=pixel_size_m,
        latitude_deg=latitude,
        mask=mask,
        geometry_3857=geometry_3857,
        area_m2=area_m2,
    )
