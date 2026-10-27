"""Questions about one named shape on a staged assignment.

WHY THIS IS HERE AND NOT IN THE APP
-----------------------------------
These two lived in the debrief app's analysis agent, wrapped as LLM tools.
They are geo computation - rasterising a polygon over an elevation grid,
sampling a channel centreline - and geo computation belongs in this
package. Leaving them in the app meant only that app's own model could ask
"how steep is that sidewall", and that anyone else wanting the same answer
would write it a second time.

The app keeps a six-line wrapper that supplies the staged directory from
its own dependencies. That is the right split: the model should never be
handling filesystem paths, and the maths should never be living in an
adapter.

THE INPUT IS A STAGED DIRECTORY
-------------------------------
Everything here reads the fetch directory this package already defines -
`manifest.json`, `dem.tif`, `subject.json` (or `assignment.json` for older
directories), and the `terrain_metrics.json`
that `metrics.compute_terrain_metrics` produces. Shapes are addressed by
the registry id that appears in those metrics, never by geometry passed in
from outside: the id is what an LLM can see in the metrics it was given,
and what a human can look up afterwards to check the answer.
"""

from __future__ import annotations

import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

METRICS_FILENAME = "terrain_metrics.json"

#: Samples along a drainage centreline. Fifty is enough to show the shape
#: of a channel without returning a wall of numbers a model has to wade
#: through to find the gradient.
DEFAULT_SAMPLES = 50


#: One loader at a time: several callers asking their first question about
#: one directory at once must not each read the DEM.
_GRID_LOCK = threading.Lock()


@lru_cache(maxsize=4)
def _grid_cached(data_dir: str, stamp: tuple | None):
    from .grid import load_assignment_grid

    return load_assignment_grid(Path(data_dir))


def _manifest_stamp(data_dir: Path) -> tuple | None:
    try:
        st = (Path(data_dir) / "manifest.json").stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def _grid_for(data_dir: str):
    """The elevation grid, cached by directory AND by its manifest's stamp.

    Small cache: the analysis agent asks several questions in a row about
    one assignment, and reloading a DEM per question is pure waste. Keyed
    by string because `lru_cache` needs something hashable.

    The stamp is what keeps a re-staged directory from being served stale:
    `prepare_terrain` writes a moved assignment back into the SAME directory,
    overwriting dem.tif, and `stage_terrain` writes manifest.json last - so a
    manifest with a new mtime and size means the raster beside it is new too.
    """
    with _GRID_LOCK:
        return _grid_cached(str(Path(data_dir).resolve()), _manifest_stamp(Path(data_dir)))


# Callers and tests reach for these on `_grid_for` itself.
_grid_for.cache_clear = _grid_cached.cache_clear
_grid_for.cache_info = _grid_cached.cache_info


def load_metrics(data_dir: Path):
    from .metrics import TerrainMetrics

    path = Path(data_dir) / METRICS_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Terrain metrics have to be computed for this "
            "assignment before its shapes can be interrogated: call the "
            "`terrain_metrics` tool on this handle first."
        )
    return TerrainMetrics.model_validate_json(path.read_text(encoding="utf-8"))


def registry_geometry(data_dir: Path, shape_id: str) -> dict:
    """One registry shape's geometry, by the id the metrics gave it.

    A missing id raises with the available ones named. A model that
    hallucinates a shape id should be told what actually exists, not handed
    an empty result it will read as "nothing there".
    """
    metrics = load_metrics(Path(data_dir))
    for shape in metrics.shapes:
        if shape.id == shape_id:
            return shape.geometry
    available = ", ".join(sorted(shape.id for shape in metrics.shapes)) or "(none)"
    raise KeyError(f"no shape {shape_id!r} in the registry. Available: {available}")


def shape_slope_statistics(data_dir: Path, shape_id: str) -> dict:
    """Slope statistics inside one registry shape - 'quantify that sidewall'."""
    from rasterio.features import rasterize
    from shapely.geometry import shape as shapely_shape

    from .slope import compute_slope_aspect

    grid = _grid_for(str(data_dir))
    geometry = shapely_shape(registry_geometry(data_dir, shape_id))

    mask = rasterize(
        [(geometry, 1)], out_shape=grid.elevation.shape,
        transform=grid.transform, fill=0, all_touched=True,
    ).astype(bool)

    slope = compute_slope_aspect(grid.elevation, grid.pixel_size_m).slope_deg
    values = slope[mask]
    values = values[~np.isnan(values)]
    if values.size == 0:
        # A shape that covers no valid pixels is a real answer, not an
        # error - it happens over a DEM void. Nulls rather than zeros,
        # because a zero mean slope reads as flat ground.
        return {"pixels": 0, "mean_deg": None, "p90_deg": None, "max_deg": None}
    return {
        "pixels": int(values.size),
        "mean_deg": float(values.mean()),
        "p90_deg": float(np.percentile(values, 90)),
        "max_deg": float(values.max()),
    }


def drainage_elevation_profile(data_dir: Path, drainage_id: str,
                               samples: int = DEFAULT_SAMPLES) -> dict:
    """Elevation along a drainage centreline - 'how steep is the channel'."""
    from shapely.geometry import shape as shapely_shape

    grid = _grid_for(str(data_dir))
    line = shapely_shape(registry_geometry(data_dir, drainage_id))

    inverse = ~grid.transform
    # The grid is in Web Mercator, whose units are inflated by
    # 1/cos(latitude); distances reported to a human have to be ground
    # metres or they are simply wrong at this latitude by about a quarter.
    ground_per_unit = grid.pixel_size_m / abs(grid.transform.a)
    length_m = line.length * ground_per_unit

    distances: list[float] = []
    elevations: list[float] = []
    for index in range(samples + 1):
        fraction = index / samples
        point = line.interpolate(fraction, normalized=True)
        col, row = inverse * (point.x, point.y)
        row_i, col_i = int(row), int(col)
        if not (0 <= row_i < grid.elevation.shape[0]
                and 0 <= col_i < grid.elevation.shape[1]):
            continue
        value = grid.elevation[row_i, col_i]
        if not np.isfinite(value):
            continue
        distances.append(round(fraction * length_m, 1))
        elevations.append(float(value))

    return {"distances_m": distances, "elevations_m": elevations}
