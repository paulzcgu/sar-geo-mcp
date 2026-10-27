"""Reading staged LANDFIRE vegetation against other ground.

The vegetation raster itself comes from `sources.landfire`; this module
matches its codes to elevation cells and summarises an area's vegetation,
which is what the walking model, site facts and sweep width read.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np

from ..sources.landfire import load_lifeform_table


@lru_cache(maxsize=2)
def _read_evt_keyed(path_str: str, mtime_ns: int, size: int):
    """The decoded raster for one (path, mtime, size). See `_read_evt`."""
    import rasterio

    with rasterio.open(path_str) as dataset:
        codes = dataset.read(1)
        transform = dataset.transform
    codes.setflags(write=False)   # the array is shared between callers
    return codes, transform


def _read_evt(path: Path):
    """`(codes, transform)` for a staged evt.tif, read once per file version.

    A hasty set asks for the composition of every corridor candidate against
    the SAME staged raster, and decoding it per candidate is the whole cost of
    the second one onward. The key carries the file's mtime and size, so a
    re-staged directory is never served the raster it replaced. Two entries is
    enough for the one directory a call works in, plus the one before it.
    """
    stat = Path(path).stat()
    return _read_evt_keyed(str(path), stat.st_mtime_ns, stat.st_size)


_read_evt.cache_clear = _read_evt_keyed.cache_clear


def codes_at_cells(evt_path: Path, grid_transform, rows, cols) -> np.ndarray:
    """LANDFIRE codes at the centres of grid cells (rows, cols) of another
    raster - the elevation model's - or -1 for a cell off the EVT raster.

    The one place a DEM cell is matched to its 30 m EVT pixel. The two
    rasters differ in transform and resolution (the DEM can be 1 m), so each
    cell centre is projected into EVT row/col space rather than either raster
    being resampled: exact, and cheaper than warping. `site_facts` reads it
    for one footprint's cells, the walking ground for every cell.
    """
    codes, evt_transform = _read_evt(evt_path)
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    xs = grid_transform.c + (cols + 0.5) * grid_transform.a
    ys = grid_transform.f + (rows + 0.5) * grid_transform.e
    evt_cols = np.floor((xs - evt_transform.c) / evt_transform.a).astype(np.int64)
    evt_rows = np.floor((ys - evt_transform.f) / evt_transform.e).astype(np.int64)
    inside = ((evt_rows >= 0) & (evt_rows < codes.shape[0])
              & (evt_cols >= 0) & (evt_cols < codes.shape[1]))
    out = np.full(rows.shape, -1, dtype=np.int64)
    out[inside] = codes[evt_rows[inside], evt_cols[inside]]
    return out


def composition(evt_path: Path, geometry_3857) -> dict:
    """Lifeform fractions and top vegetation types inside the assignment."""
    from rasterio.features import rasterize

    table = load_lifeform_table()

    codes, transform = _read_evt(evt_path)
    mask = rasterize(
        [(geometry_3857, 1)],
        out_shape=codes.shape,
        transform=transform,
        fill=0,
        all_touched=True,
    ).astype(bool)

    selected = codes[mask] if mask.any() else codes.ravel()

    lifeform_counts: dict[str, int] = {}
    name_counts: dict[str, int] = {}
    for code, count in zip(*np.unique(selected, return_counts=True)):
        lifeform, _, name = table.get(int(code), ("Unknown", "Unknown", f"code {code}"))
        lifeform_counts[lifeform] = lifeform_counts.get(lifeform, 0) + int(count)
        name_counts[name] = name_counts.get(name, 0) + int(count)

    total = sum(lifeform_counts.values())
    unknown_fraction = lifeform_counts.get("Unknown", 0) / max(total, 1)
    fractions = {lf: count / total for lf, count in lifeform_counts.items()}
    top_types = sorted(
        ((name, count / total) for name, count in name_counts.items()),
        key=lambda item: -item[1],
    )[:5]

    return {
        "fractions": fractions,
        "dominant_lifeform": max(fractions, key=fractions.get),
        "top_types": top_types,
        "unknown_fraction": unknown_fraction,
        "source": "LANDFIRE LF2024 EVT, 30 m",
    }
