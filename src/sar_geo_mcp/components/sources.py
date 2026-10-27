"""Components: pull a data source's own data directly.

The planning tools measure; these hand over what one external source holds
for an area, as the source gives it - an elevation grid, a vegetation map,
a map panel, OpenStreetMap ways with their tags, agency road records,
hospitals. Raw data carries none of the checks a planning tool makes
(closures, unknown-is-null, what was not looked at), and every answer says
so: an empty OpenStreetMap answer is not "no trail here".
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal

from ..context import ServerContext, mcp_tool
from ..geo import ground_meters_per_mercator_unit, require_point
from ..sources import (
    Area,
    FeatureSource,
    RasterSource,
    SourceUnavailable,
    source_classes,
)

#: The largest circle a pull covers: what `prepare_area` stages.
MAX_RADIUS_M = 4500.0
#: The most pixels a pulled raster has along either side.
MAX_RASTER_PX = 2048
#: The most features an answer carries inline; the file holds them all.
INLINE_FEATURES = 200

RAW_DATA_NOTE = ("Raw source data, as the source gives it: no closures, no checks, "
                 "and no unknown-is-null reading. A missing feature is not proof of "
                 "absence. Show the attribution beside anything drawn from it.")

_CLASSES = source_classes()
RASTER_NAMES = tuple(name for name, cls in _CLASSES.items()
                     if issubclass(cls, RasterSource) and cls.info.pullable)
FEATURE_NAMES = tuple(name for name, cls in _CLASSES.items()
                      if issubclass(cls, FeatureSource) and cls.info.pullable)
RasterName = Literal[RASTER_NAMES]  # type: ignore[valid-type]
FeatureName = Literal[FEATURE_NAMES]  # type: ignore[valid-type]


def _area(ctx: ServerContext, handle: str | None, center_latitude: float | None,
          center_longitude: float | None, radius_m: float | None) -> tuple[Area, dict]:
    """The area asked for, and how it was given. Exactly one of a handle or
    a centre and radius."""
    by_point = (center_latitude, center_longitude, radius_m)
    if handle is not None:
        if any(value is not None for value in by_point):
            raise ValueError("give a handle, or a centre and radius_m, not both")
        resolved = ctx.staged(handle)
        manifest = json.loads((resolved.directory / "manifest.json").read_text(encoding="utf-8"))
        return Area(tuple(manifest["bbox_3857"])), {"handle": handle,
                                                    "staleness": resolved.status}
    if any(value is None for value in by_point):
        raise ValueError("give a handle, or center_latitude, center_longitude and radius_m")
    require_point(center_latitude, center_longitude)
    if not (isinstance(radius_m, (int, float)) and math.isfinite(radius_m)
            and 0 < radius_m <= MAX_RADIUS_M):
        raise ValueError(f"radius_m must be above 0 and at most {MAX_RADIUS_M:.0f}; got {radius_m}")
    return (Area.around(center_latitude, center_longitude, float(radius_m)),
            {"center": {"latitude": center_latitude, "longitude": center_longitude},
             "radius_m": float(radius_m)})


def _source(ctx: ServerContext, name: str, names: tuple[str, ...], kind: str):
    if name not in names:
        raise ValueError(f"{name!r} is not a {kind} source that can be pulled; "
                         f"choose from {', '.join(names)}")
    if not ctx.config.cache_dir:
        raise ValueError("caching is disabled, so a pulled file has nowhere to go")
    source = _CLASSES[name](ctx.config.cache_dir)
    readiness = source.readiness()
    if not readiness.ready:
        raise ValueError(f"{name} is not ready: it {readiness.reason}")
    return source


def _raster_summary(name: str, path: Path) -> dict[str, Any]:
    """A few numbers about the image, never its pixels."""
    if path.suffix.lower() not in (".tif", ".tiff"):
        return {"format": path.suffix.lower().lstrip(".")}
    import numpy as np
    import rasterio

    with rasterio.open(path) as dataset:
        band = dataset.read(1, masked=True)
        summary: dict[str, Any] = {"format": "geotiff", "bands": dataset.count,
                                   "dtype": dataset.dtypes[0]}
    values = band.compressed()
    if values.size == 0:
        return {**summary, "values": None}
    if name == "landfire_evt":
        from ..sources.landfire import load_lifeform_table

        table = load_lifeform_table()
        codes, counts = np.unique(values, return_counts=True)
        order = np.argsort(-counts)[:10]
        summary["classes"] = [
            {"code": int(codes[i]),
             "lifeform": table.get(int(codes[i]), ("Unknown", "", ""))[0],
             "name": table.get(int(codes[i]), ("", "", f"code {int(codes[i])}"))[2],
             "share": round(float(counts[i]) / values.size, 4)} for i in order]
    else:
        summary["band_1"] = {"min": round(float(values.min()), 2),
                             "max": round(float(values.max()), 2),
                             "mean": round(float(values.mean()), 2)}
    return summary


def list_data_sources(ctx: ServerContext) -> dict[str, Any]:
    """The `list_data_sources` tool as a plain function: `ctx`; the tool's
    payload back. What each field means is in the tool's description, in
    `register` below."""
    sources = []
    for name, cls in _CLASSES.items():
        described = cls().describe()
        described["pull_with"] = ("fetch_raster" if name in RASTER_NAMES else
                                  "query_features" if name in FEATURE_NAMES else None)
        sources.append(described)
    return {"sources": sources, "note": RAW_DATA_NOTE}


def fetch_raster(ctx: ServerContext, source: RasterName, handle: str | None = None,
                 center_latitude: float | None = None,
                 center_longitude: float | None = None, radius_m: float | None = None,
                 pixel_m: float | None = None, layer: str | None = None) -> dict[str, Any]:
    """The `fetch_raster` tool as a plain function: `ctx`, then the tool's
    arguments; the tool's payload back. What each argument and field means
    is in the tool's description, in `register` below."""
    raster = _source(ctx, source, RASTER_NAMES, "raster")
    area, asked = _area(ctx, handle, center_latitude, center_longitude, radius_m)
    if layer is not None and source != "caltopo_wms":
        raise ValueError(f"layer is for caltopo_wms only; {source} has one layer")
    if source == "caltopo_wms" and layer is None:
        raise ValueError("caltopo_wms needs a layer; list_data_sources names them")
    west, south, east, north = area.bbox_3857
    scale = ground_meters_per_mercator_unit(area.latitude)
    pixel = pixel_m if pixel_m is not None else (raster.info.native_resolution_m or 5.0)
    if not (isinstance(pixel, (int, float)) and math.isfinite(pixel) and pixel > 0):
        raise ValueError(f"pixel_m must be above 0; got {pixel_m}")
    width = max(1, min(MAX_RASTER_PX, round((east - west) * scale / pixel)))
    height = max(1, min(MAX_RASTER_PX, round((north - south) * scale / pixel)))
    options = {"layer": layer} if layer is not None else {}
    try:
        result = raster.fetch(area, width, height, **options)
    except SourceUnavailable as error:
        raise ValueError(str(error)) from error
    pixel_actual = (east - west) * scale / result.width_px
    warnings = list(result.warnings)
    if result.width_px == MAX_RASTER_PX or result.height_px == MAX_RASTER_PX:
        warnings.append(f"the raster was capped at {MAX_RASTER_PX} px a side, so its "
                        f"pixel is about {pixel_actual:.1f} m, coarser than asked")
    return {"source": source, "attribution": result.attribution, "area": asked,
            "path": str(result.path), "bbox_3857": list(result.bbox_3857), "crs": result.crs,
            "width_px": result.width_px, "height_px": result.height_px,
            "pixel_m": round(pixel_actual, 2),
            "summary": _raster_summary(source, result.path),
            "warnings": warnings, "note": RAW_DATA_NOTE}


def query_features(ctx: ServerContext, source: FeatureName, handle: str | None = None,
                   center_latitude: float | None = None,
                   center_longitude: float | None = None, radius_m: float | None = None,
                   filters: dict[str, Any] | None = None,
                   limit: int = INLINE_FEATURES) -> dict[str, Any]:
    """The `query_features` tool as a plain function: `ctx`, then the tool's
    arguments; the tool's payload back. What each argument and field means
    is in the tool's description, in `register` below."""
    if not (isinstance(limit, int) and not isinstance(limit, bool) and 0 <= limit <= INLINE_FEATURES):
        raise ValueError(f"limit must be a whole number from 0 to {INLINE_FEATURES}; got {limit}")
    features = _source(ctx, source, FEATURE_NAMES, "feature")
    area, asked = _area(ctx, handle, center_latitude, center_longitude, radius_m)
    try:
        result = features.query(area, filters=filters)
    except SourceUnavailable as error:
        raise ValueError(str(error)) from error
    collection = {"type": "FeatureCollection", "features": result.features,
                  "attribution": result.attribution}
    text = json.dumps(collection)
    name = hashlib.sha256((source + json.dumps(asked, sort_keys=True, default=str)
                           + json.dumps(filters or {}, sort_keys=True)).encode()).hexdigest()[:24]
    path = features.folder() / "pulls" / f"{name}.geojson"
    path.parent.mkdir(parents=True, exist_ok=True)
    from ..cache import atomic_write_text

    atomic_write_text(path, text)
    return {"source": source, "attribution": result.attribution, "area": asked,
            "filters": filters or {}, "total": result.total,
            "returned": min(limit, len(result.features)),
            "features": result.features[:limit], "path": str(path),
            "truncated_by_source": result.truncated,
            "warnings": list(result.warnings), "note": RAW_DATA_NOTE}


def register(mcp, ctx) -> None:

    @mcp_tool(mcp, name="list_data_sources", tags={"component"})
    def _list_data_sources_tool() -> dict[str, Any]:
        """Every external data source this server reads, what it is and whether
        it can be used now.

        Each entry: `name` (what fetch_raster and query_features take), `title`,
        `kind` (raster, features or service), `provider`, `coverage`,
        `attribution` (show it beside anything drawn from the source),
        `native_resolution_m`, `key` (the variable its key is read from, or
        null for none needed - never the key itself), `ready` and
        `not_ready_because`, `pullable` (false: the source's terms keep it
        to the tools that use it, such as Google's to landing-zone chips),
        `shared_service` (a public, rate-limited service: keep requests few),
        `notes` (what it cannot tell you; for caltopo_wms, its layers), and
        `pull_with`: the tool that pulls it, or null for a service (a road
        router, a state lookup) that the planning tools call themselves."""
        return list_data_sources(ctx)

    @mcp_tool(mcp, name="fetch_raster", tags={"component"})
    def _fetch_raster_tool(source: RasterName, handle: str | None = None,
                           center_latitude: float | None = None,
                           center_longitude: float | None = None,
                           radius_m: float | None = None,
                           pixel_m: float | None = None,
                           layer: str | None = None) -> dict[str, Any]:
        """One raster source's image of an area, saved as a file: elevation
        (usgs_3dep), vegetation type (landfire_evt), a CalTopo map layer
        (caltopo_wms, with `layer`, e.g. "sf" slope shading, "imagery",
        "parcel"), NAIP four-band imagery (usgs_naip) or Mapbox satellite
        (mapbox_satellite). RAW DATA: none of the checks a planning tool
        makes - prefer lz_candidates, slope_stats and the like for a
        question they answer.

        The area: a staged `handle`, or `center_latitude`,
        `center_longitude` and `radius_m` (at most 4,500 m). `pixel_m` is
        the ground size of a pixel (default the source's own); a side is
        capped at 2,048 px and a warning says when.

        Returns `path` (a file under the cache; save_export hands it to the
        person), `bbox_3857`, `crs`, `width_px`, `height_px`, `pixel_m`,
        `attribution`, `summary` (never pixels: min, max and mean of band 1;
        for landfire_evt, the ten commonest vegetation classes with their
        shares; for a PNG or JPEG, its format alone), `warnings` and `note`."""
        return fetch_raster(ctx, source=source, handle=handle,
                            center_latitude=center_latitude,
                            center_longitude=center_longitude, radius_m=radius_m,
                            pixel_m=pixel_m, layer=layer)

    @mcp_tool(mcp, name="query_features", tags={"component"})
    def _query_features_tool(source: FeatureName, handle: str | None = None,
                             center_latitude: float | None = None,
                             center_longitude: float | None = None,
                             radius_m: float | None = None,
                             filters: dict[str, Any] | None = None,
                             limit: int = INLINE_FEATURES) -> dict[str, Any]:
        """One feature source's features over an area, as GeoJSON in WGS84:
        OpenStreetMap ways and nodes with their own tags (osm), the land
        agencies' road and trail records (agency_ways), or hospitals as HIFLD
        lists them (hifld_hospitals). RAW DATA: none of the checks a planning
        tool makes, and an empty answer is not proof of absence - prefer
        hasty_candidates, parking_candidates, unmarked_paths or
        nearest_medical_facilities for a question they answer.

        The area: a staged `handle`, or `center_latitude`,
        `center_longitude` and `radius_m` (at most 4,500 m).

        `filters`, by source:
        - osm: tag to value - true for any value, a string, or a list of
          strings, e.g. {"highway": ["track", "path"]} or {"barrier": true}.
          Only the tags the staging query fetches are there: highways,
          waterways, water edges, power lines, railways, barriers, pipelines,
          reservoirs, parking, gates and public sites. Reads the tiles
          staging cached, so over staged ground it costs Overpass nothing.
        - agency_ways: {"source": [...]} from usfs_roads, usfs_trails, blm,
          usgs_trails, usgs_roads (all when not given).
        - hifld_hospitals: {"where": "<ArcGIS where clause>"}; default open
          hospitals.

        Returns `total` (every feature found), `features` (the first
        `limit`, at most 200), `returned`, `path` (all of them, as a GeoJSON
        file save_export can hand over), `attribution`, `warnings` (an
        agency inventory that failed while others answered) and `note`."""
        return query_features(ctx, source=source, handle=handle,
                              center_latitude=center_latitude,
                              center_longitude=center_longitude, radius_m=radius_m,
                              filters=filters, limit=limit)
