"""Roads and trails from the land agencies' own records, for one staged box.

WHY THESE AND NOT ONLY OPENSTREETMAP
------------------------------------
OSM is what nearly every map app draws. A Forest Service road that was
closed thirty years ago (maintenance level 1, "basic custodial care") is
usually still a bench on the hillside and still a corridor a lost person
walks, and it is very often missing from OSM - while the agency's own road
inventory still carries it. The same goes for BLM routes and for trails an
agency manages but nobody has traced into OSM.

So these layers are read as EVIDENCE, compared against the staged OSM ways
by `unmarked_paths`, never as a replacement for them.

THE LAYERS, VERIFIED LIVE 2026-10-21
------------------------------------
- Forest Service EDW `EDW_RoadBasic_01` layer 0 (National Forest System
  roads) and layer 1 (closed to motorized use). The two are DISJOINT: over
  a Tahoe NF patch (-120.76, 39.00, -120.70, 39.04) layer 0 returned 7 road
  ids and layer 1 five others, none shared - so both are read.
- Forest Service EDW `EDW_TrailNFSPublish_01` layer 0 (system trails).
- USGS National Map `transportation` layer 37 (trails), 35 (4WD roads) and
  36 (closed roads). Layer 37 is largely a COPY of other agencies' data:
  over the same patch it returned the same three Forest Service segments,
  with `sourceoriginator` "U.S. Forest Service". Each record's lineage is
  therefore read from its originator field, so a copy is never counted as
  a second, independent sighting.
- BLM `BLM_Natl_GTLF_Public_Display` layers 0-7 (Ground Transportation
  Linear Features), verified live 2026-10-27. BLM retired the single-layer
  `BLM_Natl_GTLF` service (it now answers 404) and split its routes by
  what each is managed for: 0 and 1 roads for public and limited public
  motorized use, 2 and 3 the same for trails, 4 non-motorized and 5
  non-mechanized trails, 6 trails not assessed for public use, 7 trails
  managed for public use. All eight are read. The kind and the managed-for
  class come from the layer; the old observed mode of transport is gone,
  and `PLAN_MODE_TRNSPRT` is kept as detail. Being a public display, it is
  not known to carry routes BLM has designated closed - the ones most
  likely missing from OSM.

None needs a credential. All return at most 2000 records a page and support
`resultOffset` pagination.

THE CACHE
---------
`<staged directory>/agency_ways.json`, one file per staging, because the box
is fixed for the life of a handle. A source that failed is stored as failed
(with a reason, never a URL) and re-asked on the next call; a source that
answered is reused until `refresh=True`.
"""

from __future__ import annotations

import datetime as _dt
import functools
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..net_errors import safe_reason

USER_AGENT = "sar-geo-mcp (search and rescue planning)"

CACHE_FILENAME = "agency_ways.json"

#: Bumped when a layer, a field or the record shape changes, so a staged
#: directory fetched under the old rules is fetched again, not misread.
AGENCY_QUERY_VERSION = 2

PAGE_SIZE = 2000
MAX_PAGES = 20

_USFS = "https://apps.fs.usda.gov/arcx/rest/services/EDW"
_TNM = "https://carto.nationalmap.gov/arcgis/rest/services/transportation/MapServer"
_BLM = ("https://gis.blm.gov/arcgis/rest/services/transportation/"
        "BLM_Natl_GTLF_Public_Display/MapServer")


def _text(value) -> str | None:
    """A field as clean text: None for null, blank and the literal 'None'
    BLM writes into unnamed routes."""
    if value is None:
        return None
    text = str(value).strip()
    return text if text and text.lower() not in ("none", "null", "<null>") else None


def lineage_from_originator(originator: str | None, default: str) -> str:
    """Whose survey a USGS record really is. The National Map republishes
    other agencies' lines, so its trail layer is a copy of the Forest
    Service's wherever the Forest Service mapped first."""
    text = (originator or "").lower()
    if "forest service" in text or "usfs" in text:
        return "usfs"
    if "land management" in text or text.startswith("blm"):
        return "blm"
    if "park service" in text or text.startswith("nps"):
        return "nps"
    if "census" in text or "tiger" in text:
        return "census"
    return default


def _usfs_road(attributes: dict, *, closed_layer: bool) -> dict:
    level = _text(attributes.get("oper_maint_level"))
    open_to = _text(attributes.get("openforuseto"))
    status = []
    if level and level.startswith("1"):
        status.append("closed (maintenance level 1)")
    elif closed_layer:
        status.append("closed to motor vehicles")
    if open_to and open_to.upper() == "ADMIN":
        status.append("administrative use only")
    return {
        "kind": "road", "lineage": "usfs",
        "name": _text(attributes.get("name")), "ref": _text(attributes.get("id")),
        "status": "; ".join(status) or None,
        "detail": {"maintenance_level": level,
                   "surface": _text(attributes.get("surface_type")),
                   "open_to": open_to,
                   "route_status": _text(attributes.get("route_status"))},
    }


def _usfs_trail(attributes: dict) -> dict:
    return {
        "kind": "trail", "lineage": "usfs",
        "name": _text(attributes.get("trail_name")), "ref": _text(attributes.get("trail_no")),
        "status": None,
        "detail": {"trail_class": _text(attributes.get("trail_class")),
                   "trail_type": _text(attributes.get("trail_type")),
                   "tread_width": _text(attributes.get("typical_tread_width"))},
    }


def _usgs_trail(attributes: dict) -> dict:
    originator = _text(attributes.get("sourceoriginator"))
    return {
        "kind": "trail", "lineage": lineage_from_originator(originator, "usgs"),
        "name": _text(attributes.get("name")), "ref": _text(attributes.get("trailnumber")),
        "status": None,
        "detail": {"trail_type": _text(attributes.get("trailtype")),
                   "originator": originator},
    }


def _usgs_road(attributes: dict, *, closed_layer: bool) -> dict:
    originator = _text(attributes.get("source_originator"))
    return {
        "kind": "road", "lineage": lineage_from_originator(originator, "usgs"),
        "name": _text(attributes.get("name")), "ref": None,
        "status": "closed" if closed_layer else "4WD",
        "detail": {"mtfcc": _text(attributes.get("mtfcc_code")), "originator": originator},
    }


#: BLM access restrictions that restrict nothing, in any of the cases BLM
#: writes them ("None", "UNKNOWN", "All").
_BLM_NO_RESTRICTION = frozenset({"none", "unknown", "all"})


def _blm(attributes: dict, *, kind: str, managed_for: str,
         limited: str | None = None) -> dict:
    """A BLM route. `kind`, `managed_for` and `limited` come from the layer
    it was read from: the Public Display service files each route under
    what it is managed for."""
    restriction = _text(attributes.get("PLAN_ACCESS_RSTRCT"))
    if restriction and restriction.lower() in _BLM_NO_RESTRICTION:
        restriction = None
    status = [part for part in (limited, restriction and restriction.lower()) if part]
    accuracy = attributes.get("ACCURACY_FT")
    return {
        "kind": kind, "lineage": "blm",
        "name": _text(attributes.get("ROUTE_PRMRY_NM")), "ref": _text(attributes.get("FAMS_ID")),
        "status": "; ".join(status) or None,
        "detail": {"managed_for": managed_for,
                   "mode": _text(attributes.get("PLAN_MODE_TRNSPRT")),
                   "surface": _text(attributes.get("OBSRVE_SRFCE_TYPE")),
                   "accuracy_ft": accuracy if isinstance(accuracy, (int, float))
                   and accuracy > 0 else None},
    }


_BLM_FIELDS = ("ROUTE_PRMRY_NM,FAMS_ID,PLAN_MODE_TRNSPRT,OBSRVE_SRFCE_TYPE,"
               "PLAN_ACCESS_RSTRCT,ACCURACY_FT")

#: (layer id, kind, what BLM manages it for, the status that class implies).
_BLM_LAYERS = (
    (0, "road", "public motorized use", None),
    (1, "road", "limited public motorized use", "limited public motorized use"),
    (2, "trail", "public motorized use", None),
    (3, "trail", "limited public motorized use", "limited public motorized use"),
    (4, "trail", "public non-motorized use", None),
    (5, "trail", "public non-mechanized use", None),
    (6, "trail", "not assessed for public use", "not assessed for public use"),
    (7, "trail", "public use", None),
)


@dataclass(frozen=True)
class Layer:
    """One ArcGIS layer, and how its attributes become a record."""
    url: str
    out_fields: str
    describe: Callable[[dict], dict]
    label: str


#: Source key -> the layers read for it. The keys are what a caller names.
SOURCES: dict[str, tuple[Layer, ...]] = {
    "usfs_roads": (
        Layer(f"{_USFS}/EDW_RoadBasic_01/MapServer/0",
              "id,name,oper_maint_level,surface_type,openforuseto,route_status",
              functools.partial(_usfs_road, closed_layer=False),
              "Forest Service system roads"),
        Layer(f"{_USFS}/EDW_RoadBasic_01/MapServer/1",
              "id,name,oper_maint_level,surface_type,openforuseto,route_status",
              functools.partial(_usfs_road, closed_layer=True),
              "Forest Service roads closed to motorized use"),
    ),
    "usfs_trails": (
        Layer(f"{_USFS}/EDW_TrailNFSPublish_01/MapServer/0",
              "trail_no,trail_name,trail_type,trail_class,typical_tread_width",
              _usfs_trail, "Forest Service system trails"),
    ),
    "blm": (
        *(Layer(f"{_BLM}/{layer}", _BLM_FIELDS,
                functools.partial(_blm, kind=kind, managed_for=managed_for,
                                  limited=limited),
                f"BLM {kind}s ({managed_for})")
          for layer, kind, managed_for, limited in _BLM_LAYERS),
    ),
    "usgs_trails": (
        Layer(f"{_TNM}/37", "name,trailnumber,trailtype,sourceoriginator",
              _usgs_trail, "USGS National Map trails"),
    ),
    "usgs_roads": (
        Layer(f"{_TNM}/35", "name,mtfcc_code,source_originator",
              functools.partial(_usgs_road, closed_layer=False),
              "USGS National Map 4WD roads"),
        Layer(f"{_TNM}/36", "name,mtfcc_code,source_originator",
              functools.partial(_usgs_road, closed_layer=True),
              "USGS National Map closed roads"),
    ),
}

AGENCY_SOURCES = tuple(SOURCES)


def _http_get(url: str, params: dict | None = None, *, timeout: float = 60.0):
    """A GET with the shared retry policy and an identifying User-Agent."""
    import httpx

    from ..net import get_with_retry

    request = httpx.Request("GET", url, params=params)
    return get_with_retry(
        str(request.url), attempts=3, base_delay=1.0, timeout=timeout,
        get=functools.partial(httpx.get, headers={"User-Agent": USER_AGENT}))


def esri_paths_to_geojson(geometry: dict | None) -> dict | None:
    """An Esri polyline ({"paths": [...]}) as a GeoJSON line, or None."""
    paths = [[[float(x), float(y)] for x, y, *_ in path]
             for path in (geometry or {}).get("paths") or [] if len(path) >= 2]
    if not paths:
        return None
    if len(paths) == 1:
        return {"type": "LineString", "coordinates": paths[0]}
    return {"type": "MultiLineString", "coordinates": paths}


def query_layer(layer: Layer, bbox_wgs84: tuple[float, float, float, float], *,
                get=None) -> list[dict]:
    """Every feature of `layer` touching the box, as records with a WGS84
    GeoJSON `geometry`. Raises on a refusal, so a failed source is never
    mistaken for an empty one."""
    get = get or _http_get
    west, south, east, north = bbox_wgs84
    records: list[dict] = []
    for page in range(MAX_PAGES):
        params = {
            "where": "1=1", "geometry": f"{west},{south},{east},{north}",
            "geometryType": "esriGeometryEnvelope", "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects", "outFields": layer.out_fields,
            "returnGeometry": "true", "outSR": "4326",
            "resultOffset": str(page * PAGE_SIZE), "resultRecordCount": str(PAGE_SIZE),
            "f": "json",
        }
        payload = get(f"{layer.url}/query", params).json()
        if "error" in payload:
            raise RuntimeError(f"{layer.label} refused the query: "
                               f"{(payload['error'] or {}).get('code', 'error')}")
        for feature in payload.get("features") or []:
            geometry = esri_paths_to_geojson(feature.get("geometry"))
            if geometry is None:
                continue
            record = layer.describe(feature.get("attributes") or {})
            record["layer"] = layer.label
            record["geometry"] = geometry
            records.append(record)
        if not payload.get("exceededTransferLimit"):
            break
    return records


def fetch_source(source: str, bbox_wgs84, *, get=None) -> dict:
    """One source's answer: {"status": "ok", "features": [...]} or
    {"status": "failed", "reason": ...}. The reason never carries a URL."""
    features: list[dict] = []
    try:
        for layer in SOURCES[source]:
            for record in query_layer(layer, bbox_wgs84, get=get):
                record["source"] = source
                features.append(record)
    except Exception as error:  # noqa: BLE001 - a missing source, not a refusal
        reason = str(error) if isinstance(error, RuntimeError) else safe_reason(error)
        return {"status": "failed", "reason": reason, "features": None}
    return {"status": "ok", "reason": None, "features": features}


def load_agency_ways(directory: Path, bbox_wgs84, sources: list[str], *,
                     refresh: bool = False, get=None) -> dict[str, dict]:
    """{source: answer} for each of `sources`, from the staged cache where it
    holds a good answer for this box and from the services otherwise."""
    from ..cache import atomic_write_text

    unknown = [source for source in sources if source not in SOURCES]
    if unknown:
        raise ValueError(f"unknown agency source(s) {unknown}; choose from {list(SOURCES)}")

    path = Path(directory) / CACHE_FILENAME
    box = [round(float(value), 7) for value in bbox_wgs84]
    cached: dict[str, Any] = {}
    if path.exists() and not refresh:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}
        if (payload.get("version") == AGENCY_QUERY_VERSION
                and payload.get("bbox_wgs84") == box):
            cached = payload.get("sources") or {}

    answers: dict[str, dict] = {}
    changed = False
    for source in sources:
        held = cached.get(source)
        if held and held.get("status") == "ok":
            answers[source] = held
            continue
        answer = fetch_source(source, box, get=get)
        answer["fetched_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
        answers[source] = answer
        cached[source] = answer
        changed = True

    if changed:
        atomic_write_text(path, json.dumps({"version": AGENCY_QUERY_VERSION,
                                            "bbox_wgs84": box, "sources": cached}))
    return answers
