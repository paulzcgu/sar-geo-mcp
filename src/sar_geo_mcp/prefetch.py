"""Pull down everything a map's ground needs, before losing connectivity.

WHY
---
A command post is often somewhere with no signal. Everything this package
computes - slope, hydrology, vegetation, coverage - needs data that lives
on somebody else's server, and discovering that at the trailhead is too
late. This walks a CalTopo map, works out which ground its features sit
on, and fetches the tiles for it while there is still wifi.

It is only possible because F3 put the static layers on a fixed grid: you
cannot enumerate what you will need in advance if the cache key is the
hash of a request nobody has made yet.

ESTIMATE FIRST, ALWAYS
----------------------
`plan_prefetch` downloads nothing. It reports how many tiles are missing,
roughly how many bytes that is, which clusters of features it intends to
cover, and which it is skipping and why. `run_prefetch` then does the work
under an explicit byte budget. Nothing here ever silently pulls a
gigabyte down a tethered phone.

RESOLUTION IS CHOSEN PER FEATURE, NOT PER CLUSTER
-------------------------------------------------
Each assignment eventually gets requested at roughly `REQUEST_PIXELS`
across its own extent, so that is the level its tiles are prefetched at -
using the same `level_for` rule the real request will use. Choosing a
level from the cluster instead would fetch a whole region at one coarse
level, and every actual request would then miss.

CLUSTERS EXIST TO STOP ONE FAR-OFF MARKER COSTING A REGION
-----------------------------------------------------------
A map routinely carries a marker for a hospital, an ICP, or a staging area
tens of kilometres from the search. Taking the union of everything would
drag that whole corridor into the download. Features are grouped into
clusters instead, and a cluster holding no assignment and sitting far from
any that does is skipped - and reported as skipped, never dropped
silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .cache import CacheIndex
from .geo import Bbox, ground_meters_per_mercator_unit
from .tiles import Tile, level_for, tiles_covering

#: What a panel or a DEM is eventually requested at, across one feature.
#: Matches the debrief pipeline's own panel size, so prefetched tiles land
#: on the level the real request will ask for.
REQUEST_PIXELS = 1024

#: Ground metres of margin by class. Assignments get the most because
#: searchers leave the boundary and coverage needs the ground around it;
#: a marker needs only enough context to be readable.
BUFFERS_M: dict[str, float] = {
    "Assignment": 400.0,
    "Shape": 200.0,
    "Marker": 100.0,
}
DEFAULT_BUFFER_M = 150.0

#: Features whose buffered extents come within this of each other are one
#: cluster. Roughly "same piece of country".
CLUSTER_GAP_M = 2_000.0

#: An assignment-free cluster further than this from every cluster that
#: does hold an assignment is somebody's hospital pin, not search ground.
ORPHAN_LIMIT_M = 25_000.0

#: Which feature classes actually get terrain staged for them.
#:
#: Only assignments. Measured on a real 85-feature map, enumerating tiles
#: for every feature came to 266 DEM tiles and an estimated 1.1 GB - because
#: a 50 m marker has a tiny extent, so `level_for` picks a very deep zoom
#: for it, and those tiles are both expensive and useless: nothing in this
#: project ever renders a DEM for a marker. Non-assignment features still
#: form clusters, so they still shape what is reported and skipped; they
#: just do not drag their own private zoom level into the download.
TERRAIN_CLASSES = frozenset({"Assignment"})

#: Deepest level prefetch will enumerate. A pathologically small assignment
#: would otherwise request tiles finer than anything downstream reads, and
#: a mistake there is measured in gigabytes.
MAX_PREFETCH_LEVEL = 17

#: Measured on this project's own cache, not guessed: 9 DEM tiles came to
#: 37.8 MB and 9 slope-shading tiles to 2.6 MB; the EVT and OSM figures
#: are from 4 and 6 real tiles respectively. Imagery and multi-layer
#: panels run heavier than slope shading, hence the separate figure.
#: These drive the ESTIMATE only - the budget is enforced against bytes
#: actually written.
TILE_BYTES: dict[str, int] = {
    "dem": 4_200_000,
    "panel": 300_000,
    "panel-imagery": 900_000,
    "evt": 135_000,
    "osm": 375_000,
}


@dataclass(frozen=True)
class Cluster:
    """A piece of country the map has features on."""

    bbox: Bbox
    labels: list[str]
    has_assignment: bool

    @property
    def span_m(self) -> float:
        return max(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1])


@dataclass(frozen=True)
class LayerPlan:
    layer: str
    missing: list[Tile]
    cached: int

    @property
    def estimated_bytes(self) -> int:
        per_tile = TILE_BYTES.get(self.layer, TILE_BYTES["panel"])
        return len(self.missing) * per_tile


@dataclass(frozen=True)
class PrefetchPlan:
    map_id: str
    clusters: list[Cluster]
    skipped: list[tuple[Cluster, str]]
    layers: list[LayerPlan] = field(default_factory=list)

    @property
    def missing_tiles(self) -> int:
        return sum(len(layer.missing) for layer in self.layers)

    @property
    def cached_tiles(self) -> int:
        return sum(layer.cached for layer in self.layers)

    @property
    def estimated_bytes(self) -> int:
        return sum(layer.estimated_bytes for layer in self.layers)

    def summary(self) -> dict:
        """What a caller is shown BEFORE anything is downloaded."""
        return {
            "map_id": self.map_id,
            "tiles_missing": self.missing_tiles,
            "tiles_already_cached": self.cached_tiles,
            "estimated_bytes": self.estimated_bytes,
            "estimated_mb": round(self.estimated_bytes / 1e6, 1),
            "clusters": [
                {"labels": cluster.labels[:12],
                 "feature_count": len(cluster.labels),
                 "span_m": round(cluster.span_m),
                 "has_assignment": cluster.has_assignment}
                for cluster in self.clusters
            ],
            "skipped": [
                {"labels": cluster.labels[:12], "reason": reason}
                for cluster, reason in self.skipped
            ],
            "by_layer": {
                layer.layer: {"missing": len(layer.missing),
                              "cached": layer.cached,
                              "estimated_mb": round(layer.estimated_bytes / 1e6, 1)}
                for layer in self.layers
            },
        }


# --- working out the ground ------------------------------------------------

def _feature_bbox(geometry: dict) -> Bbox | None:
    from .geo import geometry_bounds_wgs84, to_web_mercator

    try:
        return to_web_mercator(geometry_bounds_wgs84(geometry))
    except Exception:
        # A feature with no usable geometry contributes no ground. It is
        # not an error - plenty of CalTopo objects carry none.
        return None


def buffered_bbox(feature: dict) -> tuple[Bbox, str, str] | None:
    """One feature's ground with its class margin, in EPSG:3857.

    The buffer is stated in ground metres and converted here, because
    Mercator units are inflated by 1/cos(latitude) and a fixed unit margin
    would mean 400 m in California and something else entirely in Alaska.
    """
    properties = feature.get("properties") or {}
    feature_class = properties.get("class", "")
    box = _feature_bbox(feature.get("geometry") or {})
    if box is None:
        return None

    label = (properties.get("title")
             or f"{properties.get('letter', '') or ''}"
                f"{properties.get('number', '') or ''}").strip() or "(untitled)"

    from .geo import bbox_center_wgs84

    latitude, _ = bbox_center_wgs84(box)
    metres = BUFFERS_M.get(feature_class, DEFAULT_BUFFER_M)
    margin = metres / ground_meters_per_mercator_unit(latitude)
    return ((box[0] - margin, box[1] - margin, box[2] + margin, box[3] + margin),
            feature_class, label)


def _near(a: Bbox, b: Bbox, gap: float) -> bool:
    return not (a[2] + gap < b[0] or b[2] + gap < a[0]
                or a[3] + gap < b[1] or b[3] + gap < a[1])


def _union(a: Bbox, b: Bbox) -> Bbox:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def cluster_features(entries: list[tuple[Bbox, str, str]],
                     gap_m: float = CLUSTER_GAP_M) -> list[Cluster]:
    """Group features that sit on the same piece of country.

    Repeated merging rather than a one-pass sweep: two boxes can become
    neighbours only after a third joins them, and a single pass would leave
    that chain as separate clusters.
    """
    clusters: list[dict] = []
    for box, feature_class, label in entries:
        clusters.append({"bbox": box, "labels": [label],
                         "assignment": feature_class == "Assignment"})

    merged = True
    while merged:
        merged = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if _near(clusters[i]["bbox"], clusters[j]["bbox"], gap_m):
                    clusters[i]["bbox"] = _union(clusters[i]["bbox"],
                                                 clusters[j]["bbox"])
                    clusters[i]["labels"] += clusters[j]["labels"]
                    clusters[i]["assignment"] |= clusters[j]["assignment"]
                    del clusters[j]
                    merged = True
                    break
            if merged:
                break

    return [Cluster(bbox=c["bbox"], labels=c["labels"],
                    has_assignment=c["assignment"]) for c in clusters]


def _centre(box: Bbox) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _within(box: Bbox, clusters: list[Cluster]) -> bool:
    """Is this extent inside a kept cluster?

    Matched geometrically rather than by label: two features on a map can
    share a title, and a name collision must not silently pull in - or drop
    - somebody else's ground.
    """
    return any(box[0] >= cluster.bbox[0] and box[1] >= cluster.bbox[1]
               and box[2] <= cluster.bbox[2] and box[3] <= cluster.bbox[3]
               for cluster in clusters)


def _terrain_boxes(entries: list[tuple[Bbox, str, str]],
                   clusters: list[Cluster]) -> list[Bbox]:
    """The extents worth staging terrain for - see `TERRAIN_CLASSES`.

    Falls back to every kept feature when the map holds no assignments at
    all, which is the same choice `partition` makes: a caller prefetching
    around markers and lines is doing it on purpose.
    """
    kept = [(box, feature_class) for box, feature_class, _label in entries
            if _within(box, clusters)]
    assignments = [box for box, feature_class in kept
                   if feature_class in TERRAIN_CLASSES]
    return assignments if assignments else [box for box, _ in kept]


def prefetch_level(box: Bbox) -> int:
    """The level a real request for this extent will use, capped."""
    return min(level_for(box, REQUEST_PIXELS, REQUEST_PIXELS),
               MAX_PREFETCH_LEVEL)


def partition(clusters: list[Cluster],
              orphan_limit_m: float = ORPHAN_LIMIT_M,
              ) -> tuple[list[Cluster], list[tuple[Cluster, str]]]:
    """Split clusters into those worth fetching and those to skip, with reasons.

    A cluster holding an assignment is always kept - that is search ground.
    One holding none is kept only if it sits near ground that does, which
    covers the useful cases (a trailhead, a staging point beside the
    search) while excluding the hospital pin two counties away.
    """
    anchors = [c for c in clusters if c.has_assignment]
    keep: list[Cluster] = []
    skip: list[tuple[Cluster, str]] = []

    for cluster in clusters:
        if cluster.has_assignment:
            keep.append(cluster)
            continue
        if not anchors:
            # No assignments anywhere on the map - the caller is prefetching
            # around markers and lines on purpose, so honour that.
            keep.append(cluster)
            continue

        centre = _centre(cluster.bbox)
        nearest_units = min(
            max(abs(centre[0] - _centre(a.bbox)[0]),
                abs(centre[1] - _centre(a.bbox)[1]))
            for a in anchors
        )
        # Mercator units are inflated by 1/cos(latitude); comparing them
        # against a limit in metres would make the threshold mean something
        # different at every latitude.
        from .geo import bbox_center_wgs84

        latitude, _ = bbox_center_wgs84(cluster.bbox)
        nearest = nearest_units * ground_meters_per_mercator_unit(latitude)
        if nearest <= orphan_limit_m:
            keep.append(cluster)
        else:
            skip.append((cluster, (
                f"no assignment, and {round(nearest / 1000)} km from the "
                f"nearest ground that has one")))
    return keep, skip


# --- planning --------------------------------------------------------------

def plan_prefetch(
    map_id: str,
    features: list[dict],
    cache_dir: Path | str,
    panel_sets: dict[str, list[str]] | None = None,
    include_terrain: bool = True,
) -> PrefetchPlan:
    """Work out what is missing. Downloads nothing.

    `panel_sets` maps a name to its CalTopo layer list, exactly as the
    debrief pipeline defines them. Omit it to plan elevation and landcover
    only, which needs no credentials at all.
    """
    cache_dir = Path(cache_dir)
    entries = [entry for entry in
               (buffered_bbox(feature) for feature in features)
               if entry is not None]
    clusters = cluster_features(entries)
    keep, skipped = partition(clusters)
    kept_boxes = _terrain_boxes(entries, keep)

    # Per FEATURE, not per cluster - see the module docstring.
    wanted: dict[str, set[Tile]] = {}

    def want(layer: str, tiles: list[Tile]) -> None:
        wanted.setdefault(layer, set()).update(tiles)

    for box in kept_boxes:
        level = prefetch_level(box)
        if include_terrain:
            want("dem", tiles_covering(box, level))
        for name, layers in (panel_sets or {}).items():
            want(f"panel:{'-'.join(layers)}", tiles_covering(box, level))

    plans = []
    for layer, tiles in sorted(wanted.items()):
        missing, cached = [], 0
        for tile in sorted(tiles, key=lambda t: (t.level, t.row, t.col)):
            if _tile_path(cache_dir, layer, tile).exists():
                cached += 1
            else:
                missing.append(tile)
        plans.append(LayerPlan(layer=_estimate_key(layer), missing=missing,
                               cached=cached))

    return PrefetchPlan(map_id=map_id, clusters=keep, skipped=skipped,
                        layers=plans)


def _estimate_key(layer: str) -> str:
    if layer.startswith("panel:"):
        return "panel-imagery" if "imagery" in layer else "panel"
    return layer


def _tile_path(cache_dir: Path, layer: str, tile: Tile) -> Path:
    if layer == "dem":
        return cache_dir / tile.key("dem", "tif")
    layers = layer.split(":", 1)[1]
    return cache_dir / tile.key(f"wms-{layers}", "png")


# --- doing it --------------------------------------------------------------

def run_prefetch(
    map_id: str,
    features: list[dict],
    cache_dir: Path | str,
    budget_bytes: int,
    panel_sets: dict[str, list[str]] | None = None,
    base_url: str | None = None,
    include_terrain: bool = True,
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    """Fetch the plan, stopping at `budget_bytes` actually written.

    The budget is enforced against bytes on disk, not against the estimate,
    because the estimate is an average and a mountain of imagery is not an
    average tile. Stopping early is reported, never silent: a caller who
    thinks they are provisioned for the field and is not has a worse
    problem than one who knows they ran short.

    `base_url` carries the WMS token in its path. It is passed straight to
    the fetcher and never logged, never put in a filename, and never
    returned in the result.
    """
    from .caltopo.wms import fetch_panel_tile
    from .sources.dem import fetch_dem_tile

    cache_dir = Path(cache_dir)
    plan = plan_prefetch(map_id, features, cache_dir, panel_sets, include_terrain)

    # Re-derive the concrete work, since LayerPlan collapses panel sets onto
    # shared estimate keys.
    written = 0
    fetched = 0
    failures: list[str] = []
    stopped_early = False

    entries = [entry for entry in
               (buffered_bbox(feature) for feature in features)
               if entry is not None]
    boxes = _terrain_boxes(entries, plan.clusters)

    jobs: list[tuple[str, Tile, list[str] | None]] = []
    seen: set[tuple[str, Tile]] = set()
    for box in boxes:
        level = prefetch_level(box)
        for tile in tiles_covering(box, level):
            # Elevation first: it is the measurement substrate, and a
            # budget that runs out should run out on imagery.
            if include_terrain and ("dem", tile) not in seen:
                seen.add(("dem", tile))
                jobs.append(("dem", tile, None))
    for box in boxes:
        level = prefetch_level(box)
        for name, layers in (panel_sets or {}).items():
            if not base_url:
                continue
            key = f"panel:{'-'.join(layers)}"
            for tile in tiles_covering(box, level):
                if (key, tile) in seen:
                    continue
                seen.add((key, tile))
                jobs.append((key, tile, layers))

    for layer, tile, layers in jobs:
        path = _tile_path(cache_dir, layer, tile)
        if path.exists():
            continue
        if written >= budget_bytes:
            stopped_early = True
            break
        try:
            if layer == "dem":
                path = fetch_dem_tile(tile, cache_dir)
            else:
                path = fetch_panel_tile(base_url, layers, tile, cache_dir)
            written += path.stat().st_size
            fetched += 1
            if on_progress:
                on_progress(f"{layer} {tile.level}/{tile.col}/{tile.row}")
        except Exception as error:
            # One bad tile must not abandon the rest - a partly provisioned
            # laptop still beats an unprovisioned one. The layer and tile
            # are named; the URL, which carries the token, is not.
            failures.append(f"{layer} {tile.level}/{tile.col}/{tile.row}: "
                            f"{type(error).__name__}")

    return {
        "map_id": map_id,
        "tiles_fetched": fetched,
        "bytes_written": written,
        "mb_written": round(written / 1e6, 1),
        "stopped_early": stopped_early,
        "budget_bytes": budget_bytes,
        "failures": failures[:20],
        "failure_count": len(failures),
        "cache_total_bytes": CacheIndex(cache_dir).summary()["bytes"],
    }
