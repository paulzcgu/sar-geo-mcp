"""Unmarked paths: agency records and terrain benches against mapped OSM ways.

The fusion rules are tested as geometry; the fetch layer with a fake
network; and the whole tool once on synthetic staged ground, where every
answer is known in advance.
"""

import json

import httpx
import numpy as np
import pytest
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import LineString

from test_grid import make_wgs84_square, write_fetch_dir

from sar_geo_mcp.sources.agency import (
    CACHE_FILENAME,
    SOURCES,
    esri_paths_to_geojson,
    fetch_source,
    lineage_from_originator,
    load_agency_ways,
    query_layer,
)
from sar_geo_mcp.terrain.unmarked import (
    cross_support,
    lineages_of,
    merge_same_paths,
    tier_of,
    unmapped_parts,
)
from exports import read_export

ORIGIN = (-13618826.5, 4486605.0)
TO_WGS84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)


def piece(source, lineage, coords, **extra):
    return {"source": source, "lineage": lineage, "layer": source, "kind": "road",
            "name": extra.get("name"), "ref": extra.get("ref"), "status": None,
            "detail": {}, "geom": LineString(coords)}


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


# ---------------------------------------------------------------------------
# Lineage and parsing.
# ---------------------------------------------------------------------------

def test_a_usgs_copy_of_a_forest_service_trail_is_forest_service_lineage():
    assert lineage_from_originator("U.S. Forest Service", "usgs") == "usfs"
    assert lineage_from_originator("Bureau of Land Management", "usgs") == "blm"
    assert lineage_from_originator("National Park Service", "usgs") == "nps"
    assert lineage_from_originator(None, "usgs") == "usgs"


def test_esri_paths_become_geojson_lines():
    assert esri_paths_to_geojson({"paths": [[[0, 0], [1, 1]]]})["type"] == "LineString"
    multi = esri_paths_to_geojson({"paths": [[[0, 0], [1, 1]], [[2, 2], [3, 3, 9]]]})
    assert multi["type"] == "MultiLineString"
    assert multi["coordinates"][1] == [[2.0, 2.0], [3.0, 3.0]]
    assert esri_paths_to_geojson({"paths": [[[0, 0]]]}) is None
    assert esri_paths_to_geojson(None) is None


def test_query_layer_follows_pagination_and_describes_each_record():
    layer = SOURCES["usfs_roads"][1]
    pages = [
        {"features": [{"attributes": {"id": "14N25K", "name": "CONVEYOR RUN",
                                      "oper_maint_level": "1 - BASIC CUSTODIAL CARE (CLOSED)"},
                       "geometry": {"paths": [[[-120.7, 39.0], [-120.71, 39.01]]]}}],
         "exceededTransferLimit": True},
        {"features": [{"attributes": {"id": "14N25KA", "openforuseto": "ADMIN"},
                       "geometry": {"paths": [[[-120.7, 39.0], [-120.72, 39.0]]]}}]},
    ]
    offsets = []

    def get(url, params):
        offsets.append(params["resultOffset"])
        return FakeResponse(pages[len(offsets) - 1])

    records = query_layer(layer, (-121, 38, -120, 40), get=get)
    assert offsets == ["0", "2000"]
    assert [record["ref"] for record in records] == ["14N25K", "14N25KA"]
    assert records[0]["status"] == "closed (maintenance level 1)"
    assert "administrative use only" in records[1]["status"]
    assert records[0]["lineage"] == "usfs" and records[0]["kind"] == "road"


def test_blm_reads_every_public_display_layer_and_takes_kind_from_the_layer():
    layers = SOURCES["blm"]
    assert len(layers) == 8
    assert all("BLM_Natl_GTLF_Public_Display/MapServer/" in layer.url for layer in layers)
    assert sorted(int(layer.url.rsplit("/", 1)[1]) for layer in layers) == list(range(8))
    assert all("PLAN_MODE_TRNSPRT" in layer.out_fields
               and "OBSRVE_MODE_TRNSPRT" not in layer.out_fields for layer in layers)

    seen = []

    def get(url, params):
        seen.append(url)
        layer = int(url.rsplit("/", 2)[1])
        attributes = {"FAMS_ID": f"R{layer}", "ROUTE_PRMRY_NM": "None",
                      "PLAN_MODE_TRNSPRT": "Motorized" if layer < 4 else "Non-Motorized",
                      "PLAN_ACCESS_RSTRCT": {1: "Authorized/Permitted User Only",
                                             0: "UNKNOWN", 7: "All"}.get(layer, "None"),
                      "ACCURACY_FT": 0}
        return FakeResponse({"features": [{"attributes": attributes, "geometry": {
            "paths": [[[-120.7, 39.0], [-120.71, 39.01]]]}}]})

    answer = fetch_source("blm", (-121, 38, -120, 40), get=get)
    assert answer["status"] == "ok" and len(seen) == 8
    by_ref = {record["ref"]: record for record in answer["features"]}
    assert [by_ref[f"R{n}"]["kind"] for n in range(8)] == ["road"] * 2 + ["trail"] * 6
    assert by_ref["R0"]["status"] is None, "UNKNOWN restricts nothing"
    assert by_ref["R7"]["status"] is None, "All restricts nothing"
    assert by_ref["R1"]["status"] == ("limited public motorized use; "
                                      "authorized/permitted user only")
    assert by_ref["R6"]["status"] == "not assessed for public use"
    assert by_ref["R4"]["detail"]["managed_for"] == "public non-motorized use"
    assert by_ref["R4"]["detail"]["mode"] == "Non-Motorized"
    assert by_ref["R0"]["name"] is None and by_ref["R0"]["detail"]["accuracy_ft"] is None
    assert all(record["lineage"] == "blm" for record in answer["features"])


def test_a_refused_query_is_a_failure_never_an_empty_source():
    def refuse(url, params):
        return FakeResponse({"error": {"code": 400}})

    answer = fetch_source("usfs_trails", (-121, 38, -120, 40), get=refuse)
    assert answer["status"] == "failed" and answer["features"] is None
    assert "400" in answer["reason"]


def test_a_network_failure_reason_carries_no_url():
    def down(url, params):
        raise httpx.ConnectError(f"cannot reach {url}")

    answer = fetch_source("blm", (-121, 38, -120, 40), get=down)
    assert answer["status"] == "failed"
    assert "http" not in answer["reason"].lower()


def test_agency_answers_are_cached_with_the_staging_and_failures_asked_again(tmp_path):
    calls = []

    def get(url, params):
        calls.append(url)
        if "BLM" in url:
            raise httpx.ConnectError("down")
        return FakeResponse({"features": []})

    box = (-121.0, 38.0, -120.0, 40.0)
    first = load_agency_ways(tmp_path, box, ["usfs_trails", "blm"], get=get)
    assert first["usfs_trails"]["status"] == "ok" and first["blm"]["status"] == "failed"
    assert (tmp_path / CACHE_FILENAME).exists()

    calls.clear()
    load_agency_ways(tmp_path, box, ["usfs_trails", "blm"], get=get)
    assert calls and all("BLM" in url for url in calls), "a good answer is reused"

    calls.clear()
    load_agency_ways(tmp_path, box, ["usfs_trails"], get=get, refresh=True)
    assert calls, "refresh asks again"

    with pytest.raises(ValueError):
        load_agency_ways(tmp_path, box, ["strava"], get=get)


# ---------------------------------------------------------------------------
# Fusion rules.
# ---------------------------------------------------------------------------

def test_only_the_unmapped_part_of_a_line_survives():
    known = LineString([(0, 0), (0, 100)]).buffer(10)
    along = LineString([(2, 0), (2, 100)])
    assert unmapped_parts(along, known, 50) == []

    leaving = LineString([(2, 0), (2, 100), (2, 300)])
    parts = unmapped_parts(leaving, known, 50)
    assert len(parts) == 1 and parts[0].length == pytest.approx(190, abs=1)

    assert unmapped_parts(along, None, 50)[0].length == pytest.approx(100)


def test_a_copy_of_the_same_survey_is_one_sighting_not_two():
    trail = [(0, 0), (0, 200)]
    pieces = [piece("usfs_trails", "usfs", trail),
              piece("usgs_trails", "usfs", [(3, 0), (3, 200)])]
    candidates = merge_same_paths(pieces, 20)
    assert len(candidates) == 1
    assert len(candidates[0]["sightings"]) == 2
    assert lineages_of(candidates[0]) == ["usfs"]
    assert tier_of(candidates[0]) == "agency_record"


def test_a_terrain_bench_along_an_agency_road_corroborates_it():
    pieces = [piece("terrain", "terrain", [(4, 50), (4, 150)]),
              piece("usfs_roads", "usfs", [(0, 0), (0, 200)], ref="14N25K")]
    candidates = merge_same_paths(pieces, 20)
    assert len(candidates) == 1, "the bench folds into the road"
    road = candidates[0]
    assert road["source"] == "usfs_roads", "the richer record leads"
    assert tier_of(road) == "corroborated"
    bench = [s for s in road["sightings"] if s["source"] == "terrain"][0]
    assert bench["coverage"] == pytest.approx(0.5, abs=0.1)


def test_separate_paths_stay_separate():
    pieces = [piece("usfs_roads", "usfs", [(0, 0), (0, 200)]),
              piece("blm", "blm", [(500, 0), (500, 200)])]
    candidates = merge_same_paths(pieces, 20)
    assert len(candidates) == 2
    assert all(tier_of(candidate) == "agency_record" for candidate in candidates)


def test_a_long_bench_over_a_short_record_supports_it_without_merging():
    pieces = [piece("usfs_roads", "usfs", [(0, 0), (0, 80)]),
              piece("terrain", "terrain", [(2, 0), (2, 400)])]
    candidates = merge_same_paths(pieces, 20)
    assert len(candidates) == 2, "most of the bench lies past the road"
    cross_support(candidates, 20)
    road = [c for c in candidates if c["source"] == "usfs_roads"][0]
    bench = [c for c in candidates if c["source"] == "terrain"][0]
    assert tier_of(road) == "corroborated"
    assert tier_of(bench) == "terrain_only", "a fifth of the bench is not most of it"


# ---------------------------------------------------------------------------
# The whole tool, on synthetic ground.
# ---------------------------------------------------------------------------

def wgs84_path(coords_3857):
    return [list(TO_WGS84.transform(x, y)) for x, y in coords_3857]


def staged_ground(tmp_path):
    """A hillside falling east with a level bench cut north-south at column
    80, and one mapped road at column 20."""
    rows = cols = 160
    ground = np.tile(np.linspace(80.0, 0.0, cols), (rows, 1)).astype("float64")
    for row in range(rows):
        ground[row, 80:84] = ground[row, 80]
    ox, oy = ORIGIN
    write_fetch_dir(tmp_path, ground.astype("float32"), from_origin(ox, oy, 1.0, 1.0),
                    make_wgs84_square(-122.35, 37.40, -122.34, 37.41))
    (tmp_path / "osm.json").write_text(json.dumps({"routes": [{
        "geometry": {"type": "LineString",
                     "coordinates": [[ox + 20, oy - 1], [ox + 20, oy - 159]]},
        "properties": {"name": "Mapped Road", "route_type": "ROAD"}}]}),
        encoding="utf-8")
    return tmp_path


def fake_agencies(url, params):
    ox, oy = ORIGIN

    def line(x, **attributes):
        return {"attributes": attributes,
                "geometry": {"paths": [wgs84_path([(ox + x, oy - 3), (ox + x, oy - 157)])]}}

    if "BLM" in url:
        raise httpx.ConnectError("BLM is down")
    if "RoadBasic_01/MapServer/0" in url:      # an open road OSM already has
        return FakeResponse({"features": [line(24, id="0023", name="MAPPED ROAD")]})
    if "RoadBasic_01/MapServer/1" in url:      # a closed road on the bench
        return FakeResponse({"features": [line(
            82, id="14N25K", name="CONVEYOR RUN",
            oper_maint_level="1 - BASIC CUSTODIAL CARE (CLOSED)")]})
    if "TrailNFSPublish" in url:
        return FakeResponse({"features": [line(140, trail_no="11E31",
                                                trail_name="MOSQUITO RIDGE")]})
    if "MapServer/37" in url:                   # USGS's copy of the same trail
        return FakeResponse({"features": [line(141, name="MOSQUITO RIDGE",
                                                trailnumber="11E31",
                                                sourceoriginator="U.S. Forest Service")]})
    return FakeResponse({"features": []})


def naip_tile(bbox: str, size: str, bare_x: float | None = None) -> bytes:
    """A 4-band GeoTIFF for the requested box: vegetation everywhere (NDVI
    0.6), and bare soil (NDVI 0) in a 6-unit strip at x = `bare_x`."""
    from rasterio.io import MemoryFile
    from rasterio.transform import from_bounds

    west, south, east, north = (float(value) for value in bbox.split(","))
    cols, rows = (int(value) for value in size.split(","))
    red = np.full((rows, cols), 20, np.uint8)
    nir = np.full((rows, cols), 80, np.uint8)
    if bare_x is not None:
        xs = west + (np.arange(cols) + 0.5) * (east - west) / cols
        strip = np.abs(xs - bare_x) <= 3.0
        red[:, strip] = 60
        nir[:, strip] = 60
    with MemoryFile() as memory:
        with memory.open(driver="GTiff", width=cols, height=rows, count=4, dtype="uint8",
                         crs="EPSG:3857",
                         transform=from_bounds(west, south, east, north, cols, rows)) as tif:
            tif.write(np.stack([red, red, red, nir]))
        return memory.read()


class FakeImage:
    def __init__(self, content):
        self.status_code = 200
        self.content = content


def fake_naip(url, params, timeout):
    """NAIP with a bare corridor along the closed road on the bench."""
    return FakeImage(naip_tile(params["bbox"], params["size"], bare_x=ORIGIN[0] + 82))


def run(tmp_path, **kwargs):
    from sar_geo_mcp.planning.unmarked_paths import unmarked_paths

    kwargs.setdefault("imagery_get", fake_naip)
    return unmarked_paths(staged_ground(tmp_path), get=fake_agencies, **kwargs)


ALL_BUT_IMAGERY = ["usfs_roads", "usfs_trails", "blm", "usgs_trails", "usgs_roads"]


def test_by_default_terrain_is_not_read_and_the_closed_road_is_an_agency_record(tmp_path):
    from sar_geo_mcp.planning.unmarked_paths import DEFAULT_SOURCES, TERRAIN_EXPERIMENTAL

    assert "terrain" not in DEFAULT_SOURCES and "imagery" in DEFAULT_SOURCES
    result = run(tmp_path)
    assert "terrain" not in result["sources"]
    assert result["summary"]["terrain"] is None
    assert TERRAIN_EXPERIMENTAL not in result["warnings"]
    assert all(c["tier"] != "terrain_only" for c in result["candidates"])
    road = [c for c in result["candidates"] if c["ref"] == "14N25K"][0]
    assert road["tier"] == "agency_record"
    assert road["independent_sources"] == ["usfs"]
    assert road["status"] == "closed (maintenance level 1)"


def test_terrain_named_is_read_as_experimental_and_corroborates_the_closed_road(tmp_path):
    from sar_geo_mcp.planning.unmarked_paths import TERRAIN_EXPERIMENTAL

    result = run(tmp_path, sources=[*ALL_BUT_IMAGERY, "terrain", "imagery"])
    assert TERRAIN_EXPERIMENTAL in result["warnings"]
    first = result["candidates"][0]
    assert first["tier"] == "corroborated"
    assert first["ref"] == "14N25K"
    assert first["independent_sources"] == ["terrain", "usfs"]


def test_a_road_osm_already_has_is_not_offered(tmp_path):
    result = run(tmp_path)
    assert all(candidate["ref"] != "0023" for candidate in result["candidates"])
    assert result["sources"]["usfs_roads"]["features"] == 2
    assert result["sources"]["usfs_roads"]["pieces"] == 1


def test_the_usgs_copy_of_the_trail_does_not_corroborate_it(tmp_path):
    result = run(tmp_path)
    trail = [c for c in result["candidates"] if c["ref"] == "11E31"]
    assert len(trail) == 1
    assert trail[0]["tier"] == "agency_record"
    assert {s["source"] for s in trail[0]["sightings"]} == {"usfs_trails", "usgs_trails"}


def test_a_failed_source_is_not_looked_for_rather_than_empty(tmp_path):
    result = run(tmp_path)
    assert result["sources"]["blm"]["status"] == "failed"
    assert result["sources"]["blm"]["features"] is None
    assert any("blm" in warning and "NOT LOOKED FOR" in warning
               for warning in result["warnings"])


def test_geometry_and_points_are_wgs84(tmp_path):
    result = run(tmp_path)
    for candidate in result["candidates"]:
        for lon, lat in candidate["geometry"]["coordinates"]:
            assert -123 < lon < -122 and 37 < lat < 38
        assert 37 < candidate["points"]["middle"]["latitude"] < 38


def test_the_caltopo_file_holds_one_line_shape_per_candidate(tmp_path):
    result = run(tmp_path, sources=[*ALL_BUT_IMAGERY, "terrain", "imagery"])
    collection = read_export(result["export"]["path"])
    assert len(collection["features"]) == len(result["candidates"])
    first = collection["features"][0]["properties"]
    assert first["class"] == "Shape"
    assert first["stroke"] == "#FF00FF"
    assert "14N25K" in first["title"] and "unverified" in first["description"]


def test_no_file_unless_asked_and_sources_can_be_narrowed(tmp_path):
    result = run(tmp_path, sources=["usfs_roads"], write_geojson=False)
    assert result["export"] is None
    assert set(result["sources"]) == {"usfs_roads"}
    assert result["candidates"][0]["imagery"] is None, "imagery not asked for"
    assert result["summary"]["terrain"] is None
    only = result["candidates"]
    assert len(only) == 1 and only[0]["tier"] == "agency_record"


def test_limits_and_unknown_sources_are_refused(tmp_path):
    with pytest.raises(ValueError):
        run(tmp_path, sources=["strava"])
    with pytest.raises(ValueError):
        run(tmp_path, limit=-1)
    result = run(tmp_path, limit=1)
    assert result["summary"]["count"] == 1
    assert result["summary"]["total_before_limit"] >= 2


def test_the_caltopo_file_holds_every_candidate_not_just_the_list(tmp_path):
    result = run(tmp_path, limit=1)
    assert result["summary"]["count"] == 1
    collection = read_export(result["export"]["path"])
    assert len(collection["features"]) == result["summary"]["total_before_limit"] >= 2
    assert result["export"]["features"] == len(collection["features"])
    assert any("CalTopo file holds" in warning for warning in result["warnings"])


def test_the_caltopo_file_can_be_capped(tmp_path):
    result = run(tmp_path, limit=0, geojson_limit=1)
    collection = read_export(result["export"]["path"])
    assert len(collection["features"]) == 1
    assert result["summary"]["count"] >= 2
    with pytest.raises(ValueError):
        run(tmp_path, geojson_limit=-1)


# ---------------------------------------------------------------------------
# Imagery as a check on the lines.
# ---------------------------------------------------------------------------

def test_a_bare_corridor_is_reported_but_never_counts_as_a_sighting(tmp_path):
    result = run(tmp_path)
    road = [c for c in result["candidates"] if c["ref"] == "14N25K"][0]
    assert road["imagery"]["bare"] is True
    assert road["imagery"]["flank_ndvi"] - road["imagery"]["along_ndvi"] >= 0.10
    assert "imagery" not in road["independent_sources"]
    assert all(s["source"] != "imagery" for s in road["sightings"])
    assert road["tier"] == "agency_record", "a bare corridor promotes nothing"
    bench_run = run(tmp_path, sources=["terrain", "imagery"])
    assert all(c["tier"] == "terrain_only" for c in bench_run["candidates"]), (
        "a bench on a bare corridor stays terrain_only")
    trail = [c for c in result["candidates"] if c["ref"] == "11E31"][0]
    assert trail["imagery"]["bare"] is False, "vegetated either side and along"
    assert trail["tier"] == "agency_record"
    assert result["sources"]["imagery"]["status"] == "ok"
    assert result["sources"]["imagery"]["bare"] >= 1


def test_imagery_that_cannot_be_read_is_not_looked_at(tmp_path):
    import httpx

    def down(url, params, timeout):
        raise httpx.ConnectError("NAIP is down")

    import sar_geo_mcp.sources.naip as imagery
    imagery_sleep = imagery.time.sleep
    imagery.time.sleep = lambda seconds: None
    try:
        result = run(tmp_path, imagery_get=down)
    finally:
        imagery.time.sleep = imagery_sleep
    assert result["sources"]["imagery"]["status"] == "failed"
    assert result["sources"]["imagery"]["bare"] is None
    assert all(c["imagery"] is None for c in result["candidates"])
    assert any("NOT LOOKED AT" in warning for warning in result["warnings"])
    road = [c for c in result["candidates"] if c["ref"] == "14N25K"][0]
    assert "imagery" not in road["independent_sources"]


def test_imagery_tiles_are_cached_with_the_staging(tmp_path):
    calls = []

    def counting(url, params, timeout):
        calls.append(params["bbox"])
        return fake_naip(url, params, timeout)

    directory = staged_ground(tmp_path)
    from sar_geo_mcp.planning.unmarked_paths import unmarked_paths

    unmarked_paths(directory, get=fake_agencies, imagery_get=counting)
    first = len(calls)
    assert first >= 1
    unmarked_paths(directory, get=fake_agencies, imagery_get=counting)
    assert len(calls) == first, "a second call reads the cached tiles"
    unmarked_paths(directory, get=fake_agencies, imagery_get=counting, refresh=True)
    assert len(calls) == 2 * first


def test_the_tile_grid_coarsens_rather_than_exceed_its_cap():
    from sar_geo_mcp.sources.naip import MAX_TILES, tile_grid

    small, pixel_small = tile_grid((0, 0, 3000, 3000), 0.0)
    assert pixel_small == 1.0 and len(small) == 9
    large, pixel_large = tile_grid((0, 0, 16000, 16000), 0.0)
    assert len(large) <= MAX_TILES and pixel_large > 1.0


def test_bridges_are_reported_apart_and_drawn_green(tmp_path):
    result = run(tmp_path, limit=0)
    for bridge in result["bridges"]:
        assert bridge["id"].startswith("bridge-")
        assert bridge["from"].startswith("unmarked-")
    collection = read_export(result["export"]["path"])
    titles = [f["properties"]["title"] for f in collection["features"]]
    assert sum(title.startswith("Bridge?") for title in titles) == result["export"]["bridges"]
    assert result["summary"]["bridges"] == len(result["bridges"])
