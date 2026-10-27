"""Populated blocks on real ground: Berwick Drive, Carmel Valley, a street
of single-family lots on both sides.

Marked `live` (deselected by default; run with `-m live`). It needs the
elevation service and Overpass to stage, and CalTopo's parcel and
structures layers through `ZA_WMS_TOKEN`; without the token it is skipped.
`ZA_LIVE_CACHE_DIR` keeps a cache between runs, as the other live tests do.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

BERWICK = (36.5212, -121.8105)
#: On Berwick Drive itself, between the two rows of houses.
ON_BERWICK = (36.52049, -121.81088)


class _NoMapReader:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.offline = False

    def read(self, map_id, *, public, max_age_seconds=300.0):
        raise AssertionError("no map is read for populated blocks")


@pytest.mark.live
def test_live_berwick_drive_is_one_block_across_the_street(tmp_path):
    from shapely.geometry import Point, shape

    from sar_geo_mcp.config import CaltopoConfig
    from sar_geo_mcp.planning.area import area_directory, prepare_area
    from sar_geo_mcp.planning.populated import populated_areas
    from sar_geo_mcp.staging import is_osm_outage_warning

    config = CaltopoConfig.from_env()
    if not config.wms_base_url():
        pytest.skip("ZA_WMS_TOKEN is not set")
    cache_root = Path(os.environ.get("ZA_LIVE_CACHE_DIR") or tmp_path / "cache")
    reader = _NoMapReader(cache_root)
    staged = prepare_area(reader, center_latitude=BERWICK[0], center_longitude=BERWICK[1],
                          radius_m=400.0, resolution_m=10.0)
    if not staged["osm_available"]:
        outage = [w for w in staged["warnings"] if is_osm_outage_warning(w)]
        pytest.skip(f"Overpass could not be reached: {outage or staged['warnings']}")
    directory = area_directory(reader.cache_dir, staged["handle"])

    result = populated_areas(directory, wms_base_url=config.wms_base_url(),
                             cache_dir=cache_root)

    berwick = [b for b in result["blocks"]
               if b["kind"] == "street" and b["facts"]["street"] == "Berwick Drive"]
    assert berwick, [b["name"] for b in result["blocks"]]
    biggest = max(berwick, key=lambda b: b["facts"]["parcels"])
    assert biggest["facts"]["parcels"] >= 15, biggest["facts"]
    on_street = Point(ON_BERWICK[1], ON_BERWICK[0])
    assert shape(biggest["geometry"]).contains(on_street)
    # The condominium complex off Center Street is one block, not a dozen
    # unit-sized ones.
    assert all(b["facts"]["median_parcel_m2"] >= 120 for b in result["blocks"]), \
        [(b["name"], b["facts"]["median_parcel_m2"]) for b in result["blocks"]]
    for warning in result["warnings"]:
        assert config.wms_token not in warning
    assert all("markers" not in b for b in result["blocks"])

    # Asked for, every building in a block gets one white house marker, and
    # the count agrees with the block's own `buildings`.
    marked = populated_areas(directory, wms_base_url=config.wms_base_url(),
                             cache_dir=cache_root, building_markers=True)
    assert [b["geometry"] for b in marked["blocks"]] == [b["geometry"] for b in result["blocks"]]
    for block in marked["blocks"]:
        assert len(block["markers"]) == block["facts"]["buildings"], block["name"]
        for marker in block["markers"]:
            assert marker["properties"]["marker-symbol"] == "hut"
            assert marker["properties"]["marker-color"] == "#FFFFFF"
    berwick_marked = max((b for b in marked["blocks"] if b["facts"]["street"] == "Berwick Drive"),
                         key=lambda b: b["facts"]["parcels"])
    assert len(berwick_marked["markers"]) >= 15

    # 0.20.0: one piece each, no shared ground, then grouped and exported.
    from shapely.geometry import shape as as_shape

    outlines = [as_shape(b["geometry"]) for b in result["blocks"]]
    assert all(o.geom_type == "Polygon" for o in outlines)
    for i, a in enumerate(outlines):
        for b in outlines[i + 1:]:
            assert a.intersection(b).area < 1e-12        # degrees squared: under 0.01 m2

    from sar_geo_mcp.planning.candidates import save_set
    from sar_geo_mcp.planning.export import export_set
    from sar_geo_mcp.planning.urban import urban_params, urban_set

    params = urban_params(target_acres=10.0, resource_type="GROUND", max_distance_m=None)
    cset = urban_set(directory, params, handle=staged["handle"], norms=None,
                     wms_base_url=config.wms_base_url(), cache_dir=cache_root)
    save_set(directory, cset)
    assert 1 <= len(cset.candidates) < len(result["blocks"])
    exported = export_set(directory, cset.set_id)
    written = json.loads(Path(exported["path"]).read_text(encoding="utf-8"))
    kinds = {f["geometry"]["type"] for f in written["features"]}
    assert kinds <= {"Point", "LineString", "Polygon"}
    assert exported["assignments"] >= len(cset.candidates)
    print("urban:", cset.summary)

    for block in result["blocks"]:
        facts = block["facts"]
        print(block["kind"], block["name"], facts["parcels"], facts["buildings"],
              facts["area_acres"], facts["distance_from_center_m"])
