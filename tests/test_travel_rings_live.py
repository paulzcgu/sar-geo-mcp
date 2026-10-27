"""Travel-time rings on real ground: Windy Hill, staged live at 10 m.

Marked `live` (deselected by default; run with `-m live`), and set up as the
egress live check is: elevation, LANDFIRE and Overpass, no CalTopo map and no
token. `ZA_LIVE_CACHE_DIR` keeps the cache between runs. When Overpass cannot
be reached the test is skipped, not failed: that is the network.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from shapely.geometry import shape as shapely_shape
from test_egress_live import WINDY_HILL, _mentions_a_path, _NoMapReader


@pytest.mark.live
def test_live_travel_time_rings_at_windy_hill(tmp_path):
    from sar_geo_mcp.planning.area import area_directory, prepare_area
    from sar_geo_mcp.planning.travel_rings import travel_time_rings
    from sar_geo_mcp.staging import is_osm_outage_warning

    latitude, longitude = WINDY_HILL
    cache_root = Path(os.environ.get("ZA_LIVE_CACHE_DIR") or tmp_path / "cache")
    reader = _NoMapReader(cache_root)
    staged = prepare_area(reader, center_latitude=latitude, center_longitude=longitude,
                          radius_m=3000.0, resolution_m=10.0)
    if not staged["osm_available"]:
        outage = [w for w in staged["warnings"] if is_osm_outage_warning(w)]
        pytest.skip(f"Overpass could not be reached: {outage or staged['warnings']}")
    directory = area_directory(reader.cache_dir, staged["handle"])

    result = travel_time_rings(directory, center_latitude=latitude,
                               center_longitude=longitude, minutes=[15, 30, 60])

    rings = result["rings"]
    assert [ring["minutes"] for ring in rings] == [15.0, 30.0, 60.0]
    shapes = [shapely_shape(ring["geometry"]) for ring in rings]
    assert all(shape.is_valid for shape in shapes)
    assert shapes[0].area < shapes[1].area < shapes[2].area
    # Freshly staged ground keeps waterway kinds, and has vegetation.
    assert not any("waterway kinds" in warning for warning in result["warnings"])
    assert result["ground"]["vegetation"]
    # 30 minutes on foot from a trailhead fits in a 3,000 m area.
    assert not rings[1]["reaches_edge"]
    # Trails and terrain make the ring lopsided: no circle.
    reach = rings[2]["reach_by_direction_m"].values()
    assert max(reach) > 1.3 * min(reach)
    for warning in [*result["warnings"], *staged["warnings"]]:
        assert not _mentions_a_path(warning, cache_root), warning

    for ring in rings:
        print(ring["minutes"], ring["area_ha"], ring["equivalent_radius_m"],
              ring["reach_by_direction_m"])
