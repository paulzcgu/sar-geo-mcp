"""Egress on real ground: Windy Hill, staged live at 10 m.

Marked `live` (deselected by default; run with `-m live`). It needs the
elevation service and Overpass, not CalTopo: egress reads the staged DEM and
OpenStreetMap only, and no `map_id` is given, so no CalTopo map is read and
no token is used. The cache is the test's own, so every source is fetched
fresh - unless `ZA_LIVE_CACHE_DIR` names one to keep between runs, so a
retry after a busy Overpass does not ask again for the tiles it already has.
When Overpass cannot be reached at all the test is skipped, not failed:
that is the network, not egress.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

WINDY_HILL = (37.35687, -122.24093)


class _NoMapReader:
    """`prepare_area` reads the map only when a `map_id` is given; none is."""

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.offline = False

    def read(self, map_id, *, public, max_age_seconds=300.0):
        raise AssertionError("no map is read for egress")


def _mentions_a_path(text: str, root) -> bool:
    return (str(root) in text or re.search(r"[A-Za-z]:[\\/]", text) is not None
            or "\\" in text)


@pytest.mark.live
def test_live_egress_at_windy_hill(tmp_path):
    from sar_geo_mcp.planning.area import area_directory, prepare_area
    from sar_geo_mcp.planning.egress import egress_routes
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

    results = {mode: egress_routes(directory, center_latitude=latitude,
                                   center_longitude=longitude, mode=mode)
               for mode in ("walk_out", "carry_out")}

    for mode, result in results.items():
        kinds = [exit_["kind"] for exit_ in result["exits"]]
        assert kinds == ["2wd", "4x4"], (mode, kinds)
        for exit_ in result["exits"]:
            assert exit_.get("reachable", True) is not False, (mode, exit_)
        for warning in [*result["warnings"], *staged["warnings"],
                        *(w for e in result["exits"] for w in e.get("warnings", []))]:
            assert not _mentions_a_path(warning, cache_root), warning
            assert not _mentions_a_path(warning, tmp_path), warning
        # The 2WD exit is named: by its road, or by the road its lot opens onto.
        two_wd = result["exits"][0]
        named = ((two_wd.get("road") or {}).get("name")
                 or (two_wd.get("parking_lot") or {}).get("off"))
        assert named, (mode, two_wd.get("road"), two_wd.get("parking_lot"))

    for walk, carry in zip(results["walk_out"]["exits"], results["carry_out"]["exits"]):
        assert carry["minutes"] > walk["minutes"], (walk["kind"], walk["minutes"],
                                                    carry["minutes"])

    # The numbers, for the record of a live run (no path, no credential).
    for mode, result in results.items():
        for exit_ in result["exits"]:
            road = exit_.get("road") or {}
            lot = exit_.get("parking_lot") or {}
            print(f"{mode} {exit_['kind']}: {exit_['minutes']:.1f} min, "
                  f"{exit_['distance_m']:.0f} m, +{exit_['gain_m']:.0f}/-{exit_['loss_m']:.0f} m, "
                  f"off trail {exit_['off_trail_m']:.0f} m, max {exit_['max_grade_pct']:.0f} %, "
                  f"road {road.get('name')!r} ({road.get('highway')}, {road.get('surface')}, "
                  f"access {road.get('access')}, gate_on_road {road.get('gate_on_road')}), "
                  f"lot {lot.get('id')!r} off {lot.get('off')!r}, "
                  f"gates crossed {len(exit_.get('gates_crossed') or [])}")
    print("resolution_m", results["walk_out"]["resolution_m"],
          "warnings", results["walk_out"]["warnings"])
