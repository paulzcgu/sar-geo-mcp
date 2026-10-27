"""Asking for a finer staged raster than the 1,024-pixel default.

WHY THIS EXISTS
---------------
The fixed 1,024-pixel raster plus the mandatory 1,000 m planning buffer caps
`prepare_area` at about 2.9 m per pixel however small the area is. A 2,500 m
radius over El Corte de Madera staged at 6.03 m and Purisima at 7.23 m - and
at those resolutions BOTH preserves contain no cliffs at all, because a 25 m
face across one metre of ground is invisible at six.

Every number in the two site studies therefore came from bypassing this
package and calling `fetch_dem` directly at 4,096 and 6,144 pixels. The
staging path could not answer the question its own detectors exist for. This
adds the lever.

THE KEY MUST MOVE WITH IT
-------------------------
`area_key` already hashes the buffer, on the stated grounds that the buffer is
not a detail of how the ground was fetched but how much ground there IS.
Resolution is the same kind of fact for the same reason: serving a 6 m raster
under the handle someone staged at 1 m would answer a different question under
the same name.
"""

import pytest

from sar_geo_mcp.planning.area import area_key, build_area
from sar_geo_mcp.staging import (
    MAX_STAGE_PIXELS,
    STAGE_PIXELS,
    panel_dimensions,
    pixels_for_resolution,
)

CENTRE = dict(center_latitude=37.4013, center_longitude=-122.3386)


def test_the_default_is_unchanged():
    assert STAGE_PIXELS == 1024
    assert panel_dimensions((0.0, 0.0, 1000.0, 1000.0)) == (1024, 1024)


def test_pixels_are_chosen_to_meet_the_requested_ground_resolution():
    """A 3,000 m box at 1 m wants about 3,000 pixels, not 1,024."""
    px = pixels_for_resolution(extent_m=3000.0, resolution_m=1.0)
    assert 2900 <= px <= 3100


def test_a_coarser_request_is_honoured_too():
    assert pixels_for_resolution(extent_m=4000.0, resolution_m=10.0) == pytest.approx(400, abs=20)


def test_a_request_finer_than_the_service_allows_is_refused_with_the_limit_named():
    """3DEP caps exportImage at 8,000 px per axis. Refuse rather than silently
    serve something coarser than asked for - everything downstream is a
    measurement."""
    with pytest.raises(ValueError) as caught:
        pixels_for_resolution(extent_m=40_000.0, resolution_m=0.5)
    message = str(caught.value)
    assert str(MAX_STAGE_PIXELS) in message
    assert "0.5" in message or "resolution" in message


def test_panel_dimensions_accepts_the_larger_size_and_keeps_pixels_square():
    width, height = panel_dimensions((0.0, 0.0, 2000.0, 1000.0), max_px=4096)
    assert width == 4096
    assert height == 2048


def test_resolution_changes_the_area_handle():
    """Two stagings differing only by resolution are two different grounds."""
    coarse = build_area(**CENTRE, radius_m=600.0)
    fine = build_area(**CENTRE, radius_m=600.0, resolution_m=1.0)
    assert coarse.key != fine.key
    assert fine.key.startswith("aoi-")


def test_the_same_resolution_request_is_the_same_handle():
    first = build_area(**CENTRE, radius_m=600.0, resolution_m=1.0)
    second = build_area(**CENTRE, radius_m=600.0, resolution_m=1.0)
    assert first.key == second.key


def test_omitting_resolution_keeps_the_handle_every_existing_caller_has():
    """A staged directory from before this feature must still be found."""
    legacy = area_key("circle", [37.4013, -122.3386], 600.0, None)
    assert build_area(**CENTRE, radius_m=600.0).key == legacy


def test_the_area_records_what_it_asked_for():
    area = build_area(**CENTRE, radius_m=600.0, resolution_m=1.0)
    assert area.resolution_m == 1.0
    assert build_area(**CENTRE, radius_m=600.0).resolution_m is None


def test_a_nonsense_resolution_is_refused():
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            build_area(**CENTRE, radius_m=600.0, resolution_m=bad)
