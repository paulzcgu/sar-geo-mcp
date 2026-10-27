import numpy as np
import pytest
from rasterio.transform import from_origin

from sar_geo_mcp.terrain.hydro import (
    HYDRO_ASSUMPTIONS,
    channel_mask,
    extract_channels,
    upstream_is_truncated,
)


def valley_dem(rows=60, cols=60):
    col_distance = np.abs(np.arange(cols) - cols // 2).astype("float64")
    across = col_distance[None, :].repeat(rows, axis=0) * 2.0
    along = np.arange(rows)[::-1, None].repeat(cols, axis=1) * 0.5
    return across + along


def branched_dem(rows=60, cols=60):
    """Main valley plus a short tributary entering from the east.

    The smooth V has no off-axis convergence, so it cannot produce branches of
    differing length — pruning is untestable against it.
    """
    dem = valley_dem(rows, cols)
    for col in range(30, 46):
        dem[20, col] -= 6.0
        dem[19, col] -= 3.0
        dem[21, col] -= 3.0
    return dem


def flow_for(dem, min_cells=50):
    from sar_geo_mcp.terrain._pysheds_backend import flow_accumulation

    return flow_accumulation(dem, from_origin(0, dem.shape[0], 1, 1),
                             network_min_cells=min_cells)


def test_channel_mask_uses_area_not_cell_count():
    """The same accumulation at finer resolution must not add channels."""
    accumulation = np.zeros((10, 10))
    accumulation[:, 5] = 400  # 400 cells

    coarse = channel_mask(accumulation, pixel_size_m=2.0, min_area_m2=1000.0)
    fine = channel_mask(accumulation, pixel_size_m=0.5, min_area_m2=1000.0)

    # 400 cells x 4 m2 = 1600 m2 -> channel. 400 x 0.25 = 100 m2 -> not.
    assert coarse.any()
    assert not fine.any()


def test_channel_mask_threshold_boundary():
    accumulation = np.full((4, 4), 100.0)

    exactly = channel_mask(accumulation, pixel_size_m=1.0, min_area_m2=100.0)
    above = channel_mask(accumulation, pixel_size_m=1.0, min_area_m2=99.0)
    below = channel_mask(accumulation, pixel_size_m=1.0, min_area_m2=101.0)

    assert exactly.all()
    assert above.all()
    assert not below.any()


def test_extract_channels_traces_the_valley_axis():
    flow = flow_for(valley_dem())
    transform = from_origin(0, 60, 1, 1)

    channels = extract_channels(flow.accumulation, flow.network, transform,
                                pixel_size_m=1.0, min_area_m2=50.0,
                                min_length_m=10.0)

    assert len(channels) >= 1
    main = channels[0]
    assert main["id"] == "drainage_0"
    assert main["kind"] == "drainage"
    assert main["geometry"]["type"] == "LineString"
    xs = [point[0] for point in main["geometry"]["coordinates"]]
    assert float(np.mean(xs)) == pytest.approx(30.0, abs=1.0)


def test_channel_length_is_the_centreline_not_a_blob_traversal():
    """Regression guard: the labelling approach reported 4879 m for this."""
    flow = flow_for(valley_dem())

    channels = extract_channels(flow.accumulation, flow.network,
                                from_origin(0, 60, 1, 1), pixel_size_m=1.0,
                                min_area_m2=50.0, min_length_m=10.0)

    assert channels[0]["properties"]["length_m"] == pytest.approx(58.0, abs=6.0)


def test_extract_channels_prunes_short_branches():
    flow = flow_for(branched_dem())
    transform = from_origin(0, 60, 1, 1)

    permissive = extract_channels(flow.accumulation, flow.network, transform,
                                  pixel_size_m=1.0, min_area_m2=50.0,
                                  min_length_m=1.0)
    strict = extract_channels(flow.accumulation, flow.network, transform,
                              pixel_size_m=1.0, min_area_m2=50.0,
                              min_length_m=20.0)

    assert len(permissive) > 1, "the branched fixture must produce branches"
    assert len(strict) < len(permissive)
    assert all(c["properties"]["length_m"] >= 20.0 for c in strict)


def test_channels_are_ordered_by_contributing_area():
    flow = flow_for(branched_dem())

    channels = extract_channels(flow.accumulation, flow.network,
                                from_origin(0, 60, 1, 1), pixel_size_m=1.0,
                                min_area_m2=50.0, min_length_m=1.0)

    areas = [c["properties"]["contributing_area_m2"] for c in channels]
    assert areas == sorted(areas, reverse=True)


def test_upstream_truncation_detected_when_a_channel_touches_the_edge():
    """Flow entering from outside the window means the area is understated."""
    accumulation = np.zeros((10, 10))
    accumulation[0, 5] = 5000  # a big channel cell ON the top border

    assert upstream_is_truncated(accumulation, pixel_size_m=1.0, min_area_m2=100.0)


def test_upstream_truncation_false_for_an_interior_channel():
    accumulation = np.zeros((10, 10))
    accumulation[5, 5] = 5000

    assert not upstream_is_truncated(accumulation, pixel_size_m=1.0, min_area_m2=100.0)


def test_assumptions_are_exported():
    assert "min_contributing_area_m2" in HYDRO_ASSUMPTIONS
    assert "min_channel_length_m" in HYDRO_ASSUMPTIONS


def test_clip_to_excludes_channels_outside_the_geometry():
    """The DEM covers a context buffer; only channels touching the
    assignment may be measured."""
    from shapely.geometry import box

    flow = flow_for(valley_dem())
    transform = from_origin(0, 60, 1, 1)

    # A box in the top-left corner, away from the centre-column channel.
    far_corner = box(0.0, 50.0, 8.0, 60.0)

    unclipped = extract_channels(flow.accumulation, flow.network, transform,
                                 pixel_size_m=1.0, min_area_m2=50.0,
                                 min_length_m=10.0)
    clipped = extract_channels(flow.accumulation, flow.network, transform,
                               pixel_size_m=1.0, min_area_m2=50.0,
                               min_length_m=1.0, clip_to=far_corner)

    assert len(unclipped) >= 1
    assert clipped == []


def test_clip_to_keeps_and_retrims_an_intersecting_channel():
    from shapely.geometry import box

    flow = flow_for(valley_dem())
    transform = from_origin(0, 60, 1, 1)

    # A band across the middle third of the valley.
    middle = box(25.0, 20.0, 35.0, 40.0)

    clipped = extract_channels(flow.accumulation, flow.network, transform,
                               pixel_size_m=1.0, min_area_m2=50.0,
                               min_length_m=1.0, clip_to=middle)

    assert len(clipped) == 1
    channel = clipped[0]
    # Trimmed to the box: about 20 m, not the full 58 m.
    assert channel["properties"]["length_m"] == pytest.approx(20.0, abs=6.0)
    # The full trace length is preserved for context.
    assert channel["properties"]["full_length_m"] == pytest.approx(58.0, abs=6.0)
    assert channel["properties"]["full_length_m"] > channel["properties"]["length_m"]


def test_clip_to_none_preserves_the_full_trace():
    flow = flow_for(valley_dem())

    channels = extract_channels(flow.accumulation, flow.network,
                                from_origin(0, 60, 1, 1), pixel_size_m=1.0,
                                min_area_m2=50.0, min_length_m=10.0)

    channel = channels[0]
    assert channel["properties"]["length_m"] == pytest.approx(
        channel["properties"]["full_length_m"], rel=1e-6
    )


def test_min_length_applies_to_the_clipped_portion():
    """A long channel that barely clips the corner must be prunable."""
    from shapely.geometry import box

    flow = flow_for(valley_dem())
    transform = from_origin(0, 60, 1, 1)
    sliver = box(28.0, 0.0, 32.0, 5.0)  # a ~5 m bite of the channel

    kept = extract_channels(flow.accumulation, flow.network, transform,
                            pixel_size_m=1.0, min_area_m2=50.0,
                            min_length_m=1.0, clip_to=sliver)
    pruned = extract_channels(flow.accumulation, flow.network, transform,
                              pixel_size_m=1.0, min_area_m2=50.0,
                              min_length_m=30.0, clip_to=sliver)

    assert len(kept) == 1
    assert pruned == []
