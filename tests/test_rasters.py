"""Georeference and rendering for the two raster tools.

The pixel-to-coordinate round trip is the part most likely to be quietly
wrong, and a feature seen in a picture is worthless if it cannot be turned
into something a team can navigate to. It is tested against hand-computed
corners.
"""

import json

import httpx
import numpy as np
from rasterio.transform import from_origin

from test_grid import make_wgs84_square, write_fetch_dir

from sar_geo_mcp.planning.rasters import (
    draw_ways,
    georeference,
    pixel_to_wgs84,
    render_grey,
)

ORIGIN = (-13618826.5, 4486605.0)
PIXEL_M = 2.0


def staged(tmp_path, rows=64, cols=64):
    elevation = np.tile(np.linspace(0.0, 40.0, cols), (rows, 1)).astype("float32")
    ox, oy = ORIGIN
    transform = from_origin(ox, oy, PIXEL_M, PIXEL_M)
    geometry = make_wgs84_square(-122.35, 37.40, -122.34, 37.41)
    write_fetch_dir(tmp_path, elevation, transform, geometry)
    return tmp_path, transform, (rows, cols)


def test_georeference_reports_north_up_and_both_projections(tmp_path):
    directory, transform, shape = staged(tmp_path)
    block = georeference(transform, shape, latitude_deg=37.4)
    assert block["north_up"] is True
    assert block["crs"] == "EPSG:3857"
    assert block["width_px"] == shape[1]
    assert block["height_px"] == shape[0]
    assert len(block["bbox_wgs84"]) == 4
    assert len(block["corners_wgs84"]) == 4


def test_pixel_zero_is_the_north_west_corner(tmp_path):
    _, transform, shape = staged(tmp_path)
    block = georeference(transform, shape, latitude_deg=37.4)
    lon, lat = pixel_to_wgs84(transform, 0, 0)
    west, south, east, north = block["bbox_wgs84"]
    assert abs(lon - west) < 1e-9
    assert abs(lat - north) < 1e-9


def test_the_centre_pixel_lands_inside_the_box(tmp_path):
    _, transform, shape = staged(tmp_path)
    block = georeference(transform, shape, latitude_deg=37.4)
    lon, lat = pixel_to_wgs84(transform, shape[1] / 2, shape[0] / 2)
    west, south, east, north = block["bbox_wgs84"]
    assert west < lon < east
    assert south < lat < north


def test_ground_metres_per_pixel_is_true_ground_not_mercator(tmp_path):
    _, transform, shape = staged(tmp_path)
    block = georeference(transform, shape, latitude_deg=37.4)
    # At 37.4 degrees a Mercator unit is about 0.79 ground metres.
    assert 1.4 < block["ground_m_per_px"] < 1.7


def test_bbox_3857_agrees_with_bbox_wgs84_to_floating_point_tolerance(tmp_path):
    """Two different routes to the same rectangle must land on the same rectangle.

    bbox_wgs84 comes from pixel_to_wgs84 at the pixel edges; bbox_3857 is
    computed directly from the affine transform's coefficients. Projecting
    bbox_3857's own corners back to WGS84 must reproduce bbox_wgs84, or one
    of the two routes is wrong.
    """
    from pyproj import Transformer

    _, transform, shape = staged(tmp_path)
    block = georeference(transform, shape, latitude_deg=37.4)
    to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    min_x, min_y, max_x, max_y = block["bbox_3857"]
    lon0, lat0 = to_wgs84.transform(min_x, min_y)
    lon1, lat1 = to_wgs84.transform(max_x, max_y)

    west, south, east, north = block["bbox_wgs84"]
    assert abs(lon0 - west) < 1e-9
    assert abs(lat0 - south) < 1e-9
    assert abs(lon1 - east) < 1e-9
    assert abs(lat1 - north) < 1e-9


def test_render_grey_writes_a_png_of_the_right_size(tmp_path):
    shade = np.linspace(0.0, 1.0, 64 * 64).reshape(64, 64)
    path = render_grey(shade, tmp_path / "hillshade.png")
    from PIL import Image
    with Image.open(path) as image:
        assert image.size == (64, 64)


def test_draw_ways_paints_only_where_a_way_runs(tmp_path):
    from PIL import Image
    _, transform, shape = staged(tmp_path)
    base = Image.new("RGB", (shape[1], shape[0]), (128, 128, 128))
    ox, oy = ORIGIN
    route = {"geometry": {"type": "LineString",
                          "coordinates": [[ox + 10 * PIXEL_M, oy - 2 * PIXEL_M],
                                          [ox + 10 * PIXEL_M, oy - 60 * PIXEL_M]]},
             "properties": {"name": "Skyline Trail"}}
    painted = draw_ways(base, [route], transform, grid=False)
    pixels = np.asarray(painted)
    assert not np.array_equal(pixels[30, 10], (128, 128, 128))
    assert np.array_equal(pixels[30, 50], (128, 128, 128))


def test_draw_ways_does_not_mutate_its_input(tmp_path):
    """Task 6 draws two overlays from one base image; if draw_ways mutated
    the base in place (e.g. because Image.convert("RGB") aliased an
    already-RGB image), the second, supposedly clean, render would come
    back painted."""
    from PIL import Image
    _, transform, shape = staged(tmp_path)
    base = Image.new("RGB", (shape[1], shape[0]), (128, 128, 128))
    before = np.asarray(base).copy()
    ox, oy = ORIGIN
    route = {"geometry": {"type": "LineString",
                          "coordinates": [[ox + 10 * PIXEL_M, oy - 2 * PIXEL_M],
                                          [ox + 10 * PIXEL_M, oy - 60 * PIXEL_M]]},
             "properties": {"name": "Skyline Trail"}}
    draw_ways(base, [route], transform, grid=False)
    after = np.asarray(base)
    assert np.array_equal(before, after)


def test_the_grid_annotates_only_when_asked(tmp_path):
    from PIL import Image
    _, transform, shape = staged(tmp_path)
    plain = np.asarray(draw_ways(Image.new("RGB", (shape[1], shape[0]), (128, 128, 128)),
                                 [], transform, grid=False))
    gridded = np.asarray(draw_ways(Image.new("RGB", (shape[1], shape[0]), (128, 128, 128)),
                                   [], transform, grid=True))
    assert not np.array_equal(plain, gridded)


#: Where every fixture in this file sits: ORIGIN projected back out of Web
#: Mercator. A coordinate a caller receives must land in this box, or it is
#: not WGS84 at all - EPSG:3857 metres here read about -13,618,800 by
#: 4,486,600, which no longitude or latitude can be.
FIXTURE_LON = -122.34
FIXTURE_LAT = 37.40


def wgs84_positions(geometry):
    """Every (longitude, latitude) in a GeoJSON geometry, however nested.

    A LineString nests coordinates one deep, a Polygon two and a
    MultiPolygon three; walking to the numbers means one assertion covers
    whatever either tool emits.
    """
    def walk(node):
        if isinstance(node[0], (int, float)):
            yield (float(node[0]), float(node[1]))
            return
        for child in node:
            yield from walk(child)

    return list(walk(geometry["coordinates"]))


def assert_wgs84(geometry):
    """Plausible degrees on THIS fixture's ground, not merely finite numbers.

    Both tools measure in EPSG:3857 and used to emit it under a key named
    `geometry`, inside an envelope whose every other coordinate field is
    suffixed `_wgs84`. GeoJSON with no stated CRS is WGS84 by definition
    (RFC 7946), and the advertised next step - handing a patch to
    `site_facts`, which reprojects 4326 to 3857 - turned a real patch
    coordinate into `(inf, inf)`. Asserting the type of the geometry, which
    is all this file used to do, steps straight past that.
    """
    positions = wgs84_positions(geometry)
    assert positions, "no coordinates at all"
    for longitude, latitude in positions:
        assert abs(longitude - FIXTURE_LON) < 0.1, (
            f"longitude {longitude} is not WGS84 near the fixture")
        assert abs(latitude - FIXTURE_LAT) < 0.1, (
            f"latitude {latitude} is not WGS84 near the fixture")


def test_shaded_relief_works_with_no_token_at_all(tmp_path):
    """The credential asymmetry, as a test. The local hillshade, the overlay
    and the whole lineament path come from dem.tif and osm.json."""
    from sar_geo_mcp.planning.rasters import shaded_relief

    directory, _, _ = staged(tmp_path, rows=160, cols=160)
    result = shaded_relief(directory, wms_base_url=None, cache_dir=None)

    assert result["images"]["hillshade"] is not None
    assert result["images"]["hillshade_osm"] is not None
    assert result["images"]["relief"] is None
    assert any("token" in warning.lower() for warning in result["warnings"])


def test_shaded_relief_finds_a_bench_nobody_mapped(tmp_path):
    from sar_geo_mcp.planning.rasters import shaded_relief

    rows = cols = 160
    elevation = np.tile(np.linspace(0.0, 80.0, cols), (rows, 1)).astype("float64")
    for row in range(rows):
        elevation[row, 80:84] = elevation[row, 80]
    ox, oy = ORIGIN
    transform = from_origin(ox, oy, 1.0, 1.0)
    write_fetch_dir(tmp_path, elevation.astype("float32"), transform,
                    make_wgs84_square(-122.35, 37.40, -122.34, 37.41))

    result = shaded_relief(tmp_path, wms_base_url=None, cache_dir=None)
    assert len(result["lineaments"]) >= 1
    assert result["lineaments"][0]["geometry"]["type"] == "LineString"
    assert "unverified" in json.dumps(result["assumptions"]).lower()
    # Not merely a LineString: a LineString a team can navigate to.
    for lineament in result["lineaments"]:
        assert_wgs84(lineament["geometry"])


def test_every_lineament_geometry_is_wgs84_degrees(tmp_path):
    """The CRS seam, asserted on its own so it cannot be lost in a rewrite.

    Measured before the fix: `lineaments[0].geometry.coordinates[:2]` came
    back as `((-13618745.5, 4486605.0), (-13618745.5, 4486585.0))` - Web
    Mercator metres in a field every consumer reads as degrees.
    """
    from sar_geo_mcp.planning.rasters import shaded_relief

    rows = cols = 160
    elevation = np.tile(np.linspace(0.0, 80.0, cols), (rows, 1)).astype("float64")
    for row in range(rows):
        elevation[row, 80:84] = elevation[row, 80]
    ox, oy = ORIGIN
    transform = from_origin(ox, oy, 1.0, 1.0)
    write_fetch_dir(tmp_path, elevation.astype("float32"), transform,
                    make_wgs84_square(-122.35, 37.40, -122.34, 37.41))

    result = shaded_relief(tmp_path, wms_base_url=None, cache_dir=None,
                           returns="data")
    assert result["lineaments"]
    first = wgs84_positions(result["lineaments"][0]["geometry"])[0]
    assert abs(first[0] - FIXTURE_LON) < 0.1 and abs(first[1] - FIXTURE_LAT) < 0.1


def test_returns_data_writes_no_images(tmp_path):
    from sar_geo_mcp.planning.rasters import shaded_relief

    directory, _, _ = staged(tmp_path, rows=160, cols=160)
    result = shaded_relief(directory, wms_base_url=None, cache_dir=None, returns="data")
    assert all(value is None for value in result["images"].values())


def test_returns_image_still_carries_the_georeference(tmp_path):
    from sar_geo_mcp.planning.rasters import shaded_relief

    directory, _, _ = staged(tmp_path, rows=160, cols=160)
    result = shaded_relief(directory, wms_base_url=None, cache_dir=None, returns="image")
    assert result["georeference"]["north_up"] is True


def test_returns_image_reports_lineaments_as_null_not_as_none_found(tmp_path):
    """`lineaments: []` here said "this ground has no unmapped tracks" when
    nothing had been looked for - and the no-token warning went further,
    stating by name that the lineaments were unaffected."""
    from sar_geo_mcp.planning.rasters import shaded_relief

    directory, _, _ = staged(tmp_path, rows=160, cols=160)
    result = shaded_relief(directory, wms_base_url=None, cache_dir=None,
                           returns="image")

    assert result["lineaments"] is None
    assert result["lineament_summary"]["count"] is None
    assert result["lineament_summary"]["total_before_limit"] is None
    assert any("not looked for" in warning.lower() for warning in result["warnings"])
    # The no-token warning appears in this same mode and must no longer
    # claim the lineaments are unaffected by anything.
    assert not any("lineaments" in warning and "unaffected" in warning
                   for warning in result["warnings"])


def rough_hillside(tmp_path, size=600, amplitude=6.0, correlation=6):
    """Undisturbed ground with correlated roughness - the module's own
    documented worst realistic case, which INVENTS benches rather than
    fragmenting them. Nothing is cut into this hillside.
    """
    from scipy import ndimage

    rng = np.random.default_rng(5)
    base = np.tile(np.arange(size, dtype="float64") * 1.2, (size, 1))
    rough = ndimage.gaussian_filter(rng.normal(size=(size, size)), correlation)
    elevation = (base + rough * (amplitude / rough.std())).astype("float32")
    ox, oy = ORIGIN
    write_fetch_dir(tmp_path, elevation, from_origin(ox, oy, 1.0, 1.0),
                    make_wgs84_square(-122.35, 37.40, -122.34, 37.41))
    return tmp_path


def test_the_lineament_list_is_bounded_and_says_how_many_there_were(tmp_path):
    """Unbounded, sorted longest first, this returns the longest NOISE first.

    Measured at 1 m pixels on correlated-roughness hillsides: 1000^2 -> 53
    lineaments, 2000^2 -> 245, 4000^2 -> 971, against a staging ceiling of
    8000 px. `lineaments.py` says in its own docstring that the orchestration
    layer must not hand it a whole tile without bounding the count, and this
    tool did exactly that with no bound, count, density or warning.
    """
    from sar_geo_mcp.planning.rasters import shaded_relief

    directory = rough_hillside(tmp_path)

    every = shaded_relief(directory, wms_base_url=None, cache_dir=None,
                          returns="data", limit=0)
    total = every["lineament_summary"]["total_before_limit"]
    assert total > 3, "fixture too quiet to exercise truncation"
    assert total == len(every["lineaments"])
    assert every["lineament_summary"]["truncated_to"] is None

    capped = shaded_relief(directory, wms_base_url=None, cache_dir=None,
                           returns="data", limit=3)
    summary = capped["lineament_summary"]
    assert len(capped["lineaments"]) == 3
    assert summary["total_before_limit"] == total
    assert summary["count"] == 3
    assert summary["truncated_to"] == 3
    assert any("limit" in warning for warning in capped["warnings"])
    # Truncation keeps the longest, so the count is bounded and the list is
    # still the top of the ordering rather than an arbitrary slice.
    assert ([item["length_m"] for item in capped["lineaments"]]
            == [item["length_m"] for item in every["lineaments"][:3]])

    # The density is reported over the whole list, not over the truncated one:
    # "3 lineaments" would otherwise read as a quiet hillside.
    assert summary["ground_km2"] > 0
    assert abs(summary["per_km2"] - total / summary["ground_km2"]) < 0.1


def test_the_density_is_read_against_the_noise_floor_not_alone(tmp_path):
    """What turns "14 unmapped tracks" into "this is the noise floor".

    This fixture has nothing cut into it at all. `lineaments.py` measured
    about 60 lineaments per km2 on undisturbed ground with correlated
    roughness, so a run at or above that has found the method's floor.
    """
    from sar_geo_mcp.planning.rasters import NOISE_FLOOR_PER_KM2, shaded_relief

    result = shaded_relief(rough_hillside(tmp_path), wms_base_url=None,
                           cache_dir=None, returns="data", limit=0)
    summary = result["lineament_summary"]

    assert summary["noise_floor_per_km2"] == NOISE_FLOOR_PER_KM2
    assert summary["per_km2"] >= NOISE_FLOOR_PER_KM2
    assert any("noise floor" in warning for warning in result["warnings"])


def test_one_bench_on_a_small_patch_is_not_accused_of_being_noise(tmp_path):
    """The density figure is always reported; the accusation waits for a
    population. A single real bench on a 95 m square reads 110 per km2, and
    calling that a noise floor would be arithmetic rather than evidence.
    """
    from sar_geo_mcp.planning.rasters import shaded_relief

    size = 120
    elevation = np.tile(np.linspace(0.0, size * 0.5, size), (size, 1))
    for row in range(size):
        elevation[row, 60:64] = elevation[row, 60]
    ox, oy = ORIGIN
    write_fetch_dir(tmp_path, elevation.astype("float32"),
                    from_origin(ox, oy, 1.0, 1.0),
                    make_wgs84_square(-122.35, 37.40, -122.34, 37.41))

    result = shaded_relief(tmp_path, wms_base_url=None, cache_dir=None,
                           returns="data")
    summary = result["lineament_summary"]

    assert summary["total_before_limit"] == 1
    assert summary["per_km2"] > summary["noise_floor_per_km2"]
    assert not any("noise floor" in warning for warning in result["warnings"])


def test_a_negative_limit_is_refused_rather_than_clamped(tmp_path):
    import pytest

    from sar_geo_mcp.planning.rasters import shaded_relief

    directory, _, _ = staged(tmp_path, rows=64, cols=64)
    with pytest.raises(ValueError, match="limit"):
        shaded_relief(directory, wms_base_url=None, cache_dir=None,
                      returns="data", limit=-1)


def test_slope_shading_refuses_a_returns_it_does_not_know(tmp_path):
    """The MCP schema refuses `returns="images"` before this runs, but a
    scenario calling the feature function directly gets no schema check
    (AGENTS.md, rule 1). Unchecked, the typo meant "data" with
    `images` null and no error: a quietly reduced answer. `shaded_relief`
    already refused it in these words."""
    import pytest

    from sar_geo_mcp.planning.rasters import slope_shading

    directory, _, _ = staged(tmp_path)
    with pytest.raises(ValueError, match='returns must be "data", "image" or "both"'):
        slope_shading(directory, wms_base_url=None, cache_dir=None, returns="images")


def fake_panel(tmp_path, band_index, shape):
    """A PNG painted entirely in one legend colour, standing in for a fetch."""
    from PIL import Image
    from sar_geo_mcp.caltopo.shading import SF_LEGEND

    rows, cols = shape
    pixels = np.zeros((rows, cols, 4), dtype="uint8")
    pixels[:, :, :3] = SF_LEGEND[band_index].rgb
    pixels[:, :, 3] = 255
    path = tmp_path / "panel.png"
    Image.fromarray(pixels, mode="RGBA").save(path, format="PNG")
    return path


def test_without_a_token_it_returns_nulls_and_a_warning_never_zeros(tmp_path):
    """The test's own name is the rule, and `steeper_in_caltopo: []` broke it.

    Nothing was fetched, so nothing was compared, so nothing can be said
    about what disagrees. An empty list is a different claim - "compared,
    and the two sources agree everywhere" - which on a cliff band is the
    opposite answer from the truth.
    """
    from sar_geo_mcp.planning.rasters import slope_shading

    directory, _, _ = staged(tmp_path)
    result = slope_shading(directory, wms_base_url=None, cache_dir=None)

    assert result["caltopo"]["bands"] is None
    assert result["agreement_fraction"] is None
    assert result["steeper_in_caltopo"] is None
    assert any("token" in warning.lower() for warning in result["warnings"])


def test_the_dem_below_legend_share_is_reported_not_left_to_subtraction(tmp_path):
    """`dem.bands` summing to less than 1 with no explanation is the bug.

    Measured on this gentle fixture before the fix: `caltopo` bands summed
    to 1.0 with `below_legend_fraction` 0.3125, while `dem` bands summed to
    0.0 and nothing said where the other 100% went. "The DEM reads
    shallower than CalTopo" is the whole thesis of this tool, and its
    largest instance was the one number not reported.
    """
    from sar_geo_mcp.planning.rasters import slope_shading

    rows = cols = 64
    gentle = np.tile(np.linspace(0.0, 5.0, cols), (rows, 1)).astype("float32")
    ox, oy = ORIGIN
    transform = from_origin(ox, oy, PIXEL_M, PIXEL_M)
    write_fetch_dir(tmp_path, gentle, transform,
                    make_wgs84_square(-122.35, 37.40, -122.34, 37.41))
    panel = fake_panel(tmp_path, band_index=5, shape=(rows, cols))

    result = slope_shading(tmp_path, wms_base_url="https://example/wms",
                           cache_dir=tmp_path, fetch=lambda *a, **k: panel)

    dem_total = sum(result["dem"]["bands"].values())
    below = result["dem"]["below_legend_fraction"]
    assert below is not None
    # The denominator the assumptions promise: dem bands plus the DEM's own
    # below-legend share account for every comparable pixel.
    assert abs(dem_total + below - 1.0) < 1e-9
    assert abs(sum(result["caltopo"]["bands"].values()) - 1.0) < 1e-9
    assert "comparable" in result["assumptions"]["denominator"]


def test_caltopo_reading_steeper_than_the_dem_yields_patches(tmp_path):
    """The whole point: ground the 3DEP mosaic smoothed away."""
    from sar_geo_mcp.planning.rasters import slope_shading

    rows = cols = 64
    gentle = np.tile(np.linspace(0.0, 5.0, cols), (rows, 1)).astype("float32")
    ox, oy = ORIGIN
    transform = from_origin(ox, oy, PIXEL_M, PIXEL_M)
    write_fetch_dir(tmp_path, gentle, transform,
                    make_wgs84_square(-122.35, 37.40, -122.34, 37.41))

    panel = fake_panel(tmp_path, band_index=5, shape=(rows, cols))
    result = slope_shading(tmp_path, wms_base_url="https://example/wms",
                           cache_dir=tmp_path,
                           fetch=lambda *args, **kwargs: panel)

    assert result["steeper_in_caltopo"]
    assert result["steeper_in_caltopo"][0]["band_delta"] >= 1
    assert result["agreement_fraction"] < 0.5


def test_every_disagreement_patch_is_wgs84_degrees(tmp_path):
    """`steeper_in_caltopo` is the deliverable, and the spec says WGS84.

    Measured before the fix:
    `steeper_in_caltopo[0].geometry.coordinates[0][0]` came back as
    `(-13618786.5, 4486605.0)`. `slope_shading`'s own docstring advertises
    handing a patch to `cliff_pitch` or `site_facts`, and `site_facts`
    reprojects its `geometry` from 4326 to 3857 - which on that coordinate
    returns `(inf, inf)`. The advertised workflow could not work.
    """
    from sar_geo_mcp.planning.rasters import slope_shading

    rows = cols = 64
    gentle = np.tile(np.linspace(0.0, 5.0, cols), (rows, 1)).astype("float32")
    ox, oy = ORIGIN
    transform = from_origin(ox, oy, PIXEL_M, PIXEL_M)
    write_fetch_dir(tmp_path, gentle, transform,
                    make_wgs84_square(-122.35, 37.40, -122.34, 37.41))

    panel = fake_panel(tmp_path, band_index=5, shape=(rows, cols))
    result = slope_shading(tmp_path, wms_base_url="https://example/wms",
                           cache_dir=tmp_path,
                           fetch=lambda *args, **kwargs: panel)

    assert result["steeper_in_caltopo"]
    for patch in result["steeper_in_caltopo"]:
        assert_wgs84(patch["geometry"])

    # And the round trip the docstring promises actually completes: the
    # projection `site_facts` runs on a `geometry` it is handed.
    from sar_geo_mcp.terrain.grid import project_geometry_to_3857

    projected = project_geometry_to_3857(result["steeper_in_caltopo"][0]["geometry"])
    assert projected.is_valid and not projected.is_empty
    assert all(abs(value) < 1e9 for value in projected.bounds)


def test_an_unreadable_palette_announces_itself(tmp_path):
    """A wrong legend must not mis-band confidently: it must say so."""
    from PIL import Image
    from sar_geo_mcp.planning.rasters import slope_shading

    directory, _, shape = staged(tmp_path)
    pixels = np.zeros((shape[0], shape[1], 4), dtype="uint8")
    pixels[:, :, :3] = (17, 99, 44)  # in no legend
    pixels[:, :, 3] = 255
    path = tmp_path / "odd.png"
    Image.fromarray(pixels, mode="RGBA").save(path, format="PNG")

    result = slope_shading(directory, wms_base_url="https://example/wms",
                           cache_dir=tmp_path, fetch=lambda *a, **k: path)

    assert result["caltopo"]["unmatched_fraction"] > 0.9
    assert any("unmatched" in warning.lower() for warning in result["warnings"])
    assert "unverified" in result["legend"]["provenance"].lower()
    # Nothing decoded to a band, so nothing was compared: null, not an empty
    # list, which would read as "compared, and the two sources agree".
    assert result["steeper_in_caltopo"] is None
    assert result["dem"]["below_legend_fraction"] is None
    # And the warning no longer accuses the legend alone - an interpolating
    # resample raises this figure on a perfectly correct legend too.
    unmatched_warning = next(w for w in result["warnings"] if "unmatched" in w)
    assert "resampled" in unmatched_warning


def test_a_narrow_patch_survives_ranking(tmp_path):
    """The regression that matters: an area floor hides vertical faces. A
    patch one pixel wide and forty long is exactly what must not be lost."""
    from PIL import Image
    from sar_geo_mcp.caltopo.shading import SF_LEGEND
    from sar_geo_mcp.planning.rasters import slope_shading

    rows = cols = 64
    gentle = np.zeros((rows, cols), dtype="float32")
    ox, oy = ORIGIN
    transform = from_origin(ox, oy, PIXEL_M, PIXEL_M)
    write_fetch_dir(tmp_path, gentle, transform,
                    make_wgs84_square(-122.35, 37.40, -122.34, 37.41))

    pixels = np.zeros((rows, cols, 4), dtype="uint8")
    pixels[:, :, 3] = 0                      # below legend everywhere
    pixels[10:50, 30, :3] = SF_LEGEND[5].rgb  # one pixel wide, forty long
    pixels[10:50, 30, 3] = 255
    path = tmp_path / "narrow.png"
    Image.fromarray(pixels, mode="RGBA").save(path, format="PNG")

    result = slope_shading(tmp_path, wms_base_url="https://example/wms",
                           cache_dir=tmp_path, fetch=lambda *a, **k: path)

    assert len(result["steeper_in_caltopo"]) == 1


def test_image_mode_outlines_the_disagreement(tmp_path):
    """The patches can be looked at, not only counted: a scatter of small
    patches down one gully reads differently from one solid face."""
    from sar_geo_mcp.planning.rasters import slope_shading

    rows = cols = 64
    gentle = np.tile(np.linspace(0.0, 5.0, cols), (rows, 1)).astype("float32")
    ox, oy = ORIGIN
    transform = from_origin(ox, oy, PIXEL_M, PIXEL_M)
    write_fetch_dir(tmp_path, gentle, transform,
                    make_wgs84_square(-122.35, 37.40, -122.34, 37.41))
    panel = fake_panel(tmp_path, band_index=5, shape=(rows, cols))

    result = slope_shading(tmp_path, wms_base_url="https://example/wms",
                           cache_dir=tmp_path, returns="both",
                           fetch=lambda *a, **k: panel)

    assert result["images"]["panel"] is not None
    outlined = result["images"]["disagreement"]
    assert outlined is not None

    from PIL import Image
    with Image.open(outlined) as image:
        assert image.size == (cols, rows)
        assert np.any(np.all(np.asarray(image.convert("RGB")) == (0, 255, 128),
                             axis=2))


def _token_bearing_401():
    """What httpx raises for a refused GetMap: its message carries the full
    URL, and a CalTopo WMS URL carries the account token in its path."""
    request = httpx.Request("GET", "https://caltopo.example/SECRET-TOKEN/wms?LAYERS=r3")
    response = httpx.Response(401, request=request)
    return httpx.HTTPStatusError(
        f"Client error '401 Unauthorized' for url '{request.url}'",
        request=request, response=response)


def test_a_failed_relief_fetch_is_a_warning_never_a_token(tmp_path, monkeypatch):
    """`shaded_relief` has no `fetch=` of its own - it imports `fetch_panel`
    from `caltopo.wms` at call time - so the failure is injected the way a
    caller with no injection point still can: patch it where it is defined,
    before the tool's own local import reads it."""
    from sar_geo_mcp.planning.rasters import shaded_relief

    def raising_fetch_panel(*args, **kwargs):
        raise _token_bearing_401()

    monkeypatch.setattr("sar_geo_mcp.caltopo.wms.fetch_panel", raising_fetch_panel)

    directory, _, _ = staged(tmp_path, rows=160, cols=160)
    result = shaded_relief(directory, wms_base_url="https://caltopo.example/wms",
                           cache_dir=tmp_path / "cache", returns="image")

    assert result["images"]["relief"] is None
    assert result["images"]["relief_osm"] is None
    # The clean hillshade path is untouched by a relief-panel failure.
    assert result["images"]["hillshade"] is not None
    assert any("401" in warning for warning in result["warnings"])
    everything = json.dumps(result)
    assert "401" in everything and "SECRET-TOKEN" not in everything


def test_a_failed_slope_panel_fetch_is_a_warning_never_a_token(tmp_path):
    from sar_geo_mcp.planning.rasters import slope_shading

    def raising_fetch(*args, **kwargs):
        raise _token_bearing_401()

    directory, _, _ = staged(tmp_path)
    result = slope_shading(directory, wms_base_url="https://caltopo.example/wms",
                           cache_dir=tmp_path, fetch=raising_fetch)

    # The same "not compared" shape a missing token returns.
    assert result["caltopo"]["bands"] is None
    assert result["agreement_fraction"] is None
    assert result["steeper_in_caltopo"] is None
    assert any("401" in warning for warning in result["warnings"])
    everything = json.dumps(result)
    assert "401" in everything and "SECRET-TOKEN" not in everything
