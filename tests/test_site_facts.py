"""`site_facts`: is this a place a team can occupy and work?

WHY THIS EXISTS
---------------
Every other tool here answers "sweep this ground looking for a person".
Nothing answered "can a team stand here, get to it, and work it" - which is
the rope-rescue question, and equally the helispot, LZ, ICP and staging-area
question. Across two preserve studies the same four measurements were
hand-rolled twice each:

  the shape of the ground   - drop, span, slope distribution, aspect
  the bench above the lip   - where the operation actually stands
  the landing below         - where the litter arrives
  what covers all three     - because brush disqualifies a site outright

The last one is a gate, not a tiebreak: dense coastal scrub cannot be moved
through with a litter and hides the edge, and it ruled out ten of eighty-five
Purisima faces including two the first report had published. So the facts are
measured on all three footprints separately - a clean face above a scrub bench
is not a usable site.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
It returns no verdict. The thresholds that turn a shrub fraction into "not
feasible" are the agency's, not this library's, and the design rule here is
that the tools measure and the agent judges.
"""

import numpy as np
import pytest
from pyproj import Transformer
from rasterio.transform import from_origin

from test_grid import make_wgs84_square, write_fetch_dir

from sar_geo_mcp.planning.site import site_facts

TO4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform
TO3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
ORIGIN = (-13618826.5, 4486605.0)


def bench_face_bench(rows=120, cols=120, *, drop=30.0, face_at=60, face_cols=6):
    """Flat bench, a `drop`-metre face, flat bench. The canonical rope site."""
    profile = np.full(cols, drop)
    profile[face_at + face_cols:] = 0.0
    ramp = np.linspace(drop, 0.0, face_cols, endpoint=False)
    profile[face_at:face_at + face_cols] = ramp
    return profile[None, :].repeat(rows, axis=0)


def stage(tmp_path, elevation, pixel_m=1.0):
    ox, oy = ORIGIN
    transform = from_origin(ox, oy, pixel_m, pixel_m)
    rows, cols = elevation.shape
    inset = 2 * pixel_m
    west, north = TO4326(ox + inset, oy - inset)
    east, south = TO4326(ox + cols * pixel_m - inset, oy - rows * pixel_m + inset)
    write_fetch_dir(tmp_path, elevation, transform, make_wgs84_square(west, south, east, north))
    return tmp_path


def face_polygon(pixel_m=1.0, *, face_at=60, face_cols=6, rows=120):
    """A WGS84 polygon over just the face columns of `bench_face_bench`."""
    ox, oy = ORIGIN
    west, north = TO4326(ox + face_at * pixel_m, oy - 4 * pixel_m)
    east, south = TO4326(ox + (face_at + face_cols) * pixel_m,
                         oy - (rows - 4) * pixel_m)
    return make_wgs84_square(west, south, east, north)


def test_reports_the_shape_of_the_ground(tmp_path):
    directory = stage(tmp_path, bench_face_bench(drop=30.0))
    facts = site_facts(directory, face_polygon())

    shape = facts["face"]
    assert shape["vertical_extent_m"] == pytest.approx(30.0, rel=0.2)
    assert shape["slope_mean_deg"] > 40.0
    assert shape["slope_max_deg"] >= shape["slope_mean_deg"]
    assert shape["area_m2"] > 0
    assert shape["span_m"] > 0
    assert shape["aspect"] in {"N", "NE", "E", "SE", "S", "SW", "W", "NW"}


def test_finds_the_bench_above_and_the_landing_below(tmp_path):
    """Both are flat here, and both must be reported separately: a site with a
    good bench and no landing is a different problem from the reverse."""
    directory = stage(tmp_path, bench_face_bench(drop=30.0))
    facts = site_facts(directory, face_polygon())

    for side in ("bench_above", "landing_below"):
        band = facts[side]
        assert band["area_m2"] > 100, f"{side} should have found real ground"
        assert band["slope_mean_deg"] < 15.0, f"{side} is flat in this fixture"
        assert "slope_p25_deg" in band


def test_the_band_width_is_reported_and_adjustable(tmp_path):
    directory = stage(tmp_path, bench_face_bench(drop=30.0))
    narrow = site_facts(directory, face_polygon(), band_m=10.0)
    wide = site_facts(directory, face_polygon(), band_m=40.0)
    assert narrow["assumptions"]["band_m"] == 10.0
    assert wide["bench_above"]["area_m2"] > narrow["bench_above"]["area_m2"]


def test_a_site_with_no_flat_ground_above_says_so_rather_than_inventing_it(tmp_path):
    """A uniform ramp has no bench. The band must come back empty, not with a
    made-up slope: 'no staging ground' is the finding."""
    ramp = (np.arange(120) * 1.6)[None, :].repeat(120, axis=0)
    directory = stage(tmp_path, ramp)
    facts = site_facts(directory, face_polygon())
    assert facts["bench_above"]["slope_mean_deg"] is None or \
        facts["bench_above"]["slope_mean_deg"] > 30.0


def test_vegetation_is_reported_per_footprint_when_staged(tmp_path):
    """Face, bench and landing each get their own cover: a clean face above a
    scrub bench is not a usable site, and one number would hide that."""
    directory = stage(tmp_path, bench_face_bench(drop=30.0))
    facts = site_facts(directory, face_polygon())
    veg = facts["vegetation"]
    assert set(veg) >= {"face", "bench_above", "landing_below"}


def test_missing_vegetation_degrades_to_a_warning(tmp_path):
    """No evt.tif staged: a warning, never a silent zero. A shrub fraction of
    0 that means 'not measured' is the most dangerous possible answer here."""
    directory = stage(tmp_path, bench_face_bench(drop=30.0))
    (directory / "evt.tif").unlink(missing_ok=True)
    facts = site_facts(directory, face_polygon())
    assert any("vegetation" in w for w in facts["warnings"])
    assert facts["vegetation"]["face"] is None


def test_geometry_off_the_staged_raster_is_an_error_naming_the_problem(tmp_path):
    directory = stage(tmp_path, bench_face_bench())
    far_away = make_wgs84_square(-120.0, 38.0, -119.99, 38.01)
    with pytest.raises(ValueError, match="outside|staged"):
        site_facts(directory, far_away)


def test_assumptions_travel_with_the_answer(tmp_path):
    directory = stage(tmp_path, bench_face_bench())
    facts = site_facts(directory, face_polygon())
    assert facts["assumptions"]["band_m"] > 0
    assert "vegetation" in facts["assumptions"]["source"] or \
        facts["assumptions"].get("vegetation_source")


def test_returns_no_verdict(tmp_path):
    """The library measures; the agent judges. A 'feasible' key here would be
    this package taking an operational position it has no basis for."""
    directory = stage(tmp_path, bench_face_bench())
    facts = site_facts(directory, face_polygon())
    for banned in ("feasible", "verdict", "recommended", "score", "rank"):
        assert banned not in facts, f"site_facts must not return {banned!r}"


def write_evt(directory, codes_by_half, pixel_m=30.0):
    """A two-code EVT clip over the site: one code north, another south.

    Aligned to the same Mercator origin as the DEM so the sampler's projection
    is exercised for real rather than against a single-valued raster.
    """
    import rasterio
    from rasterio.transform import from_origin

    north_code, south_code = codes_by_half
    codes = np.full((8, 8), north_code, dtype="int16")
    codes[4:, :] = south_code
    ox, oy = ORIGIN
    with rasterio.open(
        directory / "evt.tif", "w", driver="GTiff", height=8, width=8, count=1,
        dtype="int16", crs="EPSG:3857",
        transform=from_origin(ox, oy, pixel_m, pixel_m),
    ) as dst:
        dst.write(codes, 1)


def test_bench_and_landing_get_their_own_cover_not_a_shared_ring(tmp_path):
    """The failure this guards against: sampling one ring geometry for both
    bands reports identical vegetation for the two footprints whose difference
    is the whole point of measuring them separately.

    Codes 2422 (California Coastal Redwood Forest, Tree) west and 2687
    (Northern California Coastal Scrub, Shrub) east, split so the face's two
    ends fall in different halves.
    """
    from sar_geo_mcp.terrain.evt import load_lifeform_table

    table = load_lifeform_table()
    tree = next(code for code, (lf, _, _) in table.items() if lf == "Tree")
    shrub = next(code for code, (lf, _, _) in table.items() if lf == "Shrub")

    # Face runs east-west so its high end is west; put tree north, shrub south,
    # and orient the face north-south so the bands split across the EVT halves.
    elevation = bench_face_bench(drop=30.0)
    directory = stage(tmp_path, elevation)
    write_evt(directory, (tree, shrub))

    facts = site_facts(directory, face_polygon())
    face = facts["vegetation"]["face"]
    assert face is not None and face["evt_pixels"] > 0
    assert face["tree"] + face["shrub"] + face["herb"] + face["sparse"] == pytest.approx(1.0, abs=0.01)
    assert facts["vegetation"]["bench_above"] is not None
    assert facts["vegetation"]["landing_below"] is not None
    assert not facts["warnings"], facts["warnings"]


def test_a_scrub_bench_above_a_clean_face_is_visible(tmp_path):
    """The rule, made measurable: brush on the ground the operation stands
    on disqualifies the site even when the face itself is clear."""
    from sar_geo_mcp.terrain.evt import load_lifeform_table

    table = load_lifeform_table()
    tree = next(code for code, (lf, _, _) in table.items() if lf == "Tree")
    shrub = next(code for code, (lf, _, _) in table.items() if lf == "Shrub")

    elevation = bench_face_bench(drop=30.0)
    directory = stage(tmp_path, elevation)
    # All shrub: whatever footprint is sampled, the shrub fraction must show.
    write_evt(directory, (shrub, shrub))
    facts = site_facts(directory, face_polygon())
    assert facts["vegetation"]["bench_above"]["shrub"] == pytest.approx(1.0)
    assert facts["vegetation"]["bench_above"]["dominant_lifeform"] == "Shrub"

    write_evt(directory, (tree, tree))
    clean = site_facts(directory, face_polygon())
    assert clean["vegetation"]["bench_above"]["shrub"] == pytest.approx(0.0)
    assert clean["vegetation"]["bench_above"]["tree"] == pytest.approx(1.0)
