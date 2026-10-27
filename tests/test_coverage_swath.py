import numpy as np
import pytest
from shapely.geometry import LineString, Polygon

from sar_geo_mcp.coverage.swath import (
    MIN_GAP_M2,
    CoverageGrid,
    build_grid,
    burn_swath,
    close_gaps,
    find_gaps,
)


def square(size=100.0):
    return Polygon([(0, 0), (size, 0), (size, size), (0, size)])


def test_grid_cells_are_square_and_cover_the_zone():
    grid = build_grid(square(100), cell_ground_m=2.0, ground_factor=1.0)

    assert isinstance(grid, CoverageGrid)
    assert grid.zone_mask.shape == (50, 50)
    assert grid.zone_mask.all()


def test_a_track_across_the_middle_covers_a_band():
    grid = build_grid(square(100), cell_ground_m=2.0, ground_factor=1.0)
    line = LineString([(0, 50), (100, 50)])

    swath = burn_swath(grid, [line], sweep_width_ground_m=10.0)

    # A 10 m wide band across a 100 m square is a tenth of the area. The
    # distance transform realises the sweep width faithfully (no dilation
    # overshoot), so this must land close to 0.10.
    assert swath.sum() / grid.zone_mask.sum() == pytest.approx(0.10, abs=0.02)


def test_the_swath_never_escapes_the_zone():
    grid = build_grid(square(100), cell_ground_m=2.0, ground_factor=1.0)
    outside = LineString([(-500, 50), (500, 50)])

    swath = burn_swath(grid, [outside], sweep_width_ground_m=10.0)

    assert not (swath & ~grid.zone_mask).any()


def test_overlapping_tracks_do_not_double_count():
    grid = build_grid(square(100), cell_ground_m=2.0, ground_factor=1.0)
    line = LineString([(0, 50), (100, 50)])

    once = burn_swath(grid, [line], sweep_width_ground_m=10.0)
    twice = burn_swath(grid, [line, line], sweep_width_ground_m=10.0)

    assert twice.sum() == once.sum()


def test_a_wider_sweep_covers_more():
    grid = build_grid(square(100), cell_ground_m=2.0, ground_factor=1.0)
    line = LineString([(0, 50), (100, 50)])

    narrow = burn_swath(grid, [line], sweep_width_ground_m=6.0)
    wide = burn_swath(grid, [line], sweep_width_ground_m=20.0)

    assert wide.sum() > narrow.sum()


def test_no_rounding_cliff_between_adjacent_esw_values():
    # Disc dilation used radius = round((esw/2)/cell): ESW 13.9 rounded to
    # radius 3 -> a realised 12 m swath, and ESW 14.0 rounded to radius 4 ->
    # 16 m - a cliff between two ESW values 0.1 m apart. A distance
    # transform has no such rounding step, so adjacent ESWs must realise
    # nearly identical swath areas.
    grid = build_grid(square(200), cell_ground_m=2.0, ground_factor=1.0)
    line = LineString([(0, 100), (200, 100)])

    just_below = burn_swath(grid, [line], sweep_width_ground_m=13.9)
    just_above = burn_swath(grid, [line], sweep_width_ground_m=14.1)

    assert just_above.sum() == pytest.approx(just_below.sum(), rel=0.05)


def test_a_fully_covered_zone_has_no_gaps():
    grid = build_grid(square(100), cell_ground_m=2.0, ground_factor=1.0)
    swath = grid.zone_mask.copy()

    assert find_gaps(grid, swath) == []


def test_one_unsearched_half_is_one_gap():
    grid = build_grid(square(100), cell_ground_m=2.0, ground_factor=1.0)
    swath = grid.zone_mask.copy()
    swath[:, 25:] = False          # right half unsearched

    gaps = find_gaps(grid, swath)

    assert len(gaps) == 1
    assert gaps[0].area_m2 == pytest.approx(5000, rel=0.1)
    assert gaps[0].id == "gap_0"


def test_slivers_below_the_threshold_are_dropped():
    grid = build_grid(square(100), cell_ground_m=2.0, ground_factor=1.0)
    swath = grid.zone_mask.copy()
    swath[0, 0] = False            # a single 4 m2 cell

    assert find_gaps(grid, swath) == []


def test_gaps_are_ordered_largest_first():
    grid = build_grid(square(100), cell_ground_m=2.0, ground_factor=1.0)
    swath = grid.zone_mask.copy()
    swath[0:20, 0:20] = False      # 20x20 cells * 4 m2 = 1600 m2 -> kept
    swath[40:45, 40:50] = False    # 5x10 cells * 4 m2 = 200 m2 -> below 500, dropped

    gaps = find_gaps(grid, swath)

    assert [g.id for g in gaps] == ["gap_0"]
    assert gaps[0].area_m2 == pytest.approx(1600, rel=0.1)


def test_gap_geometry_is_returned_for_mapping():
    grid = build_grid(square(100), cell_ground_m=2.0, ground_factor=1.0)
    swath = grid.zone_mask.copy()
    swath[:, 25:] = False

    gap = find_gaps(grid, swath)[0]

    assert gap.geometry["type"] in ("Polygon", "MultiPolygon")
    assert MIN_GAP_M2 == 500.0


# --- GPS-noise gap closing (Phase 9i), 2-D analogue of corridor.py -------
#
# Same physics as the line-assignment rule: jitter between adjacent sweep
# lanes can leave a thin strip of cells unswept even though both flanking
# lanes were genuinely walked. `close_gaps` binary-closes the swath (dilate
# then erode with a disc sized off the sweep width - see swath.py) so a
# sliver like that gets folded into "covered", while a real, much larger
# unswept block (QC1202's actual 6.4-acre gap) survives untouched.


def test_closing_fills_a_thin_sliver_between_two_lanes():
    grid = build_grid(square(100), cell_ground_m=2.0, ground_factor=1.0)
    swath = np.zeros_like(grid.zone_mask)
    swath[20:24, :] = True   # lane 1
    swath[25:29, :] = True   # lane 2 - row 24 (one 2 m cell) left unswept between them
    swath &= grid.zone_mask

    closed, closed_regions = close_gaps(grid, swath, sweep_width_ground_m=10.0)

    # Closing only ADDS cells - nothing already swept is ever removed.
    assert (swath & ~closed).sum() == 0
    # The sliver between the lanes is now covered, away from the raster's
    # own left/right edge (a disc has less to work with right at a corner -
    # not the scenario this test is about).
    assert closed[24, 5:-5].all()
    # ...and shows up as a discrete closed region, not silently vanished.
    assert len(closed_regions) >= 1
    assert sum(r.area_m2 for r in closed_regions) > 0
    for region in closed_regions:
        assert region.geometry["type"] in ("Polygon", "MultiPolygon")


def test_closing_does_not_swallow_a_real_unswept_block():
    # A synthetic stand-in for QC1202's real ~6.4-acre (~26000 m2) gap: a
    # zone with a large block that was simply never swept, tens of metres
    # across in every direction - far too big for a sweep-width-scale disc
    # to bridge. Built independent of anything under out/, per the task.
    grid = build_grid(square(400), cell_ground_m=2.0, ground_factor=1.0)
    swath = grid.zone_mask.copy()
    swath[50:130, 50:130] = False   # an 80x80 cell (160 m x 160 m) block, ~25600 m2

    before = find_gaps(grid, swath)
    assert len(before) == 1
    real_gap_area = before[0].area_m2

    closed, closed_regions = close_gaps(grid, swath, sweep_width_ground_m=10.5)
    after = find_gaps(grid, closed)

    assert len(after) == 1
    # Within a couple of percent - closing may nibble a cell or two at the
    # block's edge, but must not meaningfully shrink a genuine gap this size.
    assert after[0].area_m2 == pytest.approx(real_gap_area, rel=0.03)
    # None of the (tiny, edge-only) area closing did claim should overlap
    # the big real gap enough to register as swallowing it.
    assert real_gap_area > 20000


def test_closing_a_fully_covered_zone_changes_nothing():
    grid = build_grid(square(100), cell_ground_m=2.0, ground_factor=1.0)
    swath = grid.zone_mask.copy()

    closed, closed_regions = close_gaps(grid, swath, sweep_width_ground_m=10.0)

    assert (closed == swath).all()
    assert closed_regions == []


def test_closing_a_zero_coverage_zone_changes_nothing():
    # Nothing to dilate FROM, so closing cannot manufacture any coverage.
    grid = build_grid(square(100), cell_ground_m=2.0, ground_factor=1.0)
    swath = np.zeros_like(grid.zone_mask)

    closed, closed_regions = close_gaps(grid, swath, sweep_width_ground_m=10.0)

    assert not closed.any()
    assert closed_regions == []
