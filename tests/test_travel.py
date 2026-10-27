"""Walking time from the centre: the shared walk (planning.walking).

Built on test_boundary's 40x40 grid of 10 m pixels (7.952 ground metres each
at this latitude). Flat open ground with no LANDFIRE staged is Tobler at zero
slope, 6 * exp(-0.175) = 5.0388 km/h, times the shared off-way factor 0.6.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from shapely.geometry import LineString, box
from test_boundary import GROUND_PER_UNIT, ORIGIN_X, ORIGIN_Y, route, staged
from test_corridors import area_subject, creek, generate

from sar_geo_mcp.planning.boundary import load_ground
from sar_geo_mcp.planning.travel import TRAVEL_ASSUMPTIONS, travel_minutes_to, travel_surface

#: Tobler on the flat, km/h, and off a way (terrain.movement.OFF_WAY_FACTOR).
FLAT_KMH = 6.0 * np.exp(-3.5 * 0.05)
FLAT_OFF_WAY_KMH = FLAT_KMH * 0.6


def origin(x_units=200, y_units=200):
    from shapely.geometry import Point

    return Point(ORIGIN_X + x_units, ORIGIN_Y - y_units)


def wall_dem(rows=40, cols=40, columns=(15, 16, 17), height_per_cell=40.0):
    """Flat ground with a wall of near-vertical cells running north-south.

    A 40 m rise across a 7.952 m cell is 79 degrees; Tobler there is about
    1e-7 km/h, so crossing the wall costs hours where the flat costs minutes.
    The wall spans every row, so there is no way around it.
    """
    dem = np.zeros((rows, cols))
    profile = np.concatenate([np.arange(len(columns) + 1), np.arange(len(columns), 0, -1) - 1])
    for step, column in enumerate(range(columns[0], columns[0] + profile.size)):
        if column < cols:
            dem[:, column] = profile[step] * height_per_cell
    dem[:, columns[0] + profile.size:] = 0.0
    return dem


# --- the surface -------------------------------------------------------------------

def test_on_flat_ground_travel_minutes_track_straight_line_distance(tmp_path):
    """The sanity case. An 8-neighbourhood over-measures a diagonal by a few
    per cent, so the tolerance is loose; what must hold is that flat ground
    costs distance over the flat off-way rate and nothing else."""
    ground = load_ground(staged(tmp_path, np.zeros((40, 40))))
    surface = travel_surface(ground, origin())

    for x_units, y_units in ((200, 50), (200, 350), (50, 200), (120, 120)):
        axis = LineString([(ORIGIN_X + x_units - 5, ORIGIN_Y - y_units),
                           (ORIGIN_X + x_units + 5, ORIGIN_Y - y_units)])
        minutes = travel_minutes_to(surface, ground, axis)
        straight_m = axis.distance(origin()) * GROUND_PER_UNIT
        expected = straight_m / 1000.0 / FLAT_OFF_WAY_KMH * 60.0
        assert minutes == pytest.approx(expected, rel=0.12), (x_units, y_units)


def test_the_origins_own_cell_costs_nothing(tmp_path):
    ground = load_ground(staged(tmp_path, np.zeros((40, 40))))
    surface = travel_surface(ground, origin())
    assert surface[20, 20] == pytest.approx(0.0)
    assert np.isfinite(surface).all()


def test_a_band_of_steep_cells_makes_a_near_candidate_slower_than_a_far_one(tmp_path):
    """The whole point of the feature. A drainage 80 m away across a steep
    band is farther, in the only sense a planner acts on, than a trail 120 m
    away along the contour. The band (about 32 degrees) is walkable, slowly."""
    ground = load_ground(staged(tmp_path, wall_dem(height_per_cell=5.0)))
    surface = travel_surface(ground, origin())

    near = LineString([(ORIGIN_X + 100, ORIGIN_Y - 150), (ORIGIN_X + 100, ORIGIN_Y - 250)])
    far = LineString([(ORIGIN_X + 350, ORIGIN_Y - 150), (ORIGIN_X + 350, ORIGIN_Y - 250)])
    assert near.distance(origin()) < far.distance(origin())

    near_minutes = travel_minutes_to(surface, ground, near)
    far_minutes = travel_minutes_to(surface, ground, far)
    assert near_minutes is not None and far_minutes is not None
    assert near_minutes > far_minutes


def test_a_cliff_band_closes_the_ground_beyond_it(tmp_path):
    """A wall far over the impassable slope (79 degrees) running the whole
    raster: what lies past it is unreachable, None - as the rings close it."""
    ground = load_ground(staged(tmp_path, wall_dem()))
    surface = travel_surface(ground, origin())
    beyond = LineString([(ORIGIN_X + 100, ORIGIN_Y - 150), (ORIGIN_X + 100, ORIGIN_Y - 250)])
    assert travel_minutes_to(surface, ground, beyond) is None


def test_a_dem_void_is_impassable_not_free(tmp_path):
    """A void is ground nobody measured, so it is not a shortcut. A band of
    void across the raster cuts everything beyond it off: the fact is None,
    never a 0.0 standing in for "not measured"."""
    dem = np.zeros((40, 40))
    dem[:, 14:17] = -999999.0            # a full-height band of void
    ground = load_ground(staged(tmp_path, dem))
    assert np.isnan(ground.slope_deg[:, 15]).any()
    surface = travel_surface(ground, origin())

    beyond = LineString([(ORIGIN_X + 60, ORIGIN_Y - 150), (ORIGIN_X + 60, ORIGIN_Y - 250)])
    this_side = LineString([(ORIGIN_X + 300, ORIGIN_Y - 150), (ORIGIN_X + 300, ORIGIN_Y - 250)])

    assert travel_minutes_to(surface, ground, beyond) is None
    assert travel_minutes_to(surface, ground, this_side) is not None


def test_a_geometry_off_the_raster_is_none_not_zero(tmp_path):
    ground = load_ground(staged(tmp_path, np.zeros((40, 40))))
    surface = travel_surface(ground, origin())
    surface[:] = np.inf
    axis = LineString([(ORIGIN_X + 100, ORIGIN_Y - 200), (ORIGIN_X + 300, ORIGIN_Y - 200)])
    assert travel_minutes_to(surface, ground, axis) is None


def test_a_corridor_is_measured_on_the_cells_it_covers(tmp_path):
    """A corridor is rasterised in its own window, the way candidate_effort
    does it, so the minutes are the minimum over the cells it occupies."""
    ground = load_ground(staged(tmp_path, np.zeros((40, 40))))
    surface = travel_surface(ground, origin())
    ribbon = box(ORIGIN_X + 50, ORIGIN_Y - 65, ORIGIN_X + 350, ORIGIN_Y - 35)

    minutes = travel_minutes_to(surface, ground, ribbon)

    from rasterio.features import rasterize
    mask = rasterize([(ribbon, 1)], out_shape=surface.shape,
                     transform=ground.grid.transform, fill=0, all_touched=False).astype(bool)
    assert mask.any()
    assert minutes == pytest.approx(float(surface[mask].min()), rel=1e-12)


def test_the_surface_agrees_with_candidate_effort_on_the_rate_it_uses(tmp_path):
    """Both read terrain.movement. A flat off-way step of one cell costs
    exactly the cell width over the off-way rate a corridor's effort reports,
    so the two numbers in one payload cannot disagree about how fast a
    searcher walks."""
    from sar_geo_mcp.planning.effort import candidate_effort

    ground = load_ground(staged(tmp_path, np.zeros((40, 40))))
    surface = travel_surface(ground, origin())
    one_cell = float(surface[20, 21])
    corridor = box(ORIGIN_X + 100, ORIGIN_Y - 300, ORIGIN_X + 300, ORIGIN_Y - 100)
    rate = candidate_effort(ground, corridor, "corridor", evt_path=None)["rate_kmh"]
    assert rate == pytest.approx(FLAT_OFF_WAY_KMH, rel=1e-9)
    cell_m = ground.grid.pixel_size_m
    assert one_cell == pytest.approx(cell_m / 1000.0 / rate * 60.0, rel=1e-9)


def test_a_trail_is_quicker_than_the_ground_beside_it(tmp_path):
    """The shared walk sees mapped ways: the same distance along a trail
    costs the off-way factor less than across open ground."""
    trail = route("route_0", "TRAIL", [(0, 200), (400, 200)], name="Flat Trail")
    ground = load_ground(staged(tmp_path, np.zeros((40, 40)), routes=[trail]))
    surface = travel_surface(ground, origin(200, 195))     # on the trail's row, 19
    along = float(surface[19, 34])                         # 15 cells east, on it
    across = float(surface[34, 19])                        # 15 cells south, off it
    assert along == pytest.approx(across * 0.6, rel=0.1)


def test_the_surface_is_built_without_scikit_image(tmp_path, monkeypatch):
    """scikit-image is in this venv only transitively; it is not a declared
    dependency, so importing it here would break a clean install."""
    import builtins

    real_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] == "skimage":
            raise AssertionError("travel.py must not import scikit-image")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    ground = load_ground(staged(tmp_path, np.zeros((40, 40))))
    assert np.isfinite(travel_surface(ground, origin())).all()


# --- wired into the candidate facts ------------------------------------------------

def test_every_candidate_carries_travel_minutes_from_center(tmp_path):
    trail = route("route_0", "TRAIL", [(0, 100), (400, 100)], name="Contour Trail")
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[trail])
    area_subject(data_dir)

    cset = generate(data_dir, include=["trail"])

    value = cset.candidates[0].facts["travel_minutes_from_center"]
    assert value > 0
    # Rounded like every other fact: minutes get one decimal.
    assert value == round(value, 1)


def test_a_candidate_that_covers_the_centre_reports_zero_which_is_a_measurement(tmp_path):
    """Zero is only ever the honest answer for a candidate the centre stands
    on. It is never the stand-in for unreachable - that is None."""
    trail = route("route_0", "TRAIL", [(0, 200), (400, 200)], name="Through The PLS")
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[trail])
    area_subject(data_dir)

    cset = generate(data_dir, include=["trail"])

    assert cset.candidates[0].facts["travel_minutes_from_center"] == 0.0
    assert cset.candidates[0].facts["distance_from_center_m"] == 0.0


def test_the_set_states_the_travel_cost_model_beside_effort(tmp_path):
    trail = route("route_0", "TRAIL", [(0, 200), (400, 200)])
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[trail])
    area_subject(data_dir)

    assumptions = generate(data_dir, include=["trail"]).assumptions["travel"]

    assert "6 * exp" in assumptions["movement"]["tobler"]
    assert assumptions["movement"]["source"] == "sar_geo_mcp.terrain.movement"
    assert "no elevation" in assumptions["dem_void"].lower()
    assert "rank" in assumptions["not_a_route"].lower()
    assert assumptions == TRAVEL_ASSUMPTIONS


def test_the_surface_is_computed_once_per_call_not_once_per_candidate(tmp_path, monkeypatch):
    """It is the same origin for every candidate in a set. Rebuilding the
    graph per candidate is the whole cost of the run, a hundred times over."""
    from sar_geo_mcp.planning import corridors as corridors_module

    routes = [route(f"route_{i}", "TRAIL", [(0, 100 + 20 * i), (400, 100 + 20 * i)],
                    name=f"Trail {i}") for i in range(6)]
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=routes)
    area_subject(data_dir)

    calls = []
    real = corridors_module.travel_surface

    def counted(ground, origin_3857, warnings=None):
        calls.append(origin_3857)
        return real(ground, origin_3857, warnings)

    monkeypatch.setattr(corridors_module, "travel_surface", counted)
    cset = generate(data_dir, include=["trail"])

    assert len(cset.candidates) > 1
    assert len(calls) == 1


def test_an_unreachable_candidate_reports_none_not_zero(tmp_path):
    dem = np.zeros((40, 40))
    dem[:, 26:29] = -999999.0            # a void band between the centre and the trail
    trail = route("route_0", "TRAIL", [(320, 100), (320, 300)], name="Far Side")
    data_dir = staged(tmp_path, dem, routes=[trail])
    area_subject(data_dir, center_units=(200, 200), radius_units=180)

    cset = generate(data_dir, include=["trail"])

    assert len(cset.candidates) == 1
    assert cset.candidates[0].facts["travel_minutes_from_center"] is None


# --- min_length_m -------------------------------------------------------------------

def test_min_length_m_defaults_to_todays_minimum(tmp_path):
    """None keeps MIN_CANDIDATE_M exactly: a 36 m stub still comes back."""
    from sar_geo_mcp.planning.corridors import MIN_CANDIDATE_M

    stub_units = 45          # 45 * 0.7952 = 35.8 ground metres
    trail = route("route_0", "TRAIL", [(200, 200), (200 + stub_units, 200)], name="Stub")
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[trail])
    area_subject(data_dir)

    cset = generate(data_dir, include=["trail"])

    assert len(cset.candidates) == 1
    assert MIN_CANDIDATE_M < cset.candidates[0].facts["length_m"] < 100
    assert cset.params["min_length_m"] is None
    assert cset.assumptions["min_candidate_m"] == MIN_CANDIDATE_M


def test_min_length_m_drops_short_line_runs(tmp_path):
    stub = route("route_0", "TRAIL", [(200, 120), (245, 120)], name="Stub")
    run = route("route_1", "TRAIL", [(60, 200), (340, 200)], name="Long Run")
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[stub, run])
    area_subject(data_dir)

    everything = generate(data_dir, include=["trail"])
    assert {c.name for c in everything.candidates} == {"Stub", "Long Run"}

    limited = generate(data_dir, include=["trail"], min_length_m=100.0)

    assert [c.name for c in limited.candidates] == ["Long Run"]
    assert limited.assumptions["min_candidate_m"] == 100.0


def test_min_length_m_leaves_corridors_alone(tmp_path):
    """It replaces the LINE minimum only. A waterway corridor keeps
    MIN_CORRIDOR_AXIS_M."""
    from test_boundary import valley_dem

    data_dir = staged(tmp_path, valley_dem(), routes=[creek()])
    area_subject(data_dir)

    everything = generate(data_dir, include=["waterway"])
    limited = generate(data_dir, include=["waterway"], min_length_m=100000.0)

    assert len(everything.candidates) >= 1
    assert [c.facts["length_m"] for c in limited.candidates] == \
           [c.facts["length_m"] for c in everything.candidates]


def test_min_length_m_changes_the_set_id(tmp_path):
    trail = route("route_0", "TRAIL", [(60, 200), (340, 200)], name="Long Run")
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[trail])
    area_subject(data_dir)

    default = generate(data_dir, include=["trail"])
    asked = generate(data_dir, include=["trail"], min_length_m=593.0)
    again = generate(data_dir, include=["trail"], min_length_m=593.0)

    assert default.set_id != asked.set_id
    assert asked.set_id == again.set_id


@pytest.mark.parametrize("bad", [0, -1, -593.0])
def test_min_length_m_must_be_a_positive_distance(tmp_path, bad):
    trail = route("route_0", "TRAIL", [(60, 200), (340, 200)])
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[trail])
    area_subject(data_dir)
    with pytest.raises(ValueError, match="min_length_m"):
        generate(data_dir, include=["trail"], min_length_m=bad)


# --- through the tool ----------------------------------------------------------------

def test_the_saved_set_round_trips_the_new_fact(tmp_path):
    """The fact travels through the CandidateSet JSON a tool reads back, so a
    reused set carries it too."""
    from sar_geo_mcp.planning.candidates import load_set, save_set

    trail = route("route_0", "TRAIL", [(60, 200), (340, 200)], name="Long Run")
    data_dir = staged(tmp_path, np.zeros((40, 40)), routes=[trail])
    area_subject(data_dir)
    cset = generate(data_dir, include=["trail"])
    save_set(data_dir, cset)

    read_back = load_set(data_dir, cset.set_id)

    assert read_back.candidates[0].facts["travel_minutes_from_center"] == \
           cset.candidates[0].facts["travel_minutes_from_center"]
    assert read_back.params["min_length_m"] is None
    assert json.loads(read_back.model_dump_json())["assumptions"]["travel"]["movement"]["tobler"]


# --- a multi-part axis -------------------------------------------------------------

def test_a_two_part_axis_reports_the_minimum_over_its_parts(tmp_path):
    """A team reaches a forked assignment at its NEAREST point, whichever
    branch that is on. The near branch here is a 2-unit stub at the centre and
    it is the SECOND part, so reading the first part - or sampling the axis as
    one concatenated line, which lands no sample on a stub that short - reports
    the far branch's minutes instead of the answer."""
    from shapely.geometry import MultiLineString

    ground = load_ground(staged(tmp_path, np.zeros((40, 40))))
    surface = travel_surface(ground, origin())
    far = LineString([(ORIGIN_X + 20, ORIGIN_Y - 20), (ORIGIN_X + 316, ORIGIN_Y - 20)])
    near = LineString([(ORIGIN_X + 199, ORIGIN_Y - 200), (ORIGIN_X + 201, ORIGIN_Y - 200)])

    far_minutes = travel_minutes_to(surface, ground, far)
    near_minutes = travel_minutes_to(surface, ground, near)
    both = travel_minutes_to(surface, ground, MultiLineString([far, near]))

    assert near_minutes == pytest.approx(0.0, abs=1e-9)
    assert far_minutes > 1.0
    assert both == pytest.approx(min(far_minutes, near_minutes))
    assert both == pytest.approx(near_minutes)


def test_a_multi_part_axis_out_of_reach_on_every_part_is_null(tmp_path):
    """None, never 0.0: unreachable is not "at the centre"."""
    from shapely.geometry import MultiLineString

    dem = np.zeros((40, 40))
    dem[:, 10:12] = -999999.0            # a void wall with no way round it
    ground = load_ground(staged(tmp_path, dem))
    surface = travel_surface(ground, origin())
    west = MultiLineString([[(ORIGIN_X + 20, ORIGIN_Y - 100), (ORIGIN_X + 60, ORIGIN_Y - 100)],
                            [(ORIGIN_X + 20, ORIGIN_Y - 300), (ORIGIN_X + 60, ORIGIN_Y - 300)]])

    assert travel_minutes_to(surface, ground, west) is None
