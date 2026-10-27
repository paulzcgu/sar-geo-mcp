"""`boundary_clearance`: how much of the boundary's inside is around this, and
does the way in stay inside it.

WHY THIS EXISTS
---------------
A candidate in El Corte de Madera Creek OSP looked like the best site found:
28 m of drop at a 62 degree mean, eight minutes from a parking lot. It survived
slope, vegetation and distance checks. What disqualified it was land tenure,
and nothing in this package could ask:

  the patch was wholly inside the preserve, but only 47 m from its edge
  the road 76 m away was 0% inside the preserve - a private road
  the parking lot was outside the boundary too
  the straight line from that lot to the patch was 50% on preserve ground

Every one of those is a geometry question with no raster in it, which is why
this lives in `geo` rather than `planning` - the design doc's stateless
accessory family.
"""

import pytest

from sar_geo_mcp.geo import boundary_clearance

# A 0.02 x 0.02 degree box, about 1.8 x 2.2 km at this latitude.
BOUNDARY = {
    "type": "Polygon",
    "coordinates": [[[-122.34, 37.40], [-122.32, 37.40],
                     [-122.32, 37.42], [-122.34, 37.42], [-122.34, 37.40]]],
}


def square(west, south, east, north):
    return {"type": "Polygon",
            "coordinates": [[[west, south], [east, south], [east, north],
                             [west, north], [west, south]]]}


CENTRAL = square(-122.331, 37.409, -122.329, 37.411)
NEAR_EDGE = square(-122.3399, 37.409, -122.3396, 37.411)
STRADDLING = square(-122.341, 37.409, -122.339, 37.411)
OUTSIDE = square(-122.36, 37.409, -122.358, 37.411)


def test_a_central_geometry_is_inside_with_real_clearance():
    result = boundary_clearance(CENTRAL, BOUNDARY)
    assert result["within"] is True
    assert result["fraction_inside"] == pytest.approx(1.0)
    assert result["clearance_m"] > 300


def test_clearance_is_the_distance_to_the_nearest_edge_not_the_centroid():
    """The C3 case: wholly inside, but tens of metres from the line."""
    result = boundary_clearance(NEAR_EDGE, BOUNDARY)
    assert result["within"] is True
    assert 0 < result["clearance_m"] < 80, result


def test_a_straddling_geometry_reports_the_fraction_and_is_not_within():
    result = boundary_clearance(STRADDLING, BOUNDARY)
    assert result["within"] is False
    assert 0.2 < result["fraction_inside"] < 0.8
    assert result["clearance_m"] == 0.0


def test_a_geometry_wholly_outside_says_so_rather_than_returning_zero():
    result = boundary_clearance(OUTSIDE, BOUNDARY)
    assert result["within"] is False
    assert result["fraction_inside"] == pytest.approx(0.0)
    assert result["distance_to_boundary_m"] > 1000


def test_radial_clearance_reports_every_compass_point():
    """One number hides a narrow finger. C3 had 60 m south-east and 1.5 km
    east, and the mean of those describes no real site."""
    radial = boundary_clearance(NEAR_EDGE, BOUNDARY)["radial_clearance_m"]
    assert set(radial) == {"N", "NE", "E", "SE", "S", "SW", "W", "NW"}
    assert radial["W"] is not None and radial["W"] < 100
    assert radial["E"] is None or radial["E"] > radial["W"]


def test_an_access_point_outside_the_boundary_is_reported_as_such():
    """The lot that demoted C3 was itself outside the preserve."""
    result = boundary_clearance(CENTRAL, BOUNDARY, access=(37.4105, -122.345))
    approach = result["approach"]
    assert approach["access_inside_boundary"] is False
    assert 0.0 < approach["fraction_inside"] < 1.0
    assert approach["distance_m"] > 0


def test_an_approach_wholly_inside_reads_as_wholly_inside():
    result = boundary_clearance(CENTRAL, BOUNDARY, access=(37.4105, -122.3325))
    assert result["approach"]["fraction_inside"] == pytest.approx(1.0)
    assert result["approach"]["access_inside_boundary"] is True


def test_no_access_point_means_no_approach_block_rather_than_a_fabricated_one():
    assert boundary_clearance(CENTRAL, BOUNDARY)["approach"] is None


def test_the_approach_is_a_straight_line_and_says_so():
    """It flags a private-land risk; it does not resolve a parcel line, and a
    reader must not mistake it for a route."""
    result = boundary_clearance(CENTRAL, BOUNDARY, access=(37.4105, -122.345))
    assert "straight line" in result["assumptions"]["approach"]


def test_a_multipolygon_boundary_works():
    """Real preserve boundaries have holes and separate parcels."""
    multi = {"type": "MultiPolygon", "coordinates": [BOUNDARY["coordinates"]]}
    assert boundary_clearance(CENTRAL, multi)["within"] is True
