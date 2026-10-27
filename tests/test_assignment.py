import json
from pathlib import Path

import pytest

from sar_geo_mcp.caltopo.assignment import Assignment

FIXTURE = Path("fixtures/assignment_sample.json")


def sample_feature(**overrides) -> dict:
    properties = {
        "class": "Assignment",
        "letter": "QA",
        "number": "",
        "title": "QA",
        "description": "Search the drainage from the road to the ridge.",
        "resourceType": "GROUND",
        "priority": "HIGH",
        "operationalPeriodId": "op-2",
        "status": "DRAFT",
        "responsivePOD": "LOW",
        "unresponsivePOD": "LOW",
        "cluePOD": "LOW",
        "teamSize": "4",
        "timeAllocated": "6",
        "transportation": "VEHICLE",
        "previousEfforts": "none",
    }
    properties.update(overrides)
    return {
        "id": "feat-123",
        "properties": properties,
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [-122.34, 37.30],
                    [-122.30, 37.30],
                    [-122.30, 37.34],
                    [-122.34, 37.34],
                    [-122.34, 37.30],
                ]
            ],
        },
    }


def test_parses_core_fields():
    a = Assignment.from_feature(sample_feature())

    assert a.id == "feat-123"
    assert a.letter == "QA"
    assert a.description.startswith("Search the drainage")
    assert a.resource_type == "GROUND"
    assert a.priority == "HIGH"


def test_label_combines_letter_and_number():
    a = Assignment.from_feature(sample_feature(letter="QA", number="3"))

    assert a.label == "QA3"


def test_label_falls_back_to_title():
    a = Assignment.from_feature(sample_feature(letter="", number="", title="North Ridge"))

    assert a.label == "North Ridge"


def test_missing_values_become_empty_strings_not_none():
    feature = sample_feature()
    del feature["properties"]["priority"]

    a = Assignment.from_feature(feature)

    assert a.priority == ""


def test_null_description_becomes_empty_string():
    a = Assignment.from_feature(sample_feature(description=None))

    assert a.description == ""


def test_rejects_non_assignment_feature():
    feature = sample_feature()
    feature["properties"]["class"] = "Marker"

    with pytest.raises(ValueError, match="not an Assignment"):
        Assignment.from_feature(feature)


def test_rejects_feature_without_geometry():
    feature = sample_feature()
    feature["geometry"] = None

    with pytest.raises(ValueError, match="no geometry"):
        Assignment.from_feature(feature)


@pytest.mark.skipif(not FIXTURE.exists(), reason="run scripts/explore_caltopo.py first")
def test_real_fixture_parses():
    feature = json.loads(FIXTURE.read_text(encoding="utf-8"))

    a = Assignment.from_feature(feature)

    assert a.id
    assert a.geometry["type"] in {"Polygon", "MultiPolygon", "LineString"}
    assert a.responsive_pod == "LOW"
    assert a.team_size == "0"


def test_parses_planner_entered_operational_fields():
    a = Assignment.from_feature(sample_feature())

    assert a.responsive_pod == "LOW"
    assert a.unresponsive_pod == "LOW"
    assert a.clue_pod == "LOW"
    assert a.team_size == "4"
    assert a.time_allocated == "6"
    assert a.transportation == "VEHICLE"
    assert a.previous_efforts == "none"


def test_operational_fields_default_to_empty_when_absent():
    feature = sample_feature()
    for key in ["responsivePOD", "teamSize", "timeAllocated"]:
        del feature["properties"][key]

    a = Assignment.from_feature(feature)

    assert a.responsive_pod == ""
    assert a.team_size == ""
    assert a.time_allocated == ""
