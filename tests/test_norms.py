"""The corpus norms: shipped as data with provenance, overridable per call."""

from __future__ import annotations

import json

import pytest

from sar_geo_mcp.planning.norms import (
    DEFAULT_NORMS_PATH,
    load_norms,
    norm_flags,
    norms_in_acres,
    target_area_range_acres,
    target_area_range_ha,
)


def test_the_shipped_file_loads_and_carries_provenance():
    norms = load_norms()
    assert norms["provenance"]["maps"] == 65 and norms["provenance"]["assignments"] == 4673
    assert norms["polygon_area_ha"]["p50"] == 9.1
    assert norms["corridor_width_m"]["by_axis"]["drainage"] == 30
    assert norms["tolerances_m"]["shared_edge"] == 15
    assert norms["grid_cell_m"] == {"min": 100, "default": 200, "max": 250}


def test_the_shipped_file_is_package_data():
    assert DEFAULT_NORMS_PATH.name == "assignment_norms.json"
    assert DEFAULT_NORMS_PATH.is_file()


def test_an_override_file_replaces_the_shipped_one(tmp_path):
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"provenance": {"corpus": "elsewhere"},
                                "polygon_area_ha": {"p10": 1, "p25": 2, "p50": 3, "p75": 4, "p90": 5}}),
                    encoding="utf-8")
    assert load_norms(path)["provenance"]["corpus"] == "elsewhere"


def test_target_range_for_a_known_resource_follows_the_ratio_rule():
    norms = load_norms()
    ground = target_area_range_ha("GROUND", norms)
    assert ground == {"p25": pytest.approx(10.0 * 0.42), "p50": 10.0, "p75": pytest.approx(10.0 * 2.15),
                      "source": "area_ha_by_resource"}
    dog = target_area_range_ha("dog_hrd", norms)   # case-insensitive
    assert dog["p50"] == 5.4


def test_target_range_for_an_unknown_resource_is_the_overall_quartiles():
    norms = load_norms()
    other = target_area_range_ha("SUBMARINE", norms)
    assert other == {"p25": 3.8, "p50": 9.1, "p75": 19.6, "source": "polygon_area_ha"}


def test_flags_raise_only_outside_the_p10_p90_band():
    norms = load_norms()
    assert norm_flags({"area_acres": 22.0, "on_feature_pct": 78.0}, norms) == []
    flags = norm_flags({"area_acres": 2.5, "on_feature_pct": 20.0, "width_m": 300.0,
                        "length_m": 20000.0}, norms)
    metrics = {f["metric"]: f for f in flags}
    assert set(metrics) == {"area_acres", "on_feature_pct", "width_m", "length_m"}
    area = metrics["area_acres"]
    # The corpus's 2, 9.1 and 31 ha, in acres.
    assert area["value"] == 2.5 and area["corpus_p10"] == 4.9 and area["corpus_p90"] == 76.6
    assert area["corpus_p50"] == 22.5 and "below" in area["note"]
    assert "above" in metrics["width_m"]["note"]


def test_flags_ignore_metrics_that_are_absent_or_null():
    norms = load_norms()
    assert norm_flags({"area_acres": None, "length_m": 5000.0}, norms) == []
    assert norm_flags({}, norms) == []


def test_config_carries_a_norms_path(monkeypatch):
    from sar_geo_mcp.config import CaltopoConfig

    assert CaltopoConfig().norms_path is None
    monkeypatch.setenv("ZA_ASSIGNMENT_NORMS", "C:/elsewhere/norms.json")
    assert str(CaltopoConfig.from_env().norms_path).endswith("norms.json")


def test_norm_flags_bands_area_against_the_resource_when_it_is_named():
    """A resource the corpus has a median for gets its own band: 22 acres is
    ordinary for GROUND (10.4 to 53.1) and 2.5 acres is not."""
    norms = load_norms()
    assert norm_flags({"area_acres": 22.0}, norms, "GROUND") == []
    flag = norm_flags({"area_acres": 2.5}, norms, "GROUND")[0]
    assert flag["metric"] == "area_acres"
    assert flag["corpus_p10"] == 10.4
    assert flag["corpus_p90"] == 53.1
    assert flag["corpus_p50"] == 24.7
    assert "resource p25-p75" in flag["band"]


def test_a_dog_trail_assignment_is_not_flagged_against_the_whole_corpus():
    """The finding: 2.2 acres is a normal DOG_TRAIL (median 1.7) and was
    being reported as below the corpus p10-p90 band of 4.9 to 76.6."""
    norms = load_norms()
    assert norm_flags({"area_acres": 2.2}, norms, "DOG_TRAIL") == []
    assert norm_flags({"area_acres": 2.2}, norms)[0]["metric"] == "area_acres"


def test_a_dog_trail_assignment_above_its_own_band_is_flagged_as_such():
    norms = load_norms()
    flag = norm_flags({"area_acres": 7.4}, norms, "DOG_TRAIL")[0]
    assert flag["metric"] == "area_acres" and flag["value"] == 7.4
    assert flag["corpus_p90"] == 3.7                     # 0.7 ha x 2.15, in acres
    assert "resource" in flag["band"]
    assert "above" in flag["note"] and "DOG_TRAIL" in flag["note"]


def test_an_unknown_resource_falls_back_to_the_corpus_band():
    norms = load_norms()
    flag = norm_flags({"area_acres": 2.5}, norms, "SUBMARINE")[0]
    assert flag["corpus_p10"] == 4.9 and flag["corpus_p90"] == 76.6
    assert flag["band"] == "corpus p10-p90"
    assert "corpus" in flag["note"]


def test_only_area_is_resource_banded():
    """The corpus has no per-resource width, length or on-feature figures, so
    those stay banded against the whole corpus."""
    norms = load_norms()
    flags = {f["metric"]: f for f in norm_flags(
        {"area_acres": 22.0, "on_feature_pct": 20.0, "width_m": 300.0}, norms, "GROUND")}
    assert set(flags) == {"on_feature_pct", "width_m"}
    assert flags["width_m"]["band"] == "corpus p10-p90"
    assert "p10" in flags["on_feature_pct"]["band"]


def test_areas_are_answered_in_acres_while_the_file_keeps_hectares():
    norms = load_norms()
    shown = norms_in_acres(norms)
    assert "polygon_area_ha" not in shown and "area_ha_by_resource" not in shown
    assert shown["polygon_area_acres"]["p50"] == 22.5
    assert shown["area_acres_by_resource"]["medians"]["GROUND"] == 24.7
    assert shown["area_acres_by_resource"]["p25_ratio"] == 0.42
    assert norms["polygon_area_ha"]["p50"] == 9.1                  # the file is untouched
    assert target_area_range_acres("GROUND", norms) == {
        "p25": 10.4, "p50": 24.7, "p75": 53.1, "source": "area_ha_by_resource"}
