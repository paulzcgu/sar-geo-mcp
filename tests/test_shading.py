"""Decoding CalTopo's slope angle shading to bands.

THE LEGEND SHIPPED HERE IS UNVERIFIED. It was written from documentation and
has never been checked against a rendered tile, because that needs a WMS
token. These tests therefore prove the DECODE, not the legend: they build
images from whatever `SF_LEGEND` currently says and assert the round trip,
plus the three behaviours that make a wrong legend safe rather than silent.
"""

import numpy as np
import pytest

from sar_geo_mcp.caltopo import shading as shading_module
from sar_geo_mcp.caltopo.shading import (
    BELOW_LEGEND,
    SF_LEGEND,
    UNKNOWN,
    Band,
    LEGEND_PROVENANCE,
    band_edges_deg,
    band_labels,
    bin_degrees_to_bands,
    decode_slope_bands,
)


def image_of(colours, alpha=255):
    """An RGBA array, one pixel per colour given."""
    pixels = np.zeros((1, len(colours), 4), dtype="uint8")
    for index, colour in enumerate(colours):
        pixels[0, index, :3] = colour
        pixels[0, index, 3] = alpha
    return pixels


def test_every_legend_colour_decodes_to_its_own_band():
    colours = [entry.rgb for entry in SF_LEGEND]
    decoded = decode_slope_bands(image_of(colours))
    assert list(decoded[0]) == list(range(len(SF_LEGEND)))


def test_a_transparent_pixel_is_below_the_legend_not_missing():
    """sf renders nothing below its lowest threshold. That is DATA - ground
    shallower than the legend starts - and must not read as a hole."""
    decoded = decode_slope_bands(image_of([(0, 0, 0)], alpha=0))
    assert decoded[0, 0] == BELOW_LEGEND


def test_transparency_wins_even_when_the_rgb_channels_match_a_legend_colour():
    """A fully transparent PNG pixel often carries arbitrary leftover RGB -
    sometimes a colour that happens to sit in the legend. Alpha must be
    checked AFTER colour matching so that case still reads BELOW_LEGEND,
    not the band its stale RGB would otherwise match."""
    decoded = decode_slope_bands(image_of([SF_LEGEND[0].rgb], alpha=0))
    assert decoded[0, 0] == BELOW_LEGEND


def test_an_antialiased_boundary_pixel_is_unknown_not_snapped():
    """The property that makes an unverified legend safe: a colour the legend
    does not contain is refused, so a wrong legend announces itself through
    unmatched_fraction instead of mis-banding every pixel confidently."""
    first, second = SF_LEGEND[0].rgb, SF_LEGEND[1].rgb
    blend = tuple(int((a + b) / 2) for a, b in zip(first, second))
    decoded = decode_slope_bands(image_of([blend]))
    assert decoded[0, 0] == UNKNOWN


def test_a_near_miss_within_tolerance_still_decodes():
    """PNG quantisation moves a colour a little; the legend is not brittle."""
    nudged = tuple(min(255, channel + 3) for channel in SF_LEGEND[0].rgb)
    decoded = decode_slope_bands(image_of([nudged]))
    assert decoded[0, 0] == 0


def test_degrees_bin_into_the_legend_edges_not_the_packages_own_bands():
    edges = band_edges_deg()
    midpoint = (edges[0] + edges[1]) / 2.0
    binned = bin_degrees_to_bands(np.array([[midpoint]]))
    assert binned[0, 0] == 0


def test_ground_shallower_than_the_legend_bins_below_legend():
    edges = band_edges_deg()
    binned = bin_degrees_to_bands(np.array([[edges[0] - 5.0]]))
    assert binned[0, 0] == BELOW_LEGEND


def test_nan_slope_is_unknown():
    binned = bin_degrees_to_bands(np.array([[np.nan]]))
    assert binned[0, 0] == UNKNOWN


def test_labels_line_up_with_the_legend():
    assert len(band_labels()) == len(SF_LEGEND)
    assert band_labels()[0].replace("_", "").isdigit() or "_" in band_labels()[0]


def test_labels_are_low_underscore_high_degrees():
    """The weaker assertion above passes for almost anything containing an
    underscore or only digits; pin down the actual `low_high` format."""
    for band, label in zip(SF_LEGEND, band_labels()):
        assert label == f"{int(band.low_deg)}_{int(band.high_deg)}"


def test_the_provenance_says_it_is_unverified():
    assert "unverified" in LEGEND_PROVENANCE.lower()


def test_slope_of_exactly_90_degrees_lands_in_the_last_band():
    """90 degrees is a real, if extreme, slope - it must not fall through
    the last band's half-open interval into BELOW_LEGEND."""
    binned = bin_degrees_to_bands(np.array([[90.0]]))
    assert binned[0, 0] == len(SF_LEGEND) - 1


def test_nearest_band_wins_when_more_than_one_is_within_tolerance(monkeypatch):
    """The shipped SF_LEGEND's minimum pairwise gap is 77.0 - over six times
    MATCH_TOLERANCE (12.0) - so no real pixel can ever land within tolerance
    of two bands at once, and this property can't be exercised through the
    real legend. Prove it with a synthetic, deliberately close-spaced
    legend instead: three bands 10 units apart on the red channel, all
    mutually within MATCH_TOLERANCE. A pixel at distance 9 / 1 / 11 from
    bands 0 / 1 / 2 must decode to band 1 - the NEAREST band, which is
    neither the first nor the last candidate a first-wins or last-wins rule
    would pick."""
    synthetic_legend = (
        Band(0.0, 1.0, (0, 0, 0)),
        Band(1.0, 2.0, (10, 0, 0)),
        Band(2.0, 3.0, (20, 0, 0)),
    )
    monkeypatch.setattr(shading_module, "SF_LEGEND", synthetic_legend)

    decoded = decode_slope_bands(image_of([(9, 0, 0)]))
    assert decoded[0, 0] == 1


def test_no_two_legend_colours_are_closer_than_twice_the_match_tolerance():
    """MATCH_TOLERANCE only keeps decode_slope_bands honest if it can never
    let two legend entries both plausibly match the same pixel. Guards
    against a future palette edit silently making the tolerance too wide."""
    from sar_geo_mcp.caltopo.shading import MATCH_TOLERANCE

    import itertools

    colours = [entry.rgb for entry in SF_LEGEND]
    distances = [
        np.linalg.norm(np.array(a, dtype="float64") - np.array(b, dtype="float64"))
        for a, b in itertools.combinations(colours, 2)
    ]
    assert min(distances) > 2 * MATCH_TOLERANCE
