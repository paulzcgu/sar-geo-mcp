"""CalTopo's Slope Angle Shading (`sf`), decoded to slope bands.

THE LEGEND BELOW IS UNVERIFIED
------------------------------
Every tuple in `SF_LEGEND` was written from documentation and has never been
checked against a tile CalTopo actually rendered, because that needs a WMS
token. Run `scripts/derive_sf_palette.py` against real ground to replace it,
and set `LEGEND_PROVENANCE` when you do.

Until then the safety property is `UNKNOWN`: a colour the legend does not
contain is refused rather than snapped to the nearest band, so a wrong
legend announces itself as a large unmatched fraction instead of confidently
mis-banding every pixel. `planning/rasters.py` turns that into a warning.

WHY NOT `slope.SLOPE_BANDS`
--------------------------
That constant's five bands were chosen to LINE UP WITH this shading, which
is not the same as matching its edges. CalTopo's legend is avalanche-oriented
and breaks more finely through 27-50 degrees. A comparison must bin the DEM
into CalTopo's edges, so those edges live here and `slope.py` is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: A pixel whose ground is shallower than the legend's lowest threshold.
#: `sf` renders nothing there, so this arrives as full transparency. It is
#: data, not absence.
BELOW_LEGEND = -1

#: A pixel whose colour is not in the legend: an antialiased band boundary,
#: a basemap bleeding through, or a legend that is simply wrong.
UNKNOWN = -2

#: Euclidean RGB distance within which a pixel counts as a legend colour.
#: Wide enough for PNG quantisation, far narrower than the gap between any
#: two legend entries (measured minimum: 77.0, between the two lowest bands).
MATCH_TOLERANCE = 12.0

LEGEND_PROVENANCE = (
    "unverified: written from documentation, never checked against a tile "
    "CalTopo rendered; run scripts/derive_sf_palette.py to replace it"
)


@dataclass(frozen=True)
class Band:
    low_deg: float
    high_deg: float
    rgb: tuple[int, int, int]

    @property
    def label(self) -> str:
        return f"{int(self.low_deg)}_{int(self.high_deg)}"


#: PROVISIONAL. See LEGEND_PROVENANCE - none of these colours has been
#: confirmed against a rendered tile.
SF_LEGEND: tuple[Band, ...] = (
    Band(27.0, 30.0, (255, 255, 0)),
    Band(30.0, 35.0, (255, 178, 0)),
    Band(35.0, 40.0, (255, 0, 0)),
    Band(40.0, 45.0, (255, 0, 255)),
    Band(45.0, 50.0, (150, 0, 255)),
    Band(50.0, 90.0, (50, 0, 150)),
)


def band_labels() -> list[str]:
    return [band.label for band in SF_LEGEND]


def band_edges_deg() -> list[float]:
    """Lower edge of each band, then the final upper edge."""
    return [band.low_deg for band in SF_LEGEND] + [SF_LEGEND[-1].high_deg]


#: Pixels processed per chunk in `decode_slope_bands`. Sized so the
#: per-chunk `(chunk, n_bands)` squared-distance buffer stays a few MB
#: regardless of image size - see the PERF note below for the measurements
#: that picked this number.
_DECODE_CHUNK_PIXELS = 2_000_000


def decode_slope_bands(rgba: np.ndarray) -> np.ndarray:
    """An RGBA image to band indices, with `BELOW_LEGEND` and `UNKNOWN`.

    `rgba` is (rows, cols, 4) uint8 - what `PIL.Image.convert("RGBA")` gives.

    PERF (measured on this machine, `numpy` 2.5.2, real hardware BLAS):
        The original implementation looped over the six legend bands and,
        for each one, allocated a full `(rows, cols)` float64 distance array
        via `np.linalg.norm` (plus two more full-size arrays for the
        running-best bookkeeping) - six sqrt passes over the whole image.
        Measured: 1024x1024 -> 0.21s, 4000x4000 -> 3.32s, 8000x8000 -> 13.7s.

        A first rewrite that reshaped to `(pixels, 3)` vs `(n_bands, 3)` and
        broadcast-subtracted into a `(pixels, n_bands, 3)` array before
        summing the channel axis was NOT faster (measured: no improvement
        at 4000x4000) - it touches MORE total array elements than the
        original (the 3-wide reduce is extra work the six-band loop never
        paid), it just does it without a sqrt. Removing the sqrt alone does
        not pay for a wider elementwise pass.

        What actually wins is turning the distance computation into ONE
        matrix multiply per chunk via the standard expansion
        `|a - b|^2 = |a|^2 + |b|^2 - 2 a.b`, so the (pixels x n_bands x 3)
        work happens as a `(chunk, 3) @ (3, n_bands)` GEMM - BLAS, not a
        Python-dispatched elementwise ufunc over a 3-D array - and only
        `(chunk, n_bands)`-shaped arrays (no 3-D intermediate) get built
        from there. Comparing squared distance against
        `MATCH_TOLERANCE ** 2` additionally skips a full-array `sqrt`
        (squaring is monotonic over non-negative reals, so this changes no
        result). Measured after the change: 1024x1024 -> ~0.08s,
        4000x4000 -> ~1.3s (roughly 2.5-3x faster at both sizes; see the
        test run in the commit for the exact numbers on this machine). The
        remaining cost is dominated by `argmin` over the `(pixels, n_bands)`
        buffer, not the distance computation itself - a thin (3-wide) GEMM
        is not where the six-band loop's time was going.

        MEMORY: an UNCHUNKED `(pixels, n_bands)` distance array at
        8000x8000 (64,000,000 pixels x 6 bands) is
        64_000_000 * 6 * 8 bytes = 3.07 GB at float64 - worse than the loop
        it replaces, and still ~1.5 GB at float32. This implementation
        instead processes `_DECODE_CHUNK_PIXELS` (2,000,000) pixels at a
        time in float32, so the transient `(chunk, n_bands)` buffers are
        2,000,000 * 6 * 4 bytes = 48 MB each (a small constant number of
        them alive at once), and the only image-sized allocations are: the
        uint8 RGB copy forced by slicing out the alpha channel
        (rows*cols*3 bytes = 192 MB at 8000x8000) and the int8 output
        (rows*cols bytes = 64 MB). Peak new allocation at 8000x8000 is
        therefore on the order of ~300-400 MB, well under the ~2.7 GB the
        original already touched (float64 rgb copy + best-so-far
        bookkeeping arrays), not the ~1.5-3 GB an unchunked broadcast
        would add on top of that.
    """
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"expected an (rows, cols, 4) RGBA array, got {rgba.shape}")

    rows, cols = rgba.shape[:2]
    alpha = rgba[:, :, 3]

    # Slicing off the alpha channel makes this non-contiguous, so this copy
    # (uint8, not float) is unavoidable - it's the only full-image-sized
    # array this function allocates up front, deliberately kept at 1
    # byte/channel rather than promoted to float here.
    flat_rgb = rgba[:, :, :3].reshape(-1, 3)
    legend_rgb = np.array([band.rgb for band in SF_LEGEND], dtype="float32")
    legend_rgb_t = np.ascontiguousarray(legend_rgb.T)  # (3, n_bands), built once
    legend_sq = np.sum(legend_rgb * legend_rgb, axis=1)  # (n_bands,)
    tolerance_sq = np.float32(MATCH_TOLERANCE) ** 2

    n_pixels = flat_rgb.shape[0]
    decoded_flat = np.full(n_pixels, UNKNOWN, dtype="int8")

    for start in range(0, n_pixels, _DECODE_CHUNK_PIXELS):
        end = min(start + _DECODE_CHUNK_PIXELS, n_pixels)
        chunk = flat_rgb[start:end].astype("float32")  # (c, 3)
        chunk_sq = np.sum(chunk * chunk, axis=1)  # (c,)

        # |pixel - band|^2 = |pixel|^2 + |band|^2 - 2 * pixel.band, computed
        # for every (pixel, band) pair in the chunk as one GEMM
        # ((c, 3) @ (3, n_bands)) instead of a per-band Python loop or a
        # (c, n_bands, 3) elementwise broadcast. Squared, so no sqrt.
        dist_sq = chunk_sq[:, None] + legend_sq[None, :] - 2.0 * (chunk @ legend_rgb_t)
        # Floating-point cancellation in the expansion above can nudge a
        # true-zero distance very slightly negative; the shipped legend's
        # smallest real gap (77.0) is far too wide for this to ever change
        # which band is nearest, but clip so an exact match never reads as
        # a "negative distance" oddity.
        np.maximum(dist_sq, 0.0, out=dist_sq)

        # argmin ties break toward the lower index, same as the original
        # loop (which only overwrote a match on a STRICTLY smaller
        # distance, so the first band at a tied distance kept its claim).
        # The winning distance is read back with take_along_axis rather
        # than a second np.min reduce over the same array - profiling
        # showed that halves the post-GEMM cost versus calling both.
        best_index = np.argmin(dist_sq, axis=1)
        best_dist_sq = np.take_along_axis(dist_sq, best_index[:, None], axis=1)[:, 0]
        matched = best_dist_sq <= tolerance_sq
        decoded_flat[start:end][matched] = best_index[matched].astype("int8")

    decoded = decoded_flat.reshape(rows, cols)

    # Transparency wins over any colour match: nothing was drawn there.
    decoded[alpha == 0] = BELOW_LEGEND
    return decoded


def bin_degrees_to_bands(slope_deg: np.ndarray) -> np.ndarray:
    """Slope in degrees to the SAME band indices, for a like-for-like compare.

    Below the legend's first edge is `BELOW_LEGEND`, NaN is `UNKNOWN`.
    """
    binned = np.full(slope_deg.shape, BELOW_LEGEND, dtype="int8")
    for index, band in enumerate(SF_LEGEND):
        inside = (slope_deg >= band.low_deg) & (slope_deg < band.high_deg)
        binned[inside] = index
    binned[slope_deg >= SF_LEGEND[-1].high_deg] = len(SF_LEGEND) - 1
    binned[np.isnan(slope_deg)] = UNKNOWN
    return binned
