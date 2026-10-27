import io
import threading
import time

import pytest
from PIL import Image

from sar_geo_mcp.tiles import TILE_PX, Tile, tile_size_m

from sar_geo_mcp.caltopo.wms import (
    MAX_PIXELS,
    PNG_MAGIC,
    ensure_png,
    fetch_panel,
    fetch_panel_tile,
)

BBOX = (-13618826.5, 4481005.8, -13614373.7, 4486604.9)


def png_bytes(width: int, height: int, colour: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buffer, format="PNG")
    return buffer.getvalue()


def test_ensure_png_accepts_a_real_png():
    ensure_png(png_bytes(4, 4, (1, 2, 3)), "http://example/test")


def test_ensure_png_rejects_html_error_page_served_as_200():
    html = b"<html lang='en-US'>\n<head>\n  <title>Error</title>"

    with pytest.raises(RuntimeError, match="did not return a PNG"):
        ensure_png(html, "http://example/test")


def test_ensure_png_error_names_the_url_but_is_still_raised():
    with pytest.raises(RuntimeError, match="example.com"):
        ensure_png(b"not an image", "https://example.com/wms?LAYERS=sf")


def test_png_magic_is_the_standard_signature():
    assert PNG_MAGIC == b"\x89PNG\r\n\x1a\n"


#: A bbox spanning exactly two grid columns within one row, at LEVEL. Built
#: from the grid rather than hand-picked so the test asserts placement, not
#: an accident of which tiles a particular bbox happened to hit.
LEVEL = 16
_SIZE = tile_size_m(LEVEL)
TWO_TILE_BBOX = (_SIZE * 0.5, _SIZE * 0.25, _SIZE * 1.5, _SIZE * 0.75)


def tile_colour_stub(colours):
    """Serve each grid column a different colour, keyed off the URL's bbox.

    Colouring by position rather than by call order is what makes this a
    placement test: an implementation that fetched the right tiles and
    pasted them in the wrong order would pass a call-order stub.
    """
    def fake_get(url: str) -> bytes:
        from urllib.parse import parse_qs, urlparse

        bbox = [float(v) for v in
                parse_qs(urlparse(url).query)["BBOX"][0].split(",")]
        column = round(bbox[0] / _SIZE)
        return png_bytes(TILE_PX, TILE_PX, colours[column])

    return fake_get


def test_fetch_panel_mosaics_tiles_in_the_right_positions(tmp_path):
    """West tile red, east tile blue - so a horizontal flip or a swapped
    paste shows up as a colour in the wrong half."""
    path = fetch_panel(
        base_url="https://example.com/wms",
        layers=["sf"],
        bbox_3857=TWO_TILE_BBOX,
        width_px=1024,
        height_px=512,
        cache_dir=tmp_path,
        get_bytes=tile_colour_stub({0: (255, 0, 0), 1: (0, 0, 255)}),
    )

    image = Image.open(path).convert("RGB")
    assert image.size == (1024, 512)
    assert image.getpixel((100, 256)) == (255, 0, 0), "west half"
    assert image.getpixel((900, 256)) == (0, 0, 255), "east half"


def test_two_overlapping_panels_share_their_tiles(tmp_path):
    """115 MB of the 179 MB measured on a real cache was panels that
    overlapped and shared nothing."""
    calls = []

    def counting(url):
        calls.append(url)
        return tile_colour_stub({0: (1, 1, 1), 1: (2, 2, 2), 2: (3, 3, 3)})(url)

    fetch_panel(base_url="https://example.com/wms", layers=["sf"],
                bbox_3857=TWO_TILE_BBOX, width_px=1024, height_px=512,
                cache_dir=tmp_path, get_bytes=counting)
    first_round = len(calls)

    shifted = tuple(v + _SIZE * 0.25 if i % 2 == 0 else v
                    for i, v in enumerate(TWO_TILE_BBOX))
    fetch_panel(base_url="https://example.com/wms", layers=["sf"],
                bbox_3857=shifted, width_px=1024, height_px=512,
                cache_dir=tmp_path, get_bytes=counting)

    assert len(calls) < first_round * 2, (
        "an overlapping panel refetched everything; the grid is not shared"
    )


def test_a_tile_is_always_a_single_server_render(tmp_path):
    """Labelled basemaps place labels per request. A tile stitched from
    sub-requests would duplicate them at the joins, so a tile must never
    exceed the GetMap ceiling."""
    requested = []

    def record(url):
        from urllib.parse import parse_qs, urlparse

        query = parse_qs(urlparse(url).query)
        requested.append((int(query["WIDTH"][0]), int(query["HEIGHT"][0])))
        return png_bytes(TILE_PX, TILE_PX, (9, 9, 9))

    fetch_panel(base_url="https://example.com/wms", layers=["mbt"],
                bbox_3857=TWO_TILE_BBOX, width_px=1024, height_px=512,
                cache_dir=tmp_path, get_bytes=record)

    assert requested, "nothing was fetched"
    for width, height in requested:
        assert width <= MAX_PIXELS and height <= MAX_PIXELS


def test_fetch_panel_uses_cache_on_second_call(tmp_path):
    calls = []

    def fake_get(url: str) -> bytes:
        calls.append(url)
        return png_bytes(10, 10, (7, 7, 7))

    kwargs = dict(
        base_url="https://example.com/wms",
        layers=["sf"],
        bbox_3857=BBOX,
        width_px=10,
        height_px=10,
        cache_dir=tmp_path,
        get_bytes=fake_get,
    )

    first = fetch_panel(**kwargs)
    second = fetch_panel(**kwargs)

    assert first == second
    assert len(calls) == 1


def test_cached_filename_does_not_contain_the_token(tmp_path):
    def fake_get(url: str) -> bytes:
        return png_bytes(10, 10, (0, 0, 0))

    path = fetch_panel(
        base_url="https://caltopo.com/api/SECRETTOKEN/wms",
        layers=["sf"],
        bbox_3857=BBOX,
        width_px=10,
        height_px=10,
        cache_dir=tmp_path,
        get_bytes=fake_get,
    )

    assert "SECRETTOKEN" not in str(path)


def test_concurrent_fetch_panel_tile_both_succeed(tmp_path):
    """The tile proxy serves Leaflet's parallel requests through a
    threadpool, so two threads can race to fetch the same tile. Each must
    get its own temp file: a shared `.part` name lets one thread's
    `write_bytes` truncate the file the other is about to `replace`, and
    an exists()-gated cache never re-validates the promoted result."""
    barrier = threading.Barrier(2)

    def slow_get(url: str) -> bytes:
        barrier.wait(timeout=5)
        time.sleep(0.2)
        return png_bytes(TILE_PX, TILE_PX, (5, 6, 7))

    tile = Tile(level=16, col=3, row=4)
    results = []
    errors = []

    def worker():
        try:
            results.append(
                fetch_panel_tile(
                    "https://example.com/wms", ["sf"], tile, tmp_path, slow_get
                )
            )
        except Exception as exc:  # pragma: no cover - surfaced via errors
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not errors, f"concurrent fetch raised: {errors}"
    assert len(results) == 2
    destination = results[0]
    assert results[1] == destination

    image = Image.open(destination)
    assert image.size == (TILE_PX, TILE_PX)

    leftover = list(tmp_path.rglob("*.part"))
    assert not leftover, f"leftover temp files: {leftover}"


def test_fetch_panel_tile_leaves_no_part_file_on_error(tmp_path):
    def failing_get(url: str) -> bytes:
        raise RuntimeError("network is down")

    tile = Tile(level=16, col=9, row=9)

    with pytest.raises(RuntimeError, match="network is down"):
        fetch_panel_tile("https://example.com/wms", ["sf"], tile, tmp_path, failing_get)

    leftover = list(tmp_path.rglob("*.part"))
    assert not leftover, f"leftover temp files: {leftover}"


# --- categorical layers and the resample that used to blend them ------------
#
# The two `slope_shading` tests inject `fetch=lambda *a, **k: path` and never
# touch `fetch_panel`, which is why nothing in the suite saw this. These go
# through the real fetch, with a stubbed transport and no token.

#: One tile of ground, requested at a pixel size that does NOT land on a
#: level: 1024 native pixels down to 611 is a 1.676x downsample, which is
#: what a staged 2.0 Mercator-unit pixel actually gets (level 17 at 1.1943
#: u/px, 1.675x). `level_for` still picks the finer level and downsamples.
ONE_TILE_BBOX = (0.0, 0.0, _SIZE, _SIZE)
OFF_LEVEL_PX = 611


def categorical_tile_bytes() -> bytes:
    """A tile painted in flat legend colours, in stripes narrow enough that a
    1.676x downsample lands between them.

    This is a PERFECTLY CORRECT render against a PERFECTLY CORRECT legend:
    every pixel is exactly one of `SF_LEGEND`'s colours. Anything the decode
    cannot match afterwards was manufactured by the resize.
    """
    import numpy as np
    from sar_geo_mcp.caltopo.shading import SF_LEGEND

    pixels = np.zeros((TILE_PX, TILE_PX, 4), dtype="uint8")
    stripe = 7
    for column in range(TILE_PX):
        band = SF_LEGEND[(column // stripe) % len(SF_LEGEND)]
        pixels[:, column, :3] = band.rgb
    pixels[:, :, 3] = 255
    buffer = io.BytesIO()
    Image.fromarray(pixels, mode="RGBA").save(buffer, format="PNG")
    return buffer.getvalue()


def unmatched_fraction(path) -> float:
    """What `slope_shading` would report as `caltopo.unmatched_fraction`."""
    import numpy as np
    from sar_geo_mcp.caltopo.shading import UNKNOWN, decode_slope_bands

    with Image.open(path) as image:
        decoded = decode_slope_bands(np.asarray(image.convert("RGBA")))
    return float((decoded == UNKNOWN).mean())


def test_the_off_level_request_really_does_resample():
    """The premise, asserted rather than assumed.

    `tiles.py` says a request "almost never lands exactly on a level", so
    the resize is the normal path. If this stopped being true the two tests
    below would pass for the wrong reason.
    """
    from sar_geo_mcp.tiles import level_for, pixel_size_m

    level = level_for(ONE_TILE_BBOX, OFF_LEVEL_PX, OFF_LEVEL_PX)
    native = pixel_size_m(level)
    requested = _SIZE / OFF_LEVEL_PX
    assert level == LEVEL
    assert 1.6 < requested / native < 1.7


def test_a_categorical_layer_survives_the_resample_intact(tmp_path):
    """`sf` is a colour-as-category raster: a blended pixel is exactly the
    colour the legend refuses, and decodes to UNKNOWN."""
    path = fetch_panel(base_url="https://example.com/wms", layers=["sf"],
                       bbox_3857=ONE_TILE_BBOX, width_px=OFF_LEVEL_PX,
                       height_px=OFF_LEVEL_PX, cache_dir=tmp_path,
                       get_bytes=lambda url: categorical_tile_bytes())

    with Image.open(path) as image:
        assert image.size == (OFF_LEVEL_PX, OFF_LEVEL_PX)
    assert unmatched_fraction(path) == 0.0


def test_lanczos_on_the_same_panel_manufactures_colours_no_legend_holds(tmp_path):
    """The defect, held in place so it cannot come back as a default.

    Measured on a synthetic `sf` render with large smooth band patches at
    this same 1.675x downsample: LANCZOS gave unmatched fractions of 0.033
    to 0.192 depending on how much band boundary the ground holds, against
    0.000 at nearest-neighbour. `UNMATCHED_WARNING_FRACTION` is 0.10, so the
    first real run with a token tripped a warning that then blamed the
    legend - which was not the culprit.
    """
    path = fetch_panel(base_url="https://example.com/wms", layers=["sf"],
                       bbox_3857=ONE_TILE_BBOX, width_px=OFF_LEVEL_PX,
                       height_px=OFF_LEVEL_PX, cache_dir=tmp_path,
                       get_bytes=lambda url: categorical_tile_bytes(),
                       resample=Image.LANCZOS)

    assert unmatched_fraction(path) > 0.10


def test_the_two_filters_do_not_share_a_cache_entry(tmp_path):
    """Otherwise the first `sf` panel ever assembled decides what every later
    caller gets, whatever filter they asked for."""
    kwargs = dict(base_url="https://example.com/wms", layers=["sf"],
                  bbox_3857=ONE_TILE_BBOX, width_px=OFF_LEVEL_PX,
                  height_px=OFF_LEVEL_PX, cache_dir=tmp_path,
                  get_bytes=lambda url: categorical_tile_bytes())

    blended = fetch_panel(**kwargs, resample=Image.LANCZOS)
    exact = fetch_panel(**kwargs)

    assert blended != exact
    assert unmatched_fraction(blended) > 0.10
    assert unmatched_fraction(exact) == 0.0


def test_the_layer_decides_the_filter_not_the_call_site():
    """A categorical layer is a fact about what CalTopo renders, so the
    default comes from the layers; the parameter only overrides it."""
    from sar_geo_mcp.caltopo.wms import CATEGORICAL_LAYERS, resample_filter_for

    assert "sf" in CATEGORICAL_LAYERS
    assert resample_filter_for(["sf"]) == Image.NEAREST
    assert resample_filter_for(["mbt", "sf"]) == Image.NEAREST
    # Everything else keeps the filter a human-viewed basemap wants, which is
    # what every caller got before the parameter existed.
    assert resample_filter_for(["r3"]) == Image.LANCZOS
    assert resample_filter_for(["mbt"]) == Image.LANCZOS


def test_a_non_categorical_panel_is_still_resampled_with_lanczos(tmp_path):
    """The default-behaviour claim, on the path `shaded_relief` uses for
    `r`/`r2`/`r3`: a relief render is a shade, not a category."""
    recorded = []
    real_resize = Image.Image.resize

    def spy(self, size, resample=None, *args, **kwargs):
        recorded.append(resample)
        return real_resize(self, size, resample, *args, **kwargs)

    Image.Image.resize = spy
    try:
        fetch_panel(base_url="https://example.com/wms", layers=["r3"],
                    bbox_3857=ONE_TILE_BBOX, width_px=OFF_LEVEL_PX,
                    height_px=OFF_LEVEL_PX, cache_dir=tmp_path,
                    get_bytes=lambda url: png_bytes(TILE_PX, TILE_PX, (60, 60, 60)))
    finally:
        Image.Image.resize = real_resize

    # Pillow can call `resize` again internally on the reduce path, so the
    # claim is about the filter, not the call count.
    assert recorded and all(flt == Image.LANCZOS for flt in recorded)


@pytest.mark.live
def test_live_fetch_against_caltopo(tmp_path):
    from sar_geo_mcp.config import CaltopoConfig

    # from_env, not the bare constructor: `CaltopoConfig()` has no token,
    # so this test used to fail on a None URL before it reached CalTopo.
    base_url = CaltopoConfig.from_env().wms_base_url()
    if base_url is None:
        pytest.skip("ZA_WMS_TOKEN is not set")
    path = fetch_panel(
        base_url=base_url,
        layers=["mbt", "sf"],
        bbox_3857=BBOX,
        width_px=512,
        height_px=512,
        cache_dir=tmp_path,
    )

    image = Image.open(path)
    assert image.size == (512, 512)



# --- thin line art read back as a mask ----------------------------------------

def outline_tile_bytes(columns) -> bytes:
    """A transparent tile with a 1 px vertical outline at alpha 64 - the low
    end of the band CalTopo draws structure footprints in - at each of
    `columns`."""
    import numpy as np

    pixels = np.zeros((TILE_PX, TILE_PX, 4), dtype="uint8")
    for column in columns:
        pixels[:, column] = (255, 171, 0, 64)
    buffer = io.BytesIO()
    Image.fromarray(pixels, mode="RGBA").save(buffer, format="PNG")
    return buffer.getvalue()


#: Lines 23 native pixels apart: 23 shares no factor with the 1.676x ratio's
#: grid, so the lines fall at every phase against the output pixels.
OUTLINE_COLUMNS = list(range(10, TILE_PX - 10, 23))


def _lowest_line_alpha(path) -> int:
    """The weakest outline's strongest alpha after the fetch: each line's
    alpha is taken as the maximum over the output columns it can land in."""
    import numpy as np

    with Image.open(path) as image:
        alpha = np.asarray(image.convert("RGBA"))[:, :, 3].astype(int)
    scale = OFF_LEVEL_PX / TILE_PX
    strongest = []
    for column in OUTLINE_COLUMNS:
        at = int(column * scale)
        strongest.append(int(alpha[:, max(0, at - 1):at + 2].max(axis=1).min()))
    return min(strongest)


def test_a_thin_structure_outline_survives_the_fetch(tmp_path):
    """The structures layer is read back as a mask with an alpha floor
    (`lz.STRUCTURE_ALPHA_MIN`, 32). Through the real fetch at the 1.676x
    downsample a staged 2 m cell gets, every 1 px outline at alpha 64 must
    still reach that floor on every row."""
    from sar_geo_mcp.planning.lz import STRUCTURE_ALPHA_MIN

    tile = outline_tile_bytes(OUTLINE_COLUMNS)
    path = fetch_panel(base_url="https://example.com/wms", layers=["structures"],
                       bbox_3857=ONE_TILE_BBOX, width_px=OFF_LEVEL_PX,
                       height_px=OFF_LEVEL_PX, cache_dir=tmp_path,
                       get_bytes=lambda url: tile)
    assert _lowest_line_alpha(path) >= STRUCTURE_ALPHA_MIN


def test_lanczos_would_lose_a_thin_structure_outline(tmp_path):
    """The premise, measured: the filter every non-categorical layer got
    before drops some of the same outlines below the floor."""
    from sar_geo_mcp.planning.lz import STRUCTURE_ALPHA_MIN

    tile = outline_tile_bytes(OUTLINE_COLUMNS)
    path = fetch_panel(base_url="https://example.com/wms", layers=["structures"],
                       bbox_3857=ONE_TILE_BBOX, width_px=OFF_LEVEL_PX,
                       height_px=OFF_LEVEL_PX, cache_dir=tmp_path,
                       get_bytes=lambda url: tile, resample=Image.LANCZOS)
    assert _lowest_line_alpha(path) < STRUCTURE_ALPHA_MIN


def test_the_structures_layer_decides_its_own_filter():
    from sar_geo_mcp.caltopo.wms import MAX_ALPHA, resample_filter_for

    assert resample_filter_for(["structures"]) == MAX_ALPHA
    assert resample_filter_for(["imagery"]) == Image.LANCZOS
