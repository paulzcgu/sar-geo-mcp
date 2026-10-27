"""Several Overpass tiles in one request (0.22.1).

No network: the batched path's `post` is injected (or `linear._http_post`
patched, for the staging path that passes nothing), and tiles are told
apart by the bbox each tile's statements carry.
"""

import json

import pytest

from sar_geo_mcp.terrain import linear
from sar_geo_mcp.terrain.linear import (
    OSM_TILE_MARKER,
    OverpassUnavailable,
    _tile_bounds_wgs84,
    _tile_path,
    _tile_statements,
    build_batched_overpass_query,
    build_overpass_query,
    fetch_osm_tiles,
    gates_for,
    parking_for,
    routes_for,
    tiles_covering_osm,
)

# Two columns by three rows of OSM tiles: the shape of the staging that
# took 205-214 s on 2026-10-21.
SIX_TILE_BBOX = (100.0, 100.0, 7900.0, 11900.0)
TILES = tiles_covering_osm(SIX_TILE_BBOX)
HEADER = {"version": 0.6, "generator": "Overpass API test", "osm3s": {"copyright": "ODbL"}}


def _area(col: int, row: int) -> str:
    south, west, north, east = _tile_bounds_wgs84(col, row)
    return f"{south:.6f},{west:.6f},{north:.6f},{east:.6f}"


def _tiles_in(query: str) -> list[tuple[int, int]]:
    """The tiles a query asks for, in the order it asks."""
    found = [(query.index(_area(*tile)), tile) for tile in TILES if _area(*tile) in query]
    return [tile for _, tile in sorted(found)]


def _elements_for(col: int, row: int) -> list[dict]:
    """A trail across the middle of this tile and a gate on it, ids naming
    the tile."""
    south, west, north, east = _tile_bounds_wgs84(col, row)
    mid_lat = (south + north) / 2
    ident = 1000 + 10 * col + row
    return [
        {"type": "way", "id": ident, "tags": {"highway": "track", "name": f"Trail {col},{row}"},
         "geometry": [{"lat": mid_lat, "lon": west + 0.2 * (east - west)},
                      {"lat": mid_lat, "lon": west + 0.8 * (east - west)}]},
        {"type": "node", "id": ident, "tags": {"barrier": "gate"},
         "lat": mid_lat, "lon": west + 0.5 * (east - west)},
    ]


def _single_answer(col: int, row: int) -> dict:
    return {**HEADER, "elements": _elements_for(col, row)}


class FakeOverpass:
    """Answers a one-tile query and a batched one as Overpass does: the
    batch is the tiles' answers in order, each behind its marker."""

    def __init__(self):
        self.queries: list[str] = []

    def __call__(self, query: str) -> dict:
        self.queries.append(query)
        tiles = _tiles_in(query)
        if OSM_TILE_MARKER not in query:
            assert len(tiles) == 1
            return _single_answer(*tiles[0])
        elements = []
        for index, (col, row) in enumerate(tiles):
            elements.append({"type": OSM_TILE_MARKER, "id": index + 1,
                             "tags": {"col": str(col), "row": str(row)}})
            elements.extend(_elements_for(col, row))
        return {**HEADER, "elements": elements}

    @property
    def batched(self) -> list[str]:
        return [q for q in self.queries if OSM_TILE_MARKER in q]


def test_the_measured_staging_shape_is_six_tiles():
    assert len(TILES) == 6


def test_a_batched_query_is_each_tile_query_behind_its_marker():
    query = build_batched_overpass_query(TILES[:3])

    assert query.startswith(f"[out:json][timeout:{linear.OVERPASS_BATCH_TIMEOUT_S}];")
    assert query.count("out geom;") == 3
    for col, row in TILES[:3]:
        marker = f"make {OSM_TILE_MARKER} col={col},row={row};out;"
        assert marker + _tile_statements(col, row) in query
    assert _tiles_in(query) == TILES[:3]


def test_the_one_tile_query_is_unchanged_by_the_split():
    """`build_overpass_query` is header plus the tile's statements; the
    statements are the same text a batch carries."""
    col, row = TILES[0]
    timeout = linear.LINEAR_ASSUMPTIONS["overpass_timeout_s"]
    assert build_overpass_query(col, row) == (
        f"[out:json][timeout:{timeout}];" + _tile_statements(col, row))


def test_six_missing_tiles_are_one_request(tmp_path):
    fake = FakeOverpass()

    payloads = fetch_osm_tiles(TILES, tmp_path, post=fake)

    assert len(fake.queries) == 1
    assert _tiles_in(fake.queries[0]) == TILES
    assert payloads == [_single_answer(*tile) for tile in TILES]


def test_each_tile_is_cached_as_a_one_tile_fetch_would_cache_it(tmp_path):
    """The cache keying is untouched: a tile file written from a batch is
    the file a one-tile fetch writes, header included."""
    fetch_osm_tiles(TILES, tmp_path / "batched", post=FakeOverpass())
    fetch_osm_tiles(TILES, tmp_path / "single", get_json=FakeOverpass())

    for col, row in TILES:
        batched = _tile_path(tmp_path / "batched", col, row)
        single = _tile_path(tmp_path / "single", col, row)
        assert batched.name == single.name == f"osm_tile_v{linear.OSM_QUERY_VERSION}_c{col}_r{row}.json"
        assert json.loads(batched.read_text()) == json.loads(single.read_text())


def test_a_staging_reads_the_same_routes_parking_and_gates_either_way(tmp_path, monkeypatch):
    """The staging path passes no fetcher: it gets the batch through
    `_http_post`, and the routes - ids included - are what the one-tile
    path gives for the same ground."""
    fake = FakeOverpass()
    monkeypatch.setattr(linear, "_http_post", fake)

    routes = routes_for(SIX_TILE_BBOX, tmp_path / "batched")
    parking = parking_for(SIX_TILE_BBOX, tmp_path / "batched")
    gates = gates_for(SIX_TILE_BBOX, tmp_path / "batched")

    assert len(fake.queries) == 1  # parking and gates read the cache
    per_tile = FakeOverpass()
    assert routes == routes_for(SIX_TILE_BBOX, tmp_path / "single", get_json=per_tile)
    assert parking == parking_for(SIX_TILE_BBOX, tmp_path / "single", get_json=per_tile)
    assert gates == gates_for(SIX_TILE_BBOX, tmp_path / "single", get_json=per_tile)
    assert len(per_tile.queries) == 6
    assert [route["properties"]["osm_id"] for route in routes] == \
        [1000 + 10 * col + row for col, row in TILES]
    assert len(gates) == 6


def test_a_warm_cache_makes_no_request(tmp_path):
    fetch_osm_tiles(TILES, tmp_path, post=FakeOverpass())
    fake = FakeOverpass()

    payloads = fetch_osm_tiles(TILES, tmp_path, post=fake)

    assert fake.queries == []
    assert payloads == [_single_answer(*tile) for tile in TILES]


def test_only_the_missing_tiles_are_asked_for(tmp_path):
    """An overlapping earlier staging cached some tiles: the batch carries
    the rest, and only the rest."""
    fetch_osm_tiles(TILES[:2], tmp_path, post=FakeOverpass())
    fake = FakeOverpass()

    payloads = fetch_osm_tiles(TILES, tmp_path, post=fake)

    assert len(fake.queries) == 1
    assert _tiles_in(fake.queries[0]) == TILES[2:]
    assert payloads == [_single_answer(*tile) for tile in TILES]


def test_one_missing_tile_gets_the_one_tile_query(tmp_path):
    fetch_osm_tiles(TILES[1:], tmp_path, post=FakeOverpass())
    fake = FakeOverpass()

    fetch_osm_tiles(TILES, tmp_path, post=fake)

    assert fake.queries == [build_overpass_query(*TILES[0])]


@pytest.mark.parametrize("size, shapes", [
    (4, [4, 2]),
    (5, [5, 1]),     # the lone remainder gets the one-tile query
    (16, [6]),
    (1, [1] * 6),    # batching off: one query per tile, as before
])
def test_batches_are_capped_at_tiles_per_query(tmp_path, size, shapes):
    fake = FakeOverpass()

    payloads = fetch_osm_tiles(TILES, tmp_path, post=fake, tiles_per_query=size)

    assert [len(_tiles_in(query)) for query in fake.queries] == shapes
    assert [tile for query in fake.queries for tile in _tiles_in(query)] == TILES
    assert payloads == [_single_answer(*tile) for tile in TILES]


def test_the_default_batch_is_sixteen_tiles():
    assert linear.OSM_TILES_PER_QUERY == 16


def test_a_failed_batch_raises_its_own_error_and_keeps_the_batches_before_it(tmp_path):
    """`_http_post`'s `OverpassUnavailable` reaches staging unchanged - its
    message is what a planner reads - and what already arrived stays cached
    for the next staging."""
    fake = FakeOverpass()

    def second_batch_fails(query: str) -> dict:
        if fake.queries:
            raise OverpassUnavailable("every mirror said 429", status=429,
                                      endpoint="https://overpass-api.de/api/interpreter")
        return fake(query)

    with pytest.raises(OverpassUnavailable, match="every mirror said 429") as raised:
        fetch_osm_tiles(TILES, tmp_path, post=second_batch_fails, tiles_per_query=3)

    assert raised.value.status == 429
    for tile in TILES[:3]:
        assert _tile_path(tmp_path, *tile).exists()
    for tile in TILES[3:]:
        assert not _tile_path(tmp_path, *tile).exists()


def _batch_answer(tiles, *, drop=None, repeat=None, extra=None, lead=None) -> dict:
    elements = [] if lead is None else [lead]
    for col, row in tiles:
        if (col, row) == drop:
            continue
        marker = {"type": OSM_TILE_MARKER, "id": 1, "tags": {"col": str(col), "row": str(row)}}
        elements.append(marker)
        elements.extend(_elements_for(col, row))
        if (col, row) == repeat:
            elements.append(marker)
    if extra is not None:
        elements.append({"type": OSM_TILE_MARKER, "id": 99,
                         "tags": {"col": str(extra[0]), "row": str(extra[1])}})
    return {**HEADER, "elements": elements}


@pytest.mark.parametrize("answer", [
    _batch_answer(TILES, drop=TILES[-1]),           # cut off before the last tile
    _batch_answer(TILES, repeat=TILES[2]),          # a tile twice
    _batch_answer(TILES, extra=(999, 999)),         # a tile nobody asked for
    _batch_answer(TILES, lead=_elements_for(*TILES[0])[0]),  # an element before any marker
    {**HEADER, "elements": []},                      # nothing at all
], ids=["missing", "repeated", "unasked", "unmarked", "empty"])
def test_an_answer_that_does_not_hold_each_tile_once_caches_nothing(tmp_path, answer):
    with pytest.raises(OverpassUnavailable, match="did not hold each tile once"):
        fetch_osm_tiles(TILES, tmp_path, post=lambda query: answer)

    assert not any(_tile_path(tmp_path, *tile).exists() for tile in TILES)


def test_a_server_runtime_error_caches_nothing(tmp_path):
    """Overpass answers 200 with whatever it gathered and a remark when it
    runs out of time: every tile is then cut short, and none is cached as
    complete."""
    answer = {**_batch_answer(TILES),
              "remark": 'runtime error: Query timed out in "query" at line 1 after 181 seconds.'}

    with pytest.raises(OverpassUnavailable, match="runtime error"):
        fetch_osm_tiles(TILES, tmp_path, post=lambda query: answer)

    assert not any(_tile_path(tmp_path, *tile).exists() for tile in TILES)


def test_a_one_tile_runtime_error_is_not_cached_either(tmp_path):
    answer = {**_single_answer(*TILES[0]),
              "remark": "runtime error: Query run out of memory using about 2048 MB of RAM."}

    with pytest.raises(OverpassUnavailable, match="runtime error"):
        fetch_osm_tiles(TILES[:1], tmp_path, post=lambda query: answer)

    assert not _tile_path(tmp_path, *TILES[0]).exists()


def test_an_injected_per_tile_fetcher_still_gets_one_query_per_tile(tmp_path):
    """The seam every older test uses: `get_json` answers one tile's query
    at a time, and gets exactly that."""
    queries = []

    def per_tile(query: str) -> dict:
        queries.append(query)
        return _single_answer(*_tiles_in(query)[0])

    fetch_osm_tiles(TILES, tmp_path, get_json=per_tile)

    assert queries == [build_overpass_query(*tile) for tile in TILES]
