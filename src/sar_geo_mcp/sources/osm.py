"""OpenStreetMap through Overpass: the source of trails, roads, water,
power lines, parking, gates and barriers.

This module asks Overpass and caches what it answers, tile by tile; it
never says what an element means for a search. `terrain.linear` reads the
cached tiles and classifies them (a walkable way, a lot, a gate), and the
planning modules build on that.

The cache is tile-keyed exactly like LANDFIRE's: one CalTopo map carries
~119 assignments in a few square kilometres, and Overpass is a shared,
rate-limited public service. The query is versioned (`OSM_QUERY_VERSION`):
the version is in every tile's file name and in every staged `osm.json`,
so an answer fetched by an older query is never read as complete.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ..cache import atomic_write_text, note_hit, note_write
from ..geo import Bbox
from .base import Area, FeatureResult, FeatureSource, SourceInfo, SourceUnavailable

#: Public Overpass instances, in the order they are tried. They speak the
#: same API against the same OSM data, so a mirror is a real answer and not
#: a degraded one. The shared endpoint at overpass-api.de stays first out of
#: courtesy to the mirrors, but it is also the one that throttles first: a
#: planning session on a laptop during an incident is exactly when a 429
#: arrives, and a 429 used to mean zero trails and zero roads in the
#: candidate list, with hundreds of DEM-derived ridges quietly taking their
#: place.
#:
#: TWO entries, not three - this used to list a third,
#: overpass.kumi.systems, which read as a second mirror but was not one.
#: DNS, checked 2026-09-20:
#:   overpass.kumi.systems   -> CNAME overpass.private.coffee
#:   overpass.private.coffee -> flanders.servers.private.coffee,
#:                              193.219.97.30, 2a0d:f302:126:78ea::1
#: Both names are the same machine. What it cost: on 2026-09-21 that host
#: completed TLS in 0.37 s and then never answered a query. With three
#: attempts under each of its two names, a staging run could spend the
#: whole 90 s Overpass budget hanging against that ONE dead machine, while
#: overpass-api.de - the other, actually-different mirror - was merely
#: slow (13.5 s for a trivial count query, 6 s just to connect). Nine tiles
#: were needed; two arrived; the staging came back with no trails, no
#: roads, and no parking.
#:
#: Adding a mirror back here later? Two checks, both by hand, because
#: `overpass_endpoints` below cannot do either automatically:
#:
#: 1. EXTENT. A fast mirror can be a trap, not a bargain. overpass.osm.ch
#:    answers a trivial count query in 0.7 s against overpass-api.de's
#:    13.5 s, which is very tempting - but it carries SWISS DATA ONLY: for
#:    bbox (37.330, -121.730, 37.355, -121.700) it returns 0 ways where
#:    overpass-api.de returns 101. A fast mirror that reports "no roads
#:    here" is far more dangerous to a search than a slow one, because the
#:    plan it produces carries no warning at all - nothing times out,
#:    nothing throttles, it just says the area is empty. Verify every
#:    candidate against a NON-EUROPEAN bbox before trusting it.
#: 2. ALIASING. `overpass_endpoints` below drops exact repeats, but it
#:    cannot see a CNAME without a DNS lookup, and this list is read on
#:    every single Overpass attempt - so that check is not something the
#:    code can do for you. Do it by hand, here, in this comment, before
#:    committing: `nslookup <host>` is enough.
OVERPASS_URLS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)

#: Backwards-compatible alias for the single endpoint this module used to have.
OVERPASS_URL = OVERPASS_URLS[0]

#: An agency running its own instance points at it here, comma-separated.
#: This module reads no settings object, so the environment is the seam.
OVERPASS_URLS_ENV = "ZA_OVERPASS_URLS"

#: Attempts per endpoint before moving on to the next one.
OVERPASS_ATTEMPTS_PER_ENDPOINT = 3

#: Wall-clock budget for the whole fetch, across every endpoint and retry.
#: Staging must not hang for minutes while a search is being planned.
OVERPASS_BUDGET_S = 90.0

#: However long a server's `Retry-After` asks for, this is the most we wait.
OVERPASS_MAX_WAIT_S = 30.0

#: The floor under a per-attempt budget share (see `_http_post`). Once the
#: budget is nearly spent, dividing what is left by several mirrors can
#: produce a timeout too thin to complete even a fast connect-and-answer -
#: this is not a promise that an attempt this short can succeed, only a
#: refusal to send one so short it cannot even try.
OVERPASS_MIN_ATTEMPT_S = 5.0

#: Statuses worth trying again: rate limiting and gateway trouble. A 400 is
#: a malformed query, which is our bug, and is never retried.
OVERPASS_RETRY_STATUS = frozenset({429, 502, 503, 504})

_USER_AGENT = "sar-geo-mcp (search and rescue planning)"

OSM_TILE_M = 4000.0  # Web Mercator units; ~3.2 km on the ground at 37 N

#: How many tiles one Overpass request carries (`fetch_osm_tiles`). A
#: staging used to send one request per tile, one after another, and on
#: overpass-api.de the time goes to waiting for the server, not to the
#: answer. Measured 2026-10-21 from one laptop, cold cache: a TRIVIAL query
#: (`node(1);out;`) took 36 s; tile queries took 24-36 s for 89-162 KB; a
#: six-tile staging took 205-214 s, nearly all of it in those six waits
#: (one run failed outright on the sixth tile after a run of 502s). One
#: request for the same six tiles took 39 s, sixteen tiles 40 s (2.5 MB):
#: the wait is per request, so asking once is the whole saving.
#:
#: Why batches and not tiles in parallel: two single-tile requests sent at
#: once that same afternoon got one answer in 35 s and one 429 Too Many
#: Requests at 49 s - overpass-api.de was granting this address one slot,
#: and a second concurrent request only spends the budget on a refusal.
#: The other mirror cannot carry the second request either: it completed
#: TLS and then answered nothing, the 2026-09-21 failure again.
#:
#: Sixteen because it was measured and covers a 4x4 block - the largest
#: staging (9,000 m plus the buffer) is five or six tiles a side, so two or
#: three requests at worst.
OSM_TILES_PER_QUERY = 16

#: The `[timeout:]` a one-tile query declares, in seconds.
OVERPASS_TIMEOUT_S = 60

#: The `[timeout:]` a batched query declares, in seconds: what both
#: measured batches (six and sixteen tiles) were sent with. A one-tile
#: query keeps `OVERPASS_ASSUMPTIONS["overpass_timeout_s"]`. The client side
#: is unchanged: `_http_post`'s per-request timeout and budget still decide
#: how long this process waits.
OVERPASS_BATCH_TIMEOUT_S = 180

#: The element type of the marker `build_batched_overpass_query` puts in
#: front of each tile's answer. Not an OSM type, so no real element can be
#: mistaken for it.
OSM_TILE_MARKER = "osm_tile"

#: Query shape version. Bumped whenever the Overpass filters change, so a
#: cached tile from the old query is not mistaken for a complete answer:
#: the file name carries it, and old files simply stop being read (they
#: remain eligible for size eviction as before). Bumped to 3 for parking
#: and public hardstand: a v2 tile has none of that, and reading it as
#: complete would say "no lots here" instead of "not fetched yet".
#: Bumped to 4 for `power=minor_line`: distribution wires, the ones most
#: often beside a meadow, close ground for a landing zone (operator's
#: ruling, 2026-09-23). A version-3 tile has none of them, and reading it
#: as complete would say "no wire" where one was never asked for.
#: 5: highway, access and tracktype kept on every route; gate nodes fetched
#: (egress, 0.18.0). A version-4 tile has no gates, and reading it as
#: complete would say "no gate on this road" where none was asked for.
OSM_QUERY_VERSION = 5

#: The versions at which the query first asked for each thing, so a reader
#: of an older `osm.json` can say exactly what it lacks rather than "older
#: than current": barriers and water edges arrived at 2, parking and public
#: hardstand at 3, distribution power lines (`power=minor_line`) at 4, road
#: class, access, tracktype and gates at 5. Compare against the one for the
#: thing you read, never against `OSM_QUERY_VERSION`: that moves every time
#: anything is added, and a file that already has what you read is not
#: stale for you.
OSM_QUERY_VERSION_BARRIERS = 2
OSM_QUERY_VERSION_PARKING = 3
OSM_QUERY_VERSION_MINOR_LINES = 4
OSM_QUERY_VERSION_ROUTE_TAGS = 5


def overpass_endpoints() -> tuple[str, ...]:
    """The endpoints to try, in order: the environment override if it is set
    and non-empty, otherwise the built-in mirror list. Exact repeats are
    dropped, first occurrence winning - an agency pasting the same URL twice
    should not pay for it twice inside the budget.

    This only catches an EXACT repeat, string for string. An aliased host -
    two different names resolving to the same machine, the way
    overpass.kumi.systems and overpass.private.coffee once did - is not
    detectable here without a DNS lookup on every attempt, which is not a
    cost this function will impose. See the note above `OVERPASS_URLS` for
    how that check is done instead: by hand, once, when a mirror is added.
    """
    raw = os.environ.get(OVERPASS_URLS_ENV, "")
    configured = tuple(part.strip() for part in raw.split(",") if part.strip())
    candidates = configured or OVERPASS_URLS
    seen: set[str] = set()
    deduped = []
    for url in candidates:
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    return tuple(deduped)


#: What `_host` says when a URL's host cannot be told apart from its userinfo.
UNREADABLE_ENDPOINT = "an endpoint whose host could not be read safely"


def _host(url: str) -> str:
    """The host, and its port when one is given, for anything a person or a
    tool response reads - never the userinfo, path, query or fragment.

    An endpoint from `ZA_OVERPASS_URLS` may carry basic-auth credentials
    (`user:password@`) or a key in its query string, and this name ends up
    in `OverpassUnavailable`'s message, which `staging` hands to a planner
    verbatim, and in `OVERPASS_ASSUMPTIONS`, which every routes result
    returns. Parsed with `urlsplit`, not split by hand: the hand-split
    version returned "h?apikey=SECRET" for a URL with no path.

    A '/', '?' or '#' in an unencoded password ends the authority early,
    and any parser then reads part of the userinfo as the host. Rather than
    guess, such a URL - an '@' left in the path, query or fragment, or a
    port that is not a number - names no host at all. (A query that
    legitimately holds an '@' loses its host name too; that is the cheaper
    mistake.) Never raises: a failure path is the worst place to
    acquire a new way to raise.
    """
    from urllib.parse import urlsplit

    try:
        parts = urlsplit(url.strip())
        host = parts.hostname
        port = parts.port
    except ValueError:
        return UNREADABLE_ENDPOINT
    # An '@' after the authority means a '/', '?' or '#' in the userinfo
    # ended it early, so what urlsplit calls the host is part of a password.
    if not host or "@" in parts.path or "@" in parts.query or "@" in parts.fragment:
        return UNREADABLE_ENDPOINT
    if ":" in host:  # IPv6: brackets back, so a port stays readable
        host = f"[{host}]"
    return f"{host}:{port}" if port is not None else host


def reported_endpoints() -> list[str]:
    """`overpass_endpoints()` as it may be reported: hosts only (`_host`)."""
    return [_host(url) for url in overpass_endpoints()]


class OverpassUnavailable(RuntimeError):
    """Every Overpass endpoint refused, timed out, or the budget ran out.

    A `RuntimeError` so the existing `except` in `staging.stage_terrain`
    still catches it; the message is written for a planner rather than for a
    stack trace, and `status`/`endpoint` carry the last thing that happened.
    """

    def __init__(self, message: str, *, status: int | None = None,
                 endpoint: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.endpoint = endpoint


OVERPASS_ASSUMPTIONS = {
    "osm_tile_m": OSM_TILE_M,
    "overpass_timeout_s": OVERPASS_TIMEOUT_S,
    "source": "OpenStreetMap via Overpass",
    "query_version": OSM_QUERY_VERSION,
    # Which mirrors a run was allowed to use, and how long it was allowed to
    # keep trying, belong in provenance: "no trails here" reads very
    # differently once you can see the fetch gave up after 90 seconds.
    # Hosts, never the configured URLs: this dict is every routes result's
    # `assumptions`, and a URL may carry credentials or a key (`_host`).
    "overpass_endpoints": reported_endpoints(),
    "overpass_budget_s": OVERPASS_BUDGET_S,
    "overpass_attempts_per_endpoint": OVERPASS_ATTEMPTS_PER_ENDPOINT,
    # The number above is a CEILING, not what every endpoint got, and the
    # difference matters to whoever is weighing "no trails here": a host
    # that was busy got up to three tries, a host that hung got one. Left
    # bare, the field reads as three attempts against a mirror that was
    # actually asked a single time - which overstates how hard the fetch
    # looked before giving up. See `_http_post`.
    "overpass_attempts_note": (
        "a ceiling, not a count: a retryable status (429/502/503/504) gets up "
        "to this many attempts with backoff, but a timeout or transport "
        "failure means the host is not answering, so it is abandoned after "
        "one attempt and the rest of the budget goes to the next mirror"),
    "overpass_attempt_share": "the budget still unspent, divided by the mirrors not yet tried",
}
BARRIER_TAGS = frozenset({"fence", "wall"})

#: `barrier` values read as a gate across a way (query version 5). A gate is
#: SAID on an egress route, never a reason to exclude one: the planner can
#: call for the combination or the ranger. A bollard stops a car but not a
#: litter team, and is not asked for.
GATE_BARRIERS = frozenset({"gate", "lift_gate", "swing_gate"})
WATER_EDGE_NATURAL = frozenset({"water", "coastline"})

#: Site tags the query fetches: public ground whose own parking OSM often
#: leaves unmapped. What they mean for a search is `terrain.linear`'s
#: (SITE_TAGS) and `planning.parking`'s to say.
SITE_TAGS = (
    ("amenity", frozenset({"school", "place_of_worship", "community_centre"})),
    ("leisure", frozenset({"park", "sports_centre", "pitch"})),
    ("landuse", frozenset({"recreation_ground"})),
    ("tourism", frozenset({"picnic_site"})),
    ("highway", frozenset({"rest_area", "services"})),
)


def tiles_covering_osm(bbox_3857: Bbox) -> list[tuple[int, int]]:
    min_col = math.floor(bbox_3857[0] / OSM_TILE_M)
    max_col = math.ceil(bbox_3857[2] / OSM_TILE_M) - 1
    min_row = math.floor(bbox_3857[1] / OSM_TILE_M)
    max_row = math.ceil(bbox_3857[3] / OSM_TILE_M) - 1
    return [
        (col, row)
        for row in range(min_row, max_row + 1)
        for col in range(min_col, max_col + 1)
    ]


def _tile_bounds_wgs84(col: int, row: int) -> tuple[float, float, float, float]:
    from pyproj import Transformer

    to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    west, south = to_wgs84.transform(col * OSM_TILE_M, row * OSM_TILE_M)
    east, north = to_wgs84.transform((col + 1) * OSM_TILE_M, (row + 1) * OSM_TILE_M)
    return (south, west, north, east)


#: Keys already fetched by their own bare-key clause in the query below - a
#: SITE_TAGS entry for one of these would just repeat that clause.
_KEYS_ALREADY_FETCHED = frozenset({"highway"})


def _way_clause(key: str, values: frozenset[str], area: str,
                element: str = "way") -> str:
    """One `way[...]` filter from a key and a value set, e.g. `("amenity",
    frozenset({"school", "place_of_worship", "community_centre"}))` becomes
    `way["amenity"~"^(community_centre|place_of_worship|school)$"](area)`.
    A single-value entry, e.g. `("tourism", frozenset({"picnic_site"}))`,
    needs no alternation and is emitted as a plain equality instead - which
    is also the form the hand-written single-value clauses already use, so
    a generated clause and a literal one read the same in a debugger.

    Sorted, not the frozenset's own iteration order: a `frozenset` iterates
    in an order that can vary between runs and Python versions, and an
    unstable query string would change what the cache filename means and
    make tile-cache tests flaky.

    `element="node"` gives the same filter on nodes, for things OSM maps as
    a point on a way - a gate - rather than as a way of their own.
    """
    ordered = sorted(values)
    if len(ordered) == 1:
        return f'{element}["{key}"="{ordered[0]}"]({area})'
    alternation = "|".join(ordered)
    return f'{element}["{key}"~"^({alternation})$"]({area})'


def _tile_statements(col: int, row: int) -> str:
    """A tile's statements, without the `[out:json]` header: the body of the
    one query a tile ever gets (`build_overpass_query`), and of its part in
    a batched query (`build_batched_overpass_query`).

    TWO GROUPS IN ONE QUERY, on purpose: the lines a team walks and the
    ground a team parks on come out of the same tile, are cached in the
    same file, and cost one request between them. A second query for
    parking would double the Overpass traffic of every staging for data
    that was already in the box.
    """
    south, west, north, east = _tile_bounds_wgs84(col, row)
    area = f"{south:.6f},{west:.6f},{north:.6f},{east:.6f}"
    hardstand_clauses = "".join(
        _way_clause(key, values, area) + ";"
        for key, values in SITE_TAGS
        if key not in _KEYS_ALREADY_FETCHED
    )
    return (
        f'(way["highway"]({area});way["waterway"]({area});'
        f'{_way_clause("natural", WATER_EDGE_NATURAL, area)};'
        f'way["power"="line"]({area});way["power"="minor_line"]({area});'
        f'way["railway"]({area});'
        f'{_way_clause("barrier", BARRIER_TAGS, area)};'
        f'way["man_made"="pipeline"]({area});'
        f'way["landuse"="reservoir"]({area});'
        f'node["amenity"="parking"]({area});way["amenity"="parking"]({area});'
        f'{_way_clause("barrier", GATE_BARRIERS, area, element="node")};'
        f"{hardstand_clauses});"
        f"out geom;"
    )


def build_overpass_query(col: int, row: int) -> str:
    """The one Overpass query a tile ever gets, on its own."""
    timeout = OVERPASS_ASSUMPTIONS["overpass_timeout_s"]
    return f"[out:json][timeout:{timeout}];" + _tile_statements(col, row)


def build_batched_overpass_query(tiles: list[tuple[int, int]]) -> str:
    """Several tiles in one request: each tile's own statements, in order,
    each preceded by a marker element naming its tile.

    `make osm_tile col=..,row=..; out;` emits a derived element of type
    `osm_tile` - a type no OSM element has - just before the tile's own
    `out geom`, so `_split_batch` can cut the answer back into the exact
    per-tile payloads the one-tile query returns, with no bbox arithmetic
    of its own (Overpass decides which ways touch which tile, as it always
    has). Checked against the live server on 2026-10-21: four tiles split
    from a six-tile answer were element-for-element identical to the four
    single-tile answers cached minutes earlier.
    """
    return f"[out:json][timeout:{OVERPASS_BATCH_TIMEOUT_S}];" + "".join(
        f"make {OSM_TILE_MARKER} col={col},row={row};out;" + _tile_statements(col, row)
        for col, row in tiles)


def _retry_after_seconds(raw: str | None) -> float | None:
    """`Retry-After` as seconds, whether the server sent seconds or a date.
    Capped: a throttled endpoint that asks for an hour is a reason to try the
    next mirror, not a reason to stall the staging run."""
    if not raw:
        return None
    raw = raw.strip()
    try:
        return max(0.0, min(float(raw), OVERPASS_MAX_WAIT_S))
    except ValueError:
        pass
    from datetime import datetime, timezone
    from email.utils import parsedate_to_datetime

    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    delay = (when - datetime.now(timezone.utc)).total_seconds()
    return max(0.0, min(delay, OVERPASS_MAX_WAIT_S))


def _backoff_seconds(attempt: int, rng) -> float:
    """Exponential, with jitter so a laptop full of assignments does not send
    every retry on the same second."""
    return min(OVERPASS_MAX_WAIT_S, 2.0 ** attempt + rng.random())


def _default_client_factory(timeout: float) -> Callable[[], "object"]:
    def build():
        import httpx

        return httpx.Client(timeout=timeout, headers={"User-Agent": _USER_AGENT})

    return build


def _http_post(query: str, *, client_factory: Callable[[], object] | None = None,
               sleep: Callable[[float], None] = time.sleep,
               monotonic: Callable[[], float] = time.monotonic,
               rng=random) -> dict:
    """POST the query to each Overpass endpoint in turn, retrying the ones
    that are merely busy, and give up with a sentence a planner can act on.

    Raises `OverpassUnavailable` when every endpoint is exhausted or the
    wall-clock budget runs out. A 400 is not retried and is re-raised as
    httpx's own `HTTPStatusError`: a malformed query is our bug, not a
    service outage, and dressing it up as one would hide it.
    """
    import httpx

    endpoints = overpass_endpoints()
    OVERPASS_ASSUMPTIONS["overpass_endpoints"] = reported_endpoints()
    request_timeout = float(OVERPASS_ASSUMPTIONS["overpass_timeout_s"]) + 30.0
    if client_factory is None:
        client_factory = _default_client_factory(request_timeout)

    started = monotonic()
    last_status: int | None = None
    last_endpoint: str | None = None
    last_reason = "no endpoint was tried"

    def remaining() -> float:
        return OVERPASS_BUDGET_S - (monotonic() - started)

    def give_up(what: str) -> OverpassUnavailable:
        where = f" from {_host(last_endpoint)}" if last_endpoint else ""
        return OverpassUnavailable(
            f"OpenStreetMap could not be reached: {what} "
            f"(last: {last_reason}{where}). Trails and roads will be missing "
            "from this area until it succeeds.",
            status=last_status, endpoint=last_endpoint,
        )

    budget_note = f"the {OVERPASS_BUDGET_S:.0f} s Overpass budget ran out"

    with client_factory() as client:
        for index, endpoint in enumerate(endpoints):
            # One dead endpoint must not be able to spend every second of
            # the budget before a healthy mirror is tried even once - that
            # is exactly how the 2026-09-21 staging lost its trails, roads,
            # and parking, while overpass.private.coffee sat completely
            # untried behind a connection that completed TLS and then just
            # went quiet. So each endpoint's attempts are capped to a SHARE
            # of whatever budget is still unspent, divided by the mirrors
            # not yet tried (this one included) rather than by the whole
            # list: dividing by what remains, not by the total, means the
            # share GROWS as the list is used up, so the last mirror still
            # gets a real attempt instead of whatever scrap the earlier
            # ones left behind.
            endpoints_not_yet_tried = len(endpoints) - index
            for attempt in range(OVERPASS_ATTEMPTS_PER_ENDPOINT):
                left = remaining()
                if left <= 0.0:
                    raise give_up(budget_note)
                share = max(OVERPASS_MIN_ATTEMPT_S, remaining() / endpoints_not_yet_tried)
                last_endpoint = endpoint
                try:
                    response = client.post(
                        endpoint, data={"data": query},
                        headers={"User-Agent": _USER_AGENT},
                        timeout=min(request_timeout, share, left),
                    )
                except (httpx.TimeoutException, httpx.TransportError) as error:
                    last_status = None
                    last_reason = type(error).__name__
                    # A STATUS says "busy, try again" - a 429 or a 503 is
                    # the server itself telling us to come back, and the
                    # attempt-and-backoff loop below exists for exactly
                    # that. A TIMEOUT or transport failure says something
                    # different: this host is not answering at all, and
                    # trying it again is not "try again", it is "wait the
                    # same silence out a second and a third time." That is
                    # what spent the 2026-09-21 budget - three attempts
                    # against flanders.servers.private.coffee, which had
                    # already gone quiet after completing TLS, with the
                    # working mirror still untried. So a hang or a
                    # transport failure abandons this endpoint after ONE
                    # attempt and moves straight to the next mirror, which
                    # the per-attempt share above then hands the budget
                    # this endpoint did not get to waste.
                    break
                if response.status_code not in OVERPASS_RETRY_STATUS:
                    response.raise_for_status()
                    return response.json()
                last_status = response.status_code
                last_reason = (
                    f"{response.status_code} {response.reason_phrase}".strip())
                wait = _retry_after_seconds(response.headers.get("Retry-After"))
                if wait is None:
                    wait = _backoff_seconds(attempt, rng)

                if attempt + 1 == OVERPASS_ATTEMPTS_PER_ENDPOINT:
                    break  # the next mirror is a better use of the budget
                left = remaining()
                if left <= 0.0:
                    raise give_up(budget_note)
                sleep(min(wait, left))

    raise give_up("every Overpass endpoint refused or timed out")


def _tile_path(cache_dir: Path, col: int, row: int) -> Path:
    return Path(cache_dir) / f"osm_tile_v{OSM_QUERY_VERSION}_c{col}_r{row}.json"


def _refuse_partial(payload: dict, what: str) -> None:
    """Raise rather than cache an answer Overpass itself says is incomplete.

    A query that runs out of time or memory on the server still comes back
    200, with whatever was gathered so far and a `remark` saying "runtime
    error: ...". Cached, that would be read for the life of the cache as
    the complete tile - "no trail here" where the server simply stopped
    looking - so it is raised as unavailable instead, and the next staging
    asks again.
    """
    remark = str(payload.get("remark") or "")
    if "error" in remark.lower():
        raise OverpassUnavailable(
            f"OpenStreetMap could not be reached: the server stopped {what} "
            "partway (a runtime error from Overpass). Trails and roads will "
            "be missing from this area until it succeeds.")


def _write_tile(cache_dir: Path, col: int, row: int, payload: dict) -> None:
    destination = _tile_path(cache_dir, col, row)
    # A unique temp name: two sessions staging overlapping ground fetch
    # this same tile at the same instant. See `cache.atomic`.
    atomic_write_text(destination, json.dumps(payload))
    note_write(cache_dir, destination, layer="osm")


def fetch_osm_tile(col: int, row: int, cache_dir: Path,
                   get_json: Callable[[str], dict] = _http_post) -> dict:
    """One cached Overpass response, shared by every assignment on the tile."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    destination = _tile_path(cache_dir, col, row)
    if destination.exists():
        note_hit(cache_dir, destination)
        return json.loads(destination.read_text(encoding="utf-8"))

    payload = get_json(build_overpass_query(col, row))
    _refuse_partial(payload, "the query for this tile")
    _write_tile(cache_dir, col, row, payload)
    return payload


def _split_batch(payload: dict, tiles: list[tuple[int, int]]) -> dict[tuple[int, int], dict]:
    """A batched answer cut back into the per-tile payloads it is made of.

    Each tile's elements follow its `osm_tile` marker (see
    `build_batched_overpass_query`), so a tile's payload is exactly what
    its own `build_overpass_query` returns - the same statements, the same
    bbox, the same `out geom` - plus Overpass's top-level `version`,
    `generator` and `osm3s`, as a single-tile file always carried.

    A marker missing, repeated, or for a tile that was not asked for means
    the answer is not the one the query describes, and nothing from it is
    cached: one tile cached from a truncated answer would be read as
    complete for as long as the cache keeps it.
    """
    _refuse_partial(payload, f"the query for {len(tiles)} tiles")
    wanted = set(tiles)
    header = {key: value for key, value in payload.items()
              if key not in ("elements", "remark")}
    split: dict[tuple[int, int], dict] = {}
    current: list[dict] | None = None
    for element in payload.get("elements", []):
        if element.get("type") == OSM_TILE_MARKER:
            tags = element.get("tags") or {}
            try:
                tile = (int(tags["col"]), int(tags["row"]))
            except (KeyError, TypeError, ValueError):
                tile = None
            if tile not in wanted or tile in split:
                break
            current = []
            split[tile] = {**header, "elements": current}
        elif current is None:
            break
        else:
            current.append(element)
    else:
        if len(split) == len(wanted):
            return split
    raise OverpassUnavailable(
        "OpenStreetMap could not be reached: the answer to a query for "
        f"{len(tiles)} tiles did not hold each tile once. Trails and roads "
        "will be missing from this area until it succeeds.")


def fetch_osm_tiles(tiles: list[tuple[int, int]], cache_dir: Path,
                    get_json: Callable[[str], dict] | None = None, *,
                    post: Callable[[str], dict] | None = None,
                    tiles_per_query: int | None = None) -> list[dict]:
    """Every tile's payload, in the order `tiles` lists them.

    The tiles not yet cached are fetched `tiles_per_query` (default
    `OSM_TILES_PER_QUERY`) to an Overpass request through `post` (default
    `_http_post`, with every mirror, retry and budget rule above), and each
    is cached in its own tile file exactly as a one-tile fetch caches it,
    so the cache keying - and every staging that overlaps this one later -
    is unchanged. A single missing tile is asked for on its own, with the
    same query it always had.

    `get_json`, when given, is the per-tile seam: one query per tile, as
    every fetch was before batching (and as the tests that inject an answer
    per query still expect). Left None, the batched path is taken.

    Batches are fetched in order, one after another. A failed batch raises
    its own error - `OverpassUnavailable` from `_http_post`, unchanged -
    and the batches before it stay cached for the next staging.
    """
    cache_dir = Path(cache_dir)
    tiles = list(tiles)
    if get_json is not None:
        return [fetch_osm_tile(col, row, cache_dir, get_json) for col, row in tiles]
    if post is None:
        post = _http_post

    size = max(1, int(tiles_per_query or OSM_TILES_PER_QUERY))
    missing = list(dict.fromkeys(
        tile for tile in tiles if not _tile_path(cache_dir, *tile).exists()))
    fetched: dict[tuple[int, int], dict] = {}
    if len(missing) > 1:
        cache_dir.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(missing), size):
            batch = missing[start:start + size]
            if len(batch) == 1:
                continue  # left to `fetch_osm_tile` below, with its own query
            for (col, row), payload in _split_batch(
                    post(build_batched_overpass_query(batch)), batch).items():
                _write_tile(cache_dir, col, row, payload)
                fetched[(col, row)] = payload
    return [fetched[tile] if tile in fetched else fetch_osm_tile(*tile, cache_dir, post)
            for tile in tiles]


#: The most elements one generic query returns, whatever `limit` asks.
QUERY_MAX_FEATURES = 5000


def _matches(tags: Mapping[str, str], filters: Mapping[str, Any]) -> bool:
    """Every filter holds: a key mapped to True (present), a string (equal)
    or a list of strings (one of them)."""
    for key, wanted in filters.items():
        value = tags.get(key)
        if wanted is True:
            if value is None:
                return False
        elif isinstance(wanted, str):
            if value != wanted:
                return False
        elif isinstance(wanted, (list, tuple, set, frozenset)):
            if value not in wanted:
                return False
        else:
            raise ValueError(f"filter {key!r} must be true, a value or a list of values")
    return True


def _feature(element: dict) -> dict | None:
    """One Overpass element as a GeoJSON feature in WGS84, or None for one
    with no geometry (a tile marker, a way Overpass returned bare)."""
    properties = {"osm_type": element.get("type"), "osm_id": element.get("id"),
                  "tags": dict(element.get("tags") or {})}
    if element.get("type") == "node":
        if element.get("lat") is None or element.get("lon") is None:
            return None
        geometry = {"type": "Point", "coordinates": [element["lon"], element["lat"]]}
    elif element.get("type") == "way":
        points = [[point["lon"], point["lat"]] for point in element.get("geometry") or []
                  if point and point.get("lat") is not None]
        if len(points) < 2:
            return None
        closed = len(points) >= 4 and points[0] == points[-1]
        geometry = ({"type": "Polygon", "coordinates": [points]} if closed
                    else {"type": "LineString", "coordinates": points})
    else:
        return None
    return {"type": "Feature", "geometry": geometry, "properties": properties}


def _touches(feature: dict, bbox_wgs84: tuple[float, float, float, float]) -> bool:
    west, south, east, north = bbox_wgs84
    geometry = feature["geometry"]
    points = ([geometry["coordinates"]] if geometry["type"] == "Point" else
              geometry["coordinates"][0] if geometry["type"] == "Polygon" else
              geometry["coordinates"])
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return not (max(xs) < west or min(xs) > east or max(ys) < south or min(ys) > north)


class OsmSource(FeatureSource):
    """OpenStreetMap through Overpass, as a source.

    `query` returns the elements the versioned tile query fetched over an
    area - ways and nodes with their own tags, as GeoJSON - filtered by tag
    when asked. It reads the same cached tiles staging reads, so a query
    over staged ground costs Overpass nothing; it can only return tags the
    tile query asks for (see `_tile_statements`)."""

    info = SourceInfo(
        name="osm",
        title="OpenStreetMap (Overpass)",
        kind="features",
        provider="OpenStreetMap contributors, through public Overpass instances",
        default_url=OVERPASS_URLS[0],
        attribution="© OpenStreetMap contributors",
        coverage="global, volunteer-mapped and uneven",
        url_env=OVERPASS_URLS_ENV,
        shared_service=True,
        notes=(
            "Fetched by a fixed, versioned query (version "
            f"{OSM_QUERY_VERSION}): ways and nodes for highways, waterways, water "
            "edges, power lines, railways, barriers, pipelines, reservoirs, parking, "
            "gates and public sites.",
            "Many social trails, ranch roads and fences are not mapped: no element "
            "is not proof of no way.",
        ),
    )

    def endpoint(self) -> str:
        """The first endpoint tried, by host: an override may carry a key."""
        configured = [part.strip() for part in
                      (self.env.get(OVERPASS_URLS_ENV) or "").split(",") if part.strip()]
        return configured[0] if configured else self.info.default_url

    def query(self, area: Area, *, filters: Mapping[str, Any] | None = None,
              limit: int | None = None) -> FeatureResult:
        from pyproj import Transformer

        cache_dir = self.require_cache()
        tiles = tiles_covering_osm(area.bbox_3857)
        post = None
        if self.transport is not None:
            transport = self.transport
            post = lambda query: transport.post_json(self.endpoint(), {"data": query})  # noqa: E731
        try:
            payloads = fetch_osm_tiles(tiles, cache_dir, post=post)
        except OverpassUnavailable as error:
            raise SourceUnavailable(self.info.name, str(error)) from error
        to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        west, south = to_wgs84.transform(area.bbox_3857[0], area.bbox_3857[1])
        east, north = to_wgs84.transform(area.bbox_3857[2], area.bbox_3857[3])
        seen: set[tuple[str, int]] = set()
        features: list[dict] = []
        for payload in payloads:
            for element in payload.get("elements", []):
                identity = (element.get("type"), element.get("id"))
                if identity in seen:
                    continue
                seen.add(identity)
                if filters and not _matches(element.get("tags") or {}, filters):
                    continue
                feature = _feature(element)
                if feature is not None and _touches(feature, (west, south, east, north)):
                    features.append(feature)
        cap = QUERY_MAX_FEATURES if limit is None else min(limit, QUERY_MAX_FEATURES)
        return FeatureResult(features=features[:cap], total=len(features),
                             attribution=self.info.attribution,
                             truncated=len(features) > cap)
