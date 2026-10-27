"""Trails, roads, waterways, and staging sites (public parking and hardstand)
from OpenStreetMap.

CalTopo's rendered road layers are pixels; measuring where a trail crosses an
assignment boundary needs vector geometry, which is what Overpass provides.

The cache is tile-keyed exactly like evt.py: one CalTopo map carries ~119
assignments in a few square kilometres, and Overpass is a shared, rate-limited
public service.
"""

import json
import math
import os
import random
import time
from collections.abc import Callable
from pathlib import Path

from ..geo import Bbox
from ..cache import atomic_write_text, note_hit, note_write

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

#: The `[timeout:]` a batched query declares, in seconds: what both
#: measured batches (six and sixteen tiles) were sent with. A one-tile
#: query keeps `LINEAR_ASSUMPTIONS["overpass_timeout_s"]`. The client side
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
    verbatim, and in `LINEAR_ASSUMPTIONS`, which every routes result
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


LINEAR_ASSUMPTIONS = {
    "osm_tile_m": OSM_TILE_M,
    "overpass_timeout_s": 60,
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

_TRAIL = {"path", "footway", "bridleway", "steps", "cycleway"}
_TRACK = {"track"}
_ROAD = {
    "service", "residential", "unclassified", "tertiary", "secondary",
    "primary", "trunk", "motorway", "road", "living_street",
    "tertiary_link", "secondary_link", "primary_link", "trunk_link",
    "motorway_link", "pedestrian",
}
_BARRIER_TAGS = frozenset({"fence", "wall"})

#: `barrier` values read as a gate across a way (query version 5). A gate is
#: SAID on an egress route, never a reason to exclude one: the planner can
#: call for the combination or the ranger. A bollard stops a car but not a
#: litter team, and is not asked for.
GATE_BARRIERS = frozenset({"gate", "lift_gate", "swing_gate"})
_WATER_EDGE_NATURAL = frozenset({"water", "coastline"})

# ---------------------------------------------------------------------------
# Site rules: what ground a search can put vehicles on.
#
# From here to `site_flags` decides whether an OSM element is staging ground
# at all, and what a planner needs to know before sending rigs there.
# `parking_for`, near the bottom of the file (it needs the tile-fetch
# machinery the routes section below defines first), is the one reader that
# turns these rules into rows. `routes_for`, also below, deliberately calls
# `is_site_shaped` rather than `classify_site` for its own exclusion guard -
# see that function's docstring for why the two are not interchangeable.
# ---------------------------------------------------------------------------

#: The two kinds of ground a search can put vehicles on. `parking` is a lot
#: OSM has actually mapped; `hardstand` is a public site whose lot it has
#: not, which is a weaker claim and is flagged as one.
SITE_PARKING = "parking"
SITE_HARDSTAND = "hardstand"

#: An `access` value that means "not you" - or not without asking first.
#: 2026-09-09 operator ruling: these are FLAGGED, not refused. Whether a lot
#: is private, permit-only, or residents/customers-only is, in the
#: operator's words, "something that just let user know" - a note on the
#: row, not a reason the planner never sees the row at all. The planner is a
#: trained search manager looking at a map at 06:00: a locked gate is a fact
#: they can verify with one phone call to the owner, faster than this module
#: can guess, and a lot this module never shows them is a lot they cannot
#: verify at all. `site_flags` names which of these values landed here so
#: the planner knows which call to make.
ACCESS_FLAGGED = frozenset({"private", "no", "permit", "customers"})

#: An `access` value that means this ground belongs to whoever lives there.
#: 2026-09-09 operator ruling: EXCLUDED from `is_site_shaped`, not flagged
#: like ACCESS_FLAGGED above - `residents` used to live in that set,
#: producing an `access_residents` flag, and it has moved out rather than
#: simply being deleted so the split stays visible to the next reader.
#:
#: The distinction is not degree of restriction, it is kind. A locked
#: commercial gate (`private`, `permit`, `customers`, `no`) is a phone call
#: the planner - a trained search manager - is qualified to make. A
#: resident's parking space is somebody's home: staging a SAR team there at
#: 06:00 is not a decision this module should hand the planner as a row on
#: a map, because getting it wrong makes trouble for the team and for the
#: agency, not just a wasted drive. The operator's ruling: avoid apartment
#: and residential parking; trailhead parking is fine. Do not fold this back into ACCESS_FLAGGED - that
#: would make every `access_residents` row visible again, right after the
#: whole point of this ruling was to make sure it never is.
ACCESS_EXCLUDED = frozenset({"residents"})

#: A `parking` sub-tag for covered or enclosed parking - no comms, no
#: helicopter, and a rig with a light bar does not fit under the entrance.
#: 2026-09-09 operator ruling: flagged (`covered_deck`), not refused, for
#: the same reason as ACCESS_FLAGGED above - deciding whether that still
#: rules a site out is the planner's call, not this module's guess. A
#: parking structure like this is a commercial building, not a home, so it
#: gets the phone-call treatment, not the exclusion below.
PARKING_FLAGGED = frozenset({"underground", "multi-storey", "rooftop"})

#: A `parking` sub-tag naming ground this module excludes outright rather
#: than flags - the same "somebody's home, not a phone call" reasoning as
#: ACCESS_EXCLUDED above, plus one more case that is not about homes at
#: all:
#: - `garage_boxes`, `carports`, `sheds` are private residential garages.
#:   `carports` and `sheds` used to live in PARKING_FLAGGED, producing
#:   `covered_deck`; they moved out on the same 2026-09-09 ruling as
#:   ACCESS_EXCLUDED, and for the same reason - do not fold them back in.
#: - `lane`, `street_side` are on-street parking spaces along a road, not a
#:   lot a team can stage in at all. There is no ground here to offer,
#:   flagged or otherwise, so these are excluded rather than flagged for a
#:   different reason than the residential values above: not "whose home is
#:   this", but "this was never a lot".
PARKING_EXCLUDED = frozenset({
    "garage_boxes", "carports", "sheds", "lane", "street_side",
})

#: Public sites whose parking OSM frequently has not mapped as its own
#: polygon: a school, a church, a ballfield. Offered only when no mapped lot
#: falls inside them (`planning.parking`), and always flagged
#: `parking_unverified` - this is "a place with room", not "a surface".
HARDSTAND_TAGS = (
    ("amenity", frozenset({"school", "place_of_worship", "community_centre"})),
    ("leisure", frozenset({"park", "sports_centre", "pitch"})),
    ("landuse", frozenset({"recreation_ground"})),
    ("tourism", frozenset({"picnic_site"})),
    ("highway", frozenset({"rest_area", "services"})),
)


def _tag(tags: dict, key: str) -> str:
    """A tag's value, stripped and lowercased, so `Parking`, `PRIVATE`, and a
    trailing space all compare the same as the canonical spelling every set
    in this module is written in. Every tag comparison below goes through
    this - `amenity`, `leisure`, `access`, `fee` included - so which tags
    happen to arrive normalised from Overpass is not something this module
    has to track."""
    return str((tags or {}).get(key) or "").strip().lower()


def _access_parts(tags: dict) -> list[str]:
    """`access` split on `;`, each part normalised. OSM stacks values there:
    `private;customers` means "private, and customers may also come" - not
    "pick whichever reads better" - so a lot tagged that way is exactly as
    closed as one tagged `private` alone, and every part has to be checked
    against ACCESS_FLAGGED, not just the one a naive read would notice
    first."""
    return [
        stripped
        for part in _tag(tags, "access").split(";")
        if (stripped := part.strip())
    ]


def is_site_shaped(tags: dict) -> bool:
    """Does this element carry one of the tags query version 3 added for
    staging ground - the structural question, independent of whether a
    search may actually use it?

    This is the ONE place that answers "is this element site-shaped";
    `classify_site` below builds on it rather than re-testing the tags
    itself. `routes_for`'s guard calls this function and not
    `classify_site`, and that choice is deliberate rather than
    interchangeable: a PRIVATE school is as much "not a route" as a public
    one, and before this function existed, `routes_for` filtered on
    `classify_site` instead - which let every refused site (a gated park,
    an underground deck, a customers-only lot) leak through as a route of
    class OTHER, which was most of the parking tagged in a suburb.
    `classify_site`'s access/deck refusals are gone as of the 2026-09-09
    ruling below, so the two functions happen to agree on every input
    today - but a future refusal added back to `classify_site` (or to this
    function) must not make `routes_for` start asking the offer question
    again, which is why the guard names `is_site_shaped` and not
    `classify_site`.

    2026-09-09 operator ruling (residential parking): an element is not
    site-shaped at all - not flagged, EXCLUDED - when `access` names
    ACCESS_EXCLUDED (`residents`) or `parking` names PARKING_EXCLUDED
    (`garage_boxes`, `carports`, `sheds`, `lane`, `street_side`). See those
    constants for why a residential value is excluded here while a private
    or customers-only one is only flagged by `site_flags` below: a locked
    commercial gate is a phone call, a resident's parking space is
    somebody's home. Because this exclusion lives in `is_site_shaped`
    itself, it reaches `classify_site`, `parking_for`, AND `routes_for`'s
    guard in one place - an excluded lot is not offered, gets no row, and
    (unlike a merely-flagged private lot) is no longer treated as
    site-shaped by `routes_for` either, so it falls through to an ordinary
    OTHER-class route instead of being dropped. That is examined and
    correct, not a leak: see `routes_for`'s own comment on this.
    """
    tags = tags or {}
    if _tag(tags, "amenity") == "parking":
        shaped = True
    else:
        shaped = any(_tag(tags, key) in values for key, values in HARDSTAND_TAGS)
    if not shaped:
        return False
    if any(part in ACCESS_EXCLUDED for part in _access_parts(tags)):
        return False
    if _tag(tags, "parking") in PARKING_EXCLUDED:
        return False
    return True


def classify_site(tags: dict) -> str:
    """`"parking"`, `"hardstand"`, or `""` for an element that is not
    site-shaped at all (`is_site_shaped(tags)` is `False`).

    Parking wins over hardstand: a lot inside a school is a lot, and the
    school is only offered when no lot was mapped inside it.

    2026-09-09 operator ruling: nothing is refused for access or for being a
    deck any more - see ACCESS_FLAGGED and PARKING_FLAGGED for why. `""`
    therefore does not mean "a search may not stage here" for one of those
    reasons - a planner reads `site_flags` for that. It DOES still mean
    "not a parking or hardstand element", and, as of the same date's
    residential-parking ruling, `""` is also what a residential lot gets:
    `is_site_shaped` excludes it outright (see ACCESS_EXCLUDED and
    PARKING_EXCLUDED), so it never reaches the parking/hardstand branches
    below at all.
    """
    tags = tags or {}
    if not is_site_shaped(tags):
        return ""
    if _tag(tags, "amenity") == "parking":
        return SITE_PARKING
    return SITE_HARDSTAND


def site_flags(tags: dict) -> list[str]:
    """What a planner must know before sending rigs there, in the order a
    row reads them: `fee`; then `permissive`, if the `access` tag carries
    it; then one `access_<value>` flag per `;`-stacked `access` part that is
    in ACCESS_FLAGGED (`access_private`, `access_no`, `access_permit`,
    `access_customers`), in the order the parts are written; then
    `access_unknown` for any part not recognised as
    `yes`/`public`/`permissive`/an ACCESS_FLAGGED value; then `covered_deck`
    if the `parking` sub-tag is in PARKING_FLAGGED.

    This function is never reached for a residential lot: `is_site_shaped`
    excludes ACCESS_EXCLUDED/PARKING_EXCLUDED tags (2026-09-09 operator
    ruling) before `parking_for` calls this, so there is no
    `access_residents` flag any more, and no flag at all for
    `garage_boxes`/`carports`/`sheds`/`lane`/`street_side` - excluded rows
    carry no flags because they carry no row.

    Structural flags (`parking_unverified`, `node_only`) are not tags and
    are added where the shape is built, not here.
    """
    tags = tags or {}
    flags: list[str] = []
    fee = _tag(tags, "fee")
    charge = _tag(tags, "charge")
    free_values = {"no", "false", "0", "none"}
    if (fee and fee not in free_values) or (charge and charge not in free_values):
        # `fee=interval`, `fee=donation`, a literal price, and a bare
        # `charge=*` with no `fee` tag at all are exactly as much "bring
        # cash" as `fee=yes` - the county-park and trailhead lots this is
        # for use all four. Only an explicit "no" / "0" / "none" (or
        # nothing tagged at all) means free.
        flags.append("fee")
    parts = _access_parts(tags)
    if "permissive" in parts:
        flags.append("permissive")
    for part in parts:
        if part in ACCESS_FLAGGED:
            # Named per value, not a shared bucket: `access_private` and
            # `access_customers` are different phone calls for the planner.
            flags.append(f"access_{part}")
    recognised = {"yes", "public", "permissive"} | ACCESS_FLAGGED
    if not parts or not all(part in recognised for part in parts):
        # A value nobody here parsed - `employees`, `destination`, the
        # `fee` in `permissive;fee` - is MORE unknown than an absent tag,
        # not less: an absent tag at least never claimed anything, where an
        # unparsed one might be a real restriction this module doesn't
        # understand. That holds even alongside a `permissive` sibling:
        # `permissive;fee` is two claims stacked, and understanding one of
        # them does not vouch for the other. A value already named by an
        # `access_*` flag above is not unknown, though - this module DID
        # understand it, which is exactly why it has its own flag.
        flags.append("access_unknown")
    if _tag(tags, "parking") in PARKING_FLAGGED:
        flags.append("covered_deck")
    return flags


# ---------------------------------------------------------------------------
# Routes: the lines a team can walk or drive, off the same OSM tiles.
#
# Classifying a way, the Overpass query and its retry machinery, the tile
# cache, and `routes_for` itself all live here - everything downstream of
# "which OSM ways are worth reading," as opposed to the site rules above,
# which answer "which OSM elements are ground to park on." `parking_for`,
# further down, reuses this section's tile machinery (`tiles_covering_osm`,
# `fetch_osm_tile`) without needing any of its route classification.
# ---------------------------------------------------------------------------

#: The classes a person can walk along - what `ingress_candidates` counts.
WALKABLE_CLASSES = frozenset({"TRAIL", "TRACK", "ROAD", "ROAD_OTHER"})


def classify_way(tags: dict) -> str:
    """Map OSM tags onto the line classes the anatomy report matched
    boundaries to: TRAIL, TRACK, ROAD, ROAD_OTHER (any other highway, treated
    as road), WATERWAY, BARRIER (power line, railway, fence, wall, pipeline),
    WATER_EDGE (lake, reservoir, coastline; a closed way contributes its
    ring), else OTHER.

    Every tag is read through `_tag` (stripped, lowercased), the way the
    stored `highway`, `access`, `tracktype` and `power` are, so a way stored
    as highway "track" is always a TRACK and a blank value is no value.
    Classification happens when a tile is read, never in the tile, so this
    needs no query-version bump."""
    highway = _tag(tags, "highway")
    if highway in _TRAIL:
        return "TRAIL"
    if highway in _TRACK:
        return "TRACK"
    if highway in _ROAD:
        return "ROAD"
    if highway:
        return "ROAD_OTHER"
    if _tag(tags, "waterway"):
        return "WATERWAY"
    if (_tag(tags, "power") in ("line", "minor_line") or _tag(tags, "railway")
            or _tag(tags, "barrier") in _BARRIER_TAGS
            or _tag(tags, "man_made") == "pipeline"):
        return "BARRIER"
    if (_tag(tags, "natural") in _WATER_EDGE_NATURAL
            or _tag(tags, "landuse") == "reservoir"):
        return "WATER_EDGE"
    return "OTHER"


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
#: HARDSTAND_TAGS entry for one of these would just repeat that clause.
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
        for key, values in HARDSTAND_TAGS
        if key not in _KEYS_ALREADY_FETCHED
    )
    return (
        f'(way["highway"]({area});way["waterway"]({area});'
        f'{_way_clause("natural", _WATER_EDGE_NATURAL, area)};'
        f'way["power"="line"]({area});way["power"="minor_line"]({area});'
        f'way["railway"]({area});'
        f'{_way_clause("barrier", _BARRIER_TAGS, area)};'
        f'way["man_made"="pipeline"]({area});'
        f'way["landuse"="reservoir"]({area});'
        f'node["amenity"="parking"]({area});way["amenity"="parking"]({area});'
        f'{_way_clause("barrier", GATE_BARRIERS, area, element="node")};'
        f"{hardstand_clauses});"
        f"out geom;"
    )


def build_overpass_query(col: int, row: int) -> str:
    """The one Overpass query a tile ever gets, on its own."""
    timeout = LINEAR_ASSUMPTIONS["overpass_timeout_s"]
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
    LINEAR_ASSUMPTIONS["overpass_endpoints"] = reported_endpoints()
    request_timeout = float(LINEAR_ASSUMPTIONS["overpass_timeout_s"]) + 30.0
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


def routes_for(bbox_3857: Bbox, cache_dir: Path,
               get_json: Callable[[str], dict] | None = None) -> list[dict]:
    """Routes clipped to the AOI, as metric shapes with stable ids."""
    from pyproj import Transformer
    from shapely.geometry import LineString, box, mapping
    from shapely.ops import transform as shapely_transform

    to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    clip = box(*bbox_3857)

    elements: list[dict] = []
    for payload in fetch_osm_tiles(tiles_covering_osm(bbox_3857), cache_dir, get_json):
        elements.extend(payload.get("elements", []))

    results: list[dict] = []
    counter = 0
    seen_ids: set[int] = set()
    for element in elements:
        if element.get("type") != "way" or element.get("id") in seen_ids:
            continue
        seen_ids.add(element.get("id"))

        tags = element.get("tags", {})
        # Query version 3 fetches parking and hardstand out of the same tile.
        # A school is not a route: it arrives classified OTHER, and it is
        # dropped here whenever it is site-shaped - whether or not a search
        # may actually stage there - so nothing the version 2 query used to
        # return has changed class.
        #
        # 2026-09-09 operator ruling: a residential lot (access=residents,
        # or a residential/on-street `parking` sub-tag) is excluded from
        # `is_site_shaped` itself, so it no longer trips this guard either.
        # That is intentional, not a gap this ruling reopened: an excluded
        # lot is not "a refused site slipping through as a route" the way
        # the version-2 gap above was - it is just an ordinary,
        # untagged-for-our-purposes closed way again, exactly as it would
        # have been read before query version 3 ever taught this module
        # about parking. It comes back as class OTHER, same as any other
        # way this module does not recognise.
        route_class = classify_way(tags)
        if route_class == "OTHER" and is_site_shaped(tags):
            continue

        coordinates = [(pt["lon"], pt["lat"]) for pt in element.get("geometry", [])]
        if len(coordinates) < 2:
            continue

        line = shapely_transform(to_3857.transform, LineString(coordinates))
        clipped = line.intersection(clip)
        if clipped.is_empty:
            continue

        results.append(
            {
                "id": f"route_{counter}",
                "kind": "route",
                "geometry": mapping(clipped),
                "properties": {
                    "route_type": route_class,
                    "name": tags.get("name"),
                    "osm_id": element.get("id"),
                    "surface": tags.get("surface"),
                    # Why a wire needs its own field: `classify_way` folds
                    # power lines, railways, fences, walls and pipelines
                    # into one BARRIER class, which is right for a boundary
                    # and useless for a helicopter. The key is ALWAYS
                    # written, null on any way that carries no `power` tag,
                    # so a record without the key is one staged before this
                    # field existed - and `planning/lz.py` reads that
                    # absence as "unknown", never as "no wires".
                    # Normalised as `classify_way` reads it, so a "Line"
                    # it classed a barrier is a "line" to `planning/lz.py`.
                    "power": _tag(tags, "power") or None,
                    # Query version 5, same rule: ALWAYS written, null where
                    # the way has no such tag, so a record without the key
                    # was staged before version 5 and its road class, access
                    # and tracktype are unknown, never "none". Normalised
                    # through `_tag` like every other tag comparison here,
                    # since egress compares them against lowercase sets.
                    "highway": _tag(tags, "highway") or None,
                    "access": _tag(tags, "access") or None,
                    "tracktype": _tag(tags, "tracktype") or None,
                    # Same rule again, for the travel-time rings (0.21.0):
                    # `waterway` tells a river (closed but at a bridge) from
                    # a creek (crossed, slowly), and `tunnel` a culvert (no
                    # crossing at all) from open water. No query change:
                    # `out geom` already returns every tag, so these come
                    # from the cached tile; a record without the keys was
                    # staged before they were kept.
                    "waterway": _tag(tags, "waterway") or None,
                    "tunnel": _tag(tags, "tunnel") or None,
                },
            }
        )
        counter += 1

    return results


def _repaired_polygon(shape):
    """`shape` made valid by geometry rather than by winding rule.

    `buffer(0)` repairs an invalid ring by winding rule, and a bowtie whose
    two lobes wind OPPOSITE ways loses one of them - a lot silently reported
    at half its real size, which is a wrong answer nobody notices rather
    than a crash somebody would. `make_valid` repairs the same ring by
    geometry and keeps both lobes. Same repair, same filtering, as
    `geo._same_ground`: a degenerate ring can make `make_valid` return a
    `GeometryCollection` mixing polygons with stray `LineString`s (a
    collinear ring, an out-and-back spike), and a line is not a lot, so only
    the polygonal parts survive. An input with no polygonal part left comes
    back empty, exactly as an unrepairable `buffer(0)` result did.
    """
    from shapely.ops import unary_union
    from shapely.validation import make_valid

    made = make_valid(shape)
    parts = [part for part in getattr(made, "geoms", [made])
             if part.geom_type in ("Polygon", "MultiPolygon")]
    return unary_union(parts)


def parking_for(bbox_3857: Bbox, cache_dir: Path,
                get_json: Callable[[str], dict] | None = None) -> list[dict]:
    """Parking lots and public hardstand from the tiles the routes come from.

    A site is kept WHOLE when it touches the box, where a route is clipped to
    it. The difference is what the two are for: a route is measured, so the
    part outside the staged ground is not ours to report; a site is
    identified, and half a lot reported as the lot would understate the room
    a planner is judging.

    An OSM area is a CLOSED way. An open one is a kerb, a barrier, or a lot
    somebody half-drew, and a polygon forced out of it would be a shape
    nobody mapped.

    A consumer of these rows has to know three things this function does not
    say by name:

    - `geometry` is metric, but not in the sense `routes_for` means it -
      it is a Web Mercator (EPSG:3857) shape, not ground metres. Area is
      `geometry.area * ground_meters_per_mercator_unit(lat) ** 2`, not
      `geometry.area` alone; anything that expects lon/lat (CalTopo, a
      GeoJSON export) needs the same reprojection to WGS84 this function's
      own inputs went through first. Guessing wrong on either gives a
      plausible-looking number off by about 1.6x at 37 N - not a crash, a
      wrong answer a planner has no way to catch.
    - `capacity` is OSM's free text, verbatim: `"40"`, `"40 (10 disabled)"`,
      `"unknown"`, `"~15"`, or absent. It is passed through unparsed - never
      call `int()` on it. It looks like a number often enough that this is
      the mistake a consumer makes once, on real data rather than on a
      fixture.
    - No row carries an area, for a Point or a Polygon alike. Computing one
      for every row, used or not, is work this function has no way to know
      is wanted; the caller derives it from `geometry` using the scale
      factor above, only when it actually needs to.
    """
    from pyproj import Transformer
    from shapely.geometry import Point, Polygon, box, mapping
    from shapely.ops import transform as shapely_transform

    to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    clip = box(*bbox_3857)

    elements: list[dict] = []
    for payload in fetch_osm_tiles(tiles_covering_osm(bbox_3857), cache_dir, get_json):
        elements.extend(payload.get("elements", []))

    results: list[dict] = []
    seen: set[tuple[str, object]] = set()
    for element in elements:
        # `node` and `way` are the only types `build_overpass_query` ever
        # asks for, so a `relation` never actually arrives here - this guard
        # is cheap insurance against that changing, not relation support
        # half-built. That distinction matters: a large park, or a lot with
        # a hole in it, is commonly mapped in OSM as a multipolygon
        # RELATION, so "no hardstand here" can quietly mean "the park is a
        # relation this module has never been asked to read."
        osm_type, osm_id = element.get("type"), element.get("id")
        if osm_type not in ("node", "way") or (osm_type, osm_id) in seen:
            continue
        tags = element.get("tags") or {}
        kind = classify_site(tags)
        if not kind:
            continue

        flags = site_flags(tags)
        if osm_type == "node":
            if element.get("lat") is None or element.get("lon") is None:
                continue
            shape = Point(element["lon"], element["lat"])
            # A dot has no area, so nothing downstream can say how many rigs
            # fit; the planner has to look at the map.
            flags.append("node_only")
        else:
            coordinates = [(pt["lon"], pt["lat"]) for pt in element.get("geometry") or []]
            if len(coordinates) < 4 or coordinates[0] != coordinates[-1]:
                continue
            shape = Polygon(coordinates)
            if not shape.is_valid:
                shape = _repaired_polygon(shape)
            if shape.is_empty:
                continue
        if kind == SITE_HARDSTAND:
            flags.append("parking_unverified")

        metric = shapely_transform(to_3857.transform, shape)
        if not metric.intersects(clip):
            continue

        # Marked seen only now, once the element has actually earned a row:
        # an element that clears `classify_site` but fails the shape checks
        # above (an open way, a node with no coordinates) must not block a
        # later tile's copy of the SAME element from being tried - which a
        # `seen.add` any earlier than this would risk, if a way's geometry
        # ever arrived incomplete from one tile and complete from another.
        seen.add((osm_type, osm_id))
        results.append({
            "id": f"{kind}-{osm_type[0]}{osm_id}",
            "kind": kind,
            "geometry": mapping(metric),
            "properties": {
                "name": tags.get("name"),
                "osm_id": osm_id,
                "osm_type": osm_type,
                "surface": tags.get("surface"),
                "capacity": tags.get("capacity"),
                "flags": flags,
            },
        })
    return results


def gates_for(bbox_3857: Bbox, cache_dir: Path,
              get_json: Callable[[str], dict] | None = None) -> list[dict]:
    """Gate nodes (`barrier` in GATE_BARRIERS) inside the box, as EPSG:3857
    points: {id: "gate-n<osm_id>", kind: "gate", geometry,
    properties: {osm_id, barrier, access}}. Read from the same cached tiles
    as routes and parking - no extra Overpass traffic.

    `access` is null where the gate carries no such tag: an unlocked-looking
    gate is not an open one, and nothing here claims it is. A gate is
    reported, never used to exclude a way."""
    from pyproj import Transformer
    from shapely.geometry import Point, box, mapping

    to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    clip = box(*bbox_3857)
    seen: set[int] = set()
    gates: list[dict] = []
    for payload in fetch_osm_tiles(tiles_covering_osm(bbox_3857), cache_dir, get_json):
        for element in payload.get("elements", []):
            if element.get("type") != "node" or element.get("id") in seen:
                continue
            tags = element.get("tags") or {}
            barrier = _tag(tags, "barrier")
            if barrier not in GATE_BARRIERS:
                continue
            if element.get("lat") is None or element.get("lon") is None:
                continue
            point = Point(*to_3857.transform(element["lon"], element["lat"]))
            if not clip.intersects(point):
                continue
            # Seen only once it has earned a row, as in `parking_for`.
            seen.add(element["id"])
            gates.append({"id": f"gate-n{element['id']}", "kind": "gate",
                          "geometry": mapping(point),
                          "properties": {"osm_id": element["id"], "barrier": barrier,
                                         "access": _tag(tags, "access") or None}})
    return gates


def ingress_candidates(routes: list[dict], assignment_3857) -> list[dict]:
    """Points where a route crosses the assignment boundary.

    These are where a team can actually walk in, which is usually a more useful
    briefing item than the assignment's centroid.

    Only walkable classes count: a creek or a fence crossing the boundary is
    not a way in, and before this filter a waterway crossing was reported as
    one.
    """
    from shapely.geometry import mapping
    from shapely.geometry import shape as shapely_shape

    boundary = assignment_3857.boundary
    results: list[dict] = []
    counter = 0

    for route in routes:
        if route["properties"].get("route_type") not in WALKABLE_CLASSES:
            continue
        line = shapely_shape(route["geometry"])
        crossing = line.intersection(boundary)
        if crossing.is_empty:
            continue

        points = [crossing] if crossing.geom_type == "Point" else list(
            getattr(crossing, "geoms", [])
        )
        for point in points:
            if point.geom_type != "Point":
                continue
            results.append(
                {
                    "id": f"ingress_{counter}",
                    "kind": "ingress",
                    "geometry": mapping(point),
                    "properties": {
                        "route_id": route["id"],
                        "route_type": route["properties"]["route_type"],
                        "route_name": route["properties"].get("name"),
                    },
                }
            )
            counter += 1

    return results
