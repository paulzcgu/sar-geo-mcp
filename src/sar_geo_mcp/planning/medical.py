"""The nearest hospital, trauma centre and paediatric trauma centre to a point,
by road, with a flight time where the hospital has a helipad.

WHAT IT ANSWERS
---------------
"Where does a patient go from here, and how long does it take?" For a
location in the field: the closest acute-care hospitals, the closest trauma
centres, and the closest paediatric trauma centres - each ranked by ROAD
time, not straight-line distance - with the trauma level, helipad, phone and
address a planner reads to a dispatcher, and a table that says it at a
glance.

FOUR PUBLIC SOURCES, NO CREDENTIAL
----------------------------------
- Hospitals: the HIFLD "Hospitals" layer (ORNL / DHS), as kept on ArcGIS
  Online by FEMA (`sources.hifld`). HIFLD Open, the original home, was
  retired in 2025; this copy carries the same schema and validation dates
  into 2025. It is the only free national list carrying trauma level,
  helipad, phone and address together. Override with `ZA_HOSPITALS_URL`.
- Road distance and time: OSRM's `table` service (`sources.osrm`), car profile,
  OpenStreetMap roads. The default is the OSRM project's public demo, which
  is fine for an occasional planning call and nothing heavier; an agency
  should point `ZA_OSRM_URL` at its own instance.
- Helipad cross-check: OpenStreetMap `aeroway=helipad|heliport` near each
  chosen hospital, through the same Overpass mirrors staging uses.
- Inside the US at all: the US Census TIGERweb States layer
  (`sources.tigerweb`), one point query. HIFLD is US-only, so a point in
  another country gets a warning that its own hospitals are missing.
  Override with `ZA_US_STATES_URL`.

WHAT IT IS NOT
--------------
Not a destination decision. Trauma designations come from state lists that
HIFLD collected at its validation date; a hospital can lose or gain a
designation, close its emergency department, or go on diversion, and none of
that is here. Road time is free-flow car time - no traffic, no lights and
sirens. Air time is the straight-line leg from the point to the hospital
alone: not dispatch, not the flight to the scene, not loading, not weather,
terrain or night. Call the receiving facility; this tells you whom to call.
"""

from __future__ import annotations

import datetime as _dt
import functools
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from ..net_errors import safe_reason
from ..sources.hifld import hospitals_url, query_hospitals
from ..sources.osm import helipads_near
from ..sources.osrm import osrm_table, osrm_url
from ..sources.tigerweb import point_state, us_states_url

CATEGORIES = ("medical_center", "trauma_center", "pediatric_trauma_center")
CATEGORY_LABELS = {
    "medical_center": "Medical center",
    "trauma_center": "Trauma center",
    "pediatric_trauma_center": "Pediatric trauma center",
}

#: HIFLD facility types that take emergency patients. Psychiatric,
#: rehabilitation, long-term and chronic-disease hospitals do not, and a
#: military hospital is counted only when it holds a trauma designation (a VA
#: medical centre generally has no civilian emergency department).
ACUTE_TYPES = frozenset({"GENERAL ACUTE CARE", "CRITICAL ACCESS", "CHILDREN"})

#: The first search; widened once to `WIDE_RADIUS_KM` for any category that
#: has nothing inside it - a rural point can be 200 km from a paediatric
#: trauma centre.
DEFAULT_RADIUS_KM = 100.0
WIDE_RADIUS_KM = 400.0
MAX_RADIUS_KM = 400.0

#: How many nearest-by-straight-line candidates per category are routed by
#: road before the closest by road are kept: road order can differ from
#: straight-line order across a bay or a ridge.
ROUTE_POOL_MIN = 6

#: Records within this distance under the same name, or at the same street
#: address, are one campus listed twice (HIFLD lists distinct parts and
#: licence variants separately - Stanford appears four times).
SAME_CAMPUS_M = 1000.0

#: An OSM helipad this close to a hospital's point is taken as its helipad.
HELIPAD_MATCH_M = 300.0

#: Air-leg defaults: about 130 knots ground speed, a typical EMS helicopter
#: cruise, and minutes for lift-off, climb, approach and landing.
DEFAULT_HELICOPTER_KMH = 240.0
DEFAULT_AIR_OVERHEAD_MIN = 5.0

#: The routed road starts this far from the point before a warning says so.
ROAD_SNAP_WARN_M = 200.0
#: A hospital this far from the road OSRM snapped it to is probably plotted
#: in the wrong place.
DEST_SNAP_WARN_M = 500.0

KM_PER_MILE = 1.609344

#: How the markers draw in CalTopo: each hospital a red first-aid cross, the
#: point a blue dot. Checked in a CalTopo import on 2026-10-21 - "firstaid"
#: and "point" are CalTopo symbols; "hospital", "medical" and "heliport" are
#: not, and draw as an empty circle.
HOSPITAL_SYMBOL, HOSPITAL_COLOR = "firstaid", "#FF0000"
ORIGIN_SYMBOL, ORIGIN_COLOR = "point", "#0000FF"

_LEVEL_NUMBER = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}
_LEVEL_ROMAN = {number: roman for roman, number in _LEVEL_NUMBER.items()}
_LEVEL_RE = re.compile(r"\bLEVEL\s+(IV|V|I{1,3})\b(.*)")
_UNKNOWN = {"", "NOT AVAILABLE", "N/A", "NA", "NONE", "UNKNOWN"}
#: Words kept in capitals when a name is title-cased.
_ACRONYMS = frozenset({
    "UC", "UCSF", "UCLA", "UCSD", "UCI", "USC", "UNC", "UAB", "UMC", "UPMC", "OHSU",
    "MUSC", "NYU", "VA", "HCA", "AHMC", "CHI", "SSM", "OSF", "LDS", "II", "III", "IV",
    "PO", "LLC", "USA", "US"})


# --- Reading HIFLD's fields ----------------------------------------------------


@dataclass(frozen=True)
class Trauma:
    """A TRAUMA field read into levels. Levels are 1 (highest) to 5."""

    raw: str | None
    adult_level: int | None = None
    pediatric_level: int | None = None
    pediatric: bool = False
    #: A state's own designation that is not a numbered level - Montana's
    #: and Wyoming's RTC/ATH/CTH/TRF, South Dakota's TRH, Maryland's PARC.
    #: Kept as written rather than guessed onto a level.
    state_designation: str | None = None

    @property
    def adult(self) -> bool:
        return self.adult_level is not None or self.state_designation is not None

    @property
    def designated(self) -> bool:
        return self.adult or self.pediatric

    def label(self) -> str:
        parts = []
        if self.adult_level is not None:
            parts.append(f"Level {_LEVEL_ROMAN[self.adult_level]}")
        elif self.state_designation:
            parts.append(f"State-designated ({self.state_designation})")
        if self.pediatric:
            level = (f"Level {_LEVEL_ROMAN[self.pediatric_level]} "
                     if self.pediatric_level is not None else "")
            parts.append(f"{level}Pediatric")
        return ", ".join(parts) if parts else "None"

    def as_dict(self) -> dict[str, Any]:
        return {"adult_level": self.adult_level,
                "pediatric_level": self.pediatric_level,
                "pediatric": self.pediatric,
                "state_designation": self.state_designation,
                "label": self.label(), "raw": self.raw}


def parse_trauma(raw: str | None) -> Trauma:
    """Read HIFLD's free-text TRAUMA field.

    Values seen in the layer (2026-10-20) include "LEVEL I", "LEVEL II
    ADULT", "LEVEL I ADULT, LEVEL II PEDIATRIC", "LEVEL II / PEDIATRIC",
    "PEDIATRIC", "Level I Trauma Center", state codes ("CTH", "TRF") and
    rehabilitation designations ("LEVEL II REHAB"), which are not trauma
    care and are ignored.
    """
    text = (raw or "").strip()
    if text.upper() in _UNKNOWN:
        return Trauma(raw=raw)
    adult: int | None = None
    pediatric_level: int | None = None
    pediatric = False
    state: list[str] = []
    for part in re.split(r"[,/;]", text.upper()):
        part = part.strip()
        if not part or "REHAB" in part:
            continue
        match = _LEVEL_RE.search(part)
        if match:
            level = _LEVEL_NUMBER[match.group(1)]
            if "PEDIATRIC" in match.group(2) or "PEDIATRIC" in part[:match.start()]:
                pediatric = True
                pediatric_level = level if pediatric_level is None else min(pediatric_level, level)
            else:
                adult = level if adult is None else min(adult, level)
        elif "PEDIATRIC" in part:
            pediatric = True
        else:
            state.append(part)
    return Trauma(raw=raw, adult_level=adult, pediatric_level=pediatric_level,
                  pediatric=pediatric,
                  state_designation=(", ".join(state) if state and adult is None else None))


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.upper() in _UNKNOWN else text


def _helipad_flag(value: Any) -> bool | None:
    text = (_clean(value) or "").upper()
    if text in {"Y", "YES"}:
        return True
    if text in {"N", "NO"}:
        return False
    return None


def _title(text: str | None) -> str | None:
    """HIFLD writes names and addresses in capitals; a table reads easier in
    title case. Short words and ordinals stay sensible ("1ST", "OF")."""
    if not text:
        return text
    small = {"of", "and", "the", "at", "for", "in", "on", "de", "la", "del"}
    words = []
    for index, word in enumerate(text.lower().split()):
        if word.upper().strip(".,()") in _ACRONYMS:
            words.append(word.upper())
        elif index and word in small:
            words.append(word)
        elif re.fullmatch(r"\d+(st|nd|rd|th)", word):
            words.append(word)
        elif re.fullmatch(r"[nsew]{1,2}", word) and index:
            words.append(word.upper())
        else:
            words.append("-".join(piece[:1].upper() + piece[1:] for piece in word.split("-")))
    return " ".join(words)


def _iso_date(epoch_ms: Any) -> str | None:
    if not isinstance(epoch_ms, (int, float)):
        return None
    return _dt.datetime.fromtimestamp(epoch_ms / 1000, _dt.timezone.utc).date().isoformat()


@dataclass
class Facility:
    """One hospital campus, merged from however many HIFLD records name it."""

    name: str
    latitude: float
    longitude: float
    facility_type: str | None
    trauma: Trauma
    hifld_helipad: bool | None
    phone: str | None
    street: str | None
    city: str | None
    state: str | None
    zip: str | None
    validated: str | None
    hifld_id: str | None
    also_listed_as: list[str] = field(default_factory=list)
    #: Corrections made to what HIFLD says, each said in a sentence.
    notes: list[str] = field(default_factory=list)
    straight_m: float = 0.0
    #: Filled by the road and helipad passes.
    road: dict[str, Any] | None = None
    osm_helipad: bool | None = None
    osm_helipad_name: str | None = None

    @property
    def address(self) -> str | None:
        line = ", ".join(part for part in (_title(self.street), _title(self.city)) if part)
        tail = " ".join(part for part in (self.state, self.zip) if part)
        return ", ".join(part for part in (line, tail) if part) or None

    @property
    def helipad(self) -> str:
        """"Y", "N" or "unknown". Either source saying yes is a yes: OSM's
        silence is not proof of absence, and HIFLD's "N" can be stale."""
        if self.hifld_helipad or self.osm_helipad:
            return "Y"
        if self.hifld_helipad is False:
            return "N"
        return "unknown"

    @property
    def helipad_source(self) -> str | None:
        sources = []
        if self.hifld_helipad is not None:
            sources.append("HIFLD")
        if self.osm_helipad:
            sources.append("OpenStreetMap")
        return " + ".join(sources) or None

    def is_acute(self) -> bool:
        return (self.facility_type or "").upper() in ACUTE_TYPES or self.trauma.designated

    def in_category(self, category: str) -> bool:
        if category == "medical_center":
            return self.is_acute()
        if category == "trauma_center":
            return self.trauma.adult
        if category == "pediatric_trauma_center":
            return self.trauma.pediatric
        raise ValueError(f"unknown category {category!r}")


def facility_from_record(record: dict) -> Facility | None:
    """One HIFLD feature (attributes plus a WGS84 point) as a Facility, or
    None for a record with no usable position."""
    attributes = record.get("attributes") or {}
    geometry = record.get("geometry") or {}
    latitude = geometry.get("y", attributes.get("LATITUDE"))
    longitude = geometry.get("x", attributes.get("LONGITUDE"))
    try:
        latitude, longitude = float(latitude), float(longitude)
    except (TypeError, ValueError):
        return None
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        return None
    zip_code = _clean(attributes.get("ZIP"))
    return Facility(
        name=_title(_clean(attributes.get("NAME"))) or "Unnamed hospital",
        latitude=latitude, longitude=longitude,
        facility_type=_clean(attributes.get("TYPE")),
        trauma=parse_trauma(attributes.get("TRAUMA")),
        hifld_helipad=_helipad_flag(attributes.get("HELIPAD")),
        phone=_clean(attributes.get("TELEPHONE")),
        street=_clean(attributes.get("ADDRESS")),
        city=_clean(attributes.get("CITY")),
        state=_clean(attributes.get("STATE")),
        zip=(zip_code.split(".")[0].zfill(5) if zip_code and zip_code.replace(".", "").isdigit()
             else zip_code),
        validated=_iso_date(attributes.get("VAL_DATE")),
        hifld_id=_clean(attributes.get("ID")),
    )


# --- Geometry ------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _geod():
    from pyproj import Geod

    return Geod(ellps="WGS84")


def ground_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Geodesic ground metres on WGS84 - never a planar projection, which
    over-reads by 1/cos(latitude)."""
    _forward, _back, metres = _geod().inv(lon1, lat1, lon2, lat2)
    return float(metres)


def _merge(primary: Facility, other: Facility) -> None:
    """Fold a second record of the same campus into the first."""
    if other.name != primary.name and other.name not in primary.also_listed_as:
        primary.also_listed_as.append(other.name)
    t, o = primary.trauma, other.trauma

    def best(a, b):
        return b if a is None else a if b is None else min(a, b)

    primary.trauma = Trauma(
        raw=" | ".join(r for r in dict.fromkeys([t.raw, o.raw]) if _clean(r)) or t.raw,
        adult_level=best(t.adult_level, o.adult_level),
        pediatric_level=best(t.pediatric_level, o.pediatric_level),
        pediatric=t.pediatric or o.pediatric,
        state_designation=(None if best(t.adult_level, o.adult_level) is not None
                           else t.state_designation or o.state_designation))
    if primary.hifld_helipad is not True and other.hifld_helipad is not None:
        primary.hifld_helipad = other.hifld_helipad or primary.hifld_helipad
    primary.phone = primary.phone or other.phone
    primary.street = primary.street or other.street
    primary.validated = max(filter(None, [primary.validated, other.validated]), default=None)


def _campus_rank(facility: Facility) -> tuple:
    """Which record names a campus: a trauma designation first, then an
    acute type, then one with a phone number."""
    return (not facility.trauma.designated,
            (facility.facility_type or "").upper() != "GENERAL ACUTE CARE",
            facility.phone is None)


def is_childrens(facility: Facility) -> bool:
    names = [facility.name, *facility.also_listed_as]
    return ((facility.facility_type or "").upper() == "CHILDREN"
            or any("CHILDREN" in name.upper() for name in names))


def correct_childrens_adult_level(facility: Facility) -> None:
    """A children's hospital is not an adult trauma centre, whatever its
    record says. HIFLD lists Lucile Packard Children's (Stanford) as "LEVEL I
    ADULT, LEVEL I PEDIATRIC" - the adult designation belongs to Stanford
    Health Care next door - and sending an adult there on the strength of it
    is the mistake this prevents. The raw value stays in `trauma.raw`."""
    trauma = facility.trauma
    if not is_childrens(facility) or not trauma.adult:
        return
    dropped = trauma.label().split(",")[0]
    facility.trauma = Trauma(raw=trauma.raw, pediatric=True,
                             pediatric_level=trauma.pediatric_level)
    facility.notes.append(
        f"HIFLD also lists an adult designation ({dropped}) for this children's "
        "hospital; it is not counted as an adult trauma center here.")


def merge_campuses(facilities: Iterable[Facility]) -> list[Facility]:
    """One entry per campus. Records share a campus when they give the same
    street address in the same city, or the same name within
    `SAME_CAMPUS_M`."""
    ordered = sorted(facilities, key=_campus_rank)
    merged: list[Facility] = []
    for facility in ordered:
        address_key = ((facility.street or "").upper(), (facility.city or "").upper())
        for kept in merged:
            same_address = (facility.street is not None and address_key ==
                            ((kept.street or "").upper(), (kept.city or "").upper()))
            same_name = (facility.name.upper() == kept.name.upper() and
                         ground_m(facility.latitude, facility.longitude,
                                  kept.latitude, kept.longitude) <= SAME_CAMPUS_M)
            if same_address or same_name:
                _merge(kept, facility)
                break
        else:
            merged.append(facility)
    return merged


# --- The four network calls -----------------------------------------------------


# --- The answer ------------------------------------------------------------------


def _minutes(seconds: float | None) -> float | None:
    return None if seconds is None else round(seconds / 60.0, 1)


def _fmt_minutes(minutes: float | None) -> str:
    if minutes is None:
        return "—"
    whole = int(round(minutes))
    if whole < 60:
        return f"{whole} min"
    return f"{whole // 60} h {whole % 60:02d} min"


def _fmt_km(metres: float | None) -> str:
    if metres is None:
        return "—"
    km = metres / 1000.0
    return f"{km:.1f} km / {km / KM_PER_MILE:.1f} mi"


def _cell(text: str | None) -> str:
    return (text or "—").replace("|", "/")


def _row(rank: int, category: str, facility: Facility, air_minutes: float | None) -> dict:
    road = facility.road or {}
    road_minutes = _minutes(road.get("duration_s"))
    faster = None
    if air_minutes is not None and road_minutes is not None:
        faster = "air" if air_minutes < road_minutes else "road"
    return {
        "category": category,
        "rank": rank,
        "name": facility.name,
        "also_listed_as": facility.also_listed_as,
        "facility_type": facility.facility_type,
        "trauma": facility.trauma.as_dict(),
        "helipad": facility.helipad,
        "helipad_source": facility.helipad_source,
        "helipad_osm_name": facility.osm_helipad_name,
        "road_distance_m": (None if road.get("distance_m") is None
                            else round(road["distance_m"])),
        "road_minutes": road_minutes,
        "air_minutes": air_minutes,
        "faster": faster,
        "straight_line_m": round(facility.straight_m),
        "phone": facility.phone,
        "address": facility.address,
        "latitude": facility.latitude,
        "longitude": facility.longitude,
        "hifld_id": facility.hifld_id,
        "validated": facility.validated,
        "notes": facility.notes,
    }


def render_table(rows: list[dict]) -> str:
    """The rows as one Markdown table, grouped by category."""
    header = ("| Category | # | Hospital | Trauma level | Helipad | Road distance | "
              "Road time | Air time | Phone | Address |")
    lines = [header, "|" + "---|" * 10]
    for row in rows:
        lines.append("| " + " | ".join([
            CATEGORY_LABELS[row["category"]], str(row["rank"]), _cell(row["name"]),
            _cell(row["trauma"]["label"]), row["helipad"],
            _fmt_km(row["road_distance_m"]), _fmt_minutes(row["road_minutes"]),
            _fmt_minutes(row["air_minutes"]), _cell(row["phone"]), _cell(row["address"]),
        ]) + " |")
    return "\n".join(lines)


def _hospital_description(row: dict, ranks: list[str]) -> str:
    """What a CalTopo marker shows when opened: the table row, one fact a line.
    CalTopo keeps only `title` and `description` of a marker, so everything a
    planner reads off the table has to be in here."""
    air = (_fmt_minutes(row["air_minutes"]) if row["air_minutes"] is not None
           else "— (no helipad known)")
    lines = [
        ", ".join(ranks),
        f"Trauma: {row['trauma']['label']}",
        f"Helipad: {row['helipad']}",
        f"Road: {_fmt_km(row['road_distance_m'])}, {_fmt_minutes(row['road_minutes'])}",
        f"Air: {air}",
        f"Phone: {row['phone'] or '—'}",
        f"Address: {row['address'] or '—'}",
    ]
    if row["validated"]:
        lines.append(f"HIFLD validated {row['validated']}. Confirm before transport.")
    return "\n".join(lines)


#: What the point's marker says of how the hospitals beside it were ranked
#: (`_ranking_line`). Each says only what is true of every hospital drawn.
RANKED_BY_ROAD = "Nearest hospitals ranked by road time from here."
RANKED_BY_ROAD_SOME_UNROUTED = (
    "Nearest hospitals ranked by road time from here; a hospital with no road route "
    "comes after every routed one in its category, by straight-line distance.")
RANKED_STRAIGHT_ROUTER_DOWN = (
    "Nearest hospitals ranked by straight-line distance from here: the road router could "
    "not be reached, so no road time is known and the nearest may not be the quickest "
    "by road.")
RANKED_STRAIGHT_NO_ROUTE = (
    "Nearest hospitals ranked by straight-line distance from here: no road route was "
    "found to any of them, so the nearest may not be the quickest by road.")


def _ranking_line(rows: list[dict], *, router_reached: bool) -> str | None:
    """How the hospitals on the map were ranked, for the point's marker -
    which a team in CalTopo reads beside "#1", "#2" on each hospital - or
    None when no hospital is drawn, since a ranking line would describe
    nothing on the map (outside the US, or nothing found). By road time only
    when every hospital drawn has one: with the router down the categories
    are in straight-line order, and saying "road time" there is the claim
    the medical module was written to prevent - across a bay or a ridge the
    nearest as the crow flies can be the slowest by road."""
    if not rows:
        return None
    if not router_reached:
        return RANKED_STRAIGHT_ROUTER_DOWN
    routed = sum(row["road_minutes"] is not None for row in rows)
    if routed == 0:
        return RANKED_STRAIGHT_NO_ROUTE
    return RANKED_BY_ROAD if routed == len(rows) else RANKED_BY_ROAD_SOME_UNROUTED


def _markers(latitude: float, longitude: float, rows: list[dict],
             road_start: dict | None = None, *, router_reached: bool = True) -> dict:
    """A FeatureCollection CalTopo imports as Markers: the point (a blue dot)
    and one red first-aid cross per hospital, carrying every category it was
    chosen for. `role`, `categories` and the numbers stay as properties for a
    caller drawing them itself; CalTopo reads the title, description, symbol
    and colour. The point's description says how the hospitals were ranked
    (`_ranking_line`; `router_reached` false when the road router raised)
    and where the road route starts when that is far from the point - and
    is empty when neither applies."""
    by_place: dict[tuple, dict] = {}
    ranks: dict[tuple, list[str]] = {}
    for row in rows:
        key = (row["latitude"], row["longitude"], row["name"])
        if key not in by_place:
            by_place[key] = {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [row["longitude"], row["latitude"]]},
                "properties": {"class": "Marker", "title": row["name"], "description": "",
                               "marker-symbol": HOSPITAL_SYMBOL,
                               "marker-color": HOSPITAL_COLOR, "marker-size": 1,
                               "role": "hospital", "categories": [],
                               "trauma": row["trauma"]["label"], "helipad": row["helipad"],
                               "phone": row["phone"], "address": row["address"],
                               "road_minutes": row["road_minutes"],
                               "air_minutes": row["air_minutes"]},
            }
            ranks[key] = []
        by_place[key]["properties"]["categories"].append(row["category"])
        ranks[key].append(f"{CATEGORY_LABELS[row['category']]} #{row['rank']}")
    for key, feature in by_place.items():
        row = next(r for r in rows if (r["latitude"], r["longitude"], r["name"]) == key)
        feature["properties"]["description"] = _hospital_description(row, ranks[key])

    ranking = _ranking_line(rows, router_reached=router_reached)
    origin_lines = [ranking] if ranking else []
    if road_start and (road_start.get("distance_from_point_m") or 0) > ROAD_SNAP_WARN_M:
        origin_lines.append(f"Road route starts {road_start['distance_from_point_m']:,} m "
                            "away; the walk or carry to it is not counted.")
    origin = {"type": "Feature",
              "geometry": {"type": "Point", "coordinates": [longitude, latitude]},
              "properties": {"class": "Marker", "title": "Point",
                             "description": "\n".join(origin_lines),
                             "marker-symbol": ORIGIN_SYMBOL, "marker-color": ORIGIN_COLOR,
                             "marker-size": 1, "role": "origin"}}
    return {"type": "FeatureCollection", "features": [origin, *by_place.values()]}


def _assumptions(helicopter_speed_kmh: float, air_overhead_minutes: float) -> dict[str, Any]:
    # Hosts only, never a full URL: an override may carry a key or userinfo.
    from ..sources.osm import _host as host
    from ..sources.osm import reported_endpoints

    return {
        "hospitals": {
            "source": "HIFLD Hospitals (ORNL / DHS), FEMA-maintained copy on ArcGIS Online",
            "host": host(hospitals_url()),
            "open_only": True,
            "medical_center_types": sorted(ACUTE_TYPES),
            "medical_center_rule": ("an open hospital of an acute type, or any open hospital "
                                    "holding a trauma designation"),
            "trauma_center_rule": "an adult trauma level (I-V) or a state trauma designation",
            "pediatric_trauma_center_rule": "a pediatric trauma designation",
            "childrens_hospital_rule": ("a children's hospital is never counted as an adult "
                                        "trauma center, whatever its record says"),
            "same_campus_m": SAME_CAMPUS_M,
        },
        "coverage": {
            "source": "US Census TIGERweb States layer (states, DC, Puerto Rico, island areas)",
            "host": host(us_states_url()),
            "meaning": ("`point.in_us` is false when the point is in no US state, DC or "
                        "territory - another country, or offshore - and null when that "
                        "could not be checked; the hospital list is US-only"),
        },
        "road": {
            "source": "OSRM table service, car profile, OpenStreetMap roads",
            "host": host(osrm_url()),
            "meaning": ("free-flow car time over mapped roads from where the point joins "
                        "the road network; no traffic, no lights and sirens"),
        },
        "air": {
            "helicopter_speed_kmh": helicopter_speed_kmh,
            "overhead_minutes": air_overhead_minutes,
            "meaning": ("straight-line distance from the point to the hospital at the "
                        "helicopter speed, plus the overhead for lift-off, climb, approach "
                        "and landing; only where the hospital has a helipad. Dispatch, the "
                        "flight to the scene, loading, weather, terrain and night are not "
                        "counted"),
        },
        "helipad": {
            "rule": ("Y when HIFLD says Y or OpenStreetMap maps a helipad or heliport within "
                     f"{HELIPAD_MATCH_M:.0f} m; N when HIFLD says N and OSM shows none; "
                     "otherwise unknown"),
            "overpass_endpoints": reported_endpoints(),
        },
        "ranking": ("within each category, by road time; a hospital the router could not "
                    "reach comes after every routed one, by straight-line distance"),
    }


NOT_A_DECISION = (
    "Not a destination decision: trauma designations, helipads and phone numbers are "
    "as HIFLD last validated them (see each row's `validated`), and emergency department "
    "status and diversion are not known here. Confirm with the receiving facility and "
    "follow the local EMS destination policy.")


def check_params(*, per_category: int = 3, search_radius_km: float = DEFAULT_RADIUS_KM,
                 helicopter_speed_kmh: float = DEFAULT_HELICOPTER_KMH,
                 air_overhead_minutes: float = DEFAULT_AIR_OVERHEAD_MIN) -> None:
    """The range checks `nearest_medical_facilities` makes before it asks
    anything over the network: a ValueError naming the first one out of
    range. Public so a scenario refuses a bad override in these words
    before it stages any ground, rather than as a failed section after."""
    if not 1 <= per_category <= 10:
        raise ValueError("per_category must be between 1 and 10")
    if not 1 <= search_radius_km <= MAX_RADIUS_KM:
        raise ValueError(f"search_radius_km must be between 1 and {MAX_RADIUS_KM:.0f}")
    if not 50 <= helicopter_speed_kmh <= 400:
        raise ValueError("helicopter_speed_kmh must be between 50 and 400")
    if not 0 <= air_overhead_minutes <= 60:
        raise ValueError("air_overhead_minutes must be between 0 and 60")


def nearest_medical_facilities(
    latitude: float, longitude: float, *,
    per_category: int = 3,
    search_radius_km: float = DEFAULT_RADIUS_KM,
    helicopter_speed_kmh: float = DEFAULT_HELICOPTER_KMH,
    air_overhead_minutes: float = DEFAULT_AIR_OVERHEAD_MIN,
    check_osm_helipads: bool = True,
    query: Callable[..., list[dict]] = query_hospitals,
    route: Callable[..., dict] = osrm_table,
    helipads: Callable[..., list[dict]] = helipads_near,
    locate: Callable[..., str | None] = point_state,
) -> dict[str, Any]:
    """The closest medical centres, trauma centres and paediatric trauma
    centres to a point, by road. See the module docstring; `query`, `route`,
    `helipads` and `locate` are injected for tests."""
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError("latitude must be within -90..90 and longitude within -180..180")
    check_params(per_category=per_category, search_radius_km=search_radius_km,
                 helicopter_speed_kmh=helicopter_speed_kmh,
                 air_overhead_minutes=air_overhead_minutes)

    warnings: list[str] = []

    # HIFLD is US-only. A point in Canada or Mexico would otherwise be told
    # its nearest trauma center is a US one, with no word that a closer one
    # may be across the border - so this warning comes first.
    try:
        us_state = locate(latitude, longitude)
    except Exception as error:   # noqa: BLE001 - degrade, never fail the answer
        us_state, in_us = None, None
        warnings.append(
            f"Whether the point is inside the US could not be checked ({safe_reason(error)}). "
            "The hospital list covers the 50 states, DC and US territories only: near a "
            "border, a closer hospital in another country would be missing.")
    else:
        in_us = us_state is not None
        if not in_us:
            warnings.append(
                "The point is not inside any US state, DC or territory (US Census "
                "boundaries): it is in another country, or offshore. The hospital list "
                "covers the US only, so if it is in another country, that country's "
                "hospitals are missing from every category and a closer one may exist. "
                "Use local sources for where to take a patient.")

    def collect(radius_km: float, where: str) -> list[Facility]:
        found = []
        for record in query(latitude, longitude, radius_km, where):
            facility = facility_from_record(record)
            if facility is None:
                continue
            facility.straight_m = ground_m(latitude, longitude,
                                           facility.latitude, facility.longitude)
            correct_childrens_adult_level(facility)
            if facility.straight_m <= radius_km * 1000.0 and facility.is_acute():
                found.append(facility)
        return found

    facilities = collect(search_radius_km, "STATUS = 'OPEN'")
    searched_km = {category: search_radius_km for category in CATEGORIES}
    missing = [c for c in CATEGORIES
               if not any(f.in_category(c) for f in facilities)]
    if missing and search_radius_km < WIDE_RADIUS_KM:
        # Only trauma hospitals are asked for at the wider radius unless the
        # point has no hospital at all nearby - keeping the answer small.
        where = ("STATUS = 'OPEN'" if "medical_center" in missing
                 else "STATUS = 'OPEN' AND TRAUMA <> 'NOT AVAILABLE'")
        known = {(f.hifld_id, f.name) for f in facilities}
        for facility in collect(WIDE_RADIUS_KM, where):
            if (facility.hifld_id, facility.name) not in known:
                facilities.append(facility)
        for category in missing:
            searched_km[category] = WIDE_RADIUS_KM
    facilities = merge_campuses(facilities)
    for facility in facilities:
        correct_childrens_adult_level(facility)

    # Route the nearest few of each category by road, in one request.
    pool_size = max(ROUTE_POOL_MIN, per_category * 2)
    pools = {category: sorted((f for f in facilities if f.in_category(category)),
                              key=lambda f: f.straight_m)[:pool_size]
             for category in CATEGORIES}
    to_route = list({id(f): f for pool in pools.values() for f in pool}.values())
    road_start = None
    # Whether the road router answered, for the point's marker: with it down
    # the ranking is straight-line, and the marker must not say road time.
    router_reached = True
    if to_route:
        try:
            table = route((latitude, longitude),
                          [(f.latitude, f.longitude) for f in to_route])
        except Exception as error:   # noqa: BLE001 - degrade, never fail the answer
            router_reached = False
            warnings.append(
                f"The road router could not be reached ({safe_reason(error)}): road "
                "distance and time are missing, and each category is ranked by "
                "straight-line distance instead.")
        else:
            for facility, leg in zip(to_route, table["legs"]):
                facility.road = leg
                if (leg.get("snap_m") or 0) > DEST_SNAP_WARN_M:
                    warnings.append(
                        f"{facility.name} is plotted {leg['snap_m']:.0f} m from the "
                        "nearest road the router knows, so its position may be wrong "
                        "and its road time is uncertain.")
            snap_m = table.get("origin_snap_m")
            road_start = {**(table.get("origin_snap") or {}),
                          "distance_from_point_m": None if snap_m is None else round(snap_m)}
            if snap_m is not None and snap_m > ROAD_SNAP_WARN_M:
                warnings.append(
                    f"The point is {snap_m:,.0f} m from the nearest road the router "
                    "knows. Road distance and time start at that road (`road_start`), "
                    "and the walk, carry or drive to it is not counted - nor is "
                    "whether it can be reached from the point at all.")
            unrouted = [f.name for f in to_route if (f.road or {}).get("duration_s") is None]
            if unrouted:
                warnings.append("No road route was found to " + ", ".join(unrouted) +
                                "; they are ranked after the routed hospitals.")

    def road_key(facility: Facility) -> tuple:
        duration = (facility.road or {}).get("duration_s")
        return (duration is None, duration or 0.0, facility.straight_m)

    chosen = {category: sorted(pool, key=road_key)[:per_category]
              for category, pool in pools.items()}

    # Cross-check helipads on the hospitals actually returned.
    finalists = list({id(f): f for pool in chosen.values() for f in pool}.values())
    if check_osm_helipads and finalists:
        try:
            pads = helipads([(f.latitude, f.longitude) for f in finalists], HELIPAD_MATCH_M)
        except Exception as error:   # noqa: BLE001
            warnings.append(
                f"OpenStreetMap could not be asked about helipads ({safe_reason(error)}); "
                "helipad is HIFLD's answer alone.")
        else:
            for facility in finalists:
                near = [(ground_m(facility.latitude, facility.longitude,
                                  pad["latitude"], pad["longitude"]), pad) for pad in pads]
                near = [item for item in near if item[0] <= HELIPAD_MATCH_M]
                facility.osm_helipad = bool(near)
                if near:
                    facility.osm_helipad_name = min(near, key=lambda item: item[0])[1].get("name")
                if facility.hifld_helipad is False and facility.osm_helipad:
                    warnings.append(
                        f"{facility.name}: HIFLD lists no helipad but OpenStreetMap maps "
                        "one beside it; shown as Y - confirm before flying there.")

    rows: list[dict] = []
    for category in CATEGORIES:
        for rank, facility in enumerate(chosen[category], start=1):
            air = None
            if facility.helipad == "Y":
                air = round(facility.straight_m / 1000.0 / helicopter_speed_kmh * 60.0
                            + air_overhead_minutes, 1)
            rows.append(_row(rank, category, facility, air))
        if not chosen[category]:
            warnings.append(
                f"No {CATEGORY_LABELS[category].lower()} was found within "
                f"{searched_km[category]:.0f} km. The hospital list covers the 50 states, "
                "DC and US territories only.")

    if any(row["helipad"] == "unknown" for row in rows):
        warnings.append("Helipad 'unknown' means HIFLD does not say and OpenStreetMap "
                        "maps none nearby; no air time is given for those.")
    validated = sorted({row["validated"] for row in rows if row["validated"]})

    return {
        "point": {"latitude": latitude, "longitude": longitude,
                  "in_us": in_us, "us_state": us_state},
        "road_start": road_start,
        "table": render_table(rows),
        "rows": rows,
        "markers": _markers(latitude, longitude, rows, road_start,
                            router_reached=router_reached),
        "searched_radius_km": searched_km,
        "data_validated": ({"earliest": validated[0], "latest": validated[-1]}
                           if validated else None),
        "note": NOT_A_DECISION,
        "warnings": warnings,
        "assumptions": _assumptions(helicopter_speed_kmh, air_overhead_minutes),
    }
