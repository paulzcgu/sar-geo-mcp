"""A saved candidate set as a GeoJSON file CalTopo imports as draft
assignments.

NOT A WRITE TO CALTOPO
----------------------
The hub stays read-only (README, *What is not here*): this writes a file
beside the staged ground and nothing else. A planner imports it in CalTopo
(Import, choose the file) and every assignment arrives as a DRAFT, to be
checked and issued by a person. That keeps the governance the write tier
waits for - a human between the agent and a map teams navigate by.

WHAT CALTOPO IS GIVEN
---------------------
Each candidate becomes a Feature with CalTopo's own Assignment properties,
as CalTopo writes them in a map's JSON: `class` "Assignment", `letter` (the
label on the map), `number`, `title`, `description`, `resourceType`,
`status` "DRAFT", and stroke and fill. Labels are a prefix and a number in
set order - `U01 Main Street +2` - so the label a planner reads aloud is
the one on the map.

CalTopo carries one ring per polygon and one polygon per assignment, so:
  - a MultiPolygon becomes one assignment per part, `U05a`, `U05b`;
  - a hole is cut open along a zero-width bridge (`geo._cut_holes_open`, as
    `buffer_geometry(single_ring=True)` does); a hole that cannot be cut
    without changing the ground is left, with a warning;
  - a line candidate becomes a line assignment, a MultiLineString one per
    part.
A candidate's `markers` (house markers from `populated_areas`) go in as
CalTopo Markers. On an area handle the area itself goes in as a Shape - the
range circle - with a Marker on its centre.

ONE FOLDER PER FILE
-------------------
Every file this package writes for CalTopo puts all its features in one
CalTopo Folder (`in_one_folder`), titled for what the file is and when it
was written, so an import lands as one group a planner can hide, move or
delete together. A Folder is a Feature of `class` "Folder" with no
geometry; each member carries `folderId`, the folder's id, as CalTopo
writes its own GeoJSON. Checked by a real CalTopo import (2026-10-26):
the folder arrives holding a draft Assignment, a Marker and a line Shape
alike. CalTopo has no folder inside a folder, so a feature that already
carries a folder is refused rather than nested.
"""

from __future__ import annotations

import datetime as dt
import json
import string
import uuid
from pathlib import Path
from typing import Any

EXPORTS_DIRNAME = "exports"
#: The start of every folder title, so a planner can tell an import apart
#: from what people drew.
FOLDER_PREFIX = "SAR Geo"


def folder_title(what: str, now: dt.datetime | None = None) -> str:
    """`SAR Geo <what> <YYYY-MM-DD HH:MM>`: what the file holds, and when it
    was written, so two imports of the same plan stay apart on the map."""
    return f"{FOLDER_PREFIX} {what} {(now or dt.datetime.now()):%Y-%m-%d %H:%M}"


def in_one_folder(features: list[dict], title: str) -> list[dict]:
    """`features` with a CalTopo Folder first and every feature in it.

    Raises ValueError for a feature that is already a Folder or already in
    one: CalTopo cannot hold a folder inside a folder, and moving a feature
    out of a folder it was given on purpose is not this function's call."""
    for feature in features:
        properties = feature.get("properties") or {}
        if properties.get("class") == "Folder" or properties.get("folderId"):
            raise ValueError(
                f"{properties.get('title') or 'a feature'} is already a folder or in one; "
                "CalTopo has no folder inside a folder, so this file cannot be grouped")
    folder_id = str(uuid.uuid4())
    folder = {"type": "Feature", "id": folder_id, "geometry": None,
              "properties": {"class": "Folder", "title": title,
                             "visible": True, "labelVisible": True}}
    return [folder, *({**feature, "properties": {**(feature.get("properties") or {}),
                                                 "folderId": folder_id}}
                      for feature in features)]
ASSIGNMENT_STROKE = "#FF0000"
AREA_STROKE = "#000000"

__all__ = ["caltopo_collection", "export_set"]


def _describe(candidate: dict, set_id: str | None) -> str:
    """One paragraph a planner reads in CalTopo's assignment dialog: what
    the ground is, its numbers, and where it came from."""
    facts = candidate.get("facts") or {}
    parts = []
    streets = facts.get("streets")
    if streets:
        parts.append("Streets: " + ", ".join(streets) + ".")
    elif candidate.get("name"):
        parts.append(f"{candidate['name']}.")
    numbers = []
    for key, word in (("parcels", "lot(s)"), ("buildings", "building(s)"),
                      ("markers", "house marker(s)")):
        if facts.get(key) is not None:
            numbers.append(f"{facts[key]} {word}")
    if facts.get("area_acres") is not None:
        numbers.append(f"{facts['area_acres']:g} acres")
    if facts.get("length_m") is not None:
        numbers.append(f"{facts['length_m']:.0f} m long")
    if numbers:
        parts.append(", ".join(numbers) + ".")
    if facts.get("distance_from_center_m") is not None:
        # A hasty line through the centre - a place last seen at a trailhead
        # - has no direction from it, and its bearing is null: it is said
        # without one, where formatting the null raised and lost the file.
        bearing = facts.get("bearing_from_center_deg", 0)
        parts.append(f"{facts['distance_from_center_m']:.0f} m"
                     + (f" at {bearing:.0f} deg" if bearing is not None else "")
                     + " from the centre.")
    parts.extend(candidate.get("warnings") or [])
    parts.append(f"Draft from sar-geo-mcp, candidate {candidate['id']}"
                 + (f" of set {set_id}" if set_id else "")
                 + "; check it on the ground before issuing.")
    return " ".join(parts)


def _single_ring(polygon, warnings: list[str], label: str):
    """`polygon` with its holes cut open, or as it is with a warning when
    the cut would change the ground."""
    from ..geo import _cut_holes_open

    if not polygon.interiors:
        return polygon
    try:
        return _cut_holes_open(polygon)
    except ValueError:
        warnings.append(f"{label} has a hole that could not be cut open; CalTopo will "
                        "drop it on import - draw that assignment by hand")
        return polygon


def caltopo_collection(candidates: list[dict], *, label_prefix: str = "U",
                       resource_type: str = "GROUND", area: dict | None = None,
                       include_markers: bool = True, set_id: str | None = None
                       ) -> tuple[dict, list[dict], list[str]]:
    """(the FeatureCollection, `[{label, candidate_id}]` one per assignment
    drawn, warnings) for `candidates` (candidate dicts with WGS84 geometry).

    `area` is the staged area (`kind`, `center` as {latitude, longitude} or
    a (lat, lon) pair, `radius_m`, `geometry`) or None."""
    from shapely.geometry import LineString, mapping
    from shapely.geometry import shape as shapely_shape

    width = max(2, len(str(len(candidates))))
    features: list[dict] = []
    labels: list[dict] = []
    warnings: list[str] = []
    style = {"stroke": ASSIGNMENT_STROKE, "stroke-width": 2, "stroke-opacity": 1,
             "fill": ASSIGNMENT_STROKE, "fill-opacity": 0.1}
    for index, candidate in enumerate(candidates, start=1):
        code = f"{label_prefix}{index:0{width}d}"
        geometry = shapely_shape(candidate["geometry"])
        if geometry.geom_type in ("Polygon", "LineString"):
            parts = [geometry]
        elif geometry.geom_type in ("MultiPolygon", "MultiLineString"):
            parts = sorted(geometry.geoms, key=lambda part: -(part.area or part.length))
        else:
            warnings.append(f"{candidate['id']} is a {geometry.geom_type}, not an area or a "
                            "line, and was left out")
            continue
        description = _describe(candidate, set_id)
        for number, part in enumerate(parts):
            suffix = "" if len(parts) == 1 else (string.ascii_lowercase[number]
                                                if number < 26 else str(number + 1))
            label = f"{code}{suffix}" + (f" {candidate['name']}" if candidate.get("name") else "")
            if part.geom_type == "Polygon":
                part = _single_ring(part, warnings, label)
            piece = ("" if len(parts) == 1 else
                     f" Piece {number + 1} of {len(parts)} of one candidate.")
            features.append({
                "type": "Feature", "geometry": mapping(part),
                "properties": {"class": "Assignment", "letter": label, "number": "",
                               "title": label, "description": description + piece,
                               "resourceType": resource_type, "status": "DRAFT", **style}})
            labels.append({"label": label, "candidate_id": candidate["id"]})
        if include_markers:
            features.extend(candidate.get("markers") or [])

    if area is not None and area.get("geometry"):
        outline = shapely_shape(area["geometry"])
        ring = outline.exterior if outline.geom_type == "Polygon" else outline.boundary
        # A shapely ring maps to "LinearRing", which is not a GeoJSON type:
        # CalTopo, like any GeoJSON reader, wants it as a LineString.
        ring = LineString(ring.coords) if ring.geom_type == "LinearRing" else ring
        radius = area.get("radius_m")
        title = ("Search area" if radius is None
                 else f"Search area, {radius:,.0f} m ({radius / 1609.344:.2f} mi) radius")
        features.append({"type": "Feature", "geometry": json.loads(json.dumps(mapping(ring))),
                         "properties": {"class": "Shape", "title": title,
                                        "description": "The staged area the assignments were "
                                                       "drawn in",
                                        "stroke": AREA_STROKE, "stroke-width": 2,
                                        "stroke-opacity": 1}})
        centre = area.get("center")
        if centre is not None:
            latitude, longitude = ((centre["latitude"], centre["longitude"])
                                   if isinstance(centre, dict) else centre)
            features.append({"type": "Feature",
                             "geometry": {"type": "Point", "coordinates": [longitude, latitude]},
                             "properties": {"class": "Marker", "title": "Search centre",
                                            "description": "", "marker-symbol": "point",
                                            "marker-color": AREA_STROKE}})
    return {"type": "FeatureCollection", "features": features}, labels, warnings


def check_label_prefix(label_prefix) -> str:
    """The label prefix as `export_set` uses it, stripped, or ValueError.
    Public so a scenario that exports a set refuses a bad prefix before it
    stages anything, in the same words."""
    prefix = label_prefix.strip() if isinstance(label_prefix, str) else ""
    if not prefix or len(prefix) > 6 or not prefix.replace("-", "").isalnum():
        raise ValueError(f"label_prefix ({label_prefix!r}) must be 1 to 6 letters or digits, "
                         "such as U or DIV-A")
    return prefix


def export_set(directory, set_id: str, *, candidate_ids: list[str] | None = None,
               label_prefix: str = "U", resource_type: str = "GROUND",
               include_area: bool = True, include_markers: bool = True) -> dict[str, Any]:
    """Write the candidates of `set_id` (all of them, or those named in
    `candidate_ids`, in that order) to `<directory>/exports/<set_id>.geojson`
    and say what was written. Raises KeyError for a set or candidate that
    is not in this staged directory, and ValueError for a bad label prefix."""
    from ..cache import atomic_write_text
    from ..terrain.grid import read_subject
    from .candidates import find_candidate_in, load_set, set_staleness

    prefix = check_label_prefix(label_prefix)
    directory = Path(directory)
    cset = load_set(directory, set_id)
    chosen = ([find_candidate_in(cset, cid) for cid in candidate_ids] if candidate_ids
              else list(cset.candidates))
    area = None
    if include_area:
        subject = read_subject(directory)
        if subject.get("kind") in ("circle", "polygon"):          # an area handle
            area = {"kind": subject["kind"], "center": subject.get("center"),
                    "radius_m": subject.get("radius_m"), "geometry": subject["geometry"]}
    collection, labels, warnings = caltopo_collection(
        [candidate.model_dump() for candidate in chosen], label_prefix=prefix,
        resource_type=(resource_type or "GROUND").strip().upper(), area=area,
        include_markers=include_markers, set_id=cset.set_id)
    title = folder_title(cset.set_id)
    members = collection["features"]
    collection = {**collection, "features": in_one_folder(members, title)}
    path = directory / EXPORTS_DIRNAME / f"{cset.set_id}.geojson"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(collection, indent=1))
    staleness = set_staleness(directory, cset)
    if staleness:
        warnings.append(staleness)
    counts: dict[str, int] = {}
    for feature in members:
        kind = feature["properties"]["class"]
        counts[kind] = counts.get(kind, 0) + 1
    return {"path": str(path), "folder": title, "set_id": cset.set_id,
            "assignments": len(labels),
            "features": counts, "labels": labels, "area_included": area is not None,
            "warnings": warnings}
