"""What real assignments look like, as data.

The numbers come from the anatomy report over 65 real searches and ship
with provenance so an agent can tell a human where a default came from.
Flags are raised only where the corpus has a lower bound to compare
against; a table with no p90 (the boundary on_feature_pct figures top out
at p75) only ever flags "below", since there is no honest upper band to
flag against. Where a resource type is named and the corpus has a median
for it, area is banded against that resource rather than against a corpus
whose middle is a ground team's 25 acres.

The file keeps its areas in hectares (`polygon_area_ha`,
`area_ha_by_resource`), as the corpus was measured and as another agency's
file is written; every answer gives them in acres (`target_area_range_acres`,
`norms_in_acres`, an `area_acres` flag). A server for another agency replaces
the file (`--norms`, `ZA_ASSIGNMENT_NORMS`); a caller overrides a target per
call and the override wins. A candidate set the norms decide is made afresh
under another file rather than read back from the cache (`norms_digest`).
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from ..geo import HA_PER_ACRE

DEFAULT_NORMS_PATH = Path(__file__).parent / "data" / "assignment_norms.json"

#: What `band` says a flag was read against.
CORPUS_BAND = "corpus p10-p90"
CORPUS_LOW_ONLY_BAND = "corpus p10 (the corpus has no p90 for this metric)"
RESOURCE_BAND = "resource p25-p75 (ratio-derived from the resource median)"

#: metric name in `facts` -> (dotted path to its percentile table, a label,
#: what one unit of the table is in the metric's unit).
_BANDED = {
    "area_acres": ("polygon_area_ha", "polygon area (acres)", 1.0 / HA_PER_ACRE),
    "on_feature_pct": ("boundary.on_feature_pct", "share of boundary on a mapped feature",
                       1.0),
    "width_m": ("corridor_width_m", "corridor width", 1.0),
    "length_m": ("line_length_m", "line length", 1.0),
}


def load_norms(path: Path | str | None = None) -> dict:
    """The shipped norms, or the file at `path`."""
    target = Path(path) if path is not None else DEFAULT_NORMS_PATH
    return json.loads(target.read_text(encoding="utf-8"))


def norms_digest(norms: dict) -> str:
    """Twelve hex characters of a SHA-256 over the norms' content, for the
    parameters of a candidate set the norms decide.

    A set is read back by its id, and the id is a hash of its parameters
    (`candidates.make_set_id`), so whatever decides a set's content has to
    be among them. The norms do - a hasty line's flags are read against
    their bands, an urban assignment is grouped toward their median - yet
    they come from the server, not the call: a server restarted with
    another agency's `--norms` over the same cache would otherwise read
    back the old corpus's flags and grouping as current (PR #4 review B23).

    Over the content, canonically, not the file or its path: the same
    figures spaced or ordered differently are the same norms, while a
    shipped file a later release changes is other norms, its sets made
    afresh like another agency's. All of it rather than the tables one
    generator reads today, so a generator that comes to read another table
    cannot leave its sets reused across a change to that table.
    """
    canonical = json.dumps(norms, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _table(norms: dict, dotted: str) -> dict | None:
    node = norms
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node if isinstance(node, dict) else None


def resource_median_ha(resource_type: str, norms: dict) -> float | None:
    """The corpus median for a resource type, or None when it has none.

    The question "does this corpus know this resource" is asked in three
    places - the target range, the area band, and the assumption that says
    which band was used - and asking it here keeps the answer one fact.
    """
    medians = (norms.get("area_ha_by_resource") or {}).get("medians", {})
    value = medians.get((resource_type or "").strip().upper())
    return float(value) if value is not None else None


def target_area_range_ha(resource_type: str, norms: dict) -> dict:
    """p25/p50/p75 hectares for a resource type, by the stated ratio rule,
    or the overall quartiles when the corpus has no median for it."""
    by_resource = norms.get("area_ha_by_resource", {})
    medians = by_resource.get("medians", {})
    key = (resource_type or "").strip().upper()
    if key in medians:
        median = float(medians[key])
        return {"p25": median * float(by_resource["p25_ratio"]), "p50": median,
                "p75": median * float(by_resource["p75_ratio"]),
                "source": "area_ha_by_resource"}
    overall = norms["polygon_area_ha"]
    return {"p25": overall["p25"], "p50": overall["p50"], "p75": overall["p75"],
            "source": "polygon_area_ha"}


def target_area_range_acres(resource_type: str, norms: dict) -> dict:
    """`target_area_range_ha` in acres, to one decimal place."""
    spread = target_area_range_ha(resource_type, norms)
    return {**{k: round(spread[k] / HA_PER_ACRE, 1) for k in ("p25", "p50", "p75")},
            "source": spread["source"]}


def norms_in_acres(norms: dict) -> dict:
    """A copy of `norms` with its hectare tables given in acres, to one
    decimal place: `polygon_area_ha` as `polygon_area_acres`, and
    `area_ha_by_resource` as `area_acres_by_resource` (its ratios, which have
    no unit, unchanged). Every other table is as the file has it."""
    shown = copy.deepcopy(norms)
    table = shown.pop("polygon_area_ha", None)
    if isinstance(table, dict):
        shown["polygon_area_acres"] = {k: round(float(v) / HA_PER_ACRE, 1)
                                       for k, v in table.items()}
    by_resource = shown.pop("area_ha_by_resource", None)
    if isinstance(by_resource, dict):
        by_resource = dict(by_resource)
        by_resource["medians"] = {k: round(float(v) / HA_PER_ACRE, 1)
                                  for k, v in (by_resource.get("medians") or {}).items()}
        shown["area_acres_by_resource"] = by_resource
    return shown


def norm_flags(facts: dict, norms: dict, resource_type: str = "") -> list[dict]:
    """Departures from the band that applies, one entry per metric.

    `facts` holds any of `area_acres`, `on_feature_pct`, `width_m`, `length_m`;
    absent or null values are skipped. The caller decides which apply (a
    line has a length, a corridor a width) by what it puts in `facts`.

    A value below the table's p10 is always flagged. A value above the
    table's p90 is flagged only when the table has a p90 at all - the
    boundary on_feature_pct figures stop at p75, and there is no p90 to
    honestly compare "too high" against.

    `resource_type` names the resource the target would be assigned to. When
    the corpus has a median for it, `area_acres` is banded against THAT
    resource's p25-p75 range rather than the whole corpus: a 2-acre dog
    trail assignment is ordinary for a resource whose median is 1.7 acres,
    and reading it against a corpus dominated by 25-acre ground teams
    reported it as too small. The bands are the corpus's hectares in acres.
    Only area is banded this way; the corpus has no per-resource width,
    length or on-feature figures, so those stay against the whole corpus. Every flag says which band it used, in `band` and in the note.
    """
    flags: list[dict] = []
    resource_key = (resource_type or "").strip().upper()
    resource_median = resource_median_ha(resource_key, norms)
    for metric, (table_name, label, unit) in _BANDED.items():
        value = facts.get(metric)
        if value is None:
            continue
        value = float(value)
        if metric == "area_acres" and resource_median is not None:
            spread = target_area_range_ha(resource_key, norms)
            low, mid, high = (float(spread["p25"]) * unit, float(spread["p50"]) * unit,
                              float(spread["p75"]) * unit)
            band, band_label = RESOURCE_BAND, f"{resource_key} p25-p75"
        else:
            table = _table(norms, table_name)
            if not table or "p10" not in table:
                continue
            low = float(table["p10"]) * unit
            mid = float(table["p50"]) * unit if table.get("p50") is not None else None
            high = float(table["p90"]) * unit if "p90" in table else None
            band = CORPUS_BAND if high is not None else CORPUS_LOW_ONLY_BAND
            band_label = "corpus p10-p90"
        direction = None
        if value < low:
            direction = "below"
        elif high is not None and value > high:
            direction = "above"
        if direction is None:
            continue
        low, high = round(low, 1), (round(high, 1) if high is not None else None)
        mid = round(mid, 1) if mid is not None else None
        where = (f"below the corpus p10 ({low:g})" if high is None
                 else f"{direction} the {band_label} band ({low:g} to {high:g})")
        flags.append({
            "metric": metric,
            "value": value,
            "corpus_p10": low,
            "corpus_p50": mid,
            "corpus_p90": high,
            "band": band,
            "note": f"{label} {value:g} is {where}; unusual on these maps, not necessarily wrong",
        })
    return flags
