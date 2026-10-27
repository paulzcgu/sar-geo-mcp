"""The Assignment domain model, its mapping from CalTopo features, and
finding one on a map by whatever a person called it.

`find_assignment` lived in `session.py` until 0.27.1. It is parsing, not
transport - it reads labels and builds an `Assignment` - and leaving it
there meant the CalTopo session (infra) imported this module and `labels`
(domain), the wrong way down the layers (AGENTS.md). It sits
here, beside the model it returns.
"""

from pydantic import BaseModel

from .labels import label_of
from .labels import match as match_label


def _text(properties: dict, key: str) -> str:
    """CalTopo omits some keys and sends null for others; both mean 'empty'."""
    value = properties.get(key)
    return "" if value is None else str(value)


class Assignment(BaseModel):
    """A search assignment as defined on a CalTopo map."""

    id: str
    letter: str = ""
    number: str = ""
    title: str = ""
    description: str = ""
    resource_type: str = ""
    priority: str = ""
    operational_period_id: str = ""
    status: str = ""
    # Planner-entered operational values. A later analysis phase must read these
    # rather than estimate quantities a human already specified.
    responsive_pod: str = ""
    unresponsive_pod: str = ""
    clue_pod: str = ""
    team_size: str = ""
    time_allocated: str = ""
    transportation: str = ""
    previous_efforts: str = ""
    geometry: dict

    @classmethod
    def from_feature(cls, feature: dict) -> "Assignment":
        properties = feature.get("properties", {})
        feature_class = properties.get("class")
        if feature_class != "Assignment":
            raise ValueError(
                f"not an Assignment feature: class={feature_class!r}"
            )

        geometry = feature.get("geometry")
        if not geometry:
            raise ValueError(f"assignment {feature.get('id')!r} has no geometry")

        return cls(
            id=str(feature["id"]),
            letter=_text(properties, "letter"),
            number=_text(properties, "number"),
            title=_text(properties, "title"),
            description=_text(properties, "description"),
            resource_type=_text(properties, "resourceType"),
            priority=_text(properties, "priority"),
            operational_period_id=_text(properties, "operationalPeriodId"),
            status=_text(properties, "status"),
            responsive_pod=_text(properties, "responsivePOD"),
            unresponsive_pod=_text(properties, "unresponsivePOD"),
            clue_pod=_text(properties, "cluePOD"),
            team_size=_text(properties, "teamSize"),
            time_allocated=_text(properties, "timeAllocated"),
            transportation=_text(properties, "transportation"),
            previous_efforts=_text(properties, "previousEfforts"),
            geometry=geometry,
        )

    @property
    def label(self) -> str:
        """Field-facing name: 'AD3', or the title when there is no letter."""
        combined = f"{self.letter}{self.number}".strip()
        return combined or self.title


def find_assignment(features: list[dict], letter: str) -> Assignment:
    """Find an assignment by whatever a person actually called it.

    Agencies do not share a labelling convention, and searchers do not
    always follow their own. All of these find the same assignment:

        BT      bt      BT (201)    bt-201    BT201

    Matching is tiered - exact, then bracket-stripped core, then prefix -
    so an exact label always wins over something that merely starts the
    same way: `AB` finds `AB` even where `ABC` exists.

    An ambiguous query raises and lists what it matched. Picking one would
    send a team to the wrong ground, and being asked to type a fuller name
    is a far smaller cost than that.
    """
    assignments = [
        f for f in features if (f.get("properties") or {}).get("class") == "Assignment"
    ]

    matched, how = match_label(assignments, letter)

    if len(matched) == 1:
        return Assignment.from_feature(matched[0])

    if len(matched) > 1:
        names = sorted(label_of(f) or "(unlabelled)" for f in matched)
        # "Use the full label" is useless advice when the map itself gives
        # two assignments the SAME full label, which 21 of 96 real maps do
        # (see `labels.match`). Say which case this is, and name the ids -
        # an id is a valid query and is the only thing that can separate
        # them.
        if len(set(names)) == 1:
            ids = ", ".join(sorted(str(f.get("id") or "?") for f in matched))
            raise LookupError(
                f"this map has {len(matched)} different assignments all "
                f"labelled {names[0]!r}, so a name cannot pick one. Ask for "
                f"it by CalTopo id instead - they are: {ids}"
            )
        raise LookupError(
            f"assignment {letter!r} is ambiguous ({how} match); it could be: "
            f"{', '.join(names)}. Use the full label."
        )

    labelled = sorted(l for l in (label_of(f) for f in assignments) if l)
    blank = sum(1 for f in assignments if not label_of(f))
    available = ", ".join(labelled) or "(none)"
    if len(available) > 400:
        available = available[:400] + f" ... ({len(labelled)} total)"
    note = (f" {blank} assignment(s) on this map carry no label at all and "
            f"cannot be looked up by name." if blank else "")
    raise LookupError(
        f"assignment {letter!r} not found. Available: {available}.{note}"
    )
