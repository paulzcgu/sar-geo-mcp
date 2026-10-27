"""The feature registry: what this package can identify, by kind.

One detector class per feature kind, all honouring the same envelope.
Registration is this explicit dict, not decorator side effects - a
registry populated by import order is a trap. Adding a feature is one
new module plus one line here.
"""

from pathlib import Path

from .base import FeatureDetector, FeatureResult, FeatureShape
from .cliff_bands import CliffBandDetector
from .cliff_pitch import CliffPitchDetector
from .clearings import ClearingDetector
from .context import TerrainContext
from .drainages import DrainageDetector
from .ingress import IngressDetector
from .ridges import RidgeDetector
from .routes import RouteDetector
from .steep_bands import SteepBandDetector

#: Kind -> detector class. Insertion order is the order `terrain_metrics`
#: has always reported shapes in, so it is part of the output contract:
#: append, never reorder.
REGISTRY: dict[str, type[FeatureDetector]] = {
    cls.kind: cls
    for cls in (
        SteepBandDetector,
        ClearingDetector,
        DrainageDetector,
        CliffBandDetector,
        RouteDetector,
        IngressDetector,
        RidgeDetector,
        CliffPitchDetector,
    )
}


def kinds() -> list[str]:
    return list(REGISTRY)


def validate_kinds(requested: list[str]) -> None:
    """Raise on an empty request or any kind the registry does not know,
    naming what exists.

    A model that hallucinates a kind - or asks for nothing - should be
    told what actually exists, not handed an empty result it will read
    as "nothing there".
    """
    if not requested:
        raise ValueError(
            f"kinds must name at least one feature kind. "
            f"Available: {', '.join(REGISTRY)}"
        )
    unknown = [kind for kind in requested if kind not in REGISTRY]
    if unknown:
        raise ValueError(
            f"unknown feature kind(s): {', '.join(unknown)}. "
            f"Available: {', '.join(REGISTRY)}"
        )


def input_missing(kind: str, warnings) -> bool:
    """Whether a `kind`'s answer carrying `warnings` is unknown rather than
    empty: one of them is a warning its detector declares for a missing
    input (`FeatureDetector.missing_input_warnings`). Its count of 0 then
    means nothing could be looked at - a reader that keeps the count apart
    from the warning must give it as unknown, never as "none here". False
    for a kind the registry does not know: nothing was declared for it."""
    detector = REGISTRY.get(kind)
    return detector is not None and any(
        warning in detector.missing_input_warnings for warning in warnings or ())


def list_kinds() -> list[dict]:
    """One entry per registered detector, for `list_feature_kinds`."""
    return [
        {"kind": cls.kind, "description": cls.description,
         "assumptions": dict(cls.assumptions)}
        for cls in REGISTRY.values()
    ]


def identify(data_dir: Path, requested: list[str]) -> dict[str, FeatureResult]:
    """Run the requested detectors over one staged directory.

    One shared context: intermediates are computed lazily and at most
    once, so the cost is what the requested kinds actually need.
    """
    validate_kinds(requested)
    ctx = TerrainContext(data_dir)
    return {kind: REGISTRY[kind]().detect(ctx)
            for kind in dict.fromkeys(requested)}
