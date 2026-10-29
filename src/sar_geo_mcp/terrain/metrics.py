"""Assemble TerrainMetrics from a fetch output directory.

This is the deterministic half of the final ZoneAnalysis document: every number
here is computed, and every polygon carries an id the LLM's evidence lists can
cite. Missing inputs degrade into data_quality.warnings, never into silence.
"""

from pathlib import Path

import numpy as np
from pydantic import BaseModel

from .evt import composition
from .slope import aspect_statistics, slope_statistics
from .traverse import BASIS_DEFAULT, default_width_warning, traverse_metrics
from .vegetation import vegetation_statistics


class ElevationMetrics(BaseModel):
    min_m: float
    max_m: float
    relief_m: float


class SlopeMetrics(BaseModel):
    mean_deg: float | None
    p90_deg: float | None
    max_deg: float | None
    distribution: dict[str, float]


class AspectMetrics(BaseModel):
    dominant: str | None
    distribution: dict[str, float]


class VegetationMetrics(BaseModel):
    canopy_fraction: float | None = None
    density_class: str | None = None
    method: str | None = None
    lifeform_fractions: dict[str, float] | None = None
    dominant_lifeform: str | None = None
    top_types: list[tuple[str, float]] | None = None
    type_source: str | None = None


class TraverseMetrics(BaseModel):
    baseline_rate_kmh: float
    vegetation_adjusted_rate_kmh: float
    effective_sweep_width_m: float
    # How the width was reached (`traverse.sweep_width_basis`); None on
    # metrics saved before 0.43.1 said.
    sweep_width_basis: str | None = None
    sweep_time_hours_per_pass: float | None
    method: str
    assumptions: dict


class DrainageMetrics(BaseModel):
    id: str
    length_m: float
    full_length_m: float
    contributing_area_m2: float
    depth_m: dict[str, float | None]
    sidewall_slope_deg: dict[str, float | None]
    transect_count: int
    any_transect_truncated: bool
    method: str


class HydrologyMetrics(BaseModel):
    drainages: list[DrainageMetrics]
    upstream_truncated: bool
    assumptions: dict


class LandformMetrics(BaseModel):
    fractions: dict[str, float]
    cliff_band_count: int
    assumptions: dict


class AccessMetrics(BaseModel):
    routes: list[dict]
    ingress_count: int
    source: str


class MetricShape(BaseModel):
    id: str
    kind: str
    geometry: dict
    properties: dict


class DataQuality(BaseModel):
    dem_resolution_m: float
    latitude_deg: float
    area_m2: float
    warnings: list[str]


class TerrainMetrics(BaseModel):
    schema_version: str = "0.4.0"
    assignment_label: str
    elevation: ElevationMetrics
    slope: SlopeMetrics
    aspect: AspectMetrics
    vegetation: VegetationMetrics
    traverse: TraverseMetrics
    hydrology: HydrologyMetrics | None = None
    landforms: LandformMetrics | None = None
    access: AccessMetrics | None = None
    shapes: list[MetricShape]
    data_quality: DataQuality


def compute_terrain_metrics(data_dir: Path) -> TerrainMetrics:
    from ..staging import manifest_warnings
    from .features import REGISTRY
    from .features.context import TerrainContext
    from .hydro import HYDRO_ASSUMPTIONS
    from .landform import (
        LANDFORM_ASSUMPTIONS,
        classify_landforms,
        landform_fractions,
    )

    data_dir = Path(data_dir)
    ctx = TerrainContext(data_dir)
    grid = ctx.grid
    manifest = ctx.manifest
    warnings: list[str] = []

    inside = grid.elevation[grid.mask]
    inside = inside[~np.isnan(inside)]
    elevation = ElevationMetrics(
        min_m=float(inside.min()),
        max_m=float(inside.max()),
        relief_m=float(inside.max() - inside.min()),
    )

    slope_aspect = ctx.slope_aspect
    slope = SlopeMetrics(**slope_statistics(slope_aspect.slope_deg, grid.mask))
    aspect = AspectMetrics(
        **aspect_statistics(slope_aspect.aspect_deg, slope_aspect.slope_deg,
                            grid.mask)
    )

    # Every identified feature comes from the registry, in registry order -
    # which is the order this document has always reported shapes in.
    # Every registered detector runs, including the ridge detector's second
    # D8 pass over the negated surface. Accepted (planning phase 1 review):
    # the metrics bundle is computed once per staged subject and cached by
    # its consumers, ridges are part of the anatomy every planner reads, and
    # `identify_features` remains the path for asking about one kind alone.
    envelopes = {kind: cls().detect(ctx) for kind, cls in REGISTRY.items()}
    shapes = [
        MetricShape(**shape.model_dump())
        for envelope in envelopes.values()
        for shape in envelope.shapes
    ]

    vegetation = VegetationMetrics()
    panel = data_dir / "panels" / "vegetation.png"
    if panel.exists():
        vegetation = VegetationMetrics(
            **vegetation_statistics(panel, grid.mask, grid.elevation.shape)
        )
    else:
        warnings.append("vegetation panel missing; density unknown")

    evt_path = data_dir / "evt.tif"
    if evt_path.exists():
        evt = composition(evt_path, grid.geometry_3857)
        vegetation.lifeform_fractions = evt["fractions"]
        vegetation.dominant_lifeform = evt["dominant_lifeform"]
        vegetation.top_types = evt["top_types"]
        vegetation.type_source = evt["source"]
    else:
        warnings.append("evt.tif missing; vegetation type unknown")

    traverse = TraverseMetrics(
        **traverse_metrics(
            slope_deg=slope_aspect.slope_deg,
            mask=grid.mask,
            area_m2=grid.area_m2,
            density_class=vegetation.density_class or "moderate",
            lifeform_fractions=vegetation.lifeform_fractions or {},
            density_measured=vegetation.density_class is not None,
        )
    )
    if traverse.sweep_width_basis == BASIS_DEFAULT:
        warnings.append(default_width_warning(traverse.effective_sweep_width_m))

    drainage_envelope = envelopes["drainage"]
    hydrology = HydrologyMetrics(
        drainages=[DrainageMetrics(**entry)
                   for entry in drainage_envelope.summary["drainages"]],
        upstream_truncated=drainage_envelope.summary["upstream_truncated"],
        assumptions=HYDRO_ASSUMPTIONS,
    )

    classes = classify_landforms(grid.elevation, slope_aspect.slope_deg)
    landforms = LandformMetrics(
        fractions=landform_fractions(classes, grid.mask),
        cliff_band_count=envelopes["cliff_band"].summary["count"],
        assumptions=LANDFORM_ASSUMPTIONS,
    )

    access = None
    if ctx.osm_payload is not None:
        access = AccessMetrics(
            routes=ctx.osm_payload.get("routes", []),
            ingress_count=envelopes["ingress"].summary["count"],
            source="OpenStreetMap via Overpass",
        )

    # Every WARNING comes from the registry too. The envelope exists so
    # that "missing inputs degrade into warnings, never into silence", and
    # this function used to hand-pick the drainage detector's and
    # re-derive the OSM one - which meant the clearing detector's
    # "vegetation panel missing; clearings unknown" was computed and then
    # dropped on the floor, the exact silence the envelope was built to
    # prevent. Deduped, because route and ingress legitimately raise the
    # same missing-OSM warning and a reader should see it once.
    for envelope in envelopes.values():
        for warning in envelope.warnings:
            if warning not in warnings:
                warnings.append(warning)

    return TerrainMetrics(
        assignment_label=manifest.get("assignment", ""),
        elevation=elevation,
        slope=slope,
        aspect=aspect,
        vegetation=vegetation,
        traverse=traverse,
        hydrology=hydrology,
        landforms=landforms,
        access=access,
        shapes=shapes,
        data_quality=DataQuality(
            dem_resolution_m=float(manifest["dem_resolution_m"]),
            latitude_deg=grid.latitude_deg,
            area_m2=grid.area_m2,
            # As `prepare_area` does: an OSM outage recorded at staging is
            # not repeated once any osm.json exists.
            warnings=warnings + manifest_warnings(data_dir, manifest),
        ),
    )
