"""Lazily-computed shared inputs for feature detectors.

One context per identification call. Every expensive intermediate is a
cached property, so asking for `drainage` alone never reads vegetation or
roads, and asking for two slope-derived kinds computes slope once.

Optional inputs (vegetation panel, osm.json) come back as None when the
staged directory lacks them; the detector converts None into a warning
envelope. Required inputs (dem.tif) raise, exactly as `load_assignment_grid`
already does.

Construction itself costs nothing - every input is lazy - so a caller that
wants only `map_assignments` may build one per call and pay for no raster.
A caller that wants the expensive intermediates MANY times over one staged
directory should not: `planning.boundary.Ground` holds those once per
directory, because the flow routing behind them is per-directory work and
repeating it per target is what makes describing hundreds of candidates
slow.
"""

import json
from functools import cached_property
from pathlib import Path

import numpy as np


class TerrainContext:
    def __init__(self, data_dir: Path, *, map_features: list[dict] | None = None):
        self.data_dir = Path(data_dir)
        #: The features of the map this subject was staged with, when the
        #: caller read them (the server does, with forced revalidation).
        #: None means "no map, or could not be read" - the two are told
        #: apart by the caller's own warnings.
        self.map_features = map_features

    @cached_property
    def manifest(self) -> dict:
        return json.loads(
            (self.data_dir / "manifest.json").read_text(encoding="utf-8")
        )

    @cached_property
    def subject(self) -> dict:
        """subject.json, or assignment.json for directories staged before it."""
        from ..grid import read_subject

        return read_subject(self.data_dir)

    @property
    def map_assignments(self) -> list[dict] | None:
        """Assignment features on the staged map, or None without a map."""
        if self.map_features is None:
            return None
        return [f for f in self.map_features
                if (f.get("properties") or {}).get("class") == "Assignment"
                and f.get("geometry")]

    @cached_property
    def has_voids(self) -> bool:
        """Whether the staged elevation has nodata cells. The flow routines fill
        them (see _pysheds_backend.flow_accumulation) so a hole does not cut a
        channel network, but a filled hole attracts flow - drainage tracing
        on the real surface, ridge tracing on the negated one - so a detector
        that routed through one says so."""
        return bool(np.isnan(self.grid.elevation).any())

    def _min_cells(self) -> float:
        from ..hydro import HYDRO_ASSUMPTIONS

        return (HYDRO_ASSUMPTIONS["min_contributing_area_m2"]
                / (self.grid.pixel_size_m ** 2))

    @cached_property
    def grid(self):
        from ..grid import load_assignment_grid

        return load_assignment_grid(self.data_dir)

    @cached_property
    def slope_aspect(self):
        from ..slope import compute_slope_aspect

        return compute_slope_aspect(self.grid.elevation, self.grid.pixel_size_m)

    @cached_property
    def flow(self):
        """D8 routing plus traced channel network, in one pysheds pass."""
        from .._pysheds_backend import flow_accumulation

        return flow_accumulation(
            self.grid.elevation, self.grid.transform,
            network_min_cells=self._min_cells(),
        )

    @cached_property
    def inverted_flow(self):
        """D8 routing over the NEGATED elevation: where the inverted hill's
        water would collect is the ridge crest. Same threshold as `flow`, so
        a ridge needs the same contributing area a drainage does; the
        anatomy report used exactly this construction."""
        from .._pysheds_backend import flow_accumulation

        return flow_accumulation(
            -self.grid.elevation, self.grid.transform,
            network_min_cells=self._min_cells(),
        )

    @cached_property
    def vegetation_proxy(self):
        """CIR proxy on the DEM grid, or None when the panel is missing."""
        from ..vegetation import cir_proxy

        panel = self.data_dir / "panels" / "vegetation.png"
        if not panel.exists():
            return None
        return cir_proxy(panel, self.grid.elevation.shape)

    @cached_property
    def osm_payload(self):
        """The staged OSM cache, or None when it is missing."""
        path = self.data_dir / "osm.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
