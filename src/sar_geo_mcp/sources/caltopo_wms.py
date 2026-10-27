"""CalTopo's rendered map layers (WMS) as a source.

The transport - the tile grid, the cache keys, the categorical and alpha
resampling rules - stays in `caltopo.wms`, beside the rest of the CalTopo
code; this class gives it the shape every raster source has. The token
goes into the request URL and nowhere else: never a cache key, a result,
`describe()` or `endpoint()`.
"""

from __future__ import annotations

from pathlib import Path

from ..caltopo import wms
from .base import Area, RasterResult, RasterSource, SourceInfo

DEFAULT_HOST = "caltopo.com"
HOST_ENV = "ZA_CALTOPO_HOST"
TOKEN_ENV = "ZA_WMS_TOKEN"


class CaltopoWmsSource(RasterSource):
    """One panel of one or more CalTopo layers over an area, as a PNG in
    Web Mercator. `layer` (or `layers`) names it from `caltopo.wms.LAYERS`."""

    info = SourceInfo(
        name="caltopo_wms",
        title="CalTopo map layers",
        kind="raster",
        provider="CalTopo",
        default_url=f"https://{DEFAULT_HOST}/api/{{token}}/wms",
        attribution="CalTopo",
        coverage="global basemaps; US-only layers (NAIP, parcels, structures, MVUM)",
        key_env=TOKEN_ENV,
        notes=(
            "Rendered images, not data: a lot line CalTopo does not draw joins two "
            "lots, and a house missing from the structures layer reads as vacant.",
            "Layers: " + ", ".join(f"{key} ({title})" for key, title in wms.LAYERS.items()),
        ),
    )

    #: Layers drawn from USDA NAIP imagery, whose attribution says so.
    NAIP_LAYERS = frozenset({"n3", "ir"})

    def host(self) -> str:
        return (self.env.get(HOST_ENV) or "").strip() or DEFAULT_HOST

    def endpoint(self) -> str:
        """Safe to show: the host, with the token's place marked."""
        return f"https://{self.host()}/api/{{token}}/wms"

    def _base_url(self) -> str:
        token = self.key()
        if token is None:
            raise ValueError(f"caltopo_wms needs {TOKEN_ENV}")
        return f"https://{self.host()}/api/{token}/wms"

    def attribution_for(self, layers: list[str]) -> str:
        if self.NAIP_LAYERS & set(layers):
            return "CalTopo, USDA NAIP"
        return self.info.attribution

    def fetch(self, area: Area, width_px: int, height_px: int, *,
              layer: str | None = None, layers: list[str] | None = None,
              resample: int | None = None, **options) -> RasterResult:
        if options:
            raise ValueError(f"caltopo_wms takes layer, layers and resample; "
                             f"got {sorted(options)}")
        if (layer is None) == (layers is None):
            raise ValueError("caltopo_wms needs exactly one of layer or layers")
        chosen = [layer] if layer is not None else list(layers)
        wms.validate_layers(chosen)
        get_bytes = self.transport.get_bytes if self.transport else wms._http_get
        path = wms.fetch_panel(self._base_url(), chosen, area.bbox_3857, width_px,
                               height_px, self.require_cache(), get_bytes=get_bytes,
                               resample=resample)
        return RasterResult(path=Path(path), bbox_3857=area.bbox_3857,
                            width_px=width_px, height_px=height_px,
                            attribution=self.attribution_for(chosen))
