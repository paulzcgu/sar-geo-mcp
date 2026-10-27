"""Every external data source this package reads, one module each.

Infra: a source fetches, caches and returns a service's own data, and
knows nothing of search and rescue. See `sources.base` for the classes and
how to add a source.

The registry is the explicit `MODULES` tuple, as `scenarios.MODULES` is: a
source is known because its module is imported and listed here, never
because something happened to import it.
"""

from __future__ import annotations

from types import ModuleType

from .base import (
    Area,
    DataSource,
    FeatureResult,
    FeatureSource,
    RasterResult,
    RasterSource,
    Readiness,
    ServiceSource,
    SourceInfo,
    SourceUnavailable,
    Transport,
)
from . import caltopo_wms, dem, landfire

#: Source modules, each defining one or more `DataSource` subclasses.
MODULES: tuple[ModuleType, ...] = (dem, landfire, caltopo_wms)


def source_classes() -> dict[str, type[DataSource]]:
    """Every registered source class by its `info.name`, in listed order."""
    found: dict[str, type[DataSource]] = {}
    for module in MODULES:
        for value in vars(module).values():
            if (isinstance(value, type) and issubclass(value, DataSource)
                    and value.__module__ == module.__name__
                    and "info" in vars(value)):
                if value.info.name in found:
                    raise ValueError(f"two sources are named {value.info.name!r}")
                found[value.info.name] = value
    return found


def get(name: str) -> type[DataSource]:
    classes = source_classes()
    if name not in classes:
        raise ValueError(f"unknown source {name!r}; known: {', '.join(classes)}")
    return classes[name]


__all__ = [
    "Area", "DataSource", "FeatureResult", "FeatureSource", "MODULES",
    "RasterResult", "RasterSource", "Readiness", "ServiceSource", "SourceInfo",
    "SourceUnavailable", "Transport", "get", "source_classes",
]
