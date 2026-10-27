"""Feature tools: multi-step answers to one planning question each.

See AGENTS.md. Each module here defines one plain function per
tool, taking the `ServerContext` first, and `register(mcp, ctx)`, which
wraps each function as an MCP tool that only calls it. A scenario calls
the functions.
"""

from __future__ import annotations

from . import (
    boundary,
    coverage,
    egress,
    hasty,
    lz,
    medical,
    parking,
    relief,
    travel_rings,
    unmarked_paths,
    urban,
)

LAYER = "feature"
MODULES = (relief, unmarked_paths, boundary, hasty, parking, lz, egress,
           travel_rings, medical, urban, coverage)


def register(mcp, ctx) -> None:
    for module in MODULES:
        module.register(mcp, ctx)
