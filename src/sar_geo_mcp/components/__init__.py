"""Component tools: building blocks with one purpose each.

See AGENTS.md. Each module here defines one plain function per
tool, taking the `ServerContext` first, and `register(mcp, ctx)`, which
wraps each function as an MCP tool that only calls it.
"""

from __future__ import annotations

from . import (caltopo, cache, candidates, coords, files, geometry, ground,
               norms, setup, sources, staging)

LAYER = "component"
MODULES = (caltopo, coords, geometry, staging, ground, norms, candidates, cache,
           files, setup, sources)


def register(mcp, ctx) -> None:
    for module in MODULES:
        module.register(mcp, ctx)
