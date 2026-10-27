"""Scenario tools: a whole planning situation, built from several features.

See AGENTS.md for the planned scenarios. A scenario module
defines its scenario as a plain function taking the `ServerContext` first,
like every tool, and `register(mcp, ctx)`, which wraps it. The function
calls feature and component functions directly with that context, never
through MCP. What every scenario shares - sections and their status, the
refusals made before any work, staging, the export, one helper per feature
- is in `_sections.py`; AGENTS.md describes the conventions.
"""

from __future__ import annotations

from . import debrief, initial_response, medevac, urban_search

LAYER = "scenario"
MODULES = (medevac, initial_response, urban_search, debrief)


def register(mcp, ctx) -> None:
    for module in MODULES:
        module.register(mcp, ctx)
