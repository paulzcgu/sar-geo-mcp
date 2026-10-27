"""The MCP server: CalTopo reads and geo accessories, over stdio.

WHAT THIS IS FOR
----------------
One place anything talks to CalTopo. A planner agent, Claude at a command
post, or any other MCP client gets the same tools; ordinary Python code
imports the same functions directly and skips the server entirely. The
server is an adapter and holds no logic of its own - every tool is a
thin call into a module that is independently tested. The tools are split
into three layers - components, features, scenarios - in their own
packages; AGENTS.md describes them.

Transport is stdio: a subprocess on the operator's own laptop. No port, no
certificate, no new attack surface, and it costs nothing to run. That is a
deliberate match to how this project deploys everywhere else (a folder you
unzip onto a field laptop), and it is why remote transport is a separate
decision with a threat model attached rather than a flag.

SCOPE OF v1 - READ ONLY, AND WHAT IS DELIBERATELY ABSENT
--------------------------------------------------------
Nothing here writes to CalTopo. The write tier (markers, lines, polygons,
draft assignments) is designed but not built: it needs its governance kit
first - a staging folder so proposals are visually segregated from
human-authored features, an ownership ledger so the hub can only ever edit
what it created, dry-run, a map allowlist, and a separate write-capable
credential. Shipping writes before that machinery would put an agent's
mistakes onto a map that live teams navigate by.

Two further absences, both deliberate:

  Terrain for ARBITRARY geometry is staged only as a planner-chosen AREA
  (`prepare_area`): a point and radius, or a polygon, whose input is
  immutable and therefore needs no revalidation. Map context an area is
  compared against is still read fresh and reported with provenance. A
  free-floating assignment-shaped polygon still has no tool of its own -
  describe it against an area with `describe_boundary`.

  Reading the debrief app's artefacts (SAR-110 JSON, coverage summaries).
  Those belong to the app that produces them; teaching this package their
  filenames would point the dependency backwards, at the very moment the
  split was made to stop that.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

from . import components, features, scenarios
from .config import CaltopoConfig
from .context import ServerContext

SERVER_NAME = "sar-geo-mcp"

#: The tool layers, registered bottom up. AGENTS.md says what
#: belongs in each and which way they may depend.
LAYERS = (components, features, scenarios)


def build_server(config: CaltopoConfig | None = None):
    """Construct the server. Importable for tests without starting stdio."""
    from fastmcp import FastMCP

    from .updates import installed_version

    config = config or CaltopoConfig.from_env()
    # The package's own version, so a client's log and serverInfo name the
    # release running rather than fastmcp's.
    mcp = FastMCP(SERVER_NAME, version=installed_version() or "0+unknown")
    ctx = ServerContext.build(config)
    for layer in LAYERS:
        layer.register(mcp, ctx)
    return mcp


def redirect_logging_to_stderr() -> list[str]:
    """Move every stdout log handler to stderr. Returns what was moved.

    Under stdio transport, stdout IS the JSON-RPC channel - a single log
    line written there corrupts the message stream. Importing
    `caltopo_python` installs a root StreamHandler pointing at stdout, so
    merely reading a CalTopo map would inject text into the protocol. That
    is not hypothetical: it showed up the first time this server was
    launched over a real stdio client, as "Failed to parse JSONRPC message
    from server" on every call.

    Handlers are repointed rather than removed - the operator still wants
    to see what the server is doing, just not down the wire the client is
    reading. Called from `main` AFTER the imports that install them.
    """
    moved = []
    for logger in [logging.root, *(logging.Logger.manager.loggerDict.values())]:
        for handler in getattr(logger, "handlers", []):
            if getattr(handler, "stream", None) is sys.stdout:
                handler.setStream(sys.stderr)
                moved.append(f"{getattr(logger, 'name', 'root')}:{type(handler).__name__}")
    return moved


def config_from_args(args: argparse.Namespace) -> CaltopoConfig:
    """The environment's config with the command line's overrides on top.

    `replace` rather than a fresh constructor: an earlier version rebuilt
    the config from the three fields it knew about and silently dropped
    the cache directory and the WMS token, so passing `--caltopo-host`
    disabled caching.
    """
    config = CaltopoConfig.from_env()
    if getattr(args, "caltopo_host", None):
        config = dataclasses.replace(config, caltopo_host=args.caltopo_host)
    if getattr(args, "norms", None):
        config = dataclasses.replace(config, norms_path=Path(args.norms))
    return config


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="sar-geo-mcp",
        description="CalTopo reads and geo accessories over MCP (stdio).",
    )
    parser.add_argument(
        "--caltopo-host", default=None,
        help="Override the CalTopo host (default: ZA_CALTOPO_HOST or caltopo.com)",
    )
    parser.add_argument(
        "--norms", default=None,
        help="Replace the shipped assignment norms with another "
             "agency's JSON file (also ZA_ASSIGNMENT_NORMS)",
    )
    args = parser.parse_args()

    config = config_from_args(args)
    # Order matters: the imports at the top of this module have already run
    # and installed their handlers, so this catches them before the first
    # byte of protocol is written.
    redirect_logging_to_stderr()
    # The startup banner is decoration on stderr for a human, but fastmcp
    # draws it before the transport takes over stdout - suppressed rather
    # than risked.
    build_server(config).run(show_banner=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
