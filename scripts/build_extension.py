"""Build the Claude Desktop extension: dist/sar-geo.mcpb.

    .venv/bin/python scripts/build_extension.py

Refreshes the tool list in extension/manifest.json from the server as it
stands (Claude Desktop shows it on the install page), then validates and
packs the extension with the official `mcpb` CLI (through npx, so Node is
needed on the machine that builds - not on the machines that install).

The .mcpb holds only the launcher. A friend installs it once; new server
releases reach them from PyPI on their next Claude restart, so this needs
re-running only when something in extension/ changes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / "extension"
MANIFEST = EXTENSION / "manifest.json"
OUTPUT = ROOT / "dist" / "sar-geo.mcpb"
MCPB = ["npx", "--yes", "@anthropic-ai/mcpb@2"]


def tool_list() -> list[dict[str, str]]:
    """Each registered tool's name and the first line of its description."""
    import asyncio

    from sar_geo_mcp.config import CaltopoConfig
    from sar_geo_mcp.server import build_server

    async def listed():
        from fastmcp import Client

        async with Client(build_server(CaltopoConfig())) as client:
            return await client.list_tools()

    return [{"name": tool.name,
             "description": (tool.description or "").strip().splitlines()[0]}
            for tool in asyncio.run(listed())]


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["tools"] = tool_list()
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"manifest: {len(manifest['tools'])} tools")

    OUTPUT.parent.mkdir(exist_ok=True)
    subprocess.run([*MCPB, "validate", str(MANIFEST)], check=True)
    subprocess.run([*MCPB, "pack", str(EXTENSION), str(OUTPUT)], check=True)
    print(f"built {OUTPUT.relative_to(ROOT)} ({OUTPUT.stat().st_size / 1e3:.0f} kB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
