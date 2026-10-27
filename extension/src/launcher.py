"""Start the newest sar-geo-mcp from PyPI: the Claude Desktop extension's server.

Claude Desktop runs this with `uv run` (manifest.json, server type "uv").
It does not contain the server. On every start it asks PyPI for the newest
release of `sar-geo-mcp` and runs that with `uv tool run`, so a friend who
installed the extension once gets each new release by restarting Claude.

- Online: runs the newest release. The first start after a release
  downloads it (a minute or so); later starts reuse uv's cache.
- Offline: runs whichever version uv already has cached.
- Pinned: the "Pin version" setting holds one version, for a field laptop
  that must not change under an operation.

Nothing may be written to stdout here: it is the MCP channel. Messages go
to stderr, which Claude Desktop keeps in its MCP log.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.request

PACKAGE = "sar-geo-mcp"
PYPI_JSON_URL = f"https://pypi.org/pypi/{PACKAGE}/json"
#: rasterio, scipy and pysheds all publish wheels for it on every platform.
PYTHON = "3.12"
#: Settings the manifest passes as environment variables. A blank optional
#: field may arrive empty or as the unexpanded placeholder; both mean unset.
SETTINGS = ("ZA_CALTOPO_ID", "ZA_CALTOPO_KEY", "ZA_CALTOPO_ACCOUNT_ID",
            "ZA_WMS_TOKEN", "ZA_MAPBOX_TOKEN", "ZA_GOOGLE_MAPS_KEY",
            "ZA_CACHE_DIR", "SAR_GEO_EXPORTS_DIR", "SAR_GEO_PIN_VERSION")
#: Folder settings: placeholders filled here, an unusable one dropped so the
#: server picks its default (exports ~/Documents/sar_geo, cache the system
#: cache folder).
FOLDERS = ("ZA_CACHE_DIR", "SAR_GEO_EXPORTS_DIR")


def log(message: str) -> None:
    print(f"[sar-geo launcher] {message}", file=sys.stderr, flush=True)


def placeholders() -> dict[str, str]:
    """The folder placeholders Claude Desktop documents for settings. It has
    been seen passing them through unexpanded, so they are filled here."""
    home = os.path.expanduser("~")
    return {"${HOME}": home,
            "${DOCUMENTS}": os.path.join(home, "Documents"),
            "${DOWNLOADS}": os.path.join(home, "Downloads"),
            "${DESKTOP}": os.path.join(home, "Desktop")}


def expand_folder(value: str) -> str:
    """A folder setting with placeholders, ~ and $VARS filled; "" if unusable."""
    value = value.strip()
    for placeholder, folder in placeholders().items():
        value = value.replace(placeholder, folder)
    value = os.path.expanduser(os.path.expandvars(value))
    return "" if "${" in value else value


def clean_settings(env: dict[str, str]) -> dict[str, str]:
    """Drop settings left blank, so the server sees them as unset, and fill
    folder placeholders."""
    env = dict(env)
    for name in FOLDERS:
        if name in env:
            env[name] = expand_folder(env[name])
    for name in SETTINGS:
        value = env.get(name, "").strip()
        if not value or value.startswith("${"):
            env.pop(name, None)
        else:
            env[name] = value
    return env


def find_uv() -> str:
    """The uv binary: the one running us, one on PATH, or the uv package's."""
    found = os.environ.get("UV") or shutil.which("uv")
    if found:
        return found
    try:
        from uv import find_uv_bin  # the `uv` dependency in pyproject.toml

        return find_uv_bin()
    except Exception as exc:  # pragma: no cover - nothing left to try
        raise SystemExit(f"[sar-geo launcher] cannot find uv: {exc}")


def latest_version(timeout: float = 4.0) -> str | None:
    """The newest release on PyPI, or None when PyPI cannot be reached."""
    try:
        with urllib.request.urlopen(PYPI_JSON_URL, timeout=timeout) as response:
            return json.load(response)["info"]["version"]
    except Exception as exc:
        log(f"could not reach PyPI ({type(exc).__name__}); using the cached copy")
        return None


def plan(env: dict[str, str], latest=latest_version) -> tuple[str, list[str]]:
    """The launch mode and the `uv tool run` arguments after the uv binary."""
    pin = env.get("SAR_GEO_PIN_VERSION")
    common = ["tool", "run", "--python", PYTHON]
    if pin:
        return "pinned", [*common, "--from", f"{PACKAGE}[mcp]=={pin}", PACKAGE]
    version = latest()
    if version:
        return "latest", [*common, "--from", f"{PACKAGE}[mcp]=={version}", PACKAGE]
    return "offline", [*common, "--offline", "--from", f"{PACKAGE}[mcp]", PACKAGE]


def main() -> int:
    env = clean_settings(os.environ)
    mode, args = plan(env)
    env["SAR_GEO_LAUNCH_MODE"] = mode
    uv = find_uv()
    log(f"mode={mode}: {' '.join(args)}")
    if os.name == "posix":
        # Replace this process, so Claude Desktop's stop signal reaches the
        # server itself rather than a parent that leaves it running.
        os.execve(uv, [uv, *args], env)
    return subprocess.run([uv, *args], env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
