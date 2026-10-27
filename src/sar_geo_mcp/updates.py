"""Which version is running, and which is the newest on PyPI.

The Claude Desktop extension (`extension/`) starts the server through a
small launcher that asks PyPI for the newest release on every start and
runs that. This module lets the server say what happened: the version
running, the newest one published, and what the launcher did, which it
records in `SAR_GEO_LAUNCH_MODE` before starting the server.

Plumbing only: no SAR knowledge, and nothing here ever installs anything.
"""

from __future__ import annotations

import os
import re
from importlib import metadata

DISTRIBUTION = "sar-geo-mcp"
PYPI_JSON_URL = f"https://pypi.org/pypi/{DISTRIBUTION}/json"
#: Set by `extension/launcher.py` for the server it starts.
LAUNCH_MODE_ENV = "SAR_GEO_LAUNCH_MODE"
#: What each launch mode means, in words a user can act on.
LAUNCH_MODES = {
    "latest": "started from the newest release on PyPI",
    "offline": "PyPI could not be reached at start, so the copy already "
               "on this computer was started",
    "pinned": "held at a fixed version by the extension's settings",
}


def installed_version() -> str | None:
    """The running package's version, or None for an uninstalled checkout."""
    try:
        return metadata.version(DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return None


def launch_mode() -> str | None:
    """How the extension's launcher started this server; None outside it."""
    return os.environ.get(LAUNCH_MODE_ENV) or None


def version_key(version: str) -> tuple:
    """A sort key for plain release versions ("0.38.0", "1.2", "1.0rc1").

    Numeric parts compare as numbers. A pre-release (any letters after the
    numbers) sorts before the release it leads up to. Enough for this
    project's own versions, without depending on `packaging`.
    """
    match = re.match(r"^\s*v?(\d+(?:\.\d+)*)(.*)$", version or "")
    if not match:
        return ((), 0, version or "")
    numbers = tuple(int(part) for part in match.group(1).split("."))
    while numbers and numbers[-1] == 0:
        numbers = numbers[:-1]
    suffix = match.group(2).strip()
    return (numbers, 0 if suffix else 1, suffix)


def latest_release(fetch=None, timeout: float = 5.0) -> dict:
    """The newest release on PyPI: {"version", "released", "url"}.

    `fetch(url, timeout)` returns the decoded JSON; the default uses httpx.
    Raises whatever the fetch raises - the caller decides what offline
    means to its user.
    """
    if fetch is None:
        def fetch(url, timeout):
            import httpx

            response = httpx.get(url, timeout=timeout, follow_redirects=True)
            response.raise_for_status()
            return response.json()

    data = fetch(PYPI_JSON_URL, timeout)
    info = data.get("info") or {}
    version = info.get("version")
    if not version:
        raise ValueError("PyPI's reply named no version")
    files = (data.get("releases") or {}).get(version) or data.get("urls") or []
    released = min((f.get("upload_time_iso_8601") or f.get("upload_time") or ""
                    for f in files), default="") or None
    return {
        "version": version,
        "released": released,
        "url": f"https://pypi.org/project/{DISTRIBUTION}/{version}/",
    }
