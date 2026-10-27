"""Components: is this install set up, and is it the newest version.

Two tools a user's Claude calls when someone asks "why can't I see
imagery?" or "am I on the latest version?". Neither ever returns a key.
"""

from __future__ import annotations

from typing import Any

from ..context import ServerContext, mcp_tool
from ..updates import (LAUNCH_MODES, installed_version, latest_release,
                       launch_mode, version_key)

#: Each optional key: the extension setting it is typed into, what it turns
#: on, what happens without it, and where to get one.
KEYS: tuple[dict[str, str], ...] = (
    {
        "key": "caltopo_credential",
        "setting": "CalTopo credential ID + CalTopo credential key",
        "env": "ZA_CALTOPO_ID, ZA_CALTOPO_KEY (or a cts.ini file)",
        "enables": "reading private and team CalTopo maps (public=False)",
        "without": "public maps still work; private and team maps cannot be read",
        "where": "caltopo.com/map.html -> click your name -> Your Account -> "
                 "Credentials. A read-only team credential is enough.",
    },
    {
        "key": "wms_token",
        "setting": "CalTopo WMS token",
        "env": "ZA_WMS_TOKEN",
        "enables": "the raster tools (shaded_relief, slope_shading), the "
                   "canopy panel, and CalTopo imagery chips on landing zones",
        "without": "those tools report that no token is configured; "
                   "elevation, OpenStreetMap and everything else still work",
        "where": "caltopo.com/map.html -> click your name -> Your Account -> "
                 "API Access: the six characters between /api/ and /wms in "
                 "the WMS link. Needs a paid CalTopo plan or team.",
    },
    {
        "key": "mapbox_token",
        "setting": "Mapbox token",
        "env": "ZA_MAPBOX_TOKEN",
        "enables": "Mapbox satellite chips in lz_candidates",
        "without": "no Mapbox chips; the CalTopo chips still work",
        "where": "account.mapbox.com -> Tokens (the free tier is enough)",
    },
    {
        "key": "google_maps_key",
        "setting": "Google Maps key",
        "env": "ZA_GOOGLE_MAPS_KEY",
        "enables": "Google satellite chips in lz_candidates",
        "without": "no Google chips; the CalTopo chips still work",
        "where": "console.cloud.google.com -> APIs & Services -> Credentials, "
                 "with the Maps Static API enabled. Google's terms limit how "
                 "these images may be used.",
    },
)


def _is_set(ctx: ServerContext, key: str) -> tuple[bool, str | None]:
    """Whether a key is configured, and a note when only partly."""
    config = ctx.config
    if key == "caltopo_credential":
        if config.caltopo_id and config.caltopo_key:
            note = None if config.caltopo_account_id else (
                "the CalTopo account ID is empty; maps read fine, but PDF "
                "printing through CalTopo needs it")
            return True, note
        if config.caltopo_id or config.caltopo_key:
            return False, ("only one of the credential ID and key is set; "
                           "both are needed")
        if config.caltopo_config_path.exists():
            return True, f"read from {config.caltopo_config_path}"
        return False, None
    value = getattr(config, key)
    return bool(value), None


def key_status(ctx: ServerContext) -> dict[str, Any]:
    """The `key_status` tool as a plain function: `ctx` in, the tool's
    payload back. What each field means is in the tool's description, in
    `register` below."""
    keys = []
    for spec in KEYS:
        is_set, note = _is_set(ctx, spec["key"])
        entry = {**spec, "set": is_set}
        if note:
            entry["note"] = note
        keys.append(entry)
    return {
        "keys": keys,
        "missing": [entry["key"] for entry in keys if not entry["set"]],
        "cache_dir": str(ctx.config.cache_dir) if ctx.config.cache_dir else None,
        "exports_dir": str(ctx.config.exports_dir),
        "how_to_change": ("Claude Desktop -> Settings -> Extensions -> "
                          "SAR Geo -> Configure, then turn the extension off "
                          "and on. Running from source instead: set the "
                          "environment variables named in `env`."),
        "note": "Every key is optional. Values are never shown, only whether "
                "each is set.",
    }


def check_for_update(ctx: ServerContext) -> dict[str, Any]:
    """The `check_for_update` tool as a plain function: `ctx` in, the
    tool's payload back. What each field means is in the tool's
    description, in `register` below."""
    running = installed_version()
    mode = launch_mode()
    result: dict[str, Any] = {
        "running": running,
        "launch_mode": mode,
        "launch_mode_meaning": LAUNCH_MODES.get(mode) if mode else None,
        "latest": None,
        "released": None,
        "status": "unknown",
        "error": None,
    }
    try:
        latest = latest_release()
    except Exception as exc:  # offline is an answer, not a failure
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 404:
            result["error"] = "sar-geo-mcp is not published on PyPI yet"
            result["how_to_update"] = "Nothing to update to yet."
        else:
            result["error"] = f"{type(exc).__name__}: could not reach PyPI"
            result["how_to_update"] = ("Could not check: PyPI is unreachable. "
                                       "Try again when online.")
        return result

    result["latest"] = latest["version"]
    result["released"] = latest["released"]
    result["release_url"] = latest["url"]
    if running is None:
        result["status"] = "unknown"
    elif version_key(running) >= version_key(latest["version"]):
        result["status"] = "up_to_date" if (
            version_key(running) == version_key(latest["version"])) else "ahead"
    else:
        result["status"] = "update_available"

    if result["status"] != "update_available":
        result["how_to_update"] = "Nothing to do."
    elif mode == "pinned":
        result["how_to_update"] = (
            "The extension is held at a fixed version. Clear the 'Pin "
            "version' setting in Claude Desktop -> Settings -> Extensions -> "
            "SAR Geo -> Configure, then turn the extension off and on.")
    elif mode:
        result["how_to_update"] = (
            "Quit Claude Desktop completely and open it again; the extension "
            "fetches the newest version when it starts. The first start "
            "after an update can take a minute.")
    else:
        result["how_to_update"] = (
            "Not running under the Claude Desktop extension. Upgrade with "
            "`pip install -U 'sar-geo-mcp[mcp]'` (or `git pull` in a source "
            "checkout), then restart the MCP client.")
    return result


def register(mcp, ctx) -> None:

    @mcp_tool(mcp, name="key_status", tags={"component"})
    def _key_status_tool() -> dict[str, Any]:
        """Which optional keys are set, what each unlocks, and where to get it.

        Call this when a tool says a key or token is missing, when imagery
        or a private map will not load, or when the user asks how sar-geo
        is set up. Each entry in `keys` has `set` (true or false), the
        extension `setting` to type it into, what it `enables`, what still
        works `without` it, and `where` to get one. `missing` lists the
        unset ones. Values are never returned - only whether each is set.
        """
        return key_status(ctx)

    @mcp_tool(mcp, name="check_for_update", tags={"component"})
    def _check_for_update_tool() -> dict[str, Any]:
        """Is this the newest version of sar-geo, and how to get it if not.

        Compares the version `running` with the `latest` on PyPI.
        `status` is `up_to_date`, `update_available`, `ahead` (a
        development build) or `unknown` (offline, or an uninstalled
        checkout). `launch_mode` says how the Claude Desktop extension
        started the server: `latest` (it fetched the newest release),
        `offline` (PyPI was unreachable, so the copy on this computer ran)
        or `pinned` (held at a fixed version); null outside the extension.
        `how_to_update` is the instruction to give the user.
        """
        return check_for_update(ctx)
