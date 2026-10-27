"""Components: handing a file this server wrote to the person using it.

Tools write GeoJSON exports under the cache directory, which a chat client
usually cannot open or link to. `save_export` copies one into the person's
exports folder - ~/Documents/sar_geo/<chat folder> - and shows it in Finder
or Explorer, so "save that file for me" is one tool call. The server cannot
see which chat it serves, so the caller names the chat's folder.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..context import ServerContext, mcp_tool

#: What `save_export` will copy. Exports, chips and rendered layers - not
#: the cache's own bookkeeping, and nothing outside the cache directory.
ALLOWED_SUFFIXES = frozenset({".geojson", ".json", ".gpx", ".kml", ".png", ".jpg",
                              ".jpeg", ".tif", ".tiff", ".csv"})


def chat_folder_name(name: str, today: dt.date | None = None) -> str:
    """A safe folder name for one chat: no path separators, no leading dots,
    nothing Windows refuses; today's date when nothing usable is given."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', " ", name or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")[:80].rstrip(" .")
    return cleaned or (today or dt.date.today()).isoformat()


def free_name(folder: Path, name: str) -> Path:
    """`folder/name`, or `name (2)`, `name (3)`... when that is taken."""
    target = folder / name
    stem, suffix = Path(name).stem, Path(name).suffix
    n = 2
    while target.exists():
        target = folder / f"{stem} ({n}){suffix}"
        n += 1
    return target


def reveal(path: Path) -> bool:
    """Show `path` selected in Finder / Explorer. False when that failed."""
    try:
        if sys.platform == "darwin":
            command = ["open", "-R", str(path)]
        elif os.name == "nt":
            command = ["explorer", f"/select,{path}"]
        else:
            command = ["xdg-open", str(path.parent)]
        subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
        return True
    except OSError:
        return False


def save_export(ctx: ServerContext, path: str, chat_folder: str = "",
                destination: str = "", show: bool = True) -> dict[str, Any]:
    """The `save_export` tool as a plain function: `ctx`, then the tool's
    arguments; the tool's payload back. What each argument and field means
    is in the tool's description, in `register` below."""
    if not ctx.config.cache_dir:
        raise ValueError("caching is disabled, so this server wrote no files")
    root = Path(ctx.config.cache_dir).resolve()
    source = Path(path).expanduser()
    if not source.is_absolute():
        # Tolerate a path quoted relative to the cache, or one from an older
        # release that named files relative to the working directory.
        candidates = [root / source, root.parent / source]
        source = next((c for c in candidates if c.exists()), candidates[0])
    source = source.resolve()
    if not source.is_relative_to(root):
        raise ValueError(f"{path} is not a file this server wrote; only files "
                         f"under {root} can be saved")
    if not source.is_file():
        raise FileNotFoundError(f"no file at {source}")
    if source.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError(f"{source.name} is not an export; save_export copies "
                         f"{', '.join(sorted(ALLOWED_SUFFIXES))} files")

    if destination.strip():
        folder = Path(destination).expanduser()
        if not folder.is_absolute():
            folder = Path.home() / folder
    else:
        folder = Path(ctx.config.exports_dir) / chat_folder_name(chat_folder)
    folder.mkdir(parents=True, exist_ok=True)
    target = free_name(folder, source.name)
    shutil.copy2(source, target)
    return {
        "saved_to": str(target),
        "file_url": target.as_uri(),
        "folder": str(folder),
        "bytes": target.stat().st_size,
        "shown": reveal(target) if show else False,
        "source": str(source),
    }


def register(mcp, ctx) -> None:

    @mcp_tool(mcp, name="save_export", tags={"component"})
    def _save_export_tool(path: str, chat_folder: str = "", destination: str = "",
                          show: bool = True) -> dict[str, Any]:
        """Save a file a tool wrote (a GeoJSON export, an imagery chip) to the
        user's Documents/sar_geo folder, one subfolder per chat, and show it
        in Finder or Explorer.

        Call this whenever the user wants a file another tool reported in a
        `path` field: the export GeoJSON of `caltopo_export`, `plan_medevac`,
        `plan_initial_response`, `plan_urban_search` or `unmarked_paths`, or
        an `lz_candidates` chip. Chat cannot attach those files, and the
        user cannot easily find the cache folder, so do not tell them to
        search for it - call this instead, then tell them where it went.

        `path` is exactly as the other tool returned it. `chat_folder` names
        this chat's subfolder: a short title for the conversation or
        incident with its date, such as "2026-01-15 trailhead medevac".
        Pick it once and pass the SAME name for every file saved in this
        chat, so they land together; left empty it is today's date.
        `destination` replaces the whole location with a folder the user
        names (`~` works). `show` opens the folder with the file selected. A
        name already taken gets " (2)" added. Returns `saved_to`, the full
        path of the copy, and `file_url`. Only files under this server's
        cache directory can be saved.
        """
        return save_export(ctx, path=path, chat_folder=chat_folder,
                           destination=destination, show=show)
