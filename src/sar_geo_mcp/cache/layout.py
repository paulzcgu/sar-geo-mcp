"""One folder per data source under the cache, and moving an older cache in.

LAYOUT
------
Each source keeps what it fetches in a folder of its own, named for the
source: `<cache>/usgs_3dep/`, `<cache>/osm/`, `<cache>/caltopo_wms/` ...
Inside it the files keep the names they always had, so a file's index key
is the old key with the folder in front. The index stays one file at the
cache root, so `cache_status` and `evict_cache` see every source at once.
Staged ground (`staged/`) and CalTopo map reads (`caltopo-state/`) are not
a source's and stay where they are.

MIGRATION
---------
Before 0.43.0 every source wrote at the cache root. The first time a
source asks for its folder in a process (`source_folder`), its files at
the old paths are moved in - a rename on the same disk, never a download -
and their index keys rewritten. It runs every process, not once per cache,
and every step is safe to repeat: so two processes starting together, a
move interrupted half way, or files an older pinned release wrote at the
old paths later, all end in the same place. A file that cannot be moved
now (Windows will not rename a file another process holds open) is left
for the next process; it is never deleted.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import threading
from pathlib import Path

from .index import INDEX_FILENAME

_done: set[tuple[str, str]] = set()
_lock = threading.Lock()


def source_folder(cache_dir: Path | str, folder: str,
                  legacy: tuple[str, ...] = ()) -> Path:
    """`<cache_dir>/<folder>`, made if missing, with this source's files at
    their pre-0.43.0 paths (`legacy`, glob patterns relative to the cache
    root) moved in first. Cheap after the first call in a process."""
    root = Path(cache_dir)
    target = root / folder
    if legacy:
        key = (str(root.resolve()), folder)
        if key not in _done:
            with _lock:
                if key not in _done:
                    migrate(root, folder, legacy)
                    _done.add(key)
    target.mkdir(parents=True, exist_ok=True)
    return target


def migrate(root: Path, folder: str, legacy: tuple[str, ...]) -> int:
    """Move every file or directory at the cache root matching `legacy`
    into `folder`, and rewrite the index keys under it. Returns how many
    paths were moved. Safe to run again at any point."""
    if not root.is_dir():
        return 0
    moved = 0
    for pattern in legacy:
        for old in sorted(root.glob(pattern)):
            relative = old.relative_to(root)
            if relative.parts[0] == folder:
                continue
            new = root / folder / relative
            try:
                moved += _move(old, new)
            except OSError:
                continue    # held open elsewhere: the next process moves it
            _remove_empty_parents(old.parent, root)
        _rekey(root, folder, pattern)
    return moved


def _remove_empty_parents(directory: Path, root: Path) -> None:
    """Remove the folders a move emptied (the old `tiles/`), up to the root."""
    while directory != root and root in directory.parents:
        try:
            directory.rmdir()
        except OSError:
            return    # not empty: another source's tiles are still in it
        directory = directory.parent


def _move(old: Path, new: Path) -> int:
    """Move one path. A file already at `new` wins: it was moved, or
    fetched afresh, by another process, and the content is the same ground
    from the same service. A directory is merged file by file."""
    new.parent.mkdir(parents=True, exist_ok=True)
    if old.is_dir() and not old.is_symlink():
        if not new.exists():
            os.replace(old, new)
            return 1
        count = 0
        for child in sorted(old.iterdir()):
            count += _move(child, new / child.name)
        try:
            old.rmdir()
        except OSError:
            pass
        return count
    if new.exists():
        old.unlink()
        return 0
    try:
        os.replace(old, new)
    except OSError:
        # Another volume (a cache folder that is a mount point): copy, then remove.
        shutil.copy2(old, new)
        old.unlink()
    return 1


def _rekey(root: Path, folder: str, pattern: str) -> None:
    """Prefix the folder onto every index key the pattern names. GLOB is
    SQLite's own, with the shell's rules; a root-level pattern (no '/')
    matches root-level keys only."""
    index = root / INDEX_FILENAME
    if not index.exists():
        return
    try:
        connection = sqlite3.connect(index, timeout=5.0)
        try:
            if "/" in pattern:
                where = "key GLOB ? || '/*' OR key GLOB ?"
                args = (pattern, pattern)
            else:
                where = "key GLOB ? AND instr(key, '/') = 0"
                args = (pattern,)
            connection.execute(
                f"UPDATE OR REPLACE entries SET key = ? || '/' || key WHERE {where}",
                (folder, *args))
            connection.commit()
        finally:
            connection.close()
    except sqlite3.Error:
        pass    # a broken index never blocks a fetch (see cache.index)
