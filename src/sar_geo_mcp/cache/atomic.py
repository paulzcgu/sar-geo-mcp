"""Landing a cache file without two writers ruining each other's work.

WHY THIS EXISTS
---------------
Every cache writer here follows the same shape: write a temp file beside
the destination, then rename it over the destination, so an interrupted
download cannot poison the cache with a truncated file that later looks
like a hit. The temp name used to be derived from the destination -
`TESTMAP.json.part` - which is fine for one writer and wrong for two.

Two writers are normal in this system. Starting a planning session reads
the map twice at once: the web thread for the PLS echo, the staging thread
a moment behind it. Two sessions staging overlapping ground share a DEM
tile, an EVT tile and an OSM tile. Two `hasty_candidates` calls on one
handle share a candidate-set file. On a cold cache both writers ran, and
both wrote the SAME temp path. What the operator saw was a
`PermissionError` on the first click of every session and success on the
second, because by the second click the cache was warm and only one writer
was left:

    pathlib.py:1047 in write_text
    PermissionError: [Errno 13] Permission denied:
        'cache\\caltopo-state\\TESTMAP.json.part'

Measured on the developer's Windows laptop before this module existed: two
threads writing one map state collided in 58 of 60 rounds, and two threads
saving one candidate set in effectively every round.

`caltopo/wms.py` fixed this for tiles and panels in 0.5.3. This is that
fix, lifted out so there is exactly one copy of it and every writer uses
it.

WHAT IT GUARANTEES
------------------
  - the temp name carries a `uuid4().hex`, so two writers never share one;
  - a failure anywhere in the write unlinks the temp and re-raises, so a
    half-written file never survives and never becomes the destination;
  - the final rename is retried a bounded number of times on
    `PermissionError`.

WHY THE RETRY IS NOT SUPERSTITION
---------------------------------
This is a Windows problem specifically. `os.replace` is atomic against
readers, but on Windows the destination is a real named file that another
PROCESS can hold open: an antivirus scanner or the search indexer commonly
opens a file the instant it lands, and for those few milliseconds the
rename comes back as `PermissionError` (WinError 5, "Access is denied").
Two of our own writers renaming onto the same destination at the same
instant widen that window. Whose content wins does not matter - concurrent
writers of one cache key are fetching the same thing and their bytes are
identical - so waiting out the holder is the right answer. The retry is
bounded: a destination that stays locked is a real failure and is raised.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

#: Enough to outlast a scanner's grab, short enough that a genuinely
#: locked file still fails promptly: about a quarter of a second in total.
REPLACE_ATTEMPTS = 5
REPLACE_DELAY_SECONDS = 0.05


def temporary_beside(destination: Path) -> Path:
    """A temp path in the destination's directory, unique to this writer.

    Same directory, because a rename across filesystems is a copy and is
    not atomic. The suffix stays `.part` so the eviction sweep and the
    tests keep recognising a leftover for what it is.
    """
    return destination.with_name(f"{destination.name}.{uuid4().hex}.part")


def replace_onto_shared_destination(temporary: Path, destination: Path) -> None:
    """`temporary.replace(destination)`, tolerant of the destination being
    briefly held. See the module docstring for why that happens."""
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            temporary.replace(destination)
            return
        except PermissionError:
            if attempt == REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(REPLACE_DELAY_SECONDS)


@contextmanager
def atomic_output(destination: Path | str) -> Iterator[Path]:
    """Yield a unique temp path; on a clean exit, rename it onto
    `destination`.

    For a caller that cannot be handed bytes because a library writes the
    file itself - `rasterio.open(path, "w")` is the case here. On any
    exception, from the caller's body or from the rename, the temp file is
    removed and the exception re-raised, leaving the destination untouched.
    """
    destination = Path(destination)
    temporary = temporary_beside(destination)
    try:
        yield temporary
        replace_onto_shared_destination(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_bytes(destination: Path | str, data: bytes) -> Path:
    """Write `data` to `destination` through a unique temp file."""
    destination = Path(destination)
    with atomic_output(destination) as temporary:
        temporary.write_bytes(data)
    return destination


def atomic_write_text(destination: Path | str, text: str,
                      encoding: str = "utf-8") -> Path:
    """Write `text` to `destination` through a unique temp file."""
    destination = Path(destination)
    with atomic_output(destination) as temporary:
        temporary.write_text(text, encoding=encoding)
    return destination
