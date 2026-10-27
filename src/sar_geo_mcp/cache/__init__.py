"""Cache plumbing: how a file lands, and the bookkeeping about it.

The files on disk are the truth. `index` is metadata *about* them: when
each was fetched, which layer it belongs to, how big it is, when it was
last used. None of that is load-bearing - every operation there degrades
to a no-op rather than failing a fetch, because a cache that cannot
describe itself must still work.

`atomic` is the exception, and is load-bearing: it is how every writer in
the package puts a file where a reader will find it, safely against a
second writer doing the same thing at the same instant.
"""

from .atomic import (
    atomic_output,
    atomic_write_bytes,
    atomic_write_text,
    replace_onto_shared_destination,
    temporary_beside,
)
from .index import CacheEntry, CacheIndex, note_hit, note_write

__all__ = [
    "CacheEntry",
    "CacheIndex",
    "atomic_output",
    "atomic_write_bytes",
    "atomic_write_text",
    "note_hit",
    "note_write",
    "replace_onto_shared_destination",
    "temporary_beside",
]
