"""What every tool layer shares: the config, the one map reader, the norms,
and the two lookups nearly every tool starts with.

Built once by `server.build_server` and handed to each layer's
`register(mcp, ctx)`. Before this existed the same things were closures
inside `build_server`, which kept the tools in one file; with the tools
split by layer (AGENTS.md) they need one object to share.

Also here, because every tool module already imports this one and may
import no other wiring: `mcp_tool`, the one way a wrapper is registered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .caltopo.reader import MapReader
from .caltopo.state import DEFAULT_MAX_AGE_SECONDS, MapState
from .config import CaltopoConfig
from .handles import Resolved, resolve
from .planning.norms import load_norms


@dataclass
class ServerContext:
    config: CaltopoConfig
    reader: MapReader
    #: The assignment norms: the shipped corpus, or `--norms`' file.
    norms: dict = field(default_factory=dict)

    @classmethod
    def build(cls, config: CaltopoConfig) -> "ServerContext":
        return cls(config=config, reader=MapReader(config),
                   norms=load_norms(config.norms_path))

    def state(self, map_id: str, public: bool,
              max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS) -> MapState:
        """Map state, by whichever path the caller is entitled to.

        `public=True` uses the unauthenticated since endpoint and needs no
        credentials at all - which is how most of this project has been
        exercised. `public=False` needs a `cts.ini`; a read-only team key is
        sufficient and is all v1 ever wants. Either way the read goes
        through the server's one `MapReader`, which holds the prior in
        memory and, for a private map, holds the session itself.
        """
        return self.reader.read(map_id, public=public, max_age_seconds=max_age_seconds)

    def staged(self, handle: str) -> Resolved:
        """A staged directory, revalidated against the live assignment.

        Every computation goes through here. A listing may be slightly
        stale provided it says so; a computed number may not be, because a
        boundary that moved after staging would have coverage reported for
        ground nobody walked.
        """
        return resolve(handle, self.reader)


def mcp_tool(mcp, *, name: str, tags: set[str]):
    """`mcp.tool(name=name, tags=tags)`, with the wrapper renamed `name` first.

    A tool's wrapper is called `_<tool>_tool` so that it does not shadow the
    plain function of the tool's own name inside `register`
    (AGENTS.md, rule 1). FastMCP takes the tool's name from
    `name=`, but it builds the argument validator from the function itself,
    and pydantic titles every validation error with the function's
    `__name__`. Left alone, a caller who passes a bad argument would read
    "validation error for call[_lz_candidates_tool]", which names a tool
    that does not exist, and a model may then try to call it. Renaming the
    function here keeps the error naming the tool, as it did before 0.27.2.
    Only `__name__` changes: `__qualname__` still says where the wrapper is
    defined, for a repr or a debugger.
    """
    def decorate(fn):
        fn.__name__ = name
        return mcp.tool(name=name, tags=tags)(fn)
    return decorate
