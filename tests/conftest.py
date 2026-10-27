"""Async test support.

The MCP tools are exercised through fastmcp's in-memory client, which is
async. anyio's default parametrisation would run every async test twice
(asyncio and trio); trio is not a dependency here and the tools are not
event-loop specific, so this pins the backend to asyncio.
"""

import pytest


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


#: Addresses a test may connect to: this machine only.
_LOCAL_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


@pytest.fixture(autouse=True)
def _no_network(request, monkeypatch):
    """Fail any test that opens a connection off this machine, unless it is
    marked `live`. A fake that stops intercepting - a patch left on a module
    a function has moved out of - would otherwise reach Overpass, 3DEP or a
    hospital service, quietly, and pass.

    Attempts are also recorded and fail the test at teardown: much of this
    package turns a failed fetch into a warning, which would swallow the
    error raised here and let the test pass anyway."""
    if request.node.get_closest_marker("live"):
        yield
        return
    import socket

    attempts: list[str] = []

    real_connect = socket.socket.connect

    def guarded_connect(self, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        if isinstance(host, str) and host not in _LOCAL_HOSTS and self.family in (
                socket.AF_INET, socket.AF_INET6):
            attempts.append(host)
            raise RuntimeError(f"test tried to reach the network ({host}); "
                               "fake the request, or mark the test live")
        return real_connect(self, address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    real_getaddrinfo = socket.getaddrinfo

    def guarded_getaddrinfo(host, *args, **kwargs):
        if isinstance(host, str) and host not in _LOCAL_HOSTS:
            attempts.append(host)
            raise RuntimeError(f"test tried to look up {host}; "
                               "fake the request, or mark the test live")
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    yield
    if attempts:
        pytest.fail(f"test tried to reach the network: {sorted(set(attempts))}")
