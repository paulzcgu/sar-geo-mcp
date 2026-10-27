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
