"""Only infra talks to the network.

Every external service is reached through a source in `sources/` (or the
CalTopo session, the WMS transport, the shared retry policy and the
release check, which are infra too). Domain and tool modules may catch
httpx's error types, but never send a request themselves: a request made
outside a source skips its endpoint override, its key handling and its
safe failure, and is invisible to anyone reading `sources/` for what this
package reads.
"""

from __future__ import annotations

import ast

from test_layering import INFRA, modules

#: httpx names that send a request. Its exception types are not here.
SENDING = frozenset({"get", "post", "put", "patch", "delete", "head", "options",
                     "request", "stream", "Client", "AsyncClient"})


def sends_with_httpx(tree: ast.AST) -> list[int]:
    lines = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and node.attr in SENDING
                and isinstance(node.value, ast.Name) and node.value.id == "httpx"):
            lines.append(node.lineno)
        if isinstance(node, ast.ImportFrom) and node.module == "httpx":
            lines += [node.lineno for alias in node.names if alias.name in SENDING]
    return lines


def test_only_infra_sends_a_request():
    offenders = []
    for name, path in modules().items():
        if name in INFRA:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        offenders += [f"{name}:{line}" for line in sends_with_httpx(tree)]
    assert offenders == [], (
        "these send a request outside infra; read the service through a source "
        "in sources/ instead: " + ", ".join(offenders))


def test_the_check_sees_a_request():
    assert sends_with_httpx(ast.parse("import httpx\nhttpx.get('u')\n")) == [2]
    assert sends_with_httpx(ast.parse("from httpx import Client\n")) == [1]
    assert sends_with_httpx(ast.parse("import httpx\nisinstance(e, httpx.HTTPError)\n")) == []
