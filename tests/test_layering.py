"""The dependency rule in AGENTS.md, checked on every run.

    scenario -> feature -> component -> domain -> infra

A module may import from its own level or below, never above. Read
statically from each module's import statements (including imports inside
functions), so the check costs nothing and needs no fixtures. Imports made
at run time by a name in the source - `importlib.import_module("...")`,
`__import__("...")`, `sys.modules["..."]`, or an attribute of the
`sar_geo_mcp` package object itself - count as imports too; one whose name
is only known at run time cannot be checked, and is refused.

Two refinements:

- Within the feature and scenario layers, a module may import only the
  layer's private helpers (modules whose name starts with `_`), never
  another feature or scenario - nor the layer's own package, whose
  attributes are every sibling. Logic two features share moves down into
  domain; that is what keeps a feature replaceable on its own.
- `server` and `context` are the wiring. `server` may import anything but
  itself. The tool layers may import `context`, for the `ServerContext`
  every tool function takes first and the `mcp_tool` every wrapper is
  registered with (AGENTS.md, rule 1); nothing below them may,
  and nothing at all, `context` included, may import `server`, which
  imports the layers. `context` itself imports nothing above domain, or
  every tool layer importing it would be importing sideways or up.

KNOWN_EXCEPTIONS lists the places an import is allowed to point up anyway.
It has been empty since 0.27.1, when the last two (infra importing domain)
were fixed. The mechanism stays so that a debt taken on deliberately is
written down rather than hidden: the test fails if a new upward import
appears unlisted, and also if a listed one is fixed but left in the list,
so the list only ever shrinks.

The static read cannot see one kind of import: loading `a.b` runs package
`a`'s `__init__` first, with no import statement in `a.b` to say so. A
fresh interpreter importing every infra module checks that path at run time.
It honours KNOWN_EXCEPTIONS too: what a listed infra import loads, directly
or in turn, is excused there as well.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import sar_geo_mcp

PACKAGE = "sar_geo_mcp"
ROOT = Path(sar_geo_mcp.__file__).parent

#: Plumbing with no SAR knowledge. Everything not listed here, and not a
#: tool layer or the wiring, is domain.
INFRA = {
    "net", "net_errors", "config", "cache", "cache.atomic", "cache.index",
    "cache.evict", "tiles", "geo", "updates",
    "caltopo", "caltopo.session", "caltopo.wms", "caltopo.state", "caltopo.reader",
}
#: Every module of the `sources` package is infra: a data source fetches and
#: caches a service's own data and knows nothing of search and rescue. Added
#: by package rather than by name, so a new source is held to that at once.
SOURCES_PACKAGE = "sources"
INFRA |= {SOURCES_PACKAGE} | {
    f"{SOURCES_PACKAGE}.{path.stem}"
    for path in (ROOT / SOURCES_PACKAGE).glob("*.py") if path.stem != "__init__"}
WIRING = {"server", "context"}
TOOL_LAYERS = {"components": 2, "features": 3, "scenarios": 4}
RANK = {"infra": 0, "domain": 1, "component": 2, "feature": 3, "scenario": 4}
TOOL_LEVELS = {"component", "feature", "scenario"}

#: (importer, imported) pairs allowed to point up the layers anyway. Empty:
#: an entry here is a debt, and AGENTS.md must say how it is to
#: be paid off.
KNOWN_EXCEPTIONS: set[tuple[str, str]] = set()


def module_name(path: Path) -> str:
    parts = list(path.relative_to(ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def level_of(name: str) -> str:
    if name in WIRING:
        return "wiring"
    top = name.split(".")[0]
    if top in TOOL_LAYERS:
        return {"components": "component", "features": "feature",
                "scenarios": "scenario"}[top]
    if name in INFRA:
        return "infra"
    return "domain"


def modules() -> dict[str, Path]:
    return {module_name(path): path for path in ROOT.rglob("*.py")}


def resolve(importer: str, is_package: bool, node: ast.ImportFrom | ast.Import) -> list[str]:
    """The package-relative module names an import statement names."""
    known = modules()
    found = []
    if isinstance(node, ast.Import):
        targets = [alias.name for alias in node.names]
    else:
        if node.level:
            base = importer.split(".") if importer else []
            if not is_package:
                base = base[:-1]
            base = base[:len(base) - (node.level - 1)] if node.level > 1 else base
            stem = ".".join([*base, node.module] if node.module else base)
        else:
            stem = node.module or ""
            if not (stem == PACKAGE or stem.startswith(PACKAGE + ".")):
                return []
            stem = stem[len(PACKAGE) + 1:] if stem != PACKAGE else ""
        # `from .x import y` may name a submodule y, or a name inside x.
        targets = []
        for alias in node.names:
            candidate = f"{stem}.{alias.name}" if stem else alias.name
            targets.append(candidate if candidate in known else stem)
        return [t for t in targets if t in known]
    for target in targets:
        if target.startswith(PACKAGE + "."):
            target = target[len(PACKAGE) + 1:]
            if target in known:
                found.append(target)
    return found


#: Calls that load a module by a name given as a string.
IMPORT_CALLS = {"import_module", "__import__"}


def dynamic_imports(importer: str, is_package: bool, tree: ast.AST,
                    known: dict[str, Path] | None = None) -> list[tuple[int, str | None]]:
    """Modules of ours `tree` reaches with no import statement naming them.

    `importlib.import_module(name)`, `__import__(name)`, `sys.modules[name]`,
    and `getattr(sar_geo_mcp, name)` or `sar_geo_mcp.x.y` on a package bound
    by `import sar_geo_mcp`: each a `(line, module)` pair, the module
    package-relative, or None when the name is not a string in the source
    and so cannot be checked. Names outside the package are not ours and are
    left out; so is the bare package, which `import sar_geo_mcp` also loads.
    """
    known = modules() if known is None else known

    def ours(dotted: str) -> str:
        """The deepest module of ours a package-relative dotted name reaches."""
        parts = [part for part in dotted.split(".") if part]
        for end in range(len(parts), 0, -1):
            candidate = ".".join(parts[:end])
            if candidate in known:
                return candidate
        return ""

    def relative(text: str, level: int) -> str:
        base = importer.split(".") if importer else []
        if not is_package:
            base = base[:-1]
        base = base[:len(base) - (level - 1)] if level > 1 else base
        return ".".join([*base, text])

    def absolute(text: str) -> str | None:
        if text == PACKAGE or text.startswith(PACKAGE + "."):
            return text[len(PACKAGE) + 1:]
        return None

    # Names bound to the package object: `import sar_geo_mcp [as x]`, and
    # `import sar_geo_mcp.y`, which binds `sar_geo_mcp` too.
    bound = {alias.asname or PACKAGE
             for node in ast.walk(tree) if isinstance(node, ast.Import)
             for alias in node.names
             if alias.name == PACKAGE
             or (alias.name.startswith(PACKAGE + ".") and alias.asname is None)}

    found: list[tuple[int, str | None]] = []
    for node in ast.walk(tree):
        dotted: str | None = None
        if isinstance(node, ast.Call):
            func = node.func
            called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if called in IMPORT_CALLS:
                name = node.args[0] if node.args else next(
                    (k.value for k in node.keywords if k.arg == "name"), None)
                if not (isinstance(name, ast.Constant) and isinstance(name.value, str)):
                    found.append((node.lineno, None))
                    continue
                text = name.value
                level = len(text) - len(text.lstrip("."))
                if called == "__import__":
                    given = node.args[4] if len(node.args) > 4 else next(
                        (k.value for k in node.keywords if k.arg == "level"), None)
                    if isinstance(given, ast.Constant) and isinstance(given.value, int):
                        level = max(level, given.value)
                dotted = (relative(text.lstrip("."), level) if level
                          else absolute(text))
            elif (called == "getattr" and len(node.args) >= 2
                  and isinstance(node.args[0], ast.Name) and node.args[0].id in bound):
                attr = node.args[1]
                if not (isinstance(attr, ast.Constant) and isinstance(attr.value, str)):
                    found.append((node.lineno, None))
                    continue
                dotted = attr.value
        elif isinstance(node, ast.Attribute):
            chain = []
            base: ast.AST = node
            while isinstance(base, ast.Attribute):
                chain.append(base.attr)
                base = base.value
            if isinstance(base, ast.Name) and base.id in bound:
                dotted = ".".join(reversed(chain))
        elif (isinstance(node, ast.Subscript)
              and isinstance(node.value, ast.Attribute) and node.value.attr == "modules"
              and isinstance(node.value.value, ast.Name) and node.value.value.id == "sys"
              and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)):
            dotted = absolute(node.slice.value)
        if dotted is None:
            continue
        target = ours(dotted)
        if target:
            found.append((node.lineno, target))
    return found


def imports_of(name: str, is_package: bool, tree: ast.AST,
               known: dict[str, Path] | None = None) -> set[str]:
    """Every module of ours one module's source imports, statically or by a
    name given at run time, itself excluded."""
    targets = {target for node in ast.walk(tree)
               if isinstance(node, (ast.Import, ast.ImportFrom))
               for target in resolve(name, is_package, node)}
    targets |= {target for _, target in dynamic_imports(name, is_package, tree, known)
                if target is not None}
    targets.discard(name)
    return targets


def edges() -> set[tuple[str, str]]:
    result = set()
    known = modules()
    for name, path in known.items():
        is_package = path.name == "__init__.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        result |= {(name, target) for target in imports_of(name, is_package, tree, known)}
    return result


def is_violation(importer: str, imported: str) -> bool:
    """Whether `importer` importing `imported` breaks the dependency rule."""
    source, target = level_of(importer), level_of(imported)
    if imported == "server":
        # Before the wiring's own exemption: `server` imports the layers and
        # `context`, so whatever imports it, `context` included, makes a cycle.
        return importer != "server"
    if source == "wiring":
        # Every tool layer imports `context`. That stays acyclic, and a
        # feature stays unable to reach another feature through it, only
        # while `context` imports nothing above domain.
        return importer == "context" and target in TOOL_LEVELS
    if target == "wiring":
        # A tool function takes the ServerContext first, so the tool layers
        # name its type. `context` imports nothing above domain, so this
        # makes no cycle; `server` imports the layers, so importing it would.
        return not (imported == "context" and source in TOOL_LEVELS)
    if RANK[target] > RANK[source]:
        return True
    # Same tool layer: only the layer's private helpers. The package itself
    # (its `__init__`) imports every module to register them, which is the
    # wiring of that layer. A module importing that package is not exempt:
    # `from .. import features` hands over every sibling as an attribute.
    if source == target and source in ("feature", "scenario"):
        importer_top = importer.split(".")[0]
        if importer == importer_top:
            return False
        return not imported.split(".")[-1].startswith("_")
    return False


def violations() -> set[tuple[str, str]]:
    return {(importer, imported) for importer, imported in edges()
            if is_violation(importer, imported)}


def test_layer_classification_covers_real_modules():
    known = modules()
    assert not (INFRA - set(known)), f"INFRA names modules that do not exist: {INFRA - set(known)}"
    assert {"server", "context", "components", "features", "scenarios"} <= set(known)


def test_imports_point_down_only():
    new = violations() - KNOWN_EXCEPTIONS
    assert new == set(), (
        "imports that point up the layers (see AGENTS.md): "
        + ", ".join(f"{a} -> {b}" for a, b in sorted(new)))


def test_known_exceptions_are_still_real():
    fixed = KNOWN_EXCEPTIONS - violations()
    assert fixed == set(), (
        f"these are fixed - remove them from KNOWN_EXCEPTIONS: {sorted(fixed)}")


def excused_at_run_time() -> set[str]:
    """What importing infra may load because KNOWN_EXCEPTIONS lists it.

    A listed infra -> X import loads X, everything X imports in turn, and
    the parent packages of each (loading `a.b` runs `a`'s `__init__`). So
    the run-time check excuses that whole reach, or listing a debt would
    satisfy the static check and still fail this one."""
    frontier = [imported for importer, imported in KNOWN_EXCEPTIONS
                if level_of(importer) == "infra"]
    if not frontier:
        # The usual case, and reading every module's imports takes seconds.
        return set()
    known = modules()
    graph = edges()
    reach: set[str] = set()
    while frontier:
        name = frontier.pop()
        if name in reach:
            continue
        reach.add(name)
        parts = name.split(".")
        frontier.extend(parent for parent in
                        (".".join(parts[:i]) for i in range(1, len(parts)))
                        if parent in known)
        frontier.extend(target for source, target in graph if source == name)
    return reach


def test_importing_infra_loads_no_domain_module():
    """The run-time half of the check: what importing infra actually loads,
    package `__init__`s included. In a fresh interpreter, because this one
    has long since imported everything."""
    names = sorted(INFRA)
    # The child names the module that failed to import, and why, on stderr
    # itself. Left to Python, the traceback would go wherever the last
    # excepthook installed sends it - caltopo_python's logs it to stdout -
    # and `check=True` would show neither.
    script = (
        "import importlib, json, sys, traceback\n"
        # The tree this file reads statically, not whichever copy of the
        # package happens to be installed.
        f"sys.path.insert(0, {str(ROOT.parent)!r})\n"
        f"for name in {names!r}:\n"
        "    try:\n"
        f"        importlib.import_module({PACKAGE!r} + '.' + name)\n"
        "    except BaseException:\n"
        "        sys.stderr.write(f'importing {name} failed:\\n'\n"
        "                         + traceback.format_exc())\n"
        "        sys.exit(1)\n"
        # Last line, so anything a module prints while loading is ignored.
        "print('\\n' + json.dumps(sorted(sys.modules)))\n"
    )
    proc = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    loaded = json.loads(proc.stdout.strip().splitlines()[-1])
    ours = {name[len(PACKAGE) + 1:] for name in loaded
            if name.startswith(PACKAGE + ".")}
    above = {name for name in ours if level_of(name) != "infra"}
    unlisted = above - excused_at_run_time()
    assert unlisted == set(), (
        f"importing infra loaded modules above it: {sorted(unlisted)}")


def test_the_check_sees_an_upward_import():
    """Guard against the checker silently resolving nothing."""
    assert ("features.boundary", "components.candidates") in edges()
    assert ("server", "components") in edges()
    assert ("components.cache", "cache") in edges()
    assert ("features.lz", "context") in edges()


def test_only_the_tool_layers_may_import_the_context_and_nothing_the_server():
    assert not is_violation("components.ground", "context")
    assert not is_violation("features.lz", "context")
    assert not is_violation("scenarios.anything", "context")
    assert is_violation("planning.lz", "context")
    assert is_violation("cache.evict", "context")
    assert is_violation("components.ground", "server")
    assert is_violation("features.lz", "server")
    # Nothing at all, the other half of the wiring included.
    assert is_violation("context", "server")
    assert is_violation("planning.lz", "server")
    assert not is_violation("server", "context")


def test_a_feature_or_scenario_cannot_reach_a_sibling_through_its_layer_package():
    """`from .. import features` in features/lz.py, then
    `features.egress.egress_routes(...)`, is a feature calling a feature
    (rule 2) without any import statement naming the sibling. The layer's
    package may import its modules - that is how they register - but a
    module may not import the package back."""
    climbing = ast.parse("from .. import features").body[0]
    assert resolve("features.lz", False, climbing) == ["features"]
    assert is_violation("features.lz", "features")

    absolute = ast.parse("from sar_geo_mcp import scenarios").body[0]
    assert resolve("scenarios.medevac", False, absolute) == ["scenarios"]
    assert is_violation("scenarios.medevac", "scenarios")

    # A sibling by name, as before.
    assert is_violation("features.lz", "features.egress")
    assert is_violation("scenarios.medevac", "scenarios.urban_search")
    # Still allowed: the package registering its modules, and the private
    # helpers a layer shares.
    assert not is_violation("features", "features.lz")
    assert not is_violation("scenarios", "scenarios.medevac")
    assert not is_violation("features.urban", "features._sets")
    assert not is_violation("scenarios.medevac", "scenarios._sections")


def test_the_context_imports_nothing_above_domain():
    """Every tool layer may import `context` only because `context` stays
    at domain level (AGENTS.md, "Shared state"). Were it to
    import a feature, every feature importing `context` would reach that
    feature through it."""
    path = ROOT / "context.py"
    imported = imports_of("context", False, ast.parse(path.read_text(encoding="utf-8")))
    assert {"config", "handles"} <= imported, "the check sees none of context's imports"
    above = {name for name in imported if level_of(name) not in ("infra", "domain")}
    assert above == set(), f"context imports above domain: {sorted(above)}"

    assert is_violation("context", "features.lz")
    assert is_violation("context", "components.cache")
    assert is_violation("context", "scenarios._sections")
    assert not is_violation("context", "handles")
    assert not is_violation("context", "caltopo.reader")
    # `server` is the other half of the wiring and may import the layers.
    assert not is_violation("server", "features.lz")


def _imported_by(importer: str, source: str, is_package: bool = False) -> set[str]:
    return imports_of(importer, is_package, ast.parse(source))


def test_the_check_sees_an_import_made_at_run_time():
    """The run-time half of the check sees only what loads at import time,
    and the static half used to read only import statements, so a lazy
    `importlib.import_module` in infra got past both."""
    cases = {
        # (importer, source): what it reaches
        ("cache.evict", "import importlib\n"
                        "def f():\n"
                        "    return importlib.import_module('sar_geo_mcp.staging')\n"): "staging",
        ("cache.evict", "from importlib import import_module\n"
                        "import_module('..staging', __package__)\n"): "staging",
        ("caltopo.session", "__import__('sar_geo_mcp.caltopo.assignment')\n"): "caltopo.assignment",
        ("caltopo.session", "__import__('assignment', globals(), locals(), [], 1)\n"):
            "caltopo.assignment",
        ("net", "import sys\nsys.modules['sar_geo_mcp.planning.lz']\n"): "planning.lz",
        ("cache.evict", "import sar_geo_mcp\n"
                        "sar_geo_mcp.staging.tiles_in_use('.')\n"): "staging",
        ("cache.evict", "import sar_geo_mcp.cache.index\n"
                        "sar_geo_mcp.handles.STAGED_DIRNAME\n"): "handles",
        ("tiles", "import sar_geo_mcp as package\n"
                  "getattr(package, 'staging')\n"): "staging",
    }
    for (importer, source), expected in cases.items():
        assert expected in _imported_by(importer, source), (importer, source)
        assert is_violation(importer, expected), (importer, expected)

    # Not ours, and not an import: nothing reported.
    assert _imported_by("net", "import importlib\nimportlib.import_module('httpx')\n") == set()
    assert _imported_by("net", "getattr(object(), 'staging')\n") == set()


def test_no_module_imports_by_a_name_only_known_at_run_time():
    """`importlib.import_module(name)` with `name` computed cannot be checked
    against the layers at all. Write the module name as a string in the
    source, so the layer it reaches can be read."""
    unreadable = []
    known = modules()
    for name, path in known.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        unreadable += [f"{name}:{line}"
                       for line, target in dynamic_imports(
                           name, path.name == "__init__.py", tree, known)
                       if target is None]
    assert unreadable == [], (
        "imports by a computed name, which the layer check cannot follow: "
        + ", ".join(unreadable))

    computed = ast.parse("import importlib\nname = 'x'\nimportlib.import_module(name)\n")
    assert dynamic_imports("cache.evict", False, computed) == [(3, None)]
