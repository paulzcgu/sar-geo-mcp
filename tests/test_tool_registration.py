"""Every tool, in every layer, must reference only names that exist.

`site_facts` shipped calling `_staged(handle)` when the resolver parameter was
named `staged`. It closed over nothing and raised NameError before reaching
any logic. Nothing caught it: the unit tests call the library function
directly, and the server test only asserts the tool is registered by name.

This walks each tool's bytecode and checks two things resolve: every global
it loads, and every import it makes inside its body (several tools import
lazily, and a relative import that points at the wrong package fails only
when the tool is first called). It is cheap, it needs no fixtures, and it
catches the whole class of typo - a misspelled helper, a name that moved
module, a resolver that was renamed.

Scope: every tool the three layer packages register (components, features,
scenarios), each built here against a `FakeMCP` stand-in. Since the split
into layers (AGENTS.md) every tool is registered through the same
`register(mcp, ctx)` call, so all of them are walked, not just the planning
ones as before.

What is walked is the code that runs. Since 0.27.2 a tool's wrapper only
delegates to a module-level function of the tool's name (rule 1 in
AGENTS.md), so the wrapper alone holds almost nothing to check.
The walk follows every global name the wrapper loads that is a function in
the tool layers - the plain function, and in turn the helpers it calls
(`reusable_set`, `candidate_and_staleness`) - and walks each one's nested
code objects too, since a lambda, nested `def` or generator expression
compiles to its own and its names are loaded there. Functions below the tool
layers are not followed: domain and infra have their own tests, and the
check is about the tool code, where the `site_facts` slip happened.

A global is not the only name that can be misspelt. The plain functions
reach the context as `ctx.staged(handle)` and `ctx.config.cache_dir`: an
attribute load, which the globals walk cannot see, and which only fails
when the tool is called. Before 0.27.2 the same slip was an alias assigned
in `register` (`staged = ctx.stagd`), so it failed in every registration
test. `unknown_context_attributes` puts that back: every `ctx.<name>` the
walked code reads must be something `ServerContext` has, and
`ctx.config.<name>` something `CaltopoConfig` has.

`test_every_tool_is_a_plain_function_first` holds the pattern itself: the
plain function exists, takes `ctx` first and then exactly the wrapper's
parameters, and the wrapper calls it and nothing else.
`test_every_wrapper_hands_each_argument_to_its_own_parameter` checks what
the call passes, since matching signatures do not stop a wrapper from
dropping or swapping an argument. `test_every_wrapper_carries_its_tool_s_name`
checks the name pydantic puts in an argument-validation error.

The layer tags are checked twice: as `register` hands them to `FakeMCP`, and
as a real fastmcp client reads them off the real server, which is what rule
3 is for. The last tests hold the two tool tables a client author reads -
README's and AGENTS.md's "Where each tool lives" - to the tools
registered, their layers and their modules.
"""

import builtins
import dataclasses
import dis
import importlib
import inspect
import re
import sys
import types
import typing
from pathlib import Path

import pytest

from sar_geo_mcp import components, features, scenarios
from sar_geo_mcp.context import ServerContext
from sar_geo_mcp.server import LAYERS

LAYER_PACKAGES = (components, features, scenarios)
#: Module-name prefixes of the tool layers, whose functions the walk follows.
TOOL_PACKAGES = tuple(f"{package.__name__}." for package in LAYER_PACKAGES)


class FakeMCP:
    """Records each tool, under the name it is registered by, and its tags.

    The name is FastMCP's: `name=` when given, else the function's own. A
    wrapper is named `_<tool>_tool` and registered with `name="<tool>"`, so
    honouring `name=` is what keys this by the name a client sees."""

    def __init__(self):
        self.registered = {}
        self.tags = {}

    def tool(self, fn=None, *, name=None, tags=None, **_kwargs):
        def decorate(fn):
            tool_name = name or fn.__name__
            assert tool_name not in self.registered, f"{tool_name} registered twice"
            self.registered[tool_name] = fn
            self.tags[tool_name] = set(tags or ())
            return fn
        return decorate(fn) if fn is not None else decorate


class Config:
    norms_path = None
    cache_dir = None
    mapbox_token = None
    google_maps_key = None

    def wms_base_url(self):
        return None


class FakeContext:
    def __init__(self, staged=lambda handle: None):
        self.config = Config()
        self.reader = None
        self.norms = {}
        self.staged = staged
        self.state = lambda *args, **kwargs: None


def registered(package=None, ctx=None):
    mcp = FakeMCP()
    for layer in ([package] if package else LAYER_PACKAGES):
        layer.register(mcp, ctx or FakeContext())
    return mcp


def code_objects(code):
    """A code object and every one nested in it, at any depth."""
    yield code
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from code_objects(const)


def instructions(fn):
    """The function's instructions, one list per code object: an import's
    level is read off its neighbours, so the lists are kept apart."""
    return [list(dis.get_instructions(code)) for code in code_objects(fn.__code__)]


def loaded_globals(fn):
    return {instruction.argval for block in instructions(fn) for instruction in block
            if instruction.opname == "LOAD_GLOBAL"}


def code_run_by(tool):
    """The tool's wrapper, then every function in the tool layers reached
    from it by a global name, each once, in the order found."""
    found = []
    pending = [tool]
    while pending:
        fn = pending.pop(0)
        if fn in found:
            continue
        found.append(fn)
        for name in sorted(loaded_globals(fn)):
            target = fn.__globals__.get(name)
            if inspect.isfunction(target) and target.__module__.startswith(TOOL_PACKAGES):
                pending.append(target)
    return found


def undefined_globals(fn):
    """Global names the function loads that resolve nowhere."""
    return sorted(name for name in loaded_globals(fn)
                  if name not in fn.__globals__ and not hasattr(builtins, name))


def unresolvable_imports(fn):
    """Imports inside the function body that would fail when it runs."""
    package = fn.__module__.rpartition(".")[0]
    failures = []
    for block in instructions(fn):
        for i, instruction in enumerate(block):
            if instruction.opname != "IMPORT_NAME":
                continue
            # The import's level is the constant loaded two instructions before.
            level = block[i - 2].argval
            name = "." * level + (instruction.argval or "")
            try:
                importlib.import_module(name, package if level else None)
            except ImportError as error:
                failures.append(f"{name}: {error}")
    return failures


#: Instructions that load a variable by name: `ctx` as a parameter of the
#: plain function, or as a free variable in a lambda or nested def inside it.
#: Python 3.12 emits LOAD_FAST or LOAD_DEREF for those; the other two are
#: spellings 3.12 and later versions use elsewhere, listed so the check
#: does not go quiet on a newer interpreter.
LOADS_BY_NAME = {"LOAD_FAST", "LOAD_FAST_CHECK", "LOAD_FAST_BORROW", "LOAD_DEREF"}


def attribute_names(cls) -> set[str]:
    """What an instance of the dataclass `cls` has: its fields, and the
    methods and class attributes it defines."""
    return ({field.name for field in dataclasses.fields(cls)}
            | {name for name in dir(cls) if not name.startswith("_")})


def unknown_context_attributes(fn):
    """`ctx.<name>` reads the function makes that `ServerContext` does not
    have. Where the attribute is itself a dataclass field
    (`ctx.config`), the next attribute along is checked against that
    class as well, so `ctx.config.cach_dir` is caught too."""
    unknown = set()
    for block in instructions(fn):
        for i, instruction in enumerate(block):
            if instruction.opname not in LOADS_BY_NAME or instruction.argval != "ctx":
                continue
            cls, path = ServerContext, "ctx"
            for following in block[i + 1:]:
                if cls is None or following.opname != "LOAD_ATTR":
                    break
                path = f"{path}.{following.argval}"
                if following.argval not in attribute_names(cls):
                    unknown.add(path)
                    break
                field_type = typing.get_type_hints(cls).get(following.argval)
                cls = field_type if dataclasses.is_dataclass(field_type) else None
    return sorted(unknown)


def offenders(check):
    """{"tool: function": what `check` found} over the code every tool runs."""
    found = {}
    for name, tool in registered().registered.items():
        for fn in code_run_by(tool):
            result = check(fn)
            if result:
                found[f"{name}: {fn.__module__}.{fn.__qualname__}"] = result
    return found


def test_server_registers_every_layer_package():
    assert tuple(LAYERS) == LAYER_PACKAGES


def test_every_tool_references_only_defined_names():
    bad = offenders(undefined_globals)
    assert bad == {}, f"tools loading undefined globals: {bad}"


def test_every_tool_imports_only_modules_that_exist():
    bad = offenders(unresolvable_imports)
    assert bad == {}, f"tools importing modules that do not exist: {bad}"


def test_every_tool_reads_only_context_attributes_that_exist():
    bad = offenders(unknown_context_attributes)
    assert bad == {}, f"tools reading context attributes that do not exist: {bad}"


def test_the_context_check_sees_a_misspelt_attribute():
    """Guard against the check reading nothing: one slip of each kind, one of
    them in a lambda, beside reads that are right."""
    def planted(ctx, handle):
        later = lambda: ctx.nrms  # noqa: E731
        return (ctx.stagd(handle), ctx.confg, ctx.config.cach_dir, later,
                ctx.staged(handle).directory, ctx.config.cache_dir,
                ctx.config.wms_base_url(), ctx.norms, ctx.state, ctx.reader)

    assert set(unknown_context_attributes(planted)) == {
        "ctx.stagd", "ctx.confg", "ctx.config.cach_dir", "ctx.nrms"}


def test_the_walk_reaches_the_plain_function_and_its_helpers():
    """Guard against the walk stopping at the wrapper, which now holds one
    call: the checks above would pass while walking almost nothing."""
    from sar_geo_mcp.components.candidates import candidate_and_staleness
    from sar_geo_mcp.features import _sets, boundary, hasty

    tools = registered().registered
    hasty_run = code_run_by(tools["hasty_candidates"])
    assert hasty.hasty_candidates in hasty_run
    assert _sets.reusable_set in hasty_run and _sets.set_map_version in hasty_run
    assert candidate_and_staleness in code_run_by(tools["describe_boundary"])
    assert boundary.describe_boundary in code_run_by(tools["describe_boundary"])


def test_every_tool_is_a_plain_function_first():
    """Rule 1 in AGENTS.md, for every tool in every layer: the
    tool's module has a function of the tool's own name taking `ctx` and
    then exactly the wrapper's parameters - names, order, annotations and
    defaults - returning the same type; and the wrapper calls it and loads
    no other global, so nothing a scenario would miss by calling the
    function lives in the wrapper. A wrapper left with the tool's own name
    would shadow the function inside `register` and load nothing global, so
    that is caught here too."""
    problems = {name: problem for name, tool in registered().registered.items()
                if (problem := shape_problem(name, tool))}
    assert problems == {}, problems


def shape_problem(name, tool):
    """What is wrong with the shape of the tool `name`'s wrapper and plain
    function, or None."""
    module = sys.modules[tool.__module__]
    plain = getattr(module, name, None)
    if not inspect.isfunction(plain) or plain.__module__ != module.__name__:
        return f"no module-level function {name} in {module.__name__}"
    signature = inspect.signature(plain)
    parameters = list(signature.parameters.values())
    if not parameters or parameters[0].name != "ctx":
        return f"{name} does not take ctx first: {signature}"
    expected = signature.replace(parameters=parameters[1:])
    if inspect.signature(tool) != expected:
        return (f"the wrapper's signature {inspect.signature(tool)} "
                f"is not the function's after ctx, {expected}")
    loads = loaded_globals(tool)
    if not loads:
        # The name in the wrapper's body resolved without a global: to the
        # wrapper itself, through `register`'s scope, so a call would recurse.
        return (f"the wrapper does not reach the module-level function {name}: "
                f"it loads no global, most likely because it is itself named "
                f"{name} and shadows it; name it _{name}_tool and register it "
                f"with name={name!r}")
    if loads != {name}:
        return f"the wrapper does more than call {name}: it loads {sorted(loads)}"
    return None


class Sentinel:
    """A value that can only have come from one parameter, so a recorder can
    tell which argument reached which parameter."""

    def __init__(self, parameter):
        self.parameter = parameter

    def __repr__(self):
        return f"<the value passed as {self.parameter}>"


def forwarding_problem(name, tool, ctx, monkeypatch):
    """Call the wrapper with a distinct value per parameter, with the
    module's plain function `name` replaced by a recorder, and say what did
    not arrive where it should, or None. The wrapper looks the function up
    as a global when it is called, which is what lets the recorder in. The
    real function is put back before this returns."""
    received = []
    sent = {parameter: Sentinel(parameter) for parameter in inspect.signature(tool).parameters}
    with monkeypatch.context() as patch:
        patch.setattr(sys.modules[tool.__module__], name,
                      lambda *args, **kwargs: received.append((args, kwargs)))
        try:
            tool(**sent)
        except Exception as error:
            # A wrapper that shadows its function calls itself and fails on
            # the arguments; reported per tool, so the rest are still checked.
            return f"calling the wrapper raised {error!r}"
    if len(received) != 1:
        return f"the wrapper called {name} {len(received)} times, not once"
    args, kwargs = received[0]
    if len(args) != 1 or args[0] is not ctx:
        return f"the wrapper passed {args!r} positionally, not the ctx it was registered with"
    wrong = {parameter: kwargs.get(parameter, "<not passed>")
             for parameter, value in sent.items() if kwargs.get(parameter) is not value}
    extra = sorted(set(kwargs) - set(sent))
    if wrong or extra:
        return f"not passed through by keyword as given: {wrong}; unexpected: {extra}"
    return None


def test_every_wrapper_hands_each_argument_to_its_own_parameter(monkeypatch):
    """Matching signatures do not make a wrapper pass what it was given: one
    that drops `limit=limit` hands the plain function its default, and the
    planner's 3 silently becomes 10. The answer then differs between an MCP
    caller and a scenario, and only the MCP side is wrong."""
    ctx = FakeContext()
    problems = {name: problem for name, tool in registered(ctx=ctx).registered.items()
                if (problem := forwarding_problem(name, tool, ctx, monkeypatch))}
    assert problems == {}, problems


def test_every_wrapper_carries_its_tool_s_name():
    """FastMCP builds a tool's argument validator from the wrapper, and
    pydantic titles every validation error with the function's `__name__`.
    A wrapper left as `_<tool>_tool` gives a caller "validation error for
    call[_lz_candidates_tool]", which names a tool that does not exist;
    `context.mcp_tool` renames it. A tool registered with a bare
    `mcp.tool(name=...)` would not be, and fails here."""
    wrong = {name: tool.__name__ for name, tool in registered().registered.items()
             if tool.__name__ != name}
    assert wrong == {}, f"wrappers not named for their tool: {wrong}"


def _planted(ctx, a: int = 1, b: str = "x") -> int:
    """Stands in for a tool's plain function in the planted cases below."""
    return a


def _forwards_everything(ctx):
    def _planted_tool(a: int = 1, b: str = "x") -> int:
        return _planted(ctx, a=a, b=b)
    return _planted_tool


def _drops_b(ctx):
    def _planted_tool(a: int = 1, b: str = "x") -> int:
        return _planted(ctx, a=a)
    return _planted_tool


def _swaps_a_and_b(ctx):
    def _planted_tool(a: int = 1, b: str = "x") -> int:
        return _planted(ctx, a=b, b=a)
    return _planted_tool


def _shadows(ctx):
    def _planted(a: int = 1, b: str = "x") -> int:
        return _planted(ctx, a=a, b=b)
    return _planted


def test_the_shape_checks_see_planted_defects(monkeypatch):
    """Guard against the two checks above passing everything: a wrapper
    that forwards all, one that drops an argument, one that swaps two, and
    one that shadows its function."""
    ctx = FakeContext()
    assert shape_problem("_planted", _forwards_everything(ctx)) is None
    assert forwarding_problem("_planted", _forwards_everything(ctx), ctx, monkeypatch) is None

    dropped = forwarding_problem("_planted", _drops_b(ctx), ctx, monkeypatch)
    assert dropped and "'b': '<not passed>'" in dropped and "'a'" not in dropped
    swapped = forwarding_problem("_planted", _swaps_a_and_b(ctx), ctx, monkeypatch)
    assert swapped and "'a': <the value passed as b>" in swapped

    shadowing = shape_problem("_planted", _shadows(ctx))
    assert shadowing and "shadows it" in shadowing


def test_every_tool_is_tagged_with_its_own_layer_only():
    for package in LAYER_PACKAGES:
        mcp = registered(package)
        wrong = {name: tags for name, tags in mcp.tags.items() if tags != {package.LAYER}}
        assert wrong == {}, f"{package.__name__}: tools tagged other than {package.LAYER!r}: {wrong}"


def layer_of_every_tool():
    """{tool name: its layer}, from each layer package's own registration."""
    return {name: package.LAYER for package in LAYER_PACKAGES
            for name in registered(package).registered}


@pytest.mark.anyio
async def test_a_client_sees_every_tool_tagged_with_its_own_layer_only(tmp_path):
    """Rule 3 in AGENTS.md is there so a client can list one
    layer, and a client reads the tags FastMCP puts in each tool's meta,
    not the set `register` passed in. Through the real server and fastmcp's
    in-memory client: every tool is there, its layer its only tag."""
    pytest.importorskip("fastmcp")
    from fastmcp import Client

    from sar_geo_mcp.config import CaltopoConfig
    from sar_geo_mcp.server import build_server

    async with Client(build_server(CaltopoConfig(cache_dir=tmp_path / "cache"))) as client:
        tools = await client.list_tools()
    seen = {tool.name: ((tool.meta or {}).get("fastmcp") or {}).get("tags") for tool in tools}
    expected = {name: [layer] for name, layer in layer_of_every_tool().items()}
    assert expected and seen == expected


ROOT = Path(__file__).resolve().parents[1]


def markdown_section(path, heading):
    """The lines of `path` under `## heading`, up to the next `## `."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = lines.index(f"## {heading}") + 1
    end = next((i for i in range(start, len(lines)) if lines[i].startswith("## ")), len(lines))
    return lines[start:end]


_UNITS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
          "fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy",
         8: "eighty", 9: "ninety"}


def in_words(number):
    """1 to 99 as README writes a count: "forty-one"."""
    if number < 20:
        return _UNITS[number]
    tens, units = divmod(number, 10)
    return _TENS[tens] + (f"-{_UNITS[units]}" if units else "")


def test_in_words_spells_a_count_as_readme_does():
    assert [in_words(n) for n in (7, 19, 20, 41, 99)] == [
        "seven", "nineteen", "twenty", "forty-one", "ninety-nine"]


def test_readme_s_tool_table_lists_every_tool_once_scenarios_last():
    """README's table is where a client author meets the tools, and its
    first lines count them. A tool added without a row, a row left for a
    tool renamed away, or a count not moved, and nothing else would notice.
    The prose above the table puts the scenarios last in it."""
    rows = [match.group(1) for line in markdown_section(ROOT / "README.md", "The tools")
            if (match := re.match(r"\| `(\w+)` \|", line))]
    layers = layer_of_every_tool()
    assert len(rows) == len(set(rows)), sorted({r for r in rows if rows.count(r) > 1})
    assert set(rows) == set(layers), (f"no row: {sorted(set(layers) - set(rows))}; "
                                      f"no such tool: {sorted(set(rows) - set(layers))}")
    scenarios_ = [name for name, layer in layers.items() if layer == "scenario"]
    assert scenarios_ and set(rows[-len(scenarios_):]) == set(scenarios_)

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert re.search(rf"\b{in_words(len(layers))} tools\b", readme, re.IGNORECASE), (
        f"README does not say {in_words(len(layers))} tools")


def test_architecture_says_where_each_tool_lives_and_in_which_layer():
    """AGENTS.md's "Where each tool lives" names, row by row, a
    layer, a module and its tools: each tool must be registered from that
    module, tagged with that layer, and listed once, and every tool must be
    in the table."""
    rows = [line for line in markdown_section(ROOT / "AGENTS.md",
                                              "Where each tool lives")
            if line.startswith("| ") and "`" in line]
    tools = registered()
    listed = []
    wrong = {}
    for row in rows:
        layer, module, names = (cell.strip() for cell in row.strip("|").split("|"))
        module_name = "sar_geo_mcp." + module.strip("`").removesuffix(".py").replace("/", ".")
        for name in re.findall(r"`(\w+)`", names):
            listed.append(name)
            if name not in tools.registered:
                wrong[name] = "not a registered tool"
            elif tools.tags[name] != {layer}:
                wrong[name] = f"listed as {layer}, tagged {sorted(tools.tags[name])}"
            elif tools.registered[name].__module__ != module_name:
                wrong[name] = (f"listed in {module_name}, registered from "
                               f"{tools.registered[name].__module__}")
    assert wrong == {}, wrong
    assert len(listed) == len(set(listed)), sorted({n for n in listed if listed.count(n) > 1})
    assert set(listed) == set(tools.registered), sorted(set(tools.registered) - set(listed))


def test_unmarked_source_enum_matches_the_sources_the_feature_reads():
    from sar_geo_mcp.features.unmarked_paths import UnmarkedSource
    from sar_geo_mcp.planning.unmarked_paths import ALL_SOURCES

    assert set(typing.get_args(UnmarkedSource)) == set(ALL_SOURCES)


def test_site_facts_reaches_its_resolver():
    """The specific regression: the resolver is called, not a NameError raised."""
    reached = []
    mcp = registered(components, FakeContext(
        staged=lambda handle: reached.append(handle) or _stop()))
    with_handle = mcp.registered["site_facts"]
    try:
        with_handle("aoi-test", {"type": "Polygon",
                                 "coordinates": [[[0, 0], [0, 1], [1, 1], [0, 0]]]})
    except _Stop:
        pass
    assert reached == ["aoi-test"]


class _Stop(Exception):
    """Raised by the stub resolver so the tool stops before touching disk."""


def _stop():
    raise _Stop
