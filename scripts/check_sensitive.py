"""Refuse real search data in commits, PRs and release notes.

    python3 scripts/check_sensitive.py --staged            # pre-commit hook
    python3 scripts/check_sensitive.py --message FILE      # commit-msg hook
    python3 scripts/check_sensitive.py --range BASE..HEAD  # PR, release
    python3 scripts/check_sensitive.py --text FILE|-       # a PR body, notes
    python3 scripts/check_sensitive.py --tree              # every tracked file
    python3 scripts/check_sensitive.py --history           # every commit, ever
    python3 scripts/check_sensitive.py --push REMOTE       # pre-push hook

This repository is public. Real search data has no place in it: a subject's
name, a team member's name, an incident number, a real CalTopo map id, or
where someone was last seen or found. Two sets of rules find it.

Generic rules, kept here, run everywhere:
- a coordinate pair (decimal degrees, 3 or more decimals) more than
  FIXTURE_RADIUS_M from every point in tests/fixture_points.py;
- an MGRS or UTM grid reference not listed there;
- a CalTopo map id (caltopo.com/m/..., map_id="...") not listed there;
- a data file: GPX, KML, GeoJSON, GeoTIFF, a staged-ground manifest, the
  cache, exports or docs folders, credentials, a denylist.
A line that carries `fixture-ok` is excused from the coordinate, grid and
map-id rules. Nothing excuses a file rule or a denylist hit.

A private denylist, kept outside the repository, names what is real:
~/.config/sar_geo/denylist.txt, or the file SAR_SENSITIVE_DENYLIST names
(CI writes it from a repository secret). One entry per line:

    # comment
    point 37.1234 -122.5678 500    a real point: lat lon [radius m, 500]  fixture-ok
    map AB12C                      a real map id
    re  \\bTG ?12\\d\\d\\b         a regular expression
    Jane Doe                       anything else: a word or phrase

Terms match whole words, ignoring case. Findings never print the matched
text, only where it is - CI logs on a public repository are public too.
Exit status 1 when anything is found.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixture_points.py"
DEFAULT_DENYLIST = Path.home() / ".config" / "sar_geo" / "denylist.txt"
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
PRAGMA = "fixture-ok"
DEFAULT_POINT_RADIUS_M = 500.0

# A decimal number with at least three decimals (about 100 m of latitude).
NUMBER = re.compile(r"(?<![\w.])-?\d{1,3}\.\d{3,}(?![\d])")
# Two numbers this far apart or less, with no other number between, pair up.
PAIR_GAP = 60
MGRS = re.compile(
    r"\b(\d{1,2}[C-HJ-NP-X])\s?([A-HJ-NP-Z]{2})\s?(\d{3,5})\s?(\d{3,5})\b")
UTM = re.compile(r"\b(\d{1,2}[C-HJ-NP-X]?)\s+(\d{6})(?:\.\d+)?\s*m?E?\s+"
                 r"(\d{7})(?:\.\d+)?\s*m?N?\b")
MAP_ID = [
    re.compile(r"(?:caltopo|sartopo)\.com/m/([A-Za-z0-9]+)", re.I),
    re.compile(r"\bmap[_-]?id[\"']?\s*[:=]\s*[\"']([A-Za-z0-9]{1,16})[\"']",
               re.I),
]
DATA_SUFFIXES = {".gpx", ".kml", ".kmz", ".geojson", ".tif", ".tiff", ".shp",
                 ".fgb", ".mbtiles", ".gpkg"}
KEY_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".jks", ".keystore", ".ppk"}
KEY_NAMES = {"id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", ".pypirc", ".netrc",
             ".npmrc", "credentials.json", "client_secret.json", ".git-credentials"}
TOP_LEVEL_FOLDERS = {"cache", "docs", "exports"}

# Credential formats with a fixed shape, by who issues them.
TOKEN_FORMATS = [
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{50,})")),
    ("PyPI token", re.compile(r"\bpypi-AgE[A-Za-z0-9_-]{50,}")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("Mapbox token", re.compile(r"\b[ps]k\.eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("Anthropic key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}")),
    ("OpenAI key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}")),
    ("JSON web token", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    ("password in a URL", re.compile(
        r"\b[a-z][a-z0-9+.-]*://[^/\s:@'\"]+:[^/\s@'\"]{3,}@"
        # not a host reserved for examples and tests (RFC 2606, 6761)
        r"(?![^/\s'\"]*(?:\.example|\.test|\.invalid|\.local|localhost)\b)"
        r"(?!host\b)[^/\s'\"]+")),
]
# A secret-sounding name given a long literal value: token = "...".
SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b[\w-]*(?:token|secret|passw(?:or)?d|api[_-]?key|access[_-]?key|"
    r"private[_-]?key|auth[_-]?key|client[_-]?secret|caltopo[_-]?(?:id|key))[\w-]*"
    r"[\"']?\s*[:=]\s*[\"']?([A-Za-z0-9_\-+/=.]{16,})")
# cts.ini's own shape: key = ..., id = ..., accountId = ...
INI_SECRET = re.compile(r"(?i)^\s*(?:key|id|accountid)\s*=\s*([A-Za-z0-9_\-+/=.]{12,})\s*$")
PLACEHOLDER = re.compile(
    r"(?i)example|your|xxxx|test|fake|dummy|sample|placeholder|redacted|"
    r"secret|changeme|\.\.\.|^[a-z_]+$|^[A-Z_]+$")
# The local files and variables that hold this project's real credentials.
SECRET_FILES = ("cts.ini", ".env")
SECRET_VARS = re.compile(r"(?i)(token|key|secret|password|caltopo_id|account_id|"
                         r"caltopo_account)$")
SECRET_MIN_LEN = 8


@dataclass(frozen=True)
class Finding:
    where: str  # "path:line", "commit 1a2b3c4 message line 2", ...
    rule: str

    def __str__(self) -> str:
        return f"{self.where}: {self.rule}"


@dataclass
class Fixtures:
    points: List[Tuple[float, float]]
    radius_m: float
    grids: set
    map_ids: set
    files: set


@dataclass
class DenyPoint:
    lat: float
    lon: float
    radius_m: float
    line: int


@dataclass
class Denylist:
    path: Optional[Path]
    terms: List[Tuple[re.Pattern, int]]
    points: List[DenyPoint]
    # Real credential values read from this machine: (value, where from).
    secrets: List[Tuple[str, str]] = None

    def __post_init__(self) -> None:
        if self.secrets is None:
            self.secrets = []


# ---------------------------------------------------------------- loading


def load_fixtures(path: Path = FIXTURES) -> Fixtures:
    scope: dict = {}
    if path.exists():
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), scope)
    return Fixtures(
        points=[(float(a), float(b)) for a, b in scope.get("FIXTURE_POINTS", [])],
        radius_m=float(scope.get("FIXTURE_RADIUS_M", 250.0)),
        grids={_grid_key(g) for g in scope.get("FIXTURE_GRIDS", [])},
        map_ids={str(m) for m in scope.get("FIXTURE_MAP_IDS", [])},
        files={str(f) for f in scope.get("FIXTURE_FILES", [])},
    )


def denylist_path(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get("SAR_SENSITIVE_DENYLIST")
    return Path(env).expanduser() if env else DEFAULT_DENYLIST


def load_denylist(path: Optional[Path]) -> Denylist:
    deny = Denylist(path=None, terms=[], points=[])
    if path is None or not path.is_file():
        return deny
    deny.path = path
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        head, _, rest = line.partition(" ")
        rest = rest.strip()
        if head == "point":
            parts = rest.replace(",", " ").split()
            try:
                lat, lon = float(parts[0]), float(parts[1])
                radius = float(parts[2]) if len(parts) > 2 else DEFAULT_POINT_RADIUS_M
            except (IndexError, ValueError):
                raise SystemExit(f"denylist line {number}: point needs lat lon [radius_m]")
            deny.points.append(DenyPoint(lat, lon, radius, number))
        elif head == "re":
            try:
                deny.terms.append((re.compile(rest, re.I), number))
            except re.error as exc:
                raise SystemExit(f"denylist line {number}: bad regular expression ({exc})")
        else:
            term = rest if head == "map" else line
            words = r"\s+".join(re.escape(w) for w in term.split())
            deny.terms.append((re.compile(rf"(?<!\w){words}(?!\w)", re.I), number))
    return deny


def local_secrets(root: Path = ROOT, environ=None) -> List[Tuple[str, str]]:
    """The credential values this machine holds for the project: every value
    in cts.ini (and its section names, which are CalTopo account names), the
    secret-named variables in .env, and the same names in the environment.
    They are matched exactly, wherever they appear, and never printed."""
    environ = os.environ if environ is None else environ
    found: dict = {}

    def keep(value: str, source: str) -> None:
        value = value.strip().strip("\"'")
        if len(value) >= SECRET_MIN_LEN:
            found.setdefault(value, source)

    ini = Path(environ.get("ZA_CALTOPO_CONFIG_PATH") or root / "cts.ini")
    if not ini.is_absolute():
        ini = root / ini
    for path in {ini, root / "cts.ini"}:
        if path.is_file():
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if line.startswith("[") and line.endswith("]"):
                    keep(line[1:-1], f"{path.name} account name")
                elif "=" in line and not line.startswith((";", "#")):
                    name, _, value = line.partition("=")
                    keep(value, f"{path.name} {name.strip()}")
    env_file = root / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            name, sep, value = line.strip().partition("=")
            if sep and not name.startswith("#") and SECRET_VARS.search(name):
                keep(value, f".env {name}")
    for name, value in environ.items():
        if name.startswith("ZA_") and SECRET_VARS.search(name):
            keep(value, f"environment {name}")
    return sorted(found.items(), key=lambda item: -len(item[0]))


def _entropy(text: str) -> float:
    counts = {c: text.count(c) for c in set(text)}
    return -sum(n / len(text) * math.log2(n / len(text)) for n in counts.values())


def _looks_secret(value: str) -> bool:
    if PLACEHOLDER.search(value):
        return False
    classes = sum(bool(re.search(p, value)) for p in (r"[a-z]", r"[A-Z]", r"\d"))
    return classes >= 2 and _entropy(value) >= 3.5


def credential_rules(line: str, deny: Denylist, excused: bool) -> List[str]:
    """What in one line looks like a credential. A real value from this
    machine is refused whatever the line says; a format or a secret-named
    assignment is excused by `fixture-ok`, for a test's fake token."""
    rules = [f"a credential from this machine ({source}) - rotate it if it "
             f"was ever pushed" for value, source in deny.secrets if value in line]
    if excused:
        return rules
    rules += [f"looks like a {name}" for name, pattern in TOKEN_FORMATS
              if pattern.search(line)]
    for pattern in (SECRET_ASSIGNMENT, INI_SECRET):
        m = pattern.search(line)
        if m and _looks_secret(m.group(1)):
            rules.append("a secret-named setting with what looks like a real value")
    return rules


# ---------------------------------------------------------------- matching


def _grid_key(text: str) -> str:
    return re.sub(r"[\s.]", "", text).upper()


def metres_between(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6_371_000 * math.asin(min(1.0, math.sqrt(h)))


def _is_lat_lon(a: float, b: float) -> bool:
    return abs(a) <= 90 and abs(b) <= 180


def coordinate_pairs(text: str) -> Iterable[Tuple[int, int, float, float]]:
    """(offset of a, offset of b, a, b) for each pair of neighbouring numbers that could be
    a position, in either order. A pair is consumed whole, so a GeoJSON
    ring's [lon, lat], [lon, lat] reads as positions, not as lat of one
    beside lon of the next."""
    numbers = [(m.start(), m.end(), float(m.group())) for m in NUMBER.finditer(text)]
    i = 0
    while i + 1 < len(numbers):
        (s1, e1, a), (s2, _, b) = numbers[i], numbers[i + 1]
        if s2 - e1 <= PAIR_GAP and (_is_lat_lon(a, b) or _is_lat_lon(b, a)):
            yield s1, s2, a, b
            i += 2
        else:
            i += 1


def _positions(a: float, b: float) -> List[Tuple[float, float]]:
    return [p for p in ((a, b), (b, a)) if _is_lat_lon(*p)]


def _near_any(pos: Tuple[float, float], points, radius_m: float) -> bool:
    return any(metres_between(pos, p) <= radius_m for p in points)


def scan_block(lines: Sequence[Tuple[str, str]], fixtures: Fixtures,
               deny: Denylist) -> List[Finding]:
    """Scan consecutive lines, given as (where, text). Numbers pair across
    a line break, so a latitude and longitude on two lines are a pair."""
    findings: List[Finding] = []
    text = "\n".join(t for _, t in lines)
    starts, offset = [], 0
    for _, t in lines:
        starts.append(offset)
        offset += len(t) + 1

    def line_at(pos: int) -> int:
        lo, hi = 0, len(starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if starts[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def add(index: int, rule: str) -> None:
        f = Finding(lines[index][0], rule)
        if f not in findings:
            findings.append(f)

    def pragma(index: int) -> bool:
        return PRAGMA in lines[index][1]

    for index, (_, line) in enumerate(lines):
        for pattern, number in deny.terms:
            if pattern.search(line):
                add(index, f"matches private denylist entry (denylist line {number})")
        for rule in credential_rules(line, deny, pragma(index)):
            add(index, rule)
        if pragma(index):
            continue
        for pattern in MAP_ID:
            for m in pattern.finditer(line):
                if m.group(1) not in fixtures.map_ids:
                    add(index, "CalTopo map id not in tests/fixture_points.py "
                               "FIXTURE_MAP_IDS")
        for pattern in (MGRS, UTM):
            for m in pattern.finditer(line):
                if pattern is MGRS and len(m.group(3)) != len(m.group(4)):
                    continue
                if _grid_key(m.group()) not in fixtures.grids:
                    add(index, "grid reference not in tests/fixture_points.py "
                               "FIXTURE_GRIDS")

    for pos, pos_b, a, b in coordinate_pairs(text):
        index = line_at(pos)
        candidates = _positions(a, b)
        for p in deny.points:
            if any(metres_between(c, (p.lat, p.lon)) <= p.radius_m for c in candidates):
                add(index, f"position within a private denylist point "
                           f"(denylist line {p.line})")
        if pragma(index) or pragma(line_at(pos_b)):
            continue
        if max(abs(a), abs(b)) < 10:
            continue  # small numbers beside each other are rarely a place
        if not any(_near_any(c, fixtures.points, fixtures.radius_m) for c in candidates):
            add(index, f"coordinate more than {fixtures.radius_m:g} m from every "
                       f"point in tests/fixture_points.py (mark a made-up one "
                       f"`{PRAGMA}`)")
    return findings


def file_rule(path: str, fixtures: Fixtures) -> Optional[str]:
    if path in fixtures.files:
        return None
    p = Path(path)
    name = p.name.lower()
    if p.parts and p.parts[0] in TOP_LEVEL_FOLDERS:
        return f"files under {p.parts[0]}/ are machine-local, never source"
    if "exports" in p.parts[:-1]:
        return "an exports folder holds files made from a search"
    if p.suffix.lower() in DATA_SUFFIXES:
        return f"{p.suffix} files carry tracks, areas or terrain from a map"
    if name == "cts.ini" or (name.startswith(".env") and name != ".env.example"):
        return "credentials are never committed"
    if (p.suffix.lower() in KEY_SUFFIXES or name in KEY_NAMES
            or name.startswith(("service-account", "id_rsa", "id_ed25519"))):
        return "key and credential files are never committed"
    if name == "manifest.json" and path != "extension/manifest.json":
        return "a staged-ground manifest names its map and assignment"
    if "denylist" in name:
        return "the denylist lists what is real; it stays outside the repository"
    return None


# ---------------------------------------------------------------- sources


def git(*args: str) -> str:
    return subprocess.run(("git",) + args, cwd=ROOT, check=True, text=True,
                          capture_output=True, encoding="utf-8",
                          errors="replace").stdout


def scan_diff(diff: str, fixtures: Fixtures, deny: Denylist) -> List[Finding]:
    """The lines a unified diff (-U0) adds, scanned in runs of consecutive
    lines, and the paths it adds or changes."""
    findings: List[Finding] = []
    path = None
    block: List[Tuple[str, str]] = []
    new_line = 0

    def flush() -> None:
        if block:
            findings.extend(scan_block(block, fixtures, deny))
            block.clear()

    for raw in diff.splitlines():
        if raw.startswith("diff --git"):
            flush()
            path = None
        elif raw.startswith("+++ "):
            target = raw[4:]
            path = None if target == "/dev/null" else target[2:] if target.startswith("b/") else target
            if path:
                rule = file_rule(path, fixtures)
                if rule:
                    findings.append(Finding(path, rule))
        elif raw.startswith("@@"):
            flush()
            m = re.search(r"\+(\d+)", raw)
            new_line = int(m.group(1)) if m else 0
        elif raw.startswith("+") and path:
            block.append((f"{path}:{new_line}", raw[1:]))
            new_line += 1
        elif raw.startswith("Binary files") and path is None:
            m = re.search(r" and b/(.+) differ$", raw)
            if m:
                rule = file_rule(m.group(1), fixtures)
                if rule:
                    findings.append(Finding(m.group(1), rule))
        else:
            flush()
    flush()
    return findings


def scan_text(text: str, label: str, fixtures: Fixtures, deny: Denylist) -> List[Finding]:
    lines = [(f"{label} line {n}", t) for n, t in enumerate(text.splitlines(), 1)]
    return scan_block(lines, fixtures, deny)


def commit_message_text(raw: str) -> str:
    """A message as git keeps it: comment lines and the scissors cut gone."""
    kept = []
    for line in raw.splitlines():
        if line.startswith("# ------------------------ >8 ------------------------"):
            break
        if not line.startswith("#"):
            kept.append(line)
    return "\n".join(kept)


DIFF = ("diff", "-U0", "--no-color", "--no-ext-diff", "--diff-filter=ACMR")


def scan_staged(fixtures: Fixtures, deny: Denylist) -> List[Finding]:
    return scan_diff(git(*DIFF, "--cached"), fixtures, deny)


def scan_range(spec: str, fixtures: Fixtures, deny: Denylist) -> List[Finding]:
    base, _, head = spec.replace("...", "..").partition("..")
    head = head or "HEAD"
    findings = scan_diff(git(*DIFF, f"{base}...{head}"), fixtures, deny)
    log = git("log", "--format=%h%x00%B%x1e", f"{base}..{head}")
    for entry in log.split("\x1e"):
        sha, _, body = entry.strip("\n").partition("\x00")
        if sha:
            findings.extend(scan_text(body, f"commit {sha} message", fixtures, deny))
    return findings


def scan_tree(fixtures: Fixtures, deny: Denylist) -> List[Finding]:
    # Against the working copy: what is tracked, as it stands now.
    return scan_diff(git(*DIFF, EMPTY_TREE), fixtures, deny)


def scan_history(fixtures: Fixtures, deny: Denylist) -> List[Finding]:
    """Every commit on every branch and tag: what each one added, and its
    message. Once something is pushed it is public even after a later
    commit removes it, so this is the check that says whether it was."""
    return scan_commits(git("rev-list", "--all").split(), fixtures, deny)


def scan_commits(commits: Sequence[str], fixtures: Fixtures,
                 deny: Denylist) -> List[Finding]:
    findings: List[Finding] = []
    for sha in commits:
        short = sha[:7]
        parent = git("rev-list", "--parents", "-n", "1", sha).split()[1:2]
        base = parent[0] if parent else EMPTY_TREE
        for f in scan_diff(git(*DIFF, base, sha), fixtures, deny):
            findings.append(Finding(f"commit {short} {f.where}", f.rule))
        body = git("log", "-1", "--format=%B", sha)
        findings.extend(scan_text(body, f"commit {short} message", fixtures, deny))
    return findings


def scan_push(remote: str, updates: str, fixtures: Fixtures,
              deny: Denylist) -> List[Finding]:
    """What a push would send: every commit on the refs being pushed (git's
    pre-push lines on stdin) that the remote does not already have. A new
    branch sends its whole history, so an old local branch pushed to the
    public remote is checked commit by commit."""
    commits: List[str] = []
    for line in updates.splitlines():
        parts = line.split()
        if len(parts) != 4 or set(parts[1]) == {"0"}:
            continue  # a deletion sends nothing
        new = git("rev-list", parts[1], "--not", f"--remotes={remote}").split()
        commits += [c for c in new if c not in commits]
    return scan_commits(commits, fixtures, deny)


# ---------------------------------------------------------------- main


def report(findings: List[Finding], deny: Denylist, quiet: bool) -> int:
    if deny.path is None and not quiet:
        print("check_sensitive: no private denylist found; generic rules only "
              f"(set SAR_SENSITIVE_DENYLIST or create {DEFAULT_DENYLIST})",
              file=sys.stderr)
    if not findings:
        return 0
    print("check_sensitive: this looks like real search data or a credential, "
          "which this public repository must not hold:", file=sys.stderr)
    for f in findings:
        print(f"  {f}", file=sys.stderr)
    print("Replace it with a made-up value. A made-up coordinate or map id goes "
          "in tests/fixture_points.py, or its line is marked `fixture-ok`.",
          file=sys.stderr)
    return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--staged", action="store_true", help="the staged diff")
    source.add_argument("--message", metavar="FILE", help="a commit message file")
    source.add_argument("--range", metavar="BASE..HEAD",
                        help="the diff and commit messages of a range")
    source.add_argument("--text", metavar="FILE", help="any text; - for stdin")
    source.add_argument("--tree", action="store_true", help="every tracked file")
    source.add_argument("--push", metavar="REMOTE",
                        help="the commits a push would send (pre-push hook)")
    source.add_argument("--history", action="store_true",
                        help="every commit on every branch, and its message")
    parser.add_argument("--label", default="text", help="name --text in findings")
    parser.add_argument("--denylist", help="the private denylist file")
    parser.add_argument("--quiet", action="store_true",
                        help="say nothing when no denylist is found")
    args = parser.parse_args(argv)

    fixtures = load_fixtures()
    deny = load_denylist(denylist_path(args.denylist))
    deny.secrets = local_secrets()
    if args.push:
        findings = scan_push(args.push, sys.stdin.read(), fixtures, deny)
    elif args.history:
        findings = scan_history(fixtures, deny)
    elif args.staged:
        findings = scan_staged(fixtures, deny)
    elif args.message:
        raw = Path(args.message).read_text(encoding="utf-8", errors="replace")
        findings = scan_text(commit_message_text(raw), "commit message", fixtures, deny)
    elif args.range:
        findings = scan_range(args.range, fixtures, deny)
    elif args.tree:
        findings = scan_tree(fixtures, deny)
    else:
        stream = sys.stdin if args.text == "-" else open(args.text, encoding="utf-8",
                                                         errors="replace")
        with stream:
            findings = scan_text(stream.read(), args.label, fixtures, deny)
    return report(findings, deny, args.quiet)


if __name__ == "__main__":
    sys.exit(main())
