"""scripts/check_sensitive.py: real search data is refused, fixtures pass.

Every made-up "real" value below sits on a line marked fixture-ok, so this
file passes the check it tests; the strings handed to the scanner carry no
marker of their own.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
cs = importlib.import_module("check_sensitive")

FAR = "45.5231, -122.6765"  # fixture-ok: a position no fixture is near
PLS = (37.4022, -122.3164)


@pytest.fixture
def fixtures():
    return cs.load_fixtures()


@pytest.fixture
def no_deny():
    return cs.Denylist(path=None, terms=[], points=[])


def deny_from(tmp_path, text):
    path = tmp_path / "list.txt"
    path.write_text(text, encoding="utf-8")
    return cs.load_denylist(path)


def rules(findings):
    return [f.rule for f in findings]


# ------------------------------------------------------------ coordinates

def test_coordinate_far_from_every_fixture_is_refused(fixtures, no_deny):
    found = cs.scan_text(f"center = ({FAR})", "t", fixtures, no_deny)
    assert len(found) == 1 and "coordinate" in found[0].rule
    assert found[0].where == "t line 1"


def test_fixture_point_and_its_neighbourhood_pass(fixtures, no_deny):
    near = f"{PLS[0] + 0.001:.4f}, {PLS[1]:.4f}"  # about 110 m north
    assert cs.scan_text(f"p = ({near})", "t", fixtures, no_deny) == []


def test_geojson_lon_lat_order_matches_a_fixture(fixtures, no_deny):
    ring = f"[[{PLS[1]}, {PLS[0]}], [{PLS[1]}, {PLS[0]}]]"
    assert cs.scan_text(ring, "t", fixtures, no_deny) == []


def test_latitude_and_longitude_on_two_lines_pair_up(fixtures, no_deny):
    lat, lon = FAR.split(", ")
    text = f'"latitude": {lat},\n"longitude": {lon}'
    assert len(cs.scan_text(text, "t", fixtures, no_deny)) == 1


def test_pragma_excuses_a_made_up_coordinate(fixtures, no_deny):
    assert cs.scan_text(f"p = ({FAR})  # {cs.PRAGMA}", "t", fixtures, no_deny) == []


def test_small_numbers_side_by_side_are_not_a_place(fixtures, no_deny):
    assert cs.scan_text("weights = (0.125, 0.875, 1.500, 2.250)", "t",
                        fixtures, no_deny) == []


# ------------------------------------------------------- grids and map ids

def test_unlisted_mgrs_is_refused_and_a_fixture_passes(fixtures, no_deny):
    assert rules(cs.scan_text("at 11T NE 12345 67890", "t", fixtures, no_deny))  # fixture-ok
    assert cs.scan_text("at 10S EG 61234 41234", "t", fixtures, no_deny) == []


def test_real_map_id_is_refused_and_a_placeholder_passes(fixtures, no_deny):
    real = 'map_id="QZ7K2"'  # fixture-ok
    link = "see caltopo.com/m/QZ7K2"  # fixture-ok
    assert "map id" in rules(cs.scan_text(real, "t", fixtures, no_deny))[0]
    assert "map id" in rules(cs.scan_text(link, "t", fixtures, no_deny))[0]
    assert cs.scan_text('map_id="MAP1"', "t", fixtures, no_deny) == []


# ---------------------------------------------------------------- denylist

def test_denylist_terms_maps_and_patterns(tmp_path, fixtures):
    deny = deny_from(tmp_path, "# real\nJane  Doe\nmap QZ7K2\nre \\bTG ?99\\d\\d\\b\n")
    assert len(cs.scan_text("found jane\tdoe at noon", "t", fixtures, deny)) == 1
    assert cs.scan_text("Janet Doering", "t", fixtures, deny) == []
    assert "denylist line 3" in rules(cs.scan_text("map QZ7K2", "t", fixtures, deny))[0]
    assert "denylist line 4" in rules(cs.scan_text("team TG9912", "t", fixtures, deny))[0]


def test_denylist_hit_is_not_excused_and_never_printed(tmp_path, fixtures, capsys):
    deny = deny_from(tmp_path, "Jane Doe\n")
    found = cs.scan_text(f"Jane Doe  # {cs.PRAGMA}", "t", fixtures, deny)
    assert len(found) == 1
    assert cs.report(found, deny, quiet=True) == 1
    assert "Jane" not in capsys.readouterr().err


def test_denylist_point_matches_within_its_radius_even_near_a_fixture(tmp_path, fixtures):
    deny = deny_from(tmp_path, f"point {PLS[0]} {PLS[1]} 300\n")
    near = f"{PLS[0] + 0.002:.4f}, {PLS[1]:.4f}"  # about 220 m away
    far = f"{PLS[0] + 0.004:.4f}, {PLS[1]:.4f}"  # about 440 m away
    assert "denylist point" in rules(cs.scan_text(near, "t", fixtures, deny))[0]
    assert not any("denylist" in r for r in rules(cs.scan_text(far, "t", fixtures, deny)))


def test_a_bad_denylist_entry_stops_the_check(tmp_path):
    with pytest.raises(SystemExit):
        deny_from(tmp_path, "point 37.4\n")


# ---------------------------------------------------------- files and diffs

@pytest.mark.parametrize("path", [
    "tracks/team1.gpx", "area.geojson", "dem.tif", "cache/x.json", "docs/plan.md",
    "src/exports/a.json", "cts.ini", ".env.local", "staged/manifest.json",
    "my-denylist.txt",
])
def test_data_files_are_refused(fixtures, path):
    assert cs.file_rule(path, fixtures)


@pytest.mark.parametrize("path", [
    "src/sar_geo_mcp/cache/evict.py", "extension/manifest.json", ".env.example",
    "README.md",
])
def test_source_files_pass(fixtures, path):
    assert cs.file_rule(path, fixtures) is None


def test_diff_names_the_added_line_and_refuses_a_data_file(fixtures, no_deny):
    diff = "\n".join([
        "diff --git a/x.py b/x.py", "--- a/x.py", "+++ b/x.py",
        "@@ -10,0 +11,2 @@", "+ok = 1", f"+p = ({FAR})",
        "diff --git a/t.gpx b/t.gpx", "--- /dev/null", "+++ b/t.gpx",
        "@@ -0,0 +1 @@", "+<gpx/>",
    ])
    where = [f.where for f in cs.scan_diff(diff, fixtures, no_deny)]
    assert where == ["x.py:12", "t.gpx"]


def test_commit_message_drops_comments_and_the_scissors_cut():
    raw = "fix\n# Please enter\nbody\n# ------------------------ >8 ------------------------\n+diff"
    assert cs.commit_message_text(raw) == "fix\nbody"


@pytest.mark.skipif(not (ROOT / ".git").exists() or shutil.which("git") is None,
                    reason="needs the git checkout")
def test_every_tracked_file_passes(fixtures, no_deny):
    assert cs.scan_tree(fixtures, no_deny) == []


def test_assignment_norms_hold_only_aggregates(fixtures, no_deny):
    path = ROOT / "src" / "sar_geo_mcp" / "planning" / "data" / "assignment_norms.json"
    text = path.read_text(encoding="utf-8")
    assert cs.scan_text(text, "norms", fixtures, no_deny) == []
    named = {"map_id", "map_ids", "label", "labels", "title", "name", "names",
             "subject", "team", "teams", "track_names", "incident"}

    def keys(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield k
                yield from keys(v)
        elif isinstance(node, list):
            for v in node:
                yield from keys(v)

    assert named.isdisjoint(k.lower() for k in keys(json.loads(text)))


# ------------------------------------------------------------- Claude guard

def run_guard(command, env=None):
    event = json.dumps({"tool_name": "Bash", "tool_input": {"command": command},
                        "cwd": str(ROOT)})
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "claude_pr_guard.py")],
                          input=event, text=True, capture_output=True, env=env)


def test_guard_blocks_a_pr_body_with_real_data(tmp_path):
    deny = tmp_path / "list.txt"
    deny.write_text("Jane Doe\n", encoding="utf-8")
    env = {"SAR_SENSITIVE_DENYLIST": str(deny), "PATH": "/usr/bin:/bin"}
    blocked = run_guard('gh pr create --title t --body "found Jane Doe"', env)
    assert blocked.returncode == 2 and "Jane" not in blocked.stderr
    assert run_guard('gh pr create --title t --body "fine"', env).returncode == 0
    assert run_guard('echo "Jane Doe"', env).returncode == 0


# ------------------------------------------------------------- credentials
# Fake tokens are built at run time, so this file carries none whole.

TOKENS = {
    "GitHub token": "gh" + "p_" + "a1B2c3D4" * 5,
    "Google API key": "AI" + "za" + "Sy" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q",
    "Mapbox token": "pk" + ".eyJ" + "1IjoiYWJjIiwiYSI6ImRlZiJ9" + ".AbCdEfGhIjKlMn",
    "AWS access key": "AK" + "IA" + "ABCDEFGH23456789",
    "private key": "-----BEGIN RSA " + "PRIVATE KEY-----",
    "PyPI token": "pypi-" + "AgE" + "IcHlwaS5vcmc" + "x9" * 25,
}


@pytest.mark.parametrize("kind", sorted(TOKENS))
def test_token_formats_are_refused(fixtures, no_deny, kind):
    found = rules(cs.scan_text(f'value = "{TOKENS[kind]}"', "t", fixtures, no_deny))
    assert f"looks like a {kind}" in found


def test_secret_named_assignment_with_a_real_looking_value(fixtures, no_deny):
    real = "ZA_WMS_TOKEN=" + "q8Ld2Xv9Rk3Tz7Wm4Np1"
    assert rules(cs.scan_text(real, "t", fixtures, no_deny))
    for fake in ('wms_token="test-wms-token"', 'mapbox_token="SECRET-MAPBOX"',
                 "ZA_WMS_TOKEN=", "key = hashlib.sha256(text).hexdigest()",
                 "token = config.wms_token"):
        assert cs.scan_text(fake, "t", fixtures, no_deny) == [], fake


def test_password_in_a_url_unless_the_host_is_reserved(fixtures, no_deny):
    real = "https://planner:" + "hunter2@overpass.agency.org/api"
    assert "looks like a password in a URL" in rules(
        cs.scan_text(real, "t", fixtures, no_deny))
    for host in ("osm.agency.example", "router.local", "localhost:8080", "host"):
        url = f"https://u:pw12@{host}/x"  # fixture-ok: reserved hosts
        assert cs.scan_text(url, "t", fixtures, no_deny) == []


def test_this_machines_secrets_are_found_and_never_excused_or_printed(
        tmp_path, fixtures, capsys):
    value = "Zq" + "9xLm2Vb7Np4Rt8Ws"
    (tmp_path / "cts.ini").write_text(f"[me@example.org]\nid = SHORT\nkey = {value}\n")
    (tmp_path / ".env").write_text("ZA_CACHE_DIR=a-long-cache-folder\n"
                                   "ZA_MAPBOX_TOKEN=pk.from-the-env-file\n")
    secrets = cs.local_secrets(tmp_path, environ={"ZA_WMS_TOKEN": "wms-from-environ",
                                                  "HOME": "/nowhere"})
    sources = sorted(src for _, src in secrets)
    assert sources == [".env ZA_MAPBOX_TOKEN", "cts.ini account name",
                       "cts.ini key", "environment ZA_WMS_TOKEN"]
    deny = cs.Denylist(path=None, terms=[], points=[], secrets=secrets)
    found = cs.scan_text(f"url = '?k={value}'  # {cs.PRAGMA}", "t", fixtures, deny)
    assert "cts.ini key" in found[0].rule
    cs.report(found, deny, quiet=True)
    assert value not in capsys.readouterr().err


@pytest.mark.parametrize("path", ["deploy.pem", "keys/server.key", "id_ed25519",
                                  ".pypirc", "service-account-prod.json",
                                  "credentials.json"])
def test_key_files_are_refused(fixtures, path):
    assert cs.file_rule(path, fixtures)


def test_a_pushed_deletion_sends_nothing(fixtures, no_deny):
    line = "(delete) " + "0" * 40 + " refs/heads/gone " + "a" * 40
    assert cs.scan_push("origin", line, fixtures, no_deny) == []
