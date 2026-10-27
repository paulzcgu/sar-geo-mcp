"""The Claude Desktop extension's launcher (extension/src/launcher.py).

It decides, on every start, which sar-geo-mcp to run: the newest on PyPI,
the version a user pinned, or the cached copy when offline. It also turns
settings left blank back into "unset" before the server reads them.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "sar_geo_launcher", ROOT / "extension" / "src" / "launcher.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_online_runs_the_newest_release_exactly():
    mode, args = launcher.plan({}, latest=lambda: "0.38.0")
    assert mode == "latest"
    assert args[args.index("--from") + 1] == "sar-geo-mcp[mcp]==0.38.0"
    assert args[-1] == "sar-geo-mcp" and "--offline" not in args


def test_offline_runs_the_cached_copy():
    mode, args = launcher.plan({}, latest=lambda: None)
    assert mode == "offline"
    assert "--offline" in args
    assert args[args.index("--from") + 1] == "sar-geo-mcp[mcp]"


def test_a_pin_wins_and_never_asks_pypi():
    def must_not_be_called():
        raise AssertionError("a pinned launch asked PyPI")

    mode, args = launcher.plan({"SAR_GEO_PIN_VERSION": "0.37.1"},
                               latest=must_not_be_called)
    assert mode == "pinned"
    assert args[args.index("--from") + 1] == "sar-geo-mcp[mcp]==0.37.1"


def test_the_python_is_one_every_dependency_has_wheels_for():
    _, args = launcher.plan({}, latest=lambda: "0.38.0")
    assert args[args.index("--python") + 1] == "3.12"


@pytest.mark.parametrize("value", ["", "   ", "${user_config.wms_token}"])
def test_a_blank_setting_is_removed(value):
    env = launcher.clean_settings({"ZA_WMS_TOKEN": value, "OTHER": "kept"})
    assert "ZA_WMS_TOKEN" not in env and env["OTHER"] == "kept"


def test_a_filled_setting_is_trimmed():
    env = launcher.clean_settings({"ZA_WMS_TOKEN": " AB12CD\n"})
    assert env["ZA_WMS_TOKEN"] == "AB12CD"


def test_the_manifest_passes_every_setting_the_launcher_cleans():
    """A setting added to one side and not the other would be silently
    ignored, or reach the server as a literal placeholder."""
    manifest = json.loads((ROOT / "extension" / "manifest.json").read_text())
    env = manifest["server"]["mcp_config"]["env"]
    assert set(env) == set(launcher.SETTINGS)
    for value in env.values():
        key = value.removeprefix("${user_config.").removesuffix("}")
        assert key in manifest["user_config"], value


def test_every_secret_field_is_masked():
    manifest = json.loads((ROOT / "extension" / "manifest.json").read_text())
    config = manifest["user_config"]
    for key in ("caltopo_key", "wms_token", "mapbox_token", "google_maps_key"):
        assert config[key]["sensitive"] is True, key
        assert config[key]["required"] is False, "every key is optional"


@pytest.mark.parametrize("value", ["", "${user_config.cache_folder}"])
def test_an_unusable_folder_is_left_to_the_server(value):
    env = launcher.clean_settings({"ZA_CACHE_DIR": value})
    assert "ZA_CACHE_DIR" not in env


def test_folder_placeholders_are_filled(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    env = launcher.clean_settings({"ZA_CACHE_DIR": "${HOME}/c",
                                   "SAR_GEO_EXPORTS_DIR": "${DOCUMENTS}/sar_geo"})
    assert env["ZA_CACHE_DIR"] == f"{tmp_path}/c"
    assert env["SAR_GEO_EXPORTS_DIR"] == f"{tmp_path}/Documents/sar_geo"
