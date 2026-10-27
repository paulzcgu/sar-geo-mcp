"""Setting up an install: keys from the extension, key_status, check_for_update.

The Claude Desktop extension has no `cts.ini` to point at. Its settings
form hands the CalTopo credential over as environment variables, and the
server must use them in place of the file. `key_status` tells a user's
Claude which keys are set without ever returning one, and
`check_for_update` compares the running version with PyPI's newest.
"""

from __future__ import annotations

import json

import pytest

from sar_geo_mcp import updates
from sar_geo_mcp.caltopo import session as session_module
from sar_geo_mcp.components import setup
from sar_geo_mcp.config import CaltopoConfig
from sar_geo_mcp.context import ServerContext
from sar_geo_mcp.scenarios._sections import error_text

SECRETS = {
    "caltopo_id": "CRED12345678",
    "caltopo_key": "c2VjcmV0LWtleS1mb3ItdGVzdHMtb25seS0xMjM0NTY3",
    "wms_token": "AB12CD",
    "mapbox_token": "pk.mapbox-secret",
    "google_maps_key": "AIza-google-secret",
}


def ctx_with(tmp_path, **fields) -> ServerContext:
    config = CaltopoConfig(cache_dir=tmp_path / "cache",
                           caltopo_config_path=tmp_path / "no-cts.ini", **fields)
    return ServerContext.build(config)


# ---------------------------------------------------------------------------
# Credentials from the environment.
# ---------------------------------------------------------------------------

def test_from_env_reads_the_extension_s_credential(monkeypatch):
    monkeypatch.setenv("ZA_CALTOPO_ID", " CRED12345678 ")
    monkeypatch.setenv("ZA_CALTOPO_KEY", "the-key")
    monkeypatch.setenv("ZA_CALTOPO_ACCOUNT_ID", "ACC123")
    monkeypatch.setenv("ZA_WMS_TOKEN", "AB12CD\n")
    config = CaltopoConfig.from_env()
    assert config.caltopo_id == "CRED12345678"
    assert config.caltopo_key == "the-key"
    assert config.caltopo_account_id == "ACC123"
    assert config.wms_token == "AB12CD", "a pasted token keeps no whitespace"


def test_an_empty_setting_reads_as_unset(monkeypatch):
    """Claude Desktop passes an empty string for an optional field left
    blank, so empty must mean 'not configured', never a key of ''."""
    for name in ("ZA_CALTOPO_ID", "ZA_CALTOPO_KEY", "ZA_WMS_TOKEN"):
        monkeypatch.setenv(name, "")
    config = CaltopoConfig.from_env()
    assert config.caltopo_id == config.caltopo_key == config.wms_token == ""
    assert config.wms_base_url() is None


def test_the_credential_never_appears_in_a_repr():
    text = repr(CaltopoConfig(**SECRETS))
    for value in SECRETS.values():
        assert value not in text


def test_a_direct_credential_opens_the_map_without_a_file(tmp_path, monkeypatch):
    seen = {}

    class FakeSession:
        def __init__(self, host, map_id, **kwargs):
            seen.update(host=host, map_id=map_id, **kwargs)

    monkeypatch.setattr(session_module, "CaltopoSession", FakeSession)
    config = CaltopoConfig(caltopo_config_path=tmp_path / "missing.ini",
                           caltopo_id="CRED12345678", caltopo_key="k",
                           caltopo_account_id="ACC123")
    session_module.open_map("MAP1", config)
    assert seen["id"] == "CRED12345678" and seen["key"] == "k"
    assert seen["accountId"] == "ACC123"
    assert "configpath" not in seen
    assert seen["sync"] is False


def test_without_a_credential_the_file_is_still_required(tmp_path):
    config = CaltopoConfig(caltopo_config_path=tmp_path / "missing.ini")
    with pytest.raises(FileNotFoundError, match="ZA_CALTOPO_ID"):
        session_module.open_map("MAP1", config)


def test_a_scenario_error_blanks_the_caltopo_key(tmp_path):
    ctx = ctx_with(tmp_path, **SECRETS)
    text = error_text(ctx, RuntimeError(f"signed with {SECRETS['caltopo_key']}"))
    assert SECRETS["caltopo_key"] not in text


# ---------------------------------------------------------------------------
# key_status.
# ---------------------------------------------------------------------------

def test_key_status_says_what_is_set_and_never_what_it_is(tmp_path):
    result = setup.key_status(ctx_with(tmp_path, **SECRETS))
    dumped = json.dumps(result)
    for value in SECRETS.values():
        assert value not in dumped
    assert result["missing"] == []
    assert all(entry["set"] for entry in result["keys"])


def test_key_status_lists_every_missing_key_with_where_to_get_it(tmp_path):
    result = setup.key_status(ctx_with(tmp_path))
    assert result["missing"] == ["caltopo_credential", "wms_token",
                                 "mapbox_token", "google_maps_key"]
    wms = next(e for e in result["keys"] if e["key"] == "wms_token")
    assert "API Access" in wms["where"] and wms["setting"] == "CalTopo WMS token"


def test_half_a_credential_is_reported_as_unset(tmp_path):
    result = setup.key_status(ctx_with(tmp_path, caltopo_id="CRED12345678"))
    cred = result["keys"][0]
    assert cred["set"] is False and "both" in cred["note"]


def test_a_cts_ini_counts_as_a_credential(tmp_path):
    (tmp_path / "cts.ini").write_text("[a]\nid=x\nkey=y\n", encoding="utf-8")
    ctx = ServerContext.build(CaltopoConfig(caltopo_config_path=tmp_path / "cts.ini"))
    cred = setup.key_status(ctx)["keys"][0]
    assert cred["set"] is True and "cts.ini" in cred["note"]


# ---------------------------------------------------------------------------
# check_for_update.
# ---------------------------------------------------------------------------

def release(version):
    return {"version": version, "released": "2026-10-20T00:00:00Z",
            "url": f"https://pypi.org/project/sar-geo-mcp/{version}/"}


@pytest.mark.parametrize("running, latest, status", [
    ("0.38.0", "0.38.0", "up_to_date"),
    ("0.37.1", "0.38.0", "update_available"),
    ("0.39.0", "0.38.0", "ahead"),
    ("0.9.0", "0.10.0", "update_available"),
])
def test_check_for_update_compares_versions(tmp_path, monkeypatch, running, latest, status):
    monkeypatch.setattr(setup, "installed_version", lambda: running)
    monkeypatch.setattr(setup, "latest_release", lambda: release(latest))
    monkeypatch.delenv(updates.LAUNCH_MODE_ENV, raising=False)
    result = setup.check_for_update(ctx_with(tmp_path))
    assert result["status"] == status
    assert result["running"] == running and result["latest"] == latest


def test_under_the_extension_the_advice_is_to_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "installed_version", lambda: "0.37.1")
    monkeypatch.setattr(setup, "latest_release", lambda: release("0.38.0"))
    monkeypatch.setenv(updates.LAUNCH_MODE_ENV, "offline")
    result = setup.check_for_update(ctx_with(tmp_path))
    assert "Quit Claude Desktop" in result["how_to_update"]
    assert "could not be reached" in result["launch_mode_meaning"]


def test_a_pinned_extension_is_told_to_clear_the_pin(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "installed_version", lambda: "0.37.1")
    monkeypatch.setattr(setup, "latest_release", lambda: release("0.38.0"))
    monkeypatch.setenv(updates.LAUNCH_MODE_ENV, "pinned")
    assert "Pin version" in setup.check_for_update(ctx_with(tmp_path))["how_to_update"]


def test_offline_is_an_answer_not_an_exception(tmp_path, monkeypatch):
    def unreachable():
        raise OSError("no route to host")

    monkeypatch.setattr(setup, "latest_release", unreachable)
    result = setup.check_for_update(ctx_with(tmp_path))
    assert result["status"] == "unknown" and "PyPI" in result["error"]


def test_a_package_not_yet_on_pypi_is_not_called_offline(tmp_path, monkeypatch):
    class NotFound(Exception):
        response = type("R", (), {"status_code": 404})()

    def missing():
        raise NotFound()

    monkeypatch.setattr(setup, "latest_release", missing)
    result = setup.check_for_update(ctx_with(tmp_path))
    assert result["status"] == "unknown" and "not published" in result["error"]


def test_version_key_orders_releases():
    ordered = ["0.9.0", "0.10.0rc1", "0.10.0", "0.10.1", "1.0"]
    assert sorted(reversed(ordered), key=updates.version_key) == ordered
    assert updates.version_key("1.0") == updates.version_key("1.0.0")


def test_latest_release_reads_pypi_s_reply():
    reply = {"info": {"version": "0.38.0"},
             "releases": {"0.38.0": [{"upload_time_iso_8601": "2026-10-21T10:00:00Z"},
                                     {"upload_time_iso_8601": "2026-10-21T09:00:00Z"}]}}
    latest = updates.latest_release(fetch=lambda url, timeout: reply)
    assert latest["version"] == "0.38.0"
    assert latest["released"] == "2026-10-21T09:00:00Z"
