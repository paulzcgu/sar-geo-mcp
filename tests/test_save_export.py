"""Handing files to the user: absolute paths, the Data folder, save_export.

A chat client was given `cache/staged/aoi-.../exports/medevac.geojson`, a
path relative to a folder inside Claude Desktop's own data, and could only
tell the user to search their disk for it. Paths are now absolute, the
extension caches terrain in the system cache folder, and `save_export`
copies a file to ~/Documents/sar_geo/<chat folder> and shows it.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import pytest

from sar_geo_mcp import config as config_module
from sar_geo_mcp.components import files
from sar_geo_mcp.config import CaltopoConfig
from sar_geo_mcp.context import ServerContext


# ---------------------------------------------------------------------------
# The cache directory.
# ---------------------------------------------------------------------------

def test_the_cache_directory_is_always_absolute(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ZA_CACHE_DIR", raising=False)
    monkeypatch.delenv("SAR_GEO_LAUNCH_MODE", raising=False)
    assert config_module.cache_dir_from_env() == tmp_path / "cache"


def test_home_is_expanded_in_the_cache_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ZA_CACHE_DIR", "${HOME}/SAR Geo")
    assert config_module.cache_dir_from_env() == tmp_path / "SAR Geo"
    monkeypatch.setenv("ZA_CACHE_DIR", "~/elsewhere")
    assert config_module.cache_dir_from_env() == tmp_path / "elsewhere"


@pytest.mark.parametrize("value", ["", "${user_config.cache_folder}", "${NOT_SET_ANYWHERE}/x"])
def test_under_the_extension_an_unusable_folder_means_the_system_cache(monkeypatch, value):
    monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)
    monkeypatch.setenv("ZA_CACHE_DIR", value)
    monkeypatch.setenv("SAR_GEO_LAUNCH_MODE", "latest")
    assert config_module.cache_dir_from_env() == config_module.EXTENSION_CACHE_DIR


def test_the_system_cache_is_not_in_documents():
    """Gigabytes of terrain must not land where cloud sync picks them up."""
    assert "Documents" not in config_module.EXTENSION_CACHE_DIR.parts


def test_the_exports_folder_defaults_to_documents_sar_geo(monkeypatch):
    monkeypatch.delenv("SAR_GEO_EXPORTS_DIR", raising=False)
    assert config_module.exports_dir_from_env() == Path.home() / "Documents" / "sar_geo"
    monkeypatch.setenv("SAR_GEO_EXPORTS_DIR", "${user_config.exports_folder}")
    assert config_module.exports_dir_from_env() == config_module.DEFAULT_EXPORTS_DIR


def test_claude_desktop_s_documents_placeholder_is_filled(monkeypatch):
    monkeypatch.setenv("SAR_GEO_EXPORTS_DIR", "${DOCUMENTS}/sar_geo")
    assert config_module.exports_dir_from_env() == Path.home() / "Documents" / "sar_geo"


# ---------------------------------------------------------------------------
# save_export.
# ---------------------------------------------------------------------------

@pytest.fixture
def setup_dirs(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    export = cache / "staged" / "aoi-1" / "exports" / "medevac.geojson"
    export.parent.mkdir(parents=True)
    export.write_text('{"type": "FeatureCollection", "features": []}')
    exports = tmp_path / "Documents" / "sar_geo"
    monkeypatch.setattr(files, "reveal", lambda path: True)
    ctx = ServerContext.build(CaltopoConfig(cache_dir=cache, exports_dir=exports))
    return ctx, export, exports


def test_an_export_goes_in_the_chat_s_folder(setup_dirs):
    ctx, export, exports = setup_dirs
    result = files.save_export(ctx, str(export), chat_folder="2026-10-25 Example medevac")
    saved = Path(result["saved_to"])
    assert saved == exports / "2026-10-25 Example medevac" / "medevac.geojson"
    assert saved.read_text() == export.read_text()
    assert result["file_url"].startswith("file://") and result["shown"] is True


def test_without_a_chat_folder_the_date_is_used(setup_dirs):
    ctx, export, exports = setup_dirs
    saved = Path(files.save_export(ctx, str(export))["saved_to"])
    assert saved.parent.parent == exports
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", saved.parent.name)


@pytest.mark.parametrize("name, folder", [
    ("../../etc", "etc"),
    ("a/b:c?", "a b c"),
    ("  ...  ", "2026-10-25"),
])
def test_a_chat_folder_name_cannot_leave_the_exports_folder(name, folder):
    assert files.chat_folder_name(name, today=dt.date(2026, 10, 25)) == folder


def test_a_taken_name_gets_a_number(setup_dirs):
    ctx, export, _ = setup_dirs
    files.save_export(ctx, str(export), chat_folder="x")
    second = files.save_export(ctx, str(export), chat_folder="x")
    assert Path(second["saved_to"]).name == "medevac (2).geojson"


def test_a_path_relative_to_the_old_working_directory_still_works(setup_dirs):
    """0.38.0 reported `cache/staged/...`: a user's Claude may quote it."""
    ctx, export, _ = setup_dirs
    result = files.save_export(ctx, "cache/staged/aoi-1/exports/medevac.geojson")
    assert Path(result["source"]) == export.resolve()


def test_a_chosen_folder_is_used_and_made(setup_dirs, tmp_path):
    ctx, export, _ = setup_dirs
    target = tmp_path / "Desktop" / "Mission"
    result = files.save_export(ctx, str(export), destination=str(target), show=False)
    assert Path(result["saved_to"]).parent == target and result["shown"] is False


def test_nothing_outside_the_cache_can_be_copied(setup_dirs, tmp_path):
    ctx, _, _ = setup_dirs
    secret = tmp_path / "cts.ini"
    secret.write_text("[a]\nkey=secret\n")
    with pytest.raises(ValueError, match="not a file this server wrote"):
        files.save_export(ctx, str(secret))
    with pytest.raises(ValueError, match="not a file this server wrote"):
        files.save_export(ctx, str(tmp_path / "cache" / ".." / "cts.ini"))


def test_only_export_file_types_are_copied(setup_dirs):
    ctx, export, _ = setup_dirs
    index = export.parent.parent / "index.sqlite"
    index.write_text("x")
    with pytest.raises(ValueError, match="not an export"):
        files.save_export(ctx, str(index))


def test_a_missing_file_says_so(setup_dirs):
    ctx, export, _ = setup_dirs
    with pytest.raises(FileNotFoundError):
        files.save_export(ctx, str(export.with_name("nope.geojson")))
