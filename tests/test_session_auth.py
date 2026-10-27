"""Opening an authenticated CalTopo session.

TWO BUGS THIS PINS
------------------
1. `getMapData()` was called in five places and does not
   exist in caltopo_python 2.0.0 - the method is `getFeatures()`. Every
   authenticated read raised AttributeError. It stayed hidden because all
   the testing had used public maps, which take a different path entirely.

2. An unset account name refused a perfectly good credentials file.
   `caltopo_python` matches the section header against a configured
   account and rejects an empty one, so a user who filled in cts.ini but
   never set the environment variable got "specified account '' has no
   entry" - which points at the config file rather than at the setting
   that was actually missing.

Both were found by trying a real key rather than by any test, which is
why they are pinned here.
"""

from __future__ import annotations

import pytest

from sar_geo_mcp.caltopo import session as session_module
from sar_geo_mcp.caltopo.session import open_map, sole_account
from sar_geo_mcp.config import CaltopoConfig


def write_cts(path, *accounts) -> None:
    body = "".join(
        f"[{name}]\nid=ID{i}\nkey=KEY{i}\naccountId=ACC{i}\n\n"
        for i, name in enumerate(accounts)
    )
    path.write_text(body, encoding="utf-8")


# ---------------------------------------------------------------------------
# The method actually exists.
# ---------------------------------------------------------------------------

def test_the_session_api_we_call_is_the_one_caltopo_python_has():
    """A rename in the client library broke every authenticated read and no
    test noticed, because the public path does not touch this object."""
    from caltopo_python import CaltopoSession

    assert hasattr(CaltopoSession, "getFeatures")
    assert not hasattr(CaltopoSession, "getMapData"), (
        "getMapData is back - if the library restored it, check which one "
        "this project should be calling")


def test_no_caller_anywhere_still_asks_for_getMapData():
    """Five call sites had the same wrong name. A grep
    guard is cruder than a type checker and catches exactly this."""
    from pathlib import Path

    roots = [Path(session_module.__file__).resolve().parents[2]]

    offenders = [
        str(path) for root in roots for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
        and "getMapData" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"getMapData still called in: {offenders}"


# ---------------------------------------------------------------------------
# Which account.
# ---------------------------------------------------------------------------

def test_one_account_in_the_file_needs_no_setting(tmp_path):
    """There is nothing to choose between, so choosing it is not a guess."""
    cts = tmp_path / "cts.ini"
    write_cts(cts, "you@example.com")
    assert sole_account(cts) == "you@example.com"


def test_two_accounts_are_never_guessed_between(tmp_path):
    """The wrong CalTopo account is the kind of mistake that reads somebody
    else's incident."""
    cts = tmp_path / "cts.ini"
    write_cts(cts, "me@example.org", "you@example.com")
    assert sole_account(cts) is None


def test_an_unreadable_config_resolves_to_nothing(tmp_path):
    cts = tmp_path / "cts.ini"
    cts.write_text("this is not an ini [[[", encoding="utf-8")
    assert sole_account(cts) is None


def test_opening_with_several_accounts_and_no_setting_names_them(tmp_path,
                                                                 monkeypatch):
    """The old failure blamed the config file. The message has to point at
    the setting that was actually missing, and list what to set it to."""
    cts = tmp_path / "cts.ini"
    write_cts(cts, "me@example.org", "you@example.com")
    config = CaltopoConfig(caltopo_config_path=cts, caltopo_account="")

    with pytest.raises(ValueError) as raised:
        open_map("M1", config)

    message = str(raised.value)
    assert "ZA_CALTOPO_ACCOUNT" in message
    assert "me@example.org" in message and "you@example.com" in message


def test_an_explicit_account_still_wins(tmp_path, monkeypatch):
    """The fallback must never override somebody's deliberate choice."""
    cts = tmp_path / "cts.ini"
    write_cts(cts, "me@example.org", "you@example.com")

    seen = {}

    class FakeSession:
        def __init__(self, host, map_id, configpath, account, **kwargs):
            seen["account"] = account

    monkeypatch.setattr(session_module, "CaltopoSession", FakeSession)
    open_map("M1", CaltopoConfig(caltopo_config_path=cts,
                                 caltopo_account="me@example.org"))
    assert seen["account"] == "me@example.org"


def test_a_missing_config_file_still_says_so_first(tmp_path):
    """Absent credentials and ambiguous credentials are different problems
    and need different messages."""
    config = CaltopoConfig(caltopo_config_path=tmp_path / "nope.ini")
    with pytest.raises(FileNotFoundError, match="cts.ini.example"):
        open_map("M1", config)


def test_identical_labels_are_reported_as_ids_not_as_use_the_full_label():
    """Real maps give two assignments the same full label (21 of 96
    measured). Telling the caller to type the full label is advice that
    cannot work; the ids can."""
    import pytest

    from sar_geo_mcp.caltopo.assignment import find_assignment

    features = [
        {"id": "aaa-111", "properties": {"class": "Assignment", "letter": "DR"},
         "geometry": {"type": "Polygon", "coordinates": []}},
        {"id": "bbb-222", "properties": {"class": "Assignment", "letter": "DR"},
         "geometry": {"type": "Polygon", "coordinates": []}},
    ]

    with pytest.raises(LookupError) as caught:
        find_assignment(features, "DR")

    message = str(caught.value)
    assert "Use the full label" not in message
    assert "aaa-111" in message and "bbb-222" in message

    # ...and the id it names actually works.
    assert find_assignment(features, "bbb-222").id == "bbb-222"
