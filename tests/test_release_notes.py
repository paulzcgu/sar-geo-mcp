"""Release notes come from pyproject.toml's `# <version>:` comment blocks.

scripts/release.py refuses to tag without one, and the release workflow
turns the same block into the GitHub Release's notes.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "release_notes", ROOT / "scripts" / "release_notes.py")
release_notes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_notes)

SAMPLE = """\
[project]
name = "x"
# 0.9.0: the old one.
# Still the old one.
# 0.10.0: the new one, hard-wrapped
# over two lines.
# WHAT CHANGED: a label starts
# its own paragraph.
# - and so does
# - a list item
version = "0.10.0"
"""


def test_the_block_for_a_version_is_its_lines_joined():
    notes = release_notes.notes_for(SAMPLE, "0.10.0")
    assert notes.splitlines()[0] == "the new one, hard-wrapped over two lines."
    assert "WHAT CHANGED: a label starts its own paragraph." in notes.split("\n\n")
    assert "- and so does" in notes.split("\n\n")
    assert "old one" not in notes


def test_a_block_stops_at_the_next_heading():
    assert release_notes.notes_for(SAMPLE, "0.9.0") == "the old one. Still the old one."


def test_a_version_without_a_block_is_refused():
    with pytest.raises(ValueError, match="0.11.0"):
        release_notes.notes_for(SAMPLE, "0.11.0")


def test_the_current_version_has_notes():
    """A release of the version in pyproject.toml would be refused."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    version = release_notes.current_version(text)
    assert release_notes.notes_for(text, version)
