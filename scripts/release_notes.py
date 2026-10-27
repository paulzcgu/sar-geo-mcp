"""Print the release notes for a version, from pyproject.toml's comments.

    python scripts/release_notes.py            # the current version
    python scripts/release_notes.py 0.37.1

Each release is written up as a comment block above `version = ...` in
pyproject.toml, starting `# 0.38.0: ...` and running to the next such
heading (or the version line). This prints that block with the `# `
markers removed, re-wrapped into paragraphs, so the same text becomes the
annotated tag's message and the GitHub Release's notes.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
HEADING = re.compile(r"^# (\d+(?:\.\d+)+(?:[a-z]+\d*)?): ?(.*)$")


def current_version(text: str) -> str:
    return tomllib.loads(text)["project"]["version"]


def notes_for(text: str, version: str) -> str:
    """The comment block for `version`; ValueError when there is none."""
    lines, collecting = [], False
    for line in text.splitlines():
        heading = HEADING.match(line)
        if heading:
            if collecting:
                break
            if heading.group(1) == version:
                collecting = True
                lines.append(heading.group(2))
            continue
        if collecting:
            if not line.startswith("#"):
                break
            lines.append(line[1:].removeprefix(" "))
    if not collecting:
        raise ValueError(f"pyproject.toml has no '# {version}:' comment block")

    # Lines are hard-wrapped comments: join each run into one paragraph,
    # but keep a line that starts a list item or a capitalised label
    # ("WHAT CHANGED:") on its own line.
    paragraphs: list[str] = []
    for line in lines:
        starts_new = (not paragraphs or re.match(r"^(- |[A-Z][A-Z ]+:)", line))
        if starts_new:
            paragraphs.append(line.strip())
        else:
            paragraphs[-1] = f"{paragraphs[-1]} {line.strip()}".strip()
    body = "\n\n".join(p for p in paragraphs if p)
    if not body:
        raise ValueError(f"the '# {version}:' comment block is empty")
    return body


def main(argv: list[str]) -> int:
    text = PYPROJECT.read_text(encoding="utf-8")
    version = argv[1] if len(argv) > 1 else current_version(text)
    print(notes_for(text, version))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
