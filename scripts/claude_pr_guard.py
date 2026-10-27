"""Claude Code PreToolUse hook: check what a `gh` command would publish.

A git hook sees commits but never a pull request's title or body, an issue,
a comment or release notes typed into `gh`. This hook reads the Bash
command Claude is about to run and, when it is a `gh` command that
publishes text, scans the whole command - and any --body-file / -F it
names - with check_sensitive.py. A finding blocks the command (exit 2) and
tells Claude why. Wired up in .claude/settings.json.
"""

from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_sensitive as cs  # noqa: E402

PUBLISHES = re.compile(
    r"\bgh\s+(?:pr\s+(?:create|edit|comment|review|merge)"
    r"|issue\s+(?:create|edit|comment)"
    r"|release\s+(?:create|edit)"
    r"|gist\s+create"
    r"|api\b)")


def body_files(command: str) -> list:
    try:
        words = shlex.split(command)
    except ValueError:
        return []
    files = []
    for i, word in enumerate(words):
        if word in ("--body-file", "-F", "--notes-file") and i + 1 < len(words):
            files.append(words[i + 1])
        elif word.startswith(("--body-file=", "--notes-file=")):
            files.append(word.split("=", 1)[1])
    return [f for f in files if f != "-"]


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except ValueError:
        return 0
    command = (event.get("tool_input") or {}).get("command") or ""
    if not PUBLISHES.search(command):
        return 0
    fixtures = cs.load_fixtures()
    deny = cs.load_denylist(cs.denylist_path())
    deny.secrets = cs.local_secrets()
    findings = cs.scan_text(command, "gh command", fixtures, deny)
    for name in body_files(command):
        path = Path(event.get("cwd") or ".") / name
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
            findings += cs.scan_text(text, name, fixtures, deny)
    if not findings:
        return 0
    print("Blocked: this gh command would publish what looks like real search "
          "data, which this public repository must not hold:", file=sys.stderr)
    for f in findings:
        print(f"  {f}", file=sys.stderr)
    print("Rewrite the text without it (made-up values only), then run it again.",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
