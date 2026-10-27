"""Release the version in pyproject.toml: one command, checks first.

    .venv/bin/python scripts/release.py --dry-run   # check, change nothing
    .venv/bin/python scripts/release.py             # check, tag, push

Before anything is pushed, it checks that:
- the working tree is clean (what is tagged is what was tested),
- tag v<version> does not exist here or on origin,
- pyproject.toml has a `# <version>:` comment block (it becomes the notes),
- the tests pass (live tests excluded, as in CI).

Then it makes the annotated tag v<version>, its message the release notes,
and pushes the current branch and the tag. The push starts
.github/workflows/release.yml, which tests again, publishes to PyPI and
creates the GitHub Release with the wheel, sdist and sar-geo.mcpb attached.
Everyone with the Claude Desktop extension gets it on their next restart.

To release: write the `# <version>:` block, bump `version`, commit, run this.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from release_notes import PYPROJECT, current_version, notes_for  # noqa: E402


def run(*args: str, capture: bool = False) -> str:
    result = subprocess.run(args, cwd=ROOT, check=True, text=True,
                            capture_output=capture)
    return result.stdout.strip() if capture else ""


def fail(message: str) -> int:
    print(f"release: {message}", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="run every check, then stop before tagging")
    parser.add_argument("--skip-tests", action="store_true",
                        help="skip the local test run (CI still runs them)")
    args = parser.parse_args()

    text = PYPROJECT.read_text(encoding="utf-8")
    version = current_version(text)
    tag = f"v{version}"
    try:
        notes = notes_for(text, version)
    except ValueError as exc:
        return fail(f"{exc}. Write it above `version = ...` first.")

    if run("git", "status", "--porcelain", capture=True):
        return fail("the working tree has uncommitted changes; commit or stash them")
    branch = run("git", "rev-parse", "--abbrev-ref", "HEAD", capture=True)
    if run("git", "tag", "--list", tag, capture=True):
        return fail(f"tag {tag} already exists here; bump the version")
    if run("git", "ls-remote", "--tags", "origin", f"refs/tags/{tag}", capture=True):
        return fail(f"tag {tag} already exists on origin; bump the version")

    if not args.skip_tests:
        print("release: running tests ...")
        run(sys.executable, "-m", "pytest", "-q", "-m", "not live",
            "-p", "no:cacheprovider")

    print(f"\nrelease: {tag} from branch {branch}\n\n{notes}\n")
    if args.dry_run:
        print("release: dry run - every check passed; nothing tagged or pushed")
        return 0

    run("git", "tag", "-a", tag, "-m", f"{tag}\n\n{notes}")
    run("git", "push", "origin", branch)
    run("git", "push", "origin", tag)
    print(f"release: pushed {tag}. Watch it with: gh run watch "
          f"$(gh run list --workflow release.yml --limit 1 --json databaseId -q '.[0].databaseId')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
