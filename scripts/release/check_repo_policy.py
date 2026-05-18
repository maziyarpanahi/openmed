#!/usr/bin/env python3
"""Repository policy checks for release-sensitive files."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_TRACKED_PATTERNS = ("RELEASE_NOTE*.md",)


def git_ls_files(pattern: str) -> list[str]:
    """Return tracked files matching a git pathspec."""
    result = subprocess.run(
        ["git", "ls-files", "--", pattern],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git ls-files failed for {pattern}")
    return [line for line in result.stdout.splitlines() if line]


def git_deleted_files(pattern: str) -> set[str]:
    """Return files scheduled for deletion in the working tree or index."""
    deleted: set[str] = set()
    for args in (
        ["git", "diff", "--name-only", "--diff-filter=D", "--", pattern],
        ["git", "diff", "--cached", "--name-only", "--diff-filter=D", "--", pattern],
    ):
        result = subprocess.run(
            args,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                result.stderr.strip() or f"{' '.join(args[:3])} failed for {pattern}"
            )
        deleted.update(line for line in result.stdout.splitlines() if line)
    return deleted


def main() -> int:
    forbidden_files: list[str] = []
    for pattern in FORBIDDEN_TRACKED_PATTERNS:
        deleted = git_deleted_files(pattern)
        forbidden_files.extend(
            path for path in git_ls_files(pattern) if path not in deleted
        )

    if forbidden_files:
        print("Repository policy failed:", file=sys.stderr)
        print(
            "- tracked release-note drafts are not allowed; move release notes into CHANGELOG.md instead",
            file=sys.stderr,
        )
        for path in sorted(set(forbidden_files)):
            print(f"- remove tracked file: {path}", file=sys.stderr)
        return 1

    print("Repository policy passed")
    print("- no tracked RELEASE_NOTE* files found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
