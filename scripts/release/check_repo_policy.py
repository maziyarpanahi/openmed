#!/usr/bin/env python3
"""Repository policy checks for release-sensitive files."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_TRACKED_PATTERNS = ("RELEASE_NOTE*.md",)
DUA_RELATION_CORPUS_MARKERS = ("biored", "n2c2", "shac")
DUA_RELATION_DATA_EXTENSIONS = frozenset(
    {
        ".a1",
        ".a2",
        ".ann",
        ".bioc",
        ".csv",
        ".gz",
        ".json",
        ".jsonl",
        ".ndjson",
        ".tar",
        ".tsv",
        ".txt",
        ".xml",
        ".zip",
    }
)


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
        raise RuntimeError(
            result.stderr.strip() or f"git ls-files failed for {pattern}"
        )
    return [line for line in result.stdout.splitlines() if line]


def git_tracked_ignored_files() -> list[str]:
    """Return tracked files that are also matched by standard ignore rules."""
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--ignored", "--exclude-standard"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git ls-files ignored check failed")
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


def tracked_dua_relation_corpus_files() -> list[str]:
    """Return tracked BioRED/n2c2 text or annotation payload paths."""

    offenders: list[str] = []
    for value in git_ls_files("."):
        path = Path(value)
        normalized = path.as_posix().casefold()
        if path.suffix.casefold() not in DUA_RELATION_DATA_EXTENSIONS:
            continue
        if any(marker in normalized for marker in DUA_RELATION_CORPUS_MARKERS):
            offenders.append(value)
    return offenders


def main() -> int:
    forbidden_files: list[str] = []
    for pattern in FORBIDDEN_TRACKED_PATTERNS:
        deleted = git_deleted_files(pattern)
        forbidden_files.extend(
            path for path in git_ls_files(pattern) if path not in deleted
        )

    deleted_files = git_deleted_files(".")
    tracked_ignored_files = [
        path for path in git_tracked_ignored_files() if path not in deleted_files
    ]
    dua_relation_corpus_files = [
        path
        for path in tracked_dua_relation_corpus_files()
        if path not in deleted_files
    ]

    if forbidden_files or tracked_ignored_files or dua_relation_corpus_files:
        print("Repository policy failed:", file=sys.stderr)
        if forbidden_files:
            print(
                "- tracked release-note drafts are not allowed; move release notes into CHANGELOG.md instead",
                file=sys.stderr,
            )
            for path in sorted(set(forbidden_files)):
                print(f"- remove tracked file: {path}", file=sys.stderr)
        if tracked_ignored_files:
            print(
                "- tracked ignored files are not allowed; remove them from the index or update .gitignore",
                file=sys.stderr,
            )
            for path in sorted(set(tracked_ignored_files)):
                print(f"- remove tracked ignored file: {path}", file=sys.stderr)
        if dua_relation_corpus_files:
            print(
                "- BioRED/n2c2 relation corpus text and annotations must remain "
                "outside the repository",
                file=sys.stderr,
            )
            for path in sorted(set(dua_relation_corpus_files)):
                print(f"- remove tracked DUA corpus file: {path}", file=sys.stderr)
        return 1

    print("Repository policy passed")
    print("- no tracked RELEASE_NOTE* files found")
    print("- no tracked ignored files found")
    print("- no tracked BioRED/n2c2 relation corpus payloads found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
