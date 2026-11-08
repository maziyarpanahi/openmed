#!/usr/bin/env python3
"""Inventory Python floor references using paths, line numbers and kinds only."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[2]
VERSION = r"(?P<major>\d+)\.(?P<minor>\d+)(?:\.\d+)?"
GUARD = re.compile(
    r"\bsys\.version_info(?:\s*\[\s*:\s*2\s*\])?\s*"
    r"(?P<operator>>=|<=|==|!=|<|>)\s*\(\s*"
    r"(?P<major>\d+)\s*,\s*(?P<minor>\d+)\s*(?P<patch>,\s*\d+)?"
)
MARKER = re.compile(
    r"\bpython_(?:full_)?version\s*(?P<operator>>=|<=|==|!=|<|>)\s*"
    r"['\"]" + VERSION
)
REQUIRES = re.compile(r"^\s*requires-python\s*=\s*['\"][^'\"]*?" + VERSION)
WORKFLOW_KEY = re.compile(r"^\s*(?:-\s*)?python(?:-version)?\s*:\s*(?P<versions>[^#]+)")
DOCS_MENTION = re.compile(
    r"\b(?:CPython|Python)(?:\s+|-(?=\d)|(?=\d))" + VERSION,
    re.IGNORECASE,
)
SKIP_DIRS = {"build", "dist", "node_modules", "site", "__pycache__"}


@dataclass(frozen=True, order=True)
class Finding:
    """A source location relevant to raising the supported Python floor."""

    path: str
    line_number: int
    kind: str


def parse_floor(value: str) -> tuple[int, int]:
    """Read a CLI floor expressed as a major and minor version."""
    match = re.fullmatch(r"(\d+)\.(\d+)", value)
    if not match:
        raise argparse.ArgumentTypeError("use a major.minor floor such as 3.12")
    return int(match[1]), int(match[2])


def _version(match: re.Match[str]) -> tuple[int, int]:
    return int(match["major"]), int(match["minor"])


def _stale_boundary(match: re.Match[str], floor: tuple[int, int]) -> bool:
    version = _version(match)
    if version != floor:
        return version < floor
    # At the floor, an inclusive minimum or an exclusive upper bound is
    # already constant. Patch-level constraints may still matter.
    if match.groupdict().get("patch"):
        return False
    if re.search(r"\d+\.\d+\.\d+$", match[0]):
        return False
    return match["operator"] in {">=", "<"}


def _relative_paths(root: Path) -> Iterator[Path]:
    if (root / ".git").exists():
        result = subprocess.run(
            [
                "git",
                "ls-files",
                "-z",
                "--",
                "*.py",
                "*.md",
                "pyproject.toml",
                ".github/workflows",
            ],
            cwd=root,
            capture_output=True,
            check=True,
        )
        for name in result.stdout.split(b"\0"):
            if name:
                yield Path(os.fsdecode(name))
        return

    # Also work in a source archive or synthetic fixture tree.
    for directory, subdirs, filenames in os.walk(root):
        rel = Path(directory).relative_to(root)
        subdirs[:] = [
            name
            for name in subdirs
            if name not in SKIP_DIRS
            and (not name.startswith(".") or (not rel.parts and name == ".github"))
        ]
        yield from (rel / name for name in filenames)


def _finding_kinds(path: Path, line: str, floor: tuple[int, int]) -> Iterator[str]:
    if path.suffix == ".py" and not line.lstrip().startswith("#"):
        if any(_stale_boundary(m, floor) for m in GUARD.finditer(line)):
            yield "version_info_guard"
    elif path == Path("pyproject.toml") and not line.lstrip().startswith("#"):
        if any(_stale_boundary(m, floor) for m in MARKER.finditer(line)):
            yield "python_version_marker"
        if (match := REQUIRES.match(line)) and _version(match) < floor:
            yield "requires_python"
    elif path.suffix == ".md":
        if any(_version(m) < floor for m in DOCS_MENTION.finditer(line)):
            yield "docs_mention"
    elif path.parts[:2] == (".github", "workflows") and path.suffix in {
        ".yml",
        ".yaml",
    }:
        if (entry := WORKFLOW_KEY.match(line)) and any(
            _version(m) < floor for m in re.finditer(VERSION, entry["versions"])
        ):
            yield "workflow_python_version"


def collect_findings(root: Path, floor: tuple[int, int]) -> list[Finding]:
    """Return sorted locations in tracked sources or an unpacked source tree."""
    findings = set()
    for path in _relative_paths(root):
        if path.suffix not in {".py", ".md", ".yaml", ".yml"} and path != Path(
            "pyproject.toml"
        ):
            continue
        file = root / path
        if file.is_symlink() or not file.is_file():
            continue
        for number, line in enumerate(
            file.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            findings.update(
                Finding(path.as_posix(), number, kind)
                for kind in _finding_kinds(path, line, floor)
            )
    return sorted(findings)


def main(argv: list[str] | None = None) -> int:
    """Print the inventory without exposing any source line content."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floor", type=parse_floor, default="3.12")
    parser.add_argument("--json", action="store_true", help="Emit structured JSON")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        parser.error("the repository root must be a directory")
    findings = collect_findings(args.root, args.floor)
    if args.json:
        print(json.dumps([asdict(finding) for finding in findings], indent=2))
    else:
        print(f"{'Path':<60} {'Line':>6}  Kind")
        for finding in findings:
            print(f"{finding.path:<60} {finding.line_number:>6}  {finding.kind}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
