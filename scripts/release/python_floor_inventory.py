#!/usr/bin/env python3
"""Inventory version guards and markers that a Python floor makes stale."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Finding:
    """A version-related mention below the floor."""

    path: str
    line_number: int
    kind: str
    snippet: str


def parse_floor(floor: str) -> tuple[int, int]:
    """Parse '3.12' into (3, 12)."""
    parts = floor.split(".")
    if len(parts) < 2:
        raise ValueError(f"Invalid floor: {floor}")
    return int(parts[0]), int(parts[1])


def version_below_floor(major: int, minor: int, floor_major: int, floor_minor: int) -> bool:
    return (major, minor) < (floor_major, floor_minor)


VERSION_INFO_RE = re.compile(
    r"sys\.version_info\s*(?:>=|<=|>|<|==|!=)\s*\(\s*(\d+)\s*,\s*(\d+)",
    re.IGNORECASE,
)
PYTHON_VERSION_MARKER_RE = re.compile(
    r"python_version\s*(?:>=|<=|>|<|==|!=)\s*['"](\d+)\.(\d+)",
    re.IGNORECASE,
)
PYTHON_VERSION_WORKFLOW_RE = re.compile(
    r"python-version:\s*['"]?(\d+)\.(\d+)",
    re.IGNORECASE,
)
DOC_VERSION_RE = re.compile(
    r"(3\.\d+)",
    re.IGNORECASE,
)


def scan_python(path: Path, root: Path, floor: tuple[int, int]) -> List[Finding]:
    findings: List[Finding] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return findings
    for i, line in enumerate(text.splitlines(), 1):
        for m in VERSION_INFO_RE.finditer(line):
            maj, mino = int(m.group(1)), int(m.group(2))
            if version_below_floor(maj, mino, *floor):
                findings.append(
                    Finding(
                        path=str(path.relative_to(root)),
                        line_number=i,
                        kind="version_info_guard",
                        snippet=line.strip()[:120],
                    )
                )
        for m in PYTHON_VERSION_MARKER_RE.finditer(line):
            maj, mino = int(m.group(1)), int(m.group(2))
            if version_below_floor(maj, mino, *floor):
                findings.append(
                    Finding(
                        path=str(path.relative_to(root)),
                        line_number=i,
                        kind="python_version_marker",
                        snippet=line.strip()[:120],
                    )
                )
    return findings


def scan_toml(path: Path, root: Path, floor: tuple[int, int]) -> List[Finding]:
    findings: List[Finding] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return findings
    for i, line in enumerate(text.splitlines(), 1):
        for m in PYTHON_VERSION_MARKER_RE.finditer(line):
            maj, mino = int(m.group(1)), int(m.group(2))
            if version_below_floor(maj, mino, *floor):
                findings.append(
                    Finding(
                        path=str(path.relative_to(root)),
                        line_number=i,
                        kind="python_version_marker",
                        snippet=line.strip()[:120],
                    )
                )
        if "requires-python" in line and "3." in line:
            m = re.search(r"(\d+)\.(\d+)", line)
            if m:
                maj, mino = int(m.group(1)), int(m.group(2))
                if version_below_floor(maj, mino, *floor):
                    findings.append(
                        Finding(
                            path=str(path.relative_to(root)),
                            line_number=i,
                            kind="requires_python",
                            snippet=line.strip()[:120],
                        )
                    )
    return findings


def scan_yaml(path: Path, root: Path, floor: tuple[int, int]) -> List[Finding]:
    findings: List[Finding] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return findings
    for i, line in enumerate(text.splitlines(), 1):
        for m in PYTHON_VERSION_WORKFLOW_RE.finditer(line):
            maj, mino = int(m.group(1)), int(m.group(2))
            if version_below_floor(maj, mino, *floor):
                findings.append(
                    Finding(
                        path=str(path.relative_to(root)),
                        line_number=i,
                        kind="workflow_python_version",
                        snippet=line.strip()[:120],
                    )
                )
    return findings


def scan_markdown(path: Path, root: Path, floor: tuple[int, int]) -> List[Finding]:
    findings: List[Finding] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return findings
    for i, line in enumerate(text.splitlines(), 1):
        for m in DOC_VERSION_RE.finditer(line):
            ver = m.group(1)
            try:
                maj, mino = map(int, ver.split("."))
                if version_below_floor(maj, mino, *floor):
                    findings.append(
                        Finding(
                            path=str(path.relative_to(root)),
                            line_number=i,
                            kind="docs_mention",
                            snippet=line.strip()[:120],
                        )
                    )
            except ValueError:
                continue
    return findings


def collect_findings(root: Path, floor_str: str) -> List[Finding]:
    floor = parse_floor(floor_str)
    findings: List[Finding] = []

    for py in root.rglob("*.py"):
        if any(p in py.parts for p in (".git", ".venv", "node_modules", "__pycache__", "dist", "build")):
            continue
        findings.extend(scan_python(py, root, floor))

    for toml in root.rglob("pyproject.toml"):
        findings.extend(scan_toml(toml, root, floor))

    workflows = root / ".github" / "workflows"
    if workflows.exists():
        for yml in list(workflows.glob("*.yml")) + list(workflows.glob("*.yaml")):
            findings.extend(scan_yaml(yml, root, floor))

    for md in list(root.rglob("*.md")):
        if any(p in md.parts for p in (".git", "node_modules")):
            continue
        findings.extend(scan_markdown(md, root, floor))

    return sorted(findings, key=lambda f: (f.path, f.line_number))


def print_table(findings: List[Finding]) -> None:
    if not findings:
        print("No findings below the floor.")
        return
    print(f"{'Path':<60} {'Line':>5} {'Kind':<25} Snippet")
    print("-" * 120)
    for f in findings:
        print(f"{f.path:<60} {f.line_number:>5} {f.kind:<25} {f.snippet}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floor", default="3.12", help="Python version floor, e.g. 3.12")
    parser.add_argument("--json", action="store_true", help="Emit JSON (paths, lines, kinds only)")
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root")
    args = parser.parse_args(argv)

    findings = collect_findings(args.root, args.floor)
    if args.json:
        payload = [
            {"path": f.path, "line_number": f.line_number, "kind": f.kind} for f in findings
        ]
        json.dump(payload, sys.stdout, indent=2)
        print()
    else:
        print_table(findings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
