#!/usr/bin/env python3
"""Run pip-audit with a reviewed ignore list and fixable-CVE gate."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IGNORE_FILE = ROOT / "docs/security/pip-audit-ignore.toml"
DEFAULT_REPORT = ROOT / "pip-audit-report.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ignore-file",
        type=Path,
        default=DEFAULT_IGNORE_FILE,
        help="TOML file with reviewed vulnerability ignores.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="Path for the pip-audit JSON report.",
    )
    return parser.parse_args()


def parse_review_date(value: object, vuln_id: str) -> dt.date:
    """Return a review date from a TOML date or ISO date string."""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        try:
            return dt.date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                f"ignore entry {vuln_id!r} has invalid review_by date {value!r}"
            ) from exc
    raise ValueError(f"ignore entry {vuln_id!r} must include review_by as YYYY-MM-DD")


def load_ignores(path: Path, today: dt.date) -> set[str]:
    """Load non-expired vulnerability IDs from the ignore list."""
    if not path.exists():
        raise FileNotFoundError(f"ignore file not found: {path}")

    data = tomllib.loads(path.read_text())
    entries = data.get("ignore", [])
    if not isinstance(entries, list):
        raise ValueError("'ignore' must be a TOML array")

    ignored: set[str] = set()
    errors: list[str] = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            errors.append(f"ignore entry #{index} must be a table")
            continue

        vuln_id = str(entry.get("id", "")).strip()
        reason = str(entry.get("reason", "")).strip()
        review_by = entry.get("review_by")

        if not vuln_id:
            errors.append(f"ignore entry #{index} is missing id")
            continue
        if not reason:
            errors.append(f"ignore entry {vuln_id!r} is missing reason")
        try:
            review_date = parse_review_date(review_by, vuln_id)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if review_date < today:
            errors.append(
                f"ignore entry {vuln_id!r} expired on {review_date.isoformat()}"
            )
        ignored.add(vuln_id)

    if errors:
        raise ValueError("\n".join(errors))
    return ignored


def run_pip_audit(report_path: Path) -> dict[str, Any]:
    """Run pip-audit and return its JSON payload."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.unlink(missing_ok=True)
    command = [
        sys.executable,
        "-m",
        "pip_audit",
        "--format",
        "json",
        "--output",
        str(report_path),
        "--progress-spinner",
        "off",
    ]

    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if not report_path.exists():
        raise RuntimeError("pip-audit did not write a JSON report")
    if result.returncode not in (0, 1):
        raise RuntimeError(f"pip-audit exited with status {result.returncode}")

    try:
        return json.loads(report_path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"pip-audit wrote invalid JSON: {exc}") from exc


def fixable_vulnerabilities(report: dict[str, Any]) -> list[tuple[str, str, list[str]]]:
    """Return vulnerabilities that have at least one known fixed version."""
    fixable: list[tuple[str, str, list[str]]] = []
    for name, vuln_id, fix_versions in iter_vulnerabilities(report):
        if fix_versions:
            fixable.append((name, vuln_id, fix_versions))
    return fixable


def iter_vulnerabilities(report: dict[str, Any]) -> list[tuple[str, str, list[str]]]:
    """Return all vulnerabilities from a pip-audit report."""
    vulnerabilities: list[tuple[str, str, list[str]]] = []
    for dependency in report.get("dependencies", []):
        name = dependency.get("name", "<unknown>")
        for vulnerability in dependency.get("vulns", []):
            vulnerabilities.append(
                (
                    name,
                    vulnerability.get("id", "<unknown>"),
                    vulnerability.get("fix_versions") or [],
                )
            )
    return vulnerabilities


def unignored_unfixable_vulnerabilities(
    report: dict[str, Any],
    ignored_ids: set[str],
) -> list[tuple[str, str]]:
    """Return unfixable vulnerabilities that are missing an active ignore."""
    unignored: list[tuple[str, str]] = []
    for name, vuln_id, fix_versions in iter_vulnerabilities(report):
        if not fix_versions and vuln_id not in ignored_ids:
            unignored.append((name, vuln_id))
    return unignored


def main() -> int:
    """Run the dependency vulnerability gate."""
    args = parse_args()
    try:
        ignored_ids = load_ignores(args.ignore_file, dt.date.today())
        report = run_pip_audit(args.report)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"pip-audit gate failed: {exc}", file=sys.stderr)
        return 1

    fixable = fixable_vulnerabilities(report)
    unignored_unfixable = unignored_unfixable_vulnerabilities(report, ignored_ids)
    if not fixable and not unignored_unfixable:
        print("pip-audit gate passed: no unreviewed vulnerabilities found")
        return 0

    if fixable:
        print("pip-audit gate failed: fixable vulnerabilities found", file=sys.stderr)
        for package, vuln_id, fix_versions in fixable:
            print(
                f"- {package}: {vuln_id} fixed by {', '.join(fix_versions)}",
                file=sys.stderr,
            )
    if unignored_unfixable:
        print(
            "pip-audit gate failed: unfixable vulnerabilities need reviewed ignores",
            file=sys.stderr,
        )
        for package, vuln_id in unignored_unfixable:
            print(f"- {package}: {vuln_id}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
