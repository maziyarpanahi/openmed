#!/usr/bin/env python3
"""Preflight checks before creating an OpenMed release tag.

The publish workflow is tag-driven, so the tag must match the package version.
Run this before `git tag vX.Y.Z` to avoid reusing an existing release tag.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ABOUT_FILE = ROOT / "openmed" / "__about__.py"
CHANGELOG_FILE = ROOT / "CHANGELOG.md"


def read_package_version() -> str:
    content = ABOUT_FILE.read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', content)
    if not match:
        raise RuntimeError(f"Could not find __version__ in {ABOUT_FILE}")
    return match.group(1)


def read_top_changelog_version() -> str:
    content = CHANGELOG_FILE.read_text(encoding="utf-8")
    match = re.search(r"^## \[(\d+\.\d+\.\d+)\]", content, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not find a released version in {CHANGELOG_FILE}")
    return match.group(1)


def run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def tag_exists_locally(tag: str) -> bool:
    result = run_git(["rev-parse", "-q", "--verify", f"refs/tags/{tag}"])
    return result.returncode == 0


def tag_exists_on_origin(tag: str) -> bool:
    result = run_git(["ls-remote", "--exit-code", "--tags", "origin", f"refs/tags/{tag}"])
    return result.returncode == 0


def has_text(path: str, expected: str) -> bool:
    file_path = ROOT / path
    content = file_path.read_text(encoding="utf-8")
    return expected in content


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        help="Expected release version. Defaults to openmed.__version__.",
    )
    parser.add_argument(
        "--skip-origin-tag-check",
        action="store_true",
        help="Skip checking whether the release tag already exists on origin.",
    )
    args = parser.parse_args()

    package_version = read_package_version()
    changelog_version = read_top_changelog_version()
    expected_version = args.version or package_version
    tag = f"v{expected_version}"

    checks = [
        (package_version == expected_version, f"package version is {package_version}"),
        (changelog_version == expected_version, f"top CHANGELOG release is {changelog_version}"),
        (not tag_exists_locally(tag), f"local tag {tag} is not already used"),
    ]

    if not args.skip_origin_tag_check:
        checks.append((not tag_exists_on_origin(tag), f"origin tag {tag} is not already used"))

    for path, expected in (
        ("README.md", f'from: "{expected_version}"'),
        ("docs/swift-openmedkit.md", f'from: "{expected_version}"'),
        ("docs/website/index.html", f"OpenMed {expected_version}"),
        ("swift/OpenMedDemo/OpenMedDemo/Info.plist", f"<string>{expected_version}</string>"),
        ("swift/OpenMedScanDemo/OpenMedScanDemo/Info.plist", f"<string>{expected_version}</string>"),
    ):
        checks.append(
            (
                has_text(path, expected),
                f"{path} references {expected_version}",
            )
        )

    failures = [message for passed, message in checks if not passed]
    if failures:
        print("Release version preflight failed:", file=sys.stderr)
        for message in failures:
            print(f"- {message}", file=sys.stderr)
        return 1

    print(f"Release version preflight passed for {tag}")
    for _, message in checks:
        print(f"- {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
