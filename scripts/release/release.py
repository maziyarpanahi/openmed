#!/usr/bin/env python3
"""Version bump helper for OpenMed.

This script only bumps `openmed/__about__.py` and does not build or publish.
Publishing remains handled by Make targets / CI workflows.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ABOUT_FILE = Path("openmed/__about__.py")
VERSION_PATTERN = re.compile(r'(__version__\s*=\s*")([^"]+)(")')
VALID_BUMPS = {"patch", "minor", "major"}


def get_current_version() -> str:
    """Read current package version from openmed/__about__.py."""
    content = ABOUT_FILE.read_text(encoding="utf-8")
    match = VERSION_PATTERN.search(content)
    if not match:
        raise ValueError(f"Could not find __version__ in {ABOUT_FILE}")
    return match.group(2)


def bump_version(current_version: str, bump_type: str) -> str:
    """Bump version according to semantic versioning."""
    major, minor, patch = map(int, current_version.split("."))

    if bump_type == "major":
        major += 1
        minor = 0
        patch = 0
    elif bump_type == "minor":
        minor += 1
        patch = 0
    else:
        patch += 1

    return f"{major}.{minor}.{patch}"


def update_version_file(new_version: str) -> None:
    """Write new version to openmed/__about__.py."""
    content = ABOUT_FILE.read_text(encoding="utf-8")
    updated, count = VERSION_PATTERN.subn(rf"\g<1>{new_version}\3", content, count=1)
    if count != 1:
        raise ValueError(f"Failed to update __version__ in {ABOUT_FILE}")
    ABOUT_FILE.write_text(updated, encoding="utf-8")


def main() -> int:
    bump_type = sys.argv[1].lower() if len(sys.argv) > 1 else "patch"
    if bump_type not in VALID_BUMPS:
        print("Usage: python scripts/release/release.py [major|minor|patch]")
        return 1

    current_version = get_current_version()
    new_version = bump_version(current_version, bump_type)
    update_version_file(new_version)

    print(f"Current version: {current_version}")
    print(f"New version: {new_version}")
    print("Updated: openmed/__about__.py")
    print("Next: run `make build` and publish via tag-triggered CI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
