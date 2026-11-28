"""The Python 3.14 extras page must list every optional extra."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _extras() -> list[str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return sorted(data["project"]["optional-dependencies"])


def test_python_versions_page_lists_every_extra() -> None:
    page = (ROOT / "docs" / "install" / "python-versions.md").read_text(
        encoding="utf-8"
    )
    missing = [name for name in _extras() if f"`openmed[{name}]`" not in page]
    assert not missing, (
        f"extras missing from docs/install/python-versions.md: {missing}"
    )


def test_python_versions_page_is_in_the_nav() -> None:
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "install/python-versions.md" in nav
