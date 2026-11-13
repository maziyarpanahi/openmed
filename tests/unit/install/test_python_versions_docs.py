"""Keep the Python 3.14 extras table in sync with ``pyproject.toml``."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PAGE = ROOT / "docs" / "install" / "python-versions.md"
PYPROJECT = ROOT / "pyproject.toml"

_ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|(.*)\|\s*$")
_STATUSES = {"install", "fail", "platform-limited"}


def _declared_extras() -> set[str]:
    with PYPROJECT.open("rb") as handle:
        project = tomllib.load(handle)["project"]
    return set(project["optional-dependencies"])


def _rows() -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for line in PAGE.read_text(encoding="utf-8").splitlines():
        match = _ROW.match(line.strip())
        if match:
            cells = [cell.strip() for cell in match.group(2).split("|")]
            rows[match.group(1)] = cells
    return rows


def test_page_lists_every_declared_extra_exactly() -> None:
    assert set(_rows()) == _declared_extras()


def test_every_row_uses_a_known_status_and_explains_failures() -> None:
    for extra, cells in _rows().items():
        linux, macos, notes = cells[0], cells[1], cells[2]
        assert {linux, macos} <= _STATUSES, extra
        if "fail" in (linux, macos):
            assert "https://" in notes or "no wheel published" in notes, extra


def test_page_names_the_check_date_and_command() -> None:
    text = PAGE.read_text(encoding="utf-8")
    assert re.search(r"\b20\d\d-\d\d-\d\d\b", text)
    assert "uv pip install --dry-run --python 3.14" in text
