"""Regression tests for the task-oriented cookbook index."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COOKBOOK = ROOT / "docs" / "cookbook.md"
EXAMPLE_LINK = re.compile(
    r"\(https://github\.com/maziyarpanahi/openmed/blob/master/"
    r"(?P<path>examples/[^)#?]+)\)"
)


def _linked_example_paths() -> set[str]:
    content = COOKBOOK.read_text(encoding="utf-8")
    return {match.group("path") for match in EXAMPLE_LINK.finditer(content)}


def test_every_cookbook_example_link_exists() -> None:
    linked = _linked_example_paths()

    assert linked
    for relative_path in sorted(linked):
        path = ROOT / relative_path
        assert path.is_file(), f"Cookbook example does not exist: {relative_path}"


def test_cookbook_indexes_current_scripts_and_notebooks() -> None:
    top_level_scripts = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "examples").glob("*.py")
        if path.name != "__init__.py"
    }
    notebooks = {
        path.relative_to(ROOT).as_posix()
        for pattern in ("*.ipynb", "*.py")
        for path in (ROOT / "examples" / "notebooks").glob(pattern)
    }
    expected = top_level_scripts | notebooks

    assert expected <= _linked_example_paths()


def test_cookbook_is_navigated_and_cross_linked() -> None:
    mkdocs = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    examples = (ROOT / "docs" / "examples.md").read_text(encoding="utf-8")
    cookbook = COOKBOOK.read_text(encoding="utf-8")

    assert "Task-oriented Cookbook: cookbook.md" in mkdocs
    assert "[Task-oriented Cookbook](./cookbook.md)" in examples
    assert "[REST recipes](./rest-recipes.md)" in cookbook
