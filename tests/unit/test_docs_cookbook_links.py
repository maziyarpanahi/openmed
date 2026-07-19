"""Keep the task-oriented cookbook linked to real repository assets."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
COOKBOOK = ROOT / "docs" / "cookbook.md"
EXAMPLES_DOC = ROOT / "docs" / "examples.md"
MKDOCS = ROOT / "mkdocs.yml"

MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
SOURCE_PREFIX = "https://github.com/maziyarpanahi/openmed/blob/master/"
REQUIRED_ASSETS = {
    "examples/clinical_ner_families.py",
    "examples/notebooks/Deidentification_Cookbook.ipynb",
    "examples/pii_batch_processing.py",
    "examples/pii_model_comparison.py",
}


def _linked_example_paths() -> list[str]:
    text = COOKBOOK.read_text(encoding="utf-8")
    return [
        unquote(target.removeprefix(SOURCE_PREFIX).split("#", 1)[0])
        for target in MARKDOWN_LINK.findall(text)
        if target.startswith(f"{SOURCE_PREFIX}examples/")
    ]


def test_every_linked_example_asset_exists() -> None:
    paths = _linked_example_paths()

    assert paths, "cookbook must link at least one repository example"
    missing = [
        relative_path for relative_path in paths if not (ROOT / relative_path).is_file()
    ]
    assert not missing, f"cookbook links missing example assets: {missing}"


def test_cookbook_covers_required_assets_and_rest_recipes() -> None:
    text = COOKBOOK.read_text(encoding="utf-8")

    assert REQUIRED_ASSETS <= set(_linked_example_paths())
    assert "[REST API Recipes](./rest-recipes.md)" in text


def test_cookbook_is_in_nav_and_cross_linked_from_examples() -> None:
    assert "Cookbook: cookbook.md" in MKDOCS.read_text(encoding="utf-8")
    assert "[Cookbook](./cookbook.md)" in EXAMPLES_DOC.read_text(encoding="utf-8")
