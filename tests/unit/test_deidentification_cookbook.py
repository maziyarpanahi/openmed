"""Structural validation for the de-identification cookbook notebook.

The notebook's model-calling cells are not executed in CI (out of scope); this
test only confirms the notebook is valid and has the expected recipe structure.
"""

from __future__ import annotations

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "examples" / "notebooks" / "Deidentification_Cookbook.ipynb"
NB_README = ROOT / "examples" / "notebooks" / "README.md"

REQUIRED_SECTIONS = [
    "Recipe 1 — De-identify a list or CSV of clinical strings",
    "Recipe 2 — Batch-redact a directory with BatchProcessor",
    "Recipe 3 — Reversible replace + re-identify round-trip",
    "Recipe 4 — Per-language model selection with DEFAULT_PII_MODELS",
]


def _load():
    return nbformat.read(NOTEBOOK, as_version=4)


def test_notebook_exists():
    assert NOTEBOOK.exists()


def test_notebook_is_valid():
    nbformat.validate(_load())  # raises NotebookValidationError if malformed


def test_notebook_has_four_recipes():
    nb = _load()
    markdown = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    for header in REQUIRED_SECTIONS:
        assert header in markdown, f"missing recipe section: {header}"


def test_notebook_uses_synthetic_data_disclaimer():
    nb = _load()
    markdown = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    assert "Synthetic data only" in markdown


def test_notebook_outputs_are_cleared():
    nb = _load()
    for cell in nb.cells:
        if cell.cell_type == "code":
            assert cell.get("outputs") == []
            assert cell.get("execution_count") is None


def test_readme_links_cookbook():
    assert "Deidentification_Cookbook.ipynb" in NB_README.read_text(encoding="utf-8")
