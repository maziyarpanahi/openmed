"""Structural validation for the de-identification cookbook notebook.

The notebook's model-calling cells are not executed in CI (out of scope); these
tests confirm the notebook is valid JSON and has the expected recipe structure.
"""

from __future__ import annotations

import json
from pathlib import Path

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
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _cell_text(cell):
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return source


def _notebook_text():
    return "\n".join(_cell_text(cell) for cell in _load()["cells"])


def test_notebook_exists():
    assert NOTEBOOK.exists()


def test_notebook_is_valid_structure():
    nb = _load()
    assert nb["nbformat"] == 4
    assert isinstance(nb.get("nbformat_minor"), int)
    assert isinstance(nb.get("metadata"), dict)
    assert isinstance(nb.get("cells"), list)
    assert nb["cells"]

    for cell in nb["cells"]:
        assert cell["cell_type"] in {"markdown", "code"}
        assert isinstance(cell.get("metadata"), dict)
        assert isinstance(cell.get("source"), list)
        assert all(isinstance(line, str) for line in cell["source"])


def test_notebook_has_four_recipes():
    nb = _load()
    markdown = "\n".join(
        _cell_text(cell) for cell in nb["cells"] if cell["cell_type"] == "markdown"
    )
    for header in REQUIRED_SECTIONS:
        assert header in markdown, f"missing recipe section: {header}"


def test_notebook_uses_synthetic_data_disclaimer():
    assert "Synthetic data only" in _notebook_text()


def test_notebook_uses_expected_public_apis():
    source = _notebook_text()
    for snippet in [
        "deidentify(",
        "reidentify(",
        "BatchProcessor(",
        "process_directory(",
        "DEFAULT_PII_MODELS",
        'method="replace"',
        "keep_mapping=True",
        "consistent=True",
    ]:
        assert snippet in source


def test_batch_recipe_writes_redacted_files():
    source = _notebook_text()
    assert "notes_deidentified" in source
    assert "write_text(item.result.deidentified_text" in source


def test_notebook_outputs_are_cleared():
    nb = _load()
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("outputs") == []
            assert cell.get("execution_count") is None


def test_readme_links_cookbook():
    assert "Deidentification_Cookbook.ipynb" in NB_README.read_text(encoding="utf-8")
