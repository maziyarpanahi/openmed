from __future__ import annotations

import re
from pathlib import Path

from openmed.mlx.models import _SUPPORTED_TOKEN_CLASSIFICATION_MODEL_TYPES


DOC_PATH = Path(__file__).resolve().parents[2] / "docs" / "export-matrix.md"


def _matrix_family_rows() -> set[str]:
    text = DOC_PATH.read_text(encoding="utf-8")
    rows: set[str] = set()
    for line in text.splitlines():
        match = re.match(r"^\|\s*`([^`]+)`\s*\|", line)
        if match:
            rows.add(match.group(1))
    return rows


def test_export_matrix_covers_every_mlx_token_classification_family() -> None:
    rows = _matrix_family_rows()

    assert set(_SUPPORTED_TOKEN_CLASSIFICATION_MODEL_TYPES).issubset(rows)


def test_export_matrix_documents_current_platform_limitations() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")

    assert "GGUF is embedding-only" in text
    assert "CoreML is token-classification-only" in text
    assert "INT4 only-if-recall-holds" in text
