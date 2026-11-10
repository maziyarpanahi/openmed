"""Coherence guard for ``docs/export-matrix.md``.

Every alias in the MLX
``openmed.mlx.models._SUPPORTED_TOKEN_CLASSIFICATION_MODEL_TYPES`` allowlist
must appear as a backtick-wrapped token in the matrix so the doc stays honest
as backends are added. The matrix must also capture the two canonical
limitations the docs promise to state verbatim (GGUF is embedding-only; CoreML
is token-classification-only) and be reachable from the primary export docs.
"""

from __future__ import annotations

import re
from pathlib import Path

from openmed.mlx.models import _SUPPORTED_TOKEN_CLASSIFICATION_MODEL_TYPES

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "export-matrix.md"
MKDOCS = ROOT / "mkdocs.yml"
LINKING_DOCS = (
    ROOT / "docs" / "coreml-export.md",
    ROOT / "docs" / "export-mlx-quant.md",
    ROOT / "docs" / "export-onnx-webgpu.md",
    ROOT / "docs" / "export-gguf.md",
)

_BACKTICK = re.compile(r"`([^`]+)`")


def _backtick_tokens(text: str) -> set[str]:
    return {match.strip() for match in _BACKTICK.findall(text)}


def test_matrix_mentions_every_mlx_allowlist_family() -> None:
    text = DOC.read_text(encoding="utf-8")
    tokens = _backtick_tokens(text)
    missing = set(_SUPPORTED_TOKEN_CLASSIFICATION_MODEL_TYPES) - tokens
    assert not missing, (
        "docs/export-matrix.md must mention every MLX-supported family alias "
        f"as a backtick-wrapped token; missing: {sorted(missing)}"
    )


def test_matrix_documents_canonical_limitations() -> None:
    text = DOC.read_text(encoding="utf-8")
    lowered = text.lower()

    assert "gguf" in lowered and "embedding" in lowered, (
        "matrix must document the GGUF-embedding-only limitation"
    )
    assert "coreml" in lowered and "token-classification" in lowered, (
        "matrix must document the CoreML-token-classification-only limitation"
    )
    assert "int4" in lowered, "matrix must reference the INT4 recall gate"
    assert "allowlist" in lowered, "matrix must reference the MLX family allowlist"


def test_matrix_is_in_mkdocs_nav_and_cross_linked() -> None:
    nav = MKDOCS.read_text(encoding="utf-8")
    assert "export-matrix.md" in nav, (
        "docs/export-matrix.md must be listed in mkdocs.yml nav"
    )
    for path in LINKING_DOCS:
        text = path.read_text(encoding="utf-8")
        assert "export-matrix.md" in text, (
            f"{path.relative_to(ROOT)} should link to the export matrix"
        )
