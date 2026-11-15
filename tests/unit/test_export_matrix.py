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


def _modernbert_support_rows(text: str) -> dict[str, dict[str, list[str]]]:
    section = text.split("## ModernBERT runtime support", 1)[1]
    section = section.split("### ", 1)[0]
    rows: dict[str, dict[str, list[str]]] = {}
    for line in section.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 6 or cells[0] in {"Runtime", "---"}:
            continue
        runtime, precisions, quantizations, providers, shapes, target = cells
        rows[runtime] = {
            "precisions": precisions.split(", "),
            "quantizations": quantizations.split(", "),
            "providers": providers.split(", "),
            "shape_modes": shapes.split(", "),
            "target": [target],
        }
    return rows


def test_modernbert_support_table_matches_the_export_adapter() -> None:
    from openmed.torch.export_adapters import MODERNBERT_EXPORT_ADAPTER

    documented = _modernbert_support_rows(DOC.read_text(encoding="utf-8"))
    declared = MODERNBERT_EXPORT_ADAPTER.describe()["support"]
    assert set(documented) == set(declared)
    for runtime, support in declared.items():
        row = documented[runtime]
        for key in ("precisions", "quantizations", "providers", "shape_modes"):
            assert row[key] == support[key], (runtime, key)
        assert row["target"] == [support["target"]]


def test_matrix_lists_every_export_qualification_reason_code() -> None:
    from openmed.eval.export_qualification import ExportReason

    tokens = _backtick_tokens(DOC.read_text(encoding="utf-8"))
    missing = {reason.value for reason in ExportReason} - tokens
    assert not missing, f"undocumented reason codes: {sorted(missing)}"
