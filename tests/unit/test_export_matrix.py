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


def _modernbert_support_rows(
    text: str, heading: str = "## ModernBERT runtime support"
) -> dict[str, dict[str, list[str]]]:
    section = text.split(heading, 1)[1]
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


def test_longformer_support_table_matches_the_export_adapter() -> None:
    from openmed.torch.export_adapters import LONGFORMER_EXPORT_ADAPTER

    documented = _modernbert_support_rows(
        DOC.read_text(encoding="utf-8"), "## Longformer runtime support"
    )
    declared = LONGFORMER_EXPORT_ADAPTER.describe()["support"]
    assert set(documented) == set(declared)
    for runtime, support in declared.items():
        row = documented[runtime]
        for key in ("precisions", "quantizations", "providers", "shape_modes"):
            assert row[key] == support[key], (runtime, key)
        assert row["target"] == [support["target"]]


def test_longformer_bounds_table_matches_the_declared_limits() -> None:
    from types import SimpleNamespace

    from openmed.torch.export_adapters import LONGFORMER_EXPORT_ADAPTER

    text = DOC.read_text(encoding="utf-8")
    section = text.split("### Sequence bounds and memory", 1)[1].split("### ", 1)[0]
    base = SimpleNamespace(
        num_hidden_layers=12,
        num_attention_heads=12,
        attention_window=512,
        max_position_embeddings=4098,
        pad_token_id=1,
    )
    rows = 0
    for line in section.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 4 or cells[0] in {"Runtime", "---"}:
            continue
        runtime, precision, ceiling, buckets = cells
        limits = LONGFORMER_EXPORT_ADAPTER.limits_for(base, runtime, precision)
        assert int(ceiling) * 1024**2 == limits.memory_ceiling_bytes, line
        assert [int(value) for value in buckets.split(", ")] == list(
            limits.supported_buckets
        ), line
        rows += 1
    assert rows == 3
    assert "longformer" in {
        token for token in _backtick_tokens(text) if token.islower()
    }


def test_matrix_lists_every_export_qualification_reason_code() -> None:
    from openmed.eval.export_qualification import ExportReason

    tokens = _backtick_tokens(DOC.read_text(encoding="utf-8"))
    missing = {reason.value for reason in ExportReason} - tokens
    assert not missing, f"undocumented reason codes: {sorted(missing)}"
