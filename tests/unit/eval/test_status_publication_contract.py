"""Tests for the documented status and publication contracts."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STATUS_CONTRACT = ROOT / "docs" / "status" / "trust-status-contract.md"
PUBLICATION_PLAN = ROOT / "docs" / "status" / "open-benchmark-publication.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_status_contract_documents_required_row_fields() -> None:
    text = _read(STATUS_CONTRACT)

    for field in (
        "key",
        "family",
        "tier",
        "device",
        "format",
        "model_count",
        "current_leakage",
        "last_green_release",
        "last_regression",
        "last_rollback",
        "harness_freshness",
        "status",
        "evidence",
        "reproducibility_hash",
    ):
        assert f"`{field}`" in text

    for source_path in (
        "models.jsonl",
        "gates/baseline.json",
        "docs/benchmarks/golden.report.json",
        "docs/status/index.md",
        "docs/leaderboard/index.md",
        "scripts/status/generate_status.py",
    ):
        assert source_path in text

    assert "OM-021/#104" in text


def test_publication_plan_is_open_and_reproducible() -> None:
    text = _read(PUBLICATION_PLAN)
    lowered = text.lower()

    assert "not a closed leaderboard" in lowered
    assert "committed result json" in lowered
    assert "hand-edited" in lowered

    for source_path in (
        "models.jsonl",
        "gates/baseline.json",
        "docs/benchmarks/golden.report.json",
        "docs/benchmarks/golden.md",
        "docs/status/trust-status-contract.md",
        "docs/status/index.md",
        "docs/leaderboard/index.md",
    ):
        assert source_path in text


def test_reproducibility_hash_convention_is_pinned() -> None:
    combined = _read(STATUS_CONTRACT) + "\n" + _read(PUBLICATION_PLAN)

    for token in (
        "compute_reproducibility_hash",
        "recipe",
        "data_manifest",
        "base_model",
        "git_sha",
        "sha256:<64 lower hex>",
    ):
        assert token in combined
