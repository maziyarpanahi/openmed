"""Tests for the documented status and publication contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from openmed.eval.report import BenchmarkReport, render_benchmark_card
from scripts.status.generate_shield_synthetic_baseline import generate_report

ROOT = Path(__file__).resolve().parents[3]
STATUS_CONTRACT = ROOT / "docs" / "status" / "trust-status-contract.md"
PUBLICATION_PLAN = ROOT / "docs" / "status" / "open-benchmark-publication.md"
SHIELD_REPORT = ROOT / "docs" / "benchmarks" / "shield-synthetic.report.json"
SHIELD_CARD = ROOT / "docs" / "benchmarks" / "shield-synthetic.md"


def test_synthetic_baseline_hashes_ignore_checkout_newlines(tmp_path: Path) -> None:
    from scripts.status.generate_shield_synthetic_baseline import text_file_digest

    lf = tmp_path / "lf.txt"
    crlf = tmp_path / "crlf.txt"
    changed = tmp_path / "changed.txt"
    lf.write_bytes(b"synthetic\nfixture\n")
    crlf.write_bytes(b"synthetic\r\nfixture\r\n")
    changed.write_bytes(b"synthetic\nchanged\n")
    assert text_file_digest(lf) == text_file_digest(crlf)
    assert text_file_digest(lf) != text_file_digest(changed)


def test_synthetic_card_rejects_unpinned_command_revision() -> None:
    report = BenchmarkReport.read_json(SHIELD_REPORT)
    report.metadata["source_revision"] = "main; echo arbitrary-command"
    with pytest.raises(ValueError, match="source revision"):
        render_benchmark_card(report)


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


def test_committed_synthetic_shield_evidence_is_reproducible(monkeypatch) -> None:
    committed = BenchmarkReport.read_json(SHIELD_REPORT)
    # Reproduce the historical report with its original catalog dimensions.
    # New, unevaluated language routes add zero-support buckets to metrics;
    # they must not invalidate immutable evidence or imply new coverage.
    import openmed.eval.metrics as metrics

    monkeypatch.setattr(
        metrics,
        "SUPPORTED_LANGUAGES",
        tuple(committed.metrics["leakage"]["total_chars_by_language"]),
    )
    source_revision = str(committed.metadata["source_revision"])
    regenerated = generate_report(
        source_revision=source_revision,
        generated_at=committed.generated_at,
    )

    assert committed.suite == "shield-synthetic"
    assert committed.fixture_count == 2
    assert committed.metadata["source_rights"] == (
        "OpenMed-generated synthetic fixture; Apache-2.0"
    )
    assert (
        committed.metadata["reproducibility_hash"]
        == (regenerated.metadata["reproducibility_hash"])
    )
    assert committed.metrics["exact_span_f1"] == regenerated.metrics["exact_span_f1"]
    assert (
        committed.metrics["leakage"]["overall"]
        == (regenerated.metrics["leakage"]["overall"])
    )
    for name, metric in committed.metrics.items():
        if name not in {"latency", "resources"}:
            assert metric == regenerated.metrics[name]
    assert SHIELD_CARD.read_text(encoding="utf-8") == render_benchmark_card(committed)
    assert "not clinical model performance" in SHIELD_CARD.read_text(encoding="utf-8")
