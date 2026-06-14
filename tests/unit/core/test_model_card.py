"""Tests for manifest-rendered model cards."""

from __future__ import annotations

from pathlib import Path

from openmed.core.model_card import render_model_card, write_model_card


ROOT = Path(__file__).resolve().parents[3]
GOLDEN = ROOT / "tests" / "fixtures" / "model_card_expected.md"


def _fixture_row():
    return {
        "repo_id": "OpenMed/OpenMed-PII-Turkish-SuperClinical-Small-44M-v1-mlx",
        "family": "PII",
        "task": "token-classification",
        "languages": ["tr"],
        "tier": "Small",
        "param_count": 44_000_000,
        "architecture": "deberta-v2",
        "base_model": "OpenMed/OpenMed-PII-Turkish-SuperClinical-Small-44M-v1",
        "formats": ["mlx-fp", "mlx-8bit"],
        "canonical_labels": ["PERSON", "DATE", "ID_NUM"],
        "benchmark": {
            "dataset": "openmed-golden-pii",
            "micro_f1": 0.9823,
            "recall": 0.991,
        },
        "arxiv": "2508.01630",
        "license": "apache-2.0",
        "reproducibility_hash": (
            "sha256:1111111111111111111111111111111111111111111111111111111111111111"
        ),
        "released": "2026-06-14",
    }


def test_render_model_card_matches_golden_fixture():
    assert render_model_card(_fixture_row()) == GOLDEN.read_text(encoding="utf-8")


def test_render_model_card_contains_manifest_release_fields():
    card = render_model_card(_fixture_row())

    assert "| Tier | Small |" in card
    assert "| Parameters | 44M (44,000,000) |" in card
    assert "| Formats | mlx-fp, mlx-8bit |" in card
    assert "| openmed-golden-pii | 0.9823 | 0.9910 |" in card
    assert "[arXiv:2508.01630]" in card
    assert "| License | apache-2.0 |" in card
    assert _fixture_row()["reproducibility_hash"] in card


def test_write_model_card_writes_readme_from_row(tmp_path):
    target = tmp_path / "README.md"

    path = write_model_card(target, _fixture_row())

    assert path == target
    assert target.read_text(encoding="utf-8") == GOLDEN.read_text(encoding="utf-8")
