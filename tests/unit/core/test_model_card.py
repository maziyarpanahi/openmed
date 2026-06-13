"""Tests for the manifest-driven model card renderer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.core.model_card import render_model_card
from openmed.core import model_card
from openmed.core.hf_publish import publish_model_card

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    "param_count, expected",
    [
        (135_000_000, "135M"),
        (560_000_000, "560M"),
        (1_500_000_000, "1.5B"),
        (135_000, "135K"),
        (512, "512"),
    ],
)
def test_format_params(param_count, expected):
    assert model_card._format_params(param_count) == expected


def test_format_params_none_returns_none():
    assert model_card._format_params(None) is None


def test_generic_quickstart_used_for_unknown_task():
    row = {
        "repo_id": "OpenMed/falcon-ocr-bf16-mlx",
        "family": "Vision",
        "task": "image-text-to-text",
        "languages": ["en"],
        "tier": None,
        "param_count": None,
        "architecture": "falcon-ocr",
        "base_model": "tiiuae/Falcon-OCR",
        "formats": ["mlx-fp", "pytorch"],
        "canonical_labels": [],
        "benchmark": {"dataset": None, "micro_f1": None, "recall": None},
        "arxiv": None,
        "license": "apache-2.0",
        "reproducibility_hash": "sha256:" + "7" * 64,
        "released": "2026-05-02",
    }
    card = render_model_card(row)
    assert 'pipeline("image-text-to-text", model="OpenMed/falcon-ocr-bf16-mlx")' in card
    assert "## 📜 Citation" not in card
    assert "arXiv-" not in card
    assert "## 🏷️ Entity Labels" not in card
    assert "aggregation_strategy" not in card


@pytest.mark.parametrize("name", ["sparse_ner", "populated_ner"])
def test_render_matches_golden(name):
    row = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    expected = (FIXTURES / f"{name}.md").read_text(encoding="utf-8")
    assert render_model_card(row) == expected


def test_publish_uploads_rendered_card_as_readme():
    row = json.loads((FIXTURES / "sparse_ner.json").read_text(encoding="utf-8"))
    captured: dict = {}

    class RecordingApi:
        def upload_file(self, **kwargs):
            captured.update(kwargs)
            return "https://huggingface.co/OpenMed/x/commit/deadbeef"

    result = publish_model_card(row, api=RecordingApi())

    assert captured["path_in_repo"] == "README.md"
    assert captured["repo_id"] == row["repo_id"]
    assert captured["repo_type"] == "model"
    assert captured["path_or_fileobj"].decode("utf-8") == render_model_card(row)
    assert captured["commit_message"] == "Update auto-generated model card"
    assert result.endswith("deadbeef")
