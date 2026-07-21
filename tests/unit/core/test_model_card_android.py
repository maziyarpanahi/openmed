"""Tests for Android metadata in manifest-rendered model cards."""

from __future__ import annotations

from openmed.core.model_card import render_model_card


def _android_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "repo_id": "OpenMed/android-fixture-v1-onnx-android",
        "family": "NER",
        "task": "token-classification",
        "languages": ["en"],
        "tier": "Tiny",
        "param_count": 65_000_000,
        "architecture": "distilbert",
        "base_model": "OpenMed/android-fixture-v1",
        "formats": ["onnx-android", "onnx-int8", "ort-android"],
        "canonical_labels": ["PERSON", "DATE"],
        "benchmark": {"dataset": None, "micro_f1": None, "recall": None},
        "arxiv": "2508.01630",
        "license": "apache-2.0",
        "reproducibility_hash": "sha256:" + "1" * 64,
        "released": "2026-07-18",
    }
    row.update(overrides)
    return row


def test_model_card_renders_android_metadata_section() -> None:
    card = render_model_card(
        _android_row(
            nnapi_compatible=True,
            min_sdk=27,
            execution_providers=["NNAPI", "XNNPACK"],
            tokenizer_assets=["tokenizer.json", "tokenizer_config.json"],
        )
    )

    assert "\n## Android\n" in card
    assert "| Formats | `onnx-android`, `onnx-int8`, `ort-android` |" in card
    assert "| NNAPI compatible | yes |" in card
    assert "| Minimum Android SDK | 27 |" in card
    assert "| Execution providers | `NNAPI`, `XNNPACK` |" in card
    assert "| Tokenizer assets | `tokenizer.json`, `tokenizer_config.json` |" in card
    assert "Outputs are not for autonomous clinical decisions." in card


def test_model_card_omits_android_metadata_section_when_fields_are_absent() -> None:
    card = render_model_card(_android_row())

    assert "\n## Android\n" not in card
    assert "Outputs are not for autonomous clinical decisions." not in card
