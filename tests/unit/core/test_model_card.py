"""Tests for manifest-rendered model cards."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.__about__ import __version__
from openmed.core.hf_publish import (
    DEFAULT_MODEL_CARD_COMMIT_MESSAGE,
    publish_model_card,
)
from openmed.core.manifest_schema import SCRIPT_COVERAGE_TARGETS
from openmed.core.model_card import render_model_card, write_model_card
from openmed.core.model_registry import load_manifest_rows

ROOT = Path(__file__).resolve().parents[3]
GOLDEN = ROOT / "tests" / "fixtures" / "model_card_expected.md"
V2_GOLDEN = ROOT / "tests" / "fixtures" / "model_card_v2_expected.md"
CONTRIBUTOR_FIXTURES = Path(__file__).parent / "fixtures"


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


def _v2_fixture_row():
    script_coverage = {
        script: {
            "unk_rate": 0.0,
            "byte_fallback_rate": 0.0,
            "tokens_per_grapheme": 0.75,
            "verdict": (
                "supported"
                if script in {"han_simplified", "han_traditional"}
                else "unclaimed"
            ),
        }
        for script in SCRIPT_COVERAGE_TARGETS
    }
    return {
        **_fixture_row(),
        "repo_id": "OpenMed/OpenMed-PII-Chinese-BigMed-Large-560M-v1",
        "languages": ["zh"],
        "tier": "Large",
        "param_count": 560_000_000,
        "architecture": "xlm-roberta",
        "base_model": "FacebookAI/xlm-roberta-large",
        "formats": ["pytorch"],
        "benchmark": {
            "dataset": "AI4Privacy + Synthetic Chinese PII",
            "micro_f1": 0.7589,
            "recall": 0.7517,
        },
        "download_mb": 2252.843,
        "download_sizes": {
            "safetensors": 2235.723,
            "mlx": 2235.717,
            "coreml": None,
            "onnx": 1330.432,
        },
        "script_eval": {
            "han_simplified": {
                "dataset": "AI4Privacy + Synthetic Chinese PII",
                "recall": 0.7517,
                "leakage_floor": None,
            },
            "han_traditional": {
                "dataset": None,
                "recall": None,
                "leakage_floor": None,
            },
        },
        "script_coverage": script_coverage,
    }


def test_render_model_card_matches_golden_fixture():
    assert render_model_card(_fixture_row()) == GOLDEN.read_text(encoding="utf-8")


def test_render_v2_model_card_matches_golden_fixture():
    assert render_model_card(_v2_fixture_row()) == V2_GOLDEN.read_text(encoding="utf-8")


def test_render_v2_model_card_contains_release_decision_sections():
    card = render_model_card(_v2_fixture_row())

    assert "## Per-Script Evaluation" in card
    assert "| Han Simplified | AI4Privacy + Synthetic Chinese PII | 0.7517 |" in card
    assert "## Download Size by Format" in card
    assert "| Safetensors | 2,235.723 MB |" in card
    assert "| Core ML | Not published |" in card
    assert "## Tokenizer Script Coverage" in card
    assert "| Han Traditional | 0.00% | 0.00% | 0.7500 | supported |" in card
    assert "## License" in card
    assert "Declared license: `apache-2.0`." in card


@pytest.mark.parametrize(
    "repo_id",
    (
        "OpenMed/OpenMed-PII-Bengali-mSuperClinical-Large-279M-v1",
        "OpenMed/OpenMed-PII-Chinese-BigMed-Large-560M-v1",
        "OpenMed/OpenMed-PII-Tamil-mSuperClinical-Large-279M-v1",
    ),
)
def test_committed_v2_entries_render_release_decision_sections(repo_id):
    row = next(row for row in load_manifest_rows() if row["repo_id"] == repo_id)

    card = render_model_card(row)

    assert "## Per-Script Evaluation" in card
    assert "## Download Size by Format" in card
    assert "## Tokenizer Script Coverage" in card
    assert "## License" in card
    assert "Declared license: `apache-2.0`." in card


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


def test_contributor_sparse_fixture_renders_current_model_card():
    row = json.loads(
        (CONTRIBUTOR_FIXTURES / "sparse_ner.json").read_text(encoding="utf-8")
    )
    card = render_model_card(row)

    assert "# OpenMed-NER-AnatomyDetect-TinyMed-135M" in card
    assert "| Repository | `OpenMed/OpenMed-NER-AnatomyDetect-TinyMed-135M` |" in card
    assert "| Parameters | 135M (135,000,000) |" in card
    assert "| Dataset | Micro F1 | Recall |" in card
    assert "| Not reported | Not reported | Not reported |" in card
    assert "`ORGAN`, `TISSUE`, `ANATOMY`" in card


def test_contributor_populated_fixture_renders_current_model_card():
    row = json.loads(
        (CONTRIBUTOR_FIXTURES / "populated_ner.json").read_text(encoding="utf-8")
    )
    card = render_model_card(row)

    assert "# OpenMed-NER-OncologyDetect-ElectraMed-560M" in card
    assert "| Base model | google/electra-large-discriminator |" in card
    assert "| Formats | pytorch, mlx-fp |" in card
    assert "| ONCOLOGY | 0.9063 | 0.9044 |" in card
    assert "`CANCER`, `TREATMENT`" in card


def test_render_model_card_includes_distillation_evidence_when_present():
    row = {
        **_fixture_row(),
        "distillation": {
            "alpha": 0.6,
            "critical_label_drops": ["EMAIL"],
            "per_label_recall_delta": [
                {
                    "critical_drop": True,
                    "delta": -0.02,
                    "label": "EMAIL",
                    "student_recall": 0.97,
                    "teacher_recall": 0.99,
                }
            ],
            "recall_gate_passed": False,
            "student_backbone": "openmed/backbones/tiny-direct-identifier-135m",
            "teacher_id": "teacher-local",
            "temperature": 2.0,
        },
    }

    card = render_model_card(row)

    assert "## Distillation Evidence" in card
    assert "| Teacher | `teacher-local` |" in card
    assert "| Recall gate | failed |" in card
    assert "| Critical drops | `EMAIL` |" in card
    assert "| EMAIL | 0.9900 | 0.9700 | -0.0200 | yes |" in card


def test_render_model_card_includes_training_provenance_when_present():
    row = {
        **_fixture_row(),
        "training_provenance": {
            "base_model_revision": "7b4f2ca",
            "data_manifest_hash": "sha256:" + "a" * 64,
            "env_lock_digest": "sha256:" + "b" * 64,
            "git_sha": "abc123",
            "path": "checkpoints/model/training_provenance.json",
            "recipe_config_hash": "sha256:" + "c" * 64,
            "reproducibility_hash": _fixture_row()["reproducibility_hash"],
            "rng_seeds": {"numpy": 21, "python": 13, "torch": 34},
        },
    }

    card = render_model_card(row)

    assert "## Training Provenance" in card
    assert "| Provenance file | `checkpoints/model/training_provenance.json` |" in card
    assert "| Base model revision | `7b4f2ca` |" in card
    assert "| RNG seeds | `numpy`=21, `python`=13, `torch`=34 |" in card
    assert "| Data manifest hash | `sha256:" in card
    assert "| Provenance reproducibility hash | `" in card


def test_android_onnx_model_card_is_personalized_and_cross_platform():
    row = {
        **_fixture_row(),
        "repo_id": "OpenMed/example-v1-onnx-android",
        "formats": ["onnx-android", "onnx-int8", "ort-android"],
    }

    card = render_model_card(row)

    assert "## OpenMed in Python on CPU" in card
    assert 'pip install --upgrade "openmed[onnx-runtime]"' in card
    assert "openmed[onnx-runtime]>=" not in card
    assert "from openmed import OnnxModel" in card
    assert 'OnnxModel.from_pretrained("OpenMed/example-v1-onnx-android")' in card
    assert "## OpenMed in Web" in card
    assert "npm install openmed @huggingface/transformers onnxruntime-web" in card
    assert 'import { loadOnnxModel } from "openmed";' in card
    assert "@openmed/openmedkit-web" not in card
    assert 'const repo = "OpenMed/example-v1-onnx-android";' in card
    assert "const model = await loadOnnxModel(repo);" in card
    assert "## OpenMedKit for Android" in card
    assert 'url = uri("https://jitpack.io")' in card
    assert f'implementation("com.github.maziyarpanahi:openmed:v{__version__}")' in card
    assert "2,000+ medical models" in card
    assert "master-SNAPSHOT" not in card
    assert "OpenMedKit.fromDirectory(modelDirectory)" in card
    assert "suspend fun analyzeModel()" in card
    assert "# OpenMed PII Detection 44M" in card
    assert "Language | Turkish" in card
    assert "`model_int8.onnx`" in card
    assert "`model_fp16.onnx`" in card
    assert "`model.ort`" in card
    assert "## The OpenMed Ecosystem" in card
    assert "Reproducibility hash" not in card
    assert "sha256:" not in card
    assert "InferenceSession" not in card
    assert "AutoTokenizer" not in card


def test_android_onnx_model_card_identifies_ner_capability():
    row = {
        **_fixture_row(),
        "repo_id": "OpenMed/OpenMed-NER-AnatomyDetect-TinyMed-135M-v1-onnx-android",
        "family": "NER",
        "languages": ["en"],
        "param_count": 135_000_000,
        "architecture": "modernbert",
        "formats": ["onnx-android", "int8"],
        "canonical_labels": ["O", "B-ANATOMY", "I-ANATOMY"],
    }

    card = render_model_card(row)

    assert "# OpenMed Anatomy NER 135M" in card
    assert "extracting anatomy mentions in English" in card
    assert "Entity labels | `ANATOMY`" in card
    assert "The biopsy was taken from the left lung." in card
    assert "`model.ort`" not in card


def test_publish_model_card_uploads_rendered_readme():
    row = _fixture_row()
    captured: dict[str, object] = {}

    class RecordingApi:
        def upload_file(self, **kwargs):
            captured.update(kwargs)
            return "https://example.invalid/commit/deadbeef"

    result = publish_model_card(row, api=RecordingApi(), token="secret-token")

    assert result.endswith("deadbeef")
    assert captured["path_in_repo"] == "README.md"
    assert captured["repo_id"] == row["repo_id"]
    assert captured["repo_type"] == "model"
    assert captured["token"] == "secret-token"
    assert captured["commit_message"] == DEFAULT_MODEL_CARD_COMMIT_MESSAGE
    assert captured["path_or_fileobj"] == render_model_card(row).encode("utf-8")
