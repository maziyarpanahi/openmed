"""Tests for the OpenAI Privacy Filter MLX runtime path."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest


def _module_importable(module_name: str) -> bool:
    try:
        __import__(module_name)
    except Exception:
        return False
    return True


_MLX_AVAILABLE = _module_importable("mlx.core")


def _privacy_config() -> dict:
    return {
        "model_type": "openai_privacy_filter",
        "encoding": "o200k_base",
        "num_hidden_layers": 1,
        "num_experts": 3,
        "experts_per_token": 2,
        "vocab_size": 16,
        "num_labels": 5,
        "hidden_size": 8,
        "intermediate_size": 4,
        "head_dim": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "sliding_window": 3,
        "bidirectional_context": True,
        "bidirectional_left_context": 1,
        "bidirectional_right_context": 1,
        "initial_context_length": 8,
        "max_position_embeddings": 32,
        "default_n_ctx": 32,
        "rope_theta": 10000.0,
        "rope_scaling_factor": 1.0,
        "rope_ntk_alpha": 1.0,
        "rope_ntk_beta": 32.0,
        "param_dtype": "float32",
        "_mlx_task": "token-classification",
        "_mlx_family": "openai-privacy-filter",
        "_mlx_model_type": "openai-privacy-filter",
        "id2label": {
            "0": "O",
            "1": "B-private_person",
            "2": "I-private_person",
            "3": "E-private_person",
            "4": "S-private_email",
        },
    }


def test_resolves_privacy_filter_model_family():
    from openmed.mlx.models import resolve_model_type

    assert resolve_model_type({"model_type": "openai_privacy_filter"}) == "openai-privacy-filter"
    assert resolve_model_type({"model_type": "privacy_filter"}) == "openai-privacy-filter"
    assert (
        resolve_model_type(
            {"model_type": "bert"},
            manifest={"task": "token-classification", "family": "openai-privacy-filter"},
        )
        == "openai-privacy-filter"
    )


@pytest.mark.skipif(not _MLX_AVAILABLE, reason="MLX is required for model forward tests")
def test_privacy_filter_tiny_forward_shape():
    import mlx.core as mx
    from openmed.mlx.models.privacy_filter import OpenAIPrivacyFilterForTokenClassification

    model = OpenAIPrivacyFilterForTokenClassification(_privacy_config())

    logits = model(
        mx.array([[1, 2, 3, 4]], dtype=mx.int32),
        attention_mask=mx.ones((1, 4), dtype=mx.bool_),
    )
    mx.eval(logits)

    assert logits.shape == (1, 4, 5)


def test_viterbi_rejects_invalid_inside_start():
    from openmed.mlx.inference import _build_label_info, _viterbi_decode

    id2label = {
        0: "O",
        1: "B-private_person",
        2: "I-private_person",
        3: "E-private_person",
        4: "S-private_person",
    }
    label_info = _build_label_info(id2label)
    decoded = _viterbi_decode(
        [[-10.0, -10.0, 0.0, -10.0, -0.1]],
        label_info=label_info,
        biases={},
    )

    assert decoded == [4]


def test_privacy_filter_grouped_decode_handles_bioes():
    from openmed.mlx.inference import PrivacyFilterMLXPipeline, _build_label_info

    pipeline = PrivacyFilterMLXPipeline.__new__(PrivacyFilterMLXPipeline)
    pipeline.id2label = {
        0: "O",
        1: "B-private_person",
        2: "I-private_person",
        3: "E-private_person",
        4: "S-private_email",
    }
    pipeline.label_info = _build_label_info(pipeline.id2label)

    probs = [
        [0.0, 0.9, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.8, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.7, 0.0],
        [0.9, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.95],
    ]
    result = pipeline._decode_grouped(
        [1, 2, 3, 0, 4],
        probs,
        [0, 4, 8, 9, 10],
        [4, 8, 9, 10, 27],
        "John Doe, alice@example.com",
    )

    assert result == [
        {
            "entity_group": "private_person",
            "score": pytest.approx(0.8),
            "word": "John Doe,",
            "start": 0,
            "end": 9,
        },
        {
            "entity_group": "private_email",
            "score": pytest.approx(0.95),
            "word": "alice@example.com",
            "start": 10,
            "end": 27,
        },
    ]


def test_dispatches_privacy_filter_pipeline(tmp_path):
    from openmed.mlx import inference

    config = _privacy_config()
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "openmed-mlx.json").write_text(
        json.dumps(
            {
                "format": "openmed-mlx",
                "format_version": 2,
                "task": "token-classification",
                "family": "openai-privacy-filter",
                "source_model_id": "openai/privacy-filter",
                "config_path": "config.json",
                "label_map_path": None,
                "preferred_weights": "weights.safetensors",
                "fallback_weights": ["weights.npz"],
                "available_weights": [],
                "weights_format": "safetensors",
                "quantization": None,
                "max_sequence_length": 32,
                "tokenizer": {"path": ".", "files": []},
            }
        )
    )

    with patch.object(
        inference.PrivacyFilterMLXPipeline,
        "__init__",
        return_value=None,
    ) as mock_init:
        pipeline = inference.create_mlx_pipeline(str(tmp_path))

    assert isinstance(pipeline, inference.PrivacyFilterMLXPipeline)
    mock_init.assert_called_once()


@pytest.mark.integration
@pytest.mark.skipif(not _MLX_AVAILABLE, reason="MLX is required for real artifact smoke tests")
def test_privacy_filter_real_artifact_smoke(monkeypatch):
    import os
    from openmed.mlx.inference import create_mlx_pipeline

    artifact = os.environ.get("OPENMED_PRIVACY_FILTER_MLX_ARTIFACT")
    if not artifact:
        pytest.skip("Set OPENMED_PRIVACY_FILTER_MLX_ARTIFACT to run the real artifact smoke test")

    pipe = create_mlx_pipeline(artifact)
    entities = pipe(
        "My name is Alice Smith, my email is alice.smith@example.com, "
        "and my phone is 415-555-0101."
    )
    groups = {entity["entity_group"] for entity in entities}

    assert "private_person" in groups
    assert "private_email" in groups
    assert "private_phone" in groups
