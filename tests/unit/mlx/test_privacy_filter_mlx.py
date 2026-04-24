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


def _privacy_quant_config() -> dict:
    config = _privacy_config()
    config.update(
        {
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 16,
            "head_dim": 8,
            "_mlx_quantization": {
                "bits": 8,
                "group_size": 32,
                "mode": "affine",
            },
        }
    )
    return config


def _stable_privacy_weights(config: dict):
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from openmed.mlx.models.privacy_filter import OpenAIPrivacyFilterForTokenClassification

    model = OpenAIPrivacyFilterForTokenClassification(config)
    weights = {}
    for key, value in tree_flatten(model.parameters()):
        shape = value.shape
        if key.endswith("norm.scale") or key == "norm.scale" or key.endswith(".scale"):
            weights[key] = mx.ones(shape, dtype=mx.float32)
        elif key.endswith("gate.bias"):
            weights[key] = mx.array([8.0, 4.0, -4.0], dtype=mx.float32)
        elif key.endswith(".bias") or key.endswith("sinks"):
            weights[key] = mx.zeros(shape, dtype=mx.float32)
        else:
            numel = 1
            for dim in shape:
                numel *= dim
            weights[key] = (
                mx.arange(numel, dtype=mx.float32).reshape(shape)
                / max(numel - 1, 1)
                - 0.5
            ) * 0.5
    return weights


def _write_mlx_artifact(path, config: dict, weights: dict) -> None:
    import mlx.core as mx

    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(path / "weights.safetensors", weights)


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


@pytest.mark.skipif(not _MLX_AVAILABLE, reason="MLX is required for q8 loader tests")
def test_privacy_filter_q8_loader_matches_bf16_fixture(tmp_path):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from openmed.mlx.models import load_model
    from openmed.mlx.models.privacy_filter import OpenAIPrivacyFilterForTokenClassification

    bf16_config = _privacy_quant_config()
    bf16_config.pop("_mlx_quantization")
    weights = _stable_privacy_weights(bf16_config)
    _write_mlx_artifact(tmp_path / "bf16", bf16_config, weights)

    q8_config = _privacy_quant_config()
    q8_model = OpenAIPrivacyFilterForTokenClassification(bf16_config)
    q8_model.load_weights(list(weights.items()))
    nn.quantize(q8_model, group_size=32, bits=8, mode="affine")
    q8_weights = dict(tree_flatten(q8_model.parameters()))
    _write_mlx_artifact(tmp_path / "q8", q8_config, q8_weights)

    bf16_model = load_model(tmp_path / "bf16")
    q8_model = load_model(tmp_path / "q8")
    input_ids = mx.array([[1, 2, 3, 4, 5, 6]], dtype=mx.int32)
    attention_mask = mx.ones((1, 6), dtype=mx.bool_)
    bf16_logits = bf16_model(input_ids, attention_mask=attention_mask).astype(mx.float32)
    q8_logits = q8_model(input_ids, attention_mask=attention_mask).astype(mx.float32)
    mx.eval(bf16_logits, q8_logits)

    assert bf16_logits.shape == q8_logits.shape == (1, 6, 5)
    assert bool(mx.all(mx.isfinite(q8_logits)).item())
    assert float(mx.max(mx.abs(bf16_logits - q8_logits)).item()) < 0.01
    assert bool(mx.all(mx.argmax(bf16_logits, axis=-1) == mx.argmax(q8_logits, axis=-1)).item())


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
