"""Focused coverage for the ModernBERT and Longformer MLX backends."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest


def _module_importable(module_name: str) -> bool:
    """Check optional MLX/HF dependencies in the interpreter running pytest."""
    code = f"import {module_name}"
    if module_name == "mlx.core":
        code = "import mlx.core as mx; mx.array([0]).tolist()"
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


_MLX_AVAILABLE = _module_importable("mlx.core")
_HF_AVAILABLE = _module_importable("torch") and _module_importable("transformers")


def test_new_families_resolve_and_are_normalized():
    """ModernBERT and Longformer resolve with their architecture aliases."""
    from openmed.mlx.models import (
        normalize_model_config,
        resolve_artifact_family,
    )

    modern = normalize_model_config(
        {
            "model_type": "modernbert",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
        }
    )
    longformer = normalize_model_config(
        {
            "model_type": "longformer",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "attention_window": 4,
        }
    )

    assert resolve_artifact_family("modernbert") == "modernbert"
    assert resolve_artifact_family("longformer") == "longformer"
    assert modern["type_vocab_size"] == 0
    assert modern["layer_types"] == ["full_attention", "sliding_attention"]
    assert modern["rope_parameters"]["full_attention"]["rope_theta"] == 160000.0
    assert longformer["attention_window"] == 4

    with pytest.raises(ValueError) as error:
        resolve_artifact_family("unsupported-test-family")
    assert "modernbert" in str(error.value)
    assert "longformer" in str(error.value)


def test_modernbert_weight_keys_remap_to_mlx_namespace():
    """Fused ModernBERT weights map to the MLX module tree."""
    from openmed.mlx.convert import remap_key

    assert (
        remap_key("model.embeddings.tok_embeddings.weight", "modernbert")
        == "embeddings.word_embeddings.weight"
    )
    assert (
        remap_key("model.layers.0.attn.Wqkv.weight", "modernbert")
        == "encoder.layers.0.attention.qkv_proj.weight"
    )
    assert (
        remap_key("model.layers.0.mlp.Wi.bias", "modernbert")
        == "encoder.layers.0.mlp.wi_proj.bias"
    )
    assert remap_key("model.final_norm.weight", "modernbert") == "final_norm.weight"
    assert remap_key("model.final_norm.weight") == "final_norm.weight"


def test_longformer_weight_keys_remap_global_projections():
    """Longformer local and global projections map to one MLX attention block."""
    from openmed.mlx.convert import remap_key

    assert (
        remap_key(
            "longformer.encoder.layer.0.attention.self.query_global.weight",
            "longformer",
        )
        == "encoder.layers.0.attention.query_global_proj.weight"
    )
    assert (
        remap_key(
            "longformer.encoder.layer.0.attention.self.query.weight", "longformer"
        )
        == "encoder.layers.0.attention.query_proj.weight"
    )
    assert (
        remap_key("longformer.embeddings.LayerNorm.bias", "longformer")
        == "embeddings.norm.bias"
    )


@pytest.mark.skipif(
    not (_MLX_AVAILABLE and _HF_AVAILABLE),
    reason="MLX, PyTorch, and Transformers are required for parity tests",
)
def test_modernbert_load_model_matches_hf_logits(tmp_path):
    """A deterministic tiny ModernBERT artifact matches Hugging Face logits."""
    import mlx.core as mx
    import numpy as np
    import torch
    from transformers import ModernBertConfig, ModernBertForTokenClassification

    from openmed.mlx.convert import remap_key
    from openmed.mlx.models import load_model, normalize_model_config

    hf_config = ModernBertConfig(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=32,
        local_attention=4,
        global_attn_every_n_layers=2,
        embedding_dropout=0.0,
        attention_dropout=0.0,
        mlp_dropout=0.0,
        classifier_dropout=0.0,
        num_labels=3,
    )
    hf_config._attn_implementation = "eager"
    torch.manual_seed(167)
    reference = ModernBertForTokenClassification(hf_config).eval()
    input_ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        expected = reference(input_ids, attention_mask=attention_mask).logits

    config = normalize_model_config(hf_config.to_dict())
    config.update(
        {
            "_mlx_family": "modernbert",
            "_mlx_model_type": "modernbert",
            "_mlx_task": "token-classification",
        }
    )
    weights = {
        remap_key(key, "modernbert"): mx.array(value.detach().cpu().numpy())
        for key, value in reference.state_dict().items()
        if not remap_key(key, "modernbert").startswith("_")
    }
    artifact = tmp_path / "modernbert"
    artifact.mkdir()
    (artifact / "config.json").write_text(json.dumps(config), encoding="utf-8")
    mx.save_safetensors(str(artifact / "weights.safetensors"), weights)

    model = load_model(artifact)
    actual = model(
        mx.array(input_ids.numpy()),
        attention_mask=mx.array(attention_mask.numpy(), dtype=mx.float32),
    )
    mx.eval(actual)
    np.testing.assert_allclose(
        np.asarray(actual), expected.numpy(), rtol=3e-4, atol=3e-4
    )


@pytest.mark.skipif(
    not (_MLX_AVAILABLE and _HF_AVAILABLE),
    reason="MLX, PyTorch, and Transformers are required for parity tests",
)
def test_longformer_load_model_matches_hf_logits_with_global_attention(tmp_path):
    """A tiny Longformer artifact matches local and global HF attention logits."""
    import mlx.core as mx
    import numpy as np
    import torch
    from transformers import LongformerConfig, LongformerForTokenClassification

    from openmed.mlx.convert import remap_key
    from openmed.mlx.models import load_model, normalize_model_config

    hf_config = LongformerConfig(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=32,
        attention_window=[4],
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        num_labels=3,
        pad_token_id=1,
    )
    torch.manual_seed(168)
    reference = LongformerForTokenClassification(hf_config).eval()
    input_ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    global_attention_mask = torch.tensor([[1, 0, 0, 0, 0, 0, 0]], dtype=torch.long)
    with torch.no_grad():
        expected = reference(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
        ).logits

    config = normalize_model_config(hf_config.to_dict())
    config.update(
        {
            "_mlx_family": "longformer",
            "_mlx_model_type": "longformer",
            "_mlx_task": "token-classification",
        }
    )
    weights = {
        remap_key(key, "longformer"): mx.array(value.detach().cpu().numpy())
        for key, value in reference.state_dict().items()
        if not remap_key(key, "longformer").startswith("_")
    }
    artifact = tmp_path / "longformer"
    artifact.mkdir()
    (artifact / "config.json").write_text(json.dumps(config), encoding="utf-8")
    mx.save_safetensors(str(artifact / "weights.safetensors"), weights)

    model = load_model(artifact)
    actual = model(
        mx.array(input_ids.numpy()),
        attention_mask=mx.array(attention_mask.numpy(), dtype=mx.float32),
        global_attention_mask=mx.array(global_attention_mask.numpy(), dtype=mx.float32),
    )
    mx.eval(actual)
    np.testing.assert_allclose(
        np.asarray(actual), expected.numpy(), rtol=3e-4, atol=3e-4
    )
