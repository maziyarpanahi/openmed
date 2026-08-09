"""Offline coverage for the ModernBERT MLX architecture path."""

from __future__ import annotations

import numpy as np
import pytest

from openmed.mlx.convert import remap_key


def _available(module_name: str) -> bool:
    """Return whether an optional backend can be imported in this process."""
    try:
        __import__(module_name)
    except Exception:
        return False
    return True


_MLX_AVAILABLE = _available("mlx.core")
_TORCH_AVAILABLE = _available("torch")
_TRANSFORMERS_AVAILABLE = _available("transformers")


def _tiny_config() -> dict:
    return {
        "model_type": "modernbert",
        "vocab_size": 64,
        "pad_token_id": 0,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_labels": 3,
        "max_position_embeddings": 32,
        "norm_eps": 1e-5,
        "norm_bias": True,
        "attention_bias": True,
        "attention_dropout": 0.0,
        "global_attn_every_n_layers": 2,
        "local_attention": 8,
        "global_rope_theta": 160000.0,
        "local_rope_theta": 10000.0,
        "embedding_dropout": 0.0,
        "mlp_bias": True,
        "mlp_dropout": 0.0,
        "classifier_dropout": 0.0,
        "classifier_bias": False,
        "classifier_activation": "gelu",
        "hidden_activation": "gelu",
    }


def test_modernbert_dispatch_and_key_remapping() -> None:
    from openmed.mlx.models import normalize_model_config, resolve_model_type

    config = _tiny_config()
    assert resolve_model_type(config) == "modernbert"
    assert (
        resolve_model_type({"architectures": ["ModernBertForTokenClassification"]})
        == "modernbert"
    )
    normalized = normalize_model_config(config)
    assert normalized["_mlx_family"] == "modernbert"
    assert normalized["layer_norm_eps"] == pytest.approx(1e-5)

    assert (
        remap_key("model.layers.0.attn.Wqkv.weight", "modernbert")
        == "model.layers.0.attn.Wqkv.weight"
    )
    assert (
        remap_key("model.embeddings.word_embeddings.weight", "modernbert")
        == "model.embeddings.tok_embeddings.weight"
    )


@pytest.mark.skipif(not _MLX_AVAILABLE, reason="requires MLX")
def test_modernbert_fp_and_int8_logits_keep_token_shape(tmp_path) -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from openmed.mlx.convert import save_mlx_model
    from openmed.mlx.models import build_model, load_model

    config = _tiny_config()
    config["_mlx_model_type"] = "modernbert"
    config["_mlx_task"] = "token-classification"
    model = build_model(config)
    weights = dict(tree_flatten(model.parameters()))

    fp_artifact = save_mlx_model(weights, config, tmp_path / "fp")
    fp_model = load_model(fp_artifact)
    input_ids = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    mask = mx.ones(input_ids.shape, dtype=mx.float32)
    fp_logits = fp_model(input_ids, attention_mask=mask)
    mx.eval(fp_logits)
    assert tuple(fp_logits.shape) == (1, 5, config["num_labels"])

    int8_artifact = save_mlx_model(
        weights,
        config,
        tmp_path / "int8",
        quantize_bits=8,
        quantize_group_size=32,
    )
    int8_model = load_model(int8_artifact)
    int8_logits = int8_model(input_ids, attention_mask=mask)
    mx.eval(int8_logits)
    assert tuple(int8_logits.shape) == tuple(fp_logits.shape)


@pytest.mark.skipif(
    not (_MLX_AVAILABLE and _TORCH_AVAILABLE and _TRANSFORMERS_AVAILABLE),
    reason="requires MLX, PyTorch, and Transformers",
)
def test_modernbert_mlx_logits_match_pytorch_reference() -> None:
    import mlx.core as mx
    import torch
    from transformers import ModernBertConfig, ModernBertForTokenClassification

    from openmed.mlx.models import build_model

    torch.manual_seed(7)
    torch_config = ModernBertConfig(
        **_tiny_config(),
        reference_compile=False,
    )
    torch_config.num_labels = 3
    torch_config._attn_implementation = "eager"
    reference = ModernBertForTokenClassification(torch_config)
    reference.eval()

    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=torch.long)
    attention_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1, 0]],
        dtype=torch.bool,
    )
    with torch.no_grad():
        expected = reference(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits.numpy()

    weights = {
        remap_key(key, "modernbert"): mx.array(value.detach().cpu().numpy())
        for key, value in reference.state_dict().items()
    }
    mlx_config = torch_config.to_dict()
    mlx_config["num_labels"] = torch_config.num_labels
    mlx_model = build_model(mlx_config)
    mlx_model.load_weights(list(weights.items()))
    mlx_model.eval()
    actual = mlx_model(
        mx.array(input_ids.numpy(), dtype=mx.int32),
        attention_mask=mx.array(attention_mask.numpy(), dtype=mx.float32),
    )
    mx.eval(actual)

    np.testing.assert_allclose(np.asarray(actual), expected, rtol=3e-4, atol=3e-4)
