"""Tests for MLX model conversion (weight key remapping logic).

These tests don't require MLX to be installed — they test the key
remapping and config extraction logic in isolation.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from openmed.mlx.convert import remap_key


def _module_importable(module_name: str) -> bool:
    """Return True only when a module can actually be imported."""
    try:
        __import__(module_name)
    except Exception:
        return False
    return True


_NUMPY_AVAILABLE = _module_importable("numpy")
_SAFETENSORS_NUMPY_AVAILABLE = _module_importable("safetensors.numpy")
_TORCH_AVAILABLE = _module_importable("torch")


class TestWeightKeyRemapping:
    """Test HuggingFace → MLX weight key remapping."""

    def test_attention_query(self):
        assert "attention.query_proj.weight" in remap_key(
            "bert.encoder.layer.0.attention.self.query.weight"
        )

    def test_attention_key(self):
        assert "attention.key_proj.weight" in remap_key(
            "bert.encoder.layer.0.attention.self.key.weight"
        )

    def test_attention_value(self):
        assert "attention.value_proj.weight" in remap_key(
            "bert.encoder.layer.0.attention.self.value.weight"
        )

    def test_attention_output(self):
        assert "attention.out_proj.weight" in remap_key(
            "bert.encoder.layer.0.attention.output.dense.weight"
        )

    def test_attention_layernorm_to_ln1(self):
        assert ".ln1." in remap_key(
            "bert.encoder.layer.0.attention.output.LayerNorm.weight"
        )

    def test_intermediate_to_linear1(self):
        assert ".linear1." in remap_key(
            "bert.encoder.layer.0.intermediate.dense.weight"
        )

    def test_output_dense_to_linear2(self):
        assert ".linear2." in remap_key("bert.encoder.layer.0.output.dense.weight")

    def test_output_layernorm_to_ln2(self):
        assert ".ln2." in remap_key("bert.encoder.layer.0.output.LayerNorm.weight")

    def test_encoder_layers(self):
        result = remap_key("bert.encoder.layer.5.intermediate.dense.bias")
        assert result.startswith("encoder.layers.5.")

    def test_embeddings_word(self):
        result = remap_key("bert.embeddings.word_embeddings.weight")
        assert result == "embeddings.word_embeddings.weight"

    def test_embeddings_position(self):
        result = remap_key("bert.embeddings.position_embeddings.weight")
        assert result == "embeddings.position_embeddings.weight"

    def test_embeddings_layernorm(self):
        result = remap_key("bert.embeddings.LayerNorm.weight")
        assert result == "embeddings.norm.weight"

    def test_classifier_preserved(self):
        assert remap_key("classifier.weight") == "classifier.weight"
        assert remap_key("classifier.bias") == "classifier.bias"

    def test_pooler_prefixed(self):
        result = remap_key("bert.pooler.dense.weight")
        assert result.startswith("_")  # pooler is skipped

    def test_deberta_attention_output_remapped(self):
        result = remap_key(
            "deberta.encoder.layer.0.attention.output.dense.weight",
            "deberta-v2",
        )
        assert result == "deberta.encoder.layer.0.attention.out_proj.weight"

    def test_deberta_ffn_and_norm_remapped(self):
        result = remap_key(
            "deberta.encoder.layer.0.output.LayerNorm.weight",
            "deberta-v2",
        )
        assert result == "deberta.encoder.layer.0.ln2.weight"

    def test_roberta_prefixes_remapped(self):
        result = remap_key(
            "roberta.encoder.layer.0.attention.self.query.weight",
            "roberta",
        )
        assert result == "encoder.layers.0.attention.query_proj.weight"

    def test_distilbert_keys_remapped(self):
        result = remap_key(
            "distilbert.transformer.layer.0.attention.q_lin.weight",
            "distilbert",
        )
        assert result == "encoder.layers.0.attention.query_proj.weight"

    def test_distilbert_norm_remapped(self):
        result = remap_key(
            "distilbert.transformer.layer.0.output_layer_norm.weight",
            "distilbert",
        )
        assert result == "encoder.layers.0.ln2.weight"

    def test_remap_all_hf_keys(self):
        """Verify that remap_key handles all common HF BERT keys."""
        hf_keys = [
            "bert.encoder.layer.0.attention.self.query.weight",
            "bert.encoder.layer.0.attention.self.key.weight",
            "bert.encoder.layer.0.attention.self.value.weight",
            "bert.encoder.layer.0.attention.output.dense.weight",
            "bert.encoder.layer.0.attention.output.LayerNorm.weight",
            "bert.encoder.layer.0.intermediate.dense.weight",
            "bert.encoder.layer.0.output.dense.weight",
            "bert.encoder.layer.0.output.LayerNorm.weight",
            "bert.embeddings.word_embeddings.weight",
            "bert.embeddings.position_embeddings.weight",
            "bert.embeddings.LayerNorm.weight",
            "classifier.weight",
            "classifier.bias",
        ]
        for key in hf_keys:
            mlx_key = remap_key(key)
            assert not mlx_key.startswith("bert."), (
                f"Key {key!r} was not remapped: {mlx_key!r}"
            )


@pytest.mark.skipif(
    not (_NUMPY_AVAILABLE and _TORCH_AVAILABLE),
    reason="numpy and torch are required for BF16 conversion tests",
)
def test_to_numpy_casts_bfloat16_to_float32():
    """BF16 tensors need an explicit float32 cast before numpy conversion."""
    import numpy as np
    import torch

    from openmed.mlx.convert import _to_numpy

    converted = _to_numpy(torch.tensor([1.0], dtype=torch.bfloat16))

    assert converted.dtype == np.float32
    np.testing.assert_allclose(converted, np.array([1.0], dtype=np.float32))


def _tiny_opf_config() -> dict:
    return {
        "model_type": "openai_privacy_filter",
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
        "classifier_bias": True,
    }


def _tiny_opf_state_dict(config: dict):
    import numpy as np

    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    num_experts = config["num_experts"]
    num_q = config["num_attention_heads"]
    num_kv = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    q_dim = num_q * head_dim
    kv_dim = num_kv * head_dim

    state = {
        "model.embed_tokens.weight": np.zeros(
            (config["vocab_size"], hidden),
            dtype=np.float32,
        ),
        "model.norm.weight": np.ones((hidden,), dtype=np.float32),
        "score.weight": np.zeros((config["num_labels"], hidden), dtype=np.float32),
        "score.bias": np.zeros((config["num_labels"],), dtype=np.float32),
    }

    prefix = "model.layers.0"
    state.update(
        {
            f"{prefix}.input_layernorm.weight": np.ones((hidden,), dtype=np.float32),
            f"{prefix}.post_attention_layernorm.weight": np.ones(
                (hidden,),
                dtype=np.float32,
            ),
            f"{prefix}.self_attn.q_proj.weight": np.full(
                (q_dim, hidden),
                1.0,
                dtype=np.float32,
            ),
            f"{prefix}.self_attn.k_proj.weight": np.full(
                (kv_dim, hidden),
                2.0,
                dtype=np.float32,
            ),
            f"{prefix}.self_attn.v_proj.weight": np.full(
                (kv_dim, hidden),
                3.0,
                dtype=np.float32,
            ),
            f"{prefix}.self_attn.q_proj.bias": np.full(
                (q_dim,),
                1.0,
                dtype=np.float32,
            ),
            f"{prefix}.self_attn.k_proj.bias": np.full(
                (kv_dim,),
                2.0,
                dtype=np.float32,
            ),
            f"{prefix}.self_attn.v_proj.bias": np.full(
                (kv_dim,),
                3.0,
                dtype=np.float32,
            ),
            f"{prefix}.self_attn.o_proj.weight": np.zeros(
                (hidden, q_dim),
                dtype=np.float32,
            ),
            f"{prefix}.self_attn.o_proj.bias": np.zeros((hidden,), dtype=np.float32),
            f"{prefix}.self_attn.sinks": np.zeros((num_q,), dtype=np.float32),
            f"{prefix}.mlp.router.weight": np.zeros(
                (num_experts, hidden),
                dtype=np.float32,
            ),
            f"{prefix}.mlp.router.bias": np.zeros((num_experts,), dtype=np.float32),
            f"{prefix}.mlp.experts.gate_up_proj": np.zeros(
                (num_experts, hidden, intermediate * 2),
                dtype=np.float32,
            ),
            f"{prefix}.mlp.experts.gate_up_proj_bias": np.zeros(
                (num_experts, intermediate * 2),
                dtype=np.float32,
            ),
            f"{prefix}.mlp.experts.down_proj": np.zeros(
                (num_experts, intermediate, hidden),
                dtype=np.float32,
            ),
            f"{prefix}.mlp.experts.down_proj_bias": np.zeros(
                (num_experts, hidden),
                dtype=np.float32,
            ),
        }
    )
    return state


@pytest.mark.skipif(not _NUMPY_AVAILABLE, reason="numpy is required for OPF tests")
def test_opf_converter_remaps_and_validates_hf_layout():
    import numpy as np

    from openmed.mlx.convert import _convert_opf_weights, _validate_opf_weights

    config = _tiny_opf_config()
    converted = _convert_opf_weights(_tiny_opf_state_dict(config))
    _validate_opf_weights(converted, config)

    q_dim = config["num_attention_heads"] * config["head_dim"]
    kv_dim = config["num_key_value_heads"] * config["head_dim"]
    qkv_weight = converted["block.0.attn.qkv.weight"]

    assert "unembedding.bias" in converted
    assert "model.layers.0.self_attn.q_proj.weight" not in converted
    assert qkv_weight.shape == (q_dim + 2 * kv_dim, config["hidden_size"])
    np.testing.assert_array_equal(qkv_weight[:q_dim], 1.0)
    np.testing.assert_array_equal(qkv_weight[q_dim : q_dim + kv_dim], 2.0)
    np.testing.assert_array_equal(qkv_weight[q_dim + kv_dim :], 3.0)


@pytest.mark.skipif(not _NUMPY_AVAILABLE, reason="numpy is required for OPF tests")
def test_opf_validation_rejects_partial_qkv_fusion():
    from openmed.mlx.convert import _convert_opf_weights, _validate_opf_weights

    config = _tiny_opf_config()
    state = _tiny_opf_state_dict(config)
    del state["model.layers.0.self_attn.v_proj.bias"]

    converted = _convert_opf_weights(state)

    with pytest.raises(ValueError, match="Invalid OPF MLX weight shapes"):
        _validate_opf_weights(converted, config)


@pytest.mark.skipif(
    not _NUMPY_AVAILABLE,
    reason="numpy is required for NumPy save/load tests",
)
class TestSaveNumpyModel:
    """Test the NumPy fallback save path."""

    def test_saves_weights_and_config(self, tmp_path):
        import numpy as np

        from openmed.mlx.convert import save_numpy_model

        weights = {
            "classifier.weight": np.random.randn(3, 64).astype(np.float32),
            "classifier.bias": np.random.randn(3).astype(np.float32),
        }
        config = {
            "num_labels": 3,
            "id2label": {"0": "O", "1": "B-NAME", "2": "I-NAME"},
        }

        output = save_numpy_model(weights, config, tmp_path / "model")
        expected_name = (
            "weights.safetensors" if _SAFETENSORS_NUMPY_AVAILABLE else "weights.npz"
        )
        assert (output / expected_name).exists()
        assert (output / "config.json").exists()
        assert (output / "id2label.json").exists()

        with open(output / "config.json") as f:
            saved_config = json.load(f)
        assert saved_config["num_labels"] == 3
        assert saved_config["_mlx_weights_format"] in {"safetensors", "npz"}

    def test_weights_loadable(self, tmp_path):
        import numpy as np

        from openmed.mlx.convert import save_numpy_model

        original_w = np.random.randn(3, 64).astype(np.float32)
        weights = {"classifier.weight": original_w}
        config = {"num_labels": 3}

        output = save_numpy_model(weights, config, tmp_path / "model")
        if (output / "weights.safetensors").exists():
            from safetensors.numpy import load_file

            loaded = load_file(str(output / "weights.safetensors"))
            np.testing.assert_array_almost_equal(
                loaded["classifier.weight"],
                original_w,
            )
        else:
            loaded = np.load(str(output / "weights.npz"))
            np.testing.assert_array_almost_equal(
                loaded["classifier.weight"], original_w
            )

    def test_creates_parent_dirs(self, tmp_path):
        import numpy as np

        from openmed.mlx.convert import save_numpy_model

        weights = {"w": np.array([1.0, 2.0], dtype=np.float32)}
        config = {"num_labels": 1}

        output = save_numpy_model(weights, config, tmp_path / "a" / "b" / "model")
        assert (output / "weights.safetensors").exists() or (
            output / "weights.npz"
        ).exists()

    def test_writes_manifest_and_tokenizer_assets_when_source_model_id_provided(
        self,
        tmp_path,
        monkeypatch,
    ):
        import numpy as np

        from openmed.mlx.convert import save_numpy_model

        class FakeTokenizer:
            def save_pretrained(self, output_dir):
                Path(output_dir, "tokenizer.json").write_text("{}")
                Path(output_dir, "tokenizer_config.json").write_text("{}")
                Path(output_dir, "special_tokens_map.json").write_text("{}")

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = types.SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: FakeTokenizer()
        )
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

        output = save_numpy_model(
            {"classifier.weight": np.random.randn(3, 64).astype(np.float32)},
            {
                "num_labels": 3,
                "model_type": "bert",
                "_mlx_model_type": "bert",
                "max_position_embeddings": 128,
            },
            tmp_path / "model",
            source_model_id="OpenMed/test-model",
        )

        manifest = json.loads((output / "openmed-mlx.json").read_text())
        assert manifest["format"] == "openmed-mlx"
        assert manifest["formats"] == ["mlx-fp"]
        assert manifest["source_model_id"] == "OpenMed/test-model"
        assert manifest["available_weights"]
        assert manifest["tokenizer"]["files"] == [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ]
        assert (output / "tokenizer.json").exists()
        assert (output / "tokenizer_config.json").exists()
        assert (output / "special_tokens_map.json").exists()

    @pytest.mark.skipif(
        not _SAFETENSORS_NUMPY_AVAILABLE,
        reason="safetensors is required for fallback test",
    )
    def test_falls_back_to_npz_when_safetensors_save_fails(self, tmp_path, monkeypatch):
        import numpy as np
        import safetensors.numpy as st_numpy

        from openmed.mlx.convert import save_numpy_model

        def raise_on_save(*args, **kwargs):
            raise RuntimeError("forced safetensors failure")

        monkeypatch.setattr(st_numpy, "save_file", raise_on_save)

        output = save_numpy_model(
            {"classifier.weight": np.random.randn(3, 64).astype(np.float32)},
            {"num_labels": 3},
            tmp_path / "model",
        )

        assert (output / "weights.npz").exists()
        assert not (output / "weights.safetensors").exists()

        with open(output / "config.json") as f:
            saved_config = json.load(f)
        assert saved_config["_mlx_weights_format"] == "npz"


class TestModelTypeResolution:
    """Test architecture selection for MLX model loading."""

    def test_resolves_bert(self):
        from openmed.mlx.models import resolve_model_type

        assert resolve_model_type("bert") == "bert"

    def test_resolves_deberta_from_architecture(self):
        from openmed.mlx.models import resolve_model_type

        assert (
            resolve_model_type(
                {"architectures": ["DebertaV2ForTokenClassification"]},
            )
            == "deberta-v2"
        )

    def test_resolves_roberta_to_bert_family(self):
        from openmed.mlx.models import resolve_model_type

        assert resolve_model_type("roberta") == "bert"

    def test_resolves_xlm_roberta_to_bert_family(self):
        from openmed.mlx.models import resolve_model_type

        assert resolve_model_type({"model_type": "xlm-roberta"}) == "bert"

    def test_resolves_distilbert_to_bert_family(self):
        from openmed.mlx.models import resolve_model_type

        assert resolve_model_type("distilbert") == "bert"


class TestModelConfigNormalization:
    """Test config aliasing for BERT-family architectures."""

    def test_normalizes_distilbert_config(self):
        from openmed.mlx.models import normalize_model_config

        normalized = normalize_model_config(
            {
                "model_type": "distilbert",
                "dim": 768,
                "n_heads": 12,
                "n_layers": 6,
                "hidden_dim": 3072,
                "dropout": 0.1,
            }
        )

        assert normalized["hidden_size"] == 768
        assert normalized["num_attention_heads"] == 12
        assert normalized["num_hidden_layers"] == 6
        assert normalized["intermediate_size"] == 3072
        assert normalized["type_vocab_size"] == 0
        assert normalized["_mlx_position_offset"] == 0

    def test_normalizes_roberta_position_offset(self):
        from openmed.mlx.models import normalize_model_config

        normalized = normalize_model_config(
            {
                "model_type": "roberta",
                "hidden_size": 768,
                "num_attention_heads": 12,
                "num_hidden_layers": 6,
                "intermediate_size": 3072,
                "pad_token_id": 1,
            }
        )

        assert normalized["type_vocab_size"] == 1
        assert normalized["_mlx_position_offset"] == 2
