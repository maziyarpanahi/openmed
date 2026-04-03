"""Tests for MLX model conversion (weight key remapping logic).

These tests don't require MLX to be installed — they test the key
remapping and config extraction logic in isolation.
"""

from __future__ import annotations

import json
import pytest

from openmed.mlx.convert import remap_key


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
        assert ".linear2." in remap_key(
            "bert.encoder.layer.0.output.dense.weight"
        )

    def test_output_layernorm_to_ln2(self):
        assert ".ln2." in remap_key(
            "bert.encoder.layer.0.output.LayerNorm.weight"
        )

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


# ---------------------------------------------------------------------------
# Tests that require numpy (skip gracefully on minimal CI environments)
# ---------------------------------------------------------------------------

numpy = pytest.importorskip("numpy")


class TestSaveNumpyModel:
    """Test the NumPy fallback save path."""

    def test_saves_weights_and_config(self, tmp_path):
        from openmed.mlx.convert import save_numpy_model

        weights = {
            "classifier.weight": numpy.random.randn(3, 64).astype(numpy.float32),
            "classifier.bias": numpy.random.randn(3).astype(numpy.float32),
        }
        config = {
            "num_labels": 3,
            "id2label": {"0": "O", "1": "B-NAME", "2": "I-NAME"},
        }

        output = save_numpy_model(weights, config, tmp_path / "model")
        assert (output / "weights.npz").exists()
        assert (output / "config.json").exists()
        assert (output / "id2label.json").exists()

        # Verify config content
        with open(output / "config.json") as f:
            saved_config = json.load(f)
        assert saved_config["num_labels"] == 3

    def test_weights_loadable(self, tmp_path):
        from openmed.mlx.convert import save_numpy_model

        original_w = numpy.random.randn(3, 64).astype(numpy.float32)
        weights = {"classifier.weight": original_w}
        config = {"num_labels": 3}

        output = save_numpy_model(weights, config, tmp_path / "model")
        loaded = numpy.load(str(output / "weights.npz"))
        numpy.testing.assert_array_almost_equal(
            loaded["classifier.weight"], original_w,
        )

    def test_creates_parent_dirs(self, tmp_path):
        from openmed.mlx.convert import save_numpy_model

        weights = {"w": numpy.array([1.0, 2.0], dtype=numpy.float32)}
        config = {"num_labels": 1}

        output = save_numpy_model(weights, config, tmp_path / "a" / "b" / "model")
        assert (output / "weights.npz").exists()
