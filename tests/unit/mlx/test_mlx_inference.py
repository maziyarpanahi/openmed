"""Tests for MLX inference pipeline output format compatibility.

These tests mock the MLX model to verify that the pipeline produces
output in the same format as HuggingFace's token-classification pipeline.
No actual MLX installation required.
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# We test the BIO decoding and output format logic without requiring MLX.
# The actual MLX model calls are mocked.


class TestMLXPipelineOutputFormat:
    """Verify MLX pipeline produces HF-compatible output dicts."""

    def _make_mock_pipeline(self, tmp_path):
        """Create a mock MLXTokenClassificationPipeline with fake model."""
        # Write fake config
        config = {
            "id2label": {"0": "O", "1": "B-NAME", "2": "I-NAME", "3": "B-DATE"},
            "num_labels": 4,
            "hidden_size": 64,
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "vocab_size": 30522,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        return config

    def test_grouped_output_has_required_keys(self, tmp_path):
        """Grouped entities must have entity_group, score, word, start, end."""
        config = self._make_mock_pipeline(tmp_path)

        # Simulate what _decode_grouped produces
        from openmed.mlx.inference import MLXTokenClassificationPipeline

        with patch.object(MLXTokenClassificationPipeline, "__init__", lambda self, **kw: None):
            pipeline = MLXTokenClassificationPipeline.__new__(MLXTokenClassificationPipeline)
            pipeline.id2label = {int(k): v for k, v in config["id2label"].items()}
            pipeline.aggregation_strategy = "simple"

            # Test the decoding logic directly
            pred_ids = [0, 1, 2, 0, 3, 0]
            probs = [
                [0.9, 0.05, 0.03, 0.02],  # O
                [0.05, 0.9, 0.03, 0.02],   # B-NAME
                [0.03, 0.05, 0.9, 0.02],   # I-NAME
                [0.9, 0.05, 0.03, 0.02],   # O
                [0.02, 0.05, 0.03, 0.9],   # B-DATE
                [0.9, 0.05, 0.03, 0.02],   # O
            ]
            offsets = [[0, 0], [0, 4], [5, 8], [8, 9], [10, 20], [0, 0]]
            text = "John Doe, 2024-01-15"

            result = pipeline._decode_grouped(pred_ids, probs, offsets, text)

            assert len(result) == 2

            # Check NAME entity
            name_ent = result[0]
            assert "entity_group" in name_ent
            assert "score" in name_ent
            assert "word" in name_ent
            assert "start" in name_ent
            assert "end" in name_ent
            assert name_ent["entity_group"] == "NAME"
            assert name_ent["word"] == "John Doe"
            assert name_ent["start"] == 0
            assert name_ent["end"] == 8

            # Check DATE entity
            date_ent = result[1]
            assert date_ent["entity_group"] == "DATE"

    def test_raw_output_has_required_keys(self, tmp_path):
        """Raw per-token output must have entity, score, word, start, end, index."""
        config = self._make_mock_pipeline(tmp_path)

        from openmed.mlx.inference import MLXTokenClassificationPipeline

        with patch.object(MLXTokenClassificationPipeline, "__init__", lambda self, **kw: None):
            pipeline = MLXTokenClassificationPipeline.__new__(MLXTokenClassificationPipeline)
            pipeline.id2label = {int(k): v for k, v in config["id2label"].items()}
            pipeline.aggregation_strategy = None

            pred_ids = [0, 1, 0]
            probs = [
                [0.9, 0.05, 0.03, 0.02],
                [0.05, 0.9, 0.03, 0.02],
                [0.9, 0.05, 0.03, 0.02],
            ]
            offsets = [[0, 0], [0, 4], [0, 0]]
            text = "John visited"

            result = pipeline._decode_raw(pred_ids, probs, offsets, text)

            assert len(result) == 1
            ent = result[0]
            assert "entity" in ent
            assert "score" in ent
            assert "word" in ent
            assert "start" in ent
            assert "end" in ent
            assert "index" in ent

    def test_tuple_offsets_skip_special_tokens(self, tmp_path):
        """Tuple ``(0, 0)`` offsets from fast tokenizers should be ignored."""
        config = self._make_mock_pipeline(tmp_path)

        from openmed.mlx.inference import MLXTokenClassificationPipeline

        with patch.object(MLXTokenClassificationPipeline, "__init__", lambda self, **kw: None):
            pipeline = MLXTokenClassificationPipeline.__new__(MLXTokenClassificationPipeline)
            pipeline.id2label = {int(k): v for k, v in config["id2label"].items()}
            pipeline.aggregation_strategy = "simple"

            pred_ids = [1, 1, 2, 0, 1]
            probs = [
                [0.05, 0.9, 0.03, 0.02],
                [0.05, 0.9, 0.03, 0.02],
                [0.03, 0.05, 0.9, 0.02],
                [0.9, 0.05, 0.03, 0.02],
                [0.05, 0.9, 0.03, 0.02],
            ]
            offsets = [(0, 0), (0, 4), (5, 8), (8, 9), (0, 0)]
            text = "John Doe,"

            result = pipeline._decode_grouped(pred_ids, probs, offsets, text)

            assert len(result) == 1
            assert result[0]["word"] == "John Doe"

    def test_aggregation_strategies(self, tmp_path):
        """Verify first/average/max aggregation produce correct scores."""
        config = self._make_mock_pipeline(tmp_path)

        from openmed.mlx.inference import MLXTokenClassificationPipeline

        pred_ids = [1, 2]  # B-NAME, I-NAME
        probs = [
            [0.05, 0.9, 0.03, 0.02],
            [0.03, 0.05, 0.8, 0.12],
        ]
        offsets = [[0, 4], [5, 8]]
        text = "John Doe"

        for strategy, expected_score in [
            ("first", 0.9),
            ("max", 0.9),
            ("simple", (0.9 + 0.8) / 2),
        ]:
            with patch.object(MLXTokenClassificationPipeline, "__init__", lambda self, **kw: None):
                pipeline = MLXTokenClassificationPipeline.__new__(MLXTokenClassificationPipeline)
                pipeline.id2label = {int(k): v for k, v in config["id2label"].items()}
                pipeline.aggregation_strategy = strategy

                result = pipeline._decode_grouped(pred_ids, probs, offsets, text)
                assert len(result) == 1
                assert abs(result[0]["score"] - expected_score) < 0.01, \
                    f"Strategy {strategy}: expected {expected_score}, got {result[0]['score']}"


class TestMLXModelResolve:
    """Test model resolution logic."""

    def test_preconverted_repo_failure_falls_back_to_conversion(self, tmp_path):
        """A private/missing Hub snapshot should fall back to local conversion."""
        from openmed.mlx import inference

        output_dir = tmp_path / "OpenMed_OpenMed-PII-SuperClinical-Small-44M-v1"
        config = type("Config", (), {"cache_dir": str(tmp_path)})()

        with patch.dict(
            inference._MLX_MODEL_MAP,
            {"OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1": "OpenMed/private-mlx"},
            clear=True,
        ), patch.object(
            inference,
            "_download_preconverted_mlx_model",
            side_effect=RuntimeError("private repo"),
        ) as mock_download, patch(
            "openmed.mlx.convert.convert",
            side_effect=lambda model_id, output_dir, cache_dir=None: Path(output_dir).mkdir(
                parents=True, exist_ok=True
            ) or (Path(output_dir) / "config.json").write_text("{}"),
        ) as mock_convert:
            path, tok_name = inference._resolve_mlx_model(
                "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1",
                config=config,
            )

        assert path == str(output_dir)
        assert tok_name == "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"
        mock_download.assert_called_once_with(
            "OpenMed/private-mlx",
            cache_dir=str(tmp_path),
        )
        mock_convert.assert_called_once_with(
            "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1",
            output_dir,
            cache_dir=str(tmp_path),
        )

    def test_local_path_detection(self, tmp_path):
        """If model_name is a local directory with config.json, use it."""
        (tmp_path / "config.json").write_text('{"num_labels": 3}')

        from openmed.mlx.inference import _resolve_mlx_model
        path, tok_name = _resolve_mlx_model(str(tmp_path))
        assert path == str(tmp_path)
        assert tok_name == str(tmp_path)
