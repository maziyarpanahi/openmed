"""Tests for openmed.onnx.convert and related seams.

Covers:
  T1 — logit parity: ONNX CPU EP vs torch on a batch of clinical token sequences.
  T3 — manifest schema: write_manifest emits formats[], back-compat readers still resolve.
  Gate seam — quant_delta and tier_fit are fail-closed with TODO→OM-032.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest


def _importable(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


_HAS_TORCH = _importable("torch")
_HAS_ONNXRUNTIME = _importable("onnxruntime")
_HAS_TRANSFORMERS = _importable("transformers")

# Directory where T1 writes the Node fixture for T4
_NODE_FIXTURE_DIR = Path(__file__).parent.parent.parent / "node" / "fixtures"


# ---------------------------------------------------------------------------
# Module API surface (always runs)
# ---------------------------------------------------------------------------

class TestOnnxConvertModule:
    def test_module_importable(self):
        from openmed.onnx import convert
        assert hasattr(convert, "convert")
        assert hasattr(convert, "main")

    def test_convert_signature(self):
        import inspect
        from openmed.onnx.convert import convert

        sig = inspect.signature(convert)
        params = list(sig.parameters)
        assert "model_id" in params
        assert "output_path" in params
        assert "max_seq_length" in params
        assert "opset_version" in params
        assert sig.parameters["opset_version"].default == 18

    def test_main_is_callable(self):
        from openmed.onnx.convert import main
        assert callable(main)

    def test_missing_torch_raises_import_error(self, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "torch", None)
        # Re-importing after patching requires reimporting the function
        import importlib
        import openmed.onnx.convert as m
        importlib.reload(m)
        with pytest.raises(ImportError, match="openmed\\[onnx\\]"):
            m.convert("some/model", "/tmp/out.onnx")
        # Restore module state for subsequent tests
        importlib.reload(m)


# ---------------------------------------------------------------------------
# Gate seam (always runs — no heavy deps)
# ---------------------------------------------------------------------------

class TestGateSeam:
    def test_gates_importable(self):
        from openmed.eval.gates import GateFailure, quant_delta, tier_fit
        assert callable(quant_delta)
        assert callable(tier_fit)
        assert issubclass(GateFailure, Exception)

    def test_quant_delta_is_fail_closed(self):
        from openmed.eval.gates import quant_delta
        with pytest.raises(NotImplementedError, match="OM-032"):
            quant_delta(None, None, [])

    def test_tier_fit_is_fail_closed(self):
        from openmed.eval.gates import tier_fit
        with pytest.raises(NotImplementedError, match="OM-032"):
            tier_fit(None, "edge")


# ---------------------------------------------------------------------------
# T3 — manifest formats[] seam and back-compat
# ---------------------------------------------------------------------------

class TestManifestFormatsSeam:
    """T3: write_manifest emits formats[]; existing readers stay intact."""

    def test_write_manifest_emits_formats_list(self, tmp_path):
        from openmed.mlx.artifact import write_manifest

        (tmp_path / "weights.safetensors").write_bytes(b"")
        manifest_path = write_manifest(
            tmp_path,
            source_model_id="test/model",
            config={"model_type": "bert", "max_position_embeddings": 128},
        )

        manifest = json.loads(manifest_path.read_text())
        assert "formats" in manifest, "write_manifest must emit formats[]"
        assert isinstance(manifest["formats"], list)
        assert len(manifest["formats"]) >= 1

    def test_write_manifest_includes_mlx_format(self, tmp_path):
        from openmed.mlx.artifact import write_manifest, MANIFEST_FORMAT

        manifest_path = write_manifest(
            tmp_path,
            source_model_id="test/model",
            config={"model_type": "bert"},
        )
        manifest = json.loads(manifest_path.read_text())
        assert MANIFEST_FORMAT in manifest["formats"]

    def test_write_manifest_keeps_format_scalar(self, tmp_path):
        """Back-compat: format scalar must be preserved for existing readers."""
        from openmed.mlx.artifact import write_manifest, MANIFEST_FORMAT

        manifest_path = write_manifest(
            tmp_path,
            source_model_id="test/model",
            config={"model_type": "bert"},
        )
        manifest = json.loads(manifest_path.read_text())
        assert manifest["format"] == MANIFEST_FORMAT

    def test_load_artifact_config_old_manifest(self, tmp_path):
        """load_artifact_config must work with pre-formats[] manifests."""
        from openmed.mlx.artifact import load_artifact_config

        old_manifest = {
            "format": "openmed-mlx",
            "format_version": 2,
            "source_model_id": "test/model",
            "config_path": "config.json",
        }
        (tmp_path / "openmed-mlx.json").write_text(json.dumps(old_manifest))
        (tmp_path / "config.json").write_text(json.dumps({"num_labels": 3}))

        manifest, config = load_artifact_config(tmp_path)
        assert manifest is not None
        assert config["num_labels"] == 3
        assert "formats" not in manifest  # reader doesn't add it

    def test_load_artifact_config_new_manifest(self, tmp_path):
        """load_artifact_config must work with formats[] manifests."""
        from openmed.mlx.artifact import load_artifact_config

        new_manifest = {
            "format": "openmed-mlx",
            "formats": ["openmed-mlx"],
            "format_version": 2,
            "source_model_id": "test/model",
            "config_path": "config.json",
        }
        (tmp_path / "openmed-mlx.json").write_text(json.dumps(new_manifest))
        (tmp_path / "config.json").write_text(json.dumps({"num_labels": 5}))

        manifest, config = load_artifact_config(tmp_path)
        assert manifest is not None
        assert "formats" in manifest
        assert config["num_labels"] == 5

    def test_resolve_weight_candidates_old_manifest(self, tmp_path):
        """resolve_weight_candidates must work with pre-formats[] manifests."""
        from openmed.mlx.artifact import resolve_weight_candidates

        (tmp_path / "weights.safetensors").write_bytes(b"")
        old_manifest = {
            "format": "openmed-mlx",
            "preferred_weights": "weights.safetensors",
            "fallback_weights": ["weights.npz"],
            "available_weights": ["weights.safetensors"],
        }
        candidates = resolve_weight_candidates(tmp_path, manifest=old_manifest)
        assert any("weights.safetensors" in str(c) for c in candidates)

    def test_resolve_weight_candidates_new_manifest(self, tmp_path):
        """resolve_weight_candidates must work with formats[] manifests."""
        from openmed.mlx.artifact import resolve_weight_candidates

        (tmp_path / "weights.safetensors").write_bytes(b"")
        new_manifest = {
            "format": "openmed-mlx",
            "formats": ["openmed-mlx"],
            "preferred_weights": "weights.safetensors",
            "fallback_weights": ["weights.npz"],
            "available_weights": ["weights.safetensors"],
        }
        candidates = resolve_weight_candidates(tmp_path, manifest=new_manifest)
        assert any("weights.safetensors" in str(c) for c in candidates)


# ---------------------------------------------------------------------------
# T1 — logit parity: ONNX CPU EP vs torch
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (_HAS_TORCH and _HAS_ONNXRUNTIME and _HAS_TRANSFORMERS),
    reason="torch, onnxruntime, and transformers required for T1 parity test",
)
class TestOnnxLogitParity:
    """T1: Exported ONNX logits must match torch within 1e-3 on clinical token sequences."""

    def test_parity_batch_clinical_strings(self, tmp_path):
        import numpy as np
        import torch
        import onnxruntime as ort
        from transformers import BertConfig, BertForTokenClassification

        # Tiny model — no download
        config = BertConfig(
            vocab_size=512,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            num_labels=4,
            id2label={0: "O", 1: "B-NAME", 2: "I-NAME", 3: "B-DATE"},
            label2id={"O": 0, "B-NAME": 1, "I-NAME": 2, "B-DATE": 3},
        )
        torch.manual_seed(42)
        model = BertForTokenClassification(config)
        model.eval()

        class _Wrapper(torch.nn.Module):
            def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
                return model(input_ids=input_ids, attention_mask=attention_mask).logits

        wrapper = _Wrapper()
        wrapper.eval()

        # "Batch of clinical strings" as token-id sequences (vocab 512, seq 32, batch 4)
        torch.manual_seed(7)
        batch_size, seq_len = 4, 32
        input_ids = torch.randint(0, 512, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        # Torch baseline
        with torch.no_grad():
            torch_logits = wrapper(input_ids, attention_mask).numpy()

        # Export — trace with batch=2 so torch.export doesn't specialize the
        # batch dimension as a constant (BERT position embeddings trigger that
        # specialization when traced with batch=1).
        onnx_path = tmp_path / "tiny_tc.onnx"
        _major = int(torch.__version__.split(".")[0])
        _minor = int(torch.__version__.split(".")[1])
        with torch.no_grad():
            if _major > 2 or (_major == 2 and _minor >= 1):
                from torch.export import Dim as _Dim
                _batch = _Dim("batch_size")
                _seq = _Dim("sequence_length", max=seq_len)
                torch.onnx.export(
                    wrapper,
                    (input_ids[:2], attention_mask[:2]),
                    str(onnx_path),
                    input_names=["input_ids", "attention_mask"],
                    output_names=["logits"],
                    dynamic_shapes={
                        "input_ids": {0: _batch, 1: _seq},
                        "attention_mask": {0: _batch, 1: _seq},
                    },
                    opset_version=18,
                )
            else:
                torch.onnx.export(
                    wrapper,
                    (input_ids[:2], attention_mask[:2]),
                    str(onnx_path),
                    input_names=["input_ids", "attention_mask"],
                    output_names=["logits"],
                    dynamic_axes={
                        "input_ids": {0: "batch_size", 1: "sequence_length"},
                        "attention_mask": {0: "batch_size", 1: "sequence_length"},
                        "logits": {0: "batch_size", 1: "sequence_length"},
                    },
                    opset_version=14,
                    do_constant_folding=True,
                )

        # ONNX CPU EP inference
        sess = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        ort_logits = sess.run(
            ["logits"],
            {
                "input_ids": input_ids.numpy().astype(np.int64),
                "attention_mask": attention_mask.numpy().astype(np.int64),
            },
        )[0]

        # T1 assertions
        max_abs_diff = float(np.max(np.abs(torch_logits - ort_logits)))
        assert max_abs_diff < 1e-3, (
            f"max abs logit diff {max_abs_diff:.2e} >= 1e-3"
        )
        assert np.array_equal(
            np.argmax(torch_logits, axis=-1),
            np.argmax(ort_logits, axis=-1),
        ), "per-token argmax disagreement between torch and ONNX CPU EP"

        # Side-effect: write Node fixture for T4 (skipped silently if dir missing)
        self._write_node_fixture(
            onnx_path,
            torch_logits,
            input_ids.numpy(),
            attention_mask.numpy(),
        )

    @staticmethod
    def _write_node_fixture(
        onnx_path: Path,
        expected_logits: "numpy.ndarray",  # type: ignore[name-defined]
        input_ids: "numpy.ndarray",  # type: ignore[name-defined]
        attention_mask: "numpy.ndarray",  # type: ignore[name-defined]
    ) -> None:
        """Copy the ONNX file and expected values to tests/node/fixtures/ for T4."""
        import numpy as np

        fixture_dir = _NODE_FIXTURE_DIR
        try:
            fixture_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(onnx_path, fixture_dir / "tiny_model.onnx")
            batch_size, seq_len = input_ids.shape
            expected = {
                "batch_size": int(batch_size),
                "seq_len": int(seq_len),
                "num_labels": int(expected_logits.shape[-1]),
                "input_ids": input_ids.astype(np.int64).tolist(),
                "attention_mask": attention_mask.astype(np.int64).tolist(),
                "logits": expected_logits.astype(np.float32).tolist(),
            }
            (fixture_dir / "tiny_expected.json").write_text(
                json.dumps(expected, indent=2)
            )
        except OSError:
            pass  # non-fatal — T4 will skip gracefully if fixture is absent
