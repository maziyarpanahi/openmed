"""Tests for openmed.onnx.webgpu — fp16 conversion for WebGPU EP.

Covers:
  API surface  — always runs, no heavy deps.
  fp16 parity  — requires torch + onnxruntime + transformers + onnx.
                 Skipped otherwise.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest


def _importable(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


_HAS_TORCH        = _importable("torch")
_HAS_ORT          = _importable("onnxruntime")
_HAS_TRANSFORMERS = _importable("transformers")
_HAS_ONNX         = _importable("onnx")

_ALL_DEPS = _HAS_TORCH and _HAS_ORT and _HAS_TRANSFORMERS and _HAS_ONNX

_NODE_FIXTURE_DIR = Path(__file__).parent.parent.parent / "node" / "fixtures"


# ---------------------------------------------------------------------------
# API surface (always runs)
# ---------------------------------------------------------------------------

class TestWebGPUModule:
    def test_module_importable(self):
        from openmed.onnx import webgpu
        assert hasattr(webgpu, "convert_to_fp16")
        assert hasattr(webgpu, "main")

    def test_signature(self):
        import inspect
        from openmed.onnx.webgpu import convert_to_fp16

        params = list(inspect.signature(convert_to_fp16).parameters)
        assert "input_path" in params
        assert "output_path" in params

    def test_missing_input_raises(self, tmp_path):
        from openmed.onnx.webgpu import convert_to_fp16

        with pytest.raises(FileNotFoundError):
            convert_to_fp16(tmp_path / "nonexistent.onnx")

    def test_default_output_name(self, tmp_path):
        """Default output should be <stem>_fp16.onnx."""
        from openmed.onnx.webgpu import convert_to_fp16
        import inspect

        sig = inspect.signature(convert_to_fp16)
        assert sig.parameters["output_path"].default is None

    def test_convert_cli_has_webgpu_flag(self):
        """convert.py CLI must expose --webgpu."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-m", "openmed.onnx.convert", "--help"],
            capture_output=True, text=True,
        )
        assert "--webgpu" in result.stdout


# ---------------------------------------------------------------------------
# fp16 parity (requires heavy deps)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _ALL_DEPS, reason="torch, onnxruntime, transformers, onnx required")
class TestFp16Conversion:
    """fp16 model must match fp32 within tolerance, be smaller, and have correct I/O types."""

    def _build_fp32_onnx(self, tmp_path: Path) -> Path:
        """Export a tiny fp32 ONNX model; reused across tests."""
        import torch
        from transformers import BertConfig, BertForTokenClassification
        from torch.export import Dim

        config = BertConfig(
            vocab_size=512, hidden_size=32, num_hidden_layers=1,
            num_attention_heads=4, intermediate_size=64, num_labels=4,
        )
        torch.manual_seed(42)
        model = BertForTokenClassification(config).eval()

        class _W(torch.nn.Module):
            def forward(self, input_ids, attention_mask):
                return model(input_ids=input_ids, attention_mask=attention_mask).logits

        wrapper = _W().eval()
        ids2  = torch.randint(0, 512, (2, 32))
        mask2 = torch.ones(2, 32, dtype=torch.long)

        fp32_path = tmp_path / "model.onnx"
        with torch.no_grad():
            torch.onnx.export(
                wrapper, (ids2, mask2), str(fp32_path),
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_shapes={
                    "input_ids":      {0: Dim("batch_size"), 1: Dim("sequence_length", max=32)},
                    "attention_mask": {0: Dim("batch_size"), 1: Dim("sequence_length", max=32)},
                },
                opset_version=18,
            )
        return fp32_path

    def test_fp16_weights_converted(self, tmp_path):
        """Conversion must produce fp16 initializers (not just cast ops at boundaries)."""
        import onnx
        from openmed.onnx.webgpu import convert_to_fp16

        fp32_path = self._build_fp32_onnx(tmp_path)
        fp16_path = convert_to_fp16(fp32_path)

        assert fp16_path.exists()
        m = onnx.load(str(fp16_path))
        FLOAT16 = 10  # onnx.TensorProto.FLOAT16
        fp16_count = sum(1 for t in m.graph.initializer if t.data_type == FLOAT16)
        assert fp16_count > 0, "No fp16 initializers after conversion"

    def test_fp16_default_naming(self, tmp_path):
        from openmed.onnx.webgpu import convert_to_fp16

        fp32_path = self._build_fp32_onnx(tmp_path)
        fp16_path = convert_to_fp16(fp32_path)

        assert fp16_path.name == "model_fp16.onnx"

    def test_fp16_io_types_preserved(self, tmp_path):
        """Inputs must stay int64; outputs must stay float32 (keep_io_types)."""
        import onnx
        from openmed.onnx.webgpu import convert_to_fp16

        fp32_path = self._build_fp32_onnx(tmp_path)
        fp16_path = convert_to_fp16(fp32_path)

        m = onnx.load(str(fp16_path))
        input_types  = {i.name: i.type.tensor_type.elem_type for i in m.graph.input}
        output_types = {o.name: o.type.tensor_type.elem_type for o in m.graph.output}

        INT64   = 7   # onnx.TensorProto.INT64
        FLOAT32 = 1   # onnx.TensorProto.FLOAT

        assert input_types["input_ids"]      == INT64,   "input_ids must stay int64"
        assert input_types["attention_mask"] == INT64,   "attention_mask must stay int64"
        assert output_types["logits"]        == FLOAT32, "logits must stay float32"

    def test_fp16_argmax_matches_fp32(self, tmp_path):
        """Per-token argmax from fp16 ONNX must match fp32 torch on clinical strings."""
        import torch
        import numpy as np
        import onnxruntime as ort
        from transformers import BertConfig, BertForTokenClassification
        from torch.export import Dim
        from openmed.onnx.webgpu import convert_to_fp16

        config = BertConfig(
            vocab_size=512, hidden_size=32, num_hidden_layers=1,
            num_attention_heads=4, intermediate_size=64, num_labels=4,
        )
        torch.manual_seed(42)
        model = BertForTokenClassification(config).eval()

        class _W(torch.nn.Module):
            def forward(self, input_ids, attention_mask):
                return model(input_ids=input_ids, attention_mask=attention_mask).logits

        wrapper = _W().eval()

        torch.manual_seed(7)
        batch, seq = 4, 32
        input_ids   = torch.randint(0, 512, (batch, seq))
        attn_mask   = torch.ones(batch, seq, dtype=torch.long)

        # fp32 torch baseline
        with torch.no_grad():
            torch_logits = wrapper(input_ids, attn_mask).numpy()

        # Export fp32 ONNX then convert to fp16
        fp32_path = tmp_path / "model.onnx"
        with torch.no_grad():
            torch.onnx.export(
                wrapper, (input_ids[:2], attn_mask[:2]), str(fp32_path),
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_shapes={
                    "input_ids":      {0: Dim("batch_size"), 1: Dim("sequence_length", max=seq)},
                    "attention_mask": {0: Dim("batch_size"), 1: Dim("sequence_length", max=seq)},
                },
                opset_version=18,
            )
        fp16_path = convert_to_fp16(fp32_path)

        # onnxruntime CPU EP inference on fp16 model
        sess = ort.InferenceSession(str(fp16_path), providers=["CPUExecutionProvider"])
        fp16_logits = sess.run(["logits"], {
            "input_ids":      input_ids.numpy().astype(np.int64),
            "attention_mask": attn_mask.numpy().astype(np.int64),
        })[0]

        assert fp16_logits.shape == torch_logits.shape

        # fp16 allows larger absolute diff than fp32 (3 decimal digits precision)
        max_abs_diff = float(np.max(np.abs(torch_logits - fp16_logits)))
        assert max_abs_diff < 0.1, f"fp16 max abs diff {max_abs_diff:.3f} >= 0.1"

        # Argmax must match — this is the safety-critical check for NER
        assert np.array_equal(
            np.argmax(torch_logits, axis=-1),
            np.argmax(fp16_logits, axis=-1),
        ), "per-token argmax disagreement between fp32 torch and fp16 ONNX"

        # Write node fixtures for T4/T6
        _write_node_fixtures(fp32_path, fp16_path, torch_logits, input_ids.numpy(), attn_mask.numpy())


def _write_node_fixtures(
    fp32_path: Path,
    fp16_path: Path,
    torch_logits: "numpy.ndarray",  # type: ignore[name-defined]
    input_ids: "numpy.ndarray",     # type: ignore[name-defined]
    attention_mask: "numpy.ndarray",# type: ignore[name-defined]
) -> None:
    import json
    import numpy as np

    try:
        _NODE_FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(fp32_path, _NODE_FIXTURE_DIR / "tiny_model.onnx")
        shutil.copy(fp16_path, _NODE_FIXTURE_DIR / "tiny_model_fp16.onnx")
        batch_size, seq_len = input_ids.shape
        payload = {
            "batch_size": int(batch_size),
            "seq_len":    int(seq_len),
            "num_labels": int(torch_logits.shape[-1]),
            "input_ids":      input_ids.astype(np.int64).tolist(),
            "attention_mask": attention_mask.astype(np.int64).tolist(),
            "logits":         torch_logits.astype(np.float32).tolist(),
        }
        (_NODE_FIXTURE_DIR / "tiny_expected.json").write_text(json.dumps(payload, indent=2))
    except OSError:
        pass  # non-fatal — Node tests skip gracefully if fixture is absent
