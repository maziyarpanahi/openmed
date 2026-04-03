"""Tests for CoreML conversion script."""

from __future__ import annotations

import pytest


class TestCoreMLConvertModule:
    """Verify the coreml.convert module is importable and has expected API."""

    def test_module_importable(self):
        from openmed.coreml import convert
        assert hasattr(convert, "convert")
        assert hasattr(convert, "main")

    def test_convert_signature(self):
        """convert() should accept model_id, output_path, and options."""
        import inspect
        from openmed.coreml.convert import convert

        sig = inspect.signature(convert)
        params = list(sig.parameters.keys())
        assert "model_id" in params
        assert "output_path" in params
        assert "max_seq_length" in params
        assert "compute_precision" in params

    def test_main_exists(self):
        from openmed.coreml.convert import main
        assert callable(main)

    def test_uses_onnx_pipeline(self):
        """Verify the converter uses ONNX as intermediate format."""
        import inspect
        from openmed.coreml.convert import convert
        source = inspect.getsource(convert)
        assert "torch.onnx.export" in source
        assert "onnx_path" in source
