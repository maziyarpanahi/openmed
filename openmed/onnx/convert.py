"""Convert HuggingFace token-classification models to ONNX format.

Produces a ``.onnx`` file suitable for onnxruntime (CPU/CUDA/CoreML EPs)
and onnxruntime-web (WASM/WebGPU EP) in browsers.

Usage::

    python -m openmed.onnx.convert \\
        --model OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1 \\
        --output ./model.onnx
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def convert(
    model_id: str,
    output_path: str | Path,
    max_seq_length: int = 512,
    opset_version: int = 18,
    cache_dir: Optional[str] = None,
) -> Path:
    """Convert a HuggingFace token-classification model to ONNX.

    Args:
        model_id: HuggingFace model identifier or local directory.
        output_path: Destination for the ``.onnx`` file.
        max_seq_length: Maximum input sequence length (used for the export trace).
        opset_version: ONNX opset version (default 18; required by torch >= 2.1
            dynamo exporter which uses LayerNormalization-18).
        cache_dir: HuggingFace cache directory.

    Returns:
        Path to the created ``.onnx`` file.
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForTokenClassification
    except ImportError as e:
        raise ImportError(
            f"Missing dependency: {e}. "
            "Install with: pip install openmed[onnx]"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load model and tokenizer
    logger.info("Loading HuggingFace model %s ...", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForTokenClassification.from_pretrained(
        model_id, cache_dir=cache_dir,
    )
    model.eval()

    num_labels = model.config.num_labels
    id2label = model.config.id2label

    # 2. Wrapper returning bare logits (not ModelOutput) — same pattern as coreml/convert.py
    class _TokenClassificationWrapper(torch.nn.Module):
        def __init__(self, base_model: torch.nn.Module) -> None:
            super().__init__()
            self.base_model = base_model

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
        ) -> torch.Tensor:
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits

    wrapper = _TokenClassificationWrapper(model)
    wrapper.eval()

    # 3. Prepare two-example trace inputs.
    # Tracing with batch=2 (not 1) is required so that torch.export does not
    # specialize the batch dimension as a constant — BERT position embeddings
    # trigger that specialization when the traced batch size is 1.
    logger.info("Preparing export trace (max_seq_length=%d) ...", max_seq_length)
    sample_text = "Patient John Doe visited the clinic on 2024-01-15."
    sample = tokenizer(
        [sample_text, sample_text],
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    dummy_ids = sample["input_ids"]    # shape: (2, max_seq_length)
    dummy_mask = sample["attention_mask"]

    # 4. Export with dynamic batch and sequence dimensions.
    # torch >= 2.1 uses the dynamo exporter by default; dynamic_shapes with
    # torch.export.Dim objects is the correct API (dynamic_axes is deprecated
    # and silently produces broken output shapes in 2.x). Fall back to the
    # legacy TorchScript exporter for torch < 2.1.
    logger.info("Exporting to ONNX (opset %d) ...", opset_version)
    _torch_major = int(torch.__version__.split(".")[0])
    _torch_minor = int(torch.__version__.split(".")[1])
    with torch.no_grad():
        if _torch_major > 2 or (_torch_major == 2 and _torch_minor >= 1):
            from torch.export import Dim as _Dim
            _batch = _Dim("batch_size")
            _seq = _Dim("sequence_length", max=max_seq_length)
            torch.onnx.export(
                wrapper,
                (dummy_ids, dummy_mask),
                str(output_path),
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_shapes={
                    "input_ids": {0: _batch, 1: _seq},
                    "attention_mask": {0: _batch, 1: _seq},
                },
                opset_version=opset_version,
            )
        else:
            torch.onnx.export(
                wrapper,
                (dummy_ids, dummy_mask),
                str(output_path),
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "logits": {0: "batch_size", 1: "sequence_length"},
                },
                opset_version=opset_version,
                do_constant_folding=True,
            )

    # 5. Write id2label.json alongside (same convention as coreml/convert.py)
    id2label_path = output_path.parent / f"{output_path.stem}_id2label.json"
    with open(id2label_path, "w") as f:
        json.dump({str(k): v for k, v in id2label.items()}, f, indent=2)

    logger.info(
        "ONNX model saved to %s (num_labels=%d, opset=%d)",
        output_path, num_labels, opset_version,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a HuggingFace token-classification model to ONNX",
    )
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--output", required=True, help="Output path for .onnx file")
    parser.add_argument(
        "--max-seq-length", type=int, default=512,
        help="Maximum input sequence length (default: 512)",
    )
    parser.add_argument(
        "--opset", type=int, default=18,
        help="ONNX opset version (default: 18)",
    )
    parser.add_argument("--cache-dir", default=None, help="HuggingFace cache directory")
    parser.add_argument(
        "--webgpu", action="store_true",
        help="Also produce a fp16 variant for onnxruntime-web WebGPU EP",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    out = convert(
        args.model,
        args.output,
        max_seq_length=args.max_seq_length,
        opset_version=args.opset,
        cache_dir=args.cache_dir,
    )
    if args.webgpu:
        from openmed.onnx.webgpu import convert_to_fp16
        convert_to_fp16(out)


if __name__ == "__main__":
    main()
