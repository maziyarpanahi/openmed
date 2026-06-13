"""Convert an fp32 ONNX export to a fp16 artifact for onnxruntime-web WebGPU EP.

The WebGPU execution provider in onnxruntime-web operates natively in fp16.
Shipping a fp16 model file gives:
  - ~50 % smaller download than fp32
  - No client-side cast overhead — weights land directly on GPU as fp16
  - Correct I/O types: inputs remain int64, outputs remain float32 (keep_io_types)

Usage::

    # standalone CLI
    python -m openmed.onnx.webgpu --input model.onnx

    # or from Python
    from openmed.onnx.webgpu import convert_to_fp16
    fp16_path = convert_to_fp16("model.onnx")
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def convert_to_fp16(
    input_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Convert an fp32 ONNX model to fp16 for WebGPU deployment.

    Uses onnxruntime's transformer-aware float16 converter so that ops
    sensitive to fp16 overflow (e.g. Softmax, LayerNorm) are handled
    correctly. keep_io_types=True preserves int64 inputs and float32
    outputs so onnxruntime-web's WebGPU EP can bind them without casting.

    Args:
        input_path: Path to the fp32 ``.onnx`` file produced by convert().
        output_path: Destination for the fp16 ``.onnx`` file.
            Defaults to ``<stem>_fp16.onnx`` in the same directory.

    Returns:
        Path to the created fp16 ``.onnx`` file.
    """
    try:
        import onnx
        from onnxruntime.transformers.float16 import convert_float_to_float16
    except ImportError as e:
        raise ImportError(
            f"Missing dependency: {e}. "
            "Install with: pip install openmed[onnx]"
        )

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {input_path}")

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_fp16.onnx"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading fp32 ONNX model from %s ...", input_path)
    model = onnx.load(str(input_path))

    logger.info("Converting weights and activations to fp16 ...")
    fp16_model = convert_float_to_float16(
        model,
        keep_io_types=True,        # inputs stay int64, outputs stay float32
        disable_shape_infer=False, # run shape inference for better conversion
    )

    onnx.save(fp16_model, str(output_path))

    fp32_kb = input_path.stat().st_size / 1024
    fp16_kb = output_path.stat().st_size / 1024
    logger.info(
        "fp16 model saved to %s (%.0f KB → %.0f KB, %.0f%% of original)",
        output_path, fp32_kb, fp16_kb, 100 * fp16_kb / fp32_kb,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a fp32 ONNX model to fp16 for onnxruntime-web WebGPU EP",
    )
    parser.add_argument("--input", required=True, help="Input fp32 .onnx file")
    parser.add_argument(
        "--output", default=None,
        help="Output fp16 .onnx file (default: <stem>_fp16.onnx)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    convert_to_fp16(args.input, args.output)


if __name__ == "__main__":
    main()
