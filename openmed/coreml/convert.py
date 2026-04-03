"""Convert HuggingFace token-classification models to CoreML format.

Produces a ``.mlpackage`` suitable for iOS 16+ and macOS 13+ deployment.

Usage::

    python -m openmed.coreml.convert \\
        --model OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1 \\
        --output ./OpenMedPIISmall.mlpackage
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
    compute_precision: str = "float16",
    cache_dir: Optional[str] = None,
) -> Path:
    """Convert a HuggingFace token-classification model to CoreML.

    Args:
        model_id: HuggingFace model identifier.
        output_path: Destination for the ``.mlpackage`` file.
        max_seq_length: Maximum input sequence length.
        compute_precision: ``"float16"`` (Neural Engine) or ``"float32"`` (CPU).
        cache_dir: HuggingFace cache directory.

    Returns:
        Path to the created ``.mlpackage``.
    """
    try:
        import torch
        import coremltools as ct
        from transformers import AutoTokenizer, AutoModelForTokenClassification
    except ImportError as e:
        raise ImportError(
            f"Missing dependency: {e}. "
            "Install with: pip install openmed[coreml]"
        )

    output_path = Path(output_path)

    # 1. Load model and tokenizer
    logger.info("Loading HuggingFace model %s ...", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForTokenClassification.from_pretrained(
        model_id, cache_dir=cache_dir,
    )
    model.eval()

    num_labels = model.config.num_labels
    id2label = model.config.id2label

    # 2. Create wrapper that returns only logits (not ModelOutput)
    class TokenClassificationWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, input_ids, attention_mask):
            output = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            return output.logits

    wrapper = TokenClassificationWrapper(model)
    wrapper.eval()

    # 3. Trace with sample inputs
    logger.info("Tracing model with sequence length %d ...", max_seq_length)
    sample_text = "Patient John Doe visited the clinic on 2024-01-15."
    sample = tokenizer(
        sample_text,
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    traced = torch.jit.trace(
        wrapper,
        (sample["input_ids"], sample["attention_mask"]),
    )

    # 4. Convert to CoreML
    logger.info("Converting to CoreML (%s precision) ...", compute_precision)

    ct_precision = (
        ct.precision.FLOAT16 if compute_precision == "float16"
        else ct.precision.FLOAT32
    )

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="input_ids",
                shape=ct.Shape(
                    shape=(1, ct.RangeDim(lower_bound=1, upper_bound=max_seq_length, default=128)),
                ),
                dtype=int,
            ),
            ct.TensorType(
                name="attention_mask",
                shape=ct.Shape(
                    shape=(1, ct.RangeDim(lower_bound=1, upper_bound=max_seq_length, default=128)),
                ),
                dtype=int,
            ),
        ],
        outputs=[
            ct.TensorType(name="logits"),
        ],
        compute_precision=ct_precision,
        minimum_deployment_target=ct.target.iOS16,
    )

    # 5. Add metadata
    mlmodel.short_description = (
        f"OpenMed Token Classification: {model_id} "
        f"({num_labels} labels, max_seq={max_seq_length})"
    )
    mlmodel.author = "OpenMed"
    mlmodel.license = "Apache-2.0"

    # Store id2label as user-defined metadata
    mlmodel.user_defined_metadata["id2label"] = json.dumps(
        {str(k): v for k, v in id2label.items()}
    )
    mlmodel.user_defined_metadata["num_labels"] = str(num_labels)
    mlmodel.user_defined_metadata["max_seq_length"] = str(max_seq_length)
    mlmodel.user_defined_metadata["source_model"] = model_id

    # 6. Save
    logger.info("Saving to %s ...", output_path)
    mlmodel.save(str(output_path))

    # Also save id2label.json alongside for easy access
    id2label_path = output_path.parent / f"{output_path.stem}_id2label.json"
    with open(id2label_path, "w") as f:
        json.dump({str(k): v for k, v in id2label.items()}, f, indent=2)

    logger.info("CoreML model saved to %s", output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert a HuggingFace token-classification model to CoreML format",
    )
    parser.add_argument(
        "--model", required=True,
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for .mlpackage file",
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=512,
        help="Maximum input sequence length (default: 512)",
    )
    parser.add_argument(
        "--precision", choices=["float16", "float32"], default="float16",
        help="Compute precision (default: float16 for Neural Engine)",
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="HuggingFace model cache directory",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    convert(
        args.model,
        args.output,
        max_seq_length=args.max_seq_length,
        compute_precision=args.precision,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
