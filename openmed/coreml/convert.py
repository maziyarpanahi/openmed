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

from openmed.core.hf_publish import publish_artifact

logger = logging.getLogger(__name__)


def convert(
    model_id: str,
    output_path: str | Path,
    max_seq_length: int = 512,
    compute_precision: str = "float16",
    cache_dir: Optional[str] = None,
    publish_to_hub: bool = False,
    publish_repo_id: str | None = None,
    publish_org: str = "OpenMed",
    publish_version: int = 1,
    publish_manifest_path: str | Path | None = None,
    publish_token_env: str = "HF_WRITE_TOKEN",
    publish_private: bool = False,
    publish_overwrite_existing: bool = False,
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
    if publish_to_hub:
        result = publish_artifact(
            artifact_dir=output_path,
            source_model_id=model_id,
            format_name="coreml",
            repo_id=publish_repo_id,
            org=publish_org,
            version=publish_version,
            token_env=publish_token_env,
            manifest_path=publish_manifest_path,
            private=publish_private,
            skip_existing=not publish_overwrite_existing,
        )
        if result.skipped:
            logger.info("Skipped existing Hub repo %s", result.repo_id)
        else:
            logger.info("Published CoreML artifact to %s", result.repo_id)
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
    parser.add_argument(
        "--publish-to-hub",
        action="store_true",
        help="Publish the converted artifact after a successful conversion",
    )
    parser.add_argument(
        "--publish-repo-id",
        default=None,
        help="Explicit target repo id for publishing",
    )
    parser.add_argument(
        "--publish-org",
        default="OpenMed",
        help="Target organization for derived publish repo ids",
    )
    parser.add_argument(
        "--publish-version",
        type=int,
        default=1,
        help="Version suffix used when the source repo is not already versioned",
    )
    parser.add_argument(
        "--publish-manifest",
        default=None,
        help="JSONL manifest path to append or update after publishing",
    )
    parser.add_argument(
        "--publish-token-env",
        default="HF_WRITE_TOKEN",
        help="Environment variable containing the Hub write token",
    )
    parser.add_argument(
        "--publish-private",
        action="store_true",
        help="Create the target repo as private when it does not exist",
    )
    parser.add_argument(
        "--publish-overwrite-existing",
        action="store_true",
        help="Upload into an existing target repo instead of skipping it",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    convert(
        args.model,
        args.output,
        max_seq_length=args.max_seq_length,
        compute_precision=args.precision,
        cache_dir=args.cache_dir,
        publish_to_hub=args.publish_to_hub,
        publish_repo_id=args.publish_repo_id,
        publish_org=args.publish_org,
        publish_version=args.publish_version,
        publish_manifest_path=args.publish_manifest,
        publish_token_env=args.publish_token_env,
        publish_private=args.publish_private,
        publish_overwrite_existing=args.publish_overwrite_existing,
    )


if __name__ == "__main__":
    main()
