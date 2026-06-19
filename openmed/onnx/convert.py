"""Convert token-classification checkpoints to ONNX and WebGPU artifacts."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from openmed.core.hf_publish import publish_artifact
from openmed.mlx.artifact import find_tokenizer_files

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "openmed-onnx.json"
MANIFEST_FORMAT = "openmed-onnx"
MANIFEST_VERSION = 1
DEFAULT_ONNX_FILENAME = "model.onnx"
DEFAULT_WEBGPU_FILENAME = "model.webgpu.onnx"
DEFAULT_SAMPLE_TEXT = "Patient John Doe visited the clinic on 2024-01-15."


@dataclass(frozen=True)
class ExportArtifact:
    """One exported runtime artifact inside an ONNX conversion output."""

    format: str
    path: Path
    precision: str

    def to_manifest(self, root: Path) -> dict[str, str]:
        return {
            "format": self.format,
            "path": self.path.relative_to(root).as_posix(),
            "precision": self.precision,
        }


@dataclass(frozen=True)
class OnnxConversionResult:
    """Paths and manifest data produced by an ONNX conversion."""

    output_dir: Path
    manifest_path: Path
    artifacts: tuple[ExportArtifact, ...]

    @property
    def formats(self) -> list[str]:
        return [artifact.format for artifact in self.artifacts]


def export_onnx(
    model_id: str,
    output_path: str | Path,
    *,
    max_seq_length: int = 512,
    opset: int = 18,
    cache_dir: str | None = None,
    sample_text: str = DEFAULT_SAMPLE_TEXT,
) -> Path:
    """Export a Hugging Face token-classification checkpoint to ONNX."""

    try:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "torch and transformers are required for ONNX export. "
            "Install with: pip install openmed[onnx]"
        ) from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForTokenClassification.from_pretrained(
        model_id,
        cache_dir=cache_dir,
    )
    model.eval()

    sample = tokenizer(
        sample_text,
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    use_token_type_ids = "token_type_ids" in sample

    class TokenClassificationWrapper(torch.nn.Module):
        def __init__(self, base_model: Any, include_token_type_ids: bool) -> None:
            super().__init__()
            self.base_model = base_model
            self.include_token_type_ids = include_token_type_ids

        def forward(self, input_ids: Any, attention_mask: Any, token_type_ids: Any = None) -> Any:
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if self.include_token_type_ids:
                inputs["token_type_ids"] = token_type_ids
            return self.base_model(**inputs).logits

    wrapper = TokenClassificationWrapper(model, use_token_type_ids)
    wrapper.eval()

    input_names = ["input_ids", "attention_mask"]
    example_inputs: tuple[Any, ...] = (sample["input_ids"], sample["attention_mask"])
    if use_token_type_ids:
        input_names.append("token_type_ids")
        example_inputs = (*example_inputs, sample["token_type_ids"])

    dynamic_axes = {
        name: {0: "batch", 1: "sequence"} for name in [*input_names, "logits"]
    }

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            example_inputs,
            str(output_path),
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
        )

    _check_onnx_model(output_path)
    return output_path


def export_webgpu(
    onnx_path: str | Path,
    output_path: str | Path,
    *,
    keep_io_types: bool = True,
) -> Path:
    """Create a WebGPU-targeted fp16 ONNX artifact from a fp32 ONNX model."""

    try:
        import onnx
        from onnxruntime.transformers.float16 import convert_float_to_float16
    except ImportError as exc:
        raise ImportError(
            "onnx and onnxruntime are required for WebGPU export. "
            "Install with: pip install openmed[onnx]"
        ) from exc

    onnx_path = Path(onnx_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(onnx_path))
    fp16_model = convert_float_to_float16(model, keep_io_types=keep_io_types)
    onnx.save(fp16_model, str(output_path))
    _check_onnx_model(output_path)
    return output_path


def convert(
    model_id: str,
    output_dir: str | Path,
    *,
    include_webgpu: bool = True,
    max_seq_length: int = 512,
    opset: int = 18,
    cache_dir: str | None = None,
    publish_to_hub: bool = False,
    publish_repo_id: str | None = None,
    publish_org: str = "OpenMed",
    publish_version: int = 1,
    publish_manifest_path: str | Path | None = None,
    publish_token_env: str = "HF_WRITE_TOKEN",
    publish_private: bool = False,
    publish_overwrite_existing: bool = False,
) -> OnnxConversionResult:
    """Export ONNX artifacts and optionally publish the resulting directory."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = export_onnx(
        model_id,
        output_dir / DEFAULT_ONNX_FILENAME,
        max_seq_length=max_seq_length,
        opset=opset,
        cache_dir=cache_dir,
    )
    artifacts = [
        ExportArtifact(format="onnx", path=onnx_path, precision="float32"),
    ]

    if include_webgpu:
        webgpu_path = export_webgpu(
            onnx_path,
            output_dir / DEFAULT_WEBGPU_FILENAME,
        )
        artifacts.append(
            ExportArtifact(format="webgpu", path=webgpu_path, precision="float16")
        )

    config, tokenizer_files = save_source_assets(
        model_id,
        output_dir,
        cache_dir=cache_dir,
        max_seq_length=max_seq_length,
    )
    manifest_path = write_export_manifest(
        output_dir,
        source_model_id=model_id,
        config=config,
        artifacts=artifacts,
        tokenizer_files=tokenizer_files,
    )
    result = OnnxConversionResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        artifacts=tuple(artifacts),
    )

    if publish_to_hub:
        publish_result = publish_artifact(
            artifact_dir=output_dir,
            source_model_id=model_id,
            format_name=result.formats[0],
            formats=result.formats,
            repo_id=publish_repo_id,
            org=publish_org,
            version=publish_version,
            token_env=publish_token_env,
            manifest_path=publish_manifest_path,
            private=publish_private,
            skip_existing=not publish_overwrite_existing,
        )
        if publish_result.skipped:
            logger.info("Skipped existing Hub repo %s", publish_result.repo_id)
        else:
            logger.info("Published ONNX artifacts to %s", publish_result.repo_id)

    return result


def save_source_assets(
    model_id: str,
    output_dir: str | Path,
    *,
    cache_dir: str | None = None,
    max_seq_length: int = 512,
) -> tuple[dict[str, Any], list[str]]:
    """Save config, labels, and tokenizer assets beside exported artifacts."""

    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required for ONNX export metadata. "
            "Install with: pip install openmed[onnx]"
        ) from exc

    output_dir = Path(output_dir)
    config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir).to_dict()
    config["max_sequence_length"] = max_seq_length

    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    id2label = config.get("id2label")
    if isinstance(id2label, Mapping):
        with (output_dir / "id2label.json").open("w", encoding="utf-8") as handle:
            json.dump({str(key): value for key, value in id2label.items()}, handle, indent=2)

    tokenizer_files: list[str] = []
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        tokenizer.save_pretrained(output_dir)
        tokenizer_files = find_tokenizer_files(output_dir)
    except Exception as exc:
        logger.warning(
            "Could not save tokenizer assets for %s into %s: %s",
            model_id,
            output_dir,
            exc,
        )

    return config, tokenizer_files


def write_export_manifest(
    output_dir: str | Path,
    *,
    source_model_id: str,
    config: Mapping[str, Any],
    artifacts: Sequence[ExportArtifact],
    tokenizer_files: Sequence[str] | None = None,
) -> Path:
    """Write an ONNX/WebGPU artifact manifest into *output_dir*."""

    output_dir = Path(output_dir)
    formats = _dedupe_keep_order([artifact.format for artifact in artifacts])
    task = str(config.get("_mlx_task") or config.get("task") or "token-classification")
    family = str(
        config.get("_mlx_family")
        or config.get("_mlx_model_type")
        or config.get("model_type")
        or "unknown"
    )

    manifest = {
        "format": MANIFEST_FORMAT,
        "format_version": MANIFEST_VERSION,
        "formats": formats,
        "task": task,
        "family": family,
        "source_model_id": source_model_id,
        "config_path": "config.json",
        "label_map_path": "id2label.json" if (output_dir / "id2label.json").exists() else None,
        "artifacts": [artifact.to_manifest(output_dir) for artifact in artifacts],
        "max_sequence_length": config.get("max_sequence_length")
        or config.get("max_position_embeddings", 512),
        "tokenizer": {
            "path": ".",
            "files": list(tokenizer_files or find_tokenizer_files(output_dir)),
        },
    }

    manifest_path = output_dir / MANIFEST_FILENAME
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest_path


def _check_onnx_model(path: Path) -> None:
    try:
        import onnx
    except ImportError as exc:
        raise ImportError(
            "onnx is required to validate exported artifacts. "
            "Install with: pip install openmed[onnx]"
        ) from exc
    model = onnx.load(str(path))
    onnx.checker.check_model(model)


def _dedupe_keep_order(items: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a Hugging Face token-classification model to ONNX artifacts",
    )
    parser.add_argument("--model", required=True, help="Source model ID or local directory")
    parser.add_argument("--output", required=True, help="Output directory for artifacts")
    parser.add_argument(
        "--no-webgpu",
        action="store_true",
        help="Only emit the fp32 ONNX artifact",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum input sequence length",
    )
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    parser.add_argument("--cache-dir", default=None, help="Hugging Face cache directory")
    parser.add_argument(
        "--publish-to-hub",
        action="store_true",
        help="Publish the converted artifact after a successful conversion",
    )
    parser.add_argument("--publish-repo-id", default=None, help="Explicit target repo id")
    parser.add_argument("--publish-org", default="OpenMed", help="Target organization")
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
        include_webgpu=not args.no_webgpu,
        max_seq_length=args.max_seq_length,
        opset=args.opset,
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
