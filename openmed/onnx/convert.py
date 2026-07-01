"""Convert token-classification checkpoints to ONNX runtime artifacts.

Use the default profile for a standard fp32 ONNX export plus the WebGPU fp16
variant. Use ``--profile android`` for Android ONNX Runtime Mobile artifacts:
``model.onnx`` and ``model_fp16.onnx`` are exported at the fixed Android opset
with `input_ids` and `attention_mask` inputs, optional `token_type_ids`, a
`logits` output, and named dynamic `batch` and `sequence` axes. The fixed opset
keeps the graph stable for the later `.ort` mobile-format conversion step.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from openmed.core.hf_publish import publish_artifact
from openmed.mlx.artifact import find_tokenizer_files
from openmed.onnx.android_profile import (
    ANDROID_FP16_FILENAME,
    ANDROID_ONNX_FORMAT,
    ANDROID_ONNX_OPSET,
    ANDROID_PROFILE_NAME,
    export_android_fp16,
    validate_android_profile,
)
from openmed.onnx.quantize_int8 import (
    INT8_ONNX_FILENAME,
    ONNX_INT8_FORMAT,
    apply_int8_recall_certification,
    int8_artifact_metadata,
    quantize_dynamic_int8,
    write_int8_recall_delta_report,
)
from openmed.onnx.transformersjs import (
    DEFAULT_BUNDLE_DIRNAME,
    TRANSFORMERSJS_FORMAT,
    export_transformersjs_bundle,
)
from openmed.processing.tokenizer_cache import get_tokenizer_with_loader

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "openmed-onnx.json"
MANIFEST_FORMAT = "openmed-onnx"
MANIFEST_VERSION = 1
DEFAULT_ONNX_FILENAME = "model.onnx"
DEFAULT_WEBGPU_FILENAME = "model.webgpu.onnx"
DEFAULT_ONNX_OPSET = 18
DEFAULT_PROFILE_NAME = "default"
DEFAULT_SAMPLE_TEXT = "Patient John Doe visited the clinic on 2024-01-15."


@dataclass(frozen=True)
class ExportArtifact:
    """One exported runtime artifact inside an ONNX conversion output."""

    format: str
    path: Path
    precision: str
    metadata: Mapping[str, Any] | None = None

    def to_manifest(self, root: Path) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": self.format,
            "path": self.path.relative_to(root).as_posix(),
            "precision": self.precision,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class OnnxConversionResult:
    """Paths and manifest data produced by an ONNX conversion."""

    output_dir: Path
    manifest_path: Path
    artifacts: tuple[ExportArtifact, ...]

    @property
    def formats(self) -> list[str]:
        return _dedupe_keep_order([artifact.format for artifact in self.artifacts])


def export_onnx(
    model_id: str,
    output_path: str | Path,
    *,
    max_seq_length: int = 512,
    opset: int | None = None,
    profile: str = DEFAULT_PROFILE_NAME,
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
    profile = _normalize_profile(profile)
    resolved_opset = _resolve_opset(profile=profile, opset=opset)

    tokenizer = get_tokenizer_with_loader(
        model_id,
        AutoTokenizer.from_pretrained,
        cache_dir=cache_dir,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_id,
        cache_dir=cache_dir,
    )
    model.eval()

    sample = tokenizer(
        [sample_text, sample_text],
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

        def forward(
            self,
            input_ids: Any,
            attention_mask: Any,
            token_type_ids: Any = None,
        ) -> Any:
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

    with torch.no_grad():
        export_kwargs = {
            "input_names": input_names,
            "output_names": ["logits"],
            "opset_version": resolved_opset,
            "do_constant_folding": True,
        }
        export_kwargs.update(
            _dynamic_export_kwargs(
                torch,
                export_fn=torch.onnx.export,
                input_names=input_names,
                max_seq_length=max_seq_length,
                profile=profile,
            )
        )
        torch.onnx.export(
            wrapper,
            example_inputs,
            str(output_path),
            **export_kwargs,
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
    include_transformersjs: bool = False,
    include_int8: bool = True,
    max_seq_length: int = 512,
    opset: int | None = None,
    profile: str = DEFAULT_PROFILE_NAME,
    cache_dir: str | None = None,
    eval_suite_path: str | Path | None = None,
    recall_delta_report_path: str | Path | None = None,
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
    profile = _normalize_profile(profile)

    onnx_path = export_onnx(
        model_id,
        output_dir / DEFAULT_ONNX_FILENAME,
        max_seq_length=max_seq_length,
        opset=opset,
        profile=profile,
        cache_dir=cache_dir,
    )

    int8_path: Path | None = None
    int8_validation_metadata: Mapping[str, Any] | None = None
    if profile == ANDROID_PROFILE_NAME:
        android_validation = validate_android_profile(onnx_path)
        android_fp16_path = export_android_fp16(
            onnx_path,
            output_dir / ANDROID_FP16_FILENAME,
            validate=False,
        )
        android_fp16_validation = validate_android_profile(android_fp16_path)
        if include_int8:
            int8_path = quantize_dynamic_int8(
                onnx_path,
                output_dir / INT8_ONNX_FILENAME,
            )
            int8_validation = validate_android_profile(int8_path)
            int8_validation_metadata = int8_validation.to_metadata()
        artifacts = [
            ExportArtifact(
                format=ANDROID_ONNX_FORMAT,
                path=onnx_path,
                precision="float32",
                metadata=android_validation.to_metadata(),
            ),
            ExportArtifact(
                format=ANDROID_ONNX_FORMAT,
                path=android_fp16_path,
                precision="float16",
                metadata=android_fp16_validation.to_metadata(),
            ),
        ]
        if int8_path is not None:
            artifacts.append(
                ExportArtifact(
                    format=ONNX_INT8_FORMAT,
                    path=int8_path,
                    precision="int8",
                    metadata=int8_artifact_metadata(
                        validation_metadata=int8_validation_metadata,
                    ),
                )
            )
    else:
        if eval_suite_path is not None:
            raise ValueError("eval_suite_path requires profile='android'.")
        artifacts = [
            ExportArtifact(format="onnx", path=onnx_path, precision="float32"),
        ]

    if profile != ANDROID_PROFILE_NAME and include_webgpu:
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
        require_id2label=profile == ANDROID_PROFILE_NAME,
    )
    if include_transformersjs:
        transformersjs_result = export_transformersjs_bundle(
            output_dir,
            output_dir / DEFAULT_BUNDLE_DIRNAME,
            tokenizer_source=output_dir,
            config_source=output_dir / "config.json",
            update_manifest=False,
        )
        artifacts.append(
            ExportArtifact(
                format=TRANSFORMERSJS_FORMAT,
                path=transformersjs_result.output_dir,
                precision="int8",
            )
        )

    recall_report: dict[str, Any] | None = None
    if eval_suite_path is not None:
        if int8_path is None:
            raise ValueError("eval_suite_path requires Android INT8 export.")
        recall_report = write_int8_recall_delta_report(
            source_model_id=model_id,
            artifact_dir=output_dir,
            eval_suite_path=eval_suite_path,
            fp_model_path=onnx_path,
            int8_model_path=int8_path,
            output_path=recall_delta_report_path,
            cache_dir=cache_dir,
        )
        artifacts = [
            (
                ExportArtifact(
                    format=artifact.format,
                    path=artifact.path,
                    precision=artifact.precision,
                    metadata=int8_artifact_metadata(
                        validation_metadata=int8_validation_metadata,
                        recall_report=recall_report,
                    ),
                )
                if artifact.path == int8_path
                else artifact
            )
            for artifact in artifacts
        ]

    manifest_path = write_export_manifest(
        output_dir,
        source_model_id=model_id,
        config=config,
        artifacts=artifacts,
        tokenizer_files=tokenizer_files,
    )
    if recall_report is not None:
        apply_int8_recall_certification(
            output_dir,
            recall_report,
            report_relpath=str(recall_report["report_path"]),
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
    require_id2label: bool = False,
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
    if require_id2label:
        config["id2label"] = _ensure_id2label(config)

    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    id2label = config.get("id2label")
    if isinstance(id2label, Mapping):
        with (output_dir / "id2label.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {str(key): value for key, value in id2label.items()}, handle, indent=2
            )

    tokenizer_files: list[str] = []
    try:
        tokenizer = get_tokenizer_with_loader(
            model_id,
            AutoTokenizer.from_pretrained,
            cache_dir=cache_dir,
        )
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
        "label_map_path": "id2label.json"
        if (output_dir / "id2label.json").exists()
        else None,
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


def _normalize_profile(profile: str) -> str:
    normalized = profile.lower().replace("_", "-")
    if normalized in {DEFAULT_PROFILE_NAME, "onnx"}:
        return DEFAULT_PROFILE_NAME
    if normalized == ANDROID_PROFILE_NAME:
        return ANDROID_PROFILE_NAME
    raise ValueError(
        f"unsupported ONNX export profile {profile!r}; expected default or android"
    )


def _resolve_opset(*, profile: str, opset: int | None) -> int:
    if profile == ANDROID_PROFILE_NAME:
        if opset is not None and opset != ANDROID_ONNX_OPSET:
            raise ValueError(
                "Android ONNX profile requires fixed opset "
                f"{ANDROID_ONNX_OPSET}; got {opset}"
            )
        return ANDROID_ONNX_OPSET
    return DEFAULT_ONNX_OPSET if opset is None else opset


def _ensure_id2label(config: Mapping[str, Any]) -> dict[str, str]:
    id2label = config.get("id2label")
    if isinstance(id2label, Mapping) and id2label:
        return {str(key): str(value) for key, value in id2label.items()}

    label2id = config.get("label2id")
    if isinstance(label2id, Mapping) and label2id:
        labels = sorted(
            ((int(index), str(label)) for label, index in label2id.items()),
            key=lambda item: item[0],
        )
        return {str(index): label for index, label in labels}

    num_labels = config.get("num_labels")
    if isinstance(num_labels, int) and num_labels > 0:
        return {str(index): f"LABEL_{index}" for index in range(num_labels)}

    raise ValueError("Android ONNX profile requires config.json id2label metadata")


def _dynamic_export_kwargs(
    torch_module: Any,
    *,
    export_fn: Any,
    input_names: Sequence[str],
    max_seq_length: int,
    profile: str = DEFAULT_PROFILE_NAME,
) -> dict[str, Any]:
    """Return Torch ONNX dynamic-shape kwargs for the installed exporter."""

    parameters = inspect.signature(export_fn).parameters
    if profile == ANDROID_PROFILE_NAME:
        kwargs: dict[str, Any] = {
            "dynamic_axes": {
                name: {0: "batch", 1: "sequence"} for name in [*input_names, "logits"]
            }
        }
        if "dynamo" in parameters:
            kwargs["dynamo"] = False
        return kwargs

    if "dynamo" in parameters and "dynamic_shapes" in parameters:
        batch_dim = torch_module.export.Dim("batch", min=1)
        sequence_dim = torch_module.export.Dim(
            "sequence",
            min=1,
            max=max_seq_length,
        )
        return {
            "dynamo": True,
            "dynamic_shapes": {
                name: {0: batch_dim, 1: sequence_dim} for name in input_names
            },
        }

    dynamic_axes = {
        name: {0: "batch", 1: "sequence"} for name in [*input_names, "logits"]
    }
    return {"dynamic_axes": dynamic_axes}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a Hugging Face token-classification model to ONNX artifacts",
    )
    parser.add_argument(
        "--model", required=True, help="Source model ID or local directory"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for artifacts"
    )
    parser.add_argument(
        "--no-webgpu",
        action="store_true",
        help="Only emit the fp32 ONNX artifact",
    )
    parser.add_argument(
        "--include-transformersjs",
        action="store_true",
        help="Also emit a Transformers.js bundle with model_quantized.onnx",
    )
    parser.add_argument(
        "--no-int8",
        action="store_true",
        help="Do not emit model_int8.onnx for the android profile",
    )
    parser.add_argument(
        "--profile",
        choices=[DEFAULT_PROFILE_NAME, ANDROID_PROFILE_NAME],
        default=DEFAULT_PROFILE_NAME,
        help="Export profile. Use android for ONNX Runtime Mobile artifacts.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum input sequence length",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=None,
        help="ONNX opset version; android always uses its fixed mobile opset",
    )
    parser.add_argument(
        "--cache-dir", default=None, help="Hugging Face cache directory"
    )
    parser.add_argument(
        "--eval-suite",
        default=None,
        help="Benchmark fixture JSON/JSONL used to certify android INT8 recall",
    )
    parser.add_argument(
        "--recall-delta-report",
        default=None,
        help="Output path for the INT8 recall_delta.json report",
    )
    parser.add_argument(
        "--publish-to-hub",
        action="store_true",
        help="Publish the converted artifact after a successful conversion",
    )
    parser.add_argument(
        "--publish-repo-id", default=None, help="Explicit target repo id"
    )
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
        include_transformersjs=args.include_transformersjs,
        include_int8=not args.no_int8,
        max_seq_length=args.max_seq_length,
        opset=args.opset,
        profile=args.profile,
        cache_dir=args.cache_dir,
        eval_suite_path=args.eval_suite,
        recall_delta_report_path=args.recall_delta_report,
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
