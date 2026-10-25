"""Shared MLX artifact helpers for Python and Swift runtimes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from openmed.core.export_provenance import (
    SourceContext,
    context_from_metadata,
    write_canonical_manifest,
)
from openmed.processing.tokenization import validate_segmenter_resources

MANIFEST_FILENAME = "openmed-mlx.json"
MANIFEST_FORMAT = "openmed-mlx"
MANIFEST_VERSION = 2
_EXPORT_MANIFEST_FILENAMES = (
    MANIFEST_FILENAME,
    "openmed-onnx.json",
    "openmed-coreml.json",
)

_KNOWN_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "spm.model",
    "sentencepiece.bpe.model",
    "added_tokens.json",
)

_KNOWN_SEGMENTER_FILES = (
    "segmenter/han_words.txt",
    "segmenter/indic_rules.json",
    "segmenter/ICU.txt",
)

_NON_TOKENIZER_FILENAMES = {
    *_EXPORT_MANIFEST_FILENAMES,
    "recall_delta.json",
    "config.json",
    "id2label.json",
    "weights.safetensors",
    "weights.npz",
}


def _declared_report_files(model_dir: Path) -> set[str]:
    """Read report references only from known root export manifests."""
    report_files: set[str] = set()
    for name in _EXPORT_MANIFEST_FILENAMES:
        manifest_path = model_dir / name
        if manifest_path.is_symlink() or not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError):
            continue
        if not isinstance(manifest, Mapping):
            continue
        references = [manifest.get("recall_delta_path")]
        certification = manifest.get("certification")
        if isinstance(certification, Mapping):
            references.append(certification.get("report_path"))
        for reference in references:
            if not isinstance(reference, str) or not reference or "\\" in reference:
                continue
            relative = Path(reference)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not relative.parts
                or relative.as_posix() != reference
            ):
                continue
            report_files.add(reference)
    return report_files


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def find_tokenizer_files(model_dir: str | Path) -> list[str]:
    """Return tokenizer asset filenames present in *model_dir*."""
    model_dir = Path(model_dir)
    tokenizer_files: list[str] = []
    excluded = _NON_TOKENIZER_FILENAMES | _declared_report_files(model_dir)

    for name in _KNOWN_TOKENIZER_FILES:
        if (model_dir / name).exists():
            tokenizer_files.append(name)

    for path in sorted(model_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name in excluded:
            continue
        if path.suffix.lower() not in {".json", ".txt", ".model", ".vocab", ".bpe"}:
            continue
        tokenizer_files.append(path.name)

    return _dedupe_keep_order(tokenizer_files)


def has_local_tokenizer(model_dir: str | Path) -> bool:
    """Return True if *model_dir* contains local tokenizer assets."""
    return bool(find_tokenizer_files(model_dir))


def find_segmenter_files(model_dir: str | Path) -> list[str]:
    """Return known compact segmenter resources present in *model_dir*."""

    root = Path(model_dir)
    return [name for name in _KNOWN_SEGMENTER_FILES if (root / name).is_file()]


def read_manifest(model_dir: str | Path) -> dict[str, Any] | None:
    """Load ``openmed-mlx.json`` if present."""
    manifest_path = Path(model_dir) / MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def load_artifact_config(
    model_dir: str | Path,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Return ``(manifest, config)`` for a converted MLX artifact."""
    model_dir = Path(model_dir)
    manifest = read_manifest(model_dir)
    config_name = "config.json"
    if manifest is not None:
        config_name = manifest.get("config_path", config_name)
    with open(model_dir / config_name, encoding="utf-8") as f:
        config = json.load(f)
    return manifest, config


def resolve_weight_candidates(
    model_dir: str | Path,
    config: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
) -> list[Path]:
    """Return ordered candidate weight paths for a converted MLX artifact."""
    model_dir = Path(model_dir)
    candidates: list[str] = []

    if manifest is not None:
        candidates.extend(list(manifest.get("available_weights", []) or []))
        candidates.extend(
            [manifest.get("preferred_weights", "")]
            + list(manifest.get("fallback_weights", []) or [])
        )

    preferred_format = None
    if config is not None:
        preferred_format = config.get("_mlx_weights_format")
    if preferred_format == "safetensors":
        candidates.append("weights.safetensors")
    elif preferred_format == "npz":
        candidates.append("weights.npz")

    candidates.extend(["weights.safetensors", "weights.npz"])
    return [model_dir / name for name in _dedupe_keep_order(candidates)]


def resolve_tokenizer_reference(
    model_dir: str | Path,
    config: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
) -> str:
    """Return a tokenizer directory path or Hugging Face tokenizer name."""
    model_dir = Path(model_dir)
    tokenizer_subdir = "."
    tokenizer_files: list[str] = []

    if manifest is not None:
        tokenizer = manifest.get("tokenizer", {}) or {}
        tokenizer_subdir = tokenizer.get("path", ".")
        tokenizer_files = list(tokenizer.get("files", []) or [])

    tokenizer_dir = (model_dir / tokenizer_subdir).resolve()
    if tokenizer_files or has_local_tokenizer(tokenizer_dir):
        return str(tokenizer_dir)

    if config is not None:
        tokenizer_name = config.get("_name_or_path")
        if tokenizer_name:
            return tokenizer_name

    return str(model_dir)


def write_manifest(
    model_dir: str | Path,
    *,
    source_model_id: str,
    config: dict[str, Any],
    tokenizer_files: list[str] | None = None,
    segmenter: Mapping[str, Any] | None = None,
    source: SourceContext | None = None,
) -> Path:
    """Write ``openmed-mlx.json`` with provenance for a converted artifact.

    Converter entry points supply their resolved source context. Direct writer
    callers retain offline behavior and record only locally available metadata.
    """
    model_dir = Path(model_dir)
    source = source or context_from_metadata(source_model_id, config)
    preferred_format = config.get("_mlx_weights_format", "safetensors")
    preferred_weights = (
        "weights.safetensors" if preferred_format == "safetensors" else "weights.npz"
    )
    fallback_weights = (
        ["weights.npz"]
        if preferred_weights == "weights.safetensors"
        else ["weights.safetensors"]
    )
    available_weights = [
        name
        for name in (preferred_weights, *fallback_weights)
        if (model_dir / name).exists()
    ]

    tokenizer_files = tokenizer_files or find_tokenizer_files(model_dir)
    family = (
        config.get("_mlx_family")
        or config.get("_mlx_model_type")
        or config.get("model_type")
        or "unknown"
    )
    task = config.get("_mlx_task", "token-classification")
    quantization = config.get("_mlx_quantization")
    quant_bits = quantization.get("bits") if isinstance(quantization, dict) else None
    format_name = f"mlx-{quant_bits}bit" if quant_bits in {2, 4, 8} else "mlx-fp"

    manifest = {
        "format": MANIFEST_FORMAT,
        "formats": [format_name],
        "format_version": MANIFEST_VERSION,
        "task": task,
        "family": family,
        "source_model_id": source.source_model_id,
        "config_path": "config.json",
        "label_map_path": "id2label.json"
        if (model_dir / "id2label.json").exists()
        else None,
        "preferred_weights": preferred_weights,
        "fallback_weights": fallback_weights,
        "available_weights": available_weights,
        "weights_format": preferred_format,
        "quantization": quantization,
        "max_sequence_length": config.get("max_position_embeddings", 512),
        "tokenizer": {
            "path": ".",
            "files": tokenizer_files,
        },
    }

    if isinstance(quantization, dict):
        if "quant_recall_delta" in quantization:
            manifest["quant_recall_delta"] = quantization["quant_recall_delta"]
        if "certified" in quantization:
            manifest["certified"] = quantization["certified"]
        if "recall_delta_path" in quantization:
            manifest["recall_delta_path"] = quantization["recall_delta_path"]
        if "certification_limit" in quantization:
            manifest["certification"] = {
                "gate": "G4",
                "limit": quantization["certification_limit"],
                "metric": "character_recall",
                "report_path": quantization.get("recall_delta_path"),
            }

    prompt_spec = config.get("_mlx_prompt_spec")
    if prompt_spec:
        manifest["prompt_spec"] = prompt_spec

    runtime = config.get("_mlx_runtime")
    if runtime:
        manifest["runtime"] = runtime

    if segmenter is not None:
        manifest["segmenter"] = dict(segmenter)

    manifest_path = write_canonical_manifest(
        model_dir,
        MANIFEST_FILENAME,
        manifest,
        source=source,
        backend="mlx",
    )
    validate_mlx_bundle(model_dir)
    return manifest_path


def validate_mlx_bundle(model_dir: str | Path) -> dict[str, Any]:
    """Validate manifest-declared optional resources in an MLX artifact bundle."""

    root = Path(model_dir)
    manifest = read_manifest(root)
    if manifest is None:
        raise ValueError(f"MLX artifact is missing {MANIFEST_FILENAME}: {root}")
    segmenter = manifest.get("segmenter")
    if segmenter is not None:
        if not isinstance(segmenter, Mapping):
            raise ValueError("MLX segmenter descriptor must be an object")
        validate_segmenter_resources(root, segmenter)
    return manifest
