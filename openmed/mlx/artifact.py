"""Shared MLX artifact helpers for Python and Swift runtimes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = "openmed-mlx.json"
MANIFEST_FORMAT = "openmed-mlx"
MANIFEST_VERSION = 1

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

_NON_TOKENIZER_FILENAMES = {
    MANIFEST_FILENAME,
    "config.json",
    "id2label.json",
    "weights.safetensors",
    "weights.npz",
}


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

    for name in _KNOWN_TOKENIZER_FILES:
        if (model_dir / name).exists():
            tokenizer_files.append(name)

    for path in sorted(model_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name in _NON_TOKENIZER_FILENAMES:
            continue
        if path.suffix.lower() not in {".json", ".txt", ".model", ".vocab", ".bpe"}:
            continue
        tokenizer_files.append(path.name)

    return _dedupe_keep_order(tokenizer_files)


def has_local_tokenizer(model_dir: str | Path) -> bool:
    """Return True if *model_dir* contains local tokenizer assets."""
    return bool(find_tokenizer_files(model_dir))


def read_manifest(model_dir: str | Path) -> dict[str, Any] | None:
    """Load ``openmed-mlx.json`` if present."""
    manifest_path = Path(model_dir) / MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    with open(manifest_path) as f:
        return json.load(f)


def load_artifact_config(model_dir: str | Path) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Return ``(manifest, config)`` for a converted MLX artifact."""
    model_dir = Path(model_dir)
    manifest = read_manifest(model_dir)
    config_name = "config.json"
    if manifest is not None:
        config_name = manifest.get("config_path", config_name)
    with open(model_dir / config_name) as f:
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
) -> Path:
    """Write ``openmed-mlx.json`` for a converted artifact."""
    model_dir = Path(model_dir)
    preferred_format = config.get("_mlx_weights_format", "safetensors")
    preferred_weights = "weights.safetensors" if preferred_format == "safetensors" else "weights.npz"
    fallback_weights = ["weights.npz"] if preferred_weights == "weights.safetensors" else ["weights.safetensors"]
    available_weights = [name for name in (preferred_weights, *fallback_weights) if (model_dir / name).exists()]

    tokenizer_files = tokenizer_files or find_tokenizer_files(model_dir)
    family = config.get("_mlx_model_type") or config.get("model_type") or "unknown"

    manifest = {
        "format": MANIFEST_FORMAT,
        "format_version": MANIFEST_VERSION,
        "task": "token-classification",
        "family": family,
        "source_model_id": source_model_id,
        "config_path": "config.json",
        "label_map_path": "id2label.json" if (model_dir / "id2label.json").exists() else None,
        "preferred_weights": preferred_weights,
        "fallback_weights": fallback_weights,
        "available_weights": available_weights,
        "weights_format": preferred_format,
        "quantization": config.get("_mlx_quantization"),
        "max_sequence_length": config.get("max_position_embeddings", 512),
        "tokenizer": {
            "path": ".",
            "files": tokenizer_files,
        },
    }

    manifest_path = model_dir / MANIFEST_FILENAME
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path
