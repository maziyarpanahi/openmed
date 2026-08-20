"""Reproducible mixed-bit MLX export planning for Maple.

The official 2-bit artifact is already the preferred Maple checkpoint. This
module prepares pinned 4-bit and 8-bit variants from the original BF16 source
without copying any model data into the OpenMed repository.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from openmed.mlx.lm import (
    MAPLE_MLX_MODEL,
    MAPLE_MLX_REVISION,
    MAPLE_SOURCE_MODEL,
)

MAPLE_SOURCE_REVISION = "ac1ddd79d2b5cb4406f5d2bebdf95406ce505a07"
MAPLE_EXPORT_BITS = (4, 8)
MAPLE_EXPORT_GROUP_SIZE = 128
MAPLE_EXPORT_MANIFEST = "openmed-maple-export.json"

_SOURCE_ALLOW_PATTERNS = (
    "LICENSE",
    "README.md",
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "model*.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
)


@dataclass(frozen=True)
class MapleMLXVariant:
    """One planned Maple MLX quantization target."""

    bits: int
    group_size: int
    output_directory: Path

    @property
    def format(self) -> str:
        """Return the OpenMed format name."""

        return f"mlx-{self.bits}bit"


@dataclass(frozen=True)
class MapleMLXExportPlan:
    """Pinned, inspectable plan for one or more Maple MLX variants."""

    output_root: Path
    variants: tuple[MapleMLXVariant, ...]
    source_model: str = MAPLE_SOURCE_MODEL
    source_revision: str = MAPLE_SOURCE_REVISION
    runtime_model: str = MAPLE_MLX_MODEL
    runtime_revision: str = MAPLE_MLX_REVISION

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable plan."""

        return {
            "source_model": self.source_model,
            "source_revision": self.source_revision,
            "runtime_model": self.runtime_model,
            "runtime_revision": self.runtime_revision,
            "output_root": str(self.output_root),
            "variants": [
                {
                    "bits": variant.bits,
                    "group_size": variant.group_size,
                    "format": variant.format,
                    "output_directory": str(variant.output_directory),
                }
                for variant in self.variants
            ],
        }


def plan_maple_mlx_variants(
    output_root: str | Path,
    *,
    bits: Iterable[int] = MAPLE_EXPORT_BITS,
    group_size: int = MAPLE_EXPORT_GROUP_SIZE,
) -> MapleMLXExportPlan:
    """Create a validated export plan without downloading or loading weights."""

    root = Path(output_root).expanduser().resolve()
    normalized_bits = tuple(dict.fromkeys(int(bit) for bit in bits))
    if not normalized_bits:
        raise ValueError("at least one Maple MLX bit width is required")
    unsupported = sorted(set(normalized_bits) - set(MAPLE_EXPORT_BITS))
    if unsupported:
        raise ValueError(
            "Maple variant export supports 4-bit and 8-bit targets; use the "
            f"published 2-bit artifact for 2-bit inference (got {unsupported})"
        )
    if group_size not in {32, 64, 128}:
        raise ValueError("group_size must be 32, 64, or 128")

    variants = tuple(
        MapleMLXVariant(
            bits=bit,
            group_size=group_size,
            output_directory=root / f"maple-preview-{bit}bit-mlx",
        )
        for bit in normalized_bits
    )
    return MapleMLXExportPlan(output_root=root, variants=variants)


def export_maple_mlx_variants(
    output_root: str | Path,
    *,
    bits: Iterable[int] = MAPLE_EXPORT_BITS,
    group_size: int = MAPLE_EXPORT_GROUP_SIZE,
    cache_dir: str | Path | None = None,
) -> MapleMLXExportPlan:
    """Download pinned inputs and create Maple 4-bit/8-bit MLX artifacts.

    Existing output directories are rejected rather than overwritten. The
    conversion requires Apple Silicon and enough free storage for the original
    roughly 40 GB checkpoint plus each requested variant.
    """

    plan = plan_maple_mlx_variants(
        output_root,
        bits=bits,
        group_size=group_size,
    )
    for variant in plan.variants:
        if variant.output_directory.exists():
            raise FileExistsError(
                f"refusing to overwrite existing export: {variant.output_directory}"
            )

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface-hub is required for Maple export; install openmed[mlx]"
        ) from exc
    try:
        from mlx_lm import convert as mlx_lm_convert
    except ImportError as exc:
        raise ImportError(
            "mlx-lm is required for Maple export; install openmed[mlx]"
        ) from exc

    plan.output_root.mkdir(parents=True, exist_ok=True)
    resolved_cache = (
        str(Path(cache_dir).expanduser()) if cache_dir is not None else None
    )
    source_snapshot = Path(
        snapshot_download(
            repo_id=plan.source_model,
            revision=plan.source_revision,
            repo_type="model",
            cache_dir=resolved_cache,
            allow_patterns=list(_SOURCE_ALLOW_PATTERNS),
        )
    )
    maple_runtime = Path(
        hf_hub_download(
            repo_id=plan.runtime_model,
            revision=plan.runtime_revision,
            repo_type="model",
            filename="maple.py",
            cache_dir=resolved_cache,
        )
    )

    with tempfile.TemporaryDirectory(
        prefix="openmed-maple-source-",
        dir=plan.output_root,
    ) as temporary_directory:
        staged_source = Path(temporary_directory)
        _stage_source_snapshot(source_snapshot, maple_runtime, staged_source)
        for variant in plan.variants:
            mlx_lm_convert(
                hf_path=str(staged_source),
                mlx_path=str(variant.output_directory),
                quantize=True,
                q_group_size=variant.group_size,
                q_bits=variant.bits,
                q_mode="affine",
                trust_remote_code=True,
            )
            _write_variant_manifest(variant, plan)
    return plan


def _stage_source_snapshot(
    source_snapshot: Path,
    maple_runtime: Path,
    staged_source: Path,
) -> None:
    for source in source_snapshot.iterdir():
        if source.name == "config.json":
            continue
        destination = staged_source / source.name
        if source.is_dir():
            os.symlink(source, destination, target_is_directory=True)
        else:
            os.symlink(source, destination)

    with (source_snapshot / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    config["model_file"] = "maple.py"
    config["model_type"] = "maple"
    config.pop("quantization", None)
    config.pop("quantization_config", None)
    with (staged_source / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    shutil.copy2(maple_runtime, staged_source / "maple.py")


def _write_variant_manifest(
    variant: MapleMLXVariant,
    plan: MapleMLXExportPlan,
) -> None:
    payload = {
        "format_version": 1,
        "format": variant.format,
        "architecture": "MapleForCausalLM",
        "license": "MIT",
        "source_model": plan.source_model,
        "source_revision": plan.source_revision,
        "runtime_code_model": plan.runtime_model,
        "runtime_code_revision": plan.runtime_revision,
        "quantization": {
            "bits": variant.bits,
            "group_size": variant.group_size,
            "mode": "affine",
        },
        "validation": {
            "status": "unvalidated",
            "required": [
                "synthetic prompt smoke test",
                "task JSON contract tests",
                "direct-identifier recall delta",
            ],
        },
    }
    path = variant.output_directory / MAPLE_EXPORT_MANIFEST
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export pinned Maple 4-bit and 8-bit MLX variants",
    )
    parser.add_argument("--output", required=True, help="Destination root directory")
    parser.add_argument(
        "--bits",
        type=int,
        nargs="+",
        default=list(MAPLE_EXPORT_BITS),
        help="Variant widths (supported: 4 8)",
    )
    parser.add_argument("--group-size", type=int, default=MAPLE_EXPORT_GROUP_SIZE)
    parser.add_argument("--cache-dir")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the pinned plan without downloading model data",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Maple MLX variant exporter."""

    args = _build_parser().parse_args(argv)
    plan = plan_maple_mlx_variants(
        args.output,
        bits=args.bits,
        group_size=args.group_size,
    )
    if not args.dry_run:
        plan = export_maple_mlx_variants(
            args.output,
            bits=args.bits,
            group_size=args.group_size,
            cache_dir=args.cache_dir,
        )
    print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())


__all__ = [
    "MAPLE_EXPORT_BITS",
    "MAPLE_EXPORT_GROUP_SIZE",
    "MAPLE_EXPORT_MANIFEST",
    "MAPLE_SOURCE_REVISION",
    "MapleMLXExportPlan",
    "MapleMLXVariant",
    "export_maple_mlx_variants",
    "main",
    "plan_maple_mlx_variants",
]
