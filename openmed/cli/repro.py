"""Reproducibility verification CLI commands for OpenMed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

from openmed.core.model_registry import MANIFEST_PATH
from openmed.training.repro_verify import (
    ReproVerificationResult,
    verify_reproducibility_inputs,
)

from ._output import EXIT_ERROR, EXIT_OK, EXIT_USAGE, CliError, wants_json


def add_repro_command(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Register the ``openmed repro`` command group."""
    repro_parser = subparsers.add_parser(
        "repro",
        help="Manage artifact reproducibility and verification.",
    )
    repro_subparsers = repro_parser.add_subparsers(dest="repro_command")
    _add_verify_parser(repro_subparsers)
    repro_parser.set_defaults(handler=_help_handler(repro_parser))
    return repro_parser


def _add_verify_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Register the ``verify`` subcommand under ``openmed repro``."""
    verify_parser = subparsers.add_parser(
        "verify",
        help="Recompute and verify reproducibility hash against claimed manifests.",
    )
    verify_parser.add_argument(
        "--repo",
        dest="repo_id",
        required=True,
        help=(
            "Model repository ID (for example "
            "OpenMed/OpenMed-PII-Chinese-BigMed-Large-560M-v1)."
        ),
    )
    verify_parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_PATH,
        help="Path to models.jsonl.",
    )
    verify_parser.add_argument(
        "--card",
        "--model-card",
        dest="card_path",
        type=Path,
        default=None,
        help="Path to rendered model card (README.md).",
    )
    verify_parser.add_argument(
        "--recipe",
        default=None,
        help="Optional recipe configuration file, JSON string, or preset override.",
    )
    verify_parser.add_argument(
        "--data-manifest",
        default=None,
        help="Optional data manifest path, reference, or JSON string override.",
    )
    verify_parser.add_argument(
        "--base-model",
        default=None,
        help="Optional base model override.",
    )
    verify_parser.add_argument(
        "--git-sha",
        default=None,
        help="Optional git commit SHA override.",
    )
    verify_parser.set_defaults(handler=_verify_handler)
    return verify_parser


def _help_handler(
    parser: argparse.ArgumentParser,
) -> Callable[[argparse.Namespace], int]:
    def handler(args: argparse.Namespace) -> int:
        parser.print_help(sys.stderr)
        return EXIT_USAGE

    return handler


def _verify_handler(args: argparse.Namespace) -> int:
    """Handle ``openmed repro verify`` execution."""
    repo_id = args.repo_id

    manifest_row: dict[str, Any] | None = None
    if args.manifest and Path(args.manifest).is_file():
        try:
            manifest_row = _find_manifest_row(Path(args.manifest), repo_id)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            manifest_row = None

    model_card_text: str | None = None
    if args.card_path and Path(args.card_path).is_file():
        try:
            model_card_text = Path(args.card_path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            model_card_text = None

    # Parse recipe if JSON string or file path
    recipe_val = _parse_input_value(args.recipe)
    if recipe_val is None and manifest_row is not None:
        recipe_val = manifest_row.get("recipe")

    # Parse data_manifest if JSON string or file path
    data_manifest_val = _parse_input_value(args.data_manifest)
    if data_manifest_val is None and manifest_row is not None:
        data_manifest_val = manifest_row.get("data_manifest")

    # Parse base_model
    base_model_val = args.base_model
    if base_model_val is None and manifest_row is not None:
        base_model_val = manifest_row.get("base_model")

    # Parse git_sha
    git_sha_val = args.git_sha
    if git_sha_val is None and manifest_row is not None:
        git_sha_val = manifest_row.get("git_sha")
        if git_sha_val is None:
            training_prov = manifest_row.get("training_provenance")
            if isinstance(training_prov, Mapping):
                git_sha_val = training_prov.get("git_sha")

    result = verify_reproducibility_inputs(
        recipe=recipe_val,
        data_manifest=data_manifest_val,
        base_model=base_model_val,
        git_sha=git_sha_val,
        manifest_row=manifest_row,
        model_card_text=model_card_text,
    )

    if wants_json(args):
        payload = result.to_dict()
        sys.stdout.write(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        )
    else:
        if result.status == "MATCH":
            sys.stdout.write("MATCH\n")
        elif result.status == "MISMATCH":
            if result.diverging_inputs:
                diverged_str = ", ".join(result.diverging_inputs)
                sys.stdout.write(f"MISMATCH: {diverged_str} diverged\n")
            else:
                sys.stdout.write(
                    f"MISMATCH: recomputed {result.recomputed_hash} "
                    f"!= claimed {result.claimed_hash}\n"
                )
        else:
            reason = result.details.get("reason", "unknown error")
            sys.stdout.write(f"UNVERIFIABLE: {reason}\n")

    return EXIT_OK if result.status == "MATCH" else EXIT_ERROR


def _find_manifest_row(manifest_path: Path, repo_id: str) -> dict[str, Any] | None:
    """Find a row matching repo_id in models.jsonl."""
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and row.get("repo_id") == repo_id:
                    return row
    except (OSError, UnicodeDecodeError):
        return None
    return None


def _parse_input_value(value: str | None) -> Any:
    """Parse JSON string or load file if path exists, otherwise return raw string."""
    if value is None:
        return None
    val_str = str(value).strip()
    if not val_str:
        return None

    path_obj = Path(val_str)
    if path_obj.is_file():
        try:
            return json.loads(path_obj.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return path_obj

    if (val_str.startswith("{") and val_str.endswith("}")) or (
        val_str.startswith("[") and val_str.endswith("]")
    ):
        try:
            return json.loads(val_str)
        except json.JSONDecodeError:
            pass

    return val_str


__all__ = [
    "add_repro_command",
]
