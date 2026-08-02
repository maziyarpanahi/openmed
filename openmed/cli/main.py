"""Command-line interface for the OpenMed toolkit."""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import os
import sys
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from ..__about__ import __version__
from ..core.config import (
    PROFILE_PRESETS,
    OpenMedConfig,
    delete_profile,
    get_config,
    get_profile,
    list_profiles,
    load_config_from_file,
    resolve_config_path,
    save_config_to_file,
    save_profile,
    set_config,
)
from ..core.hf_hub import get_remote_model_size_mb, list_cached_models
from ..core.manifest_diff import ManifestDiff, diff_manifests
from ..core.model_card import render_model_card
from ..core.model_integrity import ModelIntegrityError, verify_cached_models
from ..core.model_registry import (
    MANIFEST_PATH,
    ModelSizeEstimate,
    estimate_model_sizes,
    get_model_info,
    load_manifest_rows,
)
from ..core.model_search import ModelSearchResult, recommend_models, search_models
from ..core.offline import OfflineModeError
from ..core.policy import CANONICAL_POLICY_NAMES, canonical_policy_name
from ._output import (
    EXIT_ERROR,
    EXIT_USAGE,
    CliError,
    add_json_flag,
    emit,
    emit_error,
    wants_json,
)
from .active_learning import add_active_learning_command
from .airgap import add_airgap_command
from .calibrate import add_calibrate_command
from .gates import add_gates_command
from .verify_pdf import add_verify_pdf_command

_ANALYZE_TEXT = None
_GET_MODEL_MAX_LENGTH = None
_LIST_MODELS = None
_BATCH_PROCESSOR = None

_AUDIT_KEY_ENV = "OPENMED_AUDIT_KEY"


class _ReleasePublicationCleanupError(OSError):
    """Raised after publication when stale backup cleanup did not finish."""


# Exposed for unit tests to patch without importing heavy modules eagerly.
analyze_text = None
get_model_max_length = None
list_models = None
BatchProcessor = None


def _lazy_api():
    global _ANALYZE_TEXT, _GET_MODEL_MAX_LENGTH, _LIST_MODELS, _BATCH_PROCESSOR

    global analyze_text, get_model_max_length, list_models, BatchProcessor

    if analyze_text is not None and analyze_text is not _ANALYZE_TEXT:
        _ANALYZE_TEXT = analyze_text

    if _ANALYZE_TEXT is None:
        if analyze_text is not None:
            _ANALYZE_TEXT = analyze_text
        else:
            from .. import analyze_text as _analyze

            _ANALYZE_TEXT = analyze_text = _analyze

    if (
        get_model_max_length is not None
        and get_model_max_length is not _GET_MODEL_MAX_LENGTH
    ):
        _GET_MODEL_MAX_LENGTH = get_model_max_length

    if _GET_MODEL_MAX_LENGTH is None:
        if get_model_max_length is not None:
            _GET_MODEL_MAX_LENGTH = get_model_max_length
        else:
            from .. import get_model_max_length as _get_max_len

            _GET_MODEL_MAX_LENGTH = get_model_max_length = _get_max_len

    if list_models is not None and list_models is not _LIST_MODELS:
        _LIST_MODELS = list_models

    if _LIST_MODELS is None:
        if list_models is not None:
            _LIST_MODELS = list_models
        else:
            from .. import list_models as _list

            _LIST_MODELS = list_models = _list

    if BatchProcessor is not None and BatchProcessor is not _BATCH_PROCESSOR:
        _BATCH_PROCESSOR = BatchProcessor

    if _BATCH_PROCESSOR is None:
        if BatchProcessor is not None:
            _BATCH_PROCESSOR = BatchProcessor
        else:
            from .. import BatchProcessor as _batch

            _BATCH_PROCESSOR = BatchProcessor = _batch

    return _ANALYZE_TEXT, _GET_MODEL_MAX_LENGTH, _LIST_MODELS, _BATCH_PROCESSOR


Handler = Callable[[argparse.Namespace], int]


class _UnavailableCommandError(NotImplementedError):
    """Signal that a recovered CLI command has no current implementation."""


COMPLIANCE_CAVEAT = (
    "No de-identification tool can guarantee compliance or zero residual risk. "
    "Validate locally before any production or clinical use."
)
_FHIR_BUNDLE_TYPES = frozenset({"transaction", "batch"})
_OMOP_WRITERS = ("duckdb", "sqlite", "parquet")

_DEFAULT_PII_MODEL = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"
_DEID_METHODS = ("mask", "remove", "replace", "hash", "shift_dates")
_MOBILE_BENCHMARK_DEVICES = ("cpu", "mlx", "coreml")
_MOBILE_BENCHMARK_TIERS = (
    "nano",
    "tiny",
    "phone",
    "mobile",
    "base",
    "laptop",
    "large",
    "workstation",
    "accurate",
    "accurate-xlarge",
    "xlarge",
    "server",
)


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be greater than or equal to 0")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError(
            "value must be a finite number greater than or equal to 0"
        )
    return parsed


def _unit_interval_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError(
            "value must be a finite number between 0 and 1"
        )
    return parsed


def _column_list_arg(value: str) -> tuple[str, ...]:
    columns = tuple(
        dict.fromkeys(item.strip() for item in value.split(",") if item.strip())
    )
    if not columns:
        raise argparse.ArgumentTypeError(
            "expected one or more comma-separated column names"
        )
    return columns


def _literal_column_arg(value: str) -> str:
    if (
        not value
        or value != value.strip()
        or any(
            unicodedata.category(character).startswith("C")
            or unicodedata.category(character) in {"Zl", "Zp"}
            for character in value
        )
    ):
        raise argparse.ArgumentTypeError(
            "literal column names must be non-empty and cannot have surrounding "
            "whitespace or control characters"
        )
    return value


def _merged_column_args(
    comma_separated: Sequence[str],
    literal_columns: Sequence[str],
) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*comma_separated, *literal_columns)))


def _release_error_message(summary: str, exc: TypeError | ValueError) -> str:
    """Attach a bounded, single-line release validation cause."""

    rendered = []
    for character in str(exc):
        category = unicodedata.category(character)
        if character in "\r\n\t":
            rendered.append(" ")
        elif category.startswith("C") or category in {"Zl", "Zp"}:
            rendered.append(f"\\u{ord(character):04x}")
        else:
            rendered.append(character)
    detail = " ".join("".join(rendered).split())
    if not detail:
        detail = "No additional validation detail was provided."
    if len(detail) > 1_000:
        detail = detail[:997] + "..."
    return f"{summary} Cause ({type(exc).__name__}): {detail}"


def _role_override_arg(value: str) -> tuple[str, tuple[str, ...]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "role overrides must use COLUMN=ROLE[,ROLE] syntax"
        )
    column, raw_roles = value.split("=", 1)
    column = column.strip()
    roles = tuple(
        dict.fromkeys(item.strip() for item in raw_roles.split(",") if item.strip())
    )
    valid = {
        "direct-id",
        "quasi-id",
        "sensitive",
        "safe",
        "internal-linkage",
        "free-text",
    }
    if not column or not roles or any(role not in valid for role in roles):
        raise argparse.ArgumentTypeError(
            "role overrides require a column and roles from: "
            + ", ".join(sorted(valid))
        )
    _literal_column_arg(column)
    return column, roles


def _named_digest_arg(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "named digests must use NAME=sha256:<64 lowercase hex digits>"
        )
    name, digest = value.split("=", 1)
    name_characters = frozenset(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.:-"
    )
    if (
        not name
        or len(name) > 128
        or not name[0].isalpha()
        or any(character not in name_characters for character in name)
        or len(digest) != 71
        or not digest.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in digest[7:])
    ):
        raise argparse.ArgumentTypeError(
            "named digests must use NAME=sha256:<64 lowercase hex digits>"
        )
    return name, digest


def _policy_name_arg(value: str) -> str:
    try:
        return canonical_policy_name(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="openmed",
        description="Command-line utilities for OpenMed medical NLP models.",
        epilog=COMPLIANCE_CAVEAT,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config-path",
        help="Override the configuration file path.",
        default=None,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"openmed {__version__}",
        help="Print the OpenMed package version and exit.",
    )

    subparsers = parser.add_subparsers(dest="command")

    _add_analyze_command(subparsers)
    _add_batch_command(subparsers)
    _add_deid_command(subparsers)
    _add_redact_dataset_command(subparsers)
    _add_pii_command(subparsers)
    _add_tui_command(subparsers)
    _add_audit_command(subparsers)
    _add_compliance_command(subparsers)
    _add_risk_command(subparsers)
    _add_policy_command(subparsers)
    _add_export_command(subparsers)
    _add_fhir_command(subparsers)
    _add_icd11_command(subparsers)
    _add_omop_command(subparsers)
    _add_ground_command(subparsers)
    _add_grounding_snapshot_command(subparsers)
    _add_benchmark_command(subparsers)
    _add_profile_command(subparsers)
    _add_eval_command(subparsers)
    _add_models_command(subparsers)
    _add_release_command(subparsers)
    _add_config_command(subparsers)
    add_airgap_command(subparsers)
    add_active_learning_command(subparsers)
    _add_doctor_command(subparsers)
    add_calibrate_command(subparsers)
    add_gates_command(subparsers)
    add_verify_pdf_command(subparsers)
    _finalize_parser(parser)
    return parser


def _find_subparsers(
    parser: argparse.ArgumentParser,
) -> Optional[argparse._SubParsersAction]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    return None


def _finalize_parser(parser: argparse.ArgumentParser) -> None:
    """Attach a uniform ``--json`` flag and a ``command_path`` to every leaf.

    Walking the built tree keeps output wiring in one place instead of scattered
    across ~40 registrars, and guarantees no scriptable subcommand is missed.
    """

    root = _find_subparsers(parser)
    if root is None:  # pragma: no cover - defensive
        return
    seen: set[int] = set()
    for name, subparser in root.choices.items():
        _finalize_subtree(subparser, name, seen)


def _finalize_subtree(
    parser: argparse.ArgumentParser,
    path: str,
    seen: set[int],
) -> None:
    if id(parser) in seen:  # guard against alias duplicates
        return
    seen.add(id(parser))

    child_action = _find_subparsers(parser)
    if child_action is not None:
        for name, child in child_action.choices.items():
            _finalize_subtree(child, f"{path} {name}", seen)
        if parser.get_default("handler") is None:
            return  # pure dispatch node, no handler of its own

    parser.set_defaults(command_path=path)
    if not any("--json" in action.option_strings for action in parser._actions):
        add_json_flag(parser)


def _add_analyze_command(subparsers: argparse._SubParsersAction) -> None:
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyse text with an OpenMed model."
    )
    analyze_parser.add_argument(
        "--model",
        default="disease_detection_superclinical",
        help="Model registry key or Hugging Face identifier.",
    )
    group = analyze_parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--text",
        help="Text to analyse.",
    )
    group.add_argument(
        "--input-file",
        type=Path,
        help="Path to a file containing text to analyse.",
    )
    analyze_parser.add_argument(
        "--output-format",
        choices=["dict", "json", "html", "csv"],
        default="dict",
        help="Desired output format.",
    )
    analyze_parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Minimum confidence score for predictions.",
    )
    analyze_parser.add_argument(
        "--group-entities",
        action="store_true",
        help="Group adjacent entities of the same label.",
    )
    analyze_parser.add_argument(
        "--no-confidence",
        action="store_true",
        help="Omit confidence scores from the output.",
    )
    analyze_parser.add_argument(
        "--use-medical-tokenizer",
        dest="use_medical_tokenizer",
        action="store_true",
        default=None,
        help="Force-enable medical token remapping in the output (default from config).",
    )
    analyze_parser.add_argument(
        "--no-medical-tokenizer",
        dest="use_medical_tokenizer",
        action="store_false",
        default=None,
        help="Disable medical token remapping in the output and fall back to raw model spans.",
    )
    analyze_parser.add_argument(
        "--medical-tokenizer-exceptions",
        default=None,
        help="Comma-separated extra terms to keep intact when remapping (e.g., MY-DRUG-123,ABC-001).",
    )
    analyze_parser.set_defaults(handler=_handle_analyze)


def _add_batch_command(subparsers: argparse._SubParsersAction) -> None:
    batch_parser = subparsers.add_parser(
        "batch", help="Process multiple texts or files in batch mode."
    )
    batch_parser.add_argument(
        "--model",
        default="disease_detection_superclinical",
        help="Model registry key or Hugging Face identifier.",
    )

    input_group = batch_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing text files to process.",
    )
    input_group.add_argument(
        "--input-files",
        nargs="+",
        type=Path,
        help="List of text files to process.",
    )
    input_group.add_argument(
        "--texts",
        nargs="+",
        help="List of text strings to process.",
    )

    batch_parser.add_argument(
        "--pattern",
        default="*.txt",
        help="Glob pattern for matching files in directory (default: *.txt).",
    )
    batch_parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively in directory.",
    )
    batch_parser.add_argument(
        "--output",
        type=Path,
        help="Output file for results (JSON format).",
    )
    batch_parser.add_argument(
        "--output-format",
        choices=["json", "summary"],
        default="summary",
        help="Output format: json (full results) or summary (default).",
    )
    batch_parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Minimum confidence score for predictions.",
    )
    batch_parser.add_argument(
        "--group-entities",
        action="store_true",
        help="Group adjacent entities of the same label.",
    )
    batch_parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Continue processing on individual item errors (default: true).",
    )
    batch_parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop processing on first error.",
    )
    batch_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )
    batch_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the checkpoint associated with --output.",
    )
    batch_parser.add_argument(
        "--checkpoint-path",
        type=Path,
        help="Checkpoint path (default: <output>.checkpoint.json).",
    )
    batch_parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Commit progress after this many items (default: 10).",
    )
    batch_parser.set_defaults(handler=_handle_batch)


def _add_tui_command(subparsers: argparse._SubParsersAction) -> None:
    """Restore the historical TUI command without reviving its removed backend."""

    tui_parser = subparsers.add_parser(
        "tui",
        help="Launch the historical interactive terminal UI.",
    )
    tui_parser.add_argument(
        "--model",
        default=None,
        help="Model registry key or Hugging Face identifier.",
    )
    tui_parser.add_argument(
        "--confidence-threshold",
        type=_unit_interval_float,
        default=0.5,
        help="Minimum confidence score for predictions (default: 0.5).",
    )
    tui_parser.set_defaults(handler=_handle_tui)


def _add_deid_command(subparsers: argparse._SubParsersAction) -> None:
    deid_parser = subparsers.add_parser(
        "deid",
        help="De-identify text with policy profiles.",
    )
    deid_parser.add_argument(
        "--policy",
        type=_policy_name_arg,
        choices=CANONICAL_POLICY_NAMES,
        default="hipaa_safe_harbor",
        help="Policy profile to apply.",
    )
    deid_parser.add_argument(
        "--method",
        choices=_DEID_METHODS,
        default="mask",
        help="De-identification method.",
    )
    deid_parser.add_argument(
        "--keep-mapping",
        action="store_true",
        help="Keep reversible mapping metadata in the de-identification result.",
    )
    deid_parser.add_argument(
        "--audit",
        action="store_true",
        help="Write an audit report and print its path instead of redacted text.",
    )
    deid_parser.add_argument(
        "--input",
        default="-",
        metavar="FILE",
        help="Input text file, or '-' for stdin (default).",
    )
    deid_parser.add_argument(
        "--output",
        default="-",
        metavar="FILE",
        help="Output file, or '-' for stdout (default).",
    )
    deid_parser.add_argument(
        "--model",
        default=_DEFAULT_PII_MODEL,
        help="PII detection model.",
    )
    deid_parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum confidence for redaction.",
    )
    deid_parser.add_argument(
        "--keep-year",
        action="store_true",
        help="Keep year in dates.",
    )
    deid_parser.set_defaults(handler=_handle_deid)


def _add_redact_dataset_command(subparsers: argparse._SubParsersAction) -> None:
    redact_parser = subparsers.add_parser(
        "redact-dataset",
        help="Redact selected free-text columns in a CSV, JSONL, or Parquet dataset.",
    )
    redact_parser.add_argument(
        "path",
        type=Path,
        help="Input .csv, .jsonl, .ndjson, or .parquet file.",
    )
    redact_parser.add_argument(
        "--text-column",
        dest="text_column",
        action="append",
        default=[],
        help="Free-text column to redact. Repeat for multiple columns.",
    )
    redact_parser.add_argument(
        "--text-columns",
        dest="text_columns",
        default=None,
        help="Comma-separated free-text columns to redact.",
    )
    redact_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output dataset path. Defaults to <stem>.redacted<suffix>.",
    )
    redact_parser.add_argument(
        "--policy",
        default=None,
        help="Policy profile name to pass to de-identification.",
    )
    redact_parser.add_argument(
        "--method",
        choices=["mask", "remove", "replace", "hash", "shift_dates"],
        default="mask",
        help="Fallback de-identification method.",
    )
    redact_parser.add_argument(
        "--model",
        default=_DEFAULT_PII_MODEL,
        help="PII detection model.",
    )
    redact_parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum confidence for redaction.",
    )
    redact_parser.add_argument(
        "--lang",
        default="en",
        help="Language hint for PII detection and redaction.",
    )
    redact_parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding for CSV and JSONL inputs.",
    )
    redact_parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Row batch size for Parquet processing.",
    )
    redact_parser.add_argument(
        "--keep-year",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep year in dates where applicable. Use --no-keep-year to disable.",
    )
    redact_parser.add_argument(
        "--no-safety-sweep",
        action="store_true",
        help="Disable deterministic structured-identifier sweep.",
    )
    redact_parser.set_defaults(handler=_handle_redact_dataset)


def _add_pii_command(subparsers: argparse._SubParsersAction) -> None:
    """Add PII extraction and de-identification commands."""
    pii_parser = subparsers.add_parser(
        "pii", help="PII extraction and de-identification."
    )
    pii_sub = pii_parser.add_subparsers(dest="pii_command")

    # PII Extract command
    extract_parser = pii_sub.add_parser(
        "extract", help="Extract PII entities from text."
    )
    extract_parser.add_argument(
        "--model",
        default="OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1",
        help="PII detection model.",
    )
    text_group = extract_parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text", help="Text to analyze.")
    text_group.add_argument("--input-file", type=Path, help="Input file.")
    extract_parser.add_argument(
        "--output",
        type=Path,
        help="Output file for results (JSON format).",
    )
    extract_parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum confidence score.",
    )
    extract_parser.set_defaults(handler=_handle_pii_extract)

    # PII De-identify command
    deid_parser = pii_sub.add_parser(
        "deidentify", help="De-identify text by redacting PII."
    )
    deid_parser.add_argument(
        "--model",
        default=_DEFAULT_PII_MODEL,
        help="PII detection model.",
    )
    deid_text_group = deid_parser.add_mutually_exclusive_group(required=True)
    deid_text_group.add_argument("--text", help="Text to de-identify.")
    deid_text_group.add_argument("--input-file", type=Path, help="Input file.")
    deid_parser.add_argument(
        "--output",
        type=Path,
        help="Output file for de-identified text.",
    )
    deid_parser.add_argument(
        "--method",
        choices=_DEID_METHODS,
        default="mask",
        help="De-identification method.",
    )
    deid_parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum confidence for redaction.",
    )
    deid_parser.add_argument(
        "--keep-year",
        action="store_true",
        help="Keep year in dates.",
    )
    deid_parser.add_argument(
        "--shift-dates",
        action="store_true",
        help="Shift dates by random offset.",
    )
    deid_parser.add_argument(
        "--keep-mapping",
        action="store_true",
        help="Keep mapping for re-identification.",
    )
    deid_parser.set_defaults(handler=_handle_pii_deidentify)

    # PII Batch command
    batch_parser = pii_sub.add_parser("batch", help="Batch de-identification of files.")
    batch_parser.add_argument(
        "--model",
        default=_DEFAULT_PII_MODEL,
        help="PII detection model.",
    )
    batch_parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory with files to process.",
    )
    batch_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for de-identified files.",
    )
    batch_parser.add_argument(
        "--pattern",
        default="*.txt",
        help="File pattern to match.",
    )
    batch_parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively.",
    )
    batch_parser.add_argument(
        "--method",
        choices=["mask", "remove", "replace", "hash"],
        default="mask",
        help="De-identification method.",
    )
    batch_parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum confidence for redaction.",
    )
    batch_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the output directory checkpoint.",
    )
    batch_parser.add_argument(
        "--checkpoint-path",
        type=Path,
        help=(
            "Checkpoint path (default: <output-dir>/.openmed-batch.checkpoint.json)."
        ),
    )
    batch_parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Commit progress after this many files (default: 10).",
    )
    batch_parser.set_defaults(handler=_handle_pii_batch)


def _add_audit_command(subparsers: argparse._SubParsersAction) -> None:
    audit_parser = subparsers.add_parser(
        "audit",
        help="Inspect and verify PHI-safe de-identification audit reports.",
    )
    audit_sub = audit_parser.add_subparsers(dest="audit_command")

    verify_parser = audit_sub.add_parser(
        "verify",
        help="Verify an audit report or tamper-evident audit chain.",
    )
    verify_parser.add_argument(
        "report",
        type=Path,
        help="Path to an audit report or audit-chain JSON file.",
    )
    verify_parser.add_argument(
        "--key",
        default=None,
        help=f"HMAC key for signed reports. Defaults to {_AUDIT_KEY_ENV}.",
    )
    verify_parser.set_defaults(handler=_handle_audit_verify)

    chain_parser = audit_sub.add_parser(
        "verify-chain",
        help="Verify an audit chain and optionally a report committed to it.",
    )
    chain_parser.add_argument(
        "chain",
        type=Path,
        help="Path to a tamper-evident audit-chain JSON file.",
    )
    chain_parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Also verify this audit report and confirm chain membership.",
    )
    chain_parser.add_argument(
        "--key",
        default=None,
        help=f"HMAC key for signed reports. Defaults to {_AUDIT_KEY_ENV}.",
    )
    chain_parser.set_defaults(handler=_handle_audit_chain_verify)

    show_parser = audit_sub.add_parser(
        "show",
        help="Print a PHI-safe summary of an audit report.",
    )
    show_parser.add_argument(
        "report",
        type=Path,
        help="Path to an audit report JSON file.",
    )
    show_parser.set_defaults(handler=_handle_audit_show)


def _add_compliance_command(subparsers: argparse._SubParsersAction) -> None:
    compliance_parser = subparsers.add_parser(
        "compliance",
        help="Generate local compliance evidence from OpenMed run artifacts.",
    )
    compliance_sub = compliance_parser.add_subparsers(dest="compliance_command")

    safe_harbor_parser = compliance_sub.add_parser(
        "safe-harbor",
        help="Generate a HIPAA Safe Harbor attestation from an audit report.",
    )
    safe_harbor_parser.add_argument(
        "report",
        type=Path,
        help="Path to a de-identification audit report JSON file.",
    )
    safe_harbor_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optional path for the attestation JSON. Defaults to stdout.",
    )
    safe_harbor_parser.set_defaults(handler=_handle_compliance_safe_harbor)

    expert_verify_parser = compliance_sub.add_parser(
        "expert-review-verify",
        help="Verify a PHI-safe de-identification expert-review evidence bundle.",
    )
    expert_verify_parser.add_argument(
        "report",
        type=Path,
        help="Path to an expert-review evidence JSON file.",
    )
    expert_verify_parser.set_defaults(handler=_handle_expert_review_verify)

    attestation_verify_parser = compliance_sub.add_parser(
        "expert-attestation-verify",
        help=(
            "Verify an expert-authored signature against its evidence and trusted key."
        ),
    )
    attestation_verify_parser.add_argument(
        "attestation",
        type=Path,
        help="Path to an expert-attestation JSON envelope.",
    )
    attestation_verify_parser.add_argument(
        "--evidence",
        type=Path,
        required=True,
        help="Expert-review evidence JSON bound by the attestation.",
    )
    attestation_verify_parser.add_argument(
        "--public-key",
        type=Path,
        required=True,
        help="Trusted Ed25519 public key in PEM or raw 32-byte form.",
    )
    attestation_verify_parser.add_argument(
        "--key-id",
        required=True,
        help="Expected trusted key identifier.",
    )
    attestation_verify_parser.add_argument(
        "--supporting-evidence",
        action="append",
        type=_named_digest_arg,
        default=[],
        metavar="NAME=SHA256_DIGEST",
        help=(
            "Expected named supporting-evidence digest; repeat for every digest "
            "bound by the attestation."
        ),
    )
    attestation_verify_parser.set_defaults(handler=_handle_expert_attestation_verify)


def _add_risk_command(subparsers: argparse._SubParsersAction) -> None:
    risk_parser = subparsers.add_parser(
        "risk",
        help="Score residual re-identification risk for text or tables.",
    )
    risk_sub = risk_parser.add_subparsers(dest="risk_command")

    text_parser = risk_sub.add_parser(
        "text",
        help="Score residual re-identification risk for text.",
    )
    text_parser.add_argument(
        "input",
        help="Text to score, or a path to a UTF-8 text file.",
    )
    text_parser.set_defaults(handler=_handle_risk_text)

    table_parser = risk_sub.add_parser(
        "table",
        help="Score residual re-identification risk for CSV records.",
    )
    table_parser.add_argument(
        "csv",
        type=Path,
        help="Path to a CSV file with a header row.",
    )
    table_parser.set_defaults(handler=_handle_risk_table)

    discover_parser = risk_sub.add_parser(
        "discover",
        help="Discover candidate quasi-identifiers in a structured table.",
    )
    discover_parser.add_argument("input", type=Path)
    discover_parser.add_argument("--output", "-o", type=Path, required=True)
    discover_parser.add_argument(
        "--sample-rows",
        type=_positive_int,
        default=10_000,
        help="Maximum rows for advisory discovery.",
    )
    discover_parser.add_argument(
        "--full-scan",
        action="store_true",
        help="Read the complete table and mark dataset coverage complete.",
    )
    discover_parser.add_argument("--privacy-unit", default=None)
    discover_parser.add_argument("--qi", type=_column_list_arg, default=())
    discover_parser.add_argument(
        "--qi-column",
        action="append",
        type=_literal_column_arg,
        default=[],
        help="Repeat for a literal QI column name, including names with commas.",
    )
    discover_parser.add_argument(
        "--sensitive",
        type=_column_list_arg,
        default=(),
    )
    discover_parser.add_argument(
        "--sensitive-column",
        action="append",
        type=_literal_column_arg,
        default=[],
        help=(
            "Repeat for a literal sensitive column name, including names with commas."
        ),
    )
    discover_parser.add_argument(
        "--role",
        action="append",
        type=_role_override_arg,
        default=[],
        metavar="COLUMN=ROLE[,ROLE]",
    )
    discover_parser.add_argument("--max-set-size", type=_positive_int, default=4)
    discover_parser.add_argument(
        "--max-candidate-columns",
        type=_positive_int,
        default=8,
    )
    discover_parser.add_argument(
        "--search-budget",
        type=_positive_int,
        default=1_000,
    )
    discover_parser.add_argument(
        "--include-safe-candidates",
        action="store_true",
        help=(
            "Include reviewed scalar columns currently classified as safe in "
            "the bounded combination search."
        ),
    )
    discover_parser.add_argument("--overwrite", action="store_true")
    discover_parser.set_defaults(handler=_handle_risk_discover)

    assess_parser = risk_sub.add_parser(
        "assess",
        help="Measure patient-level k/l/t release risk with safe output.",
    )
    assess_parser.add_argument("input", type=Path)
    assess_parser.add_argument("--output", "-o", type=Path, required=True)
    assess_parser.add_argument(
        "--dashboard",
        type=Path,
        default=None,
        help="Optional self-contained aggregate-only HTML dashboard.",
    )
    _add_release_policy_arguments(assess_parser, include_transformation=False)
    assess_parser.add_argument("--overwrite", action="store_true")
    assess_parser.set_defaults(handler=_handle_risk_assess)

    population_parser = risk_sub.add_parser(
        "population-assess",
        help="Measure exact risk against a supplied reference population.",
    )
    population_parser.add_argument("input", type=Path, help="Sample table.")
    population_parser.add_argument(
        "reference_population",
        type=Path,
        help="Caller-supplied reference-population table.",
    )
    population_parser.add_argument("--output", "-o", type=Path, required=True)
    population_parser.add_argument(
        "--qi",
        type=_column_list_arg,
        default=(),
        help="Comma-separated quasi-identifier columns.",
    )
    population_parser.add_argument(
        "--qi-column",
        action="append",
        type=_literal_column_arg,
        default=[],
        help="Repeat for a literal QI column name, including names with commas.",
    )
    population_parser.add_argument(
        "--sample-privacy-unit",
        default=None,
        help=(
            "Sample analysis-unit column; must be supplied together with "
            "--population-privacy-unit."
        ),
    )
    population_parser.add_argument(
        "--population-privacy-unit",
        default=None,
        help=(
            "Reference analysis-unit column; must be supplied together with "
            "--sample-privacy-unit."
        ),
    )
    population_parser.add_argument(
        "--k-map",
        type=_positive_int,
        required=True,
        help="Minimum reference frequency for every sample profile.",
    )
    population_parser.add_argument(
        "--max-delta-presence",
        type=_unit_interval_float,
        required=True,
        help="Maximum sample/reference profile-frequency ratio in [0, 1].",
    )
    population_parser.add_argument("--overwrite", action="store_true")
    population_parser.set_defaults(handler=_handle_risk_population_assess)

    gate_parser = risk_sub.add_parser(
        "gate",
        help="Verify and gate aggregate structured-release evidence for CI.",
    )
    gate_parser.add_argument(
        "evidence",
        type=Path,
        help="Expert-review evidence JSON produced by risk anonymize.",
    )
    gate_parser.set_defaults(handler=_handle_risk_gate)

    anonymize_parser = risk_sub.add_parser(
        "anonymize",
        help="Generalize and suppress a structured release, then revalidate it.",
    )
    anonymize_parser.add_argument("input", type=Path)
    anonymize_parser.add_argument("--output", "-o", type=Path, required=True)
    anonymize_parser.add_argument(
        "--evidence",
        type=Path,
        required=True,
        help="Path for PHI-safe expert-review evidence JSON.",
    )
    anonymize_parser.add_argument(
        "--evidence-markdown",
        type=Path,
        default=None,
        help="Optional Markdown path; defaults beside --evidence.",
    )
    anonymize_parser.add_argument(
        "--dashboard",
        type=Path,
        default=None,
        help="Optional self-contained aggregate-only HTML dashboard.",
    )
    _add_release_policy_arguments(anonymize_parser, include_transformation=True)
    anonymize_parser.add_argument(
        "--privacy-unit-kind",
        choices=(
            "row",
            "patient",
            "person",
            "encounter",
            "document",
            "event",
            "household",
            "other",
        ),
        default=None,
    )
    anonymize_parser.add_argument(
        "--population-scope",
        choices=(
            "source_population",
            "eligible_cohort",
            "sampled_cohort",
            "release_cohort",
            "external_reference_population",
            "other_documented",
        ),
        default="release_cohort",
    )
    anonymize_parser.add_argument(
        "--release-model",
        choices=("public", "restricted", "controlled", "internal", "other_documented"),
        required=True,
    )
    anonymize_parser.add_argument(
        "--recipient-model",
        choices=(
            "general_public",
            "named_researchers",
            "covered_entity",
            "authorized_internal",
            "contracted_recipient",
            "other_documented",
        ),
        required=True,
    )
    anonymize_parser.add_argument(
        "--auxiliary-data-model",
        choices=(
            "publicly_available",
            "recipient_supplied",
            "reasonably_available",
            "expert_defined",
            "none_assumed",
            "other_documented",
        ),
        required=True,
    )
    anonymize_parser.add_argument(
        "--assumptions-notes",
        type=Path,
        default=None,
        help=(
            "Optional local UTF-8 .md or .txt notes whose content is bound by "
            "digest but never copied into evidence; required for any 'other' "
            "release-context choice."
        ),
    )
    anonymize_parser.add_argument("--overwrite", action="store_true")
    anonymize_parser.set_defaults(handler=_handle_risk_anonymize)


def _add_release_policy_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_transformation: bool,
) -> None:
    parser.add_argument(
        "--qi",
        type=_column_list_arg,
        default=(),
        help="Comma-separated quasi-identifier columns.",
    )
    parser.add_argument(
        "--qi-column",
        action="append",
        type=_literal_column_arg,
        default=[],
        help="Repeat for a literal QI column name, including names with commas.",
    )
    parser.add_argument(
        "--sensitive",
        type=_column_list_arg,
        default=(),
        help="Comma-separated sensitive-attribute columns.",
    )
    parser.add_argument(
        "--sensitive-column",
        action="append",
        type=_literal_column_arg,
        default=[],
        help=(
            "Repeat for a literal sensitive column name, including names with commas."
        ),
    )
    parser.add_argument(
        "--direct-id",
        type=_column_list_arg,
        default=(),
        help="Comma-separated direct-identifier columns to remove.",
    )
    parser.add_argument(
        "--direct-id-column",
        action="append",
        type=_literal_column_arg,
        default=[],
        help=(
            "Repeat for a literal direct-identifier column name, including "
            "names with commas."
        ),
    )
    parser.add_argument(
        "--non-sensitive",
        type=_column_list_arg,
        default=(),
        help="Comma-separated reviewed non-sensitive columns.",
    )
    parser.add_argument(
        "--non-sensitive-column",
        action="append",
        type=_literal_column_arg,
        default=[],
        help=(
            "Repeat for a literal non-sensitive column name, including names "
            "with commas."
        ),
    )
    parser.add_argument(
        "--exclude",
        type=_column_list_arg,
        default=(),
        help="Comma-separated columns excluded from release.",
    )
    parser.add_argument(
        "--exclude-column",
        action="append",
        type=_literal_column_arg,
        default=[],
        help="Repeat for a literal excluded column name, including names with commas.",
    )
    parser.add_argument(
        "--privacy-unit",
        default=None,
        help="Patient/person key used only for analysis and removed from output.",
    )
    parser.add_argument(
        "--k",
        type=_positive_int,
        required=True,
        help="Explicit target k; OpenMed does not choose a regulatory default.",
    )
    parser.add_argument("--l", type=_positive_int, default=1)
    parser.add_argument(
        "--l-metric",
        choices=("distinct", "entropy"),
        default="distinct",
    )
    parser.add_argument("--t", type=_unit_interval_float, default=1.0)
    if not include_transformation:
        return
    parser.add_argument("--max-suppressed-units", type=_non_negative_int, default=None)
    parser.add_argument(
        "--max-suppression-rate",
        type=_unit_interval_float,
        default=0.0,
    )
    parser.add_argument(
        "--max-lattice-nodes",
        type=_positive_int,
        default=100_000,
    )
    parser.add_argument(
        "--max-suppression-subsets",
        type=_positive_int,
        default=100_000,
        help="Maximum equivalence-class suppression subsets evaluated.",
    )
    parser.add_argument(
        "--hierarchies",
        type=Path,
        default=None,
        help="Optional reviewed hierarchy JSON.",
    )


def _add_fhir_command(subparsers: argparse._SubParsersAction) -> None:
    """Add FHIR export commands."""
    fhir_parser = subparsers.add_parser("fhir", help="FHIR export utilities.")
    fhir_sub = fhir_parser.add_subparsers(dest="fhir_command")

    bundle_parser = fhir_sub.add_parser(
        "bundle",
        help="Assemble standalone FHIR resources into a deterministic Bundle.",
    )
    bundle_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSON result file containing standalone FHIR resources.",
    )
    bundle_parser.add_argument(
        "--type",
        dest="bundle_type",
        choices=sorted(_FHIR_BUNDLE_TYPES),
        required=True,
        help="FHIR Bundle type to emit.",
    )
    bundle_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the FHIR Bundle JSON.",
    )
    bundle_parser.set_defaults(handler=_handle_fhir_bundle)


def _add_icd11_command(subparsers: argparse._SubParsersAction) -> None:
    """Add offline ICD-11 snapshot management commands."""
    icd11_parser = subparsers.add_parser(
        "icd11",
        help="Build local ICD-11 MMS grounding snapshots.",
    )
    icd11_sub = icd11_parser.add_subparsers(dest="icd11_command")

    build_parser = icd11_sub.add_parser(
        "build-snapshot",
        help="Build a release-pinned chapter subset from the WHO ICD-API.",
    )
    build_parser.add_argument(
        "--release",
        required=True,
        help="Pinned WHO ICD-11 release in YYYY-MM form.",
    )
    build_parser.add_argument(
        "--chapter",
        dest="chapters",
        action="append",
        required=True,
        help="Chapter code to include; repeat for multiple chapters.",
    )
    build_parser.add_argument(
        "--language",
        default="en",
        help="WHO response language (default: en).",
    )
    build_parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Snapshot output directory (default: OPENMED_ICD11_CACHE_DIR).",
    )
    build_parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-request timeout in seconds (default: 30).",
    )
    build_parser.set_defaults(handler=_handle_icd11_build_snapshot)


def _add_omop_command(subparsers: argparse._SubParsersAction) -> None:
    """Add OMOP CDM load commands over the existing local-first loader."""
    omop_parser = subparsers.add_parser(
        "omop",
        help="Load grounded clinical spans into a local OMOP CDM target.",
    )
    omop_sub = omop_parser.add_subparsers(dest="omop_command")

    load_parser = omop_sub.add_parser(
        "load",
        help="Load a grounded-results JSONL file into a local OMOP CDM target.",
    )
    load_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSONL file of grounded note records to load.",
    )
    load_parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help=(
            "Local target DSN to persist into (a file for duckdb/sqlite, a "
            "directory for parquet). Omit to only report the load summary."
        ),
    )
    load_parser.add_argument(
        "--writer",
        choices=_OMOP_WRITERS,
        default="sqlite",
        help="On-device writer used when --target is set (default: sqlite).",
    )
    load_parser.add_argument(
        "--vocabulary-version",
        dest="vocabulary_version",
        default=None,
        help="Optional vocabulary version recorded in SOURCE_TO_CONCEPT_MAP rows.",
    )
    load_parser.add_argument(
        "--validate",
        dest="validate",
        action="store_true",
        help="Validate CDM constraints and report PHI-free violation counts.",
    )
    load_parser.set_defaults(handler=_handle_omop_load)


def _add_ground_command(subparsers: argparse._SubParsersAction) -> None:
    """Add the shared offline grounding command."""

    ground_parser = subparsers.add_parser(
        "ground",
        help="Ground text or pre-extracted entities against local snapshots.",
    )
    input_group = ground_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text",
        help="Synthetic or caller-owned text to treat as one span.",
    )
    input_group.add_argument(
        "--input",
        type=Path,
        help="JSON/JSONL file containing text or pre-extracted entities.",
    )
    ground_parser.add_argument(
        "--system",
        dest="systems",
        action="append",
        help="Vocabulary system to search; repeat for multiple systems.",
    )
    ground_parser.add_argument(
        "--source-language",
        default="en",
        help="Source language tag used for multilingual aliases.",
    )
    ground_parser.add_argument(
        "--top-k",
        type=_positive_int,
        default=5,
        help="Maximum ranked candidates per retrieval channel (default: 5).",
    )
    ground_parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Local grounding snapshot cache directory.",
    )
    ground_parser.add_argument(
        "--offline",
        dest="offline",
        action="store_true",
        default=True,
        help="Disable network access during grounding (default).",
    )
    ground_parser.add_argument(
        "--online",
        dest="offline",
        action="store_false",
        help="Allow configured snapshot downloads during grounding.",
    )
    ground_parser.set_defaults(handler=_handle_ground)


def _add_grounding_snapshot_command(subparsers: argparse._SubParsersAction) -> None:
    """Add explicit import/download lifecycle commands for snapshots."""

    grounding_parser = subparsers.add_parser(
        "grounding",
        help="Manage checksum-verified terminology snapshots.",
    )
    grounding_sub = grounding_parser.add_subparsers(dest="grounding_command")

    import_parser = grounding_sub.add_parser(
        "import",
        help="Import a local permissive vocabulary snapshot into the cache.",
    )
    import_parser.add_argument("--system", required=True)
    import_parser.add_argument("--input", type=Path, required=True)
    import_parser.add_argument("--version", required=True)
    import_parser.add_argument("--sha256", default=None)
    import_parser.add_argument("--cache-dir", type=Path, default=None)
    import_parser.add_argument("--license-note", default="")
    import_parser.add_argument("--replace", action="store_true")
    import_parser.set_defaults(handler=_handle_grounding_snapshot_import)

    download_parser = grounding_sub.add_parser(
        "download",
        help="Download and checksum-verify one configured public snapshot.",
    )
    download_parser.add_argument("--system", required=True)
    download_parser.add_argument("--url", required=True)
    download_parser.add_argument("--sha256", required=True)
    download_parser.add_argument("--version", default=None)
    download_parser.add_argument("--checksum-url", default=None)
    download_parser.add_argument("--artifact-name", default="concepts.tsv")
    download_parser.add_argument("--archive-member", default=None)
    download_parser.add_argument("--cache-dir", type=Path, default=None)
    download_parser.add_argument("--timeout", type=float, default=60.0)
    download_parser.set_defaults(handler=_handle_grounding_snapshot_download)

    list_parser = grounding_sub.add_parser(
        "list",
        help="List locally imported terminology snapshots.",
    )
    list_parser.add_argument("--cache-dir", type=Path, default=None)
    list_parser.set_defaults(handler=_handle_grounding_snapshot_list)


def _add_export_command(subparsers: argparse._SubParsersAction) -> None:
    """Add clinical export commands."""

    export_parser = subparsers.add_parser("export", help="Clinical export utilities.")
    export_sub = export_parser.add_subparsers(dest="export_command")

    openehr_parser = export_sub.add_parser(
        "openehr",
        help="Serialize grounded clinical entities into openEHR flat JSON.",
    )
    openehr_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSON result file containing grounded clinical entities.",
    )
    openehr_parser.add_argument(
        "--template",
        type=Path,
        required=True,
        help="EHRbase WebTemplate JSON or allowed-path template manifest.",
    )
    openehr_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the openEHR flat COMPOSITION JSON.",
    )
    openehr_parser.add_argument(
        "--doc-id",
        default=None,
        help="Stable source document id; defaults to the input payload id.",
    )
    openehr_parser.add_argument(
        "--source-text-file",
        type=Path,
        default=None,
        help="Optional de-identified note text file for offset validation.",
    )
    openehr_parser.add_argument(
        "--composer",
        default="OpenMed",
        help="Composer name for openEHR context.",
    )
    openehr_parser.add_argument(
        "--language",
        default="en",
        help="ISO language code for openEHR context.",
    )
    openehr_parser.add_argument(
        "--territory",
        default="US",
        help="ISO territory code for openEHR context.",
    )
    openehr_parser.add_argument(
        "--time",
        default=None,
        help="Composition timestamp; defaults to the current UTC time.",
    )
    openehr_parser.add_argument(
        "--vocabulary-key",
        default=None,
        help="Enable caller-supplied terminology codings when present.",
    )
    openehr_parser.set_defaults(handler=_handle_export_openehr)


def _add_models_command(subparsers: argparse._SubParsersAction) -> None:
    models_parser = subparsers.add_parser("models", help="Discover OpenMed models.")
    models_sub = models_parser.add_subparsers(dest="models_command")

    models_pull = models_sub.add_parser(
        "pull",
        help="Download and integrity-check a model for offline use.",
    )
    models_pull.add_argument(
        "model",
        help="Registry alias, bare model name, or Hugging Face repository id.",
    )
    models_pull.add_argument(
        "--revision",
        default=None,
        help="Optional branch, tag, or commit to download.",
    )
    models_pull.add_argument(
        "--max-bandwidth",
        type=_positive_int,
        default=None,
        metavar="BYTES_PER_SECOND",
        help="Limit aggregate download bandwidth in bytes per second.",
    )
    models_pull.add_argument(
        "--retries",
        type=_non_negative_int,
        default=5,
        help="Retries for transient network failures (default: 5).",
    )
    models_pull.set_defaults(handler=_handle_models_pull)

    models_list = models_sub.add_parser("list", help="List available models.")
    models_list.add_argument(
        "--include-remote",
        action="store_true",
        help="Fetch additional models from Hugging Face Hub.",
    )
    models_list.set_defaults(handler=_handle_models_list)

    models_info = models_sub.add_parser(
        "info",
        help="Show metadata for a registry model.",
    )
    models_info.add_argument(
        "model_key",
        help="Registry key defined in openmed.core.model_registry.",
    )
    models_info.set_defaults(handler=_handle_models_info)

    models_verify = models_sub.add_parser(
        "verify",
        help="Verify cached model artifacts without network access.",
    )
    models_verify.add_argument(
        "model_id",
        nargs="?",
        help="Registry model id or local model directory.",
    )
    models_verify.add_argument(
        "--all",
        dest="all_models",
        action="store_true",
        help="Verify every cached model with integrity metadata.",
    )
    models_verify.set_defaults(handler=_handle_models_verify)

    models_size = models_sub.add_parser(
        "size",
        help="Show model download, disk, and peak RAM estimates.",
    )
    models_size.add_argument(
        "model_key",
        nargs="?",
        default=None,
        help="Optional registry alias or full model repository id.",
    )
    models_size.add_argument(
        "--remote",
        action="store_true",
        help="Refine snapshot sizes from Hugging Face Hub metadata.",
    )
    models_size.add_argument(
        "--budget-mb",
        type=_non_negative_float,
        default=None,
        help="Only show models needing at most this many MB to download.",
    )
    models_size.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        dest="output_format",
        help="Output format (default: table).",
    )
    models_size.set_defaults(handler=_handle_models_size)

    models_search = models_sub.add_parser(
        "search",
        help="Search the canonical model manifest.",
    )
    models_search.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Case-insensitive substring matched against repo_id or family.",
    )
    models_search.add_argument("--task", help="Filter by model task.")
    models_search.add_argument("--language", help="Filter by language code.")
    models_search.add_argument("--tier", help="Filter by model tier.")
    models_search.add_argument(
        "--max-params",
        type=_non_negative_int,
        default=None,
        help="Maximum parameter count. Unknown counts are retained by default.",
    )
    models_search.add_argument(
        "--min-params",
        type=_non_negative_int,
        default=None,
        help="Minimum parameter count.",
    )
    models_search.add_argument(
        "--format",
        help="Filter by runtime format or device, such as mlx, coreml, onnx, or pytorch.",
    )
    models_search.add_argument("--license", help="Filter by SPDX license string.")
    models_search.add_argument(
        "--require-params",
        action="store_true",
        help="Exclude manifest rows with unknown parameter counts.",
    )
    models_search.set_defaults(handler=_handle_models_search)

    models_recommend = models_sub.add_parser(
        "recommend",
        help="Recommend the best on-device model for a task and device tier.",
    )
    models_recommend.add_argument("--task", help="Filter by model task.")
    models_recommend.add_argument("--language", help="Filter by language code.")
    models_recommend.add_argument(
        "--tier",
        required=True,
        choices=["phone", "laptop", "workstation", "server"],
        help="Target device tier the recommended model must fit.",
    )
    models_recommend.set_defaults(handler=_handle_models_recommend)

    models_card = models_sub.add_parser(
        "card",
        help="Render a README model card from the canonical manifest.",
    )
    models_card.add_argument(
        "repo_id",
        help="Hugging Face repository id to resolve from models.jsonl.",
    )
    card_output = models_card.add_mutually_exclusive_group()
    card_output.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Optional path to write the rendered README Markdown.",
    )
    card_output.add_argument(
        "--check",
        type=Path,
        metavar="README",
        help="Compare an existing README against the rendered card.",
    )
    models_card.set_defaults(handler=_handle_models_card)

    models_freshness = models_sub.add_parser(
        "freshness",
        help="Compute freshness metrics from the canonical model manifest.",
    )
    models_freshness.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to a model manifest JSONL file.",
    )
    models_freshness.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Optional path to write the metrics artifact.",
    )
    models_freshness.add_argument(
        "--format",
        dest="artifact_format",
        choices=["json", "markdown"],
        default="json",
        help="Artifact format to print or write.",
    )
    models_freshness.add_argument(
        "--as-of",
        default=None,
        help="Reference date in YYYY-MM-DD format. Defaults to today in UTC.",
    )
    models_freshness.add_argument(
        "--target-days",
        type=int,
        default=None,
        help="Reference median-age target in days.",
    )
    models_freshness.set_defaults(handler=_handle_models_freshness)

    models_diff = models_sub.add_parser(
        "diff",
        help="Diff two canonical model manifest JSONL files.",
    )
    models_diff.add_argument(
        "old_manifest",
        type=Path,
        help="Path to the older model manifest JSONL file.",
    )
    models_diff.add_argument(
        "new_manifest",
        type=Path,
        help="Path to the newer model manifest JSONL file.",
    )
    models_diff.add_argument(
        "--fail-on-removed",
        action="store_true",
        help="Exit non-zero when any repo was removed between manifests.",
    )
    models_diff.set_defaults(handler=_handle_models_diff)

    models_validate = models_sub.add_parser(
        "validate",
        help="Validate the canonical model manifest schema.",
    )
    models_validate.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to a model manifest JSONL file.",
    )
    models_validate.set_defaults(handler=_handle_models_validate)


def _add_doctor_command(
    subparsers: argparse._SubParsersAction,
) -> None:
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Inspect the OpenMed environment and dependencies.",
    )

    doctor_parser.set_defaults(
        handler=_handle_doctor,
    )


def _add_release_command(subparsers: argparse._SubParsersAction) -> None:
    from .release import add_release_command

    add_release_command(subparsers)


def _add_config_command(subparsers: argparse._SubParsersAction) -> None:
    config_parser = subparsers.add_parser(
        "config", help="Inspect or modify OpenMed CLI configuration."
    )
    config_sub = config_parser.add_subparsers(dest="config_command")

    config_show = config_sub.add_parser("show", help="Display active configuration.")
    config_show.add_argument(
        "--profile",
        help="Show configuration with a specific profile applied.",
    )
    config_show.set_defaults(handler=_handle_config_show)

    config_set = config_sub.add_parser("set", help="Persist a configuration value.")
    config_set.add_argument("key", help="Configuration key to set.")
    config_set.add_argument(
        "value",
        nargs="?",
        help="Value to store. Required unless --unset is provided.",
    )
    config_set.add_argument(
        "--unset",
        action="store_true",
        help="Clear the value for the given key.",
    )
    config_set.set_defaults(handler=_handle_config_set)

    # Profile management subcommands
    profile_list = config_sub.add_parser(
        "profiles", help="List available configuration profiles."
    )
    profile_list.set_defaults(handler=_handle_profile_list)

    profile_show = config_sub.add_parser(
        "profile-show", help="Show settings for a specific profile."
    )
    profile_show.add_argument("profile_name", help="Name of the profile to show.")
    profile_show.set_defaults(handler=_handle_profile_show)

    profile_use = config_sub.add_parser(
        "profile-use", help="Apply a profile to the current configuration."
    )
    profile_use.add_argument("profile_name", help="Name of the profile to use.")
    profile_use.set_defaults(handler=_handle_profile_use)

    profile_save = config_sub.add_parser(
        "profile-save", help="Save current configuration as a named profile."
    )
    profile_save.add_argument("profile_name", help="Name for the new profile.")
    profile_save.set_defaults(handler=_handle_profile_save)

    profile_delete = config_sub.add_parser(
        "profile-delete", help="Delete a custom profile."
    )
    profile_delete.add_argument("profile_name", help="Name of the profile to delete.")
    profile_delete.set_defaults(handler=_handle_profile_delete)


def _add_policy_command(subparsers: argparse._SubParsersAction) -> None:
    policy_parser = subparsers.add_parser(
        "policy", help="Inspect and validate OpenMed policy profiles."
    )
    policy_sub = policy_parser.add_subparsers(dest="policy_command")

    diff_parser = policy_sub.add_parser(
        "diff",
        help="Compare two policy profile configurations.",
    )
    diff_parser.add_argument(
        "base",
        help="Baseline bundled profile name or policy JSON path.",
    )
    diff_parser.add_argument(
        "candidate",
        help="Candidate bundled profile name or policy JSON path.",
    )
    diff_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        dest="output_format",
        help="Output format.",
    )
    diff_parser.set_defaults(handler=_handle_policy_diff)

    policy_lint = policy_sub.add_parser(
        "lint",
        help="Lint a bundled policy name or policy profile JSON file.",
    )
    policy_lint.add_argument(
        "target",
        help="Policy profile name or path to a policy profile JSON file.",
    )
    policy_lint.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code when warnings are present.",
    )
    policy_lint.set_defaults(handler=_handle_policy_lint)


def _add_benchmark_command(subparsers: argparse._SubParsersAction) -> None:
    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Run benchmark and adversarial evaluation suites."
    )
    benchmark_sub = benchmark_parser.add_subparsers(dest="benchmark_command")

    pii_parser = benchmark_sub.add_parser(
        "pii",
        help="Run PII benchmark suites.",
    )
    pii_parser.add_argument(
        "--attack",
        choices=["reid"],
        default=None,
        help="Optional adversarial attack mode.",
    )
    pii_parser.add_argument(
        "--suite",
        default=None,
        help="Benchmark suite to run. Defaults to shield, or golden for re-id attacks.",
    )
    pii_parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="One or more model identifiers. Comma-separated values are accepted.",
    )
    pii_parser.add_argument(
        "--device",
        default="cpu",
        help="Device tier label recorded in the benchmark report.",
    )
    pii_parser.add_argument(
        "--model",
        default=None,
        help="Model identifier to record in the re-id report.",
    )
    pii_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the BenchmarkReport JSON.",
    )
    pii_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for per-model JSON and Markdown reports.",
    )
    pii_parser.add_argument(
        "--leaderboard-output",
        type=Path,
        default=None,
        help="Optional path for a generated leaderboard table.",
    )
    pii_parser.add_argument(
        "--leaderboard-format",
        choices=["markdown", "json"],
        default="markdown",
        help="Generated leaderboard format.",
    )
    pii_parser.add_argument(
        "--full-shield",
        action="store_true",
        help="Use the approved-access full SHIELD corpus instead of the public sample.",
    )
    pii_parser.add_argument(
        "--checkpoint-manifest",
        type=Path,
        default=None,
        help=(
            "Checkpoint JSON or JSONL for the named clinical PHI flagship. "
            "This enables manifest-linked SHIELD comparison evidence."
        ),
    )
    pii_parser.add_argument(
        "--checkpoint-manifest-ref",
        default=None,
        help=(
            "Stable repository or publication link to --checkpoint-manifest, "
            "recorded in the BenchmarkReport."
        ),
    )
    pii_parser.set_defaults(handler=_handle_benchmark_pii)

    clinical_parser = benchmark_sub.add_parser(
        "clinical",
        help="Resolve clinical benchmark suites such as DrugProt.",
    )
    clinical_parser.add_argument(
        "--suite",
        default="drugprot",
        help="Clinical benchmark suite to load.",
    )
    clinical_parser.add_argument(
        "--task",
        choices=["ner", "relation"],
        default="ner",
        help="Clinical benchmark task view to load.",
    )
    clinical_parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional local corpus directory, fixture file, or DrugProt archive.",
    )
    clinical_parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional cache directory for download-on-demand public corpora.",
    )
    clinical_parser.add_argument(
        "--split",
        default=None,
        help="Optional public-corpus split to load.",
    )
    clinical_parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "One or more model identifiers for NER benchmark reports. "
            "Comma-separated values are accepted."
        ),
    )
    clinical_parser.add_argument(
        "--device",
        default="cpu",
        help="Device tier label recorded in NER benchmark reports.",
    )
    clinical_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for a JSON suite-resolution summary.",
    )
    clinical_parser.set_defaults(handler=_handle_benchmark_clinical)

    mobile_parser = benchmark_sub.add_parser(
        "mobile",
        help="Parse mobile benchmark options.",
    )
    mobile_parser.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        default=None,
        help=(
            "Model id(s), comma-separated ids, or @manifest. When omitted, "
            "the committed synthetic mobile workload runner is used."
        ),
    )
    mobile_parser.add_argument(
        "--device",
        choices=_MOBILE_BENCHMARK_DEVICES,
        required=True,
        help="Mobile runtime device.",
    )
    mobile_parser.add_argument(
        "--tier",
        choices=_MOBILE_BENCHMARK_TIERS,
        required=True,
        help="Device tier to benchmark.",
    )
    mobile_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where benchmark reports will be written.",
    )
    mobile_parser.set_defaults(handler=_handle_benchmark_mobile)

    from openmed.eval import arm_latency as arm_latency_module

    latency_parser = benchmark_sub.add_parser(
        "latency",
        help="Gate cached INT8 ONNX latency over synthetic SMS-scale text.",
    )
    latency_parser.add_argument(
        "--model",
        default=arm_latency_module.DEFAULT_ARM_LATENCY_MODEL,
        help="Cached model repository id or local ONNX artifact directory.",
    )
    latency_parser.add_argument(
        "--revision",
        default=arm_latency_module.DEFAULT_ARM_LATENCY_MODEL_REVISION,
        help="Pinned model revision recorded in the report.",
    )
    latency_parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional pre-populated Hugging Face cache directory.",
    )
    latency_parser.add_argument(
        "--corpus",
        type=Path,
        default=arm_latency_module.DEFAULT_ARM_LATENCY_CORPUS,
        help="Synthetic SMS-scale JSONL corpus.",
    )
    latency_parser.add_argument(
        "--budget",
        type=Path,
        default=arm_latency_module.DEFAULT_ARM_LATENCY_BUDGET,
        help="Committed ARM p95 budget JSON.",
    )
    latency_parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Warm-up inferences excluded from the latency distribution.",
    )
    latency_parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Measured repetitions of the committed corpus.",
    )
    latency_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the JSON report; JSON is always emitted to stdout.",
    )
    latency_parser.set_defaults(handler=_handle_benchmark_latency)

    false_negatives_parser = benchmark_sub.add_parser(
        "false-negatives",
        help="Explore missed gold PHI spans from an error-analysis report.",
    )
    false_negatives_parser.add_argument(
        "report",
        type=Path,
        help="Path to an error-analysis report JSON produced by the eval harness.",
    )
    false_negatives_parser.add_argument(
        "--fixtures",
        action="append",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Optional synthetic gold fixture file(s) used to render span text and "
            "surrounding context. Without them only offsets, labels, and hashes "
            "are shown. Repeat to combine multiple fixture files."
        ),
    )
    false_negatives_parser.add_argument(
        "--label",
        default=None,
        help="Only show missed spans for this label (case-insensitive).",
    )
    false_negatives_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the total number of missed spans shown.",
    )
    false_negatives_parser.add_argument(
        "--context-chars",
        type=int,
        default=None,
        help="Trim rendered context windows to this many characters around a span.",
    )
    false_negatives_parser.set_defaults(handler=_handle_benchmark_false_negatives)


def _add_profile_command(subparsers: argparse._SubParsersAction) -> None:
    """Register inference-path profiling commands with the CLI parser."""
    profile_parser = subparsers.add_parser(
        "profile", help="Profile the inference path."
    )
    profile_sub = profile_parser.add_subparsers(dest="profile_command")

    memory_parser = profile_sub.add_parser(
        "memory",
        help="Profile inference-path memory across load and inference phases.",
    )
    memory_parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model id or local path. When omitted, the committed synthetic "
            "one-page-note workload runner is profiled offline."
        ),
    )
    memory_parser.add_argument(
        "--top-allocators",
        type=int,
        default=None,
        help="Number of top allocators to report per phase (default: 10).",
    )
    memory_parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Output format for the memory profile (default: json).",
    )
    memory_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the profile to this file instead of stdout.",
    )
    memory_parser.set_defaults(handler=_handle_profile_memory)


def _add_eval_command(subparsers: argparse._SubParsersAction) -> None:
    """Register evaluation commands with the CLI parser."""
    eval_parser = subparsers.add_parser("eval", help="Run evaluation tools.")
    eval_sub = eval_parser.add_subparsers(dest="eval_command")

    load_parser = eval_sub.add_parser(
        "load-test", help="Load test the ASGI app in-process."
    )
    load_parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of requests to run at once (default: 4).",
    )
    load_parser.add_argument(
        "--total-requests",
        type=int,
        default=20,
        help="Total number of requests to run (default: 20).",
    )
    load_parser.set_defaults(handler=_handle_eval_load_test)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point invoked by the console script."""
    parser = build_parser()
    args = parser.parse_args(argv)

    handler: Optional[Handler] = getattr(args, "handler", None)

    if handler is None:
        parser.print_help()
        return 0

    try:
        return handler(args)
    except _UnavailableCommandError as exc:
        error = CliError(
            str(exc),
            code="not_implemented",
            exit_code=EXIT_ERROR,
        )
        return emit_error(args, error)
    except CliError as exc:
        return emit_error(args, exc)
    except Exception as exc:
        # Keep unexpected failures scriptable without echoing exception text,
        # which may contain input content or other sensitive details.
        error = CliError(
            f"Command failed with {type(exc).__name__}.",
            code="runtime_error",
            exit_code=EXIT_ERROR,
        )
        return emit_error(args, error)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _load_and_apply_config(args: argparse.Namespace) -> OpenMedConfig:
    config_path = getattr(args, "config_path", None)
    try:
        config = load_config_from_file(config_path)
        set_config(config)
        return config
    except FileNotFoundError:
        config = get_config()

    # Apply CLI overrides if present
    if (
        hasattr(args, "use_medical_tokenizer")
        and args.use_medical_tokenizer is not None
    ):
        config.use_medical_tokenizer = bool(args.use_medical_tokenizer)

    if getattr(args, "medical_tokenizer_exceptions", None):
        extras = [
            item.strip()
            for item in str(args.medical_tokenizer_exceptions).split(",")
            if item.strip()
        ]
        config.medical_tokenizer_exceptions = extras if extras else None

    set_config(config)
    return config


def _handle_analyze(args: argparse.Namespace) -> int:
    _load_and_apply_config(args)

    if args.text:
        text = args.text
    else:
        try:
            text = args.input_file.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise CliError(
                f"Input file not found: {args.input_file}",
                code="input_not_found",
                exit_code=EXIT_ERROR,
            )
        except OSError as exc:  # pragma: no cover - defensive
            raise CliError(
                f"Failed to read {args.input_file}: {exc}",
                code="read_failed",
                exit_code=EXIT_ERROR,
            )

    analyze_text, _, _, _ = _lazy_api()

    result = analyze_text(
        text,
        model_name=args.model,
        output_format=args.output_format,
        confidence_threshold=args.confidence_threshold,
        group_entities=args.group_entities,
        include_confidence=not args.no_confidence,
    )

    if isinstance(result, str):
        payload: Any = {"format": args.output_format, "output": result}
        human = result
    else:
        data = result.to_dict() if hasattr(result, "to_dict") else result
        payload = data
        human = json.dumps(data, indent=2)

    return emit(args, payload, human=human)


def _handle_batch(args: argparse.Namespace) -> int:
    config = _load_and_apply_config(args)

    _, _, _, BatchProcessor = _lazy_api()

    if args.checkpoint_interval < 1:
        sys.stderr.write("--checkpoint-interval must be positive\n")
        return 2
    if (args.resume or args.checkpoint_path is not None) and args.output is None:
        sys.stderr.write("--resume and --checkpoint-path require --output\n")
        return 2

    checkpoint_path = None
    if args.output is not None:
        checkpoint_path = args.checkpoint_path or Path(f"{args.output}.checkpoint.json")

    continue_on_error = not args.stop_on_error if args.stop_on_error else True

    processor = BatchProcessor(
        model_name=args.model,
        config=config,
        confidence_threshold=args.confidence_threshold or 0.0,
        group_entities=args.group_entities,
        continue_on_error=continue_on_error,
        checkpoint_interval=args.checkpoint_interval,
    )

    def progress_callback(current: int, total: int, result: Any) -> None:
        if args.quiet:
            return
        status = "OK" if result and result.success else "FAILED"
        item_id = result.id if result else "?"
        sys.stderr.write(f"\r[{current}/{total}] {item_id}: {status}")
        sys.stderr.flush()

    try:
        if args.texts:
            result = processor.process_texts(
                args.texts,
                progress_callback=progress_callback if not args.quiet else None,
                output_path=args.output,
                checkpoint_path=checkpoint_path,
                resume_from_checkpoint=args.resume,
                output_format=args.output_format,
            )
        elif args.input_files:
            result = processor.process_files(
                args.input_files,
                progress_callback=progress_callback if not args.quiet else None,
                output_path=args.output,
                checkpoint_path=checkpoint_path,
                resume_from_checkpoint=args.resume,
                output_format=args.output_format,
            )
        elif args.input_dir:
            if not args.input_dir.is_dir():
                raise CliError(
                    f"Not a directory: {args.input_dir}",
                    code="not_a_directory",
                    exit_code=EXIT_ERROR,
                )
            result = processor.process_directory(
                args.input_dir,
                pattern=args.pattern,
                recursive=args.recursive,
                progress_callback=progress_callback if not args.quiet else None,
                output_path=args.output,
                checkpoint_path=checkpoint_path,
                resume_from_checkpoint=args.resume,
                output_format=args.output_format,
            )
        else:
            raise CliError("No input provided.", code="no_input", exit_code=EXIT_USAGE)

    except CliError:
        raise
    except Exception as exc:
        raise CliError(
            f"\nBatch processing failed: {exc}",
            code="batch_failed",
            exit_code=EXIT_ERROR,
        )

    if not args.quiet:
        sys.stderr.write("\n")

    payload = result.to_dict()
    if args.output_format == "json":
        output = json.dumps(payload, indent=2)
    else:
        output = result.summary()

    if args.output:
        human = f"Results written to: {args.output}"
    else:
        human = output

    emit(args, payload, human=human)
    return 0 if result.failed_items == 0 else 1


def _handle_tui(args: argparse.Namespace) -> int:
    raise _UnavailableCommandError(
        "The historical OpenMed TUI is not implemented because its backend "
        "was removed; use 'openmed --help' for supported commands."
    )


def _handle_redact_dataset(args: argparse.Namespace) -> int:
    from .redact_dataset import run_from_args

    config = _load_and_apply_config(args)
    return run_from_args(args, config=config)


def _handle_audit_verify(args: argparse.Namespace) -> int:
    try:
        artifact = _load_audit_artifact(args.report)
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        raise CliError(
            f"Failed to load audit artifact: {exc}",
            code="load_failed",
            exit_code=EXIT_ERROR,
        )

    from ..core.audit_chain import AuditChain

    if isinstance(artifact, AuditChain):
        payload, human = _audit_chain_verification(artifact)
    else:
        payload, human = _audit_report_verification(artifact, args.key)
    emit(args, payload, human=human)
    return 0 if payload["verified"] else 1


def _audit_report_verification(
    report: Any,
    key_override: str | None,
) -> tuple[dict[str, Any], str]:
    repro_ok = report.repro_hash_matches()
    signature_status = "SKIPPED (report is unsigned)"
    signature_ok = True

    if report.signature is not None:
        key = key_override or os.environ.get(_AUDIT_KEY_ENV)
        if not key:
            signature_status = f"FAIL (set --key or {_AUDIT_KEY_ENV})"
            signature_ok = False
        else:
            try:
                signature_ok = report.verify(key)
            except (TypeError, ValueError) as exc:
                signature_status = f"FAIL ({exc})"
                signature_ok = False
            else:
                signature_status = _pass_fail(signature_ok)

    verified = repro_ok and signature_ok
    payload = {
        "verified": verified,
        "repro_hash_ok": repro_ok,
        "signature_ok": signature_ok,
        "signature_status": signature_status,
    }
    human = (
        f"Audit report verification: {_pass_fail(verified)}\n"
        f"Reproducibility hash: {_pass_fail(repro_ok)}\n"
        f"HMAC signature: {signature_status}"
    )
    return payload, human


def _audit_chain_verification(chain: Any) -> tuple[dict[str, Any], str]:
    result = chain.verify()
    payload = {
        "verified": result.valid,
        "entries_checked": result.checked_entries,
        "reason": result.reason,
    }
    lines = [
        f"Audit chain verification: {_pass_fail(result.valid)}",
        f"Entries checked: {result.checked_entries}",
    ]
    if not result.valid:
        lines.append(f"Reason: {result.reason}")
    return payload, "\n".join(lines)


def _handle_audit_chain_verify(args: argparse.Namespace) -> int:
    from ..core.audit_chain import AuditChain

    try:
        chain = AuditChain.load(args.chain)
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        raise CliError(
            f"Failed to load audit chain: {exc}",
            code="load_failed",
            exit_code=EXIT_ERROR,
        )

    chain_payload, chain_human = _audit_chain_verification(chain)
    payload: dict[str, Any] = {
        "verified": chain_payload["verified"],
        "chain": chain_payload,
    }
    human_parts = [chain_human]
    if args.report is None:
        emit(args, payload, human="\n".join(human_parts))
        return 0 if payload["verified"] else 1

    try:
        report = _load_audit_report(args.report)
    except (OSError, TypeError, ValueError) as exc:
        raise CliError(
            f"Failed to load audit report: {exc}",
            code="load_failed",
            exit_code=EXIT_ERROR,
        )

    report_payload, report_human = _audit_report_verification(report, args.key)
    membership_ok = chain.contains_report(report)
    payload.update(
        {
            "verified": bool(
                chain_payload["verified"]
                and report_payload["verified"]
                and membership_ok
            ),
            "report": report_payload,
            "report_membership": membership_ok,
        }
    )
    human_parts.extend(
        [
            report_human,
            f"Audit report chain membership: {_pass_fail(membership_ok)}",
        ]
    )
    emit(args, payload, human="\n".join(human_parts))
    return 0 if payload["verified"] else 1


def _handle_audit_show(args: argparse.Namespace) -> int:
    try:
        report = _load_audit_report(args.report)
    except (OSError, TypeError, ValueError) as exc:
        raise CliError(
            f"Failed to load audit report: {exc}",
            code="load_failed",
            exit_code=EXIT_ERROR,
        )

    return emit(
        args,
        _audit_summary_payload(report),
        human=_format_audit_summary(report),
    )


def _audit_summary_payload(report: Any) -> dict[str, Any]:
    span_counts = Counter(
        span.canonical_label or span.label or "UNKNOWN" for span in report.spans
    )
    action_counts = Counter(span.action or "unspecified" for span in report.spans)
    return {
        "policy": report.policy or None,
        "openmed_version": report.openmed_version or None,
        "document_length": report.document_length,
        "repro_hash_ok": report.repro_hash_matches(),
        "signature": "present" if report.signature is not None else "absent",
        "span_counts": dict(sorted(span_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "residual_risk": dict(report.residual_risk) if report.residual_risk else {},
    }


def _handle_compliance_safe_harbor(args: argparse.Namespace) -> int:
    from ..compliance.safe_harbor import generate_safe_harbor_attestation

    try:
        report = _load_audit_report(args.report)
        attestation = generate_safe_harbor_attestation(report)
    except (OSError, TypeError, ValueError) as exc:
        raise CliError(
            "Failed to generate Safe Harbor attestation.",
            code="attestation_failed",
            exit_code=EXIT_ERROR,
        ) from exc

    payload = attestation.to_dict()
    if args.output is None:
        return emit(args, payload, human=attestation.to_json())

    try:
        args.output.write_text(f"{attestation.to_json()}\n", encoding="utf-8")
    except OSError as exc:
        raise CliError(
            "Failed to write Safe Harbor attestation.",
            code="write_failed",
            exit_code=EXIT_ERROR,
        ) from exc
    return emit(
        args,
        {
            "output": str(args.output),
            "attestation_hash": attestation.attestation_hash,
        },
        human=f"Safe Harbor attestation written to: {args.output}",
    )


def _handle_expert_review_verify(args: argparse.Namespace) -> int:
    from ..compliance import ExpertReviewEvidenceReport

    try:
        report = ExpertReviewEvidenceReport.from_json(
            args.report.read_text(encoding="utf-8")
        )
    except (OSError, TypeError, ValueError) as exc:
        raise CliError(
            "Expert-review evidence verification failed.",
            code="evidence_verification_failed",
            exit_code=EXIT_ERROR,
        ) from exc
    payload = {
        "verified": report.verify(),
        "integrity_hash": report.integrity_hash,
        "schema_version": report.schema_version,
        "qualified_expert_review": "pending",
    }
    return emit(
        args,
        payload,
        human=(
            "Expert-review evidence verification: PASS\n"
            f"Integrity hash: {report.integrity_hash}\n"
            "Qualified expert review: pending\n"
        ),
    )


def _handle_expert_attestation_verify(args: argparse.Namespace) -> int:
    from ..compliance import (
        ExpertAttestationEnvelope,
        ExpertReviewEvidenceReport,
    )

    try:
        attestation = ExpertAttestationEnvelope.from_json(
            args.attestation.read_text(encoding="utf-8")
        )
        evidence = ExpertReviewEvidenceReport.from_json(
            args.evidence.read_text(encoding="utf-8")
        )
        public_key = args.public_key.read_bytes()
        verification = attestation.verify(
            evidence=evidence,
            public_key=public_key,
            expected_key_id=args.key_id,
            expected_supporting_evidence_digests=tuple(args.supporting_evidence),
        )
    except ImportError as exc:
        raise CliError(
            "Expert-attestation verification requires the 'integrity' extra.",
            code="expert_attestation_dependency_missing",
            exit_code=EXIT_ERROR,
        ) from exc
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        raise CliError(
            "Expert-attestation verification failed because an input is "
            "unreadable, malformed, or unsupported.",
            code="expert_attestation_verification_failed",
            exit_code=EXIT_ERROR,
        ) from exc

    payload = verification.to_dict()
    human = (
        "Expert-authored attestation verification\n"
        f"Cryptographic signature: "
        f"{_pass_fail(verification.cryptographically_valid)}\n"
        f"Trusted key identifier: {_pass_fail(verification.key_id_matches)}\n"
        f"Evidence integrity: "
        f"{_pass_fail(verification.evidence_integrity_valid)}\n"
        f"Evidence bindings: {_pass_fail(verification.bindings_match)}\n"
        f"Expert-stated conclusion: {verification.conclusion}\n"
        f"Freshness: {verification.freshness_status}\n"
        "These are independent verification facts, not an automated Expert "
        "Determination or release authorization.\n"
    )
    emitted = emit(args, payload, human=human)
    authenticity_valid = (
        verification.cryptographically_valid
        and verification.key_id_matches
        and verification.evidence_integrity_valid
        and verification.bindings_match
    )
    return emitted if authenticity_valid else EXIT_ERROR


def _load_audit_report(path: Path):
    from ..core.audit import AuditReport

    return AuditReport.from_json(path.read_text(encoding="utf-8"))


def _load_audit_artifact(path: Path):
    from ..core.audit import AuditReport
    from ..core.audit_chain import CHAIN_FORMAT, AuditChain

    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, MappingABC):
        raise ValueError("audit artifact JSON must contain an object")
    if parsed.get("format") == CHAIN_FORMAT or "entries" in parsed:
        return AuditChain.from_dict(parsed)
    return AuditReport.from_dict(parsed)


def _format_audit_summary(report: Any) -> str:
    span_counts = Counter(
        span.canonical_label or span.label or "UNKNOWN" for span in report.spans
    )
    action_counts = Counter(span.action or "unspecified" for span in report.spans)
    signature = "present" if report.signature is not None else "absent"

    lines = [
        "Audit report summary",
        f"Policy: {report.policy or '-'}",
        f"OpenMed version: {report.openmed_version or '-'}",
        f"Document length: {report.document_length}",
        f"Reproducibility hash: {_pass_fail(report.repro_hash_matches())}",
        f"Signature: {signature}",
        "Span counts by type:",
        *_format_count_lines(span_counts),
        "Policy actions:",
        *_format_count_lines(action_counts),
        "Residual risk:",
        *_format_residual_risk_lines(report.residual_risk),
    ]
    return "\n".join(lines) + "\n"


def _format_residual_risk_lines(residual_risk: Mapping[str, Any]) -> list[str]:
    if not residual_risk:
        return ["  none"]

    lines: list[str] = []
    projected = residual_risk.get("projected_leakage")
    if _is_number(projected):
        lines.append(f"  Projected leakage: {_format_number(projected)}")

    record_score = residual_risk.get("risk_report_record_score")
    if _is_number(record_score):
        lines.append(f"  Risk report record score: {_format_number(record_score)}")

    risk = residual_risk.get("risk_report")
    if isinstance(risk, MappingABC):
        lines.extend(f"  {line}" for line in _format_risk_summary_lines(risk))

    return lines or ["  summary unavailable"]


def _handle_risk_text(args: argparse.Namespace) -> int:
    from ..risk import risk_report, safe_risk_summary

    try:
        text = _read_text_input(args.input)
    except OSError as exc:
        raise CliError(
            f"Failed to read text input: {exc}",
            code="read_failed",
            exit_code=EXIT_ERROR,
        )

    report = risk_report(text)
    return emit(
        args,
        safe_risk_summary(report),
        human=_format_risk_summary("Text risk summary", report),
    )


def _handle_risk_table(args: argparse.Namespace) -> int:
    from ..risk import risk_report, safe_risk_summary

    try:
        records = _read_csv_records(args.csv)
    except (OSError, ValueError) as exc:
        raise CliError(
            f"Failed to read table input: {exc}",
            code="read_failed",
            exit_code=EXIT_ERROR,
        )

    report = risk_report(records)
    return emit(
        args,
        safe_risk_summary(report),
        human=_format_risk_summary("Table risk summary", report),
    )


def _handle_risk_discover(args: argparse.Namespace) -> int:
    from ..structured import (
        SUPPORTED_TABLE_SUFFIXES,
        DiscoveryConfigurationError,
        scan_table,
    )

    role_overrides = dict(args.role)
    if len(role_overrides) != len(args.role):
        raise CliError(
            "Each structured discovery column may have only one role override.",
            code="invalid_discovery_config",
            exit_code=EXIT_USAGE,
        )
    _preflight_structured_paths(
        inputs=((args.input, "Discovery input", SUPPORTED_TABLE_SUFFIXES),),
        outputs=((args.output, "Discovery output", frozenset({".json"})),),
        overwrite=args.overwrite,
    )
    try:
        manifest = scan_table(
            args.input,
            max_rows=args.sample_rows,
            max_set_size=args.max_set_size,
            max_candidate_columns=args.max_candidate_columns,
            search_budget=args.search_budget,
            full_scan=args.full_scan,
            role_overrides=role_overrides,
            quasi_identifier_columns=_merged_column_args(
                args.qi,
                args.qi_column,
            ),
            sensitive_columns=_merged_column_args(
                args.sensitive,
                args.sensitive_column,
            ),
            privacy_unit=args.privacy_unit,
            include_safe_candidates=args.include_safe_candidates,
        )
        _write_safe_text(
            args.output,
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            overwrite=args.overwrite,
        )
    except DiscoveryConfigurationError as exc:
        raise CliError(
            "The structured discovery configuration does not match the input schema.",
            code="invalid_discovery_config",
            exit_code=EXIT_USAGE,
        ) from exc
    except (ImportError, OSError, TypeError, ValueError) as exc:
        raise CliError(
            "Failed to discover structured quasi-identifiers.",
            code="qi_discovery_failed",
            exit_code=EXIT_ERROR,
        ) from exc

    discovery = manifest["discovery"]
    payload = {
        "output": str(args.output),
        "status": discovery["status"],
        "advisory": discovery["advisory"],
        "candidate_set_count": len(manifest["quasi_identifier_sets"]),
        "search": manifest["search"],
    }
    human = (
        "Quasi-identifier discovery complete\n"
        f"Status: {discovery['status']}\n"
        f"Advisory: {str(discovery['advisory']).lower()}\n"
        f"Candidate sets: {len(manifest['quasi_identifier_sets'])}\n"
        f"Manifest: {args.output}\n"
    )
    return emit(args, payload, human=human)


def _handle_risk_assess(args: argparse.Namespace) -> int:
    from ..risk import assess_release, render_release_assessment_dashboard
    from ..structured import SUPPORTED_TABLE_SUFFIXES, read_table

    policy = _validated_release_policy(args)
    outputs = [(args.output, "Assessment output", frozenset({".json"}))]
    if args.dashboard is not None:
        outputs.append((args.dashboard, "Assessment dashboard", frozenset({".html"})))
    _preflight_structured_paths(
        inputs=((args.input, "Assessment input", SUPPORTED_TABLE_SUFFIXES),),
        outputs=tuple(outputs),
        overwrite=args.overwrite,
    )
    try:
        records = read_table(args.input)
    except (ImportError, OSError, TypeError, ValueError) as exc:
        raise CliError(
            "Failed to read the structured assessment input.",
            code="release_input_failed",
            exit_code=EXIT_ERROR,
        ) from exc
    try:
        assessment = assess_release(records, policy)
    except (TypeError, ValueError) as exc:
        raise CliError(
            _release_error_message(
                "The structured release policy does not match the input schema.",
                exc,
            ),
            code="invalid_release_config",
            exit_code=EXIT_USAGE,
        ) from exc
    staged_paths: list[Path] = []
    try:
        staged_assessment = _temporary_sibling_path(args.output)
        staged_paths.append(staged_assessment)
        publications = [(staged_assessment, args.output)]
        _write_safe_text(
            staged_assessment,
            assessment.to_json() + "\n",
            overwrite=False,
        )
        if args.dashboard is not None:
            staged_dashboard = _temporary_sibling_path(args.dashboard)
            staged_paths.append(staged_dashboard)
            _write_safe_text(
                staged_dashboard,
                render_release_assessment_dashboard(assessment),
                overwrite=False,
            )
            publications.append((staged_dashboard, args.dashboard))
        _publish_release_outputs(publications, overwrite=args.overwrite)
    except (ImportError, OSError, TypeError, ValueError) as exc:
        raise CliError(
            "Failed to assess structured release risk.",
            code="release_assessment_failed",
            exit_code=EXIT_ERROR,
        ) from exc
    finally:
        for staged_path in staged_paths:
            _unlink_path(staged_path, missing_ok=True)
    if not assessment.meets_policy:
        raise CliError(
            "Structured release does not meet the configured privacy policy; "
            f"the aggregate assessment was written to {args.output}.",
            code="release_policy_failed",
            exit_code=EXIT_ERROR,
        )

    payload = {
        "output": str(args.output),
        "meets_policy": assessment.meets_policy,
        "achieved_k": assessment.achieved_k,
        "privacy_unit_count": assessment.privacy_unit_count,
        "policy_digest": assessment.policy_digest,
        "dataset_digest": assessment.dataset_digest,
        "dashboard": str(args.dashboard) if args.dashboard is not None else None,
    }
    human = (
        "Structured release risk assessment\n"
        f"Meets configured policy: {_pass_fail(assessment.meets_policy)}\n"
        f"Achieved k: {assessment.achieved_k}\n"
        f"Privacy units: {assessment.privacy_unit_count}\n"
        f"Assessment: {args.output}\n"
        + (f"Dashboard: {args.dashboard}\n" if args.dashboard is not None else "")
        + "This is not an Expert Determination; qualified expert review is required.\n"
    )
    return emit(args, payload, human=human)


def _handle_risk_population_assess(args: argparse.Namespace) -> int:
    from ..risk import assess_population_risk
    from ..structured import SUPPORTED_TABLE_SUFFIXES, read_table

    quasi_identifiers = _merged_column_args(args.qi, args.qi_column)
    if not quasi_identifiers:
        raise CliError(
            "Population-risk assessment requires at least one --qi or --qi-column.",
            code="invalid_population_risk_config",
            exit_code=EXIT_USAGE,
        )
    _preflight_structured_paths(
        inputs=(
            (args.input, "Sample input", SUPPORTED_TABLE_SUFFIXES),
            (
                args.reference_population,
                "Reference-population input",
                SUPPORTED_TABLE_SUFFIXES,
            ),
        ),
        outputs=((args.output, "Population-risk output", frozenset({".json"})),),
        overwrite=args.overwrite,
    )
    try:
        sample = read_table(args.input)
        reference_population = read_table(args.reference_population)
    except (ImportError, OSError, TypeError, ValueError) as exc:
        raise CliError(
            "Failed to read a population-risk input table.",
            code="population_risk_input_failed",
            exit_code=EXIT_ERROR,
        ) from exc
    try:
        assessment = assess_population_risk(
            sample,
            reference_population,
            quasi_identifiers,
            sample_privacy_unit=args.sample_privacy_unit,
            population_privacy_unit=args.population_privacy_unit,
            target_k_map=args.k_map,
            max_delta_presence=args.max_delta_presence,
        )
    except (TypeError, ValueError) as exc:
        raise CliError(
            "The population-risk configuration or reference model is invalid.",
            code="invalid_population_risk_config",
            exit_code=EXIT_USAGE,
        ) from exc
    try:
        _write_safe_text(
            args.output,
            assessment.to_json() + "\n",
            overwrite=args.overwrite,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise CliError(
            "Failed to write the aggregate population-risk assessment.",
            code="population_risk_output_failed",
            exit_code=EXIT_ERROR,
        ) from exc
    if not assessment.meets_policy:
        raise CliError(
            "The sample does not meet the configured reference-population risk "
            f"policy; aggregate evidence was written to {args.output}.",
            code="population_risk_policy_failed",
            exit_code=EXIT_ERROR,
        )

    payload = {
        "output": str(args.output),
        "meets_policy": assessment.meets_policy,
        "achieved_k_map": assessment.achieved_k_map,
        "max_delta_presence": assessment.max_delta_presence,
        "matched_sample_unit_count": assessment.matched_sample_unit_count,
        "unmatched_sample_unit_count": assessment.unmatched_sample_unit_count,
        "reference_model_consistent": assessment.reference_model_consistent,
        "assessment_digest": assessment.digest,
    }
    human = (
        "Reference-population risk assessment\n"
        f"Meets configured policy: {_pass_fail(assessment.meets_policy)}\n"
        f"Achieved k-map: {assessment.achieved_k_map}\n"
        f"Maximum delta-presence: {assessment.max_delta_presence:.6g}\n"
        f"Assessment: {args.output}\n"
        "The supplied reference model and assumptions require qualified expert "
        "review.\n"
    )
    return emit(args, payload, human=human)


def _handle_risk_gate(args: argparse.Namespace) -> int:
    from ..compliance import ExpertReviewEvidenceReport
    from ..eval import evaluate_release_risk_evidence

    try:
        evidence_text = args.evidence.read_text(encoding="utf-8")
        evidence = ExpertReviewEvidenceReport.from_json(evidence_text)
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        raise CliError(
            "Release-risk evidence is unreadable, malformed, or fails its "
            "structural integrity check.",
            code="invalid_release_risk_evidence",
            exit_code=EXIT_ERROR,
        ) from exc
    check = evaluate_release_risk_evidence(evidence)
    payload = check.to_dict()
    human = (
        "Structured release evidence gate\n"
        f"Technical policy: {_pass_fail(check.passed)}\n"
        f"Reason: {check.reason}\n"
        "A passing technical gate is not an Expert Determination or release "
        "authorization.\n"
    )
    emitted = emit(args, payload, human=human)
    return emitted if check.passed else EXIT_ERROR


def _handle_risk_anonymize(args: argparse.Namespace) -> int:
    from ..compliance import (
        ReleaseAssumptions,
        build_release_expert_review_evidence,
    )
    from ..core.audit import stable_hash
    from ..risk import (
        anonymize_release,
        assess_release,
        render_release_assessment_dashboard,
        validate_released_output,
    )
    from ..structured import SUPPORTED_TABLE_SUFFIXES, read_table, write_table

    markdown_path = args.evidence_markdown or args.evidence.with_suffix(".md")
    policy = _validated_release_policy(args)
    other_context_selected = (
        args.privacy_unit_kind == "other"
        or args.population_scope == "other_documented"
        or args.release_model == "other_documented"
        or args.recipient_model == "other_documented"
        or args.auxiliary_data_model == "other_documented"
    )
    if other_context_selected and args.assumptions_notes is None:
        raise CliError(
            "Documented release-context choices require --assumptions-notes.",
            code="invalid_release_assumptions",
            exit_code=EXIT_USAGE,
        )
    input_paths = [
        (args.input, "Release input", SUPPORTED_TABLE_SUFFIXES),
    ]
    if args.hierarchies is not None:
        input_paths.append(
            (args.hierarchies, "Hierarchy configuration", frozenset({".json"}))
        )
    if args.assumptions_notes is not None:
        input_paths.append(
            (
                args.assumptions_notes,
                "Assumptions notes",
                frozenset({".md", ".txt"}),
            )
        )
    output_paths = [
        (args.output, "Release output", SUPPORTED_TABLE_SUFFIXES),
        (args.evidence, "Evidence output", frozenset({".json"})),
        (markdown_path, "Evidence Markdown output", frozenset({".md"})),
    ]
    if args.dashboard is not None:
        output_paths.append((args.dashboard, "Release dashboard", frozenset({".html"})))
    _preflight_structured_paths(
        inputs=tuple(input_paths),
        outputs=tuple(output_paths),
        overwrite=args.overwrite,
    )
    hierarchies = _validated_hierarchy_config(
        args.hierarchies,
        quasi_identifiers=policy.quasi_identifiers,
    )
    privacy_unit_kind = _validated_privacy_unit_kind(args)
    assumptions_values = {
        "privacy_unit": privacy_unit_kind,
        "population_scope": args.population_scope,
        "release_model": args.release_model,
        "recipient_model": args.recipient_model,
        "auxiliary_data_model": args.auxiliary_data_model,
    }
    assumptions_notes: str | None = None
    if args.assumptions_notes is not None:
        try:
            assumptions_notes = args.assumptions_notes.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise CliError(
                "Failed to read the reviewed assumptions notes.",
                code="assumptions_notes_read_failed",
                exit_code=EXIT_ERROR,
            ) from exc
        if not assumptions_notes.strip():
            raise CliError(
                "Reviewed assumptions notes must not be empty.",
                code="invalid_release_assumptions",
                exit_code=EXIT_USAGE,
            )
    staged_paths: list[Path] = []
    try:
        records = read_table(args.input)
    except (ImportError, OSError, TypeError, ValueError) as exc:
        raise CliError(
            "Failed to read the structured release input.",
            code="release_input_failed",
            exit_code=EXIT_ERROR,
        ) from exc
    try:
        assess_release(records, policy)
    except (TypeError, ValueError) as exc:
        raise CliError(
            _release_error_message(
                "The structured release policy does not match the input schema.",
                exc,
            ),
            code="invalid_release_config",
            exit_code=EXIT_USAGE,
        ) from exc
    try:
        result = anonymize_release(records, policy, hierarchies=hierarchies)
        staged_output = _temporary_sibling_path(args.output)
        staged_evidence = _temporary_sibling_path(args.evidence)
        staged_markdown = _temporary_sibling_path(markdown_path)
        staged_paths.extend((staged_output, staged_evidence, staged_markdown))
        write_table(staged_output, result.records)
        reread = read_table(staged_output)
        preserves_types = args.output.suffix.lower() not in {".csv", ".tsv"}
        validation = validate_released_output(
            reread,
            result,
            preserve_scalar_types=preserves_types,
        )
        if not validation.passed:
            raise ValueError("materialized release failed residual validation")

        assumptions = ReleaseAssumptions(
            **assumptions_values,
            notes_digest=stable_hash(
                {
                    "kind": "openmed-release-assumptions-binding",
                    "coded_assumptions": assumptions_values,
                    "detailed_notes": assumptions_notes,
                }
            ),
        )
        evidence = build_release_expert_review_evidence(
            result,
            validation=validation,
            assumptions=assumptions,
        )
        _write_safe_text(
            staged_evidence,
            evidence.to_json() + "\n",
            overwrite=False,
        )
        _write_safe_text(
            staged_markdown,
            evidence.to_markdown(),
            overwrite=False,
        )
        publications = [
            (staged_evidence, args.evidence),
            (staged_markdown, markdown_path),
        ]
        if args.dashboard is not None:
            staged_dashboard = _temporary_sibling_path(args.dashboard)
            staged_paths.append(staged_dashboard)
            _write_safe_text(
                staged_dashboard,
                render_release_assessment_dashboard(result),
                overwrite=False,
            )
            publications.append((staged_dashboard, args.dashboard))
        # The sensitive release is published only after materialized
        # validation and every safe evidence artifact has been staged.
        publications.append((staged_output, args.output))
        _publish_release_outputs(
            publications,
            overwrite=args.overwrite,
        )
    except _ReleasePublicationCleanupError as exc:
        raise CliError(
            "Structured release outputs were published, but secure backup "
            "cleanup failed.",
            code="release_backup_cleanup_failed",
            exit_code=EXIT_ERROR,
        ) from exc
    except (TypeError, ValueError) as exc:
        raise CliError(
            _release_error_message(
                "Failed to anonymize and validate the structured release.",
                exc,
            ),
            code="release_anonymization_failed",
            exit_code=EXIT_ERROR,
        ) from exc
    except (ImportError, OSError) as exc:
        raise CliError(
            "Failed to anonymize and validate the structured release.",
            code="release_anonymization_failed",
            exit_code=EXIT_ERROR,
        ) from exc
    finally:
        for staged_path in staged_paths:
            _unlink_path(staged_path, missing_ok=True)

    payload = {
        "output": str(args.output),
        "evidence": str(args.evidence),
        "evidence_markdown": str(markdown_path),
        "dashboard": str(args.dashboard) if args.dashboard is not None else None,
        "achieved_k": result.after.achieved_k,
        "meets_policy": result.after.meets_policy,
        "released_rows": result.utility.released_rows,
        "released_privacy_units": result.utility.released_privacy_units,
        "released_dataset_digest": validation.dataset_digest,
        "evidence_integrity_hash": evidence.integrity_hash,
        "output_validation": validation.to_dict(),
    }
    human = (
        "Structured release anonymization complete\n"
        f"Meets configured policy: {_pass_fail(result.after.meets_policy)}\n"
        f"Achieved k: {result.after.achieved_k}\n"
        f"Released rows: {result.utility.released_rows}\n"
        f"Release: {args.output}\n"
        f"Expert-review evidence: {args.evidence}\n"
        f"Evidence Markdown: {markdown_path}\n"
        + (f"Dashboard: {args.dashboard}\n" if args.dashboard is not None else "")
        + "The evidence is not an Expert Determination; qualified expert review "
        + "is required.\n"
    )
    return emit(args, payload, human=human)


def _release_policy_from_args(args: argparse.Namespace):
    from ..risk import AnonymityPolicy

    return AnonymityPolicy(
        quasi_identifiers=_merged_column_args(args.qi, args.qi_column),
        sensitive_attributes=_merged_column_args(
            args.sensitive,
            args.sensitive_column,
        ),
        direct_identifiers=_merged_column_args(
            args.direct_id,
            args.direct_id_column,
        ),
        non_sensitive_attributes=_merged_column_args(
            args.non_sensitive,
            args.non_sensitive_column,
        ),
        excluded_attributes=_merged_column_args(
            args.exclude,
            args.exclude_column,
        ),
        privacy_unit=args.privacy_unit,
        target_k=args.k,
        target_l=args.l,
        l_metric=args.l_metric,
        target_t=args.t,
        suppression_limit=getattr(args, "max_suppressed_units", None),
        suppression_rate=getattr(args, "max_suppression_rate", 0.0),
        max_lattice_nodes=getattr(args, "max_lattice_nodes", 100_000),
        max_suppression_subsets=getattr(
            args,
            "max_suppression_subsets",
            100_000,
        ),
    )


def _validated_release_policy(args: argparse.Namespace):
    try:
        return _release_policy_from_args(args)
    except (TypeError, ValueError) as exc:
        raise CliError(
            _release_error_message(
                "The structured release policy is invalid.",
                exc,
            ),
            code="invalid_release_policy",
            exit_code=EXIT_USAGE,
        ) from exc


def _validated_privacy_unit_kind(args: argparse.Namespace) -> str:
    privacy_unit = args.privacy_unit
    kind = args.privacy_unit_kind
    if privacy_unit is None:
        if kind in (None, "row"):
            return "row"
        raise CliError(
            "A non-row privacy-unit kind requires --privacy-unit.",
            code="invalid_privacy_unit_kind",
            exit_code=EXIT_USAGE,
        )
    if kind is None or kind == "row":
        raise CliError(
            "A named --privacy-unit requires an explicit non-row --privacy-unit-kind.",
            code="invalid_privacy_unit_kind",
            exit_code=EXIT_USAGE,
        )
    return str(kind)


def _load_hierarchies(
    path: Path | None,
) -> Mapping[str, Sequence[Mapping[str, Any]]] | None:
    if path is None:
        return None
    from ..structured.table_io import _strict_json_loads

    payload = _strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, MappingABC):
        raise ValueError("Hierarchy JSON must contain an object")
    return payload


def _validated_hierarchy_config(
    path: Path | None,
    *,
    quasi_identifiers: Sequence[str],
) -> Mapping[str, Sequence[Mapping[str, Any]]] | None:
    if path is None:
        return None
    try:
        payload = _load_hierarchies(path)
        if payload is None:
            return None
        _validate_hierarchy_shape(payload, quasi_identifiers=quasi_identifiers)
        return payload
    except (OSError, TypeError, ValueError) as exc:
        raise CliError(
            "The hierarchy configuration is invalid.",
            code="invalid_hierarchy_config",
            exit_code=EXIT_USAGE,
        ) from exc


def _validate_hierarchy_shape(
    payload: Mapping[str, Any],
    *,
    quasi_identifiers: Sequence[str],
) -> None:
    from ..risk.kanon import _INTERNAL_QI_TOKEN_PREFIX, _user_hierarchy

    declared = set(quasi_identifiers)
    unknown = [field for field in payload if not isinstance(field, str)]
    unknown.extend(
        field for field in payload if isinstance(field, str) and field not in declared
    )
    if unknown:
        raise ValueError("hierarchy fields must be declared quasi-identifiers")

    allowed_keys = {"name", "values", "default", "loss"}
    for field, raw_levels in payload.items():
        if (
            not isinstance(raw_levels, Sequence)
            or isinstance(raw_levels, (str, bytes, bytearray))
            or not raw_levels
        ):
            raise ValueError(f"hierarchy {field!r} must contain levels")
        previous_loss = -1.0
        max_index = max(1, len(raw_levels) - 1)
        for index, level in enumerate(raw_levels):
            if not isinstance(level, MappingABC):
                raise TypeError("hierarchy levels must be objects")
            if set(level) - allowed_keys:
                raise ValueError("hierarchy level contains unsupported keys")
            values = level.get("values")
            if values is not None and not isinstance(values, MappingABC):
                raise TypeError("hierarchy level values must be an object")
            if (
                isinstance(values, MappingABC)
                and any(
                    str(value).startswith(_INTERNAL_QI_TOKEN_PREFIX)
                    for value in values.values()
                )
            ) or (
                level.get("default") is not None
                and str(level["default"]).startswith(_INTERNAL_QI_TOKEN_PREFIX)
            ):
                raise ValueError(
                    "hierarchy outputs cannot use the reserved internal namespace"
                )
            if index == 0 and (values is not None or "default" in level):
                raise ValueError(
                    "hierarchy level zero must be a canonical identity level"
                )
            loss = level.get("loss", index / max_index)
            if (
                not isinstance(loss, (int, float))
                or isinstance(loss, bool)
                or not math.isfinite(float(loss))
                or not 0.0 <= float(loss) <= 1.0
                or float(loss) < previous_loss
            ):
                raise ValueError(
                    "hierarchy losses must be finite, bounded, and non-decreasing"
                )
            if index == 0 and float(loss) != 0.0:
                raise ValueError("hierarchy identity loss must be zero")
            if index > 0 and float(loss) <= 0.0:
                raise ValueError("hierarchy coarsening loss must be positive")
            previous_loss = float(loss)
        _user_hierarchy(field, raw_levels)


def _preflight_structured_paths(
    *,
    inputs: Sequence[tuple[Path, str, frozenset[str]]],
    outputs: Sequence[tuple[Path, str, frozenset[str]]],
    overwrite: bool,
) -> None:
    all_paths = [path for path, _label, _suffixes in (*inputs, *outputs)]
    _ensure_resolved_paths_distinct(all_paths)
    for path, label, suffixes in inputs:
        _validate_structured_input(path, label=label, suffixes=suffixes)
    for path, label, suffixes in outputs:
        _validate_structured_output(
            path,
            label=label,
            suffixes=suffixes,
            overwrite=overwrite,
        )


def _ensure_resolved_paths_distinct(paths: Sequence[Path]) -> None:
    try:
        resolved = [path.resolve(strict=False) for path in paths]
    except (OSError, RuntimeError) as exc:
        raise CliError(
            "An input or output path could not be resolved.",
            code="invalid_path",
            exit_code=EXIT_USAGE,
        ) from exc
    if len(set(resolved)) != len(resolved):
        raise CliError(
            "All input and output paths must be distinct after resolving symlinks.",
            code="path_alias",
            exit_code=EXIT_USAGE,
        )
    for index, path in enumerate(paths):
        if not _path_entry_exists(path):
            continue
        for other in paths[index + 1 :]:
            if not _path_entry_exists(other):
                continue
            try:
                aliases = os.path.samefile(path, other)
            except OSError:
                aliases = False
            if aliases:
                raise CliError(
                    "All input and output paths must be distinct after resolving "
                    "symlinks.",
                    code="path_alias",
                    exit_code=EXIT_USAGE,
                )


def _validate_structured_input(
    path: Path,
    *,
    label: str,
    suffixes: frozenset[str],
) -> None:
    _validate_structured_suffix(path, label=label, suffixes=suffixes)
    if not path.exists() or not path.is_file():
        raise CliError(
            f"{label} must be an existing file.",
            code="input_path_invalid",
            exit_code=EXIT_USAGE,
        )


def _validate_structured_output(
    path: Path,
    *,
    label: str,
    suffixes: frozenset[str],
    overwrite: bool,
) -> None:
    _validate_structured_suffix(path, label=label, suffixes=suffixes)
    if not path.parent.exists() or not path.parent.is_dir():
        raise CliError(
            f"{label} directory must already exist.",
            code="output_directory_invalid",
            exit_code=EXIT_USAGE,
        )
    if path.exists() and path.is_dir():
        raise CliError(
            f"{label} must be a file path.",
            code="output_path_invalid",
            exit_code=EXIT_USAGE,
        )
    if _path_entry_exists(path) and not overwrite:
        raise CliError(
            f"{label} already exists; use --overwrite to replace it.",
            code="output_exists",
            exit_code=EXIT_USAGE,
        )


def _validate_structured_suffix(
    path: Path,
    *,
    label: str,
    suffixes: frozenset[str],
) -> None:
    if path.suffix.lower() not in suffixes:
        raise CliError(
            f"{label} has an unsupported file suffix.",
            code="unsupported_suffix",
            exit_code=EXIT_USAGE,
        )


def _path_entry_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _publish_release_outputs(
    publications: Sequence[tuple[Path, Path]],
    *,
    overwrite: bool,
) -> None:
    targets = [target for _staged, target in publications]
    had_original = {target for target in targets if _path_entry_exists(target)}
    if had_original and not overwrite:
        raise FileExistsError("Release output appeared during publication")
    backups = {
        target: _temporary_sibling_path(target)
        for target in targets
        if target in had_original
    }

    try:
        for target, backup in backups.items():
            os.replace(target, backup)
        for staged, target in publications:
            os.replace(staged, target)
    except Exception as exc:
        rollback_errors = _rollback_release_outputs(
            targets,
            had_original=had_original,
            backups=backups,
        )
        if rollback_errors:
            raise OSError("Release output rollback failed") from exc
        raise
    else:
        cleanup_errors: list[OSError] = []
        for backup in backups.values():
            try:
                _unlink_path(backup, missing_ok=True)
            except OSError as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            raise _ReleasePublicationCleanupError(
                "published release backup cleanup failed"
            )


def _rollback_release_outputs(
    targets: Sequence[Path],
    *,
    had_original: set[Path],
    backups: Mapping[Path, Path],
) -> list[OSError]:
    errors: list[OSError] = []
    for target in reversed(targets):
        backup = backups.get(target)
        try:
            if backup is not None and _path_entry_exists(backup):
                if _path_entry_exists(target):
                    _unlink_path(target, missing_ok=False)
                os.rename(backup, target)
            elif target not in had_original and _path_entry_exists(target):
                _unlink_path(target, missing_ok=False)
        except OSError as exc:
            errors.append(exc)
    return errors


def _temporary_sibling_path(path: Path) -> Path:
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=path.suffix,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    _unlink_path(temporary, missing_ok=False)
    return temporary


def _unlink_path(path: Path, *, missing_ok: bool) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        if not missing_ok:
            raise


def _write_safe_text(path: Path, text: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
    if not path.parent.exists():
        raise FileNotFoundError(f"Output directory does not exist: {path.parent}")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(text)
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _read_text_input(value: str) -> str:
    path = Path(value)
    if path.exists():
        if not path.is_file():
            raise OSError(f"not a file: {path}")
        return path.read_text(encoding="utf-8")
    return value


def _read_csv_records(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV input must include a header row")
        return [dict(row) for row in reader]


def _format_risk_summary(title: str, report: Mapping[str, Any]) -> str:
    return "\n".join([title, *_format_risk_summary_lines(report)]) + "\n"


def _format_risk_summary_lines(report: Mapping[str, Any]) -> list[str]:
    quasi_identifiers = _mapping_items(report.get("quasi_identifiers"))
    singleton_records = _mapping_items(report.get("singleton_records"))
    category_counts = Counter(
        str(item.get("category") or "unknown") for item in quasi_identifiers
    )

    lines = [
        f"Leakage rate: {_format_number(report.get('leakage_rate'))}",
        f"Re-identification rate: {_format_number(report.get('reid_rate'))}",
        f"Minimum k: {report.get('k_min', 0)}",
        f"Singleton records: {len(singleton_records)}",
        f"Quasi-identifiers: {len(quasi_identifiers)}",
    ]
    if category_counts:
        lines.append("Quasi-identifier categories:")
        lines.extend(_format_count_lines(category_counts))
    return lines


def _mapping_items(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, MappingABC)]


def _format_count_lines(counts: Counter[str]) -> list[str]:
    if not counts:
        return ["  none"]
    return [f"  {name}: {count}" for name, count in sorted(counts.items())]


def _format_number(value: Any) -> str:
    if _is_number(value):
        return f"{float(value):.3f}"
    return "n/a"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _pass_fail(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _handle_fhir_bundle(args: argparse.Namespace) -> int:
    if args.bundle_type not in _FHIR_BUNDLE_TYPES:
        allowed = ", ".join(sorted(_FHIR_BUNDLE_TYPES))
        raise CliError(
            f"--type must be one of: {allowed}",
            code="invalid_argument",
            exit_code=EXIT_USAGE,
        )

    try:
        source = json.loads(args.input.read_text(encoding="utf-8"))
        resources = _extract_fhir_resources(source)
        doc_id = _extract_fhir_doc_id(source)

        from ..clinical.exporters.fhir import to_bundle

        bundle = to_bundle(
            resources,
            doc_id=doc_id,
            bundle_type=args.bundle_type,
        )
        args.output.write_text(
            json.dumps(bundle, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except FileNotFoundError:
        raise CliError(
            f"Input file not found: {args.input}",
            code="input_not_found",
            exit_code=EXIT_ERROR,
        )
    except json.JSONDecodeError as exc:
        raise CliError(
            f"Invalid JSON in {args.input}: {exc.msg} "
            f"at line {exc.lineno} column {exc.colno}",
            code="invalid_json",
            exit_code=EXIT_ERROR,
        )
    except OSError as exc:
        raise CliError(
            f"Failed to read or write FHIR Bundle: {exc}",
            code="io_error",
            exit_code=EXIT_ERROR,
        )
    except (TypeError, ValueError) as exc:
        raise CliError(
            f"Failed to assemble FHIR Bundle: {exc}",
            code="assemble_failed",
            exit_code=EXIT_ERROR,
        )

    payload = {
        "output": str(args.output),
        "bundle_type": args.bundle_type,
        "doc_id": doc_id,
        "resource_count": len(resources),
    }
    return emit(args, payload, human=f"FHIR Bundle written to: {args.output}")


def _handle_icd11_build_snapshot(args: argparse.Namespace) -> int:
    """Build one local ICD-11 snapshot using environment-held credentials."""
    from ..interop.icd11_api import (
        CLIENT_ID_ENV,
        CLIENT_SECRET_ENV,
        ICD11APIClient,
        ICD11APIError,
        build_snapshot,
    )

    client_id = os.environ.get(CLIENT_ID_ENV, "")
    client_secret = os.environ.get(CLIENT_SECRET_ENV, "")
    if not client_id or not client_secret:
        sys.stderr.write(
            f"Set {CLIENT_ID_ENV} and {CLIENT_SECRET_ENV} before building a snapshot.\n"
        )
        return 2

    try:
        client = ICD11APIClient(
            client_id,
            client_secret,
            language=args.language,
            timeout=args.timeout,
        )
        result = build_snapshot(
            client,
            release=args.release,
            chapters=args.chapters,
            cache_dir=args.cache_dir,
        )
    except (ICD11APIError, OSError, ValueError) as exc:
        sys.stderr.write(f"Failed to build ICD-11 snapshot: {exc}\n")
        return 1

    sys.stdout.write(
        json.dumps(
            {
                "entity_count": result.entity_count,
                "manifest_path": str(result.manifest_path),
                "snapshot_path": str(result.snapshot_path),
                "snapshot_sha256": result.snapshot_sha256,
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


def _handle_omop_load(args: argparse.Namespace) -> int:
    """Load a grounded-results JSONL file into a local OMOP CDM target."""
    from ..interop.omop import (
        load_grounded_jsonl,
        validate_omop_tables,
        write_omop_duckdb,
        write_omop_parquet,
        write_omop_sqlite,
    )

    try:
        tables = load_grounded_jsonl(
            args.input,
            vocabulary_version=args.vocabulary_version,
        )
    except FileNotFoundError:
        raise CliError(
            f"Input file not found: {args.input}",
            code="input_not_found",
            exit_code=EXIT_ERROR,
        )
    except json.JSONDecodeError as exc:
        raise CliError(
            f"Invalid JSON in {args.input}: {exc.msg}",
            code="invalid_json",
            exit_code=EXIT_ERROR,
        )
    except (OSError, ValueError) as exc:
        raise CliError(
            f"Failed to load grounded notes: {exc}",
            code="load_failed",
            exit_code=EXIT_ERROR,
        )

    if args.target is not None:
        writers = {
            "duckdb": write_omop_duckdb,
            "sqlite": write_omop_sqlite,
            "parquet": write_omop_parquet,
        }
        try:
            connection = writers[args.writer](tables, str(args.target))
            if hasattr(connection, "close"):
                connection.close()
        except ImportError as exc:
            raise CliError(
                str(exc),
                code="missing_dependency",
                exit_code=EXIT_ERROR,
            )
        except (OSError, ValueError) as exc:
            raise CliError(
                f"Failed to write OMOP target: {exc}",
                code="write_failed",
                exit_code=EXIT_ERROR,
            )

    summary = tables.summary
    payload: dict[str, Any] = {
        "input": str(args.input),
        "target": str(args.target) if args.target is not None else None,
        "writer": args.writer if args.target is not None else None,
        "vocabulary_version": args.vocabulary_version,
        "row_counts": dict(summary.row_counts),
        "rejection_counts": dict(summary.rejection_counts),
        "rejected_spans": [span.to_dict() for span in summary.rejected_spans],
        "source_note_hashes": list(summary.source_note_hashes),
    }

    if args.validate:
        violations = validate_omop_tables(tables)
        by_reason: dict[str, int] = {}
        for violation in violations:
            by_reason[violation.reason] = by_reason.get(violation.reason, 0) + 1
        payload["constraint_violations"] = {
            "count": len(violations),
            "by_reason": by_reason,
        }

    counts = ", ".join(
        f"{table}={count}" for table, count in payload["row_counts"].items() if count
    )
    rejected_total = sum(payload["rejection_counts"].values())
    human = f"Loaded {args.input} -> {counts} ({rejected_total} rejected span(s))"
    return emit(args, payload, human=human)


def _handle_ground(args: argparse.Namespace) -> int:
    """Run the canonical offline grounding facade."""

    from ..clinical.grounding import RankingConfig, VocabLoader, ground_payload
    from ..clinical.grounding.vocab import RestrictedVocabularyError, VocabLoaderError
    from ..core.offline import OfflineModeError

    systems = tuple(args.systems or ("rxnorm", "icd10cm", "loinc", "hpo"))
    inputs: Any
    if args.text is not None:
        inputs = args.text
    else:
        try:
            raw = args.input.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise CliError(
                "Grounding input file was not found.",
                code="input_not_found",
                exit_code=EXIT_ERROR,
            ) from exc
        except OSError as exc:
            raise CliError(
                "Grounding input file could not be read.",
                code="input_read_failed",
                exit_code=EXIT_ERROR,
            ) from exc
        try:
            if args.input.suffix.casefold() == ".jsonl":
                inputs = [json.loads(line) for line in raw.splitlines() if line.strip()]
            else:
                inputs = json.loads(raw)
        except json.JSONDecodeError:
            inputs = raw
        if isinstance(inputs, MappingABC) and "entities" in inputs:
            inputs = inputs["entities"]
        elif isinstance(inputs, MappingABC) and "text" in inputs:
            inputs = inputs["text"]

    loader = VocabLoader(cache_dir=args.cache_dir, local_only=args.offline)
    try:
        payload = ground_payload(
            inputs,
            systems=systems,
            loader=loader,
            config=RankingConfig(k=args.top_k),
            source_language=args.source_language,
            offline=args.offline,
        )
    except RestrictedVocabularyError as exc:
        raise CliError(
            "Restricted terminology requires a configured user-supplied "
            "out-of-process endpoint.",
            code="restricted_terminology_unconfigured",
            exit_code=EXIT_ERROR,
        ) from exc
    except OfflineModeError as exc:
        raise CliError(
            "The requested vocabulary snapshot is unavailable offline.",
            code="offline_snapshot_unavailable",
            exit_code=EXIT_ERROR,
        ) from exc
    except VocabLoaderError as exc:
        raise CliError(
            "The requested vocabulary snapshot could not be verified.",
            code="snapshot_invalid",
            exit_code=EXIT_ERROR,
        ) from exc
    except (TypeError, ValueError) as exc:
        raise CliError(
            "The grounding request is invalid.",
            code="grounding_invalid_request",
            exit_code=EXIT_ERROR,
        ) from exc

    human = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
    return emit(args, payload, human=human)


def _handle_grounding_snapshot_import(args: argparse.Namespace) -> int:
    """Import one local snapshot and emit its manifest."""

    from ..clinical.grounding import VocabLoader
    from ..clinical.grounding.vocab import VocabLoaderError

    loader = VocabLoader(cache_dir=args.cache_dir, local_only=True)
    try:
        manifest = loader.import_snapshot(
            args.system,
            args.input,
            version=args.version,
            sha256=args.sha256,
            license_note=args.license_note,
            replace=args.replace,
        )
    except (OSError, TypeError, ValueError, VocabLoaderError) as exc:
        raise CliError(
            "The vocabulary snapshot could not be imported.",
            code="snapshot_import_failed",
            exit_code=EXIT_ERROR,
        ) from exc
    payload = manifest.to_dict()
    return emit(
        args,
        payload,
        human=f"Imported {manifest.system} snapshot {manifest.version}.",
    )


def _handle_grounding_snapshot_download(args: argparse.Namespace) -> int:
    """Download one explicitly configured and checksum-pinned snapshot."""

    from ..clinical.grounding import VocabLoader, VocabSource
    from ..clinical.grounding.vocab import VocabLoaderError
    from ..core.offline import OfflineModeError

    source = VocabSource(
        system=args.system,
        url=args.url,
        sha256=args.sha256,
        checksum_url=args.checksum_url,
        artifact_name=args.artifact_name,
        archive_member=args.archive_member,
        version=args.version,
    )
    loader = VocabLoader(
        cache_dir=args.cache_dir,
        local_only=False,
        registry={args.system: source},
        timeout=args.timeout,
    )
    try:
        manifest = loader.download_snapshot(args.system)
    except OfflineModeError as exc:
        raise CliError(
            "Snapshot download is blocked by offline mode.",
            code="offline_download_blocked",
            exit_code=EXIT_ERROR,
        ) from exc
    except (OSError, TypeError, ValueError, VocabLoaderError) as exc:
        raise CliError(
            "The vocabulary snapshot could not be downloaded and verified.",
            code="snapshot_download_failed",
            exit_code=EXIT_ERROR,
        ) from exc
    payload = manifest.to_dict()
    return emit(
        args,
        payload,
        human=f"Downloaded {manifest.system} snapshot {manifest.version}.",
    )


def _handle_grounding_snapshot_list(args: argparse.Namespace) -> int:
    """List checksum-pinned snapshots without touching the network."""

    from ..clinical.grounding import VocabLoader

    loader = VocabLoader(cache_dir=args.cache_dir, local_only=True)
    payload = {
        "snapshots": [manifest.to_dict() for manifest in loader.list_snapshots()]
    }
    return emit(args, payload, human=json.dumps(payload, indent=2, sort_keys=True))


def _handle_export_openehr(args: argparse.Namespace) -> int:
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        entities = _extract_clinical_entities(payload)
        source_text = _extract_openehr_source_text(payload)
        if args.source_text_file is not None:
            source_text = args.source_text_file.read_text(encoding="utf-8")
        doc_id = args.doc_id or _extract_doc_id(payload)

        from ..clinical.exporters.openehr import to_openehr_composition

        composition = to_openehr_composition(
            entities,
            operational_template=args.template,
            doc_id=doc_id,
            source_text=source_text,
            composer_name=args.composer,
            language=args.language,
            territory=args.territory,
            time=args.time,
            vocabulary_key=args.vocabulary_key,
        )
        args.output.write_text(
            json.dumps(composition, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except FileNotFoundError as exc:
        raise CliError(
            f"Input file not found: {exc.filename}",
            code="input_not_found",
            exit_code=EXIT_ERROR,
        )
    except json.JSONDecodeError as exc:
        raise CliError(
            f"Invalid JSON in {args.input}: {exc.msg} "
            f"at line {exc.lineno} column {exc.colno}",
            code="invalid_json",
            exit_code=EXIT_ERROR,
        )
    except OSError as exc:
        raise CliError(
            f"Failed to read or write openEHR COMPOSITION: {exc}",
            code="io_error",
            exit_code=EXIT_ERROR,
        )
    except (TypeError, ValueError) as exc:
        raise CliError(
            f"Failed to assemble openEHR COMPOSITION: {exc}",
            code="assemble_failed",
            exit_code=EXIT_ERROR,
        )

    result = {
        "output": str(args.output),
        "entity_count": len(entities),
        "coded_element_count": sum(path.endswith("|code") for path in composition),
    }
    return emit(
        args,
        result,
        human=f"openEHR COMPOSITION written to: {args.output}",
    )


def _extract_doc_id(payload: Any) -> str:
    """Return the stable source document id carried by a result payload."""

    if isinstance(payload, MappingABC):
        for key in ("doc_id", "document_id", "id"):
            value = payload.get(key)
            if isinstance(value, (str, int)) and str(value):
                return str(value)
    return "openmed-document"


def _extract_clinical_entities(payload: Any) -> list[Any]:
    entities = _find_clinical_entities_payload(payload)
    if not isinstance(entities, list):
        raise ValueError("clinical entities must be a JSON array")
    normalized: list[Any] = []
    for index, entity in enumerate(entities):
        if not isinstance(entity, MappingABC):
            raise ValueError(f"clinical entity at index {index} must be a JSON object")
        normalized.append(dict(entity))
    return normalized


def _find_clinical_entities_payload(payload: Any) -> Any:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, MappingABC):
        raise ValueError(
            "openEHR input must be a JSON array of clinical entities or a result object"
        )

    for key in ("entities", "clinical_entities", "clinicalEntities"):
        if key in payload:
            return payload[key]

    result_payload = payload.get("result")
    if isinstance(result_payload, MappingABC):
        for key in ("entities", "clinical_entities", "clinicalEntities"):
            if key in result_payload:
                return result_payload[key]

    raise ValueError(
        "openEHR input must contain grounded clinical entities under "
        "'entities' or 'clinical_entities'"
    )


def _extract_openehr_source_text(payload: Any) -> str | None:
    if not isinstance(payload, MappingABC):
        return None
    for key in ("source_text", "text", "note", "narrative"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    result_payload = payload.get("result")
    if isinstance(result_payload, MappingABC):
        for key in ("source_text", "text", "note", "narrative"):
            value = result_payload.get(key)
            if isinstance(value, str):
                return value
    return None


def _extract_fhir_doc_id(payload: Any) -> str:
    """Return the stable document id carried by a serialized result payload."""
    if isinstance(payload, MappingABC):
        for key in ("doc_id", "document_id", "id"):
            value = payload.get(key)
            if isinstance(value, (str, int)) and str(value):
                return str(value)
    return "openmed-document"


def _extract_fhir_resources(payload: Any) -> list[dict[str, Any]]:
    """Extract standalone FHIR resources from supported result JSON shapes."""
    resources = _find_fhir_resource_payload(payload)
    if not isinstance(resources, list):
        raise ValueError("FHIR resources must be a JSON array")

    normalized: list[dict[str, Any]] = []
    for index, resource in enumerate(resources):
        if not isinstance(resource, MappingABC):
            raise ValueError(f"FHIR resource at index {index} must be a JSON object")
        if resource.get("resourceType") == "Bundle":
            raise ValueError(
                "input resources must be standalone FHIR resources, not Bundles"
            )
        normalized.append(dict(resource))
    return normalized


def _find_fhir_resource_payload(payload: Any) -> Any:
    if isinstance(payload, list):
        return payload

    if not isinstance(payload, MappingABC):
        raise ValueError(
            "FHIR input must be a JSON array of resources or a result object"
        )

    for key in ("fhir_resources", "fhirResources", "resources"):
        if key in payload:
            return payload[key]

    fhir_payload = payload.get("fhir")
    if isinstance(fhir_payload, list):
        return fhir_payload
    if isinstance(fhir_payload, MappingABC):
        for key in ("resources", "fhir_resources", "fhirResources"):
            if key in fhir_payload:
                return fhir_payload[key]

    result_payload = payload.get("result")
    if isinstance(result_payload, MappingABC):
        for key in ("fhir_resources", "fhirResources", "resources"):
            if key in result_payload:
                return result_payload[key]

    if "resourceType" in payload:
        if payload.get("resourceType") == "Bundle":
            raise ValueError(
                "input is already a FHIR Bundle; provide standalone resources"
            )
        return [payload]

    raise ValueError(
        "FHIR input must contain standalone FHIR resources under "
        "'resources', 'fhir_resources', or 'fhir.resources'"
    )


def _handle_policy_diff(args: argparse.Namespace) -> int:
    from ..core.policy_diff import diff_policies, render

    try:
        diff = diff_policies(args.base, args.candidate)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise CliError(
            f"Policy diff failed: {exc}",
            code="diff_failed",
            exit_code=EXIT_ERROR,
        )

    payload = render(diff, fmt="dict")
    if args.output_format == "json":
        human = json.dumps(payload, indent=2, sort_keys=True)
    else:
        human = render(diff, fmt="text")
    return emit(args, payload, human=human)


def _handle_eval_load_test(args: argparse.Namespace) -> int:
    """Run the in-process ASGI load test and print its report."""
    from openmed.eval.load_test import run_load_test
    from openmed.service.app import app

    report = run_load_test(
        app,
        concurrency=args.concurrency,
        total_requests=args.total_requests,
    )
    data = vars(report)
    return emit(args, data, human=json.dumps(data, indent=2))


def _handle_benchmark_pii(args: argparse.Namespace) -> int:
    if args.attack == "reid":
        return _handle_benchmark_pii_reid(args)

    from openmed.eval.datasets import CLINICAL_PRIVACY_MODEL_ID
    from openmed.eval.harness import run_benchmark
    from openmed.eval.suites import (
        SHIELD,
        load_suite_fixtures,
        run_clinical_phi_shield_benchmark,
        suite_metadata,
    )

    try:
        models = _parse_model_args(args.models or [])
    except ValueError as exc:
        raise CliError(str(exc), code="invalid_argument", exit_code=EXIT_USAGE)
    if not models:
        raise CliError(
            "At least one model identifier is required.",
            code="missing_models",
            exit_code=EXIT_USAGE,
        )

    suite = str(args.suite or SHIELD)
    if args.checkpoint_manifest_ref and args.checkpoint_manifest is None:
        raise CliError(
            "--checkpoint-manifest-ref requires --checkpoint-manifest.",
            code="invalid_argument",
            exit_code=EXIT_USAGE,
        )
    if args.checkpoint_manifest is not None:
        if suite != SHIELD:
            raise CliError(
                "--checkpoint-manifest is supported only for the SHIELD suite.",
                code="invalid_argument",
                exit_code=EXIT_USAGE,
            )
        if args.full_shield:
            raise CliError(
                "The clinical PHI flagship report uses the public SHIELD sample; "
                "do not combine --checkpoint-manifest with --full-shield.",
                code="invalid_argument",
                exit_code=EXIT_USAGE,
            )
        if len(models) != 1 or models[0] != CLINICAL_PRIVACY_MODEL_ID:
            raise CliError(
                "--checkpoint-manifest requires exactly the named model "
                f"{CLINICAL_PRIVACY_MODEL_ID!r}.",
                code="invalid_argument",
                exit_code=EXIT_USAGE,
            )
        if not args.checkpoint_manifest_ref:
            raise CliError(
                "--checkpoint-manifest-ref is required for reproducible evidence.",
                code="invalid_argument",
                exit_code=EXIT_USAGE,
            )
    try:
        if suite == SHIELD:
            use_sample = not bool(args.full_shield)
            fixtures = load_suite_fixtures(suite, use_sample=use_sample)
            metadata = suite_metadata(suite, use_sample=use_sample)
        else:
            fixtures = load_suite_fixtures(suite)
            metadata = suite_metadata(suite)
    except (PermissionError, RuntimeError, ValueError) as exc:
        raise CliError(
            f"Failed to load benchmark suite: {exc}",
            code="load_failed",
            exit_code=EXIT_ERROR,
        )

    metadata = dict(metadata)
    metadata.setdefault("benchmark_domain", "pii")
    metadata.setdefault("source_suite", suite)

    if args.checkpoint_manifest is not None:
        try:
            reports = [
                run_clinical_phi_shield_benchmark(
                    fixtures,
                    checkpoint_manifest=args.checkpoint_manifest,
                    checkpoint_manifest_ref=args.checkpoint_manifest_ref,
                    device=args.device,
                )
            ]
        except ValueError as exc:
            raise CliError(
                f"Invalid clinical PHI checkpoint evidence: {exc}",
                code="invalid_argument",
                exit_code=EXIT_USAGE,
            ) from exc
    else:
        reports = [
            run_benchmark(
                fixtures,
                suite=suite,
                model_name=model,
                device=args.device,
                metadata=metadata,
            )
            for model in models
        ]
    if len(reports) == 1:
        payload: Any = reports[0].to_dict()
    else:
        payload = {
            "metadata": metadata,
            "reports": [report.to_dict() for report in reports],
            "suite": suite,
        }

    if args.output_dir:
        try:
            paths = _write_benchmark_report_files(
                reports,
                output_dir=args.output_dir,
                domain="pii",
                suite=suite,
                device=str(args.device),
            )
        except OSError as exc:
            raise CliError(
                f"Failed to write benchmark output: {exc}",
                code="write_failed",
                exit_code=EXIT_ERROR,
            )
        if args.output is None:
            written = [
                {"json": str(json_path), "markdown": str(markdown_path)}
                for json_path, markdown_path in paths
            ]
            human_lines = ["Benchmark reports written:"]
            for json_path, markdown_path in paths:
                human_lines.append(f"  JSON: {json_path}")
                human_lines.append(f"  Markdown: {markdown_path}")
            return emit(args, {"written": written}, human="\n".join(human_lines))

    output = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        try:
            args.output.write_text(output + "\n", encoding="utf-8")
        except OSError as exc:
            raise CliError(
                f"Failed to write benchmark output: {exc}",
                code="write_failed",
                exit_code=EXIT_ERROR,
            )
        return emit(args, payload, human=None)
    return emit(args, payload, human=output)


def _handle_benchmark_clinical(args: argparse.Namespace) -> int:
    from openmed.eval.suites import (
        BIOMEDICAL_NER,
        load_suite_fixtures,
        run_biomedical_ner_benchmark,
        suite_metadata,
    )

    try:
        suite = str(args.suite)
        task = str(args.task)
        load_kwargs: dict[str, Any] = {
            "task": task,
            "path": args.input,
            "cache_dir": args.cache_dir,
        }
        if args.split is not None:
            load_kwargs["split"] = str(args.split)
        fixtures = load_suite_fixtures(suite, **load_kwargs)
        metadata_kwargs: dict[str, Any] = {}
        if suite == "drugprot":
            metadata_kwargs["task"] = task
        if suite == BIOMEDICAL_NER and args.split is not None:
            metadata_kwargs["split"] = str(args.split)
        metadata = suite_metadata(suite, **metadata_kwargs)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise CliError(
            f"Failed to load clinical benchmark suite: {exc}",
            code="load_failed",
            exit_code=EXIT_ERROR,
        )

    if suite == BIOMEDICAL_NER and task == "ner":
        split = str(args.split) if args.split is not None else "test"
        try:
            models = _parse_model_args(args.models or [])
        except ValueError as exc:
            raise CliError(str(exc), code="invalid_argument", exit_code=EXIT_USAGE)
        if not models:
            models = ["disease_detection_superclinical"]
        reports = [
            run_biomedical_ner_benchmark(
                fixtures,
                model_name=model,
                device=str(args.device),
                metadata=metadata,
                split=split,
            )
            for model in models
        ]
        if len(reports) == 1:
            payload: Any = reports[0].to_dict()
        else:
            payload = {
                "metadata": metadata,
                "reports": [report.to_dict() for report in reports],
                "suite": suite,
            }
        return _write_json_payload(args, payload, args.output)

    payload: dict[str, Any] = {
        "fixture_count": len(fixtures),
        "metadata": metadata,
        "suite": suite,
        "task": task,
    }
    if task == "relation":
        payload["relation_count"] = sum(
            len(getattr(fixture, "relations", ())) for fixture in fixtures
        )
    else:
        payload["span_count"] = sum(
            len(getattr(fixture, "gold_spans", ())) for fixture in fixtures
        )

    return _write_json_payload(args, payload, args.output)


def _handle_benchmark_mobile(args: argparse.Namespace) -> int:
    from openmed.eval import perf as perf_module

    try:
        models = _parse_model_args(args.models or [])
    except ValueError as exc:
        raise CliError(str(exc), code="invalid_argument", exit_code=EXIT_USAGE)
    if not models:
        models = [perf_module.SYNTHETIC_PERF_MODEL_NAME]

    reports = []
    try:
        for model in models:
            runner = (
                perf_module.synthetic_perf_runner
                if model == perf_module.SYNTHETIC_PERF_MODEL_NAME
                else None
            )
            reports.append(
                perf_module.run_perf_benchmark(
                    model,
                    device=str(args.device),
                    tier=str(args.tier),
                    runner=runner,
                    metadata={"benchmark_domain": "mobile", "source_suite": "perf"},
                )
            )
    except (OSError, RuntimeError, ValueError) as exc:
        raise CliError(
            f"Mobile benchmark failed: {exc}",
            code="benchmark_failed",
            exit_code=EXIT_ERROR,
        )

    if args.output_dir:
        try:
            paths = _write_perf_report_files(
                reports,
                output_dir=args.output_dir,
                suite="perf",
            )
        except OSError as exc:
            raise CliError(
                f"Failed to write benchmark output: {exc}",
                code="write_failed",
                exit_code=EXIT_ERROR,
            )
        written = [
            {"json": str(json_path), "markdown": str(markdown_path)}
            for json_path, markdown_path in paths
        ]
        human_lines = ["Mobile benchmark reports written:"]
        for json_path, markdown_path in paths:
            human_lines.append(f"  JSON: {json_path}")
            human_lines.append(f"  Markdown: {markdown_path}")
        return emit(args, {"written": written}, human="\n".join(human_lines))

    if len(reports) == 1:
        payload: Any = reports[0].to_dict()
        human = reports[0].to_json()
    else:
        payload = {"reports": [report.to_dict() for report in reports]}
        human = json.dumps(payload, indent=2, sort_keys=True)
    return emit(args, payload, human=human)


def _handle_benchmark_latency(args: argparse.Namespace) -> int:
    from openmed.core.offline import network_blocked_if_offline
    from openmed.eval import arm_latency as arm_latency_module
    from openmed.onnx.inference import OnnxModel

    try:
        budget = arm_latency_module.load_arm_latency_budget(args.budget)
        documents = arm_latency_module.load_latency_documents(args.corpus)
        with network_blocked_if_offline(local_only=True):
            model = OnnxModel.from_pretrained(
                args.model,
                variant="int8",
                revision=str(args.revision),
                cache_dir=args.cache_dir,
                local_files_only=True,
            )
            report = arm_latency_module.run_arm_latency_benchmark(
                model,
                model_id=budget.model_id,
                model_revision=str(args.revision),
                documents=documents,
                budget=budget,
                corpus_path=args.corpus,
                warmup_runs=args.warmup_runs,
                repeat=args.repeat,
                metadata={"execution_provider": "CPUExecutionProvider"},
            )
        if args.output is not None:
            report.write_json(args.output)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        sys.stderr.write(f"ARM latency benchmark failed: {exc}\n")
        return 1

    sys.stdout.write(report.to_json() + "\n")
    if not report.passed:
        sys.stderr.write(
            "ARM latency budget exceeded: "
            f"p95 {report.p95_ms:.3f} ms > "
            f"{report.verdict.maximum_p95_ms:.3f} ms\n"
        )
        return 1
    return 0


def _handle_profile_memory(args: argparse.Namespace) -> int:
    from openmed.eval import memprofile as memprofile_module

    model = args.model or memprofile_module.SYNTHETIC_MEMPROFILE_MODEL_NAME
    loader = (
        memprofile_module.synthetic_memprofile_loader if args.model is None else None
    )
    top_allocators = (
        memprofile_module.DEFAULT_TOP_ALLOCATORS
        if args.top_allocators is None
        else args.top_allocators
    )
    if top_allocators < 1:
        raise CliError(
            "--top-allocators must be a positive integer.",
            code="invalid_argument",
            exit_code=EXIT_USAGE,
        )

    try:
        profile = memprofile_module.profile_memory(
            model,
            loader=loader,
            top_allocators=top_allocators,
            metadata={"source": "cli"},
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise CliError(
            f"Memory profile failed: {exc}",
            code="profile_failed",
            exit_code=EXIT_ERROR,
        )

    data = json.loads(profile.to_json())
    rendered = profile.to_markdown() if args.format == "markdown" else profile.to_json()
    if args.output:
        try:
            if args.format == "markdown":
                profile.write_markdown(args.output)
            else:
                profile.write_json(args.output)
        except OSError as exc:
            raise CliError(
                f"Failed to write memory profile: {exc}",
                code="write_failed",
                exit_code=EXIT_ERROR,
            )
        return emit(args, data, human=f"Memory profile written: {args.output}")

    return emit(args, data, human=rendered)


def _handle_benchmark_false_negatives(args: argparse.Namespace) -> int:
    from openmed.eval.error_analysis import ErrorAnalysisReport
    from openmed.eval.false_negatives import (
        explore_false_negatives,
        load_fixture_texts,
    )

    try:
        report = ErrorAnalysisReport.read_json(args.report)
    except FileNotFoundError:
        raise CliError(
            f"Report not found: {args.report}",
            code="report_not_found",
            exit_code=EXIT_ERROR,
        )
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        raise CliError(
            f"Failed to read error-analysis report: {exc}",
            code="load_failed",
            exit_code=EXIT_ERROR,
        )

    context_chars = getattr(args, "context_chars", None)
    if context_chars is not None and context_chars < 0:
        raise CliError(
            "context-chars must be non-negative",
            code="invalid_argument",
            exit_code=EXIT_USAGE,
        )

    fixture_texts: dict[str, str] = {}
    if args.fixtures:
        try:
            fixture_texts = load_fixture_texts(args.fixtures)
        except (OSError, ValueError) as exc:
            raise CliError(
                f"Failed to load fixtures: {exc}",
                code="load_failed",
                exit_code=EXIT_ERROR,
            )

    try:
        exploration = explore_false_negatives(
            report,
            fixture_texts=fixture_texts,
            label=args.label,
            limit=args.limit,
        )
    except ValueError as exc:
        raise CliError(str(exc), code="invalid_argument", exit_code=EXIT_USAGE)

    payload = exploration.to_dict()
    if context_chars is not None:
        for group in payload["groups"]:
            for record in group["records"]:
                _trim_record_context(record, context_chars)

    human = _render_false_negatives_table(exploration, context_chars)
    return emit(args, payload, human=human)


def _render_false_negatives_table(
    exploration: Any,
    context_chars: int | None,
) -> str:
    lines = [
        f"# False Negatives: {exploration.suite}",
        "",
        f"Model: {exploration.model_name}  Device: {exploration.device}",
        (
            f"Missed gold spans: {exploration.total_missed}  "
            f"Stored examples: {exploration.available}  Shown: {exploration.shown}"
        ),
    ]
    if exploration.label_filter is not None:
        lines.append(f"Label filter: {exploration.label_filter}")
    if exploration.limit is not None:
        lines.append(f"Limit: {exploration.limit}")
    if exploration.examples_truncated:
        lines.append(
            "Stored examples are capped by the report "
            f"(example cap: {exploration.example_cap} per label)."
        )
    if exploration.shown and not exploration.has_text:
        lines.append(
            "Verified synthetic fixture text unavailable: showing offsets, "
            "labels, and hashes only."
        )

    if not exploration.groups:
        lines.append("")
        lines.append("No missed gold spans found.")
        return "\n".join(lines) + "\n"

    for group in exploration.groups:
        lines.append("")
        lines.append(f"## {group.label} ({group.count})")
        lines.append(f"Stored examples: {group.available}  Shown: {len(group.records)}")
        if not group.records:
            if group.available:
                lines.append("- No stored example shown under the current limit.")
            else:
                lines.append("- No missed-span example is stored for this label.")
        for record in group.records:
            span = f"{record.fixture_id} [{record.start}:{record.end}]"
            if record.span_text is not None:
                span += f" {record.span_text!r}"
            lines.append(f"- {span}")
            if record.context is not None:
                context = record.context
                if context_chars is not None:
                    context = _center_context(record, context_chars)
                lines.append(f"    context: {context!r}")
            else:
                lines.append(f"    hash: {record.text_hash}")
    return "\n".join(lines) + "\n"


def _center_context(record: Any, context_chars: int) -> str:
    context = record.context or ""
    if context_chars < 0 or len(context) <= context_chars:
        return context
    span_offset = max(record.start - record.context_start, 0)
    span_length = max(record.end - record.start, 0)
    center = span_offset + span_length // 2
    half = context_chars // 2
    start = max(0, center - half)
    end = min(len(context), start + context_chars)
    start = max(0, end - context_chars)
    return context[start:end]


def _trim_record_context(record: dict[str, Any], context_chars: int) -> None:
    context = record.get("context")
    if not isinstance(context, str) or context_chars < 0:
        return
    if len(context) <= context_chars:
        return
    span_offset = max(int(record["start"]) - int(record["context_start"]), 0)
    span_length = max(int(record["end"]) - int(record["start"]), 0)
    center = span_offset + span_length // 2
    half = context_chars // 2
    start = max(0, center - half)
    end = min(len(context), start + context_chars)
    start = max(0, end - context_chars)
    record["context"] = context[start:end]


def _write_json_payload(
    args: argparse.Namespace, payload: Any, output_path: Path | None
) -> int:
    output = json.dumps(payload, indent=2, sort_keys=True)
    if output_path:
        try:
            output_path.write_text(output + "\n", encoding="utf-8")
        except OSError as exc:
            raise CliError(
                f"Failed to write benchmark output: {exc}",
                code="write_failed",
                exit_code=EXIT_ERROR,
            )
        return emit(args, payload, human=None)
    return emit(args, payload, human=output)


def _write_benchmark_report_files(
    reports: Sequence[Any],
    *,
    output_dir: Path,
    domain: str,
    suite: str,
    device: str,
) -> list[tuple[Path, Path]]:
    paths: list[tuple[Path, Path]] = []
    for report in reports:
        json_path, markdown_path = _benchmark_report_paths(
            output_dir=output_dir,
            domain=domain,
            suite=suite,
            model_name=str(report.model_name),
            device=device,
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        report.write_json(json_path)
        report.write_markdown(markdown_path)
        paths.append((json_path, markdown_path))
    return paths


def _write_perf_report_files(
    reports: Sequence[Any],
    *,
    output_dir: Path,
    suite: str,
) -> list[tuple[Path, Path]]:
    paths: list[tuple[Path, Path]] = []
    for report in reports:
        json_path, markdown_path = _benchmark_report_paths(
            output_dir=output_dir,
            domain="mobile",
            suite=suite,
            model_name=str(report.model_name),
            device=str(report.device),
        )
        report.write_json(json_path)
        report.write_markdown(markdown_path)
        paths.append((json_path, markdown_path))
    return paths


def _benchmark_report_paths(
    *,
    output_dir: Path,
    domain: str,
    suite: str,
    model_name: str,
    device: str,
) -> tuple[Path, Path]:
    stem = f"{_path_token(model_name)}-{_path_token(device)}"
    directory = output_dir / _path_token(domain) / _path_token(suite)
    return directory / f"{stem}.json", directory / f"{stem}.md"


def _path_token(value: str) -> str:
    token = "".join(
        character if character.isalnum() or character in "._-" else "-"
        for character in value
    ).strip("-")
    return token or "value"


def _parse_model_args(values: Sequence[str]) -> list[str]:
    models: list[str] = []
    for value in values:
        models.extend(item.strip() for item in value.split(",") if item.strip())
    if models == ["@manifest"]:
        manifest_models = [
            str(row["repo_id"])
            for row in load_manifest_rows(MANIFEST_PATH)
            if isinstance(row.get("repo_id"), str) and row["repo_id"]
        ]
        if not manifest_models:
            raise ValueError(f"model manifest is empty: {MANIFEST_PATH}")
        return manifest_models
    if "@manifest" in models:
        raise ValueError("--models @manifest cannot be combined with explicit ids")
    return models


def build_models_size_report(
    model_key: str | None = None,
    *,
    budget_mb: float | None = None,
    remote: bool = False,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Build the shared argparse/Typer model-size report payload."""

    if budget_mb is not None and (
        not isinstance(budget_mb, (int, float))
        or isinstance(budget_mb, bool)
        or not math.isfinite(float(budget_mb))
        or budget_mb < 0
    ):
        raise ValueError("budget_mb must be a finite non-negative number")

    estimates = estimate_model_sizes(
        model_key,
        manifest_path=manifest_path or MANIFEST_PATH,
    )
    if model_key is not None and not estimates:
        raise ValueError(f"Unknown model key: {model_key}")

    try:
        cached_by_id = {model.repo_id: model for model in list_cached_models()}
    except ImportError:
        # Size inspection remains useful in the minimal install. Cache
        # awareness activates automatically when the optional HF extra exists.
        cached_by_id = {}

    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    for estimate in estimates:
        remote_mb = _remote_size_or_none(estimate, remote=remote, warnings=warnings)
        snapshot_mb = remote_mb or estimate.download_mb
        disk_mb = remote_mb or estimate.disk_mb
        source = "remote" if remote_mb is not None else estimate.source

        cached = cached_by_id.get(estimate.repo_id)
        is_cached = cached is not None and cached.size_on_disk > 0
        if is_cached:
            disk_mb = round(cached.size_on_disk / 1_000_000, 3)
            if snapshot_mb is None:
                snapshot_mb = disk_mb

        peak_ram_mb = estimate.peak_ram_mb
        if remote_mb is not None and estimate.source == "estimated":
            peak_ram_mb = round(max(256.0, remote_mb * 1.25), 3)

        download_mb = 0.0 if is_cached else snapshot_mb
        if budget_mb is not None and (download_mb is None or download_mb > budget_mb):
            continue

        rows.append(
            {
                "repo_id": estimate.repo_id,
                "task": estimate.task,
                "download_mb": download_mb,
                "snapshot_mb": snapshot_mb,
                "disk_mb": disk_mb,
                "peak_ram_mb": peak_ram_mb,
                "cached": is_cached,
                "status": ("cached — 0 MB to download" if is_cached else "not cached"),
                "source": source,
                "recommended": False,
            }
        )

    recommendations: list[dict[str, Any]] = []
    if budget_mb is not None:
        recommendations = _size_recommendations(rows)
        recommended_ids = {
            recommendation["repo_id"] for recommendation in recommendations
        }
        for row in rows:
            row["recommended"] = row["repo_id"] in recommended_ids

    return {
        "budget_mb": budget_mb,
        "remote": remote,
        "models": rows,
        "recommendations": recommendations,
        "warnings": warnings,
    }


def _remote_size_or_none(
    estimate: ModelSizeEstimate,
    *,
    remote: bool,
    warnings: list[str],
) -> float | None:
    if not remote:
        return None
    try:
        return get_remote_model_size_mb(estimate.repo_id)
    except (ImportError, OfflineModeError):
        raise
    except Exception as exc:  # pragma: no cover - depends on remote service
        warnings.append(f"{estimate.repo_id}: {exc}")
        return None


def _size_recommendations(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_task: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        snapshot_mb = row.get("snapshot_mb")
        if not isinstance(snapshot_mb, (int, float)):
            continue
        by_task.setdefault(str(row.get("task") or "unknown"), []).append(row)

    recommendations = []
    for task, task_rows in sorted(by_task.items()):
        recommended = min(
            task_rows,
            key=lambda row: (float(row["snapshot_mb"]), str(row["repo_id"])),
        )
        recommendations.append(
            {
                "task": task,
                "repo_id": recommended["repo_id"],
                "snapshot_mb": recommended["snapshot_mb"],
            }
        )
    return recommendations


def _handle_models_size(args: argparse.Namespace) -> int:
    try:
        report = build_models_size_report(
            args.model_key,
            budget_mb=args.budget_mb,
            remote=args.remote,
        )
    except (ImportError, OfflineModeError, OSError, ValueError) as exc:
        sys.stderr.write(f"Failed to inspect model sizes: {exc}\n")
        return 1

    if not report["models"]:
        if args.budget_mb is None:
            message = "No model size metadata is available."
        else:
            message = f"No models fit the {args.budget_mb:g} MB download budget."
        sys.stderr.write(f"{message}\n")
        return 1

    for warning in report["warnings"]:
        sys.stderr.write(f"Remote size lookup warning: {warning}\n")

    if args.output_format == "json":
        sys.stdout.write(f"{json.dumps(report, indent=2)}\n")
    else:
        sys.stdout.write(_format_models_size_table(report))
    return 0


def _format_models_size_table(report: Mapping[str, Any]) -> str:
    columns = (
        ("repo_id", "model"),
        ("task", "task"),
        ("download_mb", "download_mb"),
        ("disk_mb", "disk_mb"),
        ("peak_ram_mb", "peak_ram_mb"),
        ("status", "status"),
    )
    rows = [
        {
            "repo_id": str(row["repo_id"]),
            "task": str(row["task"]),
            "download_mb": _format_models_size_mb(row["download_mb"]),
            "disk_mb": _format_models_size_mb(row["disk_mb"]),
            "peak_ram_mb": _format_models_size_mb(row["peak_ram_mb"]),
            "status": str(row["status"]),
        }
        for row in report["models"]
    ]
    widths = {
        key: max(len(header), *(len(row[key]) for row in rows))
        for key, header in columns
    }
    lines = [
        "  ".join(header.ljust(widths[key]) for key, header in columns),
        "  ".join("-" * widths[key] for key, _header in columns),
        *[
            "  ".join(row[key].ljust(widths[key]) for key, _header in columns)
            for row in rows
        ],
    ]
    if report["recommendations"]:
        lines.append("")
        lines.extend(
            "Recommended for "
            f"{recommendation['task']}: {recommendation['repo_id']} "
            f"({_format_models_size_mb(recommendation['snapshot_mb'])} MB snapshot)"
            for recommendation in report["recommendations"]
        )
    return "\n".join(lines) + "\n"


def _format_models_size_mb(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "unknown"
    return f"{float(value):.1f}"


def _handle_models_search(args: argparse.Namespace) -> int:
    if (
        args.min_params is not None
        and args.max_params is not None
        and args.min_params > args.max_params
    ):
        raise CliError(
            "--min-params must be less than or equal to --max-params",
            code="invalid_argument",
            exit_code=EXIT_USAGE,
        )

    try:
        results = search_models(
            task=args.task,
            language=args.language,
            tier=args.tier,
            max_params=args.max_params,
            min_params=args.min_params,
            format=args.format,
            license=args.license,
            query=args.query,
            require_params=args.require_params,
        )
    except (OSError, ValueError) as exc:
        raise CliError(
            f"Failed to search models: {exc}",
            code="search_failed",
            exit_code=EXIT_ERROR,
        )

    if not results:
        raise CliError(
            "No models matched the search filters.",
            code="no_results",
            exit_code=EXIT_ERROR,
        )

    payload = {
        "count": len(results),
        "models": [_recommendation_to_dict(result) for result in results],
    }
    return emit(args, payload, human=_format_model_search_table(results))


def _handle_models_recommend(args: argparse.Namespace) -> int:
    try:
        results = recommend_models(
            device_tier=args.tier,
            task=args.task,
            language=args.language,
        )
    except (OSError, ValueError) as exc:
        raise CliError(
            f"Failed to recommend models: {exc}",
            code="recommend_failed",
            exit_code=EXIT_ERROR,
        )

    if not results:
        raise CliError(
            f"No model fits the '{args.tier}' device tier for the requested filters.",
            code="no_results",
            exit_code=EXIT_ERROR,
        )

    payload = {
        "tier": args.tier,
        "task": args.task,
        "language": args.language,
        "recommended": results[0].repo_id,
        "models": [_recommendation_to_dict(result) for result in results],
    }
    human = (
        f"Recommended for {args.tier}: {results[0].repo_id}\n"
        + _format_model_search_table(results)
    )
    return emit(args, payload, human=human)


def _handle_models_card(args: argparse.Namespace) -> int:
    try:
        row = _find_manifest_row(args.repo_id)
        rendered = render_model_card(dict(row))
    except (OSError, ValueError) as exc:
        raise CliError(
            f"Failed to render model card: {exc}",
            code="render_failed",
            exit_code=EXIT_ERROR,
        )

    if args.check is not None:
        try:
            existing = args.check.read_text(encoding="utf-8")
        except OSError as exc:
            raise CliError(
                f"Failed to read README for comparison: {exc}",
                code="read_failed",
                exit_code=EXIT_ERROR,
            )

        if existing == rendered:
            return emit(
                args,
                {"repo_id": args.repo_id, "matches": True, "diff": ""},
                human=None,
            )

        diff_text = "".join(
            difflib.unified_diff(
                existing.splitlines(keepends=True),
                rendered.splitlines(keepends=True),
                fromfile=str(args.check),
                tofile=f"rendered:{args.repo_id}",
            )
        )
        emit(
            args,
            {"repo_id": args.repo_id, "matches": False, "diff": diff_text},
            human=diff_text,
        )
        return 1

    if args.output is not None:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        except OSError as exc:
            raise CliError(
                f"Failed to write model card: {exc}",
                code="write_failed",
                exit_code=EXIT_ERROR,
            )
        return emit(
            args,
            {"repo_id": args.repo_id, "output": str(args.output), "card": rendered},
            human=None,
        )

    return emit(args, {"repo_id": args.repo_id, "card": rendered}, human=rendered)


def _find_manifest_row(repo_id: str) -> Mapping[str, Any]:
    rows = load_manifest_rows(MANIFEST_PATH)
    for row in rows:
        if row.get("repo_id") == repo_id:
            return row
    raise ValueError(f"repo_id not found in model manifest: {repo_id}")


def _recommendation_to_dict(result: ModelSearchResult) -> dict[str, Any]:
    row = result.manifest_row
    return {
        "repo_id": result.repo_id,
        "family": result.family,
        "task": result.task,
        "languages": list(result.languages),
        "tier": result.tier,
        "param_count": result.param_count,
        "formats": list(result.formats),
        "license": result.license,
        "recommended_tier": row.get("recommended_tier"),
        "peak_ram_mb": row.get("peak_ram_mb"),
        "latency_ms": row.get("latency_ms"),
        "benchmark": result.benchmark,
    }


def _format_model_search_table(results: Sequence[ModelSearchResult]) -> str:
    columns = (
        ("repo_id", "repo_id"),
        ("family", "family"),
        ("task", "task"),
        ("languages", "languages"),
        ("tier", "tier"),
        ("params", "params"),
        ("formats", "formats"),
        ("license", "license"),
    )
    rows = [
        {
            "repo_id": result.repo_id,
            "family": result.family or "-",
            "task": result.task or "-",
            "languages": ",".join(result.languages) or "-",
            "tier": result.tier or "-",
            "params": _format_param_count(result.param_count),
            "formats": ",".join(result.formats) or "-",
            "license": result.license or "-",
        }
        for result in results
    ]
    widths = {
        key: max(len(header), *(len(row[key]) for row in rows))
        for key, header in columns
    }

    header = "  ".join(header.ljust(widths[key]) for key, header in columns)
    separator = "  ".join("-" * widths[key] for key, _header in columns)
    body = [
        "  ".join(row[key].ljust(widths[key]) for key, _header in columns)
        for row in rows
    ]
    return "\n".join([header, separator, *body]) + "\n"


def _format_param_count(param_count: int | None) -> str:
    if param_count is None:
        return "unknown"
    return f"{param_count:,}"


def _handle_models_list(args: argparse.Namespace) -> int:
    config = _load_and_apply_config(args)

    _, _, list_models, _ = _lazy_api()

    try:
        models = list_models(
            include_registry=True,
            include_remote=args.include_remote,
            config=config,
        )
    except Exception as exc:  # pragma: no cover - defensive
        raise CliError(
            f"Failed to list models: {exc}",
            code="load_failed",
            exit_code=EXIT_ERROR,
        )

    model_names = [str(model) for model in models]
    payload = {"count": len(model_names), "models": model_names}
    return emit(args, payload, human="\n".join(model_names))


def _handle_models_pull(args: argparse.Namespace) -> int:
    from ..core.hf_hub import DownloadProgress, prefetch_model

    config = _load_and_apply_config(args)
    completed_files = 0

    def report_progress(progress: DownloadProgress) -> None:
        nonlocal completed_files
        finished = progress.files_done > completed_files
        completed_files = max(completed_files, progress.files_done)
        line_end = "\n" if finished else "\r"
        sys.stdout.write(
            f"{progress.filename}: "
            f"{progress.bytes_done}/{progress.bytes_total} bytes; "
            f"{progress.files_done}/{progress.files_total} files"
            f"{line_end}"
        )
        sys.stdout.flush()

    try:
        path = prefetch_model(
            args.model,
            revision=args.revision,
            config=config,
            retries=args.retries,
            max_bandwidth=args.max_bandwidth,
            progress_callback=report_progress,
        )
    except Exception as exc:  # pragma: no cover - exact failures tested in helper
        sys.stderr.write(f"Failed to pull model: {exc}\n")
        return 1

    sys.stdout.write(f"Model ready: {path}\n")
    return 0


def _handle_models_info(args: argparse.Namespace) -> int:
    config = _load_and_apply_config(args)

    info = get_model_info(args.model_key)
    if not info:
        raise CliError(
            f"Unknown model key: {args.model_key}",
            code="unknown_model_key",
            exit_code=EXIT_USAGE,
        )

    _, get_model_max_length, _, _ = _lazy_api()

    max_length = get_model_max_length(args.model_key, config=config)

    payload = {
        "model_id": info.model_id,
        "display_name": info.display_name,
        "category": info.category,
        "specialization": info.specialization,
        "description": info.description,
        "entity_types": info.entity_types,
        "size_category": info.size_category,
        "recommended_confidence": info.recommended_confidence,
        "size_mb": info.size_mb,
    }
    if max_length is not None:
        payload["max_length"] = max_length
    return emit(args, payload, human=json.dumps(payload, indent=2))


def _handle_models_verify(args: argparse.Namespace) -> int:
    if (args.model_id is None) == (not args.all_models):
        sys.stderr.write("Provide MODEL_ID or --all, but not both.\n")
        return 2

    config = _load_and_apply_config(args)
    try:
        results = verify_cached_models(
            cache_dir=str(config.cache_dir),
            model_id=None if args.all_models else args.model_id,
        )
    except ModelIntegrityError as exc:
        sys.stdout.write("model_id  status  expected  actual  files\n")
        sys.stdout.write(
            f"{exc.model_id}  FAIL  {exc.expected_sha256}  {exc.actual_sha256}  -\n"
        )
        sys.stderr.write(f"{exc}\n")
        return 1
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"Model integrity verification failed: {exc}\n")
        return 1

    sys.stdout.write("model_id  status  expected  actual  files\n")
    for result in results:
        sys.stdout.write(
            f"{result.model_id}  PASS  {result.expected_sha256}  "
            f"{result.actual_sha256}  {result.files_checked}\n"
        )
    if not results:
        sys.stdout.write("No verified model caches found.\n")
    return 0


def _handle_models_freshness(args: argparse.Namespace) -> int:
    from openmed.eval.fleet_metrics import (
        MEDIAN_AGE_TARGET_DAYS,
        compute_fleet_freshness_from_manifest,
        write_fleet_freshness_artifact,
    )

    manifest_path = args.manifest
    target_days = (
        args.target_days if args.target_days is not None else MEDIAN_AGE_TARGET_DAYS
    )
    try:
        if manifest_path is None:
            metrics = compute_fleet_freshness_from_manifest(
                as_of=args.as_of,
                median_age_target_days=target_days,
            )
        else:
            metrics = compute_fleet_freshness_from_manifest(
                manifest_path,
                as_of=args.as_of,
                median_age_target_days=target_days,
            )
    except (OSError, ValueError) as exc:
        raise CliError(
            f"Failed to compute fleet freshness metrics: {exc}",
            code="compute_failed",
            exit_code=EXIT_ERROR,
        )

    data = json.loads(metrics.to_json())
    if args.output:
        try:
            write_fleet_freshness_artifact(
                metrics,
                args.output,
                output_format=args.artifact_format,
            )
        except OSError as exc:
            raise CliError(
                f"Failed to write metrics artifact: {exc}",
                code="write_failed",
                exit_code=EXIT_ERROR,
            )
        return emit(
            args,
            data,
            human=f"Fleet freshness metrics written to: {args.output}",
        )

    if args.artifact_format == "json":
        human = metrics.to_json()
    else:
        human = metrics.to_markdown()
    return emit(args, data, human=human)


def _handle_models_diff(args: argparse.Namespace) -> int:
    try:
        diff = diff_manifests(args.old_manifest, args.new_manifest)
    except (OSError, ValueError) as exc:
        raise CliError(
            f"Failed to diff manifests: {exc}",
            code="diff_failed",
            exit_code=EXIT_ERROR,
        )

    payload = diff.to_dict()
    emit(args, payload, human=_format_manifest_diff(diff))
    return 1 if args.fail_on_removed and diff.has_removed else 0


def _format_manifest_diff(diff: ManifestDiff) -> str:
    lines = [
        "Manifest diff",
        f"Added: {len(diff.added)}",
        f"Removed: {len(diff.removed)}",
        f"Changed: {len(diff.changed)}",
    ]

    if diff.added:
        lines.extend(["", "Added repos:"])
        lines.extend(f"  + {repo_id}" for repo_id in diff.added)

    if diff.removed:
        lines.extend(["", "Removed repos:"])
        lines.extend(f"  - {repo_id}" for repo_id in diff.removed)

    if diff.changed:
        lines.extend(["", "Changed repos:"])
        for repo_change in diff.changed:
            lines.append(f"  * {repo_change.repo_id}")
            for field, change in repo_change.changes.items():
                lines.append(
                    f"    - {field}: "
                    f"{_format_diff_value(change.before)} -> "
                    f"{_format_diff_value(change.after)}"
                )

    return "\n".join(lines) + "\n"


def _format_diff_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _handle_models_validate(args: argparse.Namespace) -> int:
    from openmed.core.manifest_schema import (
        MANIFEST_PATH,
        format_manifest_validation,
        validate_manifest_file,
    )

    manifest_path = args.manifest or MANIFEST_PATH
    try:
        result = validate_manifest_file(manifest_path)
    except OSError as exc:
        raise CliError(
            f"Failed to read manifest: {exc}",
            code="read_failed",
            exit_code=EXIT_ERROR,
        )

    lines = list(format_manifest_validation(result))
    if wants_json(args):
        emit(
            args,
            {
                "ok": result.ok,
                "violation_count": len(result.violations),
                "messages": lines,
            },
        )
    else:
        output = sys.stderr if result.violations else sys.stdout
        for line in lines:
            output.write(f"{line}\n")
    return 0 if result.ok else 1


def _handle_doctor(args: argparse.Namespace) -> int:
    from ..core.doctor import run_diagnostics

    results = run_diagnostics()

    has_fail = any(item["status"] == "FAIL" for item in results)

    human_lines: list[str] = []
    for item in results:
        human_lines.append(f"{item['status'][:5]} {item['name']}: {item['details']}")
        if item.get("hint"):
            human_lines.append(f"      Hint: {item['hint']}")

    emit(
        args,
        {"checks": results, "has_failure": has_fail},
        human="\n".join(human_lines),
    )
    return 1 if has_fail else 0


def _handle_benchmark_pii_reid(args: argparse.Namespace) -> int:
    from openmed.eval.attacks.reid import (
        render_reid_leaderboard,
        run_reid_benchmark,
    )

    try:
        report = run_reid_benchmark(
            suite=args.suite or "golden",
            model_name=args.model or "privacy-filter",
            output_json=args.output,
        )
        if args.leaderboard_output is not None:
            args.leaderboard_output.write_text(
                render_reid_leaderboard(
                    [report],
                    output_format=args.leaderboard_format,
                ),
                encoding="utf-8",
            )
    except Exception as exc:
        raise CliError(
            f"PII benchmark failed: {exc}",
            code="benchmark_failed",
            exit_code=EXIT_ERROR,
        )

    return emit(args, json.loads(report.to_json()), human=report.to_json())


def _handle_config_show(args: argparse.Namespace) -> int:
    config_path = resolve_config_path(getattr(args, "config_path", None))
    profile_name = getattr(args, "profile", None)

    try:
        config = load_config_from_file(config_path)
        source = str(config_path)
    except FileNotFoundError:
        config = get_config()
        source = "defaults (not yet saved)"

    # Apply profile if specified
    if profile_name:
        try:
            config = config.with_profile(profile_name)
            source = f"{source} (with profile: {profile_name})"
        except ValueError as e:
            raise CliError(str(e), code="invalid_profile", exit_code=EXIT_USAGE)

    payload = config.to_dict()
    payload["_source"] = source
    return emit(args, payload, human=json.dumps(payload, indent=2))


def _handle_config_set(args: argparse.Namespace) -> int:
    key = args.key
    unset = args.unset
    value = args.value

    config_path = resolve_config_path(getattr(args, "config_path", None))

    try:
        config = load_config_from_file(config_path)
    except FileNotFoundError:
        config = get_config()

    config_dict = config.to_dict()

    if key not in config_dict:
        raise CliError(
            f"Unknown configuration key: {key}. "
            f"Valid keys: {', '.join(sorted(config_dict.keys()))}",
            code="unknown_key",
            exit_code=EXIT_USAGE,
        )

    if unset:
        new_value: Any = None
    else:
        if value is None:
            raise CliError(
                "Value is required unless --unset is provided.",
                code="missing_value",
                exit_code=EXIT_USAGE,
            )
        try:
            new_value = _coerce_value(key, value)
        except ValueError as exc:
            raise CliError(str(exc), code="invalid_value", exit_code=EXIT_USAGE)

    config_dict[key] = new_value
    updated_config = OpenMedConfig.from_dict(config_dict)
    set_config(updated_config)
    saved_path = save_config_to_file(updated_config, config_path)

    payload = {"key": key, "value": new_value, "path": str(saved_path)}
    return emit(args, payload, human=f"Updated {key} -> {new_value} in {saved_path}")


def _coerce_value(key: str, value: str) -> Any:
    if key == "timeout":
        try:
            return int(value)
        except ValueError:
            raise ValueError("timeout must be an integer") from None
    return value


# ---------------------------------------------------------------------------
# Policy Handlers
# ---------------------------------------------------------------------------


def _handle_policy_lint(args: argparse.Namespace) -> int:
    from ..core.policy_lint import lint_policy

    report = lint_policy(args.target)
    emit(args, report, human=json.dumps(report, indent=2, sort_keys=True))
    if report["errors"]:
        return 1
    if args.strict and report["warnings"]:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Profile Handlers
# ---------------------------------------------------------------------------


def _handle_profile_list(args: argparse.Namespace) -> int:
    profiles = list_profiles()

    human_lines = ["Available profiles:"]
    profile_entries = []
    for profile in profiles:
        builtin = profile in PROFILE_PRESETS
        marker = " (built-in)" if builtin else " (custom)"
        human_lines.append(f"  - {profile}{marker}")
        profile_entries.append({"name": profile, "builtin": builtin})

    human_lines.append("")
    human_lines.append(f"Total: {len(profiles)} profiles")
    human_lines.append("")
    human_lines.append("Use 'openmed config profile-show <name>' to view settings.")

    payload = {"profiles": profile_entries, "count": len(profiles)}
    return emit(args, payload, human="\n".join(human_lines))


def _handle_profile_show(args: argparse.Namespace) -> int:
    profile_name = args.profile_name

    try:
        settings = get_profile(profile_name)
    except ValueError as e:
        raise CliError(str(e), code="unknown_profile", exit_code=EXIT_USAGE)

    builtin = profile_name in PROFILE_PRESETS
    marker = "(built-in)" if builtin else "(custom)"
    human = f"Profile: {profile_name} {marker}\n{json.dumps(settings, indent=2)}"
    payload = {"name": profile_name, "builtin": builtin, "settings": settings}
    return emit(args, payload, human=human)


def _handle_profile_use(args: argparse.Namespace) -> int:
    profile_name = args.profile_name
    config_path = resolve_config_path(getattr(args, "config_path", None))

    try:
        config = load_config_from_file(config_path)
    except FileNotFoundError:
        config = get_config()

    try:
        new_config = config.with_profile(profile_name)
    except ValueError as e:
        raise CliError(str(e), code="unknown_profile", exit_code=EXIT_USAGE)

    set_config(new_config)
    saved_path = save_config_to_file(new_config, config_path)

    payload = {"profile": profile_name, "path": str(saved_path)}
    return emit(
        args, payload, human=f"Applied profile '{profile_name}' to {saved_path}"
    )


def _handle_profile_save(args: argparse.Namespace) -> int:
    profile_name = args.profile_name
    config_path = resolve_config_path(getattr(args, "config_path", None))

    # Cannot overwrite built-in profiles
    if profile_name in PROFILE_PRESETS:
        raise CliError(
            f"Cannot overwrite built-in profile: {profile_name}",
            code="builtin_profile",
            exit_code=EXIT_USAGE,
        )

    try:
        config = load_config_from_file(config_path)
    except FileNotFoundError:
        config = get_config()

    # Get settings without profile-specific keys
    settings = config.to_dict()
    settings.pop("profile", None)  # Don't save profile reference

    saved_path = save_profile(profile_name, settings)
    payload = {"profile": profile_name, "path": str(saved_path)}
    return emit(args, payload, human=f"Saved profile '{profile_name}' to {saved_path}")


def _handle_profile_delete(args: argparse.Namespace) -> int:
    profile_name = args.profile_name

    try:
        deleted = delete_profile(profile_name)
    except ValueError as e:
        raise CliError(str(e), code="invalid_profile", exit_code=EXIT_USAGE)

    if not deleted:
        raise CliError(
            f"Profile not found: {profile_name}",
            code="profile_not_found",
            exit_code=EXIT_ERROR,
        )

    payload = {"profile": profile_name, "deleted": True}
    return emit(args, payload, human=f"Deleted profile: {profile_name}")


# ---------------------------------------------------------------------------
# PII Handlers
# ---------------------------------------------------------------------------


def _read_text_input(input_path: str) -> str:
    if input_path == "-":
        return sys.stdin.read()
    return Path(input_path).read_text(encoding="utf-8")


def _write_text_output(text: str, output_path: str) -> None:
    if output_path == "-":
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
        return

    path = Path(output_path)
    path.write_text(text, encoding="utf-8")


def _write_audit_report(report: Any, output_path: str) -> Path:
    payload = report.to_json()
    if output_path == "-":
        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            encoding="utf-8",
            prefix="openmed-deid-audit-",
            suffix=".json",
        ) as handle:
            handle.write(payload)
            handle.write("\n")
            return Path(handle.name)

    path = Path(output_path)
    path.write_text(f"{payload}\n", encoding="utf-8")
    return path


def _handle_deid(args: argparse.Namespace) -> int:
    """Handle the top-level de-identification command."""
    from ..core.pii import deidentify

    config = _load_and_apply_config(args)

    try:
        text = _read_text_input(args.input)
    except FileNotFoundError:
        raise CliError(
            f"Input file not found: {args.input}",
            code="input_not_found",
            exit_code=EXIT_ERROR,
        )

    try:
        result = deidentify(
            text,
            method=args.method,
            model_name=args.model,
            confidence_threshold=args.confidence_threshold,
            keep_year=args.keep_year,
            keep_mapping=args.keep_mapping,
            config=config,
            policy=args.policy,
            audit=args.audit,
        )
    except ValueError as exc:
        raise CliError(str(exc), code="invalid_argument", exit_code=EXIT_USAGE)

    if args.audit:
        audit_path = _write_audit_report(result, args.output)
        return emit(args, {"audit_report": str(audit_path)}, human=str(audit_path))

    payload = {"deidentified_text": result.deidentified_text, "output": args.output}
    if args.output == "-":
        return emit(args, payload, human=result.deidentified_text)
    _write_text_output(result.deidentified_text, args.output)
    return emit(args, payload, human=None)


def _handle_pii_extract(args: argparse.Namespace) -> int:
    """Handle PII extraction command."""
    from ..core.pii import extract_pii

    config = _load_and_apply_config(args)

    if args.text:
        text = args.text
    else:
        try:
            text = args.input_file.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise CliError(
                f"Input file not found: {args.input_file}",
                code="input_not_found",
                exit_code=EXIT_ERROR,
            )

    result = extract_pii(
        text,
        model_name=args.model,
        confidence_threshold=args.confidence_threshold,
        config=config,
    )

    output = {
        "text": text,
        "model": args.model,
        "entities": [
            {
                "text": e.text,
                "label": e.label,
                "start": e.start,
                "end": e.end,
                "confidence": float(e.confidence) if e.confidence else None,
            }
            for e in result.entities
        ],
        "num_entities": len(result.entities),
    }

    if args.output:
        args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
        human = f"Results written to: {args.output}"
    else:
        human = json.dumps(output, indent=2)

    return emit(args, output, human=human)


def _handle_pii_deidentify(args: argparse.Namespace) -> int:
    """Handle PII de-identification command."""
    from ..core.pii import deidentify

    config = _load_and_apply_config(args)

    if args.text:
        text = args.text
    else:
        try:
            text = args.input_file.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise CliError(
                f"Input file not found: {args.input_file}",
                code="input_not_found",
                exit_code=EXIT_ERROR,
            )

    result = deidentify(
        text,
        method=args.method,
        model_name=args.model,
        confidence_threshold=args.confidence_threshold,
        keep_year=args.keep_year,
        shift_dates=args.shift_dates,
        keep_mapping=args.keep_mapping,
        config=config,
    )

    num_entities = len(result.pii_entities)
    payload = {
        "deidentified_text": result.deidentified_text,
        "num_entities": num_entities,
        "output": str(args.output) if args.output else None,
    }

    if args.output:
        args.output.write_text(result.deidentified_text, encoding="utf-8")
        human = (
            f"De-identified text written to: {args.output}\n"
            f"Redacted {num_entities} PII entities"
        )
        return emit(args, payload, human=human)

    code = emit(args, payload, human=result.deidentified_text)
    sys.stderr.write(f"\n[Redacted {num_entities} entities]\n")
    return code


def _handle_pii_batch(args: argparse.Namespace) -> int:
    """Handle batch PII de-identification command."""
    config = _load_and_apply_config(args)

    if not args.input_dir.is_dir():
        raise CliError(
            f"Not a directory: {args.input_dir}",
            code="not_a_directory",
            exit_code=EXIT_ERROR,
        )
    if args.checkpoint_interval < 1:
        raise CliError(
            "--checkpoint-interval must be positive",
            code="invalid_argument",
            exit_code=EXIT_USAGE,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.recursive:
        files = sorted(args.input_dir.rglob(args.pattern))
    else:
        files = sorted(args.input_dir.glob(args.pattern))

    if not files:
        raise CliError(
            f"No files found matching pattern: {args.pattern}",
            code="no_files",
            exit_code=EXIT_ERROR,
        )

    json_mode = wants_json(args)
    checkpoint_path = args.checkpoint_path or (
        args.output_dir / ".openmed-batch.checkpoint.json"
    )
    _, _, _, BatchProcessor = _lazy_api()
    processor = BatchProcessor(
        model_name=args.model,
        operation="deidentify",
        config=config,
        confidence_threshold=args.confidence_threshold,
        checkpoint_interval=args.checkpoint_interval,
        method=args.method,
    )

    def progress_callback(current: int, total: int, item_result: Any) -> None:
        if json_mode:
            return
        if item_result and item_result.success:
            result_value = item_result.result
            if isinstance(result_value, MappingABC):
                entities = result_value.get("pii_entities", [])
            else:
                entities = getattr(result_value, "pii_entities", [])
            sys.stdout.write(
                f"[{current}/{total}] {item_result.id}: "
                f"{len(entities)} entities redacted\n"
            )
        else:
            item_id = item_result.id if item_result else "?"
            sys.stderr.write(f"[{current}/{total}] {item_id}: failed\n")

    try:
        result = processor.process_files_to_directory(
            files,
            input_root=args.input_dir,
            output_dir=args.output_dir,
            checkpoint_path=checkpoint_path,
            resume_from_checkpoint=args.resume,
            progress_callback=progress_callback,
        )
    except Exception as exc:
        raise CliError(
            f"Batch processing failed: {exc}",
            code="batch_failed",
            exit_code=EXIT_ERROR,
        )

    payload = result.to_dict()
    payload["output_dir"] = str(args.output_dir)
    human = (
        f"\nProcessed {result.successful_items} files, "
        f"{result.failed_items} failed\n"
        f"Output directory: {args.output_dir}"
    )
    emit(args, payload, human=human)
    return 0 if result.failed_items == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
