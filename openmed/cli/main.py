"""Command-line interface for the OpenMed toolkit."""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import sys
import tempfile
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
from ..core.manifest_diff import ManifestDiff, diff_manifests
from ..core.model_card import render_model_card
from ..core.model_registry import MANIFEST_PATH, get_model_info, load_manifest_rows
from ..core.model_search import ModelSearchResult, recommend_models, search_models
from ..core.policy import CANONICAL_POLICY_NAMES, canonical_policy_name
from .active_learning import add_active_learning_command
from .calibrate import add_calibrate_command
from .gates import add_gates_command

_ANALYZE_TEXT = None
_GET_MODEL_MAX_LENGTH = None
_LIST_MODELS = None
_BATCH_PROCESSOR = None

_AUDIT_KEY_ENV = "OPENMED_AUDIT_KEY"

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

COMPLIANCE_CAVEAT = (
    "No de-identification tool can guarantee compliance or zero residual risk. "
    "Validate locally before any production or clinical use."
)
_FHIR_BUNDLE_TYPES = frozenset({"transaction", "batch"})

_DEFAULT_PII_MODEL = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"
_DEID_METHODS = ("mask", "remove", "replace", "hash", "shift_dates")


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be greater than or equal to 0")
    return parsed


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
    _add_pii_command(subparsers)
    _add_audit_command(subparsers)
    _add_risk_command(subparsers)
    _add_policy_command(subparsers)
    _add_fhir_command(subparsers)
    _add_benchmark_command(subparsers)
    _add_eval_command(subparsers)
    _add_models_command(subparsers)
    _add_config_command(subparsers)
    add_active_learning_command(subparsers)
    _add_doctor_command(subparsers)
    add_calibrate_command(subparsers)
    add_gates_command(subparsers)
    return parser


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
    batch_parser.set_defaults(handler=_handle_batch)


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
    batch_parser.set_defaults(handler=_handle_pii_batch)


def _add_audit_command(subparsers: argparse._SubParsersAction) -> None:
    audit_parser = subparsers.add_parser(
        "audit",
        help="Inspect and verify PHI-safe de-identification audit reports.",
    )
    audit_sub = audit_parser.add_subparsers(dest="audit_command")

    verify_parser = audit_sub.add_parser(
        "verify",
        help="Verify an audit report's reproducibility hash and signature.",
    )
    verify_parser.add_argument(
        "report",
        type=Path,
        help="Path to a signed audit report JSON file.",
    )
    verify_parser.add_argument(
        "--key",
        default=None,
        help=f"HMAC key for signed reports. Defaults to {_AUDIT_KEY_ENV}.",
    )
    verify_parser.set_defaults(handler=_handle_audit_verify)

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


def _add_models_command(subparsers: argparse._SubParsersAction) -> None:
    models_parser = subparsers.add_parser("models", help="Discover OpenMed models.")
    models_sub = models_parser.add_subparsers(dest="models_command")

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
    models_recommend.add_argument(
        "--json",
        action="store_true",
        help="Emit the ranked shortlist as a single JSON document.",
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
        "--json",
        action="store_true",
        help="Emit the manifest diff as a single JSON document.",
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

    doctor_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit diagnostics as JSON.",
    )

    doctor_parser.set_defaults(
        handler=_handle_doctor,
    )


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

    return handler(args)


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
            sys.stderr.write(f"Input file not found: {args.input_file}\n")
            return 1
        except OSError as exc:  # pragma: no cover - defensive
            sys.stderr.write(f"Failed to read {args.input_file}: {exc}\n")
            return 1

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
        output = result
    elif hasattr(result, "to_dict"):
        output = json.dumps(result.to_dict(), indent=2)
    else:
        output = json.dumps(result, indent=2)

    sys.stdout.write(f"{output}\n")
    return 0


def _handle_batch(args: argparse.Namespace) -> int:
    config = _load_and_apply_config(args)

    _, _, _, BatchProcessor = _lazy_api()

    continue_on_error = not args.stop_on_error if args.stop_on_error else True

    processor = BatchProcessor(
        model_name=args.model,
        config=config,
        confidence_threshold=args.confidence_threshold or 0.0,
        group_entities=args.group_entities,
        continue_on_error=continue_on_error,
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
            )
        elif args.input_files:
            result = processor.process_files(
                args.input_files,
                progress_callback=progress_callback if not args.quiet else None,
            )
        elif args.input_dir:
            if not args.input_dir.is_dir():
                sys.stderr.write(f"Not a directory: {args.input_dir}\n")
                return 1
            result = processor.process_directory(
                args.input_dir,
                pattern=args.pattern,
                recursive=args.recursive,
                progress_callback=progress_callback if not args.quiet else None,
            )
        else:
            sys.stderr.write("No input provided.\n")
            return 1

    except Exception as exc:
        sys.stderr.write(f"\nBatch processing failed: {exc}\n")
        return 1

    if not args.quiet:
        sys.stderr.write("\n")

    if args.output_format == "json":
        output = json.dumps(result.to_dict(), indent=2)
    else:
        output = result.summary()

    if args.output:
        try:
            args.output.write_text(
                json.dumps(result.to_dict(), indent=2)
                if args.output_format == "json"
                else output,
                encoding="utf-8",
            )
            sys.stdout.write(f"Results written to: {args.output}\n")
        except OSError as exc:
            sys.stderr.write(f"Failed to write output: {exc}\n")
            return 1
    else:
        sys.stdout.write(f"{output}\n")

    return 0 if result.failed_items == 0 else 1


def _handle_audit_verify(args: argparse.Namespace) -> int:
    try:
        report = _load_audit_report(args.report)
    except (OSError, TypeError, ValueError) as exc:
        sys.stderr.write(f"Failed to load audit report: {exc}\n")
        return 1

    repro_ok = report.repro_hash_matches()
    signature_status = "SKIPPED (report is unsigned)"
    signature_ok = True

    if report.signature is not None:
        key = args.key or os.environ.get(_AUDIT_KEY_ENV)
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
    sys.stdout.write(f"Audit report verification: {_pass_fail(verified)}\n")
    sys.stdout.write(f"Reproducibility hash: {_pass_fail(repro_ok)}\n")
    sys.stdout.write(f"HMAC signature: {signature_status}\n")
    return 0 if verified else 1


def _handle_audit_show(args: argparse.Namespace) -> int:
    try:
        report = _load_audit_report(args.report)
    except (OSError, TypeError, ValueError) as exc:
        sys.stderr.write(f"Failed to load audit report: {exc}\n")
        return 1

    sys.stdout.write(_format_audit_summary(report))
    return 0


def _load_audit_report(path: Path):
    from ..core.audit import AuditReport

    return AuditReport.from_json(path.read_text(encoding="utf-8"))


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
    from ..risk import risk_report

    try:
        text = _read_text_input(args.input)
    except OSError as exc:
        sys.stderr.write(f"Failed to read text input: {exc}\n")
        return 1

    report = risk_report(text)
    sys.stdout.write(_format_risk_summary("Text risk summary", report))
    return 0


def _handle_risk_table(args: argparse.Namespace) -> int:
    from ..risk import risk_report

    try:
        records = _read_csv_records(args.csv)
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"Failed to read table input: {exc}\n")
        return 1

    report = risk_report(records)
    sys.stdout.write(_format_risk_summary("Table risk summary", report))
    return 0


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
        sys.stderr.write(f"--type must be one of: {allowed}\n")
        return 2

    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        resources = _extract_fhir_resources(payload)
        doc_id = _extract_fhir_doc_id(payload)

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
        sys.stderr.write(f"Input file not found: {args.input}\n")
        return 1
    except json.JSONDecodeError as exc:
        sys.stderr.write(
            f"Invalid JSON in {args.input}: {exc.msg} "
            f"at line {exc.lineno} column {exc.colno}\n"
        )
        return 1
    except OSError as exc:
        sys.stderr.write(f"Failed to read or write FHIR Bundle: {exc}\n")
        return 1
    except (TypeError, ValueError) as exc:
        sys.stderr.write(f"Failed to assemble FHIR Bundle: {exc}\n")
        return 1

    sys.stdout.write(f"FHIR Bundle written to: {args.output}\n")
    return 0


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
        sys.stderr.write(f"Policy diff failed: {exc}\n")
        return 1

    if args.output_format == "json":
        payload = render(diff, fmt="dict")
        sys.stdout.write(f"{json.dumps(payload, indent=2, sort_keys=True)}\n")
    else:
        sys.stdout.write(f"{render(diff, fmt='text')}\n")
    return 0


def _handle_eval_load_test(args: argparse.Namespace) -> int:
    """Run the in-process ASGI load test and print its report."""
    from openmed.eval.load_test import run_load_test
    from openmed.service.app import app

    report = run_load_test(
        app,
        concurrency=args.concurrency,
        total_requests=args.total_requests,
    )
    print(json.dumps(vars(report), indent=2))
    return 0


def _handle_benchmark_pii(args: argparse.Namespace) -> int:
    if args.attack == "reid":
        return _handle_benchmark_pii_reid(args)

    from openmed.eval.harness import run_benchmark
    from openmed.eval.suites import SHIELD, load_suite_fixtures, suite_metadata

    models = _parse_model_args(args.models or [])
    if not models:
        sys.stderr.write("At least one model identifier is required.\n")
        return 1

    suite = str(args.suite or SHIELD)
    try:
        if suite == SHIELD:
            use_sample = not bool(args.full_shield)
            fixtures = load_suite_fixtures(suite, use_sample=use_sample)
            metadata = suite_metadata(suite, use_sample=use_sample)
        else:
            fixtures = load_suite_fixtures(suite)
            metadata = suite_metadata(suite)
    except (PermissionError, RuntimeError, ValueError) as exc:
        sys.stderr.write(f"Failed to load benchmark suite: {exc}\n")
        return 1

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

    output = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        try:
            args.output.write_text(output + "\n", encoding="utf-8")
        except OSError as exc:
            sys.stderr.write(f"Failed to write benchmark output: {exc}\n")
            return 1
    else:
        sys.stdout.write(output + "\n")
    return 0


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
        sys.stderr.write(f"Failed to load clinical benchmark suite: {exc}\n")
        return 1

    if suite == BIOMEDICAL_NER and task == "ner":
        split = str(args.split) if args.split is not None else "test"
        models = _parse_model_args(args.models or [])
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
        return _write_json_payload(payload, args.output)

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

    return _write_json_payload(payload, args.output)


def _write_json_payload(payload: Any, output_path: Path | None) -> int:
    output = json.dumps(payload, indent=2, sort_keys=True)
    if output_path:
        try:
            output_path.write_text(output + "\n", encoding="utf-8")
        except OSError as exc:
            sys.stderr.write(f"Failed to write benchmark output: {exc}\n")
            return 1
    else:
        sys.stdout.write(output + "\n")
    return 0


def _parse_model_args(values: Sequence[str]) -> list[str]:
    models: list[str] = []
    for value in values:
        models.extend(item.strip() for item in value.split(",") if item.strip())
    return models


def _handle_models_search(args: argparse.Namespace) -> int:
    if (
        args.min_params is not None
        and args.max_params is not None
        and args.min_params > args.max_params
    ):
        sys.stderr.write("--min-params must be less than or equal to --max-params\n")
        return 2

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
        sys.stderr.write(f"Failed to search models: {exc}\n")
        return 1

    if not results:
        sys.stderr.write("No models matched the search filters.\n")
        return 1

    sys.stdout.write(_format_model_search_table(results))
    return 0


def _handle_models_recommend(args: argparse.Namespace) -> int:
    try:
        results = recommend_models(
            device_tier=args.tier,
            task=args.task,
            language=args.language,
        )
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"Failed to recommend models: {exc}\n")
        return 1

    if not results:
        sys.stderr.write(
            f"No model fits the '{args.tier}' device tier for the requested filters.\n"
        )
        return 1

    if args.json:
        payload = {
            "tier": args.tier,
            "task": args.task,
            "language": args.language,
            "recommended": results[0].repo_id,
            "models": [_recommendation_to_dict(result) for result in results],
        }
        sys.stdout.write(f"{json.dumps(payload, indent=2)}\n")
        return 0

    sys.stdout.write(f"Recommended for {args.tier}: {results[0].repo_id}\n")
    sys.stdout.write(_format_model_search_table(results))
    return 0


def _handle_models_card(args: argparse.Namespace) -> int:
    try:
        row = _find_manifest_row(args.repo_id)
        rendered = render_model_card(dict(row))
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"Failed to render model card: {exc}\n")
        return 1

    if args.check is not None:
        try:
            existing = args.check.read_text(encoding="utf-8")
        except OSError as exc:
            sys.stderr.write(f"Failed to read README for comparison: {exc}\n")
            return 1

        if existing == rendered:
            return 0

        diff = difflib.unified_diff(
            existing.splitlines(keepends=True),
            rendered.splitlines(keepends=True),
            fromfile=str(args.check),
            tofile=f"rendered:{args.repo_id}",
        )
        sys.stdout.writelines(diff)
        return 1

    if args.output is not None:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        except OSError as exc:
            sys.stderr.write(f"Failed to write model card: {exc}\n")
            return 1
        return 0

    sys.stdout.write(rendered)
    return 0


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
        sys.stderr.write(f"Failed to list models: {exc}\n")
        return 1

    for model in models:
        sys.stdout.write(f"{model}\n")
    return 0


def _handle_models_info(args: argparse.Namespace) -> int:
    config = _load_and_apply_config(args)

    info = get_model_info(args.model_key)
    if not info:
        sys.stderr.write(f"Unknown model key: {args.model_key}\n")
        return 1

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
    sys.stdout.write(f"{json.dumps(payload, indent=2)}\n")
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
        sys.stderr.write(f"Failed to compute fleet freshness metrics: {exc}\n")
        return 1

    if args.output:
        try:
            write_fleet_freshness_artifact(
                metrics,
                args.output,
                output_format=args.artifact_format,
            )
        except OSError as exc:
            sys.stderr.write(f"Failed to write metrics artifact: {exc}\n")
            return 1
        sys.stdout.write(f"Fleet freshness metrics written to: {args.output}\n")
        return 0

    if args.artifact_format == "json":
        sys.stdout.write(f"{metrics.to_json()}\n")
    else:
        sys.stdout.write(metrics.to_markdown())
    return 0


def _handle_models_diff(args: argparse.Namespace) -> int:
    try:
        diff = diff_manifests(args.old_manifest, args.new_manifest)
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"Failed to diff manifests: {exc}\n")
        return 1

    if args.json:
        sys.stdout.write(f"{json.dumps(diff.to_dict(), indent=2, sort_keys=True)}\n")
    else:
        sys.stdout.write(_format_manifest_diff(diff))

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
        sys.stderr.write(f"Failed to read manifest: {exc}\n")
        return 1

    output = sys.stderr if result.violations else sys.stdout
    for line in format_manifest_validation(result):
        output.write(f"{line}\n")
    return 0 if result.ok else 1


def _handle_doctor(args: argparse.Namespace) -> int:
    from ..core.doctor import run_diagnostics

    results = run_diagnostics()

    if args.json:
        sys.stdout.write(json.dumps(results, indent=2))
        sys.stdout.write("\n")

        has_fail = any(item["status"] == "FAIL" for item in results)

        return 1 if has_fail else 0

    for item in results:
        sys.stdout.write(f"{item['status'][:5]} {item['name']}: {item['details']}\n")

        if item.get("hint"):
            sys.stdout.write(f"      Hint: {item['hint']}\n")

    has_fail = any(item["status"] == "FAIL" for item in results)

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
        sys.stderr.write(f"PII benchmark failed: {exc}\n")
        return 1

    sys.stdout.write(report.to_json() + "\n")
    return 0


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
            sys.stderr.write(f"{e}\n")
            return 1

    payload = config.to_dict()
    payload["_source"] = source
    sys.stdout.write(f"{json.dumps(payload, indent=2)}\n")
    return 0


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
        sys.stderr.write(
            f"Unknown configuration key: {key}. "
            f"Valid keys: {', '.join(sorted(config_dict.keys()))}\n"
        )
        return 1

    if unset:
        new_value: Any = None
    else:
        if value is None:
            sys.stderr.write("Value is required unless --unset is provided.\n")
            return 1
        try:
            new_value = _coerce_value(key, value)
        except ValueError as exc:
            sys.stderr.write(f"{exc}\n")
            return 1

    config_dict[key] = new_value
    updated_config = OpenMedConfig.from_dict(config_dict)
    set_config(updated_config)
    saved_path = save_config_to_file(updated_config, config_path)

    sys.stdout.write(f"Updated {key} -> {new_value} in {saved_path}\n")
    return 0


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
    sys.stdout.write(f"{json.dumps(report, indent=2, sort_keys=True)}\n")
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

    sys.stdout.write("Available profiles:\n")
    for profile in profiles:
        marker = " (built-in)" if profile in PROFILE_PRESETS else " (custom)"
        sys.stdout.write(f"  - {profile}{marker}\n")

    sys.stdout.write(f"\nTotal: {len(profiles)} profiles\n")
    sys.stdout.write("\nUse 'openmed config profile-show <name>' to view settings.\n")
    return 0


def _handle_profile_show(args: argparse.Namespace) -> int:
    profile_name = args.profile_name

    try:
        settings = get_profile(profile_name)
    except ValueError as e:
        sys.stderr.write(f"{e}\n")
        return 1

    marker = "(built-in)" if profile_name in PROFILE_PRESETS else "(custom)"
    sys.stdout.write(f"Profile: {profile_name} {marker}\n")
    sys.stdout.write(f"{json.dumps(settings, indent=2)}\n")
    return 0


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
        sys.stderr.write(f"{e}\n")
        return 1

    set_config(new_config)
    saved_path = save_config_to_file(new_config, config_path)

    sys.stdout.write(f"Applied profile '{profile_name}' to {saved_path}\n")
    return 0


def _handle_profile_save(args: argparse.Namespace) -> int:
    profile_name = args.profile_name
    config_path = resolve_config_path(getattr(args, "config_path", None))

    # Cannot overwrite built-in profiles
    if profile_name in PROFILE_PRESETS:
        sys.stderr.write(f"Cannot overwrite built-in profile: {profile_name}\n")
        return 1

    try:
        config = load_config_from_file(config_path)
    except FileNotFoundError:
        config = get_config()

    # Get settings without profile-specific keys
    settings = config.to_dict()
    settings.pop("profile", None)  # Don't save profile reference

    saved_path = save_profile(profile_name, settings)
    sys.stdout.write(f"Saved profile '{profile_name}' to {saved_path}\n")
    return 0


def _handle_profile_delete(args: argparse.Namespace) -> int:
    profile_name = args.profile_name

    try:
        deleted = delete_profile(profile_name)
    except ValueError as e:
        sys.stderr.write(f"{e}\n")
        return 1

    if deleted:
        sys.stdout.write(f"Deleted profile: {profile_name}\n")
        return 0
    else:
        sys.stderr.write(f"Profile not found: {profile_name}\n")
        return 1


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
        sys.stderr.write(f"Input file not found: {args.input}\n")
        return 1

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
        sys.stderr.write(f"{exc}\n")
        return 2

    if args.audit:
        audit_path = _write_audit_report(result, args.output)
        sys.stdout.write(f"{audit_path}\n")
        return 0

    _write_text_output(result.deidentified_text, args.output)
    return 0


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
            sys.stderr.write(f"Input file not found: {args.input_file}\n")
            return 1

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
        sys.stdout.write(f"Results written to: {args.output}\n")
    else:
        sys.stdout.write(f"{json.dumps(output, indent=2)}\n")

    return 0


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
            sys.stderr.write(f"Input file not found: {args.input_file}\n")
            return 1

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

    if args.output:
        args.output.write_text(result.deidentified_text, encoding="utf-8")
        sys.stdout.write(f"De-identified text written to: {args.output}\n")
        sys.stdout.write(f"Redacted {len(result.pii_entities)} PII entities\n")
    else:
        sys.stdout.write(f"{result.deidentified_text}\n")
        sys.stderr.write(f"\n[Redacted {len(result.pii_entities)} entities]\n")

    return 0


def _handle_pii_batch(args: argparse.Namespace) -> int:
    """Handle batch PII de-identification command."""
    from ..core.pii import deidentify

    config = _load_and_apply_config(args)

    if not args.input_dir.is_dir():
        sys.stderr.write(f"Not a directory: {args.input_dir}\n")
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find files to process
    if args.recursive:
        files = list(args.input_dir.rglob(args.pattern))
    else:
        files = list(args.input_dir.glob(args.pattern))

    if not files:
        sys.stderr.write(f"No files found matching pattern: {args.pattern}\n")
        return 1

    # Process files
    processed = 0
    failed = 0

    for input_file in files:
        try:
            text = input_file.read_text(encoding="utf-8")

            result = deidentify(
                text,
                method=args.method,
                model_name=args.model,
                confidence_threshold=args.confidence_threshold,
                config=config,
            )

            # Preserve directory structure
            relative_path = input_file.relative_to(args.input_dir)
            output_file = args.output_dir / relative_path
            output_file.parent.mkdir(parents=True, exist_ok=True)

            output_file.write_text(result.deidentified_text, encoding="utf-8")

            processed += 1
            sys.stdout.write(
                f"[{processed}/{len(files)}] {input_file.name}: "
                f"{len(result.pii_entities)} entities redacted\n"
            )

        except Exception as exc:
            failed += 1
            sys.stderr.write(f"Failed to process {input_file}: {exc}\n")

    sys.stdout.write(
        f"\nProcessed {processed} files, {failed} failed\n"
        f"Output directory: {args.output_dir}\n"
    )

    return 0 if failed == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
