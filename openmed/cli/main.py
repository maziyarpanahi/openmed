"""Command-line interface for the OpenMed toolkit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from ..core.config import (
    OpenMedConfig,
    get_config,
    set_config,
    load_config_from_file,
    save_config_to_file,
    resolve_config_path,
    list_profiles,
    get_profile,
    save_profile,
    delete_profile,
    PROFILE_PRESETS,
)
from ..core.model_registry import get_model_info


_ANALYZE_TEXT = None
_GET_MODEL_MAX_LENGTH = None
_LIST_MODELS = None
_BATCH_PROCESSOR = None

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

    if get_model_max_length is not None and get_model_max_length is not _GET_MODEL_MAX_LENGTH:
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


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="openmed",
        description="Command-line utilities for OpenMed medical NLP models.",
    )
    parser.add_argument(
        "--config-path",
        help="Override the configuration file path.",
        default=None,
    )

    subparsers = parser.add_subparsers(dest="command")

    _add_analyze_command(subparsers)
    _add_batch_command(subparsers)
    _add_pii_command(subparsers)
    _add_tui_command(subparsers)
    _add_models_command(subparsers)
    _add_config_command(subparsers)
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
        default="OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1",
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
        choices=["mask", "remove", "replace", "hash", "shift_dates"],
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
    batch_parser = pii_sub.add_parser(
        "batch", help="Batch de-identification of files."
    )
    batch_parser.add_argument(
        "--model",
        default="OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1",
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


def _add_tui_command(subparsers: argparse._SubParsersAction) -> None:
    tui_parser = subparsers.add_parser(
        "tui", help="Launch interactive terminal UI for clinical NER analysis."
    )
    tui_parser.add_argument(
        "--model",
        default=None,
        help="Model registry key or Hugging Face identifier (default: disease_detection_superclinical).",
    )
    tui_parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum confidence score for predictions (default: 0.5).",
    )
    tui_parser.set_defaults(handler=_handle_tui)


def _add_models_command(subparsers: argparse._SubParsersAction) -> None:
    models_parser = subparsers.add_parser(
        "models", help="Discover OpenMed models."
    )
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point invoked by the console script."""
    parser = build_parser()
    args = parser.parse_args(argv)

    handler: Optional[Handler] = getattr(args, "handler", None)

    if handler is None:
        # No subcommand provided; launch TUI directly.
        try:
            from openmed.tui import OpenMedTUI
        except ImportError:
            sys.stderr.write(
                "TUI dependencies not installed. Install with: pip install openmed[tui]\n"
                "\nFor CLI commands, use:\n"
                "  openmed analyze --text \"...\"\n"
                "  openmed batch --input-dir ./notes\n"
                "  openmed models list\n"
                "  openmed config show\n"
                "\nRun 'openmed --help' for more options.\n"
            )
            return 1

        app = OpenMedTUI()
        app.run()
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
    if hasattr(args, "use_medical_tokenizer") and args.use_medical_tokenizer is not None:
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
                json.dumps(result.to_dict(), indent=2) if args.output_format == "json" else output,
                encoding="utf-8",
            )
            sys.stdout.write(f"Results written to: {args.output}\n")
        except OSError as exc:
            sys.stderr.write(f"Failed to write output: {exc}\n")
            return 1
    else:
        sys.stdout.write(f"{output}\n")

    return 0 if result.failed_items == 0 else 1


def _handle_tui(args: argparse.Namespace) -> int:
    try:
        from openmed.tui import OpenMedTUI
    except ImportError:
        sys.stderr.write(
            "TUI dependencies not installed. Install with: pip install openmed[tui]\n"
        )
        return 1

    app = OpenMedTUI(
        model_name=args.model,
        confidence_threshold=args.confidence_threshold,
    )
    app.run()
    return 0


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
