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
)
from ..core.model_registry import get_model_info


def _lazy_api():
    from .. import analyze_text, get_model_max_length, list_models

    return analyze_text, get_model_max_length, list_models

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
    analyze_parser.set_defaults(handler=_handle_analyze)


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


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point invoked by the console script."""
    parser = build_parser()
    args = parser.parse_args(argv)

    handler: Optional[Handler] = getattr(args, "handler", None)

    if handler is None:
        # No subcommand provided; show help and hint at interactive mode.
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
        return get_config()


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

    analyze_text, _, _ = _lazy_api()

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


def _handle_models_list(args: argparse.Namespace) -> int:
    config = _load_and_apply_config(args)

    _, _, list_models = _lazy_api()

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

    _, get_model_max_length, _ = _lazy_api()

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
    try:
        config = load_config_from_file(config_path)
        source = str(config_path)
    except FileNotFoundError:
        config = get_config()
        source = "defaults (not yet saved)"

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


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
