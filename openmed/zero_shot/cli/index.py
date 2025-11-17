"""Command-line interface for building the zero-shot model index.

Usage:
    python -m openmed.zero_shot.cli.index --models-dir /path/to/models
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from openmed.ner.indexing import ModelIndex, build_index, write_index

_MODELS_DIR_ENV = "OPENMED_ZEROSHOT_MODELS_DIR"


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m openmed.zero_shot.cli.index",
        description="Build or refresh the zero-shot model metadata index.",
    )
    parser.add_argument(
        "--models-dir",
        help=(
            "Directory containing model artefacts. "
            "Defaults to the $OPENMED_ZEROSHOT_MODELS_DIR environment variable."
        ),
    )
    parser.add_argument(
        "--output",
        help=(
            "Destination file for the generated index. "
            "Defaults to <models-dir>/index.json."
        ),
    )
    parser.add_argument(
        "--no-pretty",
        action="store_false",
        dest="pretty",
        help="Write compact JSON instead of pretty-printed output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover models and print the JSON payload without writing any files.",
    )
    parser.set_defaults(pretty=True)
    return parser.parse_args(argv)


def _resolve_models_dir(arg_value: Optional[str]) -> Path:
    if arg_value:
        return Path(arg_value).expanduser()
    env_value = os.getenv(_MODELS_DIR_ENV)
    if env_value:
        return Path(env_value).expanduser()
    raise SystemExit(
        "Models directory not provided. Use --models-dir or set "
        f"{_MODELS_DIR_ENV}."
    )


def _resolve_output_path(arg_value: Optional[str], models_dir: Path) -> Path:
    if arg_value:
        return Path(arg_value).expanduser()
    return models_dir / "index.json"


def _print_summary(index: ModelIndex) -> None:
    domain_preview = ", ".join(sorted(index.unique_domains)) or "(none)"
    sys.stdout.write(
        f"Discovered {len(index.models)} model(s) across "
        f"{len(index.unique_domains)} domain(s): {domain_preview}\n"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    models_dir = _resolve_models_dir(args.models_dir)
    output_path = _resolve_output_path(args.output, models_dir)

    index = build_index(models_dir)

    if args.dry_run:
        payload = index.to_dict()
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return 0

    write_index(index, output_path, pretty=args.pretty)
    _print_summary(index)
    sys.stdout.write(f"Index written to {output_path}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation only
    raise SystemExit(main())
