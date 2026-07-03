"""CLI handler for dataset redaction."""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from typing import Any, Sequence, TextIO

from openmed.processing.batch import redact_dataset


def parse_text_columns(
    text_columns: str | None,
    repeated_columns: Sequence[str] | None,
) -> list[str]:
    """Parse comma-separated and repeated text-column arguments."""
    columns: list[str] = []
    if text_columns:
        columns.extend(_split_columns(text_columns))
    for value in repeated_columns or ():
        columns.extend(_split_columns(value))
    return columns


def run_from_args(
    args: Namespace,
    *,
    config: Any | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run dataset redaction from argparse arguments."""
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    columns = parse_text_columns(
        getattr(args, "text_columns", None),
        getattr(args, "text_column", None),
    )
    if not columns:
        stderr.write(
            "At least one --text-column or --text-columns value is required.\n"
        )
        return 1

    try:
        result = redact_dataset(
            getattr(args, "path"),
            columns,
            output_path=getattr(args, "output", None),
            policy=getattr(args, "policy", None),
            method=getattr(args, "method", "mask"),
            model_name=getattr(args, "model"),
            confidence_threshold=getattr(args, "confidence_threshold", 0.7),
            config=config,
            lang=getattr(args, "lang", "en"),
            keep_year=getattr(args, "keep_year", True),
            use_safety_sweep=not getattr(args, "no_safety_sweep", False),
            encoding=getattr(args, "encoding", "utf-8"),
            batch_size=getattr(args, "batch_size", 512),
        )
    except Exception as exc:
        stderr.write(f"Dataset redaction failed: {exc}\n")
        return 1

    stdout.write(json.dumps(result.summary.to_dict(), indent=2, sort_keys=True))
    stdout.write("\n")
    return 0


def _split_columns(value: str) -> list[str]:
    return [column.strip() for column in value.split(",") if column.strip()]
