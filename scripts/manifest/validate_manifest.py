#!/usr/bin/env python3
"""Validate the canonical OpenMed model manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence, TextIO

from openmed.core.manifest_schema import (
    format_manifest_validation,
    validate_manifest_file,
)

DEFAULT_MANIFEST = Path("models.jsonl")


def build_parser() -> argparse.ArgumentParser:
    """Build the manifest validator argument parser."""
    parser = argparse.ArgumentParser(
        description="Validate models.jsonl against the canonical OpenMed schema."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to the manifest JSONL file to validate.",
    )
    return parser


def run(
    manifest: Path,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Validate *manifest*, write a human-readable verdict, and return an exit code."""
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    try:
        result = validate_manifest_file(manifest)
    except OSError as exc:
        stderr.write(f"Failed to read manifest: {exc}\n")
        return 1

    output = stderr if result.violations else stdout
    for line in format_manifest_validation(result):
        output.write(f"{line}\n")
    return 0 if result.ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the manifest validator CLI."""
    args = build_parser().parse_args(argv)
    return run(args.manifest)


if __name__ == "__main__":
    raise SystemExit(main())
