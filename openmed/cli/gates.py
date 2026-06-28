"""Release-gate CLI command wiring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from openmed.core import baseline as baseline_store
from openmed.eval.release_gates import RELEASABLE, format_preview, preview


def add_gates_command(subparsers: argparse._SubParsersAction) -> None:
    """Add release-gate commands to the top-level parser."""

    parser = subparsers.add_parser(
        "gates",
        help="Preview release-gate decisions.",
    )
    gates_sub = parser.add_subparsers(dest="gates_command")

    preview_parser = gates_sub.add_parser(
        "preview",
        help="Preview gate outcomes without signing or writing a report.",
    )
    preview_parser.add_argument(
        "--candidate",
        required=True,
        type=Path,
        help="Path to a candidate BenchmarkReport JSON payload.",
    )
    preview_parser.add_argument(
        "--baseline",
        type=Path,
        help="Optional baseline JSON payload. Defaults to the baseline store.",
    )
    preview_parser.add_argument(
        "--baseline-store",
        type=Path,
        default=baseline_store.BASELINE_PATH,
        help="Path to the last-green baseline store.",
    )
    preview_parser.add_argument(
        "--milestone",
        default="v1.6",
        help="Milestone version used for release thresholds.",
    )
    preview_parser.add_argument(
        "--policy",
        default="hipaa_safe_harbor",
        help="Policy profile used when the candidate report omits one.",
    )
    preview_parser.add_argument(
        "--thresholds-matrix",
        type=Path,
        help="Optional thresholds matrix JSON path.",
    )
    preview_parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any previewed gate fails.",
    )
    preview_parser.set_defaults(handler=handle_gates_preview)


def handle_gates_preview(args: argparse.Namespace) -> int:
    """Run a read-only release-gate preview."""

    if not args.candidate.is_file():
        sys.stderr.write(f"Candidate report not found: {args.candidate}\n")
        return 2

    try:
        candidate = _read_json_file(args.candidate)
        baseline = _read_json_file(args.baseline) if args.baseline else None
        report = preview(
            candidate,
            baseline,
            milestone=args.milestone,
            policy=args.policy,
            baseline_path=args.baseline_store,
            thresholds_matrix_path=args.thresholds_matrix,
        )
    except Exception as exc:
        sys.stderr.write(f"Release gate preview failed: {exc}\n")
        return 2

    sys.stdout.write(format_preview(report) + "\n")
    if args.strict and report.decision != RELEASABLE:
        return 1
    return 0


def _read_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return dict(payload)


__all__ = ["add_gates_command", "handle_gates_preview"]
