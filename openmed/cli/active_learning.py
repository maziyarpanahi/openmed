"""CLI helpers for the active-learning queue runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from openmed.training.active_learning import (
    DEFAULT_EVENT_LOG,
    ActiveLearningQueue,
)


def add_active_learning_command(subparsers: argparse._SubParsersAction) -> None:
    """Register active-learning subcommands on the argparse CLI."""

    parser = subparsers.add_parser(
        "active-learning",
        help="Manage PHI-free active-learning labeling batches.",
    )
    active_sub = parser.add_subparsers(dest="active_learning_command")

    next_batch = active_sub.add_parser(
        "next-batch",
        help="Emit the next PHI-free labeling batch as JSONL.",
    )
    next_batch.add_argument(
        "--size",
        type=_positive_int,
        required=True,
        help="Maximum number of candidates to emit.",
    )
    next_batch.add_argument(
        "--state",
        type=Path,
        default=DEFAULT_EVENT_LOG,
        help=f"Queue event-log JSONL path (default: {DEFAULT_EVENT_LOG}).",
    )
    next_batch.add_argument(
        "--gate-report",
        action="append",
        type=Path,
        default=[],
        help="Release-gate report JSON/JSONL to ingest before batching.",
    )
    next_batch.add_argument(
        "--adjudication",
        action="append",
        type=Path,
        default=[],
        help="Adjudication hook output JSON/JSONL to ingest before batching.",
    )
    next_batch.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optional path to write the JSONL batch.",
    )
    next_batch.set_defaults(handler=handle_next_batch)


def handle_next_batch(args: argparse.Namespace) -> int:
    """Ingest optional sources and emit a bounded labeling batch."""

    queue = ActiveLearningQueue(args.state)
    try:
        for report in _load_records(args.gate_report):
            queue.ingest_gate_report(report)
        for item in _load_records(args.adjudication):
            queue.ingest_adjudication(item)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"Failed to prepare active-learning queue: {exc}\n")
        return 1

    payload = queue.next_batch_jsonl(args.size)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{payload}\n" if payload else "", encoding="utf-8")
    elif payload:
        sys.stdout.write(f"{payload}\n")
    return 0


def _load_records(paths: Iterable[Path]) -> Iterable[Mapping[str, Any]]:
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":
            for line_number, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: invalid JSON: {exc}"
                    ) from exc
                if not isinstance(record, Mapping):
                    raise ValueError(f"{path} contains a non-object JSONL row")
                yield record
            continue

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: invalid JSON: {exc}") from exc
        if isinstance(payload, Mapping):
            yield payload
        elif isinstance(payload, list):
            for record in payload:
                if not isinstance(record, Mapping):
                    raise ValueError(f"{path} contains a non-object JSON item")
                yield record
        else:
            raise ValueError(f"{path} must contain a JSON object or array")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("size must be non-negative")
    return parsed


__all__ = ["add_active_learning_command", "handle_next_batch"]
