"""CLI helpers for release-gate evidence workflows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from openmed.eval.evidence_bundle import bundle_gate_evidence


def add_gates_command(subparsers: argparse._SubParsersAction) -> None:
    """Register ``openmed gates`` subcommands."""
    gates_parser = subparsers.add_parser(
        "gates",
        help="Release gate evidence utilities.",
    )
    gates_sub = gates_parser.add_subparsers(dest="gates_command")

    bundle_parser = gates_sub.add_parser(
        "bundle",
        help="Assemble a release-gate evidence bundle.",
    )
    bundle_parser.add_argument(
        "--gate-report",
        "--report",
        required=True,
        type=Path,
        help="Path to a signed GateReport JSON payload.",
    )
    bundle_parser.add_argument(
        "--output-dir",
        "--output",
        required=True,
        type=Path,
        help="Directory where the evidence bundle should be written.",
    )
    bundle_parser.add_argument(
        "--evidence-root",
        type=Path,
        default=None,
        help=(
            "Base directory for relative evidence paths. Defaults to the "
            "gate report's parent directory."
        ),
    )
    bundle_parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="ID[;GATE,...]=PATH",
        help=(
            "Additional evidence artifact. Known IDs automatically map to "
            "their gates; append ';G1a,G2' to override gates."
        ),
    )
    bundle_parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when required evidence is missing.",
    )
    bundle_parser.set_defaults(handler=_handle_bundle)


def _handle_bundle(args: argparse.Namespace) -> int:
    try:
        payload = _read_json_object(args.gate_report)
        extra_artifacts = [_parse_artifact(value) for value in args.artifact]
        extra_artifacts.append(
            {
                "artifact_id": "gate_report",
                "path": str(args.gate_report.resolve()),
                "required": True,
            }
        )
        result = bundle_gate_evidence(
            payload,
            args.output_dir,
            evidence_root=args.evidence_root or args.gate_report.parent,
            extra_artifacts=extra_artifacts,
        )
    except Exception as exc:
        sys.stderr.write(f"Failed to bundle gate evidence: {exc}\n")
        return 2

    print(result.summary)
    if args.strict and result.has_missing_required:
        return 1
    return 0


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _parse_artifact(value: str) -> dict[str, Any]:
    if "=" not in value:
        raise ValueError("--artifact must use ID[;GATE,...]=PATH")
    left, path = value.split("=", 1)
    if not left or not path:
        raise ValueError("--artifact must include a non-empty ID and PATH")

    artifact_id = left
    gates: list[str] = []
    if ";" in left:
        artifact_id, raw_gates = left.split(";", 1)
        gates = [gate.strip() for gate in raw_gates.split(",") if gate.strip()]
    if not artifact_id:
        raise ValueError("--artifact must include a non-empty ID")
    payload: dict[str, Any] = {"artifact_id": artifact_id, "path": path}
    if gates:
        payload["gates"] = gates
    return payload
