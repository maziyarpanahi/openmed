"""CLI helpers for release-gate workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from openmed.core import baseline as baseline_store
from openmed.eval.evidence_bundle import bundle_gate_evidence
from openmed.eval.release_gates import RELEASABLE, format_preview, preview

from ._output import EXIT_ERROR, EXIT_USAGE, CliError, emit


def add_gates_command(subparsers: argparse._SubParsersAction) -> None:
    """Register ``openmed gates`` subcommands."""
    gates_parser = subparsers.add_parser(
        "gates",
        help="Release gate evidence and preview utilities.",
    )
    gates_sub = gates_parser.add_subparsers(dest="gates_command")

    _add_preview_command(gates_sub)
    _add_bundle_command(gates_sub)


def _add_preview_command(subparsers: argparse._SubParsersAction) -> None:
    preview_parser = subparsers.add_parser(
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
        default="v1.7",
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


def _add_bundle_command(subparsers: argparse._SubParsersAction) -> None:
    bundle_parser = subparsers.add_parser(
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


def handle_gates_preview(args: argparse.Namespace) -> int:
    """Run a read-only release-gate preview."""

    if not args.candidate.is_file():
        raise CliError(
            f"Candidate report not found: {args.candidate}",
            code="candidate_not_found",
            exit_code=EXIT_USAGE,
        )

    try:
        candidate = _read_json_object(args.candidate)
        baseline = _read_json_object(args.baseline) if args.baseline else None
        report = preview(
            candidate,
            baseline,
            milestone=args.milestone,
            policy=args.policy,
            baseline_path=args.baseline_store,
            thresholds_matrix_path=args.thresholds_matrix,
        )
    except Exception as exc:
        raise CliError(
            f"Release gate preview failed: {exc}",
            code="preview_failed",
            exit_code=EXIT_ERROR,
        ) from exc

    emit(args, report.to_dict(), human=format_preview(report))
    if args.strict and report.decision != RELEASABLE:
        return 1
    return 0


def _handle_bundle(args: argparse.Namespace) -> int:
    try:
        gate_report = _read_json_object(args.gate_report)
        extra_artifacts = [_parse_artifact(value) for value in args.artifact]
        extra_artifacts.append(
            {
                "artifact_id": "gate_report",
                "path": str(args.gate_report.resolve()),
                "required": True,
            }
        )
        result = bundle_gate_evidence(
            gate_report,
            args.output_dir,
            evidence_root=args.evidence_root or args.gate_report.parent,
            extra_artifacts=extra_artifacts,
        )
    except Exception as exc:
        raise CliError(
            f"Failed to bundle gate evidence: {exc}",
            code="bundle_failed",
            exit_code=EXIT_ERROR,
        ) from exc

    payload = {
        "output_dir": str(result.output_dir),
        "manifest_path": str(result.manifest_path),
        "summary_path": str(result.summary_path),
        "manifest": dict(result.manifest),
        "missing_required": [dict(item) for item in result.missing_required],
        "has_missing_required": result.has_missing_required,
        "summary": result.summary,
    }
    emit(args, payload, human=result.summary)
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


__all__ = ["add_gates_command", "handle_gates_preview"]
