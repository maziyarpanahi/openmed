"""Calibration CLI command wiring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from openmed.eval.calibrate import (
    artifact_dir_for,
    default_suite_calibration_samples,
    load_calibration_samples,
    write_calibration_artifacts,
)


def add_calibrate_command(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "calibrate",
        help="Fit per-label reliability thresholds for a model artifact.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model identifier being calibrated.",
    )
    parser.add_argument(
        "--suite",
        required=True,
        help="Evaluation suite name, for example golden.",
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        type=Path,
        default=None,
        help="Optional held-out reliability sample JSON.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Directory where thresholds.json and calibration_report.json are written.",
    )
    parser.add_argument(
        "--target-leakage",
        type=float,
        default=0.0,
        help="Maximum target leakage rate before optimizing over-redaction.",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=None,
        help="Optional recall floor; defaults to 1 - target leakage.",
    )
    parser.add_argument(
        "--conformal-alpha",
        type=float,
        default=0.05,
        help="Split-conformal miscoverage target for calibration bands.",
    )
    parser.add_argument(
        "--gate-input",
        dest="gate_input_path",
        type=Path,
        default=None,
        help="Optional held-out gate sample JSON used for shift validation.",
    )
    parser.add_argument(
        "--coverage-tolerance",
        type=float,
        default=0.01,
        help="Allowed coverage gap before conformal release gates fail.",
    )
    parser.set_defaults(handler=handle_calibrate)


def handle_calibrate(args: argparse.Namespace) -> int:
    try:
        if args.input_path is not None:
            samples = load_calibration_samples(
                args.input_path,
                default_model_id=args.model,
            )
            sample_source = str(args.input_path)
        else:
            samples = default_suite_calibration_samples(args.model, args.suite)
            sample_source = f"builtin:{args.suite}"

        gate_samples = None
        if args.gate_input_path is not None:
            gate_samples = load_calibration_samples(
                args.gate_input_path,
                default_model_id=args.model,
            )

        artifact_dir = args.artifact_dir or artifact_dir_for(args.model, args.suite)
        paths = write_calibration_artifacts(
            samples,
            artifact_dir=artifact_dir,
            model_id=args.model,
            suite=args.suite,
            target_leakage=args.target_leakage,
            min_recall=args.min_recall,
            conformal_alpha=args.conformal_alpha,
            gate_samples=gate_samples,
            coverage_tolerance=args.coverage_tolerance,
            metadata={"sample_source": sample_source},
        )
    except Exception as exc:
        sys.stderr.write(f"Calibration failed: {exc}\n")
        return 1

    sys.stdout.write(
        json.dumps(
            {
                "artifact_dir": str(paths.artifact_dir),
                "thresholds": str(paths.thresholds_path),
                "report": str(paths.report_path),
                "under_shift_report": (
                    str(paths.under_shift_report_path)
                    if paths.under_shift_report_path is not None
                    else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return 0


__all__ = ["add_calibrate_command", "handle_calibrate"]
