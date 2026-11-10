"""Supported offline qualification command for local decision backends."""

import argparse
import importlib
import json
import os
from pathlib import Path

from ._output import CliError, add_json_flag, emit


def add_decision_qualification_command(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Register the local decision-backend qualification command."""
    parser = subparsers.add_parser(
        "decision-qualify",
        help="Qualify a local fixed-option decision backend on caller data.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--backend",
        choices=("deterministic",),
        help="Built-in dependency-free lexical backend.",
    )
    source.add_argument(
        "--backend-factory",
        help="Trusted installed module:function returning a local decision backend.",
    )
    parser.add_argument(
        "--development",
        required=True,
        type=Path,
        help="Development JSONL used only to fit thresholds.",
    )
    parser.add_argument(
        "--evaluation",
        required=True,
        type=Path,
        help="Held-out evaluation JSONL; never used to fit thresholds.",
    )
    parser.add_argument(
        "--calibration-id",
        required=True,
        help="Controlled profile prefix; each mode becomes <prefix>.<mode>.",
    )
    parser.add_argument(
        "--calibration-version",
        default="1.0.0",
        help="Semantic version recorded in emitted profiles.",
    )
    parser.add_argument(
        "--min-examples-per-option",
        type=int,
        default=20,
        help="Evaluation examples required for every option before a profile.",
    )
    parser.add_argument(
        "--accuracy-floor",
        type=float,
        default=0.9,
        help="Selective accuracy required on development and evaluation.",
    )
    parser.add_argument(
        "--coverage-floor",
        type=float,
        default=0.5,
        help="Evaluation coverage required at the frozen thresholds.",
    )
    parser.add_argument(
        "--profile-output",
        type=Path,
        help="New file for the emitted profiles; written only when qualified.",
    )
    add_json_flag(parser)
    parser.set_defaults(handler=handle_decision_qualification)


def handle_decision_qualification(args: argparse.Namespace) -> int:
    """Emit only the value-free report; never echo inputs or backend errors."""
    from openmed.eval.decision_qualification import (
        DecisionQualificationError,
        DecisionQualificationPolicy,
        qualify_decision_backend,
    )

    code = None
    written = False
    try:
        output = args.profile_output
        if output is not None and os.path.lexists(output):
            raise DecisionQualificationError("profile_output_exists")
        report = qualify_decision_backend(
            _backend(args, DecisionQualificationError),
            development=args.development,
            evaluation=args.evaluation,
            calibration_id=args.calibration_id,
            calibration_version=args.calibration_version,
            policy=DecisionQualificationPolicy(
                min_examples_per_option=args.min_examples_per_option,
                accuracy_floor=args.accuracy_floor,
                coverage_floor=args.coverage_floor,
            ),
        )
        if report.qualified and output is not None:
            profiles = report.to_dict()["profiles"]
            try:
                _write_new(
                    output, json.dumps(profiles, indent=2, sort_keys=True) + "\n"
                )
            except OSError:
                raise DecisionQualificationError("profile_output_unavailable") from None
            written = True
    except DecisionQualificationError as error:
        code = error.code
    except Exception:
        code = "qualification_unavailable"
    if code is not None:
        raise CliError(
            f"Decision qualification failed: {code}.",
            code="decision_qualification_failed",
        )
    payload = report.to_dict()
    notice = ""
    if output is not None:
        notice = " Profile written." if written else " Profile withheld."
    emit(
        args,
        payload,
        human=(
            f"Decision qualification: {payload['status']}; "
            f"human review required.{notice}"
        ),
    )
    return 0 if payload["qualified"] else 1


def _backend(args: argparse.Namespace, error: type[Exception]):
    from openmed.structured.decision import (
        DecisionBackend,
        DecisionBackendIdentity,
        DeterministicDecisionBackend,
    )

    if args.backend_factory is None:
        # A fresh instance; the module-level default is never touched.
        return DeterministicDecisionBackend()
    parts = args.backend_factory.split(":")
    if (
        len(parts) != 2
        or not all(part.isidentifier() for part in parts[0].split("."))
        or not parts[1].isidentifier()
    ):
        raise error("backend_factory_invalid")
    try:
        backend = getattr(importlib.import_module(parts[0]), parts[1])()
    except Exception:
        raise error("backend_unavailable") from None
    if not isinstance(backend, DecisionBackend) or not isinstance(
        getattr(backend, "identity", None), DecisionBackendIdentity
    ):
        raise error("backend_factory_invalid")
    return backend


def _write_new(path: Path, content: str) -> None:
    # Exclusive creation refuses existing files and symlinks without loss.
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
    except Exception:
        path.unlink(missing_ok=True)
        raise
