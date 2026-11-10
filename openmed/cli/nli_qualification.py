"""Supported offline qualification command for caller-supplied NLI artifacts."""

import argparse
import json
from pathlib import Path

from ._output import CliError, add_json_flag, emit


def add_nli_qualification_command(subparsers: argparse._SubParsersAction) -> None:
    """Register the local artifact qualification command."""
    parser = subparsers.add_parser(
        "nli-qualify", help="Qualify a caller-supplied local clinical NLI artifact."
    )
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--label-mapping", required=True, type=Path)
    parser.add_argument("--development", type=Path)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--runtime", choices=("torch", "onnx"), default="torch")
    add_json_flag(parser)
    parser.set_defaults(handler=handle_nli_qualification)


def handle_nli_qualification(args: argparse.Namespace, *, loader=None) -> int:
    """Emit only aggregate qualification evidence; never echo input failures."""
    from openmed.clinical.nli_qualification import (
        NLIQualificationPolicy,
        qualify_local_nli,
    )

    def read(path):
        return json.loads(path.read_text(encoding="utf-8")) if path else None

    failed = False
    try:
        options = read(args.policy) or {}
        if "required_slices" in options:
            options["required_slices"] = tuple(options["required_slices"])
        receipt = qualify_local_nli(
            args.artifact,
            label_mapping=read(args.label_mapping),
            development=read(args.development),
            evaluation=read(args.evaluation),
            policy=NLIQualificationPolicy(**options),
            runtime=args.runtime,
            loader=loader,
        )
    except Exception:
        failed = True
    if failed:
        raise CliError(
            "NLI qualification input unavailable.", code="nli_qualification_failed"
        )
    payload = receipt.to_dict()
    emit(
        args,
        payload,
        human=f"NLI qualification: {payload['status']}; human review required.",
    )
    return 0 if payload["qualified"] else 1
