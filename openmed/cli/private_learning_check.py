"""Argparse integration for offline private-learning metadata self-checks."""

from __future__ import annotations

import argparse
from typing import TextIO

from openmed.cli._output import EXIT_ERROR, EXIT_OK, CliError, emit


def add_private_learning_check_command(subparsers: argparse._SubParsersAction) -> None:
    """Register ``openmed training self-check`` without models or training execution."""
    training = subparsers.add_parser(
        "training", help="Offline private-learning metadata checks."
    )
    commands = training.add_subparsers(dest="training_command", required=True)
    check = commands.add_parser(
        "self-check", help="Check bundled private-learning metadata contracts offline."
    )
    check.set_defaults(handler=run_from_args)


def run_from_args(args: argparse.Namespace, *, stdout: TextIO | None = None) -> int:
    """Emit a deterministic report and fail the process when a mandatory check fails."""
    from openmed.training.private_learning_check import run_private_learning_check

    try:
        report = run_private_learning_check()
        payload, human, passed = report.to_dict(), report.to_text(), report.passed
    except Exception:
        failed = True
    else:
        failed = False
    if failed:
        raise CliError(
            "Private-learning self-check could not complete.",
            code="private_learning_check_unavailable",
            exit_code=EXIT_ERROR,
        ) from None
    emit(
        args,
        payload,
        human=human,
        stream=stdout,
        outcome=None if passed else "check_failed",
    )
    return EXIT_OK if passed else EXIT_ERROR
