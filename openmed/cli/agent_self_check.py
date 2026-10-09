"""Argparse surface for the bundled offline agent governance self-check."""

from __future__ import annotations

import argparse
from typing import TextIO

from ._output import EXIT_ERROR, EXIT_OK, CliError, emit


def add_agent_self_check_command(subparsers: argparse._SubParsersAction) -> None:
    """Register ``openmed agent self-check`` without models or credentials."""
    agent = subparsers.add_parser("agent", help="Offline agent governance checks.")
    commands = agent.add_subparsers(dest="agent_command", required=True)
    check = commands.add_parser(
        "self-check", help="Check bundled governance contracts offline."
    )
    check.set_defaults(handler=run_from_args)


def run_from_args(args: argparse.Namespace, *, stdout: TextIO | None = None) -> int:
    """Emit every independent result and fail the process if any check fails.

    Args:
        args: Parsed CLI arguments, including the uniform JSON flag.
        stdout: Optional caller-owned output stream.

    Returns:
        Zero for a passing report, one when any check failed.

    Raises:
        CliError: With a controlled message if report construction fails.
    """
    from openmed.agent.self_check import run_agent_self_check

    try:
        report = run_agent_self_check()
        data, human, passed = report.to_dict(), report.to_text(), report.passed
    except Exception:
        raise CliError(
            "Agent self-check could not complete.",
            code="agent_self_check_unavailable",
            exit_code=EXIT_ERROR,
        ) from None
    emit(args, data, human=human, stream=stdout)
    return EXIT_OK if passed else EXIT_ERROR
