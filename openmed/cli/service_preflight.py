"""Argparse surface for the REST service security-profile preflight."""

from __future__ import annotations

import argparse
from typing import TextIO

from ._output import EXIT_USAGE, CliError, emit


def add_service_preflight_command(subparsers: argparse._SubParsersAction) -> None:
    """Register ``openmed service preflight``.

    Args:
        subparsers: The root subparser collection.
    """
    service = subparsers.add_parser(
        "service", help="REST service security and deployment checks."
    )
    commands = service.add_subparsers(dest="service_command", required=True)
    preflight = commands.add_parser(
        "preflight",
        help="Check the service security profile without starting the server.",
        description=(
            "Build the REST app from the current environment and run the same "
            "read-only security-profile self-check as startup and /readyz. "
            "Exit 0 when the profile is valid, 1 when a required control "
            "fails, 2 when the configuration cannot be read."
        ),
    )
    preflight.add_argument(
        "--profile",
        choices=("dev", "prod"),
        default=None,
        help="Evaluate this security profile instead of the resolved one.",
    )
    preflight.add_argument(
        "--host",
        default=None,
        help="Listener host the service will bind (default: launcher value).",
    )
    preflight.add_argument(
        "--tls",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether the listener terminates TLS in-process.",
    )
    preflight.add_argument(
        "--grpc-address",
        default=None,
        help="Also check a gRPC listener bound to this address.",
    )
    preflight.set_defaults(handler=run_from_args)


def run_from_args(args: argparse.Namespace, *, stdout: TextIO | None = None) -> int:
    """Emit the value-free report and return its stable exit code.

    Args:
        args: Parsed CLI arguments, including the uniform JSON flag.
        stdout: Optional caller-owned output stream.

    Returns:
        Zero for a valid profile, one when a required control fails.

    Raises:
        CliError: Exit code two when the service configuration or the
            ``service`` extra is unavailable. Messages never echo values.
    """
    from openmed.service.security_profile import run_service_preflight

    try:
        report = run_service_preflight(
            profile=args.profile,
            listen_host=args.host,
            listen_tls=args.tls,
            grpc_address=args.grpc_address,
        )
    except ImportError:
        code, message = _MISSING_EXTRA
    except Exception:
        # Parser messages and exception text may carry configured values.
        code, message = _UNREADABLE
    else:
        emit(args, report.to_dict(), human=report.to_text(), stream=stdout)
        return report.exit_code
    raise CliError(message, code=code, exit_code=EXIT_USAGE) from None


_MISSING_EXTRA = (
    "service_extra_missing",
    "The service preflight needs the 'service' extra.",
)
_UNREADABLE = (
    "security_profile_unreadable",
    "The service configuration could not be read; fix the service environment "
    "and rerun the preflight.",
)
