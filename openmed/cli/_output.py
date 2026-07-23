"""Shared JSON-output envelope, error handling, and exit codes for the CLI.

Every scriptable subcommand accepts ``--json`` and, when set, emits a single
machine-readable document on stdout with stable top-level keys:

* success -> ``{"ok": true, "command": "<path>", "data": {...}}``
* failure -> ``{"ok": false, "command": "<path>", "error": {"code", "message"}}``

Handlers signal failure by raising :class:`CliError`, which carries a stable
error ``code`` and a documented process exit status. ``main()`` catches it and
renders the error envelope (JSON on stdout when ``--json`` is set, otherwise
plain text on stderr) so no handler has to format errors itself.

Exit-code table (documented in ``docs/output-formatting.md``):

======  ============================================================
 code    meaning
======  ============================================================
 0       success
 1       failure: runtime error, or a gate/verification negative result
         (e.g. ``--strict``/``--fail-on-*`` violations, matching the
         repository's existing release-gate convention)
 2       usage / validation error (matches argparse's own exit code)
======  ============================================================
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import IO, Any

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2


class CliError(Exception):
    """A user-facing CLI failure with a stable error code and exit status."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "error",
        exit_code: int = EXIT_ERROR,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.exit_code = exit_code


def add_json_flag(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach the uniform ``--json`` flag to a leaf subcommand parser."""

    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Emit machine-readable JSON with stable top-level keys.",
    )
    return parser


def wants_json(args: argparse.Namespace) -> bool:
    """Return whether ``--json`` machine output was requested."""

    return bool(getattr(args, "json_output", False))


def command_path(args: argparse.Namespace) -> str:
    """Return the dotted/space command path for the envelope ``command`` key."""

    return str(
        getattr(args, "command_path", None) or getattr(args, "command", "") or ""
    )


def _dump(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)


def emit(
    args: argparse.Namespace,
    payload: Any,
    *,
    human: str | None = None,
    stream: IO[str] | None = None,
) -> int:
    """Write a success result and return :data:`EXIT_OK`.

    In ``--json`` mode a ``{"ok", "command", "data"}`` envelope is written to
    ``stream`` (stdout by default). Otherwise ``human`` is written verbatim (a
    trailing newline is added when missing); ``None`` writes nothing.
    """

    out = stream if stream is not None else sys.stdout
    if wants_json(args):
        envelope = {"ok": True, "command": command_path(args), "data": payload}
        out.write(_dump(envelope) + "\n")
    elif human is not None:
        out.write(human if human.endswith("\n") else human + "\n")
    return EXIT_OK


def emit_error(
    args: argparse.Namespace,
    error: CliError,
    *,
    json_stream: IO[str] | None = None,
    text_stream: IO[str] | None = None,
) -> int:
    """Render a :class:`CliError` and return its exit code.

    In ``--json`` mode the error envelope is written to stdout (so an agent can
    parse a single stream); otherwise the message is written to stderr.
    """

    if wants_json(args):
        out = json_stream if json_stream is not None else sys.stdout
        envelope = {
            "ok": False,
            "command": command_path(args),
            "error": {"code": error.code, "message": error.message},
        }
        out.write(_dump(envelope) + "\n")
    else:
        err = text_stream if text_stream is not None else sys.stderr
        err.write(f"{error.message}\n")
    return error.exit_code


__all__ = [
    "CliError",
    "EXIT_OK",
    "EXIT_ERROR",
    "EXIT_USAGE",
    "add_json_flag",
    "command_path",
    "emit",
    "emit_error",
    "wants_json",
]
