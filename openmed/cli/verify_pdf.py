"""CLI subcommand for redacted-PDF text-layer and visual fidelity verification.

Wraps :func:`openmed.multimodal.verify_pdf.verify_redacted_pdf`. Given an
original PDF, a redacted PDF, and the redaction regions, it prints a PHI-safe
fidelity report and exits non-zero when any region still leaks selectable text
or shows no visible redaction box.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence, TextIO

from ._output import EXIT_ERROR, EXIT_USAGE, CliError, emit, emit_error


def add_verify_pdf_command(subparsers: "argparse._SubParsersAction") -> None:
    """Register the ``verify-pdf`` subcommand on the top-level parser."""
    parser = subparsers.add_parser(
        "verify-pdf",
        help="Verify a redacted PDF scrubbed its text layer and drew redaction boxes.",
    )
    parser.add_argument("original", type=Path, help="Path to the original PDF.")
    parser.add_argument("redacted", type=Path, help="Path to the redacted PDF.")
    parser.add_argument(
        "--spans",
        type=Path,
        required=True,
        metavar="FILE",
        help=(
            "JSON file listing the redacted regions: a list of "
            '{"page": int, "bbox": [x0, top, x1, bottom]} objects, or character '
            'spans {"start": int, "end": int} projected against the original.'
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optional path to write the fidelity report JSON.",
    )
    parser.set_defaults(handler=run_from_args)


def _load_spans(path: Path) -> list[Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("spans", "regions", "redaction_rectangles"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        raise ValueError(
            "spans JSON object must contain a 'spans', 'regions', or "
            "'redaction_rectangles' list"
        )
    if not isinstance(payload, list):
        raise ValueError("spans JSON must be a list or an object wrapping one")
    return payload


def run_from_args(
    args: argparse.Namespace,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run redacted-PDF verification from argparse arguments."""
    stdout = stdout or sys.stdout

    from openmed.multimodal.verify_pdf import verify_redacted_pdf

    try:
        spans = _load_spans(args.spans)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise CliError(
            f"Failed to read spans file: {exc}",
            code="invalid_spans",
            exit_code=EXIT_USAGE,
        )

    try:
        report = verify_redacted_pdf(args.original, args.redacted, spans)
    except FileNotFoundError as exc:
        raise CliError(
            f"Input PDF not found: {exc}",
            code="input_not_found",
            exit_code=EXIT_ERROR,
        )
    except Exception as exc:  # noqa: BLE001 - surface a clean CLI error.
        raise CliError(
            f"PDF fidelity verification failed: {exc}",
            code="verification_failed",
            exit_code=EXIT_ERROR,
        )

    data = report.to_dict()
    output = json.dumps(data, indent=2, sort_keys=True)
    if args.output is not None:
        try:
            args.output.write_text(output + "\n", encoding="utf-8")
        except OSError as exc:
            raise CliError(
                f"Failed to write report: {exc}",
                code="write_failed",
                exit_code=EXIT_ERROR,
            )
    emit(args, data, human=output + "\n" + report.summary(), stream=stdout)
    return 0 if report.passed else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Standalone entry point for ``python -m openmed.cli.verify_pdf``."""
    parser = argparse.ArgumentParser(
        prog="openmed verify-pdf",
        description="Verify redacted-PDF text-layer leakage and visual fidelity.",
    )
    parser.add_argument("original", type=Path, help="Path to the original PDF.")
    parser.add_argument("redacted", type=Path, help="Path to the redacted PDF.")
    parser.add_argument(
        "--spans", type=Path, required=True, help="Redacted regions JSON."
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None, help="Report JSON path."
    )
    args = parser.parse_args(argv)
    try:
        return run_from_args(args)
    except CliError as exc:
        return emit_error(args, exc)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
