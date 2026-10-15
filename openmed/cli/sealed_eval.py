"""Argparse surface for ``openmed eval sealed``: local sealed private-model evaluation.

The command is a thin, offline binding. Evaluation itself is delegated to an
injected service so the CLI stays testable without models, holdout data or a
network. Only the signed aggregate envelope is written (to the declared output
path); case-level artifacts stay in the service's site-local directory and are
never printed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, TextIO

from ._output import EXIT_ERROR, EXIT_OK, EXIT_USAGE, CliError, emit


class SealedEvaluationService(Protocol):
    """Injected sealed-evaluation backend."""

    def preflight(self, model_ref: str, holdout_manifest: Path) -> Mapping[str, Any]:
        """Return ``signed``, ``digest_match`` and ``requires_network`` booleans."""

    def evaluate(self, model_ref: str, holdout_manifest: Path) -> Mapping[str, Any]:
        """Return the signed aggregate result envelope (no case-level data)."""


ServiceFactory = Callable[[], SealedEvaluationService]


def default_service() -> SealedEvaluationService:
    """Resolve the ambient service; none ships on master yet."""
    raise CliError(
        "No sealed-evaluation service is configured.",
        code="sealed_eval_service_unavailable",
        exit_code=EXIT_ERROR,
    )


def add_sealed_eval_command(eval_sub: argparse._SubParsersAction) -> None:
    """Register ``openmed eval sealed`` under the ``eval`` command group."""
    parser = eval_sub.add_parser(
        "sealed",
        help="Run an offline sealed evaluation of a signed candidate model.",
        description=(
            "Evaluate a signed candidate against a local holdout manifest and write "
            "only the signed aggregate result envelope. Runs offline. Exit codes: "
            "0 success; 1 refused or failed (see error code); 2 usage error."
        ),
    )
    parser.add_argument("--model", required=True, help="Explicit candidate model reference.")
    parser.add_argument(
        "--holdout-manifest",
        type=Path,
        required=True,
        help="Local holdout manifest (never uploaded or printed).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the signed aggregate result envelope.",
    )
    parser.set_defaults(handler=run_from_args)


def _refuse(message: str, code: str, exit_code: int = EXIT_ERROR) -> CliError:
    return CliError(message, code=code, exit_code=exit_code)


def run_from_args(
    args: argparse.Namespace,
    *,
    stdout: TextIO | None = None,
    service: SealedEvaluationService | None = None,
) -> int:
    """Validate the candidate, evaluate, and write the aggregate envelope only.

    Raises:
        CliError: Controlled, content-free failure raised before any evaluation
            when the manifest is missing, the candidate is unsigned or its digest
            mismatches, or the configuration needs a network.
    """
    manifest: Path = args.holdout_manifest
    if not manifest.is_file():
        raise _refuse(
            "Holdout manifest not found.", "holdout_manifest_missing", EXIT_USAGE
        )
    service = service if service is not None else default_service()

    try:
        check = service.preflight(args.model, manifest)
    except CliError:
        raise
    except Exception:
        raise _refuse("Preflight could not complete.", "sealed_eval_preflight_failed") from None
    if check.get("requires_network"):
        raise _refuse("Sealed evaluation must run offline.", "network_required")
    if not check.get("signed"):
        raise _refuse("Candidate is unsigned or unverified.", "candidate_unsigned")
    if not check.get("digest_match"):
        raise _refuse("Candidate digest mismatch.", "candidate_digest_mismatch")

    try:
        envelope = dict(service.evaluate(args.model, manifest))
    except CliError:
        raise
    except Exception:
        raise _refuse("Sealed evaluation could not complete.", "sealed_eval_failed") from None

    output: Path = args.output
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(envelope, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError:
        raise _refuse("Could not write the result envelope.", "output_unwritable") from None

    data = {"status": "completed", "output_written": True}
    emit(args, data, human="Sealed evaluation completed; aggregate envelope written.", stream=stdout)
    return EXIT_OK
