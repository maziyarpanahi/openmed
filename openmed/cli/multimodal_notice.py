"""Offline CLI rendering for digest-bound multimodal review results."""

from __future__ import annotations

import argparse
from pathlib import Path

from ._output import EXIT_USAGE, CliError, emit


def add_multimodal_notice_command(subparsers: argparse._SubParsersAction) -> None:
    """Register notice-preserving JSON and text output for review references."""
    parser = subparsers.add_parser(
        "multimodal-notice", help="Validate and render a multimodal review result."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--kind",
        required=True,
        choices=("measurement_for_review", "visual_description", "draft_for_review"),
    )
    parser.set_defaults(handler=run_from_args)


def run_from_args(args: argparse.Namespace) -> int:
    """Emit only a validated notice and digest, with value-free input failures."""
    from openmed.multimodal.notices import (
        MAX_NOTICE_RESULT_JSON_BYTES,
        NOTICE_RESULT_TYPES,
        MultimodalNoticeError,
    )

    failed = False
    try:
        with args.input.open("rb") as source:
            payload = source.read(MAX_NOTICE_RESULT_JSON_BYTES + 1)
        result = NOTICE_RESULT_TYPES[args.kind].from_json(payload)
    except (OSError, MultimodalNoticeError):
        failed = True
    if failed:
        raise CliError(
            "Multimodal review result is invalid or unavailable.",
            code="invalid_multimodal_notice",
            exit_code=EXIT_USAGE,
        )
    return emit(args, result.to_dict(), human=result.render_text())
