"""``openmed service`` commands for bounded, value-free service diagnostics.

``openmed service diagnose`` collects a deterministic diagnostic and recovery
bundle with :mod:`openmed.service.diagnostics`; ``openmed service verify-bundle``
checks one offline.  Neither command opens a network connection, mutates the
inspected stores, or echoes source paths, values, or exception text.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ._output import EXIT_ERROR, EXIT_USAGE, CliError, emit, wants_json

_LIMIT_OPTIONS = (
    ("max_elapsed_ms", "Total collection time budget in milliseconds."),
    ("section_timeout_ms", "Per-category time budget in milliseconds."),
    ("max_files", "Number of source files that may be opened."),
    ("max_file_bytes", "Bytes that may be read from one source file."),
    ("max_total_bytes", "Bytes that may be read across all source files."),
    ("max_depth", "Directory recursion depth below a store root."),
    ("max_dir_entries", "Directory entries that may be visited in total."),
    ("max_items", "Items kept in one list of a section."),
    ("max_bundle_bytes", "Size of the serialized bundle."),
)


def add_service_command(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``openmed service`` command group."""

    from ..service.diagnostics import CATEGORIES

    parser = subparsers.add_parser(
        "service",
        help="Diagnose a local OpenMed service without collecting PHI.",
    )
    commands = parser.add_subparsers(dest="service_command")

    diagnose = commands.add_parser(
        "diagnose",
        help="Write a bounded, value-free diagnostic and recovery bundle.",
        description=(
            "Collect allowlisted versions, configuration key provenance, "
            "readiness, store schema versions and counts, model registry "
            "digests, audit-continuity state and recovery guidance. The bundle "
            "never contains values, paths, payloads or identifiers and is not a "
            "backup."
        ),
    )
    diagnose.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="FILE",
        help="Write the bundle to FILE (0600) instead of standard output.",
    )
    diagnose.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace FILE when it already exists.",
    )
    diagnose.add_argument(
        "--category",
        dest="categories",
        action="append",
        choices=CATEGORIES,
        help="Evidence category to collect; repeat to select several (default: all).",
    )
    diagnose.add_argument(
        "--jobs-store",
        type=Path,
        default=None,
        metavar="FILE",
        help="Async job metadata store (default: OPENMED_SERVICE_JOBS_STORE_PATH).",
    )
    diagnose.add_argument(
        "--journey-store",
        type=Path,
        default=None,
        metavar="DIR",
        help="Local Journey store root holding journey.sqlite3 and artifacts/.",
    )
    diagnose.add_argument(
        "--registry-state",
        type=Path,
        default=None,
        metavar="FILE",
        help="Model registry slot state (default: the checkout's gates file).",
    )
    diagnose.add_argument(
        "--audit-chain",
        type=Path,
        default=None,
        metavar="FILE",
        help="Tamper-evident audit chain to verify for continuity.",
    )
    for name, description in _LIMIT_OPTIONS:
        diagnose.add_argument(
            f"--{name.replace('_', '-')}",
            dest=name,
            type=int,
            default=None,
            metavar="N",
            help=description,
        )
    diagnose.set_defaults(handler=_handle_diagnose)

    verify = commands.add_parser(
        "verify-bundle",
        help="Verify a diagnostic bundle's manifest and value-safety offline.",
    )
    verify.add_argument("bundle", type=Path, help="Diagnostic bundle file.")
    verify.set_defaults(handler=_handle_verify)


def _handle_diagnose(args: argparse.Namespace) -> int:
    from ..service.diagnostics import (
        CATEGORIES,
        DiagnosticLimits,
        DiagnosticRequest,
        DiagnosticRequestError,
        DiagnosticSources,
        collect_diagnostic_bundle,
        write_diagnostic_bundle,
    )

    overrides = {
        name: getattr(args, name)
        for name, _ in _LIMIT_OPTIONS
        if getattr(args, name, None) is not None
    }
    categories = tuple(dict.fromkeys(args.categories)) if args.categories else None
    try:
        request = DiagnosticRequest(
            categories=categories or CATEGORIES,
            limits=DiagnosticLimits(**overrides),
        )
    except DiagnosticRequestError as exc:
        raise CliError(
            f"Invalid diagnostic request ({exc.code}).",
            code=exc.code,
            exit_code=EXIT_USAGE,
        ) from None
    config_path = getattr(args, "config_path", None)
    sources = DiagnosticSources(
        config_path=Path(config_path) if config_path else None,
        jobs_store_path=args.jobs_store,
        journey_store_root=args.journey_store,
        registry_state_path=args.registry_state,
        audit_chain_path=args.audit_chain,
    )
    bundle = collect_diagnostic_bundle(request, sources=sources)
    summary = bundle.summary()
    if args.out is None:
        if wants_json(args):
            return emit(args, {**summary, "written": False, "bundle": bundle.document})
        sys.stdout.write(bundle.to_bytes().decode("ascii"))
        return 0
    try:
        write_diagnostic_bundle(bundle, args.out, overwrite=args.overwrite)
    except DiagnosticRequestError as exc:
        message = (
            "Refusing to overwrite an existing bundle file; pass --overwrite."
            if exc.code == "output_exists"
            else "The diagnostic bundle file could not be written."
        )
        raise CliError(message, code=exc.code, exit_code=EXIT_ERROR) from None
    lines = [
        "Diagnostic bundle written: "
        f"{summary['entry_count']} entries, collection {summary['collection_status']}.",
        f"Root digest: {summary['root_digest']}",
    ]
    for action in summary["recovery_actions"]:
        lines.append(f"Suggested recovery: {action}")
    return emit(args, {**summary, "written": True}, human="\n".join(lines))


def _handle_verify(args: argparse.Namespace) -> int:
    from ..service.diagnostics import read_diagnostic_bundle, verify_diagnostic_bundle

    try:
        payload = read_diagnostic_bundle(args.bundle)
    except OSError:
        raise CliError(
            "The diagnostic bundle file could not be read.",
            code="bundle_unreadable",
            exit_code=EXIT_ERROR,
        ) from None
    result = verify_diagnostic_bundle(payload)
    human = (
        f"Diagnostic bundle valid: {result.entry_count} entries, "
        f"root {result.root_digest}."
        if result.valid
        else f"Diagnostic bundle invalid: {result.reason_code}."
    )
    return emit(
        args,
        result.to_dict(),
        human=human,
        outcome="completed" if result.valid else "check_failed",
    )


__all__ = ["add_service_command"]
