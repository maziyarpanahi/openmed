"""CLI commands for safe annotation interchange and pipeline migration."""

from __future__ import annotations

import argparse
from pathlib import Path

from openmed.eval.annotation.interchange import (
    AnnotationEnvelope,
    AnnotationInterchangeError,
    export_annotation_tsv,
    import_annotation_tsv,
)
from openmed.interop.bridges.pipeline_migration import (
    PipelineMigrationError,
    scan_pipeline_json,
)

from ._output import EXIT_ERROR, CliError, emit


def add_annotation_interchange_command(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Register annotation interchange and pipeline scanning commands."""

    parser = subparsers.add_parser(
        "annotation",
        help="Import, export, and assess source-text-free annotations.",
    )
    sub = parser.add_subparsers(dest="annotation_command", required=True)

    import_parser = sub.add_parser(
        "import",
        help="Import canonical annotation TSV into a versioned JSON envelope.",
    )
    import_parser.add_argument("--input", type=Path, required=True)
    import_parser.add_argument("--output", type=Path, required=True)
    import_parser.add_argument("--force", action="store_true")
    import_parser.set_defaults(handler=handle_annotation_import)

    export_parser = sub.add_parser(
        "export",
        help="Export a versioned JSON envelope as canonical annotation TSV.",
    )
    export_parser.add_argument("--input", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument(
        "--omit-embeddings",
        action="store_true",
        help="Omit embeddings and declare that loss in the command result.",
    )
    export_parser.add_argument("--loss-report", type=Path, default=None)
    export_parser.add_argument("--force", action="store_true")
    export_parser.set_defaults(handler=handle_annotation_export)

    scan_parser = sub.add_parser(
        "scan-pipeline",
        help="Classify a declarative JSON pipeline without executing it.",
    )
    scan_parser.add_argument("--input", type=Path, required=True)
    scan_parser.add_argument("--report", type=Path, required=True)
    scan_parser.add_argument("--stub", type=Path, default=None)
    scan_parser.add_argument("--force", action="store_true")
    scan_parser.set_defaults(handler=handle_pipeline_scan)


def handle_annotation_import(args: argparse.Namespace) -> int:
    """Import a canonical TSV file and persist a JSON envelope."""

    try:
        _ensure_writable((args.output,), force=args.force)
        envelope = import_annotation_tsv(args.input.read_bytes())
        _write(args.output, envelope.to_json() + "\n", force=args.force)
    except (OSError, AnnotationInterchangeError) as exc:
        raise CliError(
            "Annotation import failed; inspect input and destination permissions",
            code="annotation_import_failed",
            exit_code=EXIT_ERROR,
        ) from exc
    data = {
        "count": len(envelope.records),
        "envelope_digest": envelope.envelope_digest,
        "state": "success",
    }
    return emit(
        args,
        data,
        human=(
            f"Imported {data['count']} annotations to the requested output "
            f"({data['envelope_digest']})."
        ),
    )


def handle_annotation_export(args: argparse.Namespace) -> int:
    """Export a persisted JSON envelope to canonical TSV."""

    try:
        _ensure_writable(
            tuple(path for path in (args.output, args.loss_report) if path is not None),
            force=args.force,
        )
        envelope = AnnotationEnvelope.from_json(args.input.read_bytes())
        exported = export_annotation_tsv(
            envelope,
            include_embeddings=not args.omit_embeddings,
        )
        _write(args.output, exported.text, force=args.force)
        if args.loss_report is not None:
            _write(
                args.loss_report,
                _canonical_line(exported.report.to_dict()),
                force=args.force,
            )
    except (OSError, AnnotationInterchangeError) as exc:
        raise CliError(
            "Annotation export failed; inspect input and destination permissions",
            code="annotation_export_failed",
            exit_code=EXIT_ERROR,
        ) from exc
    data = {
        "count": len(envelope.records),
        "loss_count": len(exported.report.entries),
        "output_digest": exported.report.output_digest,
        "state": exported.report.state.value,
    }
    return emit(
        args,
        data,
        human=(
            f"Exported {data['count']} annotations to the requested output; "
            f"state={data['state']}, declared_losses={data['loss_count']}."
        ),
    )


def handle_pipeline_scan(args: argparse.Namespace) -> int:
    """Scan declarative JSON and persist report and optional native stub."""

    try:
        _ensure_writable(
            tuple(path for path in (args.report, args.stub) if path is not None),
            force=args.force,
        )
        report = scan_pipeline_json(args.input.read_bytes())
        _write(args.report, report.to_json() + "\n", force=args.force)
        if args.stub is not None:
            _write(args.stub, _canonical_line(report.native_config), force=args.force)
    except (OSError, PipelineMigrationError) as exc:
        raise CliError(
            "Pipeline scan failed; inspect input and destination permissions",
            code="pipeline_scan_failed",
            exit_code=EXIT_ERROR,
        ) from exc
    data = {
        "automatic": report.can_auto_migrate,
        "report_digest": report.report_digest,
        "stage_count": len(report.stages),
        "state": report.state.value,
    }
    return emit(
        args,
        data,
        human=(
            f"Scanned {data['stage_count']} stages; state={data['state']}, "
            f"automatic={str(data['automatic']).lower()}."
        ),
    )


def _write(path: Path, text: str, *, force: bool) -> None:
    if path.exists() and not force:
        raise OSError("refusing to overwrite an output; pass --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _ensure_writable(paths: tuple[Path, ...], *, force: bool) -> None:
    normalized = tuple(path.resolve() for path in paths)
    if len(normalized) != len(set(normalized)):
        raise OSError("output paths must be distinct")
    if not force:
        existing = next((path for path in paths if path.exists()), None)
        if existing is not None:
            raise OSError("refusing to overwrite an output; pass --force")


def _canonical_line(payload: object) -> str:
    from openmed.clinical.journey_contracts import canonical_json

    return canonical_json(payload) + "\n"


__all__ = [
    "add_annotation_interchange_command",
    "handle_annotation_export",
    "handle_annotation_import",
    "handle_pipeline_scan",
]
