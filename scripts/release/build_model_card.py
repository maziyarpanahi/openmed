"""Build or check a model card from release-gate artifacts."""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

from openmed.core.model_registry import load_manifest_rows
from openmed.eval.evidence_bundle import bundle_gate_evidence
from openmed.eval.model_card_builder import (
    MODEL_DATASHEET_FILENAME,
    ModelCardBuilderError,
    build_model_card,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an artifact-backed OpenMed model card and datasheet.",
    )
    parser.add_argument("--manifest", default="models.jsonl", help="models.jsonl path")
    parser.add_argument("--repo-id", required=True, help="Manifest repo_id to render")
    parser.add_argument(
        "--gate-report",
        required=True,
        help="Signed release gate report JSON",
    )
    parser.add_argument("--calibration-report", default=None)
    parser.add_argument("--calibration-thresholds", default=None)
    parser.add_argument("--fairness-report", default=None)
    parser.add_argument("--quant-delta", default=None)
    parser.add_argument("--training-provenance", default=None)
    parser.add_argument("--output", default=None, help="README.md output path")
    parser.add_argument(
        "--datasheet-output",
        default=None,
        help=f"Datasheet JSON output path (default beside --output as {MODEL_DATASHEET_FILENAME})",
    )
    parser.add_argument(
        "--check",
        default=None,
        help="Existing README.md to compare against generated output",
    )
    parser.add_argument(
        "--evidence-bundle",
        default=None,
        help="Optional evidence bundle directory to receive card and datasheet",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        row = _find_manifest_row(args.manifest, args.repo_id)
        result = build_model_card(
            row,
            args.gate_report,
            calibration_report=args.calibration_report,
            calibration_thresholds=args.calibration_thresholds,
            fairness_report=args.fairness_report,
            quant_delta=args.quant_delta,
            training_provenance=args.training_provenance,
        )
    except (OSError, ValueError, ModelCardBuilderError) as exc:
        sys.stderr.write(f"Failed to build model card: {exc}\n")
        return 1

    if args.check is not None:
        check_path = Path(args.check)
        try:
            existing = check_path.read_text(encoding="utf-8")
        except OSError as exc:
            sys.stderr.write(f"Failed to read card for comparison: {exc}\n")
            return 1
        if existing != result.markdown:
            diff = difflib.unified_diff(
                existing.splitlines(keepends=True),
                result.markdown.splitlines(keepends=True),
                fromfile=str(check_path),
                tofile=f"rendered:{args.repo_id}",
            )
            sys.stdout.writelines(diff)
            return 1

    output_path = Path(args.output) if args.output is not None else None
    datasheet_path = _datasheet_path(args.datasheet_output, output_path)

    if output_path is not None:
        result.write_markdown(output_path)
    if datasheet_path is not None:
        result.write_datasheet(datasheet_path)

    if args.evidence_bundle is not None:
        if output_path is None or datasheet_path is None:
            sys.stderr.write(
                "--evidence-bundle requires --output and a datasheet output path\n"
            )
            return 1
        bundle_gate_evidence(
            result.gate_report,
            args.evidence_bundle,
            extra_artifacts={
                "model_card": output_path,
                "model_datasheet": datasheet_path,
            },
        )

    if output_path is None and args.check is None:
        sys.stdout.write(result.markdown)
    return 0


def _find_manifest_row(path: str | Path, repo_id: str) -> dict[str, object]:
    rows = load_manifest_rows(Path(path))
    for row in rows:
        if row.get("repo_id") == repo_id:
            return dict(row)
    raise ValueError(f"repo_id not found in model manifest: {repo_id}")


def _datasheet_path(
    raw_path: str | None,
    output_path: Path | None,
) -> Path | None:
    if raw_path is not None:
        return Path(raw_path)
    if output_path is not None:
        return output_path.with_name(MODEL_DATASHEET_FILENAME)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
