"""Benchmark-specific CLI command wiring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from openmed.eval.cost import (
    PriceCitationError,
    cost_vs_cloud_report,
    load_cloud_prices,
)
from openmed.eval.generalization import cross_corpus_report

from ._output import EXIT_ERROR, EXIT_USAGE, CliError, emit


def add_generalization_command(subparsers: argparse._SubParsersAction) -> None:
    """Register ``openmed benchmark generalization``."""

    parser = subparsers.add_parser(
        "generalization",
        help="Compare benchmark metrics across source corpora.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model identifier to evaluate.",
    )
    parser.add_argument(
        "--in-domain",
        required=True,
        dest="in_domain",
        help="In-domain suite name or local JSON/JSONL fixture path.",
    )
    parser.add_argument(
        "--out-of-domain",
        required=True,
        nargs="+",
        dest="out_of_domain",
        help=(
            "One or more out-of-domain suite names or local JSON/JSONL fixture "
            "paths. Comma-separated values are accepted."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device tier label recorded in each benchmark report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path where the JSON generalization report is written.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for generalization.json and generalization.md.",
    )
    parser.set_defaults(handler=handle_generalization)


def add_cost_command(subparsers: argparse._SubParsersAction) -> None:
    """Register ``openmed benchmark cost``."""

    parser = subparsers.add_parser(
        "cost",
        help="Compare local benchmark throughput with cited cloud prices.",
    )
    parser.add_argument(
        "--perf",
        required=True,
        type=Path,
        help="Path to a local performance report JSON file.",
    )
    parser.add_argument(
        "--prices",
        required=True,
        type=Path,
        help="Path to a dated, citation-annotated cloud price table JSON file.",
    )
    parser.add_argument(
        "--chars-per-doc",
        type=float,
        default=None,
        help="Optional character-per-document override for older perf reports.",
    )
    parser.add_argument(
        "--hardware-cost-model",
        type=Path,
        default=None,
        help="Optional JSON hardware-cost model overriding table/report defaults.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the JSON cost report.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for cost.json and cost.md.",
    )
    parser.set_defaults(handler=handle_cost)


def handle_generalization(args: argparse.Namespace) -> int:
    """Run the cross-corpus report and emit or write its aggregate evidence."""

    if args.output is not None and args.output_dir is not None:
        raise CliError(
            "--output and --output-dir cannot be combined.",
            code="invalid_argument",
            exit_code=EXIT_USAGE,
        )

    out_of_domain = _parse_suite_args(args.out_of_domain)
    try:
        report = cross_corpus_report(
            args.model,
            args.in_domain,
            out_of_domain,
            device=args.device,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise CliError(
            f"Generalization report failed: {exc}",
            code="generalization_failed",
            exit_code=EXIT_ERROR,
        ) from exc

    if args.output is not None:
        try:
            output_path = report.write_json(args.output)
        except OSError as exc:
            raise CliError(
                f"Failed to write generalization report: {exc}",
                code="write_failed",
                exit_code=EXIT_ERROR,
            ) from exc
        return emit(
            args,
            {"written": str(output_path), "headline_gap": report.headline_gap},
            human=f"Generalization report written: {output_path}",
        )

    if args.output_dir is not None:
        try:
            json_path = report.write_json(args.output_dir / "generalization.json")
            markdown_path = report.write_markdown(args.output_dir / "generalization.md")
        except OSError as exc:
            raise CliError(
                f"Failed to write generalization report: {exc}",
                code="write_failed",
                exit_code=EXIT_ERROR,
            ) from exc
        paths = {"json": str(json_path), "markdown": str(markdown_path)}
        return emit(
            args,
            {"written": paths, "headline_gap": report.headline_gap},
            human=(
                "Generalization reports written:\n"
                f"  JSON: {json_path}\n"
                f"  Markdown: {markdown_path}"
            ),
        )

    return emit(args, report.to_dict(), human=report.to_json())


def handle_cost(args: argparse.Namespace) -> int:
    """Build and emit a local-versus-cloud cost comparison report."""

    if args.output is not None and args.output_dir is not None:
        raise CliError(
            "--output and --output-dir cannot be combined.",
            code="invalid_argument",
            exit_code=EXIT_USAGE,
        )

    perf_report: Any = args.perf
    if args.chars_per_doc is not None:
        perf_payload = _read_json_mapping(args.perf, "performance report")
        perf_payload["chars_per_doc"] = args.chars_per_doc
        perf_report = perf_payload

    hardware_cost_model: Mapping[str, Any] | None = None
    if args.hardware_cost_model is not None:
        hardware_cost_model = _read_json_mapping(
            args.hardware_cost_model,
            "hardware cost model",
        )

    try:
        report = cost_vs_cloud_report(
            perf_report,
            load_cloud_prices(args.prices),
            hardware_cost_model,
        )
    except PriceCitationError as exc:
        raise CliError(
            str(exc),
            code="invalid_price_table",
            exit_code=EXIT_USAGE,
        ) from exc
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise CliError(
            f"Cost comparison failed: {exc}",
            code="cost_comparison_failed",
            exit_code=EXIT_ERROR,
        ) from exc

    if args.output is not None:
        try:
            output_path = report.write_json(args.output)
        except OSError as exc:
            raise CliError(
                f"Failed to write cost report: {exc}",
                code="write_failed",
                exit_code=EXIT_ERROR,
            ) from exc
        return emit(
            args,
            {"written": str(output_path)},
            human=f"Cost comparison report written: {output_path}",
        )

    if args.output_dir is not None:
        try:
            json_path = report.write_json(args.output_dir / "cost.json")
            markdown_path = report.write_markdown(args.output_dir / "cost.md")
        except OSError as exc:
            raise CliError(
                f"Failed to write cost report: {exc}",
                code="write_failed",
                exit_code=EXIT_ERROR,
            ) from exc
        paths = {"json": str(json_path), "markdown": str(markdown_path)}
        return emit(
            args,
            {"written": paths},
            human=(
                "Cost comparison reports written:\n"
                f"  JSON: {json_path}\n"
                f"  Markdown: {markdown_path}"
            ),
        )

    return emit(args, report.to_dict(), human=report.to_json())


def _read_json_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CliError(
            f"Unable to read {label} {path}: {exc}",
            code="read_failed",
            exit_code=EXIT_ERROR,
        ) from exc
    except json.JSONDecodeError as exc:
        raise CliError(
            f"{label} is not valid JSON: {path}",
            code="invalid_json",
            exit_code=EXIT_USAGE,
        ) from exc
    if not isinstance(payload, dict):
        raise CliError(
            f"{label} must be a JSON object: {path}",
            code="invalid_json",
            exit_code=EXIT_USAGE,
        )
    return payload


def _parse_suite_args(values: list[str]) -> list[str]:
    suites: list[str] = []
    for value in values:
        suites.extend(item.strip() for item in value.split(",") if item.strip())
    if not suites:
        raise CliError(
            "At least one out-of-domain suite is required.",
            code="missing_suites",
            exit_code=EXIT_USAGE,
        )
    return suites


__all__ = [
    "add_cost_command",
    "add_generalization_command",
    "handle_cost",
    "handle_generalization",
]
