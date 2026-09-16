"""Benchmark-specific CLI command wiring."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from openmed.eval.generalization import cross_corpus_report

from ._output import EXIT_ERROR, EXIT_USAGE, CliError, emit


def add_cost_command(subparsers: argparse._SubParsersAction) -> None:
    """Register ``openmed benchmark cost``."""

    parser = subparsers.add_parser(
        "cost",
        help="Compare measured local throughput with cited cloud prices.",
    )
    parser.add_argument(
        "--perf",
        type=Path,
        required=True,
        help="Local PerfReport JSON with chars_per_document metadata.",
    )
    parser.add_argument(
        "--prices",
        type=Path,
        required=True,
        help="Versioned, cited cloud-price JSON table.",
    )
    parser.add_argument(
        "--hardware",
        type=Path,
        default=None,
        help=(
            "Optional hardware-cost JSON. When omitted, use the "
            "hardware_cost_model object in the perf report."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for cost-vs-cloud.json and cost-vs-cloud.md.",
    )
    parser.set_defaults(handler=handle_cost)


def handle_cost(args: argparse.Namespace) -> int:
    """Build and write a cited cost-vs-cloud report."""

    from openmed.eval.cost import (
        cost_vs_cloud_report,
        load_cloud_prices,
        load_cost_input,
    )

    try:
        perf = load_cost_input(args.perf, name="perf report")
        prices = load_cloud_prices(args.prices)
        if args.hardware is not None:
            hardware = load_cost_input(args.hardware, name="hardware cost model")
        else:
            embedded_hardware = perf.get("hardware_cost_model")
            if not isinstance(embedded_hardware, Mapping):
                raise ValueError(
                    "perf report must contain hardware_cost_model when "
                    "--hardware is omitted"
                )
            hardware = dict(embedded_hardware)
        report = cost_vs_cloud_report(perf, prices, hardware)
        json_path = report.write_json(args.output_dir / "cost-vs-cloud.json")
        markdown_path = report.write_markdown(args.output_dir / "cost-vs-cloud.md")
    except (OSError, TypeError, ValueError) as exc:
        raise CliError(
            f"Cost benchmark failed: {exc}",
            code="cost_benchmark_failed",
            exit_code=EXIT_ERROR,
        ) from exc

    written = {"json": str(json_path), "markdown": str(markdown_path)}
    return emit(
        args,
        {
            "input_fingerprint": report.input_fingerprint,
            "written": written,
        },
        human=(
            "Cost benchmark reports written:\n"
            f"  JSON: {json_path}\n"
            f"  Markdown: {markdown_path}"
        ),
    )


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
