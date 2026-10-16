"""Benchmark-specific CLI command wiring."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path

from openmed.eval.generalization import cross_corpus_report

from ._output import EXIT_ERROR, EXIT_USAGE, CliError, emit


def add_metadata_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register offline suite discovery and descriptive report comparison."""
    listing = subparsers.add_parser("list-suites", help="List registered eval suites.")
    listing.set_defaults(handler=handle_list_suites)
    describe = subparsers.add_parser(
        "describe", help="Describe a suite's access policy."
    )
    describe.add_argument("suite")
    describe.set_defaults(handler=handle_describe)
    compare = subparsers.add_parser("compare", help="Compare aggregate report metrics.")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--fail-on-regression", action="store_true")
    compare.set_defaults(handler=handle_compare)


def _describe_suite(suite: str) -> dict:
    from openmed.eval.datasets.licenses import license_for
    from openmed.eval.suites import suite_metadata, validate_suite_name

    name = validate_suite_name(suite)
    metadata = suite_metadata(name)
    try:
        license_metadata = license_for(name).to_dict()
    except ValueError:
        license_metadata = metadata.get("license", metadata.get("licenses"))
    synthetic = name == "golden" or metadata.get("synthetic") is True
    if synthetic and license_metadata is None:
        license_metadata = "repository-owned synthetic fixtures only"
    if license_metadata is None:
        license_metadata = "source-specific; consult suite access requirements"
    if metadata.get("dua"):
        access = "local-dua-required"
    elif synthetic or metadata.get("phi_kind") == "synthetic":
        access = "public-synthetic"
    elif name == "shield":
        access = "public-sample-with-data-use-terms; full-corpus-approval-required"
    else:
        access = "user-supplied; upstream-license-required"
    labels = {}
    for key in (
        "label_mapping",
        "canonical_label_mapping",
        "entity_label_mapping",
        "relation_type_mapping",
        "entity_types",
        "labels",
        "n2c2_categories",
    ):
        if key in metadata:
            labels[key] = metadata[key]
    # Only repository-owned descriptive fields; never configured paths or rows.
    return {
        "suite": name,
        "task": metadata["task"],
        "access": access,
        "license": license_metadata,
        "categories": labels,
        "requirements": {
            key: metadata[key]
            for key in ("access", "redistribution", "data_boundary", "dua_boundary")
            if key in metadata
        },
    }


def handle_list_suites(args: argparse.Namespace) -> int:
    """List every registered suite without loading any corpus or model."""
    from openmed.eval.suites import REGISTERED_EVAL_SUITES

    rows = [_describe_suite(name) for name in REGISTERED_EVAL_SUITES]
    human = "\n".join(
        f"{row['suite']} | {row['task']} | {row['access']} | "
        f"{json.dumps(row['license'], sort_keys=True)}"
        for row in rows
    )
    return emit(args, {"suites": rows}, human=human)


def handle_describe(args: argparse.Namespace) -> int:
    """Print license, category mappings and access requirements for one suite."""
    failed = False
    try:
        row = _describe_suite(args.suite)
    except (KeyError, TypeError, ValueError):
        failed = True
    if failed:
        raise CliError(
            "Unknown benchmark suite.", code="invalid_suite", exit_code=EXIT_USAGE
        )
    return emit(args, row, human=json.dumps(row, indent=2, sort_keys=True))


_COMPARISON_METRICS = {
    "leakage.overall": "lower",
    "leakage_rate.overall": "lower",
    "leakage_rate": "lower",
    "critical_leakage_rate": "lower",
    "leakage": "lower",
    "character_recall.overall": "higher",
    "exact_span_f1.recall": "higher",
    "exact_span_f1.f1": "higher",
    "relaxed_span_f1.recall": "higher",
    "relaxed_span_f1.f1": "higher",
    "span.recall": "higher",
    "span.f1": "higher",
    "micro.recall": "higher",
    "micro.f1": "higher",
    "recall": "higher",
    "f1": "higher",
    "micro_f1": "higher",
}


def _metric(metrics: Mapping, path: str) -> float | None:
    value = metrics
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    if isinstance(value, Mapping):
        return None  # A container such as leakage is not a scalar metric.
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError("invalid aggregate rate")
    return float(value)


def handle_compare(args: argparse.Namespace) -> int:
    """Compare B minus A; report missing evidence instead of a false green."""
    from openmed.eval.report import read_reports
    from openmed.eval.suites import validate_suite_name

    failed = False
    try:
        for path in (args.baseline, args.candidate):
            if not path.is_file() or path.stat().st_size > 8 * 1024**2:
                raise ValueError("invalid report file")
        baseline, candidate = read_reports([args.baseline, args.candidate])
        suite = validate_suite_name(baseline.suite)
        if (
            candidate.suite != suite
            or baseline.fixture_count != candidate.fixture_count
        ):
            raise ValueError("incomparable suites or fixture counts")
        if baseline.fixture_count <= 0:
            raise ValueError("empty report")
        if not all(
            isinstance(report.metrics, Mapping) for report in (baseline, candidate)
        ):
            raise ValueError("invalid metrics")
        rows = []
        for name, direction in _COMPARISON_METRICS.items():
            a, b = _metric(baseline.metrics, name), _metric(candidate.metrics, name)
            if a is None and b is None:
                continue
            delta = None if a is None or b is None else b - a
            status = "missing"
            if delta is not None:
                signed = delta if direction == "higher" else -delta
                status = (
                    "regression"
                    if signed < 0
                    else "improved"
                    if signed > 0
                    else "unchanged"
                )
            rows.append(
                {
                    "metric": name,
                    "baseline": a,
                    "candidate": b,
                    "delta": delta,
                    "better": direction,
                    "status": status,
                }
            )
        if not rows:
            raise ValueError("no supported aggregate metrics")
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError):
        failed = True
    if failed:
        raise CliError(
            "Reports must be bounded, valid, nonempty and comparable.",
            code="invalid_reports",
            exit_code=EXIT_USAGE,
        )
    regressions = sum(row["status"] == "regression" for row in rows)
    missing = sum(row["status"] == "missing" for row in rows)
    verdict = (
        "regression" if regressions else "incomplete" if missing else "no-regression"
    )
    payload = {
        "suite": suite,
        "metrics": rows,
        "verdict": verdict,
        "regression_count": regressions,
        "missing_count": missing,
        "scope": "descriptive comparison only; not a release gate or corpus-equivalence proof",
    }
    human = (
        "\n".join(
            f"{row['metric']}: {row['delta'] if row['delta'] is not None else 'unavailable'} "
            f"({row['status']}; {row['better']} is better)"
            for row in rows
        )
        + f"\nVerdict: {verdict}\n{payload['scope']}"
    )
    result = emit(args, payload, human=human)
    return (
        EXIT_ERROR if args.fail_on_regression and (regressions or missing) else result
    )


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
