#!/usr/bin/env python3
"""Render release status and leaderboard artifacts from source inputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from openmed.core.baseline import BASELINE_PATH, baseline_key, load_baseline_store
from openmed.core.model_registry import MANIFEST_PATH, load_manifest_rows
from openmed.eval.report import (
    BenchmarkReport,
    read_reports,
    write_benchmark_cards,
    write_leaderboard,
)


def render_status_page(
    *,
    manifest_rows: Iterable[Mapping[str, Any]],
    baseline_store: Mapping[str, Any],
    reports: Iterable[BenchmarkReport] = (),
    smoke_status: str = "green",
    smoke_failure_reason: str | None = None,
) -> str:
    """Render the release status page from manifest, baseline, and reports."""

    rows = list(manifest_rows)
    report_rows = list(reports)
    reports_by_key = _reports_by_key(report_rows, rows)
    aggregates = _manifest_aggregates(rows)

    lines = [
        "# OpenMed Release Status",
        "",
        "This page is rendered from the canonical model manifest, last-green",
        "baseline store, and latest BenchmarkReport inputs.",
        "",
        "| Family | Tier | Device | Format | Models | Current Leakage | Last Green Release | Last Regression + Rollback | Harness Freshness | Status |",
        "|---|---|---|---|---:|---:|---|---|---|---|",
    ]

    for key in sorted(aggregates):
        aggregate = aggregates[key]
        baseline = (baseline_store.get("entries") or {}).get(key, {})
        report = reports_by_key.get(key)
        status = _status_for_row(
            smoke_status=smoke_status,
            report=report,
            baseline=baseline,
        )
        lines.append(
            "| "
            f"`{aggregate['family']}` | "
            f"`{_display_value(aggregate['tier'])}` | "
            f"`{_display_value(report.device if report else aggregate['format'])}` | "
            f"`{aggregate['format']}` | "
            f"{aggregate['model_count']} | "
            f"{_format_percent(_metric_lookup(report, ('leakage.overall', 'leakage_rate.overall')))} | "
            f"`{_display_value(baseline.get('released'))}` | "
            f"`{_regression_status(baseline)}` | "
            f"`{_display_value(report.generated_at if report else None)}` | "
            f"`{status}` |"
        )

    if smoke_status == "red" and smoke_failure_reason:
        lines.extend(["", "## Smoke Test Failure", "", smoke_failure_reason])

    return "\n".join(lines) + "\n"


def write_status_page(
    output_path: str | Path,
    *,
    manifest_rows: Iterable[Mapping[str, Any]],
    baseline_store: Mapping[str, Any],
    reports: Iterable[BenchmarkReport] = (),
    smoke_status: str = "green",
    smoke_failure_reason: str | None = None,
) -> Path:
    """Write a release status page."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_status_page(
            manifest_rows=manifest_rows,
            baseline_store=baseline_store,
            reports=reports,
            smoke_status=smoke_status,
            smoke_failure_reason=smoke_failure_reason,
        ),
        encoding="utf-8",
    )
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the status surface generator CLI parser."""

    parser = argparse.ArgumentParser(
        description="Render OpenMed release status artifacts.",
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    parser.add_argument("--report", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, default=Path("docs/status/index.md"))
    parser.add_argument("--benchmarks-dir", type=Path, default=Path("docs/benchmarks"))
    parser.add_argument(
        "--leaderboard-dir", type=Path, default=Path("docs/leaderboard")
    )
    parser.add_argument(
        "--smoke-status",
        choices=["green", "red"],
        default="green",
    )
    parser.add_argument("--smoke-failure-reason", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Regenerate status, benchmark, and leaderboard artifacts."""

    args = build_arg_parser().parse_args(argv)
    manifest_rows = load_manifest_rows(args.manifest)
    baseline_store = load_baseline_store(args.baseline)
    reports = read_reports(args.report)

    write_benchmark_cards(
        reports,
        args.benchmarks_dir,
        manifest_rows=manifest_rows,
    )
    write_leaderboard(
        args.leaderboard_dir,
        manifest_rows=manifest_rows,
        reports=reports,
        baseline_store=baseline_store,
    )
    write_status_page(
        args.output,
        manifest_rows=manifest_rows,
        baseline_store=baseline_store,
        reports=reports,
        smoke_status=args.smoke_status,
        smoke_failure_reason=args.smoke_failure_reason,
    )
    return 0


def _manifest_aggregates(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    aggregates: dict[str, dict[str, Any]] = {}
    for row in rows:
        family = str(row.get("family") or "Unknown")
        tier = row.get("tier")
        for format_name in row.get("formats") or ["unknown"]:
            key = baseline_key(family, tier, str(format_name))
            aggregate = aggregates.setdefault(
                key,
                {
                    "family": family,
                    "tier": tier,
                    "format": str(format_name),
                    "model_count": 0,
                },
            )
            aggregate["model_count"] += 1
    return aggregates


def _reports_by_key(
    reports: Iterable[BenchmarkReport],
    manifest_rows: Iterable[Mapping[str, Any]],
) -> dict[str, BenchmarkReport]:
    rows = list(manifest_rows)
    reports_by_key: dict[str, BenchmarkReport] = {}
    for report in reports:
        row = _find_manifest_row(report.model_name, rows)
        if row is not None:
            family = str(
                row.get("family") or report.metadata.get("family") or "Unknown"
            )
            tier = row.get("tier")
            formats = row.get("formats") or [report.device]
            format_name = report.device if report.device in formats else formats[0]
        else:
            family = str(report.metadata.get("family") or "Unknown")
            tier = report.metadata.get("tier")
            format_name = str(report.metadata.get("format") or report.device)
        reports_by_key[baseline_key(family, tier, str(format_name))] = report
    return reports_by_key


def _find_manifest_row(
    model_name: str,
    rows: Iterable[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    for row in rows:
        if row.get("repo_id") == model_name:
            return row
    return None


def _metric_lookup(report: BenchmarkReport | None, keys: Sequence[str]) -> Any:
    if report is None:
        return None
    for key in keys:
        value = _nested_get(report.metrics, key)
        if value is not None:
            return value
    return None


def _nested_get(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _status_for_row(
    *,
    smoke_status: str,
    report: BenchmarkReport | None,
    baseline: Mapping[str, Any],
) -> str:
    if smoke_status == "red":
        return "red"
    if report is None:
        return "amber"
    if not baseline:
        return "amber"
    return "green"


def _regression_status(baseline: Mapping[str, Any]) -> str:
    metadata = baseline.get("metadata")
    if isinstance(metadata, Mapping):
        regression = metadata.get("last_regression")
        rollback = metadata.get("last_rollback")
        if regression or rollback:
            return f"{_display_value(regression)} / {_display_value(rollback)}"
    return "none recorded"


def _format_percent(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.2%}"
    return str(value)


def _display_value(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
