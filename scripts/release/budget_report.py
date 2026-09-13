#!/usr/bin/env python3
"""Record and report advisory release compute, cost, and carbon budgets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from openmed.eval.budget_tracker import (
    DEFAULT_BUDGET_LEDGER,
    DEFAULT_ORCHESTRATOR_LEDGER,
    OVER,
    BudgetPolicy,
    BudgetThresholds,
    BudgetTrackingError,
    CostCarbonFactors,
    append_budget_entry,
    build_budget_entry,
    evaluate_budget,
    load_budget_ledger,
    load_stage_timings,
    rolling_weekly_totals,
)
from openmed.eval.fleet_metrics import compute_fleet_budget_metrics


def _add_policy_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = BudgetPolicy()
    parser.add_argument(
        "--per-run-warn-cost-usd",
        type=float,
        default=defaults.per_run_warn.estimated_cost_usd,
    )
    parser.add_argument(
        "--per-run-over-cost-usd",
        type=float,
        default=defaults.per_run_over.estimated_cost_usd,
    )
    parser.add_argument(
        "--weekly-warn-cost-usd",
        type=float,
        default=defaults.weekly_warn.estimated_cost_usd,
    )
    parser.add_argument(
        "--weekly-over-cost-usd",
        type=float,
        default=defaults.weekly_over.estimated_cost_usd,
    )


def _add_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--github-output", type=Path, default=None)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the release budget-report command line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    record = commands.add_parser(
        "record",
        help="Aggregate committed stage timings and append one budget row.",
    )
    record.add_argument("--timings", type=Path, required=True)
    record.add_argument("--ledger", type=Path, default=DEFAULT_BUDGET_LEDGER)
    record.add_argument(
        "--orchestrator-ledger",
        type=Path,
        default=DEFAULT_ORCHESTRATOR_LEDGER,
    )
    record.add_argument("--recorded-at", default=None)
    record.add_argument("--gpu-cost-per-hour-usd", type=float, default=2.0)
    record.add_argument("--runner-cost-per-minute-usd", type=float, default=0.008)
    record.add_argument("--gpu-power-kw", type=float, default=0.35)
    record.add_argument("--runner-power-kw", type=float, default=0.08)
    record.add_argument(
        "--carbon-intensity-kg-per-kwh",
        type=float,
        default=0.4,
    )
    _add_policy_arguments(record)
    _add_output_arguments(record)

    status = commands.add_parser(
        "status",
        help="Report current rolling spend before selecting the next batch.",
    )
    status.add_argument("--ledger", type=Path, default=DEFAULT_BUDGET_LEDGER)
    status.add_argument("--as-of", default=None)
    status.add_argument(
        "--throttle-on-over",
        action="store_true",
        help="Recommend a one-candidate batch when rolling spend is OVER.",
    )
    _add_policy_arguments(status)
    _add_output_arguments(status)
    return parser


def _threshold_with_cost(
    original: BudgetThresholds,
    estimated_cost_usd: float,
) -> BudgetThresholds:
    values = original.to_dict()
    values["estimated_cost_usd"] = estimated_cost_usd
    return BudgetThresholds.from_mapping(values)


def _policy(args: argparse.Namespace) -> BudgetPolicy:
    defaults = BudgetPolicy()
    return BudgetPolicy(
        per_run_warn=_threshold_with_cost(
            defaults.per_run_warn,
            args.per_run_warn_cost_usd,
        ),
        per_run_over=_threshold_with_cost(
            defaults.per_run_over,
            args.per_run_over_cost_usd,
        ),
        weekly_warn=_threshold_with_cost(
            defaults.weekly_warn,
            args.weekly_warn_cost_usd,
        ),
        weekly_over=_threshold_with_cost(
            defaults.weekly_over,
            args.weekly_over_cost_usd,
        ),
        window_days=defaults.window_days,
    )


def _write_report(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_github_outputs(path: Path | None, values: dict[str, str]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for key in sorted(values):
            handle.write(f"{key}={values[key]}\n")


def _warn_over(scope: str, metrics: Sequence[str]) -> None:
    names = ", ".join(metrics) or "configured thresholds"
    print(
        "::warning title=Advisory release budget OVER::"
        f"{scope} exceeded {names}; safety gates remain independent"
    )


def _record(args: argparse.Namespace) -> int:
    timings = load_stage_timings(args.timings)
    history = load_budget_ledger(args.ledger)
    policy = _policy(args)
    factors = CostCarbonFactors(
        gpu_cost_per_hour_usd=args.gpu_cost_per_hour_usd,
        runner_cost_per_minute_usd=args.runner_cost_per_minute_usd,
        gpu_power_kw=args.gpu_power_kw,
        runner_power_kw=args.runner_power_kw,
        carbon_intensity_kg_per_kwh=args.carbon_intensity_kg_per_kwh,
    )
    entry = build_budget_entry(
        run_id=timings.run_id,
        stages=timings.stages,
        history=history,
        recorded_at=args.recorded_at,
        factors=factors,
        policy=policy,
    )
    append_budget_entry(
        entry,
        ledger_path=args.ledger,
        orchestrator_ledger_path=args.orchestrator_ledger,
    )
    fleet = compute_fleet_budget_metrics(
        (*history, entry),
        as_of=entry.recorded_at,
        window_days=policy.window_days,
    )
    payload = {
        "advisory": True,
        "fleet": fleet.to_dict(),
        "run": entry.to_dict(),
    }
    _write_report(args.output, payload)
    _write_github_outputs(
        args.github_output,
        {
            "throttle_recommended": str(entry.throttle_recommended).lower(),
            "verdict": entry.verdict,
        },
    )
    if entry.verdict == OVER:
        metrics = tuple(
            sorted(
                {
                    *entry.per_run_budget.exceeded_metrics,
                    *entry.rolling_weekly_budget.exceeded_metrics,
                }
            )
        )
        _warn_over("release run budget", metrics)
    print(
        f"release budget {entry.run_id}: {entry.verdict}; "
        f"${entry.totals.estimated_cost_usd:.6f}; "
        f"{entry.totals.carbon_kg:.6f} kgCO2e"
    )
    return 0


def _status(args: argparse.Namespace) -> int:
    entries = load_budget_ledger(args.ledger)
    policy = _policy(args)
    totals = rolling_weekly_totals(
        entries,
        as_of=args.as_of,
        window_days=policy.window_days,
    )
    decision = evaluate_budget(
        totals,
        warning=policy.weekly_warn,
        maximum=policy.weekly_over,
    )
    throttle = args.throttle_on_over and decision.verdict == OVER
    metrics = compute_fleet_budget_metrics(
        entries,
        as_of=args.as_of,
        window_days=policy.window_days,
    )
    _write_report(
        args.output,
        {
            "advisory": True,
            "budget": decision.to_dict(),
            "fleet": metrics.to_dict(),
            "throttle_recommended": decision.verdict == OVER,
            "throttle_selected": throttle,
        },
    )
    _write_github_outputs(
        args.github_output,
        {
            "max_candidates": "1" if throttle else "0",
            "throttle_recommended": str(decision.verdict == OVER).lower(),
            "verdict": decision.verdict,
        },
    )
    if decision.verdict == OVER:
        _warn_over("rolling weekly release budget", decision.exceeded_metrics)
    print(
        f"rolling release budget: {decision.verdict}; "
        f"${totals.estimated_cost_usd:.6f}; {totals.carbon_kg:.6f} kgCO2e"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected advisory budget operation."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.github_output is None:
        github_output = os.environ.get("GITHUB_OUTPUT")
        if github_output:
            args.github_output = Path(github_output)
    try:
        if args.command == "record":
            return _record(args)
        return _status(args)
    except (BudgetTrackingError, OSError, json.JSONDecodeError):
        print("release budget report failed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
