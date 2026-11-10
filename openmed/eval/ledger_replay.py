"""Backfill and deterministic replay for the benchmark run ledger.

This module adds two operations on top of the append-only
:class:`~openmed.eval.history.BenchmarkRunLedger`:

``backfill``
    Reconstruct historical ledger rows from previously emitted
    :class:`~openmed.eval.report.BenchmarkReport` JSON artifacts. Each report is
    keyed by ``(model_id, suite, manifest_hash, seed)`` and appended idempotently
    so re-running the same inputs never mutates prior rows and yields a
    byte-identical ledger.

``replay``
    Re-run a small deterministic regression detector over an existing ledger and
    emit a PHI-safe verdict report. For each ``(model_id, suite)`` group the last
    run is compared against the mean of its trailing baseline window; a seeded
    bootstrap confidence interval over the baseline window makes the ``seed``
    input load-bearing. The serialized verdict output is byte-identical for a
    fixed ledger and seed.

Expected local usage::

    python -m openmed.eval.ledger_replay backfill --ledger runs.json \
        --seed 7 report-a.json report-b.json
    python -m openmed.eval.ledger_replay replay --ledger runs.json --seed 7

Failure modes:

* A malformed ledger row (missing key fields or non-object metrics) fails the
  load with a non-zero exit and a message on stderr.
* Two historical reports that resolve to the same run key but disagree on their
  metrics raise a duplicate run-key conflict rather than silently overwriting.
* A report that does not carry (and is not given) a manifest hash is rejected;
  the run key is otherwise ambiguous.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openmed.eval.history import (
    HIGHER_IS_BETTER,
    IMPROVEMENT,
    REGRESSION,
    UNCHANGED,
    BenchmarkRunLedger,
    BenchmarkRunLedgerEntry,
    RunLedgerConflict,
    RunLedgerKey,
    load_run_ledger,
    metric_direction,
    write_run_ledger,
)
from openmed.eval.metrics import bootstrap_ci
from openmed.eval.report import BenchmarkReport

REPLAY_SCHEMA_VERSION = "openmed.ledger_replay.v1"
DEFAULT_REPLAY_SEED = 0
DEFAULT_REPLAY_RESAMPLES = 1000
DEFAULT_UNCHANGED_TOLERANCE = 1e-12


@dataclass(frozen=True)
class BackfillSummary:
    """Outcome of backfilling historical reports into a run ledger."""

    report_count: int
    added: tuple[str, ...]
    skipped: tuple[str, ...]

    @property
    def entry_count(self) -> int:
        """Return the number of ledger rows the reports resolved to."""

        return len(self.added) + len(self.skipped)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready payload."""

        return {
            "added": list(self.added),
            "report_count": self.report_count,
            "skipped": list(self.skipped),
        }


@dataclass(frozen=True)
class MetricReplayVerdict:
    """One deterministic candidate-vs-baseline verdict for a ledger metric."""

    model_id: str
    suite: str
    metric: str
    direction: str
    baseline_mean: float
    candidate: float
    delta: float
    verdict: str
    ci_lower: float
    ci_upper: float
    outside_ci: bool
    baseline_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready payload."""

        return {
            "baseline_count": self.baseline_count,
            "baseline_mean": self.baseline_mean,
            "candidate": self.candidate,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "delta": self.delta,
            "direction": self.direction,
            "metric": self.metric,
            "model_id": self.model_id,
            "outside_ci": self.outside_ci,
            "suite": self.suite,
            "verdict": self.verdict,
        }


@dataclass(frozen=True)
class LedgerReplayReport:
    """Deterministic regression verdicts replayed from a run ledger."""

    verdicts: tuple[MetricReplayVerdict, ...]
    seed: int = DEFAULT_REPLAY_SEED
    n_resamples: int = DEFAULT_REPLAY_RESAMPLES
    schema_version: str = REPLAY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready payload."""

        return {
            "n_resamples": self.n_resamples,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "verdicts": [verdict.to_dict() for verdict in self.verdicts],
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the report to byte-stable JSON with a trailing newline."""

        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                allow_nan=False,
                indent=indent,
                sort_keys=True,
            )
            + "\n"
        )


def format_run_key(key: RunLedgerKey) -> str:
    """Render a run key as a stable ``model::suite::manifest::seed`` string."""

    model_id, suite, manifest_hash, seed = key
    return f"{model_id}::{suite}::{manifest_hash}::{seed}"


def entry_from_report(
    report: BenchmarkReport | Mapping[str, Any] | str | Path,
    *,
    seed: int | None = None,
    manifest_hash: str | None = None,
) -> BenchmarkRunLedgerEntry:
    """Reconstruct a ledger entry from a historical benchmark report.

    The manifest hash and seed default to values embedded in the report metadata
    (``metadata.manifest_hash`` and ``metadata.seed``); the ``seed`` and
    ``manifest_hash`` arguments override those defaults. A manifest hash is
    required because the run key is otherwise ambiguous.
    """

    resolved = _read_report(report)
    metadata = dict(resolved.metadata)
    resolved_manifest = manifest_hash or _optional_str(metadata.get("manifest_hash"))
    if resolved_manifest is None:
        raise ValueError(
            "cannot backfill report without a manifest hash "
            "(pass --manifest-hash or set metadata.manifest_hash)"
        )
    resolved_seed = seed if seed is not None else metadata.get("seed")
    if resolved_seed is None:
        raise ValueError(
            "cannot backfill report without a seed (pass --seed or set metadata.seed)"
        )
    return BenchmarkRunLedgerEntry.from_report(
        resolved,
        manifest_hash=resolved_manifest,
        seed=int(resolved_seed),
    )


def backfill_ledger(
    ledger: BenchmarkRunLedger,
    entries: Iterable[BenchmarkRunLedgerEntry],
    *,
    report_count: int | None = None,
) -> tuple[BenchmarkRunLedger, BackfillSummary]:
    """Append *entries* to *ledger* idempotently and summarize the outcome.

    Rows whose run key already exists with identical payloads are skipped; a
    conflicting payload for an existing key raises
    :class:`~openmed.eval.history.RunLedgerConflict`.
    """

    added: list[str] = []
    skipped: list[str] = []
    current = ledger
    entry_list = list(entries)
    for entry in entry_list:
        updated = current.append(entry)
        if updated is current:
            skipped.append(format_run_key(entry.key))
        else:
            added.append(format_run_key(entry.key))
            current = updated
    summary = BackfillSummary(
        report_count=len(entry_list) if report_count is None else report_count,
        added=tuple(added),
        skipped=tuple(skipped),
    )
    return current, summary


def backfill_reports(
    ledger_path: str | Path,
    report_paths: Sequence[str | Path],
    *,
    seed: int | None = None,
    manifest_hash: str | None = None,
    indent: int = 2,
) -> BackfillSummary:
    """Backfill benchmark report JSON files into the ledger at *ledger_path*.

    The ledger file is created if absent and rewritten only when at least one new
    row is added, so repeated runs over identical inputs leave the bytes intact.
    """

    path = Path(ledger_path)
    ledger = load_run_ledger(path) if path.exists() else BenchmarkRunLedger()
    entries = [
        entry_from_report(report_path, seed=seed, manifest_hash=manifest_hash)
        for report_path in report_paths
    ]
    updated, summary = backfill_ledger(
        ledger,
        entries,
        report_count=len(report_paths),
    )
    if updated is not ledger or not path.exists():
        write_run_ledger(updated, path, indent=indent)
    return summary


def replay_ledger(
    ledger: BenchmarkRunLedger | str | Path,
    *,
    seed: int = DEFAULT_REPLAY_SEED,
    n_resamples: int = DEFAULT_REPLAY_RESAMPLES,
    metric_directions: Mapping[str, str] | None = None,
    tolerance: float = DEFAULT_UNCHANGED_TOLERANCE,
) -> LedgerReplayReport:
    """Replay a deterministic regression detector over a run ledger.

    For each ``(model_id, suite)`` group the runs are ordered by
    ``(generated_at, seed, manifest_hash)`` and the last run is scored against the
    mean of the trailing baseline window. A seeded bootstrap confidence interval
    over the baseline window is attached to every verdict, so the serialized
    report is byte-identical for a fixed ledger and seed.
    """

    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")
    resolved = load_run_ledger(ledger) if isinstance(ledger, (str, Path)) else ledger

    groups: dict[tuple[str, str], list[BenchmarkRunLedgerEntry]] = {}
    for entry in resolved.entries:
        groups.setdefault((entry.model_id, entry.suite), []).append(entry)

    verdicts: list[MetricReplayVerdict] = []
    for model_id, suite in sorted(groups):
        runs = sorted(
            groups[(model_id, suite)],
            key=lambda item: (item.generated_at or "", item.seed, item.manifest_hash),
        )
        if len(runs) < 2:
            continue
        candidate = runs[-1]
        baseline_runs = runs[:-1]
        for metric in sorted(candidate.metrics):
            baseline_values = [
                run.metrics[metric] for run in baseline_runs if metric in run.metrics
            ]
            if not baseline_values:
                continue
            candidate_value = candidate.metrics[metric]
            interval = bootstrap_ci(
                baseline_values,
                _mean,
                n_resamples=n_resamples,
                seed=seed,
            )
            baseline_mean = interval.point
            delta = candidate_value - baseline_mean
            direction = metric_direction(metric, metric_directions)
            verdicts.append(
                MetricReplayVerdict(
                    model_id=model_id,
                    suite=suite,
                    metric=metric,
                    direction=direction,
                    baseline_mean=baseline_mean,
                    candidate=candidate_value,
                    delta=delta,
                    verdict=_verdict(delta, direction, tolerance),
                    ci_lower=interval.lower,
                    ci_upper=interval.upper,
                    outside_ci=(
                        candidate_value < interval.lower
                        or candidate_value > interval.upper
                    ),
                    baseline_count=len(baseline_values),
                )
            )

    return LedgerReplayReport(
        verdicts=tuple(verdicts),
        seed=seed,
        n_resamples=n_resamples,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the benchmark ledger backfill and replay command parser."""

    parser = argparse.ArgumentParser(
        prog="python -m openmed.eval.ledger_replay",
        description=(
            "Backfill historical benchmark reports into the run ledger and "
            "replay a deterministic regression detector from it."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    backfill_parser = subparsers.add_parser(
        "backfill",
        help="Reconstruct historical ledger rows from benchmark report JSON.",
    )
    backfill_parser.add_argument(
        "--ledger",
        type=Path,
        required=True,
        help="Run ledger JSON file to create or extend.",
    )
    backfill_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run seed override applied to every report lacking metadata.seed.",
    )
    backfill_parser.add_argument(
        "--manifest-hash",
        default=None,
        help="Manifest hash override for reports lacking metadata.manifest_hash.",
    )
    backfill_parser.add_argument(
        "reports",
        nargs="+",
        type=Path,
        help="Historical BenchmarkReport JSON files to backfill.",
    )

    replay_parser = subparsers.add_parser(
        "replay",
        help="Replay a deterministic regression detector from the ledger.",
    )
    replay_parser.add_argument(
        "--ledger",
        type=Path,
        required=True,
        help="Run ledger JSON file to replay.",
    )
    replay_parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_REPLAY_SEED,
        help="Bootstrap seed; fixed seeds give byte-identical output.",
    )
    replay_parser.add_argument(
        "--resamples",
        type=int,
        default=DEFAULT_REPLAY_RESAMPLES,
        help="Bootstrap resample count for baseline-window intervals.",
    )
    replay_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the verdict JSON here instead of stdout.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark ledger backfill or replay command."""

    args = build_arg_parser().parse_args(argv)
    try:
        if args.command == "backfill":
            summary = backfill_reports(
                args.ledger,
                args.reports,
                seed=args.seed,
                manifest_hash=args.manifest_hash,
            )
            print(
                f"Backfilled {summary.report_count} report(s) into {args.ledger}: "
                f"{len(summary.added)} added, {len(summary.skipped)} unchanged"
            )
            return 0
        if args.command == "replay":  # pragma: no branch - argparse enforces command
            report = replay_ledger(
                args.ledger,
                seed=args.seed,
                n_resamples=args.resamples,
            )
            payload = report.to_json()
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(payload, encoding="utf-8")
                print(f"Wrote {len(report.verdicts)} verdict(s): {args.output}")
            else:
                print(payload, end="")
            return 0
    except (ValueError, RunLedgerConflict, OSError, json.JSONDecodeError) as exc:
        print(f"Ledger command failed: {exc}", file=sys.stderr)
        return 1
    return 2  # pragma: no cover - argparse requires a known command.


def _read_report(
    report: BenchmarkReport | Mapping[str, Any] | str | Path,
) -> BenchmarkReport:
    if isinstance(report, BenchmarkReport):
        return report
    if isinstance(report, (str, Path)):
        return BenchmarkReport.read_json(report)
    if isinstance(report, Mapping):
        return BenchmarkReport.from_dict(report)
    raise TypeError(f"unsupported benchmark report: {type(report).__name__}")


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(float(value) for value in values) / len(values)


def _verdict(delta: float, direction: str, tolerance: float) -> str:
    if abs(delta) <= tolerance:
        return UNCHANGED
    effect = delta if direction == HIGHER_IS_BETTER else -delta
    return IMPROVEMENT if effect > 0 else REGRESSION


def _optional_str(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


__all__ = [
    "DEFAULT_REPLAY_RESAMPLES",
    "DEFAULT_REPLAY_SEED",
    "DEFAULT_UNCHANGED_TOLERANCE",
    "REPLAY_SCHEMA_VERSION",
    "BackfillSummary",
    "LedgerReplayReport",
    "MetricReplayVerdict",
    "backfill_ledger",
    "backfill_reports",
    "build_arg_parser",
    "entry_from_report",
    "format_run_key",
    "main",
    "replay_ledger",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
