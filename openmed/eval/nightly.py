"""Nightly benchmark orchestration and gate-trend emission.

The nightly orchestrator drives an existing benchmark suite over multiple
seeds, appends the resulting rows to the append-only benchmark run ledger, and
emits a deterministic verdict report plus a byte-stable Markdown trend summary.

The orchestration is intentionally decoupled from any concrete model: callers
supply a per-seed runner that returns a :class:`~openmed.eval.report.BenchmarkReport`.
A thin :func:`benchmark_suite_runner` adapter is provided to wire the runner to
the existing harness (``run_benchmark``) over committed synthetic fixtures so
the whole flow stays offline and reproducible.

The verdict is consumable by the release-gate harness: it exposes the aggregate
run as a ``BenchmarkReport`` (metrics + metadata) via
:meth:`NightlyVerdict.to_benchmark_report`. The orchestrator only reads the
last-green baseline store; it never writes to ``gates/baseline.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from openmed.core.baseline import BASELINE_PATH
from openmed.eval.harness import (
    BenchmarkFixture,
    ModelRunner,
    load_fixtures,
    run_benchmark,
)
from openmed.eval.history import (
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    BenchmarkRunLedger,
    BenchmarkRunLedgerEntry,
    load_run_ledger,
    write_run_ledger,
)
from openmed.eval.report import BenchmarkReport

NIGHTLY_SCHEMA_VERSION = "openmed.nightly.v1"

GATE_PASS = "pass"
GATE_FAIL = "fail"

VERDICT_RELEASABLE = "releasable"
VERDICT_BLOCKED = "blocked"

#: Default compact ledger artifact location, alongside ``gates/baseline.json``.
DEFAULT_NIGHTLY_LEDGER_PATH = BASELINE_PATH.parent / "nightly_ledger.json"

#: Number of trailing runs surfaced in the emitted gate trend.
DEFAULT_TREND_WINDOW = 10

#: Per-seed benchmark runner: maps a seed to a benchmark report.
SeedRunner = Callable[[int], BenchmarkReport]

_VALID_DIRECTIONS = (HIGHER_IS_BETTER, LOWER_IS_BETTER)


@dataclass(frozen=True)
class GateOutcome:
    """One gate's pass/fail decision for a single benchmark run."""

    gate: str
    metric: str
    direction: str
    threshold: float
    blocking: bool
    value: float | None
    status: str

    @property
    def passed(self) -> bool:
        """Return ``True`` when the gate passed for this run."""

        return self.status == GATE_PASS

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready payload."""

        return {
            "blocking": bool(self.blocking),
            "direction": self.direction,
            "gate": self.gate,
            "metric": self.metric,
            "status": self.status,
            "threshold": float(self.threshold),
            "value": None if self.value is None else float(self.value),
        }


@dataclass(frozen=True)
class NightlyGate:
    """A metric threshold gate evaluated on each nightly benchmark run.

    ``blocking`` gates flip the nightly verdict to ``blocked`` on failure;
    non-blocking gates are advisory and never block the verdict. This mirrors
    the leakage-first-blocking, F1-advisory policy used by the release gates.
    """

    name: str
    metric: str
    threshold: float
    direction: str = HIGHER_IS_BETTER
    blocking: bool = True

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("nightly gate name must be a non-empty string")
        if not str(self.metric).strip():
            raise ValueError("nightly gate metric must be a non-empty string")
        if self.direction not in _VALID_DIRECTIONS:
            raise ValueError(f"invalid gate direction: {self.direction!r}")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "metric", str(self.metric))
        object.__setattr__(self, "threshold", float(self.threshold))
        object.__setattr__(self, "blocking", bool(self.blocking))

    def evaluate(self, metrics: Mapping[str, float]) -> GateOutcome:
        """Evaluate this gate against one run's flattened numeric metrics."""

        value = _lookup_metric(metrics, self.metric)
        if value is None:
            status = GATE_FAIL
        elif self.direction == HIGHER_IS_BETTER:
            status = GATE_PASS if value >= self.threshold else GATE_FAIL
        else:
            status = GATE_PASS if value <= self.threshold else GATE_FAIL
        return GateOutcome(
            gate=self.name,
            metric=self.metric,
            direction=self.direction,
            threshold=self.threshold,
            blocking=self.blocking,
            value=value,
            status=status,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready payload."""

        return {
            "blocking": self.blocking,
            "direction": self.direction,
            "metric": self.metric,
            "name": self.name,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class GateTrendPoint:
    """Gate outcomes for a single benchmark run in the trend window."""

    index: int
    seed: int
    manifest_hash: str
    generated_at: str | None
    outcomes: tuple[GateOutcome, ...]

    def status(self, gate_name: str) -> str | None:
        """Return the pass/fail status for *gate_name*, or ``None`` if absent."""

        for outcome in self.outcomes:
            if outcome.gate == gate_name:
                return outcome.status
        return None

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready payload."""

        return {
            "generated_at": self.generated_at,
            "index": self.index,
            "manifest_hash": self.manifest_hash,
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
            "seed": self.seed,
        }


@dataclass(frozen=True)
class NightlyVerdict:
    """Deterministic nightly verdict report with a recent gate trend."""

    model_id: str
    suite: str
    manifest_hash: str
    seeds: tuple[int, ...]
    gates: tuple[NightlyGate, ...]
    trend: tuple[GateTrendPoint, ...]
    verdict: str
    blocking_failures: tuple[str, ...]
    advisory_failures: tuple[str, ...]
    aggregate_metrics: Mapping[str, float]
    fixture_count: int | None = None
    generated_at: str | None = None
    schema_version: str = NIGHTLY_SCHEMA_VERSION

    @property
    def releasable(self) -> bool:
        """Return ``True`` when no blocking gate failed this nightly."""

        return self.verdict == VERDICT_RELEASABLE

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready payload."""

        return {
            "advisory_failures": list(self.advisory_failures),
            "aggregate_metrics": {
                metric: float(self.aggregate_metrics[metric])
                for metric in sorted(self.aggregate_metrics)
            },
            "blocking_failures": list(self.blocking_failures),
            "fixture_count": self.fixture_count,
            "gates": [gate.to_dict() for gate in self.gates],
            "generated_at": self.generated_at,
            "manifest_hash": self.manifest_hash,
            "model_id": self.model_id,
            "schema_version": self.schema_version,
            "seeds": list(self.seeds),
            "suite": self.suite,
            "trend": [point.to_dict() for point in self.trend],
            "verdict": self.verdict,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the verdict to byte-stable JSON."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write byte-stable verdict JSON to *path*."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    def to_markdown(self) -> str:
        """Render a byte-stable Markdown gate-trend summary."""

        seeds = ", ".join(str(seed) for seed in self.seeds)
        lines = [
            f"# Nightly Benchmark Trend: {self.suite}",
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| Model | `{self.model_id}` |",
            f"| Suite | `{self.suite}` |",
            f"| Manifest | `{self.manifest_hash}` |",
            f"| Seeds | `{seeds}` |",
            f"| Verdict | `{self.verdict}` |",
            f"| Schema | `{self.schema_version}` |",
            "",
            "## Gates",
            "",
            "| Gate | Metric | Direction | Threshold | Blocking |",
            "| --- | --- | --- | ---: | --- |",
        ]
        for gate in self.gates:
            lines.append(
                f"| `{gate.name}` | `{gate.metric}` | {gate.direction} | "
                f"{_format_number(gate.threshold)} | "
                f"{'yes' if gate.blocking else 'no'} |"
            )

        header = ["Run", "Seed", "Generated At"] + [gate.name for gate in self.gates]
        align = ["---", "---:", "---"] + ["---" for _ in self.gates]
        lines.extend(
            [
                "",
                "## Gate Trend",
                "",
                "| " + " | ".join(header) + " |",
                "| " + " | ".join(align) + " |",
            ]
        )
        for point in self.trend:
            cells = [
                str(point.index + 1),
                str(point.seed),
                f"`{_display(point.generated_at)}`",
            ]
            for gate in self.gates:
                cells.append(point.status(gate.name) or "n/a")
            lines.append("| " + " | ".join(cells) + " |")

        lines.extend(["", "## Blocking Failures", ""])
        lines.extend(_failure_lines(self.blocking_failures))
        lines.extend(["", "## Advisory Failures", ""])
        lines.extend(_failure_lines(self.advisory_failures))

        return "\n".join(lines) + "\n"

    def write_markdown(self, path: str | Path) -> Path:
        """Write the byte-stable Markdown trend summary to *path*."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
        return output_path

    def to_benchmark_report(self) -> BenchmarkReport:
        """Return the aggregate run as a release-gate-consumable report."""

        metadata = {
            "advisory_failures": list(self.advisory_failures),
            "blocking_failures": list(self.blocking_failures),
            "manifest_hash": self.manifest_hash,
            "nightly_schema_version": self.schema_version,
            "nightly_verdict": self.verdict,
            "seeds": list(self.seeds),
        }
        return BenchmarkReport(
            suite=self.suite,
            model_name=self.model_id,
            device="nightly",
            fixture_count=int(self.fixture_count or 0),
            metrics={
                metric: float(self.aggregate_metrics[metric])
                for metric in sorted(self.aggregate_metrics)
            },
            generated_at=self.generated_at,
            metadata=metadata,
        )


@dataclass(frozen=True)
class NightlyRun:
    """Result bundle from a nightly orchestration invocation."""

    verdict: NightlyVerdict
    ledger: BenchmarkRunLedger
    ledger_path: Path


def run_nightly(
    runner: SeedRunner,
    *,
    seeds: Sequence[int],
    gates: Sequence[NightlyGate],
    manifest_hash: str,
    model_id: str | None = None,
    ledger_path: str | Path = DEFAULT_NIGHTLY_LEDGER_PATH,
    generated_at: str | None = None,
    trend_window: int = DEFAULT_TREND_WINDOW,
    metadata: Mapping[str, Any] | None = None,
    verdict_json_path: str | Path | None = None,
    trend_markdown_path: str | Path | None = None,
) -> NightlyRun:
    """Run a nightly benchmark over *seeds* and emit a gate-trend verdict.

    The runner is invoked once per (deduplicated, sorted) seed. Each report is
    appended to the compact run ledger at *ledger_path* idempotently, so a
    re-run with identical inputs never mutates prior rows. The gate trend is the
    pass/fail history of *gates* across the trailing ``trend_window`` ledger
    rows for the resolved model and suite.
    """

    seed_list = tuple(sorted({int(seed) for seed in seeds}))
    if not seed_list:
        raise ValueError("run_nightly requires at least one seed")
    if not gates:
        raise ValueError("run_nightly requires at least one gate")

    ordered_gates = tuple(sorted(gates, key=lambda gate: gate.name))
    if len({gate.name for gate in ordered_gates}) != len(ordered_gates):
        raise ValueError("nightly gate names must be unique")

    ledger_file = Path(ledger_path)
    ledger = (
        load_run_ledger(ledger_file) if ledger_file.exists() else BenchmarkRunLedger()
    )

    appended: list[BenchmarkRunLedgerEntry] = []
    resolved_model_id = model_id
    resolved_suite: str | None = None
    fixture_count: int | None = None
    for seed in seed_list:
        report = runner(seed)
        entry_metadata: dict[str, Any] = {"nightly_seed": seed}
        if metadata:
            entry_metadata.update(dict(metadata))
        entry = BenchmarkRunLedgerEntry.from_report(
            report,
            manifest_hash=manifest_hash,
            seed=seed,
            model_id=model_id,
            metadata=entry_metadata,
        )
        ledger = ledger.append(entry)
        appended.append(entry)
        resolved_model_id = resolved_model_id or entry.model_id
        if resolved_suite is None:
            resolved_suite = entry.suite
        if fixture_count is None:
            fixture_count = entry.fixture_count

    write_run_ledger(ledger, ledger_file)

    resolved_model_id = str(resolved_model_id)
    resolved_suite = str(resolved_suite)

    window_rows = _select_rows(ledger, resolved_model_id, resolved_suite)
    if trend_window > 0:
        window_rows = window_rows[-trend_window:]
    trend = tuple(
        GateTrendPoint(
            index=index,
            seed=row.seed,
            manifest_hash=row.manifest_hash,
            generated_at=row.generated_at,
            outcomes=tuple(gate.evaluate(row.metrics) for gate in ordered_gates),
        )
        for index, row in enumerate(window_rows)
    )

    blocking_failures: list[str] = []
    advisory_failures: list[str] = []
    for gate in ordered_gates:
        failed = any(
            gate.evaluate(entry.metrics).status == GATE_FAIL for entry in appended
        )
        if not failed:
            continue
        if gate.blocking:
            blocking_failures.append(gate.name)
        else:
            advisory_failures.append(gate.name)

    verdict = VERDICT_BLOCKED if blocking_failures else VERDICT_RELEASABLE
    verdict_report = NightlyVerdict(
        model_id=resolved_model_id,
        suite=resolved_suite,
        manifest_hash=manifest_hash,
        seeds=seed_list,
        gates=ordered_gates,
        trend=trend,
        verdict=verdict,
        blocking_failures=tuple(blocking_failures),
        advisory_failures=tuple(advisory_failures),
        aggregate_metrics=_mean_metrics(appended),
        fixture_count=fixture_count,
        generated_at=generated_at,
    )

    if verdict_json_path is not None:
        verdict_report.write_json(verdict_json_path)
    if trend_markdown_path is not None:
        verdict_report.write_markdown(trend_markdown_path)

    return NightlyRun(
        verdict=verdict_report,
        ledger=ledger,
        ledger_path=ledger_file,
    )


def benchmark_suite_runner(
    fixtures: Sequence[BenchmarkFixture] | str | Path,
    *,
    suite: str,
    model_name: str,
    model_runner: ModelRunner | None = None,
    device: str = "cpu",
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SeedRunner:
    """Build a per-seed runner over committed fixtures using the harness.

    *fixtures* may be a fixture sequence or a JSON/JSONL path loaded via
    :func:`~openmed.eval.harness.load_fixtures`. The returned runner threads the
    seed into the harness bootstrap-CI and abstention seeds so each seed yields
    a deterministic, offline benchmark report.
    """

    loaded = (
        tuple(load_fixtures(fixtures))
        if isinstance(fixtures, (str, Path))
        else tuple(fixtures)
    )

    def _run(seed: int) -> BenchmarkReport:
        run_metadata: dict[str, Any] = {"nightly_seed": int(seed)}
        if metadata:
            run_metadata.update(dict(metadata))
        return run_benchmark(
            loaded,
            suite=suite,
            model_name=model_name,
            device=device,
            runner=model_runner,
            generated_at=generated_at,
            metadata=run_metadata,
            ci_seed=int(seed),
            abstention_seed=int(seed),
        )

    return _run


def _select_rows(
    ledger: BenchmarkRunLedger,
    model_id: str,
    suite: str,
) -> list[BenchmarkRunLedgerEntry]:
    rows = [
        entry
        for entry in ledger.entries
        if entry.model_id == model_id and entry.suite == suite
    ]
    return sorted(rows, key=lambda entry: (entry.generated_at or "", entry.seed))


def _mean_metrics(
    entries: Sequence[BenchmarkRunLedgerEntry],
) -> dict[str, float]:
    if not entries:
        return {}
    shared = set(entries[0].metrics)
    for entry in entries[1:]:
        shared &= set(entry.metrics)
    count = len(entries)
    return {
        metric: sum(entry.metrics[metric] for entry in entries) / count
        for metric in sorted(shared)
    }


def _lookup_metric(metrics: Mapping[str, float], metric: str) -> float | None:
    value = metrics.get(metric)
    if value is None:
        return None
    return float(value)


def _failure_lines(names: Sequence[str]) -> list[str]:
    if not names:
        return ["_None._"]
    return [f"- `{name}`" for name in names]


def _format_number(value: float) -> str:
    return f"{float(value):.6g}"


def _display(value: str | None) -> str:
    return "n/a" if value is None or value == "" else str(value)


__all__ = [
    "DEFAULT_NIGHTLY_LEDGER_PATH",
    "DEFAULT_TREND_WINDOW",
    "GATE_FAIL",
    "GATE_PASS",
    "NIGHTLY_SCHEMA_VERSION",
    "VERDICT_BLOCKED",
    "VERDICT_RELEASABLE",
    "GateOutcome",
    "GateTrendPoint",
    "NightlyGate",
    "NightlyRun",
    "NightlyVerdict",
    "SeedRunner",
    "benchmark_suite_runner",
    "run_nightly",
]
