"""Deterministic cost, compute, and carbon tracking for release runs.

The tracker consumes PHI-free stage timings emitted by the release
orchestrator. It deliberately estimates spend and carbon from reviewed local
factors instead of contacting cloud billing services. Budget verdicts are
advisory: callers may use ``throttle_recommended`` to reduce a future queue,
but a budget verdict never changes a safety-gate decision.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUDGET_LEDGER = ROOT / "gates" / "budget_ledger.jsonl"
DEFAULT_ORCHESTRATOR_LEDGER = ROOT / "gates" / "release_runs.jsonl"

BUDGET_SCHEMA_VERSION = "openmed.release_budget.v1"
STAGE_TIMINGS_SCHEMA_VERSION = "openmed.release_stage_timings.v1"

WITHIN = "WITHIN"
WARN = "WARN"
OVER = "OVER"

BENCHMARK_REFRESH = "benchmark_refresh"
TRAINING = "training"
VALID_WORKLOADS = frozenset({BENCHMARK_REFRESH, TRAINING})

_METRIC_NAMES = (
    "estimated_cost_usd",
    "carbon_kg",
    "energy_kwh",
    "gpu_hours",
    "runner_minutes",
    "wall_clock_seconds",
)
_QUANTUM = Decimal("0.000001")
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")
_PHI_PATTERNS = (
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
)


class BudgetTrackingError(ValueError):
    """Raised when a budget artifact violates its schema or safety contract."""


def _decimal(value: Any, field: str) -> Decimal:
    if isinstance(value, bool):
        raise BudgetTrackingError(f"{field} must be a finite non-negative number")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise BudgetTrackingError(
            f"{field} must be a finite non-negative number"
        ) from exc
    if not number.is_finite() or number < 0:
        raise BudgetTrackingError(f"{field} must be a finite non-negative number")
    return number


def _number(value: Any, field: str) -> float:
    rounded = _decimal(value, field).quantize(_QUANTUM, rounding=ROUND_HALF_UP)
    return 0.0 if rounded == 0 else float(rounded)


def _identifier(value: Any, field: str) -> str:
    text = str(value).strip()
    if not _SAFE_IDENTIFIER_RE.fullmatch(text):
        raise BudgetTrackingError(f"{field} must be a safe identifier")
    if any(pattern.search(text) for pattern in _PHI_PATTERNS):
        raise BudgetTrackingError(f"{field} must not contain PHI-shaped data")
    return text


def _reject_unknown_fields(
    value: Mapping[str, Any],
    allowed: set[str],
    label: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise BudgetTrackingError(f"{label} contains unsupported fields: {unknown}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _payload_hash(value: Any) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _utc_datetime(value: datetime | str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError as exc:
            raise BudgetTrackingError("timestamp must be ISO-8601") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _timestamp(value: datetime | str | None) -> str:
    return _utc_datetime(value).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class CostCarbonFactors:
    """Reviewed estimation factors applied to raw compute measurements."""

    gpu_cost_per_hour_usd: float = 2.0
    runner_cost_per_minute_usd: float = 0.008
    gpu_power_kw: float = 0.35
    runner_power_kw: float = 0.08
    carbon_intensity_kg_per_kwh: float = 0.4

    def __post_init__(self) -> None:
        for field in (
            "gpu_cost_per_hour_usd",
            "runner_cost_per_minute_usd",
            "gpu_power_kw",
            "runner_power_kw",
            "carbon_intensity_kg_per_kwh",
        ):
            object.__setattr__(self, field, _number(getattr(self, field), field))

    def to_dict(self) -> dict[str, float]:
        """Return a stable JSON-ready representation."""

        return {
            "carbon_intensity_kg_per_kwh": self.carbon_intensity_kg_per_kwh,
            "gpu_cost_per_hour_usd": self.gpu_cost_per_hour_usd,
            "gpu_power_kw": self.gpu_power_kw,
            "runner_cost_per_minute_usd": self.runner_cost_per_minute_usd,
            "runner_power_kw": self.runner_power_kw,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CostCarbonFactors":
        """Load factors from a JSON-style mapping."""

        _reject_unknown_fields(
            value,
            {
                "carbon_intensity_kg_per_kwh",
                "gpu_cost_per_hour_usd",
                "gpu_power_kw",
                "runner_cost_per_minute_usd",
                "runner_power_kw",
            },
            "cost/carbon factors",
        )

        return cls(
            gpu_cost_per_hour_usd=value.get("gpu_cost_per_hour_usd", 2.0),
            runner_cost_per_minute_usd=value.get("runner_cost_per_minute_usd", 0.008),
            gpu_power_kw=value.get("gpu_power_kw", 0.35),
            runner_power_kw=value.get("runner_power_kw", 0.08),
            carbon_intensity_kg_per_kwh=value.get("carbon_intensity_kg_per_kwh", 0.4),
        )


@dataclass(frozen=True)
class StageTiming:
    """PHI-free resource measurements for one orchestrator stage."""

    stage: str
    candidate_id: str
    family: str
    tier: str
    workload: str
    gpu_hours: float = 0.0
    runner_minutes: float = 0.0
    wall_clock_seconds: float = 0.0

    def __post_init__(self) -> None:
        for field in ("stage", "candidate_id", "family", "tier"):
            object.__setattr__(
                self,
                field,
                _identifier(getattr(self, field), field),
            )
        workload = str(self.workload).strip().lower()
        if workload not in VALID_WORKLOADS:
            raise BudgetTrackingError(
                f"workload must be one of {sorted(VALID_WORKLOADS)}"
            )
        object.__setattr__(self, "workload", workload)
        for field in ("gpu_hours", "runner_minutes", "wall_clock_seconds"):
            object.__setattr__(self, field, _number(getattr(self, field), field))

    @classmethod
    def from_elapsed(
        cls,
        *,
        stage: str,
        candidate_id: str,
        family: str,
        tier: str,
        workload: str,
        elapsed_seconds: float,
        gpu: bool,
    ) -> "StageTiming":
        """Build a timing from a monotonic elapsed duration."""

        elapsed = _number(elapsed_seconds, "elapsed_seconds")
        return cls(
            stage=stage,
            candidate_id=candidate_id,
            family=family,
            tier=tier,
            workload=workload,
            gpu_hours=elapsed / 3600 if gpu else 0.0,
            runner_minutes=0.0 if gpu else elapsed / 60,
            wall_clock_seconds=elapsed,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-ready representation."""

        return {
            "candidate_id": self.candidate_id,
            "family": self.family,
            "gpu_hours": self.gpu_hours,
            "runner_minutes": self.runner_minutes,
            "stage": self.stage,
            "tier": self.tier,
            "wall_clock_seconds": self.wall_clock_seconds,
            "workload": self.workload,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StageTiming":
        """Load one stage timing from a JSON-style mapping."""

        _reject_unknown_fields(
            value,
            {
                "candidate_id",
                "family",
                "gpu_hours",
                "runner_minutes",
                "stage",
                "tier",
                "wall_clock_seconds",
                "workload",
            },
            "stage timing",
        )

        try:
            return cls(
                stage=value["stage"],
                candidate_id=value["candidate_id"],
                family=value["family"],
                tier=value["tier"],
                workload=value["workload"],
                gpu_hours=value.get("gpu_hours", 0.0),
                runner_minutes=value.get("runner_minutes", 0.0),
                wall_clock_seconds=value["wall_clock_seconds"],
            )
        except KeyError as exc:
            raise BudgetTrackingError(
                f"stage timing is missing field {exc.args[0]!r}"
            ) from exc


def _stage_sort_key(stage: StageTiming) -> tuple[str, ...]:
    return (
        stage.candidate_id,
        stage.stage,
        stage.family,
        stage.tier,
        stage.workload,
    )


@dataclass(frozen=True)
class StageTimingDocument:
    """Committed raw timing input linked to an orchestrator run."""

    run_id: str
    orchestrator_run_id: str
    stages: tuple[StageTiming, ...]
    schema_version: str = STAGE_TIMINGS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        run_id = _identifier(self.run_id, "run_id")
        orchestrator_run_id = _identifier(
            self.orchestrator_run_id, "orchestrator_run_id"
        )
        if run_id != orchestrator_run_id:
            raise BudgetTrackingError(
                "run_id must match the linked orchestrator_run_id"
            )
        if self.schema_version != STAGE_TIMINGS_SCHEMA_VERSION:
            raise BudgetTrackingError("unsupported stage timing schema version")
        stages = tuple(sorted(self.stages, key=_stage_sort_key))
        if not stages:
            raise BudgetTrackingError("stage timing document must not be empty")
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "orchestrator_run_id", orchestrator_run_id)
        object.__setattr__(self, "stages", stages)
        _assert_no_phi(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical committed timing payload."""

        return {
            "orchestrator_run_id": self.orchestrator_run_id,
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "stages": [stage.to_dict() for stage in self.stages],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StageTimingDocument":
        """Load a timing document from a JSON-style mapping."""

        _reject_unknown_fields(
            value,
            {"orchestrator_run_id", "run_id", "schema_version", "stages"},
            "stage timing document",
        )

        raw_stages = value.get("stages")
        if not isinstance(raw_stages, Sequence) or isinstance(raw_stages, str):
            raise BudgetTrackingError("stage timings must be an array")
        if any(not isinstance(stage, Mapping) for stage in raw_stages):
            raise BudgetTrackingError("each stage timing must be an object")
        try:
            return cls(
                run_id=value["run_id"],
                orchestrator_run_id=value["orchestrator_run_id"],
                stages=tuple(
                    StageTiming.from_mapping(stage)
                    for stage in raw_stages
                    if isinstance(stage, Mapping)
                ),
                schema_version=str(value.get("schema_version", "")),
            )
        except KeyError as exc:
            raise BudgetTrackingError(
                f"stage timing document is missing {exc.args[0]!r}"
            ) from exc

    def write_json(self, path: str | Path) -> Path:
        """Write byte-stable timing JSON for review and later replay."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output


def write_stage_timings(
    stages: Iterable[StageTiming],
    *,
    run_id: str,
    path: str | Path,
) -> Path:
    """Commit raw stage timings linked to *run_id* as deterministic JSON."""

    document = StageTimingDocument(
        run_id=run_id,
        orchestrator_run_id=run_id,
        stages=tuple(stages),
    )
    return document.write_json(path)


def load_stage_timings(path: str | Path) -> StageTimingDocument:
    """Load and validate a committed timing document."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise BudgetTrackingError("stage timing document must be an object")
    return StageTimingDocument.from_mapping(payload)


@dataclass(frozen=True)
class BudgetTotals:
    """Deterministic aggregate compute, cost, energy, and carbon totals."""

    stage_count: int = 0
    gpu_hours: float = 0.0
    runner_minutes: float = 0.0
    wall_clock_seconds: float = 0.0
    energy_kwh: float = 0.0
    carbon_kg: float = 0.0
    estimated_cost_usd: float = 0.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.stage_count, int)
            or isinstance(self.stage_count, bool)
            or self.stage_count < 0
        ):
            raise BudgetTrackingError("stage_count must be a non-negative integer")
        for field in _METRIC_NAMES:
            object.__setattr__(self, field, _number(getattr(self, field), field))

    def to_dict(self) -> dict[str, int | float]:
        """Return stable JSON-ready totals."""

        return {
            "carbon_kg": self.carbon_kg,
            "energy_kwh": self.energy_kwh,
            "estimated_cost_usd": self.estimated_cost_usd,
            "gpu_hours": self.gpu_hours,
            "runner_minutes": self.runner_minutes,
            "stage_count": self.stage_count,
            "wall_clock_seconds": self.wall_clock_seconds,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BudgetTotals":
        """Load totals from a JSON-style mapping."""

        _reject_unknown_fields(
            value,
            {"stage_count", *_METRIC_NAMES},
            "budget totals",
        )

        return cls(
            stage_count=value.get("stage_count", 0),
            gpu_hours=value.get("gpu_hours", 0.0),
            runner_minutes=value.get("runner_minutes", 0.0),
            wall_clock_seconds=value.get("wall_clock_seconds", 0.0),
            energy_kwh=value.get("energy_kwh", 0.0),
            carbon_kg=value.get("carbon_kg", 0.0),
            estimated_cost_usd=value.get("estimated_cost_usd", 0.0),
        )

    @classmethod
    def combine(cls, totals: Iterable["BudgetTotals"]) -> "BudgetTotals":
        """Sum aggregates using decimal arithmetic independent of input order."""

        items = list(totals)
        return cls(
            stage_count=sum(item.stage_count for item in items),
            **{
                metric: sum(
                    (_decimal(getattr(item, metric), metric) for item in items),
                    Decimal(0),
                )
                for metric in _METRIC_NAMES
            },
        )


def aggregate_stage_timings(
    stages: Iterable[StageTiming],
    factors: CostCarbonFactors = CostCarbonFactors(),
) -> BudgetTotals:
    """Aggregate stage measurements with deterministic decimal arithmetic."""

    materialized = tuple(stages)
    gpu_hours = sum(
        (_decimal(stage.gpu_hours, "gpu_hours") for stage in materialized),
        Decimal(0),
    )
    runner_minutes = sum(
        (_decimal(stage.runner_minutes, "runner_minutes") for stage in materialized),
        Decimal(0),
    )
    wall_clock_seconds = sum(
        (
            _decimal(stage.wall_clock_seconds, "wall_clock_seconds")
            for stage in materialized
        ),
        Decimal(0),
    )
    gpu_energy = gpu_hours * _decimal(factors.gpu_power_kw, "gpu_power_kw")
    runner_energy = (
        runner_minutes
        / Decimal(60)
        * _decimal(factors.runner_power_kw, "runner_power_kw")
    )
    energy_kwh = gpu_energy + runner_energy
    carbon_kg = energy_kwh * _decimal(
        factors.carbon_intensity_kg_per_kwh,
        "carbon_intensity_kg_per_kwh",
    )
    estimated_cost_usd = gpu_hours * _decimal(
        factors.gpu_cost_per_hour_usd,
        "gpu_cost_per_hour_usd",
    ) + runner_minutes * _decimal(
        factors.runner_cost_per_minute_usd,
        "runner_cost_per_minute_usd",
    )
    return BudgetTotals(
        stage_count=len(materialized),
        gpu_hours=gpu_hours,
        runner_minutes=runner_minutes,
        wall_clock_seconds=wall_clock_seconds,
        energy_kwh=energy_kwh,
        carbon_kg=carbon_kg,
        estimated_cost_usd=estimated_cost_usd,
    )


@dataclass(frozen=True)
class BudgetThresholds:
    """Upper advisory thresholds for all tracked budget dimensions."""

    estimated_cost_usd: float
    carbon_kg: float
    energy_kwh: float
    gpu_hours: float
    runner_minutes: float
    wall_clock_seconds: float

    def __post_init__(self) -> None:
        for field in _METRIC_NAMES:
            object.__setattr__(self, field, _number(getattr(self, field), field))

    def to_dict(self) -> dict[str, float]:
        """Return stable JSON-ready thresholds."""

        return {metric: getattr(self, metric) for metric in _METRIC_NAMES}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BudgetThresholds":
        """Load thresholds from a JSON-style mapping."""

        _reject_unknown_fields(value, set(_METRIC_NAMES), "budget thresholds")

        try:
            return cls(**{metric: value[metric] for metric in _METRIC_NAMES})
        except KeyError as exc:
            raise BudgetTrackingError(
                f"budget thresholds are missing {exc.args[0]!r}"
            ) from exc


DEFAULT_PER_RUN_WARN = BudgetThresholds(20, 2, 5, 4, 240, 14_400)
DEFAULT_PER_RUN_OVER = BudgetThresholds(30, 3, 7.5, 6, 360, 21_600)
DEFAULT_WEEKLY_WARN = BudgetThresholds(100, 10, 25, 20, 1_200, 72_000)
DEFAULT_WEEKLY_OVER = BudgetThresholds(150, 15, 37.5, 30, 1_800, 108_000)


@dataclass(frozen=True)
class BudgetPolicy:
    """Per-run and rolling-week advisory budget policy."""

    per_run_warn: BudgetThresholds = DEFAULT_PER_RUN_WARN
    per_run_over: BudgetThresholds = DEFAULT_PER_RUN_OVER
    weekly_warn: BudgetThresholds = DEFAULT_WEEKLY_WARN
    weekly_over: BudgetThresholds = DEFAULT_WEEKLY_OVER
    window_days: int = 7

    def __post_init__(self) -> None:
        if (
            not isinstance(self.window_days, int)
            or isinstance(self.window_days, bool)
            or self.window_days <= 0
        ):
            raise BudgetTrackingError("window_days must be a positive integer")
        for warning, maximum, label in (
            (self.per_run_warn, self.per_run_over, "per-run"),
            (self.weekly_warn, self.weekly_over, "weekly"),
        ):
            for metric in _METRIC_NAMES:
                if getattr(warning, metric) > getattr(maximum, metric):
                    raise BudgetTrackingError(
                        f"{label} warning threshold exceeds OVER threshold for {metric}"
                    )

    def to_dict(self) -> dict[str, Any]:
        """Return the complete reviewed policy."""

        return {
            "per_run_over": self.per_run_over.to_dict(),
            "per_run_warn": self.per_run_warn.to_dict(),
            "weekly_over": self.weekly_over.to_dict(),
            "weekly_warn": self.weekly_warn.to_dict(),
            "window_days": self.window_days,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BudgetPolicy":
        """Load a policy from a JSON-style mapping."""

        _reject_unknown_fields(
            value,
            {
                "per_run_over",
                "per_run_warn",
                "weekly_over",
                "weekly_warn",
                "window_days",
            },
            "budget policy",
        )

        try:
            return cls(
                per_run_warn=BudgetThresholds.from_mapping(value["per_run_warn"]),
                per_run_over=BudgetThresholds.from_mapping(value["per_run_over"]),
                weekly_warn=BudgetThresholds.from_mapping(value["weekly_warn"]),
                weekly_over=BudgetThresholds.from_mapping(value["weekly_over"]),
                window_days=int(value.get("window_days", 7)),
            )
        except (KeyError, TypeError) as exc:
            raise BudgetTrackingError("budget policy is malformed") from exc


@dataclass(frozen=True)
class BudgetDecision:
    """Advisory threshold result with the metrics that caused it."""

    verdict: str
    exceeded_metrics: tuple[str, ...] = ()
    warned_metrics: tuple[str, ...] = ()
    gating: bool = False

    def __post_init__(self) -> None:
        if self.verdict not in {WITHIN, WARN, OVER}:
            raise BudgetTrackingError("budget verdict is invalid")
        if self.gating:
            raise BudgetTrackingError("release compute budgets must remain advisory")
        for metric in (*self.exceeded_metrics, *self.warned_metrics):
            if metric not in _METRIC_NAMES:
                raise BudgetTrackingError("budget decision names an unknown metric")

    def to_dict(self) -> dict[str, Any]:
        """Return a stable, explicitly non-gating decision."""

        return {
            "exceeded_metrics": list(self.exceeded_metrics),
            "gating": False,
            "verdict": self.verdict,
            "warned_metrics": list(self.warned_metrics),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BudgetDecision":
        """Load a decision from a JSON-style mapping."""

        _reject_unknown_fields(
            value,
            {"exceeded_metrics", "gating", "verdict", "warned_metrics"},
            "budget decision",
        )

        return cls(
            verdict=str(value.get("verdict", "")),
            exceeded_metrics=tuple(value.get("exceeded_metrics") or ()),
            warned_metrics=tuple(value.get("warned_metrics") or ()),
            gating=bool(value.get("gating", False)),
        )


def evaluate_budget(
    totals: BudgetTotals,
    *,
    warning: BudgetThresholds,
    maximum: BudgetThresholds,
) -> BudgetDecision:
    """Evaluate advisory thresholds, with OVER taking precedence over WARN."""

    exceeded = tuple(
        metric
        for metric in _METRIC_NAMES
        if getattr(totals, metric) > getattr(maximum, metric)
    )
    warned = tuple(
        metric
        for metric in _METRIC_NAMES
        if getattr(totals, metric) > getattr(warning, metric) and metric not in exceeded
    )
    if exceeded:
        verdict = OVER
    elif warned:
        verdict = WARN
    else:
        verdict = WITHIN
    return BudgetDecision(
        verdict=verdict,
        exceeded_metrics=exceeded,
        warned_metrics=warned,
    )


def _group_stage_totals(
    stages: Sequence[StageTiming],
    factors: CostCarbonFactors,
    key,
) -> dict[str, BudgetTotals]:
    grouped: dict[str, list[StageTiming]] = {}
    for stage in stages:
        grouped.setdefault(str(key(stage)), []).append(stage)
    return {
        label: aggregate_stage_timings(grouped[label], factors)
        for label in sorted(grouped)
    }


def _totals_mapping(value: Mapping[str, Any]) -> dict[str, BudgetTotals]:
    parsed: dict[str, BudgetTotals] = {}
    for label, totals in value.items():
        if not isinstance(totals, Mapping):
            raise BudgetTrackingError("budget breakdown totals must be objects")
        parsed[_identifier(label, "breakdown key")] = BudgetTotals.from_mapping(totals)
    return parsed


@dataclass(frozen=True)
class BudgetLedgerEntry:
    """One append-only, orchestrator-linked release budget record."""

    run_id: str
    orchestrator_run_id: str
    recorded_at: str
    stages: tuple[StageTiming, ...]
    factors: CostCarbonFactors
    policy: BudgetPolicy
    totals: BudgetTotals
    by_family: Mapping[str, BudgetTotals]
    by_tier: Mapping[str, BudgetTotals]
    by_family_tier: Mapping[str, BudgetTotals]
    by_workload: Mapping[str, BudgetTotals]
    rolling_weekly_totals: BudgetTotals
    per_run_budget: BudgetDecision
    rolling_weekly_budget: BudgetDecision
    verdict: str
    throttle_recommended: bool
    aggregation_hash: str
    record_hash: str
    schema_version: str = BUDGET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        run_id = _identifier(self.run_id, "run_id")
        orchestrator_run_id = _identifier(
            self.orchestrator_run_id, "orchestrator_run_id"
        )
        if run_id != orchestrator_run_id:
            raise BudgetTrackingError(
                "run_id must match the linked orchestrator_run_id"
            )
        if self.schema_version != BUDGET_SCHEMA_VERSION:
            raise BudgetTrackingError("unsupported release budget schema version")
        if self.verdict not in {WITHIN, WARN, OVER}:
            raise BudgetTrackingError("budget ledger verdict is invalid")
        if self.throttle_recommended != (self.verdict == OVER):
            raise BudgetTrackingError("throttle recommendation is inconsistent")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.aggregation_hash):
            raise BudgetTrackingError("aggregation_hash must be a sha256 digest")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.record_hash):
            raise BudgetTrackingError("record_hash must be a sha256 digest")
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "orchestrator_run_id", orchestrator_run_id)
        object.__setattr__(self, "recorded_at", _timestamp(self.recorded_at))
        object.__setattr__(
            self, "stages", tuple(sorted(self.stages, key=_stage_sort_key))
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the complete PHI-free ledger row."""

        return {
            "aggregation_hash": self.aggregation_hash,
            "breakdown": {
                "family": _serialize_totals_mapping(self.by_family),
                "family_tier": _serialize_totals_mapping(self.by_family_tier),
                "tier": _serialize_totals_mapping(self.by_tier),
                "workload": _serialize_totals_mapping(self.by_workload),
            },
            "factors": self.factors.to_dict(),
            "orchestrator_run_id": self.orchestrator_run_id,
            "per_run_budget": self.per_run_budget.to_dict(),
            "policy": self.policy.to_dict(),
            "record_hash": self.record_hash,
            "recorded_at": self.recorded_at,
            "rolling_weekly_budget": self.rolling_weekly_budget.to_dict(),
            "rolling_weekly_totals": self.rolling_weekly_totals.to_dict(),
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "stages": [stage.to_dict() for stage in self.stages],
            "throttle_recommended": self.throttle_recommended,
            "totals": self.totals.to_dict(),
            "verdict": self.verdict,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BudgetLedgerEntry":
        """Load and verify one committed budget ledger row."""

        _reject_unknown_fields(
            value,
            {
                "aggregation_hash",
                "breakdown",
                "factors",
                "orchestrator_run_id",
                "per_run_budget",
                "policy",
                "record_hash",
                "recorded_at",
                "rolling_weekly_budget",
                "rolling_weekly_totals",
                "run_id",
                "schema_version",
                "stages",
                "throttle_recommended",
                "totals",
                "verdict",
            },
            "budget ledger row",
        )
        _assert_no_phi(value)

        try:
            raw_stages = value["stages"]
            breakdown = value["breakdown"]
            if not isinstance(raw_stages, Sequence) or isinstance(raw_stages, str):
                raise BudgetTrackingError("budget ledger stages must be an array")
            if not isinstance(breakdown, Mapping):
                raise BudgetTrackingError("budget ledger breakdown must be an object")
            _reject_unknown_fields(
                breakdown,
                {"family", "family_tier", "tier", "workload"},
                "budget breakdown",
            )
            if any(not isinstance(stage, Mapping) for stage in raw_stages):
                raise BudgetTrackingError("each budget ledger stage must be an object")
            for dimension in ("family", "family_tier", "tier", "workload"):
                if not isinstance(breakdown.get(dimension), Mapping):
                    raise BudgetTrackingError(
                        f"budget breakdown {dimension!r} must be an object"
                    )
            entry = cls(
                run_id=value["run_id"],
                orchestrator_run_id=value["orchestrator_run_id"],
                recorded_at=value["recorded_at"],
                stages=tuple(
                    StageTiming.from_mapping(stage)
                    for stage in raw_stages
                    if isinstance(stage, Mapping)
                ),
                factors=CostCarbonFactors.from_mapping(value["factors"]),
                policy=BudgetPolicy.from_mapping(value["policy"]),
                totals=BudgetTotals.from_mapping(value["totals"]),
                by_family=_totals_mapping(breakdown.get("family", {})),
                by_tier=_totals_mapping(breakdown.get("tier", {})),
                by_family_tier=_totals_mapping(breakdown.get("family_tier", {})),
                by_workload=_totals_mapping(breakdown.get("workload", {})),
                rolling_weekly_totals=BudgetTotals.from_mapping(
                    value["rolling_weekly_totals"]
                ),
                per_run_budget=BudgetDecision.from_mapping(value["per_run_budget"]),
                rolling_weekly_budget=BudgetDecision.from_mapping(
                    value["rolling_weekly_budget"]
                ),
                verdict=str(value["verdict"]),
                throttle_recommended=bool(value["throttle_recommended"]),
                aggregation_hash=str(value["aggregation_hash"]),
                record_hash=str(value["record_hash"]),
                schema_version=str(value.get("schema_version", "")),
            )
        except (KeyError, TypeError) as exc:
            raise BudgetTrackingError("budget ledger row is malformed") from exc
        _verify_entry(entry)
        _assert_no_phi(entry.to_dict())
        return entry


def _serialize_totals_mapping(
    value: Mapping[str, BudgetTotals],
) -> dict[str, dict[str, int | float]]:
    return {label: value[label].to_dict() for label in sorted(value)}


def _aggregation_payload(entry: BudgetLedgerEntry) -> dict[str, Any]:
    return {
        "breakdown": {
            "family": _serialize_totals_mapping(entry.by_family),
            "family_tier": _serialize_totals_mapping(entry.by_family_tier),
            "tier": _serialize_totals_mapping(entry.by_tier),
            "workload": _serialize_totals_mapping(entry.by_workload),
        },
        "factors": entry.factors.to_dict(),
        "stages": [stage.to_dict() for stage in entry.stages],
        "totals": entry.totals.to_dict(),
    }


def _record_payload(entry: BudgetLedgerEntry) -> dict[str, Any]:
    payload = entry.to_dict()
    payload.pop("record_hash", None)
    return payload


def _verify_entry(entry: BudgetLedgerEntry) -> None:
    expected_totals = aggregate_stage_timings(entry.stages, entry.factors)
    if entry.totals != expected_totals:
        raise BudgetTrackingError("budget totals do not match committed stage timings")
    expected_breakdowns = {
        "family": _group_stage_totals(
            entry.stages, entry.factors, lambda item: item.family
        ),
        "tier": _group_stage_totals(
            entry.stages, entry.factors, lambda item: item.tier
        ),
        "family_tier": _group_stage_totals(
            entry.stages,
            entry.factors,
            lambda item: f"{item.family}/{item.tier}",
        ),
        "workload": _group_stage_totals(
            entry.stages, entry.factors, lambda item: item.workload
        ),
    }
    observed_breakdowns = {
        "family": dict(entry.by_family),
        "tier": dict(entry.by_tier),
        "family_tier": dict(entry.by_family_tier),
        "workload": dict(entry.by_workload),
    }
    if observed_breakdowns != expected_breakdowns:
        raise BudgetTrackingError(
            "budget breakdowns do not match committed stage timings"
        )
    expected_per_run = evaluate_budget(
        entry.totals,
        warning=entry.policy.per_run_warn,
        maximum=entry.policy.per_run_over,
    )
    if entry.per_run_budget != expected_per_run:
        raise BudgetTrackingError("per-run budget decision is inconsistent")
    expected_weekly = evaluate_budget(
        entry.rolling_weekly_totals,
        warning=entry.policy.weekly_warn,
        maximum=entry.policy.weekly_over,
    )
    if entry.rolling_weekly_budget != expected_weekly:
        raise BudgetTrackingError("rolling weekly budget decision is inconsistent")
    expected_verdict = _worst_verdict(
        expected_per_run.verdict,
        expected_weekly.verdict,
    )
    if entry.verdict != expected_verdict:
        raise BudgetTrackingError("combined budget verdict is inconsistent")
    if entry.aggregation_hash != _payload_hash(_aggregation_payload(entry)):
        raise BudgetTrackingError("budget aggregation hash verification failed")
    if entry.record_hash != _payload_hash(_record_payload(entry)):
        raise BudgetTrackingError("budget record hash verification failed")


def build_budget_entry(
    *,
    run_id: str,
    stages: Iterable[StageTiming],
    history: Iterable[BudgetLedgerEntry] = (),
    recorded_at: datetime | str | None = None,
    factors: CostCarbonFactors = CostCarbonFactors(),
    policy: BudgetPolicy = BudgetPolicy(),
) -> BudgetLedgerEntry:
    """Build one deterministic run record plus its rolling-week verdict."""

    safe_run_id = _identifier(run_id, "run_id")
    recorded = _timestamp(recorded_at)
    materialized = tuple(sorted(stages, key=_stage_sort_key))
    if not materialized:
        raise BudgetTrackingError("a budget run needs at least one stage timing")
    totals = aggregate_stage_timings(materialized, factors)
    by_family = _group_stage_totals(materialized, factors, lambda item: item.family)
    by_tier = _group_stage_totals(materialized, factors, lambda item: item.tier)
    by_family_tier = _group_stage_totals(
        materialized,
        factors,
        lambda item: f"{item.family}/{item.tier}",
    )
    by_workload = _group_stage_totals(
        materialized,
        factors,
        lambda item: item.workload,
    )
    previous_week = rolling_weekly_totals(
        history,
        as_of=recorded,
        window_days=policy.window_days,
    )
    current_week = BudgetTotals.combine((previous_week, totals))
    per_run_budget = evaluate_budget(
        totals,
        warning=policy.per_run_warn,
        maximum=policy.per_run_over,
    )
    weekly_budget = evaluate_budget(
        current_week,
        warning=policy.weekly_warn,
        maximum=policy.weekly_over,
    )
    verdict = _worst_verdict(per_run_budget.verdict, weekly_budget.verdict)
    placeholder = "sha256:" + ("0" * 64)
    entry = BudgetLedgerEntry(
        run_id=safe_run_id,
        orchestrator_run_id=safe_run_id,
        recorded_at=recorded,
        stages=materialized,
        factors=factors,
        policy=policy,
        totals=totals,
        by_family=by_family,
        by_tier=by_tier,
        by_family_tier=by_family_tier,
        by_workload=by_workload,
        rolling_weekly_totals=current_week,
        per_run_budget=per_run_budget,
        rolling_weekly_budget=weekly_budget,
        verdict=verdict,
        throttle_recommended=verdict == OVER,
        aggregation_hash=placeholder,
        record_hash=placeholder,
    )
    aggregation_hash = _payload_hash(_aggregation_payload(entry))
    entry = BudgetLedgerEntry(
        **{
            **entry.__dict__,
            "aggregation_hash": aggregation_hash,
        }
    )
    record_hash = _payload_hash(_record_payload(entry))
    entry = BudgetLedgerEntry(**{**entry.__dict__, "record_hash": record_hash})
    _assert_no_phi(entry.to_dict())
    return entry


def _worst_verdict(*verdicts: str) -> str:
    order = {WITHIN: 0, WARN: 1, OVER: 2}
    return max(verdicts, key=order.__getitem__)


def load_budget_ledger(
    path: str | Path = DEFAULT_BUDGET_LEDGER,
) -> tuple[BudgetLedgerEntry, ...]:
    """Load and verify every non-empty row in the append-only ledger."""

    ledger = Path(path)
    if not ledger.exists():
        return ()
    entries: list[BudgetLedgerEntry] = []
    with ledger.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BudgetTrackingError(
                    f"budget ledger line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(payload, Mapping):
                raise BudgetTrackingError(
                    f"budget ledger line {line_number} must be an object"
                )
            entry = BudgetLedgerEntry.from_mapping(payload)
            expected_week = BudgetTotals.combine(
                (
                    rolling_weekly_totals(
                        entries,
                        as_of=entry.recorded_at,
                        window_days=entry.policy.window_days,
                    ),
                    entry.totals,
                )
            )
            if entry.rolling_weekly_totals != expected_week:
                raise BudgetTrackingError(
                    "rolling weekly totals do not match committed ledger entries"
                )
            entries.append(entry)
    run_ids = [entry.run_id for entry in entries]
    if len(run_ids) != len(set(run_ids)):
        raise BudgetTrackingError("budget ledger contains duplicate run ids")
    return tuple(entries)


def _orchestrator_run_exists(path: str | Path, run_id: str) -> bool:
    ledger = Path(path)
    if not ledger.is_file():
        return False
    with ledger.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BudgetTrackingError(
                    "orchestrator ledger contains invalid JSON"
                ) from exc
            if (
                isinstance(row, Mapping)
                and row.get("record_type") == "nightly-release"
                and row.get("run_id") == run_id
            ):
                return True
    return False


def append_budget_entry(
    entry: BudgetLedgerEntry,
    *,
    ledger_path: str | Path = DEFAULT_BUDGET_LEDGER,
    orchestrator_ledger_path: str | Path = DEFAULT_ORCHESTRATOR_LEDGER,
    require_orchestrator_link: bool = True,
) -> Path:
    """Append one verified run, refusing duplicates or a missing run link."""

    _verify_entry(entry)
    _assert_no_phi(entry.to_dict())
    existing = load_budget_ledger(ledger_path)
    if any(item.run_id == entry.run_id for item in existing):
        raise BudgetTrackingError("budget ledger already contains this run id")
    expected_week = BudgetTotals.combine(
        (
            rolling_weekly_totals(
                existing,
                as_of=entry.recorded_at,
                window_days=entry.policy.window_days,
            ),
            entry.totals,
        )
    )
    if entry.rolling_weekly_totals != expected_week:
        raise BudgetTrackingError(
            "rolling weekly totals do not match committed ledger entries"
        )
    if require_orchestrator_link and not _orchestrator_run_exists(
        orchestrator_ledger_path, entry.orchestrator_run_id
    ):
        raise BudgetTrackingError("orchestrator run id is absent from release ledger")
    ledger = Path(ledger_path)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(_canonical_json(entry.to_dict()) + "\n")
    return ledger


def budget_entries_in_window(
    entries: Iterable[BudgetLedgerEntry],
    *,
    as_of: datetime | str | None = None,
    window_days: int = 7,
) -> tuple[BudgetLedgerEntry, ...]:
    """Return entries in the inclusive rolling window ending at *as_of*."""

    if window_days <= 0:
        raise BudgetTrackingError("window_days must be positive")
    end = _utc_datetime(as_of)
    start = end - timedelta(days=window_days)
    return tuple(
        entry for entry in entries if start <= _utc_datetime(entry.recorded_at) <= end
    )


def rolling_weekly_totals(
    entries: Iterable[BudgetLedgerEntry],
    *,
    as_of: datetime | str | None = None,
    window_days: int = 7,
) -> BudgetTotals:
    """Sum run totals in a rolling window (seven days by default)."""

    selected = budget_entries_in_window(
        entries,
        as_of=as_of,
        window_days=window_days,
    )
    return BudgetTotals.combine(entry.totals for entry in selected)


def rolling_budget_breakdown(
    entries: Iterable[BudgetLedgerEntry],
    *,
    dimension: str,
    as_of: datetime | str | None = None,
    window_days: int = 7,
) -> dict[str, BudgetTotals]:
    """Sum a family, tier, family/tier, or workload ledger breakdown."""

    attribute = {
        "family": "by_family",
        "tier": "by_tier",
        "family_tier": "by_family_tier",
        "workload": "by_workload",
    }.get(dimension)
    if attribute is None:
        raise BudgetTrackingError("unsupported budget breakdown dimension")
    selected = budget_entries_in_window(
        entries,
        as_of=as_of,
        window_days=window_days,
    )
    grouped: dict[str, list[BudgetTotals]] = {}
    for entry in selected:
        for label, totals in getattr(entry, attribute).items():
            grouped.setdefault(label, []).append(totals)
    return {label: BudgetTotals.combine(grouped[label]) for label in sorted(grouped)}


def _assert_no_phi(value: Any) -> None:
    def walk(key: str, item: Any) -> Iterable[str]:
        if key in {
            "aggregation_hash",
            "orchestrator_run_id",
            "record_hash",
            "recorded_at",
            "run_id",
            "schema_version",
        } or key.endswith("_hash"):
            return
        if isinstance(item, Mapping):
            for nested_key, nested_value in item.items():
                yield from walk(str(nested_key), nested_value)
        elif isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            for nested_value in item:
                yield from walk(key, nested_value)
        elif isinstance(item, str):
            yield item

    for text in walk("", value):
        if any(pattern.search(text) for pattern in _PHI_PATTERNS):
            raise BudgetTrackingError("refusing to persist PHI-shaped budget data")


__all__ = [
    "BENCHMARK_REFRESH",
    "BUDGET_SCHEMA_VERSION",
    "BudgetDecision",
    "BudgetLedgerEntry",
    "BudgetPolicy",
    "BudgetThresholds",
    "BudgetTotals",
    "BudgetTrackingError",
    "CostCarbonFactors",
    "DEFAULT_BUDGET_LEDGER",
    "DEFAULT_ORCHESTRATOR_LEDGER",
    "OVER",
    "STAGE_TIMINGS_SCHEMA_VERSION",
    "StageTiming",
    "StageTimingDocument",
    "TRAINING",
    "WARN",
    "WITHIN",
    "aggregate_stage_timings",
    "append_budget_entry",
    "budget_entries_in_window",
    "build_budget_entry",
    "evaluate_budget",
    "load_budget_ledger",
    "load_stage_timings",
    "rolling_budget_breakdown",
    "rolling_weekly_totals",
    "write_stage_timings",
]
