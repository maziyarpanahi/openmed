"""Deterministic aggregate-only retraining trigger decisions."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from openmed.core.audit import stable_hash
from openmed.eval.drift_monitor import DriftTriggerSignal, assert_no_raw_text
from openmed.eval.error_analysis import (
    RetrainingSliceArtifact,
    RetrainingSlicePriority,
)
from openmed.eval.fleet_metrics import FleetFreshnessMetrics, compute_fleet_freshness
from openmed.eval.history import REGRESSION, BenchmarkHistoryDiff, MetricDelta
from openmed.eval.release_gates import G7_RECALL_DROP_LIMIT, GateCheck, GateReport

RETRAIN_TRIGGER_INPUT_SCHEMA_VERSION = "openmed.retrain_trigger.inputs.v1"
RETRAIN_TRIGGER_EVIDENCE_SCHEMA_VERSION = "openmed.retrain_trigger.evidence.v1"
RETRAIN_QUEUE_SCHEMA_VERSION = "openmed.retrain_queue.v1"

SIGNAL_DRIFT = "drift"
SIGNAL_GATE_FAILURE = "gate-failure"
SIGNAL_STALENESS = "staleness"
SIGNAL_NONE = "none"

_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SIGNAL_ORDER = {
    SIGNAL_NONE: 0,
    SIGNAL_STALENESS: 1,
    SIGNAL_DRIFT: 2,
    SIGNAL_GATE_FAILURE: 3,
}
_INPUT_KEYS = frozenset(
    {
        "dominant_label",
        "drift_ratio",
        "failed_gates",
        "family",
        "gate_failure_ratio",
        "recipe_slices",
        "source_hashes",
        "staleness_ratio",
        "tier",
    }
)
_SLICE_KEYS = frozenset(
    {
        "candidate_count",
        "evidence_hash",
        "fixture_hashes",
        "label",
        "language",
        "priority_score",
        "rank",
    }
)


class RetrainTriggerInputError(ValueError):
    """Raised when aggregate retraining-trigger input is malformed."""


@dataclass(frozen=True)
class RetrainTriggerPolicy:
    """Configurable weights and threshold for aggregate trigger scoring."""

    threshold: float = 1.0
    drift_weight: float = 1.0
    gate_failure_weight: float = 1.0
    staleness_weight: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "threshold",
            "drift_weight",
            "gate_failure_weight",
            "staleness_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise RetrainTriggerInputError(f"{name} must be finite")
            if name == "threshold" and value <= 0.0:
                raise RetrainTriggerInputError("threshold must be positive")
            if name != "threshold" and value < 0.0:
                raise RetrainTriggerInputError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, float]:
        """Return the stable policy payload."""

        return {
            "drift_weight": self.drift_weight,
            "gate_failure_weight": self.gate_failure_weight,
            "staleness_weight": self.staleness_weight,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class RetrainRecipeSlice:
    """Recipe-safe slice priority derived from PHI-free error analysis."""

    rank: int
    label: str
    language: str
    candidate_count: int
    priority_score: float
    evidence_hash: str
    fixture_hashes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_positive_int(self.rank, "recipe_slices.rank")
        _safe_identifier(self.label, "recipe_slices.label")
        _safe_identifier(self.language, "recipe_slices.language")
        _require_non_negative_int(
            self.candidate_count,
            "recipe_slices.candidate_count",
        )
        _non_negative_float(self.priority_score, "recipe_slices.priority_score")
        _safe_hash(self.evidence_hash, "recipe_slices.evidence_hash")
        hashes = tuple(sorted(set(self.fixture_hashes)))
        for value in hashes:
            _safe_hash(value, "recipe_slices.fixture_hashes")
        object.__setattr__(self, "fixture_hashes", hashes)

    @classmethod
    def from_priority(cls, priority: RetrainingSlicePriority) -> "RetrainRecipeSlice":
        """Project one ranked slice onto the retrain recipe boundary."""

        return cls(
            rank=priority.rank,
            label=priority.label,
            language=priority.language,
            candidate_count=priority.candidate_count,
            priority_score=priority.priority_score,
            evidence_hash=stable_hash(priority.to_dict()),
            fixture_hashes=tuple(priority.fixture_hashes),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RetrainRecipeSlice":
        """Build a recipe slice from strict aggregate input."""

        _require_exact_keys(payload, _SLICE_KEYS, "recipe slice")
        fixture_hashes = payload.get("fixture_hashes")
        if not isinstance(fixture_hashes, Sequence) or isinstance(
            fixture_hashes, (str, bytes)
        ):
            raise RetrainTriggerInputError("fixture_hashes must be a list")
        return cls(
            rank=_required_int(payload.get("rank"), "recipe_slices.rank"),
            label=str(payload.get("label", "")),
            language=str(payload.get("language", "")),
            candidate_count=_required_int(
                payload.get("candidate_count"),
                "recipe_slices.candidate_count",
            ),
            priority_score=_required_float(
                payload.get("priority_score"),
                "recipe_slices.priority_score",
            ),
            evidence_hash=str(payload.get("evidence_hash", "")),
            fixture_hashes=tuple(str(value) for value in fixture_hashes),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic recipe-safe payload."""

        return {
            "candidate_count": self.candidate_count,
            "evidence_hash": self.evidence_hash,
            "fixture_hashes": list(self.fixture_hashes),
            "label": self.label,
            "language": self.language,
            "priority_score": round(self.priority_score, 6),
            "rank": self.rank,
        }


@dataclass(frozen=True)
class RetrainCandidateSignals:
    """Normalized aggregate signals for one model family and tier."""

    family: str
    tier: str
    drift_ratio: float = 0.0
    gate_failure_ratio: float = 0.0
    staleness_ratio: float = 0.0
    dominant_label: str | None = None
    failed_gates: tuple[str, ...] = ()
    source_hashes: Mapping[str, str] = field(default_factory=dict)
    recipe_slices: tuple[RetrainRecipeSlice, ...] = ()

    def __post_init__(self) -> None:
        _safe_identifier(self.family, "family")
        _safe_identifier(self.tier, "tier")
        for name in ("drift_ratio", "gate_failure_ratio", "staleness_ratio"):
            value = _non_negative_float(getattr(self, name), name)
            object.__setattr__(self, name, value)
        if self.dominant_label is not None:
            _safe_identifier(self.dominant_label, "dominant_label")
        gates = tuple(sorted(set(self.failed_gates)))
        if any(gate not in {"G3", "G7"} for gate in gates):
            raise RetrainTriggerInputError("failed_gates may contain only G3 and G7")
        object.__setattr__(self, "failed_gates", gates)
        hashes = dict(sorted(self.source_hashes.items()))
        for name, hash_value in hashes.items():
            _safe_identifier(name, "source_hashes key")
            _safe_hash(hash_value, f"source_hashes.{name}")
        object.__setattr__(self, "source_hashes", hashes)
        slices = tuple(sorted(self.recipe_slices, key=lambda item: item.rank))
        if len({item.rank for item in slices}) != len(slices):
            raise RetrainTriggerInputError("recipe slice ranks must be unique")
        if tuple(item.rank for item in slices) != tuple(range(1, len(slices) + 1)):
            raise RetrainTriggerInputError(
                "recipe slice ranks must be contiguous and start at one"
            )
        object.__setattr__(self, "recipe_slices", slices)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RetrainCandidateSignals":
        """Build candidate signals from a strict no-text aggregate mapping."""

        if not isinstance(payload, Mapping):
            raise RetrainTriggerInputError("candidate input must be an object")
        assert_no_raw_text(payload, where="retrain trigger candidate")
        _require_exact_keys(payload, _INPUT_KEYS, "candidate input")
        raw_gates = payload.get("failed_gates")
        raw_hashes = payload.get("source_hashes")
        raw_slices = payload.get("recipe_slices")
        if not isinstance(raw_gates, Sequence) or isinstance(raw_gates, (str, bytes)):
            raise RetrainTriggerInputError("failed_gates must be a list")
        if not isinstance(raw_hashes, Mapping):
            raise RetrainTriggerInputError("source_hashes must be an object")
        if not isinstance(raw_slices, Sequence) or isinstance(raw_slices, (str, bytes)):
            raise RetrainTriggerInputError("recipe_slices must be a list")
        slices: list[RetrainRecipeSlice] = []
        for row in raw_slices:
            if not isinstance(row, Mapping):
                raise RetrainTriggerInputError("recipe_slices entries must be objects")
            slices.append(RetrainRecipeSlice.from_mapping(row))
        return cls(
            family=str(payload.get("family", "")),
            tier=str(payload.get("tier", "")),
            drift_ratio=_required_float(payload.get("drift_ratio"), "drift_ratio"),
            gate_failure_ratio=_required_float(
                payload.get("gate_failure_ratio"),
                "gate_failure_ratio",
            ),
            staleness_ratio=_required_float(
                payload.get("staleness_ratio"),
                "staleness_ratio",
            ),
            dominant_label=(
                str(payload["dominant_label"])
                if payload.get("dominant_label") is not None
                else None
            ),
            failed_gates=tuple(str(value) for value in raw_gates),
            source_hashes={str(key): str(value) for key, value in raw_hashes.items()},
            recipe_slices=tuple(slices),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic aggregate trigger input."""

        payload: dict[str, Any] = {
            "dominant_label": self.dominant_label,
            "drift_ratio": round(self.drift_ratio, 12),
            "failed_gates": list(self.failed_gates),
            "family": self.family,
            "gate_failure_ratio": round(self.gate_failure_ratio, 12),
            "recipe_slices": [item.to_dict() for item in self.recipe_slices],
            "source_hashes": dict(self.source_hashes),
            "staleness_ratio": round(self.staleness_ratio, 12),
            "tier": self.tier,
        }
        assert_no_raw_text(payload, where="retrain trigger candidate")
        return payload


@dataclass(frozen=True)
class RetrainDecision:
    """Auditable trigger decision for one model family and tier."""

    candidate: RetrainCandidateSignals
    score: float
    threshold: float
    dominant_signal: str
    weighted_components: Mapping[str, float]
    queued: bool

    def to_dict(self) -> dict[str, Any]:
        """Return the all-candidate evidence record."""

        payload = {
            **self.candidate.to_dict(),
            "dominant_signal": self.dominant_signal,
            "queued": self.queued,
            "score": round(self.score, 12),
            "threshold": round(self.threshold, 12),
            "weighted_components": {
                name: round(value, 12)
                for name, value in sorted(self.weighted_components.items())
            },
        }
        assert_no_raw_text(payload, where="retrain trigger decision")
        return payload

    def to_queue_dict(self) -> dict[str, Any]:
        """Return the queue record, refusing non-queued decisions."""

        if not self.queued:
            raise RetrainTriggerInputError("only queued decisions may be serialized")
        payload = {
            "schema_version": RETRAIN_QUEUE_SCHEMA_VERSION,
            **self.to_dict(),
        }
        payload["decision_hash"] = stable_hash(payload)
        return payload


@dataclass(frozen=True)
class RetrainTriggerResult:
    """Deterministic retraining queue and all-decision evidence."""

    policy: RetrainTriggerPolicy
    decisions: tuple[RetrainDecision, ...]

    @property
    def queued(self) -> tuple[RetrainDecision, ...]:
        """Return queued decisions in deterministic family/tier order."""

        return tuple(decision for decision in self.decisions if decision.queued)

    def queue_jsonl(self) -> str:
        """Serialize queued families as deterministic JSON Lines."""

        return "".join(
            json.dumps(
                decision.to_queue_dict(),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
            for decision in self.queued
        )

    def to_evidence_dict(self) -> dict[str, Any]:
        """Return evidence for every trigger decision, including non-queued."""

        queue_rows = [decision.to_queue_dict() for decision in self.queued]
        payload: dict[str, Any] = {
            "decisions": [decision.to_dict() for decision in self.decisions],
            "policy": self.policy.to_dict(),
            "queue_count": len(queue_rows),
            "queue_hash": stable_hash(queue_rows),
            "schema_version": RETRAIN_TRIGGER_EVIDENCE_SCHEMA_VERSION,
        }
        payload["artifact_hash"] = stable_hash(payload)
        assert_no_raw_text(payload, where="retrain trigger evidence")
        return payload

    def evidence_json(self, *, indent: int = 2) -> str:
        """Serialize all-decision evidence deterministically."""

        return json.dumps(
            self.to_evidence_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=indent,
            sort_keys=True,
        )


def candidate_signals_from_artifacts(
    family: str,
    tier: str,
    *,
    drift_signal: DriftTriggerSignal | None = None,
    gate_report: GateReport | None = None,
    history_diff: BenchmarkHistoryDiff | None = None,
    fleet_freshness: FleetFreshnessMetrics | None = None,
    retraining_slices: RetrainingSliceArtifact | None = None,
) -> RetrainCandidateSignals:
    """Normalize existing aggregate eval artifacts for trigger scoring."""

    _safe_identifier(family, "family")
    _safe_identifier(tier, "tier")
    source_hashes: dict[str, str] = {}
    drift_ratio = 0.0
    dominant_label: str | None = None
    if drift_signal is not None:
        threshold = _positive_float(drift_signal.threshold, "drift threshold")
        drift_ratio = (
            _non_negative_float(
                drift_signal.max_divergence,
                "drift max_divergence",
            )
            / threshold
        )
        dominant_label = drift_signal.dominant_drifting_label
        source_hashes["drift"] = stable_hash(drift_signal.to_dict())

    gate_ratio = 0.0
    failed_gates: set[str] = set()
    gate_label: str | None = None
    if gate_report is not None:
        _require_report_identity(gate_report, family=family, tier=tier)
        gate_ratio, report_gates, gate_label = _gate_report_signal(gate_report)
        failed_gates.update(report_gates)
        source_hashes["gate"] = stable_hash(gate_report.to_dict())

    if history_diff is not None:
        history_ratio, history_gates = _history_gate_signal(history_diff)
        gate_ratio = max(gate_ratio, history_ratio)
        failed_gates.update(history_gates)
        source_hashes["history"] = stable_hash(history_diff.to_dict())

    staleness_ratio = 0.0
    if fleet_freshness is not None:
        staleness_ratio = _fleet_staleness_ratio(fleet_freshness)
        source_hashes["fleet"] = stable_hash(fleet_freshness.to_dict())

    recipe_slices: tuple[RetrainRecipeSlice, ...] = ()
    if retraining_slices is not None:
        recipe_slices = tuple(
            RetrainRecipeSlice.from_priority(item) for item in retraining_slices.slices
        )
        source_hashes["active_learning"] = retraining_slices.to_dict()["artifact_hash"]

    return RetrainCandidateSignals(
        family=family,
        tier=tier,
        drift_ratio=drift_ratio,
        gate_failure_ratio=gate_ratio,
        staleness_ratio=staleness_ratio,
        dominant_label=gate_label or dominant_label,
        failed_gates=tuple(failed_gates),
        source_hashes=source_hashes,
        recipe_slices=recipe_slices,
    )


def compute_candidate_fleet_freshness(
    rows: Iterable[Mapping[str, Any]],
    *,
    family: str,
    tier: str,
    as_of: date | datetime | str | None = None,
    median_age_target_days: int = 30,
) -> FleetFreshnessMetrics:
    """Compute freshness for rows matching one family and tier."""

    family_key = _safe_identifier(family, "family").casefold()
    tier_key = _safe_identifier(tier, "tier").casefold()
    matching = [
        row
        for row in rows
        if str(row.get("family") or "").casefold() == family_key
        and str(row.get("tier") or "").casefold() == tier_key
    ]
    return compute_fleet_freshness(
        matching,
        as_of=as_of,
        median_age_target_days=median_age_target_days,
    )


def score_retrain_candidates(
    candidates: Iterable[RetrainCandidateSignals],
    *,
    policy: RetrainTriggerPolicy | None = None,
) -> RetrainTriggerResult:
    """Score candidates and return a deterministic queue plus evidence."""

    active_policy = policy or RetrainTriggerPolicy()
    ordered = sorted(candidates, key=lambda item: (item.family, item.tier))
    identities = [(item.family, item.tier) for item in ordered]
    if len(set(identities)) != len(identities):
        raise RetrainTriggerInputError(
            "candidate family/tier identities must be unique"
        )

    decisions: list[RetrainDecision] = []
    for candidate in ordered:
        components = {
            SIGNAL_DRIFT: candidate.drift_ratio * active_policy.drift_weight,
            SIGNAL_GATE_FAILURE: (
                candidate.gate_failure_ratio * active_policy.gate_failure_weight
            ),
            SIGNAL_STALENESS: (
                candidate.staleness_ratio * active_policy.staleness_weight
            ),
        }
        score = math.fsum(components.values())
        dominant_signal = max(
            components,
            key=lambda name: (components[name], _SIGNAL_ORDER[name]),
        )
        if components[dominant_signal] == 0.0:
            dominant_signal = SIGNAL_NONE
        decisions.append(
            RetrainDecision(
                candidate=candidate,
                score=score,
                threshold=active_policy.threshold,
                dominant_signal=dominant_signal,
                weighted_components=components,
                queued=score >= active_policy.threshold,
            )
        )
    return RetrainTriggerResult(policy=active_policy, decisions=tuple(decisions))


def parse_retrain_trigger_inputs(
    payload: Mapping[str, Any],
) -> tuple[RetrainCandidateSignals, ...]:
    """Parse the strict aggregate input document used by offline automation."""

    if not isinstance(payload, Mapping):
        raise RetrainTriggerInputError("retrain trigger input must be an object")
    assert_no_raw_text(payload, where="retrain trigger input")
    _require_exact_keys(
        payload,
        {"schema_version", "candidates"},
        "retrain trigger input",
    )
    if payload.get("schema_version") != RETRAIN_TRIGGER_INPUT_SCHEMA_VERSION:
        raise RetrainTriggerInputError("unsupported retrain trigger input schema")
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, Sequence) or isinstance(
        raw_candidates, (str, bytes)
    ):
        raise RetrainTriggerInputError("candidates must be a list")
    candidates: list[RetrainCandidateSignals] = []
    for row in raw_candidates:
        if not isinstance(row, Mapping):
            raise RetrainTriggerInputError("candidate entries must be objects")
        candidates.append(RetrainCandidateSignals.from_mapping(row))
    return tuple(candidates)


def load_retrain_trigger_inputs(
    path: str | Path,
) -> tuple[RetrainCandidateSignals, ...]:
    """Load aggregate trigger candidates from a local JSON document."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RetrainTriggerInputError("retrain trigger input must be an object")
    return parse_retrain_trigger_inputs(payload)


def write_retrain_trigger_artifacts(
    result: RetrainTriggerResult,
    *,
    queue_path: str | Path,
    evidence_path: str | Path,
) -> tuple[Path, Path]:
    """Write deterministic queue JSONL and all-decision evidence JSON."""

    queue_output = Path(queue_path)
    evidence_output = Path(evidence_path)
    queue_output.parent.mkdir(parents=True, exist_ok=True)
    evidence_output.parent.mkdir(parents=True, exist_ok=True)
    queue_output.write_text(result.queue_jsonl(), encoding="utf-8")
    evidence_output.write_text(result.evidence_json() + "\n", encoding="utf-8")
    return queue_output, evidence_output


def _gate_report_signal(report: GateReport) -> tuple[float, set[str], str | None]:
    ratio = 0.0
    failed: set[str] = set()
    dominant_label: str | None = None
    for check in report.gate_results:
        if check.gate not in {"G3", "G7"} or check.passed:
            continue
        failed.add(check.gate)
        if check.gate == "G3":
            count = _numeric(check.details.get("critical_leakage_count"), default=1.0)
            ratio = max(ratio, 1.0, count)
        else:
            ratio = max(ratio, 1.0, _max_violation_ratio(check.details))
            dominant_label = _dominant_g7_label(check)
    return ratio, failed, dominant_label


def _history_gate_signal(history: BenchmarkHistoryDiff) -> tuple[float, set[str]]:
    ratio = 0.0
    failed: set[str] = set()
    for delta in history.metrics.values():
        if delta.verdict != REGRESSION:
            continue
        metric = delta.metric.casefold()
        if "critical_leakage" in metric and delta.current > 0.0:
            failed.add("G3")
            ratio = max(ratio, 1.0, delta.current)
            continue
        if "recall" not in metric and "leakage" not in metric:
            continue
        regression = _regression_magnitude(delta)
        normalized = regression / G7_RECALL_DROP_LIMIT
        if normalized >= 1.0:
            failed.add("G7")
            ratio = max(ratio, normalized)
    return ratio, failed


def _regression_magnitude(delta: MetricDelta) -> float:
    if delta.direction == "higher_is_better":
        return max(delta.baseline - delta.current, 0.0)
    return max(delta.current - delta.baseline, 0.0)


def _max_violation_ratio(payload: Any) -> float:
    ratios: list[float] = []
    if isinstance(payload, Mapping):
        limit = _numeric(payload.get("limit"), default=0.0)
        observed = _numeric(
            payload.get("drop", payload.get("observed")),
            default=0.0,
        )
        if limit > 0.0 and observed >= 0.0:
            ratios.append(observed / limit)
        for value in payload.values():
            ratios.append(_max_violation_ratio(value))
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        ratios.extend(_max_violation_ratio(value) for value in payload)
    return max(ratios, default=0.0)


def _dominant_g7_label(check: GateCheck) -> str | None:
    violations = check.details.get("violations")
    if not isinstance(violations, Mapping):
        return None
    recall_drop = violations.get("recall_drop")
    if not isinstance(recall_drop, Mapping):
        return None
    ranked: list[tuple[float, str]] = []
    for label, details in recall_drop.items():
        if not isinstance(details, Mapping):
            continue
        ranked.append((_numeric(details.get("drop"), default=0.0), str(label)))
    if not ranked:
        return None
    label = max(ranked, key=lambda item: (item[0], item[1]))[1]
    return _safe_identifier(label, "G7 dominant label")


def _fleet_staleness_ratio(metrics: FleetFreshnessMetrics) -> float:
    target = _positive_float(metrics.median_age_target_days, "fleet target days")
    if metrics.median_age_days is not None:
        return _non_negative_float(metrics.median_age_days, "fleet median age") / target
    if metrics.total_model_count > 0 and metrics.dated_model_count == 0:
        return 1.0
    return 0.0


def _require_report_identity(report: GateReport, *, family: str, tier: str) -> None:
    if report.family and report.family.casefold() != family.casefold():
        raise RetrainTriggerInputError("gate report family does not match candidate")
    if report.tier and report.tier.casefold() != tier.casefold():
        raise RetrainTriggerInputError("gate report tier does not match candidate")


def _safe_identifier(value: Any, name: str) -> str:
    text = str(value)
    if not _IDENTIFIER_RE.fullmatch(text):
        raise RetrainTriggerInputError(f"{name} must be a short safe identifier")
    return text


def _safe_hash(value: Any, name: str) -> str:
    text = str(value)
    if not _HASH_RE.fullmatch(text):
        raise RetrainTriggerInputError(f"{name} must be a sha256 hash")
    return text


def _required_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RetrainTriggerInputError(f"{name} must be numeric")
    return float(value)


def _non_negative_float(value: Any, name: str) -> float:
    number = _required_float(value, name)
    if not math.isfinite(number) or number < 0.0:
        raise RetrainTriggerInputError(f"{name} must be finite and non-negative")
    return number


def _positive_float(value: Any, name: str) -> float:
    number = _required_float(value, name)
    if not math.isfinite(number) or number <= 0.0:
        raise RetrainTriggerInputError(f"{name} must be finite and positive")
    return number


def _required_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RetrainTriggerInputError(f"{name} must be an integer")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    number = _required_int(value, name)
    if number <= 0:
        raise RetrainTriggerInputError(f"{name} must be positive")
    return number


def _require_non_negative_int(value: Any, name: str) -> int:
    number = _required_int(value, name)
    if number < 0:
        raise RetrainTriggerInputError(f"{name} must be non-negative")
    return number


def _numeric(value: Any, *, default: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _require_exact_keys(
    payload: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    name: str,
) -> None:
    keys = set(payload)
    if keys != set(expected):
        missing = sorted(set(expected) - keys)
        unexpected = sorted(keys - set(expected))
        raise RetrainTriggerInputError(
            f"{name} fields do not match schema; missing={missing}, "
            f"unexpected={unexpected}"
        )


__all__ = [
    "RETRAIN_QUEUE_SCHEMA_VERSION",
    "RETRAIN_TRIGGER_EVIDENCE_SCHEMA_VERSION",
    "RETRAIN_TRIGGER_INPUT_SCHEMA_VERSION",
    "SIGNAL_DRIFT",
    "SIGNAL_GATE_FAILURE",
    "SIGNAL_NONE",
    "SIGNAL_STALENESS",
    "RetrainCandidateSignals",
    "RetrainDecision",
    "RetrainRecipeSlice",
    "RetrainTriggerInputError",
    "RetrainTriggerPolicy",
    "RetrainTriggerResult",
    "candidate_signals_from_artifacts",
    "compute_candidate_fleet_freshness",
    "load_retrain_trigger_inputs",
    "parse_retrain_trigger_inputs",
    "score_retrain_candidates",
    "write_retrain_trigger_artifacts",
]
