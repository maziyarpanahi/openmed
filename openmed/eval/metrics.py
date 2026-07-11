"""Benchmark metrics for OpenMed de-identification evaluation."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from math import ceil, isfinite
from typing import Any, Callable, Iterable, Mapping, Sequence

from openmed.core.labels import CANONICAL_LABELS, normalize_label
from openmed.core.pii_i18n import SUPPORTED_LANGUAGES
from openmed.core.quality_gates import detect_overlapping_entities
from openmed.processing.outputs import EntityPrediction

DEVICE_TIERS: tuple[str, ...] = ("cpu", "mlx-fp", "mlx-8bit", "coreml")
ABSTENTION_ROUTE_ACCEPT = "accept"
ABSTENTION_ROUTE_REDACT = "redact"
ABSTENTION_ROUTE_REVIEW = "review"
CRITICAL_ABSTENTION_LABELS = frozenset(
    {
        "SSN",
        "ID_NUM",
        "API_KEY",
        "ACCOUNT_NUMBER",
        "PASSWORD",
        "PIN",
        "CREDIT_CARD",
        "CVV",
        "IBAN",
        "BIC",
    }
)


@dataclass(frozen=True)
class EvalSpan:
    """Normalized span record used by eval metrics."""

    start: int
    end: int
    label: str
    text: str = ""
    language: str = "en"
    device: str = "cpu"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        """Return the non-negative character span length."""
        return max(self.end - self.start, 0)

    def to_entity(self) -> EntityPrediction:
        """Convert to the project's runtime span type for span utilities."""
        return EntityPrediction(
            text=self.text,
            label=self.label,
            confidence=float(self.metadata.get("confidence", 1.0)),
            start=self.start,
            end=self.end,
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True)
class RateMetric:
    """A metric represented as numerator / denominator."""

    rate: float
    numerator: int | float
    denominator: int | float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "rate": self.rate,
            "numerator": self.numerator,
            "denominator": self.denominator,
        }

    def __getitem__(self, key: str) -> int | float:
        return self.to_dict()[key]


@dataclass(frozen=True)
class F1Metrics:
    """Precision/recall/F1 counts for span matching."""

    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int

    def to_dict(self) -> dict[str, int | float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }

    def __getitem__(self, key: str) -> int | float:
        return self.to_dict()[key]


@dataclass(frozen=True)
class RelationArgument:
    """One relation argument represented by source-text offsets."""

    start: int
    end: int
    label: str = ""
    text: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def overlaps(self, other: "RelationArgument") -> bool:
        """Return True when this argument overlaps *other* by at least one char."""
        return min(self.end, other.end) > max(self.start, other.start)

    def overlap_length(self, other: "RelationArgument") -> int:
        """Return the character overlap length with *other*."""
        return max(min(self.end, other.end) - max(self.start, other.start), 0)

    def shifted(self, offset: int) -> "RelationArgument":
        """Return the argument shifted by *offset* source characters."""
        if offset == 0:
            return self
        return RelationArgument(
            start=self.start + offset,
            end=self.end + offset,
            label=self.label,
            text=self.text,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "end": self.end,
            "label": self.label,
            "metadata": dict(self.metadata),
            "start": self.start,
            "text": self.text,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


@dataclass(frozen=True)
class RelationTriple:
    """Typed relation triple used by relation-extraction metrics."""

    relation_type: str
    arg1: RelationArgument
    arg2: RelationArgument
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def shifted(self, offset: int) -> "RelationTriple":
        """Return this triple shifted by *offset* source characters."""
        if offset == 0:
            return self
        return RelationTriple(
            relation_type=self.relation_type,
            arg1=self.arg1.shifted(offset),
            arg2=self.arg2.shifted(offset),
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "arg1": self.arg1.to_dict(),
            "arg2": self.arg2.to_dict(),
            "metadata": dict(self.metadata),
            "relation_type": self.relation_type,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


@dataclass(frozen=True)
class LeakageMetrics:
    """Character-weighted PHI leakage rate and required slices."""

    overall: float
    by_label: dict[str, float]
    by_language: dict[str, float]
    by_device: dict[str, float]
    leaked_chars: int
    total_chars: int
    leaked_chars_by_label: dict[str, int]
    total_chars_by_label: dict[str, int]
    leaked_chars_by_language: dict[str, int]
    total_chars_by_language: dict[str, int]
    leaked_chars_by_device: dict[str, int]
    total_chars_by_device: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall,
            "by_label": self.by_label,
            "by_language": self.by_language,
            "by_device": self.by_device,
            "leaked_chars": self.leaked_chars,
            "total_chars": self.total_chars,
            "leaked_chars_by_label": self.leaked_chars_by_label,
            "total_chars_by_label": self.total_chars_by_label,
            "leaked_chars_by_language": self.leaked_chars_by_language,
            "total_chars_by_language": self.total_chars_by_language,
            "leaked_chars_by_device": self.leaked_chars_by_device,
            "total_chars_by_device": self.total_chars_by_device,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


@dataclass(frozen=True)
class AbstentionDecision:
    """One span-level abstention decision without retaining raw span text."""

    label: str
    language: str
    confidence: float
    threshold: float
    nonconformity: float
    accepted: bool
    route: str
    residual_error: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "language": self.language,
            "confidence": self.confidence,
            "threshold": self.threshold,
            "nonconformity": self.nonconformity,
            "accepted": self.accepted,
            "route": self.route,
            "residual_error": self.residual_error,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


@dataclass(frozen=True)
class AbstentionMetrics:
    """Abstention-rate and residual-risk slices for span decisions."""

    abstention_rate: float
    abstained: int
    total: int
    abstention_rate_by_label: dict[str, float]
    abstention_rate_by_language: dict[str, float]
    residual_risk: float
    critical_residual_risk: float
    residual_risk_by_label: dict[str, float]
    residual_risk_by_language: dict[str, float]
    residual_errors: int
    accepted: int
    critical_residual_errors: int
    accepted_critical: int
    route_counts: dict[str, int]
    bootstrap: Mapping[str, Any] = field(default_factory=dict)
    target_risk: float | None = None
    confidence_level: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "abstention_rate": {
                "overall": self.abstention_rate,
                "by_label": self.abstention_rate_by_label,
                "by_language": self.abstention_rate_by_language,
                "abstained": self.abstained,
                "total": self.total,
            },
            "residual_risk": {
                "overall": self.residual_risk,
                "critical": self.critical_residual_risk,
                "by_label": self.residual_risk_by_label,
                "by_language": self.residual_risk_by_language,
                "residual_errors": self.residual_errors,
                "accepted": self.accepted,
                "critical_residual_errors": self.critical_residual_errors,
                "accepted_critical": self.accepted_critical,
            },
            "route_counts": self.route_counts,
        }
        if self.bootstrap:
            payload["residual_risk"]["bootstrap"] = dict(self.bootstrap)
        if self.target_risk is not None:
            payload["target_risk"] = self.target_risk
        if self.confidence_level is not None:
            payload["confidence_level"] = self.confidence_level
        return payload

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


@dataclass(frozen=True)
class RecallSlices:
    """Character recall sliced by label, language, and device."""

    overall: float
    by_label: dict[str, float]
    by_language: dict[str, float]
    by_device: dict[str, float]
    covered_chars: int
    total_chars: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall,
            "by_label": self.by_label,
            "by_language": self.by_language,
            "by_device": self.by_device,
            "covered_chars": self.covered_chars,
            "total_chars": self.total_chars,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


@dataclass(frozen=True)
class ConsistencyMetric:
    """Consistency score with concrete violation counts."""

    score: float
    consistent: int
    total: int
    violations: dict[str, list[str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "consistent": self.consistent,
            "total": self.total,
            "violations": self.violations,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


@dataclass(frozen=True)
class LatencyMetrics:
    """Latency distribution in milliseconds."""

    p50_ms: float
    p95_ms: float
    p99_ms: float
    count: int

    def to_dict(self) -> dict[str, int | float]:
        return {
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "count": self.count,
        }

    def __getitem__(self, key: str) -> int | float:
        return self.to_dict()[key]


@dataclass(frozen=True)
class ResourceMetrics:
    """Resource summary for a benchmark run."""

    peak_rss_bytes: int | None = None
    model_size_bytes: int | None = None

    @property
    def peak_rss_mib(self) -> float | None:
        if self.peak_rss_bytes is None:
            return None
        return self.peak_rss_bytes / (1024 * 1024)

    @property
    def model_size_mib(self) -> float | None:
        if self.model_size_bytes is None:
            return None
        return self.model_size_bytes / (1024 * 1024)

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_rss_mib": self.peak_rss_mib,
            "model_size_bytes": self.model_size_bytes,
            "model_size_mib": self.model_size_mib,
        }

    def __getitem__(self, key: str) -> float | int | None:
        return self.to_dict()[key]


@dataclass(frozen=True)
class BootstrapCI:
    """A bootstrap confidence interval around a point estimate.

    ``degenerate`` is set when the interval could not vary under resampling
    (an empty or single-document corpus), in which case ``lower``/``upper``
    collapse onto ``point`` -- a zero-width, explicitly flagged interval.
    """

    point: float
    lower: float
    upper: float
    n_resamples: int
    alpha: float
    degenerate: bool = False

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "point": self.point,
            "lower": self.lower,
            "upper": self.upper,
            "n_resamples": self.n_resamples,
            "alpha": self.alpha,
            "degenerate": self.degenerate,
        }

    def __getitem__(self, key: str) -> float | int | bool:
        return self.to_dict()[key]


@dataclass(frozen=True)
class ReliabilityBin:
    """One serializable bin for a reliability diagram."""

    bin_index: int
    lower_bound: float
    upper_bound: float
    mean_confidence: float
    empirical_accuracy: float
    count: int

    def to_dict(self) -> dict[str, int | float]:
        return {
            "bin_index": self.bin_index,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "mean_confidence": self.mean_confidence,
            "accuracy": self.empirical_accuracy,
            "empirical_accuracy": self.empirical_accuracy,
            "count": self.count,
        }

    def __getitem__(self, key: str) -> int | float:
        return self.to_dict()[key]


@dataclass(frozen=True)
class PairedSignificance:
    """Observed paired delta and permutation-test p-value."""

    observed_delta: float
    p_value: float
    n_resamples: int
    method: str = "paired_permutation"

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "observed_delta": self.observed_delta,
            "p_value": self.p_value,
            "n_resamples": self.n_resamples,
            "method": self.method,
        }

    def __getitem__(self, key: str) -> float | int | str:
        return self.to_dict()[key]


@dataclass(frozen=True)
class CoverageGap:
    """Observed coverage against a target for one reporting slice."""

    slice_key: str
    target_coverage: float
    realized_coverage: float
    coverage_gap: float
    covered_weight: float
    total_weight: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "slice_key": self.slice_key,
            "target_coverage": self.target_coverage,
            "realized_coverage": self.realized_coverage,
            "coverage_gap": self.coverage_gap,
            "covered_weight": self.covered_weight,
            "total_weight": self.total_weight,
        }

    def __getitem__(self, key: str) -> float | str:
        return self.to_dict()[key]


def normalize_eval_span(
    span: Any,
    *,
    default_language: str = "en",
    default_device: str = "cpu",
    source_text: str | None = None,
) -> EvalSpan:
    """Normalize dicts, runtime entities, and span-like objects for scoring."""
    if isinstance(span, EvalSpan):
        return span

    data = span if isinstance(span, Mapping) else vars(span)
    metadata = _read_mapping(data, "metadata") or {}
    if not isinstance(metadata, Mapping):
        metadata = {"value": metadata}
    metadata = dict(metadata)
    for confidence_key in ("confidence", "score", "probability"):
        confidence = _read_value(data, confidence_key)
        if confidence is not None:
            metadata.setdefault("confidence", confidence)
            break
    raw_group = _read_value(data, "group")
    if raw_group is not None and str(raw_group).strip():
        metadata["group"] = str(raw_group).strip()

    start = _read_int(data, "start")
    end = _read_int(data, "end")
    if start is None or end is None:
        raise ValueError(f"span must include integer start/end offsets: {span!r}")

    raw_label = (
        _read_value(data, "canonical_label")
        or _read_value(data, "label")
        or _read_value(data, "entity_type")
        or _read_value(data, "entity_group")
        or _read_value(data, "entity")
        or "OTHER"
    )
    raw_language = (
        _read_value(data, "language")
        or _read_value(data, "lang")
        or metadata.get("language")
        or metadata.get("lang")
        or metadata.get("sentence_language")
        or default_language
    )
    raw_device = (
        _read_value(data, "device")
        or metadata.get("device")
        or metadata.get("device_tier")
        or metadata.get("hardware_backend")
        or default_device
    )
    text = _read_value(data, "text")
    if (
        text is None
        and source_text is not None
        and 0 <= start <= end <= len(source_text)
    ):
        text = source_text[start:end]

    language = str(raw_language)
    label = normalize_label(str(raw_label), lang=language)
    return EvalSpan(
        start=start,
        end=end,
        label=label,
        text=str(text or ""),
        language=language,
        device=str(raw_device),
        metadata=metadata,
    )


def normalize_eval_spans(
    spans: Iterable[Any],
    *,
    default_language: str = "en",
    default_device: str = "cpu",
    source_text: str | None = None,
) -> list[EvalSpan]:
    """Normalize a sequence of span-like records."""
    return [
        normalize_eval_span(
            span,
            default_language=default_language,
            default_device=default_device,
            source_text=source_text,
        )
        for span in spans
    ]


def compute_leakage_rate(
    gold_spans: Iterable[Any],
    predicted_spans: Iterable[Any],
    *,
    default_language: str = "en",
    default_device: str = "cpu",
    source_text: str | None = None,
) -> LeakageMetrics:
    """Compute first-class character-weighted PHI leakage.

    Leakage is the number of gold PHI characters not covered by any
    same-label prediction divided by the total number of gold PHI characters.
    The same numerator/denominator accounting is reported overall and by
    label, language, and device.
    """
    gold = normalize_eval_spans(
        gold_spans,
        default_language=default_language,
        default_device=default_device,
        source_text=source_text,
    )
    predicted = normalize_eval_spans(
        predicted_spans,
        default_language=default_language,
        default_device=default_device,
        source_text=source_text,
    )

    leaked_by_label: defaultdict[str, int] = defaultdict(int)
    total_by_label: defaultdict[str, int] = defaultdict(int)
    leaked_by_language: defaultdict[str, int] = defaultdict(int)
    total_by_language: defaultdict[str, int] = defaultdict(int)
    leaked_by_device: defaultdict[str, int] = defaultdict(int)
    total_by_device: defaultdict[str, int] = defaultdict(int)

    total_chars = 0
    leaked_chars = 0
    for span in gold:
        covered = _covered_char_count(span, predicted)
        leaked = max(span.length - covered, 0)
        total_chars += span.length
        leaked_chars += leaked
        total_by_label[span.label] += span.length
        leaked_by_label[span.label] += leaked
        total_by_language[span.language] += span.length
        leaked_by_language[span.language] += leaked
        total_by_device[span.device] += span.length
        leaked_by_device[span.device] += leaked

    label_keys = _slice_keys(CANONICAL_LABELS, total_by_label, leaked_by_label)
    language_keys = _slice_keys(
        SUPPORTED_LANGUAGES, total_by_language, leaked_by_language
    )
    device_keys = _slice_keys(DEVICE_TIERS, total_by_device, leaked_by_device)

    return LeakageMetrics(
        overall=_safe_rate(leaked_chars, total_chars, zero_denominator=0.0),
        by_label=_rate_map(label_keys, leaked_by_label, total_by_label, 0.0),
        by_language=_rate_map(
            language_keys, leaked_by_language, total_by_language, 0.0
        ),
        by_device=_rate_map(device_keys, leaked_by_device, total_by_device, 0.0),
        leaked_chars=leaked_chars,
        total_chars=total_chars,
        leaked_chars_by_label=_count_map(label_keys, leaked_by_label),
        total_chars_by_label=_count_map(label_keys, total_by_label),
        leaked_chars_by_language=_count_map(language_keys, leaked_by_language),
        total_chars_by_language=_count_map(language_keys, total_by_language),
        leaked_chars_by_device=_count_map(device_keys, leaked_by_device),
        total_chars_by_device=_count_map(device_keys, total_by_device),
    )


def compute_character_recall(
    gold_spans: Iterable[Any],
    predicted_spans: Iterable[Any],
    *,
    default_language: str = "en",
    default_device: str = "cpu",
    source_text: str | None = None,
) -> RateMetric:
    """Compute label-aware character recall over gold PHI spans."""
    gold = normalize_eval_spans(
        gold_spans,
        default_language=default_language,
        default_device=default_device,
        source_text=source_text,
    )
    predicted = normalize_eval_spans(
        predicted_spans,
        default_language=default_language,
        default_device=default_device,
        source_text=source_text,
    )
    total_chars = sum(span.length for span in gold)
    covered_chars = sum(_covered_char_count(span, predicted) for span in gold)
    return RateMetric(
        rate=_safe_rate(covered_chars, total_chars, zero_denominator=1.0),
        numerator=covered_chars,
        denominator=total_chars,
    )


def compute_recall_slices(
    gold_spans: Iterable[Any],
    predicted_spans: Iterable[Any],
    *,
    default_language: str = "en",
    default_device: str = "cpu",
    source_text: str | None = None,
) -> RecallSlices:
    """Compute character recall sliced by canonical labels, languages, devices."""
    gold = normalize_eval_spans(
        gold_spans,
        default_language=default_language,
        default_device=default_device,
        source_text=source_text,
    )
    predicted = normalize_eval_spans(
        predicted_spans,
        default_language=default_language,
        default_device=default_device,
        source_text=source_text,
    )

    covered_by_label: defaultdict[str, int] = defaultdict(int)
    total_by_label: defaultdict[str, int] = defaultdict(int)
    covered_by_language: defaultdict[str, int] = defaultdict(int)
    total_by_language: defaultdict[str, int] = defaultdict(int)
    covered_by_device: defaultdict[str, int] = defaultdict(int)
    total_by_device: defaultdict[str, int] = defaultdict(int)

    total_chars = 0
    covered_chars = 0
    for span in gold:
        covered = _covered_char_count(span, predicted)
        total_chars += span.length
        covered_chars += covered
        total_by_label[span.label] += span.length
        covered_by_label[span.label] += covered
        total_by_language[span.language] += span.length
        covered_by_language[span.language] += covered
        total_by_device[span.device] += span.length
        covered_by_device[span.device] += covered

    label_keys = _slice_keys(CANONICAL_LABELS, total_by_label, covered_by_label)
    language_keys = _slice_keys(
        SUPPORTED_LANGUAGES, total_by_language, covered_by_language
    )
    device_keys = _slice_keys(DEVICE_TIERS, total_by_device, covered_by_device)

    return RecallSlices(
        overall=_safe_rate(covered_chars, total_chars, zero_denominator=1.0),
        by_label=_rate_map(label_keys, covered_by_label, total_by_label, 1.0),
        by_language=_rate_map(
            language_keys, covered_by_language, total_by_language, 1.0
        ),
        by_device=_rate_map(device_keys, covered_by_device, total_by_device, 1.0),
        covered_chars=covered_chars,
        total_chars=total_chars,
    )


def compute_exact_span_f1(
    gold_spans: Iterable[Any],
    predicted_spans: Iterable[Any],
    *,
    default_language: str = "en",
    default_device: str = "cpu",
    source_text: str | None = None,
) -> F1Metrics:
    """Compute strict label-aware exact span F1."""
    gold = normalize_eval_spans(
        gold_spans,
        default_language=default_language,
        default_device=default_device,
        source_text=source_text,
    )
    predicted = normalize_eval_spans(
        predicted_spans,
        default_language=default_language,
        default_device=default_device,
        source_text=source_text,
    )
    matched_predictions: set[int] = set()
    true_positives = 0
    for gold_span in gold:
        for index, pred_span in enumerate(predicted):
            if index in matched_predictions:
                continue
            if (
                gold_span.label == pred_span.label
                and gold_span.start == pred_span.start
                and gold_span.end == pred_span.end
            ):
                matched_predictions.add(index)
                true_positives += 1
                break
    return _f1_from_counts(true_positives, len(predicted), len(gold))


def compute_relaxed_span_f1(
    gold_spans: Iterable[Any],
    predicted_spans: Iterable[Any],
    *,
    default_language: str = "en",
    default_device: str = "cpu",
    source_text: str | None = None,
) -> F1Metrics:
    """Compute label-aware relaxed F1 where any character overlap matches."""
    gold = normalize_eval_spans(
        gold_spans,
        default_language=default_language,
        default_device=default_device,
        source_text=source_text,
    )
    predicted = normalize_eval_spans(
        predicted_spans,
        default_language=default_language,
        default_device=default_device,
        source_text=source_text,
    )
    matched_predictions: set[int] = set()
    true_positives = 0
    for gold_span in gold:
        candidates = [
            (index, pred_span)
            for index, pred_span in enumerate(predicted)
            if index not in matched_predictions
            and _label_aware_overlap(gold_span, pred_span)
        ]
        if not candidates:
            continue
        best_index, _ = max(
            candidates,
            key=lambda item: _overlap_len(gold_span, item[1]),
        )
        matched_predictions.add(best_index)
        true_positives += 1
    return _f1_from_counts(true_positives, len(predicted), len(gold))


def normalize_relation_triple(relation: Any) -> RelationTriple:
    """Normalize a DrugProt-style or mapping relation into a typed triple."""
    if isinstance(relation, RelationTriple):
        return relation

    if isinstance(relation, Mapping):
        return _relation_from_mapping(relation)

    if isinstance(relation, Sequence) and not isinstance(relation, str | bytes):
        return _relation_from_sequence(relation)

    relation_type = _read_attr(relation, "relation_type", "type", "label")
    if relation_type is None:
        raise ValueError(f"relation is missing relation_type: {relation!r}")
    arg1 = _read_attr(relation, "arg1", "head", "source", "subject")
    arg2 = _read_attr(relation, "arg2", "tail", "target", "object", "attribute")
    if arg1 is None or arg2 is None:
        raise ValueError(f"relation is missing arg1/arg2 spans: {relation!r}")
    metadata = _read_attr(relation, "metadata") or {}
    if not isinstance(metadata, Mapping):
        metadata = {"value": metadata}
    return RelationTriple(
        relation_type=str(relation_type).strip(),
        arg1=_normalize_relation_argument(arg1),
        arg2=_normalize_relation_argument(arg2),
        metadata=dict(metadata),
    )


def normalize_relation_triples(relations: Iterable[Any]) -> tuple[RelationTriple, ...]:
    """Normalize relation records into immutable typed triples."""
    return tuple(normalize_relation_triple(relation) for relation in relations)


def compute_strict_relation_f1(
    gold_relations: Iterable[Any],
    predicted_relations: Iterable[Any],
) -> F1Metrics:
    """Compute exact typed relation F1 over relation triples.

    A strict true positive requires an exact match on relation type plus both
    argument spans.
    """
    gold = normalize_relation_triples(gold_relations)
    predicted = normalize_relation_triples(predicted_relations)
    return _relation_f1(gold, predicted, relaxed=False)


def compute_relaxed_relation_f1(
    gold_relations: Iterable[Any],
    predicted_relations: Iterable[Any],
) -> F1Metrics:
    """Compute typed relation F1 with any-overlap argument span matching."""
    gold = normalize_relation_triples(gold_relations)
    predicted = normalize_relation_triples(predicted_relations)
    return _relation_f1(gold, predicted, relaxed=True)


def compute_relation_metrics_bundle(
    gold_relations: Iterable[Any],
    predicted_relations: Iterable[Any],
) -> dict[str, Any]:
    """Compute strict, relaxed, and per-type relation-extraction metrics."""
    gold = normalize_relation_triples(gold_relations)
    predicted = normalize_relation_triples(predicted_relations)
    relation_types = sorted(
        {relation.relation_type for relation in (*gold, *predicted)}
    )
    per_type: dict[str, Any] = {}
    for relation_type in relation_types:
        gold_for_type = tuple(
            relation for relation in gold if relation.relation_type == relation_type
        )
        predicted_for_type = tuple(
            relation
            for relation in predicted
            if relation.relation_type == relation_type
        )
        per_type[relation_type] = {
            "strict": _relation_f1(
                gold_for_type,
                predicted_for_type,
                relaxed=False,
            ).to_dict(),
            "relaxed": _relation_f1(
                gold_for_type,
                predicted_for_type,
                relaxed=True,
            ).to_dict(),
        }

    return {
        "gold_relation_count": len(gold),
        "per_relation_type": per_type,
        "predicted_relation_count": len(predicted),
        "relaxed": _relation_f1(gold, predicted, relaxed=True).to_dict(),
        "strict": _relation_f1(gold, predicted, relaxed=False).to_dict(),
    }


def compute_relation_confidence_intervals(
    per_document_relations: Sequence[tuple[Iterable[Any], Iterable[Any]]],
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Bootstrap document-level confidence intervals for strict/relaxed RE-F1."""
    strict_docs: list[tuple[int, int, int]] = []
    relaxed_docs: list[tuple[int, int, int]] = []
    for gold_relations, predicted_relations in per_document_relations:
        strict = compute_strict_relation_f1(gold_relations, predicted_relations)
        relaxed = compute_relaxed_relation_f1(gold_relations, predicted_relations)
        strict_docs.append(
            (strict.true_positives, strict.false_positives, strict.false_negatives)
        )
        relaxed_docs.append(
            (relaxed.true_positives, relaxed.false_positives, relaxed.false_negatives)
        )

    return {
        "relaxed": bootstrap_ci(
            relaxed_docs,
            _f1_over_docs,
            n_resamples=n_resamples,
            alpha=alpha,
            seed=seed,
        ).to_dict(),
        "strict": bootstrap_ci(
            strict_docs,
            _f1_over_docs,
            n_resamples=n_resamples,
            alpha=alpha,
            seed=seed,
        ).to_dict(),
    }


def compute_over_redaction_loss(
    gold_spans: Iterable[Any],
    predicted_spans: Iterable[Any],
    *,
    text_length: int | None = None,
    default_language: str = "en",
    default_device: str = "cpu",
    source_text: str | None = None,
) -> RateMetric:
    """Compute non-PHI character coverage by predicted redaction spans."""
    gold = normalize_eval_spans(
        gold_spans,
        default_language=default_language,
        default_device=default_device,
        source_text=source_text,
    )
    predicted = normalize_eval_spans(
        predicted_spans,
        default_language=default_language,
        default_device=default_device,
        source_text=source_text,
    )
    pred_positions = _positions_for_spans(predicted)
    gold_positions = _positions_for_spans(gold)
    over_redacted = len(pred_positions - gold_positions)
    if text_length is None:
        denominator = max(len(pred_positions), 0)
    else:
        denominator = max(text_length - len(gold_positions), 0)
    return RateMetric(
        rate=_safe_rate(over_redacted, denominator, zero_denominator=0.0),
        numerator=over_redacted,
        denominator=denominator,
    )


def compute_clinical_utility_loss(*args: Any, **kwargs: Any) -> RateMetric:
    """Alias for over-redaction loss used in benchmark reports."""
    return compute_over_redaction_loss(*args, **kwargs)


def abstention_route(
    label: str,
    *,
    accepted: bool,
    language: str = "en",
    critical_labels: frozenset[str] = CRITICAL_ABSTENTION_LABELS,
) -> str:
    """Route accepted spans, critical abstentions, and review abstentions."""

    if accepted:
        return ABSTENTION_ROUTE_ACCEPT
    canonical = normalize_label(label, language)
    if canonical in critical_labels:
        return ABSTENTION_ROUTE_REDACT
    return ABSTENTION_ROUTE_REVIEW


def apply_abstention_policy(
    gold_spans: Iterable[Any],
    predicted_spans: Iterable[Any],
    *,
    thresholds: Any | None = None,
    confidence_threshold: float = 0.0,
    model_id: str | None = None,
    default_language: str = "en",
    default_device: str = "cpu",
    source_text: str | None = None,
    critical_labels: frozenset[str] = CRITICAL_ABSTENTION_LABELS,
) -> list[AbstentionDecision]:
    """Apply span abstention routing and return deterministic decisions."""

    gold = normalize_eval_spans(
        gold_spans,
        default_language=default_language,
        default_device=default_device,
        source_text=source_text,
    )
    predicted = normalize_eval_spans(
        predicted_spans,
        default_language=default_language,
        default_device=default_device,
        source_text=source_text,
    )
    matched_gold: set[int] = set()
    decisions: list[AbstentionDecision] = []
    for span in predicted:
        threshold = _threshold_for_span(
            span,
            thresholds,
            default_threshold=confidence_threshold,
            model_id=model_id,
        )
        confidence = _confidence_for_span(span)
        accepted = confidence >= threshold
        exact_match = False
        for index, gold_span in enumerate(gold):
            if index in matched_gold:
                continue
            if _exact_span_match(gold_span, span):
                matched_gold.add(index)
                exact_match = True
                break
        route = abstention_route(
            span.label,
            accepted=accepted,
            language=span.language,
            critical_labels=critical_labels,
        )
        decisions.append(
            AbstentionDecision(
                label=span.label,
                language=span.language,
                confidence=confidence,
                threshold=threshold,
                nonconformity=1.0 - confidence,
                accepted=accepted,
                route=route,
                residual_error=not exact_match,
            )
        )
    return decisions


def compute_abstention_metrics(
    gold_spans: Iterable[Any],
    predicted_spans: Iterable[Any],
    *,
    thresholds: Any | None = None,
    confidence_threshold: float = 0.0,
    model_id: str | None = None,
    target_risk: float | None = None,
    confidence_level: float | None = None,
    bootstrap_resamples: int = 0,
    seed: int = 0,
    default_language: str = "en",
    default_device: str = "cpu",
    source_text: str | None = None,
    critical_labels: frozenset[str] = CRITICAL_ABSTENTION_LABELS,
) -> AbstentionMetrics:
    """Compute abstention-rate and residual-risk metrics per label/language."""

    decisions = apply_abstention_policy(
        gold_spans,
        predicted_spans,
        thresholds=thresholds,
        confidence_threshold=confidence_threshold,
        model_id=model_id,
        default_language=default_language,
        default_device=default_device,
        source_text=source_text,
        critical_labels=critical_labels,
    )
    total = len(decisions)
    abstained = sum(1 for decision in decisions if not decision.accepted)
    accepted = sum(1 for decision in decisions if decision.accepted)
    residual_errors = sum(
        1 for decision in decisions if decision.accepted and decision.residual_error
    )
    accepted_critical = sum(
        1
        for decision in decisions
        if decision.accepted and decision.label in critical_labels
    )
    critical_residual_errors = sum(
        1
        for decision in decisions
        if decision.accepted
        and decision.label in critical_labels
        and decision.residual_error
    )
    route_counts = {
        route: sum(1 for decision in decisions if decision.route == route)
        for route in (
            ABSTENTION_ROUTE_ACCEPT,
            ABSTENTION_ROUTE_REDACT,
            ABSTENTION_ROUTE_REVIEW,
        )
    }
    label_keys = sorted({decision.label for decision in decisions})
    language_keys = sorted({decision.language for decision in decisions})
    bootstrap = (
        bootstrap_abstention_residual_risk(
            decisions,
            n_resamples=bootstrap_resamples,
            seed=seed,
            critical_only=True,
            critical_labels=critical_labels,
        )
        if bootstrap_resamples
        else {}
    )
    return AbstentionMetrics(
        abstention_rate=_safe_rate(abstained, total, zero_denominator=0.0),
        abstained=abstained,
        total=total,
        abstention_rate_by_label=_abstention_rate_by(
            decisions,
            label_keys,
            group_key="label",
        ),
        abstention_rate_by_language=_abstention_rate_by(
            decisions,
            language_keys,
            group_key="language",
        ),
        residual_risk=_safe_rate(residual_errors, accepted, zero_denominator=0.0),
        critical_residual_risk=_safe_rate(
            critical_residual_errors,
            accepted_critical,
            zero_denominator=0.0,
        ),
        residual_risk_by_label=_residual_risk_by(
            decisions,
            label_keys,
            group_key="label",
        ),
        residual_risk_by_language=_residual_risk_by(
            decisions,
            language_keys,
            group_key="language",
        ),
        residual_errors=residual_errors,
        accepted=accepted,
        critical_residual_errors=critical_residual_errors,
        accepted_critical=accepted_critical,
        route_counts=route_counts,
        bootstrap=bootstrap,
        target_risk=target_risk,
        confidence_level=confidence_level,
    )


def bootstrap_abstention_residual_risk(
    decisions: Sequence[AbstentionDecision],
    *,
    n_resamples: int = 100,
    seed: int = 0,
    critical_only: bool = True,
    critical_labels: frozenset[str] = CRITICAL_ABSTENTION_LABELS,
) -> dict[str, Any]:
    """Bootstrap realized residual risk for accepted abstention decisions."""

    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")
    values = list(decisions)
    if critical_only:
        values = [decision for decision in values if decision.label in critical_labels]
    point = _decision_residual_risk(values)
    if len(values) < 2:
        return {
            "point": point,
            "max": point,
            "n_resamples": n_resamples,
            "seed": seed,
            "critical_only": critical_only,
            "degenerate": True,
        }

    rng = random.Random(seed)
    size = len(values)
    samples: list[float] = []
    for _ in range(n_resamples):
        resample = [values[rng.randrange(size)] for _ in range(size)]
        samples.append(_decision_residual_risk(resample))
    return {
        "point": point,
        "max": max(samples, default=point),
        "min": min(samples, default=point),
        "mean": sum(samples) / len(samples),
        "n_resamples": n_resamples,
        "seed": seed,
        "critical_only": critical_only,
        "degenerate": False,
    }


def compute_date_shift_consistency(
    original_dates: Sequence[str],
    shifted_dates: Sequence[str],
) -> ConsistencyMetric:
    """Score whether date replacements use one consistent day offset."""
    if len(original_dates) != len(shifted_dates):
        raise ValueError("original_dates and shifted_dates must have the same length")
    parsed: list[tuple[str, str, int]] = []
    violations: dict[str, list[str]] = {}
    for original, shifted in zip(original_dates, shifted_dates):
        original_date = _parse_date(original)
        shifted_date = _parse_date(shifted)
        if original_date is None or shifted_date is None:
            violations.setdefault(original, []).append(shifted)
            continue
        parsed.append((original, shifted, (shifted_date - original_date).days))

    if not original_dates:
        return ConsistencyMetric(score=1.0, consistent=0, total=0, violations={})
    if not parsed:
        return ConsistencyMetric(
            score=0.0,
            consistent=0,
            total=len(original_dates),
            violations=violations,
        )

    expected = parsed[0][2]
    consistent = 0
    for original, shifted, delta in parsed:
        if delta == expected:
            consistent += 1
        else:
            violations.setdefault(original, []).append(shifted)
    return ConsistencyMetric(
        score=_safe_rate(consistent, len(original_dates), zero_denominator=1.0),
        consistent=consistent,
        total=len(original_dates),
        violations=violations,
    )


def compute_surrogate_consistency(
    originals: Sequence[str],
    surrogates: Sequence[str],
) -> ConsistencyMetric:
    """Score whether repeated source values map to stable surrogates."""
    if len(originals) != len(surrogates):
        raise ValueError("originals and surrogates must have the same length")
    groups: defaultdict[str, list[str]] = defaultdict(list)
    for original, surrogate in zip(originals, surrogates):
        groups[original].append(surrogate)

    violations: dict[str, list[str]] = {}
    consistent = 0
    for original, values in groups.items():
        unique_values = sorted(set(values))
        if len(unique_values) == 1:
            consistent += 1
        else:
            violations[original] = unique_values
    return ConsistencyMetric(
        score=_safe_rate(consistent, len(groups), zero_denominator=1.0),
        consistent=consistent,
        total=len(groups),
        violations=violations,
    )


def compute_latency_summary(latencies_ms: Sequence[int | float]) -> LatencyMetrics:
    """Compute p50, p95, and p99 latency from elapsed milliseconds."""
    if not latencies_ms:
        return LatencyMetrics(p50_ms=0.0, p95_ms=0.0, p99_ms=0.0, count=0)
    values = sorted(float(value) for value in latencies_ms)
    return LatencyMetrics(
        p50_ms=_percentile(values, 50),
        p95_ms=_percentile(values, 95),
        p99_ms=_percentile(values, 99),
        count=len(values),
    )


def compute_resource_metrics(
    *,
    peak_rss_bytes: int | None = None,
    model_size_bytes: int | None = None,
) -> ResourceMetrics:
    """Build a resource summary for RSS and model artifact size."""
    return ResourceMetrics(
        peak_rss_bytes=peak_rss_bytes,
        model_size_bytes=model_size_bytes,
    )


def reliability_bins(
    predictions_with_confidence: Iterable[Any],
    n_bins: int = 10,
) -> list[dict[str, int | float]]:
    """Return binned confidence-vs-accuracy records for predictions.

    Each input record must provide a confidence score in ``[0, 1]`` and a
    boolean correctness indicator. Mappings and objects may use ``confidence``,
    ``score``, or ``probability`` for the score, and ``correct``,
    ``is_correct``, ``matched``, or ``accurate`` for correctness. Two-item
    sequences are treated as ``(confidence, correct)``.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")

    confidence_sums = [0.0 for _ in range(n_bins)]
    correct_sums = [0 for _ in range(n_bins)]
    counts = [0 for _ in range(n_bins)]

    for record in predictions_with_confidence:
        confidence, correct = _confidence_correctness(record)
        index = min(int(confidence * n_bins), n_bins - 1)
        confidence_sums[index] += confidence
        correct_sums[index] += int(correct)
        counts[index] += 1

    bins: list[dict[str, int | float]] = []
    for index, count in enumerate(counts):
        bins.append(
            ReliabilityBin(
                bin_index=index,
                lower_bound=index / n_bins,
                upper_bound=(index + 1) / n_bins,
                mean_confidence=confidence_sums[index] / count if count else 0.0,
                empirical_accuracy=_safe_rate(
                    correct_sums[index], count, zero_denominator=0.0
                ),
                count=count,
            ).to_dict()
        )
    return bins


def expected_calibration_error(
    bins: Iterable[Mapping[str, Any]],
) -> float:
    """Compute expected calibration error over reliability bins."""
    rows = list(bins)
    total = sum(int(row.get("count", 0)) for row in rows)
    if total == 0:
        return 0.0

    error = 0.0
    for row in rows:
        count = int(row.get("count", 0))
        if count <= 0:
            continue
        mean_confidence = float(row["mean_confidence"])
        accuracy_value = (
            row["empirical_accuracy"]
            if "empirical_accuracy" in row
            else row["accuracy"]
        )
        empirical_accuracy = float(accuracy_value)
        error += (count / total) * abs(empirical_accuracy - mean_confidence)
    return error


def weighted_coverage(
    rows: Iterable[Mapping[str, Any]],
    *,
    covered_key: str = "covered",
    weight_key: str = "weight",
    zero_denominator: float = 1.0,
) -> RateMetric:
    """Compute weighted coverage from rows carrying covered/weight fields."""

    covered_weight = 0.0
    total_weight = 0.0
    for row in rows:
        raw_weight = row.get(weight_key, 1.0)
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError):
            continue
        if not isfinite(weight) or weight <= 0.0:
            continue
        total_weight += weight
        if bool(row.get(covered_key, False)):
            covered_weight += weight
    return RateMetric(
        rate=_safe_rate(
            covered_weight,
            total_weight,
            zero_denominator=zero_denominator,
        ),
        numerator=covered_weight,
        denominator=total_weight,
    )


def coverage_gaps_by_language(
    rows: Iterable[Mapping[str, Any]],
    *,
    target_coverage: float,
    languages: Iterable[str] = SUPPORTED_LANGUAGES,
    covered_key: str = "covered",
    weight_key: str = "weight",
    language_key: str = "language",
) -> dict[str, dict[str, float | str]]:
    """Return byte-stable per-language coverage gaps."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        language = str(row.get(language_key) or "en").lower()
        grouped[language].append(row)

    target = _bounded_unit_interval(target_coverage, "target_coverage")
    result: dict[str, dict[str, float | str]] = {}
    languages_to_report = {str(item).lower() for item in languages} | set(grouped)
    for language in sorted(languages_to_report):
        coverage = weighted_coverage(
            grouped.get(language, ()),
            covered_key=covered_key,
            weight_key=weight_key,
        )
        realized = float(coverage.rate)
        gap = max(target - realized, 0.0)
        result[language] = CoverageGap(
            slice_key=language,
            target_coverage=target,
            realized_coverage=realized,
            coverage_gap=gap,
            covered_weight=float(coverage.numerator),
            total_weight=float(coverage.denominator),
        ).to_dict()
    return result


def compute_metrics_bundle(
    gold_spans: Iterable[Any],
    predicted_spans: Iterable[Any],
    *,
    latencies_ms: Sequence[int | float] = (),
    cold_start_ms: float | None = None,
    peak_rss_bytes: int | None = None,
    model_size_bytes: int | None = None,
    abstention_thresholds: Any | None = None,
    abstention_confidence_threshold: float = 0.0,
    abstention_model_id: str | None = None,
    abstention_target_risk: float | None = None,
    abstention_confidence_level: float | None = None,
    abstention_bootstrap_resamples: int = 0,
    abstention_seed: int = 0,
    default_language: str = "en",
    default_device: str = "cpu",
    source_text: str | None = None,
) -> dict[str, Any]:
    """Compute the standard OM-018 benchmark metric bundle.

    ``cold_start_ms`` is reported as a non-gating edge metric nested under
    ``report.metrics['latency']['cold_start_ms']``. It is the first fixture
    call's wall-clock latency (model/tokenizer load plus first forward pass)
    and is excluded from the steady-state ``p50``/``p95``/``count`` sample.
    """
    text_length = len(source_text) if source_text is not None else None
    metrics = {
        "leakage": compute_leakage_rate(
            gold_spans,
            predicted_spans,
            default_language=default_language,
            default_device=default_device,
            source_text=source_text,
        ).to_dict(),
        "character_recall": compute_character_recall(
            gold_spans,
            predicted_spans,
            default_language=default_language,
            default_device=default_device,
            source_text=source_text,
        ).to_dict(),
        "recall_slices": compute_recall_slices(
            gold_spans,
            predicted_spans,
            default_language=default_language,
            default_device=default_device,
            source_text=source_text,
        ).to_dict(),
        "exact_span_f1": compute_exact_span_f1(
            gold_spans,
            predicted_spans,
            default_language=default_language,
            default_device=default_device,
            source_text=source_text,
        ).to_dict(),
        "relaxed_span_f1": compute_relaxed_span_f1(
            gold_spans,
            predicted_spans,
            default_language=default_language,
            default_device=default_device,
            source_text=source_text,
        ).to_dict(),
        "over_redaction_loss": compute_over_redaction_loss(
            gold_spans,
            predicted_spans,
            text_length=text_length,
            default_language=default_language,
            default_device=default_device,
            source_text=source_text,
        ).to_dict(),
        "latency": {
            **compute_latency_summary(latencies_ms).to_dict(),
            "cold_start_ms": cold_start_ms,
        },
        "resources": compute_resource_metrics(
            peak_rss_bytes=peak_rss_bytes,
            model_size_bytes=model_size_bytes,
        ).to_dict(),
    }
    if abstention_thresholds is not None:
        metrics["abstention"] = compute_abstention_metrics(
            gold_spans,
            predicted_spans,
            thresholds=abstention_thresholds,
            confidence_threshold=abstention_confidence_threshold,
            model_id=abstention_model_id,
            target_risk=abstention_target_risk,
            confidence_level=abstention_confidence_level,
            bootstrap_resamples=abstention_bootstrap_resamples,
            seed=abstention_seed,
            default_language=default_language,
            default_device=default_device,
            source_text=source_text,
        ).to_dict()
    return metrics


def bootstrap_ci(
    per_document_values: Sequence[Any],
    statistic: Callable[[Sequence[Any]], float],
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapCI:
    """Non-parametric bootstrap confidence interval over documents.

    ``per_document_values`` holds one entry per benchmark document (any value
    ``statistic`` knows how to aggregate, e.g. a ``(numerator, denominator)``
    pair). ``statistic`` maps a resampled list of those entries to a scalar
    point estimate. Documents are drawn with replacement from a
    ``random.Random(seed)`` stream, so a fixed ``seed`` is fully reproducible.

    The returned interval always brackets the point estimate. Degenerate inputs
    -- an empty corpus or a single document -- cannot vary under resampling and
    yield a zero-width interval flagged with ``degenerate=True``.
    """
    values = list(per_document_values)
    point = float(statistic(values))
    if len(values) < 2:
        return BootstrapCI(
            point=point,
            lower=point,
            upper=point,
            n_resamples=n_resamples,
            alpha=alpha,
            degenerate=True,
        )

    rng = random.Random(seed)
    size = len(values)
    samples: list[float] = []
    for _ in range(n_resamples):
        resample = [values[rng.randrange(size)] for _ in range(size)]
        samples.append(float(statistic(resample)))
    samples.sort()

    lower = _percentile(samples, 100.0 * (alpha / 2.0))
    upper = _percentile(samples, 100.0 * (1.0 - alpha / 2.0))
    # Guarantee the point estimate is bracketed even when the bootstrap
    # distribution is skewed to one side of it.
    return BootstrapCI(
        point=point,
        lower=min(lower, point),
        upper=max(upper, point),
        n_resamples=n_resamples,
        alpha=alpha,
        degenerate=False,
    )


def paired_significance(
    per_document_a: Sequence[Any],
    per_document_b: Sequence[Any],
    statistic: str | Callable[[Sequence[Any]], float],
    n_resamples: int = 1000,
    seed: int = 0,
) -> PairedSignificance:
    """Run a paired permutation test between two benchmark runs.

    ``per_document_a`` and ``per_document_b`` must be aligned one-to-one by
    benchmark document. The null hypothesis is that the two systems are
    exchangeable within each aligned document pair, so each resample randomly
    swaps the A/B labels per document before recomputing the scalar statistic.

    The observed delta is ``statistic(A) - statistic(B)``. Built-in statistic
    names are ``"leakage"``/``"leakage_rate"`` for leaked-character rates,
    ``"character_recall"``/``"recall"`` for character recall, and
    ``"f1"``/``"exact_span_f1"``/``"relaxed_span_f1"`` for span F1. Rate
    inputs use per-document ``(numerator, denominator)`` pairs; F1 inputs use
    ``(true_positives, false_positives, false_negatives)`` triples.
    """
    values_a = list(per_document_a)
    values_b = list(per_document_b)
    if len(values_a) != len(values_b):
        raise ValueError("per_document_a and per_document_b must have the same length")
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")

    scorer = _paired_statistic(statistic)
    observed_delta = float(scorer(values_a)) - float(scorer(values_b))
    if not values_a:
        return PairedSignificance(
            observed_delta=observed_delta,
            p_value=1.0,
            n_resamples=n_resamples,
        )

    rng = random.Random(seed)
    extreme = 0
    observed_abs = abs(observed_delta)
    tolerance = 1e-12
    for _ in range(n_resamples):
        sample_a: list[Any] = []
        sample_b: list[Any] = []
        for value_a, value_b in zip(values_a, values_b):
            if rng.random() < 0.5:
                sample_a.append(value_b)
                sample_b.append(value_a)
            else:
                sample_a.append(value_a)
                sample_b.append(value_b)
        sample_delta = float(scorer(sample_a)) - float(scorer(sample_b))
        if abs(sample_delta) + tolerance >= observed_abs:
            extreme += 1

    return PairedSignificance(
        observed_delta=observed_delta,
        p_value=(extreme + 1) / (n_resamples + 1),
        n_resamples=n_resamples,
    )


def compute_confidence_intervals(
    per_document_spans: Sequence[tuple[Iterable[Any], Iterable[Any]]],
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    default_language: str = "en",
    default_device: str = "cpu",
) -> dict[str, dict[str, Any]]:
    """Bootstrap per-document CIs for leakage rate, character recall, and F1.

    Each entry in ``per_document_spans`` is a ``(gold_spans, predicted_spans)``
    pair for one benchmark document. Documents are resampled with replacement so
    the intervals reflect document-level uncertainty in the corpus point
    estimates. Returns a mapping keyed by metric name (``leakage``,
    ``character_recall``, ``exact_span_f1``, ``relaxed_span_f1``) whose values
    are :meth:`BootstrapCI.to_dict` payloads.
    """
    leakage_docs: list[tuple[int, int]] = []
    recall_docs: list[tuple[int, int]] = []
    exact_docs: list[tuple[int, int, int]] = []
    relaxed_docs: list[tuple[int, int, int]] = []
    for gold_spans, predicted_spans in per_document_spans:
        leakage = compute_leakage_rate(
            gold_spans,
            predicted_spans,
            default_language=default_language,
            default_device=default_device,
        )
        leakage_docs.append((leakage.leaked_chars, leakage.total_chars))
        recall = compute_character_recall(
            gold_spans,
            predicted_spans,
            default_language=default_language,
            default_device=default_device,
        )
        recall_docs.append((int(recall.numerator), int(recall.denominator)))
        exact = compute_exact_span_f1(
            gold_spans,
            predicted_spans,
            default_language=default_language,
            default_device=default_device,
        )
        exact_docs.append(
            (exact.true_positives, exact.false_positives, exact.false_negatives)
        )
        relaxed = compute_relaxed_span_f1(
            gold_spans,
            predicted_spans,
            default_language=default_language,
            default_device=default_device,
        )
        relaxed_docs.append(
            (relaxed.true_positives, relaxed.false_positives, relaxed.false_negatives)
        )

    return {
        "leakage": bootstrap_ci(
            leakage_docs,
            lambda docs: _ratio_over_docs(docs, zero_denominator=0.0),
            n_resamples=n_resamples,
            alpha=alpha,
            seed=seed,
        ).to_dict(),
        "character_recall": bootstrap_ci(
            recall_docs,
            lambda docs: _ratio_over_docs(docs, zero_denominator=1.0),
            n_resamples=n_resamples,
            alpha=alpha,
            seed=seed,
        ).to_dict(),
        "exact_span_f1": bootstrap_ci(
            exact_docs,
            _f1_over_docs,
            n_resamples=n_resamples,
            alpha=alpha,
            seed=seed,
        ).to_dict(),
        "relaxed_span_f1": bootstrap_ci(
            relaxed_docs,
            _f1_over_docs,
            n_resamples=n_resamples,
            alpha=alpha,
            seed=seed,
        ).to_dict(),
    }


def _ratio_over_docs(
    docs: Sequence[tuple[int | float, int | float]],
    *,
    zero_denominator: float,
) -> float:
    numerator = sum(item[0] for item in docs)
    denominator = sum(item[1] for item in docs)
    return _safe_rate(numerator, denominator, zero_denominator=zero_denominator)


def _f1_over_docs(docs: Sequence[tuple[int, int, int]]) -> float:
    true_positives = sum(item[0] for item in docs)
    false_positives = sum(item[1] for item in docs)
    false_negatives = sum(item[2] for item in docs)
    return _f1_from_counts(
        true_positives,
        true_positives + false_positives,
        true_positives + false_negatives,
    ).f1


def _paired_statistic(
    statistic: str | Callable[[Sequence[Any]], float],
) -> Callable[[Sequence[Any]], float]:
    if callable(statistic):
        return statistic
    if not isinstance(statistic, str):
        raise TypeError("statistic must be a supported name or callable")

    normalized = statistic.lower().replace("-", "_").replace(" ", "_")
    if normalized in {"leakage", "leakage_rate"}:
        return lambda docs: _ratio_over_docs(docs, zero_denominator=0.0)
    if normalized in {"character_recall", "char_recall", "recall"}:
        return lambda docs: _ratio_over_docs(docs, zero_denominator=1.0)
    if normalized in {"f1", "exact_span_f1", "relaxed_span_f1", "exact_f1"}:
        return _f1_over_docs
    raise ValueError(
        "statistic must be one of leakage, character_recall, f1, or a callable"
    )


def _read_value(data: Mapping[str, Any], key: str) -> Any:
    if key in data:
        return data[key]
    return None


def _read_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    value = _read_value(data, key)
    return value if isinstance(value, Mapping) else None


def _read_int(data: Mapping[str, Any], key: str) -> int | None:
    value = _read_value(data, key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _confidence_correctness(record: Any) -> tuple[float, bool]:
    if isinstance(record, tuple | list) and len(record) == 2:
        confidence = _coerce_confidence(record[0])
        return confidence, _coerce_bool(record[1])

    data = record if isinstance(record, Mapping) else vars(record)
    metadata = _read_mapping(data, "metadata") or {}

    confidence_value = None
    for key in ("confidence", "score", "probability"):
        value = _read_value(data, key)
        if value is None:
            value = metadata.get(key)
        if value is not None:
            confidence_value = value
            break
    if confidence_value is None:
        raise ValueError("prediction record must include a confidence score")

    correctness_value = None
    for key in ("correct", "is_correct", "matched", "accurate"):
        value = _read_value(data, key)
        if value is None:
            value = metadata.get(key)
        if value is not None:
            correctness_value = value
            break
    if correctness_value is None:
        raise ValueError("prediction record must include a correctness indicator")

    return _coerce_confidence(confidence_value), _coerce_bool(correctness_value)


def _relation_f1(
    gold: Sequence[RelationTriple],
    predicted: Sequence[RelationTriple],
    *,
    relaxed: bool,
) -> F1Metrics:
    matched_predictions: set[int] = set()
    true_positives = 0
    for gold_relation in gold:
        candidates = [
            (index, predicted_relation)
            for index, predicted_relation in enumerate(predicted)
            if index not in matched_predictions
            and _relation_match(gold_relation, predicted_relation, relaxed=relaxed)
        ]
        if not candidates:
            continue
        if relaxed:
            best_index, _ = max(
                candidates,
                key=lambda item: _relation_overlap_score(gold_relation, item[1]),
            )
        else:
            best_index = candidates[0][0]
        matched_predictions.add(best_index)
        true_positives += 1
    return _f1_from_counts(true_positives, len(predicted), len(gold))


def _relation_match(
    gold: RelationTriple,
    predicted: RelationTriple,
    *,
    relaxed: bool,
) -> bool:
    if gold.relation_type != predicted.relation_type:
        return False
    if relaxed:
        return gold.arg1.overlaps(predicted.arg1) and gold.arg2.overlaps(predicted.arg2)
    return (
        gold.arg1.start == predicted.arg1.start
        and gold.arg1.end == predicted.arg1.end
        and gold.arg2.start == predicted.arg2.start
        and gold.arg2.end == predicted.arg2.end
    )


def _relation_overlap_score(gold: RelationTriple, predicted: RelationTriple) -> int:
    return gold.arg1.overlap_length(predicted.arg1) + gold.arg2.overlap_length(
        predicted.arg2
    )


def _relation_from_mapping(data: Mapping[str, Any]) -> RelationTriple:
    relation_type = _relation_first_value(
        data,
        ("relation_type", "type", "label", "predicate"),
    )
    if relation_type is None:
        raise ValueError(f"relation is missing relation_type: {data!r}")
    metadata = data.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        metadata = {"value": metadata}
    return RelationTriple(
        relation_type=str(relation_type).strip(),
        arg1=_relation_argument_from_mapping(
            data,
            nested_keys=("arg1", "head", "source", "subject"),
            flat_prefixes=("arg1", "head", "source", "subject"),
        ),
        arg2=_relation_argument_from_mapping(
            data,
            nested_keys=("arg2", "tail", "target", "object", "attribute"),
            flat_prefixes=("arg2", "tail", "target", "object", "attribute"),
        ),
        metadata=dict(metadata),
    )


def _relation_from_sequence(data: Sequence[Any]) -> RelationTriple:
    if len(data) == 3:
        relation_type, arg1, arg2 = data
        return RelationTriple(
            relation_type=str(relation_type).strip(),
            arg1=_normalize_relation_argument(arg1),
            arg2=_normalize_relation_argument(arg2),
        )
    if len(data) == 5:
        relation_type, arg1_start, arg1_end, arg2_start, arg2_end = data
        return RelationTriple(
            relation_type=str(relation_type).strip(),
            arg1=RelationArgument(start=int(arg1_start), end=int(arg1_end)),
            arg2=RelationArgument(start=int(arg2_start), end=int(arg2_end)),
        )
    if len(data) == 6:
        _, relation_type, arg1_start, arg1_end, arg2_start, arg2_end = data
        return RelationTriple(
            relation_type=str(relation_type).strip(),
            arg1=RelationArgument(start=int(arg1_start), end=int(arg1_end)),
            arg2=RelationArgument(start=int(arg2_start), end=int(arg2_end)),
        )
    raise ValueError(
        "relation tuple must be (type, arg1, arg2), "
        "(type, arg1_start, arg1_end, arg2_start, arg2_end), or "
        "(id, type, arg1_start, arg1_end, arg2_start, arg2_end)"
    )


def _relation_argument_from_mapping(
    data: Mapping[str, Any],
    *,
    nested_keys: Sequence[str],
    flat_prefixes: Sequence[str],
) -> RelationArgument:
    for key in nested_keys:
        if key in data:
            return _normalize_relation_argument(data[key])
    for prefix in flat_prefixes:
        start = _relation_first_value(data, (f"{prefix}_start", f"{prefix}Start"))
        end = _relation_first_value(data, (f"{prefix}_end", f"{prefix}End"))
        if start is not None and end is not None:
            label = _relation_first_value(
                data,
                (f"{prefix}_label", f"{prefix}Label", f"{prefix}_type"),
            )
            text = _relation_first_value(data, (f"{prefix}_text", f"{prefix}Text"))
            return RelationArgument(
                start=int(start),
                end=int(end),
                label="" if label is None else str(label),
                text="" if text is None else str(text),
            )
    raise ValueError(f"relation is missing argument offsets: {data!r}")


def _normalize_relation_argument(value: Any) -> RelationArgument:
    if isinstance(value, RelationArgument):
        return value
    if isinstance(value, Mapping):
        nested = value.get("span")
        if nested is not None:
            return _normalize_relation_argument(nested)
        start = _relation_first_value(value, ("start", "begin", "offset_start"))
        end = _relation_first_value(value, ("end", "stop", "offset_end"))
        if start is None or end is None:
            raise ValueError(f"relation argument is missing start/end: {value!r}")
        metadata = value.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            metadata = {"value": metadata}
        label = _relation_first_value(value, ("label", "type", "source_label"))
        text = value.get("text", "")
        return RelationArgument(
            start=int(start),
            end=int(end),
            label="" if label is None else str(label),
            text="" if text is None else str(text),
            metadata=dict(metadata),
        )
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        if len(value) < 2:
            raise ValueError(
                f"relation argument tuple must have at least start/end: {value!r}"
            )
        label = value[2] if len(value) > 2 else ""
        text = value[3] if len(value) > 3 else ""
        return RelationArgument(
            start=int(value[0]),
            end=int(value[1]),
            label=str(label),
            text=str(text),
        )
    start = _read_attr(value, "start", "begin", "offset_start")
    end = _read_attr(value, "end", "stop", "offset_end")
    if start is None or end is None:
        raise ValueError(f"relation argument is missing start/end: {value!r}")
    label = _read_attr(value, "label", "type", "source_label") or ""
    text = _read_attr(value, "text") or ""
    metadata = _read_attr(value, "metadata") or {}
    if not isinstance(metadata, Mapping):
        metadata = {"value": metadata}
    return RelationArgument(
        start=int(start),
        end=int(end),
        label=str(label),
        text=str(text),
        metadata=dict(metadata),
    )


def _relation_first_value(data: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return value
    return None


def _read_attr(value: Any, *names: str) -> Any:
    for name in names:
        if hasattr(value, name):
            result = getattr(value, name)
            if result is not None:
                return result
    return None


def _confidence_for_span(span: EvalSpan) -> float:
    value = span.metadata.get("confidence")
    if value is None:
        value = span.metadata.get("score", span.metadata.get("probability", 1.0))
    return _coerce_confidence(value)


def _threshold_for_span(
    span: EvalSpan,
    thresholds: Any | None,
    *,
    default_threshold: float,
    model_id: str | None,
) -> float:
    if thresholds is None:
        return _coerce_confidence(default_threshold)
    lookup = getattr(thresholds, "lookup", None)
    if callable(lookup):
        return _coerce_confidence(
            lookup(
                span.label,
                span.language,
                model_id=model_id,
                default=default_threshold,
            )
        )
    if isinstance(thresholds, Mapping):
        return _threshold_from_mapping(
            thresholds,
            span.label,
            span.language,
            model_id=model_id,
            default_threshold=default_threshold,
        )
    return _coerce_confidence(default_threshold)


def _threshold_from_mapping(
    thresholds: Mapping[str, Any],
    label: str,
    language: str,
    *,
    model_id: str | None,
    default_threshold: float,
) -> float:
    candidates: list[Mapping[str, Any]] = []
    if model_id and isinstance(thresholds.get(model_id), Mapping):
        candidates.append(thresholds[model_id])
    candidates.append(thresholds)
    for candidate in candidates:
        value = _mapping_lookup(candidate, label)
        if isinstance(value, Mapping):
            if language in value:
                return _coerce_confidence(value[language])
            if "*" in value:
                return _coerce_confidence(value["*"])
        elif value is not None:
            return _coerce_confidence(value)
    return _coerce_confidence(default_threshold)


def _mapping_lookup(payload: Mapping[str, Any], label: str) -> Any:
    if label in payload:
        return payload[label]
    canonical = normalize_label(label)
    if canonical in payload:
        return payload[canonical]
    return None


def _exact_span_match(gold_span: EvalSpan, predicted_span: EvalSpan) -> bool:
    return (
        gold_span.label == predicted_span.label
        and gold_span.start == predicted_span.start
        and gold_span.end == predicted_span.end
    )


def _abstention_rate_by(
    decisions: Sequence[AbstentionDecision],
    keys: Sequence[str],
    *,
    group_key: str,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in keys:
        group = [
            decision for decision in decisions if getattr(decision, group_key) == key
        ]
        result[key] = _safe_rate(
            sum(1 for decision in group if not decision.accepted),
            len(group),
            zero_denominator=0.0,
        )
    return result


def _residual_risk_by(
    decisions: Sequence[AbstentionDecision],
    keys: Sequence[str],
    *,
    group_key: str,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in keys:
        group = [
            decision
            for decision in decisions
            if getattr(decision, group_key) == key and decision.accepted
        ]
        result[key] = _safe_rate(
            sum(1 for decision in group if decision.residual_error),
            len(group),
            zero_denominator=0.0,
        )
    return result


def _decision_residual_risk(decisions: Sequence[AbstentionDecision]) -> float:
    accepted = [decision for decision in decisions if decision.accepted]
    return _safe_rate(
        sum(1 for decision in accepted if decision.residual_error),
        len(accepted),
        zero_denominator=0.0,
    )


def _coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"confidence must be numeric: {value!r}") from exc
    if not isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
        raise ValueError(f"confidence must be between 0 and 1: {value!r}")
    return confidence


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"correctness indicator must be boolean-like: {value!r}")


def _safe_rate(
    numerator: int | float,
    denominator: int | float,
    zero_denominator: float,
) -> float:
    if denominator == 0:
        return zero_denominator
    return float(numerator) / float(denominator)


def _bounded_unit_interval(value: Any, field_name: str) -> float:
    result = float(value)
    if not isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")
    return result


def _f1_from_counts(
    true_positives: int, predicted_count: int, gold_count: int
) -> F1Metrics:
    false_positives = predicted_count - true_positives
    false_negatives = gold_count - true_positives
    precision = _safe_rate(true_positives, predicted_count, zero_denominator=1.0)
    recall = _safe_rate(true_positives, gold_count, zero_denominator=1.0)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return F1Metrics(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def _label_aware_overlap(gold_span: EvalSpan, pred_span: EvalSpan) -> bool:
    if gold_span.label != pred_span.label:
        return False
    overlaps = detect_overlapping_entities(
        [gold_span.to_entity(), pred_span.to_entity()]
    )
    return bool(overlaps)


def _overlap_len(a: EvalSpan, b: EvalSpan) -> int:
    if not _label_aware_overlap(a, b):
        return 0
    return max(min(a.end, b.end) - max(a.start, b.start), 0)


def _covered_char_count(gold_span: EvalSpan, predicted: Sequence[EvalSpan]) -> int:
    intervals: list[tuple[int, int]] = []
    for pred_span in predicted:
        if not _label_aware_overlap(gold_span, pred_span):
            continue
        start = max(gold_span.start, pred_span.start)
        end = min(gold_span.end, pred_span.end)
        if start < end:
            intervals.append((start, end))
    return _merged_interval_length(intervals)


def _merged_interval_length(intervals: Sequence[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            old_start, old_end = merged[-1]
            merged[-1] = (old_start, max(old_end, end))
    return sum(end - start for start, end in merged)


def _positions_for_spans(spans: Sequence[EvalSpan]) -> set[int]:
    positions: set[int] = set()
    for span in spans:
        positions.update(range(max(span.start, 0), max(span.end, span.start)))
    return positions


def _slice_keys(
    required: Iterable[str],
    *maps: Mapping[str, Any],
) -> list[str]:
    keys = set(required)
    for item in maps:
        keys.update(item)
    return sorted(keys)


def _rate_map(
    keys: Iterable[str],
    numerators: Mapping[str, int],
    denominators: Mapping[str, int],
    zero_denominator: float,
) -> dict[str, float]:
    return {
        key: _safe_rate(
            numerators.get(key, 0), denominators.get(key, 0), zero_denominator
        )
        for key in keys
    }


def _count_map(keys: Iterable[str], counts: Mapping[str, int]) -> dict[str, int]:
    return {key: int(counts.get(key, 0)) for key in keys}


def _parse_date(value: str) -> date | None:
    candidates = ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d")
    for fmt in candidates:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value).date()
    except ValueError:
        return None


def _percentile(values: Sequence[float], percentile: int | float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    rank = ceil((float(percentile) / 100.0) * len(values))
    index = min(max(rank - 1, 0), len(values) - 1)
    return values[index]


def relation_assertion_consistency(
    predicted: Iterable[tuple[Any, str]],
    gold: Iterable[tuple[Any, str]],
) -> ConsistencyMetric:
    """Score how consistently predicted relation-assertion tags match the gold.

    Each input is a sequence of ``(relation_key, status)`` pairs. The score is
    the fraction of gold relations whose predicted status matches; every
    mismatch is recorded as a violation keyed by the relation.
    """

    predicted_by_key = {key: status for key, status in predicted}
    gold_by_key = {key: status for key, status in gold}

    consistent = 0
    violations: dict[str, list[str]] = {}
    for key, gold_status in gold_by_key.items():
        predicted_status = predicted_by_key.get(key)
        if predicted_status == gold_status:
            consistent += 1
        else:
            violations[str(key)] = [f"expected {gold_status}, got {predicted_status}"]

    total = len(gold_by_key)
    score = 1.0 if total == 0 else consistent / total
    return ConsistencyMetric(
        score=score,
        consistent=consistent,
        total=total,
        violations=violations,
    )


__all__ = [
    "ABSTENTION_ROUTE_ACCEPT",
    "ABSTENTION_ROUTE_REDACT",
    "ABSTENTION_ROUTE_REVIEW",
    "CRITICAL_ABSTENTION_LABELS",
    "DEVICE_TIERS",
    "AbstentionDecision",
    "AbstentionMetrics",
    "EvalSpan",
    "RateMetric",
    "F1Metrics",
    "RelationArgument",
    "RelationTriple",
    "LeakageMetrics",
    "RecallSlices",
    "ConsistencyMetric",
    "LatencyMetrics",
    "ResourceMetrics",
    "BootstrapCI",
    "ReliabilityBin",
    "PairedSignificance",
    "CoverageGap",
    "normalize_eval_span",
    "normalize_eval_spans",
    "normalize_relation_triple",
    "normalize_relation_triples",
    "compute_leakage_rate",
    "compute_character_recall",
    "compute_recall_slices",
    "compute_exact_span_f1",
    "compute_relaxed_span_f1",
    "compute_strict_relation_f1",
    "compute_relaxed_relation_f1",
    "compute_relation_metrics_bundle",
    "compute_relation_confidence_intervals",
    "compute_over_redaction_loss",
    "compute_clinical_utility_loss",
    "compute_abstention_metrics",
    "compute_date_shift_consistency",
    "compute_surrogate_consistency",
    "compute_latency_summary",
    "compute_resource_metrics",
    "coverage_gaps_by_language",
    "reliability_bins",
    "expected_calibration_error",
    "weighted_coverage",
    "compute_metrics_bundle",
    "bootstrap_ci",
    "paired_significance",
    "compute_confidence_intervals",
    "abstention_route",
    "apply_abstention_policy",
    "bootstrap_abstention_residual_risk",
    "relation_assertion_consistency",
]
