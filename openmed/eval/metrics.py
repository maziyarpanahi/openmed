"""Benchmark metrics for OpenMed de-identification evaluation."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from math import ceil, isfinite
from typing import Any, Callable, Iterable, Mapping, Sequence

from openmed.core.decoding.spans import iter_grapheme_cluster_spans
from openmed.core.labels import CANONICAL_LABELS, normalize_label
from openmed.core.pii_i18n import SUPPORTED_LANGUAGES
from openmed.core.quality_gates import detect_overlapping_entities
from openmed.core.script_detect import UNKNOWN_SCRIPT, segment_by_script
from openmed.processing.outputs import EntityPrediction

DEVICE_TIERS: tuple[str, ...] = ("cpu", "mlx-fp", "mlx-8bit", "coreml")
MIXED_SCRIPT_LEAKAGE_CEILING = 0.01
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
CRITICAL_FINDING_CATEGORY_DIAGNOSIS = "critical_diagnosis"
CRITICAL_FINDING_CATEGORY_DRUG_ALLERGY = "drug_allergy"
CRITICAL_FINDING_CATEGORY_RESULT = "critical_result"
CRITICAL_FINDING_CATEGORIES: tuple[str, ...] = (
    CRITICAL_FINDING_CATEGORY_DIAGNOSIS,
    CRITICAL_FINDING_CATEGORY_DRUG_ALLERGY,
    CRITICAL_FINDING_CATEGORY_RESULT,
)
_CRITICAL_FINDING_CATEGORY_ALIASES = {
    "diagnosis": CRITICAL_FINDING_CATEGORY_DIAGNOSIS,
    "critical_diagnosis": CRITICAL_FINDING_CATEGORY_DIAGNOSIS,
    "condition": CRITICAL_FINDING_CATEGORY_DIAGNOSIS,
    "critical_condition": CRITICAL_FINDING_CATEGORY_DIAGNOSIS,
    "allergy": CRITICAL_FINDING_CATEGORY_DRUG_ALLERGY,
    "drug_allergy": CRITICAL_FINDING_CATEGORY_DRUG_ALLERGY,
    "medication_allergy": CRITICAL_FINDING_CATEGORY_DRUG_ALLERGY,
    "result": CRITICAL_FINDING_CATEGORY_RESULT,
    "critical_result": CRITICAL_FINDING_CATEGORY_RESULT,
    "lab_result": CRITICAL_FINDING_CATEGORY_RESULT,
    "critical_lab": CRITICAL_FINDING_CATEGORY_RESULT,
}
_CRITICAL_FINDING_CATEGORY_KEYS = (
    "critical_finding_category",
    "critical_category",
    "critical_finding_type",
)
_CRITICAL_FINDING_MARKER_KEYS = ("critical_finding", "critical", "must_not_miss")


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
class _GraphemeTally:
    """Internal whole-grapheme counts for one gold span."""

    matched: int
    total: int
    matched_by_script: Mapping[str, int]
    total_by_script: Mapping[str, int]


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
class LeakageMetrics:
    """Grapheme-weighted PHI leakage rate and required slices.

    The ``*_chars`` fields remain compatibility aliases, but their counts use
    complete user-perceived grapheme clusters rather than Unicode code points.
    """

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
    by_script: dict[str, float] = field(default_factory=dict)
    leaked_chars_by_script: dict[str, int] = field(default_factory=dict)
    total_chars_by_script: dict[str, int] = field(default_factory=dict)

    @property
    def leaked_graphemes(self) -> int:
        """Return the number of leaked grapheme clusters."""
        return self.leaked_chars

    @property
    def total_graphemes(self) -> int:
        """Return the number of gold grapheme clusters."""
        return self.total_chars

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall,
            "by_label": self.by_label,
            "by_language": self.by_language,
            "by_device": self.by_device,
            "by_script": self.by_script,
            "unit": "grapheme_cluster",
            "leaked_graphemes": self.leaked_graphemes,
            "total_graphemes": self.total_graphemes,
            "leaked_chars": self.leaked_chars,
            "total_chars": self.total_chars,
            "leaked_chars_by_label": self.leaked_chars_by_label,
            "total_chars_by_label": self.total_chars_by_label,
            "leaked_chars_by_language": self.leaked_chars_by_language,
            "total_chars_by_language": self.total_chars_by_language,
            "leaked_chars_by_device": self.leaked_chars_by_device,
            "total_chars_by_device": self.total_chars_by_device,
            "leaked_graphemes_by_script": self.leaked_chars_by_script,
            "total_graphemes_by_script": self.total_chars_by_script,
            "leaked_chars_by_script": self.leaked_chars_by_script,
            "total_chars_by_script": self.total_chars_by_script,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


@dataclass(frozen=True)
class MixedScriptLeakageMetrics:
    """Leakage gate result for a confusable/mixed-script attack corpus."""

    leakage: LeakageMetrics
    ceiling: float = MIXED_SCRIPT_LEAKAGE_CEILING
    pre_defense_baseline: float | None = None

    @property
    def passed(self) -> bool:
        """Return whether leakage stays at or below the strict ceiling."""
        return self.leakage.overall <= self.ceiling

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready gate payload."""
        return {
            **self.leakage.to_dict(),
            "ceiling": self.ceiling,
            "passed": self.passed,
            "pre_defense_baseline": self.pre_defense_baseline,
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
    """Grapheme recall sliced by label, language, device, and script."""

    overall: float
    by_label: dict[str, float]
    by_language: dict[str, float]
    by_device: dict[str, float]
    covered_chars: int
    total_chars: int
    by_script: dict[str, float] = field(default_factory=dict)
    covered_chars_by_script: dict[str, int] = field(default_factory=dict)
    total_chars_by_script: dict[str, int] = field(default_factory=dict)

    @property
    def covered_graphemes(self) -> int:
        """Return the number of fully covered gold grapheme clusters."""
        return self.covered_chars

    @property
    def total_graphemes(self) -> int:
        """Return the number of gold grapheme clusters."""
        return self.total_chars

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall,
            "by_label": self.by_label,
            "by_language": self.by_language,
            "by_device": self.by_device,
            "by_script": self.by_script,
            "unit": "grapheme_cluster",
            "covered_graphemes": self.covered_graphemes,
            "total_graphemes": self.total_graphemes,
            "covered_chars": self.covered_chars,
            "total_chars": self.total_chars,
            "covered_graphemes_by_script": self.covered_chars_by_script,
            "total_graphemes_by_script": self.total_chars_by_script,
            "covered_chars_by_script": self.covered_chars_by_script,
            "total_chars_by_script": self.total_chars_by_script,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


@dataclass(frozen=True)
class CriticalFindingMiss:
    """PHI-free detail for one missed critical clinical finding."""

    category: str
    fixture_id: str
    start: int
    end: int
    label: str

    def to_dict(self) -> dict[str, int | str]:
        return {
            "category": self.category,
            "fixture_id": self.fixture_id,
            "start": int(self.start),
            "end": int(self.end),
            "label": self.label,
        }

    def __getitem__(self, key: str) -> int | str:
        return self.to_dict()[key]


@dataclass(frozen=True)
class CriticalFindingRecallMetrics:
    """Span recall over gold spans marked as critical clinical findings."""

    overall: float
    by_category: dict[str, float]
    covered: int
    total: int
    covered_by_category: dict[str, int]
    total_by_category: dict[str, int]
    missed_findings: tuple[CriticalFindingMiss, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall,
            "by_category": self.by_category,
            "covered": int(self.covered),
            "total": int(self.total),
            "covered_by_category": self.covered_by_category,
            "total_by_category": self.total_by_category,
            "missed_findings": [finding.to_dict() for finding in self.missed_findings],
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
    """Compute first-class grapheme-cluster-weighted PHI leakage.

    Leakage is the number of gold PHI graphemes not fully covered by any
    same-label prediction divided by the total number of gold PHI graphemes.
    The same numerator/denominator accounting is reported overall and by
    label, language, device, and Unicode script.
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

    leakage_tallies = [
        _invert_grapheme_tally(_grapheme_coverage_tally(span, predicted))
        for span in gold
    ]
    return _leakage_metrics_from_tallies(gold, leakage_tallies)


def compute_mixed_script_leakage(
    gold_spans: Iterable[Any],
    predicted_spans: Iterable[Any],
    *,
    ceiling: float = MIXED_SCRIPT_LEAKAGE_CEILING,
    pre_defense_baseline: float | None = None,
    default_language: str = "en",
    default_device: str = "cpu",
    source_text: str | None = None,
) -> MixedScriptLeakageMetrics:
    """Compute grapheme-cluster leakage and apply the evasion ceiling."""

    if not 0.0 <= ceiling <= 1.0:
        raise ValueError("mixed-script leakage ceiling must be between 0 and 1")
    if pre_defense_baseline is not None and not 0.0 <= pre_defense_baseline <= 1.0:
        raise ValueError("pre-defense leakage baseline must be between 0 and 1")
    leakage = compute_leakage_rate(
        gold_spans,
        predicted_spans,
        default_language=default_language,
        default_device=default_device,
        source_text=source_text,
    )
    return MixedScriptLeakageMetrics(
        leakage=leakage,
        ceiling=ceiling,
        pre_defense_baseline=pre_defense_baseline,
    )


def compute_extraction_reemission_leakage(
    gold_spans: Iterable[Any],
    extraction_outputs: Any,
    *,
    default_language: str = "en",
    default_device: str = "cpu",
    source_text: str | None = None,
) -> LeakageMetrics:
    """Compute PHI re-emission leakage from extraction or grounding output.

    The scanner walks arbitrary nested extraction payloads, including fact
    values, concept text, evidence spans, and FHIR-style resources. A gold PHI
    span is counted as leaked when its surface form is re-emitted in an output
    field or when an explicit output offset overlaps that gold span. Counts use
    the same grapheme-weighted numerator and denominator as
    :func:`compute_leakage_rate`.
    """
    gold = normalize_eval_spans(
        gold_spans,
        default_language=default_language,
        default_device=default_device,
        source_text=source_text,
    )
    leaked_intervals: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
    surface_index = _surface_index(gold)

    for text in _iter_extraction_text_values(extraction_outputs):
        folded = text.casefold()
        for span_index, surface in surface_index:
            if _contains_surface(folded, surface):
                span = gold[span_index]
                leaked_intervals[span_index].append((span.start, span.end))

    for start, end in _iter_extraction_offsets(extraction_outputs):
        for span_index, span in enumerate(gold):
            overlap_start = max(start, span.start)
            overlap_end = min(end, span.end)
            if overlap_start < overlap_end:
                leaked_intervals[span_index].append((overlap_start, overlap_end))

    leakage_tallies = [
        _grapheme_overlap_tally(span, leaked_intervals.get(index, ()))
        for index, span in enumerate(gold)
    ]
    return _leakage_metrics_from_tallies(gold, leakage_tallies)


def _leakage_metrics_from_tallies(
    gold: Sequence[EvalSpan],
    tallies: Sequence[_GraphemeTally],
) -> LeakageMetrics:
    leaked_by_label: defaultdict[str, int] = defaultdict(int)
    total_by_label: defaultdict[str, int] = defaultdict(int)
    leaked_by_language: defaultdict[str, int] = defaultdict(int)
    total_by_language: defaultdict[str, int] = defaultdict(int)
    leaked_by_device: defaultdict[str, int] = defaultdict(int)
    total_by_device: defaultdict[str, int] = defaultdict(int)
    leaked_by_script: defaultdict[str, int] = defaultdict(int)
    total_by_script: defaultdict[str, int] = defaultdict(int)

    total_chars = 0
    leaked_chars = 0
    for span, tally in zip(gold, tallies):
        leaked = min(max(int(tally.matched), 0), tally.total)
        total_chars += tally.total
        leaked_chars += leaked
        total_by_label[span.label] += tally.total
        leaked_by_label[span.label] += leaked
        total_by_language[span.language] += tally.total
        leaked_by_language[span.language] += leaked
        total_by_device[span.device] += tally.total
        leaked_by_device[span.device] += leaked
        for script, total in tally.total_by_script.items():
            total_by_script[script] += int(total)
        for script, script_leaked in tally.matched_by_script.items():
            leaked_by_script[script] += int(script_leaked)

    label_keys = _slice_keys(CANONICAL_LABELS, total_by_label, leaked_by_label)
    language_keys = _slice_keys(
        SUPPORTED_LANGUAGES, total_by_language, leaked_by_language
    )
    device_keys = _slice_keys(DEVICE_TIERS, total_by_device, leaked_by_device)
    script_keys = tuple(sorted(set(total_by_script) | set(leaked_by_script)))

    return LeakageMetrics(
        overall=_safe_rate(leaked_chars, total_chars, zero_denominator=0.0),
        by_label=_rate_map(label_keys, leaked_by_label, total_by_label, 0.0),
        by_language=_rate_map(
            language_keys, leaked_by_language, total_by_language, 0.0
        ),
        by_device=_rate_map(device_keys, leaked_by_device, total_by_device, 0.0),
        by_script=_rate_map(
            script_keys,
            leaked_by_script,
            total_by_script,
            0.0,
        ),
        leaked_chars=leaked_chars,
        total_chars=total_chars,
        leaked_chars_by_label=_count_map(label_keys, leaked_by_label),
        total_chars_by_label=_count_map(label_keys, total_by_label),
        leaked_chars_by_language=_count_map(language_keys, leaked_by_language),
        total_chars_by_language=_count_map(language_keys, total_by_language),
        leaked_chars_by_device=_count_map(device_keys, leaked_by_device),
        total_chars_by_device=_count_map(device_keys, total_by_device),
        leaked_chars_by_script=_count_map(script_keys, leaked_by_script),
        total_chars_by_script=_count_map(script_keys, total_by_script),
    )


def compute_character_recall(
    gold_spans: Iterable[Any],
    predicted_spans: Iterable[Any],
    *,
    default_language: str = "en",
    default_device: str = "cpu",
    source_text: str | None = None,
) -> RateMetric:
    """Compute label-aware grapheme-cluster recall over gold PHI spans."""
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
    tallies = [_grapheme_coverage_tally(span, predicted) for span in gold]
    total_chars = sum(tally.total for tally in tallies)
    covered_chars = sum(tally.matched for tally in tallies)
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
    """Compute grapheme recall sliced by label, language, device, and script."""
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
    covered_by_script: defaultdict[str, int] = defaultdict(int)
    total_by_script: defaultdict[str, int] = defaultdict(int)

    total_chars = 0
    covered_chars = 0
    for span in gold:
        tally = _grapheme_coverage_tally(span, predicted)
        covered = tally.matched
        total_chars += tally.total
        covered_chars += covered
        total_by_label[span.label] += tally.total
        covered_by_label[span.label] += covered
        total_by_language[span.language] += tally.total
        covered_by_language[span.language] += covered
        total_by_device[span.device] += tally.total
        covered_by_device[span.device] += covered
        for script, total in tally.total_by_script.items():
            total_by_script[script] += int(total)
        for script, script_covered in tally.matched_by_script.items():
            covered_by_script[script] += int(script_covered)

    label_keys = _slice_keys(CANONICAL_LABELS, total_by_label, covered_by_label)
    language_keys = _slice_keys(
        SUPPORTED_LANGUAGES, total_by_language, covered_by_language
    )
    device_keys = _slice_keys(DEVICE_TIERS, total_by_device, covered_by_device)
    script_keys = tuple(sorted(set(total_by_script) | set(covered_by_script)))

    return RecallSlices(
        overall=_safe_rate(covered_chars, total_chars, zero_denominator=1.0),
        by_label=_rate_map(label_keys, covered_by_label, total_by_label, 1.0),
        by_language=_rate_map(
            language_keys, covered_by_language, total_by_language, 1.0
        ),
        by_device=_rate_map(device_keys, covered_by_device, total_by_device, 1.0),
        by_script=_rate_map(
            script_keys,
            covered_by_script,
            total_by_script,
            1.0,
        ),
        covered_chars=covered_chars,
        total_chars=total_chars,
        covered_chars_by_script=_count_map(script_keys, covered_by_script),
        total_chars_by_script=_count_map(script_keys, total_by_script),
    )


def normalize_critical_finding_category(value: Any) -> str:
    """Return a stable critical-finding category identifier."""
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return _CRITICAL_FINDING_CATEGORY_ALIASES.get(normalized, normalized)


def critical_finding_category(span: Any) -> str | None:
    """Return the critical-finding category for *span*, if it is marked critical."""
    if isinstance(span, EvalSpan):
        data: Mapping[str, Any] = {}
        metadata = span.metadata
    elif isinstance(span, Mapping):
        data = span
        metadata = _read_mapping(data, "metadata") or {}
        if not isinstance(metadata, Mapping):
            metadata = {}
    else:
        data = vars(span)
        metadata = _read_mapping(data, "metadata") or {}
        if not isinstance(metadata, Mapping):
            metadata = {}

    raw_category = _first_present_value(
        *(metadata.get(key) for key in _CRITICAL_FINDING_CATEGORY_KEYS),
        *(data.get(key) for key in _CRITICAL_FINDING_CATEGORY_KEYS),
    )
    marked = any(
        _truthy(_first_present_value(metadata.get(key), data.get(key)))
        for key in _CRITICAL_FINDING_MARKER_KEYS
    )
    if raw_category is None:
        return None
    category = normalize_critical_finding_category(raw_category)
    if category in CRITICAL_FINDING_CATEGORIES and (
        marked or bool(str(raw_category).strip())
    ):
        return category
    return None


def compute_critical_finding_recall(
    gold_spans: Iterable[Any],
    predicted_spans: Iterable[Any],
    *,
    default_language: str = "en",
    default_device: str = "cpu",
    source_text: str | None = None,
) -> CriticalFindingRecallMetrics:
    """Compute span recall over gold spans marked as critical findings.

    The metric keeps the output PHI-free by reporting only the missed finding's
    category, fixture id, canonical label, and offsets.
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

    covered_by_category: defaultdict[str, int] = defaultdict(int)
    total_by_category: defaultdict[str, int] = defaultdict(int)
    missed: list[CriticalFindingMiss] = []
    covered = 0
    total = 0

    for span in gold:
        category = critical_finding_category(span)
        if category is None:
            continue
        total += 1
        total_by_category[category] += 1
        found = any(
            _label_aware_overlap(span, predicted_span) for predicted_span in predicted
        )
        if found:
            covered += 1
            covered_by_category[category] += 1
            continue
        missed.append(
            CriticalFindingMiss(
                category=category,
                fixture_id=_fixture_id_for_span(span),
                start=span.start,
                end=span.end,
                label=span.label,
            )
        )

    category_keys = _slice_keys(
        CRITICAL_FINDING_CATEGORIES,
        total_by_category,
        covered_by_category,
    )
    return CriticalFindingRecallMetrics(
        overall=_safe_rate(covered, total, zero_denominator=1.0),
        by_category=_rate_map(
            category_keys,
            covered_by_category,
            total_by_category,
            1.0,
        ),
        covered=covered,
        total=total,
        covered_by_category=_count_map(category_keys, covered_by_category),
        total_by_category=_count_map(category_keys, total_by_category),
        missed_findings=tuple(missed),
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
    extraction_outputs: Any | None = None,
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
        "critical_finding_recall": compute_critical_finding_recall(
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
    if extraction_outputs is not None:
        metrics["extraction_reemission_leakage"] = (
            compute_extraction_reemission_leakage(
                gold_spans,
                extraction_outputs,
                default_language=default_language,
                default_device=default_device,
                source_text=source_text,
            ).to_dict()
        )
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


_RADIOLOGY_SECTION_KEYS = ("findings", "impression", "recommendation")


def _normalize_section_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def section_boundary_accuracy(
    predicted: Iterable[Mapping[str, Any]],
    gold: Iterable[Mapping[str, Any]],
    *,
    section_keys: Sequence[str] = _RADIOLOGY_SECTION_KEYS,
) -> RateMetric:
    """Fraction of report sections whose predicted text matches the gold text.

    ``predicted`` and ``gold`` are parallel iterables of per-report mappings,
    each carrying ``<section>_text`` keys (e.g. ``findings_text``). Text is
    whitespace-normalized before comparison so segmentation is scored, not
    incidental spacing. Parser-agnostic: callers pass already-parsed sections.
    """
    correct = 0
    total = 0
    for predicted_row, gold_row in zip(predicted, gold, strict=True):
        for key in section_keys:
            total += 1
            field = f"{key}_text"
            if _normalize_section_text(
                predicted_row.get(field)
            ) == _normalize_section_text(gold_row.get(field)):
                correct += 1
    return RateMetric(
        rate=_safe_rate(correct, total, zero_denominator=1.0),
        numerator=correct,
        denominator=total,
    )


def stated_category_accuracy(
    predicted: Iterable[Mapping[str, Any]],
    gold: Iterable[Mapping[str, Any]],
) -> RateMetric:
    """Fraction of reports whose stated ``(system, category)`` matches the gold.

    A report with no stated category (both ``None``) counts as correct only when
    the gold also has none, so inferring a category where none is written is
    penalized. Parser-agnostic: callers pass already-parsed rows carrying
    ``assessment_system`` and ``assessment_category``.
    """
    correct = 0
    total = 0
    for predicted_row, gold_row in zip(predicted, gold, strict=True):
        total += 1
        predicted_pair = (
            predicted_row.get("assessment_system"),
            predicted_row.get("assessment_category"),
        )
        gold_pair = (
            gold_row.get("assessment_system"),
            gold_row.get("assessment_category"),
        )
        if predicted_pair == gold_pair:
            correct += 1
    return RateMetric(
        rate=_safe_rate(correct, total, zero_denominator=1.0),
        numerator=correct,
        denominator=total,
    )


_RADIOLOGY_FINDING_TUPLE_FIELDS = (
    "finding",
    "laterality",
    "size_value",
    "size_unit",
    "location",
)


def _radiology_finding_tuple(item: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the deterministic evaluation key for one radiology finding."""
    return tuple(item.get(field) for field in _RADIOLOGY_FINDING_TUPLE_FIELDS)


def radiology_finding_tuple_f1(
    predicted: Iterable[Mapping[str, Any]],
    gold: Iterable[Mapping[str, Any]],
) -> F1Metrics:
    """Compute exact tuple precision, recall, and F1 for radiology findings.

    A match requires equality of ``(finding, laterality, size_value,
    size_unit, location)``. ``radlex_code`` and provenance are deliberately
    excluded: callers may use different caller-supplied terminology mappings
    and source offsets while agreeing on the extracted clinical attributes.
    Duplicate tuples are counted with multiset semantics.
    """
    predicted_counts: dict[tuple[Any, ...], int] = defaultdict(int)
    gold_counts: dict[tuple[Any, ...], int] = defaultdict(int)
    for item in predicted:
        predicted_counts[_radiology_finding_tuple(item)] += 1
    for item in gold:
        gold_counts[_radiology_finding_tuple(item)] += 1
    true_positives = sum(
        min(count, gold_counts.get(key, 0)) for key, count in predicted_counts.items()
    )
    return _f1_from_counts(
        true_positives,
        sum(predicted_counts.values()),
        sum(gold_counts.values()),
    )


TNM_FIELDS: tuple[str, ...] = (
    "basis",
    "t",
    "t_subcategory",
    "n",
    "n_subcategory",
    "m",
    "m_subcategory",
)


def tnm_field_accuracy(
    predicted: Iterable[Mapping[str, Any]],
    gold: Iterable[Mapping[str, Any]],
    *,
    fields: Sequence[str] = TNM_FIELDS,
) -> dict[str, RateMetric]:
    """Per-field accuracy for parsed TNM stages against gold.

    ``predicted`` and ``gold`` are parallel iterables of stage mappings, each
    carrying ``basis``/``t``/``n``/``m`` and the ``*_subcategory`` fields. For
    every field the accuracy is the fraction of rows whose predicted value
    equals the gold value; ``None == None`` counts as correct so a genuinely
    absent category is not penalized. Parser-agnostic: callers pass already-
    parsed stage mappings.
    """
    correct = {field: 0 for field in fields}
    total = 0
    for predicted_row, gold_row in zip(predicted, gold, strict=True):
        total += 1
        for field in fields:
            if predicted_row.get(field) == gold_row.get(field):
                correct[field] += 1
    return {
        field: RateMetric(
            rate=_safe_rate(correct[field], total, zero_denominator=1.0),
            numerator=correct[field],
            denominator=total,
        )
        for field in fields
    }


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


def _first_present_value(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _fixture_id_for_span(span: EvalSpan) -> str:
    for key in ("fixture_id", "fixture", "source_fixture_id", "id"):
        value = span.metadata.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return "unknown"


def _overlap_len(a: EvalSpan, b: EvalSpan) -> int:
    if not _label_aware_overlap(a, b):
        return 0
    return max(min(a.end, b.end) - max(a.start, b.start), 0)


def _covered_char_count(gold_span: EvalSpan, predicted: Sequence[EvalSpan]) -> int:
    """Return fully covered graphemes (legacy private helper name)."""
    return _grapheme_coverage_tally(gold_span, predicted).matched


def _grapheme_count(gold_span: EvalSpan) -> int:
    return len(_grapheme_units(gold_span))


def _grapheme_coverage_tally(
    gold_span: EvalSpan,
    predicted: Sequence[EvalSpan],
) -> _GraphemeTally:
    intervals: list[tuple[int, int]] = []
    for pred_span in predicted:
        if not _label_aware_overlap(gold_span, pred_span):
            continue
        start = max(gold_span.start, pred_span.start)
        end = min(gold_span.end, pred_span.end)
        if start < end:
            intervals.append((start, end))
    merged = _merge_intervals(intervals)
    units = _grapheme_units(gold_span)
    total_by_script: defaultdict[str, int] = defaultdict(int)
    covered_by_script: defaultdict[str, int] = defaultdict(int)
    covered = 0
    for start, end, script in units:
        total_by_script[script] += 1
        if any(left <= start and end <= right for left, right in merged):
            covered += 1
            covered_by_script[script] += 1
    return _GraphemeTally(
        matched=covered,
        total=len(units),
        matched_by_script=dict(covered_by_script),
        total_by_script=dict(total_by_script),
    )


def _grapheme_overlap_tally(
    gold_span: EvalSpan,
    intervals: Sequence[tuple[int, int]],
) -> _GraphemeTally:
    merged = _merge_intervals(intervals)
    units = _grapheme_units(gold_span)
    total_by_script: defaultdict[str, int] = defaultdict(int)
    leaked_by_script: defaultdict[str, int] = defaultdict(int)
    leaked = 0
    for start, end, script in units:
        total_by_script[script] += 1
        if any(max(start, left) < min(end, right) for left, right in merged):
            leaked += 1
            leaked_by_script[script] += 1
    return _GraphemeTally(
        matched=leaked,
        total=len(units),
        matched_by_script=dict(leaked_by_script),
        total_by_script=dict(total_by_script),
    )


def _invert_grapheme_tally(tally: _GraphemeTally) -> _GraphemeTally:
    scripts = set(tally.total_by_script) | set(tally.matched_by_script)
    return _GraphemeTally(
        matched=max(tally.total - tally.matched, 0),
        total=tally.total,
        matched_by_script={
            script: max(
                int(tally.total_by_script.get(script, 0))
                - int(tally.matched_by_script.get(script, 0)),
                0,
            )
            for script in scripts
        },
        total_by_script=dict(tally.total_by_script),
    )


def _grapheme_units(gold_span: EvalSpan) -> list[tuple[int, int, str]]:
    surface = gold_span.text
    if not surface or len(surface) != gold_span.length:
        return [
            (offset, offset + 1, UNKNOWN_SCRIPT)
            for offset in range(gold_span.start, gold_span.end)
        ]

    script_runs = tuple(segment_by_script(surface))
    units: list[tuple[int, int, str]] = []
    for local_start, local_end in iter_grapheme_cluster_spans(surface):
        script = _script_for_grapheme(local_start, local_end, script_runs)
        units.append(
            (
                gold_span.start + local_start,
                gold_span.start + local_end,
                script,
            )
        )
    return units


def _script_for_grapheme(
    start: int,
    end: int,
    script_runs: Sequence[tuple[int, int, str]],
) -> str:
    overlaps = [
        (max(0, min(end, run_end) - max(start, run_start)), -run_start, script)
        for run_start, run_end, script in script_runs
        if max(start, run_start) < min(end, run_end)
    ]
    if not overlaps:
        return UNKNOWN_SCRIPT
    return max(overlaps)[2]


_OFFSET_START_KEYS = (
    "start",
    "span_start",
    "offset_start",
    "start_offset",
    "char_start",
    "text_start",
    "source_start",
    "begin",
)
_OFFSET_END_KEYS = (
    "end",
    "span_end",
    "offset_end",
    "end_offset",
    "char_end",
    "text_end",
    "source_end",
    "stop",
)
_OFFSET_PAIR_KEYS = (
    "offset",
    "offsets",
    "span",
    "source_span",
    "evidence_span",
    "char_offsets",
    "text_offsets",
)
_OFFSET_SCALAR_KEYS = (*_OFFSET_START_KEYS, *_OFFSET_END_KEYS)


def _surface_index(gold: Sequence[EvalSpan]) -> list[tuple[int, str]]:
    surfaces: list[tuple[int, str]] = []
    for index, span in enumerate(gold):
        surface = span.text.strip().casefold()
        if surface:
            surfaces.append((index, surface))
    return surfaces


def _iter_extraction_text_values(
    value: Any,
    *,
    field_name: str | None = None,
) -> Iterable[str]:
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, str):
        if field_name in _OFFSET_SCALAR_KEYS:
            return
        yield value
        return
    if isinstance(value, int):
        if field_name in _OFFSET_SCALAR_KEYS:
            return
        yield str(value)
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            nested_field = str(key).strip().casefold()
            yield from _iter_extraction_text_values(
                nested,
                field_name=nested_field,
            )
        return
    if isinstance(value, Sequence) and not isinstance(
        value,
        (bytes, bytearray, str),
    ):
        if field_name in _OFFSET_PAIR_KEYS and _span_pair_from_value(value) is not None:
            return
        for nested in value:
            yield from _iter_extraction_text_values(nested)


def _iter_extraction_offsets(value: Any) -> Iterable[tuple[int, int]]:
    if isinstance(value, Mapping):
        pair = _span_pair_from_mapping(value)
        if pair is not None:
            yield pair
        for key in _OFFSET_PAIR_KEYS:
            nested_pair = _span_pair_from_value(value.get(key))
            if nested_pair is not None:
                yield nested_pair
        for nested in value.values():
            yield from _iter_extraction_offsets(nested)
        return
    if isinstance(value, Sequence) and not isinstance(
        value,
        (bytes, bytearray, str),
    ):
        for nested in value:
            yield from _iter_extraction_offsets(nested)


def _span_pair_from_mapping(value: Mapping[str, Any]) -> tuple[int, int] | None:
    start = _first_int_for_keys(value, _OFFSET_START_KEYS)
    end = _first_int_for_keys(value, _OFFSET_END_KEYS)
    if start is None or end is None or end <= start:
        return None
    return start, end


def _span_pair_from_value(value: Any) -> tuple[int, int] | None:
    if isinstance(value, Mapping):
        return _span_pair_from_mapping(value)
    if isinstance(value, Sequence) and not isinstance(
        value,
        (bytes, bytearray, str),
    ):
        if len(value) < 2:
            return None
        start = _coerce_int(value[0])
        end = _coerce_int(value[1])
        if start is not None and end is not None and end > start:
            return start, end
    return None


def _first_int_for_keys(
    value: Mapping[str, Any],
    keys: Sequence[str],
) -> int | None:
    for key in keys:
        parsed = _coerce_int(value.get(key))
        if parsed is not None:
            return parsed
    return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _contains_surface(text: str, surface: str) -> bool:
    if not surface:
        return False
    start = text.find(surface)
    while start != -1:
        end = start + len(surface)
        if _surface_boundaries_match(text, start, end, surface):
            return True
        start = text.find(surface, start + 1)
    return False


def _surface_boundaries_match(
    text: str,
    start: int,
    end: int,
    surface: str,
) -> bool:
    if surface[0].isalnum() and start > 0 and text[start - 1].isalnum():
        return False
    if surface[-1].isalnum() and end < len(text) and text[end].isalnum():
        return False
    return True


def _merged_interval_length(intervals: Sequence[tuple[int, int]]) -> int:
    return sum(end - start for start, end in _merge_intervals(intervals))


def _merge_intervals(
    intervals: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            old_start, old_end = merged[-1]
            merged[-1] = (old_start, max(old_end, end))
    return merged


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


# ---------------------------------------------------------------------------
# Inter-annotator agreement
# ---------------------------------------------------------------------------

#: Category assigned to an item an annotator did not label.
_UNLABELED = "∅"  # empty-set symbol


@dataclass(frozen=True)
class InterAnnotatorAgreement:
    """Agreement across two or more annotators on an extraction gold set.

    ``cohen_kappa`` is populated for exactly two annotators; ``fleiss_kappa`` for
    three or more. ``disagreements`` carries offsets and labels only -- never raw
    clinical text.
    """

    n_annotators: int
    n_items: int
    cohen_kappa: float | None
    fleiss_kappa: float | None
    mean_span_f1: float
    per_label: Mapping[str, float]
    per_relation: Mapping[str, float]
    disagreements: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_annotators": self.n_annotators,
            "n_items": self.n_items,
            "cohen_kappa": self.cohen_kappa,
            "fleiss_kappa": self.fleiss_kappa,
            "mean_span_f1": self.mean_span_f1,
            "per_label": dict(self.per_label),
            "per_relation": dict(self.per_relation),
            "disagreements": [
                {"offset": list(item["offset"]), "labels": list(item["labels"])}
                for item in self.disagreements
            ],
        }


def _coerce_annotation(span: Any) -> tuple[int, int, str]:
    """Coerce a span (tuple or ``EvalSpan``) to ``(start, end, label)``."""

    if isinstance(span, EvalSpan):
        return span.start, span.end, span.label
    if isinstance(span, Mapping):
        return int(span["start"]), int(span["end"]), str(span["label"])
    start, end, label = span
    return int(start), int(end), str(label)


def _annotation_map(annotator: Iterable[Any]) -> dict[tuple[int, int], str]:
    return {(s, e): label for s, e, label in map(_coerce_annotation, annotator)}


def _aligned_categories(
    annotators: Sequence[Iterable[Any]],
) -> tuple[list[tuple[int, int]], list[list[str]]]:
    """Return the sorted item universe and the per-item annotator categories."""

    maps = [_annotation_map(annotator) for annotator in annotators]
    items = sorted({offset for mapping in maps for offset in mapping})
    rows = [[mapping.get(item, _UNLABELED) for mapping in maps] for item in items]
    return items, rows


def _kappa_from_agreement(observed: float, expected: float) -> float:
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def cohen_kappa_agreement(*annotators: Iterable[Any]) -> float:
    """Cohen kappa between exactly two annotators over offset-aligned labels."""

    if len(annotators) != 2:
        raise ValueError("cohen_kappa_agreement requires exactly two annotators")
    items, rows = _aligned_categories(annotators)
    if not items:
        return 1.0

    total = len(rows)
    observed = sum(1 for a, b in rows if a == b) / total
    first = defaultdict(float)
    second = defaultdict(float)
    for a, b in rows:
        first[a] += 1 / total
        second[b] += 1 / total
    expected = sum(first[c] * second[c] for c in set(first) | set(second))
    return _kappa_from_agreement(observed, expected)


def fleiss_kappa_agreement(annotators: Sequence[Iterable[Any]]) -> float:
    """Fleiss kappa across three or more annotators over offset-aligned labels."""

    if len(annotators) < 3:
        raise ValueError("fleiss_kappa_agreement requires at least three annotators")
    items, rows = _aligned_categories(annotators)
    if not items:
        return 1.0

    n_items = len(rows)
    n_raters = len(annotators)
    categories = {category for row in rows for category in row}

    agreement_sum = 0.0
    category_totals = defaultdict(int)
    for row in rows:
        counts = defaultdict(int)
        for category in row:
            counts[category] += 1
            category_totals[category] += 1
        agreement_sum += (
            sum(count * count for count in counts.values()) - n_raters
        ) / (n_raters * (n_raters - 1))

    mean_agreement = agreement_sum / n_items
    expected = sum((category_totals[c] / (n_items * n_raters)) ** 2 for c in categories)
    return _kappa_from_agreement(mean_agreement, expected)


def _mean_pairwise_span_f1(annotators: Sequence[Iterable[Any]]) -> float:
    coerced = [
        [
            EvalSpan(start=s, end=e, label=label)
            for s, e, label in map(_coerce_annotation, annotator)
        ]
        for annotator in annotators
    ]
    scores: list[float] = []
    for i in range(len(coerced)):
        for j in range(i + 1, len(coerced)):
            scores.append(compute_relaxed_span_f1(coerced[i], coerced[j]).f1)
    return sum(scores) / len(scores) if scores else 1.0


def _label_agreement(items: Sequence[tuple[int, int]], rows: Sequence[Sequence[str]]):
    """Per-label observed agreement over items involving that label."""

    involved = defaultdict(list)
    for item, row in zip(items, rows):
        agreed = all(category == row[0] for category in row)
        for label in set(row) - {_UNLABELED}:
            involved[label].append(1.0 if agreed else 0.0)
    return {
        label: sum(involved[label]) / len(involved[label]) for label in sorted(involved)
    }


def inter_annotator_agreement(
    annotators: Sequence[Iterable[Any]],
    *,
    relations: Sequence[Mapping[str, Iterable[Any]]] | None = None,
) -> InterAnnotatorAgreement:
    """Compute inter-annotator agreement over an extraction gold set.

    ``annotators`` is one span set per annotator (tuples ``(start, end, label)``
    or ``EvalSpan``). With two annotators Cohen kappa is reported; with three or
    more, Fleiss kappa. ``relations`` optionally supplies, per annotator, a
    mapping of relation type to its labeled spans for per-relation agreement.
    """

    if len(annotators) < 2:
        raise ValueError("inter_annotator_agreement requires at least two annotators")
    if relations is not None and len(relations) != len(annotators):
        raise ValueError("relations must contain one mapping per annotator")

    materialized = [tuple(annotator) for annotator in annotators]
    items, rows = _aligned_categories(materialized)
    n_annotators = len(materialized)

    cohen = cohen_kappa_agreement(*materialized) if n_annotators == 2 else None
    fleiss = fleiss_kappa_agreement(materialized) if n_annotators >= 3 else None

    disagreements = tuple(
        {"offset": item, "labels": tuple(sorted(set(row) - {_UNLABELED}))}
        for item, row in zip(items, rows)
        if any(category != row[0] for category in row)
    )

    per_relation: dict[str, float] = {}
    if relations is not None:
        relation_types = {rt for mapping in relations for rt in mapping}
        for relation_type in sorted(relation_types):
            _rel_items, rel_rows = _aligned_categories(
                [mapping.get(relation_type, []) for mapping in relations]
            )
            if not rel_rows:
                continue
            per_relation[relation_type] = sum(
                1 for row in rel_rows if all(c == row[0] for c in row)
            ) / len(rel_rows)

    return InterAnnotatorAgreement(
        n_annotators=n_annotators,
        n_items=len(items),
        cohen_kappa=cohen,
        fleiss_kappa=fleiss,
        mean_span_f1=_mean_pairwise_span_f1(materialized),
        per_label=_label_agreement(items, rows),
        per_relation=per_relation,
        disagreements=disagreements,
    )


__all__ = [
    "ABSTENTION_ROUTE_ACCEPT",
    "ABSTENTION_ROUTE_REDACT",
    "ABSTENTION_ROUTE_REVIEW",
    "CRITICAL_ABSTENTION_LABELS",
    "CRITICAL_FINDING_CATEGORIES",
    "CRITICAL_FINDING_CATEGORY_DIAGNOSIS",
    "CRITICAL_FINDING_CATEGORY_DRUG_ALLERGY",
    "CRITICAL_FINDING_CATEGORY_RESULT",
    "DEVICE_TIERS",
    "MIXED_SCRIPT_LEAKAGE_CEILING",
    "AbstentionDecision",
    "AbstentionMetrics",
    "CriticalFindingMiss",
    "CriticalFindingRecallMetrics",
    "EvalSpan",
    "RateMetric",
    "F1Metrics",
    "LeakageMetrics",
    "MixedScriptLeakageMetrics",
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
    "compute_extraction_reemission_leakage",
    "compute_leakage_rate",
    "compute_mixed_script_leakage",
    "compute_character_recall",
    "compute_critical_finding_recall",
    "compute_recall_slices",
    "compute_exact_span_f1",
    "section_boundary_accuracy",
    "stated_category_accuracy",
    "radiology_finding_tuple_f1",
    "TNM_FIELDS",
    "tnm_field_accuracy",
    "compute_relaxed_span_f1",
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
    "critical_finding_category",
    "normalize_critical_finding_category",
    "abstention_route",
    "apply_abstention_policy",
    "bootstrap_abstention_residual_risk",
    "relation_assertion_consistency",
    "InterAnnotatorAgreement",
    "cohen_kappa_agreement",
    "fleiss_kappa_agreement",
    "inter_annotator_agreement",
]
