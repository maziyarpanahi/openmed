"""Fairness metrics for de-identification leakage across surrogate groups."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from openmed.core.quality_gates import validate_entity_spans
from openmed.eval.golden import load_benchmark_fixtures
from openmed.eval.harness import (
    BenchmarkFixture,
    ModelRunner,
    default_model_runner,
)
from openmed.eval.metrics import (
    EvalSpan,
    compute_character_recall,
    compute_leakage_rate,
    normalize_eval_spans,
)
from openmed.eval.suites import GOLDEN, load_suite_fixtures, validate_suite_name

UNSPECIFIED_GROUP = "unspecified"

_GROUP_METADATA_KEYS = ("group", "demographic_group", "surrogate_group")


@dataclass(frozen=True)
class FairnessGroupMetrics:
    """Leakage and recall for one demographic surrogate group."""

    leakage_rate: float
    recall: float
    leaked_chars: int
    covered_chars: int
    total_chars: int
    span_count: int

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-ready mapping."""
        return {
            "leakage_rate": self.leakage_rate,
            "recall": self.recall,
            "leaked_chars": self.leaked_chars,
            "covered_chars": self.covered_chars,
            "total_chars": self.total_chars,
            "span_count": self.span_count,
        }

    def __getitem__(self, key: str) -> int | float:
        return self.to_dict()[key]


@dataclass(frozen=True)
class FairnessReport:
    """Fairness report over gold PHI groups for one model and suite."""

    suite: str
    model_name: str
    fixture_count: int
    per_group: dict[str, FairnessGroupMetrics]
    leakage_disparity: float
    worst_group_leakage: float
    worst_group: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready mapping."""
        return {
            "suite": self.suite,
            "model_name": self.model_name,
            "fixture_count": self.fixture_count,
            "per_group": {
                group: metrics.to_dict()
                for group, metrics in sorted(self.per_group.items())
            },
            "leakage_disparity": self.leakage_disparity,
            "worst_group_leakage": self.worst_group_leakage,
            "worst_group": self.worst_group,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


@dataclass
class _GroupCounts:
    leaked_chars: int = 0
    covered_chars: int = 0
    total_chars: int = 0
    span_count: int = 0


def fairness_report(
    model: str | ModelRunner,
    suite: str | Sequence[BenchmarkFixture | Mapping[str, Any]],
    *,
    runner: ModelRunner | None = None,
    device: str = "cpu",
    suite_kwargs: Mapping[str, Any] | None = None,
) -> FairnessReport:
    """Run a model and report leakage/recall by gold-span group.

    Args:
        model: Model identifier, or a runner callable with the harness
            ``ModelRunner`` signature.
        suite: Named suite such as ``"golden"``, or concrete benchmark
            fixtures/mappings.
        runner: Optional runner to use when ``model`` is a string identifier.
        device: Device tag passed to the model runner and prediction
            normalization.
        suite_kwargs: Optional keyword arguments for named suite loaders.

    Returns:
        A fairness report with per-group leakage, recall, leakage disparity,
        and worst-group leakage.
    """
    suite_name, fixtures = _load_fairness_fixtures(suite, suite_kwargs=suite_kwargs)
    model_name, model_runner = _resolve_model_runner(model, runner)
    counts: defaultdict[str, _GroupCounts] = defaultdict(_GroupCounts)

    for fixture in fixtures:
        predicted_spans = _predict_fixture(
            fixture,
            model_name=model_name,
            model_runner=model_runner,
            device=device,
        )
        for group, gold_spans in _gold_spans_by_group(fixture.gold_spans).items():
            leakage = compute_leakage_rate(
                gold_spans,
                predicted_spans,
                default_language=fixture.language,
                default_device=device,
                source_text=fixture.text,
            )
            recall = compute_character_recall(
                gold_spans,
                predicted_spans,
                default_language=fixture.language,
                default_device=device,
                source_text=fixture.text,
            )
            group_counts = counts[group]
            group_counts.leaked_chars += leakage.leaked_chars
            group_counts.covered_chars += int(recall.numerator)
            group_counts.total_chars += leakage.total_chars
            group_counts.span_count += len(gold_spans)

    per_group = {
        group: _group_metrics(group_counts)
        for group, group_counts in sorted(counts.items())
    }
    leakage_rates = [metrics.leakage_rate for metrics in per_group.values()]
    worst_group = _worst_group(per_group)
    worst_group_leakage = (
        per_group[worst_group].leakage_rate if worst_group is not None else 0.0
    )

    return FairnessReport(
        suite=suite_name,
        model_name=model_name,
        fixture_count=len(fixtures),
        per_group=per_group,
        leakage_disparity=(
            max(leakage_rates) - min(leakage_rates) if leakage_rates else 0.0
        ),
        worst_group_leakage=worst_group_leakage,
        worst_group=worst_group,
    )


def _load_fairness_fixtures(
    suite: str | Sequence[BenchmarkFixture | Mapping[str, Any]],
    *,
    suite_kwargs: Mapping[str, Any] | None,
) -> tuple[str, list[BenchmarkFixture]]:
    if not isinstance(suite, str):
        return "custom", [_coerce_fixture(item) for item in suite]

    suite_name = validate_suite_name(suite)
    kwargs = dict(suite_kwargs or {})
    if suite_name == GOLDEN:
        return suite_name, load_benchmark_fixtures(**kwargs)
    return suite_name, load_suite_fixtures(suite_name, **kwargs)


def _coerce_fixture(
    fixture: BenchmarkFixture | Mapping[str, Any],
) -> BenchmarkFixture:
    if isinstance(fixture, BenchmarkFixture):
        return fixture
    return BenchmarkFixture.from_mapping(fixture)


def _resolve_model_runner(
    model: str | ModelRunner,
    runner: ModelRunner | None,
) -> tuple[str, ModelRunner]:
    if runner is not None:
        return str(model), runner
    if not isinstance(model, str) and callable(model):
        name = getattr(model, "__name__", model.__class__.__name__)
        return str(name), model
    return str(model), default_model_runner


def _predict_fixture(
    fixture: BenchmarkFixture,
    *,
    model_name: str,
    model_runner: ModelRunner,
    device: str,
) -> tuple[EvalSpan, ...]:
    raw_predictions = list(model_runner(fixture, model_name, device))
    predicted_spans = tuple(
        normalize_eval_spans(
            raw_predictions,
            default_language=fixture.language,
            default_device=device,
            source_text=fixture.text,
        )
    )
    validate_entity_spans(
        [span.to_entity() for span in predicted_spans],
        fixture.text,
    )
    return predicted_spans


def _gold_spans_by_group(spans: Sequence[EvalSpan]) -> dict[str, list[EvalSpan]]:
    grouped: defaultdict[str, list[EvalSpan]] = defaultdict(list)
    for span in spans:
        grouped[_group_for_span(span)].append(span)
    return dict(grouped)


def _group_for_span(span: EvalSpan) -> str:
    for key in _GROUP_METADATA_KEYS:
        value = span.metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return UNSPECIFIED_GROUP


def _group_metrics(counts: _GroupCounts) -> FairnessGroupMetrics:
    leakage_rate = _safe_rate(counts.leaked_chars, counts.total_chars, 0.0)
    recall = _safe_rate(counts.covered_chars, counts.total_chars, 1.0)
    return FairnessGroupMetrics(
        leakage_rate=leakage_rate,
        recall=recall,
        leaked_chars=counts.leaked_chars,
        covered_chars=counts.covered_chars,
        total_chars=counts.total_chars,
        span_count=counts.span_count,
    )


def _worst_group(
    per_group: Mapping[str, FairnessGroupMetrics],
) -> str | None:
    if not per_group:
        return None
    return max(per_group, key=lambda group: per_group[group].leakage_rate)


def _safe_rate(numerator: int, denominator: int, zero_denominator: float) -> float:
    if denominator == 0:
        return zero_denominator
    return float(numerator) / float(denominator)


__all__ = [
    "UNSPECIFIED_GROUP",
    "FairnessGroupMetrics",
    "FairnessReport",
    "fairness_report",
]
