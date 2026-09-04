"""Synthetic consistency evaluation for ConText assertion axes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from openmed.clinical.context import (
    CERTAINTY_VALUES,
    TEMPORALITY_VALUES,
    apply_context_rules,
    resolve_span_context,
)
from openmed.eval.golden import GoldenFixture, load_golden_fixtures
from openmed.eval.report import BenchmarkReport

TEMPORAL_CONSISTENCY = "temporal_consistency"
TEMPORAL_CONSISTENCY_SCHEMA_VERSION = 1
TEMPORAL_CONSISTENCY_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "golden"
    / "fixtures"
    / "temporal_consistency.jsonl"
)

SCORED_AXES: tuple[str, ...] = ("temporality", "uncertainty")
UNCERTAINTY_VALUES = CERTAINTY_VALUES
TEMPORAL_CONSISTENCY_AXES = SCORED_AXES


@dataclass(frozen=True)
class TemporalConsistencyFixture:
    """One golden clinical finding with a consistency-group expectation."""

    fixture_id: str
    group_id: str
    variant: str
    language: str
    text: str
    target_text: str
    target_start: int
    target_end: int
    expected_axes: Mapping[str, str]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    golden_fixture: GoldenFixture | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not self.fixture_id:
            raise ValueError("temporal consistency fixture id is required")
        if not self.group_id:
            raise ValueError("temporal consistency group id is required")
        if not self.variant:
            raise ValueError("temporal consistency fixture variant is required")
        if not self.text:
            raise ValueError("temporal consistency fixture text is required")
        if not self.target_text:
            raise ValueError("temporal consistency target text is required")
        if not 0 <= self.target_start < self.target_end <= len(self.text):
            raise ValueError("temporal consistency target offsets are invalid")
        if self.text[self.target_start : self.target_end] != self.target_text:
            raise ValueError("temporal consistency target offsets do not match text")
        if set(self.expected_axes) != set(SCORED_AXES):
            raise ValueError(
                "temporal consistency expected axes must be temporality and uncertainty"
            )
        object.__setattr__(
            self,
            "expected_axes",
            MappingProxyType(dict(self.expected_axes)),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def from_golden(cls, fixture: GoldenFixture) -> "TemporalConsistencyFixture":
        """Build a consistency fixture from the shared golden fixture schema."""

        if fixture.category != TEMPORAL_CONSISTENCY:
            raise ValueError(
                f"expected {TEMPORAL_CONSISTENCY!r} fixture, got {fixture.category!r}"
            )
        if fixture.metadata.get("contains_real_phi") is not False:
            raise ValueError(
                "temporal consistency fixtures must declare contains_real_phi=false"
            )
        if len(fixture.gold_spans) != 1:
            raise ValueError(
                "temporal consistency fixtures require exactly one target span"
            )

        expected_output = fixture.expected_output
        if expected_output.get("text") != fixture.text:
            raise ValueError(
                "temporal consistency expected_output.text must equal fixture text"
            )
        raw_expected = fixture.metadata.get("expected_axes")
        if not isinstance(raw_expected, Mapping):
            raw_expected = expected_output.get("axes")
        if not isinstance(raw_expected, Mapping):
            raise ValueError(
                "temporal consistency fixtures require expected_output.axes"
            )
        expected = _normalize_axes(raw_expected, fixture.fixture_id)

        group_id = str(
            fixture.metadata.get("consistency_group")
            or fixture.metadata.get("group_id")
            or ""
        ).strip()
        variant = str(fixture.metadata.get("variant") or "").strip()
        if not group_id or not variant:
            raise ValueError(
                "temporal consistency fixtures require consistency_group and variant"
            )
        if (
            fixture.metadata.get("trap") == "hypothetical_not_recent"
            and expected["temporality"] == "recent"
        ):
            raise ValueError(
                "hypothetical_not_recent fixtures must not expect recent temporality"
            )

        span = fixture.gold_spans[0]
        return cls(
            fixture_id=fixture.fixture_id,
            group_id=group_id,
            variant=variant,
            language=fixture.language,
            text=fixture.text,
            target_text=span.text,
            target_start=span.start,
            target_end=span.end,
            expected_axes=expected,
            metadata=fixture.metadata,
            golden_fixture=fixture,
        )

    @property
    def context_span(self) -> dict[str, Any]:
        """Return the offset-aware span consumed by the ConText resolver."""

        return {
            "context": self.text,
            "end": self.target_end,
            "start": self.target_start,
            "text": self.target_text,
        }


@dataclass(frozen=True)
class AxisAccuracy:
    """Accuracy counts for one assertion axis."""

    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        """Return the exact-match accuracy for this axis."""

        return self.correct / self.total if self.total else 1.0

    def to_dict(self) -> dict[str, int | float]:
        """Return JSON-ready counts and score."""

        return {
            "accuracy": self.accuracy,
            "correct": self.correct,
            "total": self.total,
        }


@dataclass(frozen=True)
class TemporalConsistencyResult:
    """Raw-text-free aggregate result for a temporal consistency run."""

    fixture_count: int
    group_count: int
    per_axis: Mapping[str, AxisAccuracy]
    consistency_score: float
    consistent_group_count: int
    group_scores: Mapping[str, Mapping[str, Any]]
    mismatches: tuple[Mapping[str, Any], ...] = ()

    def to_metrics(self) -> dict[str, Any]:
        """Return metrics suitable for :class:`BenchmarkReport`."""

        axis_metrics = {axis: self.per_axis[axis].to_dict() for axis in SCORED_AXES}
        per_axis_accuracy = {
            axis: float(self.per_axis[axis].accuracy) for axis in SCORED_AXES
        }
        return {
            "axis_metrics": axis_metrics,
            "consistency": {
                "consistent_group_count": self.consistent_group_count,
                "group_count": self.group_count,
                "groups": {
                    group_id: dict(self.group_scores[group_id])
                    for group_id in sorted(self.group_scores)
                },
                "mismatches": [dict(mismatch) for mismatch in self.mismatches],
                "score": self.consistency_score,
            },
            "consistency_score": self.consistency_score,
            "per_axis": axis_metrics,
            "per_axis_accuracy": per_axis_accuracy,
            "temporality_accuracy": per_axis_accuracy["temporality"],
            "uncertainty_accuracy": per_axis_accuracy["uncertainty"],
        }


PredictionResolver = Callable[[TemporalConsistencyFixture], Mapping[str, str] | Any]


def load_temporal_consistency_fixtures(
    path: str | Path | None = None,
) -> tuple[TemporalConsistencyFixture, ...]:
    """Load the committed synthetic temporal-consistency fixtures.

    The rows use the shared :class:`~openmed.eval.golden.GoldenFixture` format;
    this loader adds only the consistency-group contract needed by this suite.
    """

    fixture_path = Path(path) if path is not None else TEMPORAL_CONSISTENCY_FIXTURE_PATH
    golden = load_golden_fixtures(fixture_path)
    fixtures = tuple(TemporalConsistencyFixture.from_golden(row) for row in golden)
    if not fixtures:
        raise ValueError("temporal consistency fixture file must not be empty")

    fixture_ids = [fixture.fixture_id for fixture in fixtures]
    if len(fixture_ids) != len(set(fixture_ids)):
        raise ValueError("temporal consistency fixture ids must be unique")

    variants: set[tuple[str, str]] = set()
    group_counts: defaultdict[str, int] = defaultdict(int)
    for fixture in fixtures:
        key = (fixture.group_id, fixture.variant)
        if key in variants:
            raise ValueError(
                "temporal consistency group variants must be unique: "
                f"{fixture.group_id}/{fixture.variant}"
            )
        variants.add(key)
        group_counts[fixture.group_id] += 1
    if any(count < 2 for count in group_counts.values()):
        raise ValueError("temporal consistency groups require at least two variants")
    return fixtures


def evaluate_temporal_consistency(
    fixtures: Sequence[TemporalConsistencyFixture] | str | Path | None = None,
    *,
    predictions_by_fixture: Mapping[str, Any] | None = None,
    resolver: PredictionResolver | None = None,
) -> TemporalConsistencyResult:
    """Score ConText temporality and uncertainty across golden variants.

    ``predictions_by_fixture`` is an offline test hook for a caller-supplied
    model or a deliberately altered prediction. When it does not contain a
    fixture, the deterministic ConText resolver scores that row.
    """

    resolved = _coerce_fixtures(fixtures)
    if predictions_by_fixture is not None:
        unknown = set(predictions_by_fixture).difference(
            fixture.fixture_id for fixture in resolved
        )
        if unknown:
            raise ValueError(
                "predictions reference unknown fixture ids: "
                + ", ".join(sorted(unknown))
            )

    expected_by_group: dict[str, Mapping[str, str]] = {}
    predictions: dict[str, Mapping[str, str]] = {}
    for fixture in resolved:
        group_expected = expected_by_group.setdefault(
            fixture.group_id, fixture.expected_axes
        )
        if dict(group_expected) != dict(fixture.expected_axes):
            raise ValueError(
                "all variants in a temporal consistency group must share "
                f"expected axes: {fixture.group_id}"
            )
        if predictions_by_fixture and fixture.fixture_id in predictions_by_fixture:
            raw_prediction = predictions_by_fixture[fixture.fixture_id]
        elif resolver is not None:
            raw_prediction = resolver(fixture)
        else:
            raw_prediction = _resolve_fixture(fixture)
        predictions[fixture.fixture_id] = _normalize_axes(
            raw_prediction,
            fixture.fixture_id,
        )

    correct_by_axis = dict.fromkeys(SCORED_AXES, 0)
    mismatches: list[Mapping[str, Any]] = []
    grouped_predictions: defaultdict[str, list[tuple[str, Mapping[str, str]]]] = (
        defaultdict(list)
    )
    for fixture in resolved:
        predicted = predictions[fixture.fixture_id]
        grouped_predictions[fixture.group_id].append((fixture.fixture_id, predicted))
        for axis in SCORED_AXES:
            if predicted[axis] == fixture.expected_axes[axis]:
                correct_by_axis[axis] += 1
            else:
                mismatches.append(
                    {
                        "axis": axis,
                        "expected": fixture.expected_axes[axis],
                        "fixture_id": fixture.fixture_id,
                        "group_id": fixture.group_id,
                        "predicted": predicted[axis],
                        "variant": fixture.variant,
                    }
                )

    per_axis = {
        axis: AxisAccuracy(correct=correct_by_axis[axis], total=len(resolved))
        for axis in SCORED_AXES
    }
    group_scores: dict[str, Mapping[str, Any]] = {}
    consistent_group_count = 0
    group_score_values: list[float] = []
    for group_id in sorted(grouped_predictions):
        expected = expected_by_group[group_id]
        rows = grouped_predictions[group_id]
        total = len(rows) * len(SCORED_AXES)
        correct = sum(
            predicted[axis] == expected[axis]
            for _, predicted in rows
            for axis in SCORED_AXES
        )
        score = correct / total if total else 1.0
        passed = correct == total
        if passed:
            consistent_group_count += 1
        group_score_values.append(score)
        group_scores[group_id] = {
            "consistent": passed,
            "correct": correct,
            "score": score,
            "total": total,
            "variant_count": len(rows),
        }

    consistency_score = (
        sum(group_score_values) / len(group_score_values) if group_score_values else 1.0
    )
    return TemporalConsistencyResult(
        fixture_count=len(resolved),
        group_count=len(grouped_predictions),
        per_axis=per_axis,
        consistency_score=consistency_score,
        consistent_group_count=consistent_group_count,
        group_scores=group_scores,
        mismatches=tuple(mismatches),
    )


def run_temporal_consistency_suite(
    fixtures: Sequence[TemporalConsistencyFixture] | str | Path | None = None,
    *,
    path: str | Path | None = None,
    model_name: str = "deterministic-context",
    device: str = "local",
    predictions_by_fixture: Mapping[str, Any] | None = None,
    resolver: PredictionResolver | None = None,
    generated_at: str | None = None,
) -> BenchmarkReport:
    """Run the synthetic temporal-consistency suite as a benchmark report."""

    if fixtures is not None and path is not None:
        raise ValueError("pass fixtures or path, not both")
    source = path if path is not None else fixtures
    loaded = _coerce_fixtures(source)
    result = evaluate_temporal_consistency(
        loaded,
        predictions_by_fixture=predictions_by_fixture,
        resolver=resolver,
    )
    metadata = temporal_consistency_metadata()
    metadata["fixture_ids"] = [fixture.fixture_id for fixture in loaded]
    metadata["consistency_groups"] = sorted({fixture.group_id for fixture in loaded})
    return BenchmarkReport(
        suite=TEMPORAL_CONSISTENCY,
        model_name=model_name,
        device=device,
        fixture_count=result.fixture_count,
        metrics=result.to_metrics(),
        generated_at=generated_at,
        metadata=metadata,
    )


def temporal_consistency_metadata() -> dict[str, Any]:
    """Return provenance-safe metadata describing this suite."""

    return {
        "contains_real_phi": False,
        "fixture_format": "GoldenFixture",
        "schema_version": TEMPORAL_CONSISTENCY_SCHEMA_VERSION,
        "scored_axes": list(SCORED_AXES),
        "suite": TEMPORAL_CONSISTENCY,
        "synthetic": True,
    }


def _coerce_fixtures(
    fixtures: Sequence[TemporalConsistencyFixture] | str | Path | None,
) -> tuple[TemporalConsistencyFixture, ...]:
    if fixtures is None:
        return load_temporal_consistency_fixtures()
    if isinstance(fixtures, (str, Path)):
        return load_temporal_consistency_fixtures(fixtures)
    resolved: list[TemporalConsistencyFixture] = []
    for fixture in fixtures:
        if isinstance(fixture, TemporalConsistencyFixture):
            resolved.append(fixture)
        elif isinstance(fixture, GoldenFixture):
            resolved.append(TemporalConsistencyFixture.from_golden(fixture))
        elif isinstance(fixture, Mapping):
            resolved.append(
                TemporalConsistencyFixture.from_golden(
                    GoldenFixture.from_mapping(fixture)
                )
            )
        else:
            raise TypeError("temporal consistency fixtures must use golden mappings")
    if not resolved:
        raise ValueError("temporal consistency fixtures must not be empty")
    return tuple(resolved)


def _resolve_fixture(fixture: TemporalConsistencyFixture) -> Mapping[str, str]:
    span = fixture.context_span
    _, modifier_hits = apply_context_rules(fixture.text, [span])[0]
    context = resolve_span_context(
        span,
        modifier_hits,
        language=fixture.language,
    )
    return {
        "temporality": context.temporality,
        "uncertainty": context.certainty,
    }


def _normalize_axes(raw: Any, fixture_id: str) -> dict[str, str]:
    if isinstance(raw, Mapping):
        temporality = raw.get("temporality")
        uncertainty = raw.get("uncertainty", raw.get("certainty"))
    else:
        temporality = getattr(raw, "temporality", None)
        uncertainty = getattr(raw, "uncertainty", getattr(raw, "certainty", None))
    values = {
        "temporality": str(temporality or ""),
        "uncertainty": str(uncertainty or ""),
    }
    if values["temporality"] not in TEMPORALITY_VALUES:
        raise ValueError(
            f"fixture {fixture_id} has invalid temporality: {values['temporality']!r}"
        )
    if values["uncertainty"] not in UNCERTAINTY_VALUES:
        raise ValueError(
            f"fixture {fixture_id} has invalid uncertainty: {values['uncertainty']!r}"
        )
    return values


load_temporal_consistency_fixture = load_temporal_consistency_fixtures
run_temporal_consistency = run_temporal_consistency_suite
score_temporal_consistency = evaluate_temporal_consistency


__all__ = [
    "SCORED_AXES",
    "TEMPORAL_CONSISTENCY",
    "TEMPORAL_CONSISTENCY_AXES",
    "TEMPORAL_CONSISTENCY_FIXTURE_PATH",
    "TEMPORAL_CONSISTENCY_SCHEMA_VERSION",
    "UNCERTAINTY_VALUES",
    "AxisAccuracy",
    "TemporalConsistencyFixture",
    "TemporalConsistencyResult",
    "evaluate_temporal_consistency",
    "load_temporal_consistency_fixture",
    "load_temporal_consistency_fixtures",
    "run_temporal_consistency",
    "run_temporal_consistency_suite",
    "score_temporal_consistency",
    "temporal_consistency_metadata",
]
