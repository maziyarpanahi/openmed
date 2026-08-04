"""Naamapadam-style span evaluation for the 11 optional Indic NER languages."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from openmed.core.audit import hash_text
from openmed.core.labels import LOCATION, ORGANIZATION, PERSON
from openmed.core.pii_i18n import INDIC_NER_LANGUAGES, INDIC_NER_MODEL_ENV
from openmed.eval.golden import GoldenFixture, load_golden_fixtures
from openmed.eval.metrics import (
    EvalSpan,
    F1Metrics,
    compute_exact_span_f1,
    normalize_eval_spans,
)
from openmed.ner.families.indic import (
    IndicNerWeightsUnavailable,
    load_indic_ner_adapter,
)

NAAMAPADAM = "naamapadam"
NAAMAPADAM_LANGUAGES: tuple[str, ...] = tuple(sorted(INDIC_NER_LANGUAGES))
NAAMAPADAM_LABELS: tuple[str, ...] = (LOCATION, ORGANIZATION, PERSON)
NAAMAPADAM_MINIMUM_RECALL: Mapping[str, float] = {
    language: 0.80 for language in NAAMAPADAM_LANGUAGES
}
NAAMAPADAM_BASELINE_SLICE = "baseline"
NAAMAPADAM_ROBUSTNESS_SLICES: tuple[str, ...] = (
    "code_mixing",
    "combining_marks",
    "punctuation_adjacency",
    "repeated_entities",
    "script_boundary",
)
NAAMAPADAM_MINIMUM_SLICE_RECALL = 0.80
NAAMAPADAM_MAX_SLICE_RECALL_DROP: Mapping[str, float] = {
    slice_name: 0.10 for slice_name in NAAMAPADAM_ROBUSTNESS_SLICES
}
NAAMAPADAM_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "golden" / "fixtures" / "naamapadam.jsonl"
)


class IndicPredictor(Protocol):
    """Minimal predictor contract used by the offline evaluation suite."""

    def predict(self, text: str) -> Sequence[Any]: ...


@dataclass(frozen=True)
class NaamapadamSliceMetrics:
    """Aggregate robustness metrics for one language and deterministic slice."""

    slice_name: str
    precision: float
    recall: float
    f1: float
    recall_delta: float
    true_positives: int
    false_positives: int
    false_negatives: int
    leaked_entities: int
    total_entities: int
    fixture_hashes: tuple[str, ...] = field(default_factory=tuple)
    failed_fixture_hashes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def gate_passed(self) -> bool:
        """Return whether this slice satisfies its release thresholds."""

        return (
            self.recall >= NAAMAPADAM_MINIMUM_SLICE_RECALL
            and self.recall_delta <= NAAMAPADAM_MAX_SLICE_RECALL_DROP[self.slice_name]
            and self.leaked_entities == 0
        )

    def to_dict(self) -> dict[str, Any]:
        """Return aggregate counts and hashes without source or entity text."""

        return {
            "f1": self.f1,
            "failure_summary": {
                "failed_fixture_hashes": list(self.failed_fixture_hashes),
                "false_negatives": self.false_negatives,
                "false_positives": self.false_positives,
                "leaked_entities": self.leaked_entities,
            },
            "fixture_count": len(self.fixture_hashes),
            "fixture_hashes": list(self.fixture_hashes),
            "gate_passed": self.gate_passed,
            "maximum_recall_drop": NAAMAPADAM_MAX_SLICE_RECALL_DROP[self.slice_name],
            "minimum_recall": NAAMAPADAM_MINIMUM_SLICE_RECALL,
            "precision": self.precision,
            "recall": self.recall,
            "recall_delta": self.recall_delta,
            "slice": self.slice_name,
            "total_entities": self.total_entities,
            "true_positives": self.true_positives,
        }


@dataclass(frozen=True)
class NaamapadamLanguageMetrics:
    """Aggregate span and leakage metrics for one language."""

    language: str
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    leaked_entities: int
    total_entities: int
    baseline_recall: float
    slices: Mapping[str, NaamapadamSliceMetrics] = field(default_factory=dict)
    fixture_hashes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def leakage_rate(self) -> float:
        """Return the fraction of gold entity surfaces surviving de-id."""

        if self.total_entities == 0:
            return 0.0
        return self.leaked_entities / self.total_entities

    def to_dict(self) -> dict[str, Any]:
        """Return aggregate-only metrics with hashes instead of source text."""

        return {
            "f1": self.f1,
            "baseline_recall": self.baseline_recall,
            "false_negatives": self.false_negatives,
            "false_positives": self.false_positives,
            "fixture_hashes": list(self.fixture_hashes),
            "language": self.language,
            "leakage_rate": self.leakage_rate,
            "leaked_entities": self.leaked_entities,
            "precision": self.precision,
            "recall": self.recall,
            "slices": {
                slice_name: self.slices[slice_name].to_dict()
                for slice_name in sorted(self.slices)
            },
            "total_entities": self.total_entities,
            "true_positives": self.true_positives,
        }


@dataclass(frozen=True)
class NaamapadamReport:
    """Aggregate-only result for the 11-language Naamapadam-style suite."""

    status: str
    skip_reason: str | None = None
    languages: Mapping[str, NaamapadamLanguageMetrics] = field(default_factory=dict)
    micro: F1Metrics | None = None
    gate_passed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic metrics without raw token or entity text."""

        return {
            "gate_passed": self.gate_passed,
            "languages": {
                language: self.languages[language].to_dict()
                for language in sorted(self.languages)
            },
            "micro": self.micro.to_dict() if self.micro is not None else None,
            "maximum_slice_recall_drop": dict(
                sorted(NAAMAPADAM_MAX_SLICE_RECALL_DROP.items())
            ),
            "minimum_recall": dict(sorted(NAAMAPADAM_MINIMUM_RECALL.items())),
            "minimum_slice_recall": NAAMAPADAM_MINIMUM_SLICE_RECALL,
            "robustness_slices": list(NAAMAPADAM_ROBUSTNESS_SLICES),
            "schema_version": "openmed.eval.naamapadam.v2",
            "skip_reason": self.skip_reason,
            "status": self.status,
            "suite": NAAMAPADAM,
        }


def load_naamapadam_fixtures(
    path: str | Path = NAAMAPADAM_FIXTURE_PATH,
) -> list[GoldenFixture]:
    """Load and validate the committed synthetic fixtures for all 11 languages."""

    fixtures = load_golden_fixtures(path)
    observed_languages = {fixture.language for fixture in fixtures}
    if observed_languages != set(NAAMAPADAM_LANGUAGES):
        raise ValueError(
            "Naamapadam fixtures must cover exactly the configured 11 languages"
        )
    for fixture in fixtures:
        labels = {span.label for span in fixture.gold_spans}
        if labels != set(NAAMAPADAM_LABELS):
            raise ValueError(
                f"Naamapadam fixture {fixture.fixture_id!r} must cover PER/LOC/ORG"
            )
        _validate_robustness_fixture(fixture)
    for language in NAAMAPADAM_LANGUAGES:
        observed_slices = {
            slice_name
            for fixture in fixtures
            if fixture.language == language
            for slice_name in _fixture_slices(fixture)
        }
        expected_slices = {
            NAAMAPADAM_BASELINE_SLICE,
            *NAAMAPADAM_ROBUSTNESS_SLICES,
        }
        if observed_slices != expected_slices:
            raise ValueError(
                f"Naamapadam fixtures for {language!r} must cover baseline and "
                "all robustness slices"
            )
    return fixtures


def naamapadam_suite_metadata() -> dict[str, Any]:
    """Return aggregate-only suite metadata and the documented recall gates."""

    return {
        "fixture_kind": "committed synthetic Naamapadam-style records",
        "labels": list(NAAMAPADAM_LABELS),
        "languages": list(NAAMAPADAM_LANGUAGES),
        "minimum_recall": dict(sorted(NAAMAPADAM_MINIMUM_RECALL.items())),
        "minimum_slice_recall": NAAMAPADAM_MINIMUM_SLICE_RECALL,
        "maximum_slice_recall_drop": dict(
            sorted(NAAMAPADAM_MAX_SLICE_RECALL_DROP.items())
        ),
        "model_config": INDIC_NER_MODEL_ENV,
        "redistribution": "no Naamapadam corpus rows or model weights are bundled",
        "robustness_slices": list(NAAMAPADAM_ROBUSTNESS_SLICES),
        "source_url": "https://huggingface.co/datasets/ai4bharat/naamapadam",
        "suite": NAAMAPADAM,
    }


def run_naamapadam(
    predictor: IndicPredictor | None = None,
    *,
    model_path: str | None = None,
    fixture_path: str | Path = NAAMAPADAM_FIXTURE_PATH,
) -> NaamapadamReport:
    """Run per-language exact-span micro-F1 and zero-leakage gates.

    Each language has a documented minimum recall gate of 0.80. Every
    robustness slice also allows at most a 0.10 recall drop from that
    language's baseline. When neither ``predictor`` nor explicitly configured
    weights are available, the suite returns a structured skip result instead
    of downloading a default model.
    """

    if predictor is None:
        try:
            predictor = load_indic_ner_adapter(model_path)
        except IndicNerWeightsUnavailable as exc:
            return NaamapadamReport(status="skipped", skip_reason=str(exc))

    fixtures = load_naamapadam_fixtures(fixture_path)
    predictions_by_fixture: dict[str, list[EvalSpan]] = {}
    for fixture in fixtures:
        predictions_by_fixture[fixture.fixture_id] = normalize_eval_spans(
            predictor.predict(fixture.text),
            default_language=fixture.language,
        )

    language_metrics: dict[str, NaamapadamLanguageMetrics] = {}
    for language in NAAMAPADAM_LANGUAGES:
        rows = [fixture for fixture in fixtures if fixture.language == language]
        gold = [span for fixture in rows for span in fixture.gold_spans]
        score = _score_rows(rows, predictions_by_fixture)
        leaked_entities = sum(
            _surviving_gold_entities(
                fixture,
                predictions_by_fixture[fixture.fixture_id],
            )
            for fixture in rows
        )
        baseline_rows = [
            fixture
            for fixture in rows
            if NAAMAPADAM_BASELINE_SLICE in _fixture_slices(fixture)
        ]
        baseline_score = _score_rows(
            baseline_rows,
            predictions_by_fixture,
        )
        slice_metrics = {
            slice_name: _slice_metrics(
                slice_name,
                [fixture for fixture in rows if slice_name in _fixture_slices(fixture)],
                predictions_by_fixture,
                baseline_recall=baseline_score.recall,
            )
            for slice_name in NAAMAPADAM_ROBUSTNESS_SLICES
        }
        language_metrics[language] = _language_metrics(
            language,
            score,
            leaked_entities=leaked_entities,
            total_entities=len(gold),
            baseline_recall=baseline_score.recall,
            slices=slice_metrics,
            fixture_hashes=tuple(hash_text(fixture.text) for fixture in rows),
        )
    micro = _score_rows(fixtures, predictions_by_fixture)
    gate_passed = all(
        row.recall >= NAAMAPADAM_MINIMUM_RECALL[language]
        and row.baseline_recall >= NAAMAPADAM_MINIMUM_RECALL[language]
        and row.leaked_entities == 0
        and all(slice_row.gate_passed for slice_row in row.slices.values())
        for language, row in language_metrics.items()
    )
    return NaamapadamReport(
        status="completed",
        languages=language_metrics,
        micro=micro,
        gate_passed=gate_passed,
    )


def _language_metrics(
    language: str,
    score: F1Metrics,
    *,
    leaked_entities: int,
    total_entities: int,
    baseline_recall: float,
    slices: Mapping[str, NaamapadamSliceMetrics],
    fixture_hashes: tuple[str, ...],
) -> NaamapadamLanguageMetrics:
    return NaamapadamLanguageMetrics(
        language=language,
        precision=score.precision,
        recall=score.recall,
        f1=score.f1,
        true_positives=score.true_positives,
        false_positives=score.false_positives,
        false_negatives=score.false_negatives,
        leaked_entities=leaked_entities,
        total_entities=total_entities,
        baseline_recall=baseline_recall,
        slices=slices,
        fixture_hashes=fixture_hashes,
    )


def _slice_metrics(
    slice_name: str,
    fixtures: Sequence[GoldenFixture],
    predictions_by_fixture: Mapping[str, Sequence[EvalSpan]],
    *,
    baseline_recall: float,
) -> NaamapadamSliceMetrics:
    score = _score_rows(fixtures, predictions_by_fixture)
    leaked_entities = sum(
        _surviving_gold_entities(
            fixture,
            predictions_by_fixture[fixture.fixture_id],
        )
        for fixture in fixtures
    )
    failed_fixture_hashes = tuple(
        hash_text(fixture.text)
        for fixture in fixtures
        if _fixture_failed(
            fixture,
            predictions_by_fixture[fixture.fixture_id],
        )
    )
    return NaamapadamSliceMetrics(
        slice_name=slice_name,
        precision=score.precision,
        recall=score.recall,
        f1=score.f1,
        recall_delta=baseline_recall - score.recall,
        true_positives=score.true_positives,
        false_positives=score.false_positives,
        false_negatives=score.false_negatives,
        leaked_entities=leaked_entities,
        total_entities=sum(len(fixture.gold_spans) for fixture in fixtures),
        fixture_hashes=tuple(hash_text(fixture.text) for fixture in fixtures),
        failed_fixture_hashes=failed_fixture_hashes,
    )


def _score_rows(
    fixtures: Sequence[GoldenFixture],
    predictions_by_fixture: Mapping[str, Sequence[EvalSpan]],
) -> F1Metrics:
    scores = [
        compute_exact_span_f1(
            fixture.gold_spans,
            predictions_by_fixture[fixture.fixture_id],
            default_language=fixture.language,
        )
        for fixture in fixtures
    ]
    return _f1_from_counts(
        sum(score.true_positives for score in scores),
        sum(score.false_positives for score in scores),
        sum(score.false_negatives for score in scores),
    )


def _fixture_failed(
    fixture: GoldenFixture,
    predicted: Sequence[EvalSpan],
) -> bool:
    score = compute_exact_span_f1(
        fixture.gold_spans,
        predicted,
        default_language=fixture.language,
    )
    return bool(
        score.false_negatives
        or score.false_positives
        or _surviving_gold_entities(fixture, predicted)
    )


def _f1_from_counts(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> F1Metrics:
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = true_positives / precision_denominator if precision_denominator else 0.0
    recall = true_positives / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return F1Metrics(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def _fixture_slices(fixture: GoldenFixture) -> tuple[str, ...]:
    raw_slices = fixture.metadata.get("robustness_slices")
    if raw_slices is None:
        return (NAAMAPADAM_BASELINE_SLICE,)
    if isinstance(raw_slices, (str, bytes)) or not isinstance(raw_slices, Sequence):
        raise ValueError(
            f"Naamapadam fixture {fixture.fixture_id!r} robustness_slices must "
            "be a sequence"
        )
    slices = tuple(str(value) for value in raw_slices)
    allowed = {NAAMAPADAM_BASELINE_SLICE, *NAAMAPADAM_ROBUSTNESS_SLICES}
    if not slices or len(slices) != len(set(slices)) or set(slices) - allowed:
        raise ValueError(
            f"Naamapadam fixture {fixture.fixture_id!r} has invalid robustness slices"
        )
    if NAAMAPADAM_BASELINE_SLICE in slices and len(slices) != 1:
        raise ValueError("Naamapadam baseline fixtures cannot mix robustness slices")
    return slices


def _validate_robustness_fixture(fixture: GoldenFixture) -> None:
    slices = _fixture_slices(fixture)
    if NAAMAPADAM_BASELINE_SLICE in slices:
        return
    surfaces = [fixture.text[span.start : span.end] for span in fixture.gold_spans]
    if "combining_marks" in slices and not any(
        unicodedata.category(character).startswith("M")
        for surface in surfaces
        for character in surface
    ):
        raise ValueError(
            f"Naamapadam fixture {fixture.fixture_id!r} lacks a combining mark"
        )
    if "punctuation_adjacency" in slices and not any(
        _span_touches_punctuation(fixture.text, span) for span in fixture.gold_spans
    ):
        raise ValueError(
            f"Naamapadam fixture {fixture.fixture_id!r} lacks punctuation adjacency"
        )
    if "repeated_entities" in slices and max(Counter(surfaces).values()) < 2:
        raise ValueError(
            f"Naamapadam fixture {fixture.fixture_id!r} lacks a repeated surface"
        )
    if "code_mixing" in slices and not (
        re.search(r"[A-Za-z]", fixture.text)
        and any(ord(character) > 127 for character in fixture.text)
    ):
        raise ValueError(f"Naamapadam fixture {fixture.fixture_id!r} lacks code mixing")
    if "script_boundary" in slices and not any(
        _span_touches_latin(fixture.text, span) for span in fixture.gold_spans
    ):
        raise ValueError(
            f"Naamapadam fixture {fixture.fixture_id!r} lacks a script boundary"
        )


def _span_touches_punctuation(text: str, span: EvalSpan) -> bool:
    neighbors = (
        text[max(0, span.start - 1) : span.start] + text[span.end : span.end + 1]
    )
    return any(
        unicodedata.category(character).startswith("P") for character in neighbors
    )


def _span_touches_latin(text: str, span: EvalSpan) -> bool:
    neighbors = (
        text[max(0, span.start - 1) : span.start] + text[span.end : span.end + 1]
    )
    return bool(re.search(r"[A-Za-z]", neighbors))


def _surviving_gold_entities(
    fixture: GoldenFixture,
    predicted: Sequence[EvalSpan],
) -> int:
    deidentified = _mask_spans(fixture.text, predicted)
    return sum(
        fixture.text[span.start : span.end] in deidentified
        for span in fixture.gold_spans
    )


def _mask_spans(text: str, spans: Sequence[EvalSpan]) -> str:
    output = text
    for span in sorted(spans, key=lambda item: (item.start, item.end), reverse=True):
        if span.label not in NAAMAPADAM_LABELS:
            continue
        if 0 <= span.start < span.end <= len(text):
            output = output[: span.start] + f"[{span.label}]" + output[span.end :]
    return output


__all__ = [
    "NAAMAPADAM",
    "NAAMAPADAM_FIXTURE_PATH",
    "NAAMAPADAM_LABELS",
    "NAAMAPADAM_LANGUAGES",
    "NAAMAPADAM_MAX_SLICE_RECALL_DROP",
    "NAAMAPADAM_MINIMUM_RECALL",
    "NAAMAPADAM_MINIMUM_SLICE_RECALL",
    "NAAMAPADAM_ROBUSTNESS_SLICES",
    "NaamapadamLanguageMetrics",
    "NaamapadamReport",
    "NaamapadamSliceMetrics",
    "load_naamapadam_fixtures",
    "naamapadam_suite_metadata",
    "run_naamapadam",
]
