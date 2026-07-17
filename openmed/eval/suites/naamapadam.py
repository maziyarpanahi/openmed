"""Naamapadam-style span evaluation for the 11 optional Indic NER languages."""

from __future__ import annotations

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
NAAMAPADAM_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "golden" / "fixtures" / "naamapadam.jsonl"
)


class IndicPredictor(Protocol):
    """Minimal predictor contract used by the offline evaluation suite."""

    def predict(self, text: str) -> Sequence[Any]: ...


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
            "false_negatives": self.false_negatives,
            "false_positives": self.false_positives,
            "fixture_hashes": list(self.fixture_hashes),
            "language": self.language,
            "leakage_rate": self.leakage_rate,
            "leaked_entities": self.leaked_entities,
            "precision": self.precision,
            "recall": self.recall,
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
            "minimum_recall": dict(sorted(NAAMAPADAM_MINIMUM_RECALL.items())),
            "schema_version": "openmed.eval.naamapadam.v1",
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
    return fixtures


def naamapadam_suite_metadata() -> dict[str, Any]:
    """Return aggregate-only suite metadata and the documented recall gates."""

    return {
        "fixture_kind": "committed synthetic Naamapadam-style records",
        "labels": list(NAAMAPADAM_LABELS),
        "languages": list(NAAMAPADAM_LANGUAGES),
        "minimum_recall": dict(sorted(NAAMAPADAM_MINIMUM_RECALL.items())),
        "model_config": INDIC_NER_MODEL_ENV,
        "redistribution": "no Naamapadam corpus rows or model weights are bundled",
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

    Each language has a documented minimum recall gate of 0.80. When neither
    ``predictor`` nor explicitly configured weights are available, the suite
    returns a structured skip result instead of downloading a default model.
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
    all_gold: list[EvalSpan] = []
    all_predictions: list[EvalSpan] = []
    for language in NAAMAPADAM_LANGUAGES:
        rows = [fixture for fixture in fixtures if fixture.language == language]
        gold = [span for fixture in rows for span in fixture.gold_spans]
        predicted = [
            span
            for fixture in rows
            for span in predictions_by_fixture[fixture.fixture_id]
        ]
        score = compute_exact_span_f1(
            gold,
            predicted,
            default_language=language,
        )
        leaked_entities = sum(
            _surviving_gold_entities(
                fixture,
                predictions_by_fixture[fixture.fixture_id],
            )
            for fixture in rows
        )
        language_metrics[language] = _language_metrics(
            language,
            score,
            leaked_entities=leaked_entities,
            total_entities=len(gold),
            fixture_hashes=tuple(hash_text(fixture.text) for fixture in rows),
        )
        all_gold.extend(gold)
        all_predictions.extend(predicted)

    micro = compute_exact_span_f1(all_gold, all_predictions)
    gate_passed = all(
        row.recall >= NAAMAPADAM_MINIMUM_RECALL[language] and row.leaked_entities == 0
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
        fixture_hashes=fixture_hashes,
    )


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
    "NAAMAPADAM_MINIMUM_RECALL",
    "NaamapadamLanguageMetrics",
    "NaamapadamReport",
    "load_naamapadam_fixtures",
    "naamapadam_suite_metadata",
    "run_naamapadam",
]
