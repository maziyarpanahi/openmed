"""Offline synthetic recall-delta check for Indic encoder-backed NER."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from openmed.core.pii_i18n import get_patterns_for_language
from openmed.eval.metrics import EvalSpan, compute_leakage_rate, normalize_eval_spans
from openmed.ner.families.base import EncoderOutput, SupportsEncoding

_PATTERN_LABELS = {
    "date": "DATE",
    "phone_number": "PHONE",
    "national_id": "ID_NUM",
    "street_address": "STREET_ADDRESS",
    "postcode": "ZIPCODE",
}


@dataclass(frozen=True)
class SyntheticIndicFixture:
    """One synthetic Indic or code-mixed note with exact gold spans."""

    fixture_id: str
    language: str
    text: str
    gold_spans: tuple[EvalSpan, ...]


@dataclass(frozen=True)
class IndicRecallDeltaReport:
    """Entity recall and leakage comparison against the regex-only path."""

    gold_entities: int
    regex_matched_entities: int
    encoder_matched_entities: int
    regex_recall: float
    encoder_recall: float
    recall_delta: float
    entity_leakage: int
    encoder_leakage_rate: float

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-serializable report."""

        return {
            "gold_entities": self.gold_entities,
            "regex_matched_entities": self.regex_matched_entities,
            "encoder_matched_entities": self.encoder_matched_entities,
            "regex_recall": self.regex_recall,
            "encoder_recall": self.encoder_recall,
            "recall_delta": self.recall_delta,
            "entity_leakage": self.entity_leakage,
            "encoder_leakage_rate": self.encoder_leakage_rate,
        }


class IndicNerAdapter(Protocol):  # pragma: no cover - interface only
    """Adapter contract used by the deterministic Indic recall suite."""

    encoder: SupportsEncoding

    def predict_entities(
        self,
        text: str,
        *,
        language: str,
        encoding: EncoderOutput,
    ) -> Sequence[Any]: ...


def _fixture(
    fixture_id: str,
    language: str,
    text: str,
    entities: Sequence[tuple[str, str]],
) -> SyntheticIndicFixture:
    spans = tuple(
        EvalSpan(
            start=text.index(value),
            end=text.index(value) + len(value),
            label=label,
            text=value,
            language=language,
        )
        for value, label in entities
    )
    return SyntheticIndicFixture(fixture_id, language, text, spans)


SYNTHETIC_INDIC_GOLD: tuple[SyntheticIndicFixture, ...] = (
    _fixture(
        "hi-devanagari",
        "hi",
        "रोगी आरव मेहता का फ़ोन 9876543210 है।",
        (("आरव मेहता", "PERSON"), ("9876543210", "PHONE")),
    ),
    _fixture(
        "hi-hinglish",
        "hi",
        "Patient Asha Verma ka phone 9123456780 hai.",
        (("Asha Verma", "PERSON"), ("9123456780", "PHONE")),
    ),
    _fixture(
        "te-native",
        "te",
        "రోగి అనన్య రెడ్డి ఫోన్ 9988776655.",
        (("అనన్య రెడ్డి", "PERSON"), ("9988776655", "PHONE")),
    ),
)


def run_indic_encoder_recall_delta(
    adapter: IndicNerAdapter,
    *,
    fixtures: Sequence[SyntheticIndicFixture] = SYNTHETIC_INDIC_GOLD,
) -> IndicRecallDeltaReport:
    """Compare an encoder-backed adapter with deterministic regex detection."""

    gold_count = 0
    regex_matches = 0
    encoder_matches = 0
    encoder_leaked_chars = 0
    gold_chars = 0

    for fixture in fixtures:
        encoding = adapter.encoder.encode(fixture.text)
        encoding.validate()
        predicted = normalize_eval_spans(
            adapter.predict_entities(
                fixture.text,
                language=fixture.language,
                encoding=encoding,
            ),
            default_language=fixture.language,
            source_text=fixture.text,
        )
        regex_predicted = _regex_spans(fixture)
        gold_count += len(fixture.gold_spans)
        regex_matches += _matched_gold_count(fixture.gold_spans, regex_predicted)
        encoder_matches += _matched_gold_count(fixture.gold_spans, predicted)
        leakage = compute_leakage_rate(
            fixture.gold_spans,
            predicted,
            default_language=fixture.language,
            source_text=fixture.text,
        )
        encoder_leaked_chars += leakage.leaked_chars
        gold_chars += leakage.total_chars

    regex_recall = regex_matches / gold_count if gold_count else 1.0
    encoder_recall = encoder_matches / gold_count if gold_count else 1.0
    return IndicRecallDeltaReport(
        gold_entities=gold_count,
        regex_matched_entities=regex_matches,
        encoder_matched_entities=encoder_matches,
        regex_recall=regex_recall,
        encoder_recall=encoder_recall,
        recall_delta=encoder_recall - regex_recall,
        entity_leakage=gold_count - encoder_matches,
        encoder_leakage_rate=(encoder_leaked_chars / gold_chars if gold_chars else 0.0),
    )


def _regex_spans(fixture: SyntheticIndicFixture) -> list[EvalSpan]:
    observed: dict[tuple[int, int, str], EvalSpan] = {}
    for pattern in get_patterns_for_language(fixture.language):
        label = _PATTERN_LABELS.get(pattern.entity_type)
        if label is None:
            continue
        for match in re.finditer(pattern.pattern, fixture.text, pattern.flags):
            value = match.group(0)
            if pattern.validator is not None and not pattern.validator(value):
                continue
            key = (match.start(), match.end(), label)
            observed[key] = EvalSpan(
                start=match.start(),
                end=match.end(),
                label=label,
                text=value,
                language=fixture.language,
            )
    return list(observed.values())


def _matched_gold_count(
    gold_spans: Sequence[EvalSpan],
    predicted_spans: Sequence[EvalSpan],
) -> int:
    predicted = {
        (span.start, span.end, span.label.casefold()) for span in predicted_spans
    }
    return sum(
        (span.start, span.end, span.label.casefold()) in predicted
        for span in gold_spans
    )


__all__ = [
    "INDIC_ENCODER_RECALL_DELTA",
    "IndicNerAdapter",
    "IndicRecallDeltaReport",
    "SYNTHETIC_INDIC_GOLD",
    "SyntheticIndicFixture",
    "run_indic_encoder_recall_delta",
]

INDIC_ENCODER_RECALL_DELTA = "indic_encoder_recall_delta"
