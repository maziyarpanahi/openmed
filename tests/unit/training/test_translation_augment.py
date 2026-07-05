from __future__ import annotations

import socket
from pathlib import Path
from typing import Iterable

import pytest

from openmed.eval.golden.loader import load_golden_fixtures
from openmed.eval.metrics import compute_exact_span_f1
from openmed.training.synthetic import (
    DictionaryTranslator,
    SpanProjectionError,
    TranslationAugmentedExample,
    augment_span_annotated_examples,
    load_span_jsonl,
    normalize_span_annotations,
    span_corruption_count,
    validate_span_integrity,
    write_augmented_jsonl,
)

LOWRESOURCE_FIXTURE = Path("openmed/eval/golden/fixtures/lowresource_ner_gold.jsonl")


def _span(text: str, value: str, label: str) -> dict[str, object]:
    start = text.index(value)
    return {"end": start + len(value), "label": label, "start": start, "text": value}


def _seed_example() -> dict[str, object]:
    text = "Patient has diabetes and takes metformin."
    return {
        "gold_spans": [
            _span(text, "diabetes", "CONDITION"),
            _span(text, "metformin", "MEDICATION"),
        ],
        "id": "seed-en-001",
        "language": "en",
        "metadata": {"split": "train", "synthetic": True},
        "text": text,
    }


def _predict_with_training_lexicon(
    text: str,
    *,
    language: str,
    training_items: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    terms: list[tuple[str, str]] = []
    for item in training_items:
        if item["language"] != language:
            continue
        for span in item.get("gold_spans", []):
            if isinstance(span, dict):
                terms.append((str(span["text"]), str(span["label"])))

    predictions = []
    for term, label in sorted(set(terms), key=lambda row: len(row[0]), reverse=True):
        start = text.find(term)
        if start == -1:
            continue
        predictions.append(
            {"end": start + len(term), "label": label, "start": start, "text": term}
        )
    return predictions


def test_offline_dictionary_pipeline_emits_offset_correct_augmented_examples():
    augmented = augment_span_annotated_examples(
        [_seed_example()],
        include_backtranslations=True,
    )

    by_language = {example.language: example for example in augmented}
    assert {"hi", "te", "en"}.issubset(by_language)

    for example in augmented:
        validate_span_integrity(example.text, example.gold_spans)
        assert span_corruption_count(example.text, example.gold_spans) == 0
        item = example.to_training_item()
        assert item["labels"] == item["gold_spans"]
        assert item["synthetic_source"] == "translation_backtranslation"
        assert item["metadata"]["synthetic"] is True
        assert item["metadata"]["provenance"]["source_id"] == "seed-en-001"

    hi = by_language["hi"]
    assert hi.text == "रोगी है मधुमेह और लेता है मेटफॉर्मिन."
    assert [(span.text, span.start, span.end) for span in hi.gold_spans] == [
        ("मधुमेह", 8, 14),
        ("मेटफॉर्मिन", 26, 36),
    ]


def test_offline_backend_runs_without_network(monkeypatch):
    def fail_socket(*_args, **_kwargs):
        raise AssertionError("offline dictionary backend opened a socket")

    monkeypatch.setattr(socket, "socket", fail_socket)

    augmented = augment_span_annotated_examples(
        [_seed_example()],
        translator=DictionaryTranslator(),
        include_backtranslations=False,
    )

    assert {example.language for example in augmented} == {"hi", "te"}


def test_lowresource_gold_f1_improves_when_augmented_terms_are_added():
    fixtures = load_golden_fixtures(LOWRESOURCE_FIXTURE)
    seed = _seed_example()
    augmented = augment_span_annotated_examples(
        [seed],
        include_backtranslations=False,
    )
    augmented_items = [example.to_training_item() for example in augmented]

    for language in ("hi", "te"):
        fixture = next(item for item in fixtures if item.language == language)
        baseline_predictions = _predict_with_training_lexicon(
            fixture.text,
            language=language,
            training_items=[seed],
        )
        augmented_predictions = _predict_with_training_lexicon(
            fixture.text,
            language=language,
            training_items=[seed, *augmented_items],
        )

        baseline = compute_exact_span_f1(
            fixture.gold_spans,
            baseline_predictions,
            default_language=language,
            source_text=fixture.text,
        )
        augmented_score = compute_exact_span_f1(
            fixture.gold_spans,
            augmented_predictions,
            default_language=language,
            source_text=fixture.text,
        )

        assert baseline.f1 == 0.0
        assert augmented_score.f1 == 1.0
        assert augmented_score.f1 > baseline.f1
        assert augmented_score.f1 >= baseline.f1


def test_leakage_guard_skips_eval_derived_rows_and_marks_provenance():
    train_seed = _seed_example()
    eval_text = "Patient has fever and takes aspirin."
    eval_seed = {
        "gold_spans": [
            _span(eval_text, "fever", "CONDITION"),
            _span(eval_text, "aspirin", "MEDICATION"),
        ],
        "id": "eval-row",
        "language": "en",
        "metadata": {"split": "eval", "synthetic": True},
        "text": eval_text,
    }

    augmented = augment_span_annotated_examples(
        [train_seed, eval_seed],
        heldout_eval_ids={"eval-row"},
        heldout_eval_texts={eval_text},
        include_backtranslations=False,
    )

    assert augmented
    assert {example.source_id for example in augmented} == {"seed-en-001"}
    for example in augmented:
        item = example.to_training_item()
        assert item["metadata"]["synthetic"] is True
        assert item["metadata"]["synthetic_source"] == "translation_backtranslation"
        assert item["metadata"]["provenance"]["source_text_hash"]
        assert eval_text not in item["text"]


def test_unrecoverable_translated_entity_span_is_dropped():
    class InconsistentTranslator:
        def translate(self, text: str, source_lang: str, target_lang: str) -> str:
            del source_lang, target_lang
            if text == "diabetes":
                return "diabete"
            if text == "metformin":
                return "metformina"
            return text.replace("diabetes", "condition").replace("metformin", "therapy")

    augmented = augment_span_annotated_examples(
        [_seed_example()],
        translator=InconsistentTranslator(),
        target_languages=("it",),
        include_backtranslations=False,
    )

    assert augmented == ()


def test_jsonl_helpers_round_trip_augmented_examples(tmp_path):
    augmented = augment_span_annotated_examples(
        [_seed_example()],
        include_backtranslations=False,
    )
    path = tmp_path / "augmented.jsonl"

    write_augmented_jsonl(augmented, path)
    rows = load_span_jsonl(path)

    assert len(rows) == len(augmented)
    assert rows[0]["synthetic_source"] == "translation_backtranslation"
    spans = normalize_span_annotations(rows[0], source_text=str(rows[0]["text"]))
    assert spans


def test_source_span_text_mismatch_is_rejected():
    text = "Patient has diabetes."
    with pytest.raises(SpanProjectionError, match="source span text mismatch"):
        normalize_span_annotations(
            {
                "gold_spans": [
                    {"end": 12, "label": "CONDITION", "start": 8, "text": "fever"}
                ],
                "text": text,
            }
        )
