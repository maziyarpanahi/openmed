"""Tests for the clinical terminology guard around local translation."""

from __future__ import annotations

import json
import socket
from collections.abc import Callable

import pytest

from openmed.training.languages.translation_guard import (
    ProtectedKind,
    TranslationGuard,
    TranslationGuardConfig,
    TranslationGuardError,
    TranslationGuardReason,
    guard_round_trip_examples,
)
from openmed.training.synthetic.offset_projection import validate_span_integrity
from openmed.training.synthetic.translation_augment import DictionaryTranslator

Reason = TranslationGuardReason

# Synthetic note: no real patient, identifier, or clinical record.
NOTE = (
    "Patient [NAME] (MRN-0042817) denies chest pain. Takes metformin 500 mg BID. "
    "BP 128/82 mmHg, temp 37.5 °C. No fever. Left knee swelling, HbA1c 7.2%. "
    "History of asthma."
)
SENSITIVE_VALUES = (
    "[NAME]",
    "MRN-0042817",
    "metformin",
    "500 mg",
    "chest pain",
    "128/82",
    "37.5",
    "HbA1c",
    "asthma",
)
LEXICON = {
    ("en", "es"): (
        ("patient", "paciente"),
        ("denies", "niega"),
        ("takes", "toma"),
        ("no", "sin"),
        ("fever", "fiebre"),
        ("cough", "tos"),
        ("left", "izquierda"),
        ("knee", "rodilla"),
        ("swelling", "hinchazón"),
        ("history of", "antecedentes de"),
        ("asthma", "asma"),
    ),
    ("es", "en"): (
        ("paciente", "patient"),
        ("niega", "denies"),
        ("toma", "takes"),
        ("sin", "no"),
        ("fiebre", "fever"),
        ("tos", "cough"),
        ("izquierda", "left"),
        ("rodilla", "knee"),
        ("hinchazón", "swelling"),
        ("antecedentes de", "history of"),
        ("asma", "asthma"),
    ),
    ("en", "te"): (
        ("patient", "రోగి"),
        ("fever", "జ్వరం"),
        ("knee", "మోకాలు"),
    ),
    ("te", "en"): (
        ("రోగి", "patient"),
        ("జ్వరం", "fever"),
        ("మోకాలు", "knee"),
    ),
}
GLOSSARY = {("en", "es"): {"metformin": "metformina"}}


def _span(text: str, surface: str, label: str) -> dict[str, object]:
    start = text.index(surface)
    return {"start": start, "end": start + len(surface), "label": label}


def _note_spans() -> list[dict[str, object]]:
    return [
        _span(NOTE, "chest pain", "PROBLEM"),
        _span(NOTE, "metformin", "DRUG"),
        _span(NOTE, "asthma", "PROBLEM"),
    ]


class RecordingTranslator:
    """Deterministic local lexicon translator that records what it receives."""

    def __init__(
        self,
        rewrite: Callable[[str, str, str], str] | None = None,
    ) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self._backend = DictionaryTranslator(lexicon=LEXICON)
        self._rewrite = rewrite

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        self.calls.append((text, source_lang, target_lang))
        translated = self._backend.translate(text, source_lang, target_lang)
        if self._rewrite is None:
            return translated
        return self._rewrite(translated, source_lang, target_lang)


def _guard(
    rewrite: Callable[[str, str, str], str] | None = None,
    **config: object,
) -> tuple[TranslationGuard, RecordingTranslator]:
    translator = RecordingTranslator(rewrite)
    options = {"glossary": GLOSSARY, **config}
    return TranslationGuard(
        translator, config=TranslationGuardConfig(**options)
    ), translator


def _leaks(payload: str) -> list[str]:
    return [value for value in SENSITIVE_VALUES if value in payload]


def test_protect_freezes_each_invariant_kind() -> None:
    guard, _ = _guard()

    protected = guard.protect(
        NOTE, language="en", target_language="es", spans=_note_spans()
    )

    kinds = {item.kind for item in protected.items}
    assert kinds == {
        ProtectedKind.ANNOTATED_SPAN,
        ProtectedKind.PHI_PLACEHOLDER,
        ProtectedKind.IDENTIFIER,
        ProtectedKind.MEASUREMENT,
        ProtectedKind.ABBREVIATION,
    }
    surfaces = {item.surface: item.kind for item in protected.items}
    assert surfaces["[NAME]"] is ProtectedKind.PHI_PLACEHOLDER
    assert surfaces["MRN-0042817"] is ProtectedKind.IDENTIFIER
    assert surfaces["500 mg"] is ProtectedKind.MEASUREMENT
    assert surfaces["128/82 mmHg"] is ProtectedKind.MEASUREMENT
    assert surfaces["37.5 °C"] is ProtectedKind.MEASUREMENT
    assert surfaces["7.2%"] is ProtectedKind.MEASUREMENT
    assert surfaces["BID"] is ProtectedKind.ABBREVIATION
    assert surfaces["BP"] is ProtectedKind.ABBREVIATION
    assert surfaces["HbA1c"] is ProtectedKind.IDENTIFIER
    metformin = next(item for item in protected.items if item.surface == "metformin")
    assert metformin.replacement == "metformina"
    assert [item.index for item in protected.items] == list(range(len(protected.items)))
    assert [item.start for item in protected.items] == sorted(
        item.start for item in protected.items
    )
    assert not _leaks(protected.masked_text)
    assert "denies" in protected.masked_text
    assert "No fever" in protected.masked_text


def test_translator_never_receives_protected_surfaces() -> None:
    guard, translator = _guard()

    result = guard.round_trip(
        NOTE, source_language="en", pivot_language="es", spans=_note_spans()
    )

    assert result.accepted
    assert [call[1:] for call in translator.calls] == [("en", "es"), ("es", "en")]
    for text, _, _ in translator.calls:
        assert not _leaks(text)
        assert "metformina" not in text


def test_round_trip_accepts_and_preserves_span_map() -> None:
    guard, _ = _guard()

    result = guard.round_trip(
        NOTE, source_language="en", pivot_language="es", spans=_note_spans()
    )

    assert result.accepted, result.reasons
    assert result.reasons == ()
    assert result.pivot_text is not None and result.text is not None
    assert "niega" in result.pivot_text and "sin fiebre" in result.pivot_text
    assert "metformina" in result.pivot_text
    items = {item.index: item for item in result.forward.items}
    assert len(result.span_map) == len(items)
    for entry in result.span_map:
        item = items[entry.index]
        assert NOTE[entry.source_start : entry.source_end] == item.surface
        pivot_surface = result.pivot_text[entry.pivot_start : entry.pivot_end]
        assert pivot_surface == item.replacement
        assert result.text[entry.back_start : entry.back_end] == item.surface

    validate_span_integrity(result.text, result.spans)
    assert [(span.label, span.text) for span in result.spans] == [
        ("PROBLEM", "chest pain"),
        ("DRUG", "metformin"),
        ("PROBLEM", "asthma"),
    ]
    drug = result.spans[1]
    assert drug.metadata["projected_from_start"] == NOTE.index("metformin")
    pivot_drug = next(span for span in result.forward.spans if span.label == "DRUG")
    assert pivot_drug.text == "metformina"
    validate_span_integrity(result.pivot_text, result.forward.spans)


def test_one_way_translation_applies_glossary_and_flags_modifiers() -> None:
    guard, _ = _guard()

    result = guard.translate(
        NOTE, source_language="en", target_language="es", spans=_note_spans()
    )

    assert result.accepted
    assert result.text is not None
    assert Reason.MODIFIER_UNVERIFIABLE in result.warnings
    drug = next(span for span in result.spans if span.label == "DRUG")
    assert result.text[drug.start : drug.end] == "metformina"
    assert drug.metadata["projected_from_end"] == NOTE.index("metformin") + 9
    report = result.to_report()
    assert report["accepted"] is True
    assert report["warnings"] == ["modifier_unverifiable"]
    assert not _leaks(json.dumps(report, ensure_ascii=False))


def _drop_placeholder(text: str, *_: str) -> str:
    return text.replace("__OMP4__", "", 1)


def _duplicate_placeholder(text: str, *_: str) -> str:
    return text + " __OMP4__"


def _alter_placeholder(text: str, *_: str) -> str:
    return text.replace("__OMP4__", "__OMP 4__", 1)


def _zero_padded_placeholder(text: str, *_: str) -> str:
    return text.replace("__OMP4__", "__OMP04__", 1)


def _unknown_placeholder(text: str, *_: str) -> str:
    return text + " __OMP99__"


def _placeholder_residue(text: str, *_: str) -> str:
    return text + " omp-4"


def _inject_number(text: str, *_: str) -> str:
    return text + " x2"


def _drop_negation(text: str, source: str, target: str) -> str:
    if (source, target) == ("en", "es"):
        return text.replace("niega", "reporta", 1)
    return text


def _flip_laterality_back(text: str, source: str, target: str) -> str:
    if (source, target) == ("es", "en"):
        return text.replace("left", "right", 1)
    return text


def _drop_history_back(text: str, source: str, target: str) -> str:
    if (source, target) == ("es", "en"):
        return text.replace("history of", "with", 1)
    return text


def _empty(*_: str) -> str:
    return "   "


@pytest.mark.parametrize(
    ("rewrite", "expected", "stage"),
    [
        (_drop_placeholder, {Reason.PROTECTED_ITEM_DROPPED}, "forward"),
        (_duplicate_placeholder, {Reason.PROTECTED_ITEM_DUPLICATED}, "forward"),
        (_alter_placeholder, {Reason.PROTECTED_ITEM_DROPPED}, "forward"),
        (
            _zero_padded_placeholder,
            {Reason.PROTECTED_ITEM_DROPPED, Reason.UNKNOWN_PLACEHOLDER},
            "forward",
        ),
        (_unknown_placeholder, {Reason.UNKNOWN_PLACEHOLDER}, "forward"),
        (
            _placeholder_residue,
            {Reason.PLACEHOLDER_RESIDUE, Reason.NUMBER_CHANGED},
            "forward",
        ),
        (_inject_number, {Reason.NUMBER_CHANGED}, "forward"),
        (_drop_negation, {Reason.NEGATION_CHANGED}, "forward"),
        (_flip_laterality_back, {Reason.LATERALITY_CHANGED}, "round_trip"),
        (_drop_history_back, {Reason.HISTORICAL_CHANGED}, "round_trip"),
        (_empty, {Reason.EMPTY_TRANSLATION}, "forward"),
    ],
)
def test_negative_control_translators_are_rejected(
    rewrite: Callable[..., str],
    expected: set[TranslationGuardReason],
    stage: str,
) -> None:
    guard, _ = _guard(rewrite)

    result = guard.round_trip(
        NOTE, source_language="en", pivot_language="es", spans=_note_spans()
    )

    assert not result.accepted
    assert set(result.reasons) == expected
    assert result.text is None
    assert result.pivot_text is None
    assert result.spans == ()
    report = result.to_report()
    assert set(report["stages"][stage]["reasons"]) == {r.value for r in expected}
    assert not _leaks(json.dumps(report, ensure_ascii=False))


def test_dropped_placeholder_reports_the_item_offsets_only() -> None:
    guard, _ = _guard(_drop_placeholder)

    result = guard.translate(
        NOTE, source_language="en", target_language="es", spans=_note_spans()
    )

    assert result.reasons == (Reason.PROTECTED_ITEM_DROPPED,)
    assert result.problem_indices == (4,)
    item = result.to_report()["protected_items"][4]
    assert item == {
        "end": NOTE.index("500 mg") + len("500 mg"),
        "index": 4,
        "kind": "measurement",
        "label": None,
        "mapped": False,
        "start": NOTE.index("500 mg"),
    }


def test_translator_errors_fail_closed_without_echoing_the_message() -> None:
    def failing(text: str, source: str, target: str) -> str:
        raise RuntimeError(f"backend crashed on {NOTE}")

    guard = TranslationGuard(failing)

    result = guard.round_trip(NOTE, source_language="en", pivot_language="es")

    assert result.reasons == (Reason.TRANSLATOR_ERROR,)
    payload = json.dumps(result.to_report(), ensure_ascii=False) + repr(result)
    assert "backend crashed" not in payload
    assert not _leaks(payload)


def test_non_string_translator_output_is_rejected() -> None:
    guard = TranslationGuard(lambda text, source, target: None)

    result = guard.translate("No fever.", source_language="en", target_language="es")

    assert result.reasons == (Reason.TRANSLATOR_ERROR,)


def test_placeholder_collision_rejects_before_calling_the_translator() -> None:
    guard, translator = _guard()

    result = guard.translate(
        "No fever. Ref __OMP3__ pending.",
        source_language="en",
        target_language="es",
    )

    assert result.reasons == (Reason.PLACEHOLDER_COLLISION,)
    assert translator.calls == []


def test_round_trip_through_pivot_without_cue_pack_is_verified_at_the_end() -> None:
    guard, _ = _guard()

    result = guard.round_trip(
        "Patient denies fever. Left knee pain.",
        source_language="en",
        pivot_language="te",
    )

    assert result.accepted
    assert result.warnings == (Reason.ASSERTION_UNVERIFIABLE,)
    assert result.forward.warnings == (Reason.ASSERTION_UNVERIFIABLE,)
    assert result.text == "patient denies fever. Left knee pain."


def test_one_way_translation_without_cue_pack_fails_closed_by_default() -> None:
    strict, _ = _guard()
    lenient, _ = _guard(fail_on_unverifiable=False)

    rejected = strict.translate(
        "Patient denies fever.", source_language="en", target_language="te"
    )
    flagged = lenient.translate(
        "Patient denies fever.", source_language="en", target_language="te"
    )

    assert rejected.reasons == (Reason.ASSERTION_UNVERIFIABLE,)
    assert rejected.text is None
    assert flagged.accepted
    assert flagged.warnings == (Reason.ASSERTION_UNVERIFIABLE,)


def test_same_language_paraphrase_checks_assertion_axes() -> None:
    def paraphrase(text: str, source: str, target: str) -> str:
        return text.replace("not ruled out", "ruled out")

    guard = TranslationGuard(paraphrase)

    result = guard.translate(
        "Pneumonia not ruled out.", source_language="en", target_language="en"
    )

    assert set(result.reasons) == {
        Reason.NEGATION_CHANGED,
        Reason.UNCERTAINTY_CHANGED,
    }


def test_pseudo_negation_is_not_counted_as_negation() -> None:
    def paraphrase(text: str, source: str, target: str) -> str:
        return text.replace("No increase", "No interval increase")

    guard = TranslationGuard(paraphrase)

    result = guard.translate(
        "No increase in swelling.", source_language="en", target_language="en"
    )

    assert result.accepted


def test_unprotected_digits_are_still_verified() -> None:
    def alter_digits(text: str, source: str, target: str) -> str:
        return text.replace("128", "182")

    kinds = {ProtectedKind.ANNOTATED_SPAN, ProtectedKind.ABBREVIATION}
    guard = TranslationGuard(
        alter_digits, config=TranslationGuardConfig(protected_kinds=kinds)
    )

    protected = guard.protect("BP 128/82 today.", language="en")
    result = guard.translate(
        "BP 128/82 today.", source_language="en", target_language="es"
    )

    assert "128/82" in protected.masked_text
    assert result.reasons == (Reason.NUMBER_CHANGED,)


def test_protected_terms_and_lexicon_abbreviations_are_frozen() -> None:
    guard = TranslationGuard(
        lambda text, source, target: text,
        config=TranslationGuardConfig(protected_terms=("chest pain",)),
    )

    protected = guard.protect(
        "Chest pain resolved; b/p stable, aspirin bid and the cough.", language="en"
    )

    by_surface = {item.surface: item.kind for item in protected.items}
    assert by_surface == {
        "Chest pain": ProtectedKind.CLINICAL_TERM,
        "b/p": ProtectedKind.ABBREVIATION,
        "bid": ProtectedKind.ABBREVIATION,
    }


@pytest.mark.parametrize(
    ("language", "text", "expected"),
    [
        (
            "en",
            "5-year-old with 2 x 5 mg q8h prn, see 2024-03-01.",
            {
                "5": ProtectedKind.NUMBER,
                "2 x 5 mg": ProtectedKind.MEASUREMENT,
                "q8h": ProtectedKind.IDENTIFIER,
                "prn": ProtectedKind.ABBREVIATION,
                "2024-03-01": ProtectedKind.NUMBER,
            },
        ),
        (
            "es",
            "Paciente sin fiebre, TA 120/80 mmHg, fc 88.",
            {
                "TA": ProtectedKind.ABBREVIATION,
                "120/80 mmHg": ProtectedKind.MEASUREMENT,
                "fc": ProtectedKind.ABBREVIATION,
                "88": ProtectedKind.NUMBER,
            },
        ),
        (
            "zh",
            "患者否认发热，体温37.5°C，血压120/80毫米汞柱，MRN-77。",
            {
                "37.5°C": ProtectedKind.MEASUREMENT,
                "120/80毫米汞柱": ProtectedKind.MEASUREMENT,
                "MRN-77": ProtectedKind.IDENTIFIER,
            },
        ),
        (
            "hi",
            "रोगी को बुखार नहीं है, तापमान ३७.५ C, MRN-77।",
            {
                "३७.५ C": ProtectedKind.MEASUREMENT,
                "MRN-77": ProtectedKind.IDENTIFIER,
            },
        ),
    ],
)
def test_multilingual_detection_protects_ids_doses_and_numbers(
    language: str,
    text: str,
    expected: dict[str, ProtectedKind],
) -> None:
    guard = TranslationGuard(lambda value, source, target: value)

    protected = guard.protect(text, language=language)

    assert {item.surface: item.kind for item in protected.items} == expected


_ZH_EN_PAIRS = (
    ("患者", "patient "),
    ("否认", "denies "),
    ("发热", "fever"),
    ("，", ", "),
    ("体温", "temperature "),
    ("。", "."),
)


def _zh_en_translator(text: str, source: str, target: str) -> str:
    for chinese, english in _ZH_EN_PAIRS:
        if (source, target) == ("zh", "en"):
            text = text.replace(chinese, english)
        else:
            text = text.replace(english, chinese)
    return text


def test_unspaced_script_round_trip_restores_codes_and_units() -> None:
    text = "患者否认发热，体温37.5°C，MRN-77。"
    guard = TranslationGuard(_zh_en_translator)

    result = guard.round_trip(text, source_language="zh", pivot_language="en")

    assert result.accepted, result.reasons
    assert result.pivot_text == "patient denies fever, temperature 37.5°C, MRN-77."
    assert result.text == text
    assert result.warnings == (Reason.MODIFIER_UNVERIFIABLE,)


def test_invalid_spans_raise_without_echoing_text() -> None:
    guard, translator = _guard()
    overlapping = [
        {"start": 0, "end": 12, "label": "A"},
        {"start": 5, "end": 15, "label": "B"},
    ]

    with pytest.raises(TranslationGuardError) as error:
        guard.translate(
            NOTE, source_language="en", target_language="es", spans=overlapping
        )

    assert not _leaks(str(error.value))
    assert "Patient" not in str(error.value)
    assert translator.calls == []


def test_round_trip_requires_a_distinct_pivot_language() -> None:
    guard, _ = _guard()

    with pytest.raises(TranslationGuardError):
        guard.round_trip(NOTE, source_language="en", pivot_language="EN-us")


def _rewrite_chills_example(text: str, source: str, target: str) -> str:
    if "chills" in text.casefold():
        return text.replace("__OMP0__", "", 1)
    return text


def test_batch_gate_rejects_changed_examples_with_reason_codes() -> None:
    translator = RecordingTranslator(_rewrite_chills_example)
    chills = "Chills since 3 days, takes metformin 500 mg."
    examples = [
        {"id": "seed-ok", "text": NOTE, "language": "en", "gold_spans": _note_spans()},
        {
            "id": "seed-chills",
            "text": chills,
            "language": "en",
            "gold_spans": [_span(chills, "metformin", "DRUG")],
        },
        {
            "id": "seed-bad-span",
            "text": "No fever.",
            "gold_spans": [{"start": 0, "end": 40, "label": "PROBLEM"}],
        },
        {"id": "seed-spanish", "text": "Sin fiebre.", "language": "es"},
        "not-a-mapping",
    ]

    batch = guard_round_trip_examples(
        examples,
        translator=translator,
        pivot_language="es",
        config=TranslationGuardConfig(glossary=GLOSSARY),
    )

    report = batch.report
    assert (report.total, report.accepted, report.rejected) == (5, 1, 4)
    assert report.reason_counts == {
        "invalid_input": 2,
        "protected_item_dropped": 1,
        "same_language": 1,
    }
    assert [d.reasons for d in report.decisions] == [
        (),
        (Reason.PROTECTED_ITEM_DROPPED,),
        (Reason.INVALID_INPUT,),
        (Reason.SAME_LANGUAGE,),
        (Reason.INVALID_INPUT,),
    ]
    assert report.protected_counts["annotated_span"] == 4

    [example] = batch.accepted
    assert example.index == 0 and example.example_id == "seed-ok"
    item = example.to_training_item()
    assert item["source_id"] == "seed-ok"
    assert item["metadata"]["transform"] == "backtranslation:en->es->en"
    span_map = item["metadata"]["translation_guard"]["span_map"]
    assert len(span_map) == len(example.span_map)
    for span in item["gold_spans"]:
        assert item["text"][span["start"] : span["end"]] == span["text"]
    assert "metformina" in example.pivot_text

    payload = report.to_json()
    assert not _leaks(payload)
    assert "seed-ok" not in payload and "Chills" not in payload
    assert json.loads(payload)["decisions"][1]["reasons"] == ["protected_item_dropped"]


def test_reports_are_deterministic() -> None:
    def run() -> tuple[str, str]:
        translator = RecordingTranslator()
        batch = guard_round_trip_examples(
            [{"text": NOTE, "gold_spans": _note_spans()}],
            translator=translator,
            pivot_language="es",
            config=TranslationGuardConfig(glossary=GLOSSARY),
        )
        return batch.report.fingerprint, batch.accepted[0].text

    first = run()
    second = run()

    assert first == second
    assert first[0].startswith("sha256:")


def test_result_reprs_exclude_raw_text() -> None:
    guard, _ = _guard()

    result = guard.round_trip(
        NOTE, source_language="en", pivot_language="es", spans=_note_spans()
    )
    protected = guard.protect(NOTE, language="en", spans=_note_spans())

    assert result.accepted
    assert not _leaks(repr(result))
    assert not _leaks(repr(protected))


def test_guard_runs_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_socket(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("translation guard opened a socket")

    monkeypatch.setattr(socket, "socket", fail_socket)
    monkeypatch.setattr(socket, "create_connection", fail_socket)
    guard, _ = _guard()

    result = guard.round_trip(
        NOTE, source_language="en", pivot_language="es", spans=_note_spans()
    )

    assert result.accepted


@pytest.mark.parametrize(
    "options",
    [
        {"placeholder_template": "__OMP__"},
        {"placeholder_template": "{index}__"},
        {"placeholder_template": "__OMP{index}{index}__"},
        {"placeholder_template": "__OMP {index}__"},
        {"placeholder_template": "__OMP1{index}__"},
        {"assertion_axes": ("negation", "recent")},
        {"cross_lingual_assertion_axes": ("laterality",)},
        {"protected_kinds": {"dose"}},
        {"protected_terms": ("  ",)},
        {"protected_terms": "chest pain"},
        {"glossary": {"en": {"metformin": "metformina"}}},
        {"glossary": {("en", "es"): {"metformin": ""}}},
    ],
)
def test_invalid_configuration_is_rejected(options: dict[str, object]) -> None:
    with pytest.raises(TranslationGuardError):
        TranslationGuardConfig(**options)


def test_custom_placeholder_template_round_trips() -> None:
    translator = RecordingTranslator()
    guard = TranslationGuard(
        translator,
        config=TranslationGuardConfig(placeholder_template="<x{index}/>"),
    )

    result = guard.round_trip(NOTE, source_language="en", pivot_language="es")

    assert result.accepted
    assert "<x0/>" in translator.calls[0][0]


def test_translator_must_be_callable() -> None:
    with pytest.raises(TypeError):
        TranslationGuard(object())  # type: ignore[arg-type]
