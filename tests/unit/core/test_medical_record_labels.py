"""Localized medical-record-number labels in deterministic detection (#3862).

Every row in ``tests/fixtures/pii/medical_record_labels_synthetic.jsonl`` is a
synthetic note with labelled record numbers and negative controls (unlabelled
numbers, doses, lab values, phone numbers and postcodes). Each table language
must have a row, so adding a language needs one data entry and one fixture.
"""

from __future__ import annotations

import json
import re
from functools import partial
from pathlib import Path
from types import MappingProxyType

import pytest

from openmed.core import medical_record_labels as labels_module
from openmed.core.medical_record_labels import (
    MEDICAL_RECORD_ENTITY_TYPE,
    MEDICAL_RECORD_LABELS,
    MEDICAL_RECORD_LABELS_VERSION,
    MedicalRecordLabels,
    medical_record_label_coverage,
    medical_record_label_matches,
    medical_record_label_patterns,
    registered_label_languages,
)
from openmed.core.pii_entity_merger import (
    PIIPattern,
    iter_pattern_spans,
    merge_entities_with_semantic_units,
)
from openmed.core.pii_i18n import LANGUAGE_PII_PATTERNS, get_patterns_for_language
from openmed.core.safety_sweep import safety_sweep
from openmed.processing.outputs import PredictionResult

FIXTURE_PATH = Path("tests/fixtures/pii/medical_record_labels_synthetic.jsonl")

# Registered non-English packs that still rely on a model alone for record
# numbers. Shrink this list as table entries land; a newly registered pack
# must either add labels or be acknowledged here.
PACKS_WITHOUT_RECORD_LABELS = (
    "af",
    "as",
    "bg",
    "bn",
    "cs",
    "da",
    "de",
    "el",
    "es",
    "et",
    "fi",
    "fr",
    "gu",
    "hr",
    "hu",
    "ig",
    "it",
    "ja",
    "kn",
    "ko",
    "lv",
    "ml",
    "mr",
    "ms",
    "ne",
    "nl",
    "no",
    "or",
    "pa",
    "pl",
    "pt",
    "ro",
    "ru",
    "rw",
    "sk",
    "sr",
    "sv",
    "ta",
    "te",
    "tl",
    "tr",
    "uk",
    "xh",
    "yo",
    "zu",
)


def _rows() -> list[dict]:
    return [
        json.loads(line)
        for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


ROWS = _rows()


def _sweep_labels(text: str, lang: str) -> dict[tuple[int, int], str]:
    return {
        (entity.start, entity.end): entity.label
        for entity in safety_sweep(text, [], lang=lang)
    }


def _detector_labels(text: str, lang: str) -> dict[tuple[int, int], str]:
    merged = merge_entities_with_semantic_units(
        [],
        text,
        patterns=get_patterns_for_language(lang),
        use_semantic_patterns=True,
        prefer_model_labels=True,
        allow_semantic_only_matches=True,
        allow_label_expansion=True,
    )
    return {
        (entity["start"], entity["end"]): entity["entity_type"] for entity in merged
    }


def _record_spans(text: str, lang: str) -> set[tuple[int, int]]:
    return {
        span
        for span, label in _sweep_labels(text, lang).items()
        if label == MEDICAL_RECORD_ENTITY_TYPE
    }


def _empty_prediction(text, **kwargs):
    return PredictionResult(
        text=text, entities=[], model_name="synthetic", timestamp="2026-01-01"
    )


# ---------------------------------------------------------------------------
# Fixture contract
# ---------------------------------------------------------------------------


def test_fixture_rows_are_synthetic_with_exact_offsets():
    assert ROWS
    for row in ROWS:
        assert row["synthetic"] is True
        assert row["record_numbers"] or row["negative_controls"]
        entry = MEDICAL_RECORD_LABELS[row["language"]]
        for span in row["record_numbers"] + row["negative_controls"]:
            assert row["text"][span["start"] : span["end"]] == span["text"]
        for span in row["record_numbers"]:
            assert span["label"] in entry.all_labels
            # A unique surface lets masking assertions use plain substrings.
            assert row["text"].count(span["text"]) == 1


def test_every_table_language_has_a_fixture_with_labelled_and_unlabelled_numbers():
    languages = {row["language"] for row in ROWS}
    assert languages == set(MEDICAL_RECORD_LABELS)
    for language in MEDICAL_RECORD_LABELS:
        rows = [row for row in ROWS if row["language"] == language]
        assert any(row["record_numbers"] for row in rows), language
        kinds = {span["kind"] for row in rows for span in row["negative_controls"]}
        assert {"unlabelled", "dose", "lab_value", "phone"} <= kinds, language


# ---------------------------------------------------------------------------
# Positive recall: sweep, detector and report agree on the number's span
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("row", ROWS, ids=lambda row: row["id"])
def test_labelled_record_numbers_are_detected_with_exact_offsets(row):
    text, lang = row["text"], row["language"]
    swept = _sweep_labels(text, lang)
    detected = _detector_labels(text, lang)
    reported = {
        (match.start, match.end): match.label
        for match in medical_record_label_matches(text, lang)
    }

    for span in row["record_numbers"]:
        key = (span["start"], span["end"])
        assert swept.get(key) == MEDICAL_RECORD_ENTITY_TYPE, span
        assert detected.get(key) == MEDICAL_RECORD_ENTITY_TYPE, span
        assert reported.get(key) == span["label"], span


@pytest.mark.parametrize("row", ROWS, ids=lambda row: row["id"])
def test_negative_controls_are_never_relabelled_as_record_numbers(row):
    text, lang = row["text"], row["language"]
    observed = [
        *_sweep_labels(text, lang).items(),
        *_detector_labels(text, lang).items(),
    ]

    for span in row["negative_controls"]:
        overlapping = {
            label
            for (start, end), label in observed
            if start < span["end"] and end > span["start"]
        }
        assert MEDICAL_RECORD_ENTITY_TYPE not in overlapping, span["kind"]


@pytest.mark.parametrize("row", ROWS, ids=lambda row: row["id"])
def test_public_paths_mask_record_numbers_with_an_empty_model(monkeypatch, row):
    from openmed.core.pii import deidentify
    from openmed.core.pipeline import Pipeline

    text, lang = row["text"], row["language"]
    monkeypatch.setattr("openmed.analyze_text", _empty_prediction)
    direct = deidentify(text, lang=lang, model_name="unit-test-model")
    staged = Pipeline(
        lang=lang, model_name="unit-test-model", model_detector=_empty_prediction
    ).run(text)

    record_entities = {
        (entity.start, entity.end)
        for entity in direct.pii_entities
        if entity.entity_type == MEDICAL_RECORD_ENTITY_TYPE
    }
    for span in row["record_numbers"]:
        assert (span["start"], span["end"]) in record_entities
        assert span["text"] not in direct.deidentified_text
        assert span["text"] not in staged.redacted_text
    for span in row["negative_controls"]:
        if span["kind"] in {"unlabelled", "dose", "lab_value"}:
            assert span["text"] in direct.deidentified_text
            assert span["text"] in staged.redacted_text


# ---------------------------------------------------------------------------
# Issue examples and tolerated variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lang", "text", "value"),
    [
        ("th", "HN 6501234", "6501234"),
        ("th", "HN: 65-012345", "65-012345"),
        ("th", "ผู้ป่วยHN6501234 มาตรวจ", "6501234"),
        ("th", "H.N. 12345/65", "12345/65"),
        ("th", "HN ๖๕๐๑๒๓๔ มาตรวจ", "๖๕๐๑๒๓๔"),
        ("th", "HN# 6501234", "6501234"),
        ("id", "No. RM 123456", "123456"),
        ("id", "No. RM: 12-34-56", "12-34-56"),
        ("id", "No.RM 001234", "001234"),
        ("id", "no rm: 001234", "001234"),
        ("vi", "Mã bệnh nhân: 2345678", "2345678"),
        ("fa", "شماره پرونده: ۱۲۳۴۵۶۷", "۱۲۳۴۵۶۷"),
        ("ar", "رقم الملف الطبي: 1234567", "1234567"),
        ("ar", "رقم الملف الطبي: ١٢٣٤٥٦٧", "١٢٣٤٥٦٧"),
        ("ur", "ایم آر نمبر: 1234567", "1234567"),
        ("sw", "Namba ya faili: 1234567", "1234567"),
        ("am", "የካርድ ቁጥር፡ 1234567", "1234567"),
        ("hi", "पंजीकरण संख्या: 1234567", "1234567"),
        ("hi", "पंजीकरण संख्या: १२३४५६७", "१२३४५६७"),
        ("ha", "Lambar fayil: 1234567", "1234567"),
        ("he", "מספר תיק: 12345", "12345"),
        ("he", "מס׳ תיק 7654321", "7654321"),
        ("he", "במספר תיק 7654321 נרשם", "7654321"),
        ("zh", "病历号：20260018", "20260018"),
    ],
)
def test_issue_examples_are_masked_as_record_numbers(lang, text, value):
    start = text.index(value)
    end = start + len(value)

    assert _sweep_labels(text, lang).get((start, end)) == MEDICAL_RECORD_ENTITY_TYPE
    assert _detector_labels(text, lang).get((start, end)) == MEDICAL_RECORD_ENTITY_TYPE


def test_hebrew_file_number_is_no_longer_reported_as_postcode():
    text = "מספר תיק: 12345; מיקוד 61000"

    swept = _sweep_labels(text, "he")
    detected = _detector_labels(text, "he")

    assert swept[(10, 15)] == MEDICAL_RECORD_ENTITY_TYPE
    assert detected[(10, 15)] == MEDICAL_RECORD_ENTITY_TYPE
    postcode = (text.index("61000"), len(text))
    assert swept[postcode] == "postcode"
    assert detected[postcode] == "postcode"


@pytest.mark.parametrize(
    ("lang", "label", "separator"),
    [
        ("ar", "رقم الملف الطبي", " \u200f: "),
        ("ar", "رقم الملف الطبي", "\u061c: "),
        ("he", "מספר תיק", "\u2067: \u2069"),
        ("fa", "شماره پرونده", ":\u200e "),
        ("ur", "فائل نمبر", " : "),
        ("zh", "住院号", "："),
        ("am", "የካርድ ቁጥር", " ፡ "),
        ("th", "HN", " # "),
    ],
)
def test_label_punctuation_and_bidi_controls_are_tolerated(lang, label, separator):
    text = f"{label}{separator}7654321"

    assert _record_spans(text, lang) == {(len(text) - 7, len(text))}


@pytest.mark.parametrize(
    ("lang", "text"),
    [
        ("he", "7654321 :מספר תיק"),
        ("ar", "١٢٣٤٥٦٧ :رقم الملف الطبي"),
        ("fa", "۱۲۳۴۵۶۷ :شماره پرونده"),
        ("ur", "1234567 :ایم آر نمبر"),
    ],
)
def test_right_to_left_number_before_its_label_is_detected(lang, text):
    assert _record_spans(text, lang) == {(0, 7)}
    assert [match.start for match in medical_record_label_matches(text, lang)] == [0]


@pytest.mark.parametrize(
    ("lang", "text"),
    [
        ("fa", "كد پرونده: 1234567"),
        ("fa", "شماره بيمار: 1234567"),
        ("ur", "داخله نمبر: 1234567"),
        ("ur", "مريض نمبر: 1234567"),
    ],
)
def test_arabic_keyboard_variants_are_caught_by_the_sweep(lang, text):
    assert _record_spans(text, lang) == {(len(text) - 7, len(text))}


@pytest.mark.parametrize(
    "text",
    [
        "Mã bệnh nhân: 2345678",
        "Ma benh nhan: 2345678",
        "MÃ BỆNH NHÂN: 2345678",
    ],
)
def test_vietnamese_labels_match_nfc_nfd_and_unmarked_spellings(text):
    import unicodedata

    for variant in {text, unicodedata.normalize("NFD", text)}:
        assert _record_spans(variant, "vi") == {(len(variant) - 7, len(variant))}


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lang", "text"),
    [
        # Unlabelled numbers, doses, lab values and phones.
        ("th", "ให้ยา 1500 mg ครีเอตินีน 1.25 โทร 081-234-5678"),
        ("id", "Pasien kamar 3021, Hb 11,2 g/dL, telp 0812-3456-7890"),
        ("ar", "الهاتف 0501234567 والجرعة 1000 ملغ"),
        ("sw", "simu 0712345678; dozi 1000 mg"),
        # Short Thai abbreviations are case-sensitive: "an" is an English word.
        ("th", "ให้อาหาร an 1800 kcal ต่อวัน"),
        # Count words never act as record labels.
        ("hi", "कुल रोगी संख्या 1250"),
        ("vi", "Số bệnh nhân hôm nay: 1250"),
        ("he", "מספר מטופלים: 1250"),
        # A label followed by unrelated words does not reach a later number.
        ("he", "מספר תיק לא ידוע, טלפון 0501234567"),
        ("ar", "رقم الملف غير متوفر 1234567"),
        # Labels inside longer Latin words are not labels.
        ("id", "Panno RM 123456"),
        # Record-like shapes longer than twenty digits are not truncated.
        ("th", "HN 123456789012345678901"),
        # Too short to be a record number.
        ("th", "HN 123"),
    ],
)
def test_unlabelled_or_unrelated_numbers_are_not_record_numbers(lang, text):
    assert _record_spans(text, lang) == set()
    assert MEDICAL_RECORD_ENTITY_TYPE not in _detector_labels(text, lang).values()


def test_number_ending_a_clause_does_not_borrow_the_next_label():
    text = "טלפון 0501234567: מספר תיק 7654321"
    phone = (6, 16)
    record = (text.index("7654321"), len(text))

    swept = _sweep_labels(text, "he")

    assert swept[record] == MEDICAL_RECORD_ENTITY_TYPE
    assert swept[phone] == "phone_number"


def test_record_labels_are_not_applied_to_other_languages():
    assert _record_spans("HN 6501234 and No. RM 123456", "en") == set()
    assert medical_record_label_patterns("en") == []
    assert medical_record_label_matches("HN 6501234", "en") == ()


def test_split_english_mrn_sweep_keeps_its_confidence_in_labelled_packs():
    for lang in MEDICAL_RECORD_LABELS:
        entities = [
            entity
            for entity in safety_sweep("MRN 1,2,3,4,5,6", [], lang=lang)
            if entity.label == MEDICAL_RECORD_ENTITY_TYPE
        ]
        assert len(entities) == 1, lang
        assert entities[0].confidence >= 0.8, lang


# ---------------------------------------------------------------------------
# Table, compiled patterns and value-group spans
# ---------------------------------------------------------------------------


def test_compiled_patterns_are_context_gated_and_name_the_value():
    for language in MEDICAL_RECORD_LABELS:
        patterns = medical_record_label_patterns(language)
        assert patterns, language
        for pattern in patterns:
            assert pattern.entity_type == MEDICAL_RECORD_ENTITY_TYPE
            assert pattern.context_required
            assert pattern.safety_sweep_requires_context
            assert "value" in re.compile(pattern.pattern, pattern.flags).groupindex
            # The language pack carries the same objects the table compiled.
            assert any(
                existing is pattern for existing in LANGUAGE_PII_PATTERNS[language]
            )


def test_right_to_left_languages_also_get_a_trailing_label_pattern():
    counts = {
        language: len(medical_record_label_patterns(language))
        for language in MEDICAL_RECORD_LABELS
    }
    assert {language for language, count in counts.items() if count == 2} == {
        "ar",
        "fa",
        "he",
        "ur",
    }


def test_caseless_scripts_do_not_request_case_insensitive_matching():
    for language in ("am", "ar", "fa", "he", "th", "ur", "zh"):
        assert all(
            not pattern.flags & re.IGNORECASE
            for pattern in medical_record_label_patterns(language)
        ), language


@pytest.mark.parametrize("language", ["fa", "ur"])
def test_perso_arabic_labels_are_stable_under_keyboard_folding(language):
    from openmed.core.script_detect import fold_perso_arabic_letters

    fold = partial(fold_perso_arabic_letters, lang=language)
    for label in MEDICAL_RECORD_LABELS[language].all_labels:
        assert fold(label) == label
    for pattern in medical_record_label_patterns(language):
        assert any(
            folded.pattern == fold(pattern.pattern)
            for folded in get_patterns_for_language(language)
        )


def test_value_group_spans_exclude_the_required_label():
    labelled = PIIPattern(r"ref:\s*(?P<value>\d{4})", "account_number")
    plain = PIIPattern(r"ref:\s*\d{4}", "account_number")
    text = "see ref: 1234 today"

    assert list(iter_pattern_spans(text, labelled)) == [(9, 13)]
    assert list(iter_pattern_spans(text, plain)) == [(4, 13)]


def test_a_new_language_needs_only_a_table_entry(monkeypatch):
    table = dict(MEDICAL_RECORD_LABELS)
    table["de"] = MedicalRecordLabels(labels=("Fallnummer", "Patientennummer"))
    monkeypatch.setattr(labels_module, "MEDICAL_RECORD_LABELS", MappingProxyType(table))
    labels_module._compiled_patterns.cache_clear()
    labels_module._canonical_label_patterns.cache_clear()
    try:
        patterns = medical_record_label_patterns("de")
        text = "Fallnummer: 2024-001234, Dosis 500 mg"

        swept = safety_sweep(text, [], lang="de", patterns=patterns)
        matches = medical_record_label_matches(text, "de_DE")
        coverage = labels_module.medical_record_label_coverage()
    finally:
        labels_module._compiled_patterns.cache_clear()
        labels_module._canonical_label_patterns.cache_clear()

    assert [(entity.start, entity.end) for entity in swept] == [(12, 23)]
    assert [match.to_dict()["label"] for match in matches] == ["Fallnummer"]
    assert "de" not in coverage.registered_without_labels


# ---------------------------------------------------------------------------
# Coverage and reports
# ---------------------------------------------------------------------------


def test_coverage_lists_registered_packs_without_labels():
    coverage = medical_record_label_coverage()
    registered = registered_label_languages()

    assert set(MEDICAL_RECORD_LABELS) <= registered
    assert coverage.registered_without_labels == PACKS_WITHOUT_RECORD_LABELS
    assert "en" not in coverage.registered_without_labels
    assert set(coverage.registered_without_labels).isdisjoint(MEDICAL_RECORD_LABELS)
    assert set(coverage.registered_without_labels) | set(MEDICAL_RECORD_LABELS) | {
        "en"
    } == set(registered)


def test_coverage_report_contains_codes_and_counts_only():
    report = medical_record_label_coverage().to_dict()

    assert report["version"] == MEDICAL_RECORD_LABELS_VERSION
    assert report["languages"] == sorted(MEDICAL_RECORD_LABELS)
    assert report["label_counts"] == {
        language: len(entry.all_labels)
        for language, entry in sorted(MEDICAL_RECORD_LABELS.items())
    }
    assert report["label_total"] == sum(report["label_counts"].values())
    serialized = json.dumps(report, ensure_ascii=False)
    assert not re.search(r"\d{4}", serialized.replace("medical-record-labels-v1", ""))


@pytest.mark.parametrize("row", ROWS, ids=lambda row: row["id"])
def test_match_report_contains_labels_offsets_and_counts_only(row):
    matches = medical_record_label_matches(row["text"], row["language"])
    payload = {
        "match_count": len(matches),
        "matches": [match.to_dict() for match in matches],
    }

    assert payload["match_count"] == len(row["record_numbers"])
    for match in payload["matches"]:
        assert set(match) == {"entity_type", "label", "language", "start", "end"}
    serialized = json.dumps(payload, ensure_ascii=False)
    for span in row["record_numbers"] + row["negative_controls"]:
        assert span["text"] not in serialized


def test_sweep_metadata_reports_counts_without_record_numbers():
    from openmed.core.pii import _apply_safety_sweep_to_result

    text = "HN: 6501234 โทร 081-234-5678"
    swept, added = _apply_safety_sweep_to_result(
        text, _empty_prediction(text), lang="th"
    )

    sweep = swept.metadata["safety_sweep"]
    assert added == sweep["spans_added"] == 2
    assert "6501234" not in json.dumps(sweep, ensure_ascii=False)
