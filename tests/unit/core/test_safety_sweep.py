"""Tests for the deterministic post-ML PII safety sweep."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pytest

from openmed.core.anonymizer.providers import clinical_ids
from openmed.core.labels import ID_SUBTYPE_NPI
from openmed.core.pii import _apply_safety_sweep_to_result, deidentify
from openmed.core.pii_entity_merger import PIIPattern
from openmed.core.quality_gates import detect_overlapping_entities
from openmed.core.safety_sweep import (
    SAFETY_SWEEP_PATTERNS_VERSION,
    SAFETY_SWEEP_SOURCE,
    safety_sweep,
)
from openmed.processing.outputs import EntityPrediction, PredictionResult


def _empty_prediction(text, **kwargs):
    return PredictionResult(
        text=text, entities=[], model_name="synthetic", timestamp="2026-01-01"
    )


@pytest.mark.parametrize(
    ("lang", "prefix", "digits", "zero"),
    [
        ("th", "เลขประจำตัวประชาชน ", "1101700203450", 0x0E50),
        ("th", "โทร ", "0812345678", 0x0E50),
        ("hi", "आधार ", "2345 6789 0124", 0x0966),
        ("te", "ఆధార్ ", "2345 6789 0124", 0x0C66),
    ],
)
def test_native_digits_masked_in_both_public_paths(
    monkeypatch, lang, prefix, digits, zero
):
    from openmed.core.pii_i18n import validate_thai_national_id
    from openmed.core.pipeline import Pipeline

    if lang == "th" and len(digits) == 13:
        check = (
            11 - sum(int(v) * w for v, w in zip(digits[:12], range(13, 1, -1))) % 11
        ) % 10
        digits = digits[:12] + str(check)
    native = "".join(chr(zero + int(c)) if c.isdecimal() else c for c in digits)
    for value in (native, native[:5] + digits[5:]):
        text = prefix + value + " stable"
        entities = safety_sweep(text, [], lang=lang)
        assert any(
            (e.start, e.end, e.text) == (len(prefix), len(prefix) + len(value), value)
            for e in entities
        )
        monkeypatch.setattr("openmed.analyze_text", _empty_prediction)
        direct = deidentify(text, lang=lang, model_name="unit-test-model")
        staged = Pipeline(lang=lang, model_detector=_empty_prediction).run(text)
        assert value not in direct.deidentified_text
        assert value not in staged.redacted_text
        assert staged.redacted_text.endswith(" stable")
    if len(digits) == 13:
        assert validate_thai_national_id(native)
        invalid = native[:-1] + chr(zero + (int(digits[-1]) + 1) % 10)
        assert not validate_thai_national_id(invalid)
        assert not any(
            e.label == "national_id"
            for e in safety_sweep(prefix + invalid, [], lang=lang)
        )
    if len(digits.replace(" ", "")) in {12, 13}:
        invalid = native[:-1] + chr(zero + (int(digits[-1]) + 1) % 10)
        source = prefix + invalid
        assert (
            invalid
            in deidentify(
                source, lang=lang, model_name="unit-test-model"
            ).deidentified_text
        )
        assert (
            invalid
            in Pipeline(lang=lang, model_detector=_empty_prediction)
            .run(source)
            .redacted_text
        )


@pytest.mark.parametrize(
    "controls",
    [
        ("\u200e", ""),
        ("\u200f", ""),
        ("\u061c", ""),
        ("\u202a", "\u202c"),
        ("\u202b", "\u202c"),
        ("\u202d", "\u202c"),
        ("\u202e", "\u202c"),
        ("\u2066", "\u2069"),
        ("\u2067", "\u2069"),
        ("\u2068", "\u2069"),
    ],
)
@pytest.mark.parametrize(
    ("lang", "prefix", "value"),
    [
        ("fa", "کد ملی ", "0012345679"),
        ("en", "SSN ", "123-45-6789"),
        ("he", "טלפון ", "0501234567"),
        ("ur", "آدھار ", "2345 6789 0124"),
        ("ar", "الرقم القومي ", "29001011234562"),
    ],
)
def test_in_value_bidi_marks_are_masked_in_both_public_paths(
    monkeypatch, controls, lang, prefix, value
):
    from openmed.core.pipeline import Pipeline
    from openmed.core.rtl_render import strip_unbalanced_bidi_controls

    left, right = controls
    marked = value[:3] + left + value[3:6] + right + value[6:]
    text = prefix + marked + " stable"
    baseline = safety_sweep(prefix + value, [], lang=lang)
    assert baseline, "synthetic control must be detectable"
    entities = safety_sweep(text, [], lang=lang)
    assert any(
        e.start == len(prefix) and e.end == len(prefix) + len(marked) for e in entities
    )
    monkeypatch.setattr("openmed.analyze_text", _empty_prediction)
    for result in (
        deidentify(text, lang=lang, model_name="unit-test-model"),
        Pipeline(lang=lang, model_detector=_empty_prediction)
        .run(text)
        .deidentification_result,
    ):
        assert marked not in result.deidentified_text
        assert (
            strip_unbalanced_bidi_controls(result.deidentified_text)
            == result.deidentified_text
        )


@pytest.mark.parametrize(
    "lang",
    ["no", "he", "id", "vi", "th", "fa", "bn", "ta", "hi", "yo", "ha", "ig", "ne"],
)
def test_dot_dates_have_validated_coverage_and_no_version_matches(monkeypatch, lang):
    from openmed.core.pipeline import Pipeline

    assert any(
        e.text == "15.01.1980" and e.label == "date"
        for e in safety_sweep("DOB 15.01.1980", [], lang=lang)
    )
    assert not any(
        e.label == "date"
        for e in safety_sweep(
            "version 1.2.3, 15.01, section 4.2.1; 31.02.1980", [], lang=lang
        )
    )
    monkeypatch.setattr("openmed.analyze_text", _empty_prediction)
    source = "DOB 15.01.1980"
    assert (
        "15.01.1980"
        not in deidentify(
            source, lang=lang, model_name="unit-test-model"
        ).deidentified_text
    )
    assert (
        "15.01.1980"
        not in Pipeline(
            lang=lang, model_name="unit-test-model", model_detector=_empty_prediction
        )
        .run(source)
        .redacted_text
    )


@pytest.mark.parametrize("prefix", ["ב", "ל", "מ", "ה", "ו", "כ", "ש"])
@pytest.mark.parametrize("separator", ["", "-", "־"])
def test_hebrew_date_prefixes_are_not_in_spans(prefix, separator):
    for value in ("15.01.1980", "15 בינואר 2026"):
        text = "נולד " + prefix + separator + value
        assert any(
            e.text == value and e.start == len("נולד " + prefix + separator)
            for e in safety_sweep(text, [], lang="he")
        )


@pytest.mark.parametrize(
    ("lang", "text", "value"),
    [
        ("fa", "بيمار كد ملي ۰۰۱۲۳۴۵۶۷۹", "۰۰۱۲۳۴۵۶۷۹"),
        ("fa", "كد پستي ۱۴۳۹۸۱۴۵۶۷", "۱۴۳۹۸۱۴۵۶۷"),
        ("ur", "مريض آدهار 2345 6789 0124", "2345 6789 0124"),
    ],
)
def test_arabic_keyboard_variants_match_context_without_rewriting_output(
    monkeypatch, lang, text, value
):
    from openmed.core.pipeline import Pipeline

    monkeypatch.setattr("openmed.analyze_text", _empty_prediction)
    for result in (
        deidentify(text, lang=lang, model_name="unit-test-model"),
        Pipeline(lang=lang, model_detector=_empty_prediction)
        .run(text)
        .deidentification_result,
    ):
        assert value not in result.deidentified_text
        assert result.deidentified_text.startswith(text[: text.index(value)])


def _entity_for(text: str, value: str, label: str = "MODEL") -> EntityPrediction:
    start = text.index(value)
    return EntityPrediction(
        text=value,
        label=label,
        start=start,
        end=start + len(value),
        confidence=0.99,
        metadata={"source": "model"},
    )


def _swept_by_label(entities):
    return {
        entity.label: entity
        for entity in entities
        if (entity.metadata or {}).get("source") == SAFETY_SWEEP_SOURCE
    }


def test_safety_sweep_recovers_ml_missed_deterministic_identifiers():
    text = (
        "Card 4111 1111 1111 1111. "
        "IBAN GB82 WEST 1234 5698 7654 32. "
        "SSN 123-45-6789. "
        "Email jane.patient@example.com. "
        "Phone 415-555-2671."
    )

    entities = safety_sweep(text, [])
    swept = _swept_by_label(entities)

    assert "credit_debit_card" in swept
    assert "iban" in swept
    assert "ssn" in swept
    assert "email" in swept
    assert "phone_number" in swept
    assert swept["iban"].metadata["patterns_version"] == SAFETY_SWEEP_PATTERNS_VERSION
    assert swept["email"].metadata["source"] == SAFETY_SWEEP_SOURCE


def test_safety_sweep_rejects_invalid_checksum_identifiers():
    text = "Card 4111 1111 1111 1112. IBAN GB83 WEST 1234 5698 7654 32."

    swept = _swept_by_label(safety_sweep(text, []))

    assert "credit_debit_card" not in swept
    assert "iban" not in swept


def test_safety_sweep_delegates_checksum_validation_to_clinical_ids(monkeypatch):
    luhn_calls = []
    iban_calls = []

    def fake_luhn(value: str) -> bool:
        luhn_calls.append(value)
        return True

    def fake_iban(value: str) -> bool:
        iban_calls.append(value)
        return False

    monkeypatch.setattr(clinical_ids, "validate_luhn", fake_luhn)
    monkeypatch.setattr(clinical_ids, "validate_iban", fake_iban)

    text = "Card 1234 5678 9012 3456. IBAN GB82 WEST 1234 5698 7654 32."
    swept = _swept_by_label(safety_sweep(text, []))

    assert luhn_calls == ["1234 5678 9012 3456"]
    assert iban_calls == ["GB82 WEST 1234 5698 7654 32"]
    assert "credit_debit_card" in swept
    assert "iban" not in swept


def test_safety_sweep_includes_id_subtype_metadata_for_id_matches():
    text = "Provider NPI TEST-12345"
    pattern = PIIPattern(
        r"\bTEST-\d{5}\b",
        "npi",
        priority=1,
        base_score=0.8,
        context_words=["npi"],
        context_boost=0.1,
    )

    swept = _swept_by_label(safety_sweep(text, [], patterns=[pattern]))

    assert swept["npi"].metadata["id_subtype"] == ID_SUBTYPE_NPI
    assert swept["npi"].metadata["safety_sweep"]["id_subtype"] == ID_SUBTYPE_NPI


def test_safety_sweep_skips_bare_postcode_numbers_without_context():
    five_digit_text = "Calibration count was 10001 before discharge."
    six_digit_text = "Calibration count was 110001 before discharge."

    assert "postcode" not in _swept_by_label(safety_sweep(five_digit_text, []))
    assert "postcode" not in _swept_by_label(
        safety_sweep(six_digit_text, [], lang="hi")
    )


def test_safety_sweep_keeps_postcode_when_context_present():
    english_swept = _swept_by_label(safety_sweep("ZIP code 10001.", []))
    italian_swept = _swept_by_label(safety_sweep("CAP 00100.", [], lang="it"))

    assert english_swept["postcode"].text == "10001"
    assert italian_swept["postcode"].text == "00100"


def test_safety_sweep_keeps_valid_national_id_and_mrn_with_context():
    national_id_swept = _swept_by_label(
        safety_sweep("Steuer-ID 12345678912 verified.", [], lang="de")
    )
    mrn_swept = _swept_by_label(safety_sweep("MRN: 123456 verified.", []))

    assert national_id_swept["national_id"].text == "12345678912"
    assert mrn_swept["medical_record_number"].text == "MRN: 123456"


def test_safety_sweep_never_adds_spans_overlapping_model_spans():
    text = "Card 4111 1111 1111 1111. Email jane.patient@example.com."
    model_card = _entity_for(text, "4111 1111 1111 1111", label="ID_NUM")

    entities = safety_sweep(text, [model_card])
    swept = _swept_by_label(entities)

    assert "credit_debit_card" not in swept
    assert "email" in swept
    for entity in swept.values():
        assert entity.start >= model_card.end or entity.end <= model_card.start


def test_safety_sweep_resolves_overlapping_existing_spans():
    text = "Patient SSN 123-45-6789 was verified."
    broad = EntityPrediction(
        text="Patient SSN 123-45-6789",
        label="OTHER",
        start=0,
        end=24,
        confidence=0.99,
    )
    sensitive = EntityPrediction(
        text="123-45-6789",
        label="SSN",
        start=12,
        end=23,
        confidence=0.50,
    )

    entities = safety_sweep(text, [broad, sensitive])

    assert detect_overlapping_entities(entities) == []
    assert [entity.label for entity in entities] == ["SSN"]


def test_safety_sweep_counts_added_spans_independently_of_overlap_resolution():
    text = "Patient SSN 123-45-6789. Email jane.patient@example.com."
    broad = _entity_for(text, "Patient SSN 123-45-6789", label="OTHER")
    sensitive = _entity_for(text, "123-45-6789", label="SSN")
    pii_result = PredictionResult(
        text=text,
        entities=[broad, sensitive],
        model_name="stub",
        timestamp=datetime.now().isoformat(),
    )

    swept_result, added_count = _apply_safety_sweep_to_result(
        text,
        pii_result,
        lang="en",
    )

    assert added_count == 1
    assert swept_result.metadata["safety_sweep"]["spans_added"] == 1
    assert "email" in _swept_by_label(swept_result.entities)
    assert detect_overlapping_entities(swept_result.entities) == []


@patch("openmed.core.pii.extract_pii")
def test_deidentify_runs_safety_sweep_by_default_for_ml_misses(mock_extract):
    text = "Email: jane.patient@example.com"
    mock_extract.return_value = PredictionResult(
        text=text,
        entities=[],
        model_name="stub",
        timestamp=datetime.now().isoformat(),
    )

    result = deidentify(text, method="mask")

    assert result.deidentified_text == "Email: [email]"
    assert result.metadata["safety_sweep"]["spans_added"] == 1
    assert result.pii_entities[0].metadata["source"] == SAFETY_SWEEP_SOURCE
    assert result.to_dict()["pii_entities"][0]["metadata"]["patterns_version"] == (
        SAFETY_SWEEP_PATTERNS_VERSION
    )


@patch("openmed.core.pii.extract_pii")
def test_deidentify_can_disable_safety_sweep(mock_extract):
    text = "Email: jane.patient@example.com"
    mock_extract.return_value = PredictionResult(
        text=text,
        entities=[],
        model_name="stub",
        timestamp=datetime.now().isoformat(),
    )

    result = deidentify(text, method="mask", use_safety_sweep=False)

    assert result.deidentified_text == text
    assert result.pii_entities == []


@patch("openmed.core.pii.extract_pii")
def test_deidentify_resolves_overlaps_before_redaction(mock_extract):
    text = "Patient SSN 123-45-6789 was verified."
    mock_extract.return_value = PredictionResult(
        text=text,
        entities=[
            EntityPrediction(
                text="Patient SSN 123-45-6789",
                label="OTHER",
                start=0,
                end=24,
                confidence=0.99,
            ),
            EntityPrediction(
                text="123-45-6789",
                label="SSN",
                start=12,
                end=23,
                confidence=0.50,
            ),
        ],
        model_name="stub",
        timestamp=datetime.now().isoformat(),
    )

    result = deidentify(text, method="mask", use_safety_sweep=False)

    assert detect_overlapping_entities(result.pii_entities) == []
    assert [entity.label for entity in result.pii_entities] == ["SSN"]
    assert result.deidentified_text == "Patient SSN [SSN] was verified."
