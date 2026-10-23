"""Nepali deterministic phone, identifier, and Bikram Sambat date coverage.

Nepali is a user-supplied-model route. These checks pin the deterministic
safety net that runs with an empty model detector: ``+977`` mobile and
landline numbers, context-gated citizenship-certificate and national identity
card numbers, and numeric Bikram Sambat dates, in Devanagari and ASCII digits.
All values are synthetic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import openmed
from openmed.core.language_pack_catalog import (
    DEFAULT_PII_MODELS as BUILTIN_DEFAULT_PII_MODELS,
)
from openmed.core.locale_formats import LOCALE_DATE_ORDER, LOCALE_NUMBER_SEP
from openmed.core.model_registry import get_default_pii_model
from openmed.core.pii import deidentify
from openmed.core.pii_i18n import (
    INDIAN_MULTI_ID_PII_PATTERNS,
    LANGUAGE_PII_PATTERNS,
    SUPPORTED_LANGUAGES,
    USER_SUPPLIED_MODEL_LANGUAGES,
    get_patterns_for_language,
)
from openmed.core.pipeline import Pipeline
from openmed.core.safety_sweep import safety_sweep
from openmed.processing.outputs import PredictionResult

FIXTURE_PATH = (
    Path(__file__).resolve().parents[3] / "openmed/eval/golden/fixtures/i18n/ne.jsonl"
)
_MODEL = "unit-test-model"
_FIXTURE_LABEL_TO_ENTITY_TYPE = {
    "DATE": "date",
    "PHONE": "phone_number",
    "ID_NUM": "national_id",
    "STREET_ADDRESS": "street_address",
    "ZIPCODE": "postcode",
}


def _empty_prediction(text, **kwargs):
    return PredictionResult(
        text=text, entities=[], model_name="synthetic", timestamp="2026-01-01"
    )


def _redacted_outputs(text: str, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    monkeypatch.setattr(openmed, "analyze_text", _empty_prediction)
    direct = deidentify(text, lang="ne", model_name=_MODEL).deidentified_text
    staged = (
        Pipeline(lang="ne", model_name=_MODEL, model_detector=_empty_prediction)
        .run(text)
        .redacted_text
    )
    return [direct, staged]


def _fixture_rows() -> list[dict]:
    return [
        json.loads(line)
        for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_nepali_route_has_a_dedicated_pattern_list() -> None:
    nepali = LANGUAGE_PII_PATTERNS["ne"]
    combined = get_patterns_for_language("ne")

    assert {pattern.entity_type for pattern in nepali} >= {
        "phone_number",
        "national_id",
        "date",
        "postcode",
    }
    assert all(any(p is q for q in combined) for p in nepali)
    assert len(combined) > len(get_patterns_for_language("en"))
    # Nepal is not India: the India-only multi-ID pack stays off this route.
    assert not any(any(p is q for q in INDIAN_MULTI_ID_PII_PATTERNS) for p in combined)
    # No checksum is claimed for Nepali identifiers.
    assert all(
        pattern.validator is None
        for pattern in nepali
        if pattern.entity_type == "national_id"
    )


def test_nepali_locale_tables_have_entries() -> None:
    assert LOCALE_DATE_ORDER["ne"] == "dmy"
    assert LOCALE_NUMBER_SEP["ne"] == LOCALE_NUMBER_SEP["hi"]


@pytest.mark.parametrize(
    ("text", "value", "label"),
    [
        ("मोबाइल ९८४१२३४५६७ छ।", "९८४१२३४५६७", "phone_number"),
        ("मोबाइल ९८४१-२३४५६७ छ।", "९८४१-२३४५६७", "phone_number"),
        ("मोबाइल 98412 34567 छ।", "98412 34567", "phone_number"),
        ("फोन +977 9841234567 छ।", "+977 9841234567", "phone_number"),
        ("फोन +९७७-९८५१०२३४५६ छ।", "+९७७-९८५१०२३४५६", "phone_number"),
        ("फोन 00977 9741234567 छ।", "00977 9741234567", "phone_number"),
        ("फोन +977-1-4412345 छ।", "+977-1-4412345", "phone_number"),
        ("फोन +९७७ ६१ ५२३४५६ छ।", "+९७७ ६१ ५२३४५६", "phone_number"),
        ("फोन ०१-४४१२३४५ छ।", "०१-४४१२३४५", "phone_number"),
        ("ल्यान्डलाइन 061-523456 छ।", "061-523456", "phone_number"),
        (
            "नागरिकता प्रमाणपत्र नं. 27-01-75-05678 छ।",
            "27-01-75-05678",
            "national_id",
        ),
        ("नागरिकता नं. २७-०१-७५-०५६७८ छ।", "२७-०१-७५-०५६७८", "national_id"),
        ("ना.प्र.नं. 123456/789 छ।", "123456/789", "national_id"),
        ("नागरिकता नं: ४५६७८९ छ।", "४५६७८९", "national_id"),
        (
            "राष्ट्रिय परिचयपत्र नं. 123-456-789-0 छ।",
            "123-456-789-0",
            "national_id",
        ),
        (
            "राष्ट्रिय परिचयपत्र नम्बर १२३४५६७८९० छ।",
            "१२३४५६७८९०",
            "national_id",
        ),
        ("जन्म मिति २०४५/०३/३२ हो।", "२०४५/०३/३२", "date"),
        ("जन्म मिति 2045/03/32 हो।", "2045/03/32", "date"),
        ("जन्म मिति २०८०-०५-१२ हो।", "२०८०-०५-१२", "date"),
        ("भर्ना मिति 2080.05.12 हो।", "2080.05.12", "date"),
        ("जन्म मिति ३२/०३/२०४५ हो।", "३२/०३/२०४५", "date"),
        ("वि.सं. २०८०/५/१२ मा भर्ना।", "२०८०/५/१२", "date"),
    ],
)
def test_nepali_values_are_swept_whole_and_masked_with_an_empty_detector(
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    value: str,
    label: str,
) -> None:
    start = text.index(value)
    entities = safety_sweep(text, [], lang="ne")

    assert any(
        (e.start, e.end, e.text, e.label) == (start, start + len(value), value, label)
        for e in entities
    ), [(e.label, e.text) for e in entities]
    # No competing rule keeps a fragment of the value under another label.
    assert not any(
        e.start < start + len(value) and e.end > start and e.label != label
        for e in entities
    )
    for output in _redacted_outputs(text, monkeypatch):
        assert value not in output
        # The whole number is masked, not only the digits after +977.
        assert "977" not in output and "९७७" not in output
        assert output.startswith(text[:start])


@pytest.mark.parametrize(
    "text",
    [
        "नागरिकता प्रमाणपत्र नं. 27-01-75-05678 छ।",
        "नागरिकता नं. २७-०१-७५-०५६७८ छ।",
    ],
)
def test_citizenship_numbers_are_not_half_masked_as_dates(
    monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    assert not any(e.label == "date" for e in safety_sweep(text, [], lang="ne"))
    for output in _redacted_outputs(text, monkeypatch):
        assert "date" not in output.lower()
        assert "postcode" not in output.lower()


@pytest.mark.parametrize(
    "text",
    [
        "औषधि 500 mg दिनको 2 पटक।",
        "हेमोग्लोबिन 12.5 g/dL, प्लेटलेट 250000 /µL।",
        "क्रिएटिनिन १.२ mg/dL, सोडियम १३८ mmol/L, पोटासियम ४.१।",
        "कोठा नं. 4412345 मा भर्ना।",
        "रिपोर्ट 27-01-75-05678 संलग्न।",
        "नमूना 123-456-789-0 प्रयोगशालामा।",
        "रक्तचाप 120/80, नाडी 72, तौल 65.5 kg।",
        "संस्करण 2.0.1, खण्ड 4.2.1, अनुपात 2080/13/12।",
        "प्लेटलेट 44600 /µL, हेमाटोक्रिट 33700।",
        # Citizenship context nearby does not turn lab values into identifiers.
        "नागरिकता नं. 27-01-75-05678। प्लेटलेट 250000, ग्लुकोज 5400।",
    ],
)
def test_numbers_without_nepali_identifier_context_are_not_identifiers(
    text: str,
) -> None:
    entities = safety_sweep(text, [], lang="ne")
    flagged = {
        e.text
        for e in entities
        if e.label in {"national_id", "phone_number", "postcode"}
    }
    flagged.discard("27-01-75-05678")

    assert not flagged, flagged


@pytest.mark.parametrize(
    "text",
    [
        "अनुपात 2080/13/12 मात्र।",
        "अनुपात २०८०/०५/३३ मात्र।",
        "रक्तचाप 120/80 छ।",
        # Pre-2000 years must still be valid Gregorian dates.
        "मिति 1980/02/31 मात्र।",
    ],
)
def test_invalid_bikram_sambat_shapes_are_not_dates(text: str) -> None:
    assert not any(e.label == "date" for e in safety_sweep(text, [], lang="ne"))


def test_nepali_stays_a_user_supplied_model_route() -> None:
    assert "ne" in USER_SUPPLIED_MODEL_LANGUAGES
    assert "ne" not in SUPPORTED_LANGUAGES
    assert "ne" not in BUILTIN_DEFAULT_PII_MODELS
    assert get_default_pii_model("ne") is None


@pytest.mark.parametrize(
    ("lang", "text", "expected"),
    [
        (
            "hi",
            "रोगी का जन्म 12/05/1985 को हुआ। फोन +91 98765 43210. पिन 110001.",
            [("date", 13, 23), ("postcode", 57, 63)],
        ),
        (
            "hi",
            "नागरिकता नं. 27-01-75-05678, मोबाइल 9841234567, जन्म मिति २०४५/०३/३२",
            [("date", 13, 21), ("phone_number", 36, 46)],
        ),
        (
            "mr",
            "रुग्णाचे नाव सौ. वैशाली सुरेश देशमुख. जन्मतारीख १४ फेब्रुवारी १९८५. "
            "मोबाइल +९१ ९८७६५ ४३२१०.",
            [("name", 13, 36), ("date", 48, 66), ("phone_number", 75, 90)],
        ),
        (
            "mr",
            "फोन +977 9841234567, तारीख ०५/०७/२००१",
            [("phone_number", 9, 19), ("date", 27, 37)],
        ),
    ],
)
def test_hindi_and_marathi_sweeps_are_unchanged(
    lang: str, text: str, expected: list[tuple[str, int, int]]
) -> None:
    nepali = LANGUAGE_PII_PATTERNS.get("ne", [])
    patterns = get_patterns_for_language(lang)

    assert not any(any(p is q for q in nepali) for p in patterns)
    assert [(e.label, e.start, e.end) for e in safety_sweep(text, [], lang=lang)] == (
        expected
    )


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("रोगी स्थिर है। उसका घर दिल्ली में है।", "hi"),
        ("रुग्णाचे नाव आहे आणि तो बरा आहे.", "mr"),
    ],
)
def test_hindi_and_marathi_routing_is_unchanged(text: str, lang: str) -> None:
    assert Pipeline(lang="auto").stage2_language_script(text).lang == lang


def test_nepali_fixture_spans_are_swept_exactly() -> None:
    rows = _fixture_rows()

    assert len(rows) >= 3
    for row in rows:
        assert row["language"] == "ne"
        assert row["metadata"]["synthetic"] is True
        text = row["text"]
        swept = {(e.label, e.start, e.end) for e in safety_sweep(text, [], lang="ne")}
        for span in row["gold_spans"]:
            assert text[span["start"] : span["end"]] == span["text"]
            key = (
                _FIXTURE_LABEL_TO_ENTITY_TYPE[span["label"]],
                span["start"],
                span["end"],
            )
            assert key in swept, (row["id"], span["label"], span["text"])


def test_nepali_fixture_is_leak_free_with_an_empty_detector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _fixture_rows()
    digit_forms = {"ascii": 0, "devanagari": 0}
    for row in rows:
        text = row["text"]
        outputs = _redacted_outputs(text, monkeypatch)
        for span in row["gold_spans"]:
            value = span["text"]
            if any("०" <= char <= "९" for char in value):
                digit_forms["devanagari"] += 1
            if any("0" <= char <= "9" for char in value):
                digit_forms["ascii"] += 1
            for output in outputs:
                assert value not in output, (row["id"], span["label"])

    assert all(digit_forms.values()), digit_forms
