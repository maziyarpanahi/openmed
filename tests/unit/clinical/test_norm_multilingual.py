"""Tests for multilingual clinical units and abbreviation normalization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.clinical import (
    ClinicalNormLexicon,
    derive_abnormal_flag,
    normalize_duration,
    normalize_frequency,
    parse_measurement,
    parse_reference_range,
    register_clinical_norm_lexicon,
    structure_vital_sign,
)
from openmed.eval.normalization import (
    DEFAULT_MULTILINGUAL_NORM_FIXTURE,
    score_multilingual_norm_fixture,
    score_multilingual_norm_records,
)

RESTRICTED_MARKERS = (
    "mimic",
    "i2b2",
    "n2c2",
    "umls",
    "snomed",
    "cpt",
    "medical record number",
    "discharge summary",
)


@pytest.mark.parametrize(
    ("language", "text", "canonical_magnitude", "canonical_unit"),
    [
        ("en", "5 mg/dL", 0.05, "g/L"),
        ("es", "5,0 mg/dl", 0.05, "g/L"),
        ("fr", "5,0 mg par dL", 0.05, "g/L"),
        ("de", "1.234,5 mg/dl", 12.345, "g/L"),
        ("zh", "５.０ 毫克/分升", 0.05, "g/L"),
    ],
)
def test_locale_measurements_parse_to_canonical_ucum(
    language,
    text,
    canonical_magnitude,
    canonical_unit,
):
    parsed = parse_measurement(text, language=language)

    assert parsed["status"] == "ok"
    assert parsed["canonical_magnitude"] == pytest.approx(canonical_magnitude)
    assert parsed["canonical_unit"] == canonical_unit
    assert parsed["provenance"]["source_language"] == language


def test_lab_range_parsing_uses_locale_numbers_and_units():
    parsed_range = parse_reference_range("3,5-5,5 mmol/l", language="de")

    assert parsed_range["low"] == 3.5
    assert parsed_range["high"] == 5.5
    assert parsed_range["unit"] == "mmol/l"
    assert (
        derive_abnormal_flag(
            "6,0",
            parsed_range,
            value_unit="mmol/l",
            language="de",
        )
        == "high"
    )


@pytest.mark.parametrize(
    ("language", "text", "expected_per_day"),
    [
        ("es", "dos veces al día", 2.0),
        ("fr", "deux fois par jour", 2.0),
        ("de", "2x täglich", 2.0),
        ("zh", "每日两次", 2.0),
    ],
)
def test_medication_frequency_uses_localized_sig_aliases(
    language,
    text,
    expected_per_day,
):
    parsed = normalize_frequency(text, language=language)

    assert parsed["recognized"] is True
    assert parsed["frequency_per_day"] == expected_per_day


def test_medication_duration_uses_localized_unit_aliases():
    parsed = normalize_duration("7 días", language="es")

    assert parsed["recognized"] is True
    assert parsed["unit"] == "d"
    assert parsed["days"] == 7


@pytest.mark.parametrize(
    ("language", "text", "kind", "unit"),
    [
        ("es", "TA: 120/80 mmHg", "blood_pressure", "mmHg"),
        ("fr", "TA: 120/80 mm Hg", "blood_pressure", "mmHg"),
        ("de", "RR: 120/80 mmHg", "blood_pressure", "mmHg"),
        ("zh", "血压: １２０/８０ 毫米汞柱", "blood_pressure", "mmHg"),
    ],
)
def test_vital_signs_use_localized_abbreviation_expansions(
    language,
    text,
    kind,
    unit,
):
    parsed = structure_vital_sign(text, language=language)

    assert parsed["kind"] == kind
    assert parsed["unit"] == unit
    assert parsed["components"] == [
        {"kind": "systolic", "value": 120, "unit": unit},
        {"kind": "diastolic", "value": 80, "unit": unit},
    ]


def test_multilingual_normalization_gold_meets_per_language_accuracy_gate():
    score = score_multilingual_norm_fixture()

    assert score.accuracy >= 0.90
    for language in ("en", "es", "fr", "de", "zh"):
        assert score.per_language[language] >= 0.90


def test_english_default_behavior_stays_stable():
    assert normalize_frequency("BID") == {
        "raw": "BID",
        "recognized": True,
        "confidence": 1.0,
        "period": None,
        "period_unit": None,
        "frequency_per_day": 2.0,
        "as_needed": False,
        "cue": "bid",
    }
    assert structure_vital_sign("RR 18 /min") == {
        "kind": "respiratory_rate",
        "value": 18,
        "unit": "/min",
        "components": [],
    }
    assert derive_abnormal_flag(13.5, "120-160 g/L", value_unit="g/dL") == "normal"


def test_canonical_ucum_output_is_language_independent_for_equivalent_values():
    english = parse_measurement("5 mg/dL", language="en")
    spanish = parse_measurement("5,0 mg/dl", language="es")
    chinese = parse_measurement("５.０ 毫克/分升", language="zh")

    assert spanish["canonical_unit"] == english["canonical_unit"]
    assert chinese["canonical_unit"] == english["canonical_unit"]
    assert spanish["canonical_magnitude"] == pytest.approx(
        english["canonical_magnitude"]
    )
    assert chinese["canonical_magnitude"] == pytest.approx(
        english["canonical_magnitude"]
    )


def test_registering_stub_language_plus_fixture_is_enough():
    register_clinical_norm_lexicon(
        ClinicalNormLexicon(
            language="zz",
            decimal_separator=",",
            unit_aliases={
                "zz sugar": "mg/dL",
                "zz pressure": "mmHg",
            },
            abbreviation_expansions={"zzbp": "blood_pressure"},
            frequency_aliases={"zz twice": "twice daily"},
        )
    )
    records = [
        {
            "language": "zz",
            "task": "measurement",
            "text": "5,0 zz sugar",
            "expected": {"canonical_magnitude": 0.05, "canonical_unit": "g/L"},
        },
        {
            "language": "zz",
            "task": "vital",
            "text": "zzbp 120/80 zz pressure",
            "expected": {
                "kind": "blood_pressure",
                "value": None,
                "unit": "mmHg",
                "components": [
                    {"kind": "systolic", "value": 120, "unit": "mmHg"},
                    {"kind": "diastolic", "value": 80, "unit": "mmHg"},
                ],
            },
        },
        {
            "language": "zz",
            "task": "frequency",
            "text": "zz twice",
            "expected": {"frequency_per_day": 2.0, "as_needed": False},
        },
    ]

    score = score_multilingual_norm_records(records)

    assert score.per_language["zz"] == 1.0


def test_multilingual_norm_fixture_contains_only_synthetic_permissive_content():
    fixture = Path(DEFAULT_MULTILINGUAL_NORM_FIXTURE)
    text = fixture.read_text(encoding="utf-8").casefold()

    for marker in RESTRICTED_MARKERS:
        assert marker not in text
    for line in fixture.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        assert json.loads(line)["provenance"] == "synthetic-permissive"
