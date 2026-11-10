from __future__ import annotations

import pytest

from openmed.core.pii_i18n import validate_chinese_resident_identity_card
from openmed.eval.release_gates import (
    SURROGATE_QUALITY_GATE,
    evaluate_surrogate_quality_gate,
)
from openmed.eval.surrogate_quality import (
    DEFAULT_SURROGATE_QUALITY_LOCALES,
    SURROGATE_QUALITY_DIMENSIONS,
    SurrogateQualityRecord,
    evaluate_surrogate_quality,
    load_surrogate_quality_records,
)


def test_surrogate_quality_fixture_passes_all_required_locales() -> None:
    report = evaluate_surrogate_quality()

    assert report.passed is True
    assert report.missing_locales == ()
    assert set(report.locale_reports) == set(DEFAULT_SURROGATE_QUALITY_LOCALES)

    for language in DEFAULT_SURROGATE_QUALITY_LOCALES:
        locale_report = report.locale_reports[language]
        assert locale_report.pass_rate >= 0.90
        assert set(locale_report.dimension_scores) == set(SURROGATE_QUALITY_DIMENSIONS)
        assert all(score == 1.0 for score in locale_report.dimension_scores.values())


def test_surrogate_quality_gate_fails_with_injected_bad_surrogate() -> None:
    records = list(load_surrogate_quality_records())
    records.append(
        {
            "record_id": "sq-zh-bad",
            "language": "zh",
            "locale": "zh_CN",
            "surrogates": {
                "name": "John Doe",
                "date_of_birth": "04/12/1990",
                "national_id": "110105199004123416",
            },
            "expected": {
                "birth_date": "1990-04-12",
                "gender": "female",
                "region_code": "110105",
            },
            "metadata": {
                "synthetic": True,
                "contains_real_phi": False,
                "synthetic_source": "surrogate_quality_bad_fixture",
            },
        }
    )

    report = evaluate_surrogate_quality(records)
    gate = evaluate_surrogate_quality_gate(report)

    assert report.passed is False
    assert report.locale_reports["zh"].pass_rate == 0.5
    assert gate.gate == SURROGATE_QUALITY_GATE
    assert gate.passed is False
    assert gate.details["failing_locales"] == {"zh": 0.5}


def test_surrogate_quality_validates_chinese_resident_id_checksum() -> None:
    assert validate_chinese_resident_identity_card("110105199004123424")
    assert not validate_chinese_resident_identity_card("110105199004123425")
    assert not validate_chinese_resident_identity_card("110105199002303424")


def test_surrogate_quality_record_requires_explicit_synthetic_marker() -> None:
    with pytest.raises(ValueError, match="explicitly marked synthetic"):
        SurrogateQualityRecord.from_mapping(
            {
                "record_id": "sq-unmarked",
                "language": "en",
                "surrogates": {
                    "name": "Avery Morgan",
                    "date_of_birth": "04/12/1990",
                    "national_id": "123-45-6789",
                },
                "metadata": {"contains_real_phi": False},
            }
        )
