"""Reference coverage for the shared identifier-validator parity vectors.

Every value in ``tests/fixtures/parity/identifier_validators_v1.json`` is
synthetic: it comes from a documented test range or from one of the
repository's seeded ``generate_*`` helpers. Each case names the Python
validator that is the reference implementation, so the Android and Swift
ports (#5419-#5421) can consume the same vectors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest

from openmed.core import pii_i18n
from openmed.core.anonymizer.providers import clinical_ids

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "parity"
    / "identifier_validators_v1.json"
)

REFERENCES: dict[str, Callable[[str], bool]] = {
    "openmed.core.anonymizer.providers.clinical_ids.validate_canadian_sin": (
        clinical_ids.validate_canadian_sin
    ),
    "openmed.core.anonymizer.providers.clinical_ids.validate_iban": (
        clinical_ids.validate_iban
    ),
    "openmed.core.anonymizer.providers.clinical_ids.validate_luhn": (
        clinical_ids.validate_luhn
    ),
    "openmed.core.anonymizer.providers.clinical_ids.validate_npi": (
        clinical_ids.validate_npi
    ),
    "openmed.core.anonymizer.providers.clinical_ids.validate_ssn": (
        clinical_ids.validate_ssn
    ),
    "openmed.core.anonymizer.providers.clinical_ids.validate_uk_nhs_number": (
        clinical_ids.validate_uk_nhs_number
    ),
    "openmed.core.pii_i18n.validate_aadhaar": pii_i18n.validate_aadhaar,
    "openmed.core.pii_i18n.validate_chinese_resident_id": (
        pii_i18n.validate_chinese_resident_id
    ),
    "openmed.core.pii_i18n.validate_dutch_bsn": pii_i18n.validate_dutch_bsn,
    "openmed.core.pii_i18n.validate_french_nir": pii_i18n.validate_french_nir,
    "openmed.core.pii_i18n.validate_spanish_dni": pii_i18n.validate_spanish_dni,
}

FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _is_single_digit_change(left: str, right: str) -> bool:
    if len(left) != len(right):
        return False
    return sum(a != b for a, b in zip(left, right)) == 1


CASES = [
    pytest.param(
        case["validator"],
        case["reference"],
        value,
        expected,
        id=f"{case['validator']}-{'valid' if expected else 'invalid'}-{index:02d}",
    )
    for case in FIXTURE["cases"]
    for expected, key in ((True, "valid"), (False, "invalid"))
    for index, value in enumerate(case[key], start=1)
]


def test_fixture_declares_synthetic_reference_vectors() -> None:
    assert FIXTURE["fixture"] == "identifier_validators_v1"
    assert FIXTURE["synthetic"] is True
    assert "synthetic" in FIXTURE["description"].lower()
    for case in FIXTURE["cases"]:
        assert case["reference"] in REFERENCES, case["validator"]
        assert len(case["valid"]) >= 3, case["validator"]
        assert len(case["invalid"]) >= 3, case["validator"]


def test_invalid_vectors_include_a_single_digit_change() -> None:
    for case in FIXTURE["cases"]:
        assert any(
            _is_single_digit_change(valid, invalid)
            for valid in case["valid"]
            for invalid in case["invalid"]
        ), case["validator"]


@pytest.mark.parametrize("validator,reference,value,expected", CASES)
def test_reference_validator_returns_expected_result(
    validator: str,
    reference: str,
    value: str,
    expected: bool,
) -> None:
    assert REFERENCES[reference](value) is expected, (validator, value)
