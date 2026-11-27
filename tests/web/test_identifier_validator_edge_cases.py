"""The web validator edge cases record what the Python references return.

``tests/web/fixtures/identifier_validator_edge_cases.json`` holds synthetic
inputs beyond the shared ``identifier_validators_v1`` vectors (separators,
case, Python whitespace, Unicode digits). The web spec
``tests/web/test_identifier_validators.spec.ts`` replays them against the web
port; this test keeps every expected value equal to the Python reference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest

from openmed.core import pii_i18n
from openmed.core.anonymizer.providers import clinical_ids

FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "identifier_validator_edge_cases.json"
)
VECTORS = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "parity"
    / "identifier_validators_v1.json"
)

REFERENCES: dict[str, Callable[[str], bool]] = {
    "validate_aadhaar": pii_i18n.validate_aadhaar,
    "validate_canadian_sin": clinical_ids.validate_canadian_sin,
    "validate_chinese_resident_id": pii_i18n.validate_chinese_resident_id,
    "validate_dutch_bsn": pii_i18n.validate_dutch_bsn,
    "validate_french_nir": pii_i18n.validate_french_nir,
    "validate_iban": clinical_ids.validate_iban,
    "validate_luhn": clinical_ids.validate_luhn,
    "validate_npi": clinical_ids.validate_npi,
    "validate_spanish_dni": pii_i18n.validate_spanish_dni,
    "validate_ssn": clinical_ids.validate_ssn,
    "validate_uk_nhs_number": clinical_ids.validate_uk_nhs_number,
}

DOCUMENT = json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_edge_cases_cover_every_shared_validator() -> None:
    shared = json.loads(VECTORS.read_text(encoding="utf-8"))
    references = {case["reference"].rsplit(".", 1)[1] for case in shared["cases"]}

    assert DOCUMENT["synthetic"] is True
    assert references == set(REFERENCES)
    assert {case["reference"] for case in DOCUMENT["cases"]} == set(REFERENCES)
    assert {case["expected"] for case in DOCUMENT["cases"]} == {True, False}


@pytest.mark.parametrize(
    "case",
    DOCUMENT["cases"],
    ids=[
        f"{case['reference']}-{index:02d}"
        for index, case in enumerate(DOCUMENT["cases"])
    ],
)
def test_expected_values_are_the_python_results(case: dict[str, object]) -> None:
    reference = REFERENCES[str(case["reference"])]

    assert reference(str(case["value"])) is case["expected"]
