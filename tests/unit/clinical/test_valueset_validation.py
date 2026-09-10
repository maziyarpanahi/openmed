"""Focused tests for offline FHIR ValueSet membership validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.clinical.exporters import (
    VALUESET_VALIDATION_EXTENSION_URL,
    load_valueset,
    validate_code,
    validate_codeable_concept,
)

FIXTURE_ROOT = (
    Path(__file__).parents[3] / "tests" / "fixtures" / "clinical" / "valuesets"
)
FINDINGS_PATH = FIXTURE_ROOT / "clinical-findings.json"
FINDINGS_SYSTEM = "https://openmed.example/fhir/CodeSystem/clinical-findings"


def test_load_valueset_accepts_local_json_and_inline_json():
    loaded = load_valueset(FINDINGS_PATH)
    inline = load_valueset(json.dumps(loaded))

    assert loaded == inline
    assert loaded["resourceType"] == "ValueSet"


def test_validate_code_accepts_member_from_compose_concepts():
    result = validate_code(FINDINGS_SYSTEM, "synthetic-finding-a", FINDINGS_PATH)

    assert result.valid is True
    assert result.ok is True
    assert "member" in result.message


def test_validate_code_rejects_non_member_with_message():
    result = validate_code(FINDINGS_SYSTEM, "not-a-synthetic-finding", FINDINGS_PATH)

    assert result.valid is False
    assert result.ok is False
    assert result.message
    assert "not a member" in result.message


def test_validate_code_supports_local_code_filter_and_exclude():
    valueset = {
        "resourceType": "ValueSet",
        "compose": {
            "include": [
                {
                    "system": FINDINGS_SYSTEM,
                    "filter": [
                        {
                            "property": "code",
                            "op": "regex",
                            "value": "^synthetic-finding-",
                        }
                    ],
                }
            ],
            "exclude": [
                {
                    "system": FINDINGS_SYSTEM,
                    "concept": [{"code": "synthetic-finding-b"}],
                }
            ],
        },
    }

    assert validate_code(FINDINGS_SYSTEM, "synthetic-finding-a", valueset).valid
    excluded = validate_code(FINDINGS_SYSTEM, "synthetic-finding-b", valueset)
    assert excluded.valid is False
    assert "excluded" in excluded.message


def test_validate_codeable_concept_annotates_without_dropping_invalid_coding():
    concept = {
        "coding": [
            {
                "system": FINDINGS_SYSTEM,
                "code": "synthetic-finding-a",
            },
            {
                "system": FINDINGS_SYSTEM,
                "code": "not-a-synthetic-finding",
            },
        ],
        "text": "Synthetic finding",
    }

    result = validate_codeable_concept(concept, FINDINGS_PATH)

    assert result.valid is False
    assert len(result["coding"]) == 2
    assert result["coding"][0] == concept["coding"][0]
    assert result["coding"][1]["code"] == "not-a-synthetic-finding"
    assert any(
        extension["url"] == VALUESET_VALIDATION_EXTENSION_URL
        for extension in result["coding"][1]["extension"]
    )
    assert result.issues[0]["expression"] == ["CodeableConcept.coding[1]"]
    assert "extension" not in concept["coding"][1]


def test_validate_codeable_concept_supports_drop_and_downgrade_policies():
    concept = {
        "coding": [{"system": FINDINGS_SYSTEM, "code": "not-a-synthetic-finding"}]
    }

    dropped = validate_codeable_concept(concept, FINDINGS_PATH, policy="drop")
    downgraded = validate_codeable_concept(
        concept,
        FINDINGS_PATH,
        policy="downgrade",
    )

    assert dropped["coding"] == []
    assert downgraded.issues[0]["severity"] == "warning"
    assert downgraded["coding"][0]["extension"]


@pytest.mark.parametrize("policy", ["unknown", "", None])
def test_validate_codeable_concept_rejects_unknown_policy(policy):
    with pytest.raises(ValueError, match="policy must be"):
        validate_codeable_concept({}, FINDINGS_PATH, policy=policy)  # type: ignore[arg-type]
