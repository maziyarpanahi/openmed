"""Tests for the packaged reviewer-disagreement report JSON Schema.

The schema is the contract consumers validate a report against, so it must
accept exactly what ``ReviewerDisagreementReport.to_dict()`` produces and
reject any extra key — in particular reviewer or case identifiers, which the
report deliberately never contains.
"""

from __future__ import annotations

import json
from importlib import resources

import pytest
from jsonschema.validators import validator_for

from openmed.eval.reviewer_disagreement import (
    DisagreementReason,
    ReviewerDecision,
    reviewer_disagreement_report,
)

SCHEMA_PACKAGE = "openmed.core.schemas.json"
SCHEMA_NAME = "reviewer_disagreement_report.schema.json"


def _load_schema() -> dict:
    resource = resources.files(SCHEMA_PACKAGE).joinpath(SCHEMA_NAME)
    return json.loads(resource.read_text(encoding="utf-8"))


def _synthetic_report() -> dict:
    def case(
        number: int,
        left: str,
        right: str,
        *,
        reason: DisagreementReason | None = None,
        adjudicated: bool = False,
    ) -> list[ReviewerDecision]:
        case_id = f"synthetic-case-{number}"
        return [
            ReviewerDecision(
                case_id=case_id,
                reviewer_id="synthetic-reviewer-a",
                decision=left,
                reason=reason,
                adjudicated=adjudicated,
            ),
            ReviewerDecision(
                case_id=case_id,
                reviewer_id="synthetic-reviewer-b",
                decision=right,
                reason=reason,
                adjudicated=adjudicated,
            ),
        ]

    decisions = [
        *case(1, "accept", "accept"),
        *case(2, "accept", "accept"),
        *case(
            3,
            "accept",
            "revise",
            reason=DisagreementReason.EVIDENCE_QUALITY,
            adjudicated=True,
        ),
        *case(
            4,
            "revise",
            "accept",
            reason=DisagreementReason.EVIDENCE_QUALITY,
            adjudicated=True,
        ),
        *case(
            5,
            "accept",
            "revise",
            reason=DisagreementReason.EVIDENCE_QUALITY,
        ),
        *case(
            6,
            "revise",
            "accept",
            reason=DisagreementReason.EVIDENCE_QUALITY,
        ),
    ]
    return reviewer_disagreement_report(decisions, minimum_cell_size=2).to_dict()


def test_schema_is_a_valid_json_schema() -> None:
    schema = _load_schema()
    validator_for(schema).check_schema(schema)


def test_synthetic_report_validates_against_schema() -> None:
    schema = _load_schema()
    validator = validator_for(schema)(schema)

    assert not list(validator.iter_errors(_synthetic_report()))


@pytest.mark.parametrize(
    "extra_key",
    ["reviewer_id", "case_id", "decision"],
)
def test_report_with_identifier_key_fails_validation(extra_key: str) -> None:
    schema = _load_schema()
    validator = validator_for(schema)(schema)
    report = _synthetic_report()
    report[extra_key] = "synthetic-identifier"

    assert list(validator.iter_errors(report))
