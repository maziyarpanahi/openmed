"""Focused tests for nested structured-redaction idempotence checks."""

from __future__ import annotations

import json
from dataclasses import dataclass

from openmed.risk import check_idempotence

_POLICY = "sha256:" + "a" * 64


def _fhir_pass(*, surrogate: str, action: str = "replace") -> dict[str, object]:
    """Return a synthetic FHIR-shaped redaction result."""

    return {
        "resource": {
            "resourceType": "Bundle",
            "entry": [
                {
                    "fullUrl": "urn:synthetic:entry-a",
                    "resource": {
                        "resourceType": "Patient",
                        "id": "synthetic-patient-a",
                        "name": [{"text": surrogate}],
                    },
                }
            ],
        },
        "report": {
            "policy_fingerprint": _POLICY,
            "counts": {"changed_value_count": 1, "redacted": 1},
            "actions": [
                {
                    "path": "entry[0].resource.name[0].text",
                    "action": action,
                    "surrogate": surrogate,
                }
            ],
        },
    }


def _omop_pass(*, surrogate: str) -> dict[str, object]:
    """Return a synthetic OMOP-shaped redaction result."""

    return {
        "data": {
            "tables": {
                "person": [
                    {
                        "person_id": surrogate,
                        "gender_concept_id": 0,
                    }
                ],
                "visit_occurrence": [{"person_id": surrogate, "visit_concept_id": 0}],
            }
        },
        "report": {
            "policy_fingerprint": _POLICY,
            "counts": {"redacted": 2},
            "redactions": [
                {
                    "path": "tables.person[0].person_id",
                    "action": "replace",
                    "surrogate": surrogate,
                },
                {
                    "path": "tables.visit_occurrence[0].person_id",
                    "action": "replace",
                    "surrogate": surrogate,
                },
            ],
        },
    }


def test_identical_nested_fhir_passes_are_idempotent_and_deterministic() -> None:
    first = _fhir_pass(surrogate="[SYNTHETIC-NAME]")
    second = _fhir_pass(surrogate="[SYNTHETIC-NAME]")

    report = check_idempotence(first, second)

    assert report.is_idempotent is True
    assert report.shape_match is True
    assert report.counts_match is True
    assert report.actions_match is True
    assert report.surrogates_match is True
    assert report.policy_fingerprint_match is True
    assert report.non_idempotent_paths == ()
    assert report.to_dict() == check_idempotence(first, second).to_dict()


def test_omop_surrogate_change_is_classified_without_echoing_values() -> None:
    first = _omop_pass(surrogate="synthetic-subject-surrogate-a")
    second = _omop_pass(surrogate="synthetic-subject-surrogate-b")

    report = check_idempotence(first, second)
    payload = report.to_dict()
    serialized = json.dumps(payload, sort_keys=True)

    assert report.is_idempotent is False
    assert report.shape_match is True
    assert report.counts_match is True
    assert report.actions_match is True
    assert report.surrogates_match is False
    assert "$.tables.person[0].person_id" in report.non_idempotent_paths
    assert "$.tables.visit_occurrence[0].person_id" in report.non_idempotent_paths
    assert "synthetic-subject-surrogate-a" not in serialized
    assert "synthetic-subject-surrogate-b" not in serialized
    assert "surrogate" in serialized


def test_shape_change_and_count_change_are_reported_at_safe_paths() -> None:
    first = _fhir_pass(surrogate="[SYNTHETIC-NAME]")
    second = _fhir_pass(surrogate="[SYNTHETIC-NAME]")
    second_resource = second["resource"]
    assert isinstance(second_resource, dict)
    entries = second_resource["entry"]
    assert isinstance(entries, list)
    entries.append(
        {
            "fullUrl": "urn:synthetic:entry-b",
            "resource": {"resourceType": "Patient", "id": "synthetic-patient-b"},
        }
    )
    second_report = second["report"]
    assert isinstance(second_report, dict)
    second_report["counts"] = {"changed_value_count": 2, "redacted": 2}

    report = check_idempotence(first, second)
    dimensions = {item.dimension for item in report.differences}

    assert report.shape_match is False
    assert report.counts_match is False
    assert "shape" in dimensions
    assert "count" in dimensions
    assert "$.entry" in report.non_idempotent_paths
    assert "synthetic-patient-b" not in report.to_json()


@dataclass(frozen=True)
class _ResultObject:
    resource: object
    report: object


@dataclass(frozen=True)
class _ReportObject:
    policy_fingerprint: str
    counts: dict[str, int]
    redactions: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_fingerprint": self.policy_fingerprint,
            "counts": self.counts,
            "redactions": list(self.redactions),
        }


def test_result_objects_and_path_metadata_are_supported() -> None:
    resource = {
        "resourceType": "Observation",
        "component": [{"value": "synthetic-value"}],
    }
    report_data = _ReportObject(
        policy_fingerprint=_POLICY,
        counts={"redacted": 1},
        redactions=(
            {
                "path": "component[0].value",
                "action": "replace",
                "surrogate": "[SYNTHETIC-VALUE]",
            },
        ),
    )

    report = check_idempotence(
        _ResultObject(resource, report_data),
        _ResultObject(resource, report_data),
    )

    assert report.passed is True
    assert report.first_pass.events[0].path == "$.component[0].value"
    assert report.first_pass.events[0].surrogate_fingerprint.startswith("sha256:")


def test_unknown_action_and_policy_metadata_are_fingerprinted() -> None:
    first = {
        "resource": {"payload": "synthetic-payload-a"},
        "report": {
            "policy": "synthetic-private-policy-name",
            "redactions": [
                {
                    "path": "payload",
                    "action": "synthetic-private-action",
                    "surrogate": "synthetic-surrogate-a",
                }
            ],
        },
    }
    second = {
        "resource": {"payload": "synthetic-payload-a"},
        "report": {
            "policy": "synthetic-private-policy-name",
            "redactions": [
                {
                    "path": "payload",
                    "action": "synthetic-private-action",
                    "surrogate": "synthetic-surrogate-a",
                }
            ],
        },
    }

    report = check_idempotence(first, second)
    serialized = report.to_json()

    assert report.is_idempotent is True
    assert "synthetic-private-policy-name" not in serialized
    assert "synthetic-private-action" not in serialized
    assert "synthetic-surrogate-a" not in serialized
