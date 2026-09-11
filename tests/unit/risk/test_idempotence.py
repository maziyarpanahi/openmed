"""Focused tests for nested structured-redaction idempotence checks."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from openmed.risk import (
    IdempotenceDifference,
    IdempotenceInputError,
    RedactionEvent,
    ShapeNode,
    check_idempotence,
)

_POLICY = "sha256:" + "a" * 64


class _ExplodingMapping(Mapping[str, Any]):
    def __getitem__(self, key: str) -> Any:
        raise RuntimeError("synthetic-sensitive-exception")

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError("synthetic-sensitive-exception")

    def __len__(self) -> int:
        return 1


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


def test_scalar_change_without_event_metadata_is_not_idempotent() -> None:
    first_value = "synthetic-private-first-value"
    second_value = "synthetic-private-second-value"

    report = check_idempotence(
        {"resource": {"patient_name": first_value}},
        {"resource": {"patient_name": second_value}},
    )
    serialized = report.to_json()

    assert report.is_idempotent is False
    assert report.surrogates_match is False
    assert any(item.path.startswith("$.key:") for item in report.differences)
    assert first_value not in serialized
    assert second_value not in serialized
    assert "patient_name" not in serialized


def test_cycles_depth_and_hostile_mappings_fail_with_sanitized_errors() -> None:
    cyclic: list[object] = []
    cyclic.append(cyclic)
    with pytest.raises(IdempotenceInputError):
        check_idempotence(cyclic, [])

    nested: object = None
    for _ in range(70):
        nested = [nested]
    with pytest.raises(IdempotenceInputError):
        check_idempotence(nested, [])

    with pytest.raises(IdempotenceInputError) as error:
        check_idempotence(_ExplodingMapping(), {})
    assert "synthetic-sensitive-exception" not in str(error.value)


def test_duplicate_json_keys_and_nonfinite_values_are_rejected(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"resource":{"id":"synthetic-a","id":"synthetic-b"}}',
        encoding="utf-8",
    )
    with pytest.raises(IdempotenceInputError):
        check_idempotence(duplicate, duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"resource":{"value":NaN}}', encoding="utf-8")
    with pytest.raises(IdempotenceInputError):
        check_idempotence(nonfinite, nonfinite)


def test_ambiguous_resource_and_metadata_aliases_are_rejected() -> None:
    with pytest.raises(IdempotenceInputError):
        check_idempotence(
            {"resource": {"id": "synthetic-a"}, "data": {"id": "synthetic-b"}},
            {"resource": {"id": "synthetic-a"}},
        )

    ambiguous_event = {
        "resource": {"id": "synthetic-a"},
        "report": {
            "redactions": [
                {
                    "path": "$.id",
                    "action": "replace",
                    "operation": "remove",
                }
            ]
        },
    }
    with pytest.raises(IdempotenceInputError):
        check_idempotence(ambiguous_event, ambiguous_event)


def test_public_evidence_objects_reject_raw_values() -> None:
    with pytest.raises(ValueError):
        RedactionEvent(path="$.synthetic_private_name")
    with pytest.raises(ValueError):
        ShapeNode(path="$", kind="object", keys=("synthetic_private_name",))
    with pytest.raises(ValueError):
        IdempotenceDifference(
            dimension="surrogate",
            path="$.id",
            before="synthetic-private-first-value",
            after="synthetic-private-second-value",
            classification="changed",
        )

    report = check_idempotence({"resource": {}}, {"resource": {}})
    with pytest.raises(ValueError):
        replace(
            report.first_pass,
            shape_fingerprint="sha256:synthetic-private-shape",
        )


@pytest.mark.parametrize(
    ("resource_type", "field", "value"),
    [("Binary", "data", "U1lOVEhFVElD"), ("DiagnosticReport", "result", [])],
)
def test_bare_fhir_resource_fields_are_not_wrapper_aliases(resource_type, field, value):
    first = {"resourceType": resource_type, "id": "synthetic-first", field: value}
    second = first | {"id": "synthetic-second"}
    report = check_idempotence(first, second)
    assert not report.passed
    assert "$.id" in report.non_idempotent_paths
    assert "synthetic-first" not in report.to_json()
    assert "synthetic-second" not in report.to_json()
