"""Offline tests for the FHIR Bundle reference-integrity report."""

from __future__ import annotations

import copy
import json

import pytest

from openmed.interop.fhir.reference_integrity import reference_integrity_report


def _valid_bundle() -> dict:
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "fullUrl": "Patient/synthetic-patient",
                "resource": {
                    "resourceType": "Patient",
                    "id": "synthetic-patient",
                    "contained": [
                        {
                            "resourceType": "Organization",
                            "id": "synthetic-organization",
                        }
                    ],
                    "managingOrganization": {"reference": "#synthetic-organization"},
                },
            },
            {
                "fullUrl": "Observation/synthetic-observation",
                "resource": {
                    "resourceType": "Observation",
                    "id": "synthetic-observation",
                    "subject": {"reference": "Patient/synthetic-patient"},
                },
            },
        ],
    }


@pytest.mark.parametrize("version", ["R4", "R5"])
def test_valid_r4_and_r5_bundle_is_deterministic(version: str) -> None:
    bundle = _valid_bundle()

    first = reference_integrity_report(bundle, fhir_version=version)
    second = reference_integrity_report(copy.deepcopy(bundle), fhir_version=version)

    assert first == second
    assert first.valid is True
    assert first.entry_count == 2
    assert first.resource_count == 2
    assert first.contained_resource_count == 1
    assert first.full_url_count == 2
    assert first.reference_count == 2
    assert first.findings == ()


def test_report_counts_duplicate_and_dangling_references_without_values() -> None:
    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "fullUrl": "Patient/synthetic-duplicate",
                "resource": {
                    "resourceType": "Patient",
                    "id": "synthetic-duplicate",
                    "contained": [
                        {"resourceType": "Device", "id": "synthetic-local"},
                        {"resourceType": "Observation", "id": "synthetic-local"},
                    ],
                    "link": {"other": {"reference": "Patient/synthetic-missing"}},
                },
            },
            {
                "fullUrl": "Patient/synthetic-duplicate",
                "resource": {
                    "resourceType": "Patient",
                    "id": "synthetic-duplicate",
                },
            },
            {
                "fullUrl": "Patient/synthetic-stale",
                "resource": {
                    "resourceType": "Patient",
                    "id": "synthetic-current",
                },
            },
            {
                "fullUrl": "Observation/synthetic-observation",
                "resource": {
                    "resourceType": "Observation",
                    "id": "synthetic-observation",
                    "subject": {"reference": "Patient/synthetic-duplicate"},
                },
            },
        ],
    }

    report = reference_integrity_report(bundle)

    assert report.valid is False
    assert report.counts == {
        "ambiguous_reference": 1,
        "dangling_reference": 1,
        "duplicate_contained_identity": 1,
        "duplicate_full_url": 1,
        "duplicate_resource_identity": 1,
        "full_url_identity_mismatch": 1,
    }
    assert report.to_dict()["findings"] == [
        {
            "code": "ambiguous_reference",
            "count": 1,
            "paths": [
                "Bundle.entry[3].resource.subject.reference",
            ],
        },
        {
            "code": "dangling_reference",
            "count": 1,
            "paths": [
                "Bundle.entry[0].resource.link.other.reference",
            ],
        },
        {
            "code": "duplicate_contained_identity",
            "count": 1,
            "paths": [
                "Bundle.entry[0].resource.contained[1]",
            ],
        },
        {
            "code": "duplicate_full_url",
            "count": 1,
            "paths": ["Bundle.entry[1].fullUrl"],
        },
        {
            "code": "duplicate_resource_identity",
            "count": 1,
            "paths": ["Bundle.entry[1].resource"],
        },
        {
            "code": "full_url_identity_mismatch",
            "count": 1,
            "paths": ["Bundle.entry[2].fullUrl"],
        },
    ]

    serialized = json.dumps(report.to_dict(), sort_keys=True)
    assert "synthetic-duplicate" not in serialized
    assert "synthetic-current" not in serialized
    assert "synthetic-missing" not in serialized


def test_report_does_not_mutate_bundle_or_leak_invalid_input_values() -> None:
    bundle = _valid_bundle()
    before = copy.deepcopy(bundle)
    report = reference_integrity_report(bundle)

    assert bundle == before
    assert report.to_json() == report.to_json()

    with pytest.raises(ValueError) as error:
        reference_integrity_report(
            {"resourceType": "Patient", "id": "synthetic-secret"}
        )
    assert "synthetic-secret" not in str(error.value)


def test_invalid_reference_is_reported_at_a_stable_path() -> None:
    bundle = {
        "resourceType": "Bundle",
        "entry": [
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "synthetic-observation",
                    "subject": {"reference": ""},
                }
            }
        ],
    }

    report = reference_integrity_report(bundle)

    assert report.counts == {"invalid_reference": 1}
    assert report.findings[0].paths == ("Bundle.entry[0].resource.subject.reference",)
