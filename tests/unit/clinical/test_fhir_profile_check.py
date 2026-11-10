"""Tests for the offline SMART Guidelines profile checker (OM-874)."""

from __future__ import annotations

import copy
import json
import shutil
import socket
from dataclasses import dataclass
from pathlib import Path

import pytest

from openmed.clinical.exporters.fhir import check_bundle, to_bundle
from openmed.interop.fhir_operations import de_identify_bundle
from openmed.interop.gateway import assert_redacted

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "smart_profiles"
PATIENT_PROFILE = "https://openmed.example/fhir/StructureDefinition/smart-anc-patient"
OBSERVATION_PROFILE = (
    "https://openmed.example/fhir/StructureDefinition/smart-anc-observation"
)


def _resources() -> list[dict]:
    return [
        {
            "resourceType": "Patient",
            "id": "patient-1",
            "meta": {"profile": [f"{PATIENT_PROFILE}|0.0.1"]},
            "identifier": [
                {
                    "system": "https://openmed.example/smart-anc/identifier",
                    "value": "ANC-001",
                }
            ],
            "name": [{"family": "Roe"}],
        },
        {
            "resourceType": "Observation",
            "id": "observation-1",
            "meta": {"profile": [OBSERVATION_PROFILE]},
            "status": "final",
            "category": [
                {
                    "coding": [
                        {
                            "system": ("https://openmed.example/smart-anc/category"),
                            "code": "anc-contact",
                            "display": "ANC contact",
                        }
                    ],
                    "text": "Antenatal contact",
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": ("https://openmed.example/smart-anc/observations"),
                        "code": "anc-weight",
                    }
                ]
            },
        },
    ]


def _bundle(resources: list[dict] | None = None) -> dict:
    return to_bundle(resources or _resources(), doc_id="synthetic-anc")


def _failures(outcome: dict) -> list[dict]:
    return [
        issue
        for issue in outcome["issue"]
        if issue["severity"] in {"fatal", "error", "warning"}
    ]


def _expressions(outcome: dict) -> set[str]:
    return {
        expression
        for issue in outcome["issue"]
        for expression in issue.get("expression", [])
    }


def test_compliant_synthetic_anc_bundle_passes():
    outcome = check_bundle(_bundle(), FIXTURE_ROOT)

    assert outcome == {
        "resourceType": "OperationOutcome",
        "issue": [
            {
                "severity": "information",
                "code": "informational",
                "diagnostics": "No issues detected.",
            }
        ],
    }


def test_missing_required_element_has_precise_fhirpath_location():
    resources = _resources()
    del resources[0]["name"]

    outcome = check_bundle(_bundle(resources), FIXTURE_ROOT)

    assert _failures(outcome) == [
        {
            "severity": "error",
            "code": "required",
            "diagnostics": "Required element cardinality is not met.",
            "expression": ["Bundle.entry[0].resource.name"],
        }
    ]


def test_wrong_fixed_code_has_precise_fhirpath_location():
    resources = _resources()
    resources[1]["status"] = "preliminary"

    outcome = check_bundle(_bundle(resources), FIXTURE_ROOT)

    assert _failures(outcome) == [
        {
            "severity": "error",
            "code": "value",
            "diagnostics": (
                "Element does not match the profile's fixedCode constraint."
            ),
            "expression": ["Bundle.entry[1].resource.status"],
        }
    ]


def test_required_identifier_and_category_slices_are_checked():
    resources = _resources()
    resources[0]["identifier"] = []
    resources[1]["category"][0]["coding"][0]["code"] = "other"

    outcome = check_bundle(_bundle(resources), FIXTURE_ROOT)

    assert _expressions(outcome) == {
        "Bundle.entry[0].resource.identifier",
        "Bundle.entry[1].resource.category",
    }
    assert all(issue["code"] == "required" for issue in _failures(outcome))


def test_required_slice_descendant_is_checked():
    resources = _resources()
    del resources[0]["identifier"][0]["value"]

    outcome = check_bundle(_bundle(resources), FIXTURE_ROOT)

    assert _failures(outcome) == [
        {
            "severity": "error",
            "code": "required",
            "diagnostics": "Required element cardinality is not met.",
            "expression": ["Bundle.entry[0].resource.identifier[0].value"],
        }
    ]


def test_local_required_binding_is_checked():
    resources = _resources()
    resources[1]["code"]["coding"][0]["code"] = "not-in-local-value-set"

    outcome = check_bundle(_bundle(resources), FIXTURE_ROOT)

    assert _failures(outcome) == [
        {
            "severity": "error",
            "code": "code-invalid",
            "diagnostics": (
                "Coded element is outside the locally available required binding."
            ),
            "expression": ["Bundle.entry[1].resource.code"],
        }
    ]


def test_systemless_coding_does_not_match_a_system_specific_value_set():
    resources = _resources()
    del resources[1]["code"]["coding"][0]["system"]

    outcome = check_bundle(_bundle(resources), FIXTURE_ROOT)

    assert _failures(outcome) == [
        {
            "severity": "error",
            "code": "code-invalid",
            "diagnostics": (
                "Coded element is outside the locally available required binding."
            ),
            "expression": ["Bundle.entry[1].resource.code"],
        }
    ]


def test_partial_value_set_expansion_is_non_blocking(tmp_path):
    ig_root = tmp_path / "ig"
    shutil.copytree(FIXTURE_ROOT / "package", ig_root / "package")
    value_set_path = ig_root / "package" / "ValueSet-smart-anc-observation-codes.json"
    value_set = json.loads(value_set_path.read_text(encoding="utf-8"))
    value_set["expansion"] = {
        "offset": 0,
        "total": 2,
        "contains": [
            {
                "system": "https://openmed.example/smart-anc/observations",
                "code": "first-page-code",
            }
        ],
    }
    value_set_path.write_text(json.dumps(value_set), encoding="utf-8")

    outcome = check_bundle(_bundle(), ig_root)

    assert _failures(outcome) == []
    assert any(
        issue["severity"] == "information" and issue["code"] == "not-supported"
        for issue in outcome["issue"]
    )


def test_whole_code_system_compose_is_not_treated_as_empty_value_set(tmp_path):
    ig_root = tmp_path / "ig"
    shutil.copytree(FIXTURE_ROOT / "package", ig_root / "package")
    value_set_path = ig_root / "package" / "ValueSet-smart-anc-observation-codes.json"
    value_set = json.loads(value_set_path.read_text(encoding="utf-8"))
    value_set["compose"]["include"][0]["concept"] = []
    value_set_path.write_text(json.dumps(value_set), encoding="utf-8")

    outcome = check_bundle(_bundle(), ig_root)

    assert _failures(outcome) == []
    assert any(
        issue["severity"] == "information" and issue["code"] == "not-supported"
        for issue in outcome["issue"]
    )


def test_abstract_expansion_code_is_not_valid_for_selection(tmp_path):
    ig_root = tmp_path / "ig"
    shutil.copytree(FIXTURE_ROOT / "package", ig_root / "package")
    value_set_path = ig_root / "package" / "ValueSet-smart-anc-observation-codes.json"
    value_set = json.loads(value_set_path.read_text(encoding="utf-8"))
    value_set["expansion"] = {
        "total": 1,
        "contains": [
            {
                "system": "https://openmed.example/smart-anc/observations",
                "code": "anc-weight",
                "abstract": True,
            }
        ],
    }
    value_set_path.write_text(json.dumps(value_set), encoding="utf-8")

    outcome = check_bundle(_bundle(), ig_root)

    assert [issue["code"] for issue in _failures(outcome)] == ["code-invalid"]


def test_slice_matching_uses_only_the_declared_discriminator_path(tmp_path):
    ig_root = tmp_path / "ig"
    shutil.copytree(FIXTURE_ROOT / "package", ig_root / "package")
    profile_path = ig_root / "package" / "StructureDefinition-smart-anc-patient.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["snapshot"]["element"].append(
        {
            "id": "Patient.identifier:anc-id.value",
            "path": "Patient.identifier.value",
            "fixedString": "required-synthetic-value",
        }
    )
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    outcome = check_bundle(_bundle(), ig_root)

    assert _failures(outcome) == [
        {
            "severity": "error",
            "code": "value",
            "diagnostics": (
                "Element does not match the profile's fixedString constraint."
            ),
            "expression": ["Bundle.entry[0].resource.identifier[0].value"],
        }
    ]


def test_value_discriminator_accepts_a_pattern_constraint(tmp_path):
    ig_root = tmp_path / "ig"
    shutil.copytree(FIXTURE_ROOT / "package", ig_root / "package")
    profile_path = ig_root / "package" / "StructureDefinition-smart-anc-patient.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    system_element = profile["snapshot"]["element"][3]
    system_element["patternUri"] = system_element.pop("fixedUri")
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    outcome = check_bundle(_bundle(), ig_root)

    assert _failures(outcome) == []


@dataclass
class _DeidentificationResult:
    deidentified_text: str


def _remove_family_name(text, **_kwargs):
    return _DeidentificationResult("" if text == "Roe" else text)


def _pseudonymize_family_name(text, **_kwargs):
    return _DeidentificationResult("PERSON" if text == "Roe" else text)


def test_post_deid_audit_flags_exactly_the_removed_required_element():
    original = _bundle()
    deidentified = de_identify_bundle(
        original,
        method="remove",
        deidentifier=_remove_family_name,
    )

    outcome = check_bundle(
        deidentified,
        FIXTURE_ROOT,
        original_bundle=original,
    )

    assert _failures(outcome) == [
        {
            "severity": "error",
            "code": "required",
            "diagnostics": (
                "De-identification introduced profile violation: Required "
                "element cardinality is not met."
            ),
            "expression": ["Bundle.entry[0].resource.name[0].family"],
        }
    ]


def test_pseudonymize_policy_variant_preserves_required_element():
    original = _bundle()
    deidentified = de_identify_bundle(
        original,
        method="replace",
        deidentifier=_pseudonymize_family_name,
    )

    outcome = check_bundle(
        deidentified,
        FIXTURE_ROOT,
        original_bundle=original,
    )

    assert _failures(outcome) == []


def test_post_deid_audit_marks_unchanged_violations_as_preexisting():
    resources = _resources()
    resources[0]["identifier"] = []
    original = _bundle(resources)
    deidentified = de_identify_bundle(
        original,
        method="replace",
        deidentifier=_pseudonymize_family_name,
    )

    outcome = check_bundle(
        deidentified,
        FIXTURE_ROOT,
        original_bundle=original,
    )

    failures = _failures(outcome)
    assert len(failures) == 1
    assert failures[0]["diagnostics"].startswith("Pre-existing profile violation:")
    assert failures[0]["expression"] == ["Bundle.entry[0].resource.identifier"]


def test_outcome_never_quotes_checked_phi_values():
    resources = _resources()
    resources[0]["name"][0]["family"] = "Jane Roe"
    resources[1]["status"] = "Jane Roe"
    resources[1]["code"]["coding"][0]["code"] = "Jane Roe"

    outcome = check_bundle(_bundle(resources), FIXTURE_ROOT)
    serialized = json.dumps(outcome, sort_keys=True)

    assert_redacted(serialized, {"Jane Roe": "PERSON"})
    assert "Jane Roe" not in serialized


def test_checker_is_pure_and_performs_no_network_access(monkeypatch):
    bundle = _bundle()
    snapshot = copy.deepcopy(bundle)

    def reject_network(*_args, **_kwargs):
        raise AssertionError("profile checker attempted network access")

    monkeypatch.setattr(socket, "create_connection", reject_network)
    check_bundle(bundle, FIXTURE_ROOT)

    assert bundle == snapshot


def test_unknown_profile_constraint_is_informational(tmp_path):
    ig_root = tmp_path / "ig"
    shutil.copytree(FIXTURE_ROOT / "package", ig_root / "package")
    profile_path = (
        ig_root / "package" / "StructureDefinition-smart-anc-observation.json"
    )
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["snapshot"]["element"][1]["constraint"] = [
        {"key": "synthetic-invariant", "expression": "status.exists()"}
    ]
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    outcome = check_bundle(_bundle(), ig_root)

    assert _failures(outcome) == []
    assert any(
        issue["severity"] == "information" and issue["code"] == "not-supported"
        for issue in outcome["issue"]
    )


def test_bundle_profile_check_hook_is_opt_in_and_cannot_mutate_output():
    outcomes = []

    def profile_hook(candidate):
        outcomes.append(check_bundle(candidate, FIXTURE_ROOT))
        candidate["type"] = "history"

    with_hook = to_bundle(
        _resources(),
        doc_id="synthetic-anc",
        profile_check=profile_hook,
    )
    without_hook = to_bundle(_resources(), doc_id="synthetic-anc")

    assert outcomes and _failures(outcomes[0]) == []
    assert with_hook == without_hook
    assert with_hook["type"] == "transaction"


def test_bundle_profile_check_hook_rejects_non_callable():
    with pytest.raises(TypeError, match="profile_check must be callable"):
        to_bundle(_resources(), profile_check=FIXTURE_ROOT)  # type: ignore[arg-type]
