"""Tests for user-supplied AYUSH and Indian drug terminology grounding."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.clinical import (
    deduplicate_problem_list,
    problem_mentions_from_grounded_terms,
)
from openmed.clinical.exporters import (
    USER_SUPPLIED_TERMINOLOGY_PROVENANCE_EXTENSION_URL,
)
from openmed.clinical.normalization import (
    AYUSH,
    INDIAN_DRUG,
    IndiaTerminologyDictionary,
    IndiaTerminologyLoader,
)
from openmed.core.labels import (
    CLINICAL_CONCEPT,
    CONDITION,
    MEDICATION,
    normalize_label,
    policy_label_for,
)
from openmed.core.policy import PolicyProfile
from openmed.eval.datasets import (
    TERMINOLOGY_REDISTRIBUTION_PERMITTED,
    TERMINOLOGY_REDISTRIBUTION_RESTRICTED,
    RestrictedTerminologyLocationError,
    TerminologyLicense,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "clinical" / "india_terminology"
PUBLIC_LICENSE = TerminologyLicense(
    license_id="CC0-1.0",
    redistribution=TERMINOLOGY_REDISTRIBUTION_PERMITTED,
    notes="Synthetic public-domain test terms.",
)


def _synthetic_dictionaries() -> tuple[IndiaTerminologyDictionary, ...]:
    return (
        IndiaTerminologyDictionary(
            source_name="synthetic AYUSH example",
            kind=AYUSH,
            license=PUBLIC_LICENSE,
            path=FIXTURE_ROOT / "ayush.csv",
            system_uri="https://example.test/CodeSystem/ayush",
            version="synthetic-1",
        ),
        IndiaTerminologyDictionary(
            source_name="synthetic Indian drug example",
            kind=INDIAN_DRUG,
            license=PUBLIC_LICENSE,
            path=FIXTURE_ROOT / "indian_drug.csv",
            system_uri="https://example.test/CodeSystem/indian-drug",
            version="synthetic-1",
        ),
    )


def _load_synthetic():
    result = IndiaTerminologyLoader().load(_synthetic_dictionaries())
    assert result.skipped == ()
    return result.terminology


def test_synthetic_ayush_and_drug_surfaces_map_to_canonical_labels():
    terminology = _load_synthetic()

    ayush = terminology.ground_surface("VATA IMBALANCE", start=12)[0]
    drug = terminology.ground_surface("lotus brand tablet", start=40)[0]

    assert ayush.canonical_label == CONDITION
    assert ayush.code == "SYN-AYUSH-001"
    assert ayush.start == 12
    assert ayush.end == 26
    assert drug.canonical_label == MEDICATION
    assert drug.code == "SYN-INDIA-DRUG-001"
    assert normalize_label("AYUSH_MORBIDITY") == CONDITION
    assert normalize_label("NAMASTE morbidity") == CONDITION
    assert normalize_label("INDIAN_DRUG_BRAND") == MEDICATION


def test_grounded_terms_export_source_license_and_assist_only_provenance():
    terminology = _load_synthetic()

    for term in (
        terminology.ground_surface("vata imbalance")[0],
        terminology.ground_surface("lotus brand tablet")[0],
    ):
        concept = term.to_codeable_concept()
        coding = concept["coding"][0]
        provenance = next(
            extension
            for extension in coding["extension"]
            if extension["url"] == USER_SUPPLIED_TERMINOLOGY_PROVENANCE_EXTENSION_URL
        )
        values = {
            item["url"]: item.get("valueString", item.get("valueBoolean"))
            for item in provenance["extension"]
        }

        assert coding["code"] == term.code
        assert coding["version"] == "synthetic-1"
        assert values["sourceName"] == term.source_name
        assert values["license"] == "CC0-1.0"
        assert values["restricted"] is False
        assert values["assistOnly"] is True
        assert "not a clinical coding" in values["disclaimer"]


def test_ayush_grounding_flows_into_problem_list_but_medication_does_not():
    grounded = _load_synthetic().ground_text(
        "Vata imbalance was noted; lotus brand tablet was reviewed."
    )

    mentions = problem_mentions_from_grounded_terms(grounded)
    problems = deduplicate_problem_list(mentions)

    assert len(mentions) == 1
    assert mentions[0].code == "SYN-AYUSH-001"
    assert problems[0].system == "https://example.test/CodeSystem/ayush"
    assert problems[0].clinical_status == "active"


def test_grounded_labels_follow_india_dpdp_clinical_concept_action_contract():
    terminology = _load_synthetic()
    grounded = (
        terminology.ground_surface("vata imbalance")[0],
        terminology.ground_surface("lotus brand tablet")[0],
    )
    india_profile = PolicyProfile(
        name="india_dpdp_act",
        schema_version=1,
        posture="india_dpdp_act_deidentification",
        threshold_profile="balanced",
        default_action="replace",
        default_action_bias="replace",
        arbitration_mode="balanced",
        safety_sweep_mandatory=True,
        keep_mapping=False,
        reversible_id=False,
        forced_cascade_tiers=("R0", "R1", "R2", "R3"),
        actions={},
        policy_label_actions={CLINICAL_CONCEPT: "keep"},
    )

    for term in grounded:
        assert policy_label_for(term.canonical_label) == CLINICAL_CONCEPT
        assert india_profile.action_for(term.canonical_label) == "keep"


def test_grounding_does_not_reintroduce_patient_identifier():
    identifier = "2345 6789 0123"
    text = f"Patient Aadhaar {identifier}; vata imbalance; lotus brand tablet reviewed."

    grounded = _load_synthetic().ground_text(text)
    exported = [term.to_codeable_concept() for term in grounded]

    assert [term.canonical_label for term in grounded] == [CONDITION, MEDICATION]
    assert identifier not in json.dumps(exported, sort_keys=True)
    assert all(identifier not in term.text for term in grounded)


def test_absent_dictionaries_degrade_with_explicit_skip_and_no_leak(tmp_path):
    result = IndiaTerminologyLoader().load(
        (
            IndiaTerminologyDictionary(
                source_name="missing AYUSH",
                kind=AYUSH,
                license=PUBLIC_LICENSE,
                path=None,
            ),
            IndiaTerminologyDictionary(
                source_name="missing Indian drug",
                kind=INDIAN_DRUG,
                license=PUBLIC_LICENSE,
                path=tmp_path / "does-not-exist.csv",
            ),
        )
    )

    assert result.terminology.term_count == 0
    assert [notice.reason for notice in result.skipped] == [
        "dictionary_absent",
        "dictionary_absent",
    ]
    assert result.terminology.ground_text("Aadhaar 2345 6789 0123") == ()
    assert result.terminology.audit_summary() == {"term_count": 0, "sources": []}


def test_restricted_dictionary_stays_external_and_out_of_logs(caplog, tmp_path):
    secret_surface = "restricted secret vata syndrome"
    secret_code = "RESTRICTED-SECRET-674"
    dictionary_path = tmp_path / "restricted-ayush.csv"
    dictionary_path.write_text(
        f"code,display,aliases\n{secret_code},Secret concept,{secret_surface}\n",
        encoding="utf-8",
    )
    restricted_license = TerminologyLicense(
        license_id="user-contract-restricted",
        redistribution=TERMINOLOGY_REDISTRIBUTION_RESTRICTED,
    )

    with caplog.at_level("DEBUG"):
        result = IndiaTerminologyLoader().load(
            (
                IndiaTerminologyDictionary(
                    source_name="local restricted AYUSH",
                    kind=AYUSH,
                    license=restricted_license,
                    path=dictionary_path,
                ),
            )
        )
        grounded = result.terminology.ground_surface(secret_surface)[0]
        audit_json = json.dumps(grounded.to_audit_dict(), sort_keys=True)
        safe_repr = repr(grounded) + repr(result.terminology)

    fixture_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in (REPO_ROOT / "tests" / "fixtures").rglob("*")
        if path.is_file()
    )
    logged = "\n".join(record.getMessage() for record in caplog.records)

    assert grounded.restricted is True
    assert secret_surface not in audit_json
    assert secret_code not in audit_json
    assert str(dictionary_path) not in audit_json
    assert secret_surface not in safe_repr
    assert secret_code not in safe_repr
    assert secret_surface not in logged
    assert secret_code not in logged
    assert secret_surface not in fixture_text
    assert secret_code not in fixture_text


def test_restricted_dictionary_inside_repository_is_rejected():
    restricted_license = TerminologyLicense(
        license_id="user-contract-restricted",
        redistribution=TERMINOLOGY_REDISTRIBUTION_RESTRICTED,
    )

    with pytest.raises(
        RestrictedTerminologyLocationError,
        match="outside the OpenMed repository",
    ):
        IndiaTerminologyLoader().load(
            (
                IndiaTerminologyDictionary(
                    source_name="misplaced restricted AYUSH",
                    kind=AYUSH,
                    license=restricted_license,
                    path=FIXTURE_ROOT / "ayush.csv",
                ),
            )
        )
