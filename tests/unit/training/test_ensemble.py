"""Unit tests for teacher ensemble loader, registry, and agreement policies."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from openmed.training.ensemble import (
    AgreementPolicy,
    EnsembleConfigError,
    EnsembleManifestError,
    EnsembleMember,
    EnsembleValidatorError,
    FamilyEnsembleConfig,
    TeacherEnsembleConfig,
    build_span_validators,
    load_teacher_ensemble_config,
    resolve_family_agreement_policy,
    validate_ensemble_against_manifest,
)
from openmed.training.weak_labeling import WeakLabelSpan, weak_label_document


@pytest.fixture
def manifest_fixture(tmp_path: Path) -> Path:
    """Fixture supplying a populated models.jsonl file."""
    p = tmp_path / "models.jsonl"
    rows = [
        {"model_id": "OpenMed/OpenMed-PII-BigMed-Large-560M-v1"},
        {"model_id": "OpenMed/OpenMed-PII-BioClinicalBERT-Base-110M-v1"},
        {"model_id": "OpenMed/OpenMed-ClinicalNER-SuperClinical-Large-434M-v1"},
    ]
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


@pytest.fixture
def custom_yaml_fixture(tmp_path: Path) -> Path:
    """Fixture supplying a valid teacher ensemble configuration YAML file."""
    p = tmp_path / "teacher_ensemble.yaml"
    data = {
        "schema_version": "openmed.training.teacher_ensemble.v1",
        "families": {
            "ClinicalPrivacy": {
                "family": "ClinicalPrivacy",
                "description": "Test ClinicalPrivacy",
                "agreement_threshold": 0.60,
                "members": [
                    {
                        "id": "OpenMed/OpenMed-PII-BigMed-Large-560M-v1",
                        "type": "model",
                        "weight": 1.0,
                    },
                    {
                        "id": "OpenMed/OpenMed-PII-BioClinicalBERT-Base-110M-v1",
                        "type": "model",
                        "weight": 0.9,
                    },
                    {
                        "id": "privacy-filter-heuristics-v1",
                        "type": "filter",
                        "weight": 0.8,
                    },
                ],
                "validators": ["validate_ssn"],
            },
            "DirectID": {
                "family": "DirectID",
                "description": "Test DirectID",
                "agreement_threshold": 0.66,
                "members": [
                    {
                        "id": "OpenMed/OpenMed-ClinicalNER-SuperClinical-Large-434M-v1",
                        "type": "model",
                        "weight": 1.0,
                    },
                    {
                        "id": "direct-id-regex-v1",
                        "type": "filter",
                        "weight": 0.9,
                    },
                ],
                "validators": [
                    "validate_ssn",
                    "validate_npi",
                    "validate_uk_nhs_number",
                ],
            },
        },
    }
    p.write_text(yaml.dump(data), encoding="utf-8")
    return p


def test_load_default_teacher_ensemble_yaml() -> None:
    """Test loading the default repository teacher_ensemble.yaml configuration file."""
    config = load_teacher_ensemble_config()
    assert config.schema_version == "openmed.training.teacher_ensemble.v1"
    assert "ClinicalPrivacy" in config.families
    assert "DirectID" in config.families

    # Validate against actual models.jsonl
    validate_ensemble_against_manifest(config)


def test_default_seeded_families_present(custom_yaml_fixture: Path) -> None:
    """Test that ClinicalPrivacy and DirectID families are present with valid properties."""
    config = load_teacher_ensemble_config(custom_yaml_fixture)
    cp = config.families["ClinicalPrivacy"]
    direct = config.families["DirectID"]

    assert cp.family == "ClinicalPrivacy"
    assert cp.agreement_threshold == 0.60
    assert len(cp.members) == 3

    assert direct.family == "DirectID"
    assert direct.agreement_threshold == 0.66
    assert len(direct.members) == 2

    # Verify serialization
    cp_dict = cp.to_dict()
    assert cp_dict["family"] == "ClinicalPrivacy"
    assert len(cp_dict["members"]) == 3
    assert config.to_dict()["schema_version"] == "openmed.training.teacher_ensemble.v1"


def test_resolve_members_against_manifest_fixture(
    custom_yaml_fixture: Path, manifest_fixture: Path
) -> None:
    """Test resolving valid model members against a manifest fixture."""
    config = load_teacher_ensemble_config(custom_yaml_fixture)
    validate_ensemble_against_manifest(config, manifest_fixture)


def test_unknown_member_id_fails_manifest_validation(tmp_path: Path) -> None:
    """Test that declaring an unmanifested model ID raises EnsembleManifestError."""
    yaml_path = tmp_path / "bad_member.yaml"
    data = {
        "schema_version": "openmed.training.teacher_ensemble.v1",
        "families": {
            "ClinicalPrivacy": {
                "family": "ClinicalPrivacy",
                "agreement_threshold": 0.5,
                "members": [
                    {"id": "unmanifested-model-v1", "type": "model", "weight": 1.0}
                ],
                "validators": [],
            }
        },
    }
    yaml_path.write_text(yaml.dump(data), encoding="utf-8")
    config = load_teacher_ensemble_config(yaml_path)

    empty_manifest = tmp_path / "empty.jsonl"
    empty_manifest.write_text("", encoding="utf-8")

    with pytest.raises(EnsembleManifestError) as exc_info:
        validate_ensemble_against_manifest(config, empty_manifest)
    assert "was not found in the model manifest" in str(exc_info.value)


def test_unknown_validator_fails_validation(tmp_path: Path) -> None:
    """Test that declaring an unregistered validator function raises EnsembleValidatorError."""
    yaml_path = tmp_path / "bad_validator.yaml"
    data = {
        "schema_version": "openmed.training.teacher_ensemble.v1",
        "families": {
            "ClinicalPrivacy": {
                "family": "ClinicalPrivacy",
                "agreement_threshold": 0.5,
                "members": [{"id": "f1", "type": "filter", "weight": 1.0}],
                "validators": ["invalid_checksum_fn"],
            }
        },
    }
    yaml_path.write_text(yaml.dump(data), encoding="utf-8")

    with pytest.raises(EnsembleValidatorError) as exc_info:
        load_teacher_ensemble_config(yaml_path)
    assert "is not registered in VALIDATOR_REGISTRY" in str(exc_info.value)


@pytest.mark.parametrize("invalid_threshold", [0.0, -0.5, 1.2, float("nan")])
def test_out_of_range_threshold_fails(tmp_path: Path, invalid_threshold: float) -> None:
    """Test that agreement_threshold <= 0, > 1.0, or NaN raises EnsembleConfigError."""
    yaml_path = tmp_path / "bad_threshold.yaml"
    data = {
        "schema_version": "openmed.training.teacher_ensemble.v1",
        "families": {
            "ClinicalPrivacy": {
                "family": "ClinicalPrivacy",
                "agreement_threshold": invalid_threshold,
                "members": [{"id": "f1", "type": "filter", "weight": 1.0}],
                "validators": [],
            }
        },
    }
    yaml_path.write_text(yaml.dump(data), encoding="utf-8")

    with pytest.raises(EnsembleConfigError) as exc_info:
        load_teacher_ensemble_config(yaml_path)
    assert "Agreement threshold" in str(exc_info.value) or str(
        invalid_threshold
    ) in str(exc_info.value)


@pytest.mark.parametrize("invalid_weight", [0.0, -1.0, float("nan")])
def test_non_positive_weight_fails(tmp_path: Path, invalid_weight: float) -> None:
    """Test that member weight <= 0.0 or NaN raises EnsembleConfigError."""
    yaml_path = tmp_path / "bad_weight.yaml"
    data = {
        "schema_version": "openmed.training.teacher_ensemble.v1",
        "families": {
            "ClinicalPrivacy": {
                "family": "ClinicalPrivacy",
                "agreement_threshold": 0.5,
                "members": [{"id": "f1", "type": "filter", "weight": invalid_weight}],
                "validators": [],
            }
        },
    }
    yaml_path.write_text(yaml.dump(data), encoding="utf-8")

    with pytest.raises(EnsembleConfigError) as exc_info:
        load_teacher_ensemble_config(yaml_path)
    assert "Member weight" in str(exc_info.value)


def test_invalid_member_type_or_missing_id_fails() -> None:
    """Test invalid member types or empty IDs raise EnsembleConfigError."""
    with pytest.raises(EnsembleConfigError, match="Member ID cannot be empty"):
        EnsembleMember(id="", member_type="model", weight=1.0)

    with pytest.raises(EnsembleConfigError, match="Invalid member type"):
        EnsembleMember(id="m1", member_type="invalid_type", weight=1.0)


def test_invalid_schema_version_fails(tmp_path: Path) -> None:
    """Test unsupported schema version raises EnsembleConfigError."""
    yaml_path = tmp_path / "bad_version.yaml"
    data = {
        "schema_version": "openmed.training.teacher_ensemble.v999",
        "families": {
            "ClinicalPrivacy": {
                "family": "ClinicalPrivacy",
                "agreement_threshold": 0.5,
                "members": [{"id": "f1", "type": "filter", "weight": 1.0}],
                "validators": [],
            }
        },
    }
    yaml_path.write_text(yaml.dump(data), encoding="utf-8")
    with pytest.raises(EnsembleConfigError, match="Unsupported schema_version"):
        load_teacher_ensemble_config(yaml_path)


def test_agreement_policy_weak_label_integration(custom_yaml_fixture: Path) -> None:
    """End-to-end integration test passing resolved policy into weak_label_document()."""
    config = load_teacher_ensemble_config(custom_yaml_fixture)
    policy = resolve_family_agreement_policy(config, "ClinicalPrivacy")

    assert policy.min_agreeing_models == 2
    assert len(policy.validators) == 1
    assert policy.to_dict()["min_agreeing_models"] == 2

    # 123-45-6789 is a valid SSN (passes validate_ssn)
    valid_ssn = "123-45-6789"
    # 000-12-3456 is an invalid SSN (rejected by validate_ssn)
    invalid_ssn = "000-12-3456"

    text = f"{valid_ssn} ... {invalid_ssn} ... Alice"

    detector_outputs = {
        "OpenMed/OpenMed-PII-BigMed-Large-560M-v1": [
            {"start": 0, "end": 11, "label": "SSN", "text": valid_ssn, "score": 0.95},
            {
                "start": 16,
                "end": 27,
                "label": "SSN",
                "text": invalid_ssn,
                "score": 0.95,
            },
            {"start": 32, "end": 37, "label": "PERSON", "text": "Alice", "score": 0.90},
        ],
        "OpenMed/OpenMed-PII-BioClinicalBERT-Base-110M-v1": [
            {"start": 0, "end": 11, "label": "SSN", "text": valid_ssn, "score": 0.92},
            {
                "start": 16,
                "end": 27,
                "label": "SSN",
                "text": invalid_ssn,
                "score": 0.92,
            },
            {"start": 32, "end": 37, "label": "PERSON", "text": "Alice", "score": 0.88},
        ],
        "privacy-filter-heuristics-v1": [
            {"start": 0, "end": 11, "label": "SSN", "text": valid_ssn, "score": 0.80},
        ],
    }

    decision = weak_label_document(
        text=text,
        detector_outputs=detector_outputs,
        min_agreeing_models=policy.min_agreeing_models,
        validators=policy.validators,
    )

    accepted_texts = [s.text for s in decision.accepted_spans]
    assert valid_ssn in accepted_texts
    assert "Alice" in accepted_texts
    # Invalid SSN was agreed by 2 models but rejected by validate_ssn
    assert invalid_ssn not in accepted_texts

    rejected_texts = [s.text for s in decision.rejected_spans]
    assert invalid_ssn in rejected_texts


def test_directid_agreement_policy_integration(custom_yaml_fixture: Path) -> None:
    """Test DirectID policy with multiple checksum validators (NPI, NHS number)."""
    config = load_teacher_ensemble_config(custom_yaml_fixture)
    policy = resolve_family_agreement_policy(config, "DirectID")

    assert policy.family == "DirectID"
    assert policy.min_agreeing_models == 2

    valid_npi = "1234567893"
    invalid_npi = "1234567890"
    valid_nhs = "943 476 5919"
    invalid_nhs = "943 476 5910"

    text = f"{valid_npi} {invalid_npi} {valid_nhs} {invalid_nhs}"

    detector_outputs = {
        "OpenMed/OpenMed-ClinicalNER-SuperClinical-Large-434M-v1": [
            WeakLabelSpan(
                start=0,
                end=10,
                label="NPI",
                text=valid_npi,
                score=0.95,
                source="OpenMed/OpenMed-ClinicalNER-SuperClinical-Large-434M-v1",
            ),
            WeakLabelSpan(
                start=11,
                end=21,
                label="NPI",
                text=invalid_npi,
                score=0.95,
                source="OpenMed/OpenMed-ClinicalNER-SuperClinical-Large-434M-v1",
            ),
            WeakLabelSpan(
                start=22,
                end=34,
                label="NHS_NUMBER",
                text=valid_nhs,
                score=0.90,
                source="OpenMed/OpenMed-ClinicalNER-SuperClinical-Large-434M-v1",
            ),
            WeakLabelSpan(
                start=35,
                end=47,
                label="NHS_NUMBER",
                text=invalid_nhs,
                score=0.90,
                source="OpenMed/OpenMed-ClinicalNER-SuperClinical-Large-434M-v1",
            ),
        ],
        "direct-id-regex-v1": [
            WeakLabelSpan(
                start=0,
                end=10,
                label="NPI",
                text=valid_npi,
                score=0.90,
                source="direct-id-regex-v1",
            ),
            WeakLabelSpan(
                start=11,
                end=21,
                label="NPI",
                text=invalid_npi,
                score=0.90,
                source="direct-id-regex-v1",
            ),
            WeakLabelSpan(
                start=22,
                end=34,
                label="NHS_NUMBER",
                text=valid_nhs,
                score=0.85,
                source="direct-id-regex-v1",
            ),
            WeakLabelSpan(
                start=35,
                end=47,
                label="NHS_NUMBER",
                text=invalid_nhs,
                score=0.85,
                source="direct-id-regex-v1",
            ),
        ],
    }

    decision = weak_label_document(
        text=text,
        detector_outputs=detector_outputs,
        min_agreeing_models=policy.min_agreeing_models,
        validators=policy.validators,
    )

    accepted_texts = [s.text for s in decision.accepted_spans]
    assert valid_npi in accepted_texts
    assert valid_nhs in accepted_texts
    assert invalid_npi not in accepted_texts
    assert invalid_nhs not in accepted_texts

    rejected_texts = [s.text for s in decision.rejected_spans]
    assert invalid_npi in rejected_texts
    assert invalid_nhs in rejected_texts


@pytest.mark.parametrize(
    ("num_generators", "threshold", "expected_k"),
    [
        (2, 0.50, 2),
        (2, 0.60, 2),
        (3, 0.60, 2),
        (3, 0.70, 3),
        (4, 0.50, 2),
        (4, 0.75, 3),
        (5, 0.66, 4),
        (10, 0.20, 2),
        (10, 0.85, 9),
    ],
)
def test_agreement_policy_k_threshold_calculation(
    tmp_path: Path, num_generators: int, threshold: float, expected_k: int
) -> None:
    """Test mathematical mapping of continuous threshold to integer min_agreeing_models."""
    members = [
        {"id": f"gen_{i}", "type": "model", "weight": 1.0}
        for i in range(num_generators)
    ]
    data = {
        "schema_version": "openmed.training.teacher_ensemble.v1",
        "families": {
            "TestFamily": {
                "family": "TestFamily",
                "agreement_threshold": threshold,
                "members": members,
                "validators": [],
            }
        },
    }
    yaml_path = tmp_path / "k_calc.yaml"
    yaml_path.write_text(yaml.dump(data), encoding="utf-8")

    config = load_teacher_ensemble_config(yaml_path)
    policy = resolve_family_agreement_policy(config, "TestFamily")
    assert policy.min_agreeing_models == expected_k


def test_single_generator_member_fails_agreement_policy(tmp_path: Path) -> None:
    """Test that a family with N=1 generator member fails with EnsembleConfigError."""
    data = {
        "schema_version": "openmed.training.teacher_ensemble.v1",
        "families": {
            "SingleModelFamily": {
                "family": "SingleModelFamily",
                "agreement_threshold": 0.5,
                "members": [{"id": "single_model", "type": "model", "weight": 1.0}],
                "validators": [],
            }
        },
    }
    yaml_path = tmp_path / "single_gen.yaml"
    yaml_path.write_text(yaml.dump(data), encoding="utf-8")

    config = load_teacher_ensemble_config(yaml_path)
    with pytest.raises(
        EnsembleConfigError, match="requires at least 2 generator members"
    ):
        resolve_family_agreement_policy(config, "SingleModelFamily")


def test_zero_generator_members_fails_agreement_policy(tmp_path: Path) -> None:
    """Test that a family with N=0 generator members fails with EnsembleConfigError."""
    data = {
        "schema_version": "openmed.training.teacher_ensemble.v1",
        "families": {
            "NoGenFamily": {
                "family": "NoGenFamily",
                "agreement_threshold": 0.5,
                "members": [
                    {"id": "only_validator", "type": "validator", "weight": 1.0}
                ],
                "validators": [],
            }
        },
    }
    yaml_path = tmp_path / "no_gen.yaml"
    yaml_path.write_text(yaml.dump(data), encoding="utf-8")

    config = load_teacher_ensemble_config(yaml_path)
    with pytest.raises(
        EnsembleConfigError, match="requires at least 2 generator members"
    ):
        resolve_family_agreement_policy(config, "NoGenFamily")
