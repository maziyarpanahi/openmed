"""Tests for policy metadata attached to canonical labels."""

from collections import Counter

from openmed.core.labels import (
    AGE,
    CANONICAL_LABELS,
    CLINICAL_CONCEPT,
    DIRECT_IDENTIFIER,
    HIPAA_SAFE_HARBOR_CLASSES,
    LABEL_METADATA,
    LABEL_TO_HIPAA,
    POLICY_LABELS,
    QUASI_IDENTIFIER,
    RISK_LEVELS,
    RISK_MEDIUM,
    ZIPCODE,
    hipaa_class_for,
    normalize_label,
    policy_label_for,
    risk_level_for,
    system_hints_for,
)

ALLOWED_SYSTEM_HINTS = {"RxNorm", "CVX", "LOINC", "ICD-10-CM", "HPO", "SNOMED"}


def test_metadata_tables_cover_canonical_labels_exactly():
    assert len(CANONICAL_LABELS) == 88
    assert set(LABEL_METADATA) == CANONICAL_LABELS
    assert set(LABEL_TO_HIPAA) == CANONICAL_LABELS


def test_every_label_resolves_policy_risk_hipaa_and_hints():
    for label in CANONICAL_LABELS:
        policy_label = policy_label_for(label)
        risk_level = risk_level_for(label)
        system_hints = system_hints_for(label)
        hipaa_class = hipaa_class_for(label)

        assert policy_label in POLICY_LABELS
        assert risk_level in RISK_LEVELS
        assert hipaa_class in HIPAA_SAFE_HARBOR_CLASSES
        assert isinstance(system_hints, tuple)

        if policy_label == CLINICAL_CONCEPT:
            assert system_hints
            assert set(system_hints) <= ALLOWED_SYSTEM_HINTS
        else:
            assert system_hints == ()


def test_accessors_normalize_before_lookup():
    assert policy_label_for("medical_record_number") == DIRECT_IDENTIFIER
    assert policy_label_for("FIRSTNAME", lang="pt") == DIRECT_IDENTIFIER
    assert hipaa_class_for("B-EMAIL") == "EMAIL_ADDRESS"

    assert normalize_label("not_a_real_label") == "OTHER"
    assert policy_label_for("not_a_real_label") == CLINICAL_CONCEPT
    assert set(system_hints_for("not_a_real_label")) <= ALLOWED_SYSTEM_HINTS


def test_acceptance_specific_direct_and_quasi_labels():
    assert policy_label_for("ID_NUM") == DIRECT_IDENTIFIER
    for label in (AGE, "DATE", ZIPCODE):
        assert label in CANONICAL_LABELS
        assert policy_label_for(label) == QUASI_IDENTIFIER
        assert risk_level_for(label) == RISK_MEDIUM


def test_label_to_hipaa_is_many_to_one():
    class_counts = Counter(LABEL_TO_HIPAA.values())

    assert len(HIPAA_SAFE_HARBOR_CLASSES) == 18
    assert set(LABEL_TO_HIPAA.values()) <= HIPAA_SAFE_HARBOR_CLASSES
    assert len(class_counts) < len(LABEL_TO_HIPAA)
    assert any(count > 1 for count in class_counts.values())
