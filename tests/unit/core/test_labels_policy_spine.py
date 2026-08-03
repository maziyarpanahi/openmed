"""Tests for policy metadata attached to canonical labels."""

from collections import Counter

from openmed.core.labels import (
    AGE,
    CANONICAL_LABELS,
    CHINESE_DRUG,
    CHINESE_ICD_10,
    CLINICAL_CONCEPT,
    CLINICAL_SYSTEM_HINTS,
    CONDITION,
    DIRECT_IDENTIFIER,
    ETHNICITY,
    GI_SYMPTOM,
    HIPAA_SAFE_HARBOR_CLASSES,
    LABEL_METADATA,
    LABEL_TO_HIPAA,
    MEDICATION,
    NDPA_RACE_OR_ETHNIC_ORIGIN,
    OTHER,
    POLICY_LABELS,
    POPIA_DEMOGRAPHIC_AND_HISTORY_ATTRIBUTES,
    QUASI_IDENTIFIER,
    RISK_LEVELS,
    RISK_MEDIUM,
    SENSITIVE_ATTRIBUTE,
    ZIPCODE,
    hipaa_class_for,
    ndpa_classes_for,
    normalize_label,
    policy_label_for,
    popia_class_for,
    risk_level_for,
    system_hints_for,
)

ALLOWED_SYSTEM_HINTS = set(CLINICAL_SYSTEM_HINTS) | {
    CHINESE_DRUG,
    CHINESE_ICD_10,
}

POLICY_SPINE_BASELINE_LABELS = frozenset(
    {
        "PERSON",
        "FIRST_NAME",
        "LAST_NAME",
        "MIDDLE_NAME",
        "PREFIX",
        "USERNAME",
        "EMAIL",
        "PHONE",
        "URL",
        "LOCATION",
        "STREET_ADDRESS",
        "BUILDING_NUMBER",
        "ZIPCODE",
        "GPS_COORDINATES",
        "ORDINAL_DIRECTION",
        "DATE",
        "DATE_OF_BIRTH",
        "TIME",
        "AGE",
        "ID_NUM",
        "SSN",
        "ACCOUNT_NUMBER",
        "PASSWORD",
        "PIN",
        "API_KEY",
        "CREDIT_CARD",
        "CREDIT_CARD_ISSUER",
        "CVV",
        "IBAN",
        "BIC",
        "AMOUNT",
        "CURRENCY",
        "BITCOIN_ADDRESS",
        "ETHEREUM_ADDRESS",
        "LITECOIN_ADDRESS",
        "MASKED_NUMBER",
        "GENDER",
        "ETHNICITY",
        "EYE_COLOR",
        "HEIGHT",
        "ORGANIZATION",
        "JOB_TITLE",
        "JOB_DEPARTMENT",
        "OCCUPATION",
        "IP_ADDRESS",
        "MAC_ADDRESS",
        "USER_AGENT",
        "VIN",
        "VEHICLE_REGISTRATION",
        "IMEI",
    }
)


def test_metadata_tables_cover_canonical_labels_exactly():
    assert len(POLICY_SPINE_BASELINE_LABELS) == 50
    assert POLICY_SPINE_BASELINE_LABELS <= CANONICAL_LABELS
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


def test_chinese_system_hints_are_scoped_to_supported_labels():
    assert CHINESE_ICD_10 in system_hints_for(CONDITION)
    assert CHINESE_DRUG in system_hints_for(MEDICATION)

    for generic_label in (GI_SYMPTOM, OTHER):
        assert CHINESE_ICD_10 not in system_hints_for(generic_label)
        assert CHINESE_DRUG not in system_hints_for(generic_label)


def test_acceptance_specific_direct_and_quasi_labels():
    assert policy_label_for("ID_NUM") == DIRECT_IDENTIFIER
    for label in (AGE, "DATE", ZIPCODE):
        assert label in CANONICAL_LABELS
        assert policy_label_for(label) == QUASI_IDENTIFIER
        assert risk_level_for(label) == RISK_MEDIUM


def test_ethnicity_resolves_to_sensitive_attribute_policy_class():
    assert normalize_label("ethnic_origin") == ETHNICITY
    assert normalize_label("tribal_affiliation") == ETHNICITY
    assert policy_label_for(ETHNICITY) == SENSITIVE_ATTRIBUTE
    assert popia_class_for(ETHNICITY) == POPIA_DEMOGRAPHIC_AND_HISTORY_ATTRIBUTES
    assert NDPA_RACE_OR_ETHNIC_ORIGIN in ndpa_classes_for(ETHNICITY)


def test_label_to_hipaa_is_many_to_one():
    class_counts = Counter(LABEL_TO_HIPAA.values())

    assert len(HIPAA_SAFE_HARBOR_CLASSES) == 18
    assert set(LABEL_TO_HIPAA.values()) <= HIPAA_SAFE_HARBOR_CLASSES
    assert len(class_counts) < len(LABEL_TO_HIPAA)
    assert any(count > 1 for count in class_counts.values())
