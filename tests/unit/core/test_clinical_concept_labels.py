"""Regression tests for the relation-extraction clinical label vocabulary."""

import pytest

from openmed.core.labels import (
    ABNORMAL_FLAG,
    BODY_SITE,
    CANONICAL_LABELS,
    CLINICAL_CONCEPT,
    CLINICAL_CONCEPT_LABELS,
    DOSAGE,
    DURATION,
    FORM,
    FREQUENCY,
    INDICATION,
    LAB_VALUE,
    MEDICATION,
    PROBLEM,
    REFERENCE_RANGE,
    RISK_LOW,
    RISK_MEDIUM,
    ROUTE,
    SEVERITY,
    STRENGTH,
    UNIT,
    normalize_label,
    policy_label_for,
    risk_level_for,
)
from openmed.core.policy import load_policy

RELATION_LABELS = frozenset(
    {
        PROBLEM,
        MEDICATION,
        DOSAGE,
        ROUTE,
        FREQUENCY,
        DURATION,
        FORM,
        STRENGTH,
        INDICATION,
        LAB_VALUE,
        UNIT,
        REFERENCE_RANGE,
        ABNORMAL_FLAG,
        BODY_SITE,
        SEVERITY,
    }
)

INTRODUCED_LABELS = RELATION_LABELS - {MEDICATION, BODY_SITE}

# Import-stable PII/PHI vocabulary that existed before clinical labels were
# added. This snapshot prevents additive clinical work from removing or
# reclassifying one of the original 50 canonical labels.
ORIGINAL_PII_LABELS = frozenset(
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


def test_exported_relation_vocabulary_is_canonical_and_policy_classified():
    assert CLINICAL_CONCEPT_LABELS == RELATION_LABELS
    assert INTRODUCED_LABELS <= CANONICAL_LABELS

    for label in CLINICAL_CONCEPT_LABELS:
        assert label in CANONICAL_LABELS
        assert normalize_label(label) == label
        assert policy_label_for(label) == CLINICAL_CONCEPT


def test_free_text_and_distinctive_values_are_quasi_identifier_aware():
    medium_risk = {PROBLEM, INDICATION, LAB_VALUE, ABNORMAL_FLAG}
    assert {
        label for label in RELATION_LABELS if risk_level_for(label) == RISK_MEDIUM
    } == (medium_risk)
    assert all(
        risk_level_for(label) == RISK_LOW for label in RELATION_LABELS - medium_risk
    )


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("dx", PROBLEM),
        ("diagnosis", PROBLEM),
        ("problem", PROBLEM),
        ("problem list item", PROBLEM),
        ("med", MEDICATION),
        ("drug", MEDICATION),
        ("dose", DOSAGE),
        ("dosage", DOSAGE),
        ("freq", FREQUENCY),
        ("frequency", FREQUENCY),
        ("route of administration", ROUTE),
        ("duration", DURATION),
        ("dose form", FORM),
        ("strength", STRENGTH),
        ("indication", INDICATION),
        ("lab value", LAB_VALUE),
        ("unit", UNIT),
        ("reference range", REFERENCE_RANGE),
        ("abnormal flag", ABNORMAL_FLAG),
        ("body site", BODY_SITE),
        ("severity", SEVERITY),
    ],
)
def test_common_surface_forms_normalize(alias, expected):
    assert normalize_label(alias) == expected


def test_original_pii_vocabulary_and_normalization_remain_stable():
    assert len(ORIGINAL_PII_LABELS) == 50
    assert ORIGINAL_PII_LABELS <= CANONICAL_LABELS
    assert ORIGINAL_PII_LABELS.isdisjoint(INTRODUCED_LABELS)

    assert normalize_label("first_name") == "FIRST_NAME"
    assert normalize_label("FIRSTNAME", lang="pt") == "FIRST_NAME"
    assert normalize_label("B-EMAIL") == "EMAIL"
    assert normalize_label("ssn") == "SSN"


def test_relation_labels_are_not_redacted_by_the_default_clinical_policy():
    profile = load_policy("clinical_minimal_redaction")

    assert profile.action_for("PERSON") == "mask"
    assert all(profile.action_for(label) == "keep" for label in RELATION_LABELS)
