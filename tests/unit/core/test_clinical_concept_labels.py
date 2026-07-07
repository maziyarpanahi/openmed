"""Tests for the clinical-concept relation-extraction labels (issue #252).

OM-087 adds the head/attribute label vocabulary that clinical relation
extraction (OM-043) and SDOH (OM-056) bind together. These tests assert that
the new labels:

- are importable and members of :data:`CANONICAL_LABELS`;
- round-trip through :func:`normalize_label`;
- resolve ``policy_label_for == CLINICAL_CONCEPT`` via the OM-010 accessors;
- carry non-empty coding-system hints and a HIPAA Safe Harbor class;
- are excluded from the default de-identification redaction set (they resolve
  to a non-redacting action under the clinical minimal-redaction policy);
- do NOT collide with, or change the behavior of, the pre-existing PII labels
  and the ``CONDITION`` grounding aliases.
"""

import pytest

from openmed.core.labels import (
    ABNORMAL_FLAG,
    BODY_SITE,
    CANONICAL_LABELS,
    CLINICAL_CONCEPT,
    CLINICAL_CONCEPT_LABELS,
    CONDITION,
    DIRECT_IDENTIFIER,
    DOSAGE,
    DURATION,
    FORM,
    FREQUENCY,
    HIPAA_SAFE_HARBOR_CLASSES,
    INDICATION,
    LAB_VALUE,
    MEDICATION,
    PERSON,
    PROBLEM,
    QUASI_IDENTIFIER,
    REFERENCE_RANGE,
    ROUTE,
    SEVERITY,
    STATUS,
    STRENGTH,
    UNIT,
    hipaa_class_for,
    normalize_label,
    policy_label_for,
    system_hints_for,
)
from openmed.core.policy import load_policy

# The 14 relation-extraction labels introduced by OM-087. ``MEDICATION`` and
# ``BODY_SITE`` are relation heads/attributes too, but they pre-date OM-087 and
# already live in CANONICAL_LABELS, so they are intentionally not in this set.
NEW_LABELS = (
    PROBLEM,
    SEVERITY,
    STATUS,
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
)


class TestClinicalConceptRelationLabels:
    """The additive relation-extraction label vocabulary (OM-087)."""

    def test_exported_set_matches_new_labels(self):
        assert CLINICAL_CONCEPT_LABELS == frozenset(NEW_LABELS)
        assert len(CLINICAL_CONCEPT_LABELS) == 14

    def test_new_labels_are_canonical(self):
        for label in NEW_LABELS:
            assert label in CANONICAL_LABELS
            assert label in CLINICAL_CONCEPT_LABELS

    def test_new_labels_round_trip(self):
        for label in NEW_LABELS:
            assert normalize_label(label) == label

    def test_new_labels_resolve_clinical_concept(self):
        # Acceptance criterion: every new label resolves to CLINICAL_CONCEPT.
        for label in NEW_LABELS:
            assert policy_label_for(label) == CLINICAL_CONCEPT

    def test_new_labels_have_grounding_hints_and_hipaa_class(self):
        for label in NEW_LABELS:
            assert system_hints_for(label)  # non-empty coding-system hints
            assert hipaa_class_for(label) in HIPAA_SAFE_HARBOR_CLASSES

    @pytest.mark.parametrize(
        "alias,expected",
        [
            ("dx", PROBLEM),
            ("problem", PROBLEM),
            ("problem list item", PROBLEM),
            ("active problem", PROBLEM),
            ("severity", SEVERITY),
            ("status", STATUS),
            ("clinical status", STATUS),
            ("dosage", DOSAGE),
            ("dose", DOSAGE),
            ("dosing", DOSAGE),
            ("route", ROUTE),
            ("route of administration", ROUTE),
            ("frequency", FREQUENCY),
            ("freq", FREQUENCY),
            ("duration", DURATION),
            ("form", FORM),
            ("dose form", FORM),
            ("strength", STRENGTH),
            ("indication", INDICATION),
            ("lab value", LAB_VALUE),
            ("lab result", LAB_VALUE),
            ("unit", UNIT),
            ("uom", UNIT),
            ("reference range", REFERENCE_RANGE),
            ("normal range", REFERENCE_RANGE),
            ("abnormal flag", ABNORMAL_FLAG),
        ],
    )
    def test_aliases_resolve(self, alias, expected):
        assert normalize_label(alias) == expected


class TestNoCollisionWithPiiLabels:
    """The new labels must not perturb the pre-existing label space."""

    def test_new_labels_are_additive(self):
        # The 50 PII direct/quasi identifiers plus the earlier clinical
        # concepts are unchanged; OM-087 only appends 14 labels.
        assert len(CANONICAL_LABELS) == 93
        assert frozenset(NEW_LABELS) <= CANONICAL_LABELS

    def test_no_new_label_is_a_direct_or_quasi_identifier(self):
        for label in NEW_LABELS:
            assert policy_label_for(label) not in (
                DIRECT_IDENTIFIER,
                QUASI_IDENTIFIER,
            )

    def test_condition_grounding_aliases_unchanged(self):
        # OM-087 must not change how CONDITION synonyms normalize.
        for alias in ("diagnosis", "disease", "disorder", "finding", "syndrome"):
            assert normalize_label(alias) == CONDITION

    def test_existing_pii_labels_unchanged(self):
        assert normalize_label("first_name") == "FIRST_NAME"
        assert normalize_label("FIRSTNAME", lang="pt") == "FIRST_NAME"
        assert normalize_label("B-EMAIL") == "EMAIL"
        assert normalize_label("ssn") == "SSN"
        assert policy_label_for("PERSON") == DIRECT_IDENTIFIER
        assert policy_label_for("ID_NUM") == DIRECT_IDENTIFIER


class TestExcludedFromDefaultRedaction:
    """Clinical concepts are kept, not redacted, under a clinical policy."""

    def test_new_labels_kept_under_clinical_minimal_redaction(self):
        # Under the clinical minimal-redaction posture, direct identifiers are
        # masked but clinical concepts (including the new relation labels)
        # resolve to a non-redacting ``keep`` action via policy_label fallback.
        profile = load_policy("clinical_minimal_redaction")

        assert profile.action_for(PERSON) == "mask"  # control: PII is redacted

        for label in NEW_LABELS:
            assert profile.action_for(label) == "keep"

    def test_new_labels_track_existing_clinical_concepts(self):
        # The new labels are handled exactly like the pre-existing clinical
        # concepts (MEDICATION, BODY_SITE, CONDITION) under the same policy.
        profile = load_policy("clinical_minimal_redaction")
        clinical_action = profile.action_for(MEDICATION)
        assert profile.action_for(BODY_SITE) == clinical_action
        assert profile.action_for(CONDITION) == clinical_action
        for label in NEW_LABELS:
            assert profile.action_for(label) == clinical_action
