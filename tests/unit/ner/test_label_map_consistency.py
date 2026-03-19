"""Label-map consistency tests.

Validates invariants across NER label maps, PII label normalization,
and model registry entity_types to catch configuration drift.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.core.pii_entity_merger import normalize_label, is_more_specific
from openmed.core.model_registry import OPENMED_MODELS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LABEL_MAPS_PATH = (
    Path(__file__).resolve().parents[3]
    / "openmed"
    / "zero_shot"
    / "data"
    / "label_maps"
    / "defaults.json"
)


@pytest.fixture
def label_maps():
    with open(LABEL_MAPS_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# defaults.json invariants
# ---------------------------------------------------------------------------


class TestDefaultsJsonInvariants:

    def test_every_domain_has_at_least_one_label(self, label_maps):
        for domain, labels in label_maps.items():
            assert len(labels) >= 1, f"Domain {domain!r} has no labels"

    def test_no_duplicate_labels_within_domain(self, label_maps):
        for domain, labels in label_maps.items():
            lower_labels = [l.lower() for l in labels]
            assert len(lower_labels) == len(set(lower_labels)), (
                f"Domain {domain!r} has duplicate labels (case-insensitive): {labels}"
            )

    def test_generic_domain_exists(self, label_maps):
        assert "generic" in label_maps, (
            "defaults.json must include a 'generic' fallback domain"
        )

    def test_generic_domain_has_labels(self, label_maps):
        assert len(label_maps["generic"]) >= 1


# ---------------------------------------------------------------------------
# normalize_label idempotency
# ---------------------------------------------------------------------------


class TestNormalizeLabelIdempotency:

    @pytest.mark.parametrize("label", [
        "date_of_birth",
        "phone_number",
        "email",
        "ssn",
        "social_security_number",
        "national_id",
        "nir",
        "insee",
        "steuer_id",
        "steuernummer",
        "codice_fiscale",
        "postcode",
        "zipcode",
        "zip",
        "postal_code",
        "address",
        "street_address",
        "fax",
        "first_name",
        "last_name",
        "date",
        "phone",
    ])
    def test_normalize_is_idempotent(self, label):
        once = normalize_label(label)
        twice = normalize_label(once)
        assert once == twice, (
            f"normalize_label is not idempotent for {label!r}: "
            f"first={once!r}, second={twice!r}"
        )


# ---------------------------------------------------------------------------
# Specificity hierarchy resolves to valid canonical forms
# ---------------------------------------------------------------------------


class TestSpecificityHierarchy:

    HIERARCHY = {
        "date": ["date_of_birth", "date_time"],
        "name": ["first_name", "last_name", "full_name"],
        "phone": ["phone_number", "fax_number", "mobile_number"],
        "address": ["street_address", "home_address", "billing_address"],
        "id": ["ssn", "medical_record_number", "account_number", "employee_id"],
        "national_id": ["nir", "insee", "steuer_id", "steuernummer", "codice_fiscale"],
    }

    def test_specific_labels_resolve_to_canonical(self):
        for general, specifics in self.HIERARCHY.items():
            for specific in specifics:
                canonical = normalize_label(specific)
                assert isinstance(canonical, str) and len(canonical) > 0, (
                    f"Specific label {specific!r} did not normalize to a valid string"
                )

    def test_is_more_specific_agrees_with_hierarchy(self):
        for general, specifics in self.HIERARCHY.items():
            for specific in specifics:
                assert is_more_specific(specific, general), (
                    f"is_more_specific({specific!r}, {general!r}) should be True"
                )

    def test_general_not_more_specific_than_specific(self):
        for general, specifics in self.HIERARCHY.items():
            for specific in specifics:
                assert not is_more_specific(general, specific), (
                    f"is_more_specific({general!r}, {specific!r}) should be False"
                )


# ---------------------------------------------------------------------------
# Model registry entity_types recognized by normalize_label
# ---------------------------------------------------------------------------


class TestModelRegistryEntityTypes:

    def _pii_models(self):
        return {
            key: info for key, info in OPENMED_MODELS.items()
            if key.startswith("pii_") and info.entity_types
        }

    def test_all_pii_entity_types_recognized(self):
        """Every entity_type in PII model entries should normalize without error."""
        pii_models = self._pii_models()
        assert len(pii_models) > 0, "No PII models found in OPENMED_MODELS"
        for key, info in pii_models.items():
            for et in info.entity_types:
                canonical = normalize_label(et)
                assert isinstance(canonical, str) and len(canonical) > 0, (
                    f"entity_type {et!r} in {key!r} did not normalize"
                )

    def test_pii_entity_types_normalize_idempotently(self):
        """Normalization of registry entity_types must be idempotent."""
        for key, info in self._pii_models().items():
            for et in info.entity_types:
                once = normalize_label(et)
                twice = normalize_label(once)
                assert once == twice, (
                    f"normalize_label not idempotent for {et!r} (model {key!r}): "
                    f"{once!r} != {twice!r}"
                )

    def test_at_least_one_pii_model_per_supported_language(self):
        """Every supported language should have at least one PII model in the registry."""
        from openmed.core.pii_i18n import SUPPORTED_LANGUAGES
        pii_keys = [k for k in OPENMED_MODELS if k.startswith("pii_")]
        for lang in SUPPORTED_LANGUAGES:
            if lang == "en":
                # English models don't have a language infix
                assert any(
                    k.startswith("pii_") and not any(
                        f"pii_{l}_" in k for l in SUPPORTED_LANGUAGES if l != "en"
                    )
                    for k in pii_keys
                ), "No English PII model found"
            else:
                # Non-English keys use pii_{lang}_ prefix, e.g. pii_de_superclinical_small
                assert any(
                    k.startswith(f"pii_{lang}_") for k in pii_keys
                ), f"No PII model found for language {lang!r}"
