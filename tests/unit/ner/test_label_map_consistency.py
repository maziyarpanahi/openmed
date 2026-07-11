"""Label-map consistency tests.

Validates invariants across NER label maps, PII label normalization,
and model registry entity_types to catch configuration drift.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from openmed.core.labels import (
    CANONICAL_LABELS,
    policy_label_for,
    system_hints_for,
)
from openmed.core.labels import normalize_label as normalize_canonical_label
from openmed.core.model_registry import (
    _CATEGORY_ENTITY_TYPES,
    OPENMED_MODELS,
    _match_categories,
    get_model_suggestions,
)
from openmed.core.pii_entity_merger import is_more_specific, normalize_label
from openmed.ner.labels import (
    available_domains,
    get_default_labels,
    load_default_label_map,
)

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
CLINICAL_FIXTURES_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "clinical"


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

    def test_labels_follow_consistent_style(self, label_maps):
        """Every domain label is a single letters-only token (no spaces/digits)."""
        for domain, labels in label_maps.items():
            for label in labels:
                assert re.fullmatch(r"[A-Za-z]+", label), (
                    f"Domain {domain!r} label {label!r} drifts from the "
                    "letters-only display-label style"
                )


def test_load_default_label_map_rejects_malformed_override(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON in label file") as exc_info:
        load_default_label_map(path)

    assert str(path) in str(exc_info.value)


# ---------------------------------------------------------------------------
# Cardiology domain (issue #317)
# ---------------------------------------------------------------------------


class TestCardiologyDomain:
    EXPECTED_LABELS = [
        "CardiacFinding",
        "ECGFinding",
        "EjectionFraction",
        "CardiacProcedure",
        "CardiacDevice",
        "Anatomy",
    ]

    def test_cardiology_in_available_domains(self):
        assert "cardiology" in available_domains()

    def test_get_default_labels_returns_cardiology_set(self):
        labels = get_default_labels("cardiology")
        assert labels  # non-empty
        assert labels == self.EXPECTED_LABELS

    def test_cardiology_labels_have_no_duplicates(self):
        labels = get_default_labels("cardiology")
        lowered = [label.lower() for label in labels]
        assert len(lowered) == len(set(lowered))


# ---------------------------------------------------------------------------
# Cardiology routing in model_registry (issue #317)
# ---------------------------------------------------------------------------


class TestCardiologyRouting:
    CARDIO_TEXT = "Echocardiogram shows reduced ejection fraction of 35%"

    def test_match_categories_routes_cardiology(self):
        categories = [
            category for category, _reason in _match_categories(self.CARDIO_TEXT)
        ]
        assert "Cardiology" in categories

    def test_cardiology_is_registry_metadata_not_a_live_category(self):
        # Forward metadata for future models; no Cardiology model exists today.
        assert "Cardiology" in _CATEGORY_ENTITY_TYPES
        from openmed.core.model_registry import CATEGORIES

        assert "Cardiology" not in CATEGORIES

    def test_get_model_suggestions_behavior_unchanged_for_cardiology(self):
        # With no Cardiology model registered, suggestions fall back to general
        # medical models rather than surfacing unrelated cardiology results.
        suggestions = get_model_suggestions(self.CARDIO_TEXT)
        assert suggestions  # still returns something useful
        assert all(info.category != "Cardiology" for _key, info, _reason in suggestions)


# ---------------------------------------------------------------------------
# Microbiology domain (issue #314)
# ---------------------------------------------------------------------------


class TestMicrobiologyDomain:
    EXPECTED_LABELS = [
        "Microorganism",
        "Antibiotic",
        "Susceptibility",
        "SpecimenSource",
        "CultureResult",
    ]

    MICROBIOLOGY_TEXT = "Blood culture grew MRSA, resistant to oxacillin"

    def test_microbiology_in_available_domains(self):
        assert "microbiology" in available_domains()

    def test_get_default_labels_returns_microbiology_set(self):
        labels = get_default_labels("microbiology")
        assert labels == self.EXPECTED_LABELS

    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            ("susceptibility", "SUSCEPTIBILITY"),
            ("antibiotic", "ANTIBIOTIC"),
            ("microorganism", "MICROORGANISM"),
            ("organism", "MICROORGANISM"),
        ],
    )
    def test_microbiology_labels_normalize(self, label, expected):
        assert normalize_canonical_label(label) == expected

    def test_match_categories_routes_microbiology(self):
        categories = [
            category for category, _reason in _match_categories(self.MICROBIOLOGY_TEXT)
        ]
        assert categories[0] == "Microbiology"

    def test_microbiology_is_registry_metadata_not_a_live_category(self):
        # Domain metadata is ready before a dedicated Microbiology model is
        # registered, matching the Cardiology forward-metadata pattern.
        assert "Microbiology" in _CATEGORY_ENTITY_TYPES
        from openmed.core.model_registry import CATEGORIES

        assert "Microbiology" not in CATEGORIES

    def test_get_model_suggestions_still_returns_live_models(self):
        suggestions = get_model_suggestions(self.MICROBIOLOGY_TEXT)
        assert suggestions
        assert all(
            info.category != "Microbiology" for _key, info, _reason in suggestions
        )


# ---------------------------------------------------------------------------
# Dermatology and ophthalmology domains (issue #318)
# ---------------------------------------------------------------------------


class TestSpecialtyDomains:
    EXPECTED = {
        "dermatology": ["SkinLesion", "Morphology", "Distribution", "Anatomy"],
        "ophthalmology": [
            "EyeFinding",
            "VisualAcuity",
            "IntraocularPressure",
            "Anatomy",
        ],
    }

    def test_domains_in_available_domains(self):
        domains = available_domains()
        assert "dermatology" in domains
        assert "ophthalmology" in domains

    def test_get_default_labels_returns_expected_sets(self):
        for domain, expected in self.EXPECTED.items():
            labels = get_default_labels(domain)
            assert labels  # non-empty
            assert labels == expected

    def test_specialty_labels_have_no_duplicates(self):
        for domain in self.EXPECTED:
            labels = get_default_labels(domain)
            lowered = [label.lower() for label in labels]
            assert len(lowered) == len(set(lowered))


# ---------------------------------------------------------------------------
# Dermatology and ophthalmology routing in model_registry (issue #318)
# ---------------------------------------------------------------------------


class TestSpecialtyRouting:
    DERM_TEXT = "Erythematous macule with pruritus noted on the forearm"
    OPHTH_TEXT = (
        "Fundus exam shows elevated intraocular pressure consistent with glaucoma"
    )

    def test_match_categories_routes_dermatology(self):
        categories = [
            category for category, _reason in _match_categories(self.DERM_TEXT)
        ]
        assert "Dermatology" in categories

    def test_match_categories_routes_ophthalmology(self):
        categories = [
            category for category, _reason in _match_categories(self.OPHTH_TEXT)
        ]
        assert "Ophthalmology" in categories

    def test_specialties_are_registry_metadata_not_live_categories(self):
        # Forward metadata for future models; no specialty model exists today.
        from openmed.core.model_registry import CATEGORIES

        for category in ("Dermatology", "Ophthalmology"):
            assert category in _CATEGORY_ENTITY_TYPES
            assert category not in CATEGORIES

    def test_get_model_suggestions_behavior_unchanged_for_specialties(self):
        for text in (self.DERM_TEXT, self.OPHTH_TEXT):
            suggestions = get_model_suggestions(text)
            assert suggestions  # still returns something useful
            assert all(
                info.category not in {"Dermatology", "Ophthalmology"}
                for _key, info, _reason in suggestions
            )


# ---------------------------------------------------------------------------
# Nutrition and Diet-Order domains (issue #951)
# ---------------------------------------------------------------------------
class TestNutritionDomain:
    EXPECTED_LABELS = [
        "DietType",
        "NutritionTarget",
        "Supplement",
        "FeedingRoute",
        "IntakeFinding",
        "NutritionalStatus",
        "FluidRestriction",
    ]

    def test_nutrition_in_available_domains(self):
        assert "nutrition_diet" in available_domains()

    def test_get_default_labels_returns_nutrition_set(self):
        assert get_default_labels("nutrition_diet") == self.EXPECTED_LABELS

    def test_nutrition_labels_have_no_duplicates(self):
        labels = get_default_labels("nutrition_diet")
        lowered = [l.lower() for l in labels]
        assert len(lowered) == len(set(lowered))


# ---------------------------------------------------------------------------
# Nutrition and Diet-Order routing in model_registry (issue #951)
# ---------------------------------------------------------------------------


class TestNutritionRouting:
    NUTRITION_TEXT = "1800 kcal diabetic diet, PEG feeds at 60 mL/hr"

    def test_match_categories_routes_nutrition(self):
        categories = [c for c, _ in _match_categories(self.NUTRITION_TEXT)]
        assert "Nutrition" in categories

    def test_nutrition_is_registry_metadata_not_a_live_category(self):
        assert "Nutrition" in _CATEGORY_ENTITY_TYPES
        from openmed.core.model_registry import CATEGORIES

        assert "Nutrition" not in CATEGORIES

    def test_get_model_suggestions_behavior_unchanged_for_nutrition(self):
        suggestions = get_model_suggestions(self.NUTRITION_TEXT)
        assert suggestions
        assert all(info.category != "Nutrition" for _k, info, _r in suggestions)


# ---------------------------------------------------------------------------
# Immunization and Vaccine-Administration domain (issue #897)
# ---------------------------------------------------------------------------
class TestImmunizationDomain:
    EXPECTED_LABELS = [
        "VaccineName",
        "DoseNumber",
        "AdministrationRoute",
        "AdministrationSite",
        "VaccineLot",
        "AdministrationDate",
        "VaccineSeries",
    ]

    def test_immunization_in_available_domains(self):
        assert "immunization" in available_domains()

    def test_get_default_labels_returns_immunization_set(self):
        assert get_default_labels("immunization") == self.EXPECTED_LABELS

    def test_immunization_labels_have_no_duplicates(self):
        labels = get_default_labels("immunization")
        lowered = [l.lower() for l in labels]
        assert len(lowered) == len(set(lowered))


# ---------------------------------------------------------------------------
# Immunization and Vaccine-Administration routing in model_registry (issue #897)
# ---------------------------------------------------------------------------
class TestImmunizationRouting:
    IMMUNIZATION_TEXT = "Tdap, dose 1 of 1, IM, left deltoid"

    def test_match_categories_routes_immunization(self):
        categories = [c for c, _ in _match_categories(self.IMMUNIZATION_TEXT)]
        assert "Immunization" in categories

    def test_immunization_is_registry_metadata_not_a_live_category(self):
        assert "Immunization" in _CATEGORY_ENTITY_TYPES

        from openmed.core.model_registry import CATEGORIES

        assert "Immunization" not in CATEGORIES

    def test_get_model_suggestions_behavior_unchanged_for_immunization(self):
        suggestions = get_model_suggestions(self.IMMUNIZATION_TEXT)
        assert suggestions
        assert all(info.category != "Immunization" for _k, info, _r in suggestions)


# ---------------------------------------------------------------------------
# normalize_label idempotency
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Anesthesia domain (issue #952)
# ---------------------------------------------------------------------------


class TestAnesthesiaDomain:
    EXPECTED_LABELS = [
        "AnesthesiaType",
        "AnestheticAgent",
        "AirwayManagement",
        "ASAClass",
        "MonitoringModality",
        "IntraoperativeEvent",
    ]
    CANONICAL_LABELS_BY_DISPLAY = {
        "AnesthesiaType": "ANESTHESIA_TYPE",
        "AnestheticAgent": "ANESTHETIC_AGENT",
        "AirwayManagement": "AIRWAY_MANAGEMENT",
        "ASAClass": "ASA_CLASS",
        "MonitoringModality": "OTHER",
        "IntraoperativeEvent": "OTHER",
    }

    def test_anesthesia_in_available_domains(self):
        assert "anesthesia" in available_domains()

    def test_get_default_labels_returns_anesthesia_set(self):
        assert get_default_labels("anesthesia") == self.EXPECTED_LABELS

    def test_anesthesia_labels_have_no_duplicates(self):
        labels = get_default_labels("anesthesia")
        lowered = [l.lower() for l in labels]
        assert len(lowered) == len(set(lowered))

    @pytest.mark.parametrize(
        ("label", "expected"),
        sorted(CANONICAL_LABELS_BY_DISPLAY.items()),
    )
    def test_anesthesia_labels_normalize_to_canonical(self, label, expected):
        assert normalize_canonical_label(label) == expected

        if expected != "OTHER":
            assert expected in CANONICAL_LABELS
            assert policy_label_for(expected) == "CLINICAL_CONCEPT"
            assert system_hints_for(expected)

    def test_anesthesia_fixture_reports_per_label_coverage(self):
        path = CLINICAL_FIXTURES_PATH / "anesthesia.jsonl"
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert rows

        observed_labels = set()
        for row in rows:
            assert row["metadata"]["synthetic"] is True
            disclaimer = row["metadata"]["disclaimer"]
            assert "not clinical guidance" in disclaimer
            assert "does not infer ASA class or dosing" in disclaimer
            text = row["text"]
            for entity in row["entities"]:
                start = entity["start"]
                end = entity["end"]
                assert text[start:end] == entity["text"]
                observed_labels.add(entity["label"])

        assert observed_labels == set(self.EXPECTED_LABELS)


# ---------------------------------------------------------------------------
# Anesthesia routing in model_registry (issue #952)
# ---------------------------------------------------------------------------


class TestAnesthesiaRouting:
    ANESTHESIA_TEXT = "General anesthesia with sevoflurane, ETT, and ASA II"

    def test_match_categories_routes_anesthesia(self):
        categories = [c for c, _ in _match_categories(self.ANESTHESIA_TEXT)]
        assert "Anesthesia" in categories

    def test_anesthesia_is_registry_metadata_not_a_live_category(self):
        assert "Anesthesia" in _CATEGORY_ENTITY_TYPES
        from openmed.core.model_registry import CATEGORIES

        assert "Anesthesia" not in CATEGORIES

    def test_get_model_suggestions_behavior_unchanged_for_anesthesia(self):
        suggestions = get_model_suggestions(self.ANESTHESIA_TEXT)
        assert suggestions
        assert all(info.category != "Anesthesia" for _k, info, _r in suggestions)


# ---------------------------------------------------------------------------
# Endocrinology domain (issue #895)
# ---------------------------------------------------------------------------


class TestEndocrinologyDomain:
    """Endocrinology domain labels, canonical mappings, and fixture coverage."""

    EXPECTED_LABELS = [
        "GlycemicMeasure",
        "ThyroidFunctionMeasure",
        "HormoneLevel",
        "InsulinRegimen",
        "MetabolicFinding",
        "EndocrineGland",
    ]
    CANONICAL_LABELS_BY_DISPLAY = {
        "GlycemicMeasure": "GLYCEMIC_MEASURE",
        "ThyroidFunctionMeasure": "THYROID_MEASURE",
        "HormoneLevel": "HORMONE_LEVEL",
        "InsulinRegimen": "INSULIN_REGIMEN",
        "MetabolicFinding": "CONDITION",
        "EndocrineGland": "BODY_SITE",
    }
    EXPECTED_ENTITIES = [
        ("GlycemicMeasure", 0, 10, "HbA1c 8.2%"),
        ("GlycemicMeasure", 15, 32, "glucose 186 mg/dL"),
        ("ThyroidFunctionMeasure", 50, 63, "TSH 6.1 mIU/L"),
        ("HormoneLevel", 68, 87, "cortisol 4.8 mcg/dL"),
        ("InsulinRegimen", 127, 148, "glargine 24 units qHS"),
        ("MetabolicFinding", 150, 168, "Metabolic syndrome"),
        ("EndocrineGland", 173, 180, "thyroid"),
    ]

    def test_endocrinology_domain_has_expected_labels(self):
        assert "endocrinology" in available_domains()
        assert get_default_labels("endocrinology") == self.EXPECTED_LABELS

    def test_endocrinology_labels_have_no_duplicates(self):
        labels = get_default_labels("endocrinology")
        lowered = [label.lower() for label in labels]
        assert len(lowered) == len(set(lowered))

    @pytest.mark.parametrize(
        ("label", "expected"),
        sorted(CANONICAL_LABELS_BY_DISPLAY.items()),
    )
    def test_endocrinology_labels_normalize_to_canonical(self, label, expected):
        assert normalize_canonical_label(label) == expected
        assert expected in CANONICAL_LABELS
        assert policy_label_for(expected) == "CLINICAL_CONCEPT"
        assert system_hints_for(expected)

    def test_endocrinology_fixture_has_expected_output(self):
        path = CLINICAL_FIXTURES_PATH / "endocrinology.jsonl"
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(rows) == 1

        row = rows[0]
        assert row["metadata"]["synthetic"] is True
        disclaimer = row["metadata"]["disclaimer"]
        assert "not clinical guidance" in disclaimer
        assert "does not infer insulin dosing or glycemic-control targets" in disclaimer

        actual_entities = [
            (entity["label"], entity["start"], entity["end"], entity["text"])
            for entity in row["entities"]
        ]
        assert actual_entities == self.EXPECTED_ENTITIES

        text = row["text"]
        for label, start, end, entity_text in actual_entities:
            assert text[start:end] == entity_text
            assert label in self.EXPECTED_LABELS

        observed_labels = {entity[0] for entity in actual_entities}
        assert observed_labels == set(self.EXPECTED_LABELS)


# ---------------------------------------------------------------------------
# Endocrinology routing in model_registry (issue #895)
# ---------------------------------------------------------------------------


class TestEndocrinologyRouting:
    ENDOCRINOLOGY_TEXT = "HbA1c 8.2%, TSH 6.1, on glargine 24 units qHS"

    def test_match_categories_routes_endocrinology(self):
        categories = [c for c, _ in _match_categories(self.ENDOCRINOLOGY_TEXT)]
        assert "Endocrinology" in categories

    def test_endocrinology_is_registry_metadata_not_a_live_category(self):
        assert "Endocrinology" in _CATEGORY_ENTITY_TYPES
        from openmed.core.model_registry import CATEGORIES

        assert "Endocrinology" not in CATEGORIES

    def test_get_model_suggestions_behavior_unchanged_for_endocrinology(self):
        suggestions = get_model_suggestions(self.ENDOCRINOLOGY_TEXT)
        assert suggestions
        assert all(info.category != "Endocrinology" for _k, info, _r in suggestions)


# ---------------------------------------------------------------------------
# Gastroenterology domain (issue #894)
# ---------------------------------------------------------------------------


class TestGastroenterologyDomain:
    """Gastroenterology domain labels, canonical mappings, and fixture coverage."""

    EXPECTED_LABELS = [
        "EndoscopicFinding",
        "GISymptom",
        "BowelPrepQuality",
        "BiopsySite",
        "GIScore",
        "LesionMorphology",
        "PolypDescriptor",
    ]
    CANONICAL_LABELS_BY_DISPLAY = {
        "EndoscopicFinding": "ENDOSCOPIC_FINDING",
        "GISymptom": "GI_SYMPTOM",
        "BowelPrepQuality": "GI_SCORE",
        "BiopsySite": "BODY_SITE",
        "GIScore": "GI_SCORE",
        "LesionMorphology": "POLYP_DESCRIPTOR",
        "PolypDescriptor": "POLYP_DESCRIPTOR",
    }
    EXPECTED_ENTITIES = [
        ("EndoscopicFinding", 0, 11, "Colonoscopy"),
        ("PolypDescriptor", 18, 41, "two 5 mm sessile polyps"),
        ("BiopsySite", 49, 62, "sigmoid colon"),
        ("LesionMorphology", 68, 81, "mild erythema"),
        ("BowelPrepQuality", 83, 112, "Boston bowel prep score was 8"),
        ("GIScore", 114, 134, "Bristol stool type 4"),
        ("GISymptom", 171, 179, "cramping"),
    ]

    def test_gastroenterology_domain_has_expected_labels(self):
        assert "gastroenterology" in available_domains()
        assert get_default_labels("gastroenterology") == self.EXPECTED_LABELS

    def test_gastroenterology_labels_have_no_duplicates(self):
        labels = get_default_labels("gastroenterology")
        lowered = [label.lower() for label in labels]
        assert len(lowered) == len(set(lowered))

    @pytest.mark.parametrize(
        ("label", "expected"),
        sorted(CANONICAL_LABELS_BY_DISPLAY.items()),
    )
    def test_gastroenterology_labels_normalize_to_canonical(self, label, expected):
        assert normalize_canonical_label(label) == expected
        assert expected in CANONICAL_LABELS
        assert policy_label_for(expected) == "CLINICAL_CONCEPT"
        assert system_hints_for(expected)

    def test_gastroenterology_fixture_has_expected_output(self):
        path = CLINICAL_FIXTURES_PATH / "gastroenterology.jsonl"
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(rows) == 1

        row = rows[0]
        assert row["metadata"]["synthetic"] is True
        disclaimer = row["metadata"]["disclaimer"]
        assert "not clinical guidance" in disclaimer
        assert "does not infer diagnosis" in disclaimer
        assert "polyp-risk stratification" in disclaimer

        actual_entities = [
            (entity["label"], entity["start"], entity["end"], entity["text"])
            for entity in row["entities"]
        ]
        assert actual_entities == self.EXPECTED_ENTITIES

        text = row["text"]
        for label, start, end, entity_text in actual_entities:
            assert text[start:end] == entity_text
            assert label in self.EXPECTED_LABELS

        observed_labels = {entity[0] for entity in actual_entities}
        assert observed_labels == set(self.EXPECTED_LABELS)


# ---------------------------------------------------------------------------
# Gastroenterology routing in model_registry (issue #894)
# ---------------------------------------------------------------------------


class TestGastroenterologyRouting:
    GASTROENTEROLOGY_TEXT = (
        "Colonoscopy found sessile polyps with Boston bowel prep score 8"
    )

    def test_match_categories_routes_gastroenterology(self):
        categories = [c for c, _ in _match_categories(self.GASTROENTEROLOGY_TEXT)]
        assert "Gastroenterology" in categories

    def test_gastroenterology_is_registry_metadata_not_a_live_category(self):
        assert "Gastroenterology" in _CATEGORY_ENTITY_TYPES
        from openmed.core.model_registry import CATEGORIES

        assert "Gastroenterology" not in CATEGORIES

    def test_get_model_suggestions_behavior_unchanged_for_gastroenterology(self):
        suggestions = get_model_suggestions(self.GASTROENTEROLOGY_TEXT)
        assert suggestions
        assert all(info.category != "Gastroenterology" for _k, info, _r in suggestions)


class TestNormalizeLabelIdempotency:
    @pytest.mark.parametrize(
        "label",
        [
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
            "bsn",
            "dni",
            "nie",
            "aadhaar",
            "cpf",
            "cnpj",
            "teudat_zehut",
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
            "medical_record_number",
            "mrn",
            "medical_record",
            "account_number",
            "account",
            "credit_debit_card",
            "credit_card",
            "debit_card",
            "payment_card",
        ],
    )
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
        "national_id": [
            "nir",
            "insee",
            "steuer_id",
            "steuernummer",
            "codice_fiscale",
            "cpf",
            "cnpj",
            "teudat_zehut",
        ],
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
            key: info
            for key, info in OPENMED_MODELS.items()
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
        from openmed.core.pii_i18n import DEFAULT_PII_MODELS, SUPPORTED_LANGUAGES

        pii_keys = [k for k in OPENMED_MODELS if k.startswith("pii_")]
        for lang in SUPPORTED_LANGUAGES:
            if lang == "en":
                # English models don't have a language infix
                assert any(
                    k.startswith("pii_")
                    and not any(
                        f"pii_{l}_" in k for l in SUPPORTED_LANGUAGES if l != "en"
                    )
                    for k in pii_keys
                ), "No English PII model found"
            else:
                # Non-English keys use pii_{lang}_ prefix, e.g. pii_de_superclinical_small
                has_language_key = any(k.startswith(f"pii_{lang}_") for k in pii_keys)
                default_model_id = DEFAULT_PII_MODELS.get(lang)
                has_default_model = any(
                    info.model_id == default_model_id and lang in info.languages
                    for info in OPENMED_MODELS.values()
                    if info.category == "Privacy"
                )
                assert has_language_key or has_default_model, (
                    f"No PII model found for language {lang!r}"
                )
