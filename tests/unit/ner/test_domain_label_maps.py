"""Clinical-domain label-map and offset-stability tests.

No restricted clinical or variant database is bundled; fixtures are synthetic
text used only for label-map and offset coverage.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.core.labels import (
    AIRWAY_MANAGEMENT,
    ALLERGEN,
    ALLERGY_CRITICALITY,
    BODY_SITE,
    CANONICAL_LABELS,
    CKD_STAGE,
    CLINICAL_CONCEPT,
    CLINICAL_SIGNIFICANCE,
    CONDITION,
    DEVELOPMENTAL_MILESTONE,
    DIALYSIS_MODALITY,
    DYSPNEA_GRADE,
    GENE,
    GENE_SYMBOL,
    GROWTH_PARAMETER,
    GROWTH_PERCENTILE,
    MEASUREMENT,
    NUTRITIONAL_STATUS,
    OTHER,
    OXYGEN_SUPPORT,
    PROTEIN_CHANGE,
    REACTION_MANIFESTATION,
    REACTION_SEVERITY,
    RECEPTOR_STATUS,
    RENAL_FUNCTION_MEASURE,
    RESPIRATORY_FINDING,
    SPIROMETRY_MEASURE,
    STAGE_GROUP,
    TNM_M,
    TNM_N,
    TNM_T,
    TUMOR_GRADE,
    URINE_FINDING,
    VARIANT_DESCRIPTOR,
    ZYGOSITY,
    hipaa_class_for,
    normalize_label,
    policy_label_for,
    risk_level_for,
    system_hints_for,
)
from openmed.core.pipeline import Pipeline
from openmed.ner.labels import available_domains, get_default_labels

NEW_LABELS = (
    GENE_SYMBOL,
    VARIANT_DESCRIPTOR,
    PROTEIN_CHANGE,
    ZYGOSITY,
    CLINICAL_SIGNIFICANCE,
    CKD_STAGE,
    DIALYSIS_MODALITY,
    RENAL_FUNCTION_MEASURE,
    URINE_FINDING,
    SPIROMETRY_MEASURE,
    OXYGEN_SUPPORT,
    RESPIRATORY_FINDING,
    DYSPNEA_GRADE,
)
FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "clinical"
    / "genomic_variant.jsonl"
)


class TestGenomicVariantDomain:
    def test_domain_coexists_with_genomic(self):
        domains = available_domains()
        assert "genomic_variant" in domains
        assert "genomic" in domains
        # The existing coarse genomic map is not overwritten.
        assert get_default_labels("genomic") == [
            "Variant",
            "Gene",
            "Transcript",
            "Phenotype",
        ]

    def test_genomic_variant_labels_non_empty(self):
        labels = get_default_labels("genomic_variant")
        assert labels
        assert "VariantDescriptor" in labels


class TestGenomicCanonicalLabels:
    def test_new_labels_in_canonical_set(self):
        for label in NEW_LABELS:
            assert label in CANONICAL_LABELS

    def test_new_labels_round_trip(self):
        for label in NEW_LABELS:
            assert normalize_label(label) == label

    @pytest.mark.parametrize(
        "alias,expected",
        [
            ("gene", GENE),
            ("gene symbol", GENE_SYMBOL),
            ("variant descriptor", VARIANT_DESCRIPTOR),
            ("hgvs", VARIANT_DESCRIPTOR),
            ("protein change", PROTEIN_CHANGE),
            ("zygosity", ZYGOSITY),
            ("clinical significance", CLINICAL_SIGNIFICANCE),
        ],
    )
    def test_aliases_resolve(self, alias, expected):
        assert normalize_label(alias) == expected

    def test_new_labels_are_clinical_concepts(self):
        for label in NEW_LABELS:
            assert policy_label_for(label) == CLINICAL_CONCEPT


class TestHgvsOffsetStability:
    def _fixtures(self):
        return [
            json.loads(line)
            for line in FIXTURE.read_text().splitlines()
            if line.strip()
        ]

    def test_fixture_loads(self):
        assert len(self._fixtures()) == 2

    def test_hgvs_spans_keep_stable_offsets_through_normalization(self):
        pipeline = Pipeline()
        for row in self._fixtures():
            document = pipeline.stage1_normalize(row["text"])
            for span in row["spans"]:
                ns, ne = document.offset_map.original_span_to_normalized(
                    span["start"], span["end"]
                )
                # HGVS punctuation (: . ( ) > _) is preserved by normalization.
                assert document.normalized_text[ns:ne] == span["text"], span
                # And the normalized span round-trips to the original offsets.
                assert document.offset_map.normalized_span_to_original_offsets(
                    ns, ne
                ) == (span["start"], span["end"])


PULMONOLOGY_FIXTURE = (
    Path(__file__).resolve().parents[2] / "fixtures" / "clinical" / "pulmonology.jsonl"
)


class TestPulmonologyDomain:
    def test_domain_resolves(self):
        assert "pulmonology" in available_domains()
        assert get_default_labels("pulmonology") == [
            "SpirometryMeasure",
            "OxygenSupport",
            "RespiratoryFinding",
            "DyspneaGrade",
            "LungAuscultation",
            "PFTInterpretation",
            "AirwayDevice",
        ]

    @pytest.mark.parametrize(
        "alias,expected",
        [
            ("spirometry measure", SPIROMETRY_MEASURE),
            ("spirometry", SPIROMETRY_MEASURE),
            ("fev1", SPIROMETRY_MEASURE),
            ("fvc", SPIROMETRY_MEASURE),
            ("oxygen support", OXYGEN_SUPPORT),
            ("oxygen therapy", OXYGEN_SUPPORT),
            ("nasal cannula", OXYGEN_SUPPORT),
            ("respiratory finding", RESPIRATORY_FINDING),
            ("wheeze", RESPIRATORY_FINDING),
            ("crackles", RESPIRATORY_FINDING),
            ("lung auscultation", RESPIRATORY_FINDING),
            ("dyspnea grade", DYSPNEA_GRADE),
            ("mmrc", DYSPNEA_GRADE),
            ("dyspnea", DYSPNEA_GRADE),
            ("airway device", AIRWAY_MANAGEMENT),
        ],
    )
    def test_aliases_resolve(self, alias, expected):
        assert normalize_label(alias) == expected

    def test_new_labels_are_clinical_concepts(self):
        for label in (
            SPIROMETRY_MEASURE,
            OXYGEN_SUPPORT,
            RESPIRATORY_FINDING,
            DYSPNEA_GRADE,
        ):
            assert label in CANONICAL_LABELS
            assert normalize_label(label) == label
            assert policy_label_for(label) == CLINICAL_CONCEPT

    def _fixtures(self):
        return [
            json.loads(line)
            for line in PULMONOLOGY_FIXTURE.read_text().splitlines()
            if line.strip()
        ]

    def test_fixture_loads(self):
        assert len(self._fixtures()) == 2

    def test_fixture_covers_spirometry_oxygen_and_dyspnea_spans(self):
        labels = {span["label"] for row in self._fixtures() for span in row["spans"]}
        assert {"SpirometryMeasure", "OxygenSupport", "DyspneaGrade"} <= labels

    def test_fixture_spans_keep_stable_offsets_through_normalization(self):
        pipeline = Pipeline()
        for row in self._fixtures():
            document = pipeline.stage1_normalize(row["text"])
            for span in row["spans"]:
                assert row["text"][span["start"] : span["end"]] == span["text"], span
                ns, ne = document.offset_map.original_span_to_normalized(
                    span["start"], span["end"]
                )
                assert document.normalized_text[ns:ne] == span["text"], span
                assert document.offset_map.normalized_span_to_original_offsets(
                    ns, ne
                ) == (span["start"], span["end"])


PEDIATRICS_GROWTH_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "clinical"
    / "pediatrics_growth.jsonl"
)


ONCOLOGY_STAGING_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "clinical"
    / "oncology_staging.jsonl"
)


ALLERGY_INTOLERANCE_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "clinical"
    / "allergy_intolerance.jsonl"
)


class TestAllergyIntoleranceDomain:
    """FHIR AllergyIntolerance-aligned labels and offline fixture coverage."""

    EXPECTED_LABELS = [
        "Allergen",
        "ReactionManifestation",
        "ReactionSeverity",
        "Criticality",
        "AllergyType",
        "OnsetContext",
    ]
    CANONICAL_LABELS_BY_DISPLAY = {
        "Allergen": ALLERGEN,
        "ReactionManifestation": REACTION_MANIFESTATION,
        "ReactionSeverity": REACTION_SEVERITY,
        "Criticality": ALLERGY_CRITICALITY,
        "AllergyType": "OTHER",
        "OnsetContext": "OTHER",
    }
    EXPECTED_ENTITIES = [
        ("Allergen", 0, 10, "Penicillin"),
        ("ReactionManifestation", 18, 38, "hives and angioedema"),
        ("ReactionSeverity", 40, 46, "severe"),
        ("Criticality", 57, 61, "high"),
        ("AllergyType", 75, 82, "allergy"),
        ("OnsetContext", 98, 107, "childhood"),
    ]

    def _fixtures(self):
        return [
            json.loads(line)
            for line in ALLERGY_INTOLERANCE_FIXTURE.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

    def test_domain_resolves(self):
        assert "allergy_intolerance" in available_domains()
        assert get_default_labels("allergy_intolerance") == self.EXPECTED_LABELS

    @pytest.mark.parametrize(
        ("label", "expected"),
        sorted(CANONICAL_LABELS_BY_DISPLAY.items()),
    )
    def test_labels_normalize_with_metadata(self, label, expected):
        assert normalize_label(label) == expected
        assert expected in CANONICAL_LABELS
        assert policy_label_for(expected) == CLINICAL_CONCEPT
        assert risk_level_for(expected) == "low"
        assert system_hints_for(expected)
        assert hipaa_class_for(expected)

    def test_fixture_reports_offline_per_label_coverage(self):
        rows = self._fixtures()
        assert len(rows) == 1

        row = rows[0]
        assert row["metadata"]["synthetic"] is True
        disclaimer = row["metadata"]["disclaimer"]
        assert "not clinical guidance" in disclaimer
        assert "not a contraindication check" in disclaimer

        observed_labels = {entity["label"] for entity in row["entities"]}
        assert observed_labels == set(self.EXPECTED_LABELS)

    def test_fixture_entities_match_expected(self):
        row = self._fixtures()[0]
        actual_entities = [
            (entity["label"], entity["start"], entity["end"], entity["text"])
            for entity in row["entities"]
        ]
        assert actual_entities == self.EXPECTED_ENTITIES

    def test_fixture_spans_keep_stable_offsets_through_normalization(self):
        pipeline = Pipeline()
        for row in self._fixtures():
            document = pipeline.stage1_normalize(row["text"])
            for entity in row["entities"]:
                assert row["text"][entity["start"] : entity["end"]] == entity["text"], (
                    entity
                )
                ns, ne = document.offset_map.original_span_to_normalized(
                    entity["start"], entity["end"]
                )
                assert document.normalized_text[ns:ne] == entity["text"], entity
                assert document.offset_map.normalized_span_to_original_offsets(
                    ns, ne
                ) == (entity["start"], entity["end"])


class TestPediatricsGrowthDomain:
    """Pediatric growth and developmental-surveillance domain (issue #896)."""

    EXPECTED_LABELS = [
        "GrowthParameter",
        "GrowthPercentile",
        "GrowthZScore",
        "DevelopmentalMilestone",
        "FeedingHistory",
        "PediatricFinding",
    ]
    CANONICAL_LABELS_BY_DISPLAY = {
        "GrowthParameter": GROWTH_PARAMETER,
        "GrowthPercentile": GROWTH_PERCENTILE,
        "GrowthZScore": GROWTH_PERCENTILE,
        "DevelopmentalMilestone": DEVELOPMENTAL_MILESTONE,
        "FeedingHistory": NUTRITIONAL_STATUS,
        "PediatricFinding": CONDITION,
    }
    EXPECTED_ENTITIES = [
        ("GrowthParameter", 0, 14, "Weight 14.2 kg"),
        ("GrowthPercentile", 35, 58, "45th percentile for age"),
        ("GrowthZScore", 67, 96, "height-for-age z-score of 0.4"),
        ("DevelopmentalMilestone", 109, 128, "walks independently"),
        ("FeedingHistory", 147, 183, "exclusively breastfed until 6 months"),
        ("PediatricFinding", 185, 218, "Anterior fontanelle open and soft"),
    ]

    def test_domain_resolves(self):
        assert "pediatrics_growth" in available_domains()
        assert get_default_labels("pediatrics_growth") == self.EXPECTED_LABELS

    @pytest.mark.parametrize(
        ("label", "expected"),
        sorted(CANONICAL_LABELS_BY_DISPLAY.items()),
    )
    def test_labels_normalize_to_canonical(self, label, expected):
        assert normalize_label(label) == expected

    def test_new_labels_are_clinical_concepts(self):
        for label in (GROWTH_PARAMETER, GROWTH_PERCENTILE, DEVELOPMENTAL_MILESTONE):
            assert label in CANONICAL_LABELS
            assert normalize_label(label) == label
            assert policy_label_for(label) == CLINICAL_CONCEPT

    def _fixtures(self):
        return [
            json.loads(line)
            for line in PEDIATRICS_GROWTH_FIXTURE.read_text().splitlines()
            if line.strip()
        ]

    def test_fixture_loads(self):
        assert len(self._fixtures()) == 1

    def test_fixture_reports_disclaimer(self):
        row = self._fixtures()[0]
        assert row["metadata"]["synthetic"] is True
        disclaimer = row["metadata"]["disclaimer"]
        assert "not clinical guidance" in disclaimer
        assert "percentile or z-score" in disclaimer

    def test_fixture_covers_percentile_and_milestone_spans(self):
        labels = {
            entity["label"] for row in self._fixtures() for entity in row["entities"]
        }
        assert {"GrowthPercentile", "DevelopmentalMilestone"} <= labels
        assert labels == set(self.EXPECTED_LABELS)

    def test_fixture_entities_match_expected(self):
        row = self._fixtures()[0]
        actual_entities = [
            (entity["label"], entity["start"], entity["end"], entity["text"])
            for entity in row["entities"]
        ]
        assert actual_entities == self.EXPECTED_ENTITIES

    def test_fixture_spans_keep_stable_offsets_through_normalization(self):
        pipeline = Pipeline()
        for row in self._fixtures():
            document = pipeline.stage1_normalize(row["text"])
            for entity in row["entities"]:
                assert row["text"][entity["start"] : entity["end"]] == entity["text"], (
                    entity
                )
                ns, ne = document.offset_map.original_span_to_normalized(
                    entity["start"], entity["end"]
                )
                assert document.normalized_text[ns:ne] == entity["text"], entity
                assert document.offset_map.normalized_span_to_original_offsets(
                    ns, ne
                ) == (entity["start"], entity["end"])


class TestOncologyStagingDomain:
    """TNM and tumor-descriptor domain coverage for issue #864."""

    EXPECTED_LABELS = [
        "TumorCategory",
        "NodeCategory",
        "MetastasisCategory",
        "StageGroup",
        "TumorGrade",
        "TumorSize",
        "ReceptorStatus",
        "ResponseAssessment",
        "PrimarySite",
    ]
    CANONICAL_LABELS_BY_DISPLAY = {
        "TumorCategory": TNM_T,
        "NodeCategory": TNM_N,
        "MetastasisCategory": TNM_M,
        "StageGroup": STAGE_GROUP,
        "TumorGrade": TUMOR_GRADE,
        "TumorSize": MEASUREMENT,
        "ReceptorStatus": RECEPTOR_STATUS,
        "ResponseAssessment": OTHER,
        "PrimarySite": BODY_SITE,
    }
    EXPECTED_ENTITIES = [
        ("TumorCategory", 25, 28, "pT2"),
        ("NodeCategory", 29, 31, "N0"),
        ("MetastasisCategory", 32, 34, "M0"),
        ("StageGroup", 36, 45, "Stage IIA"),
        ("TumorGrade", 47, 60, "tumor grade 2"),
        ("TumorSize", 73, 79, "3.4 cm"),
        ("ReceptorStatus", 81, 92, "ER-positive"),
        ("ResponseAssessment", 94, 110, "partial response"),
        ("PrimarySite", 125, 131, "breast"),
    ]

    def _fixtures(self):
        return [
            json.loads(line)
            for line in ONCOLOGY_STAGING_FIXTURE.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

    def test_domain_resolves_with_all_display_labels(self):
        assert "oncology_staging" in available_domains()
        assert get_default_labels("oncology_staging") == self.EXPECTED_LABELS

    @pytest.mark.parametrize(
        ("label", "expected"),
        sorted(CANONICAL_LABELS_BY_DISPLAY.items()),
    )
    def test_labels_have_canonical_policy_metadata(self, label, expected):
        assert normalize_label(label) == expected
        assert expected in CANONICAL_LABELS
        assert policy_label_for(expected) == CLINICAL_CONCEPT
        assert risk_level_for(expected) == "low"
        assert system_hints_for(expected)

    def test_fixture_covers_every_label_and_stage_group_offsets(self):
        rows = self._fixtures()
        assert len(rows) == 1
        row = rows[0]
        assert row["metadata"]["synthetic"] is True
        disclaimer = row["metadata"]["disclaimer"]
        assert "descriptive" in disclaimer
        assert "human review" in disclaimer
        assert "medical decisions" in disclaimer

        actual_entities = [
            (entity["label"], entity["start"], entity["end"], entity["text"])
            for entity in row["entities"]
        ]
        assert actual_entities == self.EXPECTED_ENTITIES
        assert {entity[0] for entity in actual_entities} == set(self.EXPECTED_LABELS)

        text = row["text"]
        for label, start, end, entity_text in actual_entities:
            assert text[start:end] == entity_text
            assert label in self.EXPECTED_LABELS

    def test_fixture_spans_keep_stable_offsets_through_normalization(self):
        pipeline = Pipeline()
        for row in self._fixtures():
            document = pipeline.stage1_normalize(row["text"])
            for entity in row["entities"]:
                ns, ne = document.offset_map.original_span_to_normalized(
                    entity["start"], entity["end"]
                )
                assert document.normalized_text[ns:ne] == entity["text"], entity
                assert document.offset_map.normalized_span_to_original_offsets(
                    ns, ne
                ) == (entity["start"], entity["end"])
