"""Genomic-variant domain and HGVS offset-stability tests (issue #906).

No ClinVar/HGMD/dbSNP/COSMIC or any restricted variant database is bundled;
the fixture is synthetic HGVS-style text only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.core.labels import (
    AIRWAY_MANAGEMENT,
    CANONICAL_LABELS,
    CKD_STAGE,
    CLINICAL_CONCEPT,
    CLINICAL_SIGNIFICANCE,
    DIALYSIS_MODALITY,
    DYSPNEA_GRADE,
    GENE_SYMBOL,
    OXYGEN_SUPPORT,
    PROTEIN_CHANGE,
    RENAL_FUNCTION_MEASURE,
    RESPIRATORY_FINDING,
    SPIROMETRY_MEASURE,
    URINE_FINDING,
    VARIANT_DESCRIPTOR,
    ZYGOSITY,
    normalize_label,
    policy_label_for,
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
            ("gene", GENE_SYMBOL),
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
