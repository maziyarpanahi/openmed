"""Tests for the grounding-aware FHIR CodeableConcept core (issue #271)."""

from __future__ import annotations

import pytest

from openmed.clinical.exporters.codeable_concept import (
    SYSTEM_URI,
    GroundedSpan,
    build_reverse_index,
    to_codeable_concept,
)
from openmed.clinical.grounding import Candidate

RXNORM_URI = "http://www.nlm.nih.gov/research/umls/rxnorm"
SNOMED_URI = "http://snomed.info/sct"


def _metformin_span() -> GroundedSpan:
    return GroundedSpan(
        text="metformin",
        start=10,
        end=19,
        candidates=(
            Candidate("RXNORM", "6809", "metformin", 1.0),
            Candidate("SNOMED", "372567009", "Metformin", 0.8),
        ),
    )


class TestSystemUri:
    def test_canonical_hl7_uris(self):
        assert SYSTEM_URI["RXNORM"] == RXNORM_URI
        assert SYSTEM_URI["ICD10CM"] == "http://hl7.org/fhir/sid/icd-10-cm"
        assert SYSTEM_URI["LOINC"] == "http://loinc.org"
        assert SYSTEM_URI["SNOMED"] == SNOMED_URI
        assert SYSTEM_URI["HPO"] == "http://human-phenotype-ontology.org"
        assert SYSTEM_URI["UMLS"] == "http://terminology.hl7.org/CodeSystem/umls"


class TestToCodeableConcept:
    def test_multi_system_span_emits_canonical_uris_and_text(self):
        cc = to_codeable_concept(_metformin_span())
        assert cc["text"] == "metformin"
        systems = {c["system"] for c in cc["coding"]}
        assert systems == {RXNORM_URI, SNOMED_URI}
        codes = {c["system"]: c["code"] for c in cc["coding"]}
        assert codes[RXNORM_URI] == "6809"

    def test_coding_order_is_deterministic_by_system_priority(self):
        cc = to_codeable_concept(_metformin_span())
        # SNOMED sorts before RxNorm in the shared default system priority.
        assert cc["coding"][0]["system"] == SNOMED_URI
        assert to_codeable_concept(_metformin_span()) == cc  # deterministic

    def test_linker_score_carried_in_internal_field(self):
        cc = to_codeable_concept(_metformin_span())
        by_system = {c["system"]: c for c in cc["coding"]}
        assert by_system[RXNORM_URI]["_score"] == 1.0
        assert by_system[SNOMED_URI]["_score"] == 0.8

    def test_span_without_candidates_is_text_only(self):
        span = GroundedSpan(text="unknown drug", start=0, end=12, candidates=())
        cc = to_codeable_concept(span)
        assert cc == {"text": "unknown drug"}

    def test_unknown_system_raises(self):
        span = GroundedSpan(
            text="x", start=0, end=1, candidates=(Candidate("NOPE", "1", "x", 1.0),)
        )
        with pytest.raises(ValueError):
            to_codeable_concept(span)


class TestReverseIndex:
    def test_maps_system_code_to_source_offsets(self):
        index = build_reverse_index([_metformin_span()])
        assert index[(RXNORM_URI, "6809")] == [(10, 19)]
        assert index[(SNOMED_URI, "372567009")] == [(10, 19)]

    def test_accumulates_multiple_spans_for_same_code(self):
        span_a = GroundedSpan(
            text="aspirin",
            start=0,
            end=7,
            candidates=(Candidate("RXNORM", "1191", "aspirin", 1.0),),
        )
        span_b = GroundedSpan(
            text="aspirin",
            start=20,
            end=27,
            candidates=(Candidate("RXNORM", "1191", "aspirin", 1.0),),
        )
        index = build_reverse_index([span_a, span_b])
        assert index[(RXNORM_URI, "1191")] == [(0, 7), (20, 27)]
