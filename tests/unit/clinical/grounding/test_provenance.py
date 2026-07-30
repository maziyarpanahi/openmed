"""Tests for grounding provenance and its feed into the signed audit report.

All vocabulary content is synthetic and algorithmically generated; no real
patient data or licensed terminology is used.
"""

from __future__ import annotations

import json

import pytest

from openmed.clinical.exporters import stamp_coding_provenance
from openmed.clinical.exporters.codeable_concept import GroundedSpan
from openmed.clinical.exporters.codeable_concept_simple import coding
from openmed.clinical.grounding import (
    GROUNDING_ASSIST_ONLY_ADVISORY,
    GROUNDING_METHODS,
    Candidate,
    GroundingProvenance,
    grounding_provenance,
    provenance_version_pins,
    scan_provenance_for_raw_text,
)
from openmed.core.audit import (
    AuditReport,
    AuditSpan,
    DetectorInfo,
    hash_text,
    stable_hash,
)


def _candidates() -> tuple[Candidate, ...]:
    return (
        Candidate(
            system="LOINC",
            code="L0001",
            display="Synthetic serum marker",
            score=1.0,
            source="sparse",
            matched_alias="serummarker",
            match_kind="exact",
            vocab_version="loinc-syn-2024a",
        ),
        Candidate(
            system="LOINC",
            code="L0001-alt",
            display="Synthetic serum marker (alternative)",
            score=0.4,
            source="sparse",
            matched_alias="serummarker",
            match_kind="fuzzy",
            vocab_version="loinc-syn-2024a",
        ),
    )


def _grounded_spans() -> list[GroundedSpan]:
    # Mention surfaces are opaque tokens disjoint from the vocabulary displays so
    # the no-PHI scan proves the raw surface itself is never stored.
    return [
        GroundedSpan(
            text="mzx-alpha",
            start=10,
            end=19,
            candidates=_candidates(),
        ),
        GroundedSpan(text="mzx-beta", start=30, end=38, candidates=()),
    ]


def test_from_candidates_captures_full_decision_chain():
    record = GroundingProvenance.from_candidates(
        start=10,
        end=22,
        span_text="serum marker",
        candidates=_candidates(),
        method="rerank",
        encoder_index_key="idx-1",
    )

    assert record.system == "LOINC"
    assert record.code == "L0001"
    assert record.score == 1.0
    assert record.method == "rerank"
    assert record.vocab_version == "loinc-syn-2024a"
    assert record.encoder_index_key == "idx-1"
    assert record.abstained is False
    assert record.text_hash == hash_text("serum marker")
    # Chosen concept is the top candidate; the rest become alternatives.
    assert [alt.code for alt in record.alternatives] == ["L0001-alt"]


def test_to_dict_is_serializable_and_carries_advisory():
    record = GroundingProvenance.from_candidates(
        start=0,
        end=5,
        span_text="hello",
        candidates=_candidates(),
        method="sparse",
    )

    payload = record.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["advisory"] == GROUNDING_ASSIST_ONLY_ADVISORY
    assert set(payload) >= {
        "start",
        "end",
        "text_hash",
        "system",
        "code",
        "display",
        "score",
        "method",
        "vocab_version",
        "encoder_index_key",
        "abstained",
        "alternatives",
    }


def test_grounding_provenance_returns_complete_chain_for_every_code():
    chain = grounding_provenance(_grounded_spans(), method="composite")

    assert len(chain) == 2
    emitted = [record for record in chain if not record.abstained]
    assert len(emitted) == 1
    for record in emitted:
        assert record.vocab_version
        assert record.method in GROUNDING_METHODS
        assert isinstance(record.score, float)


def test_span_without_candidates_is_recorded_as_abstention():
    chain = grounding_provenance(_grounded_spans())

    abstained = [record for record in chain if record.abstained]
    assert len(abstained) == 1
    assert abstained[0].code == ""
    assert abstained[0].system == ""


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="unknown grounding method"):
        GroundingProvenance.from_candidates(
            start=0,
            end=1,
            span_text="x",
            candidates=_candidates(),
            method="telepathy",
        )


def test_calibrated_score_overrides_candidate_score():
    record = GroundingProvenance.from_candidates(
        start=0,
        end=5,
        span_text="hello",
        candidates=_candidates(),
        method="dense",
        calibrated_score=0.73,
    )

    assert record.score == 0.73


def test_provenance_is_reproducible_for_identical_input():
    first = grounding_provenance(_grounded_spans(), method="composite")
    second = grounding_provenance(_grounded_spans(), method="composite")

    first_payload = [record.to_dict() for record in first]
    second_payload = [record.to_dict() for record in second]

    assert first_payload == second_payload
    assert stable_hash(first_payload) == stable_hash(second_payload)


def test_no_raw_note_text_enters_provenance():
    note = "Patient mzx-alpha and mzx-beta noted."
    chain = grounding_provenance(_grounded_spans())

    forbidden = [note, "mzx-alpha", "mzx-beta"]
    assert scan_provenance_for_raw_text(chain, forbidden) == ()

    serialized = json.dumps([record.to_dict() for record in chain])
    assert "mzx-alpha" not in serialized
    assert "mzx-beta" not in serialized


def test_leaked_raw_text_is_detected_by_the_scan():
    # A hand-built record that (incorrectly) stored a raw surface as its display
    # must be caught by the no-PHI scanner.
    leaky = GroundingProvenance(
        start=0,
        end=12,
        text_hash=hash_text("serum marker"),
        system="LOINC",
        code="L0001",
        display="serum marker",
        score=1.0,
        method="sparse",
    )

    assert scan_provenance_for_raw_text([leaky], ["serum marker"]) == ("serum marker",)


def test_non_ascii_leaked_text_is_detected_by_the_scan():
    # The no-PHI scanner must not be blind to accented / non-ASCII PHI.
    leaky = GroundingProvenance(
        start=0,
        end=13,
        text_hash=hash_text("Müller-Straße"),
        system="LOINC",
        code="L0002",
        display="Müller-Straße",
        score=1.0,
        method="sparse",
    )

    assert scan_provenance_for_raw_text([leaky], ["Müller-Straße"]) == (
        "Müller-Straße",
    )


def _base_report(grounding: list[dict] | None = None) -> AuditReport:
    text = "Patient serum marker noted."
    return AuditReport(
        policy="hipaa_safe_harbor",
        resolved_profile={"method": "mask"},
        detectors=[
            DetectorInfo(
                source="ml",
                model_id="unit-test-model",
                model_format="transformers",
            )
        ],
        safety_sweep={"source": "safety_sweep", "spans_added": 0},
        spans=[
            AuditSpan(
                start=8,
                end=20,
                label="OBSERVATION",
                canonical_label="OBSERVATION",
                sources=["ml"],
                confidence=0.9,
                threshold=0.5,
                action="retain",
                surrogate=None,
                text_hash=hash_text("serum marker"),
            )
        ],
        thresholds={"OBSERVATION": 0.5},
        residual_risk={"projected_leakage": 0.0},
        openmed_version="1.5.5",
        manifest_hash="sha256:manifest",
        document_length=len(text),
        input_hash=hash_text(text),
        deidentified_text_hash=hash_text(text),
        grounding=grounding or [],
    )


def test_grounding_feeds_into_signed_audit_and_verifies():
    chain = grounding_provenance(_grounded_spans(), method="rerank")
    report = _base_report([record.to_dict() for record in chain]).sign("release-key")

    assert report.grounding  # present in the signed report
    assert report.verify("release-key")

    restored = AuditReport.from_json(report.to_json())
    assert restored == report
    assert restored.verify("release-key")


def test_tampering_with_grounding_breaks_the_signature():
    chain = grounding_provenance(_grounded_spans(), method="rerank")
    report = _base_report([record.to_dict() for record in chain]).sign("release-key")

    tampered = AuditReport.from_json(report.to_json())
    tampered.grounding[0]["code"] = "TAMPERED"

    assert not tampered.verify("release-key")


def test_empty_grounding_leaves_report_hash_unchanged():
    # Backward compatibility: a report with no grounding hashes exactly as before
    # (the key is omitted from the signed payload entirely).
    without = _base_report()
    payload = json.loads(without.to_json())

    assert "grounding" not in payload
    assert without.verify("release-key") is False  # unsigned
    assert without.sign("release-key").verify("release-key")


def test_emitted_coding_carries_matching_system_version():
    chain = grounding_provenance(_grounded_spans(), method="sparse")
    pins = provenance_version_pins(chain)

    base = coding("loinc", "L0001", "Synthetic serum marker")
    stamped = stamp_coding_provenance(base, pins, source_label="synthetic manifest")

    assert stamped["version"] == "loinc-syn-2024a"
