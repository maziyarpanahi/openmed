"""Synthetic offline controls for Journey chart-abstraction production."""

import builtins
import urllib.request
from dataclasses import replace

import pytest

from openmed.agent.workflows import (
    AbstractionEvidenceError,
    AbstractionFieldBinding,
    AbstractionReviewReceipt,
    ReviewerState,
    SourceKind,
    TransformationKind,
    build_journey_abstraction_evidence,
)
from openmed.clinical.journey import JourneySnapshot
from openmed.clinical.journey_contracts import (
    ClinicalArtifact,
    ClinicalFact,
    ConflictSet,
    EvidenceLocator,
    canonical_digest,
    sha256_digest,
)

FIELD = "registry.primary_diagnosis"
SUBJECT = "subject_aaaaaaaaaaaaaaaa"
CANARIES = (
    "Synthetic Patient X",
    "synthetic@example.invalid",
    "/private/synthetic/chart",
    "患者合成１２３",
    "2026-02-03",
    "Bearer synthetic-secret",
)


def records():
    """Return synthetic Journey metadata containing deliberate private canaries."""
    artifact = ClinicalArtifact(
        artifact_id="artifact_aaaaaaaaaaaaaaaa",
        artifact_type="clinical_note",
        media_type="text/plain",
        content_hash=sha256_digest(" | ".join(CANARIES)),
        byte_size=200,
        source_id="source_aaaaaaaaaaaaaaaa",
        subject_id=SUBJECT,
        recorded_at="2026-01-01T00:00:00Z",
        attributes={"private_path": CANARIES[2]},
    )
    locator = EvidenceLocator(
        locator_id="evidence_aaaaaaaaaaaaaaaa",
        artifact_id=artifact.artifact_id,
        location_type="text_span",
        location={"start": 4, "end": 18},
        transform={"private_payload": CANARIES[5]},
    )
    fact = ClinicalFact(
        fact_id="fact_aaaaaaaaaaaaaaaa",
        subject_id=SUBJECT,
        fact_type="condition",
        value={"canaries": CANARIES},
        status="active",
        evidence_ids=(locator.locator_id,),
        derivation_hash=sha256_digest("synthetic-extractor"),
        confidence=0.95,
        attributes={"private_value": CANARIES[0]},
    )
    return artifact, locator, fact


def build(**changes):
    artifact, locator, fact = records()
    arguments = {
        "snapshot": JourneySnapshot.at(SUBJECT, 1),
        "fields": (
            AbstractionFieldBinding(FIELD, (fact.fact_id,), TransformationKind.RULE),
        ),
        "facts": (fact,),
        "locators": (locator,),
        "artifacts": (artifact,),
        "source_kinds": {artifact.artifact_id: SourceKind.CLINICAL_RECORD},
    }
    arguments.update(changes)
    return build_journey_abstraction_evidence(**arguments)


def approval(evidence):
    return AbstractionReviewReceipt(
        FIELD, evidence.chains[0].chain_digest, ReviewerState.APPROVED
    )


def codes(evidence):
    return {issue.code for issue in evidence.evaluate((FIELD,)).issues}


def test_explicit_approval_is_required_and_all_required_fields_are_checked():
    pending = build()
    assert codes(pending) == {"review_not_approved"}
    approved = build(review_receipts=(approval(pending),))
    assert approved.finalize((FIELD,)).chain_count == 1
    report = approved.evaluate((FIELD, "registry.stage"))
    assert {item.code for item in report.issues} == {"missing_field_evidence"}
    with pytest.raises(AbstractionEvidenceError) as caught:
        approved.finalize((FIELD, "registry.stage"))
    assert caught.value.report == report
    chain = approved.chains[0]
    artifact, _, fact = records()
    assert chain.normalized_fact_digest == canonical_digest(fact)
    assert chain.source_locations[0].source_digest == artifact.content_hash
    assert (
        chain.source_locations[0].start_offset,
        chain.source_locations[0].end_offset,
    ) == (4, 18)
    assert chain.uncertainty == pytest.approx(0.05)


@pytest.mark.parametrize(
    "case,expected",
    [
        ("missing_fact", "missing_field_evidence"),
        ("missing_locator", "missing_locator_evidence"),
        ("non_text", "non_text_locator_evidence"),
        ("missing_artifact", "missing_artifact_evidence"),
        ("generated", "generated_only_evidence"),
        ("derived", "derived_only_evidence"),
        ("unknown_origin", "source_kind_undeclared"),
        ("subject", "subject_mismatch"),
        ("artifact_subject", "subject_mismatch"),
        ("conflict", "conflicting_facts"),
        ("candidates", "conflicting_facts"),
    ],
)
def test_distinct_coverage_blockers_cannot_be_approved_away(case, expected):
    artifact, locator, fact = records()
    changes = {}
    if case == "missing_fact":
        changes["facts"] = ()
    elif case == "missing_locator":
        changes["locators"] = ()
    elif case == "non_text":
        changes["locators"] = (
            replace(
                locator,
                location_type="json_pointer",
                location={"pointer": "/synthetic"},
            ),
        )
    elif case == "missing_artifact":
        changes["artifacts"] = ()
    elif case == "generated":
        changes["source_kinds"] = {artifact.artifact_id: SourceKind.GENERATED_TEXT}
    elif case == "derived":
        changes.update(
            facts=(replace(fact, parent_fact_ids=("fact_bbbbbbbbbbbbbbbb",)),),
            locators=(),
        )
    elif case == "unknown_origin":
        changes["source_kinds"] = {}
    elif case == "subject":
        changes["facts"] = (replace(fact, subject_id="subject_bbbbbbbbbbbbbbbb"),)
    elif case == "artifact_subject":
        changes["artifacts"] = (
            replace(artifact, subject_id="subject_bbbbbbbbbbbbbbbb"),
        )
    elif case == "conflict":
        changes["conflicts"] = (
            ConflictSet(
                conflict_id="conflict_aaaaaaaaaaaaaaaa",
                subject_id=SUBJECT,
                conflict_type="value",
                fact_ids=(fact.fact_id, "fact_bbbbbbbbbbbbbbbb"),
                status="open",
                detected_by="synthetic",
                derivation_hash=sha256_digest("conflict"),
            ),
        )
    elif case == "candidates":
        other = replace(fact, fact_id="fact_bbbbbbbbbbbbbbbb", value="synthetic-other")
        changes.update(
            facts=(fact, other),
            fields=(
                AbstractionFieldBinding(
                    FIELD, (fact.fact_id, other.fact_id), TransformationKind.RULE
                ),
            ),
        )
    pending = build(**changes)
    if pending.chains:
        changes["review_receipts"] = (approval(pending),)
    evidence = build(**changes)
    assert expected in codes(evidence)
    with pytest.raises(AbstractionEvidenceError, match="evidence_not_finalizable"):
        evidence.finalize((FIELD,))
    # Persistent producer blockers apply even when the caller changes requirements.
    assert expected in {
        issue.code for issue in evidence.evaluate(("registry.stage",)).issues
    }


@pytest.mark.parametrize(
    "case",
    [
        "value",
        "attributes",
        "span",
        "source",
        "transform",
        "derivation",
        "snapshot",
        "kind",
    ],
)
def test_stale_review_receipts_fail_closed(case):
    artifact, locator, fact = records()
    changes = {"review_receipts": (approval(build()),)}
    if case == "value":
        changes["facts"] = (replace(fact, value="synthetic-changed"),)
    elif case == "attributes":
        changes["facts"] = (
            replace(fact, attributes={"private_value": "synthetic-changed"}),
        )
    elif case == "span":
        changes["locators"] = (replace(locator, location={"start": 5, "end": 18}),)
    elif case == "source":
        changes["artifacts"] = (
            replace(artifact, content_hash=sha256_digest("changed")),
        )
    elif case == "transform":
        changes["locators"] = (replace(locator, transform={"step": "changed"}),)
    elif case == "derivation":
        changes["facts"] = (replace(fact, derivation_hash=sha256_digest("changed")),)
    elif case == "snapshot":
        changes["snapshot"] = JourneySnapshot.at(SUBJECT, 2)
    elif case == "kind":
        changes["fields"] = (
            AbstractionFieldBinding(FIELD, (fact.fact_id,), TransformationKind.MODEL),
        )
    assert {"review_receipt_mismatch", "review_not_approved"} <= codes(build(**changes))


def test_rejection_and_pending_receipts_never_approve():
    for state in (ReviewerState.PENDING, ReviewerState.REJECTED):
        receipt = replace(approval(build()), reviewer_state=state)
        assert codes(build(review_receipts=(receipt,))) == {"review_not_approved"}


def test_every_locator_is_checked_and_duplicate_spans_are_deduplicated():
    artifact, locator, fact = records()
    second = replace(locator, locator_id="evidence_bbbbbbbbbbbbbbbb")
    fact = replace(fact, evidence_ids=(locator.locator_id, second.locator_id))
    assert "missing_locator_evidence" in codes(build(facts=(fact,)))
    evidence = build(facts=(fact,), locators=(locator, second))
    assert len(evidence.chains[0].source_locations) == 1
    reordered = build(facts=(fact,), locators=(second, locator))
    assert evidence.evidence_digest == reordered.evidence_digest
    assert (
        evidence.evaluate((FIELD,)).to_json() == reordered.evaluate((FIELD,)).to_json()
    )


def test_no_private_values_in_chains_reports_receipts_or_repr():
    pending = build()
    receipt = approval(pending)
    evidence = build(review_receipts=(receipt,))
    outputs = str(
        (
            evidence.chains[0].to_dict(),
            evidence.evaluate((FIELD,)).to_json(),
            evidence.finalize((FIELD,)).to_json(),
            repr(evidence),
            repr(receipt),
        )
    )
    for value in CANARIES:
        assert value not in outputs
    artifact, locator, fact = records()
    for opaque_id in (artifact.artifact_id, locator.locator_id, fact.fact_id, SUBJECT):
        assert opaque_id not in outputs


def test_invalid_and_duplicate_declarations_use_value_free_errors():
    with pytest.raises(AbstractionEvidenceError) as caught:
        AbstractionFieldBinding(CANARIES[0], (), TransformationKind.RULE)
    assert CANARIES[0] not in str(caught.value)
    artifact, locator, fact = records()
    for changes in (
        {"facts": (fact, fact)},
        {"locators": (locator, locator)},
        {"artifacts": (artifact, artifact)},
        {"facts": (CANARIES[0],)},
        {"source_kinds": {artifact.artifact_id: "clinical_record"}},
        {"review_receipts": (approval(build()), approval(build()))},
    ):
        with pytest.raises(AbstractionEvidenceError) as caught:
            build(**changes)
        assert all(value not in str(caught.value) for value in CANARIES)


def test_mixed_clinical_and_generated_sources_keep_all_offsets_and_block_non_text():
    artifact, locator, fact = records()
    generated = replace(artifact, artifact_id="artifact_bbbbbbbbbbbbbbbb")
    supplemental = replace(
        locator,
        locator_id="evidence_bbbbbbbbbbbbbbbb",
        artifact_id=generated.artifact_id,
        location={"start": 30, "end": 45},
    )
    fact = replace(fact, evidence_ids=(locator.locator_id, supplemental.locator_id))
    args = dict(
        facts=(fact,),
        locators=(locator, supplemental),
        artifacts=(artifact, generated),
        source_kinds={
            artifact.artifact_id: SourceKind.CLINICAL_RECORD,
            generated.artifact_id: SourceKind.GENERATED_TEXT,
        },
    )
    pending = build(**args)
    approved = build(**args, review_receipts=(approval(pending),))
    assert approved.finalize((FIELD,)).chain_count == 1
    assert {
        (span.start_offset, span.end_offset, span.kind)
        for span in approved.chains[0].source_locations
    } == {
        (4, 18, SourceKind.CLINICAL_RECORD),
        (30, 45, SourceKind.GENERATED_TEXT),
    }
    args["locators"] = (
        locator,
        replace(
            supplemental,
            location_type="json_pointer",
            location={"pointer": "/synthetic"},
        ),
    )
    partial = build(**args)
    assert "non_text_locator_evidence" in codes(
        build(**args, review_receipts=(approval(partial),))
    )


def test_producer_and_review_finalization_perform_no_io(monkeypatch):
    def unexpected_io(*_args, **_kwargs):
        raise AssertionError("unexpected I/O")

    monkeypatch.setattr(builtins, "open", unexpected_io)
    monkeypatch.setattr(urllib.request, "urlopen", unexpected_io)
    pending = build()
    evidence = build(review_receipts=(approval(pending),))
    assert evidence.finalize((FIELD,)).chain_count == 1


def test_derived_fact_with_direct_clinical_evidence_can_finalize():
    _, _, fact = records()
    args = dict(
        facts=(
            replace(fact, parent_fact_ids=("fact_bbbbbbbbbbbbbbbb",), confidence=None),
        )
    )
    pending = build(**args)
    assert pending.chains[0].uncertainty == 1.0
    approved = build(**args, review_receipts=(approval(pending),))
    assert approved.finalize((FIELD,)).chain_count == 1
