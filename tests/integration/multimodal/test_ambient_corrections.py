"""Offline authority/verification/review composition over synthetic metadata."""

import pytest

from openmed.multimodal.ambient_corrections import (
    AmbientCorrectionLedger,
    ClaimReviewEvent,
    CorrectionEvent,
    CorrectionKind,
    DraftClaim,
    EvidenceChange,
    EvidenceRevision,
    ReviewAction,
    ReviewConflict,
)


@pytest.mark.integration
def test_correction_requires_fresh_local_verification_and_explicit_review():
    def reference(number):
        return f"00000000-0000-4000-8000-{number:012x}"

    reviewer = reference(900)
    grants = {"correct", "verify", "approve", "export"}
    calls = []

    def authority(handle, action):
        calls.append((handle, action))
        return handle == reviewer and action in grants

    evidence = EvidenceRevision(reference(1), 0, "sha256:" + "a" * 64)
    claim = DraftClaim(reference(2), (evidence,))
    book = AmbientCorrectionLedger((evidence,), (claim,), authority=authority)

    def review(action):
        book.apply(
            ClaimReviewEvent(
                event_id=reference(100 + len(book.events)),
                reviewer_id=reviewer,
                base_revision=book.current.revision,
                transcript_revision=book.current.transcript_revision,
                claim_id=claim.claim_id,
                claim_revision=book.current.claims[0].revision,
                action=action,
                verification_digest="sha256:" + "c" * 64
                if action is ReviewAction.VERIFY
                else None,
                reviewer_confirmed=action is ReviewAction.APPROVE,
            )
        )

    review(ReviewAction.VERIFY)
    review(ReviewAction.APPROVE)
    old_revision = book.current.revision
    book.apply(
        CorrectionEvent(
            event_id=reference(500),
            reviewer_id=reviewer,
            base_revision=old_revision,
            transcript_revision=0,
            kind=CorrectionKind.NEGATION,
            changes=(EvidenceChange(evidence, "sha256:" + "b" * 64),),
        )
    )
    with pytest.raises(ReviewConflict, match="stale_revision"):
        book.export_reviewed(
            old_revision, reviewer_id=reviewer, reviewer_confirmed=True
        )
    with pytest.raises(ReviewConflict, match="review_required"):
        book.export_reviewed(
            book.current.revision, reviewer_id=reviewer, reviewer_confirmed=True
        )
    review(ReviewAction.VERIFY)
    review(ReviewAction.APPROVE)
    output = book.export_reviewed(
        book.current.revision, reviewer_id=reviewer, reviewer_confirmed=True
    )
    assert output["draft"]["claims"][0]["claim"]["citations"][0]["revision"] == 1
    assert output["non_diagnostic"] and output["reviewer_confirmed"]
    assert {action for _, action in calls} == grants
    replay = AmbientCorrectionLedger.replay(
        (evidence,), (claim,), book.events, authority=authority
    )
    assert replay.graph == book.graph
