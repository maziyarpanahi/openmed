"""Synthetic, offline correction propagation and negative controls."""

import hashlib
import json
from dataclasses import asdict, replace

import pytest

from openmed.multimodal.ambient_corrections import (
    ADVISORY,
    AmbientCorrectionLedger,
    ClaimReviewEvent,
    ConflictCode,
    CorrectionEvent,
    CorrectionKind,
    DraftClaim,
    EvidenceChange,
    EvidenceRevision,
    ReviewAction,
    ReviewConflict,
)


def opaque(number):
    return f"00000000-0000-4000-8000-{number:012x}"


def digest(value):
    # Synthetic test strings only; applications should use caller-keyed digests.
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


REVIEWER = opaque(900)


def authority(reviewer, action):
    return reviewer == REVIEWER and action in {"correct", "verify", "approve", "export"}


def fixture():
    evidence = (
        EvidenceRevision(opaque(1), 0, digest("synthetic medication A")),
        EvidenceRevision(opaque(2), 0, digest("synthetic denies symptom")),
        EvidenceRevision(opaque(3), 0, digest("synthetic patient speaker")),
        EvidenceRevision(opaque(4), 0, digest("synthetic unrelated claim")),
    )
    claims = (
        DraftClaim(opaque(11), evidence[:3]),
        DraftClaim(opaque(12), evidence[:3]),
        DraftClaim(opaque(13), (evidence[3],)),
    )
    return evidence, claims


def ledger():
    return AmbientCorrectionLedger(*fixture(), authority=authority)


def review(book, claim_id, action, *, confirmed=True):
    state = next(s for s in book.current.claims if s.claim.claim_id == claim_id)
    event = ClaimReviewEvent(
        event_id=opaque(100 + len(book.events)),
        reviewer_id=REVIEWER,
        base_revision=book.current.revision,
        transcript_revision=book.current.transcript_revision,
        claim_id=claim_id,
        claim_revision=state.revision,
        action=action,
        verification_digest=digest("synthetic verification")
        if action is ReviewAction.VERIFY
        else None,
        reviewer_confirmed=confirmed if action is ReviewAction.APPROVE else False,
    )
    book.apply(event)
    return event


def approve_all(book):
    for state in book.current.claims:
        review(book, state.claim.claim_id, ReviewAction.VERIFY)
        review(book, state.claim.claim_id, ReviewAction.APPROVE)


def correction(book, index=0, kind=CorrectionKind.MEDICATION):
    return CorrectionEvent(
        event_id=opaque(500 + len(book.events)),
        reviewer_id=REVIEWER,
        base_revision=book.current.revision,
        transcript_revision=book.current.transcript_revision,
        kind=kind,
        changes=(
            EvidenceChange(
                book.current.evidence[index], digest("synthetic correction")
            ),
        ),
    )


def assert_conflict(code, operation):
    with pytest.raises(ReviewConflict) as exc:
        operation()
    assert exc.value.code is code
    assert str(exc.value) == code.value


@pytest.mark.parametrize(
    "index,kind",
    list(
        enumerate(
            (CorrectionKind.MEDICATION, CorrectionKind.NEGATION, CorrectionKind.SPEAKER)
        )
    ),
)
def test_every_dependent_claim_loses_verification_and_approval(index, kind):
    book = ledger()
    approve_all(book)
    old = book.current
    old_export = book.export_reviewed(
        old.revision, reviewer_id=REVIEWER, reviewer_confirmed=True
    )
    node = book.apply(correction(book, index, kind))
    assert node.parent_revision == old.revision
    assert node.transcript_revision == old.transcript_revision + 1
    for before, after in zip(old.claims[:2], node.claims[:2]):
        assert after.claim.claim_id == before.claim.claim_id
        assert after.revision == before.revision + 1
        assert after.verification_event is None and after.approval_event is None
        assert after.claim.citations[index].revision == 1
        for citation_index in set(range(3)) - {index}:
            assert (
                after.claim.citations[citation_index]
                is before.claim.citations[citation_index]
            )
    assert node.claims[2] is old.claims[2]
    assert old.evidence[index].revision == 0
    assert all(s.approval_event for s in old.claims)
    assert old_export["advisory"] == ADVISORY
    assert_conflict(
        ConflictCode.STALE_REVISION,
        lambda: book.export_reviewed(
            old.revision, reviewer_id=REVIEWER, reviewer_confirmed=True
        ),
    )
    assert_conflict(
        ConflictCode.REVIEW_REQUIRED,
        lambda: book.export_reviewed(
            node.revision, reviewer_id=REVIEWER, reviewer_confirmed=True
        ),
    )
    for state in node.claims[:2]:
        assert_conflict(
            ConflictCode.VERIFICATION_REQUIRED,
            lambda: review(book, state.claim.claim_id, ReviewAction.APPROVE),
        )
        review(book, state.claim.claim_id, ReviewAction.VERIFY)
        assert_conflict(
            ConflictCode.REVIEW_REQUIRED,
            lambda: book.export_reviewed(
                book.current.revision, reviewer_id=REVIEWER, reviewer_confirmed=True
            ),
        )
        review(book, state.claim.claim_id, ReviewAction.APPROVE)
    exported = book.export_reviewed(
        book.current.revision, reviewer_id=REVIEWER, reviewer_confirmed=True
    )
    assert exported["non_diagnostic"] is True and exported["reviewer_confirmed"] is True


def test_replay_reconstructs_identical_graph_including_old_approvals():
    book = ledger()
    approve_all(book)
    book.apply(correction(book))
    approve_all(book)
    book.apply(correction(book, 2, CorrectionKind.SPEAKER))
    rebuilt = AmbientCorrectionLedger.replay(
        *fixture(), book.events, authority=authority
    )
    assert rebuilt.graph == book.graph
    assert rebuilt.events == book.events
    assert json.dumps(
        [asdict(node) for node in rebuilt.graph], sort_keys=True
    ) == json.dumps([asdict(node) for node in book.graph], sort_keys=True)
    assert_conflict(
        ConflictCode.REVIEW_REQUIRED,
        lambda: rebuilt.export_reviewed(
            rebuilt.current.revision, reviewer_id=REVIEWER, reviewer_confirmed=True
        ),
    )
    assert_conflict(
        ConflictCode.AUTHORITY_REQUIRED,
        lambda: AmbientCorrectionLedger.replay(
            *fixture(), book.events, authority=lambda *_: False
        ),
    )


def test_conflicting_stale_duplicate_and_atomic_edits():
    book = ledger()
    first = correction(book)
    simultaneous = replace(first, event_id=opaque(700))
    book.apply(first)
    graph, events = book.graph, book.events
    assert_conflict(ConflictCode.DUPLICATE_EVENT, lambda: book.apply(first))
    assert_conflict(ConflictCode.STALE_REVISION, lambda: book.apply(simultaneous))
    rebound = replace(
        simultaneous,
        base_revision=book.current.revision,
        transcript_revision=book.current.transcript_revision,
    )
    assert_conflict(ConflictCode.STALE_EVIDENCE, lambda: book.apply(rebound))
    unknown = EvidenceRevision(opaque(999), 0, digest("unknown"))
    atomic = replace(
        correction(book, 1),
        changes=(
            EvidenceChange(book.current.evidence[1], digest("changed")),
            EvidenceChange(unknown, digest("new")),
        ),
    )
    assert_conflict(ConflictCode.UNKNOWN_REFERENCE, lambda: book.apply(atomic))
    assert book.graph == graph and book.events == events


def test_multi_evidence_correction_and_unused_evidence():
    evidence, claims = fixture()
    book = AmbientCorrectionLedger(evidence, claims[:2], authority=authority)
    approve_all(book)
    old = book.current
    book.apply(correction(book, 3))
    assert book.current.claims == old.claims
    book.export_reviewed(
        book.current.revision, reviewer_id=REVIEWER, reviewer_confirmed=True
    )
    event = correction(book)
    book.apply(
        replace(
            event,
            changes=(
                event.changes[0],
                EvidenceChange(book.current.evidence[1], digest("second change")),
            ),
        )
    )
    assert all(
        s.revision == 1 and s.approval_event is None for s in book.current.claims
    )
    assert [e.revision for e in book.current.evidence] == [1, 1, 0, 1]


def test_reverification_clears_approval_and_confirmation_is_explicit():
    book = ledger()
    approve_all(book)
    assert_conflict(
        ConflictCode.CONFIRMATION_REQUIRED,
        lambda: book.export_reviewed(
            book.current.revision, reviewer_id=REVIEWER, reviewer_confirmed=False
        ),
    )
    review(book, opaque(11), ReviewAction.VERIFY)
    assert book.current.claims[0].approval_event is None
    assert_conflict(
        ConflictCode.CONFIRMATION_REQUIRED,
        lambda: review(book, opaque(11), ReviewAction.APPROVE, confirmed=False),
    )
    assert_conflict(
        ConflictCode.REVIEW_REQUIRED,
        lambda: book.export_reviewed(
            book.current.revision, reviewer_id=REVIEWER, reviewer_confirmed=True
        ),
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("base_revision", 42),
        ("transcript_revision", 42),
        ("claim_revision", 42),
        ("claim_id", opaque(999)),
    ],
)
def test_review_is_bound_to_exact_revision_and_claim(field, value):
    book = ledger()
    event = review(book, opaque(11), ReviewAction.VERIFY)
    event = (
        replace(
            event,
            event_id=opaque(750),
            base_revision=book.current.revision,
            **{field: value},
        )
        if field != "base_revision"
        else replace(event, event_id=opaque(750), base_revision=value)
    )
    code = (
        ConflictCode.UNKNOWN_REFERENCE
        if field == "claim_id"
        else ConflictCode.STALE_REVISION
    )
    prior = book.graph
    assert_conflict(code, lambda: book.apply(event))
    assert book.graph == prior


@pytest.mark.parametrize("grant", [False, "yes", 1, None])
def test_authority_denies_non_boolean_grants(grant):
    book = AmbientCorrectionLedger(*fixture(), authority=lambda *_: grant)
    assert_conflict(
        ConflictCode.AUTHORITY_REQUIRED, lambda: book.apply(correction(book))
    )
    assert len(book.graph) == 1


def test_authority_failure_does_not_disclose_private_details():
    private = "Synthetic Alice /private/source token=example"

    def deny(*_):
        raise RuntimeError(private)

    book = AmbientCorrectionLedger(*fixture(), authority=deny)
    with pytest.raises(ReviewConflict) as exc:
        book.apply(correction(book))
    assert private not in str(exc.value)
    assert exc.value.__suppress_context__
    assert book.events == ()


@pytest.mark.parametrize(
    "value",
    [
        "Synthetic Alice",
        "/private/note",
        "Bearer synthetic",
        "123456",
        "00000000-0000-4000-8000-000000000001\n",
        None,
    ],
)
def test_identifiers_are_opaque_and_errors_never_echo_input(value):
    assert_conflict(
        ConflictCode.INVALID_METADATA, lambda: EvidenceRevision(value, 0, digest("x"))
    )


@pytest.mark.parametrize("value", [True, -1, 1.0, 1 << 53, "0"])
def test_revision_validation(value):
    assert_conflict(
        ConflictCode.INVALID_METADATA,
        lambda: EvidenceRevision(opaque(1), value, digest("x")),
    )


def test_invalid_initial_dependencies_and_events_fail_closed():
    evidence, claims = fixture()
    invalid = DraftClaim(opaque(21), (replace(evidence[0], revision=1),))
    assert_conflict(
        ConflictCode.INVALID_METADATA,
        lambda: AmbientCorrectionLedger(evidence, (invalid,), authority=authority),
    )
    assert_conflict(
        ConflictCode.INVALID_METADATA,
        lambda: AmbientCorrectionLedger(
            evidence + evidence, claims, authority=authority
        ),
    )
    assert_conflict(
        ConflictCode.INVALID_METADATA,
        lambda: AmbientCorrectionLedger(evidence, claims + claims, authority=authority),
    )
    assert_conflict(
        ConflictCode.INVALID_METADATA,
        lambda: DraftClaim(opaque(11), (evidence[0], evidence[0])),
    )
    assert_conflict(ConflictCode.INVALID_METADATA, lambda: DraftClaim(opaque(11), ()))
    assert_conflict(
        ConflictCode.INVALID_METADATA,
        lambda: EvidenceChange(evidence[0], evidence[0].digest),
    )
    book = ledger()
    event = correction(book)
    for changes in ((), event.changes + event.changes):
        assert_conflict(
            ConflictCode.INVALID_METADATA, lambda: replace(event, changes=changes)
        )
    assert_conflict(
        ConflictCode.INVALID_METADATA, lambda: replace(event, schema_version=True)
    )
    assert_conflict(
        ConflictCode.INVALID_METADATA, lambda: replace(event, schema_version=2)
    )
    assert_conflict(
        ConflictCode.INVALID_METADATA, lambda: replace(event, kind="medication")
    )
    assert_conflict(
        ConflictCode.INVALID_METADATA, lambda: book.apply("private payload")
    )


def test_safe_reports_events_and_repr_exclude_synthetic_source_payloads():
    book = ledger()
    approve_all(book)
    book.apply(correction(book))
    safe = (
        repr(book.graph)
        + repr(book.events)
        + json.dumps([asdict(e) for e in book.events], sort_keys=True)
    )
    for source in (
        "synthetic medication A",
        "synthetic denies symptom",
        "synthetic patient speaker",
        "synthetic correction",
        "synthetic verification",
    ):
        assert source not in safe
    assert not hasattr(book, "source_text")


def test_initial_provider_transcript_revision_is_bound_and_replayed():
    book = AmbientCorrectionLedger(
        *fixture(), authority=authority, transcript_revision=17
    )
    event = correction(book)
    assert event.transcript_revision == 17
    assert_conflict(
        ConflictCode.STALE_REVISION,
        lambda: book.apply(replace(event, transcript_revision=0)),
    )
    book.apply(event)
    assert book.current.transcript_revision == 18
    replay = AmbientCorrectionLedger.replay(
        *fixture(), book.events, authority=authority, transcript_revision=17
    )
    assert replay.graph == book.graph
