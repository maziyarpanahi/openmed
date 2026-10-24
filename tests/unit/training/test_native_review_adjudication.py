"""Tests for explicit adjudication of native-review language labels."""

from __future__ import annotations

import json
import socket
from dataclasses import replace
from itertools import permutations

import pytest

from openmed.clinical.journey_contracts import derived_opaque_id, sha256_digest
from openmed.eval.annotation.interchange import (
    export_annotation_tsv,
    import_annotation_tsv,
)
from openmed.eval.reviewer_disagreement import (
    DisagreementReason,
    reviewer_disagreement_report,
)
from openmed.training.native_review_adjudication import (
    AdjudicationEvent,
    AdjudicationOutcome,
    AdjudicationPolicy,
    AdjudicationRevision,
    DecisionOverlay,
    ItemAdjudication,
    LabelProjection,
    LabelState,
    NativeReviewAdjudicationError,
    Resolution,
    ReviewSpan,
    build_adjudication_event,
    export_adjudicated_annotations,
    merge_native_review_labels,
    native_review_document_id,
)

# Synthetic Swahili segment: "The patient has had a high fever since yesterday."
TEXT = "Mgonjwa ana homa kali tangu jana."
SOURCE = "synthetic-batch:item-0001"
FEVER = (12, 16, "CONDITION")
HIGH_FEVER = (12, 21, "CONDITION")
POLICY = AdjudicationPolicy(
    policy_id="sw_label_review",
    policy_version="1",
    label_codes=("CONDITION", "MEDICATION"),
    adjudicator_roles=("clinical_linguist",),
)
HASH_SECRET = b"synthetic-annotation-key-0123456789"

# Produced by the native-review packet importer for TEXT with the FEVER
# projection, a "revise" decision and a HIGH_FEVER correction. Pinning it
# keeps the two modules interchangeable without importing the packet module.
GOLDEN_NATIVE_REVIEW_DECISION = {
    "corrections": [{"end": 21, "label": "CONDITION", "start": 12}],
    "decision": "revise",
    "decision_id": "nrd_dbf48f0f5ed5baa323ebaabd37a7035b",
    "item_id": "nri_21256cbd1bf5e44f9fc5f4f878f3bbb4",
    "language": "sw",
    "packet_digest": (
        "sha256:e84ebc80e9cab4c86c884a9d1583deec90183eccd03a527e71fdf3926db84326"
    ),
    "packet_ref": "sw-assignment-0001",
    "privacy_status": "synthetic",
    "projection_digest": (
        "sha256:e987c3874a6fefa8812592609cf8c2e6434633471a7fc08cec85f65d2ec0e674"
    ),
    "provenance_digest": "sha256:" + "a" * 64,
    "reason_code": "boundary",
    "reviewer_role": "native_clinician",
    "schema_version": "openmed.eval.native_review.decision.v1",
    "source_ref": SOURCE,
    "submission_digest": (
        "sha256:083a00b94b4b336f63861a9503be345808d9ee9e4bb9d02c95df233722f6cba0"
    ),
    "task_code": "label_check",
    "text_digest": (
        "sha256:35a448ea2c787c66e934c31a5db6ef15f343a6ef8c482591ad904ee1a5e8de49"
    ),
}
EMPTY_PROJECTION_DIGEST = (
    "sha256:0fde9da45bf02d919637e5e1746a2e2833c097b10496f0c4f497ba23b571a252"
)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("adjudication must not use the network")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def _span(values):
    return ReviewSpan(*values)


def projection(
    *,
    text=TEXT,
    spans=(FEVER,),
    source_ref=SOURCE,
    language="sw",
    task_code="label_check",
):
    return LabelProjection(
        source_ref=source_ref,
        language=language,
        task_code=task_code,
        text=text,
        spans=tuple(_span(span) for span in spans),
    )


def decision(
    base,
    *,
    packet="sw-assignment-0001",
    code="accept",
    role="native_clinician",
    corrections=(),
    submission="1",
    language=None,
    text_digest=None,
    projection_digest=None,
    supersedes=None,
):
    packet_digest = sha256_digest(f"synthetic-packet:{packet}")
    item_id = derived_opaque_id("nri", packet, base.source_ref)
    submission_digest = sha256_digest(f"synthetic-submission:{packet}:{submission}")
    document = {
        "corrections": [_span(span).to_dict() for span in corrections],
        "decision": code,
        "decision_id": derived_opaque_id(
            "nrd", packet_digest, item_id, submission_digest
        ),
        "item_id": item_id,
        "language": language or base.language,
        "packet_digest": packet_digest,
        "packet_ref": packet,
        "privacy_status": "synthetic",
        "projection_digest": projection_digest or base.projection_digest,
        "provenance_digest": "sha256:" + "b" * 64,
        "reason_code": None,
        "reviewer_role": role,
        "schema_version": "openmed.eval.native_review.decision.v1",
        "source_ref": base.source_ref,
        "submission_digest": submission_digest,
        "task_code": base.task_code,
        "text_digest": text_digest or base.text_digest,
    }
    if supersedes is not None:
        document["supersedes_decision_id"] = supersedes
    return document


def conflicting_pair(base):
    return [
        decision(base, packet="sw-assignment-0001", code="accept"),
        decision(
            base,
            packet="sw-assignment-0002",
            code="revise",
            corrections=(HIGH_FEVER,),
        ),
    ]


def merge(projections, decisions=(), **kwargs):
    kwargs.setdefault("policy", POLICY)
    return merge_native_review_labels(projections, decisions, **kwargs)


def only(revision):
    assert len(revision.items) == 1
    return revision.items[0]


def test_projection_digest_matches_native_review_contract():
    assert (
        projection().projection_digest
        == (GOLDEN_NATIVE_REVIEW_DECISION["projection_digest"])
    )
    assert projection(spans=()).projection_digest == EMPTY_PROJECTION_DIGEST
    assert projection().text_digest == GOLDEN_NATIVE_REVIEW_DECISION["text_digest"]


def test_imported_native_review_decision_document_is_accepted():
    overlay = DecisionOverlay.from_decision(GOLDEN_NATIVE_REVIEW_DECISION)

    assert overlay.assignment_key == ("sw-assignment-0001", overlay.item_id)
    assert overlay.corrections == (_span(HIGH_FEVER),)
    item = only(merge([projection()], [GOLDEN_NATIVE_REVIEW_DECISION]))
    assert item.state is LabelState.PENDING
    assert item.reason_codes == ("insufficient_reviews",)
    assert item.votes[0].decision_id == GOLDEN_NATIVE_REVIEW_DECISION["decision_id"]


def test_objects_exposing_to_dict_are_accepted():
    class ImportedDecision:
        def to_dict(self):
            return dict(GOLDEN_NATIVE_REVIEW_DECISION)

    overlay = DecisionOverlay.from_decision(ImportedDecision())

    assert overlay.decision_id == GOLDEN_NATIVE_REVIEW_DECISION["decision_id"]
    assert DecisionOverlay.from_decision(overlay.to_dict()) == overlay


def test_unanimous_reviewers_resolve_only_under_policy():
    base = projection()
    decisions = [
        decision(base, packet="sw-assignment-0001"),
        decision(base, packet="sw-assignment-0002", role="clinical_linguist"),
    ]

    item = only(merge([base], decisions))

    assert item.state is LabelState.ACCEPTED
    assert item.resolution is Resolution.UNANIMOUS_AGREEMENT
    assert item.labels == (_span(FEVER),)
    assert item.label_set_digest == base.projection_digest
    assert item.reason_codes == ()
    assert {vote.reviewer_role for vote in item.votes} == {
        "clinical_linguist",
        "native_clinician",
    }

    strict = replace(POLICY, resolve_unanimous=False)
    pending = only(merge([base], decisions, policy=strict))
    assert pending.state is LabelState.PENDING
    assert pending.reason_codes == ("adjudication_required",)
    event = build_adjudication_event(
        pending,
        policy=strict,
        adjudicator_role="clinical_linguist",
        outcome="accept_projection",
    )
    accepted = only(merge([base], decisions, policy=strict, events=[event]))
    assert accepted.state is LabelState.ACCEPTED
    assert accepted.resolution is Resolution.ADJUDICATION_EVENT


def test_single_reviewer_stays_pending():
    base = projection()

    item = only(merge([base], [decision(base)]))

    assert item.state is LabelState.PENDING
    assert item.labels == ()
    assert item.reason_codes == ("insufficient_reviews",)
    assert only(merge([base])).reason_codes == ("insufficient_reviews",)


def test_unanimous_rejection_is_resolved_but_not_exported():
    base = projection()
    decisions = [
        decision(base, packet="sw-assignment-0001", code="reject"),
        decision(base, packet="sw-assignment-0002", code="reject"),
    ]

    revision = merge([base], decisions)

    assert only(revision).state is LabelState.REJECTED
    assert revision.export_labels()["labels"] == []


def test_abstention_routes_to_adjudication():
    base = projection()
    decisions = [
        decision(base, packet="sw-assignment-0001"),
        decision(base, packet="sw-assignment-0002", code="unsure"),
    ]

    item = only(merge([base], decisions))

    assert item.state is LabelState.UNRESOLVED
    assert item.reason_codes == ("reviewer_abstained",)


def test_conflicting_reviewers_remain_unresolved_until_valid_event():
    base = projection()
    decisions = conflicting_pair(base)
    revision = merge([base], decisions)
    item = only(revision)

    assert item.state is LabelState.UNRESOLVED
    assert item.reason_codes == ("reviewer_disagreement",)
    assert item.labels == ()
    assert revision.export_labels()["labels"] == []

    other_policy = replace(POLICY, policy_version="2")
    early = only(merge([base], decisions[:1]))
    invalid_events = {
        "adjudicator_role_not_allowed": build_adjudication_event(
            item,
            policy=POLICY,
            adjudicator_role="native_speaker",
            outcome="accept_projection",
        ),
        "adjudication_policy_mismatch": build_adjudication_event(
            item,
            policy=other_policy,
            adjudicator_role="clinical_linguist",
            outcome="accept_projection",
        ),
        "adjudication_stale": build_adjudication_event(
            early,
            policy=POLICY,
            adjudicator_role="clinical_linguist",
            outcome="accept_projection",
        ),
        "unknown_label": build_adjudication_event(
            item,
            policy=POLICY,
            adjudicator_role="clinical_linguist",
            outcome="accept_spans",
            spans=(ReviewSpan(12, 16, "SYMPTOM"),),
        ),
        "offset_out_of_range": build_adjudication_event(
            item,
            policy=POLICY,
            adjudicator_role="clinical_linguist",
            outcome="accept_spans",
            spans=(ReviewSpan(12, 99, "CONDITION"),),
        ),
    }
    for reason, event in invalid_events.items():
        blocked = only(merge([base], decisions, events=[event]))
        assert blocked.state is LabelState.UNRESOLVED, reason
        assert reason in blocked.reason_codes
        assert "reviewer_disagreement" in blocked.reason_codes
        assert blocked.ignored_event_ids == (event.event_id,)
        assert blocked.event_id is None

    deferred = build_adjudication_event(
        item, policy=POLICY, adjudicator_role="clinical_linguist", outcome="defer"
    )
    held = only(merge([base], decisions, events=[deferred]))
    assert held.state is LabelState.UNRESOLVED
    assert held.reason_codes == ("adjudication_deferred",)
    assert held.event_id == deferred.event_id

    valid = build_adjudication_event(
        item,
        policy=POLICY,
        adjudicator_role="clinical_linguist",
        outcome=AdjudicationOutcome.ACCEPT_SPANS,
        spans=(_span(HIGH_FEVER),),
        reason_code="boundary",
    )
    resolved = only(merge([base], decisions, events=[valid], previous=revision))
    assert resolved.state is LabelState.ACCEPTED
    assert resolved.resolution is Resolution.ADJUDICATION_EVENT
    assert resolved.labels == (_span(HIGH_FEVER),)
    assert resolved.event_id == valid.event_id
    assert resolved.adjudicator_role == "clinical_linguist"
    assert resolved.prior_state is LabelState.UNRESOLVED
    assert not resolved.prior_acceptance_revoked


def test_adjudicated_rejection_resolves_without_labels():
    base = projection()
    decisions = conflicting_pair(base)
    item = only(merge([base], decisions))
    event = build_adjudication_event(
        item,
        policy=POLICY,
        adjudicator_role="clinical_linguist",
        outcome="reject_projection",
    )

    rejected = only(merge([base], decisions, events=[event]))

    assert rejected.state is LabelState.REJECTED
    assert rejected.resolution is Resolution.ADJUDICATION_EVENT
    assert rejected.labels == ()


def test_conflicting_adjudication_heads_stay_unresolved():
    base = projection()
    decisions = conflicting_pair(base)
    item = only(merge([base], decisions))
    first = build_adjudication_event(
        item,
        policy=POLICY,
        adjudicator_role="clinical_linguist",
        outcome="accept_projection",
    )
    second = build_adjudication_event(
        item,
        policy=POLICY,
        adjudicator_role="clinical_linguist",
        outcome="accept_spans",
        spans=(_span(HIGH_FEVER),),
    )

    forked = only(merge([base], decisions, events=[first, second]))
    assert forked.state is LabelState.UNRESOLVED
    assert "conflicting_adjudication" in forked.reason_codes
    assert forked.ignored_event_ids == tuple(sorted([first.event_id, second.event_id]))

    revision = build_adjudication_event(
        item,
        policy=POLICY,
        adjudicator_role="clinical_linguist",
        outcome="accept_spans",
        spans=(_span(HIGH_FEVER),),
        supersedes_event_id=first.event_id,
    )
    chained = only(merge([base], decisions, events=[first, revision]))
    assert chained.state is LabelState.ACCEPTED
    assert chained.event_id == revision.event_id
    assert chained.superseded_event_ids == (first.event_id,)


def test_superseding_event_revokes_previously_accepted_labels():
    base = projection()
    decisions = conflicting_pair(base)
    item = only(merge([base], decisions))
    first = build_adjudication_event(
        item,
        policy=POLICY,
        adjudicator_role="clinical_linguist",
        outcome="accept_projection",
    )
    accepted = merge([base], decisions, events=[first])
    revised_event = build_adjudication_event(
        item,
        policy=POLICY,
        adjudicator_role="clinical_linguist",
        outcome="accept_spans",
        spans=(_span(HIGH_FEVER),),
        supersedes_event_id=first.event_id,
    )

    revised = merge([base], decisions, events=[first, revised_event], previous=accepted)
    entry = only(revised)

    assert revised.parent_revision_id == accepted.revision_id
    assert entry.state is LabelState.ACCEPTED
    assert entry.prior_label_set_digest == only(accepted).label_set_digest
    assert entry.prior_acceptance_revoked
    assert entry.labels == (_span(HIGH_FEVER),)


def _accepted_revision(base):
    decisions = [
        decision(base, packet="sw-assignment-0001"),
        decision(base, packet="sw-assignment-0002"),
    ]
    revision = merge([base], decisions)
    assert only(revision).state is LabelState.ACCEPTED
    return revision, decisions


def test_stale_projection_invalidates_prior_acceptance():
    base = projection()
    revision, decisions = _accepted_revision(base)
    regenerated = projection(spans=(HIGH_FEVER,))

    item = only(merge([regenerated], decisions, previous=revision))

    assert item.state is LabelState.INVALIDATED
    assert item.reason_codes == ("stale_projection",)
    assert item.labels == ()
    assert item.prior_acceptance_revoked
    assert item.excluded_decision_ids == tuple(
        sorted(entry["decision_id"] for entry in decisions)
    )


def test_changed_text_invalidates_prior_acceptance():
    base = projection()
    revision, decisions = _accepted_revision(base)
    edited = projection(text="Mgonjwa ana homa kali tangu juzi.")

    next_revision = merge([edited], decisions, previous=revision)
    item = only(next_revision)

    assert item.state is LabelState.INVALIDATED
    assert item.reason_codes == ("text_changed",)
    assert item.prior_acceptance_revoked
    assert next_revision.export_labels()["labels"] == []


def test_fresh_reviews_replace_stale_evidence():
    base = projection()
    revision, stale = _accepted_revision(base)
    edited = projection(text="Mgonjwa ana homa kali tangu juzi.")
    fresh = [
        decision(edited, packet="sw-assignment-0003"),
        decision(edited, packet="sw-assignment-0004"),
    ]

    item = only(merge([edited], stale + fresh, previous=revision))

    assert item.state is LabelState.ACCEPTED
    assert item.reason_codes == ("text_changed",)
    assert len(item.excluded_decision_ids) == 2
    assert {vote.decision_id for vote in item.votes} == {
        entry["decision_id"] for entry in fresh
    }
    # Same labels on different text do not carry the prior acceptance over.
    assert item.prior_acceptance_revoked


def test_duplicate_decisions_invalidate_prior_acceptance():
    base = projection()
    revision, decisions = _accepted_revision(base)
    resubmission = decision(base, packet="sw-assignment-0001", submission="2")

    item = only(merge([base], [*decisions, resubmission], previous=revision))

    assert item.state is LabelState.INVALIDATED
    assert item.reason_codes == ("duplicate_decision",)
    assert item.prior_acceptance_revoked

    explicit = decision(
        base,
        packet="sw-assignment-0001",
        submission="2",
        supersedes=decisions[0]["decision_id"],
    )
    restored = only(merge([base], [*decisions, explicit], previous=revision))
    assert restored.state is LabelState.ACCEPTED
    assert restored.superseded_decision_ids == (decisions[0]["decision_id"],)
    assert explicit["decision_id"] in {vote.decision_id for vote in restored.votes}
    assert not restored.prior_acceptance_revoked


def test_conflicting_copies_of_one_decision_invalidate():
    base = projection()
    _, decisions = _accepted_revision(base)
    tampered = dict(decisions[0], decision="reject")

    item = only(merge([base], [*decisions, tampered]))

    assert item.state is LabelState.INVALIDATED
    assert item.reason_codes == ("duplicate_decision",)


def test_identical_reimport_is_idempotent():
    base = projection()
    decisions = conflicting_pair(base)

    once = merge([base], decisions)
    twice = merge([base], decisions + decisions)

    assert once.revision_id == twice.revision_id
    assert once.to_json() == twice.to_json()


def test_supersession_cycles_and_cross_assignment_pointers_fail_closed():
    base = projection()
    first = decision(base, submission="1")
    second = decision(base, submission="2", supersedes=first["decision_id"])
    cyclic_first = dict(first, supersedes_decision_id=second["decision_id"])

    cycle = only(merge([base], [cyclic_first, second]))
    assert cycle.state is LabelState.INVALIDATED
    assert cycle.reason_codes == ("duplicate_decision",)

    other = decision(base, packet="sw-assignment-0002")
    crossing = decision(base, submission="3", supersedes=other["decision_id"])
    crossed = only(merge([base], [other, crossing]))
    assert crossed.state is LabelState.INVALIDATED
    assert "supersession_scope_mismatch" in crossed.reason_codes


def test_decision_scope_and_correction_contract_are_enforced():
    base = projection()
    good = decision(base, packet="sw-assignment-0002")
    cases = {
        "decision_scope_mismatch": decision(base, language="sw-KE"),
        "invalid_correction": decision(base, corrections=(HIGH_FEVER,)),
        "unknown_label": decision(
            base, code="revise", corrections=((12, 16, "SYMPTOM"),)
        ),
        "offset_out_of_range": decision(
            base, code="revise", corrections=((12, 80, "CONDITION"),)
        ),
    }
    for reason, bad in cases.items():
        item = only(merge([base], [good, bad]))
        assert item.state is LabelState.INVALIDATED, reason
        assert reason in item.reason_codes


def test_offsets_are_revalidated_on_grapheme_boundaries():
    # Synthetic Hindi: "The patient has a fever."
    text = "मरीज़ को बुखार है।"
    word = "बुखार"
    start = text.index(word)
    whole = (start, start + len(word), "CONDITION")
    split = (start, start + 1, "CONDITION")  # Cuts the vowel sign off its base.
    base = projection(text=text, spans=(whole,), language="hi")
    decisions = [
        decision(base, packet="hi-assignment-0001"),
        decision(
            base, packet="hi-assignment-0002", code="revise", corrections=(split,)
        ),
    ]

    item = only(merge([base], decisions))
    assert item.state is LabelState.INVALIDATED
    assert item.reason_codes == ("split_grapheme",)

    bad_projection = projection(text=text, spans=(split,), language="hi")
    assert only(merge([bad_projection])).state is LabelState.INVALIDATED

    disagreement = [
        decision(base, packet="hi-assignment-0001"),
        decision(base, packet="hi-assignment-0002", code="reject"),
    ]
    unresolved = only(merge([base], disagreement))
    event = build_adjudication_event(
        unresolved,
        policy=POLICY,
        adjudicator_role="clinical_linguist",
        outcome="accept_spans",
        spans=(_span(split),),
    )
    held = only(merge([base], disagreement, events=[event]))
    assert held.state is LabelState.UNRESOLVED
    assert "split_grapheme" in held.reason_codes


def test_reviewer_roles_outside_policy_are_not_counted():
    base = projection()
    policy = replace(POLICY, reviewer_roles=("native_clinician",))
    decisions = [
        decision(base, packet="sw-assignment-0001"),
        decision(base, packet="sw-assignment-0002", role="native_speaker"),
    ]

    item = only(merge([base], decisions, policy=policy))

    assert item.state is LabelState.PENDING
    assert item.reason_codes == ("insufficient_reviews", "reviewer_role_not_counted")
    assert len(item.excluded_decision_ids) == 1


@pytest.mark.parametrize(
    ("field", "code"),
    [
        ("reviewer_name", "reviewer_identity_forbidden"),
        ("reviewer_email", "reviewer_identity_forbidden"),
        ("annotator_id", "reviewer_identity_forbidden"),
        ("comment", "free_text_forbidden"),
        ("snippet", "free_text_forbidden"),
        ("concordance", "free_text_forbidden"),
        ("extra", "unknown_field"),
    ],
)
def test_identity_and_free_text_fields_are_refused(field, code):
    document = dict(GOLDEN_NATIVE_REVIEW_DECISION, **{field: "Asha Mwangi"})

    with pytest.raises(NativeReviewAdjudicationError, match=code) as error:
        DecisionOverlay.from_decision(document)

    assert "Asha" not in str(error.value)


def test_decision_lineage_and_privacy_status_are_validated():
    with pytest.raises(NativeReviewAdjudicationError, match="lineage_mismatch"):
        DecisionOverlay.from_decision(
            dict(GOLDEN_NATIVE_REVIEW_DECISION, submission_digest="sha256:" + "c" * 64)
        )
    with pytest.raises(NativeReviewAdjudicationError, match="not_reviewable"):
        DecisionOverlay.from_decision(
            dict(GOLDEN_NATIVE_REVIEW_DECISION, privacy_status="identified")
        )
    with pytest.raises(NativeReviewAdjudicationError, match="self_supersession"):
        DecisionOverlay.from_decision(
            GOLDEN_NATIVE_REVIEW_DECISION,
            supersedes_decision_id=GOLDEN_NATIVE_REVIEW_DECISION["decision_id"],
        )


def test_merge_rejects_unknown_sources_and_duplicate_projections():
    base = projection()
    stranger = projection(source_ref="synthetic-batch:item-0002")

    with pytest.raises(NativeReviewAdjudicationError, match="unknown_source"):
        merge([base], [decision(stranger)])
    with pytest.raises(NativeReviewAdjudicationError, match="duplicate_source_ref"):
        merge([base, base])
    with pytest.raises(NativeReviewAdjudicationError, match="invalid_policy"):
        merge_native_review_labels([base], policy=POLICY.to_dict())
    previous = merge([base, stranger])
    with pytest.raises(NativeReviewAdjudicationError, match="missing_projection"):
        merge([base], previous=previous)


def test_policy_contract_is_explicit_and_digest_bound():
    with pytest.raises(NativeReviewAdjudicationError, match="overlapping_codes"):
        replace(POLICY, reject_decision_codes=("accept",))
    with pytest.raises(NativeReviewAdjudicationError, match="case_collision"):
        replace(POLICY, label_codes=("CONDITION", "condition"))
    with pytest.raises(NativeReviewAdjudicationError, match="invalid_count"):
        replace(POLICY, minimum_reviewers=True)
    with pytest.raises(NativeReviewAdjudicationError, match="invalid_count"):
        replace(POLICY, adjudicator_roles=())

    assert AdjudicationPolicy.from_dict(POLICY.to_dict()) == POLICY
    assert replace(POLICY, policy_version="2").policy_digest != POLICY.policy_digest


def test_events_are_content_addressed():
    base = projection()
    item = only(merge([base], conflicting_pair(base)))
    event = build_adjudication_event(
        item, policy=POLICY, adjudicator_role="clinical_linguist", outcome="defer"
    )

    assert AdjudicationEvent.from_dict(event.to_dict()) == event
    with pytest.raises(NativeReviewAdjudicationError, match="lineage_mismatch"):
        AdjudicationEvent.from_dict(dict(event.to_dict(), outcome="reject_projection"))
    with pytest.raises(NativeReviewAdjudicationError, match="outcome_mismatch"):
        build_adjudication_event(
            item,
            policy=POLICY,
            adjudicator_role="clinical_linguist",
            outcome="accept_spans",
        )
    with pytest.raises(NativeReviewAdjudicationError, match="reviewer_identity"):
        AdjudicationEvent.from_dict(dict(event.to_dict(), adjudicator_name="x"))


def test_revision_round_trips_and_detects_tampering():
    base = projection()
    revision = merge([base], conflicting_pair(base))
    document = json.loads(revision.to_json())

    assert AdjudicationRevision.from_dict(document) == revision

    forged_id = dict(document, revision_id="nrr_" + "0" * 32)
    with pytest.raises(NativeReviewAdjudicationError, match="lineage_mismatch"):
        AdjudicationRevision.from_dict(forged_id)
    forged_item = dict(document["items"][0], state="accepted")
    with pytest.raises(NativeReviewAdjudicationError, match="inconsistent"):
        ItemAdjudication.from_dict(forged_item)
    forged_basis = dict(document["items"][0], votes=document["items"][0]["votes"][:1])
    with pytest.raises(NativeReviewAdjudicationError, match="lineage_mismatch"):
        ItemAdjudication.from_dict(forged_basis)


def test_merge_is_deterministic_across_input_order():
    first = projection()
    second = projection(source_ref="synthetic-batch:item-0002", spans=())
    decisions = conflicting_pair(first) + [
        decision(second, packet="sw-assignment-0003", code="reject"),
        decision(second, packet="sw-assignment-0004", code="reject"),
    ]
    expected = merge([first, second], decisions).to_json()

    for order in permutations(decisions):
        assert merge([second, first], list(order)).to_json() == expected


def test_audit_and_label_exports_keep_lineage_without_identity_or_text():
    base = projection()
    other = projection(source_ref="synthetic-batch:item-0002")
    decisions = conflicting_pair(base) + [
        decision(other, packet="sw-assignment-0003"),
        decision(other, packet="sw-assignment-0004", role="clinical_linguist"),
    ]
    item = merge([base, other], decisions).item(SOURCE)
    event = build_adjudication_event(
        item,
        policy=POLICY,
        adjudicator_role="clinical_linguist",
        outcome="accept_spans",
        spans=(_span(HIGH_FEVER),),
    )
    revision = merge([base, other], decisions, events=[event])

    labels = revision.export_labels()
    audit = revision.to_audit_dict()
    label_json = json.dumps(labels)
    audit_json = json.dumps(audit)

    assert [entry["source_ref"] for entry in labels["labels"]] == [
        SOURCE,
        other.source_ref,
    ]
    lineage = labels["labels"][0]["lineage"]
    assert {entry["decision_id"] for entry in lineage["decisions"]} == {
        entry["decision_id"] for entry in conflicting_pair(base)
    }
    assert lineage["event_id"] == event.event_id
    assert lineage["adjudicator_role"] == "clinical_linguist"
    assert audit["state_counts"]["accepted"] == 2
    assert audit["items"][0]["subject_digest"].startswith("sha256:")
    for artifact in (label_json, audit_json):
        for forbidden in ("Mgonjwa", "homa", "kali", "sw-assignment", "packet_ref"):
            assert forbidden not in artifact
    assert SOURCE not in audit_json


def test_reviewer_disagreement_metric_reuses_counted_votes():
    projections = [projection(source_ref=f"synthetic-batch:item-{n}") for n in range(4)]
    decisions = []
    for index, base in enumerate(projections):
        if index < 2:
            decisions += [
                decision(base, packet=f"sw-assignment-{index}a"),
                decision(base, packet=f"sw-assignment-{index}b"),
            ]
        else:
            decisions += [
                decision(base, packet=f"sw-assignment-{index}a"),
                decision(base, packet=f"sw-assignment-{index}b", code="reject"),
            ]
    revision = merge(projections, decisions)
    events = [
        build_adjudication_event(
            item,
            policy=POLICY,
            adjudicator_role="clinical_linguist",
            outcome="reject_projection",
        )
        for item in revision.items
        if item.state is LabelState.UNRESOLVED
    ]
    adjudicated = merge(projections, decisions, events=events, previous=revision)

    records = adjudicated.to_reviewer_decisions(
        reason=DisagreementReason.LABEL_DEFINITION
    )
    report = reviewer_disagreement_report(records, minimum_cell_size=2).to_dict()

    assert len(records) == 8
    assert report["agreement"]["numerator"] == 2
    assert report["agreement"]["denominator"] == 4
    assert report["adjudication"]["rate"] == 1.0
    assert all(record.case_id.startswith("sha256:") for record in records)


def test_accepted_labels_export_to_annotation_interchange():
    base = projection(spans=(FEVER,))
    revision, _ = _accepted_revision(base)

    envelope = export_adjudicated_annotations(revision, [base], hash_secret=HASH_SECRET)
    exported = export_annotation_tsv(envelope)
    record = envelope.records[0]

    assert import_annotation_tsv(exported.text).envelope_digest == (
        envelope.envelope_digest
    )
    assert record.document_id == native_review_document_id(SOURCE)
    assert record.result["label"] == "condition"
    assert dict(record.metadata) == {
        "batch_id": revision.revision_id,
        "review_state": "unanimous_agreement",
        "source_format": "native_review",
    }
    assert "homa" not in exported.text and SOURCE not in exported.text
    other_key = export_adjudicated_annotations(
        revision, [base], hash_secret=b"another-synthetic-key-0123456789"
    )
    assert other_key.records[0].result["surface_hash"] != record.result["surface_hash"]

    edited = projection(text="Mgonjwa ana homa kali tangu juzi.")
    with pytest.raises(NativeReviewAdjudicationError, match="text_changed"):
        export_adjudicated_annotations(revision, [edited], hash_secret=HASH_SECRET)
    with pytest.raises(NativeReviewAdjudicationError, match="missing_projection"):
        export_adjudicated_annotations(revision, [], hash_secret=HASH_SECRET)
    with pytest.raises(NativeReviewAdjudicationError, match="invalid_key"):
        export_adjudicated_annotations(revision, [base], hash_secret=b"")


def test_projection_repr_and_errors_do_not_expose_text():
    base = projection()

    assert "Mgonjwa" not in repr(base)
    with pytest.raises(NativeReviewAdjudicationError) as error:
        projection(spans=((12, 99, "CONDITION"),))
    assert "Mgonjwa" not in str(error.value)
    with pytest.raises(NativeReviewAdjudicationError) as error:
        projection(source_ref="Asha Mwangi, bed 4")
    assert "Asha" not in str(error.value)
