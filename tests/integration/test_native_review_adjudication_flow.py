"""Offline revision chain for adjudicated native-review language labels."""

from __future__ import annotations

import json
import socket

import pytest

from openmed.clinical.journey_contracts import derived_opaque_id, sha256_digest
from openmed.eval.annotation.interchange import (
    export_annotation_tsv,
    import_annotation_tsv,
)
from openmed.eval.reviewer_disagreement import reviewer_disagreement_report
from openmed.training.native_review_adjudication import (
    AdjudicationPolicy,
    AdjudicationRevision,
    LabelProjection,
    LabelState,
    ReviewSpan,
    build_adjudication_event,
    export_adjudicated_annotations,
    merge_native_review_labels,
)

pytestmark = pytest.mark.integration

# Synthetic Swahili segments; none describe a real person.
FEVER = "Mgonjwa ana homa kali tangu jana."
MEDICINE = "Anatumia dawa ya paracetamol kila siku."
COUGH = "Mtoto ana kikohozi kikavu."
POLICY = AdjudicationPolicy(
    policy_id="sw_label_review",
    policy_version="2026.10",
    label_codes=("CONDITION", "MEDICATION"),
    adjudicator_roles=("clinical_linguist",),
)


def _projection(source_ref, text, start, end, label):
    return LabelProjection(
        source_ref=source_ref,
        language="sw",
        task_code="label_check",
        text=text,
        spans=(ReviewSpan(start, end, label),),
    )


def _decision(base, packet, code="accept", corrections=(), submission="1", **extra):
    packet_digest = sha256_digest(f"synthetic-packet:{packet}")
    item_id = derived_opaque_id("nri", packet, base.source_ref)
    submission_digest = sha256_digest(f"synthetic-submission:{packet}:{submission}")
    document = {
        "corrections": [span.to_dict() for span in corrections],
        "decision": code,
        "decision_id": derived_opaque_id(
            "nrd", packet_digest, item_id, submission_digest
        ),
        "item_id": item_id,
        "language": "sw",
        "packet_digest": packet_digest,
        "packet_ref": packet,
        "privacy_status": "synthetic",
        "projection_digest": base.projection_digest,
        "provenance_digest": "sha256:" + "d" * 64,
        "reason_code": None,
        "reviewer_role": "native_clinician",
        "schema_version": "openmed.eval.native_review.decision.v1",
        "source_ref": base.source_ref,
        "submission_digest": submission_digest,
        "task_code": "label_check",
        "text_digest": base.text_digest,
    }
    document.update(extra)
    return document


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("native-review adjudication must stay offline")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def test_revision_chain_resolves_invalidates_and_reaccepts(tmp_path):
    fever = _projection("synthetic:fever", FEVER, 12, 16, "CONDITION")
    medicine = _projection("synthetic:medicine", MEDICINE, 17, 28, "MEDICATION")
    cough = _projection("synthetic:cough", COUGH, 11, 25, "CONDITION")
    projections = [fever, medicine, cough]
    decisions = [
        _decision(fever, "sw-a1"),
        _decision(
            fever,
            "sw-a2",
            code="revise",
            corrections=(ReviewSpan(12, 21, "CONDITION"),),
        ),
        _decision(medicine, "sw-b1"),
        _decision(medicine, "sw-b2"),
        _decision(cough, "sw-c1"),
        _decision(cough, "sw-c2"),
    ]

    # Revision 1: agreement resolves, disagreement waits for adjudication.
    first = merge_native_review_labels(projections, decisions, policy=POLICY)
    assert first.state_counts()["accepted"] == 2
    assert first.item("synthetic:fever").state is LabelState.UNRESOLVED

    # Revision 2: an explicit adjudication event resolves the conflict.
    event = build_adjudication_event(
        first.item("synthetic:fever"),
        policy=POLICY,
        adjudicator_role="clinical_linguist",
        outcome="accept_spans",
        spans=(ReviewSpan(12, 21, "CONDITION"),),
        reason_code="boundary",
    )
    second = merge_native_review_labels(
        projections, decisions, policy=POLICY, events=[event], previous=first
    )
    assert second.parent_revision_id == first.revision_id
    assert second.state_counts()["accepted"] == 3
    stored = tmp_path / "revision.json"
    stored.write_text(second.to_json(), encoding="utf-8")
    reloaded = AdjudicationRevision.from_dict(
        json.loads(stored.read_text(encoding="utf-8"))
    )
    assert reloaded == second

    # Revision 3: a regenerated projection, a duplicate resubmission and a
    # text edit each invalidate an accepted item.
    regenerated = _projection("synthetic:fever", FEVER, 12, 21, "CONDITION")
    edited = _projection(
        "synthetic:cough", "Mtoto ana kikohozi kikavu usiku.", 11, 25, "CONDITION"
    )
    duplicate = _decision(medicine, "sw-b1", submission="2")
    third = merge_native_review_labels(
        [regenerated, medicine, edited],
        [*decisions, duplicate],
        policy=POLICY,
        events=[event],
        previous=reloaded,
    )
    reasons = {item.source_ref: item.reason_codes for item in third.items}
    assert third.state_counts()["invalidated"] == 3
    assert reasons == {
        "synthetic:cough": ("text_changed",),
        "synthetic:fever": ("stale_projection",),
        "synthetic:medicine": ("duplicate_decision",),
    }
    assert third.item("synthetic:fever").ignored_event_ids == (event.event_id,)
    assert third.to_audit_dict()["prior_acceptance_revoked_count"] == 3
    assert third.export_labels()["labels"] == []

    # Revision 4: fresh reviews and an explicit supersession restore labels.
    restored = [
        _decision(regenerated, "sw-a3"),
        _decision(regenerated, "sw-a4"),
        _decision(
            medicine,
            "sw-b1",
            submission="2",
            supersedes_decision_id=decisions[2]["decision_id"],
        ),
        _decision(edited, "sw-c3"),
        _decision(edited, "sw-c4"),
    ]
    fourth = merge_native_review_labels(
        [regenerated, medicine, edited],
        [*decisions, *restored],
        policy=POLICY,
        events=[event],
        previous=third,
    )
    assert fourth.parent_revision_id == third.revision_id
    assert fourth.state_counts()["accepted"] == 3
    assert not any(item.prior_acceptance_revoked for item in fourth.items)
    medicine_item = fourth.item("synthetic:medicine")
    assert medicine_item.superseded_decision_ids == (decisions[2]["decision_id"],)

    # Exports keep lineage while audits and interchange stay text-free.
    labels = fourth.export_labels()
    audit = json.dumps(fourth.to_audit_dict())
    envelope = export_adjudicated_annotations(
        fourth,
        [regenerated, medicine, edited],
        hash_secret=b"synthetic-annotation-key-0123456789",
    )
    table = export_annotation_tsv(envelope).text
    assert import_annotation_tsv(table).envelope_digest == envelope.envelope_digest
    assert len(envelope.records) == 3
    assert all(entry["lineage"]["decisions"] for entry in labels["labels"])
    for artifact in (json.dumps(labels), audit, table):
        for fragment in ("homa", "paracetamol", "kikohozi", "sw-a1", "sw-b1"):
            assert fragment not in artifact
    for source_ref in ("synthetic:fever", "synthetic:medicine", "synthetic:cough"):
        assert source_ref not in audit
        assert source_ref not in table

    report = reviewer_disagreement_report(
        fourth.to_reviewer_decisions(), minimum_cell_size=2
    ).to_dict()
    assert report["agreement"]["denominator"] == 3
    assert "synthetic" not in json.dumps(report)
