"""Offline tests for field-level FHIR write provenance."""

from __future__ import annotations

import socket
from dataclasses import replace
from uuid import UUID

import pytest

from openmed.agent.approvals.side_effect_preview import (
    ResourceWrite,
    WorkflowState,
    WriteIntent,
    WriteKind,
)
from openmed.interop.fhir.conditional_writes import (
    ConditionalWriteKind,
    build_conditional_write_plan,
)
from openmed.interop.fhir.write_provenance import (
    ApprovalBinding,
    EvidenceSpan,
    FieldLineage,
    ProvenanceError,
    require_provenance_target,
    require_write_provenance,
    write_provenance_digest,
)

SECRET = b"synthetic-private-key-for-write-provenance"
ACTION_ID = "00000000-0000-4000-8000-000000000001"
HANDLE = "res_" + "a" * 32
SOURCE = "a" * 64
STEP = "b" * 64
POLICY = "c" * 64
RECEIPT = "d" * 64
TARGET = "Observation/synthetic-123"


def _inputs():
    intent = WriteIntent(
        action_id=ACTION_ID,
        writes=(
            ResourceWrite(
                WriteKind.CREATE,
                "Observation",
                HANDLE,
                before=None,
                after={"status": "final", "valueString": "synthetic-private-marker"},
            ),
        ),
        workflow_before=WorkflowState.APPROVED,
        workflow_after=WorkflowState.IN_PROGRESS,
    )
    plan = build_conditional_write_plan(
        ConditionalWriteKind.CREATE,
        "Observation",
        {"identifier": "urn:synthetic|private-marker"},
        operation_id=UUID(ACTION_ID),
        secret=SECRET,
    )
    links = tuple(
        FieldLineage(HANDLE, path, (EvidenceSpan(SOURCE, 5, 12),), (STEP,), POLICY)
        for path in ("status", "valueString")
    )
    approval = ApprovalBinding(
        write_provenance_digest(
            intent, {HANDLE: plan}, links, {HANDLE: TARGET}, secret=SECRET
        ),
        RECEIPT,
    )
    return intent, {HANDLE: plan}, links, {HANDLE: TARGET}, approval


def _require(*, inputs=None, verifier=lambda _: True):
    intent, plans, links, targets, approval = inputs or _inputs()
    return require_write_provenance(
        intent,
        plans,
        links,
        targets,
        approval,
        secret=SECRET,
        verify_approval=verifier,
    )


def test_every_changed_field_has_deterministic_value_free_links() -> None:
    packet = _require()
    assert packet == _require()
    assert tuple(record.field_path for record in packet.records) == (
        "status",
        "valueString",
    )
    assert all(record.approval_receipt_digest == RECEIPT for record in packet.records)
    assert all(
        record.target_digest == packet.records[0].target_digest
        for record in packet.records
    )
    assert all(
        record.evidence == (EvidenceSpan(SOURCE, 5, 12),) for record in packet.records
    )
    assert "synthetic-private-marker" not in repr(packet)
    assert "synthetic-123" not in repr(packet)
    assert "private-marker" not in repr(packet)


def test_rejects_missing_duplicate_and_unplanned_field_lineage() -> None:
    intent, plans, links, targets, approval = _inputs()
    for bad_links in (
        links[:1],
        links + links[:1],
        links + (replace(links[0], field_path="note.text"),),
    ):
        with pytest.raises(ProvenanceError):
            _require(inputs=(intent, plans, bad_links, targets, approval))


def test_rejects_changed_payload_and_unverified_approval() -> None:
    intent, plans, links, targets, approval = _inputs()
    changed = replace(
        intent,
        writes=(
            replace(
                intent.writes[0], after={"status": "final", "valueString": "changed"}
            ),
        ),
    )
    with pytest.raises(ProvenanceError, match="approval_action_mismatch"):
        _require(inputs=(changed, plans, links, targets, approval))
    with pytest.raises(ProvenanceError, match="approval_verification_failed"):
        _require(verifier=lambda _: False)
    with pytest.raises(ProvenanceError, match="approval_verification_failed") as error:
        _require(verifier=lambda _: (_ for _ in ()).throw(ValueError("private-marker")))
    assert "private-marker" not in str(error.value)


def test_approval_binds_plan_lineage_and_target() -> None:
    intent, plans, links, targets, approval = _inputs()
    changed_plan = build_conditional_write_plan(
        ConditionalWriteKind.CREATE,
        "Observation",
        {"identifier": "urn:synthetic|different"},
        operation_id=UUID(ACTION_ID),
        secret=SECRET,
    )
    for changed_plans, changed_links, changed_targets in (
        ({HANDLE: changed_plan}, links, targets),
        (
            plans,
            (replace(links[0], evidence=(EvidenceSpan(SOURCE, 6, 12),)), links[1]),
            targets,
        ),
        (plans, links, {HANDLE: "Observation/different"}),
    ):
        with pytest.raises(ProvenanceError, match="approval_action_mismatch"):
            _require(
                inputs=(
                    intent,
                    changed_plans,
                    changed_links,
                    changed_targets,
                    approval,
                )
            )


def test_rejects_missing_or_mismatched_plan_and_target() -> None:
    intent, plans, links, targets, approval = _inputs()
    for bad_plans, bad_targets in (
        ({}, targets),
        (plans, {}),
        (plans, {HANDLE: "Patient/private-marker"}),
        (plans, {HANDLE: "Observation/private-marker?x=y"}),
    ):
        with pytest.raises(ProvenanceError) as error:
            _require(inputs=(intent, bad_plans, links, bad_targets, approval))
        assert "private-marker" not in str(error.value)

    update_plan = build_conditional_write_plan(
        ConditionalWriteKind.UPDATE,
        "Observation",
        {"identifier": "urn:synthetic|private-marker"},
        operation_id=UUID(ACTION_ID),
        secret=SECRET,
    )
    with pytest.raises(ProvenanceError, match="write_plan_mismatch"):
        _require(inputs=(intent, {HANDLE: update_plan}, links, targets, approval))


def test_provenance_target_must_match_the_reviewed_reference() -> None:
    record = _require().records[0]
    require_provenance_target(record, TARGET, "Observation", secret=SECRET)
    with pytest.raises(ProvenanceError, match="provenance_target_mismatch"):
        require_provenance_target(
            record, "Observation/another-private-marker", "Observation", secret=SECRET
        )
    with pytest.raises(ProvenanceError, match="invalid_provenance_target") as error:
        require_provenance_target(
            record, "Patient/private-marker", "Observation", secret=SECRET
        )
    assert "private-marker" not in str(error.value)


def test_rejects_incomplete_evidence_and_untraceable_values() -> None:
    with pytest.raises(ProvenanceError, match="invalid_evidence_chain"):
        replace(_inputs()[2][0], evidence=())
    with pytest.raises(ProvenanceError, match="invalid_transformation_chain"):
        replace(_inputs()[2][0], step_digests=())
    with pytest.raises(ProvenanceError, match="invalid_evidence_span"):
        EvidenceSpan(SOURCE, 12, 5)
    with pytest.raises(ProvenanceError, match="invalid_policy_digest"):
        replace(_inputs()[2][0], policy_digest="private-marker")


def test_no_network_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_socket(*args, **kwargs):
        raise AssertionError("unexpected network access")

    monkeypatch.setattr(socket, "socket", fail_socket)
    assert len(_require().records) == 2
