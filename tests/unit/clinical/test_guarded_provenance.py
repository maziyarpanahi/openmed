"""Focused tests for guarded clinical-output provenance records."""

from __future__ import annotations

import json
import socket
from dataclasses import FrozenInstanceError

import pytest

from openmed.clinical import (
    GUARDED_PROVENANCE_SCHEMA_VERSION,
    EvidenceReference,
    GuardedProvenanceError,
    ProvenanceEvidenceReference,
    build_guarded_provenance_manifest,
    check_guarded_provenance,
    fingerprint_input,
    fingerprint_policy,
    fingerprint_provenance_policy,
    load_guarded_provenance_manifest,
    write_guarded_provenance_manifest,
)
from openmed.clinical import (
    ProvenanceReviewState as ReviewState,
)
from openmed.clinical.evidence_packet import (
    EvidenceReference as PacketEvidenceReference,
)
from openmed.clinical.evidence_packet import (
    fingerprint_policy as packet_fingerprint_policy,
)
from openmed.clinical.guarded_provenance import (
    EvidenceReference as GuardedEvidenceReference,
)
from openmed.clinical.guarded_provenance import (
    GuardedProvenanceRecord,
    IntegritySummary,
    ModelProvenance,
    ReviewTransition,
)
from openmed.clinical.guarded_provenance import (
    fingerprint_policy as guarded_fingerprint_policy,
)

SYNTHETIC_INPUT = "SYNTHETIC_CLINICAL_INPUT"
SYNTHETIC_OUTPUT = "SYNTHETIC_GENERATED_CLINICAL_OUTPUT"
SYNTHETIC_EVIDENCE = "SYNTHETIC_EVIDENCE_FRAGMENT"


def test_public_exports_preserve_packet_and_provenance_types() -> None:
    assert EvidenceReference is PacketEvidenceReference
    assert ProvenanceEvidenceReference is GuardedEvidenceReference
    assert fingerprint_policy is packet_fingerprint_policy
    assert fingerprint_provenance_policy is guarded_fingerprint_policy


def _evidence(text: str = SYNTHETIC_EVIDENCE) -> dict[str, object]:
    return {
        "id": "synthetic-evidence-1",
        "start": 4,
        "end": 18,
        "text": text,
    }


def _manifest(*, evidence: object = None, output: object = None):
    return build_guarded_provenance_manifest(
        output=output
        if output is not None
        else {
            "output_kind": "summary",
            "input_text": SYNTHETIC_INPUT,
            "summary_text": SYNTHETIC_OUTPUT,
            "evidence_ids": ["synthetic-evidence-1"],
        },
        evidence=[_evidence()] if evidence is None else evidence,
        model={"model_id": "OpenMed/synthetic-summary-model", "revision": "v1"},
        policy={"profile": "synthetic-local-policy", "revision": 1},
    )


def test_manifest_is_deterministic_and_value_free() -> None:
    first = _manifest()
    second = build_guarded_provenance_manifest(
        output={
            "summary_text": SYNTHETIC_OUTPUT,
            "input_text": SYNTHETIC_INPUT,
            "evidence_ids": ["synthetic-evidence-1"],
            "output_kind": "summary",
        },
        evidence=[_evidence()],
        model={"revision": "v1", "model_id": "OpenMed/synthetic-summary-model"},
        policy={"revision": 1, "profile": "synthetic-local-policy"},
    )

    assert first.to_json() == second.to_json()
    payload = first.to_json()
    assert json.loads(payload)["schema_version"] == GUARDED_PROVENANCE_SCHEMA_VERSION
    assert SYNTHETIC_INPUT not in payload
    assert SYNTHETIC_OUTPUT not in payload
    assert SYNTHETIC_EVIDENCE not in payload
    assert first.records[0].output_hash is not None
    assert first.records[0].input_hash == fingerprint_input(SYNTHETIC_INPUT)
    assert first.records[0].output_kind == "summary"
    assert first.records[0].review_status == ReviewState.QUEUED
    assert first.records[0].requires_human_review
    assert not first.release_ready


def test_record_binds_model_policy_evidence_and_review_transitions() -> None:
    manifest = build_guarded_provenance_manifest(
        output={
            "output_kind": "nli",
            "input_text": SYNTHETIC_INPUT,
            "text": SYNTHETIC_OUTPUT,
            "evidence_ids": ["synthetic-evidence-1"],
        },
        evidence=[_evidence()],
        model={"model_id": "OpenMed/synthetic-nli-model", "revision": "2026.09"},
        policy={"name": "synthetic-review-policy", "version": 1},
        review_transitions=[
            {
                "from_state": "queued",
                "to_state": "in_review",
                "reviewer_id": "synthetic-reviewer",
            },
            {
                "from_state": "in_review",
                "to_state": "approved",
                "reviewer_id": "synthetic-reviewer",
            },
        ],
    )
    record = manifest.records[0]

    assert record.review_status == ReviewState.APPROVED
    assert len(record.review_transitions) == 2
    assert record.review_transitions[0].reviewer_fingerprint is not None
    assert record.model.model_id.startswith("sha256:")
    assert record.model.revision.startswith("sha256:")
    assert record.model.model_fingerprint is not None
    assert record.policy_fingerprint is not None
    assert record.integrity.ok
    assert check_guarded_provenance(manifest).ok
    assert check_guarded_provenance(manifest).release_ready


def test_missing_evidence_is_reported_without_echoing_reference() -> None:
    sentinel = "synthetic-missing-evidence-reference"
    manifest = build_guarded_provenance_manifest(
        output={
            "input_text": SYNTHETIC_INPUT,
            "summary_text": SYNTHETIC_OUTPUT,
            "evidence_ids": [sentinel],
        },
        evidence=[],
        model="OpenMed/synthetic-summary-model",
        policy={"name": "synthetic-policy"},
    )

    report = check_guarded_provenance(manifest)
    assert not report.ok
    assert report.missing_evidence_count == 1
    assert "missing_evidence" in report.reason_codes
    assert sentinel not in report.to_json()
    assert report.requires_human_review


def test_value_free_surfaces_cover_markdown_repr_and_exceptions() -> None:
    sentinel = "SYNTHETIC_SECRET_CLINICAL_VALUE"
    manifest = build_guarded_provenance_manifest(
        output={"input_text": sentinel, "text": sentinel},
        evidence=[],
        model="synthetic-model",
        policy={"profile": "synthetic-policy"},
    )

    surfaces = (
        manifest.to_json(),
        manifest.to_markdown(),
        repr(manifest),
        repr(manifest.records[0]),
        check_guarded_provenance(manifest).to_json(),
    )
    assert all(sentinel not in surface for surface in surfaces)

    with pytest.raises(GuardedProvenanceError) as error:
        build_guarded_provenance_manifest(
            output={"input_text": sentinel, "text": sentinel},
            evidence=[{"id": "synthetic-evidence", "start": 4, "end": 4}],
            model="synthetic-model",
            policy={"profile": "synthetic-policy"},
        )
    assert sentinel not in str(error.value)


def test_changed_input_and_evidence_fail_closed() -> None:
    manifest = _manifest()
    report = check_guarded_provenance(
        manifest,
        current_input="SYNTHETIC_CHANGED_INPUT",
        current_evidence=[_evidence("SYNTHETIC_CHANGED_EVIDENCE")],
    )

    assert not report.ok
    assert report.input_changed
    assert report.changed_evidence_count == 1
    assert set(report.reason_codes) == {"changed_evidence", "changed_input"}
    assert "SYNTHETIC_CHANGED_INPUT" not in report.to_json()
    assert "SYNTHETIC_CHANGED_EVIDENCE" not in report.to_json()


def test_review_state_machine_rejects_disconnected_or_unreviewed_approval() -> None:
    with pytest.raises(GuardedProvenanceError, match="review transition"):
        build_guarded_provenance_manifest(
            output={"input_text": SYNTHETIC_INPUT, "text": SYNTHETIC_OUTPUT},
            evidence=[_evidence()],
            model="OpenMed/synthetic-model",
            policy={"name": "synthetic-policy"},
            review_transitions=[{"from_state": "queued", "to_state": "approved"}],
        )

    with pytest.raises(GuardedProvenanceError, match="approved output"):
        build_guarded_provenance_manifest(
            output={"input_text": SYNTHETIC_INPUT, "text": SYNTHETIC_OUTPUT},
            evidence=[_evidence()],
            model="OpenMed/synthetic-model",
            policy={"name": "synthetic-policy"},
            review_status="approved",
        )


def test_manifest_round_trip_and_tamper_detection(tmp_path) -> None:
    manifest = _manifest()
    path = write_guarded_provenance_manifest(tmp_path / "provenance.json", manifest)
    loaded = load_guarded_provenance_manifest(path)

    assert loaded.to_json() == manifest.to_json()
    assert loaded.verify_hash()
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["records"][0]["review_status"] = "rejected"
    report = check_guarded_provenance(tampered)
    assert not report.ok
    assert "invalid_manifest_hash" in report.reason_codes


def test_records_and_reports_are_immutable_and_offline(monkeypatch) -> None:
    def fail_network(*args, **kwargs):
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    manifest = _manifest()
    manifest.to_json()
    manifest.to_markdown()
    check_guarded_provenance(manifest).to_json()

    with pytest.raises(FrozenInstanceError):
        manifest.records = ()  # type: ignore[misc]


def test_caller_metadata_cannot_leak_through_safe_looking_tokens():
    marker = "synthetic_person_name"
    manifest = build_guarded_provenance_manifest(
        output={"input_text": SYNTHETIC_INPUT, "text": SYNTHETIC_OUTPUT},
        output_kind=marker,
        evidence=[dict(_evidence(), kind=marker)],
        model={"model_id": marker, "revision": marker, "tokenizer_id": marker},
        policy={"profile": "synthetic"},
        review_transitions=[
            {"from_state": "queued", "to_state": "in_review", "reason_code": marker}
        ],
    )
    assert marker not in manifest.to_json() + manifest.to_markdown() + repr(manifest)


def test_supplied_integrity_cannot_approve_missing_inputs_and_evidence():
    record = GuardedProvenanceRecord(
        review_status="approved",
        review_transitions=(
            ReviewTransition(0, "queued", "in_review"),
            ReviewTransition(1, "in_review", "approved"),
        ),
        integrity=IntegritySummary(
            model_present=True,
            policy_present=True,
            input_present=True,
            output_present=True,
        ),
    )
    manifest = build_guarded_provenance_manifest(records=[record])
    assert record.requires_human_review
    assert not manifest.release_ready
    assert not check_guarded_provenance(manifest).release_ready


def test_evidence_offset_drift_is_detected_even_when_content_hash_matches():
    manifest = _manifest()
    report = check_guarded_provenance(
        manifest, current_evidence=[dict(_evidence(), start=5, end=19)]
    )
    assert not report.ok
    assert report.changed_evidence_count == 1


def test_ambiguous_duplicate_evidence_is_rejected():
    with pytest.raises(GuardedProvenanceError, match="duplicate evidence"):
        _manifest(evidence=[_evidence(), _evidence("synthetic changed evidence")])


def test_nested_mutated_model_is_revalidated_before_serialization():
    model = ModelProvenance(model_id="synthetic-model")
    object.__setattr__(model, "model_id", "synthetic_person_name")
    record = GuardedProvenanceRecord(model=model)
    assert "synthetic_person_name" not in json.dumps(record.to_dict())


def test_rejected_iterators_and_paths_retain_no_exception_context(tmp_path):
    class InvalidEvidence:
        def __iter__(self):
            raise ValueError("synthetic_private_marker")

    for operation in (
        lambda: _manifest(evidence=InvalidEvidence()),
        lambda: load_guarded_provenance_manifest(tmp_path / "synthetic_private_marker"),
    ):
        with pytest.raises(GuardedProvenanceError) as caught:
            operation()
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None
        assert "synthetic_private_marker" not in str(caught.value)
