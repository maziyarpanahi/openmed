from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from jsonschema import Draft202012Validator

from openmed.clinical.journey_contracts import (
    ClinicalArtifact,
    ClinicalFact,
    EvidenceLocator,
    canonical_digest,
    derived_opaque_id,
    sha256_digest,
)
from openmed.clinical.review_transitions import transition_review_packet
from openmed.structured.facts import (
    FactReconciler,
    FactReconciliationInput,
    FactReconciliationPolicy,
    build_human_resolution,
    load_fact_reconciliation_schema,
    persist_fact_reconciliation,
)
from openmed.structured.store import LocalJourneyStore, StorePoint, StoreState

T0 = "2026-09-21T10:00:00Z"
T1 = "2026-09-21T11:00:00Z"
SUBJECT_ID = "subject_aaaaaaaaaaaaaaaa"
ENCOUNTER_ID = "encounter_aaaaaaaaaaaaaaaa"
RECONCILIATION_ID = "canonical_aaaaaaaaaaaaaaaa"
CANARY = "SYNTHETIC-PRIVATE-FACT-CANARY"
FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "fixtures"
    / "structured"
    / "fact_conflict_matrix.json"
)


def _fact(
    suffix: str,
    *,
    subject_id: str = SUBJECT_ID,
    encounter_id: str | None = ENCOUNTER_ID,
    value: object = None,
    status: str = "final",
    unit: str | None = "mg/dL",
    effective_time: dict[str, object] | None = None,
    confidence: float = 0.9,
    parent_fact_ids: tuple[str, ...] = (),
    attributes: dict[str, object] | None = None,
) -> ClinicalFact:
    fact_id = derived_opaque_id("fact", suffix)
    evidence_id = derived_opaque_id("evidence", suffix)
    actual_value = {"code": "synthetic-a"} if value is None else value
    return ClinicalFact(
        fact_id=fact_id,
        subject_id=subject_id,
        encounter_id=encounter_id,
        fact_type="laboratory",
        value=actual_value,
        status=status,
        evidence_ids=(evidence_id,),
        derivation_hash=canonical_digest({"fixture": suffix}),
        parent_fact_ids=parent_fact_ids,
        effective_time=effective_time or {"instant": "2026-01-01T00:00:00Z"},
        unit=unit,
        confidence=confidence,
        attributes=attributes or {"assertion": "affirmed", "certainty": "certain"},
    )


def _input(
    fact: ClinicalFact,
    *,
    source: str,
    amendment_of: str | None = None,
    reconciliation_id: str = RECONCILIATION_ID,
) -> FactReconciliationInput:
    return FactReconciliationInput(
        fact=fact,
        reconciliation_id=reconciliation_id,
        source=source,
        amendment_of=amendment_of,
    )


def _conflict_types(result) -> set[str]:
    assert result.value is not None
    return {item.conflict_type for item in result.value.conflicts}


def _matrix_cases() -> list[dict[str, object]]:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return payload["cases"]


def test_exact_duplicates_preserve_every_evidence_reference() -> None:
    first = _fact("exact-a")
    second = _fact("exact-b")
    result = FactReconciler().reconcile(
        SUBJECT_ID,
        (_input(first, source="source.a"), _input(second, source="source.a")),
        occurred_at=T0,
    )

    assert result.ok and result.value is not None
    group = result.value.equivalence_groups[0]
    assert group.kind == "exact_duplicate"
    assert group.fact_ids == tuple(sorted((first.fact_id, second.fact_id)))
    assert group.evidence_ids == tuple(
        sorted((*first.evidence_ids, *second.evidence_ids))
    )
    assert result.value.current_fact_ids == (group.representative_fact_id,)


def test_policy_equivalent_status_and_units_deduplicate() -> None:
    first = _fact("equivalent-a", status="final", unit="mg/dL")
    second = _fact("equivalent-b", status="amended", unit="mg / dl")
    policy = FactReconciliationPolicy(
        status_equivalence={"final": "complete", "amended": "complete"},
        unit_equivalence={"mg/dl": "mg/dl", "mg / dl": "mg/dl"},
    )
    result = FactReconciler(policy).reconcile(
        SUBJECT_ID,
        (_input(first, source="source.a"), _input(second, source="source.a")),
        occurred_at=T0,
    )

    assert result.ok and result.value is not None
    assert result.value.equivalence_groups[0].kind == "policy_equivalent"
    assert not result.value.conflicts


@pytest.mark.parametrize("case", _matrix_cases(), ids=lambda item: item["name"])
def test_synthetic_conflict_matrix_abstains_when_policy_is_insufficient(
    case: dict[str, object],
) -> None:
    first = _fact("matrix-base")
    changes = dict(case["changes"])
    if "value" in changes:
        changes["value"] = changes["value"]
    second = replace(_fact("matrix-other"), **changes)
    result = FactReconciler().reconcile(
        SUBJECT_ID,
        (_input(first, source="source.a"), _input(second, source="source.b")),
        occurred_at=T0,
    )

    assert result.state is StoreState.CONFLICT
    assert _conflict_types(result) == set(case["expected"])
    assert result.value is not None
    assert not result.value.current_fact_ids
    assert all(item.action == "defer" for item in result.value.resolutions)
    assert len(result.value.review_packets) == len(result.value.conflicts)


@given(priority=st.integers(min_value=1, max_value=1000))
def test_unique_source_priority_property_resolves_deterministically(
    priority: int,
) -> None:
    first = _fact("property-a", value={"code": "synthetic-a"})
    second = _fact("property-b", value={"code": "synthetic-b"})
    policy = FactReconciliationPolicy(
        source_priority={"source.a": priority + 1, "source.b": priority},
        auto_resolve_conflicts=frozenset({"value", "source"}),
    )
    inputs = (_input(first, source="source.a"), _input(second, source="source.b"))

    forward = FactReconciler(policy).reconcile(SUBJECT_ID, inputs, occurred_at=T0)
    reverse = FactReconciler(policy).reconcile(
        SUBJECT_ID, tuple(reversed(inputs)), occurred_at=T0
    )

    assert forward.ok and reverse.ok
    assert forward.value is not None and reverse.value is not None
    assert forward.value.to_dict() == reverse.value.to_dict()
    assert forward.value.current_fact_ids == (first.fact_id,)


def test_identity_conflict_cannot_be_auto_resolved() -> None:
    first = _fact("identity-a")
    second = _fact("identity-b", encounter_id="encounter_bbbbbbbbbbbbbbbb")
    policy = FactReconciliationPolicy(
        source_priority={"source.a": 20, "source.b": 10},
        auto_resolve_conflicts=frozenset({"identity", "source"}),
    )

    result = FactReconciler(policy).reconcile(
        SUBJECT_ID,
        (_input(first, source="source.a"), _input(second, source="source.b")),
        occurred_at=T0,
    )

    assert result.state is StoreState.CONFLICT
    assert result.value is not None and not result.value.current_fact_ids


@pytest.mark.parametrize(
    ("field", "first_value", "second_value"),
    (
        ("assertion", "affirmed", "negated"),
        ("certainty", "certain", "uncertain"),
        ("relation_participants", ["fact_a"], ["fact_b"]),
        ("mapping", {"code": "A"}, {"code": "B"}),
    ),
)
def test_semantic_attribute_disagreement_is_a_value_conflict(
    field: str, first_value: object, second_value: object
) -> None:
    first = _fact("semantic-a", attributes={field: first_value})
    second = _fact("semantic-b", attributes={field: second_value})

    result = FactReconciler().reconcile(
        SUBJECT_ID,
        (_input(first, source="source.a"), _input(second, source="source.a")),
        occurred_at=T0,
    )

    assert result.state is StoreState.CONFLICT
    assert _conflict_types(result) == {"value"}


def test_valid_correction_is_new_fact_and_resolution_event() -> None:
    original = _fact("correction-original", value={"code": "synthetic-old"})
    corrected = _fact(
        "correction-new",
        value={"code": "synthetic-new"},
        parent_fact_ids=(original.fact_id,),
    )
    policy = FactReconciliationPolicy(prefer_valid_amendment=True)

    result = FactReconciler(policy).reconcile(
        SUBJECT_ID,
        (
            _input(original, source="source.a"),
            _input(
                corrected,
                source="source.a",
                amendment_of=original.fact_id,
            ),
        ),
        occurred_at=T0,
    )

    assert result.ok and result.value is not None
    assert result.value.current_fact_ids == (corrected.fact_id,)
    assert all(
        event.selected_fact_ids == (corrected.fact_id,)
        for event in result.value.resolutions
    )
    assert all(
        original.fact_id in event.rejected_fact_ids
        for event in result.value.resolutions
    )


def test_competing_corrections_emit_amendment_conflict() -> None:
    original = _fact("amendment-original")
    first = _fact(
        "amendment-first",
        value={"code": "synthetic-first"},
        parent_fact_ids=(original.fact_id,),
    )
    second = _fact(
        "amendment-second",
        value={"code": "synthetic-second"},
        parent_fact_ids=(original.fact_id,),
    )

    result = FactReconciler(
        FactReconciliationPolicy(prefer_valid_amendment=True)
    ).reconcile(
        SUBJECT_ID,
        (
            _input(original, source="source.a"),
            _input(first, source="source.a", amendment_of=original.fact_id),
            _input(second, source="source.a", amendment_of=original.fact_id),
        ),
        occurred_at=T0,
    )

    assert result.state is StoreState.CONFLICT
    assert "amendment" in _conflict_types(result)


def test_policy_change_supersedes_resolution_and_preserves_point_in_time_state(
    tmp_path: Path,
) -> None:
    first = _fact("policy-a", value={"code": "synthetic-a"})
    second = _fact("policy-b", value={"code": "synthetic-b"})
    inputs = (_input(first, source="source.a"), _input(second, source="source.b"))
    policy_one = FactReconciliationPolicy(
        version="1.0.0",
        source_priority={"source.a": 20, "source.b": 10},
        auto_resolve_conflicts=frozenset({"value", "source"}),
    )
    first_run = FactReconciler(policy_one).reconcile(SUBJECT_ID, inputs, occurred_at=T0)
    assert first_run.ok and first_run.value is not None

    store = _store_with_facts(tmp_path, first, second)
    stored_first = persist_fact_reconciliation(
        first_run.value, store.metadata, committed_at=T0
    )
    assert stored_first.ok and stored_first.revision is not None
    point = StorePoint(stored_first.revision)

    previous = {event.conflict_id: event for event in first_run.value.resolutions}
    policy_two = FactReconciliationPolicy(
        version="2.0.0",
        source_priority={"source.a": 10, "source.b": 20},
        auto_resolve_conflicts=frozenset({"value", "source"}),
    )
    second_run = FactReconciler(policy_two).reconcile(
        SUBJECT_ID,
        inputs,
        occurred_at=T1,
        previous_resolutions=previous,
    )
    assert second_run.ok and second_run.value is not None
    stored_second = persist_fact_reconciliation(
        second_run.value, store.metadata, committed_at=T1
    )
    assert stored_second.ok

    earlier = store.metadata.get_canonical(RECONCILIATION_ID, as_of=point)
    latest = store.metadata.get_canonical(RECONCILIATION_ID)
    history = store.metadata.list_canonical_versions(RECONCILIATION_ID)

    assert earlier.value is not None and earlier.value.record.fact_id == first.fact_id
    assert latest.value is not None and latest.value.record.fact_id == second.fact_id
    assert history.value is not None and len(history.value) == 2
    second_events = {event.conflict_id: event for event in second_run.value.resolutions}
    assert all(
        second_events[key].supersedes_resolution_id == previous[key].resolution_id
        for key in previous
    )
    store.close()


def test_unresolved_plan_persists_review_packets_without_raw_values(
    tmp_path: Path,
) -> None:
    first = _fact("review-a", value={"code": CANARY})
    second = _fact("review-b", value={"code": "synthetic-b"})
    result = FactReconciler().reconcile(
        SUBJECT_ID,
        (_input(first, source="source.a"), _input(second, source="source.b")),
        occurred_at=T0,
    )
    assert result.value is not None
    store = _store_with_facts(tmp_path, first, second)

    persisted = persist_fact_reconciliation(
        result.value, store.metadata, committed_at=T0
    )

    assert persisted.state is StoreState.CONFLICT
    assert persisted.revision is not None
    assert result.value.review_packets
    queued = store.metadata.get_job(result.value.review_packets[0].packet_id)
    assert queued.ok and queued.value is not None
    assert CANARY not in queued.value.to_json()
    assert CANARY not in result.value.to_json()
    store.close()


def test_public_schema_validates_reconciliation_result() -> None:
    first = _fact("schema-a")
    second = _fact("schema-b")
    result = FactReconciler().reconcile(
        SUBJECT_ID,
        (_input(first, source="source.a"), _input(second, source="source.a")),
        occurred_at=T0,
    )
    assert result.value is not None
    schema = load_fact_reconciliation_schema()

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(result.value.to_dict())


def test_human_resolution_requires_completed_review() -> None:
    first = _fact("human-a", value={"code": "synthetic-a"})
    second = _fact("human-b", value={"code": "synthetic-b"})
    result = FactReconciler().reconcile(
        SUBJECT_ID,
        (_input(first, source="source.a"), _input(second, source="source.b")),
        occurred_at=T0,
    )
    assert result.value is not None
    packet = result.value.review_packets[0]
    deferred = next(
        item
        for item in result.value.resolutions
        if item.conflict_id == packet.conflict_id
    )

    unresolved = build_human_resolution(
        packet,
        selected_fact_ids=(first.fact_id,),
        rejected_fact_ids=(second.fact_id,),
        occurred_at=T1,
        policy_id="openmed.review.human",
        policy_version="1.0.0",
        rationale_code="review_confirmed",
        supersedes_resolution_id=deferred.resolution_id,
    )

    assert unresolved.state is StoreState.CONFLICT
    assert unresolved.code == "review_not_complete"


def test_completed_review_builds_append_only_human_resolution() -> None:
    first = _fact("human-complete-a", value={"code": "synthetic-a"})
    second = _fact("human-complete-b", value={"code": "synthetic-b"})
    result = FactReconciler().reconcile(
        SUBJECT_ID,
        (_input(first, source="source.a"), _input(second, source="source.b")),
        occurred_at=T0,
    )
    assert result.value is not None
    packet = result.value.review_packets[0]
    deferred = next(
        item
        for item in result.value.resolutions
        if item.conflict_id == packet.conflict_id
    )
    started = transition_review_packet(
        packet,
        to_state="in_review",
        occurred_at="2026-09-21T10:30:00Z",
        policy_id="openmed.review.transitions",
        policy_version="1.0.0",
        provenance_fingerprint=canonical_digest({"state": "in_review"}),
        reason_code="review_started",
    )
    assert started.value is not None
    approved = transition_review_packet(
        started.value,
        to_state="approved",
        occurred_at=T1,
        policy_id="openmed.review.transitions",
        policy_version="1.0.0",
        provenance_fingerprint=canonical_digest({"state": "approved"}),
        reason_code="review_approved",
    )
    assert approved.value is not None

    resolution = build_human_resolution(
        approved.value,
        selected_fact_ids=(first.fact_id,),
        rejected_fact_ids=(second.fact_id,),
        occurred_at="2026-09-21T11:30:00Z",
        policy_id="openmed.review.human",
        policy_version="1.0.0",
        rationale_code="review_confirmed",
        supersedes_resolution_id=deferred.resolution_id,
    )

    assert resolution.ok and resolution.value is not None
    assert resolution.value.actor_type == "human"
    assert resolution.value.selected_fact_ids == (first.fact_id,)
    assert resolution.value.supersedes_resolution_id == deferred.resolution_id


def _store_with_facts(tmp_path: Path, *facts: ClinicalFact) -> LocalJourneyStore:
    store = LocalJourneyStore(
        tmp_path / derived_opaque_id("store", *(f.fact_id for f in facts))
    )
    content = b"synthetic reconciliation artifact"
    artifact = ClinicalArtifact(
        artifact_id="artifact_aaaaaaaaaaaaaaaa",
        artifact_type="synthetic",
        media_type="application/json",
        content_hash=sha256_digest(content),
        byte_size=len(content),
        source_id="source_aaaaaaaaaaaaaaaa",
        recorded_at=T0,
        subject_id=SUBJECT_ID,
    )
    evidence = tuple(
        EvidenceLocator(
            locator_id=fact.evidence_ids[0],
            artifact_id=artifact.artifact_id,
            location_type="json_pointer",
            location={"pointer": f"/{index}"},
        )
        for index, fact in enumerate(facts)
    )
    ingested = store.ingest_graph(
        artifact,
        content,
        evidence=evidence,
        facts=facts,
        committed_at="2026-09-21T09:00:00Z",
    )
    assert ingested.ok
    return store
