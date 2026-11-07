"""Tests for point-in-time longitudinal journey materialization."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator

from openmed.clinical.journey import (
    JourneyQuery,
    JourneySnapshot,
    load_journey_view_schema,
    query_journey,
)
from openmed.clinical.journey_contracts import (
    ClinicalArtifact,
    ClinicalFact,
    EvidenceLocator,
    canonical_digest,
    derived_opaque_id,
    sha256_digest,
)
from openmed.structured.facts import (
    FactReconciler,
    FactReconciliationInput,
    FactReconciliationPolicy,
    persist_fact_reconciliation,
)
from openmed.structured.store import (
    DenyStorageOperations,
    LocalJourneyStore,
    StoreState,
)

SUBJECT_ID = "subject_aaaaaaaaaaaaaaaa"
ENCOUNTER_A = "encounter_aaaaaaaaaaaaaaaa"
ENCOUNTER_B = "encounter_bbbbbbbbbbbbbbbb"
T0 = "2026-09-21T08:00:00Z"
T1 = "2026-09-21T09:00:00Z"
T2 = "2026-09-21T10:00:00Z"
CANARY = "SYNTHETIC-JOURNEY-CANARY"


def test_equal_timestamps_order_by_fact_id_without_inventing_precision(
    tmp_path: Path,
) -> None:
    store = LocalJourneyStore(tmp_path / "equal-time")
    second = _fact("equal-b", effective_time={"precision": "month", "start": "2026-06"})
    first = _fact("equal-a", effective_time={"precision": "month", "start": "2026-06"})
    _ingest(store, second, source_suffix="source-b", committed_at=T0)
    _ingest(store, first, source_suffix="source-a", committed_at=T1)

    result = query_journey(store.metadata, JourneyQuery(subject_id=SUBJECT_ID))

    assert result.ok and result.value is not None
    assert [item.fact.fact_id for item in result.value.events] == sorted(
        (first.fact_id, second.fact_id)
    )
    assert all(
        item.fact.effective_time == {"precision": "month", "start": "2026-06"}
        for item in result.value.events
    )
    assert not result.value.graph.edges
    store.close()


def test_point_in_time_replays_before_correction_and_preserves_history(
    tmp_path: Path,
) -> None:
    store = LocalJourneyStore(tmp_path / "correction")
    original = _fact(
        "correction-original",
        value={"code": "synthetic-old"},
        status="final",
    )
    _ingest(store, original, source_suffix="source-a", committed_at=T0)
    reconciliation_id = derived_opaque_id("canonical", SUBJECT_ID, "laboratory")
    first_plan = FactReconciler().reconcile(
        SUBJECT_ID,
        (_input(original, reconciliation_id=reconciliation_id),),
        occurred_at=T1,
    )
    assert first_plan.ok and first_plan.value is not None
    first_write = persist_fact_reconciliation(
        first_plan.value,
        store.metadata,
        committed_at=T1,
    )
    assert first_write.ok and first_write.revision is not None
    before_correction = JourneySnapshot.at(SUBJECT_ID, first_write.revision)

    corrected = _fact(
        "correction-new",
        value={"code": "synthetic-new"},
        status="corrected",
        parent_fact_ids=(original.fact_id,),
    )
    _ingest(store, corrected, source_suffix="source-b", committed_at=T2)
    second_plan = FactReconciler(
        FactReconciliationPolicy(prefer_valid_amendment=True)
    ).reconcile(
        SUBJECT_ID,
        (
            _input(original, reconciliation_id=reconciliation_id),
            _input(
                corrected,
                reconciliation_id=reconciliation_id,
                amendment_of=original.fact_id,
            ),
        ),
        occurred_at="2026-09-21T11:00:00Z",
    )
    assert second_plan.ok and second_plan.value is not None
    assert persist_fact_reconciliation(
        second_plan.value,
        store.metadata,
        committed_at="2026-09-21T11:00:00Z",
    ).ok

    earlier = query_journey(
        store.metadata,
        JourneyQuery(subject_id=SUBJECT_ID, snapshot=before_correction),
    )
    current = query_journey(store.metadata, JourneyQuery(subject_id=SUBJECT_ID))

    assert earlier.ok and earlier.value is not None
    assert [
        (item.fact.fact_id, item.journey_state) for item in earlier.value.events
    ] == [(original.fact_id, "current")]
    assert current.ok and current.value is not None
    by_id = {item.fact.fact_id: item for item in current.value.events}
    assert by_id[original.fact_id].journey_state == "historical"
    assert by_id[original.fact_id].correction_state == "superseded"
    assert by_id[corrected.fact_id].journey_state == "current"
    assert by_id[corrected.fact_id].correction_state == "amends"
    assert by_id[corrected.fact_id].resolutions
    assert any(edge.relation_type == "corrects" for edge in current.value.graph.edges)
    store.close()


def test_unresolved_conflict_is_reviewable_and_never_selects_truth(
    tmp_path: Path,
) -> None:
    store = LocalJourneyStore(tmp_path / "review")
    first = _fact("review-a", value={"code": "synthetic-a"})
    second = _fact("review-b", value={"code": "synthetic-b"})
    _ingest(store, first, source_suffix="source-a", committed_at=T0)
    _ingest(store, second, source_suffix="source-b", committed_at=T1)
    reconciliation_id = derived_opaque_id("canonical", SUBJECT_ID, "review")
    plan = FactReconciler().reconcile(
        SUBJECT_ID,
        (
            _input(first, reconciliation_id=reconciliation_id, source="source.a"),
            _input(second, reconciliation_id=reconciliation_id, source="source.b"),
        ),
        occurred_at=T2,
    )
    assert plan.state is StoreState.CONFLICT and plan.value is not None
    persisted = persist_fact_reconciliation(plan.value, store.metadata, committed_at=T2)
    assert persisted.state is StoreState.CONFLICT

    result = query_journey(
        store.metadata,
        JourneyQuery(subject_id=SUBJECT_ID, review_states=("queued",)),
    )

    assert result.ok and result.value is not None
    assert len(result.value.events) == 2
    assert {item.journey_state for item in result.value.events} == {"conflicted"}
    assert {item.review_states for item in result.value.events} == {("queued",)}
    assert all(not item.canonical_records for item in result.value.events)
    store.close()


def test_filters_and_snapshot_bound_cursor_are_deterministic(tmp_path: Path) -> None:
    store = LocalJourneyStore(tmp_path / "filters")
    condition = _fact(
        "filter-condition",
        fact_type="condition",
        encounter_id=ENCOUNTER_A,
        effective_time={"precision": "day", "start": "2026-01-01"},
        status="active",
    )
    medication = _fact(
        "filter-medication",
        fact_type="medication",
        encounter_id=ENCOUNTER_B,
        effective_time={"precision": "day", "start": "2026-02-01"},
        status="active",
    )
    _ingest(store, condition, source_suffix="source-a", committed_at=T0)
    _ingest(store, medication, source_suffix="source-b", committed_at=T1)

    first_page = query_journey(
        store.metadata,
        JourneyQuery(subject_id=SUBJECT_ID, limit=1),
    )
    assert first_page.ok and first_page.value is not None
    assert first_page.value.next_cursor is not None
    second_page = query_journey(
        store.metadata,
        JourneyQuery(
            subject_id=SUBJECT_ID,
            snapshot=first_page.value.snapshot,
            limit=1,
            cursor=first_page.value.next_cursor,
        ),
    )
    assert second_page.ok and second_page.value is not None
    assert second_page.value.events[0].fact.fact_id == medication.fact_id
    assert second_page.value.events[0].position == 1
    assert second_page.value.graph.nodes[0].position == 1

    source_b = derived_opaque_id("source", "source-b")
    filtered = query_journey(
        store.metadata,
        JourneyQuery(
            subject_id=SUBJECT_ID,
            snapshot=first_page.value.snapshot,
            encounter_ids=(ENCOUNTER_B,),
            start_time="2026-02",
            end_time="2026-02-28",
            event_types=("medication",),
            statuses=("active",),
            source_ids=(source_b,),
        ),
    )
    assert filtered.ok and filtered.value is not None
    assert [item.fact.fact_id for item in filtered.value.events] == [medication.fact_id]

    mismatch = query_journey(
        store.metadata,
        JourneyQuery(
            subject_id=SUBJECT_ID,
            snapshot=first_page.value.snapshot,
            limit=2,
            cursor=first_page.value.next_cursor,
        ),
    )
    assert mismatch.state is StoreState.CONFLICT
    assert mismatch.code == "journey_cursor_mismatch"
    store.close()


def test_time_filter_abstains_when_fact_time_is_unknown(tmp_path: Path) -> None:
    store = LocalJourneyStore(tmp_path / "unknown-time")
    fact = _fact("unknown-time", effective_time={})
    _ingest(store, fact, source_suffix="source-a", committed_at=T0)

    result = query_journey(
        store.metadata,
        JourneyQuery(subject_id=SUBJECT_ID, start_time="2026-01-01"),
    )

    assert result.state is StoreState.PARTIAL
    assert result.code == "journey_time_unknown"
    store.close()


def test_unsupported_fact_type_returns_typed_outcome(tmp_path: Path) -> None:
    store = LocalJourneyStore(tmp_path / "unsupported-type")
    fact = _fact("unsupported-type", fact_type="relation")
    _ingest(store, fact, source_suffix="source-a", committed_at=T0)

    result = query_journey(store.metadata, JourneyQuery(subject_id=SUBJECT_ID))

    assert result.state is StoreState.UNSUPPORTED
    assert result.code == "journey_event_type_unsupported"
    store.close()


def test_read_policy_denial_stays_typed(tmp_path: Path) -> None:
    store = LocalJourneyStore(tmp_path / "denied")
    fact = _fact("denied")
    _ingest(store, fact, source_suffix="source-a", committed_at=T0)
    store.metadata.policy = DenyStorageOperations(frozenset({"read"}))

    result = query_journey(store.metadata, JourneyQuery(subject_id=SUBJECT_ID))

    assert result.state is StoreState.DENIED
    assert result.code == "policy_denied"
    store.close()


def test_event_drills_to_artifact_locator_and_derivation_versions(
    tmp_path: Path,
) -> None:
    store = LocalJourneyStore(tmp_path / "provenance")
    fact = _fact("provenance", value={"code": CANARY})
    _ingest(store, fact, source_suffix="source-a", committed_at=T0)
    result = query_journey(store.metadata, JourneyQuery(subject_id=SUBJECT_ID))

    assert result.ok and result.value is not None
    event = result.value.events[0]
    assert event.evidence_paths[0].locator.locator_id == fact.evidence_ids[0]
    assert event.evidence_paths[0].artifact.content_hash.startswith("sha256:")
    assert event.fact.derivation_hash.startswith("sha256:")
    assert CANARY not in json.dumps(result.value.to_safe_dict(), sort_keys=True)
    assert CANARY not in repr(result.value)

    schema = load_journey_view_schema()
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(result.value.to_dict())
    store.close()


def test_public_journey_contracts_reject_malformed_nested_values(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="JourneySnapshot"):
        JourneyQuery(
            subject_id=SUBJECT_ID,
            snapshot=cast(Any, "malformed"),
        )

    store = LocalJourneyStore(tmp_path / "malformed-contracts")
    fact = _fact("malformed-contracts")
    _ingest(store, fact, source_suffix="source-a", committed_at=T0)
    result = query_journey(store.metadata, JourneyQuery(subject_id=SUBJECT_ID))
    assert result.ok and result.value is not None
    page = result.value

    with pytest.raises(TypeError, match="JourneyEvidencePath"):
        replace(page.events[0], evidence_paths=(cast(Any, "malformed"),))
    with pytest.raises(TypeError, match="JourneyEvent"):
        replace(page, events=(cast(Any, "malformed"),))
    store.close()


def test_semantic_axes_intervals_relations_and_mappings_survive_materialization(
    tmp_path: Path,
) -> None:
    store = LocalJourneyStore(tmp_path / "semantic-provenance")
    condition = _fact(
        "semantic-condition",
        fact_type="condition",
        status="active",
        attributes={
            "assertion": "negated",
            "certainty": "uncertain",
            "experiencer": "patient",
            "mapping": {"code": "synthetic-condition", "system": "synthetic"},
        },
    )
    procedure = _fact(
        "semantic-procedure",
        fact_type="procedure",
        status="completed",
        effective_time={
            "precision": "month",
            "start": "2026-03",
            "end": "2026-04",
        },
    )
    medication = _fact(
        "semantic-medication",
        fact_type="medication",
        status="active",
        attributes={
            "assertion": "affirmed",
            "certainty": "certain",
            "experiencer": "patient",
            "relation_participants": [
                {
                    "role": "indication",
                    "target_id": condition.fact_id,
                    "target_type": "fact",
                }
            ],
        },
    )
    for index, fact in enumerate((condition, procedure, medication)):
        _ingest(
            store,
            fact,
            source_suffix=f"semantic-source-{index}",
            committed_at=f"2026-09-21T{8 + index:02d}:00:00Z",
        )

    result = query_journey(store.metadata, JourneyQuery(subject_id=SUBJECT_ID))

    assert result.ok and result.value is not None
    by_fact = {item.fact.fact_id: item for item in result.value.events}
    assert by_fact[condition.fact_id].fact.attributes["assertion"] == "negated"
    assert by_fact[condition.fact_id].fact.attributes["certainty"] == "uncertain"
    assert by_fact[condition.fact_id].fact.attributes["mapping"] == {
        "code": "synthetic-condition",
        "system": "synthetic",
    }
    assert by_fact[procedure.fact_id].fact.effective_time == {
        "end": "2026-04",
        "precision": "month",
        "start": "2026-03",
    }
    assert any(
        edge.relation_type == "relation.indication"
        and edge.source_event_id == by_fact[medication.fact_id].event_id
        and edge.target_event_id == by_fact[condition.fact_id].event_id
        for edge in result.value.graph.edges
    )
    store.close()


def _fact(
    suffix: str,
    *,
    fact_type: str = "laboratory",
    encounter_id: str | None = ENCOUNTER_A,
    value: object | None = None,
    status: str = "final",
    effective_time: dict[str, object] | None = None,
    parent_fact_ids: tuple[str, ...] = (),
    attributes: dict[str, object] | None = None,
) -> ClinicalFact:
    return ClinicalFact(
        fact_id=derived_opaque_id("fact", suffix),
        subject_id=SUBJECT_ID,
        encounter_id=encounter_id,
        fact_type=fact_type,
        value=value or {"code": f"synthetic-{suffix}"},
        status=status,
        evidence_ids=(derived_opaque_id("evidence", suffix),),
        derivation_hash=canonical_digest({"fixture": suffix}),
        parent_fact_ids=parent_fact_ids,
        effective_time=(
            {"precision": "day", "start": "2026-01-01"}
            if effective_time is None
            else effective_time
        ),
        unit="mg/dL" if fact_type == "laboratory" else None,
        confidence=0.9,
        attributes=attributes
        or {
            "assertion": "affirmed",
            "certainty": "certain",
            "experiencer": "patient",
        },
    )


def _input(
    fact: ClinicalFact,
    *,
    reconciliation_id: str,
    source: str = "source.synthetic",
    amendment_of: str | None = None,
) -> FactReconciliationInput:
    return FactReconciliationInput(
        fact=fact,
        reconciliation_id=reconciliation_id,
        source=source,
        amendment_of=amendment_of,
    )


def _ingest(
    store: LocalJourneyStore,
    fact: ClinicalFact,
    *,
    source_suffix: str,
    committed_at: str,
) -> None:
    content = f"synthetic-{source_suffix}-{fact.fact_id}".encode()
    artifact = ClinicalArtifact(
        artifact_id=derived_opaque_id("artifact", fact.fact_id),
        artifact_type="synthetic",
        media_type="application/json",
        content_hash=sha256_digest(content),
        byte_size=len(content),
        source_id=derived_opaque_id("source", source_suffix),
        recorded_at=committed_at,
        subject_id=fact.subject_id,
        encounter_id=fact.encounter_id,
    )
    locator = EvidenceLocator(
        locator_id=fact.evidence_ids[0],
        artifact_id=artifact.artifact_id,
        location_type="json_pointer",
        location={"pointer": "/synthetic"},
    )
    result = store.ingest_graph(
        artifact,
        content,
        evidence=(locator,),
        facts=(fact,),
        committed_at=committed_at,
    )
    assert result.ok
