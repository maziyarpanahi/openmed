"""Focused synthetic tests for medication reconciliation confidence."""

from __future__ import annotations

from openmed.clinical.medication_reconciliation import (
    MedicationReconciliationPolicy,
    reconcile_medications,
    score_medication_match,
)


def _candidate(
    candidate_id: str,
    *,
    name: str = "Synthetic Medication Alpha",
    dose: str | None = "500 mg",
    route: str | None = "PO",
    date_value: str | None = "2026-01-15",
    status: str | None = "current",
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "normalized_name": name,
        "dose": dose,
        "route": route,
        "event_date": date_value,
        "temporal_label": status,
        "source_document_id": f"synthetic-document-{candidate_id}",
    }


def test_exact_name_dose_route_and_temporal_evidence_matches() -> None:
    decision = score_medication_match(
        _candidate("mention-a"),
        _candidate("mention-b", route="oral", dose="0.5 g"),
    )

    assert decision.matched
    assert decision.confidence == 1.0
    assert decision.evidence == {
        "name": "match",
        "dose": "match",
        "route": "match",
        "temporal": "match",
    }
    assert decision.abstention_reasons == ()


def test_name_only_match_abstains_for_insufficient_evidence() -> None:
    decision = score_medication_match(
        _candidate("mention-a", dose=None, route=None, date_value=None, status=None),
        _candidate("mention-b", dose=None, route=None, date_value=None, status=None),
    )

    assert decision.abstained
    assert decision.confidence == 0.45
    assert decision.reason == "insufficient_evidence"
    assert decision.evidence["dose"] == "unknown"
    assert decision.evidence["route"] == "unknown"
    assert decision.evidence["temporal"] == "unknown"


def test_conflicting_dose_abstains_and_is_not_merged() -> None:
    result = reconcile_medications(
        [
            _candidate("mention-a"),
            _candidate("mention-b", dose="750 mg"),
        ]
    )

    assert len(result.groups) == 2
    assert result.merged_groups == ()
    assert len(result.abstentions) == 1
    assert result.abstentions[0].reason == "dose_conflict"
    assert result.abstentions[0].evidence["dose"] == "mismatch"


def test_conflicting_route_abstains_even_when_other_features_match() -> None:
    decision = score_medication_match(
        _candidate("mention-a"),
        _candidate("mention-b", route="intravenous"),
    )

    assert decision.abstained
    assert decision.reason == "route_conflict"
    assert "route_conflict" in decision.abstention_reasons


def test_current_and_stopped_overlapping_regimens_abstain() -> None:
    decision = score_medication_match(
        _candidate("mention-a", status="current"),
        {
            **_candidate("mention-b", status="discontinued"),
            "temporal": {"start": "2026-01-01", "end": "2026-01-31"},
        },
    )

    assert decision.abstained
    assert decision.reason == "temporal_conflict"
    assert decision.evidence["temporal"] == "mismatch"


def test_reconciliation_all_pairs_gate_blocks_transitive_conflict() -> None:
    policy = MedicationReconciliationPolicy(
        name_weight=0.0,
        dose_weight=0.0,
        route_weight=0.0,
        temporal_weight=1.0,
        match_threshold=0.8,
        temporal_gap_days=1.0,
    )
    result = reconcile_medications(
        [
            _candidate(
                "mention-a",
                date_value=None,
                status=None,
            )
            | {"temporal": {"start": "2026-01-01", "end": "2026-01-10"}},
            _candidate(
                "mention-b",
                date_value=None,
                status=None,
            )
            | {"temporal": {"start": "2026-01-05", "end": "2026-01-15"}},
            _candidate(
                "mention-c",
                date_value=None,
                status=None,
            )
            | {"temporal": {"start": "2026-01-12", "end": "2026-01-20"}},
        ],
        policy=policy,
    )

    assert len(result.merged_groups) == 1
    assert result.merged_groups[0].candidate_ids == ("mention-a", "mention-b")
    assert {"mention-c"} in [set(group.candidate_ids) for group in result.groups]
    assert any(
        "temporal_gap" in decision.abstention_reasons for decision in result.abstentions
    )
    assert any(
        "transitive_conflict" in group.abstention_reasons for group in result.groups
    )


def test_audit_serialization_hashes_sensitive_candidate_values() -> None:
    result = reconcile_medications([_candidate("private-mention")])

    report = result.to_dict()
    rendered = repr(report)
    assert "Synthetic Medication Alpha" not in rendered
    assert "private-mention" not in rendered
    assert "synthetic-document-private-mention" not in rendered
    assert report["candidate_count"] == 1
    assert report["groups"][0]["candidate_count"] == 1
