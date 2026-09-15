"""Tests for deterministic relation-candidate audit aggregation."""

from __future__ import annotations

import json
from dataclasses import dataclass

from openmed.eval.relation_audit import (
    ACCEPTED_FILTERING_REASON,
    RelationCandidateAuditRecord,
    RelationCandidateAuditReport,
    audit_relation_candidates,
    build_relation_candidate_audit_report,
)


@dataclass(frozen=True)
class _SyntheticEndpoint:
    section: str
    text: str = "synthetic endpoint"


@dataclass(frozen=True)
class _SyntheticCandidate:
    relation_type: str
    head: _SyntheticEndpoint
    attribute: _SyntheticEndpoint
    status: str = "confirmed"
    candidate_id: str = "synthetic-candidate-id"


def test_audit_aggregates_family_section_and_filter_reason() -> None:
    candidates = [
        {
            "relation_type": "drug_to_dose",
            "section": "Medications",
            "filtering_reason": "accepted",
            "candidate_id": "synthetic-id-1",
            "text": "synthetic source text",
        },
        {
            "relation_type": "problem_to_status",
            "section": "Assessment",
            "filter_reason": "assertion_refuted",
            "head_id": "synthetic-head-id",
            "tail_id": "synthetic-tail-id",
        },
        {
            "family": "temporal",
            "section": "History of Present Illness",
            "filtered": True,
        },
        _SyntheticCandidate(
            relation_type="drug_to_route",
            head=_SyntheticEndpoint("Medications"),
            attribute=_SyntheticEndpoint("Medications"),
        ),
    ]

    report = audit_relation_candidates(candidates)

    assert report.candidate_count == 4
    assert report.total_candidates == 4
    assert report.by_relation_family == {
        "drug": 2,
        "problem": 1,
        "temporal": 1,
    }
    assert report.by_section == {
        "assessment": 1,
        "history_of_present_illness": 1,
        "medications": 2,
    }
    assert report.by_filtering_reason == {
        "accepted": 2,
        "assertion_refuted": 1,
        "filtered": 1,
    }
    assert report.relation_family_counts == report.by_relation_family
    assert report.section_counts == report.by_section
    assert report.filtering_reason_counts == report.by_filtering_reason


def test_audit_is_order_independent_and_serializes_stably() -> None:
    candidates = [
        RelationCandidateAuditRecord("problem", "assessment", "accepted"),
        RelationCandidateAuditRecord("drug", "medications", "filtered"),
        RelationCandidateAuditRecord("drug", "medications", "accepted"),
    ]

    first = audit_relation_candidates(candidates)
    second = audit_relation_candidates(reversed(candidates))

    assert first == second
    assert build_relation_candidate_audit_report(candidates) == first
    assert json.loads(first.to_json()) == {
        "artifact": "relation_candidate_audit",
        "by_filtering_reason": {"accepted": 2, "filtered": 1},
        "by_relation_family": {"drug": 2, "problem": 1},
        "by_section": {"assessment": 1, "medications": 2},
        "candidate_count": 3,
        "schema_version": 1,
    }
    assert first.to_json() == first.to_json()
    assert first.to_markdown() == first.to_markdown()


def test_report_omits_source_text_and_identifiers(tmp_path) -> None:
    source_text = "synthetic source text that must not be emitted"
    identifier = "synthetic-record-identifier"
    report = audit_relation_candidates(
        {
            "relation_family": "laboratory",
            "section": "Results",
            "filtering_reason": ACCEPTED_FILTERING_REASON,
            "text": source_text,
            "source_text": source_text,
            "record_id": identifier,
            "head_id": identifier,
            "tail_id": identifier,
            "offsets": [3, 9],
        }
    )

    json_report = report.to_json()
    markdown_report = report.to_markdown()
    for forbidden in (source_text, identifier, "record_id", "head_id", "tail_id"):
        assert forbidden not in json_report
        assert forbidden not in markdown_report

    output_path = report.write_json(tmp_path / "nested" / "audit.json")
    assert RelationCandidateAuditReport.read_json(output_path) == report


def test_from_dict_round_trips_aggregate_only_payload() -> None:
    payload = {
        "artifact": "relation_candidate_audit",
        "candidate_count": 2,
        "schema_version": 1,
        "by_relation_family": {"drug": 1, "problem": 1},
        "by_section": {"assessment": 2},
        "by_filtering_reason": {"accepted": 1, "filtered": 1},
        "candidate_ids": ["synthetic-id-that-is-ignored"],
    }

    report = RelationCandidateAuditReport.from_dict(payload)

    assert report.candidate_count == 2
    assert "candidate_ids" not in report.to_dict()
    assert report.counts["by_section"] == {"assessment": 2}


def test_empty_input_is_a_valid_deterministic_report() -> None:
    report = audit_relation_candidates(None)

    assert report.candidate_count == 0
    assert report.by_relation_family == {}
    assert report.by_section == {}
    assert report.by_filtering_reason == {}
