"""Tests for deterministic relation-candidate audit aggregation."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import pytest

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


def test_category_fields_cannot_turn_identifiers_into_report_keys() -> None:
    identifier = "synthetic_patient_name_123"
    report = audit_relation_candidates(
        [
            {
                "relation_family": identifier,
                "section": identifier,
                "filtering_reason": identifier,
            },
            RelationCandidateAuditRecord(identifier, identifier, identifier),
        ]
    )

    assert report.by_relation_family == {"unknown": 2}
    assert report.by_section == {"unsectioned": 2}
    assert report.by_filtering_reason == {"other": 2}
    assert identifier not in report.to_json()
    assert identifier not in report.to_markdown()

    restored = RelationCandidateAuditReport.from_dict(
        {
            "candidate_count": 1,
            "by_relation_family": {identifier: 1},
            "by_section": {identifier: 1},
            "by_filtering_reason": {identifier: 1},
        }
    )
    assert identifier not in restored.to_json()


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


def test_unreadable_input_never_echoes_sensitive_exception_text() -> None:
    sensitive_value = "synthetic-sensitive-identifier"

    class _UnreadableCandidate:
        @property
        def relation_type(self) -> str:
            raise RuntimeError(sensitive_value)

    report = audit_relation_candidates([_UnreadableCandidate()])
    assert sensitive_value not in report.to_json()
    assert report.by_relation_family == {"unknown": 1}

    def _unreadable_records() -> Iterator[object]:
        yield {"relation_family": "drug"}
        raise RuntimeError(sensitive_value)

    with pytest.raises(ValueError, match="input is unreadable") as error:
        audit_relation_candidates(_unreadable_records())
    assert sensitive_value not in str(error.value)

    class _UnreadableCounts(Mapping[str, int]):
        def __getitem__(self, key: str) -> int:
            raise RuntimeError(sensitive_value)

        def __iter__(self) -> Iterator[str]:
            raise RuntimeError(sensitive_value)

        def __len__(self) -> int:
            return 1

    with pytest.raises(ValueError, match="dimension is unreadable") as error:
        RelationCandidateAuditReport(
            candidate_count=1,
            by_section=_UnreadableCounts(),
        )
    assert sensitive_value not in str(error.value)


def test_report_dimensions_must_equal_the_candidate_total():
    with pytest.raises(ValueError):
        RelationCandidateAuditReport(2, {"drug": 1}, {"assessment": 2}, {"accepted": 2})


def test_boolean_schema_is_not_a_version_number():
    with pytest.raises(ValueError):
        RelationCandidateAuditReport(0, schema_version=True)


def test_conflicting_total_aliases_are_rejected():
    with pytest.raises(ValueError):
        RelationCandidateAuditReport.from_dict(
            {"candidate_count": 0, "total_candidates": 1}
        )


@pytest.mark.parametrize("value", ["synthetic-sensitive-marker", b"synthetic-marker"])
def test_scalar_text_is_not_a_candidate_batch(value):
    with pytest.raises(ValueError):
        audit_relation_candidates(value)


def test_invalid_json_discards_private_decoder_context(tmp_path):
    source = tmp_path / "bad.json"
    source.write_text('{"synthetic-sensitive-marker":', encoding="utf-8")
    with pytest.raises(ValueError) as error:
        RelationCandidateAuditReport.read_json(source)
    assert error.value.__context__ is None
    assert not hasattr(error.value, "doc")


def test_error_context_does_not_retain_iterator_failure():
    def values():
        raise RuntimeError("synthetic-sensitive-marker")
        yield

    with pytest.raises(ValueError) as error:
        audit_relation_candidates(values())
    assert error.value.__context__ is None


def test_serialization_indent_cannot_inject_source_values():
    report = audit_relation_candidates([{}])
    with pytest.raises(ValueError):
        report.to_json(indent="synthetic-sensitive-marker")


def test_conflicting_filter_flags_cannot_be_counted_as_accepted():
    with pytest.raises(ValueError):
        audit_relation_candidates([{"filtered": False, "rejected": True}])


def test_non_boolean_filter_flag_goes_to_other_bucket():
    report = audit_relation_candidates([{"filtered": "false"}])
    assert report.by_filtering_reason == {"other": 1}


def test_audit_iteration_is_bounded(monkeypatch):
    from itertools import repeat

    import openmed.eval.relation_audit as module

    monkeypatch.setattr(module, "MAX_AUDIT_RECORDS", 2)
    with pytest.raises(ValueError):
        audit_relation_candidates(repeat({}))


def test_typed_records_are_renormalized():
    record = RelationCandidateAuditRecord()
    object.__setattr__(record, "relation_family", "synthetic-sensitive-marker")
    report = audit_relation_candidates([record])
    assert report.by_relation_family == {"unknown": 1}
    assert "synthetic-sensitive-marker" not in record.to_dict().values()
