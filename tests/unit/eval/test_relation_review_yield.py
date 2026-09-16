"""Focused offline tests for guarded clinical relation review yield."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping

import pytest

from openmed.eval.relation_review_yield import (
    RelationReviewCandidate,
    RelationReviewDisposition,
    RelationReviewYieldError,
    build_relation_review_yield_report,
    compute_relation_review_yield,
)


def test_counts_are_reported_by_relation_class_with_explicit_denominators():
    report = compute_relation_review_yield(
        [
            {"relation_class": "treats", "disposition": "accepted"},
            {"relation_class": "TREATS", "disposition": "corrected"},
            {"relation_class": "TREATS", "disposition": "rejected"},
            {"relation_class": "TREATS", "disposition": "duplicate"},
            {"relation_class": "TREATS", "disposition": "deferred"},
            {"relation_type": "causes", "review_status": "rejected"},
        ]
    )

    treats = report.by_relation_class["TREATS"]
    assert treats.accepted_count == 1
    assert treats.corrected_count == 1
    assert treats.rejected_count == 1
    assert treats.duplicate_count == 1
    assert treats.deferred_count == 1
    assert treats.candidate_count == 5
    assert treats.reviewed_count == 4
    assert treats.review_yield == pytest.approx(0.5)
    assert treats.surface_yield == pytest.approx(0.4)
    assert treats.reviewed_rate == pytest.approx(0.8)

    assert report.overall.accepted == 1
    assert report.overall.corrected == 1
    assert report.overall.rejected == 2
    assert report.overall.duplicate == 1
    assert report.overall.deferred == 1
    assert report.overall.review_yield == pytest.approx(2 / 5)
    assert report.overall.surface_yield == pytest.approx(2 / 6)


def test_declared_empty_classes_and_enum_dispositions_are_deterministic():
    records = [
        RelationReviewCandidate("procedure-indication", "accepted"),
        {"relation": {"type": "TREATS"}, "review": {"status": "deferred"}},
    ]
    report = build_relation_review_yield_report(
        records,
        relation_classes=("CAUSES", "TREATS", "PROCEDURE_INDICATION"),
    )

    assert report.relation_classes == (
        "CAUSES",
        "PROCEDURE_INDICATION",
        "TREATS",
    )
    assert report.by_relation_class["CAUSES"].candidate_count == 0
    assert report.by_relation_class["TREATS"].deferred_count == 1
    assert report.by_relation_class["PROCEDURE_INDICATION"].accepted_count == 1
    assert report.to_json() == report.to_json()
    assert report.to_dict() == json.loads(report.to_json())

    enum_report = compute_relation_review_yield(
        [
            {
                "relation_class": "TREATS",
                "disposition": RelationReviewDisposition.ACCEPTED,
            }
        ]
    )
    assert enum_report.overall.accepted == 1


def test_outputs_drop_case_and_reviewer_fields():
    report = compute_relation_review_yield(
        [
            {
                "relation_class": "TREATS",
                "disposition": "accepted",
                "case_id": "synthetic-case-001",
                "reviewer_id": "synthetic-reviewer-001",
                "source_text": "synthetic fixture content",
                "score": 0.99,
            }
        ]
    )

    serialized = report.to_json() + report.to_markdown()
    assert "synthetic-case-001" not in serialized
    assert "synthetic-reviewer-001" not in serialized
    assert "synthetic fixture content" not in serialized
    assert "reviewer" not in serialized.lower()
    assert report["by_relation_class"]["TREATS"]["accepted"] == 1


def test_invalid_control_metadata_fails_without_echoing_values():
    with pytest.raises(RelationReviewYieldError, match="disposition"):
        compute_relation_review_yield(
            [{"relation_class": "TREATS", "disposition": "not-a-disposition"}]
        )

    with pytest.raises(RelationReviewYieldError, match="relation class"):
        compute_relation_review_yield(
            [{"relation_class": "<synthetic-patient>", "disposition": "accepted"}]
        )

    with pytest.raises(RelationReviewYieldError, match="relation class") as error:
        compute_relation_review_yield([{"disposition": "accepted"}])
    assert "accepted" not in str(error.value)


def test_empty_input_has_zero_safe_rates_and_stable_markdown():
    report = compute_relation_review_yield([], relation_classes=("TREATS",))

    assert report.candidate_count == 0
    assert report.review_yield == 0.0
    assert report.surface_yield == 0.0
    assert report.by_relation_class["TREATS"].to_dict()["deferred_rate"] == 0.0
    markdown = report.to_markdown()
    assert "| `Overall` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.000000 | 0.000000 |" in markdown


def test_unreadable_input_never_echoes_sensitive_exception_text():
    sensitive_value = "synthetic-sensitive-identifier"

    class _UnreadableMapping(Mapping[str, object]):
        def __getitem__(self, key: str) -> object:
            raise RuntimeError(sensitive_value)

        def __iter__(self) -> Iterator[str]:
            return iter(("relation_class",))

        def __len__(self) -> int:
            return 1

    with pytest.raises(RelationReviewYieldError, match="field is unreadable") as error:
        compute_relation_review_yield([_UnreadableMapping()])
    assert sensitive_value not in str(error.value)

    def _unreadable_records() -> Iterator[object]:
        yield {"relation_class": "TREATS", "disposition": "accepted"}
        raise RuntimeError(sensitive_value)

    with pytest.raises(RelationReviewYieldError, match="input is unreadable") as error:
        compute_relation_review_yield(_unreadable_records())
    assert sensitive_value not in str(error.value)

    def _unreadable_classes() -> Iterator[str]:
        yield "TREATS"
        raise RuntimeError(sensitive_value)

    with pytest.raises(RelationReviewYieldError, match="input is unreadable") as error:
        compute_relation_review_yield([], relation_classes=_unreadable_classes())
    assert sensitive_value not in str(error.value)
