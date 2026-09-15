"""Synthetic offline tests for clinical section-boundary provenance."""

from __future__ import annotations

import json

import pytest

from openmed.clinical.sections.provenance import (
    SectionProvenanceError,
    SectionRange,
    validate_section_provenance,
)
from openmed.core.audit import hash_text


def test_valid_ranges_are_deterministic_and_source_text_free() -> None:
    text = "HPI: synthetic cough.\nPLAN: synthetic follow-up."
    ranges = (
        SectionRange(label="hpi", start=0, end=23, section_id="a"),
        SectionRange(label="plan", start=23, end=len(text), section_id="b"),
    )

    first = validate_section_provenance(text, ranges)
    second = validate_section_provenance(text, ranges)

    assert first.valid
    assert first.to_json() == second.to_json()
    assert first.document_hash == hash_text(text)
    assert text not in first.to_json()
    assert "synthetic cough" not in json.dumps(first.to_dict())


def test_gap_is_reported_with_offsets_and_hash_not_section_text() -> None:
    text = "HPI: synthetic gap.\nPLAN: synthetic plan."
    plan_start = text.index("PLAN")
    gap_start = text.index("\n")
    ranges = (
        {"id": "hpi", "start": 0, "end": gap_start},
        {"id": "plan", "start": plan_start, "end": len(text)},
    )

    report = validate_section_provenance(text, ranges)

    assert not report.valid
    gap = next(issue for issue in report.issues if issue.code == "gap")
    assert (gap.start, gap.end) == (gap_start, plan_start)
    assert gap.source_hash == hash_text(text[gap_start:plan_start])
    assert "synthetic" not in gap.message
    assert "synthetic gap" not in report.to_json()


def test_order_and_overlap_conflicts_are_distinct() -> None:
    text = "0123456789"
    ranges = (
        {"id": "later", "start": 6, "end": 10},
        {"id": "earlier", "start": 2, "end": 7},
    )

    report = validate_section_provenance(text, ranges, require_coverage=False)

    assert {issue.code for issue in report.issues} >= {"out_of_order", "overlap"}
    assert "ordering" in report.categories


def test_parent_containment_uses_offsets_and_never_parent_text() -> None:
    text = "ROOT synthetic parent content."
    ranges = (
        {"id": "root", "start": 0, "end": len(text)},
        {
            "id": "child",
            "parent_id": "root",
            "start": 0,
            "end": len(text) + 1,
        },
    )

    report = validate_section_provenance(text, ranges)

    issue = next(issue for issue in report.issues if issue.code == "outside_parent")
    assert issue.related_index == 0
    assert issue.start == 0
    assert issue.end == len(text) + 1
    assert "ROOT synthetic" not in report.to_json()


def test_source_map_reference_conflict_and_hash_mismatch_are_safe() -> None:
    text = "abcdef"
    ranges = (
        {
            "id": "first",
            "start": 0,
            "end": 3,
            "source_start": 0,
            "source_end": 3,
            "source_ref": "synthetic-ref",
            "source_hash": hash_text("wrong"),
        },
        {
            "id": "second",
            "start": 3,
            "end": 6,
            "source_start": 3,
            "source_end": 6,
            "source_ref": "synthetic-ref",
        },
    )

    report = validate_section_provenance(text, ranges, require_source_map=True)

    assert {issue.code for issue in report.issues} >= {
        "reference_conflict",
        "hash_mismatch",
    }
    assert "synthetic-ref" not in report.to_json()
    assert hash_text("synthetic-ref") in report.to_json()


def test_top_level_source_map_is_validated_without_network_or_text_copy() -> None:
    text = "abcdef"
    ranges = (
        {"id": "left", "start": 0, "end": 3},
        {"id": "right", "start": 3, "end": 6},
    )
    source_map = {
        "left": {"source_start": 0, "source_end": 3, "source_ref": "left-ref"},
        "right": {"source_start": 3, "source_end": 6, "source_ref": "right-ref"},
    }

    report = validate_section_provenance(
        text,
        ranges,
        source_map,
        require_source_map=True,
    )

    assert report.valid
    assert all(record.source_hash for record in report.ranges)
    assert "left-ref" not in report.to_json()


def test_strict_mode_raises_only_a_sanitized_error() -> None:
    with pytest.raises(SectionProvenanceError, match="issue"):
        validate_section_provenance(
            "sensitive synthetic text",
            [{"start": 2, "end": 3}],
            strict=True,
        )
