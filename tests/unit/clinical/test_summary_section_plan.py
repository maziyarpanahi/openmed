"""Offline regressions for the section-preserving summary planner."""

from __future__ import annotations

import json

import pytest

from openmed.clinical import (
    SUMMARY_PLAN_STATUS_READY,
    SUMMARY_PLAN_STATUS_REFUSED,
    SUMMARY_SECTION_PLAN_DISCLAIMER,
    SummaryPlanRefusalReason,
    SummarySectionPlanError,
    build_summary_section_plan,
    require_summary_section_plan,
)

PRIVATE_SENTINELS = (
    "SYNTHETIC_SOURCE_SURFACE",
    "SYNTHETIC_EVIDENCE_VALUE",
)


def _evidence(
    evidence_id: str,
    section_id: str,
    start: int,
    end: int,
    *,
    approved: bool = True,
) -> dict[str, object]:
    return {
        "evidence_id": evidence_id,
        "section_id": section_id,
        "source_offset": {"start": start, "end": end},
        "approved": approved,
        "text": PRIVATE_SENTINELS[0],
        "value": PRIVATE_SENTINELS[1],
    }


def test_planner_groups_approved_evidence_by_section_and_orders_it() -> None:
    evidence = [
        _evidence("e-2", "s-2", 30, 35),
        _evidence("e-1b", "s-1", 20, 25),
        _evidence("e-1a", "s-1", 10, 15),
        _evidence("e-rejected", "s-1", 40, 45, approved=False),
    ]

    plan = build_summary_section_plan(reversed(evidence))

    assert plan.status == SUMMARY_PLAN_STATUS_READY
    assert plan.refusal is None
    assert [group.section_id for group in plan.groups] == ["s-1", "s-2"]
    assert [item.evidence_id for item in plan.groups[0].evidence] == [
        "e-1a",
        "e-1b",
    ]
    assert plan.input_evidence_count == 4
    assert plan.approved_evidence_count == 3
    assert plan.excluded_evidence_count == 1
    assert plan.requires_clinician_review is True
    assert plan.autonomous_decision is False


def test_order_and_json_are_input_order_independent() -> None:
    evidence = [
        _evidence("e-b", "s-b", 40, 45),
        _evidence("e-a", "s-a", 5, 12),
    ]

    first = build_summary_section_plan(evidence)
    second = build_summary_section_plan(reversed(evidence))

    assert first == second
    assert first.to_json() == second.to_json()
    assert json.loads(first.to_json()) == first.to_dict()


def test_missing_section_id_returns_typed_refusal_without_echoing_source() -> None:
    plan = build_summary_section_plan(
        [
            {
                "evidence_id": "e-unscoped",
                "start": 3,
                "end": 9,
                "text": PRIVATE_SENTINELS[0],
                "value": PRIVATE_SENTINELS[1],
            }
        ]
    )

    assert plan.status == SUMMARY_PLAN_STATUS_REFUSED
    assert plan.refusal_reason is SummaryPlanRefusalReason.MISSING_SECTION_ID
    assert plan.refusal is not None
    assert plan.refusal.reason_code == "missing_section_id"
    serialized = plan.to_json()
    assert PRIVATE_SENTINELS[0] not in serialized
    assert PRIVATE_SENTINELS[1] not in serialized
    assert all(sentinel not in str(plan) for sentinel in PRIVATE_SENTINELS)


def test_section_label_is_not_accepted_as_a_stable_section_identifier() -> None:
    plan = build_summary_section_plan([{"evidence_id": "e-1", "section": "assessment"}])

    assert plan.refusal_reason is SummaryPlanRefusalReason.MISSING_SECTION_ID


def test_strict_consumer_gets_fixed_code_error_for_refused_plan() -> None:
    with pytest.raises(
        SummarySectionPlanError,
        match="summary_section_plan_missing_section_id",
    ) as error:
        require_summary_section_plan([{"evidence_id": "e-1"}])

    assert error.value.reason is SummaryPlanRefusalReason.MISSING_SECTION_ID
    assert "e-1" not in str(error.value)


def test_section_metadata_orders_groups_without_replacing_evidence_ids() -> None:
    plan = build_summary_section_plan(
        [
            _evidence("e-later", "section-later", 8, 12),
            _evidence("e-earlier", "section-earlier", 80, 84),
        ],
        sections=[
            {"id": "section-earlier", "start": 4, "end": 20},
            {"id": "section-later", "start": 60, "end": 100},
        ],
    )

    assert [group.section_id for group in plan.groups] == [
        "section-earlier",
        "section-later",
    ]
    assert plan.groups[0].source_offset == (4, 20)
    assert [item.evidence_id for group in plan.groups for item in group.evidence] == [
        "e-earlier",
        "e-later",
    ]


def test_canonical_section_identifiers_with_underscores_are_preserved() -> None:
    plan = build_summary_section_plan([_evidence("e-1", "assessment_and_plan", 10, 14)])

    assert plan.groups[0].section_id == "assessment_and_plan"


def test_named_approved_collection_and_detected_sections_are_supported() -> None:
    plan = build_summary_section_plan(
        approved_evidence={
            "records": [_evidence("e-1", "s-1", 10, 14)],
        },
        sections={
            "sections": [{"id": "s-1", "start": 0, "end": 20}],
        },
    )

    assert plan.ready is True
    assert plan.groups[0].source_section_id == "s-1"


def test_evidence_with_unknown_section_is_refused_when_sections_are_supplied() -> None:
    plan = build_summary_section_plan(
        [_evidence("e-1", "s-unknown", 10, 14)],
        sections=[{"id": "s-known", "start": 0, "end": 20}],
    )

    assert plan.refusal_reason is SummaryPlanRefusalReason.UNKNOWN_SECTION_ID


@pytest.mark.parametrize(
    "row",
    [
        _evidence("e-1", "", 1, 2),
        _evidence("e-1", "s 1", 1, 2),
        _evidence("not a stable id", "s-1", 1, 2),
    ],
)
def test_unstable_identifiers_are_refused_without_value_echo(
    row: dict[str, object],
) -> None:
    plan = build_summary_section_plan([row])

    assert plan.refused is True
    assert plan.refusal_reason in {
        SummaryPlanRefusalReason.INVALID_SECTION_ID,
        SummaryPlanRefusalReason.INVALID_EVIDENCE_ID,
    }
    assert all(sentinel not in plan.to_json() for sentinel in PRIVATE_SENTINELS)


def test_plan_disclaimer_is_fixed_and_value_free() -> None:
    plan = build_summary_section_plan([_evidence("e-1", "s-1", 1, 2)])

    assert plan.to_dict()["disclaimer"] == SUMMARY_SECTION_PLAN_DISCLAIMER
    assert "clinical" in SUMMARY_SECTION_PLAN_DISCLAIMER.casefold()
