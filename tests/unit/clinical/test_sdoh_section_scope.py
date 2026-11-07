"""Synthetic offline tests for strict SDOH section boundaries."""

from __future__ import annotations

import json
from dataclasses import dataclass

from openmed.clinical.sdoh_section_scope import (
    SDOH_SCOPE_FALLBACK_DOCUMENT,
    SDOH_SCOPE_FALLBACK_REJECT,
    SDOH_SCOPE_FALLBACK_UNSECTIONED,
    SDOHSectionScopePolicy,
    scope_sdoh_candidates,
)
from openmed.clinical.sections import detect_sections


def _candidate(text: str, value: str) -> dict[str, object]:
    start = text.index(value)
    return {"start": start, "end": start + len(value), "text": value}


def test_default_scope_keeps_only_social_history_candidates() -> None:
    text = (
        "Assessment: SYNTHETIC_ASSESSMENT_VALUE.\n"
        "Social History: SYNTHETIC_SOCIAL_VALUE.\n"
        "Plan: SYNTHETIC_PLAN_VALUE."
    )
    candidates = tuple(
        _candidate(text, value)
        for value in (
            "SYNTHETIC_ASSESSMENT_VALUE",
            "SYNTHETIC_SOCIAL_VALUE",
            "SYNTHETIC_PLAN_VALUE",
        )
    )

    result = scope_sdoh_candidates(text, candidates, detect_sections(text))

    assert result.candidates == (candidates[1],)
    assert result.candidate_offsets == ((candidates[1]["start"], candidates[1]["end"]),)
    assert dict(result.excluded_candidate_counts) == {
        "assessment": 1,
        "plan": 1,
    }
    assert result.report.fallback_used is False

    report_json = result.to_json()
    assert all(
        value not in report_json
        for value in (
            "SYNTHETIC_ASSESSMENT_VALUE",
            "SYNTHETIC_SOCIAL_VALUE",
            "SYNTHETIC_PLAN_VALUE",
        )
    )
    assert json.loads(report_json)["excluded_candidate_count"] == 2


def test_allowed_sections_are_configurable_without_changing_boundaries() -> None:
    text = (
        "Assessment: SYNTHETIC_ASSESSMENT_VALUE.\n"
        "Social History: SYNTHETIC_SOCIAL_VALUE.\n"
        "Plan: SYNTHETIC_PLAN_VALUE."
    )
    candidates = tuple(
        _candidate(text, value)
        for value in (
            "SYNTHETIC_ASSESSMENT_VALUE",
            "SYNTHETIC_SOCIAL_VALUE",
            "SYNTHETIC_PLAN_VALUE",
        )
    )

    result = scope_sdoh_candidates(
        text,
        candidates,
        detect_sections(text),
        allowed_sections=("social_history", "assessment"),
    )

    assert result.candidates == (candidates[0], candidates[1])
    assert dict(result.report.excluded_candidate_counts) == {"plan": 1}


def test_missing_configured_section_fails_closed_by_default() -> None:
    text = "Plan: SYNTHETIC_PLAN_VALUE."
    result = scope_sdoh_candidates(text, (_candidate(text, "SYNTHETIC_PLAN_VALUE"),))

    assert result.candidates == ()
    assert result.report.policy.fallback == SDOH_SCOPE_FALLBACK_REJECT
    assert result.report.fallback_used is True
    assert dict(result.report.excluded_candidate_counts) == {"plan": 1}
    assert dict(result.report.excluded_reason_counts) == {"fallback_rejected": 1}


def test_document_fallback_is_explicit_and_value_free() -> None:
    text = "Plan: SYNTHETIC_PLAN_VALUE."
    result = scope_sdoh_candidates(
        text,
        (_candidate(text, "SYNTHETIC_PLAN_VALUE"),),
        fallback=SDOH_SCOPE_FALLBACK_DOCUMENT,
    )

    assert len(result) == 1
    assert result.report.fallback_used is True
    assert result.report.excluded_candidate_count == 0
    assert "SYNTHETIC_PLAN_VALUE" not in result.report.to_markdown()


def test_unsectioned_fallback_is_explicit() -> None:
    text = "SYNTHETIC_UNSECTIONED_VALUE."
    result = scope_sdoh_candidates(
        text,
        (_candidate(text, "SYNTHETIC_UNSECTIONED_VALUE"),),
        fallback=SDOH_SCOPE_FALLBACK_UNSECTIONED,
    )

    assert len(result) == 1
    assert result.report.policy.fallback == SDOH_SCOPE_FALLBACK_UNSECTIONED
    assert result.report.fallback_used is True


def test_candidates_crossing_a_section_boundary_are_rejected() -> None:
    text = "Social History: SYNTHETIC_VALUE\nPlan: SYNTHETIC_PLAN"
    social_start = text.index("Social History")
    plan_start = text.index("Plan")
    candidate_start = text.index("SYNTHETIC_VALUE")
    sections = (
        {"label": "social_history", "start": social_start, "end": plan_start},
        {"label": "plan", "start": plan_start, "end": len(text)},
    )
    candidates = (
        {"start": candidate_start, "end": plan_start + 2, "text": "SYNTHETIC"},
    )

    result = scope_sdoh_candidates(text, candidates, sections)

    assert result.candidates == ()
    assert dict(result.report.excluded_candidate_counts) == {"social_history": 1}
    assert dict(result.report.excluded_reason_counts) == {"cross_section_boundary": 1}


@dataclass(frozen=True)
class _SpanCandidate:
    span: tuple[int, int]
    value: str


def test_scope_is_deterministic_and_accepts_span_objects() -> None:
    text = "Social History: SYNTHETIC_VALUE."
    start = text.index("SYNTHETIC_VALUE")
    candidate = _SpanCandidate(
        span=(start, start + len("SYNTHETIC_VALUE")),
        value="SYNTHETIC_VALUE",
    )
    policy = SDOHSectionScopePolicy()
    sections = detect_sections(text)

    first = scope_sdoh_candidates(text, (candidate,), sections, policy=policy)
    second = scope_sdoh_candidates(text, (candidate,), sections, policy=policy)

    assert first.to_dict() == second.to_dict()
    assert first.candidate_offsets == (candidate.span,)


def test_invalid_candidate_is_counted_without_persisting_value() -> None:
    text = "Social History: SYNTHETIC_VALUE."
    result = scope_sdoh_candidates(
        text,
        ({"start": "invalid", "end": 2, "text": "SYNTHETIC_VALUE"},),
        detect_sections(text),
    )

    assert result.candidates == ()
    assert dict(result.report.excluded_candidate_counts) == {"invalid": 1}
    assert "SYNTHETIC_VALUE" not in result.report.to_json()
