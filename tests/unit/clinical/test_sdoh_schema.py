"""Synthetic offline tests for the SHAC-aligned SDOH skeleton."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from openmed.clinical.sdoh import (
    SDOHFinding,
    available_determinant_extractors,
    extract_sdoh,
    register_determinant_extractor,
    unregister_determinant_extractor,
)
from openmed.clinical.sections import detect_sections

_SOCIAL_EXTRACTORS = frozenset({"employment", "food_insecurity", "living_status"})


def test_sdoh_finding_round_trips_through_dict() -> None:
    finding = SDOHFinding(
        category="tobacco",
        value="smoking",
        status="past",
        extent="synthetic 10 pack-years",
        temporality="historical",
        span=(16, 23),
        score=0.91,
    )

    payload = finding.to_dict()

    assert payload == {
        "category": "tobacco",
        "value": "smoking",
        "status": "past",
        "extent": "synthetic 10 pack-years",
        "temporality": "historical",
        "span": [16, 23],
        "score": 0.91,
    }
    assert SDOHFinding.from_dict(payload) == finding


def test_extract_sdoh_without_matching_cues_returns_empty() -> None:
    assert _SOCIAL_EXTRACTORS <= set(available_determinant_extractors())
    assert extract_sdoh("Synthetic Social History note.", spans=[]) == []


def test_registered_extractor_is_scoped_to_social_history_section() -> None:
    text = (
        "Assessment: Synthetic dummy-marker mention.\n"
        "Social History: Synthetic dummy-marker mention.\n"
        "Plan: Synthetic dummy-marker mention."
    )
    trigger = "dummy-marker"
    all_spans = [
        {"start": index, "end": index + len(trigger)}
        for index in _substring_offsets(text, trigger)
    ]
    received_spans: list[Sequence[Any]] = []
    registered_before = available_determinant_extractors()

    def dummy_extractor(
        source_text: str,
        candidate_spans: Sequence[Any],
    ) -> list[SDOHFinding]:
        assert source_text is text
        received_spans.append(candidate_spans)
        return [
            SDOHFinding(
                category="synthetic",
                value=source_text[span["start"] : span["end"]],
                status=None,
                extent=None,
                temporality=None,
                span=(span["start"], span["end"]),
                score=1.0,
            )
            for span in all_spans
        ]

    register_determinant_extractor("synthetic-dummy", dummy_extractor)
    try:
        findings = extract_sdoh(
            text,
            spans=all_spans,
            sections=detect_sections(text),
        )
    finally:
        unregister_determinant_extractor("synthetic-dummy")

    social_section = next(
        section
        for section in detect_sections(text)
        if section["label"] == "social_history"
    )
    assert len(received_spans) == 1
    assert received_spans[0] == (all_spans[1],)
    assert [finding.span for finding in findings] == [
        (all_spans[1]["start"], all_spans[1]["end"])
    ]
    assert social_section["start"] <= findings[0].span[0]
    assert findings[0].span[1] <= social_section["end"]
    assert available_determinant_extractors() == registered_before


def _substring_offsets(text: str, substring: str) -> list[int]:
    offsets: list[int] = []
    cursor = 0
    while (offset := text.find(substring, cursor)) >= 0:
        offsets.append(offset)
        cursor = offset + len(substring)
    return offsets
