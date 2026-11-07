"""Synthetic offline tests for the SHAC-aligned SDOH skeleton."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

import openmed.clinical.sdoh as sdoh
from openmed.clinical.sdoh import (
    SDOHFinding,
    available_determinant_extractors,
    extract_sdoh,
    register_determinant_extractor,
    unregister_determinant_extractor,
)
from openmed.clinical.sections import detect_sections

_SOCIAL_EXTRACTORS = frozenset({"employment", "food_insecurity", "living_status"})
SDOH_GUIDE = (
    Path(__file__).resolve().parents[3] / "docs" / "clinical" / "sdoh-extraction.md"
)


def _assert_documented_determinants(markdown: str) -> None:
    heading = "## Built-in determinants and statuses\n"
    assert heading in markdown, "SDOH determinant table is missing"
    section = markdown.split(heading, 1)[1].split("\n## ", 1)[0]
    categories = re.findall(r"(?m)^\| `([a-z_]+)` \|", section)
    assert tuple(sorted(categories)) == available_determinant_extractors(), (
        "documented SDOH determinants differ from the registry"
    )


def _run_documented_examples(markdown: str) -> list[dict[str, Any]]:
    from openmed.core.offline import network_blocked_if_offline

    examples = re.findall(r"```python\n(.*?)\n```", markdown, flags=re.DOTALL)
    assert examples, "SDOH guide has no runnable Python examples"
    namespaces = []
    with network_blocked_if_offline(local_only=True):
        for index, source in enumerate(examples):
            namespace: dict[str, Any] = {"__name__": "__sdoh_docs_example__"}
            exec(compile(source, f"<sdoh-doc-example-{index}>", "exec"), namespace)
            namespaces.append(namespace)
    return namespaces


@pytest.fixture
def isolated_documentation_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = sdoh.DeterminantExtractorRegistry()
    for name, extractor in sdoh._DETERMINANT_EXTRACTORS.items():
        registry.register(name, extractor)
    monkeypatch.setattr(sdoh, "_DETERMINANT_EXTRACTORS", registry)


def test_sdoh_guide_lists_exactly_the_registered_determinants() -> None:
    _assert_documented_determinants(SDOH_GUIDE.read_text(encoding="utf-8"))


def test_sdoh_guide_examples_run_offline_and_preserve_registry(
    isolated_documentation_registry: None,
) -> None:
    before = sdoh._DETERMINANT_EXTRACTORS.items()
    examples = _run_documented_examples(SDOH_GUIDE.read_text(encoding="utf-8"))
    assert len(examples) == 3
    assert examples[0]["summary"] == [
        ("employment", "unemployed", (45, 55)),
        ("food_insecurity", "current", (70, 85)),
        ("living_status", "lives_alone", (57, 68)),
    ]
    assert len(examples[1]["custom_summary"]) == 1
    assert examples[1]["custom_summary"][0][:2] == ("transportation", "unknown")
    assert examples[2]["loaded"]["determinants"]["food_insecurity"]["cues"] == [
        "synthetic pantry cue"
    ]
    assert not examples[2]["path"].exists()
    assert sdoh._DETERMINANT_EXTRACTORS.items() == before


def test_sdoh_guide_check_rejects_an_unlisted_new_determinant(
    isolated_documentation_registry: None,
) -> None:
    sdoh.register_determinant_extractor("synthetic_new_determinant", lambda *_: [])
    with pytest.raises(AssertionError, match="differ from the registry"):
        _assert_documented_determinants(SDOH_GUIDE.read_text(encoding="utf-8"))


def test_sdoh_guide_check_rejects_incorrect_table_labels() -> None:
    markdown = SDOH_GUIDE.read_text(encoding="utf-8").replace(
        "| `drug` |", "| `synthetic_unknown` |", 1
    )
    with pytest.raises(AssertionError, match="differ from the registry"):
        _assert_documented_determinants(markdown)


def test_sdoh_guide_check_rejects_duplicate_table_rows() -> None:
    markdown = SDOH_GUIDE.read_text(encoding="utf-8").replace(
        "| `drug` |", "| `drug` |\n| `drug` |", 1
    )
    with pytest.raises(AssertionError, match="differ from the registry"):
        _assert_documented_determinants(markdown)


def test_sdoh_guide_check_rejects_missing_table() -> None:
    with pytest.raises(AssertionError, match="table is missing"):
        _assert_documented_determinants("No determinant table.\n")


def test_sdoh_guide_example_check_rejects_section_scope_drift(
    isolated_documentation_registry: None,
) -> None:
    markdown = SDOH_GUIDE.read_text(encoding="utf-8").replace(
        "sections=detect_sections(text)", "sections=[]", 1
    )
    with pytest.raises(AssertionError):
        _run_documented_examples(markdown)


def test_sdoh_guide_example_check_rejects_output_drift(
    isolated_documentation_registry: None,
) -> None:
    markdown = SDOH_GUIDE.read_text(encoding="utf-8").replace(
        "assert len(findings) == 3", "assert len(findings) == 4", 1
    )
    with pytest.raises(AssertionError):
        _run_documented_examples(markdown)


def test_sdoh_guide_example_check_rejects_missing_examples() -> None:
    with pytest.raises(AssertionError, match="no runnable Python examples"):
        _run_documented_examples("No Python example.\n")


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


def test_substance_extractors_are_registered_by_default() -> None:
    available = available_determinant_extractors()

    assert "tobacco" in available
    assert "alcohol" in available
    assert "drug" in available


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

    synthetic_findings = [
        finding for finding in findings if finding.category == "synthetic"
    ]

    assert [finding.span for finding in synthetic_findings] == [
        (all_spans[1]["start"], all_spans[1]["end"])
    ]
    assert social_section["start"] <= synthetic_findings[0].span[0]
    assert synthetic_findings[0].span[1] <= social_section["end"]

    available = available_determinant_extractors()

    assert _SOCIAL_EXTRACTORS <= set(available)
    assert "tobacco" in available
    assert "alcohol" in available
    assert "drug" in available
    assert available == registered_before


def _substring_offsets(text: str, substring: str) -> list[int]:
    offsets: list[int] = []
    cursor = 0
    while (offset := text.find(substring, cursor)) >= 0:
        offsets.append(offset)
        cursor = offset + len(substring)
    return offsets
