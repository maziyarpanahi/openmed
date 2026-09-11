"""Focused tests for the deterministic radiology finding profile."""

from __future__ import annotations

import json

import pytest

from openmed.clinical import (
    RADIOLOGY_FINDING_PROFILE,
    RADIOLOGY_FINDING_PROFILE_ADVISORY,
    extract_radiology_profile,
)


def test_profile_binds_sections_attributes_and_negated_assertion() -> None:
    text = (
        "FINDINGS:\n"
        "No focal consolidation in the left lower lobe.\n"
        "Possible 8 mm nodule in the right upper lobe.\n"
        "RECOMMENDATION:\nRoutine synthetic follow-up."
    )

    records = extract_radiology_profile(text)

    assert [
        (
            record["finding"],
            record["laterality"],
            record["size_value"],
            record["size_unit"],
            record["location"],
            record["assertion"],
            record["section"],
        )
        for record in records
    ] == [
        ("consolidation", "left", None, None, "lower lobe", "negated", "findings"),
        ("nodule", "right", 8.0, "mm", "upper lobe", "unknown", "findings"),
    ]
    assert records[0]["evidence"]["assertion"] is not None
    assert records[1]["evidence"]["assertion"] is not None


def test_profile_uses_explicit_unknown_fields_and_excludes_recommendations() -> None:
    text = (
        "FINDINGS: A synthetic lesion is present.\n"
        "RECOMMENDATION: Consider a 4 mm nodule for future review."
    )

    [record] = extract_radiology_profile(text)

    assert record["laterality"] == "unknown"
    assert record["size_value"] is None
    assert record["size_unit"] is None
    assert record["location"] is None
    assert record["assertion"] == "affirmed"
    assert record["unknown_fields"] == ["laterality", "size", "location"]
    assert record["evidence"]["laterality"] is None
    assert record["evidence"]["size_value"] is None
    assert record["evidence"]["location"] is None
    assert "recommendation" not in record["section"]


def test_hypothetical_finding_keeps_unknown_assertion_evidence() -> None:
    text = "If a synthetic nodule is present, obtain local follow-up."

    [record] = extract_radiology_profile(text)

    assert record["assertion"] == "unknown"
    assertion_evidence = record["evidence"]["assertion"]
    assert assertion_evidence is not None
    assert text[assertion_evidence["start"] : assertion_evidence["end"]].lower() == "if"


def test_profile_evidence_points_into_the_original_source_without_copying_it() -> None:
    text = "IMPRESSION: A 5 mm right upper lobe nodule is present."

    [record] = RADIOLOGY_FINDING_PROFILE(text)
    evidence = record["evidence"]

    assert (
        text[evidence["finding"]["start"] : evidence["finding"]["end"]].lower()
        == "nodule"
    )
    assert text[evidence["size_value"]["start"] : evidence["size_value"]["end"]] == "5"
    assert (
        text[evidence["size_unit"]["start"] : evidence["size_unit"]["end"]].lower()
        == "mm"
    )
    assert (
        text[evidence["laterality"]["start"] : evidence["laterality"]["end"]].lower()
        == "right"
    )
    assert (
        text[evidence["location"]["start"] : evidence["location"]["end"]].lower()
        == "upper lobe"
    )
    assert "synthetic" not in json.dumps(record)


def test_profile_is_deterministic_and_aliases_match() -> None:
    text = "FINDINGS: Bilateral 2 cm cysts."

    first = extract_radiology_profile(text)
    second = extract_radiology_profile(text)

    assert first == second
    assert first == RADIOLOGY_FINDING_PROFILE.extract(text)
    assert first[0]["laterality"] == "bilateral"
    assert first[0]["unknown_fields"] == ["location"]


def test_profile_rejects_non_string_input_without_echoing_source() -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        extract_radiology_profile(17)  # type: ignore[arg-type]


def test_profile_advisory_is_review_only() -> None:
    lowered = RADIOLOGY_FINDING_PROFILE_ADVISORY.lower()
    assert "deterministic" in lowered
    assert "not diagnostic" in lowered
    assert "treatment recommendation" in lowered
