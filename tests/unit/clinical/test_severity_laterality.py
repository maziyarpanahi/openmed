"""Tests for the deterministic severity / laterality modifier extractor."""

from __future__ import annotations

import pytest

from openmed.clinical import (
    LATERALITY_BILATERAL,
    LATERALITY_LEFT,
    LATERALITY_RIGHT,
    LATERALITY_UNSPECIFIED,
    SEVERITY_LATERALITY_ADVISORY,
    SEVERITY_MILD,
    SEVERITY_MODERATE,
    SEVERITY_NONE,
    SEVERITY_ORDINAL,
    SEVERITY_SEVERE,
    extract_severity_laterality,
)


def _span(text: str, sub: str, label: str) -> dict:
    start = text.index(sub)
    return {"start": start, "end": start + len(sub), "label": label, "text": sub}


def _by_kind(attachments, kind):
    return [a for a in attachments if a["kind"] == kind]


# --------------------------------------------------------------------------
# Severity: descriptive / graded / numeric-pain scales
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cue", "expected"),
    [
        ("mild", SEVERITY_MILD),
        ("slight", SEVERITY_MILD),
        ("moderate", SEVERITY_MODERATE),
        ("severe", SEVERITY_SEVERE),
        ("marked", SEVERITY_SEVERE),
    ],
)
def test_descriptive_severity_normalizes_to_ordinal_bucket(cue, expected):
    text = f"Patient reports {cue} headache today."
    finding = _span(text, "headache", "CONDITION")

    attachments = extract_severity_laterality([finding], text)

    severities = _by_kind(attachments, "severity")
    assert len(severities) == 1
    hit = severities[0]
    assert hit["normalized"] == expected
    assert hit["ordinal"] == SEVERITY_ORDINAL[expected]
    assert hit["scale"] == "descriptive"
    assert hit["target_offset"] == (finding["start"], finding["end"])


@pytest.mark.parametrize(
    ("cue", "expected"),
    [
        ("grade 1", SEVERITY_MILD),
        ("grade II", SEVERITY_MODERATE),
        ("grade III", SEVERITY_SEVERE),
        ("grade IV", SEVERITY_SEVERE),
    ],
)
def test_graded_severity_maps_roman_and_arabic(cue, expected):
    text = f"Biopsy shows {cue} tumor in the specimen."
    finding = _span(text, "tumor", "CONDITION")

    attachments = extract_severity_laterality([finding], text)

    severities = _by_kind(attachments, "severity")
    assert len(severities) == 1
    assert severities[0]["normalized"] == expected
    assert severities[0]["scale"] == "graded"


@pytest.mark.parametrize(
    ("cue", "expected"),
    [
        ("0/10", SEVERITY_NONE),
        ("2/10", SEVERITY_MILD),
        ("5/10", SEVERITY_MODERATE),
        ("9/10", SEVERITY_SEVERE),
    ],
)
def test_numeric_pain_score_buckets(cue, expected):
    text = f"Chest pain rated {cue} on presentation."
    finding = _span(text, "Chest pain", "CONDITION")

    attachments = extract_severity_laterality([finding], text)

    severities = _by_kind(attachments, "severity")
    assert len(severities) == 1
    assert severities[0]["normalized"] == expected
    assert severities[0]["scale"] == "numeric_pain"
    assert severities[0]["ordinal"] == SEVERITY_ORDINAL[expected]


# --------------------------------------------------------------------------
# Laterality / position attachment to the nearest governing finding
# --------------------------------------------------------------------------


def test_laterality_attaches_to_nearest_finding_not_distant():
    text = "left knee effusion and right shoulder pain noted."
    knee = _span(text, "knee effusion", "CONDITION")
    shoulder = _span(text, "shoulder pain", "CONDITION")

    attachments = extract_severity_laterality([knee, shoulder], text)

    lat = {a["normalized"]: a for a in _by_kind(attachments, "laterality")}
    assert set(lat) == {LATERALITY_LEFT, LATERALITY_RIGHT}
    assert lat[LATERALITY_LEFT]["target_offset"] == (knee["start"], knee["end"])
    assert lat[LATERALITY_RIGHT]["target_offset"] == (
        shoulder["start"],
        shoulder["end"],
    )


@pytest.mark.parametrize(
    ("cue", "expected"),
    [
        ("bilateral", LATERALITY_BILATERAL),
        ("b/l", LATERALITY_BILATERAL),
        ("left-sided", LATERALITY_LEFT),
        ("right", LATERALITY_RIGHT),
        ("unilateral", LATERALITY_UNSPECIFIED),
    ],
)
def test_laterality_controlled_set(cue, expected):
    text = f"There is {cue} pneumonia on the chest film."
    finding = _span(text, "pneumonia", "CONDITION")

    attachments = extract_severity_laterality([finding], text)

    lat = _by_kind(attachments, "laterality")
    assert len(lat) == 1
    assert lat[0]["normalized"] == expected


def test_position_modifier_is_separate_axis():
    text = "Tenderness noted at the proximal femur fracture site."
    finding = _span(text, "femur fracture", "CONDITION")

    attachments = extract_severity_laterality([finding], text)

    pos = _by_kind(attachments, "position")
    assert len(pos) == 1
    assert pos[0]["normalized"] == "proximal"
    assert pos[0]["target_offset"] == (finding["start"], finding["end"])


# --------------------------------------------------------------------------
# Robustness: absent / distant modifiers must not misattach
# --------------------------------------------------------------------------


def test_modifier_beyond_scope_window_is_dropped():
    filler = "x" * 200
    text = f"severe {filler} headache"
    finding = _span(text, "headache", "CONDITION")

    attachments = extract_severity_laterality([finding], text, max_distance=40)

    assert attachments == []


def test_no_modifiers_returns_empty():
    text = "The headache resolved without intervention."
    finding = _span(text, "headache", "CONDITION")

    assert extract_severity_laterality([finding], text) == []


def test_output_contains_only_controlled_values_and_offsets():
    text = "severe left knee pain, 8/10."
    finding = _span(text, "knee pain", "CONDITION")

    attachments = extract_severity_laterality([finding], text)

    allowed_keys = {
        "kind",
        "text",
        "offset",
        "normalized",
        "ordinal",
        "scale",
        "target_offset",
        "target_label",
    }
    for a in attachments:
        assert set(a) == allowed_keys
        assert isinstance(a["offset"], tuple) and len(a["offset"]) == 2
        assert isinstance(a["target_offset"], tuple)
        # The governing finding's raw surface text is never echoed back.
        assert "knee pain" not in a["text"]


def test_advisory_is_exposed():
    assert isinstance(SEVERITY_LATERALITY_ADVISORY, str)
    assert SEVERITY_LATERALITY_ADVISORY
