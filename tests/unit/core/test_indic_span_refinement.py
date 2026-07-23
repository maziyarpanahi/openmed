"""Boundary-refinement acceptance gates for Indic inflected names."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.core.decoding.spans import (
    refine_indic_name_span,
    refine_privacy_filter_span,
)
from openmed.core.labels import supports_name_boundary_refinement
from openmed.processing.morphology import grapheme_boundaries

_FIXTURE_PATH = Path(__file__).parents[2] / "fixtures" / "indic_morphology.json"


@pytest.fixture(scope="module")
def morphology_fixture() -> dict[str, list[dict[str, str]]]:
    """Load the synthetic cross-script morphology fixture."""
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def test_refinement_tightens_fixture_spans_and_reconstructs_source_exactly(
    morphology_fixture,
) -> None:
    for case in morphology_fixture["cases"]:
        text = case["context"]
        original_start = text.index(case["surface"])
        original_end = original_start + len(case["surface"])
        result = refine_indic_name_span(
            "B-NAME",
            original_start,
            original_end,
            text,
            enabled=True,
            language=case["language"],
            confidence=0.99,
            allowed_stems={case["stem"]},
        )

        assert result.applied
        assert text[result.start : result.end] == case["stem"]
        assert result.start == original_start
        assert result.end < original_end
        assert result.start in grapheme_boundaries(text)
        assert result.end in grapheme_boundaries(text)
        assert result.to_source_span(0, result.end - result.start) == (
            result.start,
            result.end,
        )
        assert (
            "".join(text[start:end] for start, end in result.grapheme_origins)
            == (case["stem"])
        )


def test_privacy_filter_hook_is_opt_in_and_disabled_path_is_identical(
    morphology_fixture,
) -> None:
    baselines: list[tuple[int, int]] = []
    explicit_disabled: list[tuple[int, int]] = []
    for case in morphology_fixture["cases"]:
        text = case["context"]
        start = text.index(case["surface"])
        end = start + len(case["surface"])
        baselines.append(refine_privacy_filter_span("NAME", start, end, text))
        explicit_disabled.append(
            refine_privacy_filter_span(
                "NAME",
                start,
                end,
                text,
                indic_morphology=False,
                language=case["language"],
                confidence=0.99,
                morphology_allowlist={case["stem"]},
            )
        )

    regression_samples = [
        ("EMAIL", "alice@hospital.org and another"),
        ("PHONE", "+1 212-555-0100"),
        ("NAME", "  Alice and "),
    ]
    for label, text in regression_samples:
        start, end = 0, len(text)
        baselines.append(refine_privacy_filter_span(label, start, end, text))
        explicit_disabled.append(
            refine_privacy_filter_span(
                label,
                start,
                end,
                text,
                indic_morphology=False,
            )
        )

    assert json.dumps(baselines).encode() == json.dumps(explicit_disabled).encode()


def test_privacy_filter_hook_refines_when_explicitly_enabled() -> None:
    text = "रोगी रामको आज देखा गया।"
    start = text.index("रामको")
    end = start + len("रामको")

    refined = refine_privacy_filter_span(
        "PERSON",
        start,
        end,
        text,
        indic_morphology=True,
        language="hi",
        confidence=0.99,
        morphology_allowlist={"राम"},
    )

    assert text[slice(*refined)] == "राम"


def test_non_name_labels_low_confidence_and_missing_allowlist_are_noops() -> None:
    text = "रामको"
    original = (0, len(text))

    non_name = refine_indic_name_span(
        "LOCATION",
        *original,
        text,
        enabled=True,
        language="hi",
        confidence=1.0,
        allowed_stems={"राम"},
    )
    assert (non_name.start, non_name.end) == original

    for confidence, allowlist in ((0.5, {"राम"}), (1.0, set())):
        result = refine_indic_name_span(
            "PERSON",
            *original,
            text,
            enabled=True,
            language="hi",
            confidence=confidence,
            allowed_stems=allowlist,
        )
        assert not result.applied
        assert (result.start, result.end) == original


def test_model_span_inside_a_grapheme_is_rejected() -> None:
    text = "रामको"
    result = refine_indic_name_span(
        "PERSON",
        1,
        len(text),
        text,
        enabled=True,
        language="hi",
        confidence=1.0,
        allowed_stems={"ाम"},
    )

    assert not result.applied
    assert result.reason == "unaligned_source_span"
    assert (result.start, result.end) == (1, len(text))


@pytest.mark.parametrize(
    "label,expected",
    [
        ("PERSON", True),
        ("B-NAME", True),
        ("FIRST_NAME", True),
        ("PREFIX", False),
        ("USERNAME", False),
        ("LOCATION", False),
    ],
)
def test_label_allowlist_is_restricted_to_person_names(
    label: str, expected: bool
) -> None:
    assert supports_name_boundary_refinement(label) is expected
