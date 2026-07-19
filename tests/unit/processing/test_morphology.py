"""Precision and offset gates for conservative Indic morphology."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.processing.morphology import (
    INDIC_SUFFIX_TABLES,
    SUPPORTED_INDIC_MORPHOLOGY_LANGUAGES,
    grapheme_boundaries,
    split_sandhi,
    stem_token,
)

_FIXTURE_PATH = Path(__file__).parents[2] / "fixtures" / "indic_morphology.json"


@pytest.fixture(scope="module")
def morphology_fixture() -> dict[str, list[dict[str, str]]]:
    """Load the synthetic cross-script morphology fixture."""
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def test_fixture_covers_every_supported_language(morphology_fixture) -> None:
    assert {case["language"] for case in morphology_fixture["cases"]} == set(
        SUPPORTED_INDIC_MORPHOLOGY_LANGUAGES
    )


def test_allowlisted_stems_strip_without_losing_a_grapheme(
    morphology_fixture,
) -> None:
    for case in morphology_fixture["cases"]:
        surface = case["surface"]
        stem = case["stem"]
        result = stem_token(
            surface,
            case["language"],
            confidence=0.99,
            allowed_stems={stem},
        )

        assert result.applied
        assert result.stem == stem
        assert result.stem_span == (0, len(stem))
        assert result.stripped_suffixes == (case["suffix"],)
        assert result.offset_map.to_source_span(0, len(result.stem)) == (
            0,
            len(stem),
        )
        assert surface[slice(*result.stem_span)] == stem
        assert len(stem) in grapheme_boundaries(surface)


def test_sandhi_split_maps_rendered_parts_back_to_the_joined_source(
    morphology_fixture,
) -> None:
    for case in morphology_fixture["cases"]:
        surface = case["surface"]
        stem = case["stem"]
        source_suffix = case["suffix"]
        split_suffix = case.get("split_suffix", source_suffix)
        result = split_sandhi(
            surface,
            case["language"],
            confidence=0.99,
            allowed_stems={stem},
        )

        assert result.applied
        assert result.parts == (stem, split_suffix)
        assert result.text == f"{stem} {split_suffix}"
        assert result.offset_map.to_source_span(0, len(result.text)) == (
            0,
            len(surface),
        )
        assert surface[slice(*result.part_spans[0])] == stem
        assert surface[slice(*result.part_spans[1])] == source_suffix
        output_suffix_start = len(stem) + 1
        assert (
            result.offset_map.to_source_span(output_suffix_start, len(result.text))
            == result.part_spans[1]
        )
        if split_suffix != source_suffix:
            assert result.offset_map.to_source_span(
                output_suffix_start, output_suffix_start + 1
            ) == (len(stem), len(stem) + 2)


def test_stemming_and_sandhi_splitting_are_idempotent(morphology_fixture) -> None:
    for case in morphology_fixture["cases"]:
        language = case["language"]
        stem = case["stem"]
        first_stem = stem_token(
            case["surface"],
            language,
            confidence=0.99,
            allowed_stems={stem},
        )
        second_stem = stem_token(
            first_stem.stem,
            language,
            confidence=0.99,
            allowed_stems={stem},
        )
        assert not second_stem.applied
        assert second_stem.stem == first_stem.stem

        first_split = split_sandhi(
            case["surface"],
            language,
            confidence=0.99,
            allowed_stems={stem},
        )
        second_split = split_sandhi(
            first_split.text,
            language,
            confidence=0.99,
            allowed_stems={stem},
        )
        assert second_split.already_split
        assert second_split.text == first_split.text
        assert second_split.parts == first_split.parts


def test_held_out_suffix_like_names_are_never_over_stripped(
    morphology_fixture,
) -> None:
    all_fixture_stems = {case["stem"] for case in morphology_fixture["cases"]}
    for case in morphology_fixture["held_out_names"]:
        assert any(
            case["name"].endswith(suffix)
            for suffix in INDIC_SUFFIX_TABLES[case["language"]]
        )
        result = stem_token(
            case["name"],
            case["language"],
            confidence=1.0,
            allowed_stems=all_fixture_stems,
        )
        assert not result.applied
        assert result.stem == case["name"]
        assert result.stripped_suffix_spans == ()


def test_both_confidence_and_allowlist_gates_are_required() -> None:
    low_confidence = stem_token(
        "रामको",
        "hi",
        confidence=0.89,
        allowed_stems={"राम"},
    )
    missing_allowlist = stem_token(
        "रामको",
        "hi",
        confidence=0.99,
        allowed_stems=(),
    )

    assert not low_confidence.applied
    assert not missing_allowlist.applied


@pytest.mark.parametrize("language", ["bn", "pa", "or"])
def test_out_of_scope_languages_fail_closed(language: str) -> None:
    with pytest.raises(ValueError, match="unsupported morphology language"):
        stem_token("example", language, confidence=1.0, allowed_stems={"exam"})
