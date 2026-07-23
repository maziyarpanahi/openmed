from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import pytest

from openmed.core.decoding import (
    build_label_info,
    is_grapheme_boundary,
    is_indic_text,
    iter_grapheme_clusters,
    labels_to_char_spans,
    refine_privacy_filter_span,
    snap_span_to_graphemes,
    trim_span_whitespace,
)

FIXTURE_PATH = Path(__file__).parents[2] / "fixtures" / "indic_grapheme_spans.jsonl"


def _fixture_rows() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.mark.parametrize("row", _fixture_rows(), ids=lambda row: str(row["script"]))
def test_fixture_spans_snap_to_complete_indic_aksharas(
    row: dict[str, object],
) -> None:
    text = str(row["text"])
    expected_start = int(row["expected_start"])
    expected_end = int(row["expected_end"])

    start, end = snap_span_to_graphemes(
        int(row["model_start"]),
        int(row["model_end"]),
        text,
    )

    assert (start, end) == (expected_start, expected_end)
    assert text[start:end] == row["phi_text"]
    assert is_grapheme_boundary(start, text)
    assert is_grapheme_boundary(end, text)


@pytest.mark.parametrize("row", _fixture_rows(), ids=lambda row: str(row["script"]))
def test_fixture_redaction_leaves_no_orphaned_indic_marks(
    row: dict[str, object],
) -> None:
    text = str(row["text"])
    start, end = snap_span_to_graphemes(
        int(row["model_start"]),
        int(row["model_end"]),
        text,
    )
    redacted = text[:start] + "[NAME]" + text[end:]

    for index, char in enumerate(redacted):
        if not unicodedata.category(char).startswith("M"):
            continue
        assert index > 0
        assert is_indic_text(redacted[index - 1])


@pytest.mark.parametrize("row", _fixture_rows(), ids=lambda row: str(row["script"]))
def test_fixture_round_trip_preserves_non_phi_aksharas(
    row: dict[str, object],
) -> None:
    text = str(row["text"])
    expected_start = int(row["expected_start"])
    expected_end = int(row["expected_end"])
    start, end = snap_span_to_graphemes(
        int(row["model_start"]),
        int(row["model_end"]),
        text,
    )
    redacted = text[:start] + "[NAME]" + text[end:]

    assert redacted[:start] == text[:expected_start]
    assert redacted[start + len("[NAME]") :] == text[expected_end:]
    assert is_grapheme_boundary(expected_start, text)
    assert is_grapheme_boundary(expected_end, text)


@pytest.mark.parametrize(
    "akshara",
    [
        "र्क्षा",  # Reph followed by a conjunct and dependent vowel.
        "र्\u200dका",  # Explicit joiner after the Reph virama.
        "ক্ষ",  # Bengali conjunct.
        "க்ஷ",  # Tamil conjunct.
        "ക്ഷ",  # Malayalam conjunct.
    ],
)
def test_iterator_keeps_reph_and_consonant_conjuncts_together(akshara: str) -> None:
    assert list(iter_grapheme_clusters(akshara)) == [(0, len(akshara))]


def test_iterator_covers_core_extended_grapheme_rules() -> None:
    clusters = [
        text[start:end]
        for text in ("a\u0301", "\r\n", "각", "👩🏽\u200d⚕️", "🇮🇳")
        for start, end in iter_grapheme_clusters(text)
    ]

    assert clusters == ["á", "\r\n", "각", "👩🏽\u200d⚕️", "🇮🇳"]


def test_whitespace_trimming_does_not_split_ksha_conjunct() -> None:
    text = "  नाम क्ष   "
    conjunct_start = text.index("क्ष")
    start, end = trim_span_whitespace(0, conjunct_start + 2, text)

    assert text[start:end] == "नाम क्ष"
    assert is_grapheme_boundary(start, text)
    assert is_grapheme_boundary(end, text)


def test_privacy_refinement_returns_grapheme_boundaries() -> None:
    text = "  रोगी क्षमा and  "
    model_start = text.index("क्ष") + 1
    start, end = refine_privacy_filter_span("NAME", model_start, len(text), text)

    assert text[start:end] == "क्षमा"
    assert is_grapheme_boundary(start, text)
    assert is_grapheme_boundary(end, text)


def test_viterbi_character_span_emission_snaps_indic_offsets() -> None:
    text = "रोगी क्षमा आई।"
    phi_start = text.index("क्षमा")
    phi_end = phi_start + len("क्षमा")
    label_info = build_label_info({0: "O", 1: "S-NAME"})

    spans = labels_to_char_spans(
        {0: 1},
        label_info,
        [(phi_start + 1, phi_end - 1)],
        text,
    )

    assert spans == [(label_info.span_label_lookup["NAME"], phi_start, phi_end)]
