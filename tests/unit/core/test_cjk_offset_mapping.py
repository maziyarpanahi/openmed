from __future__ import annotations

import unicodedata

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from openmed.core.decoding import (
    CjkOffsetMap,
    project_word_spans_to_char_spans,
    refine_privacy_filter_span,
    snap_char_span_to_word_boundaries,
    trim_span_whitespace,
)
from openmed.processing.tokenization import (
    SpanToken,
    remap_predictions_to_chinese_words,
)


def _tokens_from_parts(parts: list[str]) -> tuple[str, list[SpanToken]]:
    text = "".join(parts)
    tokens: list[SpanToken] = []
    cursor = 0
    for part in parts:
        end = cursor + len(part)
        if part and not part.isspace():
            tokens.append(SpanToken(part, cursor, end))
        cursor = end
    return text, tokens


def test_name_projection_covers_exact_chinese_code_points() -> None:
    text, tokens = _tokens_from_parts(["患者", "王芳", "入院"])
    offset_map = CjkOffsetMap(text, tokens)

    assert project_word_spans_to_char_spans([(1, 2)], offset_map) == [(2, 4)]
    start, end = project_word_spans_to_char_spans([(1, 2)], offset_map)[0]
    assert text[start:end] == "王芳"
    assert (start, end) == (text.index("王芳"), text.index("王芳") + 2)


def test_partial_han_prediction_snaps_to_whole_word() -> None:
    text, tokens = _tokens_from_parts(["患者", "心房颤动", "入院"])
    offset_map = CjkOffsetMap(text, tokens)
    partial_start = text.index("心房")
    partial_end = partial_start + len("心房")

    assert snap_char_span_to_word_boundaries(
        partial_start,
        partial_end,
        offset_map,
    ) == (text.index("心房颤动"), text.index("心房颤动") + len("心房颤动"))


def test_multibyte_name_offsets_follow_python_slicing_not_utf8_bytes() -> None:
    text, tokens = _tokens_from_parts(["李雷", " ", "said", " ", "hello"])
    offset_map = CjkOffsetMap(text, tokens)

    assert offset_map.word_to_char[0] == (0, 2)
    assert project_word_spans_to_char_spans([(0, 1)], offset_map) == [(0, 2)]
    assert text[slice(*offset_map.word_to_char[0])] == "李雷"
    assert len("李雷".encode("utf-8")) == 6


def test_full_width_space_is_a_gap_not_a_redactable_word() -> None:
    text, tokens = _tokens_from_parts(["王芳", "\u3000", "心房颤动"])
    offset_map = CjkOffsetMap(text, tokens)

    assert offset_map.char_to_word[2] is None
    assert snap_char_span_to_word_boundaries(2, 3, offset_map) == (2, 3)
    assert trim_span_whitespace(0, len(text), text) == (0, len(text))
    assert refine_privacy_filter_span(
        "private_person",
        0,
        len(text),
        text,
    ) == (0, len(text))


def test_unicode_edge_spaces_trim_without_dropping_joined_han_text() -> None:
    text = "\u3000王\u200d芳\u3000"

    assert trim_span_whitespace(0, len(text), text) == (1, len(text) - 1)
    assert text[slice(*trim_span_whitespace(0, len(text), text))] == "王\u200d芳"


def test_chinese_remap_expands_partial_subword_prediction() -> None:
    text, tokens = _tokens_from_parts(["患者", "心房颤动", "入院"])
    start = text.index("心房")
    predictions = [
        {
            "start": start,
            "end": start + len("心房"),
            "entity": "B-CONDITION",
            "score": 0.95,
        }
    ]

    remapped = remap_predictions_to_chinese_words(predictions, text, tokens)

    assert remapped == [
        {
            "start": text.index("心房颤动"),
            "end": text.index("心房颤动") + len("心房颤动"),
            "score": 0.95,
            "entity_group": "CONDITION",
            "word": "心房颤动",
            "metadata": {},
        }
    ]


def test_chinese_remap_does_not_merge_across_full_width_space() -> None:
    text, tokens = _tokens_from_parts(["王芳", "\u3000", "李雷"])
    predictions = [
        {"start": 0, "end": 2, "entity_group": "NAME", "score": 0.9},
        {"start": 3, "end": 5, "entity_group": "NAME", "score": 0.8},
    ]

    remapped = remap_predictions_to_chinese_words(predictions, text, tokens)

    assert [(item["word"], item["start"], item["end"]) for item in remapped] == [
        ("王芳", 0, 2),
        ("李雷", 3, 5),
    ]


def test_offset_map_rejects_non_nfc_source_and_invalid_tokens() -> None:
    decomposed = "e\u0301患者"
    assert unicodedata.normalize("NFC", decomposed) != decomposed
    with pytest.raises(ValueError, match="NFC-normalized"):
        CjkOffsetMap(decomposed, [SpanToken(decomposed, 0, len(decomposed))])

    with pytest.raises(ValueError, match="does not match"):
        CjkOffsetMap("王芳", [SpanToken("李雷", 0, 2)])


_TOKEN_PARTS = st.sampled_from(
    [
        "患者",
        "王芳",
        "李雷",
        "心房颤动",
        "入院",
        "A",
        "Latin",
        "B2",
        "Ａ",
        "１２",
    ]
)
_SPACE_PARTS = st.sampled_from([" ", "\u3000"])


@pytest.mark.fuzz
@settings(max_examples=2000, deadline=None)
@given(
    st.lists(
        st.one_of(_TOKEN_PARTS, _SPACE_PARTS),
        min_size=1,
        max_size=20,
    )
)
def test_mixed_code_point_word_round_trips_are_exact(parts: list[str]) -> None:
    text, tokens = _tokens_from_parts(parts)
    offset_map = CjkOffsetMap(text, tokens)

    for char_index, word_index in enumerate(offset_map.char_to_word):
        if word_index is None:
            assert text[char_index].isspace()
            continue
        word_position = offset_map.word_position_for_char(char_index)
        assert word_position is not None
        assert offset_map.char_for_word_position(*word_position) == char_index
        word_start, word_end = offset_map.char_span_for_word(word_index)
        assert text[word_start:word_end] == tokens[word_index].text
