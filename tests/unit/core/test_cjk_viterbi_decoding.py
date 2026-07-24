from __future__ import annotations

from itertools import accumulate

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from openmed.core.decoding import (
    CjkOffsetMap,
    assert_cjk_span_boundaries,
    build_label_info,
    is_grapheme_boundary,
    iter_grapheme_cluster_spans,
    labels_to_char_spans,
    labels_to_token_spans,
    snap_span_to_graphemes,
)
from openmed.processing.tokenization import SpanToken


def _word_tokens(parts: list[str]) -> tuple[str, list[SpanToken]]:
    text = "".join(parts)
    tokens: list[SpanToken] = []
    cursor = 0
    for part in parts:
        end = cursor + len(part)
        tokens.append(SpanToken(part, cursor, end))
        cursor = end
    return text, tokens


def test_han_subwords_decode_to_one_exact_segmenter_word() -> None:
    text, words = _word_tokens(["患者", "心房颤动", "入院"])
    phi_start = text.index("心房颤动")
    phi_end = phi_start + len("心房颤动")
    offsets = [(phi_start, phi_start + 2), (phi_start + 2, phi_end)]
    label_info = build_label_info({0: "O", 1: "S-CONDITION"})

    token_spans = labels_to_token_spans(
        {0: 1, 1: 1},
        label_info,
        token_offsets=offsets,
        text=text,
        language_hint="zh-CN",
        segmenter_word_tokens=words,
    )
    char_spans = labels_to_char_spans(
        {0: 1, 1: 1},
        label_info,
        offsets,
        text,
        language_hint="zh-CN",
        segmenter_word_tokens=words,
    )

    condition_label = label_info.span_label_lookup["CONDITION"]
    assert token_spans == [(condition_label, 0, 2)]
    assert char_spans == [(condition_label, phi_start, phi_end)]
    assert text[phi_start:phi_end] == "心房颤动"


def test_partial_han_subword_expands_to_the_complete_segmenter_word() -> None:
    text, words = _word_tokens(["患者", "心房颤动", "入院"])
    phi_start = text.index("心房颤动")
    phi_end = phi_start + len("心房颤动")
    offsets = [(phi_start, phi_start + 2), (phi_start + 2, phi_end)]
    label_info = build_label_info({0: "O", 1: "S-CONDITION"})

    spans = labels_to_char_spans(
        {0: 1, 1: 0},
        label_info,
        offsets,
        text,
        language_hint="zh",
        segmenter_word_tokens=words,
    )

    assert spans == [
        (
            label_info.span_label_lookup["CONDITION"],
            phi_start,
            phi_end,
        )
    ]
    assert text[slice(*spans[0][1:])] == "心房颤动"


def test_cjk_merges_contiguous_same_label_without_changing_latin_defaults() -> None:
    label_info = build_label_info({0: "O", 1: "S-NAME"})
    labels = {0: 1, 1: 1}

    assert labels_to_token_spans(labels, label_info) == [
        (label_info.span_label_lookup["NAME"], 0, 1),
        (label_info.span_label_lookup["NAME"], 1, 2),
    ]
    assert labels_to_char_spans(
        labels,
        label_info,
        [(0, 1), (1, 2)],
        "AB",
        language_hint="en",
    ) == [
        (label_info.span_label_lookup["NAME"], 0, 1),
        (label_info.span_label_lookup["NAME"], 1, 2),
    ]

    text, words = _word_tokens(["王", "芳"])
    assert labels_to_char_spans(
        labels,
        label_info,
        [(0, 1), (1, 2)],
        text,
        language_hint="zh",
        segmenter_word_tokens=words,
    ) == [(label_info.span_label_lookup["NAME"], 0, 2)]


def test_cjk_same_label_spans_do_not_merge_across_full_width_space() -> None:
    text = "王\u3000芳"
    words = [SpanToken("王", 0, 1), SpanToken("芳", 2, 3)]
    label_info = build_label_info({0: "O", 1: "S-NAME"})

    spans = labels_to_char_spans(
        {0: 1, 1: 1},
        label_info,
        [(0, 1), (2, 3)],
        text,
        language_hint="zh",
        segmenter_word_tokens=words,
    )

    name_label = label_info.span_label_lookup["NAME"]
    assert spans == [(name_label, 0, 1), (name_label, 2, 3)]


@pytest.mark.parametrize(
    "description",
    [
        "⿱⿰木木日",
        "⿲木水火",
        "⿼木丶",
        "⿾木",
        "㇯木一",
    ],
)
def test_ideographic_description_sequences_are_atomic_boundaries(
    description: str,
) -> None:
    text = f"{description}患者"
    description_end = len(description)

    assert list(iter_grapheme_cluster_spans(text))[0] == (0, description_end)
    assert snap_span_to_graphemes(1, 2, text) == (
        0,
        description_end,
    )
    for index in range(1, description_end):
        assert not is_grapheme_boundary(index, text)


def test_segmenter_boundary_invariant_rejects_partial_words() -> None:
    text = "心房颤动"
    offset_map = CjkOffsetMap(text, [SpanToken(text, 0, len(text))])

    with pytest.raises(AssertionError, match="segmenter word boundary"):
        assert_cjk_span_boundaries(1, len(text), text, offset_map)

    variation_text = "心\ufe00房"
    with pytest.raises(AssertionError, match="grapheme"):
        assert_cjk_span_boundaries(1, len(variation_text), variation_text)


_CJK_GRAPHEME_WORDS = st.sampled_from(
    [
        "患",
        "者",
        "心\ufe00",
        "房\u0301",
        "⿰木木",
        "⿾日",
    ]
)


@st.composite
def _synthetic_han_decode_case(draw):
    words = draw(st.lists(_CJK_GRAPHEME_WORDS, min_size=1, max_size=10))
    text, word_tokens = _word_tokens(words)
    raw_start = draw(st.integers(min_value=0, max_value=len(text) - 1))
    raw_end = draw(st.integers(min_value=raw_start + 1, max_value=len(text)))
    return text, words, word_tokens, raw_start, raw_end


@pytest.mark.fuzz
@settings(max_examples=2000, deadline=None)
@given(_synthetic_han_decode_case())
def test_synthetic_han_decodes_never_split_graphemes(case) -> None:
    text, words, word_tokens, raw_start, raw_end = case
    label_info = build_label_info({0: "O", 1: "S-PHI"})
    spans = labels_to_char_spans(
        {0: 1},
        label_info,
        [(raw_start, raw_end)],
        text,
        language_hint="zh",
        segmenter_word_tokens=word_tokens,
    )
    expected_boundaries = {0, *accumulate(map(len, words))}

    assert len(spans) == 1
    _, start, end = spans[0]
    assert 0 <= start < end <= len(text)
    assert start in expected_boundaries
    assert end in expected_boundaries
    assert is_grapheme_boundary(start, text)
    assert is_grapheme_boundary(end, text)
