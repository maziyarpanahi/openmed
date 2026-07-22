from itertools import accumulate

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from openmed.core.decoding import (
    iter_grapheme_cluster_spans,
    refine_privacy_filter_span,
    snap_span_to_grapheme_boundaries,
    trim_span_whitespace,
)
from openmed.core.decoding import spans as grapheme_spans

_SYNTHETIC_CLUSTERS = (
    "क्षि",  # Devanagari consonant + virama + consonant + dependent vowel
    "क्\u200dष",  # Devanagari conjunct with an explicit ZWJ
    "कि",  # Devanagari spacing-mark vowel
    "ক্ষি",  # Bengali conjunct
    "ਕ੍ਸ਼ਿ",  # Gurmukhi conjunct
    "ક્ષિ",  # Gujarati conjunct
    "କ୍ଷି",  # Odia conjunct
    "க்ஷி",  # Tamil conjunct
    "త్రా",  # Telugu consonant + virama + consonant + dependent vowel
    "తా",  # Telugu dependent vowel
    "ಕ್ಷಿ",  # Kannada conjunct
    "ക്ഷി",  # Malayalam conjunct
    "患",
    "者",
    "か",
    "한",
    "각",  # decomposed Hangul L + V + T
    "👩\u200d⚕️",
    "👨\u200d👩\u200d👧\u200d👦",
    "👍🏽",
    "🇫🇷",
    " ",
    " \u0301",  # whitespace plus a combining mark is one cluster
)


@st.composite
def _synthetic_multiscript_spans(draw):
    clusters = draw(
        st.lists(
            st.sampled_from(_SYNTHETIC_CLUSTERS),
            min_size=1,
            max_size=10,
        )
    )
    text = "".join(clusters)
    start = draw(st.integers(min_value=0, max_value=len(text)))
    end = draw(st.integers(min_value=start, max_value=len(text)))
    return clusters, text, start, end


@pytest.mark.fuzz
@settings(max_examples=600, deadline=None)
@given(_synthetic_multiscript_spans())
def test_generated_multiscript_spans_never_split_graphemes(case):
    clusters, text, raw_start, raw_end = case
    expected_ends = list(accumulate(len(cluster) for cluster in clusters))
    expected_boundaries = {0, *expected_ends}
    expected_spans = list(zip([0, *expected_ends[:-1]], expected_ends, strict=True))

    assert list(iter_grapheme_cluster_spans(text)) == expected_spans

    snapped = snap_span_to_grapheme_boundaries(raw_start, raw_end, text)
    trimmed = trim_span_whitespace(raw_start, raw_end, text)
    refined = refine_privacy_filter_span(
        "private_person",
        raw_start,
        raw_end,
        text,
    )

    for start, end in (snapped, trimmed, refined):
        assert 0 <= start <= end <= len(text)
        assert start in expected_boundaries
        assert end in expected_boundaries
        assert text[start:end] == text[slice(start, end)]

    assert trim_span_whitespace(*trimmed, text) == trimmed
    assert refine_privacy_filter_span("private_person", *refined, text) == refined


@pytest.mark.parametrize(
    ("label", "text", "expected_text"),
    [
        ("private_email", " alice@example.test and ", "alice@example.test"),
        ("private_url", " https://example.test/path, ", "https://example.test/path"),
        ("private_phone", " +1 (415) 555-0123 and ", "+1 (415) 555-0123"),
        ("private_person", " Jane Doe and ", "Jane Doe"),
        ("private_person", " Jane Doe or ", "Jane Doe"),
    ],
)
def test_latin_refinement_regression_bytes_are_unchanged(label, text, expected_text):
    start, end = refine_privacy_filter_span(label, 0, len(text), text)

    assert (start, end) == (
        text.index(expected_text),
        text.index(expected_text) + len(expected_text),
    )
    assert text[start:end].encode("utf-8") == expected_text.encode("utf-8")


@pytest.mark.parametrize("prefix", ["患者", "रवि"])
def test_script_adjacency_refines_latin_email_without_word_spacing(prefix):
    text = f"{prefix}alice@example.test連絡"

    start, end = refine_privacy_filter_span("private_email", 0, len(text), text)

    assert (start, end) == (len(prefix), len(prefix) + len("alice@example.test"))
    assert text[start:end] == "alice@example.test"


def test_cjk_adjacency_does_not_leak_into_latin_url():
    text = "患者 https://example.test/path連絡"

    start, end = refine_privacy_filter_span("private_url", 0, len(text), text)

    assert text[start:end] == "https://example.test/path"


def test_cjk_span_does_not_use_latin_glue_word_heuristic():
    text = "患者 and"

    assert refine_privacy_filter_span("private_person", 0, len(text), text) == (
        0,
        len(text),
    )


def test_cross_script_fixture_records_broken_codepoint_trimmer_baseline():
    text = "患者क्\u200dष👩\u200d⚕️"
    raw_start = text.index("ष")
    raw_end = text.index("⚕")

    legacy_baseline = _legacy_codepoint_whitespace_trim(raw_start, raw_end, text)
    boundaries = {
        0,
        *(end for _, end in iter_grapheme_cluster_spans(text)),
    }

    assert legacy_baseline == (5, 8)
    assert legacy_baseline[0] not in boundaries
    assert legacy_baseline[1] not in boundaries

    expected = (2, len(text))
    assert trim_span_whitespace(raw_start, raw_end, text) == expected
    assert (
        refine_privacy_filter_span(
            "private_person",
            raw_start,
            raw_end,
            text,
        )
        == expected
    )
    assert text[slice(*expected)] == "क्\u200dष👩\u200d⚕️"


def test_grapheme_iterator_handles_core_uax29_sequences():
    text = "\r\n🇫🇷👩\u200d⚕️각"
    clusters = [text[start:end] for start, end in iter_grapheme_cluster_spans(text)]

    assert clusters == ["\r\n", "🇫🇷", "👩\u200d⚕️", "각"]


@pytest.mark.parametrize(
    "text",
    ["क्षि", "ক্ষি", "ਕ੍ਸ਼ਿ", "ક્ષિ", "କ୍ଷି", "க்ஷி", "క్షి", "ಕ್ಷಿ", "ക്ഷി"],
)
def test_grapheme_iterator_keeps_supported_indic_conjuncts_whole(text):
    assert list(iter_grapheme_cluster_spans(text)) == [(0, len(text))]


def test_span_snapping_only_inspects_clusters_touching_boundaries(monkeypatch):
    text = ("Patient record 🧬 é ‍ 北京 مستشفى. " * 256) + "Zzyxx Qwerty"
    start = text.index("Zzyxx")
    calls: list[int] = []
    original_has_grapheme_break = grapheme_spans._has_grapheme_break

    def counted_has_grapheme_break(value: str, index: int) -> bool:
        calls.append(index)
        return original_has_grapheme_break(value, index)

    monkeypatch.setattr(
        grapheme_spans,
        "_has_grapheme_break",
        counted_has_grapheme_break,
    )

    assert snap_span_to_grapheme_boundaries(start, len(text), text) == (
        start,
        len(text),
    )
    assert len(calls) < 10


def _legacy_codepoint_whitespace_trim(
    start: int,
    end: int,
    text: str,
) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end
