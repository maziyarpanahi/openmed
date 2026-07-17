import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from openmed.core.decoding.spans import (
    is_grapheme_boundary,
    iter_grapheme_clusters,
)
from openmed.processing.sentences import SentenceSpan, segment_indic_text, segment_text
from openmed.processing.tokenization import indic_word_tokenize, medical_tokenize


def _fixture() -> dict[str, object]:
    path = Path(__file__).parents[2] / "fixtures" / "indic_sentence_tokenization.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_exact_spans(text: str, spans: list[SentenceSpan]) -> None:
    assert "".join(span.text for span in spans) == text
    assert all(text[span.start : span.end] == span.text for span in spans)
    assert all(left.end == right.start for left, right in zip(spans, spans[1:]))
    assert all(is_grapheme_boundary(span.start, text) for span in spans)
    assert all(is_grapheme_boundary(span.end, text) for span in spans)


def test_indic_fixture_splits_danda_and_double_danda_without_honorific_split():
    fixture = _fixture()["sentences"]
    assert isinstance(fixture, dict)
    text = fixture["text"]
    expected = fixture["expected"]
    assert isinstance(text, str)
    assert isinstance(expected, list)

    with patch("openmed.processing.sentences._get_segmenter") as get_segmenter:
        spans = segment_text(text)

    get_segmenter.assert_not_called()
    assert [span.text for span in spans] == expected
    assert spans[0].text.startswith("डॉ. सीमा")
    _assert_exact_spans(text, spans)


def test_indic_segmentation_is_idempotent():
    fixture = _fixture()["sentences"]
    assert isinstance(fixture, dict)
    text = fixture["text"]
    assert isinstance(text, str)

    for span in segment_indic_text(text):
        assert segment_indic_text(span.text) == [
            SentenceSpan(span.text, 0, len(span.text))
        ]


def test_honorific_using_danda_does_not_create_a_false_boundary():
    text = "डॉ। सीमा ने जाँच की। अगला वाक्य॥"

    spans = segment_indic_text(text)

    assert [span.text for span in spans] == [
        "डॉ। सीमा ने जाँच की। ",
        "अगला वाक्य॥",
    ]
    _assert_exact_spans(text, spans)


def test_mixed_latin_telugu_and_devanagari_digits_keep_exact_offsets():
    fixture = _fixture()["mixed"]
    assert isinstance(fixture, dict)
    text = fixture["text"]
    expected = fixture["expected"]
    assert isinstance(text, str)
    assert isinstance(expected, list)

    spans = segment_text(text)

    assert [span.text for span in spans] == expected
    _assert_exact_spans(text, spans)


def test_indic_word_tokenizer_preserves_date_identifier_and_conjunct():
    fixture = _fixture()["words"]
    assert isinstance(fixture, dict)
    text = fixture["text"]
    assert isinstance(text, str)

    tokens = indic_word_tokenize(text)
    token_texts = [token.text for token in tokens]

    assert fixture["date"] in token_texts
    assert fixture["identifier"] in token_texts
    assert token_texts.count(fixture["conjunct_word"]) == 1
    for token in tokens:
        assert text[token.start : token.end] == token.text
        assert is_grapheme_boundary(token.start, text)
        assert is_grapheme_boundary(token.end, text)

    conjunct = str(fixture["conjunct_word"])
    conjunct_clusters = [
        conjunct[start:end] for start, end in iter_grapheme_clusters(conjunct)
    ]
    assert conjunct_clusters[0].startswith("क्ष")


def test_medical_tokenizer_routes_indic_runs_and_keeps_latin_exception_shape():
    text = "COVID-19 के बाद क्षेत्र जाँच १२/०५/२०२६।"

    tokens = medical_tokenize(text)
    token_texts = [token.text for token in tokens]

    assert "COVID-19" in token_texts
    assert "क्षेत्र" in token_texts
    assert "१२/०५/२०२६" in token_texts
    assert all(is_grapheme_boundary(token.start, text) for token in tokens)
    assert all(is_grapheme_boundary(token.end, text) for token in tokens)


def test_latin_only_text_stays_on_pysbd_path():
    text = "Patient is stable. Follow up tomorrow."
    sentence_objects = [
        SimpleNamespace(sent="Patient is stable. ", start=0, end=19),
        SimpleNamespace(sent="Follow up tomorrow.", start=19, end=len(text)),
    ]
    pysbd_segmenter = Mock()
    pysbd_segmenter.segment.return_value = sentence_objects

    with patch(
        "openmed.processing.sentences._get_segmenter",
        return_value=pysbd_segmenter,
    ) as get_segmenter:
        spans = segment_text(text)

    get_segmenter.assert_called_once()
    pysbd_segmenter.segment.assert_called_once_with(text)
    assert [span.text for span in spans] == [
        "Patient is stable. ",
        "Follow up tomorrow.",
    ]
