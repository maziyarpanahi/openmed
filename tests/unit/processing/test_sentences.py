from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from openmed.processing import SentenceSpan, segment_chinese_text, segment_text


def _assert_exact_round_trip(text: str, spans: list[SentenceSpan]) -> None:
    assert all(text[span.start : span.end] == span.text for span in spans)
    assert "".join(text[span.start : span.end] for span in spans) == text
    assert all(left.end == right.start for left, right in zip(spans, spans[1:]))


def test_chinese_three_sentence_note_has_exact_offsets():
    text = "患者发热。血压稳定！是否复诊？"

    spans = segment_text(text, language="zh")

    assert [span.text for span in spans] == ["患者发热。", "血压稳定！", "是否复诊？"]
    _assert_exact_round_trip(text, spans)


def test_chinese_nested_quotes_defer_boundary_until_outer_quote_closes():
    text = "医生记录：「患者说『胸痛缓解！』，生命体征稳定。」次日复诊。"

    spans = segment_chinese_text(text)

    assert [span.text for span in spans] == [
        "医生记录：「患者说『胸痛缓解！』，生命体征稳定。」",
        "次日复诊。",
    ]
    _assert_exact_round_trip(text, spans)


def test_chinese_book_title_period_does_not_split_sentence():
    text = "依据《临床指南。第二版》调整用药。患者情况稳定！"

    spans = segment_chinese_text(text)

    assert [span.text for span in spans] == [
        "依据《临床指南。第二版》调整用药。",
        "患者情况稳定！",
    ]
    _assert_exact_round_trip(text, spans)


def test_chinese_closing_quote_stays_with_preceding_sentence():
    text = "医生说：「情况稳定。」患者可以出院。"

    spans = segment_chinese_text(text)

    assert spans[0].text == "医生说：「情况稳定。」"
    assert len(spans) == 2
    _assert_exact_round_trip(text, spans)


@pytest.mark.parametrize(
    "text",
    [
        "医嘱（立即复诊！）患者知情。",
        "记录〔血压正常；〕继续观察。",
        "依据《临床指南。》调整用药。",
    ],
)
def test_non_quote_bracket_punctuation_does_not_end_outer_sentence(text: str):
    spans = segment_chinese_text(text)

    assert [span.text for span in spans] == [text]
    _assert_exact_round_trip(text, spans)


def test_closing_quote_followed_by_comma_continues_the_outer_sentence():
    text = "医生说：「情况稳定！」，建议明日出院。患者知情。"

    spans = segment_chinese_text(text)

    assert [span.text for span in spans] == [
        "医生说：「情况稳定！」，建议明日出院。",
        "患者知情。",
    ]
    _assert_exact_round_trip(text, spans)


def test_chinese_fullwidth_decimal_does_not_split():
    text = "剂量为1．5毫克。复诊安排！"

    spans = segment_chinese_text(text)

    assert [span.text for span in spans] == ["剂量为1．5毫克。", "复诊安排！"]
    _assert_exact_round_trip(text, spans)


def test_chinese_fullwidth_abbreviation_and_ascii_terminator():
    text = "Dr．Wang记录剂量。情况稳定!"

    spans = segment_chinese_text(text)

    assert [span.text for span in spans] == ["Dr．Wang记录剂量。", "情况稳定!"]
    _assert_exact_round_trip(text, spans)


def test_chinese_spans_preserve_inter_sentence_and_trailing_whitespace():
    text = "患者稳定。  血压正常！\n"

    spans = segment_chinese_text(text)

    assert [span.text for span in spans] == ["患者稳定。  ", "血压正常！\n"]
    _assert_exact_round_trip(text, spans)


def test_han_dominant_text_uses_chinese_segmenter_automatically():
    text = "患者稳定。次日复诊！"

    with patch("openmed.processing.sentences._get_segmenter") as get_segmenter:
        spans = segment_text(text)

    get_segmenter.assert_not_called()
    assert [span.text for span in spans] == ["患者稳定。", "次日复诊！"]


def test_non_chinese_language_keeps_pysbd_path():
    text = "Patient is stable. Follow up tomorrow."
    sentence_objects = [
        SimpleNamespace(sent="Patient is stable. ", start=0, end=19),
        SimpleNamespace(sent="Follow up tomorrow.", start=19, end=len(text)),
    ]
    pysbd_segmenter = Mock()
    pysbd_segmenter.segment.return_value = sentence_objects

    spans = segment_text(text, language="en", segmenter=pysbd_segmenter)

    pysbd_segmenter.segment.assert_called_once_with(text)
    assert spans == [
        SentenceSpan("Patient is stable. ", 0, 19),
        SentenceSpan("Follow up tomorrow.", 19, len(text)),
    ]


def test_explicit_segmenter_override_is_preserved_for_han_text():
    text = "患者稳定。次日复诊！"
    sentence_object = SimpleNamespace(sent=text, start=0, end=len(text))
    explicit_segmenter = Mock()
    explicit_segmenter.segment.return_value = [sentence_object]

    spans = segment_text(text, language="zh", segmenter=explicit_segmenter)

    explicit_segmenter.segment.assert_called_once_with(text)
    assert spans == [SentenceSpan(text, 0, len(text))]
