import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from openmed.processing import (
    SentenceSpan,
    parse_lists,
    segment_chinese_text,
    segment_clinical_text,
    segment_text,
    sentences,
)


@pytest.fixture(autouse=True)
def clear_segmenter_cache():
    sentences._SEGMENTER_CACHE.clear()
    yield
    sentences._SEGMENTER_CACHE.clear()


@pytest.fixture
def fake_yasbd_segmenter():
    detector_cls = Mock(name="BoundaryDetector")

    fake_yasbd = types.ModuleType("yasbd")
    fake_yasbd.BoundaryDetector = detector_cls

    fake_modules = {
        "yasbd": fake_yasbd,
    }
    with patch.dict(sys.modules, fake_modules):
        yield detector_cls


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


def test_top_level_list_items_are_single_segmentation_units() -> None:
    text = (
        "- Metformin 500 mg.\n  a) Take twice daily.\n- Lisinopril 10 mg. Take daily."
    )
    segmenter = Mock()

    spans = segment_text(text, segmenter=segmenter, list_items=parse_lists(text))

    assert [span.text for span in spans] == [
        "- Metformin 500 mg.\n  a) Take twice daily.\n",
        "- Lisinopril 10 mg. Take daily.",
    ]
    segmenter.segment.assert_not_called()
    _assert_exact_round_trip(text, spans)


def test_clinical_segmentation_scopes_list_parsing_to_medication_section() -> None:
    text = (
        "HPI: Synthetic cough. It is improving.\n"
        "MEDICATIONS:\n"
        "Metformin 500 mg. Take twice daily.\n"
        "Lisinopril 10 mg. Take daily.\n"
        "PLAN: Follow up. Continue care."
    )

    class WholeRegionSegmenter:
        def segment(self, region: str):
            return [SimpleNamespace(sent=region, start=0, end=len(region))]

    spans = segment_clinical_text(text, segmenter=WholeRegionSegmenter())
    medication_spans = [
        span for span in spans if "Metformin" in span.text or "Lisinopril" in span.text
    ]

    assert len(medication_spans) == 2
    assert "Lisinopril" not in medication_spans[0].text
    assert "Metformin" not in medication_spans[1].text
    assert all(
        "Take" in span.text and span.text.count("Take") == 1
        for span in medication_spans
    )
    _assert_exact_round_trip(text, spans)


def test_explicit_segmenter_override_is_preserved_for_han_text():
    text = "患者稳定。次日复诊！"
    sentence_object = SimpleNamespace(sent=text, start=0, end=len(text))
    explicit_segmenter = Mock()
    explicit_segmenter.segment.return_value = [sentence_object]

    spans = segment_text(text, language="zh", segmenter=explicit_segmenter)

    explicit_segmenter.segment.assert_called_once_with(text)
    assert spans == [SentenceSpan(text, 0, len(text))]


def test_explicit_segmenter_override_is_preserved_for_indic_text():
    text = "रोगी स्थिर है। कल समीक्षा करें।"
    sentence_object = SimpleNamespace(sent=text, start=0, end=len(text))
    explicit_segmenter = Mock()
    explicit_segmenter.segment.return_value = [sentence_object]

    spans = segment_text(text, language="hi", segmenter=explicit_segmenter)

    explicit_segmenter.segment.assert_called_once_with(text)
    assert spans == [SentenceSpan(text, 0, len(text))]


@pytest.mark.parametrize("backend", ["fast", "", None])
@pytest.mark.parametrize("text", ["", "Patient is stable."])
def test_unknown_backend_raises_value_error(backend, text):
    with pytest.raises(ValueError, match="Unknown segmentation backend"):
        segment_text(text, backend=backend)


def test_preconstructed_segmenter_with_yasbd_backend_raises():
    with pytest.raises(ValueError, match="cannot be combined"):
        segment_text("Patient is stable.", segmenter=Mock(), backend="yasbd")


def test_yasbd_backend_routes_through_yasbd_adapter(fake_yasbd_segmenter):
    detector_cls = fake_yasbd_segmenter
    text = "Patient is stable. Follow up tomorrow."
    first_end = len("Patient is stable. ")
    instance = detector_cls.return_value
    instance.detect.return_value = [first_end, len(text)]

    spans = segment_text(text, backend="yasbd")

    detector_cls.assert_called_once_with(lang="en", hook=sentences._yasbd_boundary_hook)
    instance.detect.assert_called_once_with(text)
    assert spans == [
        SentenceSpan("Patient is stable. ", 0, 19),
        SentenceSpan("Follow up tomorrow.", 19, len(text)),
    ]
    assert ("yasbd", "en", False) in sentences._SEGMENTER_CACHE


def test_yasbd_backend_normalizes_whitespace_and_trailing_offsets(
    fake_yasbd_segmenter,
):
    detector_cls = fake_yasbd_segmenter
    text = "Patient is stable.\n\nFollow up tomorrow.\n"
    first_end = text.index("\n")
    final_newline = len(text) - 1
    instance = detector_cls.return_value
    instance.detect.return_value = [first_end, final_newline]

    spans = segment_text(text, backend="yasbd")

    assert [span.text for span in spans] == [
        "Patient is stable.\n\n",
        "Follow up tomorrow.\n",
    ]
    _assert_exact_round_trip(text, spans)
    assert not any(span.text.isspace() for span in spans)


def test_yasbd_backend_preserves_offsets_with_leading_blank_lines(
    fake_yasbd_segmenter,
):
    detector_cls = fake_yasbd_segmenter
    text = "\n\nPatient is stable. Follow up tomorrow.\n"
    first_end = text.index(" ", len("\n\nPatient is stable."))
    final_newline = len(text) - 1
    instance = detector_cls.return_value
    instance.detect.return_value = [first_end, final_newline]

    spans = segment_text(text, backend="yasbd")

    instance.detect.assert_called_once_with(text)
    assert [span.text for span in spans] == [
        "\n\nPatient is stable. ",
        "Follow up tomorrow.\n",
    ]
    _assert_exact_round_trip(text, spans)


def test_yasbd_backend_fails_closed_when_non_whitespace_text_has_no_spans(
    fake_yasbd_segmenter,
):
    instance = fake_yasbd_segmenter.return_value
    instance.detect.return_value = []

    with pytest.raises(ValueError, match="no spans for non-whitespace text"):
        segment_text("Patient is stable.", backend="yasbd")


def test_yasbd_chinese_semicolon_boundary_has_no_global_rule_mutation(
    fake_yasbd_segmenter,
):
    detector_cls = fake_yasbd_segmenter
    text = "第一项完成；第二项完成。"
    instance = detector_cls.return_value

    def detect(source):
        semicolon_end = source.index("；") + 1
        context = {
            "text": source,
            "lang": "zh",
            # Model BoundaryDetector's required sentinels and prove the hook
            # does not duplicate a boundary added by an upstream rule.
            "boundaries": [0, semicolon_end, len(source)],
        }
        detector_cls.call_args.kwargs["hook"](context)
        return sorted(context["boundaries"])[1:]

    instance.detect.side_effect = detect

    spans = segment_text(text, language="zh", backend="yasbd")

    assert [span.text for span in spans] == ["第一项完成；", "第二项完成。"]
    _assert_exact_round_trip(text, spans)


def test_missing_yasbd_dependency_names_openmed_extra():
    with patch.dict(sys.modules, {"yasbd": None}):
        with pytest.raises(ImportError, match=r"openmed\[yasbd\]"):
            segment_text("Patient is stable.", backend="yasbd")


@pytest.mark.integration
def test_yasbd_real_adapter_preserves_openmed_span_contract():
    pytest.importorskip("yasbd", reason="requires the optional openmed[yasbd] extra")
    base_rules = pytest.importorskip("yasbd.rules.base")
    terminators_before = set(base_rules.Rules.TERMINATORS)
    cases = [
        ("en", "Patient is stable.\n\nFollow up tomorrow.\n"),
        ("en", "\n\nPatient is stable.\n\nFollow up tomorrow.\n\n"),
        ("de", "Dr. Müller kam um 8.30 Uhr. Danach ging er."),
        ("es", "El paciente está estable. Seguimiento mañana."),
        ("zh", "第一项完成；第二项完成。"),
    ]

    for language, text in cases:
        spans = segment_text(text, language=language, backend="yasbd")
        _assert_exact_round_trip(text, spans)
        assert not any(span.text.isspace() for span in spans)

    chinese = segment_text(cases[-1][1], language="zh", backend="yasbd")
    assert [span.text for span in chinese] == ["第一项完成；", "第二项完成。"]
    assert set(base_rules.Rules.TERMINATORS) == terminators_before


@pytest.mark.integration
def test_yasbd_real_adapter_rejects_clean_with_exact_spans():
    pytest.importorskip("yasbd", reason="requires the optional openmed[yasbd] extra")

    with pytest.raises(ValueError, match="char_span must be False if clean is True"):
        segment_text("Patient is stable.", clean=True, backend="yasbd")
