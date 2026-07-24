"""Tests for word-aware Chinese Pinyin romanization."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from openmed.processing import zh_pinyin
from openmed.processing.tokenization import SpanToken
from openmed.processing.zh_pinyin import (
    PinyinStyle,
    PinyinUnavailableError,
    PinyinUnavailableWarning,
    pinyin_fuzzy_key,
    to_pinyin,
)


def test_tone_styles_match_synthetic_name_readings():
    assert to_pinyin("王芳", style=PinyinStyle.NORMAL) == ("wang", "fang")
    assert to_pinyin("王芳", style=PinyinStyle.TONE_MARK) == ("wáng", "fāng")
    assert to_pinyin("王芳", style=PinyinStyle.TONE3) == ("wang2", "fang1")


def test_polyphone_uses_whole_segmented_word_context():
    class Segmenter:
        def segment(self, text: str) -> list[SpanToken]:
            assert text == "重庆"
            return [SpanToken("重庆", 0, 2)]

    assert to_pinyin("重庆", segmenter=Segmenter()) == ("chong", "qing")


def test_heteronym_output_keeps_all_reported_readings():
    readings = to_pinyin("重", heteronym=True)

    assert "zhong" in readings[0]
    assert "chong" in readings[0]


def test_word_tokens_are_romanized_as_units(monkeypatch):
    calls = []

    class Segmenter:
        def segment(self, text: str) -> list[SpanToken]:
            return [SpanToken("重庆", 0, 2), SpanToken("王芳", 2, 4)]

    fake_module = SimpleNamespace(
        Style=SimpleNamespace(NORMAL=0, TONE=1, TONE3=8),
        pinyin=lambda word, **kwargs: (
            calls.append((word, kwargs)) or [[word, "unused"]]
        ),
    )
    monkeypatch.setattr(zh_pinyin, "_import_module", lambda name: fake_module)

    assert to_pinyin("重庆王芳", segmenter=Segmenter()) == ("重庆", "王芳")
    assert [word for word, _kwargs in calls] == ["重庆", "王芳"]
    assert all(kwargs["heteronym"] is False for _word, kwargs in calls)


def test_missing_optional_dependency_is_actionable_and_key_falls_back(monkeypatch):
    def missing(_name: str):
        raise ModuleNotFoundError("pypinyin")

    monkeypatch.setattr(zh_pinyin, "_import_module", missing)
    monkeypatch.setattr(zh_pinyin, "_PINYIN_NOTICE_EMITTED", False)

    with pytest.raises(PinyinUnavailableError, match=r"openmed\[zh\]"):
        to_pinyin("王芳")
    with pytest.warns(PinyinUnavailableWarning, match=r"openmed\[zh\]"):
        first = pinyin_fuzzy_key("王芳")
    assert first == pinyin_fuzzy_key("王芳")
    assert first.startswith("zh-han-v1:")


def test_fuzzy_key_normalizes_han_marks_numbers_case_and_spacing():
    expected = pinyin_fuzzy_key("王芳")

    assert expected == pinyin_fuzzy_key("Wáng Fāng")
    assert expected == pinyin_fuzzy_key("wang2 fang1")
    assert expected == pinyin_fuzzy_key("WANG-FANG")
