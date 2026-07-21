from __future__ import annotations

import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

from openmed.interop import adapter_spec, get_adapter, indic, zh


def test_language_adapters_are_registered_lazily():
    assert get_adapter("zh") is zh
    assert get_adapter("indic") is indic
    assert adapter_spec("zh").extra == "zh"
    assert adapter_spec("indic").extra == "indic"


def test_import_openmed_does_not_import_language_dependencies():
    code = """
    import sys
    import openmed

    optional_modules = {"jieba", "opencc", "pypinyin", "indicnlp"}
    assert not optional_modules & set(sys.modules)
    """

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=False,
        cwd=".",
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_zh_segment_missing_extra_raises_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "jieba", None)

    with pytest.raises(ImportError, match=r"pip install openmed\[zh\]"):
        zh.segment("患者张伟")


def test_indic_segment_missing_extra_raises_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "indicnlp", None)

    with pytest.raises(ImportError, match=r"pip install openmed\[indic\]"):
        indic.segment("रोगी रवि")


def test_zh_helpers_load_only_the_requested_dependency(monkeypatch):
    requested = []
    dependencies = {
        "jieba": SimpleNamespace(
            lcut=lambda text, **kwargs: [text[:2], text[2:], kwargs]
        ),
        "opencc": SimpleNamespace(
            OpenCC=lambda config: SimpleNamespace(
                convert=lambda text: f"{config}:{text}"
            )
        ),
        "pypinyin": SimpleNamespace(lazy_pinyin=lambda text: [text, "pin"]),
    }

    def load(name: str):
        requested.append(name)
        return dependencies[name]

    monkeypatch.setattr(zh, "_import_module", load)

    assert zh.segment("患者张伟")[:2] == ("患者", "张伟")
    assert zh.convert_script("汉字", config="s2t") == "s2t:汉字"
    assert zh.to_pinyin("患者") == ("患者", "pin")
    assert requested == ["jieba", "opencc", "pypinyin"]


def test_indic_helpers_load_only_the_requested_dependency(monkeypatch):
    requested = []
    tokenizer = SimpleNamespace(
        trivial_tokenize=lambda text, lang: [lang, *text.split()]
    )

    class Transliterator:
        @staticmethod
        def transliterate(text: str, source: str, target: str) -> str:
            return f"{source}:{target}:{text}"

    transliteration = SimpleNamespace(UnicodeIndicTransliterator=Transliterator)

    def load(name: str):
        requested.append(name)
        if name == "indicnlp.tokenize.indic_tokenize":
            return tokenizer
        return transliteration

    monkeypatch.setattr(indic, "_import_module", load)

    assert indic.segment("रोगी रवि", language="hi") == ("hi", "रोगी", "रवि")
    assert indic.transliterate("रवि", source="hi", target="ta") == "hi:ta:रवि"
    assert requested == [
        "indicnlp.tokenize.indic_tokenize",
        "indicnlp.transliterate.unicode_transliterate",
    ]
