from __future__ import annotations

import importlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from openmed.core.config import OpenMedConfig
from openmed.processing.tokenization import SpanToken
from openmed.processing.zh_segmentation import (
    ChineseSegmenter,
    HanLPSegmenter,
    JiebaSegmenter,
    PkusegSegmenter,
    UserDictionaryEntry,
    create_chinese_segmenter,
    create_chinese_segmenter_from_config,
    load_user_dictionary,
    segmentation_boundary_f1,
    validate_segmentation,
)

FIXTURE_PATH = (
    Path(__file__).parents[2] / "fixtures" / "processing" / "zh_segmentation_gold.json"
)


def _span_tokens(text: str, words: list[str]) -> list[SpanToken]:
    tokens = []
    cursor = 0
    for word in words:
        start = text.index(word, cursor)
        end = start + len(word)
        tokens.append(SpanToken(word, start, end))
        cursor = end
    return tokens


def test_default_jieba_backend_keeps_seeded_clinical_terms_and_offsets():
    text = "患者王芳因心房颤动入院"

    segmenter = create_chinese_segmenter()
    tokens = segmenter.segment(text)

    assert isinstance(segmenter, ChineseSegmenter)
    assert "王芳" in [token.text for token in tokens]
    assert "心房颤动" in [token.text for token in tokens]
    assert all(text[token.start : token.end] == token.text for token in tokens)
    validate_segmentation(text, tokens)


def test_user_dictionary_loader_and_custom_jieba_dictionary(tmp_path):
    dictionary = tmp_path / "custom.txt"
    dictionary.write_text(
        "心脏超声 90000 nz\n临床路径\n",
        encoding="utf-8",
    )

    assert load_user_dictionary(dictionary) == (
        UserDictionaryEntry("心脏超声", 90000, "nz"),
        UserDictionaryEntry("临床路径"),
    )

    tokens = JiebaSegmenter(user_dict_path=dictionary).segment("完成心脏超声检查")
    assert "心脏超声" in [token.text for token in tokens]


@pytest.mark.parametrize("frequency", ["zero 0", "negative -1", "bad nope"])
def test_user_dictionary_rejects_invalid_frequency(tmp_path, frequency):
    dictionary = tmp_path / "invalid.txt"
    dictionary.write_text(f"{frequency}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="frequency"):
        load_user_dictionary(dictionary)


def test_validation_rejects_wrong_offsets_overlap_and_non_whitespace_gaps():
    with pytest.raises(ValueError, match="does not match"):
        validate_segmentation("患者", [SpanToken("病人", 0, 2)])

    with pytest.raises(ValueError, match="overlaps"):
        validate_segmentation(
            "患者王芳",
            [SpanToken("患者", 0, 2), SpanToken("者王", 1, 3)],
        )

    with pytest.raises(ValueError, match="uncovered"):
        validate_segmentation("患者王芳", [SpanToken("王芳", 2, 4)])


def test_round_trip_invariant_for_1000_random_han_strings():
    randomizer = random.Random(639)
    alphabet = "患者王芳心房颤动入院高血压糖尿病肺炎肾功能正常异常检查治疗"
    segmenter = JiebaSegmenter()

    for _ in range(1000):
        text = "".join(
            randomizer.choice(alphabet) for _ in range(randomizer.randint(1, 64))
        )
        tokens = segmenter.segment(text)
        assert tokens == sorted(tokens, key=lambda token: (token.start, token.end))
        assert all(text[token.start : token.end] == token.text for token in tokens)
        validate_segmentation(text, tokens)


def test_held_out_200_sentence_boundary_f1_gate():
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    segmenter = JiebaSegmenter()
    scores = []

    for name in payload["names"]:
        for condition in payload["conditions"]:
            words = [
                word.format(name=name, condition=condition)
                for word in payload["gold_tokens"]
            ]
            text = "".join(words)
            gold = _span_tokens(text, words)
            predicted = segmenter.segment(text)
            scores.append(segmentation_boundary_f1(gold, predicted))

    assert len(scores) == 200
    assert sum(scores) / len(scores) >= 0.90


@pytest.mark.parametrize(
    ("backend", "module_name", "extra", "license_name"),
    [
        ("pkuseg", "pkuseg", "zh-pkuseg", "MIT"),
        ("hanlp", "hanlp", "zh-hanlp", "Apache-2.0"),
    ],
)
def test_missing_optional_backend_error_names_extra_and_license(
    monkeypatch,
    backend,
    module_name,
    extra,
    license_name,
):
    real_import = importlib.import_module

    def missing_dependency(name):
        if name == module_name:
            raise ModuleNotFoundError(name)
        return real_import(name)

    monkeypatch.setattr(importlib, "import_module", missing_dependency)

    with pytest.raises(ImportError) as exc_info:
        create_chinese_segmenter(backend)

    message = str(exc_info.value)
    assert extra in message
    assert license_name in message


def test_pkuseg_adapter_is_model_lazy_and_receives_seeded_dictionary(monkeypatch):
    observed = {}

    class FakePkuseg:
        def cut(self, text):
            assert text == "患者王芳入院"
            return ["患者", "王芳", "入院"]

    def build_pkuseg(**kwargs):
        observed.update(kwargs)
        return FakePkuseg()

    real_import = importlib.import_module
    fake_module = SimpleNamespace(pkuseg=build_pkuseg)
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake_module if name == "pkuseg" else real_import(name),
    )

    segmenter = PkusegSegmenter(model_name="medicine")
    assert observed == {}
    tokens = segmenter.segment("患者王芳入院")

    assert observed["model_name"] == "medicine"
    assert "心房颤动" in observed["user_dict"]
    assert [token.text for token in tokens] == ["患者", "王芳", "入院"]


def test_pkuseg_domain_model_is_never_downloaded_implicitly(monkeypatch, tmp_path):
    called = False

    def build_pkuseg(**kwargs):
        nonlocal called
        called = True
        return kwargs

    fake_config = SimpleNamespace(
        available_models={"medicine"},
        pkuseg_home=str(tmp_path),
    )
    fake_module = SimpleNamespace(pkuseg=build_pkuseg, config=fake_config)
    real_import = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake_module if name == "pkuseg" else real_import(name),
    )

    segmenter = PkusegSegmenter(model_name="medicine")
    with pytest.raises(FileNotFoundError, match="does not download"):
        segmenter.segment("患者王芳入院")
    assert called is False


def test_hanlp_adapter_accepts_preloaded_model_and_applies_dictionary(monkeypatch):
    real_import = importlib.import_module
    fake_module = SimpleNamespace(load=lambda path: pytest.fail(path))
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake_module if name == "hanlp" else real_import(name),
    )
    segmenter = HanLPSegmenter(
        model=lambda text: ["患者", "王", "芳", "因", "心房", "颤动", "入院"]
    )

    tokens = segmenter.segment("患者王芳因心房颤动入院")

    assert [token.text for token in tokens] == [
        "患者",
        "王芳",
        "因",
        "心房颤动",
        "入院",
    ]


def test_chinese_segmentation_config_round_trips_and_validates(monkeypatch, tmp_path):
    dictionary = tmp_path / "terms.txt"
    dictionary.write_text("心房颤动\n", encoding="utf-8")
    config = OpenMedConfig.from_dict(
        {
            "chinese_segmentation_backend": "PKUSEG",
            "chinese_user_dict_path": str(dictionary),
            "chinese_pkuseg_domain": "medicine",
        }
    )

    assert config.chinese_segmentation_backend == "pkuseg"
    assert config.to_dict()["chinese_user_dict_path"] == str(dictionary)

    real_import = importlib.import_module
    fake_module = SimpleNamespace(pkuseg=lambda **kwargs: kwargs)
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake_module if name == "pkuseg" else real_import(name),
    )
    configured_segmenter = create_chinese_segmenter_from_config(config)
    assert isinstance(configured_segmenter, PkusegSegmenter)

    monkeypatch.setenv("OPENMED_CHINESE_SEGMENTATION_BACKEND", "hanlp")
    monkeypatch.setenv("OPENMED_CHINESE_USER_DICT", "/tmp/custom.txt")
    monkeypatch.setenv("OPENMED_CHINESE_PKUSEG_DOMAIN", "web")
    env_config = OpenMedConfig()
    assert env_config.chinese_segmentation_backend == "hanlp"
    assert env_config.chinese_user_dict_path == "/tmp/custom.txt"
    assert env_config.chinese_pkuseg_domain == "web"

    monkeypatch.setenv("OPENMED_CHINESE_SEGMENTATION_BACKEND", "unknown")
    with pytest.raises(ValueError, match="chinese_segmentation_backend"):
        OpenMedConfig()
    monkeypatch.delenv("OPENMED_CHINESE_SEGMENTATION_BACKEND")
