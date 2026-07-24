"""Chinese Pinyin name-surrogate and vault regression tests."""

from __future__ import annotations

import re

import pytest
from faker import Faker

from openmed.core.anonymizer.providers.clinical_ids import (
    ChineseNameProvider,
    generate_chinese_name,
)
from openmed.core.pii import _build_deidentification_result
from openmed.core.surrogate_vault import SurrogateVault
from openmed.processing import zh_pinyin
from openmed.processing.outputs import EntityPrediction, PredictionResult
from openmed.processing.zh_pinyin import PinyinUnavailableWarning, pinyin_fuzzy_key

_HAN_NAME = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff]+$")


def _prediction_result(text: str, surface: str) -> PredictionResult:
    entities = []
    cursor = 0
    while (start := text.find(surface, cursor)) >= 0:
        end = start + len(surface)
        entities.append(
            EntityPrediction(
                text=surface,
                label="PERSON",
                start=start,
                end=end,
                confidence=0.99,
            )
        )
        cursor = end
    return PredictionResult(
        text=text,
        entities=entities,
        model_name="synthetic-chinese-name",
        timestamp="now",
    )


def test_provider_is_seeded_by_pinyin_key_and_source_disjoint():
    first = Faker("zh_CN")
    second = Faker("zh_CN")
    first.add_provider(ChineseNameProvider)
    second.add_provider(ChineseNameProvider)
    first.seed_instance(1)
    second.seed_instance(999)

    first_surrogate = first.chinese_name("王芳")
    second_surrogate = second.chinese_name("Wáng Fāng")

    assert first_surrogate == second_surrogate
    assert _HAN_NAME.fullmatch(first_surrogate)
    assert set(first_surrogate).isdisjoint("王芳")


def test_pinyin_surface_variants_share_one_vault_key():
    vault = SurrogateVault.in_memory("synthetic-chinese-name-secret")
    surfaces = ("王芳", "Wáng Fāng", "wang2 fang1")

    assert len({pinyin_fuzzy_key(surface) for surface in surfaces}) == 1
    assert (
        len({vault.key_for(surface, label="PERSON", lang="zh") for surface in surfaces})
        == 1
    )


def test_repeated_name_uses_one_vault_surrogate_and_keeps_source_offsets():
    text = "患者王芳复诊，王芳已知情。"
    surface = "王芳"
    vault = SurrogateVault.in_memory("synthetic-chinese-offset-secret")

    result = _build_deidentification_result(
        text,
        _prediction_result(text, surface),
        effective_method="replace",
        keep_year=False,
        date_shift_days=None,
        keep_mapping=True,
        lang="zh",
        consistent=False,
        seed=None,
        locale=None,
        surrogate_vault=vault,
    )

    assert len(result.pii_entities) == 2
    assert len({entity.surrogate for entity in result.pii_entities}) == 1
    assert len(vault.entries()) == 1
    for entity in result.pii_entities:
        assert text[entity.start : entity.end] == surface
        assert entity.end - entity.start == len(surface)
        assert entity.surrogate is not None
        assert _HAN_NAME.fullmatch(entity.surrogate)
    assert surface not in result.deidentified_text


def test_provider_remains_deterministic_without_pypinyin(monkeypatch):
    def missing(_name: str):
        raise ModuleNotFoundError("pypinyin")

    monkeypatch.setattr(zh_pinyin, "_import_module", missing)
    monkeypatch.setattr(zh_pinyin, "_PINYIN_NOTICE_EMITTED", False)

    with pytest.warns(PinyinUnavailableWarning, match=r"openmed\[zh\]"):
        first = generate_chinese_name("王芳")
    second = generate_chinese_name("王芳")

    assert first == second
    assert _HAN_NAME.fullmatch(first)
    assert set(first).isdisjoint("王芳")
