"""Synthetic-only Chinese personal-name detection and surrogate tests."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from importlib import resources
from pathlib import Path

from openmed.core.anonymizer import Anonymizer
from openmed.core.anonymizer.locales import resolve_locale
from openmed.core.pii import _build_deidentification_result
from openmed.core.pii_entity_merger import merge_entities_with_semantic_units
from openmed.core.pii_i18n import (
    CHINESE_COMPOUND_SURNAMES,
    CHINESE_SURNAMES,
    get_patterns_for_language,
)
from openmed.core.safety_sweep import safety_sweep
from openmed.processing.outputs import PredictionResult

_FIXTURE_PATH = (
    Path(__file__).parents[2] / "fixtures" / "pii" / "chinese_names_synthetic.json"
)
_HAN_NAME = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff]+$")


def _fixture() -> dict:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _prediction_result(text: str):
    entities = safety_sweep(text, [], lang="zh")
    return PredictionResult(
        text=text,
        entities=entities,
        model_name="synthetic-detector",
        timestamp="2026-07-13T00:00:00Z",
        metadata={"fixture": "synthetic-only"},
    )


def test_surname_gazetteer_is_large_packaged_and_explicitly_licensed():
    resource = resources.files("openmed.clinical").joinpath("data/chinese_surnames.txt")
    header = resource.read_text(encoding="utf-8").splitlines()[:10]

    assert "# SPDX-License-Identifier: CC0-1.0" in header
    assert len(CHINESE_SURNAMES) >= 400
    assert {"欧阳", "司马", "诸葛", "上官"} <= CHINESE_COMPOUND_SURNAMES
    assert all(_HAN_NAME.fullmatch(surname) for surname in CHINESE_SURNAMES)
    assert all(len(surname) in {1, 2} for surname in CHINESE_SURNAMES)


def test_zh_language_resolution_selects_the_chinese_surrogate_path():
    assert resolve_locale("zh") == "zh_CN"

    anonymizer = Anonymizer(lang="zh", consistent=True, seed=651)
    surrogate = anonymizer.surrogate("赵雨辰", "PERSON")
    assert 2 <= len(surrogate) <= 3
    assert _HAN_NAME.fullmatch(surrogate)
    assert any(surrogate.startswith(surname) for surname in CHINESE_SURNAMES)
    assert set(surrogate).isdisjoint("赵雨辰")


def test_zh_first_and_last_name_generators_preserve_shape_without_latin_text():
    anonymizer = Anonymizer(lang="zh", consistent=True, seed=652)

    first_name = anonymizer.surrogate("雨辰", "FIRST_NAME")
    last_name = anonymizer.surrogate("王", "LAST_NAME")
    compound_last_name = anonymizer.surrogate("欧阳", "LAST_NAME")

    assert len(first_name) == 2 and _HAN_NAME.fullmatch(first_name)
    assert len(last_name) == 1 and last_name in CHINESE_SURNAMES
    assert compound_last_name in CHINESE_COMPOUND_SURNAMES
    assert set(first_name).isdisjoint("雨辰")
    assert last_name != "王"
    assert set(compound_last_name).isdisjoint("欧阳")


def test_compound_surname_person_input_gets_compound_surname_surrogate():
    for original in ("欧阳清", "司马澄宁"):
        anonymizer = Anonymizer(lang="zh", consistent=True, seed=653)
        surrogate = anonymizer.surrogate(original, "PERSON")
        assert len(surrogate) == 3
        assert any(
            surrogate.startswith(surname) for surname in CHINESE_COMPOUND_SURNAMES
        )
        assert _HAN_NAME.fullmatch(surrogate)
        assert set(surrogate).isdisjoint(original)


def test_surname_assist_expands_an_existing_fragmented_person_detection():
    text = "患者赵雨辰今日复诊。"
    fragment_start = text.index("雨辰")
    model_entities = [
        {
            "entity_type": "PERSON",
            "score": 0.81,
            "start": fragment_start,
            "end": fragment_start + len("雨辰"),
            "word": "雨辰",
        }
    ]

    merged = merge_entities_with_semantic_units(
        model_entities,
        text,
        patterns=get_patterns_for_language("zh"),
        prefer_model_labels=True,
        allow_semantic_only_matches=False,
        allow_label_expansion=False,
    )

    assert len(merged) == 1
    assert merged[0]["word"] == "赵雨辰"
    assert merged[0]["entity_type"] == "PERSON"


def test_synthetic_note_recall_meets_threshold():
    fixture = _fixture()
    expected = 0
    detected = 0

    for case in fixture["cases"]:
        found = [entity.text for entity in safety_sweep(case["text"], [], lang="zh")]
        expected += len(case["names"])
        remaining = list(found)
        for name in case["names"]:
            if name in remaining:
                detected += 1
                remaining.remove(name)

    recall = detected / expected
    assert recall >= fixture["recall_threshold"]


def test_replace_has_zero_source_character_leakage_and_consistent_mapping():
    for case in _fixture()["cases"]:
        pii_result = _prediction_result(case["text"])
        result = _build_deidentification_result(
            case["text"],
            pii_result,
            effective_method="replace",
            keep_year=False,
            date_shift_days=None,
            keep_mapping=True,
            lang="zh",
            consistent=False,
            seed=None,
            locale=None,
        )

        surrogates_by_source: dict[str, set[str]] = defaultdict(set)
        for entity in result.pii_entities:
            surrogates_by_source[entity.original_text].add(entity.surrogate)

        assert set(surrogates_by_source) == set(case["names"])
        assert all(len(values) == 1 for values in surrogates_by_source.values())
        assert len(result.mapping) == len(set(case["names"]))
        for original in set(case["names"]):
            assert original not in result.deidentified_text
            assert set(original).isdisjoint(result.deidentified_text)
