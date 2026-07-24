"""Synthetic span-level coverage for the optional CJK and Indic adapters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.core.safety_sweep import safety_sweep
from openmed.core.script_detect import (
    detect_chinese_script,
    detect_script,
    segment_by_script,
)
from openmed.interop import indic, zh

_FIXTURE_DIR = Path(__file__).parents[2] / "fixtures" / "i18n"
_FIXTURE_NAMES = ("zh-Hans.json", "zh-Hant.json", "hi.json", "ta.json", "hinglish.json")
_FIXTURE_PATHS = tuple(_FIXTURE_DIR / name for name in _FIXTURE_NAMES)


def _load_fixture(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _token_spans(text: str, tokens: tuple[str, ...]) -> list[dict[str, object]]:
    spans: list[dict[str, object]] = []
    cursor = 0
    for token in tokens:
        start = text.index(token, cursor)
        end = start + len(token)
        spans.append({"start": start, "end": end, "text": token})
        cursor = end
    return spans


@pytest.mark.parametrize("path", _FIXTURE_PATHS, ids=lambda path: path.stem)
def test_i18n_fixtures_have_exact_script_and_entity_spans(path: Path) -> None:
    fixture = _load_fixture(path)
    text = str(fixture["text"])

    assert fixture["schema_version"] == 1
    assert fixture["synthetic"] is True
    assert fixture["contains_real_phi"] is False
    assert detect_script(text) == fixture["expected_script"]
    assert [
        {"start": start, "end": end, "script": script}
        for start, end, script in segment_by_script(text)
    ] == fixture["expected_segments"]

    entities = safety_sweep(text, [], lang=str(fixture["language"]))
    assert [
        {
            "label": entity.label,
            "start": entity.start,
            "end": entity.end,
            "text": entity.text,
        }
        for entity in entities
    ] == fixture["expected_entities"]

    for span in [*fixture["expected_segments"], *fixture["expected_entities"]]:
        assert 0 <= span["start"] < span["end"] <= len(text)
    for entity in fixture["expected_entities"]:
        assert text[entity["start"] : entity["end"]] == entity["text"]


@pytest.mark.parametrize("path", _FIXTURE_PATHS, ids=lambda path: path.stem)
def test_optional_language_adapters_match_fixture_token_spans(path: Path) -> None:
    fixture = _load_fixture(path)
    text = str(fixture["text"])
    language = str(fixture["language"])
    tokens = (
        zh.segment(text)
        if language == "zh"
        else indic.segment(
            text,
            language=language,
        )
    )

    assert _token_spans(text, tokens) == fixture["expected_tokens"]


def test_opencc_round_trips_simplified_and_traditional_fixtures() -> None:
    simplified = _load_fixture(_FIXTURE_DIR / "zh-Hans.json")
    traditional = _load_fixture(_FIXTURE_DIR / "zh-Hant.json")

    assert (
        zh.convert_script(str(simplified["text"]), config="s2t") == traditional["text"]
    )
    assert (
        zh.convert_script(str(traditional["text"]), config="t2s") == simplified["text"]
    )
    assert detect_chinese_script(str(simplified["text"])).variant.value == "simplified"
    assert (
        detect_chinese_script(str(traditional["text"])).variant.value == "traditional"
    )
