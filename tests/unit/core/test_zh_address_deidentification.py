"""Chinese address-surrogate, assist, and leakage regression tests."""

from __future__ import annotations

import json
import re
from datetime import datetime
from importlib import resources
from pathlib import Path
from unittest.mock import patch

import pytest

from openmed.core.anonymizer import Anonymizer
from openmed.core.anonymizer.locales import ZH_CN_ADDRESS_LOCALE, resolve_locale
from openmed.core.pii import deidentify
from openmed.core.safety_sweep import SAFETY_SWEEP_SOURCE, safety_sweep
from openmed.processing.outputs import EntityPrediction, PredictionResult

_FIXTURE_PATH = (
    Path(__file__).resolve().parents[2] / "fixtures" / "pii" / "zh_address_cases.json"
)
_ZH_STREET_TAIL_SURROGATE = re.compile(
    r"^(?:安澜路|和景街|康悦路|清禾大道|瑞宁街|新岚路|云栖大道|竹安路)"
    r"\d+号，邮编\d{6}$"
)


@pytest.fixture(scope="module")
def address_fixture() -> dict[str, object]:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def address_division_prefixes() -> tuple[str, ...]:
    resource = resources.files("openmed.clinical").joinpath(
        "data/zh_cn_administrative_divisions.json"
    )
    payload = json.loads(resource.read_text(encoding="utf-8"))
    return tuple(
        f"{province_record['province']}{city_record['city']}{district}"
        for province_record in payload["divisions"]
        for city_record in province_record["cities"]
        for district in city_record["districts"]
    )


def _assert_zh_surrogate_hierarchy(
    value: str,
    division_prefixes: tuple[str, ...],
) -> None:
    matched_prefixes = [
        prefix for prefix in division_prefixes if value.startswith(prefix)
    ]
    assert len(matched_prefixes) == 1
    tail = value[len(matched_prefixes[0]) :]
    assert _ZH_STREET_TAIL_SURROGATE.fullmatch(tail)
    assert re.search(r"[A-Za-z]", value) is None


def _model_prefix_span(case: dict[str, object]) -> EntityPrediction:
    text = str(case["text"])
    prefix = str(case["model_prefix"])
    start = text.index(prefix)
    return EntityPrediction(
        text=prefix,
        label="location",
        start=start,
        end=start + len(prefix),
        confidence=0.99,
        metadata={"source": "model"},
    )


def _covered_token_recall(
    text: str,
    tokens: list[str],
    spans: list[EntityPrediction],
) -> float:
    covered = 0
    for token in tokens:
        start = text.index(token)
        end = start + len(token)
        if any(span.start <= start and span.end >= end for span in spans):
            covered += 1
    return covered / len(tokens)


def test_zh_locale_routes_to_hierarchical_address_generators(
    address_division_prefixes: tuple[str, ...],
) -> None:
    assert resolve_locale("zh") == ZH_CN_ADDRESS_LOCALE

    for seed in range(12):
        anonymizer = Anonymizer(lang="zh", consistent=True, seed=seed)
        location = anonymizer.surrogate("澄海省新川市", "LOCATION")
        address = anonymizer.surrogate(
            "澄海省新川市星河区云杉街道康悦路88号",
            "STREET_ADDRESS",
        )

        assert location in address_division_prefixes
        _assert_zh_surrogate_hierarchy(address, address_division_prefixes)
        assert address.index("省") < address.index("市") < address.index("区")


def test_zh_building_and_postcode_surrogates_have_native_shape() -> None:
    anonymizer = Anonymizer(lang="zh", consistent=True, seed=652)

    assert re.fullmatch(r"\d{3}号", anonymizer.surrogate("88号", "BUILDING_NUMBER"))
    assert re.fullmatch(r"\d{6}", anonymizer.surrogate("518000", "ZIPCODE"))

    for source_number in ("1", "17", "88", "306"):
        for seed in range(24):
            seeded = Anonymizer(lang="zh", consistent=True, seed=seed)
            surrogate = seeded.surrogate(
                f"{source_number}号",
                "BUILDING_NUMBER",
            )
            assert source_number not in surrogate


def test_keep_mapping_reuses_one_surrogate_for_repeated_locality() -> None:
    text = "患者来自新川市，复诊地址仍为新川市。"
    entities = [
        EntityPrediction(
            text=match.group(),
            label="location",
            start=match.start(),
            end=match.end(),
            confidence=0.99,
        )
        for match in re.finditer("新川市", text)
    ]
    prediction = PredictionResult(
        text=text,
        entities=entities,
        model_name="stub",
        timestamp=datetime.now().isoformat(),
    )

    with patch("openmed.core.pii.extract_pii", return_value=prediction):
        result = deidentify(
            text,
            method="replace",
            lang="en",
            locale=ZH_CN_ADDRESS_LOCALE,
            keep_mapping=True,
            use_safety_sweep=False,
        )

    replacements = {entity.redacted_text for entity in result.pii_entities}
    assert len(replacements) == 1
    replacement = replacements.pop()
    assert result.deidentified_text.count(replacement) == 2
    assert result.mapping == {replacement: "新川市"}


def test_address_resource_is_permissive_and_contains_no_patient_addresses() -> None:
    resource = resources.files("openmed.clinical").joinpath(
        "data/zh_cn_administrative_divisions.json"
    )
    payload = json.loads(resource.read_text(encoding="utf-8"))

    assert payload["license"] == "Apache-2.0"
    assert "no patient" in payload["privacy"].casefold()
    assert len(payload["divisions"]) >= 10
    for province_record in payload["divisions"]:
        assert province_record["province"].endswith("省")
        for city_record in province_record["cities"]:
            assert city_record["city"].endswith("市")
            assert city_record["districts"]
            assert all(
                value.endswith(("区", "县")) for value in city_record["districts"]
            )
            assert set(city_record) == {"city", "districts"}
            assert not re.search(
                r"\d|街道|号|室", json.dumps(city_record, ensure_ascii=False)
            )


def test_zh_address_assist_adds_tail_without_moving_model_span(
    address_fixture: dict[str, object],
) -> None:
    for case in address_fixture["cases"]:
        text = case["text"]
        model_span = _model_prefix_span(case)
        original_bounds = (model_span.start, model_span.end)

        spans = safety_sweep(text, [model_span], lang="zh")

        retained = next(
            span for span in spans if span.metadata.get("source") == "model"
        )
        assisted = next(
            span
            for span in spans
            if span.metadata.get("source") == SAFETY_SWEEP_SOURCE
            and span.label == "street_address"
        )
        assert (retained.start, retained.end) == original_bounds
        assert assisted.start == model_span.end
        assert assisted.end == text.index(case["address"]) + len(case["address"])


def test_zh_address_assist_covers_full_hierarchy_without_model_span(
    address_fixture: dict[str, object],
) -> None:
    for case in address_fixture["cases"]:
        text = case["text"]
        address = case["address"]

        spans = safety_sweep(text, [], lang="zh")

        assisted = next(
            span
            for span in spans
            if span.metadata.get("source") == SAFETY_SWEEP_SOURCE
            and span.label == "street_address"
        )
        assert assisted.text == address
        assert assisted.start == text.index(address)
        assert assisted.end == assisted.start + len(address)


def test_synthetic_address_leakage_gate(
    address_fixture: dict[str, object],
    address_division_prefixes: tuple[str, ...],
) -> None:
    recalls = []
    for case in address_fixture["cases"]:
        text = case["text"]
        model_span = _model_prefix_span(case)
        spans = safety_sweep(text, [model_span], lang="zh")
        recalls.append(_covered_token_recall(text, case["labelled_tokens"], spans))

        prediction = PredictionResult(
            text=text,
            entities=spans,
            model_name="stub",
            timestamp=datetime.now().isoformat(),
        )
        with patch("openmed.core.pii.extract_pii", return_value=prediction):
            result = deidentify(
                text,
                method="replace",
                lang="en",
                locale=ZH_CN_ADDRESS_LOCALE,
                keep_mapping=True,
                use_safety_sweep=False,
            )

        surviving_tokens = [
            token
            for token in case["labelled_tokens"]
            if token in result.deidentified_text
        ]
        assert surviving_tokens == []

        combined_surrogate = "".join(
            entity.redacted_text
            for entity in sorted(result.pii_entities, key=lambda item: item.start)
            if entity.label.casefold() in {"location", "street_address"}
        )
        _assert_zh_surrogate_hierarchy(
            combined_surrogate,
            address_division_prefixes,
        )

    assert (
        sum(recalls) / len(recalls) >= address_fixture["minimum_address_token_recall"]
    )
