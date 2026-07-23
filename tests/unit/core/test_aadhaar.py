"""Aadhaar detection, masking, and surrogate hardening tests."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import pytest

from openmed.core.anonymizer import Anonymizer
from openmed.core.anonymizer.format_preserve import mask_aadhaar
from openmed.core.anonymizer.providers.clinical_ids import generate_aadhaar
from openmed.core.anonymizer.providers.registry_ids import get_national_id
from openmed.core.pii_entity_merger import find_semantic_units
from openmed.core.pii_i18n import get_patterns_for_language, validate_aadhaar
from openmed.core.pipeline import Pipeline
from openmed.core.safety_sweep import safety_sweep
from openmed.core.surrogate_vault import HMAC_SCHEME, SurrogateVault
from openmed.processing.outputs import PredictionResult

FIXTURE_PATH = Path("openmed/eval/golden/fixtures/aadhaar.json")


def _fixture_rows() -> list[dict[str, object]]:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return payload["fixtures"]


def _empty_detector(text: str, **_kwargs: object) -> PredictionResult:
    return PredictionResult(
        text=text,
        entities=[],
        model_name="synthetic-empty-detector",
        timestamp="2026-01-01T00:00:00Z",
    )


def _pipeline(*, lang: str = "en") -> Pipeline:
    return Pipeline(
        lang=lang,
        model_detector=_empty_detector,
        use_smart_merging=True,
        use_safety_sweep=True,
    )


def test_fixture_validators_and_invalid_corpus_agree() -> None:
    invalid_values: list[str] = []
    for row in _fixture_rows():
        text = str(row["text"])
        for span in row["gold_spans"]:
            value = span["text"]
            assert text[span["start"] : span["end"]] == value
            assert validate_aadhaar(value)
        for negative in row["metadata"]["hard_negatives"]:
            value = negative["text"]
            assert text[negative["start"] : negative["end"]] == value
            if negative["reason"] == "fails_verhoeff":
                invalid_values.append(value)

    invalid_values.extend(
        [
            "681937567570",
            "123456789012",
            "081937567577",
            "+91 6819375675",
            "6819-3756-7577",
        ]
    )
    assert invalid_values
    assert not any(validate_aadhaar(value) for value in invalid_values)


def test_generated_aadhaar_corpus_is_always_verhoeff_valid() -> None:
    source = random.Random(666)
    generated = [generate_aadhaar(rng=source) for _ in range(500)]

    assert len(set(generated)) == len(generated)
    assert all(value[0] not in "01" for value in generated)
    assert all(validate_aadhaar(value) for value in generated)


@pytest.mark.parametrize("alias", ["en", "hi", "in", "india", "en_IN", "hi_IN"])
def test_aadhaar_registry_maps_english_hindi_and_india_aliases(alias: str) -> None:
    spec = get_national_id(alias, "aadhaar")

    assert spec is not None
    assert spec.faker_method == "aadhaar"
    assert spec.validate(generate_aadhaar(rng=random.Random(42)))


@pytest.mark.parametrize("lang", ["en", "hi", "te"])
def test_aadhaar_pattern_is_language_agnostic_and_strict(lang: str) -> None:
    patterns = get_patterns_for_language(lang)
    valid_text = "Aadhaar पहचान: 9664 8088 5109"
    invalid_text = "UID: 919876543210"

    valid_units = [
        unit
        for unit in find_semantic_units(valid_text, patterns)
        if unit[2] == "national_id"
    ]
    invalid_units = [
        unit
        for unit in find_semantic_units(invalid_text, patterns)
        if unit[2] == "national_id"
    ]

    assert len(valid_units) == 1
    assert valid_text[valid_units[0][0] : valid_units[0][1]] == "9664 8088 5109"
    assert valid_units[0][5] is True
    assert invalid_units == []


def test_safety_sweep_rejects_phone_like_random_and_masked_values() -> None:
    text = (
        "Aadhaar 966480885109; phone-like 919876543210; "
        "random 234567890123; masked XXXX XXXX 5109."
    )
    entities = safety_sweep(text, [], lang="en")
    national_ids = [entity for entity in entities if entity.label == "national_id"]

    assert [entity.text for entity in national_ids] == ["966480885109"]


def test_safety_sweep_requires_aadhaar_context_even_for_valid_checksum() -> None:
    assert safety_sweep("Reference 966480885109", [], lang="en") == []


def test_uidai_masking_is_exact_idempotent_and_never_returns_first_eight() -> None:
    assert mask_aadhaar("966480885109") == "XXXX XXXX 5109"
    assert mask_aadhaar("9664 8088 5109") == "XXXX XXXX 5109"
    assert mask_aadhaar("xxxx-xxxx-5109") == "XXXX XXXX 5109"
    assert "96648088" not in mask_aadhaar("966480885109")

    with pytest.raises(ValueError, match="12 digits"):
        mask_aadhaar("9198765432")
    with pytest.raises(ValueError, match="Verhoeff"):
        mask_aadhaar("919876543210")


def test_anonymizer_exposes_mask_and_valid_surrogate_strategies() -> None:
    anonymizer = Anonymizer(lang="en", consistent=True, seed=666)

    masked = anonymizer.anonymize_aadhaar("9664 8088 5109")
    surrogate = anonymizer.anonymize_aadhaar(
        "9664 8088 5109",
        strategy="surrogate",
    )

    assert masked == "XXXX XXXX 5109"
    assert validate_aadhaar(surrogate)
    assert surrogate != "966480885109"

    with pytest.raises(ValueError, match="Verhoeff"):
        anonymizer.anonymize_aadhaar("919876543210")


@pytest.mark.parametrize("row", _fixture_rows(), ids=lambda row: row["id"])
def test_pipeline_detects_and_uidai_masks_english_and_code_mixed_notes(
    row: dict[str, object],
) -> None:
    text = str(row["text"])
    result = _pipeline(lang=str(row["language"])).run(text, method="aadhaar_mask")
    deidentified = result.deidentification_result

    assert deidentified.deidentified_text == row["metadata"]["expected_output"]["text"]
    for span in row["gold_spans"]:
        assert span["text"] not in deidentified.deidentified_text
    for negative in row["metadata"]["hard_negatives"]:
        assert negative["text"] in deidentified.deidentified_text


def test_english_replace_uses_valid_vault_stable_aadhaar_without_logging_source(
    caplog: pytest.LogCaptureFixture,
) -> None:
    raw_aadhaar = "9664 8088 5109"
    text = f"Aadhaar: {raw_aadhaar}"
    vault = SurrogateVault.in_memory("synthetic-unit-test-secret")
    pipeline = _pipeline()

    with caplog.at_level(logging.DEBUG):
        first = pipeline.run(
            text,
            method="replace",
            surrogate_vault=vault,
        ).deidentification_result
        second = pipeline.run(
            text,
            method="replace",
            surrogate_vault=vault,
        ).deidentification_result

    first_surrogate = first.pii_entities[0].surrogate
    second_surrogate = second.pii_entities[0].surrogate
    assert first_surrogate == second_surrogate
    assert first_surrogate is not None and validate_aadhaar(first_surrogate)
    assert raw_aadhaar not in first.deidentified_text
    assert raw_aadhaar not in caplog.text
    assert len(vault.entries()) == 1
    assert vault.entries()[0].key.text_hash.startswith(f"{HMAC_SCHEME}:")
