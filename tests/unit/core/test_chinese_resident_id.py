"""Chinese Resident ID coverage using algorithmically generated fixtures only.

No identifier in this module is copied from a person or external dataset.
Every valid-looking value is constructed at test runtime from a seeded PRNG.
"""

from __future__ import annotations

import json
import random
from datetime import datetime

from openmed.core.anonymizer import Anonymizer
from openmed.core.anonymizer.providers.clinical_ids import (
    ChineseResidentIdProvider,
    generate_chinese_resident_id,
    register_clinical_providers,
)
from openmed.core.anonymizer.providers.registry_ids import get_national_id
from openmed.core.labels import ID_NUM, normalize_label
from openmed.core.pii_i18n import (
    chinese_resident_id_check_character,
    get_patterns_for_language,
    validate_chinese_resident_id,
    validate_chinese_resident_identity_card,
)
from openmed.core.safety_sweep import safety_sweep
from openmed.processing.outputs import PredictionResult


def _replace_body_segment(
    value: str,
    *,
    start: int,
    end: int,
    replacement: str,
) -> str:
    body = f"{value[:start]}{replacement}{value[end:17]}"
    return f"{body}{chinese_resident_id_check_character(body)}"


def _synthetic_x_case() -> str:
    rng = random.Random(649_001)
    for _ in range(100):
        candidate = generate_chinese_resident_id(rng=rng)
        if candidate.endswith("X"):
            return candidate
    raise AssertionError("seeded generator did not produce an X check character")


def _deidentify_synthetic_note(text: str, *, audit: bool = False):
    from openmed.core import pii

    prediction = PredictionResult(
        text=text,
        entities=safety_sweep(text, [], lang="zh"),
        model_name="synthetic-stub",
        timestamp=datetime.now().isoformat(),
    )
    return pii._build_deidentification_result(
        text,
        prediction,
        effective_method="replace",
        keep_year=False,
        date_shift_days=None,
        keep_mapping=False,
        lang="zh",
        consistent=True,
        seed=649,
        locale=None,
        model_name="synthetic-stub",
        audit=audit,
    )


def test_validator_accepts_generated_numeric_and_x_check_characters() -> None:
    numeric = generate_chinese_resident_id(rng=random.Random(649_002))
    x_case = _synthetic_x_case()

    assert numeric[-1].isdigit()
    assert validate_chinese_resident_id(numeric)
    assert validate_chinese_resident_id(x_case)
    assert validate_chinese_resident_id(f"{x_case[:-1]}x")
    assert validate_chinese_resident_identity_card(x_case)


def test_validator_rejects_checksum_date_region_and_sequence_errors() -> None:
    valid = generate_chinese_resident_id(rng=random.Random(649_003))
    wrong_checksum = f"{valid[:-1]}{'0' if valid[-1] != '0' else '1'}"
    bad_date = _replace_body_segment(
        valid,
        start=6,
        end=14,
        replacement="20210229",
    )
    bad_region = _replace_body_segment(
        valid,
        start=0,
        end=6,
        replacement="990001",
    )
    bad_region_hierarchy = _replace_body_segment(
        valid,
        start=2,
        end=6,
        replacement="0001",
    )
    zero_sequence = _replace_body_segment(
        valid,
        start=14,
        end=17,
        replacement="000",
    )

    assert not validate_chinese_resident_id(wrong_checksum)
    assert not validate_chinese_resident_id(bad_date)
    assert not validate_chinese_resident_id(bad_region)
    assert not validate_chinese_resident_id(bad_region_hierarchy)
    assert not validate_chinese_resident_id(zero_sequence)
    assert not validate_chinese_resident_id(f"{valid[:6]}-{valid[6:]}")
    assert not validate_chinese_resident_id(f" {valid}")


def test_provider_registry_and_locale_generator_round_trip_10k_samples() -> None:
    spec = get_national_id("zh-CN", "resident id")
    assert spec is not None
    assert spec.faker_provider is ChineseResidentIdProvider
    assert spec.faker_method == "chinese_resident_id"

    source_rng = random.Random(649_004)
    anonymizer = Anonymizer(lang="zh", consistent=True, seed=649)
    for _ in range(10_000):
        original = generate_chinese_resident_id(rng=source_rng)
        surrogate = anonymizer.surrogate(original, ID_NUM)

        assert len(surrogate) == 18
        assert surrogate != original
        assert validate_chinese_resident_id(surrogate)
        assert spec.validate(surrogate)


def test_registered_faker_provider_generates_valid_ids() -> None:
    from faker import Faker

    faker = Faker("zh_CN")
    register_clinical_providers(faker)
    faker.seed_instance(649)

    assert validate_chinese_resident_id(faker.chinese_resident_id())


def test_zh_pattern_detects_contextual_id_with_exact_offsets() -> None:
    value = generate_chinese_resident_id(rng=random.Random(649_005))
    text = f"患者身份证号：{value}，请核对。"
    entities = safety_sweep(text, [], lang="zh")

    assert len(entities) == 1
    entity = entities[0]
    assert entity.text == value
    assert entity.start == text.index(value)
    assert entity.end == entity.start + len(value)
    assert normalize_label(entity.label, "zh") == ID_NUM

    patterns = get_patterns_for_language("zh")
    assert any(
        pattern.validator is validate_chinese_resident_id for pattern in patterns
    )


def test_deidentification_replaces_all_ids_with_valid_distinct_surrogates() -> None:
    source_rng = random.Random(649_006)
    originals = [generate_chinese_resident_id(rng=source_rng) for _ in range(3)]
    text = (
        f"患者身份证号：{originals[0]}；家属居民身份证：{originals[1]}；"
        f"复核公民身份号码：{originals[2]}。"
    )

    result = _deidentify_synthetic_note(text)

    replacements = [
        entity.redacted_text
        for entity in result.pii_entities
        if entity.canonical_label == ID_NUM
    ]
    assert len(replacements) == len(originals)
    assert all(original not in result.deidentified_text for original in originals)
    assert all(replacement is not None for replacement in replacements)
    assert all(
        validate_chinese_resident_id(replacement)
        for replacement in replacements
        if replacement is not None
    )
    assert set(replacements).isdisjoint(originals)

    audit_result = _deidentify_synthetic_note(text, audit=True)
    assert audit_result.audit_report is not None
    serialized_audit = json.dumps(
        audit_result.audit_report.to_dict(), ensure_ascii=False
    )
    assert all(original not in serialized_audit for original in originals)
