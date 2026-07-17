"""Chinese mobile, bank-card, passport, and permit regression tests."""

from __future__ import annotations

import json
import random
import re
from unittest.mock import patch

import pytest
from faker import Faker

import openmed
from openmed.core.anonymizer import Anonymizer
from openmed.core.anonymizer.providers.clinical_ids import (
    ChineseIdentifierProvider,
    generate_chinese_bank_card,
    generate_chinese_mobile_number,
    generate_chinese_passport,
    generate_hong_kong_macau_permit,
    generate_taiwan_compatriot_permit,
    register_clinical_providers,
)
from openmed.core.anonymizer.providers.registry_ids import get_national_id
from openmed.core.labels import (
    ID_NUM,
    ID_SUBTYPE_CHINESE_PASSPORT,
    ID_SUBTYPE_HONG_KONG_MACAU_PERMIT,
    ID_SUBTYPE_TAIWAN_PERMIT,
    ID_SUBTYPES,
    id_subtype_for,
    normalize_label,
)
from openmed.core.pii_i18n import (
    LANGUAGE_PII_PATTERNS,
    get_patterns_for_language,
    validate_chinese_bank_card,
    validate_chinese_mobile_number,
    validate_chinese_passport,
    validate_hong_kong_macau_permit,
    validate_taiwan_compatriot_permit,
)
from openmed.core.safety_sweep import safety_sweep
from openmed.processing.outputs import PredictionResult


def _synthetic_values(seed: int = 654) -> dict[str, str]:
    rng = random.Random(seed)
    return {
        "mobile": generate_chinese_mobile_number(rng=rng),
        "card": generate_chinese_bank_card(length=18, rng=rng),
        "passport": generate_chinese_passport(rng=rng),
        "home_return_permit": generate_hong_kong_macau_permit(rng=rng),
        "taiwan_permit": generate_taiwan_compatriot_permit(rng=rng),
    }


def _synthetic_note(values: dict[str, str]) -> str:
    return (
        f"手机：{values['mobile']}；银行卡号：{values['card']}；"
        f"护照号：{values['passport']}；回乡证：{values['home_return_permit']}；"
        f"台胞证：{values['taiwan_permit']}。"
    )


def _empty_model_result(text: str, *_args, **_kwargs) -> PredictionResult:
    return PredictionResult(
        text=text,
        entities=[],
        model_name="synthetic-offline-fixture",
        timestamp="2026-01-01T00:00:00",
    )


class TestChineseValidators:
    def test_mobile_accepts_domestic_and_country_prefix_forms(self) -> None:
        mobile = generate_chinese_mobile_number(rng=random.Random(1))

        assert validate_chinese_mobile_number(mobile)
        assert validate_chinese_mobile_number(f"+86{mobile}")
        assert validate_chinese_mobile_number(f"+86 {mobile}")
        assert not validate_chinese_mobile_number(mobile[:-1])
        assert not validate_chinese_mobile_number("0" + mobile)

    @pytest.mark.parametrize("length", range(16, 20))
    def test_bank_cards_are_luhn_valid_for_every_supported_length(
        self,
        length: int,
    ) -> None:
        card = generate_chinese_bank_card(
            length=length,
            rng=random.Random(600 + length),
        )
        bad_check_digit = str((int(card[-1]) + 1) % 10)

        assert len(card) == length
        assert validate_chinese_bank_card(card)
        assert not validate_chinese_bank_card(card[:-1] + bad_check_digit)

    def test_travel_document_validators_reject_wrong_shapes(self) -> None:
        values = _synthetic_values()

        assert validate_chinese_passport(values["passport"])
        assert validate_hong_kong_macau_permit(values["home_return_permit"])
        assert validate_taiwan_compatriot_permit(values["taiwan_permit"])
        assert not validate_chinese_passport(values["passport"][1:])
        assert not validate_hong_kong_macau_permit(
            "X" + values["home_return_permit"][1:]
        )
        assert not validate_taiwan_compatriot_permit(values["taiwan_permit"] + "0")


class TestChineseRecognition:
    def test_language_and_locale_paths_expose_the_chinese_patterns(self) -> None:
        zh_patterns = LANGUAGE_PII_PATTERNS["zh"]

        assert all(
            pattern in get_patterns_for_language("zh") for pattern in zh_patterns
        )
        assert all(
            pattern in get_patterns_for_language("en", locale="zh_CN")
            for pattern in zh_patterns
        )

    def test_detection_has_exact_offsets_and_distinct_id_subtypes(self) -> None:
        values = _synthetic_values()
        note = _synthetic_note(values)
        spans = safety_sweep(note, [], lang="zh")
        by_text = {span.text: span for span in spans}

        assert set(values.values()) <= set(by_text)
        for value in values.values():
            span = by_text[value]
            assert (span.start, span.end) == (
                note.index(value),
                note.index(value) + len(value),
            )

        assert by_text[values["mobile"]].label == "phone_number"
        assert by_text[values["card"]].label == "credit_card"
        assert (
            by_text[values["passport"]].metadata["id_subtype"]
            == ID_SUBTYPE_CHINESE_PASSPORT
        )
        assert (
            by_text[values["home_return_permit"]].metadata["id_subtype"]
            == ID_SUBTYPE_HONG_KONG_MACAU_PERMIT
        )
        assert (
            by_text[values["taiwan_permit"]].metadata["id_subtype"]
            == ID_SUBTYPE_TAIWAN_PERMIT
        )

    def test_context_gates_bank_card_passport_and_permits(self) -> None:
        values = _synthetic_values()
        patterns = LANGUAGE_PII_PATTERNS["zh"]

        for key in ("card", "passport", "home_return_permit", "taiwan_permit"):
            assert safety_sweep(values[key], [], patterns=patterns) == []

    def test_mobile_pattern_rejects_short_and_leading_zero_noise(self) -> None:
        mobile = generate_chinese_mobile_number(rng=random.Random(2))
        patterns = LANGUAGE_PII_PATTERNS["zh"]

        assert safety_sweep(f"电话：{mobile[:-1]}", [], patterns=patterns) == []
        assert safety_sweep(f"手机：0{mobile}", [], patterns=patterns) == []


class TestChineseSubtypesAndProviders:
    @pytest.mark.parametrize(
        ("label", "subtype"),
        [
            ("chinese_passport", ID_SUBTYPE_CHINESE_PASSPORT),
            ("home_return_permit", ID_SUBTYPE_HONG_KONG_MACAU_PERMIT),
            ("taiwan_compatriot_permit", ID_SUBTYPE_TAIWAN_PERMIT),
        ],
    )
    def test_travel_documents_keep_flat_id_num_contract(
        self,
        label: str,
        subtype: str,
    ) -> None:
        assert normalize_label(label) == ID_NUM
        assert id_subtype_for(label) == subtype
        assert subtype in ID_SUBTYPES

    @pytest.mark.parametrize(
        ("id_type", "validator"),
        [
            ("chinese_passport", validate_chinese_passport),
            ("hong_kong_macau_permit", validate_hong_kong_macau_permit),
            ("taiwan_permit", validate_taiwan_compatriot_permit),
        ],
    )
    def test_registry_specs_generate_valid_surrogates(self, id_type, validator) -> None:
        spec = get_national_id("zh_CN", id_type)
        assert spec is not None

        faker = Faker("zh_CN")
        register_clinical_providers(faker)
        faker.seed_instance(654)
        assert validator(getattr(faker, spec.faker_method)())

    def test_provider_registers_all_chinese_methods(self) -> None:
        faker = Faker("zh_CN")
        faker.add_provider(ChineseIdentifierProvider)
        faker.seed_instance(654)

        assert validate_chinese_mobile_number(faker.chinese_mobile_number())
        assert validate_chinese_bank_card(faker.chinese_bank_card())
        assert validate_chinese_passport(faker.chinese_passport())
        assert validate_hong_kong_macau_permit(faker.hong_kong_macau_permit())
        assert validate_taiwan_compatriot_permit(faker.taiwan_compatriot_permit())

    def test_anonymizer_surrogates_are_valid_and_distinct(self) -> None:
        values = _synthetic_values()
        anonymizer = Anonymizer(locale="zh_CN", consistent=True, seed=654)

        mobile = anonymizer.surrogate(values["mobile"], "PHONE")
        card = anonymizer.surrogate(values["card"], "CREDIT_CARD")
        passport = anonymizer.surrogate(values["passport"], "chinese_passport")
        home_return = anonymizer.surrogate(
            values["home_return_permit"],
            "home_return_permit",
        )
        taiwan = anonymizer.surrogate(
            values["taiwan_permit"],
            "taiwan_compatriot_permit",
        )

        assert len(mobile) == 11
        assert validate_chinese_mobile_number(mobile)
        assert len(card) == len(values["card"])
        assert validate_chinese_bank_card(card)
        assert validate_chinese_passport(passport)
        assert validate_hong_kong_macau_permit(home_return)
        assert validate_taiwan_compatriot_permit(taiwan)
        assert {mobile, card, passport, home_return, taiwan}.isdisjoint(values.values())

        prefixed_mobile = anonymizer.surrogate(
            "+86 " + values["mobile"],
            "PHONE",
        )
        assert len(prefixed_mobile) == 11
        assert validate_chinese_mobile_number(prefixed_mobile)


class TestChineseLeakageGate:
    def test_standard_deidentification_cascade_replaces_every_original(self) -> None:
        values = _synthetic_values()
        note = _synthetic_note(values)

        with patch("openmed.core.pii.extract_pii", side_effect=_empty_model_result):
            result = openmed.deidentify(
                note,
                method="replace",
                model_name="synthetic-offline-fixture",
                lang="en",
                locale="zh_CN",
                seed=654,
            )

        assert all(value not in result.deidentified_text for value in values.values())
        by_original = {
            entity.original_text: entity.surrogate for entity in result.pii_entities
        }
        assert set(values.values()) <= set(by_original)
        assert validate_chinese_mobile_number(by_original[values["mobile"]])
        assert validate_chinese_bank_card(by_original[values["card"]])
        assert validate_chinese_passport(by_original[values["passport"]])
        assert validate_hong_kong_macau_permit(
            by_original[values["home_return_permit"]]
        )
        assert validate_taiwan_compatriot_permit(by_original[values["taiwan_permit"]])

    def test_audit_output_contains_hashes_and_offsets_not_originals(self) -> None:
        values = _synthetic_values()
        note = _synthetic_note(values)

        with patch("openmed.core.pii.extract_pii", side_effect=_empty_model_result):
            audit = openmed.deidentify(
                note,
                method="replace",
                model_name="synthetic-offline-fixture",
                lang="en",
                locale="zh_CN",
                seed=654,
                audit=True,
            )

        payload = audit.to_dict()
        serialized = json.dumps(payload, ensure_ascii=False)
        assert all(value not in serialized for value in values.values())
        for span in payload["spans"]:
            assert isinstance(span["start"], int)
            assert isinstance(span["end"], int)
            assert re.fullmatch(r"sha256:[0-9a-f]{64}", span["text_hash"])
