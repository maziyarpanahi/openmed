"""Tests for the Unified Social Credit Code (USCC) recognizer and validator."""

from __future__ import annotations

import random

import pytest

import openmed
from openmed.core.anonymizer.providers.clinical_ids import (
    UnifiedSocialCreditCodeProvider,
    generate_unified_social_credit_code,
)
from openmed.core.labels import (
    ID_SUBTYPE_NATIONAL_ID,
    ID_SUBTYPE_SOCIAL_CREDIT_CODE,
    ID_SUBTYPES,
    id_subtype_for,
)
from openmed.core.pii_entity_merger import find_semantic_units
from openmed.core.pii_i18n import (
    USCC_ALPHABET,
    get_patterns_for_language,
    uscc_check_char,
    validate_unified_social_credit_code,
)
from openmed.processing.outputs import PredictionResult

EXCLUDED = "IOZSV"


def _synthetic_code_with_check_char(check_char: str) -> str:
    """Construct a synthetic valid code with the requested check character."""

    for suffix in USCC_ALPHABET:
        body = f"91123456ABCDEFGH{suffix}"
        if uscc_check_char(body) == check_char:
            return body + check_char
    raise AssertionError(f"could not construct check character {check_char!r}")


VALID_Y = _synthetic_code_with_check_char("Y")
VALID_0 = _synthetic_code_with_check_char("0")


# --------------------------------------------------------------------------
# Validator
# --------------------------------------------------------------------------


@pytest.mark.parametrize("code", [VALID_Y, VALID_0])
def test_accepts_valid_codes_including_edge_cases(code):
    assert validate_unified_social_credit_code(code) is True


def test_rejects_wrong_checksum():
    tampered = VALID_Y[:-1] + ("0" if VALID_Y[-1] != "0" else "1")
    assert validate_unified_social_credit_code(tampered) is False


@pytest.mark.parametrize("bad", list(EXCLUDED))
def test_rejects_excluded_letters(bad):
    # Place an excluded letter in the organization segment (position 11).
    code = VALID_Y[:10] + bad + VALID_Y[11:]
    assert validate_unified_social_credit_code(code) is False


def test_rejects_bad_length_and_non_alphabet():
    assert validate_unified_social_credit_code("ABC") is False
    assert validate_unified_social_credit_code(VALID_Y + "1") is False
    assert validate_unified_social_credit_code(VALID_Y[:-1] + "!") is False


def test_region_segment_must_be_digits():
    # Positions 3-8 are the 6-digit administrative region code.
    code = VALID_Y[:3] + "A" + VALID_Y[4:]
    assert validate_unified_social_credit_code(code) is False


@pytest.mark.parametrize("prefix", ["01", "9A", "B1"])
def test_rejects_invalid_department_category_pair(prefix):
    body = prefix + "123456ABCDEFGH0"
    code = body + uscc_check_char(body)
    assert validate_unified_social_credit_code(code) is False


def test_check_char_rejects_malformed_body():
    with pytest.raises(ValueError, match="17 characters"):
        uscc_check_char("ABC")
    with pytest.raises(ValueError, match="USCC_ALPHABET"):
        uscc_check_char("91123456ABCDEFGI0")


# --------------------------------------------------------------------------
# ID subtype (no new canonical label)
# --------------------------------------------------------------------------


def test_social_credit_subtype_registered():
    assert ID_SUBTYPE_SOCIAL_CREDIT_CODE in ID_SUBTYPES


def test_entity_type_maps_to_distinct_subtype():
    assert id_subtype_for("social_credit_code") == ID_SUBTYPE_SOCIAL_CREDIT_CODE
    # Distinct from a personal Resident/National ID.
    assert ID_SUBTYPE_SOCIAL_CREDIT_CODE != ID_SUBTYPE_NATIONAL_ID


# --------------------------------------------------------------------------
# Surrogate provider / generator
# --------------------------------------------------------------------------


def test_generated_code_validates():
    code = generate_unified_social_credit_code(rng=random.Random(42))
    assert len(code) == 18
    assert validate_unified_social_credit_code(code) is True


def test_10k_surrogates_valid_and_never_reuse_excluded_letters():
    rng = random.Random(7)
    excluded = set(EXCLUDED)
    for _ in range(10_000):
        code = generate_unified_social_credit_code(rng=rng)
        assert validate_unified_social_credit_code(code) is True
        assert not (set(code) & excluded)


def test_provider_method_returns_valid_code():
    from faker import Faker

    faker = Faker()
    faker.add_provider(UnifiedSocialCreditCodeProvider)
    faker.seed_instance(3)
    code = faker.unified_social_credit_code()
    assert validate_unified_social_credit_code(code) is True


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


def test_zh_pattern_detects_in_context_with_correct_offsets():
    # USCC is checksum-guarded and language-agnostic, so it joins the universal
    # base set (like the passport MRZ patterns) rather than a zh-only list.
    text = f"登记机关：统一社会信用代码 {VALID_Y}"
    patterns = get_patterns_for_language("en")

    units = find_semantic_units(text, patterns)

    matched = [u for u in units if text[u[0] : u[1]] == VALID_Y]
    assert matched, "in-context USCC should be detected"
    # The detected span's entity type resolves to the social-credit ID subtype.
    entity_type = matched[0][2]
    assert id_subtype_for(entity_type) == ID_SUBTYPE_SOCIAL_CREDIT_CODE


def test_zh_pattern_allows_adjacent_han_context():
    text = f"登记机关：统一社会信用代码{VALID_Y}。"
    units = find_semantic_units(text, get_patterns_for_language("en"))

    matches = [u for u in units if text[u[0] : u[1]] == VALID_Y]
    assert len(matches) == 1


def test_surrogate_replaces_uscc_with_a_valid_distinct_code():
    from openmed.core.anonymizer import Anonymizer

    anon = Anonymizer(seed=5)
    surrogate = anon.surrogate(VALID_Y, "social_credit_code")

    assert validate_unified_social_credit_code(surrogate) is True
    assert surrogate != VALID_Y  # original does not survive


def test_deidentify_registrant_block_replaces_original_with_valid_code(monkeypatch):
    def fake_analyze_text(text, **_kwargs):
        return PredictionResult(
            text=text,
            entities=[],
            model_name="fixture-pii-model",
            timestamp="2026-01-01T00:00:00",
        )

    monkeypatch.setattr(openmed, "analyze_text", fake_analyze_text)
    original = f"登记机关：统一社会信用代码{VALID_Y}。"

    result = openmed.deidentify(
        original,
        method="replace",
        model_name="fixture-pii-model",
        seed=17,
        use_safety_sweep=False,
    )

    assert VALID_Y not in result.deidentified_text
    assert len(result.pii_entities) == 1
    surrogate = result.pii_entities[0].surrogate
    assert surrogate is not None
    assert validate_unified_social_credit_code(surrogate)
