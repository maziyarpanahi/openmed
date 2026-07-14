"""Tests for the Unified Social Credit Code (USCC) recognizer and validator."""

from __future__ import annotations

import random

import pytest

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
    get_patterns_for_language,
    validate_unified_social_credit_code,
)

# Algorithmically generated valid codes (GB 32100 MOD-31-3), including the two
# check-character edge cases.
VALID_Y = "9135010000ABCDEF3Y"  # check char -> Y (value 30)
VALID_0 = "9135010000ABCDEFQ0"  # check char -> 0
EXCLUDED = "IOZSV"


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


def test_surrogate_replaces_uscc_with_a_valid_distinct_code():
    from openmed.core.anonymizer import Anonymizer

    anon = Anonymizer(seed=5)
    surrogate = anon.surrogate(VALID_Y, "social_credit_code")

    assert validate_unified_social_credit_code(surrogate) is True
    assert surrogate != VALID_Y  # original does not survive
