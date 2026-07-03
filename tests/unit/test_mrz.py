"""Tests for ICAO 9303 passport/ID MRZ detection and validation (issue #899)."""

from __future__ import annotations

import pytest

from openmed.core.labels import ID_NUM, id_subtype_for, normalize_label
from openmed.core.pii_entity_merger import find_semantic_units
from openmed.core.pii_i18n import (
    generate_mrz_td1,
    generate_mrz_td3,
    get_patterns_for_language,
    validate_mrz_td1,
    validate_mrz_td3,
)

# Canonical ICAO 9303 specimen MRZ blocks (synthetic; no real passport data).
VALID_TD3 = (
    "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\n"
    "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
)
VALID_TD1 = (
    "I<UTOD231458907<<<<<<<<<<<<<<<\n"
    "7408122F1204159UTO<<<<<<<<<<<6\n"
    "ERIKSSON<<ANNA<MARIA<<<<<<<<<<"
)


class TestMrzTd3Validator:
    def test_accepts_valid_block(self):
        assert validate_mrz_td3(VALID_TD3) is True

    def test_rejects_wrong_document_number_check_digit(self):
        line1, line2 = VALID_TD3.split("\n")
        # Corrupt the document-number check digit (index 9 on line 2).
        bad = line2[:9] + ("7" if line2[9] != "7" else "5") + line2[10:]
        assert validate_mrz_td3(f"{line1}\n{bad}") is False

    def test_rejects_wrong_composite_check_digit(self):
        line1, line2 = VALID_TD3.split("\n")
        bad = line2[:43] + ("1" if line2[43] != "1" else "2")
        assert validate_mrz_td3(f"{line1}\n{bad}") is False

    def test_rejects_wrong_line_length(self):
        assert validate_mrz_td3("P<UTO\nL898902C3") is False


class TestMrzTd1Validator:
    def test_accepts_valid_block(self):
        assert validate_mrz_td1(VALID_TD1) is True

    def test_rejects_wrong_dob_check_digit(self):
        lines = VALID_TD1.split("\n")
        line2 = lines[1]
        lines[1] = line2[:6] + ("3" if line2[6] != "3" else "5") + line2[7:]
        assert validate_mrz_td1("\n".join(lines)) is False

    def test_rejects_wrong_line_count(self):
        assert validate_mrz_td1(VALID_TD1.rsplit("\n", 1)[0]) is False


class TestCrossType:
    def test_td3_validator_rejects_td1_block(self):
        assert validate_mrz_td3(VALID_TD1) is False

    def test_td1_validator_rejects_td3_block(self):
        assert validate_mrz_td1(VALID_TD3) is False


class TestMrzLabel:
    def test_passport_mrz_normalizes_to_id_num(self):
        assert normalize_label("passport_mrz") == ID_NUM

    def test_passport_mrz_subtype(self):
        assert id_subtype_for("passport_mrz") == "passport_mrz"


class TestMrzDetection:
    def test_mrz_patterns_present_for_all_languages(self):
        for lang in ("en", "de", "fr"):
            types = {p.entity_type for p in get_patterns_for_language(lang)}
            assert "passport_mrz" in types, lang

    def test_td3_detected_with_multiline_offsets(self):
        text = f"Scanned document\n{VALID_TD3}\nend of page"
        patterns = get_patterns_for_language("en")
        mrz = [u for u in find_semantic_units(text, patterns) if u[2] == "passport_mrz"]
        assert mrz
        start, end = mrz[0][0], mrz[0][1]
        assert text[start:end] == VALID_TD3

    def test_td1_detected_with_multiline_offsets(self):
        text = f"ID card\n{VALID_TD1}\nfooter"
        patterns = get_patterns_for_language("en")
        mrz = [u for u in find_semantic_units(text, patterns) if u[2] == "passport_mrz"]
        assert mrz
        start, end = mrz[0][0], mrz[0][1]
        assert text[start:end] == VALID_TD1

    def test_invalid_check_digit_scores_below_valid(self):
        line1, line2 = VALID_TD3.split("\n")
        bad_cd = "7" if line2[9] != "7" else "5"
        bad = f"{line1}\n{line2[:9]}{bad_cd}{line2[10:]}"
        patterns = get_patterns_for_language("en")
        good = [
            u
            for u in find_semantic_units(VALID_TD3, patterns)
            if u[2] == "passport_mrz"
        ]
        bad_units = [
            u for u in find_semantic_units(bad, patterns) if u[2] == "passport_mrz"
        ]
        assert good
        if bad_units:
            assert good[0][3] > bad_units[0][3]


class TestMrzSurrogate:
    def test_generators_round_trip_validators(self):
        import random

        rng = random.Random(7)
        for _ in range(50):
            assert validate_mrz_td3(generate_mrz_td3(rng))
            assert validate_mrz_td1(generate_mrz_td1(rng))

    def test_replace_path_emits_valid_surrogate_mrz(self):
        from openmed.core.anonymizer import Anonymizer

        anon = Anonymizer()
        surrogate = anon.surrogate(VALID_TD3, "passport_mrz")
        assert validate_mrz_td3(surrogate)
        assert surrogate != VALID_TD3


class TestMrzGoldenFixture:
    def _fixtures(self):
        from openmed.eval.golden import load_golden_fixtures

        return load_golden_fixtures("openmed/eval/golden/passport_mrz.jsonl")

    def test_fixtures_load(self):
        assert len(self._fixtures()) == 2

    def test_mrz_detected_and_redacts_without_leakage(self):
        patterns = get_patterns_for_language("en")
        for fixture in self._fixtures():
            text = fixture.text
            gold = fixture.gold_spans[0]
            units = [
                u for u in find_semantic_units(text, patterns) if u[2] == "passport_mrz"
            ]
            assert units, fixture.fixture_id
            start, end = units[0][0], units[0][1]
            # The detected span covers the full gold MRZ block.
            assert start <= gold.start and end >= gold.end
            # Redacting the detected span leaves no residual MRZ characters.
            redacted = text[:start] + "[ID_NUM]" + text[end:]
            assert gold.text not in redacted
            assert "<<<" not in redacted  # MRZ filler runs are gone
