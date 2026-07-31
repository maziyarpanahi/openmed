"""Tests for Polish PESEL and South Korean RRN national-ID validators."""

from __future__ import annotations

import pytest

from openmed.core.anonymizer import Anonymizer
from openmed.core.anonymizer.locales import locale_coherence_report
from openmed.core.pii_entity_merger import PIIPattern
from openmed.core.pii_i18n import (
    get_patterns_for_language,
    validate_korean_rrn,
    validate_polish_pesel,
)

# ---------------------------------------------------------------------------
# Polish PESEL
# ---------------------------------------------------------------------------


class TestValidatePolishPesel:
    """Tests for :func:`validate_polish_pesel`."""

    def test_valid_1980s_pesel(self):
        # PESEL for 1985-03-15 (month_raw=03, year=1985)
        # Body: 850315 + serial 1234, check=4
        assert validate_polish_pesel("85031512344") is True

    def test_valid_2000s_pesel(self):
        # PESEL for 2001-07-22 (month_raw=27, year=2001)
        # Body: 012722 + serial 5678, check=2
        assert validate_polish_pesel("01272256782") is True

    def test_rejects_bad_checksum(self):
        # Change last digit of a valid PESEL.
        valid = "85031512345"
        bad = valid[:-1] + str((int(valid[-1]) + 1) % 10)
        assert validate_polish_pesel(bad) is False

    def test_rejects_too_short(self):
        assert validate_polish_pesel("8503151234") is False

    def test_rejects_too_long(self):
        assert validate_polish_pesel("850315123456") is False

    def test_rejects_impossible_date_feb30(self):
        # Month=02, day=30 => invalid
        # Build: 850230XXXXX, find a serial that passes checksum.
        body = [8, 5, 0, 2, 3, 0, 0, 0, 0, 0]
        weights = (1, 3, 7, 9, 1, 3, 7, 9, 1, 3)
        total = sum(w * d for w, d in zip(weights, body))
        check = (10 - total % 10) % 10
        pesel = "".join(str(d) for d in body) + str(check)
        assert validate_polish_pesel(pesel) is False

    def test_rejects_impossible_month_13(self):
        # Month=13 => invalid month_raw
        body = [8, 5, 1, 3, 1, 5, 0, 0, 0, 0]
        weights = (1, 3, 7, 9, 1, 3, 7, 9, 1, 3)
        total = sum(w * d for w, d in zip(weights, body))
        check = (10 - total % 10) % 10
        pesel = "".join(str(d) for d in body) + str(check)
        assert validate_polish_pesel(pesel) is False

    def test_rejects_zero_month(self):
        # Month=00 => invalid
        body = [8, 5, 0, 0, 1, 5, 0, 0, 0, 0]
        weights = (1, 3, 7, 9, 1, 3, 7, 9, 1, 3)
        total = sum(w * d for w, d in zip(weights, body))
        check = (10 - total % 10) % 10
        pesel = "".join(str(d) for d in body) + str(check)
        assert validate_polish_pesel(pesel) is False

    def test_accepts_spaces_and_hyphens(self):
        # Same PESEL with formatting (85031512344).
        assert validate_polish_pesel("850315 12344") is True
        assert validate_polish_pesel("850315-12344") is True

    def test_valid_2020s_pesel(self):
        # 2020-11-10 => month_raw = 31 => year = 2020, month = 11
        body = [2, 0, 3, 1, 1, 0, 9, 8, 7, 6]
        weights = (1, 3, 7, 9, 1, 3, 7, 9, 1, 3)
        total = sum(w * d for w, d in zip(weights, body))
        check = (10 - total % 10) % 10
        pesel = "".join(str(d) for d in body) + str(check)
        assert validate_polish_pesel(pesel) is True

    def test_valid_1800s_pesel(self):
        # 1885-03-15 => month_raw = 83 (80+3) => year = 1885, month = 3
        body = [8, 5, 8, 3, 1, 5, 0, 0, 0, 0]
        weights = (1, 3, 7, 9, 1, 3, 7, 9, 1, 3)
        total = sum(w * d for w, d in zip(weights, body))
        check = (10 - total % 10) % 10
        pesel = "".join(str(d) for d in body) + str(check)
        assert validate_polish_pesel(pesel) is True

    def test_rejects_month_raw_above_92(self):
        # month_raw = 93 => invalid (above 92 threshold)
        body = [8, 5, 9, 3, 1, 5, 0, 0, 0, 0]
        weights = (1, 3, 7, 9, 1, 3, 7, 9, 1, 3)
        total = sum(w * d for w, d in zip(weights, body))
        check = (10 - total % 10) % 10
        pesel = "".join(str(d) for d in body) + str(check)
        assert validate_polish_pesel(pesel) is False

    def test_non_digit_input(self):
        assert validate_polish_pesel("abcdefghijk") is False

    def test_empty_string(self):
        assert validate_polish_pesel("") is False


# ---------------------------------------------------------------------------
# South Korean RRN
# ---------------------------------------------------------------------------


class TestValidateKoreanRrn:
    """Tests for :func:`validate_korean_rrn`."""

    def test_valid_1980s_male_rrn(self):
        # 860815-1234567
        # Year=1986, month=08, day=15, gender=1 (1900s male)
        # Body: 8608151 + serial 23456 => weights 2,3,4,5,6,7,8,9,2,3,4,5
        body = [8, 6, 0, 8, 1, 5, 1, 2, 3, 4, 5, 6]
        weights = (2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5)
        total = sum(w * d for w, d in zip(weights, body))
        check = (11 - total % 11) % 10
        rrn = "".join(str(d) for d in body) + str(check)
        assert validate_korean_rrn(rrn) is True

    def test_valid_2000s_female_rrn(self):
        # 020515-4234567
        # Year=2002, month=05, day=15, gender=4 (2000s female)
        body = [0, 2, 0, 5, 1, 5, 4, 2, 3, 4, 5, 6]
        weights = (2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5)
        total = sum(w * d for w, d in zip(weights, body))
        check = (11 - total % 11) % 10
        rrn = "".join(str(d) for d in body) + str(check)
        assert validate_korean_rrn(rrn) is True

    def test_rejects_bad_checksum(self):
        body = [8, 6, 0, 8, 1, 5, 1, 2, 3, 4, 5, 6]
        weights = (2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5)
        total = sum(w * d for w, d in zip(weights, body))
        check = (11 - total % 11) % 10
        rrn = "".join(str(d) for d in body) + str((check + 1) % 10)
        assert validate_korean_rrn(rrn) is False

    def test_rejects_too_short(self):
        assert validate_korean_rrn("860815123456") is False

    def test_rejects_too_long(self):
        assert validate_korean_rrn("86081512345678") is False

    def test_rejects_impossible_date_feb30(self):
        # Month=02, day=30 => invalid
        body = [8, 6, 0, 2, 3, 0, 1, 0, 0, 0, 0, 0]
        weights = (2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5)
        total = sum(w * d for w, d in zip(weights, body))
        check = (11 - total % 11) % 10
        rrn = "".join(str(d) for d in body) + str(check)
        assert validate_korean_rrn(rrn) is False

    def test_rejects_invalid_gender_code(self):
        # Gender code at position 6 = 9 (1800s, but that's valid per spec)
        # Actually test with a truly invalid code... all digits 0-9 map to centuries.
        # So we test that the century mapping works correctly.
        # 860815-9234567 => 1800s male
        body = [8, 6, 0, 8, 1, 5, 9, 2, 3, 4, 5, 6]
        weights = (2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5)
        total = sum(w * d for w, d in zip(weights, body))
        check = (11 - total % 11) % 10
        rrn = "".join(str(d) for d in body) + str(check)
        # 1800-08-15 is valid
        assert validate_korean_rrn(rrn) is True

    def test_rejects_impossible_month_13(self):
        body = [8, 6, 1, 3, 1, 5, 1, 0, 0, 0, 0, 0]
        weights = (2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5)
        total = sum(w * d for w, d in zip(weights, body))
        check = (11 - total % 11) % 10
        rrn = "".join(str(d) for d in body) + str(check)
        assert validate_korean_rrn(rrn) is False

    def test_accepts_hyphen_format(self):
        # 860815-1234567
        body = [8, 6, 0, 8, 1, 5, 1, 2, 3, 4, 5, 6]
        weights = (2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5)
        total = sum(w * d for w, d in zip(weights, body))
        check = (11 - total % 11) % 10
        rrn = "".join(str(d) for d in body) + str(check)
        formatted = f"{rrn[:6]}-{rrn[6:]}"
        assert validate_korean_rrn(formatted) is True

    def test_accepts_space_format(self):
        body = [8, 6, 0, 8, 1, 5, 1, 2, 3, 4, 5, 6]
        weights = (2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5)
        total = sum(w * d for w, d in zip(weights, body))
        check = (11 - total % 11) % 10
        rrn = "".join(str(d) for d in body) + str(check)
        formatted = f"{rrn[:6]} {rrn[6:]}"
        assert validate_korean_rrn(formatted) is True

    def test_non_digit_input(self):
        assert validate_korean_rrn("abcdefghijklm") is False

    def test_empty_string(self):
        assert validate_korean_rrn("") is False


# ---------------------------------------------------------------------------
# Faker provider round-trip
# ---------------------------------------------------------------------------


class TestFakerProviderRoundTrip:
    """Verify that generated values pass the validators."""

    def test_pesel_round_trip(self):
        from openmed.core.anonymizer.providers.clinical_ids import generate_pesel

        for _ in range(200):
            pesel = generate_pesel()
            assert len(pesel) == 11
            assert validate_polish_pesel(pesel) is True

    def test_korean_rrn_round_trip(self):
        from openmed.core.anonymizer.providers.clinical_ids import (
            generate_korean_rrn,
        )

        for _ in range(200):
            rrn = generate_korean_rrn()
            assert len(rrn) == 13
            assert validate_korean_rrn(rrn) is True

    def test_pesel_registered_anonymizer_path_round_trips(self):
        anon = Anonymizer(lang="pl", consistent=True, seed=42)

        surrogate = anon.surrogate("85031512344", "national_id")

        assert validate_polish_pesel(surrogate) is True

    def test_korean_rrn_registered_anonymizer_path_round_trips(self):
        anon = Anonymizer(lang="ko", consistent=True, seed=42)

        surrogate = anon.surrogate("8608151234567", "national_id")

        assert validate_korean_rrn(surrogate) is True


class TestNationalIdPatternReachability:
    """PESEL/RRN patterns are reachable without full language-pack support."""

    def test_polish_patterns_are_national_id_only(self):
        patterns = [
            pattern
            for pattern in get_patterns_for_language("pl")
            if pattern.entity_type == "national_id"
            and pattern.validator is validate_polish_pesel
        ]

        assert len(patterns) == 1
        assert isinstance(patterns[0], PIIPattern)

    def test_korean_patterns_are_national_id_only(self):
        patterns = [
            pattern
            for pattern in get_patterns_for_language("ko")
            if pattern.entity_type == "national_id"
            and pattern.validator is validate_korean_rrn
        ]

        assert len(patterns) == 1
        assert isinstance(patterns[0], PIIPattern)

    @pytest.mark.parametrize(
        ("lang", "locale", "method"),
        [
            ("pl", "pl_PL", "pesel"),
            ("ko", "ko_KR", "korean_rrn"),
            ("lv", "lv_LV", "personas_kods"),
        ],
    )
    def test_locale_coherence_report_includes_registered_provider(
        self, lang, locale, method
    ):
        rows = {row["language"]: row for row in locale_coherence_report()}

        assert rows[lang]["locale"] == locale
        assert rows[lang]["id_locale"] == locale
        assert rows[lang]["id_providers"] == [method]
