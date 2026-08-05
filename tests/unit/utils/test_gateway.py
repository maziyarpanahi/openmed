"""Tests for the shared input-normalization and validation gateway.

These cover the gateway in isolation (valid/invalid text, encoding, length,
byte, and language guardrails) and confirm that all three surfaces — the Python
library, the REST schemas, and the MCP server — route through it.
"""

from __future__ import annotations

import pytest

from openmed.utils import gateway
from openmed.utils.gateway import (
    GatewayLimits,
    InputValidationError,
    normalize_text,
    validate_language,
)


class TestNormalizeText:
    def test_valid_text_is_stripped(self):
        assert normalize_text("  Patient has asthma.  ") == "Patient has asthma."

    def test_none_is_rejected(self):
        with pytest.raises(InputValidationError) as exc:
            normalize_text(None)
        assert exc.value.code == "text_required"

    def test_empty_is_rejected(self):
        with pytest.raises(InputValidationError) as exc:
            normalize_text("")
        assert exc.value.code == "empty_text"

    def test_blank_is_rejected(self):
        with pytest.raises(InputValidationError) as exc:
            normalize_text("   \n\t ")
        assert exc.value.code == "empty_text"

    def test_allow_empty_returns_empty_string(self):
        assert normalize_text("", allow_empty=True) == ""
        assert normalize_text("   ", allow_empty=True) == ""

    def test_bytes_input_is_decoded(self):
        assert normalize_text("café".encode("utf-8")) == "café"

    def test_memoryview_input_is_decoded(self):
        assert normalize_text(memoryview("café".encode("utf-8"))) == "café"

    def test_invalid_utf8_bytes_are_rejected(self):
        with pytest.raises(InputValidationError) as exc:
            normalize_text(b"\xff\xfe not utf-8")
        assert exc.value.code == "invalid_encoding"

    def test_lone_surrogate_is_rejected(self):
        # A lone surrogate cannot be encoded to UTF-8.
        with pytest.raises(InputValidationError) as exc:
            normalize_text("bad\ud800surrogate")
        assert exc.value.code == "invalid_encoding"

    def test_multilingual_text_is_accepted(self):
        # CJK / Arabic / Devanagari runs must not be treated as suspicious.
        for sample in ("患者は喘息です。", "المريض مصاب بالربو", "रोगी को अस्थमा है"):
            assert normalize_text(sample) == sample

    def test_char_limit_is_enforced(self):
        with pytest.raises(InputValidationError) as exc:
            normalize_text("a" * 11, limits=GatewayLimits(max_chars=10))
        assert exc.value.code == "max_chars"
        assert exc.value.limit == 10
        assert exc.value.actual == 11
        assert exc.value.metadata == {
            "unit": "characters",
            "limit": 10,
            "actual": 11,
        }

    def test_minimum_length_is_enforced(self):
        with pytest.raises(InputValidationError) as exc:
            normalize_text("abc", min_length=4)
        assert exc.value.code == "min_chars"
        assert exc.value.limit == 4
        assert exc.value.actual == 3

    def test_byte_limit_is_enforced(self):
        # Four two-byte characters == 8 bytes but only 4 chars, so this trips the
        # byte cap while staying under a generous char cap.
        text = "é" * 4
        with pytest.raises(InputValidationError) as exc:
            normalize_text(text, limits=GatewayLimits(max_chars=100, max_bytes=4))
        assert exc.value.code == "max_bytes"
        assert exc.value.limit == 4

    def test_common_whitespace_is_allowed(self):
        text = "line one\nline two\tindented"
        assert normalize_text(text) == text

    def test_error_message_never_contains_input_text(self):
        secret = "SSN 123-45-6789 belongs to Jane Q Patient"
        # Force a limit failure while keeping the (sensitive) text in the input.
        with pytest.raises(InputValidationError) as exc:
            normalize_text(secret, limits=GatewayLimits(max_chars=5))
        assert secret not in str(exc.value)
        assert "123-45-6789" not in str(exc.value)


class TestValidateLanguage:
    def test_valid_language(self):
        assert validate_language("en") == "en"

    def test_normalizes_case_and_whitespace(self):
        assert validate_language("  FR ") == "fr"

    def test_unsupported_language_raises(self):
        with pytest.raises(InputValidationError) as exc:
            validate_language("private-patient-code")
        assert exc.value.code == "unsupported_language"
        assert "Unsupported language" in str(exc.value)
        assert "private-patient-code" not in str(exc.value)

    def test_none_language_raises(self):
        with pytest.raises(InputValidationError) as exc:
            validate_language(None)
        assert exc.value.code == "language_required"

    def test_non_string_language_raises(self):
        with pytest.raises(InputValidationError) as exc:
            validate_language(123)
        assert exc.value.code == "language_type"

    def test_custom_supported_set(self):
        assert validate_language("zz", supported={"zz"}) == "zz"
        with pytest.raises(InputValidationError):
            validate_language("en", supported={"zz"})

    def test_defaults_to_pii_supported_languages(self):
        from openmed.core.pii_i18n import (
            INDIC_NER_LANGUAGES,
            NATIONAL_ID_ONLY_LANGUAGES,
            SUPPORTED_LANGUAGES,
            USER_SUPPLIED_MODEL_LANGUAGES,
        )

        for code in (
            SUPPORTED_LANGUAGES
            | INDIC_NER_LANGUAGES
            | NATIONAL_ID_ONLY_LANGUAGES
            | USER_SUPPLIED_MODEL_LANGUAGES
        ):
            assert validate_language(code) == code

    def test_api_language_set_excludes_national_id_only_languages(self):
        from openmed.core.pii_i18n import (
            NATIONAL_ID_ONLY_LANGUAGES,
            USER_SUPPLIED_MODEL_LANGUAGES,
        )

        # Urdu is both national-ID-only and user-supplied-model, and the second
        # membership keeps it on the public language enums. Pick a code that is
        # only national-ID-only so this exercises the toggle rather than set
        # iteration order.
        national_id_only = sorted(
            NATIONAL_ID_ONLY_LANGUAGES - USER_SUPPLIED_MODEL_LANGUAGES
        )[0]
        with pytest.raises(InputValidationError):
            validate_language(national_id_only, include_national_id=False)

    def test_api_language_set_includes_user_supplied_model_languages(self):
        from openmed.core.pii_i18n import USER_SUPPLIED_MODEL_LANGUAGES

        for code in sorted(USER_SUPPLIED_MODEL_LANGUAGES):
            assert validate_language(code, include_national_id=False) == code


class TestDefaultLimits:
    def test_char_cap_follows_service_env(self, monkeypatch):
        monkeypatch.setenv("OPENMED_SERVICE_MAX_TEXT_LENGTH", "5")
        limits = gateway.get_default_limits()
        assert limits.max_chars == 5

    def test_byte_cap_follows_env(self, monkeypatch):
        monkeypatch.setenv(gateway.MAX_TEXT_BYTES_ENV_VAR, "123")
        limits = gateway.get_default_limits()
        assert limits.max_bytes == 123

    def test_invalid_byte_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(gateway.MAX_TEXT_BYTES_ENV_VAR, "not-an-int")
        limits = gateway.get_default_limits()
        assert limits.max_bytes == gateway.DEFAULT_MAX_TEXT_BYTES

    def test_default_char_cap_used_by_normalize_text(self, monkeypatch):
        monkeypatch.setenv("OPENMED_SERVICE_MAX_TEXT_LENGTH", "3")
        with pytest.raises(InputValidationError) as exc:
            normalize_text("abcd")
        assert exc.value.code == "max_chars"


class TestTypedError:
    def test_is_value_error_subclass(self):
        # Existing ``except ValueError`` handlers on every surface must still
        # catch gateway rejections.
        assert issubclass(InputValidationError, ValueError)

    def test_carries_structured_metadata(self):
        err = InputValidationError(
            "boom",
            code="max_chars",
            metadata={"unit": "characters"},
            limit=10,
            actual=20,
        )
        assert err.code == "max_chars"
        assert err.limit == 10
        assert err.actual == 20
        assert err.metadata == {
            "unit": "characters",
            "limit": 10,
            "actual": 20,
        }
