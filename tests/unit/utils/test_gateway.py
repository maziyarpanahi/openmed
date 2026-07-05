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
    NormalizedInput,
    ensure_valid_encoding,
    normalize_input,
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

    def test_byte_limit_is_enforced(self):
        # Four two-byte characters == 8 bytes but only 4 chars, so this trips the
        # byte cap while staying under a generous char cap.
        text = "é" * 4
        with pytest.raises(InputValidationError) as exc:
            normalize_text(text, limits=GatewayLimits(max_chars=100, max_bytes=4))
        assert exc.value.code == "max_bytes"
        assert exc.value.limit == 4

    def test_control_character_run_is_rejected(self):
        with pytest.raises(InputValidationError) as exc:
            normalize_text("ok" + "\x00" * 12)
        assert exc.value.code == "control_characters"

    def test_single_control_character_is_rejected(self):
        with pytest.raises(InputValidationError) as exc:
            normalize_text("hello\x07world")
        assert exc.value.code == "control_characters"

    def test_common_whitespace_is_allowed(self):
        text = "line one\nline two\tindented"
        assert normalize_text(text) == text

    def test_abusive_repeated_run_is_rejected(self):
        with pytest.raises(InputValidationError) as exc:
            normalize_text("a" * 200)
        assert exc.value.code == "repeated_characters"

    def test_error_message_never_contains_input_text(self):
        secret = "SSN 123-45-6789 belongs to Jane Q Patient"
        # Force a limit failure while keeping the (sensitive) text in the input.
        with pytest.raises(InputValidationError) as exc:
            normalize_text(secret, limits=GatewayLimits(max_chars=5))
        assert secret not in str(exc.value)
        assert "123-45-6789" not in str(exc.value)


class TestEnsureValidEncoding:
    def test_passes_valid_text(self):
        assert ensure_valid_encoding("hello") == "hello"

    def test_decodes_bytes(self):
        assert ensure_valid_encoding("café".encode("utf-8")) == "café"

    def test_rejects_none(self):
        with pytest.raises(InputValidationError):
            ensure_valid_encoding(None)

    def test_rejects_invalid_bytes(self):
        with pytest.raises(InputValidationError) as exc:
            ensure_valid_encoding(b"\xff\xfe")
        assert exc.value.code == "invalid_encoding"


class TestValidateLanguage:
    def test_valid_language(self):
        assert validate_language("en") == "en"

    def test_normalizes_case_and_whitespace(self):
        assert validate_language("  FR ") == "fr"

    def test_unsupported_language_raises(self):
        with pytest.raises(InputValidationError) as exc:
            validate_language("xx")
        assert exc.value.code == "unsupported_language"
        # The canonical library message contract is preserved.
        assert "Unsupported language" in str(exc.value)

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
        from openmed.core.pii_i18n import SUPPORTED_LANGUAGES

        for code in SUPPORTED_LANGUAGES:
            assert validate_language(code) == code


class TestNormalizeInput:
    def test_returns_normalized_text_and_language(self):
        result = normalize_input("  hello  ", lang="EN")
        assert isinstance(result, NormalizedInput)
        assert result.text == "hello"
        assert result.language == "en"

    def test_language_optional(self):
        result = normalize_input("hello")
        assert result.language is None

    def test_propagates_text_failure(self):
        with pytest.raises(InputValidationError):
            normalize_input("", lang="en")

    def test_propagates_language_failure(self):
        with pytest.raises(InputValidationError):
            normalize_input("hello", lang="xx")


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
        err = InputValidationError("boom", code="max_chars", limit=10, actual=20)
        assert err.code == "max_chars"
        assert err.limit == 10
        assert err.actual == 20
