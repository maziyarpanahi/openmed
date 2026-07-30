"""Tests for the shared input-normalization and validation gateway.

The gateway in :mod:`openmed.utils.validation` is the single path all three
entry points (library, REST schemas, MCP handlers) use to normalize and
validate request input. These tests exercise each guardrail directly and then
confirm the REST and MCP surfaces route through it.

All inputs are synthetic and generated algorithmically -- no real PHI.
"""

from __future__ import annotations

import pytest

from openmed.utils.validation import (
    DEFAULT_MAX_TEXT_BYTES,
    InputValidationError,
    get_max_text_bytes,
    validate_input,
    validate_language,
    validate_text_input,
)


class TestTypedError:
    def test_is_value_error_subclass(self):
        # Backward compatibility: existing ``except ValueError`` keeps working.
        assert issubclass(InputValidationError, ValueError)

    def test_carries_code_and_metadata(self):
        err = InputValidationError("boom", code="text_too_long", max_length=5)
        assert err.code == "text_too_long"
        assert err.metadata == {"max_length": 5}
        assert str(err) == "boom"


class TestTextGuardrails:
    def test_valid_text_is_stripped(self):
        assert validate_text_input("  hello  ") == "hello"

    def test_none_rejected(self):
        with pytest.raises(InputValidationError) as exc:
            validate_text_input(None)
        assert exc.value.code == "text_required"

    def test_none_allowed_when_empty_ok(self):
        assert validate_text_input(None, allow_empty=True) == ""

    def test_empty_rejected(self):
        with pytest.raises(InputValidationError) as exc:
            validate_text_input("   ")
        assert exc.value.code == "text_empty"

    def test_too_short(self):
        with pytest.raises(InputValidationError) as exc:
            validate_text_input("hi", min_length=5)
        assert exc.value.code == "text_too_short"
        assert exc.value.metadata["min_length"] == 5

    def test_char_length_cap(self):
        with pytest.raises(InputValidationError) as exc:
            validate_text_input("a" * 11, max_length=10)
        assert exc.value.code == "text_too_long"
        assert exc.value.metadata["max_length"] == 10
        assert exc.value.metadata["length"] == 11

    def test_byte_size_cap(self):
        # Four-byte code points blow the byte cap while staying under any
        # reasonable character cap.
        text = "\U0001f600" * 4  # 4 emoji -> 16 UTF-8 bytes
        with pytest.raises(InputValidationError) as exc:
            validate_text_input(text, max_length=None, max_bytes=8)
        assert exc.value.code == "text_too_large"
        assert exc.value.metadata["max_bytes"] == 8
        assert exc.value.metadata["byte_size"] == 16

    def test_none_caps_disable_limits(self):
        big = "a" * 5000
        assert validate_text_input(big, max_length=None, max_bytes=None) == big


class TestEncodingGuardrails:
    def test_unpaired_surrogate_rejected(self):
        with pytest.raises(InputValidationError) as exc:
            validate_text_input("valid\ud800text")
        assert exc.value.code == "invalid_encoding"

    def test_invalid_utf8_bytes_rejected(self):
        with pytest.raises(InputValidationError) as exc:
            validate_text_input(b"\xff\xfe invalid")
        assert exc.value.code == "invalid_encoding"

    def test_valid_utf8_bytes_decoded(self):
        assert validate_text_input("café".encode("utf-8")) == "café"


class TestLanguageGuardrails:
    def test_supported_language_passes(self):
        assert validate_language("en") == "en"

    def test_unsupported_language_rejected(self):
        with pytest.raises(InputValidationError) as exc:
            validate_language("zz")
        assert exc.value.code == "unsupported_language"
        assert "en" in exc.value.metadata["supported"]

    def test_explicit_accepted_set(self):
        assert validate_language("fr", accepted={"fr", "de"}) == "fr"
        with pytest.raises(InputValidationError):
            validate_language("en", accepted={"fr", "de"})

    def test_national_id_only_can_be_excluded(self):
        # ``ur`` is a pattern-only national-ID language: accepted by default,
        # rejected when national-ID languages are excluded.
        assert validate_language("ur") == "ur"
        with pytest.raises(InputValidationError):
            validate_language("ur", include_national_id=False)


class TestNoRawPhiInErrors:
    def test_error_never_echoes_input_text(self):
        phi = "MRN-9988776655-JohnDoe"
        payload = phi * 500  # exceed the default character/byte cap
        with pytest.raises(InputValidationError) as exc:
            validate_text_input(payload, max_length=100)
        rendered = str(exc.value) + repr(exc.value.metadata)
        assert phi not in rendered


class TestConfiguredCaps:
    def test_default_byte_cap(self):
        assert get_max_text_bytes() == DEFAULT_MAX_TEXT_BYTES

    def test_byte_cap_env_override(self, monkeypatch):
        monkeypatch.setenv("OPENMED_SERVICE_MAX_TEXT_BYTES", "123")
        assert get_max_text_bytes() == 123

    def test_byte_cap_env_invalid_falls_back(self, monkeypatch):
        monkeypatch.setenv("OPENMED_SERVICE_MAX_TEXT_BYTES", "not-a-number")
        assert get_max_text_bytes() == DEFAULT_MAX_TEXT_BYTES


class TestValidateInputWrapper:
    def test_delegates_and_keeps_messages(self):
        assert validate_input("  ok  ") == "ok"
        with pytest.raises(ValueError, match="Input text too long"):
            validate_input("a" * 200, max_length=100)

    def test_suspicious_content_still_enforced(self):
        with pytest.raises(InputValidationError) as exc:
            validate_input("!" * 50)
        assert exc.value.code == "suspicious_content"


class TestRestRoutesThroughGateway:
    def test_normalize_text_uses_gateway(self):
        from openmed.service.schemas import _normalize_text

        # Surrogate rejection is a gateway-only behavior the old REST
        # normalizer lacked -- proves REST now routes through it.
        with pytest.raises(InputValidationError) as exc:
            _normalize_text("bad\ud800text")
        assert exc.value.code == "invalid_encoding"

    def test_pii_extract_schema_rejects_bad_encoding(self):
        from openmed.service.schemas import PIIExtractRequest

        with pytest.raises(Exception):
            PIIExtractRequest(text="bad\ud800text")


class TestMcpRoutesThroughGateway:
    def test_list_models_language_uses_gateway(self):
        from openmed.mcp.server import openmed_list_models

        with pytest.raises(InputValidationError) as exc:
            openmed_list_models(pii_language="zz")
        assert exc.value.code == "unsupported_language"
