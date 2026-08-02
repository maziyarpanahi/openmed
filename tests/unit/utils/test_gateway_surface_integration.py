"""Confirm the library, REST, and MCP surfaces all route through the gateway.

The gateway (``openmed.utils.gateway``) is the single input-normalization and
validation path. These tests assert that each of the three entry points rejects
the same malformed inputs by delegating to it, rather than re-validating
ad-hoc.
"""

from __future__ import annotations

import pytest

from openmed.utils.gateway import InputValidationError


# ---------------------------------------------------------------------------
# Library surface
# ---------------------------------------------------------------------------
class TestLibrarySurface:
    def test_validate_input_delegates_to_gateway(self, monkeypatch):
        from openmed.utils import gateway, validation

        captured = {}

        def fake_normalize_text(value, **kwargs):
            captured["value"] = value
            captured["kwargs"] = kwargs
            return "normalized text"

        monkeypatch.setattr(gateway, "normalize_text", fake_normalize_text)

        assert validation.validate_input("raw text") == "normalized text"
        assert captured["value"] == "raw text"
        assert captured["kwargs"]["min_length"] == 1

    def test_validate_input_rejects_invalid_utf8_bytes(self):
        from openmed.utils.validation import validate_input

        # The library entry point delegates encoding validation to the gateway.
        with pytest.raises(ValueError):
            validate_input(b"\xff\xfe not utf-8")

    def test_validate_input_rejects_lone_surrogate(self):
        from openmed.utils.validation import validate_input

        with pytest.raises(ValueError):
            validate_input("bad\ud800surrogate")

    def test_validate_input_still_strips_and_accepts_clean_text(self):
        from openmed.utils.validation import validate_input

        assert validate_input("  hello  ") == "hello"

    def test_validate_input_enforces_configured_byte_cap(self, monkeypatch):
        from openmed.utils.validation import validate_input

        monkeypatch.setenv("OPENMED_MAX_TEXT_BYTES", "5")
        with pytest.raises(InputValidationError) as exc:
            validate_input("é" * 3)
        assert exc.value.code == "max_bytes"

    def test_extract_pii_language_guardrail_uses_gateway(self):
        from openmed.core.pii import extract_pii

        # ``_resolve_effective_pii_model`` now validates the language through the
        # shared gateway, preserving the canonical "Unsupported language" error.
        with pytest.raises(InputValidationError, match="Unsupported language"):
            extract_pii("test", lang="xx")

    def test_resolve_effective_pii_model_uses_gateway(self, monkeypatch):
        import openmed.utils.gateway as gateway
        from openmed.core.pii import _resolve_effective_pii_model

        called = {}
        real = gateway.validate_language

        def spy(lang, **kwargs):
            called["lang"] = lang
            return real(lang, **kwargs)

        # The function imports ``validate_language`` from the gateway module at
        # call time, so patching the source module is picked up.
        monkeypatch.setattr(gateway, "validate_language", spy)

        result = _resolve_effective_pii_model("some/model", "en")
        assert result == "some/model"
        assert called["lang"] == "en"


# ---------------------------------------------------------------------------
# REST surface
# ---------------------------------------------------------------------------
class TestRestSurface:
    def test_schemas_normalize_text_delegates_to_gateway(self, monkeypatch):
        from openmed.service import schemas

        sentinel = object()
        captured = {}

        def fake_normalize_text(value, **kwargs):
            captured["value"] = value
            return "NORMALIZED"

        monkeypatch.setattr(schemas, "normalize_text", fake_normalize_text)
        assert schemas._normalize_text(sentinel) == "NORMALIZED"
        assert captured["value"] is sentinel

    def test_analyze_request_rejects_blank_text(self):
        from openmed.service.schemas import AnalyzeRequest

        with pytest.raises(ValueError):
            AnalyzeRequest(text="   ")

    def test_pii_extract_request_enforces_byte_cap(self, monkeypatch):
        from openmed.service.schemas import PIIExtractRequest

        monkeypatch.setenv("OPENMED_MAX_TEXT_BYTES", "5")
        with pytest.raises(ValueError):
            PIIExtractRequest(text="é" * 3)

    def test_rest_validation_messages_do_not_echo_text(self, monkeypatch):
        from openmed.service.schemas import AnalyzeRequest

        monkeypatch.setenv("OPENMED_SERVICE_MAX_TEXT_LENGTH", "5")
        secret = "Synthetic MRN 9988776655"
        with pytest.raises(ValueError) as exc:
            AnalyzeRequest(text=secret)

        messages = [str(error.get("msg", "")) for error in exc.value.errors()]
        assert all(secret not in message for message in messages)
        assert all("9988776655" not in message for message in messages)

    def test_pii_extract_request_rejects_invalid_utf8_bytes(self):
        from openmed.service.schemas import PIIExtractRequest

        with pytest.raises(ValueError):
            PIIExtractRequest(text=b"\xff\xfe")

    def test_analyze_request_accepts_clean_text(self):
        from openmed.service.schemas import AnalyzeRequest

        req = AnalyzeRequest(text="  Patient has asthma.  ")
        assert req.text == "Patient has asthma."

    def test_pii_language_delegates_to_gateway(self, monkeypatch):
        from openmed.service import schemas

        captured = {}
        real_validate_language = schemas.validate_language

        def spy(value, **kwargs):
            captured["value"] = value
            captured["kwargs"] = kwargs
            return real_validate_language(value, **kwargs)

        monkeypatch.setattr(schemas, "validate_language", spy)

        request = schemas.PIIExtractRequest(text="synthetic note", lang="EN")
        assert request.lang == "en"
        assert captured == {
            "value": "EN",
            "kwargs": {"include_national_id": False},
        }

    def test_char_cap_honours_service_env(self, monkeypatch):
        monkeypatch.setenv("OPENMED_SERVICE_MAX_TEXT_LENGTH", "5")
        from openmed.service.schemas import AnalyzeRequest

        with pytest.raises(ValueError):
            AnalyzeRequest(text="x" * 6)


# ---------------------------------------------------------------------------
# MCP surface
# ---------------------------------------------------------------------------
def _exploding_runtime_provider():
    """A runtime provider that fails if the gateway did not short-circuit."""

    def _provider():
        raise AssertionError(
            "runtime must not be reached; gateway should reject input first"
        )

    return _provider


class TestMcpSurface:
    def test_analyze_rejects_blank_text_before_runtime(self):
        from openmed.mcp.server import openmed_analyze_text

        with pytest.raises(InputValidationError):
            openmed_analyze_text(
                "   ",
                runtime_provider=_exploding_runtime_provider(),
            )

    def test_analyze_rejects_invalid_utf8_before_runtime(self):
        from openmed.mcp.server import openmed_analyze_text

        with pytest.raises(InputValidationError):
            openmed_analyze_text(
                b"\xff\xfe",
                runtime_provider=_exploding_runtime_provider(),
            )

    def test_extract_pii_rejects_blank_text_before_runtime(self):
        from openmed.mcp.server import openmed_extract_pii

        with pytest.raises(InputValidationError):
            openmed_extract_pii(
                "   ",
                runtime_provider=_exploding_runtime_provider(),
            )

    def test_extract_pii_rejects_unsupported_language_before_runtime(self):
        from openmed.mcp.server import openmed_extract_pii

        with pytest.raises(InputValidationError):
            openmed_extract_pii(
                "Patient John Doe",
                lang="xx",
                runtime_provider=_exploding_runtime_provider(),
            )

    def test_deidentify_rejects_blank_text_before_runtime(self):
        from openmed.mcp.server import openmed_deidentify

        with pytest.raises(InputValidationError):
            openmed_deidentify(
                "   ",
                runtime_provider=_exploding_runtime_provider(),
            )

    def test_deidentify_rejects_unsupported_language_before_runtime(self):
        from openmed.mcp.server import openmed_deidentify

        with pytest.raises(InputValidationError):
            openmed_deidentify(
                "Patient John Doe",
                lang="xx",
                runtime_provider=_exploding_runtime_provider(),
            )

    def test_list_models_rejects_unsupported_language_through_gateway(self):
        from openmed.mcp.server import openmed_list_models

        secret = "synthetic-private-language-code"
        with pytest.raises(InputValidationError) as exc:
            openmed_list_models(pii_language=secret)
        assert secret not in str(exc.value)

    def test_signed_audit_rejects_blank_text_before_runtime(self):
        from openmed.mcp.server import openmed_signed_audit_report

        with pytest.raises(InputValidationError):
            openmed_signed_audit_report(
                "   ",
                signing_key="synthetic-signing-key",
                runtime_provider=_exploding_runtime_provider(),
            )

    def test_signed_audit_rejects_language_before_runtime(self):
        from openmed.mcp.server import openmed_signed_audit_report

        with pytest.raises(InputValidationError):
            openmed_signed_audit_report(
                "Synthetic patient note",
                lang="xx",
                signing_key="synthetic-signing-key",
                runtime_provider=_exploding_runtime_provider(),
            )

    def test_error_from_mcp_does_not_leak_input_text(self, monkeypatch):
        from openmed.mcp.server import openmed_extract_pii

        monkeypatch.setenv("OPENMED_SERVICE_MAX_TEXT_LENGTH", "5")
        secret = "Synthetic MRN 9988776655"
        with pytest.raises(InputValidationError) as exc:
            openmed_extract_pii(
                secret,
                runtime_provider=_exploding_runtime_provider(),
            )
        assert "9988776655" not in str(exc.value)
