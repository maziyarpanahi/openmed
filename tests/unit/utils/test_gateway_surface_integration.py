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

    def test_extract_pii_language_guardrail_uses_gateway(self):
        from openmed.core.pii import extract_pii

        # ``_resolve_effective_pii_model`` now validates the language through the
        # shared gateway, preserving the canonical "Unsupported language" error.
        with pytest.raises(ValueError, match="Unsupported language"):
            extract_pii("test", lang="ko")

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

    def test_pii_extract_request_rejects_control_characters(self):
        from openmed.service.schemas import PIIExtractRequest

        with pytest.raises(ValueError):
            PIIExtractRequest(
                text="clinical note\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            )

    def test_pii_extract_request_rejects_invalid_utf8_bytes(self):
        from openmed.service.schemas import PIIExtractRequest

        with pytest.raises(ValueError):
            PIIExtractRequest(text=b"\xff\xfe")

    def test_analyze_request_accepts_clean_text(self):
        from openmed.service.schemas import AnalyzeRequest

        req = AnalyzeRequest(text="  Patient has asthma.  ")
        assert req.text == "Patient has asthma."

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

    def test_error_from_mcp_does_not_leak_input_text(self):
        from openmed.mcp.server import openmed_extract_pii

        secret = "SSN 123-45-6789 " + "x" * 10_000_000
        with pytest.raises(InputValidationError) as exc:
            openmed_extract_pii(
                secret,
                runtime_provider=_exploding_runtime_provider(),
            )
        assert "123-45-6789" not in str(exc.value)
