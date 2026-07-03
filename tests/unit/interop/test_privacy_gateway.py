from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from openmed.core.pii import reidentify
from openmed.interop import (
    PrivacyGateway,
    RedactionMapping,
    assert_redacted,
)
from openmed.interop import (
    gateway as gateway_module,
)

ORIGINAL_TEXT = "Patient Casey Example can be reached at casey@example.test."
REDACTED_TEXT = "Patient [NAME] can be reached at [EMAIL]."
MAPPING = {
    "[NAME]": "Casey Example",
    "[EMAIL]": "casey@example.test",
}


def fake_deidentify(text: str, **kwargs):
    assert text == ORIGINAL_TEXT
    assert kwargs["method"] == "mask"
    assert kwargs["keep_mapping"] is True
    assert kwargs["audit"] is False
    return SimpleNamespace(deidentified_text=REDACTED_TEXT, mapping=MAPPING)


def test_guarded_passes_only_redacted_text_and_restores_response():
    gateway = PrivacyGateway(deidentifier=fake_deidentify)
    seen_by_external: list[str] = []

    def external_call(text: str) -> str:
        seen_by_external.append(text)
        assert "Casey Example" not in text
        assert "casey@example.test" not in text
        return "Send a portal message to [NAME] at [EMAIL]."

    guarded = gateway.guarded(external_call)

    result = guarded(ORIGINAL_TEXT)

    assert seen_by_external == [REDACTED_TEXT]
    assert result == "Send a portal message to Casey Example at casey@example.test."


def test_redact_restore_round_trip_reproduces_original_identifiers():
    gateway = PrivacyGateway(deidentifier=fake_deidentify)

    clean_text, mapping = gateway.redact(ORIGINAL_TEXT)
    restored = gateway.restore(clean_text, mapping)

    assert clean_text == REDACTED_TEXT
    assert isinstance(mapping, RedactionMapping)
    assert mapping["[NAME]"] == "Casey Example"
    assert restored == ORIGINAL_TEXT


def test_mapping_representation_and_gateway_logs_do_not_expose_identifiers(caplog):
    caplog.set_level(logging.DEBUG)
    gateway = PrivacyGateway(deidentifier=fake_deidentify)

    _, mapping = gateway.redact(ORIGINAL_TEXT)
    logging.getLogger(__name__).warning("mapping=%s", mapping)
    logging.getLogger(__name__).warning("mapping=%r", mapping)

    assert "Casey Example" not in repr(mapping)
    assert "casey@example.test" not in repr(mapping)
    assert "Casey Example" not in caplog.text
    assert "casey@example.test" not in caplog.text


def test_input_guardrail_rejects_known_identifier_leak_without_echoing_phi():
    mapping = RedactionMapping(MAPPING)

    with pytest.raises(ValueError) as exc_info:
        assert_redacted("Patient Casey Example reached out.", mapping)

    message = str(exc_info.value)
    assert "Casey Example" not in message
    assert "casey@example.test" not in message
    assert "1 original identifier" in message


def test_input_and_output_guardrail_bindings_are_composable():
    gateway = PrivacyGateway(deidentifier=fake_deidentify, reidentifier=reidentify)
    clean_text, mapping = gateway.redact(ORIGINAL_TEXT)

    protected_prompt = gateway.input_guardrail(mapping)(clean_text)
    external_response = protected_prompt.replace("Patient", "Summary for")
    restored = gateway.output_guardrail(mapping)(external_response)

    assert protected_prompt == REDACTED_TEXT
    assert restored == "Summary for Casey Example can be reached at casey@example.test."


def test_gateway_exports_are_available_without_adapter_registry_churn():
    assert gateway_module.PrivacyGateway is PrivacyGateway
