"""Synthetic offline tests for MCP consent receipts and policy wiring."""

from __future__ import annotations

import inspect
import json
import logging
from copy import deepcopy
from typing import Any

import pytest

from openmed.mcp.consent_receipts import (
    ConsentReceiptBindingError,
    ConsentReceiptExpiredError,
    ConsentReceiptIssuer,
    ConsentReceiptPolicy,
    ConsentReceiptReplayError,
    ConsentReceiptRequiredError,
    ConsentReceiptVerificationResult,
    ConsentReceiptVerifier,
    MappingConsentKeyProvider,
    canonical_argument_digest,
)

_KEY = "synthetic-consent-signing-key"
_CLIENT = "synthetic-client"
_TOOL = "openmed_unload_model"
_RESOURCE = "synthetic-resource"
_SCOPE = "synthetic:models.write"
_ARGUMENTS = {
    "model_name": "synthetic-model",
    "all_models": False,
    "clinical_text": "synthetic clinical text must not be retained",
    "bearer": "synthetic-bearer-value",
}


def _issuer(clock: list[float]) -> ConsentReceiptIssuer:
    return ConsentReceiptIssuer(
        MappingConsentKeyProvider({"synthetic": _KEY}),
        key_id="synthetic",
        clock=lambda: clock[0],
        receipt_id_factory=lambda: "synthetic-receipt-001",
    )


def test_receipt_serialization_is_bound_and_phi_free() -> None:
    clock = [1_000.0]
    receipt = _issuer(clock).issue(
        _CLIENT,
        _TOOL,
        _RESOURCE,
        _SCOPE,
        _ARGUMENTS,
        ttl_seconds=60,
    )

    serialized = receipt.to_json()
    assert receipt.argument_digest == canonical_argument_digest(_ARGUMENTS)
    assert receipt.canonical_argument_digest == receipt.argument_digest
    assert receipt.to_dict()["signature"].startswith("hmac-sha256:")
    assert "synthetic clinical text" not in serialized
    assert "synthetic-bearer-value" not in serialized
    assert "model_name" not in serialized
    assert "arguments" not in receipt.to_dict()

    restored = receipt.from_json(serialized)
    assert restored == receipt


def test_verifier_accepts_once_and_rejects_binding_variants_and_replay() -> None:
    clock = [1_000.0]
    receipt = _issuer(clock).issue(
        _CLIENT,
        _TOOL,
        _RESOURCE,
        _SCOPE,
        _ARGUMENTS,
        ttl_seconds=60,
    )
    verifier = ConsentReceiptVerifier(
        MappingConsentKeyProvider({"synthetic": _KEY}),
        clock=lambda: clock[0],
    )

    with pytest.raises(ConsentReceiptBindingError):
        verifier.verify(
            receipt,
            _CLIENT,
            "openmed_run_workflow",
            _RESOURCE,
            _SCOPE,
            _ARGUMENTS,
        )
    with pytest.raises(ConsentReceiptBindingError):
        verifier.verify(
            receipt,
            _CLIENT,
            _TOOL,
            "synthetic-other-resource",
            _SCOPE,
            _ARGUMENTS,
        )

    assert (
        verifier.verify(
            receipt,
            _CLIENT,
            _TOOL,
            _RESOURCE,
            _SCOPE,
            _ARGUMENTS,
        )
        == receipt
    )
    with pytest.raises(ConsentReceiptReplayError):
        verifier.verify(
            receipt,
            _CLIENT,
            _TOOL,
            _RESOURCE,
            _SCOPE,
            _ARGUMENTS,
        )


def test_verifier_result_is_non_throwing_stable_and_single_use() -> None:
    clock = [1_000.0]
    receipt = _issuer(clock).issue(
        _CLIENT,
        _TOOL,
        _RESOURCE,
        _SCOPE,
        _ARGUMENTS,
        ttl_seconds=60,
    )
    verifier = ConsentReceiptVerifier(
        MappingConsentKeyProvider({"synthetic": _KEY}),
        clock=lambda: clock[0],
    )

    assert verifier.verify_result(
        None, _CLIENT, _TOOL, _RESOURCE, _SCOPE, _ARGUMENTS
    ) == ConsentReceiptVerificationResult(False, "missing_receipt")
    assert verifier.verify_result(
        receipt, _CLIENT, "other-tool", _RESOURCE, _SCOPE, _ARGUMENTS
    ) == ConsentReceiptVerificationResult(False, "binding_mismatch")

    success = verifier.verify_result(
        receipt, _CLIENT, _TOOL, _RESOURCE, _SCOPE, _ARGUMENTS
    )
    assert success == ConsentReceiptVerificationResult(True, "verified", receipt)
    assert verifier.verify_result(
        receipt, _CLIENT, _TOOL, _RESOURCE, _SCOPE, _ARGUMENTS
    ) == ConsentReceiptVerificationResult(False, "replay")


def test_verifier_result_maps_typed_failures_to_stable_codes() -> None:
    clock = [1_000.0]
    receipt = _issuer(clock).issue(
        _CLIENT, _TOOL, _RESOURCE, _SCOPE, _ARGUMENTS, ttl_seconds=10
    )

    def result_for(candidate, *, now=1_000.0, keys=None):
        verifier = ConsentReceiptVerifier(
            keys or MappingConsentKeyProvider({"synthetic": _KEY}),
            clock=lambda: now,
        )
        return verifier.verify_result(
            candidate, _CLIENT, _TOOL, _RESOURCE, _SCOPE, _ARGUMENTS
        )

    tampered = receipt.from_dict(
        {**receipt.to_dict(), "signature": f"hmac-sha256:{'0' * 64}"}
    )
    denied = _issuer(clock).issue(
        _CLIENT,
        _TOOL,
        _RESOURCE,
        _SCOPE,
        _ARGUMENTS,
        decision="deny",
        ttl_seconds=10,
    )

    assert result_for(tampered).code == "invalid_signature"
    assert result_for(receipt, now=999.0).code == "not_yet_valid"
    assert result_for(receipt, now=1_010.0).code == "expired"
    assert result_for(denied).code == "decision_denied"
    assert (
        result_for(receipt, keys=MappingConsentKeyProvider({})).code
        == "key_unavailable"
    )
    assert result_for("not-json").code == "invalid_receipt"


def test_verifier_rejects_expiry_and_does_not_log_request_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = [1_000.0]
    receipt = _issuer(clock).issue(
        _CLIENT,
        _TOOL,
        _RESOURCE,
        _SCOPE,
        _ARGUMENTS,
        ttl_seconds=10,
    )
    verifier = ConsentReceiptVerifier(
        MappingConsentKeyProvider({"synthetic": _KEY}),
        clock=lambda: clock[0],
    )
    policy = ConsentReceiptPolicy(
        verifier=verifier,
        client=_CLIENT,
        resource=_RESOURCE,
        scope=_SCOPE,
    )

    with caplog.at_level(logging.DEBUG):
        policy.authorize(tool=_TOOL, arguments=deepcopy(_ARGUMENTS), receipt=receipt)
    clock[0] = 1_010.0
    with pytest.raises(ConsentReceiptExpiredError):
        verifier.verify(
            receipt,
            _CLIENT,
            _TOOL,
            _RESOURCE,
            _SCOPE,
            _ARGUMENTS,
        )

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "synthetic clinical text" not in logged
    assert "synthetic-bearer-value" not in logged
    assert "argument_digest" in logged
    assert "policy_version" in logged


def test_policy_fails_closed_when_receipt_is_required() -> None:
    verifier = ConsentReceiptVerifier(
        MappingConsentKeyProvider({"synthetic": _KEY}),
        clock=lambda: 1_000.0,
    )
    policy = ConsentReceiptPolicy(
        verifier=verifier,
        client=_CLIENT,
        resource=_RESOURCE,
        scope=_SCOPE,
    )

    with pytest.raises(ConsentReceiptRequiredError):
        policy.authorize(tool=_TOOL, arguments={"all_models": False})


def test_mcp_policy_adds_receipts_only_to_state_changing_tools() -> None:
    from openmed.mcp import server as mcp_server

    clock = [1_000.0]
    key_provider = MappingConsentKeyProvider({"synthetic": _KEY})
    verifier = ConsentReceiptVerifier(key_provider, clock=lambda: clock[0])
    policy = ConsentReceiptPolicy(
        verifier=verifier,
        client=_CLIENT,
        resource=_RESOURCE,
        scope=_SCOPE,
    )

    class Runtime:
        def loaded_models(self) -> dict[str, Any]:
            return {"models": {"synthetic-model": {"active_requests": 0}}}

        def unload_model(self, model_name: str) -> dict[str, Any]:
            return {"status": "unloaded", "model": model_name}

    class FakeServer:
        def __init__(self) -> None:
            self.tools: dict[str, Any] = {}

        def tool(self, *, name: str, **metadata: Any):
            del metadata

            def decorator(handler):
                self.tools[name] = handler
                return handler

            return decorator

    class FakeResult:
        def __init__(self, payload: dict[str, Any], is_error: bool) -> None:
            self.structuredContent = payload
            self.isError = is_error

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        mcp_server,
        "_call_tool_result",
        lambda payload, *, is_error: FakeResult(payload, is_error),
    )
    try:
        server = FakeServer()
        mcp_server._register_tools(server, Runtime, policy)
        assert (
            "consent_receipt"
            not in inspect.signature(server.tools["openmed_loaded_models"]).parameters
        )
        assert "consent_receipt" in inspect.signature(server.tools[_TOOL]).parameters

        readonly = server.tools["openmed_loaded_models"]()
        assert readonly.isError is False
        missing = server.tools[_TOOL](model_name="synthetic-model")
        assert missing.isError is True
        assert missing.structuredContent["error"]["code"] == "consent_required"

        receipt = ConsentReceiptIssuer(
            key_provider,
            key_id="synthetic",
            clock=lambda: clock[0],
            receipt_id_factory=lambda: "server-receipt-001",
        ).issue(
            _CLIENT,
            _TOOL,
            _RESOURCE,
            _SCOPE,
            {"model_name": "synthetic-model", "all_models": False},
            ttl_seconds=60,
        )
        allowed = server.tools[_TOOL](
            model_name="synthetic-model",
            consent_receipt=receipt.to_dict(),
        )
        assert allowed.isError is False
    finally:
        monkeypatch.undo()
