"""Smoke test for the local privacy-proxy echo transport example."""

from __future__ import annotations

import json

from examples import privacy_proxy_echo as example


def test_privacy_proxy_echo_redacts_before_transport_and_restores_reply(capsys):
    result = example.main()
    captured = capsys.readouterr().out

    transport_payload = result["transport_saw"]
    client_response = result["client_received"]
    serialized_payload = json.dumps(transport_payload, sort_keys=True)
    serialized_reply = json.dumps(client_response, sort_keys=True)
    reply_content = client_response["choices"][0]["message"]["content"]

    # The echo transport only sees placeholder-bearing text.
    assert "OPENMED_PHI_EMAIL" in serialized_payload
    assert "OPENMED_PHI_MEDICAL_RECORD_NUMBER" in serialized_payload
    assert example.SYNTHETIC_EMAIL not in serialized_payload
    assert example.SYNTHETIC_MEDICAL_RECORD_NUMBER not in serialized_payload

    # The local client gets the original identifiers restored.
    assert client_response["object"] == "chat.completion"
    assert example.SYNTHETIC_EMAIL in reply_content
    assert example.SYNTHETIC_MEDICAL_RECORD_NUMBER in reply_content
    assert "OPENMED_PHI" not in serialized_reply

    # Both boundary views are printed by the runnable example.
    assert "=== Payload received by the echo transport ===" in captured
    assert "=== Reply returned to the local client ===" in captured
    assert "OPENMED_PHI" in captured
    assert example.SYNTHETIC_EMAIL in captured
    assert example.SYNTHETIC_MEDICAL_RECORD_NUMBER in captured


def test_privacy_proxy_echo_is_deterministic():
    first = example.run_echo_example()
    second = example.run_echo_example()

    assert first == second
