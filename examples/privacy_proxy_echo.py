#!/usr/bin/env python3
"""Runnable local privacy-proxy example with an in-process echo transport.

The proxy redacts a synthetic chat request, hands the placeholder-bearing
payload to an echo transport that never leaves the process, and restores the
original identifiers only in the local response. The request is sent through
the FastAPI test client, so the example opens no port and makes no network
call. Redaction uses the deterministic safety-sweep detector, which needs no
model download.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from openmed.service.privacy_gateway import safety_sweep_tripwire
from openmed.service.privacy_proxy import create_app

SYNTHETIC_EMAIL = "avery@example.com"
SYNTHETIC_MEDICAL_RECORD_NUMBER = "MRN 12345678"
# The "medical record number" context keeps the deterministic MRN match above
# the gateway's default confidence threshold without loading any model.
SYNTHETIC_PROMPT = (
    f"Contact Avery at {SYNTHETIC_EMAIL}. "
    f"The medical record number {SYNTHETIC_MEDICAL_RECORD_NUMBER} is on file."
)
MODEL_NAME = "local-fixture"
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"


def run_echo_example() -> dict[str, Any]:
    """Build the proxy, send one synthetic request, and capture both views.

    Returns:
        A mapping with the placeholder-bearing payload the echo transport
        received and the restored completion returned to the local client.
    """
    seen_payloads: list[dict[str, Any]] = []

    def echo_transport(payload: dict[str, Any], **_: Any) -> str:
        """Record the redacted payload and echo its last message text."""
        seen_payloads.append(payload)
        redacted_text = payload["messages"][-1]["content"]
        if (
            SYNTHETIC_EMAIL in redacted_text
            or SYNTHETIC_MEDICAL_RECORD_NUMBER in redacted_text
        ):
            raise RuntimeError("echo transport received an unredacted identifier")
        return redacted_text

    app = create_app(
        transport=echo_transport,
        extractor=safety_sweep_tripwire,
    )
    request_body = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": SYNTHETIC_PROMPT}],
    }

    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post(CHAT_COMPLETIONS_PATH, json=request_body)
    assert response.status_code == 200, response.text

    return {
        "transport_saw": seen_payloads[0],
        "client_received": response.json(),
    }


def main() -> dict[str, Any]:
    """Print the redacted transport payload and the restored local reply."""
    result = run_echo_example()

    print("=== Payload received by the echo transport ===")
    print(json.dumps(result["transport_saw"], indent=2, sort_keys=True))
    print("\n=== Reply returned to the local client ===")
    print(json.dumps(result["client_received"], indent=2, sort_keys=True))

    return result


if __name__ == "__main__":
    main()
