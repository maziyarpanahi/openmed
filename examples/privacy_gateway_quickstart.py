"""Synthetic privacy-gateway quickstart.

This example shows the local flow for external model calls:

1. redact text on device,
2. send only redacted text to a user-supplied callable,
3. restore identifiers locally from the in-memory mapping.
"""

from __future__ import annotations

from openmed.interop.gateway import PrivacyGateway


def external_model_stub(redacted_text: str) -> str:
    """Stand in for a cloud model without making a network call."""

    if "Casey Example" in redacted_text or "555-0100" in redacted_text:
        raise RuntimeError("external callable received an unredacted identifier")
    return f"Follow-up summary for: {redacted_text}"


def main() -> None:
    gateway = PrivacyGateway()
    text = "Patient Casey Example can be reached at 555-0100."

    redacted_text, mapping = gateway.redact(text)
    gateway.input_guardrail(mapping)(redacted_text)

    model_response = external_model_stub(redacted_text)
    restored_response = gateway.output_guardrail(mapping)(model_response)

    print(restored_response)


if __name__ == "__main__":
    main()
