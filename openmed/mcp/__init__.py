"""Model Context Protocol integration for OpenMed."""

from __future__ import annotations

from typing import Any

__all__ = [
    "ConsentDecision",
    "ConsentReceipt",
    "ConsentReceiptIssuer",
    "ConsentReceiptPolicy",
    "ConsentReceiptVerifier",
    "canonical_argument_digest",
    "create_mcp_server",
    "main",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        if name in {
            "ConsentDecision",
            "ConsentReceipt",
            "ConsentReceiptIssuer",
            "ConsentReceiptPolicy",
            "ConsentReceiptVerifier",
            "canonical_argument_digest",
        }:
            from .consent_receipts import (
                ConsentDecision,
                ConsentReceipt,
                ConsentReceiptIssuer,
                ConsentReceiptPolicy,
                ConsentReceiptVerifier,
                canonical_argument_digest,
            )

            exports = {
                "ConsentDecision": ConsentDecision,
                "ConsentReceipt": ConsentReceipt,
                "ConsentReceiptIssuer": ConsentReceiptIssuer,
                "ConsentReceiptPolicy": ConsentReceiptPolicy,
                "ConsentReceiptVerifier": ConsentReceiptVerifier,
                "canonical_argument_digest": canonical_argument_digest,
            }
            return exports[name]
        from .server import create_mcp_server, main

        exports = {"create_mcp_server": create_mcp_server, "main": main}
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
