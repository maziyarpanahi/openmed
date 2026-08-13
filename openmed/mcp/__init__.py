"""Model Context Protocol integration for OpenMed.

The MCP server depends on the optional ``mcp`` extra. Importing this package
stays lightweight: server and consent-receipt APIs are resolved lazily, and
:func:`is_mcp_available` / :func:`ensure_mcp_available` provide the same
capability-probe convention as the other optional backends.

Install with: ``pip install openmed[mcp]``.
"""

from __future__ import annotations

from typing import Any

from openmed.core.capabilities import is_backend_available as _is_backend_available
from openmed.core.capabilities import require_backend as _require_backend

_SERVER_EXPORTS = ("create_mcp_server", "main")
_CONSENT_EXPORTS = (
    "ConsentDecision",
    "ConsentReceipt",
    "ConsentReceiptIssuer",
    "ConsentReceiptPolicy",
    "ConsentReceiptVerifier",
    "canonical_argument_digest",
)

__all__ = [
    *_CONSENT_EXPORTS,
    *_SERVER_EXPORTS,
    "ensure_mcp_available",
    "is_mcp_available",
]


def is_mcp_available() -> bool:
    """Return True when the ``mcp`` extra is importable, without importing it."""

    return _is_backend_available("mcp")


def ensure_mcp_available() -> None:
    """Raise an actionable error when the ``mcp`` extra is not installed."""

    _require_backend("mcp", feature="The OpenMed MCP server")


def __getattr__(name: str) -> Any:
    if name in _CONSENT_EXPORTS:
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
    if name in _SERVER_EXPORTS:
        from .server import create_mcp_server, main

        exports = {"create_mcp_server": create_mcp_server, "main": main}
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
