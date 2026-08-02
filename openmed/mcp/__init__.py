"""Model Context Protocol integration for OpenMed."""

from __future__ import annotations

from typing import Any

__all__ = [
    "MCPAuthConfig",
    "MCPAuthorizationConfig",
    "MCPTokenVerifier",
    "MCPToolPolicy",
    "create_mcp_server",
    "main",
]


def __getattr__(name: str) -> Any:
    if name in {"create_mcp_server", "main"}:
        from .server import create_mcp_server, main

        exports = {"create_mcp_server": create_mcp_server, "main": main}
        return exports[name]
    if name in {
        "MCPAuthConfig",
        "MCPAuthorizationConfig",
        "MCPTokenVerifier",
        "MCPToolPolicy",
    }:
        from openmed.service.security import (
            MCPAuthConfig,
            MCPAuthorizationConfig,
            MCPTokenVerifier,
            MCPToolPolicy,
        )

        exports = {
            "MCPAuthConfig": MCPAuthConfig,
            "MCPAuthorizationConfig": MCPAuthorizationConfig,
            "MCPTokenVerifier": MCPTokenVerifier,
            "MCPToolPolicy": MCPToolPolicy,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
