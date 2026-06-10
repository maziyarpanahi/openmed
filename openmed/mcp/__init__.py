"""Model Context Protocol integration for OpenMed."""

from __future__ import annotations

from typing import Any

__all__ = ["create_mcp_server", "main"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .server import create_mcp_server, main

        exports = {"create_mcp_server": create_mcp_server, "main": main}
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
