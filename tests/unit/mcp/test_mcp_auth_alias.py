import asyncio
import warnings

import pytest

from openmed.core.errors import ConfigurationError
from openmed.mcp.server import MCPAuthorizationConfig, create_mcp_server


def test_legacy_and_canonical_names_configure_the_same_authorization():
    pytest.importorskip("mcp.server.fastmcp")
    config = MCPAuthorizationConfig(
        enabled=True,
        resource_url="https://private-fixture.example/mcp",
        authorization_server_url="https://issuer.example",
    )
    with pytest.warns(DeprecationWarning, match="authorization_config") as captured:
        legacy = create_mcp_server(auth_config=config)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", message=".*auth_config.*", category=DeprecationWarning
        )
        canonical = create_mcp_server(authorization_config=config)
    alias_warnings = [
        str(item.message) for item in captured if "auth_config" in str(item.message)
    ]
    assert len(alias_warnings) == 1 and "private-fixture" not in alias_warnings[0]
    for server in (legacy, canonical):
        result = asyncio.run(server.call_tool("openmed_loaded_models", {}))
        assert (
            result.isError
            and result.structuredContent["error"]["code"] == "invalid_token"
        )
        assert "/.well-known/oauth-protected-resource/mcp" in {
            route.path for route in server.streamable_http_app().routes
        }


def test_both_names_keep_the_existing_configuration_error():
    config = MCPAuthorizationConfig()
    with (
        pytest.warns(DeprecationWarning, match="authorization_config"),
        pytest.raises(
            ConfigurationError, match="Specify one MCP authorization configuration"
        ),
    ):
        create_mcp_server(auth_config=config, authorization_config=config)
