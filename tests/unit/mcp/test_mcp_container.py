"""Smoke coverage for the OpenMed MCP container surfaces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

BASE_DIR = Path(__file__).resolve().parents[3]
REST_DOCKERFILE = BASE_DIR / "Dockerfile"
MCP_DOCKERFILE = BASE_DIR / "Dockerfile.mcp"
COMPOSE_FILE = BASE_DIR / "docker-compose.yml"
CLIENT_DOCS = BASE_DIR / "docs" / "mcp-clients.md"


def _json_instruction(contents: str, instruction: str) -> list[str]:
    prefix = f"{instruction} "
    line = next(line for line in contents.splitlines() if line.startswith(prefix))
    value = json.loads(line.removeprefix(prefix))
    assert isinstance(value, list)
    return value


def _load_compose() -> dict[str, Any]:
    payload = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_mcp_image_defaults_to_stdio_and_supports_http_override() -> None:
    dockerfile = MCP_DOCKERFILE.read_text(encoding="utf-8")

    assert 'pip install --no-cache-dir ".[hf,mcp,service]"' in dockerfile
    assert "EXPOSE 8081" in dockerfile
    assert _json_instruction(dockerfile, "ENTRYPOINT") == ["openmed-mcp"]
    assert _json_instruction(dockerfile, "CMD") == ["--transport", "stdio"]


def test_rest_image_command_remains_unchanged() -> None:
    dockerfile = REST_DOCKERFILE.read_text(encoding="utf-8")

    assert "EXPOSE 8080" in dockerfile
    assert _json_instruction(dockerfile, "CMD") == [
        "uvicorn",
        "openmed.service.app:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
    ]


def test_compose_declares_healthy_mcp_service_with_persistent_cache() -> None:
    compose = _load_compose()
    mcp = compose["services"]["mcp"]

    assert mcp["build"] == {"context": ".", "dockerfile": "Dockerfile.mcp"}
    assert mcp["command"] == [
        "--transport",
        "streamable-http",
        "--host",
        "0.0.0.0",
        "--port",
        "8081",
        "--streamable-http-path",
        "/mcp",
    ]
    assert "127.0.0.1:${OPENMED_MCP_PORT:-8081}:8081" in mcp["ports"]
    assert "hf-cache:/root/.cache/huggingface" in mcp["volumes"]
    assert "hf-cache" in compose["volumes"]
    assert mcp["healthcheck"]["test"][:2] == ["CMD", "python"]


def test_container_client_commands_are_documented() -> None:
    docs = CLIENT_DOCS.read_text(encoding="utf-8")

    assert "docker build -f Dockerfile.mcp -t openmed-mcp:local ." in docs
    assert '"command": "docker"' in docs
    assert "docker run -i --rm" in docs
    assert "--transport streamable-http" in docs
    assert "docker compose up --build mcp" in docs
    assert "http://127.0.0.1:8081/mcp" in docs
