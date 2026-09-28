"""Focused tests for the PHI-safe agent tool inventory CLI."""

from __future__ import annotations

import builtins
import json
import socket
from pathlib import Path

import pytest
from typer.testing import CliRunner

from openmed.agent.tool_inventory import (
    SideEffectClass,
    ToolInventory,
    ToolInventoryRecord,
)
from openmed.cli import typer_app

SCHEMA_A = "sha256:" + "a" * 64
SCHEMA_B = "sha256:" + "b" * 64


def _inventory() -> ToolInventory:
    return ToolInventory.from_records(
        [
            ToolInventoryRecord(
                tool_id="tool:org.example/summarize",
                version="1.0.0",
                capability_class=("capability:org.example/clinical-transform@1.0.0"),
                side_effect_class=SideEffectClass.NONE,
                schema_digest=SCHEMA_A,
            ),
            ToolInventoryRecord(
                tool_id="tool:org.example/export",
                version="2.0.0",
                capability_class="capability:org.example/clinical-write@1.0.0",
                side_effect_class=SideEffectClass.IDEMPOTENT_WRITE,
                schema_digest=SCHEMA_B,
            ),
        ]
    )


def test_empty_registry_has_stable_text_snapshot() -> None:
    result = CliRunner().invoke(typer_app.build_app(), ["agents", "tools"])

    assert result.exit_code == 0, result.output
    assert result.stdout == (
        "# Agent tool inventory\n"
        "\n"
        "Schema: `openmed.agent.tool_inventory.v1`\n"
        "\n"
        "Registered tools: 0\n"
        "\n"
        "_No registered tools._\n"
    )


def test_populated_registry_has_stable_json_snapshot(tmp_path: Path) -> None:
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(_inventory().to_json(), encoding="utf-8")

    result = CliRunner().invoke(
        typer_app.build_app(),
        [
            "agents",
            "tools",
            "--inventory",
            str(inventory_path),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == _inventory().to_json() + "\n"
    assert json.loads(result.stdout) == _inventory().to_dict()


def test_explicit_output_path_receives_exact_snapshot(tmp_path: Path) -> None:
    inventory_path = tmp_path / "inventory.json"
    output_path = tmp_path / "tools.txt"
    inventory_path.write_text(_inventory().to_json(), encoding="utf-8")

    result = CliRunner().invoke(
        typer_app.build_app(),
        [
            "agents",
            "tools",
            "--inventory",
            str(inventory_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == ""
    assert output_path.read_text(encoding="utf-8") == _inventory().to_markdown()


@pytest.mark.parametrize(
    "forbidden_field",
    ["endpoint", "credential", "arguments", "example", "clinical_data"],
)
def test_content_bearing_registry_fields_never_enter_output(
    forbidden_field: str,
    tmp_path: Path,
) -> None:
    sentinel = "Synthetic Patient at https://private.example.test token-secret"
    payload = _inventory().to_dict()
    payload["tools"][0][forbidden_field] = sentinel
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(payload), encoding="utf-8")

    result = CliRunner().invoke(
        typer_app.build_app(),
        ["agents", "tools", "--inventory", str(inventory_path)],
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "tools: unreadable_records" in result.stderr
    assert sentinel not in result.output


def test_inventory_command_is_offline_and_provider_independent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(_inventory().to_json(), encoding="utf-8")

    def fail_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("agent tool inventory attempted network access")

    original_import = builtins.__import__

    def reject_provider_import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith(("openmed.agent.providers", "openmed.mcp")):
            raise AssertionError("agent tool inventory loaded an agent provider")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(socket, "create_connection", fail_network)
    monkeypatch.setattr(builtins, "__import__", reject_provider_import)

    result = CliRunner().invoke(
        typer_app.build_app(),
        ["agents", "tools", "--inventory", str(inventory_path)],
    )

    assert result.exit_code == 0, result.output
    assert "tool:org.example/export" in result.stdout
    assert "tool:org.example/summarize" in result.stdout


def test_invalid_format_fails_with_value_free_error() -> None:
    result = CliRunner().invoke(
        typer_app.build_app(),
        ["agents", "tools", "--format", "yaml"],
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "format: invalid_format" in result.stderr
