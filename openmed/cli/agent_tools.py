"""CLI helpers for content-free local-agent tool inventories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from openmed.agent.tool_inventory import (
    ToolInventory,
    ToolInventoryError,
    render_tool_inventory_json,
    render_tool_inventory_markdown,
)

MAX_TOOL_INVENTORY_INPUT_BYTES: Final = 10 * 1024 * 1024
_OUTPUT_FORMATS: Final = frozenset({"json", "text"})


class AgentToolsCliError(ValueError):
    """Report a stable CLI failure without retaining rejected content."""

    def __init__(self, code: str, field_name: str) -> None:
        self.code = code
        self.field_name = field_name
        super().__init__(f"{field_name}: {code}")


def load_tool_inventory(path: Path | None) -> ToolInventory:
    """Load a content-free inventory document without importing providers.

    A missing path represents an empty local registry. Invalid JSON, unknown
    fields, and content-bearing metadata fail with value-free errors.

    Args:
        path: Optional path to a content-free inventory JSON document.

    Returns:
        A validated, deterministically ordered tool inventory.

    Raises:
        AgentToolsCliError: If the document cannot be read or parsed safely.
    """

    if path is None:
        return ToolInventory.from_records(())

    try:
        with path.open("rb") as stream:
            payload = stream.read(MAX_TOOL_INVENTORY_INPUT_BYTES + 1)
    except OSError:
        raise AgentToolsCliError("unreadable_file", "inventory") from None
    if len(payload) > MAX_TOOL_INVENTORY_INPUT_BYTES:
        raise AgentToolsCliError("file_too_large", "inventory")

    try:
        document = json.loads(payload)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise AgentToolsCliError("invalid_json", "inventory") from None

    try:
        return ToolInventory.from_dict(document)
    except ToolInventoryError as exc:
        raise AgentToolsCliError(exc.code, exc.field_name) from None


def render_tool_inventory(inventory: ToolInventory, output_format: str) -> str:
    """Render a validated inventory in a deterministic CLI format."""

    if output_format not in _OUTPUT_FORMATS:
        raise AgentToolsCliError("invalid_format", "format")
    if output_format == "json":
        return render_tool_inventory_json(inventory) + "\n"
    return render_tool_inventory_markdown(inventory)


def write_tool_inventory(rendered: str, output: Path) -> None:
    """Write rendered inventory text to an explicit local path."""

    try:
        output.write_text(rendered, encoding="utf-8")
    except OSError:
        raise AgentToolsCliError("write_failed", "output") from None


def add_agent_tools_command(app: Any, typer_module: Any) -> None:
    """Register the ``openmed agents tools`` Typer command group."""

    agents_app = typer_module.Typer(help="Local-agent governance commands.")

    @agents_app.command("tools")
    def agent_tools(
        inventory: Path | None = typer_module.Option(
            None,
            "--inventory",
            help="Content-free tool inventory JSON. Omit for an empty registry.",
        ),
        output_format: str = typer_module.Option(
            "text",
            "--format",
            help="Output format: text or json.",
        ),
        output: Path | None = typer_module.Option(
            None,
            "--output",
            "-o",
            help="Write output to this path instead of stdout.",
        ),
    ) -> None:
        """List registered tools without exposing runtime or clinical data."""

        try:
            loaded = load_tool_inventory(inventory)
            rendered = render_tool_inventory(loaded, output_format)
            if output is None:
                typer_module.echo(rendered, nl=False)
            else:
                write_tool_inventory(rendered, output)
        except AgentToolsCliError as exc:
            typer_module.echo(
                f"Unable to render agent tool inventory: {exc}",
                err=True,
            )
            raise typer_module.Exit(code=1) from None

    app.add_typer(agents_app, name="agents")


__all__ = [
    "AgentToolsCliError",
    "MAX_TOOL_INVENTORY_INPUT_BYTES",
    "add_agent_tools_command",
    "load_tool_inventory",
    "render_tool_inventory",
    "write_tool_inventory",
]
