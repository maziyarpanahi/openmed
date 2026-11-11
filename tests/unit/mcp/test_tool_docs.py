"""Keep the MCP tool hints page in sync with the tool registry."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

from openmed.mcp.tool_registry import TOOL_REGISTRY, ToolSpec

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_PAGE = REPO_ROOT / "docs" / "mcp-tools.md"
TABLE_START = "<!-- mcp-tools-table:start -->"
TABLE_END = "<!-- mcp-tools-table:end -->"
HINT_KEYS = ("readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint")


def _first_party_specs() -> tuple[ToolSpec, ...]:
    """Return the latest built-in tools; optional plugin tools are excluded."""

    return tuple(spec for spec in TOOL_REGISTRY.latest_specs() if not spec.plugin_id)


def _cell(value: str) -> str:
    return value.replace("|", "\\|")


def render_tool_table(specs: Iterable[ToolSpec]) -> str:
    """Render the Markdown table committed in ``docs/mcp-tools.md``."""

    lines = [
        "| Tool | Title | Version | Read-only | Destructive | Idempotent "
        "| Open-world |",
        "| --- | --- | --- | :---: | :---: | :---: | :---: |",
    ]
    for spec in specs:
        annotations = spec.annotations()
        hints = " | ".join("yes" if annotations[key] else "no" for key in HINT_KEYS)
        lines.append(
            f"| `{spec.name}` | {_cell(annotations['title'])} "
            f"| {spec.version} | {hints} |"
        )
    return "\n".join(lines)


def _committed_table() -> str:
    text = TOOLS_PAGE.read_text(encoding="utf-8")
    assert text.count(TABLE_START) == 1, f"missing {TABLE_START} marker"
    assert text.count(TABLE_END) == 1, f"missing {TABLE_END} marker"
    body = text.split(TABLE_START, 1)[1].split(TABLE_END, 1)[0]
    return body.strip()


def test_tools_page_matches_registry() -> None:
    expected = render_tool_table(_first_party_specs())

    assert _committed_table() == expected, (
        "docs/mcp-tools.md is out of date with TOOL_REGISTRY. Replace the "
        f"table between the markers with:\n\n{expected}\n"
    )


def test_tools_page_lists_every_registered_tool() -> None:
    table = _committed_table()
    specs = _first_party_specs()

    assert specs
    missing = [spec.name for spec in specs if f"| `{spec.name}` |" not in table]
    assert missing == []
    assert table.count("\n| `") == len(specs)


def test_changed_hint_is_detected_as_drift() -> None:
    specs = _first_party_specs()
    flipped = replace(specs[0], read_only_hint=not specs[0].read_only_hint)

    assert render_tool_table((flipped, *specs[1:])) != _committed_table()


def test_tools_page_links_back_to_client_setup() -> None:
    assert "](mcp-clients.md)" in TOOLS_PAGE.read_text(encoding="utf-8")
