"""Protocol tests for the public grounding MCP facade."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from openmed.clinical.grounding import VocabLoader
from openmed.mcp.server import create_mcp_server
from openmed.mcp.tool_registry import (
    TOOL_REGISTRY,
    validate_registered_tool_output,
)

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "openmed/eval/golden/fixtures/grounding_vocab_synthetic.jsonl"


@pytest.fixture()
def grounding_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache_dir = tmp_path / "grounding"
    VocabLoader(cache_dir=cache_dir, local_only=True).import_snapshot(
        "icd10cm",
        FIXTURE,
        version="synthetic-fixture-1",
    )
    monkeypatch.setenv("OPENMED_GROUNDING_CACHE_DIR", str(cache_dir))
    return cache_dir


def test_ground_tool_is_listed_and_round_trips_without_logging_phi(
    grounding_cache: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    del grounding_cache
    source_text = "Synthetic patient Rowan Example has type 2 diabetes in assessment."
    surface = "type 2 diabetes"
    server = create_mcp_server()

    tools = asyncio.run(server.list_tools())
    advertised = {tool.name: tool for tool in tools}
    assert len(advertised) == len(TOOL_REGISTRY.latest_specs())
    assert "openmed_ground_concepts" in advertised
    spec = TOOL_REGISTRY.get("openmed_ground_concepts")
    assert advertised[spec.name].inputSchema == spec.input_schema
    assert advertised[spec.name].outputSchema == spec.mcp_output_schema()

    with caplog.at_level(logging.DEBUG):
        response = asyncio.run(
            server.call_tool(
                "openmed_ground_concepts",
                {
                    "text": source_text,
                    "systems": ["icd10cm"],
                    "lang": "en",
                    "top_k": 1,
                    "offline": True,
                },
            )
        )

    assert response.isError is False
    payload = response.structuredContent
    assert validate_registered_tool_output("openmed_ground_concepts", payload) == (
        payload
    )
    concept = payload["concepts"][0]
    assert concept["system"] == "icd10cm"
    assert concept["code"] == "E11.9"
    assert concept["span"] == {
        "start": source_text.index(surface),
        "end": source_text.index(surface) + len(surface),
    }
    assert 0.0 <= concept["confidence"] <= 1.0
    assert concept["provenance"]["vocabulary_snapshot_version"] == (
        "synthetic-fixture-1"
    )

    rendered = "\n".join(record.getMessage() for record in caplog.records)
    assert source_text not in rendered
    assert "Rowan Example" not in rendered
    assert surface not in rendered
