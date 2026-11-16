"""Shared request fixtures are parseable, synthetic and free of secrets."""

import json
from pathlib import Path

import pytest

from openmed.core.secrets import detect_secrets
from tests.fixtures.privacy_proxy import load_fixture

FIXTURE_ROOT = Path(__file__).resolve().parents[3] / "fixtures/privacy_proxy"


@pytest.mark.parametrize(
    "path", sorted(FIXTURE_ROOT.glob("*.json")), ids=lambda p: p.stem
)
def test_proxy_fixture_is_synthetic_and_secret_free(path):
    fixture = load_fixture(path.stem)
    assert fixture["synthetic"] is True
    assert detect_secrets(json.dumps(fixture)) == []
    assert "request" in fixture or "chunks" in fixture


def test_fixture_inventory_and_protocol_shapes():
    assert {path.stem for path in FIXTURE_ROOT.glob("*.json")} == {
        "single_turn",
        "multipart_text",
        "tool_exchange",
        "image_part",
        "split_placeholder",
    }
    assert len(load_fixture("multipart_text")["request"]["messages"][0]["content"]) == 2
    messages = load_fixture("tool_exchange")["request"]["messages"]
    call = messages[1]["tool_calls"][0]
    assert messages[2]["tool_call_id"] == call["id"]
    assert json.loads(call["function"]["arguments"]) == {"name": "Avery Example"}
    chunks = load_fixture("split_placeholder")["chunks"]
    parts = [chunk["choices"][0]["delta"]["content"] for chunk in chunks]
    assert "".join(parts) == "Contact <<OPENMED_PHI_NAME_A1B2C3D4_000001>>."
    assert all("<<OPENMED_PHI_NAME_A1B2C3D4_000001>>" not in part for part in parts)
