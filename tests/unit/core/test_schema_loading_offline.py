"""Bundled JSON Schemas load and validate without resolving their ``$id`` URLs.

The documentation site publishes copies of the packaged schemas at their
``$id`` URLs for external tools. OpenMed itself must keep reading the packaged
files: these tests block every socket and URL opener, then load and validate.
"""

from __future__ import annotations

import json
import socket
import urllib.request
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from jsonschema.validators import validator_for

from openmed.core.config import (
    OpenMedConfig,
    config_schema_path,
    load_config_from_file,
)
from openmed.core.schemas import (
    SCHEMA_NAMES,
    OpenMedSpan,
    hmac_text_hash,
    load_schema,
    load_schema_bundle,
)

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def blocked_network(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    attempts: list[str] = []

    def deny(name: str) -> Any:
        def blocked(*args: object, **kwargs: object) -> None:
            attempts.append(name)
            raise AssertionError(f"network access attempted through {name}")

        return blocked

    monkeypatch.setattr(socket.socket, "connect", deny("socket.connect"))
    monkeypatch.setattr(socket.socket, "connect_ex", deny("socket.connect_ex"))
    monkeypatch.setattr(socket, "create_connection", deny("create_connection"))
    monkeypatch.setattr(socket, "getaddrinfo", deny("getaddrinfo"))
    monkeypatch.setattr(urllib.request, "urlopen", deny("urlopen"))
    return attempts


def _span_record() -> dict[str, Any]:
    return OpenMedSpan(
        doc_id="synthetic-doc",
        start=0,
        end=4,
        text_hash=hmac_text_hash("SYNT", "synthetic-key"),
        entity_type="PERSON",
        canonical_label="PERSON",
    ).to_dict()


def test_load_schema_reads_package_files_without_network(
    blocked_network: list[str],
) -> None:
    for name in SCHEMA_NAMES:
        schema = load_schema(name)
        assert schema["$id"].startswith("https://")
        validator_for(schema).check_schema(schema)

    bundle = load_schema_bundle()
    validator = Draft202012Validator(bundle["schemas"]["span"])
    assert list(validator.iter_errors(_span_record())) == []
    assert blocked_network == []


def test_every_packaged_schema_checks_against_its_metaschema_offline(
    blocked_network: list[str],
) -> None:
    paths = sorted(
        path
        for path in (ROOT / "openmed").rglob("*.schema.json")
        if "__pycache__" not in path.parts
    )
    assert paths
    for path in paths:
        schema = json.loads(path.read_text(encoding="utf-8"))
        validator_for(schema).check_schema(schema)
        assert all(ref.startswith("#") for ref in _refs(schema)), path.name
    assert blocked_network == []


def _refs(node: Any) -> list[str]:
    if isinstance(node, dict):
        found = [node["$ref"]] if isinstance(node.get("$ref"), str) else []
        return found + [ref for value in node.values() for ref in _refs(value)]
    if isinstance(node, list):
        return [ref for value in node for ref in _refs(value)]
    return []


def test_config_validation_uses_the_bundled_schema_offline(
    blocked_network: list[str], tmp_path: Path
) -> None:
    config_schema = json.loads(config_schema_path().read_text(encoding="utf-8"))
    assert config_schema["$id"].startswith("https://openmed.life/")

    config = OpenMedConfig()
    assert config.validate() is None
    assert OpenMedConfig.from_dict({"timeout": 30}).timeout == 30
    path = tmp_path / "config.toml"
    path.write_text("timeout = 45\n", encoding="utf-8")
    assert load_config_from_file(path).timeout == 45
    validator = Draft202012Validator(config_schema)
    assert list(validator.iter_errors(config.to_dict())) == []
    assert blocked_network == []


def test_network_guard_is_active(blocked_network: list[str]) -> None:
    with pytest.raises(AssertionError, match="network access attempted"):
        socket.create_connection(("openmed.life", 443))
    with pytest.raises(AssertionError, match="network access attempted"):
        urllib.request.urlopen("https://openmed.life/schemas/index.json")
    assert blocked_network == ["create_connection", "urlopen"]
