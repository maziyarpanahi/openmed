"""Version pins for the decision contract documents.

Each supported schema version names exactly one document per contract file.
The current documents are pinned by fingerprint, and every earlier supported
version is frozen in a compatibility fixture whose documents are pinned too.
Changing a document therefore fails here until ``DECISION_SCHEMA_VERSION`` is
bumped, the previous documents are frozen, and the new fingerprints are added.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from openmed.core.schemas import schema_fingerprint
from openmed.mcp.tool_registry import TOOL_REGISTRY
from openmed.structured.decision import (
    DECISION_BASE_SCHEMA_VERSION,
    DECISION_SCHEMA_VERSION,
    SUPPORTED_DECISION_SCHEMA_VERSIONS,
    decision_request_schema,
    decision_result_schema,
    load_decision_schema,
)
from openmed.structured.decision_interop import (
    openai_decisions_request_schema,
    openai_decisions_response_schema,
)

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests" / "fixtures" / "structured"

# Append-only: one entry per published decision contract version. Never edit
# the fingerprints of an existing version; bump the version instead.
PINNED_FINGERPRINTS = {
    "1.0.0": {
        "decision_request": (
            "sha256:17bb87c480a49a5de85cbc7feed3eb246243a8995b9a8e9c43a3d23651bd5d89"
        ),
        "decision_result": (
            "sha256:9044ef01a1ed02adbc290247ff2a9ab30e53cf3fb68cc85bd7bba3a5f5ddfc67"
        ),
    },
    "1.1.0": {
        "decision_request": (
            "sha256:f0e791027c7e0088dafa63a2f241dd37d602db0b111fde99366b2281b940997e"
        ),
        "decision_result": (
            "sha256:1e263c27a1670d1b6d1c9c0983bc3318e7fea62a5812386e15341d47491250b6"
        ),
    },
    "1.2.0": {
        "decision_request": (
            "sha256:2e6c794333af434f5d225264dd80dd17bd037211a168681fd789a6d8e84adc9f"
        ),
        "decision_result": (
            "sha256:3dce9b93a1334a057b8c8eb81fb199e9b2e8b04004da72376574156eebaa8515"
        ),
        "decision_openai_request": (
            "sha256:6165dfef9aab5f3d0c26a6255a047e2cd83670751dfaee371b755c7b23165279"
        ),
        "decision_openai_response": (
            "sha256:7d69b861225d2933fc098a9391ee0473e1c234bebebd33751f50fd0e9999c657"
        ),
    },
}
GENERATORS = {
    "decision_request": decision_request_schema,
    "decision_result": decision_result_schema,
    "decision_openai_request": openai_decisions_request_schema,
    "decision_openai_response": openai_decisions_response_schema,
}
ID_SLUGS = {
    "decision_request": "decision-request",
    "decision_result": "decision-result",
    "decision_openai_request": "decision-openai-request",
    "decision_openai_response": "decision-openai-response",
}


def _frozen(version: str) -> dict:
    path = FIXTURES / f"decision_compat_v{version.replace('.', '_')}.json"
    assert path.exists(), f"schema {version} needs a frozen compatibility fixture"
    return json.loads(path.read_text(encoding="utf-8"))


def test_every_supported_version_is_pinned() -> None:
    assert tuple(PINNED_FINGERPRINTS) == SUPPORTED_DECISION_SCHEMA_VERSIONS
    assert SUPPORTED_DECISION_SCHEMA_VERSIONS[0] == DECISION_BASE_SCHEMA_VERSION
    assert SUPPORTED_DECISION_SCHEMA_VERSIONS[-1] == DECISION_SCHEMA_VERSION
    assert list(SUPPORTED_DECISION_SCHEMA_VERSIONS) == sorted(
        SUPPORTED_DECISION_SCHEMA_VERSIONS,
        key=lambda version: tuple(int(part) for part in version.split(".")),
    )


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_current_documents_match_their_version_pin(name: str) -> None:
    pinned = PINNED_FINGERPRINTS[DECISION_SCHEMA_VERSION][name]

    assert schema_fingerprint(GENERATORS[name]()) == pinned, (
        f"{name} changed without a decision schema version bump: bump "
        "DECISION_SCHEMA_VERSION, freeze the previous documents in a "
        "compatibility fixture and pin the new fingerprints"
    )


def test_an_unversioned_document_change_breaks_the_pin() -> None:
    changed = decision_request_schema()
    changed["properties"]["namespace"]["description"] = "Synthetic drift."

    assert (
        schema_fingerprint(changed)
        != (PINNED_FINGERPRINTS[DECISION_SCHEMA_VERSION]["decision_request"])
    )
    assert changed["$id"] == decision_request_schema()["$id"]


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_documents_name_the_current_version(name: str) -> None:
    document = GENERATORS[name]()

    assert document["$id"] == (
        f"https://openmed.dev/schemas/{ID_SLUGS[name]}/{DECISION_SCHEMA_VERSION}"
    )
    assert document["$schema"] == "https://json-schema.org/draft/2020-12/schema"


@pytest.mark.parametrize("version", SUPPORTED_DECISION_SCHEMA_VERSIONS[:-1])
def test_earlier_versions_are_frozen_with_their_pinned_documents(
    version: str,
) -> None:
    frozen = _frozen(version)

    for name, key in (
        ("decision_request", "request_schema"),
        ("decision_result", "result_schema"),
    ):
        assert frozen[key]["$id"].endswith(f"/{version}")
        assert schema_fingerprint(frozen[key]) == PINNED_FINGERPRINTS[version][name]
        assert frozen[key]["$id"] != GENERATORS[name]()["$id"]


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_committed_documents_match_their_generators(name: str) -> None:
    assert load_decision_schema(name) == GENERATORS[name]()


def test_generator_writes_every_document_byte_for_byte() -> None:
    script = ROOT / "scripts" / "generate_decision_schemas.py"
    spec = importlib.util.spec_from_file_location("generate_decision_schemas", script)
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)

    assert {path.name for path in generator.OUTPUTS} == {
        f"{name}.schema.json" for name in GENERATORS
    }
    for path, render in generator.OUTPUTS.items():
        expected = json.dumps(render(), indent=2, sort_keys=True) + "\n"
        assert path.read_text(encoding="utf-8") == expected, path.name


def test_mcp_decision_tool_carries_the_current_contract_version() -> None:
    spec = TOOL_REGISTRY.get("openmed_decide")

    assert spec.version == DECISION_SCHEMA_VERSION
    assert spec.output_schema == decision_result_schema()
