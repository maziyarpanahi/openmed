"""MCP discovery and golden-fixture tests for the clinical workflow."""

from __future__ import annotations

import json
import logging
import socket
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from openmed.mcp import server as mcp_server
from openmed.mcp.clinical_workflow import load_golden_agent_run
from openmed.mcp.tool_registry import (
    CLINICAL_STAGE_ORDER,
    CLINICAL_WORKFLOW_NAME,
    CLINICAL_WORKFLOW_SPEC,
    TOOL_REGISTRY,
    ToolSchemaValidationError,
    render_tool_registry_document,
    validate_registered_workflow_artifact,
)


def _install_synthetic_vocabulary(cache_root: Path) -> None:
    vocabulary = cache_root / "grounding" / "icd10cm" / "concepts.jsonl"
    vocabulary.parent.mkdir(parents=True)
    vocabulary.write_text(
        json.dumps(
            {
                "aliases": ["CONDITION"],
                "canonical_term": "Aster syndrome",
                "concept_id": "SYN-ASTER-1251",
                "system": "icd10cm",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


class FakeFastMCP:
    """Capture registered resources and prompts without importing the MCP SDK."""

    def __init__(self) -> None:
        self.resources: dict[str, Any] = {}
        self.prompts: dict[str, Any] = {}

    def resource(self, uri: str, **metadata: Any):
        del metadata

        def _decorator(func: Any) -> Any:
            self.resources[uri] = func
            return func

        return _decorator

    def prompt(self, *, name: str, **metadata: Any):
        del metadata

        def _decorator(func: Any) -> Any:
            self.prompts[name] = func
            return func

        return _decorator


def test_clinical_workflow_is_discoverable_without_hardcoded_tool_schemas() -> None:
    registry = render_tool_registry_document()
    workflow = next(
        item for item in registry["workflows"] if item["name"] == CLINICAL_WORKFLOW_NAME
    )

    assert registry["schema_version"] == "1.1.0"
    assert workflow["prompt_name"] == CLINICAL_WORKFLOW_SPEC.prompt_name
    assert workflow["resource_uri"] == CLINICAL_WORKFLOW_SPEC.resource_uri
    assert workflow["fixture_uri"] == CLINICAL_WORKFLOW_SPEC.fixture_uri
    assert workflow["default_execution"] == "local"
    assert workflow["tools"] == list(CLINICAL_WORKFLOW_SPEC.tools)
    assert workflow["stage_order"] == list(CLINICAL_WORKFLOW_SPEC.stage_order)
    assert [item["name"] for item in workflow["artifacts"]] == list(
        CLINICAL_WORKFLOW_SPEC.stage_order
    )
    assert all("schema_id" in item for item in workflow["artifacts"])
    assert all("schema" not in item for item in workflow["artifacts"])


def test_clinical_prompt_and_resources_are_exposed_with_privacy_guidance() -> None:
    fake = FakeFastMCP()
    mcp_server._register_resources(fake)
    mcp_server._register_prompts(fake)

    assert CLINICAL_WORKFLOW_SPEC.prompt_name in fake.prompts
    assert CLINICAL_WORKFLOW_SPEC.resource_uri in fake.resources
    assert CLINICAL_WORKFLOW_SPEC.fixture_uri in fake.resources

    prompt = fake.prompts[CLINICAL_WORKFLOW_SPEC.prompt_name]()
    assert "openmed_deidentify" in prompt
    assert "openmed_clinical_pipeline" in prompt
    assert "allow_external_llm=false" in prompt
    assert "zero network egress" in prompt
    assert "never log" in prompt

    document = json.loads(fake.resources[CLINICAL_WORKFLOW_SPEC.resource_uri]())
    assert document["execution"]["default"] == "local"
    assert document["execution"]["network_egress"] is False
    assert document["stage_order"] == list(CLINICAL_WORKFLOW_SPEC.stage_order)
    assert [item["name"] for item in document["artifacts"]] == list(
        CLINICAL_WORKFLOW_SPEC.stage_order
    )
    assert all("schema" in item for item in document["artifacts"])
    assert "Human review required" in document["output"]["clinical_use"]


def test_golden_agent_run_validates_every_registered_artifact_schema() -> None:
    fixture = load_golden_agent_run()
    artifacts = fixture["artifacts"]

    for artifact_name in CLINICAL_WORKFLOW_SPEC.stage_order:
        assert (
            validate_registered_workflow_artifact(
                CLINICAL_WORKFLOW_NAME,
                artifact_name,
                artifacts[artifact_name],
            )
            == artifacts[artifact_name]
        )

    assert artifacts["export"]["bundle"]["resourceType"] == "Bundle"
    assert artifacts["export"]["resource_count"] == 1
    assert artifacts["risk"]["risk_report"]["detail_level"] == "aggregate_phi_safe"
    assert artifacts["risk"]["risk_report"]["leakage_rate"] == 0.0


def test_golden_final_outputs_match_standalone_local_handlers() -> None:
    fixture = load_golden_agent_run()
    artifacts = fixture["artifacts"]
    grounded_spans = artifacts["ground"]["spans"]

    exported = mcp_server.openmed_export_fhir(
        spans=grounded_spans,
        doc_id="synthetic-agent-run-1303",
        bundle_type="collection",
    )
    risk = mcp_server.openmed_risk_score(spans=grounded_spans)

    assert exported == artifacts["export"]
    assert risk == artifacts["risk"]


def test_golden_note_pipeline_matches_chained_handlers_without_egress_or_phi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fixture = load_golden_agent_run()
    raw_note = fixture["input"]["note"]
    deidentified_text = fixture["artifacts"]["deidentify"]["deidentified_text"]
    surface = "aster syndrome"
    start = deidentified_text.index(surface)

    class SyntheticRuntime:
        config: dict[str, Any] = {}

        @staticmethod
        def get_loader() -> None:
            return None

        @staticmethod
        def run_model_request(
            model_name: str,
            keep_alive: Any,
            operation: Any,
        ) -> dict[str, Any]:
            del model_name, keep_alive
            return operation()

    def deidentify(text: str, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        assert text == raw_note
        return {
            "original_text": text,
            "deidentified_text": deidentified_text,
            "pii_entities": [],
            "method": "mask",
            "timestamp": "2026-08-03T00:00:00+00:00",
            "num_entities_redacted": 2,
            "metadata": {"synthetic": True},
            "audit_report": None,
        }

    def analyze(text: str, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        assert text == deidentified_text
        return {
            "text": text,
            "entities": [
                {
                    "text": surface,
                    "label": "CONDITION",
                    "confidence": 0.99,
                    "start": start,
                    "end": start + len(surface),
                    "metadata": {"synthetic": True},
                }
            ],
            "model_name": "synthetic-offline-model",
        }

    def fail_network(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("canonical clinical workflow attempted network egress")

    _install_synthetic_vocabulary(tmp_path)
    monkeypatch.setenv("OPENMED_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(mcp_server.openmed, "deidentify", deidentify)
    monkeypatch.setattr(mcp_server, "openmed_analyze_text", analyze)
    monkeypatch.setattr(socket.socket, "connect", fail_network)
    monkeypatch.setattr(socket.socket, "connect_ex", fail_network)
    monkeypatch.setattr(socket, "create_connection", fail_network)

    deidentified = mcp_server.openmed_deidentify(
        raw_note,
        runtime_provider=SyntheticRuntime,
    )
    assert deidentified["deidentified_text"] == deidentified_text

    provided = mcp_server.openmed_clinical_pipeline(
        stages=["detect"],
        spans=fixture["artifacts"]["detect"]["spans"],
    )
    assert provided["status"] == "completed"
    assert provided["artifacts"]["detect"]["model_name"] == "provided-openmed-spans"

    early = mcp_server.openmed_clinical_pipeline(
        stages=list(CLINICAL_STAGE_ORDER[:4]),
        text=deidentified_text,
        options={"detect": {"doc_id": "synthetic-agent-run-1251"}},
    )
    assert early["status"] == "completed"
    chained_spans = early["artifacts"]["relations"]["spans"]
    expected_ground = mcp_server.openmed_ground(
        spans=chained_spans,
        vocabularies=["icd10cm"],
        max_candidates=1,
    )
    expected_export = mcp_server.openmed_export_fhir(
        spans=expected_ground["spans"],
        doc_id="synthetic-agent-run-1251",
    )
    expected_risk = mcp_server.openmed_risk_score(spans=expected_export["spans"])
    assert expected_ground["status"] == "completed"
    assert expected_ground["grounded_concepts"][0]["code"] == "SYN-ASTER-1251"
    assert expected_export["status"] == "completed"
    assert expected_export["resource_count"] == 1
    assert expected_risk["status"] == "completed"

    with caplog.at_level(logging.DEBUG):
        composed = mcp_server.openmed_clinical_pipeline(
            stages=list(CLINICAL_STAGE_ORDER),
            text=deidentified_text,
            options={
                "detect": {"doc_id": "synthetic-agent-run-1251"},
                "ground": {
                    "vocabularies": ["icd10cm"],
                    "max_candidates": 1,
                },
                "export": {"doc_id": "synthetic-agent-run-1251"},
            },
        )

    assert composed["status"] == "completed"
    for stage in CLINICAL_STAGE_ORDER:
        assert (
            validate_registered_workflow_artifact(
                CLINICAL_WORKFLOW_NAME,
                stage,
                composed["artifacts"][stage],
            )
            == composed["artifacts"][stage]
        )
    assert composed["artifacts"]["ground"] == expected_ground
    assert composed["artifacts"]["export"] == expected_export
    assert composed["artifacts"]["risk"] == expected_risk
    assert composed["final_output"] == expected_risk
    composed_entries = composed["artifacts"]["export"]["bundle"]["entry"]
    assert composed_entries == expected_export["bundle"]["entry"]

    serialized = json.dumps(composed, sort_keys=True)
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert raw_note not in serialized
    assert raw_note not in logged
    for marker in fixture["input"]["synthetic_direct_identifiers"]:
        assert marker not in serialized
        assert marker not in logged


@pytest.mark.parametrize("artifact_name", CLINICAL_WORKFLOW_SPEC.stage_order)
def test_golden_agent_run_artifact_schemas_reject_invalid_payloads(
    artifact_name: str,
) -> None:
    fixture = load_golden_agent_run()
    invalid = deepcopy(fixture["artifacts"][artifact_name])
    invalid.clear()

    with pytest.raises(ToolSchemaValidationError, match=artifact_name):
        validate_registered_workflow_artifact(
            CLINICAL_WORKFLOW_NAME,
            artifact_name,
            invalid,
        )


def test_golden_post_deidentification_outputs_and_logs_exclude_source_markers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake = FakeFastMCP()
    mcp_server._register_resources(fake)

    with caplog.at_level(logging.DEBUG):
        fixture = json.loads(fake.resources[CLINICAL_WORKFLOW_SPEC.fixture_uri]())

    artifacts = json.dumps(fixture["artifacts"], sort_keys=True)
    for marker in fixture["input"]["synthetic_direct_identifiers"]:
        assert marker in fixture["input"]["note"]
        assert marker not in artifacts
    assert fixture["input"]["note"] not in artifacts
    assert "original_text" not in artifacts
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert fixture["input"]["note"] not in logged
    assert all(
        marker not in logged
        for marker in fixture["input"]["synthetic_direct_identifiers"]
    )


def test_workflow_registry_requires_known_tools() -> None:
    assert {
        name for name in CLINICAL_WORKFLOW_SPEC.tools if TOOL_REGISTRY.get(name)
    } == set(CLINICAL_WORKFLOW_SPEC.tools)
