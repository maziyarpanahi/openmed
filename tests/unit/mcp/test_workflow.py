"""Tests for MCP workflow orchestration."""

from __future__ import annotations

import json
import socket
from collections import Counter
from types import SimpleNamespace
from typing import Any

import pytest

from openmed.mcp import server as mcp_server
from openmed.mcp.server import _register_tools
from openmed.mcp.tool_registry import (
    CLINICAL_STAGE_ORDER,
    TOOL_REGISTRY,
    ToolSchemaValidationError,
    validate_registered_tool_output,
)
from openmed.mcp.workflow import (
    ClinicalStageOrderError,
    TransientWorkflowError,
    WorkflowRunner,
    WorkflowStateStore,
    builtin_workflow_step_executors,
    execute_clinical_pipeline,
    plan_clinical_pipeline,
    validate_clinical_stage_order,
)

PHI_NOTE = "Patient Jane Doe called 555-1212 about diabetes."
SYNTHETIC_CLINICAL_SPAN = {
    "schema_version": 1,
    "doc_id": "synthetic-note-1302",
    "start": 0,
    "end": 9,
    "text_hash": f"hmac-sha256:{'0' * 64}",
    "entity_type": "CONDITION",
    "canonical_label": "CONDITION",
    "policy_label": "CLINICAL_CONCEPT",
    "regulatory_tags": [],
    "score": 0.95,
    "detector": "synthetic-offline-detector",
    "evidence": {"synthetic": True},
    "action": "keep",
    "replacement": None,
    "reversible_id": None,
    "section": None,
    "metadata": {"synthetic": True},
}


def _redact(text: str) -> str:
    return (
        text.replace("Jane Doe", "[NAME]")
        .replace("555-1212", "[PHONE]")
        .replace("MRN123", "[MRN]")
    )


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def test_four_step_pipeline_passes_intermediates_by_handle_without_phi_egress():
    store = WorkflowStateStore()
    counters: Counter[str] = Counter()

    def extract(text: str) -> dict[str, Any]:
        counters["extract"] += 1
        assert text == PHI_NOTE
        return {
            "entities": [
                {"text": "Jane Doe", "label": "PERSON", "start": 8, "end": 16},
                {
                    "text": "diabetes",
                    "label": "DISEASE",
                    "system": "snomed",
                    "code": "73211009",
                    "display": "Diabetes mellitus",
                },
            ],
            "source_text": text,
        }

    def deidentify(text: Any) -> dict[str, str]:
        counters["deidentify"] += 1
        raw = _json_text(text)
        return {"original_text": raw, "deidentified_text": _redact(raw)}

    executors = builtin_workflow_step_executors()
    executors.update(
        {
            "openmed_extract_pii": extract,
            "openmed_deidentify": deidentify,
        }
    )
    runner = WorkflowRunner(
        store=store,
        executors=executors,
        deidentifier=_redact,
    )

    pipeline = {
        "session_id": "session-637",
        "workflow_id": "extract-map-export-deidentify",
        "steps": [
            {
                "id": "extract",
                "tool": "openmed_extract_pii",
                "inputs": {"text": PHI_NOTE},
                "return_output": True,
            },
            {
                "id": "map",
                "tool": "openmed_map_concepts",
                "inputs": {"entities": {"from_step": "extract", "path": "entities"}},
            },
            {
                "id": "export",
                "tool": "openmed_export_fhir",
                "inputs": {
                    "concepts": {"from_step": "map", "path": "concepts"},
                    "doc_id": "doc-637",
                },
            },
            {
                "id": "deidentify",
                "tool": "openmed_deidentify",
                "inputs": {"text": {"from_step": "export"}},
            },
        ],
    }

    result = runner.run(pipeline)

    assert result["status"] == "completed"
    assert set(result["handles"]) == {"extract", "map", "export", "deidentify"}
    assert result["final_handle"] == result["handles"]["deidentify"]
    assert counters == {"extract": 1, "deidentify": 1}
    assert validate_registered_tool_output("openmed_run_workflow", result) == result

    raw_extract = store.get(result["session_id"], result["handles"]["extract"])
    assert raw_extract["source_text"] == PHI_NOTE

    surfaced = json.dumps(result, sort_keys=True)
    assert "Jane Doe" not in surfaced
    assert "555-1212" not in surfaced
    assert "[NAME]" in surfaced
    assert "[PHONE]" in surfaced


def test_transient_failure_at_step_three_retries_without_rerunning_prior_steps():
    store = WorkflowStateStore()
    counters: Counter[str] = Counter()

    def step_one() -> dict[str, str]:
        counters["step1"] += 1
        return {"value": "one"}

    def step_two(value: str) -> dict[str, str]:
        counters["step2"] += 1
        return {"value": f"{value}-two"}

    def step_three(value: str) -> dict[str, str]:
        counters["step3"] += 1
        if counters["step3"] == 1:
            raise TransientWorkflowError("temporary backend timeout")
        return {"value": f"{value}-three"}

    def step_four(value: str) -> dict[str, str]:
        counters["step4"] += 1
        return {"value": f"{value}-four"}

    runner = WorkflowRunner(
        store=store,
        executors={
            "openmed_extract_pii": step_one,
            "openmed_map_concepts": step_two,
            "openmed_export_fhir": step_three,
            "openmed_deidentify": step_four,
        },
        deidentifier=_redact,
    )
    result = runner.run(
        {
            "session_id": "retry-session",
            "workflow_id": "retry-workflow",
            "steps": [
                {"id": "step1", "tool": "openmed_extract_pii"},
                {
                    "id": "step2",
                    "tool": "openmed_map_concepts",
                    "inputs": {"value": {"from_step": "step1", "path": "value"}},
                },
                {
                    "id": "step3",
                    "tool": "openmed_export_fhir",
                    "inputs": {"value": {"from_step": "step2", "path": "value"}},
                    "retry": {"max_retries": 1},
                },
                {
                    "id": "step4",
                    "tool": "openmed_deidentify",
                    "inputs": {"value": {"from_step": "step3", "path": "value"}},
                },
            ],
        }
    )

    assert result["status"] == "completed"
    assert counters == {"step1": 1, "step2": 1, "step3": 2, "step4": 1}
    step3_trace = next(item for item in result["trace"] if item["step_id"] == "step3")
    assert step3_trace["status"] == "completed"
    assert step3_trace["retry_count"] == 1
    assert step3_trace["attempt_count"] == 2


def test_failed_workflow_resumes_prior_completed_steps_on_next_call():
    store = WorkflowStateStore()
    counters: Counter[str] = Counter()
    fail_step_three = True

    def step_one() -> dict[str, str]:
        counters["step1"] += 1
        return {"value": "one"}

    def step_two(value: str) -> dict[str, str]:
        counters["step2"] += 1
        return {"value": f"{value}-two"}

    def step_three(value: str) -> dict[str, str]:
        counters["step3"] += 1
        if fail_step_three:
            raise TransientWorkflowError("temporary backend timeout")
        return {"value": f"{value}-three"}

    runner = WorkflowRunner(
        store=store,
        executors={
            "step1": step_one,
            "step2": step_two,
            "step3": step_three,
        },
        deidentifier=_redact,
    )
    pipeline = {
        "session_id": "resume-session",
        "workflow_id": "resume-workflow",
        "steps": [
            {"id": "step1", "tool": "step1"},
            {
                "id": "step2",
                "tool": "step2",
                "inputs": {"value": {"from_step": "step1", "path": "value"}},
            },
            {
                "id": "step3",
                "tool": "step3",
                "inputs": {"value": {"from_step": "step2", "path": "value"}},
            },
        ],
    }

    first = runner.run(pipeline)
    fail_step_three = False
    second = runner.run(pipeline)

    assert first["status"] == "failed"
    assert second["status"] == "completed"
    assert counters == {"step1": 1, "step2": 1, "step3": 2}
    assert [item["status"] for item in second["trace"]] == [
        "resumed",
        "resumed",
        "completed",
    ]


def test_conditional_gate_skips_step_without_materializing_output():
    runner = WorkflowRunner(
        store=WorkflowStateStore(),
        executors={
            "gate": lambda: {"run": False},
            "echo": lambda: {"message": "should not run"},
        },
        deidentifier=_redact,
    )

    result = runner.run(
        {
            "steps": [
                {"id": "gate", "tool": "gate"},
                {
                    "id": "echo",
                    "tool": "echo",
                    "condition": {
                        "from_step": "gate",
                        "path": "run",
                        "operator": "truthy",
                    },
                },
            ]
        }
    )

    assert result["status"] == "completed"
    assert set(result["handles"]) == {"gate"}
    assert result["trace"][1]["status"] == "skipped"


def test_returned_outputs_are_redacted_by_default_and_raw_only_when_allowed():
    def echo() -> dict[str, str]:
        return {"message": "Jane Doe has MRN123"}

    redacted_runner = WorkflowRunner(
        store=WorkflowStateStore(),
        executors={"echo": echo},
        deidentifier=_redact,
    )
    redacted = redacted_runner.run(
        {
            "steps": [
                {"id": "echo", "tool": "echo", "return_output": True},
            ]
        },
        workflow_id="redacted-egress",
    )

    assert redacted["outputs"]["echo"]["message"] == "[NAME] has [MRN]"
    assert redacted["final_output"]["message"] == "[NAME] has [MRN]"

    raw_runner = WorkflowRunner(
        store=WorkflowStateStore(),
        executors={"echo": echo},
        deidentifier=_redact,
    )
    raw = raw_runner.run(
        {
            "steps": [
                {
                    "id": "echo",
                    "tool": "echo",
                    "return_output": True,
                    "allow_raw_output": True,
                },
            ]
        },
        workflow_id="raw-egress",
    )

    assert raw["outputs"]["echo"]["message"] == "Jane Doe has MRN123"
    assert raw["final_output"]["message"] == "Jane Doe has MRN123"


def test_execution_trace_contains_only_phi_free_metadata():
    runner = WorkflowRunner(
        store=WorkflowStateStore(),
        executors={"echo": lambda text: {"text": text}},
        deidentifier=_redact,
    )
    result = runner.run(
        {
            "steps": [
                {
                    "id": "extract",
                    "tool": "echo",
                    "inputs": {"text": PHI_NOTE},
                    "return_output": True,
                }
            ]
        }
    )

    trace_text = json.dumps(result["trace"], sort_keys=True)
    assert "Jane Doe" not in trace_text
    assert "555-1212" not in trace_text
    assert "extract" in trace_text
    assert "output_handle" in trace_text


def test_workflow_tool_is_registered_and_schema_validated():
    definition = TOOL_REGISTRY.get("openmed_run_workflow")
    assert definition.name == "openmed_run_workflow"
    assert definition.output_schema["required"]

    valid_payload = {
        "schema_version": "openmed.workflow.v1",
        "session_id": "s",
        "workflow_id": "w",
        "status": "completed",
        "handles": {},
        "final_handle": None,
        "final_output": None,
        "outputs": {},
        "trace": [],
    }
    assert (
        validate_registered_tool_output("openmed_run_workflow", valid_payload)
        == valid_payload
    )

    invalid_payload = dict(valid_payload)
    invalid_payload.pop("trace")
    with pytest.raises(ToolSchemaValidationError):
        validate_registered_tool_output("openmed_run_workflow", invalid_payload)

    class FakeServer:
        def __init__(self) -> None:
            self.tools: list[str] = []

        def tool(self, *, name: str, **metadata: Any):
            del metadata
            self.tools.append(name)

            def decorator(func):
                return func

            return decorator

    fake_server = FakeServer()
    _register_tools(fake_server, runtime_provider=None)

    assert "openmed_run_workflow" in fake_server.tools


def test_clinical_stage_order_accepts_canonical_subsequences() -> None:
    stages = [" Detect ", "sections", "ground", "risk"]

    normalized = validate_clinical_stage_order(stages)
    plan = plan_clinical_pipeline(stages)

    assert normalized == ("detect", "sections", "ground", "risk")
    assert plan["status"] == "planned"
    assert plan["stages"] == list(normalized)
    assert [item["status"] for item in plan["trace"]] == ["planned"] * 4
    assert validate_registered_tool_output("openmed_clinical_pipeline", plan) == plan


def test_invalid_clinical_stage_order_returns_error_without_work() -> None:
    calls: Counter[str] = Counter()

    def record(stage: str) -> dict[str, str]:
        calls[stage] += 1
        return {"stage": stage}

    callbacks = {
        stage: (lambda stage=stage: record(stage)) for stage in CLINICAL_STAGE_ORDER
    }

    plan = plan_clinical_pipeline(
        ["detect", "risk", "ground"],
        stage_callbacks=callbacks,
    )
    execution = execute_clinical_pipeline(
        ["detect", "risk", "ground"],
        text=None,
        spans=[SYNTHETIC_CLINICAL_SPAN],
        stage_handlers={
            stage: (lambda artifact, options, stage=stage: record(stage))
            for stage in CLINICAL_STAGE_ORDER
        },
    )

    assert plan["status"] == "rejected"
    assert execution["status"] == "rejected"
    assert execution["error"] == plan["error"]
    assert plan["stages"] == ["detect", "risk", "ground"]
    assert plan["error"] == {
        "code": "invalid_stage_order",
        "message": "Clinical pipeline stages are not in canonical order.",
        "stage": "ground",
        "details": {
            "allowed_order": list(CLINICAL_STAGE_ORDER),
            "declared_index": 2,
            "previous_stage": "risk",
        },
    }
    assert calls == {}
    assert plan["artifacts"] == {}
    assert plan["trace"] == []
    assert validate_registered_tool_output("openmed_clinical_pipeline", plan) == plan


def test_unknown_clinical_stage_error_does_not_echo_input() -> None:
    private_marker = "SYNTHETIC-PRIVATE-MARKER"

    with pytest.raises(ClinicalStageOrderError) as exc_info:
        validate_clinical_stage_order(["detect", private_marker])

    plan = plan_clinical_pipeline(["detect", private_marker])
    serialized = json.dumps(plan, sort_keys=True)

    assert exc_info.value.error["code"] == "unknown_stage"
    assert plan["status"] == "rejected"
    assert plan["stages"] == ["detect"]
    assert private_marker not in serialized
    assert validate_registered_tool_output("openmed_clinical_pipeline", plan) == plan


def test_clinical_pipeline_preserves_text_free_spans_across_local_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    note = "Assessment:\nNo synthetic pneumonia finding."
    surface = "pneumonia"
    start = note.index(surface)

    def analyze(text: str, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {
            "text": text,
            "entities": [
                {
                    "text": surface,
                    "label": "CONDITION",
                    "confidence": 0.97,
                    "start": start,
                    "end": start + len(surface),
                    "metadata": {},
                }
            ],
            "model_name": "synthetic-offline-model",
        }

    monkeypatch.setattr(mcp_server, "openmed_analyze_text", analyze)

    result = mcp_server.openmed_clinical_pipeline(
        stages=["detect", "context", "sections", "relations"],
        text=note,
        options={
            "detect": {"doc_id": "synthetic-note-1302"},
            "context": {"language": "en"},
            "sections": {"language": "en"},
            "relations": {"language": "en"},
        },
    )

    assert result["status"] == "completed"
    assert [item["stage"] for item in result["trace"]] == [
        "detect",
        "context",
        "sections",
        "relations",
    ]
    for stage in result["stages"]:
        spans = result["artifacts"][stage]["spans"]
        assert len(spans) == 1
        assert set(spans[0]) == set(SYNTHETIC_CLINICAL_SPAN)
    context = result["artifacts"]["context"]["spans"][0]["metadata"]
    assert context["clinical_context"]["negation"] == "negated"
    assert result["artifacts"]["sections"]["spans"][0]["section"] is not None
    assert result["artifacts"]["relations"]["relations"] == []
    assert surface not in json.dumps(result, sort_keys=True)
    assert note not in json.dumps(result, sort_keys=True)


def test_clinical_pipeline_matches_chained_standalone_stage_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def ground(
        spans: list[dict[str, Any]],
        vocabularies: list[str] | None = None,
        max_candidates: int = 5,
        allow_external_llm: bool = False,
    ) -> dict[str, Any]:
        calls.append("ground")
        assert vocabularies == ["icd10cm"]
        assert max_candidates == 2
        assert allow_external_llm is False
        return {
            "schema_version": "openmed.ground.v1",
            "status": "completed",
            "spans": spans,
            "grounded_concepts": [{"system": "ICD10CM", "code": "SYNTHETIC-001"}],
            "error": None,
        }

    def export(
        spans: list[dict[str, Any]],
        resources: list[dict[str, Any]] | None = None,
        doc_id: str = "workflow",
        bundle_type: str = "collection",
    ) -> dict[str, Any]:
        del resources
        calls.append("export")
        assert doc_id == "synthetic-note-1302"
        assert bundle_type == "collection"
        bundle = {"resourceType": "Bundle", "type": bundle_type, "entry": []}
        return {
            "schema_version": "openmed.export_fhir.v1",
            "status": "completed",
            "spans": spans,
            "bundle": bundle,
            "resource_count": 0,
            "error": None,
        }

    def risk(
        spans: list[dict[str, Any]],
        deidentified_text: str | None = None,
        records: list[dict[str, Any]] | None = None,
        quasi_identifiers: list[str] | None = None,
    ) -> dict[str, Any]:
        del deidentified_text, records, quasi_identifiers
        calls.append("risk")
        return {
            "schema_version": "openmed.risk_score.v1",
            "status": "completed",
            "spans": spans,
            "risk_report": {"risk": 0.0},
            "error": None,
        }

    monkeypatch.setattr(mcp_server, "openmed_ground", ground)
    monkeypatch.setattr(mcp_server, "openmed_export_fhir", export)
    monkeypatch.setattr(mcp_server, "openmed_risk_score", risk)
    span = dict(SYNTHETIC_CLINICAL_SPAN)
    expected_ground = ground([span], ["icd10cm"], 2, False)
    expected_export = export(
        expected_ground["spans"],
        doc_id="synthetic-note-1302",
    )
    expected_risk = risk(expected_export["spans"])
    calls.clear()

    result = mcp_server.openmed_clinical_pipeline(
        stages=["ground", "export", "risk"],
        spans=[span],
        options={
            "ground": {"vocabularies": ["icd10cm"], "max_candidates": 2},
            "export": {"doc_id": "synthetic-note-1302"},
        },
    )

    assert result["status"] == "completed"
    assert calls == ["ground", "export", "risk"]
    assert result["artifacts"] == {
        "ground": expected_ground,
        "export": expected_export,
        "risk": expected_risk,
    }
    assert result["final_output"] == expected_risk


def test_external_grounding_is_reachable_only_through_privacy_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []
    span = dict(SYNTHETIC_CLINICAL_SPAN)
    external_output = {
        "schema_version": "openmed.ground.v1",
        "status": "completed",
        "spans": [span],
        "grounded_concepts": [{"system": "ICD10CM", "code": "SYNTHETIC-GATEWAY"}],
        "error": None,
    }

    class Gateway:
        def complete(self, prompt: str) -> SimpleNamespace:
            requests.append(json.loads(prompt))
            return SimpleNamespace(
                reidentified_text=json.dumps(external_output, sort_keys=True)
            )

    def direct_ground(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        raise AssertionError("external grounding bypassed the privacy gateway")

    monkeypatch.setattr(mcp_server, "openmed_ground", direct_ground)

    result = mcp_server.openmed_clinical_pipeline(
        stages=["ground"],
        spans=[span],
        options={"ground": {"vocabularies": ["icd10cm"]}},
        allow_external_llm=True,
        privacy_gateway_provider=Gateway,
    )

    assert result["status"] == "completed"
    assert result["artifacts"]["ground"] == external_output
    assert len(requests) == 1
    assert requests[0] == {
        "schema_version": "openmed.clinical_artifact.v1",
        "stage": "ground",
        "spans": [span],
        "options": {"vocabularies": ["icd10cm"]},
    }
    assert "text" not in requests[0]


def test_external_pipeline_fails_closed_before_work_without_gateway() -> None:
    calls: Counter[str] = Counter()

    def handler(
        artifact: Any,
        stage_options: Any,
    ) -> dict[str, Any]:
        del artifact, stage_options
        calls["handler"] += 1
        return {}

    result = execute_clinical_pipeline(
        ["detect", "ground"],
        text=None,
        spans=[SYNTHETIC_CLINICAL_SPAN],
        stage_handlers={"detect": handler, "ground": handler},
        allow_external_llm=True,
        external_stage_gateway=None,
    )

    assert result["status"] == "failed"
    assert result["error"]["code"] == "privacy_gateway_required"
    assert result["trace"] == []
    assert calls == {}


def test_default_local_pipeline_runs_with_all_socket_connections_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_network(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("default clinical pipeline attempted network egress")

    monkeypatch.setattr(socket.socket, "connect", fail_network)
    monkeypatch.setattr(socket.socket, "connect_ex", fail_network)
    monkeypatch.setattr(socket, "create_connection", fail_network)

    result = mcp_server.openmed_clinical_pipeline(
        stages=["ground", "export", "risk"],
        spans=[dict(SYNTHETIC_CLINICAL_SPAN)],
    )

    assert result["status"] == "completed"
    assert [item["status"] for item in result["trace"]] == ["completed"] * 3


def test_invalid_intermediate_span_stops_downstream_work_without_echoing() -> None:
    calls: Counter[str] = Counter()
    private_marker = "SYNTHETIC-PRIVATE-SURFACE"

    def context_stage(artifact: Any, options: Any) -> dict[str, Any]:
        del artifact, options
        calls["context"] += 1
        invalid = dict(SYNTHETIC_CLINICAL_SPAN)
        invalid["text"] = private_marker
        return {"spans": [invalid]}

    def sections_stage(artifact: Any, options: Any) -> dict[str, Any]:
        del artifact, options
        calls["sections"] += 1
        return {"spans": [SYNTHETIC_CLINICAL_SPAN]}

    result = execute_clinical_pipeline(
        ["context", "sections"],
        text=None,
        spans=[SYNTHETIC_CLINICAL_SPAN],
        stage_handlers={
            "context": context_stage,
            "sections": sections_stage,
        },
    )

    assert result["status"] == "failed"
    assert result["error"]["code"] == "stage_execution_failed"
    assert result["error"]["stage"] == "context"
    assert result["artifacts"] == {}
    assert calls == {"context": 1}
    assert private_marker not in json.dumps(result, sort_keys=True)


def test_invalid_intermediate_artifact_schema_stops_downstream_work() -> None:
    calls: Counter[str] = Counter()

    def context_stage(artifact: Any, options: Any) -> dict[str, Any]:
        del artifact, options
        calls["context"] += 1
        return {"spans": [SYNTHETIC_CLINICAL_SPAN]}

    def sections_stage(artifact: Any, options: Any) -> dict[str, Any]:
        del artifact, options
        calls["sections"] += 1
        return {"spans": [SYNTHETIC_CLINICAL_SPAN], "sections": []}

    result = execute_clinical_pipeline(
        ["context", "sections"],
        text=None,
        spans=[SYNTHETIC_CLINICAL_SPAN],
        stage_handlers={
            "context": context_stage,
            "sections": sections_stage,
        },
    )

    assert result["status"] == "failed"
    assert result["error"] == {
        "code": "stage_execution_failed",
        "message": "A clinical pipeline stage failed.",
        "stage": "context",
        "details": {"error_type": "ToolSchemaValidationError"},
    }
    assert result["artifacts"] == {}
    assert result["trace"] == []
    assert calls == {"context": 1}
