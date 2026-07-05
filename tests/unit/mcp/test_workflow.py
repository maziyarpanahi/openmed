"""Tests for MCP workflow orchestration."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import pytest

from openmed.mcp.server import _register_tools
from openmed.mcp.tool_registry import (
    TOOL_REGISTRY,
    ToolSchemaValidationError,
    validate_registered_tool_output,
)
from openmed.mcp.workflow import (
    TransientWorkflowError,
    WorkflowRunner,
    WorkflowStateStore,
    builtin_workflow_step_executors,
    plan_clinical_pipeline,
)

PHI_NOTE = "Patient Jane Doe called 555-1212 about diabetes."


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

        def tool(self, *, name: str):
            self.tools.append(name)

            def decorator(func):
                return func

            return decorator

    fake_server = FakeServer()
    _register_tools(fake_server, runtime_provider=None)

    assert "openmed_run_workflow" in fake_server.tools


def test_clinical_pipeline_rejects_illegal_order_without_running_callbacks():
    counters: Counter[str] = Counter()

    result = plan_clinical_pipeline(
        ["ground", "detect"],
        stage_callbacks={
            "ground": lambda: counters.update(["ground"]),
            "detect": lambda: counters.update(["detect"]),
        },
    )

    assert result["status"] == "rejected"
    assert result["error"]["code"] == "invalid_stage_order"
    assert result["error"]["stage"] == "detect"
    assert counters == {}
    assert (
        validate_registered_tool_output("openmed_clinical_pipeline", result) == result
    )


def test_clinical_pipeline_accepts_ordered_stage_subsequence():
    counters: Counter[str] = Counter()

    def _stage(name: str) -> dict[str, str]:
        counters.update([name])
        return {"stage": name}

    result = plan_clinical_pipeline(
        ["detect", "ground", "risk"],
        stage_callbacks={
            "detect": lambda: _stage("detect"),
            "ground": lambda: _stage("ground"),
            "risk": lambda: _stage("risk"),
        },
    )

    assert result["status"] == "completed"
    assert result["stages"] == ["detect", "ground", "risk"]
    assert result["final_output"] == {"stage": "risk"}
    assert [item["stage"] for item in result["trace"]] == [
        "detect",
        "ground",
        "risk",
    ]
    assert counters == {"detect": 1, "ground": 1, "risk": 1}
    assert (
        validate_registered_tool_output("openmed_clinical_pipeline", result) == result
    )
