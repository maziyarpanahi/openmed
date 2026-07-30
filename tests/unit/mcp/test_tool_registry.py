from __future__ import annotations

import inspect
import json
from copy import deepcopy
from typing import Any

import pytest

import openmed
from openmed.core.audit import (
    AuditReport,
    AuditSignature,
    AuditSpan,
    DetectorInfo,
    hash_text,
    recompute_repro_hash,
    stable_hash,
    verify_repro_hash,
)
from openmed.interop import adapter_tool_definitions, langchain, presidio
from openmed.mcp import server as mcp_server
from openmed.mcp.tool_registry import (
    TOOL_REGISTRY,
    ToolCompatibilityError,
    ToolRegistry,
    ToolSchemaValidationError,
    ToolSpec,
    check_tool_registry_compatibility,
    invoke_tool,
)


class FakeFastMCP:
    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}
        self.tool_metadata: dict[str, dict[str, Any]] = {}
        self.resources: dict[str, Any] = {}

    def tool(self, *, name: str, **metadata: Any):
        def _decorator(func):
            self.tools[name] = func
            self.tool_metadata[name] = metadata
            return func

        return _decorator

    def resource(self, uri: str, **kwargs):
        del kwargs

        def _decorator(func):
            self.resources[uri] = func
            return func

        return _decorator


def test_mcp_registers_all_tools_from_registry() -> None:
    fake = FakeFastMCP()

    mcp_server._register_tools(fake, runtime_provider=None)

    expected = {spec.name for spec in TOOL_REGISTRY.latest_specs()}
    assert set(fake.tools) == expected
    assert len(fake.tools) == len(expected)

    deidentify_signature = inspect.signature(fake.tools["openmed_deidentify"])
    assert list(deidentify_signature.parameters) == [
        parameter.name
        for parameter in TOOL_REGISTRY.get("openmed_deidentify").parameters
    ]
    assert deidentify_signature.parameters["method"].default == "mask"


def test_registered_tool_invocation_validates_structured_output() -> None:
    spec = TOOL_REGISTRY.get("openmed_list_models")

    def bad_handler(**kwargs):
        del kwargs
        return {"count": "not-an-integer", "returned": 0, "models": []}

    with pytest.raises(ToolSchemaValidationError, match="openmed_list_models"):
        invoke_tool(spec, bad_handler, category=None, pii_language=None, limit=50)


def test_mcp_language_listing_discovers_v2_registry_entries() -> None:
    payload = mcp_server.openmed_list_pii_languages()
    languages = {item["code"]: item for item in payload["languages"]}

    for code in ("bn", "ta", "zh"):
        assert languages[code]["default_pii_model"].startswith("OpenMed/OpenMed-PII-")
        assert languages[code]["model_count"] >= 1


def test_tool_registry_resource_is_generated_from_specs() -> None:
    fake = FakeFastMCP()

    mcp_server._register_resources(fake)
    payload = json.loads(fake.resources["openmed://tool-registry"]())

    assert payload["schema_version"] == "1.0.0"
    assert [tool["name"] for tool in payload["tools"]] == [
        spec.name for spec in TOOL_REGISTRY.all_specs()
    ]
    assert all(tool["version"] for tool in payload["tools"])
    assert all(tool["stability"] for tool in payload["tools"])


def test_adapter_tool_definitions_match_registry_schemas() -> None:
    langchain_defs = langchain.create_tool_definitions()
    presidio_defs = presidio.create_tool_definitions()
    registry_defs = adapter_tool_definitions("presidio")

    assert len(langchain_defs) == len(TOOL_REGISTRY.latest_specs())
    assert _schema_projection(langchain_defs) == _schema_projection(presidio_defs)
    assert _schema_projection(presidio_defs) == _schema_projection(registry_defs)
    assert _schema_projection(langchain_defs) == [
        {
            "name": spec.name,
            "input_schema": spec.input_schema,
            "output_schema": spec.output_schema,
        }
        for spec in TOOL_REGISTRY.latest_specs()
    ]


def test_breaking_schema_change_without_major_bump_fails() -> None:
    previous = TOOL_REGISTRY.latest_specs()
    target = TOOL_REGISTRY.get("openmed_analyze_text")
    input_schema = deepcopy(dict(target.input_schema))
    input_schema["properties"]["text"]["type"] = "integer"
    broken = _replace_spec(target, input_schema=input_schema)

    current = [broken if spec.name == broken.name else spec for spec in previous]

    with pytest.raises(ToolCompatibilityError, match="without version bump"):
        check_tool_registry_compatibility(previous, current)


def test_breaking_schema_change_with_major_bump_passes() -> None:
    previous = TOOL_REGISTRY.latest_specs()
    target = TOOL_REGISTRY.get("openmed_analyze_text")
    input_schema = deepcopy(dict(target.input_schema))
    input_schema["properties"]["text"]["type"] = "integer"
    bumped = _replace_spec(target, input_schema=input_schema, version="2.0.0")

    current = [bumped if spec.name == bumped.name else spec for spec in previous]

    check_tool_registry_compatibility(previous, current)


def test_registry_supports_multiple_versions_side_by_side() -> None:
    original = TOOL_REGISTRY.get("openmed_analyze_text")
    bumped = _replace_spec(original, version="2.0.0")
    registry = ToolRegistry([original, bumped])

    assert registry.get("openmed_analyze_text", "1.0.0") == original
    assert registry.get("openmed_analyze_text", "2.0.0") == bumped
    assert registry.get("openmed_analyze_text") == bumped
    assert [tool["version"] for tool in registry.document()["tools"]] == [
        "1.0.0",
        "2.0.0",
    ]


# Issue #1741
NEW_TOOLS = {
    "openmed_fhir_bundle",
    "openmed_risk_report",
    "openmed_signed_audit_report",
    "openmed_search_models",
}


def test_new_tools_have_correct_annotations() -> None:
    for tool in NEW_TOOLS:
        annotations = TOOL_REGISTRY.get(tool).annotations()
        assert annotations["readOnlyHint"] is True
        assert annotations["destructiveHint"] is False
        assert annotations["openWorldHint"] is False


# openmed_fhir_bundle specific tests.
def test_fhir_bundle_assembles_a_valid_bundle() -> None:
    bundle = mcp_server.openmed_fhir_bundle(
        resources=[
            {"resourceType": "Patient", "id": "patient-1"},
            {
                "resourceType": "Observation",
                "id": "obs-1",
                "status": "final",
                "subject": {"reference": "Patient/patient-1"},
            },
        ],
        doc_id="doc-1",
    )
    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "transaction"
    assert len(bundle["entry"]) == 2
    patient_full_url = bundle["entry"][0]["fullUrl"]
    subject = bundle["entry"][1]["resource"]["subject"]["reference"]
    assert subject == patient_full_url


def test_fhir_bundle_raises_value_error_without_resource_type() -> None:
    with pytest.raises(ValueError):
        mcp_server.openmed_fhir_bundle(resources=[{"id": "patient-1"}])


# openmed_risk_report specific tests
def _span(text, label, value, *, section="assessment"):
    start = text.index(value)
    return {
        "label": label,
        "start": start,
        "end": start + len(value),
        "metadata": {"section": section},
    }


def test_risk_report_is_schema_valid() -> None:
    text = (
        "Assessment: 94-year-old seen at North Clinic on 2024-02-03 "
        "with [RARE_CONDITION]."
    )
    report = mcp_server.openmed_risk_report(
        deidentified={
            "doc_id": "note-1",
            "text": text,
            "entities": [
                _span(text, "AGE", "94-year-old"),
                _span(text, "ORGANIZATION", "North Clinic"),
                _span(text, "DATE", "2024-02-03"),
                _span(text, "RARE_CONDITION", "[RARE_CONDITION]"),
            ],
        }
    )

    assert set(report) == {
        "leakage_rate",
        "reid_rate",
        "k_min",
        "singleton_records",
        "quasi_identifiers",
    }
    assert report["k_min"] == 1
    assert report["singleton_records"][0]["record_id"] == "note-1"
    categories = {qi["category"] for qi in report["quasi_identifiers"]}
    assert categories == {"age", "provider_institution", "date", "rare_condition"}
    age_qi = next(qi for qi in report["quasi_identifiers"] if qi["category"] == "age")
    assert age_qi["value"] == "94-year-old"
    assert age_qi["start"] == text.index("94-year-old")
    assert age_qi["section"] == "assessment"


def test_risk_report_output_is_phi_safe() -> None:
    text = (
        "Assessment: 94-year-old seen at North Clinic on 2024-02-03 "
        "with [RARE_CONDITION]."
    )
    report = mcp_server.openmed_risk_report(
        deidentified={
            "doc_id": "note-1",
            "text": text,
            "entities": [
                _span(text, "AGE", "94-year-old"),
                _span(text, "ORGANIZATION", "North Clinic"),
                _span(text, "DATE", "2024-02-03"),
                _span(text, "RARE_CONDITION", "[RARE_CONDITION]"),
            ],
        }
    )
    serialized = json.dumps(report)
    assert text not in serialized
    for quasi_identifier in report["quasi_identifiers"]:
        assert quasi_identifier["value"] != text
        assert len(quasi_identifier["value"]) < len(text)


# openmed_signed_audit_report specific tests
class _StubRuntime:
    config = None

    def get_loader(self):
        return None

    def run_model_request(self, model_name, keep_alive, operation):
        del model_name, keep_alive
        return operation()


def _sample_audit_report(text: str) -> AuditReport:
    text = "Patient John Doe called 555-1234."
    return AuditReport(
        policy="hipaa_safe_harbor",
        resolved_profile={
            "method": "mask",
            "confidence_threshold": 0.7,
            "language": "en",
        },
        detectors=[
            DetectorInfo(
                source="ml",
                model_id="unit-test-model",
                model_format="transformers",
            )
        ],
        safety_sweep={
            "source": "safety_sweep",
            "patterns_version": "safety-sweep-v1",
            "spans_added": 0,
        },
        spans=[
            AuditSpan(
                start=8,
                end=16,
                label="NAME",
                canonical_label="PERSON",
                sources=["ml"],
                confidence=0.95,
                threshold=0.7,
                action="mask",
                surrogate="[NAME]",
                text_hash=hash_text("John Doe"),
                evidence={"raw_label": "NAME", "model_id": "unit-test-model"},
                context={"before": "Patient ", "after": " called 555-1234."},
            )
        ],
        thresholds={"PERSON": 0.7},
        residual_risk={
            "projected_leakage": 0.05,
            "risk_report_record_score": 0.0,
            "risk_report": {
                "leakage_rate": 0.0,
                "reid_rate": 0.0,
                "k_min": 0,
                "singleton_records": [],
                "quasi_identifiers": [],
            },
        },
        openmed_version="2.0.0",
        manifest_hash="sha256:manifest",
        document_length=len(text),
        input_hash=hash_text(text),
        deidentified_text_hash=hash_text("Patient [NAME] called [PHONE]."),
    )


def test_signed_audit_report_is_schema_valid_and_signed(monkeypatch) -> None:
    text = "John Doe visited the clinic."

    def fake_deidentify(*args, **kwargs):
        assert kwargs.get("audit") is True
        return _sample_audit_report(text)

    monkeypatch.setattr(openmed, "deidentify", fake_deidentify)

    report = mcp_server.openmed_signed_audit_report(
        text=text,
        signing_key="unit-test-key",
        runtime_provider=_StubRuntime,
    )

    assert report["signature"]["key_id"] == "release"
    assert report["signature"]["algorithm"]
    assert report["signature"]["value"]
    assert len(report["spans"]) == 1


def test_signed_audit_report_output_is_phi_safe(monkeypatch) -> None:
    text = "John Doe visited the clinic."
    monkeypatch.setattr(
        openmed, "deidentify", lambda *a, **k: _sample_audit_report(text)
    )

    report = mcp_server.openmed_signed_audit_report(
        text=text,
        signing_key="unit-test-key",
        runtime_provider=_StubRuntime,
    )

    serialized = json.dumps(report)
    assert "John Doe" not in serialized
    for span in report["spans"]:
        assert set(span) >= {"start", "end", "label", "text_hash"}
        assert isinstance(span["start"], int)
        assert isinstance(span["end"], int)
        assert isinstance(span["text_hash"], str) and span["text_hash"]


def _replace_spec(
    spec: ToolSpec,
    *,
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
    version: str | None = None,
) -> ToolSpec:
    return ToolSpec(
        name=spec.name,
        description=spec.description,
        input_schema=input_schema or spec.input_schema,
        output_schema=output_schema or spec.output_schema,
        version=version or spec.version,
        stability=spec.stability,
        parameters=spec.parameters,
    )


def _schema_projection(definitions: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    return [
        {
            "name": definition["name"],
            "input_schema": definition["input_schema"],
            "output_schema": definition["output_schema"],
        }
        for definition in definitions
    ]
