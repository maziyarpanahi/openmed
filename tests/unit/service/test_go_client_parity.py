"""Parity checks for the Go REST client SDK.

These tests are a structural drift guard: they assert that the committed Go
client in ``clients/go`` stays aligned with the service surface published in the
OpenAPI spec. They do not require a Go toolchain -- they read the Go source as
text so they run inside the standard Python test suite.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
OPENAPI_PATH = ROOT / "docs/api/openapi.json"
SDK_ROOT = ROOT / "clients/go"
SDK_SOURCE_PATH = SDK_ROOT / "client.go"
SDK_README_PATH = SDK_ROOT / "README.md"
SDK_GOMOD_PATH = SDK_ROOT / "go.mod"

# Every OpenAPI path maps to exactly one exported Go client method.
CLIENT_METHOD_BY_PATH = {
    "/analyze": "Analyze",
    "/fhir/smart-backend/ingestions": "StartSMARTBackendIngestion",
    "/fhir/smart-backend/ingestions/{job_id}": "SMARTBackendIngestionStatus",
    "/fhir/smart-backend/ingestions/{job_id}/summary": "SMARTBackendIngestionSummary",
    "/health": "Health",
    "/jobs": "CreateJob",
    "/jobs/{job_id}": "GetJob",
    "/livez": "Livez",
    "/models/loaded": "LoadedModels",
    "/models/unload": "UnloadModels",
    "/pii/deidentify": "Deidentify",
    "/pii/extract": "ExtractPII",
    "/pii/extract/stream": "ExtractPIIStream",
    "/privacy-gateway/complete": "PrivacyGateway",
    "/readyz": "Readyz",
}

GO_REQUEST_STRUCT_BY_SCHEMA = {
    "AnalyzeRequest": "AnalyzeRequest",
    "DeidentifyJobDocument": "DeidentifyJobDocument",
    "DeidentifyJobRequest": "DeidentifyJobRequest",
    "JobWebhookRequest": "JobWebhookRequest",
    "ModelUnloadRequest": "ModelUnloadRequest",
    "PIIDeidentifyRequest": "PIIDeidentifyRequest",
    "PIIExtractRequest": "PIIExtractRequest",
    "PIIExtractStreamRequest": "PIIExtractStreamRequest",
    "PrivacyGatewayRequest": "PrivacyGatewayRequest",
    "SMARTBackendIngestionRequest": "SMARTBackendIngestionRequest",
}


def test_go_client_covers_openapi_paths_and_request_fields() -> None:
    spec = json.loads(OPENAPI_PATH.read_text(encoding="utf-8"))
    source = SDK_SOURCE_PATH.read_text(encoding="utf-8")

    assert set(spec["paths"]) == set(CLIENT_METHOD_BY_PATH)

    for path, method_name in CLIENT_METHOD_BY_PATH.items():
        assert re.search(
            rf"^func \(c \*Client\) {re.escape(method_name)}\(", source, re.MULTILINE
        ), f"{path} is missing Client.{method_name}()."
        assert f'"{path}"' in source, f"{path} is not called by the SDK."

        operation = _first_operation(spec["paths"][path], spec)
        request_schema = _request_schema(operation, spec)
        if request_schema is None:
            continue

        schema_name = _request_schema_name(operation)
        assert schema_name is not None
        assert schema_name in GO_REQUEST_STRUCT_BY_SCHEMA


def test_go_request_structs_match_openapi_components() -> None:
    spec = json.loads(OPENAPI_PATH.read_text(encoding="utf-8"))
    source = SDK_SOURCE_PATH.read_text(encoding="utf-8")

    assert set(GO_REQUEST_STRUCT_BY_SCHEMA) == _openapi_request_schema_names(spec)

    for schema_name, struct_name in GO_REQUEST_STRUCT_BY_SCHEMA.items():
        openapi_fields = set(
            spec["components"]["schemas"][schema_name].get("properties", {})
        )
        go_fields = _go_struct_json_fields(source, struct_name)
        assert go_fields == openapi_fields, (
            f"Go {struct_name} fields differ from OpenAPI {schema_name}: "
            f"missing={sorted(openapi_fields - go_fields)}, "
            f"extra={sorted(go_fields - openapi_fields)}"
        )


def test_go_pii_languages_match_core_and_openapi() -> None:
    from openmed.core.pii_i18n import SUPPORTED_LANGUAGES

    source = SDK_SOURCE_PATH.read_text(encoding="utf-8")
    go_languages = set(
        re.findall(
            r'^\s*Lang[A-Z]{2}\s+PIILanguage\s*=\s*"([a-z]{2})"$',
            source,
            flags=re.MULTILINE,
        )
    )

    spec = json.loads(OPENAPI_PATH.read_text(encoding="utf-8"))
    openapi_language_sets = {
        frozenset(properties["lang"]["enum"])
        for schema in spec["components"]["schemas"].values()
        if isinstance(schema, dict)
        for properties in [schema.get("properties", {})]
        if isinstance(properties.get("lang"), dict) and "enum" in properties["lang"]
    }

    assert go_languages == SUPPORTED_LANGUAGES
    assert openapi_language_sets == {frozenset(SUPPORTED_LANGUAGES)}


def test_go_module_has_no_external_dependencies() -> None:
    gomod = SDK_GOMOD_PATH.read_text(encoding="utf-8")

    assert re.search(r"^module\s+\S+", gomod, re.MULTILINE), (
        "go.mod must declare a module path."
    )
    assert re.search(r"^go\s+1\.\d+", gomod, re.MULTILINE), (
        "go.mod must declare a Go language version."
    )
    # No require directives => no external dependencies.
    assert "require" not in gomod, "The Go client must not declare dependencies."
    assert not (SDK_ROOT / "go.sum").exists(), (
        "A go.sum indicates external dependencies were added."
    )


def test_go_client_error_type_is_implemented_and_documented() -> None:
    source = SDK_SOURCE_PATH.read_text(encoding="utf-8")
    readme = SDK_README_PATH.read_text(encoding="utf-8")

    # Typed error carrying the service envelope, with a context on every call.
    assert "type APIError struct" in source
    assert "func (e *APIError) Error() string" in source
    assert "type ErrorEnvelope struct" in source
    assert "func AsAPIError(err error) (*APIError, bool)" in source
    assert source.count("ctx context.Context") >= len(CLIENT_METHOD_BY_PATH)

    assert "APIError" in readme
    assert "AsAPIError" in readme
    assert "error.code" in readme
    assert "error.message" in readme
    assert "error.details" in readme


def _first_operation(path_item: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    for method_name in ("get", "post", "put", "patch", "delete"):
        operation = path_item.get(method_name)
        if operation is not None:
            return operation
    raise AssertionError(f"No HTTP operation found in path item: {path_item}")


def _request_schema(
    operation: dict[str, Any], spec: dict[str, Any]
) -> dict[str, Any] | None:
    request_body = operation.get("requestBody")
    if request_body is None:
        return None

    content = request_body["content"]["application/json"]
    schema = content["schema"]
    ref = schema.get("$ref")
    if not ref:
        return schema

    schema_name = ref.removeprefix("#/components/schemas/")
    return spec["components"]["schemas"][schema_name]


def _request_schema_name(operation: dict[str, Any]) -> str | None:
    request_body = operation.get("requestBody")
    if request_body is None:
        return None
    ref = request_body["content"]["application/json"]["schema"].get("$ref")
    if ref is None:
        return None
    return ref.removeprefix("#/components/schemas/")


def _go_struct_json_fields(source: str, struct_name: str) -> set[str]:
    match = re.search(
        rf"^type {re.escape(struct_name)} struct \{{(?P<body>.*?)^\}}$",
        source,
        flags=re.DOTALL | re.MULTILINE,
    )
    assert match is not None, f"Go request struct {struct_name} is missing."
    return set(re.findall(r'`json:"([^",]+)', match.group("body")))


def _openapi_request_schema_names(spec: dict[str, Any]) -> set[str]:
    schema_names: set[str] = set()
    pending = [
        schema_name
        for path_item in spec["paths"].values()
        for operation in [_first_operation(path_item, spec)]
        if (schema_name := _request_schema_name(operation)) is not None
    ]

    while pending:
        schema_name = pending.pop()
        if schema_name in schema_names:
            continue
        schema_names.add(schema_name)
        schema = spec["components"]["schemas"][schema_name]
        pending.extend(_schema_references(schema))
    return schema_names


def _schema_references(value: Any) -> list[str]:
    if isinstance(value, dict):
        references = []
        ref = value.get("$ref")
        if isinstance(ref, str):
            references.append(ref.removeprefix("#/components/schemas/"))
        for nested in value.values():
            references.extend(_schema_references(nested))
        return references
    if isinstance(value, list):
        return [reference for item in value for reference in _schema_references(item)]
    return []
