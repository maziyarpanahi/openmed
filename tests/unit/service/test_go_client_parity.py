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
    "/fhir/smart-backend/ingestions/{job_id}/summary": (
        "SMARTBackendIngestionSummary"
    ),
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

        for field_name in sorted(request_schema.get("properties", {})):
            assert re.search(rf'`json:"{re.escape(field_name)}[",]', source), (
                f"{field_name!r} from {path} is missing a matching "
                f"json tag in the Go source."
            )


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
