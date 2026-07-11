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

# Every OpenAPI operation maps to exactly one exported Go client method.
CLIENT_METHOD_BY_OPERATION = {
    ("post", "/analyze"): "Analyze",
    ("post", "/fhir/smart-backend/ingestions"): "StartSMARTBackendIngestion",
    ("get", "/fhir/smart-backend/ingestions/{job_id}"): ("SMARTBackendIngestionStatus"),
    ("get", "/fhir/smart-backend/ingestions/{job_id}/summary"): (
        "SMARTBackendIngestionSummary"
    ),
    ("get", "/health"): "Health",
    ("post", "/jobs"): "CreateJob",
    ("get", "/jobs/{job_id}"): "GetJob",
    ("get", "/livez"): "Livez",
    ("get", "/models/loaded"): "LoadedModels",
    ("post", "/models/unload"): "UnloadModels",
    ("post", "/pii/deidentify"): "Deidentify",
    ("post", "/pii/extract"): "ExtractPII",
    ("post", "/pii/extract/stream"): "ExtractPIIStream",
    ("post", "/privacy-gateway/complete"): "PrivacyGateway",
    ("get", "/readyz"): "Readyz",
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

GO_NAMED_STRING_TYPE_BY_FIELD = {
    "aggregation_strategy": "AggregationStrategy",
    "lang": "PIILanguage",
    "method": "DeidentificationMethod",
    "policy": "PrivacyPolicy",
}


def test_go_client_covers_openapi_paths_and_request_fields() -> None:
    spec = json.loads(OPENAPI_PATH.read_text(encoding="utf-8"))
    source = SDK_SOURCE_PATH.read_text(encoding="utf-8")
    string_constants = dict(
        re.findall(r'^\s*(\w+)\s*=\s*"([^"]+)"$', source, re.MULTILINE)
    )

    assert set(_openapi_operations(spec)) == set(CLIENT_METHOD_BY_OPERATION)

    for (http_method, path), method_name in CLIENT_METHOD_BY_OPERATION.items():
        assert re.search(
            rf"^func \(c \*Client\) {re.escape(method_name)}\(", source, re.MULTILINE
        ), f"{http_method.upper()} {path} is missing Client.{method_name}()."
        method_body = _go_client_method_body(source, method_name)
        verb_markers = {
            "get": ("c.get(", "http.MethodGet"),
            "post": ("c.post(", "http.MethodPost"),
        }
        assert any(marker in method_body for marker in verb_markers[http_method]), (
            f"Client.{method_name} does not issue {http_method.upper()}."
        )
        assert f'"{path}"' in method_body or any(
            constant_name in method_body and constant_value == path
            for constant_name, constant_value in string_constants.items()
        ), f"Client.{method_name} does not call {path}."

        operation = spec["paths"][path][http_method]
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


def test_go_request_field_types_match_openapi() -> None:
    spec = json.loads(OPENAPI_PATH.read_text(encoding="utf-8"))
    source = SDK_SOURCE_PATH.read_text(encoding="utf-8")

    for schema_name, struct_name in GO_REQUEST_STRUCT_BY_SCHEMA.items():
        schema = spec["components"]["schemas"][schema_name]
        fields = _go_struct_fields(source, struct_name)
        for field_name, property_schema in schema.get("properties", {}).items():
            go_type = fields[field_name]["type"]
            assert _go_type_matches_schema(go_type, property_schema), (
                f"Go {struct_name}.{field_name} type {go_type!r} differs from "
                f"OpenAPI {schema_name}.{field_name}."
            )
            expected_named_type = GO_NAMED_STRING_TYPE_BY_FIELD.get(field_name)
            if expected_named_type is not None:
                assert go_type.removeprefix("*") == expected_named_type, (
                    f"Go {struct_name}.{field_name} must use "
                    f"{expected_named_type}, not {go_type}."
                )


def test_go_request_requiredness_and_zero_defaults_match_openapi() -> None:
    spec = json.loads(OPENAPI_PATH.read_text(encoding="utf-8"))
    source = SDK_SOURCE_PATH.read_text(encoding="utf-8")

    for schema_name, struct_name in GO_REQUEST_STRUCT_BY_SCHEMA.items():
        schema = spec["components"]["schemas"][schema_name]
        required = set(schema.get("required", ()))
        fields = _go_struct_fields(source, struct_name)

        for field_name, field in fields.items():
            assert field["omitempty"] is (field_name not in required), (
                f"Go {struct_name}.{field_name} requiredness differs from "
                f"OpenAPI {schema_name}."
            )

        for field_name, property_schema in schema.get("properties", {}).items():
            property_type = _schema_type(property_schema)
            is_optional = field_name not in required
            if (
                is_optional
                and property_type == "boolean"
                and property_schema.get("default") is not False
            ):
                assert fields[field_name]["type"] == "*bool", (
                    f"Go {struct_name}.{field_name} must be *bool so explicit false "
                    "is distinguishable from omission."
                )
            if property_type not in {"integer", "number"}:
                continue
            default_is_zero = property_schema.get("default") in (0, 0.0)
            if (
                is_optional
                and _schema_accepts_zero(property_schema)
                and not default_is_zero
            ):
                assert fields[field_name]["type"].startswith("*"), (
                    f"Go {struct_name}.{field_name} must be a pointer so an "
                    "explicit zero is distinguishable from omission."
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


def test_go_openapi_enum_constants_match() -> None:
    source = SDK_SOURCE_PATH.read_text(encoding="utf-8")
    spec = json.loads(OPENAPI_PATH.read_text(encoding="utf-8"))

    go_aggregation = set(
        re.findall(
            r'^\s*Aggregation\w+\s+AggregationStrategy\s*=\s*"([^"]+)"$',
            source,
            flags=re.MULTILINE,
        )
    )
    openapi_aggregation = set(
        spec["components"]["schemas"]["AnalyzeRequest"]["properties"][
            "aggregation_strategy"
        ]["anyOf"][0]["enum"]
    )
    assert go_aggregation == openapi_aggregation

    go_methods = set(
        re.findall(
            r'^\s*Method\w+\s+DeidentificationMethod\s*=\s*"([^"]+)"$',
            source,
            flags=re.MULTILINE,
        )
    )
    openapi_method_sets = {
        frozenset(properties["method"]["enum"])
        for schema in spec["components"]["schemas"].values()
        if isinstance(schema, dict)
        for properties in [schema.get("properties", {})]
        if isinstance(properties.get("method"), dict) and "enum" in properties["method"]
    }
    assert openapi_method_sets == {frozenset(go_methods)}


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
    assert source.count("ctx context.Context") >= len(CLIENT_METHOD_BY_OPERATION)

    assert "APIError" in readme
    assert "AsAPIError" in readme
    assert "error.code" in readme
    assert "error.message" in readme
    assert "error.details" in readme


def _openapi_operations(spec: dict[str, Any]) -> list[tuple[str, str]]:
    methods = ("get", "post", "put", "patch", "delete")
    return [
        (method, path)
        for path, path_item in spec["paths"].items()
        for method in methods
        if method in path_item
    ]


def _go_client_method_body(source: str, method_name: str) -> str:
    signature = re.search(
        rf"^func \(c \*Client\) {re.escape(method_name)}\(.*?\) .*?\{{",
        source,
        flags=re.DOTALL | re.MULTILINE,
    )
    assert signature is not None, f"Go client method {method_name} is missing."
    opening_brace = signature.end() - 1
    depth = 0
    for index in range(opening_brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening_brace + 1 : index]
    raise AssertionError(f"Go client method {method_name} has an unclosed body.")


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
    return set(_go_struct_fields(source, struct_name))


def _go_struct_fields(source: str, struct_name: str) -> dict[str, dict[str, Any]]:
    match = re.search(
        rf"^type {re.escape(struct_name)} struct \{{(?P<body>.*?)^\}}$",
        source,
        flags=re.DOTALL | re.MULTILINE,
    )
    assert match is not None, f"Go request struct {struct_name} is missing."
    fields: dict[str, dict[str, Any]] = {}
    for field_match in re.finditer(
        r'^\s*\w+\s+(?P<type>\S+)\s+`json:"(?P<name>[^",]+)(?P<opts>[^"]*)"`',
        match.group("body"),
        flags=re.MULTILINE,
    ):
        options = {option for option in field_match.group("opts").split(",") if option}
        fields[field_match.group("name")] = {
            "type": field_match.group("type"),
            "omitempty": "omitempty" in options,
        }
    return fields


def _openapi_request_schema_names(spec: dict[str, Any]) -> set[str]:
    schema_names: set[str] = set()
    pending = [
        schema_name
        for method, path in _openapi_operations(spec)
        for operation in [spec["paths"][path][method]]
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


def _schema_type(schema: dict[str, Any]) -> str | None:
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        return schema_type
    for candidate in schema.get("anyOf", ()):
        candidate_type = candidate.get("type")
        if candidate_type != "null":
            return candidate_type
    return None


def _go_type_matches_schema(go_type: str, schema: dict[str, Any]) -> bool:
    base_go_type = go_type.removeprefix("*")
    variants = [
        variant
        for variant in schema.get("anyOf", [schema])
        if variant.get("type") != "null"
    ]
    if len(variants) > 1:
        return base_go_type == "any"
    if not variants:
        return False

    active = variants[0]
    ref = active.get("$ref")
    if isinstance(ref, str):
        return base_go_type == ref.removeprefix("#/components/schemas/")

    schema_type = active.get("type")
    if schema_type == "array":
        return base_go_type.startswith("[]") and _go_type_matches_schema(
            base_go_type[2:], active["items"]
        )

    go_types_by_schema_type = {
        "boolean": {"bool"},
        "integer": {"int"},
        "number": {"float64"},
        "string": {
            "string",
            "AggregationStrategy",
            "DeidentificationMethod",
            "PIILanguage",
            "PrivacyPolicy",
        },
    }
    return base_go_type in go_types_by_schema_type.get(schema_type, set())


def _schema_accepts_zero(schema: dict[str, Any]) -> bool:
    variants = [
        variant
        for variant in schema.get("anyOf", [schema])
        if variant.get("type") in {"integer", "number"}
    ]
    if len(variants) != 1:
        return False
    active = variants[0]
    minimum = active.get("minimum")
    maximum = active.get("maximum")
    exclusive_minimum = active.get("exclusiveMinimum")
    exclusive_maximum = active.get("exclusiveMaximum")
    return (
        (minimum is None or minimum <= 0)
        and (maximum is None or maximum >= 0)
        and (exclusive_minimum is None or exclusive_minimum < 0)
        and (exclusive_maximum is None or exclusive_maximum > 0)
    )


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
