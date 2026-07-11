"""Validation and drift-guard tests for the committed Postman collection.

The collection in ``docs/api/openmed.postman_collection.json`` is a hand-curated
Postman v2.1 export used to try the OpenMed REST service. These tests keep it
honest: they assert the file is valid JSON, declares the v2.1 schema, exposes a
``{{base_url}}`` collection variable, covers every path in the committed OpenAPI
spec (so the collection cannot silently drift from the API), and contains only
synthetic example payloads (no real PHI).
"""

from __future__ import annotations

import inspect
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

import pytest
import yaml
from jsonschema import Draft202012Validator

from openmed.service.client import CLIENT_ENDPOINTS, OpenMedClient

BASE_DIR = Path(__file__).resolve().parents[3]
COLLECTION_FILE = BASE_DIR / "docs" / "api" / "openmed.postman_collection.json"
OPENAPI_FILE = BASE_DIR / "docs" / "api" / "openapi.json"
REST_DOCS_FILE = BASE_DIR / "docs" / "rest-service.md"
ROOT_DOCKERFILE = BASE_DIR / "Dockerfile"
HARDENED_DOCKERFILE = BASE_DIR / "deploy" / "docker" / "Dockerfile"
COMPOSE_FILE = BASE_DIR / "docker-compose.yml"
HELM_VALUES_FILE = BASE_DIR / "deploy" / "helm" / "openmed-service" / "values.yaml"
HTTP_METHODS = {"get", "post", "put", "patch", "delete"}
UNSCOPED_SECRET_REFERENCES = {"smart_private_key_pem"}

# The endpoints the issue explicitly requires the collection to exercise.
REQUIRED_ENDPOINTS = (
    "/health",
    "/models/loaded",
    "/models/unload",
    "/analyze",
    "/pii/extract",
    "/pii/deidentify",
)


def _load_collection() -> dict:
    with COLLECTION_FILE.open(encoding="utf-8") as handle:
        collection = json.load(handle)
    assert isinstance(collection, dict)
    return collection


def _load_openapi() -> dict:
    with OPENAPI_FILE.open(encoding="utf-8") as handle:
        spec = json.load(handle)
    assert isinstance(spec, dict)
    return spec


def _iter_requests(items: list) -> Iterator[dict]:
    """Yield every leaf request item, recursing through folders."""
    for item in items:
        if "request" in item:
            yield item
        for child in item.get("item", []) or []:
            if "request" in child:
                yield child
            else:
                yield from _iter_requests([child])


def _request_operation(request_item: dict) -> tuple[str, str]:
    """Return the normalized OpenAPI operation for one collection item."""
    request = request_item["request"]
    url = request["url"]
    assert isinstance(url, dict)
    segments = url.get("path")
    assert isinstance(segments, list)
    path = "/" + "/".join(segments)
    path = re.sub(r"\{\{(\w+)\}\}", r"{\1}", path)
    return request["method"].lower(), path


def _request_operations() -> set[tuple[str, str]]:
    """Return normalized ``(method, path)`` pairs for every collection request."""
    collection = _load_collection()
    return {
        _request_operation(request_item)
        for request_item in _iter_requests(collection.get("item", []))
    }


def _request_paths() -> set[str]:
    """Return the normalized ``/`` path for every request in the collection."""
    return {path for _, path in _request_operations()}


def _openapi_operations() -> set[tuple[str, str]]:
    """Return the public HTTP operations declared by the committed OpenAPI spec."""
    operations: set[tuple[str, str]] = set()
    for path, path_item in _load_openapi().get("paths", {}).items():
        operations.update(
            (method.lower(), path)
            for method in path_item
            if method.lower() in HTTP_METHODS
        )
    return operations


def _client_default_base_url() -> str:
    default = inspect.signature(OpenMedClient.__init__).parameters["base_url"].default
    assert isinstance(default, str)
    return default


def _resolve_local_schema(spec: dict, schema: dict) -> dict:
    """Resolve a local OpenAPI JSON Pointer used by a request body schema."""
    ref = schema.get("$ref")
    if ref is None:
        return schema
    assert ref.startswith("#/")
    resolved: Any = spec
    for component in ref[2:].split("/"):
        resolved = resolved[component.replace("~1", "/").replace("~0", "~")]
    assert isinstance(resolved, dict)
    return resolved


def _iter_string_values(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _iter_string_values(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _iter_string_values(nested)


def _iter_mapping_nodes(value: Any) -> Iterator[dict]:
    """Yield every mapping in an arbitrarily nested collection document."""
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _iter_mapping_nodes(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _iter_mapping_nodes(nested)


def test_collection_is_valid_json_and_v21_schema() -> None:
    collection = _load_collection()
    assert collection["info"]["schema"] == (
        "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
    )


def test_collection_description_tracks_openapi_without_version_literal() -> None:
    description = _load_collection()["info"]["description"]

    assert "docs/api/openapi.json" in description
    assert re.search(r"\bOpenAPI\s+v?\d+\.\d+\.\d+\b", description) is None
    assert _client_default_base_url() in description
    assert "--host 127.0.0.1" in description
    assert "0.0.0.0" not in description
    assert "`https://` base URL" in description
    assert "plaintext HTTP" in description
    assert "Postman history" in description


def test_collection_declares_base_url_variable() -> None:
    collection = _load_collection()
    variables = {var["key"]: var for var in collection.get("variable", [])}
    assert "base_url" in variables
    assert variables["base_url"]["value"] == _client_default_base_url()


def test_rest_docs_link_collection_and_explain_runtime_variables() -> None:
    docs = REST_DOCS_FILE.read_text(encoding="utf-8")

    assert "[Postman collection](api/openmed.postman_collection.json)" in docs
    assert _client_default_base_url() in docs
    assert "`job_id`" in docs
    assert "`{{smart_private_key_pem}}`" in docs
    assert "never" in docs and "real key" in docs
    assert "`https://` base URL" in docs
    assert "plaintext HTTP" in docs
    assert "Postman history" in docs


def test_base_url_matches_deployment_runtime_port() -> None:
    collection = _load_collection()
    variables = {var["key"]: var for var in collection.get("variable", [])}
    default_base_url = _client_default_base_url()
    service_port = urlparse(default_base_url).port
    assert service_port is not None
    assert variables["base_url"]["value"] == default_base_url
    for dockerfile in (ROOT_DOCKERFILE, HARDENED_DOCKERFILE):
        assert f"EXPOSE {service_port}" in dockerfile.read_text(encoding="utf-8")

    compose = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    assert f"{service_port}:{service_port}" in compose["services"]["app"]["ports"]

    helm_values = yaml.safe_load(HELM_VALUES_FILE.read_text(encoding="utf-8"))
    assert helm_values["service"]["port"] == service_port
    assert helm_values["service"]["targetPort"] == service_port


def test_template_variables_are_declared_or_intentionally_local_secret_only() -> None:
    collection = _load_collection()
    consumers = json.dumps(
        {
            "event": collection.get("event", []),
            "item": collection.get("item", []),
        }
    )
    declared = {variable["key"] for variable in collection.get("variable", [])}
    used = set(re.findall(r"\{\{([^{}]+)\}\}", consumers))

    assert used == declared | UNSCOPED_SECRET_REFERENCES
    assert declared.isdisjoint(UNSCOPED_SECRET_REFERENCES)
    assert "smart_private_key_pem" not in declared


def test_requests_reference_base_url_variable() -> None:
    collection = _load_collection()
    requests = list(_iter_requests(collection.get("item", [])))
    assert requests, "collection should contain at least one request"
    for request_item in requests:
        url = request_item["request"]["url"]
        raw = url["raw"] if isinstance(url, dict) else url
        assert "{{base_url}}" in raw, (
            f"{request_item['name']} must use {{{{base_url}}}}"
        )


def test_collection_covers_required_endpoints() -> None:
    paths = _request_paths()
    missing = [endpoint for endpoint in REQUIRED_ENDPOINTS if endpoint not in paths]
    assert not missing, f"collection missing required endpoints: {missing}"


def test_collection_covers_all_openapi_paths() -> None:
    """Drift guard: every OpenAPI path must have a matching request item."""
    spec_paths = set(_load_openapi().get("paths", {}))
    collection_paths = _request_paths()
    missing = sorted(spec_paths - collection_paths)
    assert not missing, (
        "Postman collection is missing requests for OpenAPI paths: "
        f"{missing}. Update docs/api/openmed.postman_collection.json."
    )


def test_collection_methods_match_openapi_operations() -> None:
    assert _request_operations() == _openapi_operations()


def test_collection_covers_every_typed_client_operation() -> None:
    client_operations = {
        (endpoint.method.lower(), endpoint.path)
        for endpoint in CLIENT_ENDPOINTS.values()
    }

    assert client_operations <= _request_operations()


def test_collection_has_exactly_one_request_per_openapi_operation() -> None:
    collection = _load_collection()
    operation_counts = Counter(
        _request_operation(request_item)
        for request_item in _iter_requests(collection.get("item", []))
    )

    assert operation_counts == Counter(
        {operation: 1 for operation in _openapi_operations()}
    )


def test_collection_requests_are_well_formed() -> None:
    collection = _load_collection()
    valid_methods = {"GET", "POST", "PUT", "PATCH", "DELETE"}
    for request_item in _iter_requests(collection.get("item", [])):
        request = request_item["request"]
        assert request["method"] in valid_methods
        body = request.get("body")
        if body and body.get("mode") == "raw":
            # Raw JSON bodies must themselves parse.
            json.loads(body["raw"])


def test_structured_urls_match_raw_urls_and_openapi_parameters() -> None:
    collection = _load_collection()
    spec = _load_openapi()

    for request_item in _iter_requests(collection.get("item", [])):
        request = request_item["request"]
        method, path = _request_operation(request_item)
        url = request["url"]
        segments = url["path"]
        assert url["host"] == ["{{base_url}}"]
        assert url["raw"] == "{{base_url}}/" + "/".join(segments)

        path_item = spec["paths"][path]
        parameters = [
            *path_item.get("parameters", []),
            *path_item[method].get("parameters", []),
        ]
        declared_path_parameters = {
            parameter["name"] for parameter in parameters if parameter["in"] == "path"
        }
        assert all(
            parameter.get("required") is True
            for parameter in parameters
            if parameter["in"] == "path"
        )
        actual_path_parameters = {
            match.group(1)
            for segment in segments
            if (match := re.fullmatch(r"\{\{(\w+)\}\}", segment))
        }
        assert actual_path_parameters == declared_path_parameters

        declared_query_parameters = {
            parameter["name"] for parameter in parameters if parameter["in"] == "query"
        }
        required_query_parameters = {
            parameter["name"]
            for parameter in parameters
            if parameter["in"] == "query" and parameter.get("required")
        }
        active_query_parameters = {
            parameter["key"]
            for parameter in url.get("query", [])
            if not parameter.get("disabled", False)
        }
        assert required_query_parameters <= active_query_parameters
        assert active_query_parameters <= declared_query_parameters


def test_request_bodies_match_openapi_schemas_and_content_types() -> None:
    collection = _load_collection()
    spec = _load_openapi()
    root_validator = Draft202012Validator(spec)

    for request_item in _iter_requests(collection.get("item", [])):
        request = request_item["request"]
        method, path = _request_operation(request_item)
        operation = spec["paths"][path][method]
        request_body = operation.get("requestBody")
        body = request.get("body")

        if request_body is None:
            assert body is None, f"{request_item['name']} must not send a body"
            continue

        assert body is not None and body.get("mode") == "raw"
        headers = {
            header["key"].lower(): header["value"]
            for header in request.get("header", [])
        }
        assert headers.get("content-type") == "application/json"
        content = request_body["content"]
        assert "application/json" in content
        payload = json.loads(body["raw"])
        schema = content["application/json"]["schema"]
        errors = sorted(
            root_validator.evolve(schema=schema).iter_errors(payload),
            key=lambda error: str(list(error.absolute_path)),
        )
        messages = "; ".join(
            f"{list(error.absolute_path)}: {error.message}" for error in errors
        )
        assert not errors, f"{request_item['name']} violates OpenAPI: {messages}"


def test_explicit_example_defaults_match_openapi_defaults() -> None:
    collection = _load_collection()
    spec = _load_openapi()

    for request_item in _iter_requests(collection.get("item", [])):
        body = request_item["request"].get("body")
        if body is None:
            continue
        method, path = _request_operation(request_item)
        schema = spec["paths"][path][method]["requestBody"]["content"]
        schema = _resolve_local_schema(spec, schema["application/json"]["schema"])
        payload = json.loads(body["raw"])

        for field_name, field_value in payload.items():
            property_schema = schema.get("properties", {}).get(field_name, {})
            if "default" in property_schema:
                assert field_value == property_schema["default"], (
                    f"{request_item['name']}.{field_name} must track the OpenAPI "
                    "default"
                )


def test_accept_headers_match_json_and_ndjson_response_handling() -> None:
    collection = _load_collection()
    for request_item in _iter_requests(collection.get("item", [])):
        headers = {
            header["key"].lower(): header["value"]
            for header in request_item["request"].get("header", [])
        }
        expected = (
            "application/x-ndjson"
            if _request_operation(request_item) == ("post", "/pii/extract/stream")
            else "application/json"
        )
        assert headers["accept"] == expected


def test_collection_has_no_scripts_or_saved_response_data() -> None:
    collection = _load_collection()

    for node in _iter_mapping_nodes(collection):
        assert not node.get("event"), (
            "portable collection must not execute or log data through scripts"
        )

    for request_item in _iter_requests(collection.get("item", [])):
        assert request_item.get("response", []) == []


# Tokens that would indicate a real-PHI leak slipped into an example body.
# The examples deliberately use example.org / 555-01xx reserved ranges and an
# obviously synthetic placeholder name.
_PHI_LEAK_PATTERNS = (
    re.compile(r"@(?!example\.(?:org|com|net))[\w.-]+\.\w+"),  # non-example email
    re.compile(r"\bssn\b", re.IGNORECASE),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # US SSN pattern
    re.compile(r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----"),
)
_PHONE_PATTERN = re.compile(r"\b\d{3}-\d{4}\b")


def test_example_bodies_are_synthetic_only() -> None:
    collection = _load_collection()
    for request_item in _iter_requests(collection.get("item", [])):
        body = request_item["request"].get("body")
        if not body or body.get("mode") != "raw":
            continue
        payload = json.loads(body["raw"])
        for string_value in _iter_string_values(payload):
            for pattern in _PHI_LEAK_PATTERNS:
                match = pattern.search(string_value)
                assert match is None, (
                    f"{request_item['name']} body looks like it contains real "
                    f"PHI or key material ({match.group(0)!r}); use synthetic "
                    "example data or an unresolved environment variable only."
                )
            for match in _PHONE_PATTERN.finditer(string_value):
                assert re.fullmatch(r"555-01\d{2}", match.group(0)), (
                    f"{request_item['name']} must use the reserved 555-01xx "
                    "fictional phone range"
                )


def test_collection_does_not_persist_credentials() -> None:
    collection = _load_collection()
    variables = {variable["key"] for variable in collection.get("variable", [])}
    assert "smart_private_key_pem" not in variables

    smart_body_seen = False
    for request_item in _iter_requests(collection.get("item", [])):
        request = request_item["request"]
        assert "auth" not in request
        headers = {header["key"].lower() for header in request.get("header", [])}
        assert headers.isdisjoint({"authorization", "x-api-key"})
        body = request.get("body")
        if body and "smart_private_key_pem" in body.get("raw", ""):
            smart_body_seen = True
            payload = json.loads(body["raw"])
            assert payload["private_key_pem"] == "{{smart_private_key_pem}}"
            assert payload["output_dir"] == "./openmed-smart-export"

    assert smart_body_seen


@pytest.mark.parametrize("endpoint", REQUIRED_ENDPOINTS)
def test_each_required_endpoint_has_a_request(endpoint: str) -> None:
    assert endpoint in _request_paths()
