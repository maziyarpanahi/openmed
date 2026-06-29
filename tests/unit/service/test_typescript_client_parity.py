"""Parity checks for the TypeScript REST client SDK."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
OPENAPI_PATH = ROOT / "docs/api/openapi.json"
SDK_ROOT = ROOT / "clients/typescript"
SDK_SOURCE_PATH = SDK_ROOT / "src/index.ts"
SDK_README_PATH = SDK_ROOT / "README.md"
SDK_TSCONFIG_PATH = SDK_ROOT / "tsconfig.json"
SDK_PACKAGE_PATH = SDK_ROOT / "package.json"

CLIENT_METHOD_BY_PATH = {
    "/analyze": "analyze",
    "/health": "health",
    "/livez": "livez",
    "/models/loaded": "loadedModels",
    "/models/unload": "unloadModels",
    "/pii/deidentify": "deidentify",
    "/pii/extract": "extractPii",
    "/privacy-gateway/complete": "privacyGateway",
    "/readyz": "readyz",
}


def test_typescript_client_covers_openapi_paths_and_request_fields() -> None:
    spec = json.loads(OPENAPI_PATH.read_text(encoding="utf-8"))
    source = SDK_SOURCE_PATH.read_text(encoding="utf-8")

    assert set(spec["paths"]) == set(CLIENT_METHOD_BY_PATH)

    for path, method_name in CLIENT_METHOD_BY_PATH.items():
        assert re.search(rf"^\s{{2}}async {method_name}\(", source, re.MULTILINE), (
            f"{path} is missing OpenMedClient.{method_name}()."
        )
        assert f'"{path}"' in source, f"{path} is not called by the SDK."

        operation = _first_operation(spec["paths"][path])
        request_schema = _request_schema(operation)
        if request_schema is None:
            continue

        for field_name in sorted(request_schema.get("properties", {})):
            assert re.search(rf"\b{re.escape(field_name)}\??:", source), (
                f"{field_name!r} from {path} is missing from TypeScript types."
            )


def test_typescript_client_package_is_dependency_light_and_strict() -> None:
    package = json.loads(SDK_PACKAGE_PATH.read_text(encoding="utf-8"))
    tsconfig = json.loads(SDK_TSCONFIG_PATH.read_text(encoding="utf-8"))

    assert package["dependencies"] == {}
    assert package["scripts"]["typecheck"] == "tsc -p tsconfig.json --noEmit"
    assert tsconfig["compilerOptions"]["strict"] is True
    assert "src/**/*.ts" in tsconfig["include"]


def test_typescript_client_error_mapping_is_implemented_and_documented() -> None:
    source = SDK_SOURCE_PATH.read_text(encoding="utf-8")
    readme = SDK_README_PATH.read_text(encoding="utf-8")

    assert "class OpenMedApiError extends Error" in source
    assert "code = envelope.error.code" in source
    assert "message = payload.error.message" in source
    assert "details" in source
    assert "response.ok" in source

    assert "OpenMedApiError" in readme
    assert "error.code" in readme
    assert "error.message" in readme
    assert "error.details" in readme


def _first_operation(path_item: dict[str, Any]) -> dict[str, Any]:
    for method_name in ("get", "post", "put", "patch", "delete"):
        operation = path_item.get(method_name)
        if operation is not None:
            return operation
    raise AssertionError(f"No HTTP operation found in path item: {path_item}")


def _request_schema(operation: dict[str, Any]) -> dict[str, Any] | None:
    request_body = operation.get("requestBody")
    if request_body is None:
        return None

    content = request_body["content"]["application/json"]
    schema = content["schema"]
    ref = schema.get("$ref")
    if not ref:
        return schema

    schema_name = ref.removeprefix("#/components/schemas/")
    spec = json.loads(OPENAPI_PATH.read_text(encoding="utf-8"))
    return spec["components"]["schemas"][schema_name]
