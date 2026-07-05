"""Validation and drift-guard tests for the committed Postman collection.

The collection in ``docs/api/openmed.postman_collection.json`` is a hand-curated
Postman v2.1 export used to try the OpenMed REST service. These tests keep it
honest: they assert the file is valid JSON, declares the v2.1 schema, exposes a
``{{base_url}}`` collection variable, covers every path in the committed OpenAPI
spec (so the collection cannot silently drift from the API), and contains only
synthetic example payloads (no real PHI).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

import pytest

BASE_DIR = Path(__file__).resolve().parents[3]
COLLECTION_FILE = BASE_DIR / "docs" / "api" / "openmed.postman_collection.json"
OPENAPI_FILE = BASE_DIR / "docs" / "api" / "openapi.json"

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


def _request_paths() -> set[str]:
    """Return the normalized ``/`` path for every request in the collection."""
    collection = _load_collection()
    paths: set[str] = set()
    for request_item in _iter_requests(collection.get("item", [])):
        url = request_item["request"]["url"]
        segments = url["path"] if isinstance(url, dict) else []
        path = "/" + "/".join(segments)
        # Normalize Postman ``{{job_id}}`` placeholders to OpenAPI ``{job_id}``.
        path = re.sub(r"\{\{(\w+)\}\}", r"{\1}", path)
        paths.add(path)
    return paths


def test_collection_is_valid_json_and_v21_schema() -> None:
    collection = _load_collection()
    assert collection["info"]["schema"] == (
        "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
    )


def test_collection_declares_base_url_variable() -> None:
    collection = _load_collection()
    variables = {var["key"]: var for var in collection.get("variable", [])}
    assert "base_url" in variables
    assert variables["base_url"]["value"].startswith("http")


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


# Tokens that would indicate a real-PHI leak slipped into an example body.
# The examples deliberately use example.org / 555-01xx reserved ranges and an
# obviously synthetic placeholder name.
_PHI_LEAK_PATTERNS = (
    re.compile(r"@(?!example\.(?:org|com|net))[\w.-]+\.\w+"),  # non-example email
    re.compile(r"\bssn\b", re.IGNORECASE),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # US SSN pattern
    re.compile(r"BEGIN (?:RSA |EC )?PRIVATE KEY-----\n(?!SYNTHETIC)"),  # real key
)


def test_example_bodies_are_synthetic_only() -> None:
    collection = _load_collection()
    for request_item in _iter_requests(collection.get("item", [])):
        body = request_item["request"].get("body")
        if not body or body.get("mode") != "raw":
            continue
        raw = body["raw"]
        for pattern in _PHI_LEAK_PATTERNS:
            match = pattern.search(raw)
            assert match is None, (
                f"{request_item['name']} body looks like it contains real PHI "
                f"({match.group(0)!r}); use synthetic example data only."
            )


@pytest.mark.parametrize("endpoint", REQUIRED_ENDPOINTS)
def test_each_required_endpoint_has_a_request(endpoint: str) -> None:
    assert endpoint in _request_paths()
