"""Offline tests for the FHIR R4 server de-identification connector."""

from __future__ import annotations

import base64
import copy
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from openmed.interop.fhir_server import (
    FHIRServerClient,
    FHIRServerConfig,
    deidentify_bundle,
    deidentify_resource,
)

FIXTURE = Path(__file__).parent / "fixtures" / "fhir_server_bundle.json"
BASE_URL = "https://fhir.example.test/fhir"

_REPLACEMENTS = (
    ("John Doe", "[NAME]"),
    ("123 Main Street", "[ADDRESS]"),
)


@dataclass(frozen=True)
class _FakeResult:
    deidentified_text: str


def _fake_deidentify(text: str, **_: Any) -> _FakeResult:
    transformed = text
    for original, replacement in _REPLACEMENTS:
        transformed = transformed.replace(original, replacement)
    return _FakeResult(transformed)


def _load_bundle() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _resource(bundle: dict[str, Any], resource_type: str) -> dict[str, Any]:
    for entry in bundle["entry"]:
        candidate = entry["resource"]
        if candidate["resourceType"] == resource_type:
            return candidate
    raise AssertionError(f"fixture lacks {resource_type}")


def _assert_redacted(value: Any) -> None:
    encoded = json.dumps(value, ensure_ascii=False)
    assert "John Doe" not in encoded
    assert "123 Main Street" not in encoded


def test_import_and_registry_are_http_client_lazy() -> None:
    script = """
import sys
import openmed
import openmed.interop as interop
assert 'httpx' not in sys.modules
assert 'openmed.interop.fhir_server' not in sys.modules
assert 'fhir_server' in interop.available_adapters()
assert interop.adapter_spec('fhir_server').extra == 'fhir'
module = interop.get_adapter('fhir_server')
assert module.__name__ == 'openmed.interop.fhir_server'
assert 'httpx' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_resource_redacts_narrative_attachment_and_conclusion_without_mutation() -> (
    None
):
    bundle = _load_bundle()
    document = _resource(bundle, "DocumentReference")
    original = copy.deepcopy(document)

    transformed = deidentify_resource(
        document,
        "unit_test_policy",
        deidentifier=_fake_deidentify,
    )

    assert document == original
    assert transformed["id"] == "doc-1"
    assert transformed["status"] == "current"
    assert transformed["type"]["coding"][0]["code"] == "34133-9"
    assert transformed["description"] == "Visit note for [NAME] at [ADDRESS]."
    assert "John Doe" not in transformed["text"]["div"]
    assert 'title="[NAME]"' in transformed["text"]["div"]
    assert transformed["subject"]["reference"] == "Patient/pat-1"
    attachment_data = base64.b64decode(
        transformed["content"][0]["attachment"]["data"]
    ).decode("utf-8")
    assert attachment_data == "Clinical note for [NAME] at [ADDRESS]."
    assert transformed["content"][0]["attachment"]["creation"] == (
        "2024-01-15T09:00:00Z"
    )
    _assert_redacted(transformed)


def test_bundle_redacts_document_reference_and_diagnostic_report_preserving_shape() -> (
    None
):
    bundle = _load_bundle()
    original = copy.deepcopy(bundle)

    transformed = deidentify_bundle(
        bundle,
        "unit_test_policy",
        deidentifier=_fake_deidentify,
    )

    assert bundle == original
    assert transformed["resourceType"] == "Bundle"
    assert transformed["type"] == "searchset"
    assert [entry["fullUrl"] for entry in transformed["entry"]] == [
        entry["fullUrl"] for entry in bundle["entry"]
    ]
    report = _resource(transformed, "DiagnosticReport")
    assert report["conclusion"] == (
        "[NAME]'s diagnostic report was reviewed at [ADDRESS]."
    )
    assert report["code"]["coding"][0]["display"] == "Lung cancer narrative"
    assert report["effectiveDateTime"] == "2024-01-15T10:00:00Z"
    assert "John Doe" not in transformed["text"]["div"]
    _assert_redacted(transformed)


class _FixtureServer:
    def __init__(self, bundle: dict[str, Any]) -> None:
        self.bundle = bundle
        self.requests: list[httpx.Request] = []
        self.writes: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method == "PUT":
            payload = json.loads(request.content)
            self.writes.append(payload)
            return httpx.Response(200, json=payload)
        if request.method != "GET":
            return httpx.Response(405)
        if request.url.path.endswith("/DocumentReference/doc-1"):
            return httpx.Response(200, json=_resource(self.bundle, "DocumentReference"))
        if request.url.params.get("page") == "2":
            return httpx.Response(
                200,
                json={
                    "resourceType": "Bundle",
                    "type": "searchset",
                    "entry": [],
                },
            )
        page = copy.deepcopy(self.bundle)
        page["link"].append(
            {
                "relation": "next",
                "url": f"{BASE_URL}/DocumentReference?page=2",
            }
        )
        return httpx.Response(200, json=page)


def _make_client(server: _FixtureServer) -> FHIRServerClient:
    return FHIRServerClient(
        FHIRServerConfig(BASE_URL),
        client=httpx.Client(transport=httpx.MockTransport(server)),
    )


def test_client_fetches_resource_and_only_puts_with_explicit_write_flag() -> None:
    server = _FixtureServer(_load_bundle())
    client = _make_client(server)

    redacted = client.fetch_and_deidentify(
        "DocumentReference",
        "doc-1",
        policy="unit_test_policy",
        deidentifier=_fake_deidentify,
    )

    assert redacted["description"] == "Visit note for [NAME] at [ADDRESS]."
    assert server.writes == []

    client.fetch_and_deidentify(
        "DocumentReference",
        "doc-1",
        policy="unit_test_policy",
        deidentifier=_fake_deidentify,
        write=True,
    )

    assert len(server.writes) == 1
    assert server.writes[0]["id"] == "doc-1"
    _assert_redacted(server.writes[0])
    assert [request.method for request in server.requests] == ["GET", "GET", "PUT"]
    client.close()


def test_client_follows_next_links_and_writes_bundle_entries_when_enabled() -> None:
    server = _FixtureServer(_load_bundle())
    client = _make_client(server)

    transformed = client.fetch_and_deidentify(
        "DocumentReference",
        policy="unit_test_policy",
        deidentifier=_fake_deidentify,
        write=True,
    )

    assert transformed["resourceType"] == "Bundle"
    assert len(transformed["entry"]) == 2
    assert all(
        str(link.get("relation")) != "next" for link in transformed.get("link", [])
    )
    assert {payload["id"] for payload in server.writes} == {"doc-1", "report-1"}
    assert len([request for request in server.requests if request.method == "GET"]) == 2
    assert len([request for request in server.requests if request.method == "PUT"]) == 2
    client.close()
