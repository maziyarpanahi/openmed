"""Offline round-trip coverage for the OpenMRS REST/FHIR2 adapter."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from openmed.interop import assert_redacted
from openmed.interop.openmrs import (
    OpenMRSAdapter,
    OpenMRSClient,
    OpenMRSConfig,
    de_identify_rest_resource,
    de_identify_rest_resource_with_manifest,
    manifest_paths,
)

FIXTURES = Path(__file__).parent / "fixtures" / "openmrs"
SOURCE_BASE = "https://facility.test/openmrs"
DESTINATION_BASE = "https://handoff.test/openmrs"

_REPLACEMENTS = (
    ("+254 712 345 678", "[PHONE]"),
    ("Amina Wanjiku", "[NAME]"),
    ("12345678", "[NATIONAL_ID]"),
    ("Wanjiku", "[FAMILY]"),
    ("Amina", "[GIVEN]"),
)
_SEEDED_IDENTIFIERS = {
    "[NAME]": "Amina Wanjiku",
    "[PHONE]": "+254 712 345 678",
    "[NATIONAL_ID]": "12345678",
}


@dataclass(frozen=True)
class _FakeResult:
    deidentified_text: str


def _fake_deidentify(text: str, **_: Any) -> _FakeResult:
    transformed = text
    for original, replacement in _REPLACEMENTS:
        transformed = transformed.replace(original, replacement)
    return _FakeResult(deidentified_text=transformed)


def _load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _assert_no_seeded_phi(payload: Any) -> None:
    assert_redacted(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        _SEEDED_IDENTIFIERS,
    )


class _FixtureServer:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.writes: list[tuple[str, str, dict[str, Any]]] = []
        self.rest_obs_failures = 1

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        query = request.url.params

        if request.method in {"POST", "PUT"}:
            payload = json.loads(request.content)
            self.writes.append((request.method, path, payload))
            return httpx.Response(200, json=payload)

        if path.endswith("/ws/rest/v1/obs"):
            if self.rest_obs_failures:
                self.rest_obs_failures -= 1
                return httpx.Response(503, headers={"Retry-After": "0"})
            name = (
                "rest_obs_page2.json"
                if query.get("startIndex") == "1"
                else "rest_obs_page1.json"
            )
            return httpx.Response(200, json=_load(name))
        if path.endswith("/ws/rest/v1/encounter"):
            return httpx.Response(200, json=_load("rest_encounters.json"))
        if path.endswith("/ws/rest/v1/patient"):
            return httpx.Response(200, json=_load("rest_patients.json"))

        if path.endswith("/ws/fhir2/R4/Patient"):
            return httpx.Response(200, json=_load("fhir_patient_bundle.json"))
        if path.endswith("/ws/fhir2/R4/Encounter"):
            return httpx.Response(200, json=_load("fhir_encounter_bundle.json"))
        if path.endswith("/ws/fhir2/R4/Observation"):
            name = (
                "fhir_observation_page2.json"
                if query.get("page") == "2"
                else "fhir_observation_page1.json"
            )
            return httpx.Response(200, json=_load(name))

        return httpx.Response(404)


def _make_adapter(
    server: _FixtureServer,
    *,
    session_token: str | None = None,
    sleeps: list[float] | None = None,
) -> tuple[OpenMRSClient, OpenMRSAdapter]:
    auth: dict[str, str] = (
        {"session_token": session_token}
        if session_token
        else {"username": "adapter-user", "password": "adapter-password"}
    )
    config = OpenMRSConfig(
        base_url=SOURCE_BASE,
        destination_url=DESTINATION_BASE,
        page_size=2,
        max_retries=2,
        backoff_factor=0,
        **auth,
    )
    http_client = httpx.Client(transport=httpx.MockTransport(server))
    client = OpenMRSClient(
        config,
        client=http_client,
        sleep=(sleeps.append if sleeps is not None else lambda _: None),
    )
    return client, OpenMRSAdapter(client, deidentifier=_fake_deidentify)


def test_registry_is_lazy_and_does_not_import_openmrs_or_httpx() -> None:
    script = """
import sys
import openmed.interop as interop
assert 'openmed.interop.openmrs' not in sys.modules
assert 'httpx' not in sys.modules
assert 'openmrs' in interop.available_adapters()
assert interop.adapter_spec('openmrs').extra == 'openmrs'
module = interop.get_adapter('openmrs')
assert module.__name__ == 'openmed.interop.openmrs'
assert 'httpx' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_rest_walker_changes_only_text_value_comment_and_note_fields() -> None:
    resource = _load("rest_obs_page1.json")["results"][0]
    original = copy.deepcopy(resource)

    transformed, manifest = de_identify_rest_resource_with_manifest(
        resource,
        resource_name="obs",
        deidentifier=_fake_deidentify,
    )

    assert resource == original
    assert transformed["uuid"] == original["uuid"]
    assert transformed["concept"] == original["concept"]
    assert transformed["person"] == original["person"]
    assert transformed["encounter"] == original["encounter"]
    assert transformed["obsDatetime"] == original["obsDatetime"]
    assert transformed["value"] == (
        "[NAME] called from [PHONE]; Kenya ID [NATIONAL_ID]."
    )
    assert transformed["comment"] == "Follow up with [NAME] next week."
    assert manifest_paths(manifest) == (
        "OpenMRS.obs.value",
        "OpenMRS.obs.comment",
    )
    _assert_no_seeded_phi(transformed)


def test_rest_walker_handles_encounter_note_lists_and_preserves_numeric_obs() -> None:
    encounter = _load("rest_encounters.json")["results"][0]
    transformed = de_identify_rest_resource(
        encounter,
        resource_name="encounter",
        deidentifier=_fake_deidentify,
    )
    numeric = _load("rest_obs_page2.json")["results"][0]
    numeric_result = de_identify_rest_resource(
        numeric,
        resource_name="obs",
        deidentifier=_fake_deidentify,
    )

    assert transformed["uuid"] == encounter["uuid"]
    assert transformed["patient"] == encounter["patient"]
    assert transformed["encounterType"] == encounter["encounterType"]
    assert transformed["encounterNotes"] == [
        "[NAME] confirmed phone [PHONE].",
        "Kenya ID [NATIONAL_ID] verified locally.",
    ]
    assert numeric_result["value"] == 42
    assert numeric_result["concept"] == numeric["concept"]
    _assert_no_seeded_phi(transformed)
    _assert_no_seeded_phi(numeric_result)


def test_rest_walker_handles_structured_note_text_without_touching_author() -> None:
    resource = {
        "uuid": "note-resource-uuid",
        "notes": [
            {
                "text": "Amina Wanjiku called +254 712 345 678.",
                "author": {"uuid": "author-uuid", "display": "Clinical officer"},
            }
        ],
    }

    transformed = de_identify_rest_resource(
        resource,
        resource_name="encounter",
        deidentifier=_fake_deidentify,
    )

    assert transformed["notes"][0]["text"] == "[NAME] called [PHONE]."
    assert transformed["notes"][0]["author"] == resource["notes"][0]["author"]


def test_rest_pull_paginates_retries_and_uses_basic_auth() -> None:
    server = _FixtureServer()
    sleeps: list[float] = []
    client, adapter = _make_adapter(server, sleeps=sleeps)

    records = adapter.pull_rest("obs")

    assert len(records) == 2
    assert sleeps == [0.0]
    get_requests = [request for request in server.requests if request.method == "GET"]
    assert len(get_requests) == 3
    assert all(
        request.headers["authorization"].startswith("Basic ")
        for request in get_requests
    )
    assert get_requests[-1].url.params["startIndex"] == "1"
    assert records[0].source_id == "11111111-1111-4111-8111-111111111111"
    assert records[0].transformed_paths == (
        "OpenMRS.obs.value",
        "OpenMRS.obs.comment",
    )
    assert records[1].resource["value"] == 42
    client.close()


@pytest.mark.parametrize(
    ("resource_name", "path_suffix"),
    [
        ("obs", "/ws/rest/v1/obs"),
        ("encounter", "/ws/rest/v1/encounter"),
        ("patient", "/ws/rest/v1/patient"),
    ],
)
def test_legacy_client_targets_all_required_endpoints(
    resource_name: str,
    path_suffix: str,
) -> None:
    server = _FixtureServer()
    server.rest_obs_failures = 0
    client, _ = _make_adapter(server)

    resources = client.pull_rest(resource_name)

    assert resources
    assert server.requests[0].url.path.endswith(path_suffix)
    client.close()


def test_session_token_uses_openmrs_session_cookie() -> None:
    server = _FixtureServer()
    server.rest_obs_failures = 0
    client, _ = _make_adapter(server, session_token="facility-session-token")

    client.pull_rest("patient")

    assert server.requests[0].headers["cookie"] == ("JSESSIONID=facility-session-token")
    assert "authorization" not in server.requests[0].headers
    client.close()


def test_fhir_round_trip_bundle_and_ndjson_are_valid_and_phi_free(
    tmp_path: Path,
) -> None:
    server = _FixtureServer()
    client, adapter = _make_adapter(server)

    patients = adapter.pull_fhir("Patient")
    encounters = adapter.pull_fhir("Encounter")
    observations = adapter.pull_fhir("Observation")
    records = patients + encounters + observations

    assert len(records) == 4
    assert records[0].resource["identifier"][0]["system"] == (
        "https://facility.test/openmrs/id/kenya-national-id"
    )
    assert (
        records[1].resource["type"][0]["coding"]
        == (
            _load("fhir_encounter_bundle.json")["entry"][0]["resource"]["type"][0][
                "coding"
            ]
        )
    )
    assert (
        records[2].resource["code"]["coding"]
        == (
            _load("fhir_observation_page1.json")["entry"][0]["resource"]["code"][
                "coding"
            ]
        )
    )
    assert records[2].resource["subject"]["reference"] == "Patient/patient-1"
    assert records[2].resource["encounter"]["reference"] == ("Encounter/encounter-1")
    assert "Observation.valueString" in records[2].transformed_paths
    for record in records:
        _assert_no_seeded_phi(record.resource)

    bundle = adapter.export_bundle(records, doc_id="synthetic-openmrs-roundtrip")
    repeated = adapter.export_bundle(records, doc_id="synthetic-openmrs-roundtrip")
    assert bundle == repeated
    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "transaction"
    assert all(entry["fullUrl"].startswith("urn:uuid:") for entry in bundle["entry"])
    assert all("request" in entry for entry in bundle["entry"])
    full_urls = {entry["fullUrl"] for entry in bundle["entry"]}
    references = _collect_references(bundle)
    assert references
    assert all(reference in full_urls for reference in references)
    _assert_no_seeded_phi(bundle)

    output = tmp_path / "openmrs-deidentified.ndjson"
    summary = adapter.export_ndjson(records, output)
    lines = [json.loads(line) for line in output.read_text().splitlines()]
    assert summary.resources_deidentified == 4
    assert summary.error_count == 0
    assert lines == [record.resource for record in records]
    _assert_no_seeded_phi(lines)
    client.close()


def test_write_back_preserves_resources_and_dry_run_sends_nothing() -> None:
    server = _FixtureServer()
    server.rest_obs_failures = 0
    client, adapter = _make_adapter(server)
    rest_records = adapter.pull_rest("obs") + adapter.pull_rest("encounter")
    fhir_records = adapter.pull_fhir("Patient") + adapter.pull_fhir("Observation")
    requests_before_dry_run = len(server.requests)

    dry_results = adapter.write_back(rest_records, dry_run=True)

    assert all(result.dry_run and result.status_code is None for result in dry_results)
    assert len(server.requests) == requests_before_dry_run

    rest_results = adapter.write_back(rest_records)
    fhir_results = adapter.write_back(fhir_records)

    assert all(result.status_code == 200 for result in rest_results + fhir_results)
    assert any(
        method == "POST"
        and path.endswith("/ws/rest/v1/obs/11111111-1111-4111-8111-111111111111")
        for method, path, _ in server.writes
    )
    assert any(
        method == "PUT" and path.endswith("/ws/fhir2/R4/Patient/patient-1")
        for method, path, _ in server.writes
    )
    for _, _, payload in server.writes:
        _assert_no_seeded_phi(payload)

    rest_payload = next(
        payload
        for method, path, payload in server.writes
        if method == "POST" and path.endswith("11111111-1111-4111-8111-111111111111")
    )
    original_rest = _load("rest_obs_page1.json")["results"][0]
    assert rest_payload["uuid"] == original_rest["uuid"]
    assert rest_payload["concept"] == original_rest["concept"]
    assert rest_payload["person"] == original_rest["person"]
    client.close()


def test_obs_comment_write_back_only_posts_deidentified_comment() -> None:
    server = _FixtureServer()
    server.rest_obs_failures = 0
    client, adapter = _make_adapter(server)
    record = adapter.pull_rest("obs")[0]

    results = adapter.write_back([record], obs_comment=True)

    assert results[0].method == "POST"
    assert server.writes[-1][2] == {
        "comment": "[NAME] called from [PHONE]; Kenya ID [NATIONAL_ID]."
    }
    client.close()


def test_legacy_patient_write_back_is_blocked_in_favor_of_safe_fhir_path() -> None:
    server = _FixtureServer()
    client, adapter = _make_adapter(server)
    patients = adapter.pull_rest("patient")

    with pytest.raises(ValueError, match="FHIR2 Patient"):
        adapter.write_back(patients)
    assert not server.writes
    client.close()


def test_pagination_rejects_cross_origin_next_link() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        payload = _load("fhir_patient_bundle.json")
        payload["link"] = [
            {"relation": "next", "url": "https://outside.example/Patient?page=2"}
        ]
        return httpx.Response(200, json=payload)

    config = OpenMRSConfig(base_url=SOURCE_BASE, max_retries=0)
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = OpenMRSClient(config, client=http_client)

    with pytest.raises(ValueError, match="changed origin"):
        client.pull_fhir("Patient")
    client.close()


def test_config_rejects_ambiguous_auth_and_hides_secrets() -> None:
    with pytest.raises(ValueError, match="either basic authentication"):
        OpenMRSConfig(
            base_url=SOURCE_BASE,
            username="user",
            password="password",
            session_token="token",
        )

    config = OpenMRSConfig(
        base_url=SOURCE_BASE,
        username="user",
        password="not-in-repr",
    )
    assert "not-in-repr" not in repr(config)


def _collect_references(node: Any) -> list[str]:
    references: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "reference" and isinstance(value, str):
                references.append(value)
            else:
                references.extend(_collect_references(value))
    elif isinstance(node, list):
        for value in node:
            references.extend(_collect_references(value))
    return references
