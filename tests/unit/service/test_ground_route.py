"""Focused tests for the shared REST grounding contract."""

from __future__ import annotations

import json
import logging
import socket
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from openmed.clinical.grounding import VocabLoader
from openmed.service.app import create_app
from openmed.service.logging import ACCESS_LOGGER_NAME

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "openmed/eval/golden/fixtures/grounding_vocab_synthetic.jsonl"


@pytest.fixture()
def grounding_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache_dir = tmp_path / "grounding"
    VocabLoader(cache_dir=cache_dir, local_only=True).import_snapshot(
        "icd10cm",
        FIXTURE,
        version="synthetic-fixture-1",
    )
    monkeypatch.setenv("OPENMED_GROUNDING_CACHE_DIR", str(cache_dir))
    monkeypatch.delenv("OPENMED_SERVICE_TRUSTED_HOSTS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_CORS_ORIGINS", raising=False)
    return cache_dir


def test_ground_route_matches_python_result_and_keeps_logs_phi_free(
    grounding_cache: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    del grounding_cache
    source_text = "Synthetic patient Juniper Example has type 2 diabetes in assessment."
    surface = "type 2 diabetes"
    start = source_text.index(surface)
    caplog.set_level(logging.INFO, logger=ACCESS_LOGGER_NAME)
    with TestClient(
        create_app(),
        base_url="http://127.0.0.1",
        raise_server_exceptions=False,
    ) as client:
        response = client.post(
            "/ground",
            json={
                "text": source_text,
                "systems": ["icd10cm"],
                "lang": "en",
                "offline": True,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == "openmed.grounding.v1"
    result = payload["results"][0]
    assert result["start"] == start
    assert result["end"] == start + len(surface)
    assert result["system"] == "icd10cm"
    assert result["code"] == "E11.9"
    assert result["system_uri"] == "http://hl7.org/fhir/sid/icd-10-cm"
    assert result["snapshot_provenance"]["icd10cm"]["version"] == (
        "synthetic-fixture-1"
    )
    assert "calibrated_confidence" not in result
    assert "confidence_band" not in result

    rendered = "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.name == ACCESS_LOGGER_NAME
    )
    assert source_text not in rendered
    assert "Juniper Example" not in rendered
    assert surface not in rendered
    access_record = next(
        record for record in caplog.records if record.name == ACCESS_LOGGER_NAME
    )
    assert access_record.openmed_access_log["grounding_input_count"] == 1
    assert access_record.openmed_access_log["grounding_result_count"] == 1
    assert access_record.openmed_access_log["grounding_systems"] == ("icd10cm",)
    assert access_record.openmed_access_log["grounding_lang"] == "en"


def test_ground_route_returns_typed_restricted_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENMED_GROUNDING_CACHE_DIR", raising=False)
    with TestClient(
        create_app(),
        base_url="http://127.0.0.1",
        raise_server_exceptions=False,
    ) as client:
        response = client.post(
            "/ground",
            json={"text": "synthetic finding", "systems": ["snomed"]},
        )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "restricted_terminology_unconfigured"
    assert "synthetic finding" not in json.dumps(body)


# Each restricted URI paired with the short name it must behave like.
RESTRICTED_URI_PAIRS = [
    ("snomed", "http://snomed.info/sct"),
    ("snomed", "http://snomed.info/sct/"),
    ("umls", "http://terminology.hl7.org/CodeSystem/umls"),
    ("cpt", "urn:openmed:restricted:cpt"),
    ("cpt", "http://www.ama-assn.org/go/cpt"),
]

# One synthetic alias per free system, matched by FIXTURE.
FREE_SYSTEM_CASES = [
    (
        "rxnorm",
        "http://www.nlm.nih.gov/research/umls/rxnorm",
        "renal disorder zeta001",
        "RXNORM-0001",
    ),
    ("icd10cm", "http://hl7.org/fhir/sid/icd-10-cm", "type 2 diabetes", "E11.9"),
    ("loinc", "http://loinc.org", "cardiac disorder zeta002", "LOINC-0002"),
    (
        "hpo",
        "http://human-phenotype-ontology.org",
        "pulmonary disorder zeta003",
        "HPO-0003",
    ),
    ("mesh", "https://www.nlm.nih.gov/mesh", "gastric disorder zeta004", "MESH-0004"),
]


@pytest.fixture()
def free_grounding_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache_dir = tmp_path / "free-grounding"
    for system, *_ in FREE_SYSTEM_CASES:
        VocabLoader(cache_dir=cache_dir, local_only=True).import_snapshot(
            system,
            FIXTURE,
            version="synthetic-fixture-1",
        )
    monkeypatch.setenv("OPENMED_GROUNDING_CACHE_DIR", str(cache_dir))
    monkeypatch.delenv("OPENMED_SERVICE_TRUSTED_HOSTS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_CORS_ORIGINS", raising=False)
    return cache_dir


@pytest.mark.parametrize(("short_name", "uri"), RESTRICTED_URI_PAIRS)
def test_ground_route_restricted_system_uri_matches_short_name_error(
    short_name: str,
    uri: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted: list[str] = []

    def fail_connect(*_: object, **__: object) -> None:
        attempted.append(uri)
        raise AssertionError("grounding attempted network access")

    monkeypatch.delenv("OPENMED_GROUNDING_CACHE_DIR", raising=False)
    source_text = "Synthetic finding zeta for Juniper Example."
    with TestClient(
        create_app(),
        base_url="http://127.0.0.1",
        raise_server_exceptions=False,
    ) as client:
        # Patch only the requests; TestClient startup opens a loopback pair.
        with monkeypatch.context() as request_patch:
            request_patch.setattr(socket.socket, "connect", fail_connect)
            by_name = client.post(
                "/ground",
                json={"text": source_text, "systems": [short_name]},
            )
            by_uri = client.post(
                "/ground",
                json={"text": source_text, "systems": [uri]},
            )

    assert attempted == []
    assert by_name.status_code == 400
    assert by_uri.status_code == 400
    name_error = by_name.json()["error"]
    uri_error = by_uri.json()["error"]
    assert uri_error["code"] == "restricted_terminology_unconfigured"
    for key in ("code", "message", "details"):
        assert uri_error[key] == name_error[key]
    rendered = json.dumps(by_uri.json())
    assert source_text not in rendered
    assert "Juniper Example" not in rendered


@pytest.mark.parametrize(("short_name", "uri", "surface", "code"), FREE_SYSTEM_CASES)
def test_ground_route_free_system_uri_returns_the_short_name_coding(
    short_name: str,
    uri: str,
    surface: str,
    code: str,
    free_grounding_cache: Path,
) -> None:
    del free_grounding_cache
    text = f"Synthetic note: {surface}."
    with TestClient(
        create_app(),
        base_url="http://127.0.0.1",
        raise_server_exceptions=False,
    ) as client:
        by_name = client.post(
            "/ground",
            json={"text": text, "systems": [short_name], "offline": True},
        )
        by_uri = client.post(
            "/ground",
            json={"text": text, "systems": [uri], "offline": True},
        )

    assert by_name.status_code == 200
    assert by_uri.status_code == 200
    name_results = by_name.json()["results"]
    assert name_results
    result = name_results[0]
    assert (result["system"], result["code"], result["system_uri"]) == (
        short_name,
        code,
        uri,
    )
    assert result["display"]
    assert by_uri.json()["results"] == name_results


def test_ground_route_cvx_short_name_and_uri_stay_unsupported() -> None:
    with TestClient(
        create_app(),
        base_url="http://127.0.0.1",
        raise_server_exceptions=False,
    ) as client:
        by_name = client.post(
            "/ground",
            json={"text": "synthetic vaccine", "systems": ["cvx"]},
        )
        by_uri = client.post(
            "/ground",
            json={
                "text": "synthetic vaccine",
                "systems": ["http://hl7.org/fhir/sid/cvx"],
            },
        )

    assert by_name.status_code == 400
    assert by_uri.status_code == 400
    name_error = by_name.json()["error"]
    uri_error = by_uri.json()["error"]
    assert name_error["code"] == "grounding_invalid_request"
    for key in ("code", "message", "details"):
        assert uri_error[key] == name_error[key]
    assert "synthetic vaccine" not in json.dumps(by_uri.json())


@pytest.mark.parametrize(
    "system",
    ["not-a-system", "http://example.invalid/fhir/CodeSystem/synthetic"],
)
def test_ground_route_unknown_system_uri_stays_a_validation_error(
    system: str,
) -> None:
    with TestClient(
        create_app(),
        base_url="http://127.0.0.1",
        raise_server_exceptions=False,
    ) as client:
        response = client.post(
            "/ground",
            json={"text": "synthetic finding", "systems": [system]},
        )

    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "validation_error"
    assert payload["error"]["details"][0]["field"] == "body.systems"
    assert "synthetic finding" not in json.dumps(payload)


def test_ground_route_rejects_invalid_system_via_validation_handler() -> None:
    with TestClient(
        create_app(),
        base_url="http://127.0.0.1",
        raise_server_exceptions=False,
    ) as client:
        response = client.post(
            "/ground",
            json={"text": "synthetic finding", "systems": ["not-a-system"]},
        )

    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "validation_error"
    assert payload["error"]["details"][0]["field"] == "body.systems"


def test_ground_route_rejects_oversized_request_before_json_parsing() -> None:
    with TestClient(
        create_app(max_request_body_bytes=128),
        base_url="http://127.0.0.1",
        raise_server_exceptions=False,
    ) as client:
        response = client.post(
            "/ground",
            content=json.dumps({"text": "x" * 512, "systems": ["rxnorm"]}),
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"


def test_ground_route_is_throttled(
    grounding_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del grounding_cache
    # One token per 100 s, so a slow first lookup cannot refill the bucket.
    monkeypatch.setenv("OPENMED_SERVICE_RATE_LIMIT_RPS", "0.01")
    monkeypatch.setenv("OPENMED_SERVICE_RATE_LIMIT_BURST", "1")

    with TestClient(
        create_app(),
        base_url="http://127.0.0.1",
        raise_server_exceptions=False,
    ) as client:
        first = client.post(
            "/ground",
            json={"text": "type 2 diabetes", "systems": ["icd10cm"]},
        )
        second = client.post(
            "/ground",
            json={"text": "type 2 diabetes", "systems": ["icd10cm"]},
        )

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["error"]["code"] == "rate_limited"


def test_ground_route_missing_snapshot_stays_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted_network = False

    def fail_connect(*_: object, **__: object) -> None:
        nonlocal attempted_network
        attempted_network = True
        raise AssertionError("grounding attempted network access")

    monkeypatch.setenv("OPENMED_GROUNDING_CACHE_DIR", str(tmp_path / "empty"))
    with TestClient(
        create_app(),
        base_url="http://127.0.0.1",
        raise_server_exceptions=False,
    ) as client:
        # Windows creates an asyncio loopback socketpair when TestClient starts.
        # Intercept only the request, after the test harness is initialized.
        with monkeypatch.context() as request_patch:
            request_patch.setattr(socket.socket, "connect", fail_connect)
            response = client.post(
                "/ground",
                json={
                    "text": "synthetic medication",
                    "systems": ["rxnorm"],
                    "offline": True,
                },
            )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "offline_snapshot_unavailable"
    assert attempted_network is False
