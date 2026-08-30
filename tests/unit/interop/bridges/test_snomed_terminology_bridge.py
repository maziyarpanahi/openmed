"""Offline tests for the user-supplied SNOMED terminology bridge."""

from __future__ import annotations

import json
from urllib.parse import parse_qs, urlsplit

import pytest

from openmed.interop.bridges import (
    SNOMED_SYSTEM_URI,
    SNOMEDTerminologyBridge,
    SNOMEDTerminologyConfigurationError,
)


class _Response:
    def __init__(self, payload: dict[str, object], *, status: int = 200) -> None:
        self.status = status
        self.headers = {"Content-Length": str(len(json.dumps(payload).encode()))}
        self._body = json.dumps(payload).encode()

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self, limit: int = -1) -> bytes:
        return self._body if limit < 0 else self._body[:limit]


class _TerminologyServer:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, list[str]], dict[str, str]]] = []

    def __call__(self, request: object, *, timeout: float) -> _Response:
        del timeout
        full_url = request.full_url  # type: ignore[attr-defined]
        headers = dict(request.header_items())  # type: ignore[attr-defined]
        parsed = urlsplit(full_url)
        query = parse_qs(parsed.query)
        self.requests.append((parsed.path, query, headers))
        if parsed.path.endswith("CodeSystem/$lookup"):
            return _Response(
                {
                    "resourceType": "Parameters",
                    "parameter": [
                        {"name": "name", "valueString": "SNOMED CT"},
                        {
                            "name": "display",
                            "valueString": "Synthetic finding alpha",
                        },
                        {
                            "name": "designation",
                            "part": [
                                {
                                    "name": "value",
                                    "valueString": "Synthetic finding alias",
                                }
                            ],
                        },
                    ],
                }
            )
        return _Response(
            {
                "resourceType": "Bundle",
                "type": "searchset",
                "entry": [
                    {
                        "resource": {
                            "resourceType": "CodeSystem",
                            "url": SNOMED_SYSTEM_URI,
                            "concept": [
                                {
                                    "code": "SYNTHETIC-SCTID-001",
                                    "display": "Synthetic finding alpha",
                                },
                                {
                                    "code": "SYNTHETIC-SCTID-002",
                                    "display": "Synthetic finding beta",
                                },
                            ],
                        }
                    }
                ],
            }
        )


def test_text_lookup_uses_codesystem_search_and_returns_concept_match() -> None:
    server = _TerminologyServer()
    bridge = SNOMEDTerminologyBridge(
        endpoint="http://terminology.test/fhir",
        headers={"X-Test-Credential": "synthetic"},
        opener=server,
    )

    matches = bridge.lookup("Synthetic finding alpha", limit=2)

    assert matches[0].system_uri == SNOMED_SYSTEM_URI
    assert matches[0].code == "SYNTHETIC-SCTID-001"
    assert matches[0].display == "Synthetic finding alpha"
    assert matches[0].match_type == "exact"
    assert matches[0].score == 1.0
    assert matches[0].metadata == {"source": "fhir-code-system-search"}
    assert len(server.requests) == 1
    path, query, headers = server.requests[0]
    assert path == "/fhir/CodeSystem"
    assert query == {
        "url": [SNOMED_SYSTEM_URI],
        "filter": ["Synthetic finding alpha"],
        "_count": ["2"],
    }
    assert headers["X-test-credential"] == "synthetic"
    assert "patient" not in query


def test_numeric_lookup_uses_fhir_lookup_without_caching_response() -> None:
    server = _TerminologyServer()
    bridge = SNOMEDTerminologyBridge(
        endpoint="http://terminology.test/fhir",
        opener=server,
    )

    first = bridge.lookup_code("SYNTHETIC-SCTID-001")
    second = bridge.lookup_code("SYNTHETIC-SCTID-001")

    assert first == second
    assert first[0].code == "SYNTHETIC-SCTID-001"
    assert first[0].display == "Synthetic finding alpha"
    assert first[0].match_type == "exact"
    assert len(server.requests) == 2
    for path, query, _ in server.requests:
        assert path == "/fhir/CodeSystem/$lookup"
        assert query == {
            "system": [SNOMED_SYSTEM_URI],
            "code": ["SYNTHETIC-SCTID-001"],
        }


def test_missing_endpoint_fails_before_any_network_fallback() -> None:
    with pytest.raises(
        SNOMEDTerminologyConfigurationError,
        match="user-supplied SNOMED terminology endpoint is required",
    ):
        SNOMEDTerminologyBridge()


def test_credentials_are_sent_only_as_headers_and_not_repr() -> None:
    config_bridge = SNOMEDTerminologyBridge(
        endpoint="http://terminology.test/fhir",
        bearer_token="synthetic-token",
    )

    assert "synthetic-token" not in repr(config_bridge.config)
    assert "Bearer synthetic-token" in config_bridge._request_headers()["Authorization"]
