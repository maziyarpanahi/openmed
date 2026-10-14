"""Tests for the shared ``X-Request-ID`` grammar and its consumers."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from openmed.service.client import OpenMedAPIError, OpenMedClient
from openmed.service.privacy_gateway import coerce_request_id
from openmed.service.privacy_proxy.app import create_app as create_proxy_app
from openmed.service.request_id import (
    REQUEST_ID_HEADER,
    REQUEST_ID_MAX_LENGTH,
    generate_request_id,
    is_valid_request_id,
    request_id_or_generated,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
# Synthetic MRN-like sentinel embedded in every rejected request id.
REQUEST_ID_SENTINEL = "MRN-SYNTH-884213"
VALID_REQUEST_IDS = (
    "a",
    "req-" + "a" * 124,
    "req_ABC-123.trace:7",
    "0123456789",
    "12345678-1234-1234-1234-123456789abc",
)
INVALID_REQUEST_IDS = (
    "",
    " ",
    REQUEST_ID_SENTINEL + "-" + "x" * 8010,
    REQUEST_ID_SENTINEL + "-" + "b" * 112,
    REQUEST_ID_SENTINEL + "-\u00e9",
    REQUEST_ID_SENTINEL + "-\u00c3\u00a9",
    REQUEST_ID_SENTINEL + "-\u0661",
    REQUEST_ID_SENTINEL + "\r\nX-Injected: 1",
    REQUEST_ID_SENTINEL + "\n",
    "\n" + REQUEST_ID_SENTINEL,
    "req-ok," + REQUEST_ID_SENTINEL,
    "req-ok " + REQUEST_ID_SENTINEL,
    "req-ok\t" + REQUEST_ID_SENTINEL,
    "req-ok;" + REQUEST_ID_SENTINEL,
    "mrn=" + REQUEST_ID_SENTINEL,
    "req/" + REQUEST_ID_SENTINEL,
    "req-ok\x00" + REQUEST_ID_SENTINEL,
)


def test_grammar_boundary_is_128_characters() -> None:
    assert REQUEST_ID_MAX_LENGTH == 128
    assert is_valid_request_id("a" * 128)
    assert not is_valid_request_id("a" * 129)
    assert len(INVALID_REQUEST_IDS[3]) == 129


@pytest.mark.parametrize("value", VALID_REQUEST_IDS)
def test_conforming_request_ids_are_accepted_and_preserved(value: str) -> None:
    assert is_valid_request_id(value)
    assert request_id_or_generated(value) == value


@pytest.mark.parametrize("value", INVALID_REQUEST_IDS)
def test_non_conforming_request_ids_are_rejected_and_replaced(value: str) -> None:
    assert not is_valid_request_id(value)
    replacement = request_id_or_generated(value)
    assert str(uuid.UUID(replacement)) == replacement
    assert REQUEST_ID_SENTINEL not in replacement


@pytest.mark.parametrize("value", [None, b"req-bytes", 123, ["req-list"]])
def test_non_string_request_ids_are_rejected(value: Any) -> None:
    assert not is_valid_request_id(value)
    assert is_valid_request_id(request_id_or_generated(value))


def test_generated_request_ids_conform_and_are_fresh() -> None:
    generated = {generate_request_id() for _ in range(16)}

    assert len(generated) == 16
    assert all(is_valid_request_id(value) for value in generated)


def test_grammar_module_does_not_import_web_framework_dependencies() -> None:
    probe = (
        "import sys\n"
        "import openmed.service.request_id\n"
        "loaded = [name for name in ('fastapi', 'starlette', 'httpx', 'uvicorn')"
        " if name in sys.modules]\n"
        "print(','.join(loaded))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        check=True,
        cwd=REPO_ROOT,
        text=True,
    )

    assert result.stdout.strip() == ""


def test_service_and_proxy_share_one_request_id_grammar() -> None:
    grammar = re.compile(r"A-Za-z0-9_\.:-")
    owners = sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "openmed" / "service").rglob("*.py")
        if grammar.search(path.read_text(encoding="utf-8"))
    )

    assert owners == ["openmed/service/request_id.py"]
    assert REQUEST_ID_HEADER == "X-Request-ID"


@pytest.mark.parametrize("value", [v for v in INVALID_REQUEST_IDS if v])
def test_privacy_gateway_rejects_non_conforming_request_ids(value: str) -> None:
    with pytest.raises(ValueError) as caught:
        coerce_request_id(value)

    assert REQUEST_ID_SENTINEL not in str(caught.value)


def test_privacy_gateway_keeps_conforming_and_generates_missing_ids() -> None:
    assert coerce_request_id("req-" + "a" * 124) == "req-" + "a" * 124
    assert is_valid_request_id(coerce_request_id(None))


def _recording_client(
    seen: list[str | None],
    *,
    request_id: str | None = None,
    response_request_id: str | None = None,
    status_code: int = 200,
) -> OpenMedClient:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get(REQUEST_ID_HEADER))
        headers = {}
        if response_request_id is not None:
            headers[REQUEST_ID_HEADER] = response_request_id
        if status_code >= 400:
            return httpx.Response(
                status_code,
                headers=headers,
                json={"error": {"code": "bad_request", "message": "Rejected"}},
            )
        return httpx.Response(status_code, headers=headers, json={"models": {}})

    return OpenMedClient(
        base_url="http://testserver",
        request_id=request_id,
        transport=httpx.MockTransport(handler),
    )


@pytest.mark.parametrize("value", VALID_REQUEST_IDS)
def test_client_sends_conforming_request_ids_unchanged(value: str) -> None:
    seen: list[str | None] = []

    with _recording_client(seen, request_id=value) as client:
        client.loaded_models()
        client.loaded_models(request_id=value)

    assert seen == [value, value]


@pytest.mark.parametrize("value", [v for v in INVALID_REQUEST_IDS if v])
def test_client_rejects_non_conforming_default_request_id(value: str) -> None:
    with pytest.raises(ValueError) as caught:
        _recording_client([], request_id=value)

    assert REQUEST_ID_SENTINEL not in str(caught.value)


@pytest.mark.parametrize("value", [v for v in INVALID_REQUEST_IDS if v])
def test_client_rejects_non_conforming_per_call_request_id(value: str) -> None:
    seen: list[str | None] = []

    with _recording_client(seen) as client:
        with pytest.raises(ValueError) as caught:
            client.loaded_models(request_id=value)
        with pytest.raises(ValueError):
            list(client.extract_pii_stream("synthetic note", request_id=value))

    assert seen == []
    assert REQUEST_ID_SENTINEL not in str(caught.value)


def test_client_omits_header_when_no_request_id_is_set() -> None:
    seen: list[str | None] = []

    with _recording_client(seen) as client:
        client.loaded_models()
        client.loaded_models(request_id="")

    assert seen == [None, None]


def test_client_ignores_non_conforming_response_request_id() -> None:
    seen: list[str | None] = []

    with _recording_client(
        seen,
        request_id="req-outgoing",
        response_request_id="req-ok " + REQUEST_ID_SENTINEL,
        status_code=400,
    ) as client:
        with pytest.raises(OpenMedAPIError) as caught:
            client.loaded_models()

    assert caught.value.request_id == "req-outgoing"
    assert REQUEST_ID_SENTINEL not in str(caught.value)


def test_client_keeps_conforming_response_request_id() -> None:
    with _recording_client(
        [],
        request_id="req-outgoing",
        response_request_id="req-response",
        status_code=400,
    ) as client:
        with pytest.raises(OpenMedAPIError) as caught:
            client.loaded_models()

    assert caught.value.request_id == "req-response"


def _proxy_client() -> TestClient:
    def transport(payload: dict[str, Any], **_: Any) -> dict[str, Any]:
        del payload
        return {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "synthetic reply"},
                    "finish_reason": "stop",
                }
            ]
        }

    return TestClient(
        create_proxy_app(
            transport=transport,
            extractor=lambda *_args, **_kwargs: [],
            tripwire_extractor=lambda *_args, **_kwargs: [],
        ),
        base_url="http://127.0.0.1",
    )


@pytest.mark.parametrize(
    "value",
    [
        REQUEST_ID_SENTINEL + "-" + "x" * 8010,
        REQUEST_ID_SENTINEL + "-" + "b" * 112,
        (REQUEST_ID_SENTINEL + "-\u00e9").encode("utf-8"),
        REQUEST_ID_SENTINEL + "\n",
        "req-ok," + REQUEST_ID_SENTINEL,
        "req-ok " + REQUEST_ID_SENTINEL,
    ],
    ids=["oversized", "one_over_limit", "non_ascii", "line_feed", "comma", "space"],
)
def test_proxy_rejects_non_conforming_request_id_without_echo(
    value: str | bytes,
) -> None:
    with _proxy_client() as client:
        response = client.post(
            "/v1/chat/completions",
            headers={REQUEST_ID_HEADER: value},
            json={"messages": [{"role": "user", "content": "synthetic note"}]},
        )

    assert response.status_code == 400
    assert response.json()["error"]["details"] == {"reason": "request_id_invalid"}
    rendered = response.text + json.dumps(dict(response.headers))
    assert REQUEST_ID_SENTINEL not in rendered


def test_proxy_accepts_max_length_request_id_as_stable_digest_source() -> None:
    value = "req-" + "a" * 124
    body = {"messages": [{"role": "user", "content": "synthetic note"}]}

    with _proxy_client() as client:
        first = client.post(
            "/v1/chat/completions", headers={REQUEST_ID_HEADER: value}, json=body
        )
        second = client.post(
            "/v1/chat/completions", headers={REQUEST_ID_HEADER: value}, json=body
        )

    assert first.status_code == 200
    proxy_request_id = first.headers[REQUEST_ID_HEADER]
    assert re.fullmatch(r"[0-9a-f]{32}", proxy_request_id)
    assert second.headers[REQUEST_ID_HEADER] == proxy_request_id
