"""Global request-body limits run before route parsing."""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import tracemalloc
from dataclasses import replace
from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import BaseModel, field_validator
from starlette.types import Message, Scope

from openmed.guard.operational_limits import OperationalLimits
from openmed.service.request_limits import (
    JSON_STRUCTURE_ROUTE_OVERRIDES,
    BoundedRequestBodyMiddleware,
)


def _client(*, maximum: int) -> TestClient:
    app = FastAPI()

    @app.post("/parse")
    async def parse(request: Request) -> dict[str, int]:
        payload = await request.body()
        return {"byte_count": len(payload)}

    app.add_middleware(
        BoundedRequestBodyMiddleware,
        limits=OperationalLimits(max_request_bytes=maximum),
    )
    return TestClient(app, raise_server_exceptions=False)


def test_declared_oversize_is_rejected_before_receive_or_route() -> None:
    with _client(maximum=8) as client:
        response = client.post(
            "/parse",
            headers={"content-length": "9"},
            content=b"ignored",
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_size_exceeded"


def test_observed_body_limit_and_valid_body_are_deterministic() -> None:
    with _client(maximum=8) as client:
        rejected = client.post("/parse", content=b"123456789")
        accepted = client.post("/parse", content=b"12345678")

    assert rejected.status_code == 413
    assert rejected.json()["error"]["code"] == "request_size_exceeded"
    assert accepted.status_code == 200
    assert accepted.json() == {"byte_count": 8}


def test_invalid_content_length_is_content_free() -> None:
    with _client(maximum=8) as client:
        response = client.post(
            "/parse",
            headers={"content-length": "not-a-count"},
            content=b"synthetic-sensitive-canary",
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "request_size_invalid"
    assert "synthetic-sensitive" not in response.text


def test_fragmented_body_replays_in_original_order() -> None:
    chunks: list[Message] = [
        {"type": "http.request", "body": b"abc", "more_body": True},
        {"type": "http.request", "body": b"def", "more_body": False},
    ]
    observed: list[Message] = []

    async def app(scope, receive, send) -> None:
        observed.append(await receive())
        observed.append(await receive())

    async def receive() -> Message:
        return chunks.pop(0)

    async def send(message: Message) -> None:
        raise AssertionError(f"unexpected ASGI response: {message['type']}")

    scope: Scope = {"type": "http", "method": "POST", "headers": []}
    middleware = BoundedRequestBodyMiddleware(
        app, limits=OperationalLimits(max_request_bytes=6)
    )
    asyncio.run(middleware(scope, receive, send))

    assert [item["body"] for item in observed] == [b"abc", b"def"]


_SENTINEL = "synthetic-structure-sentinel-7319"


def _json_client(
    limits: OperationalLimits,
    *,
    overrides: dict[str, dict[str, int]] | None = None,
) -> tuple[TestClient, list[bytes]]:
    app = FastAPI()
    seen: list[bytes] = []

    async def echo(request: Request) -> dict[str, int]:
        payload = await request.body()
        seen.append(payload)
        return {"byte_count": len(payload)}

    app.add_api_route("/parse", echo, methods=["POST"])
    app.add_api_route("/wide", echo, methods=["POST"])
    kwargs: dict[str, object] = {"limits": limits}
    if overrides is not None:
        kwargs["json_route_overrides"] = overrides
    app.add_middleware(BoundedRequestBodyMiddleware, **kwargs)
    return TestClient(app, raise_server_exceptions=False), seen


def _assert_structure_rejection(
    response, *, category: str, ceiling: int, seen: list[bytes]
) -> None:
    assert response.status_code == 413
    assert response.json() == {
        "error": {
            "code": "json_structure_exceeded",
            "details": {"category": category, "ceiling": ceiling},
            "message": "Request rejected by an operational limit",
        }
    }
    assert seen == []


@pytest.mark.parametrize(
    ("limits", "body", "category", "ceiling"),
    (
        (OperationalLimits(max_json_depth=3), b"[[[[]]]]", "depth", 3),
        (OperationalLimits(max_json_depth=3), b'{"a":{"b":{"c":{}}}}', "depth", 3),
        (OperationalLimits(max_json_nodes=3), b"[1,2,3]", "nodes", 3),
        (OperationalLimits(max_json_nodes=3), b'{"a":1,"b":2,"c":3}', "nodes", 3),
        (
            OperationalLimits(max_json_string_bytes=8),
            b'{"v":"012345678"}',
            "string_bytes",
            8,
        ),
        (
            OperationalLimits(max_json_string_bytes=8),
            b'{"012345678":1}',
            "string_bytes",
            8,
        ),
    ),
)
def test_json_structure_violations_return_stable_category_and_ceiling(
    limits: OperationalLimits, body: bytes, category: str, ceiling: int
) -> None:
    client, seen = _json_client(limits)
    with client:
        response = client.post(
            "/parse", content=body, headers={"content-type": "application/json"}
        )

    _assert_structure_rejection(response, category=category, ceiling=ceiling, seen=seen)


def test_json_bodies_at_each_ceiling_are_forwarded_unchanged() -> None:
    limits = OperationalLimits(
        max_json_depth=3, max_json_nodes=4, max_json_string_bytes=8
    )
    body = b'{"k": [["01234567"]]}'
    client, seen = _json_client(limits)
    with client:
        response = client.post(
            "/parse", content=body, headers={"content-type": "application/json"}
        )

    assert response.status_code == 200
    assert response.json() == {"byte_count": len(body)}
    assert seen == [body]


@pytest.mark.parametrize(
    ("raw", "decoded_bytes"),
    (
        ("\\u00e9\\u00e9\\u00e9\\u00e9", 8),
        ("éééé", 8),
        ("\\ud83d\\ude00\\ud83d\\ude00", 8),
        ('\\n\\t\\"\\\\\\/\\b\\f\\r', 8),
        ("\\u20ac\\u0041\\u0042\\u0043\\u0044\\u0045", 8),
    ),
)
def test_escaped_and_unicode_strings_are_measured_as_decoded_utf8(
    raw: str, decoded_bytes: int
) -> None:
    body = ('{"v":"' + raw + '"}').encode("utf-8")
    assert len(json.loads(body)["v"].encode("utf-8")) == decoded_bytes
    accepted, accepted_seen = _json_client(
        OperationalLimits(max_json_string_bytes=decoded_bytes)
    )
    rejected, rejected_seen = _json_client(
        OperationalLimits(max_json_string_bytes=decoded_bytes - 1)
    )
    with accepted, rejected:
        ok = accepted.post(
            "/parse", content=body, headers={"content-type": "application/json"}
        )
        denied = rejected.post(
            "/parse", content=body, headers={"content-type": "application/json"}
        )

    assert ok.status_code == 200
    assert accepted_seen == [body]
    _assert_structure_rejection(
        denied,
        category="string_bytes",
        ceiling=decoded_bytes - 1,
        seen=rejected_seen,
    )


@pytest.mark.parametrize(
    "content_type",
    (None, "application/json; charset=utf-8", "application/fhir+json"),
)
def test_json_family_and_untyped_bodies_are_scanned(content_type: str | None) -> None:
    client, seen = _json_client(OperationalLimits(max_json_depth=2))
    headers = {} if content_type is None else {"content-type": content_type}
    with client:
        response = client.post("/parse", content=b"[[[1]]]", headers=headers)

    _assert_structure_rejection(response, category="depth", ceiling=2, seen=seen)


@pytest.mark.parametrize(
    "content_type",
    ("text/plain", "application/octet-stream", "multipart/form-data; boundary=x"),
)
def test_non_json_content_types_are_not_scanned(content_type: str) -> None:
    body = b"[[[[[[" + b",".join(b"1" for _ in range(20)) + b"]]]]]]"
    client, seen = _json_client(OperationalLimits(max_json_depth=1, max_json_nodes=1))
    with client:
        response = client.post(
            "/parse", content=body, headers={"content-type": content_type}
        )

    assert response.status_code == 200
    assert seen == [body]


def test_utf16_json_cannot_bypass_the_structure_scan() -> None:
    body = "[[[[1]]]]".encode("utf-16")
    client, seen = _json_client(OperationalLimits(max_json_depth=3))
    with client:
        response = client.post(
            "/parse", content=body, headers={"content-type": "application/json"}
        )

    _assert_structure_rejection(response, category="depth", ceiling=3, seen=seen)


def test_structure_rejection_never_echoes_request_values(
    caplog: pytest.LogCaptureFixture,
) -> None:
    body = json.dumps(
        {"note": _SENTINEL * 4, "nested": [[[[_SENTINEL]]]], "list": [_SENTINEL] * 9}
    ).encode()
    cases = (
        OperationalLimits(max_json_string_bytes=len(_SENTINEL)),
        OperationalLimits(max_json_depth=2),
        OperationalLimits(max_json_nodes=4),
    )
    caplog.set_level(logging.DEBUG)
    for limits in cases:
        client, seen = _json_client(limits)
        with client:
            response = client.post(
                "/parse", content=body, headers={"content-type": "application/json"}
            )
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "json_structure_exceeded"
        assert _SENTINEL not in response.text
        assert seen == []

    assert _SENTINEL not in caplog.text
    for record in caplog.records:
        assert _SENTINEL not in repr(record.__dict__)


def test_per_route_ceiling_overrides_apply_only_to_the_named_path() -> None:
    body = b"[1,2,3,4,5,6,7,8]"
    client, seen = _json_client(
        OperationalLimits(max_json_nodes=3),
        overrides={"/wide": {"max_json_nodes": 9}},
    )
    with client:
        wide = client.post(
            "/wide", content=body, headers={"content-type": "application/json"}
        )
        default = client.post(
            "/parse", content=body, headers={"content-type": "application/json"}
        )

    assert wide.status_code == 200
    assert seen == [body]
    seen.clear()
    _assert_structure_rejection(default, category="nodes", ceiling=3, seen=seen)


@pytest.mark.parametrize(
    "overrides",
    (
        {"/wide": {"max_request_bytes": 10}},
        {"/wide": {"max_json_nodes": 0}},
        {"/wide": {"max_json_depth": 10_000}},
    ),
)
def test_per_route_overrides_are_restricted_to_safe_json_ceilings(
    overrides: dict[str, dict[str, int]],
) -> None:
    with pytest.raises(ValueError):
        BoundedRequestBodyMiddleware(
            FastAPI(), limits=OperationalLimits(), json_route_overrides=overrides
        )


def test_default_route_ceilings_are_documented_safe_limits() -> None:
    assert set(JSON_STRUCTURE_ROUTE_OVERRIDES) == {"/cohort/resolve"}
    for fields in JSON_STRUCTURE_ROUTE_OVERRIDES.values():
        replace(OperationalLimits(), **fields)
        assert set(fields) <= {
            "max_json_depth",
            "max_json_nodes",
            "max_json_string_bytes",
        }


_VALIDATED_EDGES: list[int] = []


class _Edge(BaseModel):
    ancestor_concept_id: int
    descendant_concept_id: int

    @field_validator("ancestor_concept_id")
    @classmethod
    def _spy(cls, value: int) -> int:
        _VALIDATED_EDGES.append(value)
        return value


class _CohortShaped(BaseModel):
    phenotype: dict[str, Any]
    records_jsonl: str
    concept_ancestors: list[_Edge]


def _edge_body(edge_count: int) -> bytes:
    edges = b",".join(
        b'{"ancestor_concept_id":%d,"descendant_concept_id":%d}' % (index + 1, 7)
        for index in range(edge_count)
    )
    return b'{"phenotype":{},"records_jsonl":"","concept_ancestors":[' + edges + b"]}"


def test_node_heavy_body_is_rejected_before_validation_with_bounded_memory() -> None:
    _VALIDATED_EDGES.clear()
    handled: list[int] = []
    app = FastAPI()

    @app.post("/cohort/resolve")
    async def resolve(payload: _CohortShaped) -> dict[str, int]:
        handled.append(len(payload.concept_ancestors))
        return {"edges": len(payload.concept_ancestors)}

    app.add_middleware(
        BoundedRequestBodyMiddleware,
        limits=OperationalLimits(),
        json_route_overrides={},
    )
    body = _edge_body(40_000)
    assert len(body) > 2_000_000
    sent: list[Message] = []

    async def receive() -> Message:
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: Message) -> None:
        sent.append(message)

    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/cohort/resolve",
        "raw_path": b"/cohort/resolve",
        "root_path": "",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", 80),
    }

    tracemalloc.start()
    try:
        asyncio.run(app(scope, receive, send))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert sent[0]["status"] == 413
    payload = json.loads(b"".join(item.get("body", b"") for item in sent[1:]))
    assert payload["error"]["code"] == "json_structure_exceeded"
    assert payload["error"]["details"] == {"category": "nodes", "ceiling": 100_000}
    assert _VALIDATED_EDGES == []
    assert handled == []
    assert peak < len(body) // 4


def test_service_cohort_route_rejects_reproduction_before_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service_app = importlib.import_module("openmed.service.app")

    def _never(*_args: object, **_kwargs: object) -> dict[str, Any]:
        raise AssertionError("cohort handler must not run")

    monkeypatch.setattr(service_app, "_cohort_resolve_summary", _never)
    body = _edge_body(90_000)
    with TestClient(
        service_app.create_app(),
        base_url="http://127.0.0.1",
        raise_server_exceptions=False,
    ) as client:
        response = client.post(
            "/cohort/resolve",
            content=body,
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 413
    assert response.json()["error"] == {
        "code": "json_structure_exceeded",
        "details": {"category": "nodes", "ceiling": 250_000},
        "message": "Request rejected by an operational limit",
    }
