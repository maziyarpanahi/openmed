"""Synthetic fragmented ASGI transport tests for the opt-in REST adapter."""

from __future__ import annotations

import asyncio
import hashlib
import json

import pytest

from openmed.multimodal.preflight import preflight_asset
from openmed.service.app import create_app

pytestmark = pytest.mark.integration
PATH = "/experimental/multimodal/preflight"
PNG = b"\x89PNG\r\n\x1a\n" + b"synthetic-sensitive-payload"


def invoke(app, chunks, content_length=None):
    data = {
        "asset_id": "synthetic-rest-1",
        "media_type": "image/png",
        "byte_size": len(PNG),
        "sha256": hashlib.sha256(PNG).hexdigest(),
        "width": 8,
        "height": 8,
    }
    headers = [
        (b"host", b"127.0.0.1"),
        (b"content-type", b"application/octet-stream"),
        (b"x-openmed-manifest", json.dumps(data).encode()),
    ]
    if content_length is not None:
        headers.append((b"content-length", content_length))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": PATH,
        "raw_path": PATH.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": ("127.0.0.1", 4321),
        "server": ("127.0.0.1", 80),
    }
    messages = [
        {"type": "http.request", "body": chunk, "more_body": i < len(chunks) - 1}
        for i, chunk in enumerate(chunks)
    ]
    output = []
    reads = 0

    async def receive():
        nonlocal reads
        reads += 1
        if messages:
            return messages.pop(0)
        # Starlette can listen for disconnects while streaming a response.
        await asyncio.Event().wait()

    async def send(message):
        output.append(message)

    asyncio.run(app(scope, receive, send))
    status = next(
        item["status"] for item in output if item["type"] == "http.response.start"
    )
    body = b"".join(
        item.get("body", b"") for item in output if item["type"] == "http.response.body"
    )
    return status, body, reads, data


@pytest.fixture(autouse=True)
def offline_env(monkeypatch):
    for name in (
        "OPENMED_SERVICE_AUTH_ENABLED",
        "OPENMED_SERVICE_TRUSTED_HOSTS",
        "OPENMED_SERVICE_MAX_REQUEST_BYTES",
    ):
        monkeypatch.delenv(name, raising=False)


def test_fragmented_binary_body_preserves_direct_report_bytes(monkeypatch):
    monkeypatch.setenv("OPENMED_SERVICE_MAX_REQUEST_BYTES", str(len(PNG)))
    app = create_app(experimental_multimodal_preflight=True)
    status, body, _, data = invoke(app, [PNG[:3], PNG[3:10], PNG[10:]])
    assert status == 200
    assert body == preflight_asset(data, PNG).to_json().encode()


@pytest.mark.parametrize("declared", [None, b"1"])
def test_actual_chunked_size_wins_over_missing_or_false_content_length(
    monkeypatch, declared
):
    monkeypatch.setenv("OPENMED_SERVICE_MAX_REQUEST_BYTES", str(len(PNG)))
    app = create_app(experimental_multimodal_preflight=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("oversized requests must never enter preflight")

    monkeypatch.setattr("openmed.multimodal.preflight.preflight_asset", forbidden)
    status, body, reads, _ = invoke(app, [PNG, b"x", b"never-read"], declared)
    assert status == 413
    assert json.loads(body)["error"]["code"] == "request_size_exceeded"
    assert reads == 2
    assert b"synthetic-sensitive-payload" not in body


def test_declared_oversize_is_rejected_without_reading(monkeypatch):
    monkeypatch.setenv("OPENMED_SERVICE_MAX_REQUEST_BYTES", "8")
    app = create_app(experimental_multimodal_preflight=True)
    status, body, reads, _ = invoke(app, [PNG], b"9")
    assert status == 413
    assert reads == 0
    assert json.loads(body)["error"]["code"] == "request_size_exceeded"
