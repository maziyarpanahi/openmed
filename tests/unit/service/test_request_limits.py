"""Global request-body limits run before route parsing."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from openmed.guard.operational_limits import OperationalLimits
from openmed.service.request_limits import BoundedRequestBodyMiddleware


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
