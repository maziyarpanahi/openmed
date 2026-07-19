"""Signed webhook delivery for async OpenMed service jobs."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import httpx

from .signing import (
    NONCE_HEADER,
    SIGNATURE_HEADER,
    SIGNATURE_PREFIX,
    TIMESTAMP_HEADER,
    sign_request,
)

EVENT_HEADER = "X-OpenMed-Event"
DEFAULT_WEBHOOK_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class WebhookDeliveryResult:
    """Terminal webhook delivery status."""

    success: bool
    attempts: int
    status_code: Optional[int] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe delivery summary."""
        return {
            "success": self.success,
            "attempts": self.attempts,
            "status_code": self.status_code,
            "error": self.error,
        }


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize a webhook payload deterministically."""
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sign_webhook_payload(
    payload: Any,
    *,
    secret: str,
    path: str = "/",
    timestamp: Optional[int] = None,
    nonce: Optional[str] = None,
) -> tuple[bytes, str, str, str]:
    """Return body bytes plus timestamp, nonce, and signature header values."""
    body = canonical_json_bytes(payload)
    headers = sign_request(
        "POST",
        path,
        body,
        secret=secret,
        timestamp=timestamp,
        nonce=nonce,
    )
    return (
        body,
        headers[TIMESTAMP_HEADER],
        headers[NONCE_HEADER],
        headers[SIGNATURE_HEADER],
    )


def deliver_webhook(
    url: str,
    payload: dict[str, Any],
    *,
    secret: str,
    max_attempts: int = 3,
    backoff_seconds: float = 0.5,
    timeout_seconds: float = DEFAULT_WEBHOOK_TIMEOUT_SECONDS,
    client: Optional[httpx.Client] = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> WebhookDeliveryResult:
    """POST a signed webhook payload, retrying non-2xx responses and I/O errors."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if backoff_seconds < 0:
        raise ValueError("backoff_seconds must be greater than or equal to 0")

    request_path = httpx.URL(url).raw_path.decode("ascii")
    body, timestamp, nonce, signature = sign_webhook_payload(
        payload,
        secret=secret,
        path=request_path,
    )
    headers = {
        "Content-Type": "application/json",
        EVENT_HEADER: str(payload.get("event", "job.terminal")),
        TIMESTAMP_HEADER: timestamp,
        NONCE_HEADER: nonce,
        SIGNATURE_HEADER: signature,
    }

    owns_client = client is None
    active_client = client or httpx.Client(timeout=timeout_seconds)
    attempts = 0
    last_status_code: Optional[int] = None
    last_error: Optional[str] = None
    try:
        for attempt in range(1, max_attempts + 1):
            attempts = attempt
            try:
                response = active_client.post(url, content=body, headers=headers)
                last_status_code = response.status_code
                if 200 <= response.status_code < 300:
                    return WebhookDeliveryResult(
                        success=True,
                        attempts=attempts,
                        status_code=response.status_code,
                        error=None,
                    )
                last_error = f"HTTP {response.status_code}"
            except httpx.HTTPError as exc:
                last_error = str(exc)

            if attempt < max_attempts and backoff_seconds:
                sleeper(backoff_seconds * (2 ** (attempt - 1)))
    finally:
        if owns_client:
            active_client.close()

    return WebhookDeliveryResult(
        success=False,
        attempts=attempts,
        status_code=last_status_code,
        error=last_error,
    )
