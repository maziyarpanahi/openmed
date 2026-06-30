"""A small, dependency-free load test for the OpenMed ASGI app."""

import asyncio
import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from openmed.eval.metrics import compute_latency_summary


@dataclass(frozen=True)
class LoadReport:
    """Throughput, latency percentiles, and error rate for a load-test run."""

    requests_per_second: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    error_rate: float


DEFAULT_PATH = "/pii/extract"
DEFAULT_PAYLOAD = {
    "text": (
        "Jane Smith, DOB 04/12/1975, was seen at Boston Clinic by Dr. Patel. "
        "Her phone number is 617-555-0142 and her MRN is 448219."
    ),
    "lang": "en",
}


def run_load_test(
    app: Any,
    concurrency: int,
    total_requests: int,
    payload: dict[str, Any] = DEFAULT_PAYLOAD,
    path: str = DEFAULT_PATH,
) -> LoadReport:
    """Run the requests and return their speed, latency, and error rate."""
    if concurrency < 1 or total_requests < 1:
        raise ValueError("concurrency and total_requests must be at least 1")

    return asyncio.run(_run_requests(app, concurrency, total_requests, payload, path))


async def _run_requests(
    app: Any,
    concurrency: int,
    total_requests: int,
    payload: dict[str, Any],
    path: str,
) -> LoadReport:
    """Run requests in concurrent groups and summarize their results."""
    results = []
    test_started = perf_counter()

    # Run one group at a time. The final group may contain fewer requests.
    for start in range(0, total_requests, concurrency):
        group_size = min(concurrency, total_requests - start)
        group = [_timed_request(app, path, payload) for _ in range(group_size)]
        results.extend(await asyncio.gather(*group))

    total_seconds = perf_counter() - test_started
    latencies = [latency for latency, _ in results]
    errors = sum(failed for _, failed in results)
    summary = compute_latency_summary(latencies)

    return LoadReport(
        requests_per_second=total_requests / total_seconds,
        p50_ms=summary.p50_ms,
        p95_ms=summary.p95_ms,
        p99_ms=summary.p99_ms,
        error_rate=errors / total_requests,
    )


async def _timed_request(
    app: Any, path: str, payload: dict[str, Any]
) -> tuple[float, bool]:
    """Time one request and report whether it failed."""
    started = perf_counter()
    try:
        status = await _post(app, path, payload)
        failed = status >= 400
    except Exception:
        failed = True
    return (perf_counter() - started) * 1000, failed


async def _post(app: Any, path: str, payload: dict[str, Any]) -> int:
    """Make one POST request using ASGI's scope, receive, and send functions."""
    body = json.dumps(payload).encode()
    status = None
    body_sent = False

    async def receive():
        nonlocal body_sent
        if body_sent:
            return {"type": "http.disconnect"}
        body_sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        nonlocal status
        if message["type"] == "http.response.start":
            status = message["status"]

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [
            (b"host", b"127.0.0.1"),
            (b"content-type", b"application/json"),
        ],
        "client": ("127.0.0.1", 0),
        "server": ("load-test", 80),
    }
    await app(scope, receive, send)

    if status is None:
        raise RuntimeError("The app did not return an HTTP status")
    return status


__all__ = ["DEFAULT_PATH", "DEFAULT_PAYLOAD", "LoadReport", "run_load_test"]
