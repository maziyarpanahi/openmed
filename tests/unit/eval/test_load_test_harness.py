"""Tests for the in-process ASGI load harness."""

import asyncio
import os
from types import SimpleNamespace

import pytest

from openmed.cli import main_module as cli_main
from openmed.eval import load_test
from openmed.service.app import create_app


class FakeApp:
    """Minimal ASGI app for concurrency and failure tests."""

    def __init__(self, failing_requests: int = 0) -> None:
        """Initialize request counters and the number of simulated failures."""
        self.failing_requests = failing_requests
        self.calls = 0
        self.active = 0
        self.peak = 0

    async def __call__(self, scope, receive, send) -> None:
        """Handle one synthetic ASGI request."""
        await receive()
        self.calls += 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.001)
        self.active -= 1
        status = 500 if self.calls <= self.failing_requests else 200
        await send({"type": "http.response.start", "status": status, "headers": []})
        await send({"type": "http.response.body", "body": b""})


def test_load_report_and_concurrency() -> None:
    """Report metrics and honor the configured concurrency level."""
    app = FakeApp()

    report = load_test.run_load_test(app, concurrency=3, total_requests=8)

    assert app.calls == 8
    assert app.peak == 3
    assert report.requests_per_second > 0
    assert report.p50_ms > 0
    assert report.p95_ms > 0
    assert report.p99_ms > 0
    assert report.error_rate == 0


def test_failures_raise_error_rate() -> None:
    """Include failed HTTP responses in the reported error rate."""
    report = load_test.run_load_test(
        FakeApp(failing_requests=2), concurrency=1, total_requests=4
    )

    assert report.error_rate == 0.5


@pytest.mark.skipif(
    os.getenv("OPENMED_RUN_MODEL_TESTS") != "1",
    reason="requires a real model",
)
@pytest.mark.integration
@pytest.mark.slow
def test_real_app_handles_real_model_requests(monkeypatch) -> None:
    """Exercise the real FastAPI app and default PII model end to end."""
    monkeypatch.setenv("OPENMED_TORCH_ATTENTION_BACKEND", "eager")
    app = create_app()

    report = load_test.run_load_test(app, concurrency=2, total_requests=2)

    assert report.requests_per_second > 0
    assert report.error_rate == 0


def test_percentiles_come_from_shared_metrics(monkeypatch) -> None:
    """Populate report percentiles with the shared latency helper."""
    monkeypatch.setattr(
        load_test,
        "compute_latency_summary",
        lambda values: SimpleNamespace(p50_ms=10, p95_ms=20, p99_ms=30),
    )

    report = load_test.run_load_test(FakeApp(), concurrency=1, total_requests=1)

    assert (report.p50_ms, report.p95_ms, report.p99_ms) == (10, 20, 30)


def test_eval_load_test_cli_arguments() -> None:
    """Parse concurrency and request-count arguments for the eval command."""
    args = cli_main.build_parser().parse_args(
        ["eval", "load-test", "--concurrency", "8", "--total-requests", "50"]
    )

    assert args.concurrency == 8
    assert args.total_requests == 50
