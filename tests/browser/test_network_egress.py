"""Offline regression tests for the browser network-egress proof harness."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from scripts.web.network_egress_check import (
    NetworkEgressProbe,
    NetworkEgressViolation,
    assert_no_unexpected_requests,
    capture_browser_requests,
    check_network_egress,
    main,
)

MODEL_ROOT = "http://127.0.0.1:8000/models/synthetic-redactor/"


class _SyntheticPage:
    """Small Playwright-shaped event source for an offline browser session."""

    def __init__(self) -> None:
        self._listeners: dict[str, list[object]] = {}

    def on(self, event: str, listener: object) -> None:
        self._listeners.setdefault(event, []).append(listener)

    def remove_listener(self, event: str, listener: object) -> None:
        self._listeners[event].remove(listener)

    def emit(self, event: str, request: object) -> None:
        for listener in tuple(self._listeners.get(event, ())):
            listener(request)  # type: ignore[operator]


def test_allowlisted_model_assets_pass_without_opening_a_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicitly configured local model asset is the only network event."""

    def fail_socket(*args: object, **kwargs: object) -> None:
        raise AssertionError("the proof harness must not open a socket")

    monkeypatch.setattr(socket, "socket", fail_socket)

    report = check_network_egress(
        [
            {
                "method": "GET",
                "resource_type": "fetch",
                "url": f"{MODEL_ROOT}model.onnx?cache=1",
            }
        ],
        allowed_model_assets=(MODEL_ROOT,),
    )

    assert report.passed
    assert report.allowed_model_asset_count == 1
    assert report.network_request_count == 1


def test_unexpected_remote_data_call_fails_closed_and_stays_safe() -> None:
    """Unexpected calls fail while payload and URL values stay out of output."""

    synthetic_value = "synthetic-redaction-value-42"
    report = check_network_egress(
        [
            {
                "method": "POST",
                "resource_type": "fetch",
                "url": "https://data.example.invalid/redact",
                "headers": {"x-synthetic": synthetic_value},
                "post_data": synthetic_value,
            }
        ]
    )

    with pytest.raises(NetworkEgressViolation) as raised:
        report.assert_clean()

    rendered = json.dumps(report.to_dict())
    assert not report.passed
    assert report.unexpected_request_count == 1
    assert synthetic_value not in rendered
    assert synthetic_value not in str(raised.value)
    assert "https://data.example.invalid/redact" not in rendered
    assert "url_digest" in rendered


def test_allowlist_is_explicit_and_does_not_match_a_sibling_path() -> None:
    """A directory prefix permits assets below it but not an unrelated path."""

    report = check_network_egress(
        [
            {"url": f"{MODEL_ROOT}config.json", "resource_type": "fetch"},
            {
                "url": "http://127.0.0.1:8000/models/synthetic-redactor-copy/model.onnx",
                "resource_type": "fetch",
            },
        ],
        allowed_model_assets=(MODEL_ROOT,),
    )

    assert report.allowed_model_asset_count == 1
    assert report.unexpected_request_count == 1


def test_internal_browser_schemes_are_not_network_egress() -> None:
    """Data and blob URLs used inside the page do not count as remote calls."""

    report = check_network_egress(
        [
            {"url": "data:text/plain,synthetic", "resource_type": "other"},
            {
                "url": "blob:https://page.example.invalid/local-id",
                "resource_type": "other",
            },
        ]
    )

    assert report.passed
    assert report.network_request_count == 0
    assert {request.classification for request in report.requests} == {
        "browser-internal"
    }


def test_probe_can_attach_to_and_detach_from_a_browser_page() -> None:
    """The callback path records a synthetic redaction-session request."""

    page = _SyntheticPage()
    probe = NetworkEgressProbe(allowed_model_assets=MODEL_ROOT)
    probe.attach(page)
    page.emit("request", {"url": f"{MODEL_ROOT}tokenizer.json"})
    probe.detach()
    page.emit("request", {"url": "https://unexpected.example.invalid/data"})

    assert probe.assert_clean().allowed_model_asset_count == 1
    assert probe.request_count == 1


def test_capture_context_manager_is_offline_and_assertable() -> None:
    """A browser action can be scoped without retaining request payloads."""

    page = _SyntheticPage()
    with capture_browser_requests(page, allowed_model_assets=(MODEL_ROOT,)) as probe:
        page.emit("request", {"url": f"{MODEL_ROOT}vocab.json"})

    assert probe.assert_clean().passed
    assert page._listeners["request"] == []


def test_report_is_deterministic_for_the_same_request_sequence() -> None:
    """The proof artifact has no timestamps, random IDs, or input-order drift."""

    requests = [
        {"url": f"{MODEL_ROOT}model.onnx", "method": "GET"},
        {"url": "https://unexpected.example.invalid/submit", "method": "POST"},
    ]

    first = check_network_egress(requests, allowed_model_assets=(MODEL_ROOT,)).to_json()
    second = check_network_egress(
        requests, allowed_model_assets=(MODEL_ROOT,)
    ).to_json()

    assert first == second


def test_cli_reads_local_trace_and_writes_only_the_safe_report(tmp_path: Path) -> None:
    """The command-line proof path is local-only and returns a passing result."""

    trace = tmp_path / "trace.json"
    output = tmp_path / "report.json"
    trace.write_text(
        json.dumps(
            {
                "requests": [
                    {
                        "url": f"{MODEL_ROOT}model.onnx",
                        "method": "GET",
                        "post_data": "synthetic-value-not-collected",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert (
        main([str(trace), "--allow-model-asset", MODEL_ROOT, "--report", str(output)])
        == 0
    )
    rendered = output.read_text(encoding="utf-8")
    assert json.loads(rendered)["passed"] is True
    assert "synthetic-value-not-collected" not in rendered


def test_assert_helper_returns_a_passing_report() -> None:
    """The assertion helper is convenient for a focused browser test."""

    report = assert_no_unexpected_requests(
        [{"url": f"{MODEL_ROOT}config.json"}],
        allowed_model_assets=(MODEL_ROOT,),
    )

    assert report.passed
