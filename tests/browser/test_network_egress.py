"""Offline regression tests for the browser network-egress proof harness."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from scripts.web import network_egress_check as egress_module
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

    asset_url = f"{MODEL_ROOT}model.onnx?cache=1"
    report = check_network_egress(
        [
            {
                "method": "GET",
                "resource_type": "fetch",
                "url": asset_url,
            }
        ],
        allowed_model_assets=(asset_url,),
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


def test_model_asset_prefix_rejects_queries_and_non_get_requests() -> None:
    """A directory prefix cannot carry query data or permit request bodies."""

    report = check_network_egress(
        [
            {
                "method": "GET",
                "url": f"{MODEL_ROOT}model.onnx?synthetic=value",
            },
            {
                "method": "POST",
                "url": f"{MODEL_ROOT}model.onnx",
            },
        ],
        allowed_model_assets=(MODEL_ROOT,),
    )

    assert report.allowed_model_asset_count == 0
    assert report.unexpected_request_count == 2


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


def test_file_urls_fail_closed() -> None:
    """A file URL may address a remote share and is not browser-internal."""

    report = check_network_egress(
        [{"url": "file://remote-share.invalid/synthetic/model.onnx"}]
    )

    assert not report.passed
    assert report.requests[0].classification == "unexpected-network"


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


def test_probe_discards_raw_urls_after_each_event() -> None:
    """A recorded request leaves only safe summary fields in probe memory."""

    synthetic_secret = "synthetic-patient-identifier-48291"
    raw_url = f"https://unexpected.example.invalid/?note={synthetic_secret}"
    probe = NetworkEgressProbe()

    probe.record({"url": raw_url, "resource_type": "fetch"})

    assert probe.report().unexpected_request_count == 1
    assert raw_url not in repr(probe.__dict__)
    assert synthetic_secret not in repr(probe.__dict__)


def test_callable_request_attributes_are_never_executed() -> None:
    """Request-shaped objects cannot run code through callable attributes."""

    class _CallableRequest:
        method = "GET"
        resource_type = "fetch"

        def __init__(self) -> None:
            self.called = False

        def url(self) -> str:
            self.called = True
            raise AssertionError("callable request values must not be invoked")

    request = _CallableRequest()
    report = check_network_egress([request])

    assert request.called is False
    assert report.unexpected_request_count == 1
    assert report.requests[0].classification == "invalid-url"


@pytest.mark.parametrize(
    "entry",
    [
        "https://models.example.invalid",
        "https://models.example.invalid/",
        "https://models.example.invalid/models/*",
        "https://*.example.invalid/models/redactor/",
    ],
)
def test_model_allowlist_rejects_host_wide_and_wildcard_entries(entry: str) -> None:
    """Only exact assets and explicitly bounded directory paths are accepted."""

    with pytest.raises(ValueError, match="model asset"):
        NetworkEgressProbe(allowed_model_assets=(entry,))


def test_request_count_is_bounded_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An untrusted or runaway trace cannot grow probe memory without bound."""

    monkeypatch.setattr(egress_module, "_MAX_REQUESTS", 2)
    probe = NetworkEgressProbe()
    probe.record("data:text/plain,one")
    probe.record("data:text/plain,two")

    with pytest.raises(ValueError, match="event count"):
        probe.record("data:text/plain,three")

    assert probe.request_count == 2


def test_oversized_url_fails_closed_without_being_retained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Oversized event values become safe invalid summaries."""

    monkeypatch.setattr(egress_module, "_MAX_URL_LENGTH", 64)
    synthetic_secret = "synthetic-secret-" * 16
    probe = NetworkEgressProbe()
    probe.record(f"https://unexpected.example.invalid/{synthetic_secret}")

    report = probe.report()
    assert report.requests[0].classification == "invalid-url"
    assert synthetic_secret not in repr(probe.__dict__)


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


def test_cli_rejects_an_oversized_trace_with_a_safe_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The local trace reader has a deterministic file-size budget."""

    monkeypatch.setattr(egress_module, "_MAX_TRACE_BYTES", 32)
    trace = tmp_path / "oversized-trace.json"
    trace.write_bytes(b"[" + (b" " * 40) + b"]")

    assert main([str(trace)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "Network egress check could not read or validate the supplied trace.\n"
    )


def test_assert_helper_returns_a_passing_report() -> None:
    """The assertion helper is convenient for a focused browser test."""

    report = assert_no_unexpected_requests(
        [{"url": f"{MODEL_ROOT}config.json"}],
        allowed_model_assets=(MODEL_ROOT,),
    )

    assert report.passed
