"""Tests for the offline desktop sidecar JSON-lines protocol."""

from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

from openmed.service.sidecar import SidecarRuntime, SidecarServer

ROOT = Path(__file__).resolve().parents[3]
GOLDEN_PATH = Path(__file__).parent / "fixtures" / "sidecar_stdio_golden.json"
HASH_SECRET = b"sidecar-golden-secret"
PHI_FRAGMENTS = ("425-555-0100", "rowan@example.test")


class _FakeModelPipeline:
    def __call__(self, text: str, **_: Any) -> list[dict[str, Any]]:
        entities = []
        for value, label, score in (
            ("425-555-0100", "PHONE", 0.85),
            ("rowan@example.test", "EMAIL", 1.0),
        ):
            if value in text:
                start = text.index(value)
                entities.append(
                    {
                        "entity_group": label,
                        "score": score,
                        "word": value,
                        "start": start,
                        "end": start + len(value),
                    }
                )
        return entities


class _FakeLoader:
    def __init__(self, config: Any, *, network_probe: bool = False) -> None:
        self.config = config
        self.network_probe = network_probe
        self.create_calls = 0
        self.unload_calls = 0
        self.network_attempted = False
        self.pipeline = _FakeModelPipeline()

    def create_pipeline(self, *_: Any, **__: Any) -> Any:
        self.create_calls += 1
        if self.network_probe:
            self.network_attempted = True
            socket.create_connection(("example.invalid", 443), timeout=0.1)
        return self.pipeline

    def get_max_sequence_length(self, *_: Any, **__: Any) -> None:
        return None

    def unload_all_models(self) -> dict[str, int]:
        self.unload_calls += 1
        return {"models": 0, "tokenizers": 0, "pipelines": 1}


class _CleanHostError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("The OpenMed sidecar terminated before responding.")
        self.code = "SIDECAR_TERMINATED"


def _serve(
    requests: list[dict[str, Any]],
    runtime: SidecarRuntime,
) -> tuple[list[dict[str, Any]], str]:
    stdin = io.StringIO("".join(f"{json.dumps(request)}\n" for request in requests))
    stdout = io.StringIO()
    stderr = io.StringIO()
    server = SidecarServer(runtime, stdin=stdin, stdout=stdout, stderr=stderr)

    assert server.serve() == 0

    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
    return responses, stderr.getvalue()


def _read_host_response(stdout: str) -> dict[str, Any]:
    if not stdout.strip():
        raise _CleanHostError
    return json.loads(stdout.splitlines()[0])


def test_stdio_deidentification_matches_committed_golden_within_tolerance() -> None:
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    runtime = SidecarRuntime(hmac_secret=HASH_SECRET)

    responses, stderr = _serve([golden["request"]], runtime)

    assert len(responses) == 1
    response = responses[0]
    assert response["id"] == golden["request"]["id"]
    assert response["ok"] is True
    actual = response["result"]
    expected = golden["expected"]
    assert actual["deidentified_text"] == expected["deidentified_text"]
    assert len(actual["spans"]) == len(expected["spans"])
    tolerance = expected["score_tolerance"]
    for actual_span, expected_span in zip(actual["spans"], expected["spans"]):
        for key, value in expected_span.items():
            if key == "score":
                assert actual_span[key] == pytest.approx(value, abs=tolerance)
            else:
                assert actual_span[key] == value

    rendered_response = json.dumps(response)
    rendered_logs = stderr
    assert not any(fragment in rendered_response for fragment in PHI_FRAGMENTS)
    assert not any(fragment in rendered_logs for fragment in PHI_FRAGMENTS)
    log_records = [json.loads(line) for line in rendered_logs.splitlines()]
    assert [record["event"] for record in log_records] == [
        "sidecar_started",
        "request_completed",
        "sidecar_stopped",
    ]
    request_log = log_records[1]
    assert request_log["input_length"] == len(golden["request"]["text"])
    assert request_log["span_count"] == 2
    assert request_log["request_id_hash"].startswith("hmac-sha256:")
    assert golden["request"]["id"] not in rendered_logs


def test_model_loader_is_local_only_lazy_reused_and_unloaded() -> None:
    loaders: list[_FakeLoader] = []

    def loader_factory(config: Any) -> _FakeLoader:
        loader = _FakeLoader(config)
        loaders.append(loader)
        return loader

    runtime = SidecarRuntime(loader_factory=loader_factory, hmac_secret=HASH_SECRET)
    assert loaders == []
    request = {
        "id": "model-1",
        "operation": "deidentify",
        "text": "Callback 425-555-0100.",
        "options": {
            "model_name": "local-pii",
            "use_smart_merging": False,
        },
    }

    responses, _ = _serve(
        [request, {**request, "id": "model-2"}],
        runtime,
    )

    assert [response["ok"] for response in responses] == [True, True]
    assert len(loaders) == 1
    loader = loaders[0]
    assert loader.config.local_only is True
    assert loader.create_calls == 2
    assert loader.unload_calls == 1
    assert len(runtime._pipelines) == 0


def test_network_egress_is_blocked_and_failure_log_contains_no_phi() -> None:
    loader: _FakeLoader | None = None

    def loader_factory(config: Any) -> _FakeLoader:
        nonlocal loader
        loader = _FakeLoader(config, network_probe=True)
        return loader

    runtime = SidecarRuntime(loader_factory=loader_factory, hmac_secret=HASH_SECRET)
    request = {
        "id": "network-guard",
        "operation": "deidentify",
        "text": "Callback 425-555-0100.",
        "options": {"model_name": "local-pii"},
    }

    responses, stderr = _serve([request], runtime)

    assert loader is not None and loader.network_attempted is True
    assert responses == [
        {
            "id": "network-guard",
            "ok": False,
            "error": {
                "code": "PROCESSING_FAILED",
                "message": ("De-identification failed; verify the local model bundle."),
            },
        }
    ]
    assert "425-555-0100" not in stderr
    assert "example.invalid" not in stderr
    for line in stderr.splitlines():
        json.loads(line)


def test_shutdown_request_is_acknowledged_and_stops_before_later_input() -> None:
    runtime = SidecarRuntime(hmac_secret=HASH_SECRET)

    responses, stderr = _serve(
        [
            {"id": "ping-1", "operation": "ping"},
            {"id": "stop-1", "operation": "shutdown"},
            {"id": "ping-never", "operation": "ping"},
        ],
        runtime,
    )

    assert [response["id"] for response in responses] == ["ping-1", "stop-1"]
    assert responses[0]["result"] == {"offline": True, "protocol_version": 1}
    assert responses[1]["result"] == {"shutdown": True}
    assert "ping-never" not in stderr


def test_untrusted_operation_value_is_not_copied_to_structured_logs() -> None:
    runtime = SidecarRuntime(hmac_secret=HASH_SECRET)
    untrusted_operation = "rowan@example.test"

    responses, stderr = _serve(
        [{"id": "invalid-operation", "operation": untrusted_operation}],
        runtime,
    )

    assert responses[0]["error"]["code"] == "INVALID_REQUEST"
    assert untrusted_operation not in stderr
    failed_record = json.loads(stderr.splitlines()[1])
    assert failed_record["operation"] == "unknown"


def test_frozen_binary_entrypoint_emits_only_structured_stderr() -> None:
    requests = "\n".join(
        (
            json.dumps({"id": "ping-entry", "operation": "ping"}),
            json.dumps({"id": "stop-entry", "operation": "shutdown"}),
            "",
        )
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT)

    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/openmed_sidecar_entry.py")],
        input=requests,
        capture_output=True,
        text=True,
        env=environment,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0
    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    assert [response["id"] for response in responses] == [
        "ping-entry",
        "stop-entry",
    ]
    log_records = [json.loads(line) for line in completed.stderr.splitlines()]
    assert log_records[0]["event"] == "sidecar_started"
    assert log_records[-1]["event"] == "sidecar_stopped"


def test_killing_sidecar_mid_request_returns_clean_host_error() -> None:
    child_code = textwrap.dedent(
        """
        import logging
        import time

        logging.disable(logging.CRITICAL)
        from openmed.service.sidecar import SidecarServer

        class BlockingRuntime:
            def deidentify(self, text, options):
                time.sleep(60)
                return {"deidentified_text": text, "spans": []}

            def close(self):
                return None

        raise SystemExit(SidecarServer(BlockingRuntime()).serve())
        """
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT)
    process = subprocess.Popen(
        [sys.executable, "-c", child_code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=environment,
    )
    assert process.stdin is not None
    assert process.stderr is not None
    try:
        started = json.loads(process.stderr.readline())
        assert started["event"] == "sidecar_started"
        process.stdin.write(
            json.dumps(
                {
                    "id": "kill-1",
                    "operation": "deidentify",
                    "text": "Synthetic patient note",
                }
            )
            + "\n"
        )
        process.stdin.flush()
        time.sleep(0.15)
        process.kill()
        stdout, _ = process.communicate(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    assert process.returncode != 0
    with pytest.raises(_CleanHostError) as exc_info:
        _read_host_response(stdout)
    assert exc_info.value.code == "SIDECAR_TERMINATED"
    assert str(exc_info.value) == ("The OpenMed sidecar terminated before responding.")
