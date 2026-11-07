"""Connection bounds applied by the reference launcher and its images."""

from __future__ import annotations

import http.client
import logging
import re
import shutil
import socket
import subprocess
import threading
import time
from collections.abc import Callable
from importlib import import_module
from pathlib import Path

import pytest

from openmed.service import logging as service_logging
from openmed.service.metrics import (
    CONNECTION_REJECTED_NAME,
    REQUEST_BODY_TIMEOUT_NAME,
    REQUEST_BODY_TIMEOUT_REASONS,
    PrometheusMetricsRegistry,
)
from openmed.service.request_limits import (
    BODY_TIMEOUT_REASONS,
    LIMIT_CONCURRENCY_ENV_VAR,
    REQUEST_BODY_GRACE_ENV_VAR,
    REQUEST_BODY_MIN_RATE_ENV_VAR,
    REQUEST_BODY_TIMEOUT_ENV_VAR,
    TIMEOUT_GRACEFUL_SHUTDOWN_ENV_VAR,
    TIMEOUT_KEEP_ALIVE_ENV_VAR,
    ConnectionLimitLogFilter,
    RequestBodyReadLimits,
    ServerConnectionLimits,
    connection_rejection_count,
    request_body_read_limits_from_env,
    server_connection_limits_from_env,
)

ROOT = Path(__file__).resolve().parents[3]
DOCKERFILES = (
    ROOT / "Dockerfile",
    ROOT / "deploy" / "docker" / "Dockerfile",
    ROOT / "deploy" / "docker" / "Dockerfile.distroless",
)
SERVER_ENV_VARS = (
    LIMIT_CONCURRENCY_ENV_VAR,
    TIMEOUT_KEEP_ALIVE_ENV_VAR,
    TIMEOUT_GRACEFUL_SHUTDOWN_ENV_VAR,
)


def _clear_server_env(monkeypatch) -> None:
    for name in SERVER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _capture_uvicorn_run(monkeypatch) -> list[dict]:
    uvicorn = pytest.importorskip("uvicorn")
    calls: list[dict] = []
    monkeypatch.setattr(uvicorn, "run", lambda **kwargs: calls.append(kwargs))
    return calls


def test_server_limits_default_to_bounded_values() -> None:
    limits = server_connection_limits_from_env({})

    assert limits == ServerConnectionLimits()
    assert limits.uvicorn_options() == {
        "limit_concurrency": 128,
        "timeout_keep_alive": 5,
        "timeout_graceful_shutdown": 30,
    }


def test_server_limits_load_from_environment_without_echoing_values() -> None:
    limits = server_connection_limits_from_env(
        {
            LIMIT_CONCURRENCY_ENV_VAR: "64",
            TIMEOUT_KEEP_ALIVE_ENV_VAR: "15",
            TIMEOUT_GRACEFUL_SHUTDOWN_ENV_VAR: " 20 ",
        }
    )
    assert limits == ServerConnectionLimits(64, 15, 20)

    for name in SERVER_ENV_VARS:
        for raw in ("synthetic-canary", "0", "-3", "2.5", "999999999"):
            with pytest.raises(ValueError) as caught:
                server_connection_limits_from_env({name: raw})
            assert str(caught.value).startswith(f"{name} must be an integer")
            assert "synthetic" not in str(caught.value)


def test_launcher_passes_bounded_connection_limits_to_uvicorn(monkeypatch) -> None:
    _clear_server_env(monkeypatch)
    calls = _capture_uvicorn_run(monkeypatch)

    service_logging.main([])

    assert len(calls) == 1
    options = calls[0]
    assert options["limit_concurrency"] == 128
    assert options["timeout_keep_alive"] == 5
    assert options["timeout_graceful_shutdown"] == 30
    assert options["host"] == "127.0.0.1"
    assert options["access_log"] is False


def test_launcher_reads_environment_and_flags_override_it(monkeypatch) -> None:
    _clear_server_env(monkeypatch)
    monkeypatch.setenv(LIMIT_CONCURRENCY_ENV_VAR, "32")
    monkeypatch.setenv(TIMEOUT_KEEP_ALIVE_ENV_VAR, "9")
    monkeypatch.setenv(TIMEOUT_GRACEFUL_SHUTDOWN_ENV_VAR, "12")
    calls = _capture_uvicorn_run(monkeypatch)

    service_logging.main([])
    service_logging.main(["--limit-concurrency", "8", "--timeout-keep-alive", "2"])

    assert [
        (
            call["limit_concurrency"],
            call["timeout_keep_alive"],
            call["timeout_graceful_shutdown"],
        )
        for call in calls
    ] == [(32, 9, 12), (8, 2, 12)]


@pytest.mark.parametrize(
    ("argv", "environment"),
    [
        (["--limit-concurrency", "0"], {}),
        (["--timeout-graceful-shutdown", "-1"], {}),
        ([], {TIMEOUT_KEEP_ALIVE_ENV_VAR: "synthetic-canary"}),
    ],
)
def test_launcher_rejects_unbounded_values_with_a_usage_error(
    monkeypatch, capsys, argv, environment
) -> None:
    _clear_server_env(monkeypatch)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    calls = _capture_uvicorn_run(monkeypatch)

    with pytest.raises(SystemExit) as caught:
        service_logging.main(argv)

    assert caught.value.code == 2
    assert calls == []
    assert "synthetic" not in capsys.readouterr().err


def test_launcher_log_config_counts_connection_limit_rejections() -> None:
    config = service_logging._launcher_log_config()
    factory_path = config["filters"]["connection_limit"]["()"]
    module_name, _, class_name = factory_path.rpartition(".")

    assert getattr(import_module(module_name), class_name) is ConnectionLimitLogFilter
    assert config["loggers"]["uvicorn.error"] == {"filters": ["connection_limit"]}
    assert config["loggers"]["uvicorn.access"]["level"] == "CRITICAL"


def test_connection_limit_filter_counts_only_the_fixed_warning() -> None:
    log_filter = ConnectionLimitLogFilter()

    def record(level: int, message: str) -> logging.LogRecord:
        return logging.LogRecord("uvicorn.error", level, __file__, 1, message, (), None)

    before = connection_rejection_count()
    assert log_filter.filter(record(logging.WARNING, "Exceeded concurrency limit."))
    assert log_filter.filter(record(logging.WARNING, "synthetic other warning"))
    assert log_filter.filter(record(logging.INFO, "Exceeded concurrency limit."))

    assert connection_rejection_count() == before + 1


def test_registry_renders_bounded_transport_counters() -> None:
    assert REQUEST_BODY_TIMEOUT_REASONS == BODY_TIMEOUT_REASONS
    registry = PrometheusMetricsRegistry()
    registry.record_request_body_timeout(reason="min_rate")
    registry.record_request_body_timeout(reason="synthetic-free-text")
    registry.set_connection_rejected_total(3)

    rendered = registry.render()

    assert f"# TYPE {REQUEST_BODY_TIMEOUT_NAME} counter" in rendered
    assert f'{REQUEST_BODY_TIMEOUT_NAME}{{reason="deadline"}} 0' in rendered
    assert f'{REQUEST_BODY_TIMEOUT_NAME}{{reason="min_rate"}} 1' in rendered
    assert f'{REQUEST_BODY_TIMEOUT_NAME}{{reason="other"}} 1' in rendered
    assert f"{CONNECTION_REJECTED_NAME} 3" in rendered
    assert "synthetic" not in rendered


def test_metrics_endpoint_exposes_connection_rejections(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from openmed.service.app import create_app

    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.setenv("OPENMED_SERVICE_METRICS_ENABLED", "true")
    ConnectionLimitLogFilter().filter(
        logging.LogRecord(
            "uvicorn.error",
            logging.WARNING,
            __file__,
            1,
            "Exceeded concurrency limit.",
            (),
            None,
        )
    )
    expected = connection_rejection_count()

    with TestClient(create_app(), base_url="http://127.0.0.1") as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert f"{CONNECTION_REJECTED_NAME} {expected}" in response.text


def _wait_until(condition: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 10
    while not condition():
        if time.monotonic() >= deadline:
            raise AssertionError("loopback test server did not reach its state")
        time.sleep(0.01)


def _get(port: int) -> tuple[int, bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request("GET", "/", headers={"Connection": "close"})
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def test_concurrency_limit_returns_503_and_is_counted() -> None:
    uvicorn = pytest.importorskip("uvicorn")

    async def app(scope, receive, send) -> None:
        assert scope["type"] == "http"
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-length", b"2")],
            }
        )
        await send({"type": "http.response.body", "body": b"ok"})

    limits = server_connection_limits_from_env({LIMIT_CONCURRENCY_ENV_VAR: "2"})
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            lifespan="off",
            log_config=None,
            access_log=False,
            **limits.uvicorn_options(),
        )
    )
    error_logger = logging.getLogger("uvicorn.error")
    previous_level = error_logger.level
    log_filter = ConnectionLimitLogFilter()
    error_logger.setLevel(logging.WARNING)
    error_logger.addFilter(log_filter)
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [listener]}, daemon=True
    )
    thread.start()
    try:
        _wait_until(lambda: server.started)
        connections = server.server_state.connections

        assert _get(port) == (200, b"ok")
        _wait_until(lambda: not connections)
        before = connection_rejection_count()
        with socket.create_connection(("127.0.0.1", port), timeout=5):
            # An idle connection holds one of the two slots.
            _wait_until(lambda: len(connections) == 1)
            status, body = _get(port)

        assert status == 503
        assert connection_rejection_count() == before + 1
        assert b"127.0.0.1" not in body
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        error_logger.removeFilter(log_filter)
        error_logger.setLevel(previous_level)
        listener.close()


def test_container_images_set_the_same_connection_limits() -> None:
    expected = {
        LIMIT_CONCURRENCY_ENV_VAR: "128",
        TIMEOUT_KEEP_ALIVE_ENV_VAR: "5",
        TIMEOUT_GRACEFUL_SHUTDOWN_ENV_VAR: "30",
    }
    assert {
        name: str(value)
        for name, value in zip(
            SERVER_ENV_VARS, ServerConnectionLimits().uvicorn_options().values()
        )
    } == expected

    for dockerfile in DOCKERFILES:
        text = dockerfile.read_text(encoding="utf-8")
        for name, value in expected.items():
            assert re.search(rf"^\s+{name}={value}\b", text, re.MULTILINE), (
                dockerfile.name,
                name,
            )


TRANSPORT_DEFAULTS = {
    LIMIT_CONCURRENCY_ENV_VAR: "128",
    TIMEOUT_KEEP_ALIVE_ENV_VAR: "5",
    TIMEOUT_GRACEFUL_SHUTDOWN_ENV_VAR: "30",
    REQUEST_BODY_TIMEOUT_ENV_VAR: "60",
    REQUEST_BODY_GRACE_ENV_VAR: "10",
    REQUEST_BODY_MIN_RATE_ENV_VAR: "1024",
}


def _resolve_compose_defaults(environment: dict[str, str]) -> dict[str, str]:
    resolved = {}
    for name in TRANSPORT_DEFAULTS:
        match = re.fullmatch(
            r"\$\{(?P<name>\w+):-(?P<default>[^}]*)\}", environment[name]
        )
        assert match is not None, name
        assert match["name"] == name
        resolved[name] = match["default"]
    return resolved


@pytest.mark.parametrize(
    ("compose_file", "service"),
    [
        (ROOT / "docker-compose.yml", "app"),
        (ROOT / "deploy" / "openmed-compose.yaml", "openmed"),
    ],
)
def test_compose_renders_overridable_transport_limits(compose_file, service) -> None:
    yaml = pytest.importorskip("yaml")
    compose = yaml.safe_load(compose_file.read_text(encoding="utf-8"))
    environment = compose["services"][service]["environment"]

    resolved = _resolve_compose_defaults(environment)

    assert resolved == TRANSPORT_DEFAULTS
    assert server_connection_limits_from_env(resolved) == ServerConnectionLimits()
    assert request_body_read_limits_from_env(resolved) == RequestBodyReadLimits()


def _render_helm_configmap(*args: str) -> dict[str, str]:
    yaml = pytest.importorskip("yaml")
    helm = shutil.which("helm")
    if helm is None:
        pytest.skip("helm is not installed")
    result = subprocess.run(
        [
            helm,
            "template",
            "openmed-service",
            str(ROOT / "deploy" / "helm" / "openmed-service"),
            "--namespace",
            "openmed",
            *args,
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    configmaps = [
        manifest
        for manifest in yaml.safe_load_all(result.stdout)
        if isinstance(manifest, dict) and manifest.get("kind") == "ConfigMap"
    ]
    assert len(configmaps) == 1
    return configmaps[0]["data"]


def test_helm_renders_default_transport_limits() -> None:
    data = _render_helm_configmap()

    assert {name: data[name] for name in TRANSPORT_DEFAULTS} == TRANSPORT_DEFAULTS
    assert server_connection_limits_from_env(data) == ServerConnectionLimits()
    assert request_body_read_limits_from_env(data) == RequestBodyReadLimits()


def test_helm_renders_large_transport_overrides_as_plain_numbers(tmp_path) -> None:
    values = tmp_path / "synthetic-transport-values.yaml"
    values.write_text(
        "config:\n"
        "  transport:\n"
        "    limitConcurrency: 100000\n"
        "    timeoutKeepAliveSeconds: 75\n"
        "    timeoutGracefulShutdownSeconds: 20\n"
        "    requestBodyTimeoutSeconds: 300\n"
        "    requestBodyGraceSeconds: 2.5\n"
        "    requestBodyMinBytesPerSecond: 1073741824\n",
        encoding="utf-8",
    )

    data = _render_helm_configmap("--values", str(values))

    assert server_connection_limits_from_env(data) == ServerConnectionLimits(
        limit_concurrency=100_000,
        timeout_keep_alive=75,
        timeout_graceful_shutdown=20,
    )
    assert request_body_read_limits_from_env(data) == RequestBodyReadLimits(
        timeout_seconds=300,
        grace_seconds=2.5,
        min_bytes_per_second=1024 * 1024 * 1024,
    )
