"""Admission, retention and tuning bounds for FHIR bulk and SMART service jobs."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from openmed.service.app import create_app
from openmed.service.bulk_data import (
    FHIRBulkJobConfig,
    FHIRBulkJobManager,
    FHIRBulkJobStatus,
)
from openmed.service.smart_backend import (
    SMARTBackendConfig,
    SMARTBackendJobManager,
    SMARTBackendJobStatus,
)

SECRET_KEY = "SENTINEL-PRIVATE-KEY-MATERIAL"
SECRET_CLIENT = "sentinel-client-credential"
SECRET_HOST = "sentinel-fhir.example.test"
SECRET_PATH = "sentinel-output-path"
SECRET_ASSERTION = "sentinel-client-assertion"
SENTINELS = (SECRET_KEY, SECRET_CLIENT, SECRET_HOST, SECRET_PATH, SECRET_ASSERTION)
KINDS = ("bulk", "smart")
BULK_ROUTE = "/fhir/bulk/exports"
SMART_ROUTE = "/fhir/smart-backend/ingestions"


def _bounds(**kwargs: Any) -> Any:
    from openmed.service.job_bounds import BulkJobBounds

    return BulkJobBounds(**kwargs)


def _saturated_error() -> type[Exception]:
    from openmed.service.job_bounds import BulkJobsSaturatedError

    return BulkJobsSaturatedError


class _Clock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _manager(kind: str, **kwargs: Any) -> Any:
    if kind == "bulk":
        return FHIRBulkJobManager(**kwargs)
    return SMARTBackendJobManager(**kwargs)


def _config(kind: str, tmp_path: Path, name: str) -> Any:
    fields: dict[str, Any] = {
        "output_dir": str(tmp_path / name),
        "fhir_base_url": f"https://{SECRET_HOST}",
        "token_url": f"https://{SECRET_HOST}/token",
        "client_id": SECRET_CLIENT,
        "private_key_pem": SECRET_KEY,
        "poll_interval_seconds": 0,
    }
    if kind == "bulk":
        return FHIRBulkJobConfig(**fields)
    return SMARTBackendConfig(**fields)


def _install_synthetic_runner(
    manager: Any,
    clock: _Clock,
    *,
    held: dict[str, asyncio.Event] | None = None,
) -> None:
    """Replace the job body with a synthetic one; no transport is touched."""

    async def run(job_id: str, config: Any, deidentifier: Any) -> None:
        if held is not None and job_id in held:
            await held[job_id].wait()
        record = manager._jobs[job_id]
        record.status.status = "succeeded"
        record.status.updated_at = clock()

    manager._run = run


def _assert_no_sentinels(*texts: str) -> None:
    rendered = "\n".join(texts)
    for sentinel in SENTINELS:
        assert sentinel not in rendered


@pytest.mark.parametrize("kind", KINDS)
def test_running_ceiling_rejects_next_start_before_creating_a_task(
    kind: str,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    calls: list[str] = []
    saturated = _saturated_error()

    async def scenario() -> None:
        gate = asyncio.Event()

        async def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.method)
            await gate.wait()
            return httpx.Response(500)

        manager = _manager(
            kind,
            transport=httpx.MockTransport(handler),
            client_assertion_builder=lambda _: SECRET_ASSERTION,
            bounds=_bounds(max_running_jobs=2, max_retained_jobs=4),
        )
        manager.start(_config(kind, tmp_path, "one"), job_id="one")
        manager.start(_config(kind, tmp_path, "two"), job_id="two")
        for _ in range(50):
            await asyncio.sleep(0)
        assert calls == ["POST", "POST"]

        tasks_before = len(asyncio.all_tasks())
        with pytest.raises(saturated) as caught:
            manager.start(_config(kind, tmp_path, "three"), job_id="three")
        with pytest.raises(saturated):
            manager.ensure_capacity()
        assert len(asyncio.all_tasks()) == tasks_before
        assert set(manager._jobs) == {"one", "two"}
        assert not (tmp_path / "three").exists()
        details = caught.value.to_details()
        assert details["max_running_jobs"] == 2
        assert caught.value.retry_after_seconds > 0
        _assert_no_sentinels(str(caught.value), json.dumps(details))

        gate.set()
        await asyncio.gather(*(record.task for record in manager._jobs.values()))
        assert calls == ["POST", "POST"]
        assert {manager.get(job).status for job in ("one", "two")} == {"failed"}

        # A finished job frees its slot for the next admission.
        manager.ensure_capacity()
        status = manager.start(_config(kind, tmp_path, "three"), job_id="three")
        assert status.status == "running"
        await manager.cancel_all()

    asyncio.run(scenario())
    _assert_no_sentinels(caplog.text)


@pytest.mark.parametrize("kind", KINDS)
def test_terminal_records_expire_after_ttl_under_injected_clock(
    kind: str, tmp_path: Path
) -> None:
    clock = _Clock()

    async def scenario() -> None:
        held = {"held": asyncio.Event()}
        manager = _manager(
            kind,
            bounds=_bounds(terminal_ttl_seconds=60),
            clock=clock,
        )
        _install_synthetic_runner(manager, clock, held=held)
        config = _config(kind, tmp_path, "out")
        manager.start(config, job_id="held")
        started = manager.start(config, job_id="done")
        assert started.created_at == clock.now
        await manager._jobs["done"].task
        assert manager.get("done").status == "succeeded"

        clock.now += 59
        assert manager.get("done").status == "succeeded"
        clock.now += 1
        with pytest.raises(KeyError):
            manager.get("done")
        assert "done" not in manager._jobs
        with pytest.raises(KeyError):
            manager.get("never-started")

        # Running jobs never expire, whatever the clock says.
        clock.now += 10_000
        assert manager.get("held").status == "running"
        held["held"].set()
        await manager._jobs["held"].task
        assert manager.get("held").status == "succeeded"
        clock.now += 60
        with pytest.raises(KeyError):
            manager.get("held")
        assert not manager._jobs

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", KINDS)
def test_retained_records_stay_bounded_across_ten_thousand_jobs(
    kind: str, tmp_path: Path
) -> None:
    clock = _Clock()

    async def scenario() -> None:
        held = {"held": asyncio.Event()}
        manager = _manager(
            kind,
            bounds=_bounds(
                max_running_jobs=2,
                max_retained_jobs=8,
                terminal_ttl_seconds=10**9,
            ),
            clock=clock,
        )
        _install_synthetic_runner(manager, clock, held=held)
        config = _config(kind, tmp_path, "out")
        manager.start(config, job_id="held")
        peak = 0
        for index in range(10_000):
            clock.now += 1
            manager.start(config, job_id=f"job-{index}")
            peak = max(peak, len(manager._jobs))
            await manager._jobs[f"job-{index}"].task
            assert "held" in manager._jobs
        assert peak <= 8
        assert set(manager._jobs) == {"held"} | {
            f"job-{index}" for index in range(9_993, 10_000)
        }
        assert manager.get("held").status == "running"
        held["held"].set()
        await manager._jobs["held"].task

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", KINDS)
def test_retained_cap_never_evicts_running_jobs(kind: str, tmp_path: Path) -> None:
    clock = _Clock()

    async def scenario() -> None:
        held = {name: asyncio.Event() for name in ("first", "second")}
        manager = _manager(
            kind,
            bounds=_bounds(max_running_jobs=2, max_retained_jobs=2),
            clock=clock,
        )
        _install_synthetic_runner(manager, clock, held=held)
        config = _config(kind, tmp_path, "out")
        manager.start(config, job_id="first")
        manager.start(config, job_id="second")
        with pytest.raises(_saturated_error()):
            manager.start(config, job_id="third")
        assert set(manager._jobs) == {"first", "second"}

        clock.now += 1
        held["second"].set()
        await manager._jobs["second"].task
        manager.start(config, job_id="third")
        assert set(manager._jobs) == {"first", "third"}
        held["first"].set()
        await asyncio.gather(*(record.task for record in manager._jobs.values()))

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_running_jobs": 0},
        {"max_retained_jobs": 0},
        {"terminal_ttl_seconds": 0},
        {"retry_after_seconds": 0},
        {"max_running_jobs": True},
        {"max_running_jobs": 9, "max_retained_jobs": 8},
    ],
)
def test_bounds_reject_invalid_configuration(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _bounds(**kwargs)


def test_service_reads_bulk_job_bounds_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.setenv("OPENMED_SERVICE_BULK_MAX_RUNNING_JOBS", "3")
    monkeypatch.setenv("OPENMED_SERVICE_BULK_MAX_RETAINED_JOBS", "12")
    monkeypatch.setenv("OPENMED_SERVICE_BULK_JOB_TTL_SECONDS", "90")
    app = create_app()
    with TestClient(app, base_url="http://127.0.0.1"):
        managers = (app.state.fhir_bulk_jobs, app.state.smart_backend_jobs)
    for manager in managers:
        assert manager.bounds.max_running_jobs == 3
        assert manager.bounds.max_retained_jobs == 12
        assert manager.bounds.terminal_ttl_seconds == 90


@pytest.mark.parametrize(
    "env",
    [
        {"OPENMED_SERVICE_BULK_MAX_RUNNING_JOBS": "zero"},
        {"OPENMED_SERVICE_BULK_MAX_RETAINED_JOBS": "0"},
        {"OPENMED_SERVICE_BULK_JOB_TTL_SECONDS": "-5"},
        {
            "OPENMED_SERVICE_BULK_MAX_RUNNING_JOBS": "9",
            "OPENMED_SERVICE_BULK_MAX_RETAINED_JOBS": "8",
        },
    ],
)
def test_invalid_bulk_job_bounds_fail_at_config_time(
    env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="OPENMED_SERVICE_BULK_"):
        create_app()


async def _hang(_: httpx.Request) -> httpx.Response:
    await asyncio.sleep(3600)
    return httpx.Response(500)


def _route_payload(output_dir: str) -> dict[str, Any]:
    return {
        "output_dir": output_dir,
        "fhir_base_url": f"https://{SECRET_HOST}",
        "token_url": f"https://{SECRET_HOST}/token",
        "client_id": SECRET_CLIENT,
        "private_key_pem": SECRET_KEY,
    }


def _route_app(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manager: Any,
) -> Any:
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.setenv("OPENMED_SERVICE_BULK_OUTPUT_ROOT", str(tmp_path))
    app = create_app()
    if kind == "bulk":
        app.state.fhir_bulk_jobs = manager
    else:
        app.state.smart_backend_jobs = manager
    return app


@pytest.mark.parametrize("kind", KINDS)
def test_saturated_route_returns_503_with_retry_after_and_no_side_effects(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    manager = _manager(
        kind,
        transport=httpx.MockTransport(_hang),
        client_assertion_builder=lambda _: SECRET_ASSERTION,
        bounds=_bounds(max_running_jobs=1, max_retained_jobs=4),
    )
    app = _route_app(kind, tmp_path, monkeypatch, manager)
    route = BULK_ROUTE if kind == "bulk" else SMART_ROUTE
    with TestClient(app, base_url="http://127.0.0.1") as client:
        first = client.post(route, json=_route_payload("first-run"))
        assert first.status_code in {200, 202}
        first_id = first.json()["job_id"]

        rejected = client.post(route, json=_route_payload(SECRET_PATH))
        assert rejected.status_code == 503
        body = rejected.json()["error"]
        assert body["code"] == "bulk_jobs_saturated"
        assert int(rejected.headers["Retry-After"]) >= 1
        assert body["details"]["max_running_jobs"] == 1
        assert not (tmp_path / SECRET_PATH).exists()
        assert set(manager._jobs) == {first_id}
        _assert_no_sentinels(
            rejected.text,
            json.dumps(dict(rejected.headers)),
        )

        # Cancelling the running job frees the slot for the next start.
        assert client.delete(f"{route}/{first_id}").status_code == 202
        second = client.post(route, json=_route_payload("second-run"))
        assert second.status_code in {200, 202}
    _assert_no_sentinels(caplog.text)


@pytest.mark.parametrize("kind", KINDS)
def test_expired_job_id_matches_unknown_job_id_response(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    manager = _manager(kind, bounds=_bounds(terminal_ttl_seconds=30), clock=clock)
    _install_synthetic_runner(manager, clock)
    app = _route_app(kind, tmp_path, monkeypatch, manager)
    route = BULK_ROUTE if kind == "bulk" else SMART_ROUTE
    with TestClient(app, base_url="http://127.0.0.1") as client:
        job_id = client.post(route, json=_route_payload("run")).json()["job_id"]
        for _ in range(100):
            if client.get(f"{route}/{job_id}").json()["status"] != "running":
                break
        assert client.get(f"{route}/{job_id}").json()["status"] == "succeeded"
        clock.now += 31
        expired = client.get(f"{route}/{job_id}")
        unknown = client.get(f"{route}/{'0' * 32}")

    def stable(response: httpx.Response) -> dict[str, Any]:
        error = dict(response.json()["error"])
        error.pop("request_id", None)
        return error

    assert expired.status_code == unknown.status_code == 404
    assert stable(expired) == stable(unknown)


_SHARED_OUT_OF_RANGE = [
    ("max_inflight_downloads", 0),
    ("max_inflight_downloads", 17),
    ("poll_interval_seconds", 0),
    ("poll_interval_seconds", 0.5),
    ("poll_interval_seconds", 301),
    ("request_timeout_seconds", 0),
    ("request_timeout_seconds", 301),
]
_OUT_OF_RANGE = [("smart", field, value) for field, value in _SHARED_OUT_OF_RANGE]
_OUT_OF_RANGE += [("bulk", field, value) for field, value in _SHARED_OUT_OF_RANGE]
_OUT_OF_RANGE += [
    ("bulk", "max_buffered_resources", 0),
    ("bulk", "max_buffered_resources", 1001),
]
_BOUND_CODES = {"greater_than", "greater_than_equal", "less_than_equal"}


@pytest.mark.parametrize(("kind", "field", "value"), _OUT_OF_RANGE)
def test_out_of_range_tuning_is_rejected_with_controlled_codes(
    kind: str,
    field: str,
    value: float,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    manager = _manager(
        kind,
        transport=httpx.MockTransport(_hang),
        client_assertion_builder=lambda _: SECRET_ASSERTION,
    )
    app = _route_app(kind, tmp_path, monkeypatch, manager)
    route = BULK_ROUTE if kind == "bulk" else SMART_ROUTE
    payload = _route_payload(SECRET_PATH)
    payload[field] = value
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post(route, json=payload)

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert [detail["field"] for detail in error["details"]] == [f"body.{field}"]
    assert error["details"][0]["type"] in _BOUND_CODES
    assert not manager._jobs
    assert not (tmp_path / SECRET_PATH).exists()
    _assert_no_sentinels(response.text, caplog.text)


@pytest.mark.parametrize("kind", KINDS)
def test_non_finite_poll_interval_is_rejected(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(kind, transport=httpx.MockTransport(_hang))
    app = _route_app(kind, tmp_path, monkeypatch, manager)
    route = BULK_ROUTE if kind == "bulk" else SMART_ROUTE
    # 1e999 parses to infinity; JSON encoders refuse to emit it directly.
    body = json.dumps(_route_payload("run"))[:-1] + ', "poll_interval_seconds": 1e999}'
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post(
            route,
            content=body,
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert not manager._jobs


class _RecordingManager:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.configs: list[Any] = []

    def ensure_capacity(self) -> None:
        return None

    def start(self, config: Any, *, deidentifier: Any = None, job_id: Any = None):
        self.configs.append(config)
        status_type = (
            FHIRBulkJobStatus if self.kind == "bulk" else SMARTBackendJobStatus
        )
        return status_type(
            job_id=job_id or "recorded",
            status="running",
            created_at=1.0,
            updated_at=1.0,
        )

    async def cancel_all(self) -> None:
        return None


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "tuning",
    [
        {},
        {
            "max_inflight_downloads": 16,
            "poll_interval_seconds": 1.0,
            "request_timeout_seconds": 300,
        },
        {
            "max_inflight_downloads": 1,
            "poll_interval_seconds": 300,
            "request_timeout_seconds": 0.5,
        },
    ],
)
def test_in_range_tuning_is_accepted_and_forwarded_unchanged(
    kind: str,
    tuning: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _RecordingManager(kind)
    app = _route_app(kind, tmp_path, monkeypatch, manager)
    route = BULK_ROUTE if kind == "bulk" else SMART_ROUTE
    payload = _route_payload("run")
    payload.update(tuning)
    if kind == "bulk" and tuning:
        payload["max_buffered_resources"] = 1000
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post(route, json=payload)

    assert response.status_code in {200, 202}
    (config,) = manager.configs
    assert config.max_inflight_downloads == tuning.get("max_inflight_downloads", 2)
    assert config.poll_interval_seconds == tuning.get("poll_interval_seconds", 1.0)
    assert config.request_timeout_seconds == tuning.get("request_timeout_seconds", 30.0)
    if kind == "bulk":
        assert config.max_buffered_resources == (1000 if tuning else 1)
