"""Tests for submission-time capacity limits on async de-identification jobs."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator

import pytest
from fastapi.testclient import TestClient

import openmed
from openmed.service import jobs
from openmed.service import runtime as service_runtime
from openmed.service.app import create_app
from openmed.service.job_admission import (
    DEFAULT_MAX_DOCUMENTS_PER_JOB,
    DEFAULT_MAX_PENDING_JOBS,
    DEFAULT_MAX_QUEUED_TEXT_BYTES,
    JobAdmission,
    JobAdmissionLimits,
    JobCost,
    JobLimitExceededError,
    JobQueueFullError,
    JobTooLargeError,
    JobTooManyDocumentsError,
)
from openmed.service.metrics import (
    JOB_QUEUE_DEPTH_NAME,
    JOB_QUEUE_TEXT_BYTES_NAME,
    PrometheusMetricsRegistry,
)
from openmed.service.schemas import DeidentifyJobDocument, DeidentifyJobRequest

LOOPBACK_BASE_URL = "http://127.0.0.1"
SYNTHETIC_PHI = "Paciente: Maria Garcia"
WAIT_SECONDS = 5.0


class FakeLoader:
    """Minimal model loader double for warm-pool bookkeeping."""

    def __init__(self, config: Any) -> None:
        self.config = config

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def loaded_models(self) -> dict[str, dict[str, int]]:
        return {}


@pytest.fixture(autouse=True)
def clear_service_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [
        "OPENMED_PROFILE",
        "OPENMED_SERVICE_PRELOAD_MODELS",
        "OPENMED_SERVICE_KEEP_ALIVE",
        "OPENMED_SERVICE_MAX_RESIDENT_MODELS",
        "OPENMED_SERVICE_MAX_TEXT_LENGTH",
        "OPENMED_SERVICE_BATCHING_ENABLED",
        "OPENMED_SERVICE_CORS_ORIGINS",
        "OPENMED_SERVICE_TRUSTED_HOSTS",
        "OPENMED_SERVICE_COALESCING_ENABLED",
        "OPENMED_SERVICE_RATE_LIMIT_RPS",
        "OPENMED_SERVICE_RATE_LIMIT_BURST",
        "OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY",
        "OPENMED_SERVICE_METRICS_ENABLED",
        jobs.SERVICE_JOBS_STORE_ENV_VAR,
        jobs.SERVICE_JOBS_TTL_ENV_VAR,
        jobs.SERVICE_JOBS_WORKERS_ENV_VAR,
        jobs.SERVICE_JOBS_MAX_PENDING_ENV_VAR,
        jobs.SERVICE_JOBS_MAX_QUEUED_BYTES_ENV_VAR,
        jobs.SERVICE_JOBS_MAX_DOCUMENTS_ENV_VAR,
    ]:
        monkeypatch.delenv(name, raising=False)


def _wait_until(predicate: Callable[[], bool]) -> None:
    deadline = time.monotonic() + WAIT_SECONDS
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached")


def _gauges(registry: PrometheusMetricsRegistry) -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in registry.render().splitlines():
        name, _, value = line.partition(" ")
        if name in {JOB_QUEUE_DEPTH_NAME, JOB_QUEUE_TEXT_BYTES_NAME}:
            values[name] = int(value)
    return values[JOB_QUEUE_DEPTH_NAME], values[JOB_QUEUE_TEXT_BYTES_NAME]


def _request(*texts: str) -> DeidentifyJobRequest:
    return DeidentifyJobRequest(
        documents=[
            DeidentifyJobDocument(id=f"synthetic-{index}", text=text)
            for index, text in enumerate(texts)
        ]
    )


class _BlockingDeidentifier:
    """Controlled worker body that blocks until the test releases it."""

    def __init__(self, *, fail: bool = False) -> None:
        self.release = threading.Event()
        self.started = threading.Event()
        self.fail = fail
        self.seen: list[str] = []
        self._lock = threading.Lock()

    def __call__(self, text: str, **_kwargs: Any) -> Any:
        with self._lock:
            self.seen.append(text)
        self.started.set()
        if not self.release.wait(WAIT_SECONDS):
            raise AssertionError("worker was never released")
        if self.fail:
            raise RuntimeError("synthetic model failure")
        return SimpleNamespace(pii_entities=[])


def _fake_runtime(
    deidentifier: Callable[..., Any],
    metrics: PrometheusMetricsRegistry,
) -> SimpleNamespace:
    return SimpleNamespace(
        metrics=metrics,
        validate_served_model=lambda model_name: model_name,
        run_model_request=lambda _model, _keep_alive, operation: operation(),
        deidentifier=deidentifier,
    )


@pytest.fixture
def controlled_queue(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Iterator[Callable[..., tuple[jobs.DeidentifyJobQueue, _BlockingDeidentifier]]]:
    created: list[tuple[jobs.DeidentifyJobQueue, _BlockingDeidentifier]] = []

    def build(
        limits: JobAdmissionLimits,
        *,
        fail: bool = False,
        registry: PrometheusMetricsRegistry | None = None,
    ) -> tuple[jobs.DeidentifyJobQueue, _BlockingDeidentifier]:
        deidentifier = _BlockingDeidentifier(fail=fail)
        metrics = PrometheusMetricsRegistry() if registry is None else registry
        runtime = _fake_runtime(deidentifier, metrics)
        queue = jobs.DeidentifyJobQueue(
            runtime,  # type: ignore[arg-type]
            store=jobs.LocalJobStore(tmp_path / f"jobs-{len(created)}.json"),
            max_workers=1,
            admission=JobAdmission(limits, metrics=metrics),
        )

        def process(_payload: DeidentifyJobRequest, document: DeidentifyJobDocument):
            return deidentifier(document.text)

        monkeypatch.setattr(queue, "_deidentify_document", process)
        created.append((queue, deidentifier))
        return queue, deidentifier

    yield build
    for queue, deidentifier in created:
        deidentifier.release.set()
        queue.shutdown()


def test_limits_have_safe_defaults_and_reject_non_positive_values() -> None:
    limits = JobAdmissionLimits()
    assert limits.max_pending_jobs == DEFAULT_MAX_PENDING_JOBS
    assert limits.max_queued_text_bytes == DEFAULT_MAX_QUEUED_TEXT_BYTES
    assert limits.max_documents_per_job == DEFAULT_MAX_DOCUMENTS_PER_JOB

    for field_name in [
        "max_pending_jobs",
        "max_queued_text_bytes",
        "max_documents_per_job",
        "retry_after_seconds",
    ]:
        for value in (0, -1, True, 1.5):
            with pytest.raises(ValueError, match=field_name):
                JobAdmissionLimits(**{field_name: value})


def test_env_limits_parse_overrides_and_reject_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert jobs.job_admission_limits_from_env() == JobAdmissionLimits()

    monkeypatch.setenv(jobs.SERVICE_JOBS_MAX_PENDING_ENV_VAR, "3")
    monkeypatch.setenv(jobs.SERVICE_JOBS_MAX_QUEUED_BYTES_ENV_VAR, "4096")
    monkeypatch.setenv(jobs.SERVICE_JOBS_MAX_DOCUMENTS_ENV_VAR, " ")
    limits = jobs.job_admission_limits_from_env()
    assert limits.max_pending_jobs == 3
    assert limits.max_queued_text_bytes == 4096
    assert limits.max_documents_per_job == DEFAULT_MAX_DOCUMENTS_PER_JOB

    for env_var in [
        jobs.SERVICE_JOBS_MAX_PENDING_ENV_VAR,
        jobs.SERVICE_JOBS_MAX_QUEUED_BYTES_ENV_VAR,
        jobs.SERVICE_JOBS_MAX_DOCUMENTS_ENV_VAR,
    ]:
        for raw in ("0", "-2", "many"):
            with monkeypatch.context() as patched:
                patched.setenv(env_var, raw)
                with pytest.raises(ValueError, match=env_var) as excinfo:
                    jobs.job_admission_limits_from_env()
                assert raw not in str(excinfo.value)


def test_job_cost_counts_documents_and_utf8_bytes() -> None:
    assert JobCost.for_texts(["ab", "é", "\ud800"]) == JobCost(
        documents=3,
        text_bytes=2 + 2 + 3,
    )
    assert JobCost.for_texts([]) == JobCost(documents=0, text_bytes=0)


def test_pending_job_ceiling_rejects_until_a_reservation_is_released() -> None:
    registry = PrometheusMetricsRegistry()
    admission = JobAdmission(
        JobAdmissionLimits(max_pending_jobs=2, retry_after_seconds=7),
        metrics=registry,
    )
    cost = JobCost(documents=1, text_bytes=10)
    first = admission.reserve(cost)
    admission.reserve(cost)

    with pytest.raises(JobQueueFullError) as excinfo:
        admission.reserve(cost)
    error = excinfo.value
    assert error.status_code == 503
    assert error.error_code == "job_queue_full"
    assert error.to_details() == {
        "reason": "pending_jobs",
        "limit": 2,
        "retry_after_seconds": 7,
    }
    assert admission.snapshot().pending_jobs == 2
    assert _gauges(registry) == (2, 20)

    first.release()
    first.release()
    assert admission.snapshot().pending_jobs == 1
    assert _gauges(registry) == (1, 10)
    admission.reserve(cost)
    assert _gauges(registry) == (2, 20)


def test_queued_byte_ceiling_counts_all_pending_jobs() -> None:
    admission = JobAdmission(JobAdmissionLimits(max_queued_text_bytes=100))
    held = admission.reserve(JobCost(documents=1, text_bytes=60))

    with pytest.raises(JobQueueFullError) as excinfo:
        admission.reserve(JobCost(documents=1, text_bytes=41))
    assert excinfo.value.to_details()["reason"] == "queued_text_bytes"
    assert admission.snapshot().queued_text_bytes == 60

    admission.reserve(JobCost(documents=1, text_bytes=40))
    held.release()
    assert admission.snapshot().queued_text_bytes == 40


def test_per_job_ceilings_reject_without_claiming_capacity() -> None:
    admission = JobAdmission(
        JobAdmissionLimits(max_documents_per_job=2, max_queued_text_bytes=50)
    )
    with pytest.raises(JobLimitExceededError) as too_many:
        admission.reserve(JobCost(documents=3, text_bytes=3))
    assert too_many.value.status_code == 422
    assert isinstance(too_many.value, JobTooManyDocumentsError)
    assert too_many.value.retry_after_seconds is None
    assert too_many.value.to_details() == {"reason": "documents_per_job", "limit": 2}

    with pytest.raises(JobLimitExceededError) as too_large:
        admission.reserve(JobCost(documents=1, text_bytes=51))
    assert isinstance(too_large.value, JobTooLargeError)
    assert too_large.value.to_details() == {"reason": "job_text_bytes", "limit": 50}

    assert admission.snapshot().pending_jobs == 0
    assert admission.snapshot().queued_text_bytes == 0


def test_reservation_from_another_admission_is_rejected() -> None:
    foreign = JobAdmission().reserve(JobCost(documents=1, text_bytes=1))
    admission = JobAdmission()
    with pytest.raises(ValueError, match="another job admission"):
        admission.release(foreign)
    assert not foreign.released


def test_concurrent_reservations_never_exceed_the_pending_ceiling() -> None:
    admission = JobAdmission(JobAdmissionLimits(max_pending_jobs=5))
    barrier = threading.Barrier(16)
    accepted: list[object] = []
    rejected: list[object] = []
    lock = threading.Lock()

    def attempt() -> None:
        barrier.wait()
        try:
            reservation = admission.reserve(JobCost(documents=1, text_bytes=1))
        except JobQueueFullError as exc:
            with lock:
                rejected.append(exc)
        else:
            with lock:
                accepted.append(reservation)

    threads = [threading.Thread(target=attempt) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(WAIT_SECONDS)

    assert len(accepted) == 5
    assert len(rejected) == 11
    assert admission.snapshot().pending_jobs == 5


def test_blocked_queue_rejects_without_record_then_accepts_after_drain(
    controlled_queue,
) -> None:
    queue, worker = controlled_queue(JobAdmissionLimits(max_pending_jobs=2))
    registry = queue.admission.metrics
    running = queue.submit(_request("synthetic running"))
    assert worker.started.wait(WAIT_SECONDS)
    queued = queue.submit(_request("synthetic queued"))
    records_before = json.loads(queue.store.path.read_text(encoding="utf-8"))

    with pytest.raises(JobQueueFullError):
        queue.submit(_request(SYNTHETIC_PHI))

    assert json.loads(queue.store.path.read_text(encoding="utf-8")) == records_before
    assert set(records_before["jobs"]) == {running["id"], queued["id"]}
    assert _gauges(registry) == (2, len("synthetic running") + len("synthetic queued"))

    worker.release.set()
    _wait_until(lambda: queue.admission.snapshot().pending_jobs == 0)
    assert _gauges(registry) == (0, 0)
    assert SYNTHETIC_PHI not in worker.seen
    for job_id in (running["id"], queued["id"]):
        record = queue.get(job_id)
        assert record is not None and record["status"] == "done"

    accepted = queue.submit(_request("synthetic after drain"))
    assert accepted["status"] == "queued"
    _wait_until(lambda: queue.admission.snapshot().pending_jobs == 0)


def test_failed_jobs_release_their_capacity(controlled_queue) -> None:
    queue, worker = controlled_queue(JobAdmissionLimits(max_pending_jobs=1), fail=True)
    failed = queue.submit(_request("synthetic failing"))
    assert worker.started.wait(WAIT_SECONDS)
    assert _gauges(queue.admission.metrics) == (1, len("synthetic failing"))

    worker.release.set()
    _wait_until(lambda: queue.admission.snapshot().pending_jobs == 0)

    record = queue.get(failed["id"])
    assert record is not None and record["status"] == "failed"
    assert _gauges(queue.admission.metrics) == (0, 0)


def test_unexpected_worker_error_still_releases_capacity(
    controlled_queue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue, worker = controlled_queue(JobAdmissionLimits(max_pending_jobs=1))
    worker.release.set()
    original_update = queue.store.update

    def failing_update(job_id: str, **changes: Any) -> dict[str, Any]:
        if changes.get("status") == "running":
            raise OSError("synthetic store failure")
        return original_update(job_id, **changes)

    monkeypatch.setattr(queue.store, "update", failing_update)
    queue.submit(_request("synthetic crash"))
    _wait_until(lambda: queue.admission.snapshot().pending_jobs == 0)
    assert _gauges(queue.admission.metrics) == (0, 0)


def test_cancelled_jobs_release_capacity_on_shutdown(controlled_queue) -> None:
    queue, worker = controlled_queue(JobAdmissionLimits(max_pending_jobs=3))
    queue.submit(_request("synthetic running"))
    assert worker.started.wait(WAIT_SECONDS)
    queue.submit(_request("synthetic cancelled one"))
    queue.submit(_request("synthetic cancelled two"))
    assert queue.admission.snapshot().pending_jobs == 3

    queue.shutdown()
    assert queue.admission.snapshot().pending_jobs == 1
    assert _gauges(queue.admission.metrics) == (1, len("synthetic running"))

    worker.release.set()
    _wait_until(lambda: queue.admission.snapshot().pending_jobs == 0)
    assert _gauges(queue.admission.metrics) == (0, 0)
    assert worker.seen == ["synthetic running"]


def test_store_failure_releases_reservation_and_queues_no_work(
    controlled_queue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue, worker = controlled_queue(JobAdmissionLimits(max_pending_jobs=1))
    worker.release.set()

    def fail_create(_record: dict[str, Any]) -> dict[str, Any]:
        raise OSError("synthetic store failure")

    with monkeypatch.context() as patched:
        patched.setattr(queue.store, "create", fail_create)
        with pytest.raises(OSError, match="synthetic store failure"):
            queue.submit(_request(SYNTHETIC_PHI))

    assert queue.admission.snapshot().pending_jobs == 0
    assert _gauges(queue.admission.metrics) == (0, 0)
    assert worker.seen == []
    queue.submit(_request("synthetic retry"))
    _wait_until(lambda: queue.admission.snapshot().pending_jobs == 0)
    assert worker.seen == ["synthetic retry"]


def test_from_env_builds_admission_from_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(jobs.SERVICE_JOBS_STORE_ENV_VAR, str(tmp_path / "jobs.json"))
    monkeypatch.setenv(jobs.SERVICE_JOBS_MAX_PENDING_ENV_VAR, "9")
    registry = PrometheusMetricsRegistry()
    queue = jobs.DeidentifyJobQueue.from_env(
        SimpleNamespace(metrics=registry)  # type: ignore[arg-type]
    )
    try:
        assert queue.admission.limits.max_pending_jobs == 9
        assert queue.admission.metrics is registry
    finally:
        queue.shutdown()


def _configure_app(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    **limits: str,
) -> Path:
    store_path = tmp_path / "jobs.json"
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.setenv("OPENMED_SERVICE_METRICS_ENABLED", "true")
    monkeypatch.setenv(jobs.SERVICE_JOBS_STORE_ENV_VAR, str(store_path))
    monkeypatch.setenv(jobs.SERVICE_JOBS_WORKERS_ENV_VAR, "1")
    for env_var, value in limits.items():
        monkeypatch.setenv(env_var, value)
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)
    return store_path


def _metric_value(client: TestClient, name: str) -> int:
    for line in client.get("/metrics").text.splitlines():
        metric, _, value = line.partition(" ")
        if metric == name:
            return int(value)
    raise AssertionError(f"metric {name} not rendered")


def _stored_job_ids(store_path: Path) -> set[str]:
    if not store_path.exists():
        return set()
    return set(json.loads(store_path.read_text(encoding="utf-8"))["jobs"])


def test_rest_queue_full_returns_503_with_retry_after_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_path = _configure_app(
        monkeypatch,
        tmp_path,
        **{jobs.SERVICE_JOBS_MAX_PENDING_ENV_VAR: "2"},
    )
    worker = _BlockingDeidentifier()
    monkeypatch.setattr(openmed, "deidentify", worker)

    with TestClient(
        create_app(),
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        try:
            first = client.post("/jobs", json={"documents": [{"text": "synthetic a"}]})
            assert first.status_code == 202
            assert worker.started.wait(WAIT_SECONDS)
            second = client.post("/jobs", json={"documents": [{"text": "synthetic b"}]})
            assert second.status_code == 202
            accepted_ids = {first.json()["id"], second.json()["id"]}

            rejected = client.post(
                "/jobs",
                json={"documents": [{"id": "mrn-0001", "text": SYNTHETIC_PHI}]},
            )

            assert rejected.status_code == 503
            assert rejected.headers["Retry-After"] == "5"
            error = rejected.json()["error"]
            assert error["code"] == "job_queue_full"
            assert error["details"] == {
                "reason": "pending_jobs",
                "limit": 2,
                "retry_after_seconds": 5,
            }
            assert "Maria" not in rejected.text
            assert "mrn-0001" not in rejected.text
            assert _stored_job_ids(store_path) == accepted_ids
            assert _metric_value(client, JOB_QUEUE_DEPTH_NAME) == 2
            assert _metric_value(client, JOB_QUEUE_TEXT_BYTES_NAME) == 22
        finally:
            worker.release.set()

        _wait_until(lambda: _metric_value(client, JOB_QUEUE_DEPTH_NAME) == 0)
        assert _metric_value(client, JOB_QUEUE_TEXT_BYTES_NAME) == 0
        retried = client.post("/jobs", json={"documents": [{"text": "synthetic c"}]})
        assert retried.status_code == 202
        _wait_until(lambda: _metric_value(client, JOB_QUEUE_DEPTH_NAME) == 0)

    assert SYNTHETIC_PHI not in worker.seen
    assert "Maria" not in store_path.read_text(encoding="utf-8")


def test_rest_queued_byte_ceiling_returns_503(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_path = _configure_app(
        monkeypatch,
        tmp_path,
        **{jobs.SERVICE_JOBS_MAX_QUEUED_BYTES_ENV_VAR: "40"},
    )
    worker = _BlockingDeidentifier()
    monkeypatch.setattr(openmed, "deidentify", worker)

    with TestClient(
        create_app(),
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        try:
            held = client.post("/jobs", json={"documents": [{"text": "x" * 30}]})
            assert held.status_code == 202
            rejected = client.post(
                "/jobs", json={"documents": [{"text": SYNTHETIC_PHI}]}
            )
            assert rejected.status_code == 503
            assert rejected.json()["error"]["details"]["reason"] == "queued_text_bytes"
            assert "Maria" not in rejected.text
            assert _stored_job_ids(store_path) == {held.json()["id"]}
        finally:
            worker.release.set()
        _wait_until(lambda: _metric_value(client, JOB_QUEUE_TEXT_BYTES_NAME) == 0)


def test_rest_per_job_ceilings_return_422_without_a_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_path = _configure_app(
        monkeypatch,
        tmp_path,
        **{
            jobs.SERVICE_JOBS_MAX_DOCUMENTS_ENV_VAR: "2",
            jobs.SERVICE_JOBS_MAX_QUEUED_BYTES_ENV_VAR: "16",
        },
    )
    worker = _BlockingDeidentifier()
    worker.release.set()
    monkeypatch.setattr(openmed, "deidentify", worker)

    with TestClient(
        create_app(),
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        too_many = client.post(
            "/jobs",
            json={"documents": [{"text": "a"}, {"text": "b"}, {"text": "Maria"}]},
        )
        too_large = client.post("/jobs", json={"documents": [{"text": SYNTHETIC_PHI}]})
        depth = _metric_value(client, JOB_QUEUE_DEPTH_NAME)

    assert too_many.status_code == 422
    assert "Retry-After" not in too_many.headers
    assert too_many.json()["error"]["code"] == "job_too_many_documents"
    assert too_many.json()["error"]["details"] == {
        "reason": "documents_per_job",
        "limit": 2,
    }
    assert too_large.status_code == 422
    assert too_large.json()["error"]["code"] == "job_too_large"
    assert too_large.json()["error"]["details"] == {
        "reason": "job_text_bytes",
        "limit": 16,
    }
    for response in (too_many, too_large):
        assert "Maria" not in response.text
    assert depth == 0
    assert _stored_job_ids(store_path) == set()
    assert worker.seen == []
