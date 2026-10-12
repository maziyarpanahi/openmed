"""Tests for async de-identification job APIs."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

import openmed
from openmed.core.pii import DeidentificationResult, PIIEntity
from openmed.service import jobs
from openmed.service import runtime as service_runtime
from openmed.service.app import create_app
from openmed.service.webhooks import WebhookDeliveryResult

LOOPBACK_BASE_URL = "http://127.0.0.1"


class FakeLoader:
    """Minimal model loader double for warm-pool bookkeeping."""

    def __init__(self, config):
        self.config = config

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def loaded_models(self) -> dict[str, dict[str, int]]:
        return {}


@pytest.fixture(autouse=True)
def clear_service_env(monkeypatch: pytest.MonkeyPatch) -> None:
    names = [
        "OPENMED_PROFILE",
        "OPENMED_SERVICE_PRELOAD_MODELS",
        "OPENMED_SERVICE_KEEP_ALIVE",
        "OPENMED_SERVICE_MAX_RESIDENT_MODELS",
        "OPENMED_SERVICE_MAX_TEXT_LENGTH",
        "OPENMED_SERVICE_BATCHING_ENABLED",
        "OPENMED_SERVICE_BATCH_MAX_SIZE",
        "OPENMED_SERVICE_BATCH_MAX_WAIT_MS",
        "OPENMED_SERVICE_CORS_ORIGINS",
        "OPENMED_SERVICE_TRUSTED_HOSTS",
        "OPENMED_SERVICE_COALESCING_ENABLED",
        "OPENMED_SERVICE_RATE_LIMIT_RPS",
        "OPENMED_SERVICE_RATE_LIMIT_BURST",
        "OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY",
        "OPENMED_SERVICE_THROTTLE_KEY",
        "OPENMED_SERVICE_CONCURRENCY_WAIT_SECONDS",
        "OPENMED_SERVICE_JOBS_STORE_PATH",
        "OPENMED_SERVICE_JOBS_TTL_SECONDS",
        "OPENMED_SERVICE_JOBS_WORKERS",
    ]
    for name in names:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("{broken", "invalid JSON"),
        ("[]", "JSON object"),
        ("{}", "jobs object"),
        ('{"jobs": []}', "jobs object"),
        ('{"jobs": {"synthetic": null}}', "invalid job record"),
    ],
)
def test_job_store_rejects_damaged_existing_file_without_overwriting_it(
    tmp_path,
    contents: str,
    message: str,
) -> None:
    from openmed.service import jobs

    path = tmp_path / "jobs.json"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        jobs.LocalJobStore(path)

    assert path.read_text(encoding="utf-8") == contents


def test_job_store_propagates_existing_file_read_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from pathlib import Path

    from openmed.service import jobs

    path = tmp_path / "jobs.json"
    path.write_text('{"jobs": {}}', encoding="utf-8")
    original = Path.read_text

    def fail_for_store(self: Path, *args: Any, **kwargs: Any) -> str:
        if self == path:
            raise OSError("synthetic unreadable store")
        return original(self, *args, **kwargs)

    with monkeypatch.context() as patched:
        patched.setattr(Path, "read_text", fail_for_store)
        with pytest.raises(OSError, match="synthetic unreadable store"):
            jobs.LocalJobStore(path)

    assert path.read_text(encoding="utf-8") == '{"jobs": {}}'


def test_missing_job_store_starts_empty(tmp_path) -> None:
    from openmed.service import jobs

    path = tmp_path / "jobs.json"
    store = jobs.LocalJobStore(path)
    assert store._records == {}
    assert not path.exists()


def _sample_deid_result(text: str, *, label: str = "NAME") -> DeidentificationResult:
    entity_text = "Maria Garcia" if "Maria Garcia" in text else "555-1212"
    start = text.index(entity_text)
    end = start + len(entity_text)
    return DeidentificationResult(
        original_text=text,
        deidentified_text=text.replace(entity_text, f"[{label}]"),
        pii_entities=[
            PIIEntity(
                text=entity_text,
                label=label,
                confidence=0.99,
                start=start,
                end=end,
                entity_type=label,
                redacted_text=f"[{label}]",
                original_text=entity_text,
            )
        ],
        method="mask",
        timestamp=datetime.now(),
    )


def test_invalid_result_span_fails_only_its_document(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from types import SimpleNamespace

    from openmed.service import jobs
    from openmed.service.schemas import DeidentifyJobDocument, DeidentifyJobRequest

    path = tmp_path / "jobs.json"
    store = jobs.LocalJobStore(path)
    queue = jobs.DeidentifyJobQueue(SimpleNamespace(), store=store)
    payload = DeidentifyJobRequest(
        documents=[
            DeidentifyJobDocument(id="synthetic-0", text="synthetic first"),
            DeidentifyJobDocument(id="synthetic-1", text="synthetic second"),
        ]
    )
    record = queue._new_record(payload)
    store.create(record)
    seen: list[str] = []

    def entity(start: int | str) -> SimpleNamespace:
        return SimpleNamespace(
            label="NAME", text="synthetic", start=start, end=4, confidence=0.9
        )

    def process(_payload: DeidentifyJobRequest, document: DeidentifyJobDocument):
        seen.append(document.id or "")
        spans = [entity(0), entity("invalid-offset")] if len(seen) == 1 else [entity(0)]
        return SimpleNamespace(pii_entities=spans)

    monkeypatch.setattr(queue, "_deidentify_document", process)
    try:
        queue._run_job(jobs._JobWorkItem(record["id"], payload))
    finally:
        queue.shutdown()

    final = store.get(record["id"])
    assert final is not None
    assert seen == ["synthetic-0", "synthetic-1"]
    assert final["status"] == "failed"
    assert final["processed_count"] == 1
    assert final["failed_count"] == 1
    assert final["progress_percent"] == 100.0
    assert final["label_histogram"] == {"NAME": 1}
    assert len(final["spans"]) == 1
    assert final["spans"][0]["document_id"] == "synthetic-1"
    assert final["error"]["type"] == "ValueError"
    assert "synthetic first" not in path.read_text(encoding="utf-8")


def test_terminal_record_keeps_its_full_metadata_ttl_after_completion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from openmed.service.schemas import DeidentifyJobDocument, DeidentifyJobRequest

    now = [datetime(2026, 1, 1, tzinfo=timezone.utc)]
    store = jobs.LocalJobStore(
        tmp_path / "jobs.json",
        ttl_seconds=10,
        clock=lambda: now[0],
    )
    queue = jobs.DeidentifyJobQueue(
        SimpleNamespace(),
        store=store,
        clock=lambda: now[0],
    )
    payload = DeidentifyJobRequest(
        documents=[DeidentifyJobDocument(id="synthetic-0", text="synthetic sample")]
    )
    record = queue._new_record(payload)
    store.create(record)

    def process(_payload: DeidentifyJobRequest, _document: DeidentifyJobDocument):
        now[0] += timedelta(seconds=11)
        return SimpleNamespace(pii_entities=[])

    monkeypatch.setattr(queue, "_deidentify_document", process)
    try:
        queue._run_job(jobs._JobWorkItem(record["id"], payload))
    finally:
        queue.shutdown()

    completed = store.get(record["id"])
    assert completed is not None
    assert completed["completed_at"] == "2026-01-01T00:00:11Z"
    assert completed["expires_at"] == "2026-01-01T00:00:21Z"
    assert completed["status"] == "done"


def _wait_for_job(
    client: TestClient,
    job_id: str,
    predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for _ in range(100):
        response = client.get(f"/jobs/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        if predicate(payload):
            return payload
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not reach expected state: {payload}")


def test_jobs_api_runs_batch_and_posts_no_phi_webhook(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    store_path = tmp_path / "jobs.json"
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.setenv("OPENMED_SERVICE_JOBS_STORE_PATH", str(store_path))
    monkeypatch.setenv("OPENMED_SERVICE_JOBS_WORKERS", "1")
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)

    def fake_deidentify(text: str, **kwargs: Any) -> DeidentificationResult:
        assert kwargs["loader"].loader.__class__ is FakeLoader
        time.sleep(0.05)
        return _sample_deid_result(text, label="NAME")

    monkeypatch.setattr(openmed, "deidentify", fake_deidentify)
    callbacks: list[dict[str, Any]] = []

    def webhook_sender(
        url: str,
        payload: dict[str, Any],
        *,
        secret: str,
        max_attempts: int,
        backoff_seconds: float,
    ) -> WebhookDeliveryResult:
        callbacks.append(
            {
                "url": url,
                "payload": payload,
                "secret": secret,
                "max_attempts": max_attempts,
                "backoff_seconds": backoff_seconds,
            }
        )
        return WebhookDeliveryResult(success=True, attempts=1, status_code=204)

    app = create_app()
    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        app.state.job_queue.webhook_sender = webhook_sender
        response = client.post(
            "/jobs",
            json={
                "documents": [
                    {"id": "doc-1", "text": "Paciente: Maria Garcia"},
                    {"id": "doc-2", "text": "Call Maria Garcia at 555-1212"},
                ],
                "webhook": {
                    "url": "https://callbacks.example.test/openmed",
                    "secret": "top-secret",
                    "max_attempts": 2,
                    "backoff_seconds": 0,
                },
            },
        )

        assert response.status_code == 202
        created = response.json()
        assert created["status"] == "queued"
        assert created["status_url"] == f"/jobs/{created['id']}"

        running = _wait_for_job(
            client,
            created["id"],
            lambda payload: payload["status"] in {"running", "done"},
        )
        assert running["status"] in {"running", "done"}

        done = _wait_for_job(
            client,
            created["id"],
            lambda payload: (
                payload["status"] == "done" and payload["webhook_delivery"] is not None
            ),
        )

    assert done["progress_percent"] == 100.0
    assert done["processed_count"] == 2
    assert done["failed_count"] == 0
    assert done["label_histogram"] == {"NAME": 2}
    assert done["webhook"]["url_hash"].startswith("sha256:")
    assert done["webhook_delivery"]["success"] is True
    assert callbacks[0]["payload"]["status"] == "done"
    assert callbacks[0]["payload"]["label_histogram"] == {"NAME": 2}

    callback_json = json.dumps(callbacks[0]["payload"], sort_keys=True)
    store_json = store_path.read_text(encoding="utf-8")
    for raw_phi in ("Maria Garcia", "555-1212", "Paciente"):
        assert raw_phi not in callback_json
        assert raw_phi not in store_json
    assert "top-secret" not in store_json
    assert "https://callbacks.example.test/openmed" not in store_json


def test_failed_job_surfaces_error_and_records_webhook_delivery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.setenv("OPENMED_SERVICE_JOBS_STORE_PATH", str(tmp_path / "jobs.json"))
    monkeypatch.setenv("OPENMED_SERVICE_JOBS_WORKERS", "1")
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)
    monkeypatch.setattr(
        openmed,
        "deidentify",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("model offline")),
    )
    callbacks: list[dict[str, Any]] = []

    def webhook_sender(
        url: str,
        payload: dict[str, Any],
        *,
        secret: str,
        max_attempts: int,
        backoff_seconds: float,
    ) -> WebhookDeliveryResult:
        del url, secret, max_attempts, backoff_seconds
        callbacks.append(payload)
        return WebhookDeliveryResult(success=True, attempts=2, status_code=200)

    app = create_app()
    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        app.state.job_queue.webhook_sender = webhook_sender
        response = client.post(
            "/jobs",
            json={
                "documents": [{"text": "Paciente: Maria Garcia"}],
                "webhook": {
                    "url": "https://callbacks.example.test/openmed",
                    "secret": "top-secret",
                    "max_attempts": 2,
                    "backoff_seconds": 0,
                },
            },
        )
        job_id = response.json()["id"]
        failed = _wait_for_job(
            client,
            job_id,
            lambda payload: (
                payload["status"] == "failed"
                and payload["webhook_delivery"] is not None
            ),
        )

    assert failed["failed_count"] == 1
    assert failed["processed_count"] == 0
    assert failed["error"] == {
        "type": "RuntimeError",
        "message": "Document failed during de-identification",
    }
    assert failed["webhook_delivery"]["attempts"] == 2
    assert callbacks[0]["status"] == "failed"
    assert callbacks[0]["error"]["type"] == "RuntimeError"


def test_missing_job_returns_standard_404(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.setenv("OPENMED_SERVICE_JOBS_STORE_PATH", str(tmp_path / "jobs.json"))
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)
    app = create_app()

    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        response = client.get("/jobs/missing")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "bad_request"


@pytest.mark.parametrize("operation", ["create", "update", "cleanup"])
@pytest.mark.parametrize("failure_point", ["serialize", "replace"])
def test_job_store_failed_publication_preserves_memory_and_disk(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    operation: str,
    failure_point: str,
) -> None:
    now = [datetime(2026, 1, 1, tzinfo=timezone.utc)]
    path = tmp_path / "jobs.json"
    store = jobs.LocalJobStore(path, clock=lambda: now[0])
    store.create(
        {
            "id": "synthetic-existing",
            "status": "done",
            "expires_at": "2026-01-01T00:00:10Z",
        }
    )
    original_records = json.loads(path.read_text(encoding="utf-8"))["jobs"]
    original_bytes = path.read_bytes()
    if operation != "update":
        now[0] += timedelta(seconds=11)

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("synthetic publication failure")

    with monkeypatch.context() as patched:
        if failure_point == "serialize":
            patched.setattr(jobs.json, "dump", fail)
        else:
            patched.setattr(jobs.os, "replace", fail)
        with pytest.raises(OSError, match="synthetic publication failure"):
            if operation == "create":
                store.create({"id": "synthetic-new", "status": "queued"})
            elif operation == "update":
                store.update("synthetic-existing", status="running")
            else:
                store.cleanup_expired()

    assert store._records == original_records
    assert path.read_bytes() == original_bytes
    assert list(tmp_path.iterdir()) == [path]


class _StoreWriteSpy:
    """Record every published job-store payload without changing behavior."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.payloads: list[bytes] = []
        self._replace = jobs.os.replace
        monkeypatch.setattr(jobs.os, "replace", self)

    def __call__(self, src: Any, dst: Any) -> None:
        self.payloads.append(Path(src).read_bytes())
        self._replace(src, dst)

    @property
    def count(self) -> int:
        return len(self.payloads)

    @property
    def bytes_written(self) -> int:
        return sum(len(payload) for payload in self.payloads)

    def job_snapshots(self, job_id: str) -> list[dict[str, Any]]:
        return [json.loads(payload)["jobs"][job_id] for payload in self.payloads]


def _synthetic_job_payload(document_count: int) -> Any:
    from openmed.service.schemas import DeidentifyJobDocument, DeidentifyJobRequest

    return DeidentifyJobRequest(
        documents=[
            DeidentifyJobDocument(
                id=f"synthetic-{index}",
                text=f"Synthetic patient Alex Rivera visit {index}",
            )
            for index in range(document_count)
        ]
    )


def _run_synthetic_job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    document_count: int,
    *,
    seconds_per_document: float = 1.0,
    failing_documents: frozenset[int] = frozenset(),
    **queue_kwargs: Any,
) -> tuple[_StoreWriteSpy, jobs.LocalJobStore, str]:
    """Run one fake-runtime job and spy on every store publication."""
    from types import SimpleNamespace

    from openmed.service.schemas import DeidentifyJobDocument, DeidentifyJobRequest

    now = [datetime(2026, 1, 1, tzinfo=timezone.utc)]
    ticks = [0.0]
    store = jobs.LocalJobStore(tmp_path / "jobs.json", clock=lambda: now[0])
    queue = jobs.DeidentifyJobQueue(
        SimpleNamespace(),
        store=store,
        clock=lambda: now[0],
        monotonic=lambda: ticks[0],
        **queue_kwargs,
    )
    payload = _synthetic_job_payload(document_count)
    record = queue._new_record(payload)
    calls = [0]

    def process(_payload: DeidentifyJobRequest, document: DeidentifyJobDocument):
        index = calls[0]
        calls[0] += 1
        ticks[0] += seconds_per_document
        now[0] += timedelta(seconds=seconds_per_document)
        if index in failing_documents:
            raise RuntimeError("synthetic model failure")
        start = document.text.index("Alex Rivera")
        return SimpleNamespace(
            pii_entities=[
                SimpleNamespace(
                    label="NAME",
                    text="Alex Rivera",
                    start=start,
                    end=start + len("Alex Rivera"),
                    confidence=0.9,
                )
            ]
        )

    monkeypatch.setattr(queue, "_deidentify_document", process)
    spy = _StoreWriteSpy(monkeypatch)
    store.create(record)
    try:
        queue._run_job(jobs._JobWorkItem(record["id"], payload))
    finally:
        queue.shutdown()
    return spy, store, record["id"]


def _assert_consistent_progress(snapshot: dict[str, Any]) -> None:
    attempted = snapshot["processed_count"] + snapshot["failed_count"]
    assert 0 <= attempted <= snapshot["document_count"]
    assert len(snapshot["spans"]) == snapshot["processed_count"]
    assert sum(snapshot["label_histogram"].values()) == snapshot["processed_count"]
    if snapshot["status"] == "running" and snapshot["document_count"]:
        expected = round(attempted / snapshot["document_count"] * 100, 2)
        assert snapshot["progress_percent"] == expected
    span_ids = [span["document_id"] for span in snapshot["spans"]]
    assert span_ids == sorted(span_ids, key=lambda value: int(value.split("-")[1]))


@pytest.mark.parametrize("document_count", [40, 160, 640])
def test_job_store_writes_stay_under_ceiling_independent_of_document_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    document_count: int,
) -> None:
    spy, store, job_id = _run_synthetic_job(
        monkeypatch,
        tmp_path,
        document_count,
        progress_interval_seconds=0.5,
        max_progress_writes=8,
    )

    # create + running + at most 8 progress checkpoints + terminal.
    assert spy.count <= 11
    final = store.get(job_id)
    assert final is not None
    assert final["status"] == "done"
    assert final["processed_count"] == document_count
    assert final["progress_percent"] == 100.0
    assert len(final["spans"]) == document_count


def test_job_store_bytes_written_grow_linearly_with_document_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    measured: dict[int, int] = {}
    for document_count in (100, 400):
        run_path = tmp_path / str(document_count)
        run_path.mkdir()
        with monkeypatch.context() as patched:
            spy, _store, _job_id = _run_synthetic_job(
                patched,
                run_path,
                document_count,
                progress_interval_seconds=0.5,
                max_progress_writes=8,
            )
        measured[document_count] = spy.bytes_written

    # Four times the documents must cost about four times the bytes; the
    # per-document full rewrite cost about sixteen times as much.
    assert measured[400] < measured[100] * 5


def test_job_progress_writes_coalesce_on_injected_clock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spy, store, job_id = _run_synthetic_job(
        monkeypatch,
        tmp_path,
        20,
        seconds_per_document=0.25,
        progress_interval_seconds=1.0,
        max_progress_writes=100,
    )

    snapshots = spy.job_snapshots(job_id)
    statuses = [snapshot["status"] for snapshot in snapshots]
    progress = [snapshot for snapshot in snapshots if snapshot["status"] == "running"]
    # One checkpoint per elapsed second of synthetic time, not one per document.
    assert statuses[0] == "queued"
    assert statuses[-1] == "done"
    assert [snapshot["processed_count"] for snapshot in progress] == [0, 4, 8, 12, 16]
    for snapshot in snapshots:
        _assert_consistent_progress(snapshot)
    persisted = json.loads(store.path.read_text(encoding="utf-8"))["jobs"]
    assert persisted == store._records


def test_job_progress_writes_skip_checkpoints_before_interval_elapses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spy, store, job_id = _run_synthetic_job(
        monkeypatch,
        tmp_path,
        30,
        seconds_per_document=0.0,
        progress_interval_seconds=1.0,
        max_progress_writes=100,
    )

    statuses = [snapshot["status"] for snapshot in spy.job_snapshots(job_id)]
    assert statuses == ["queued", "running", "done"]
    final = store.get(job_id)
    assert final is not None
    assert final["processed_count"] == 30
    assert len(final["spans"]) == 30


def test_job_progress_without_coalescing_writes_after_each_document(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Negative control: disabling both limits restores one write per document.
    spy, _store, job_id = _run_synthetic_job(
        monkeypatch,
        tmp_path,
        12,
        progress_interval_seconds=0.0,
        max_progress_writes=12,
    )

    progress = [
        snapshot["processed_count"]
        for snapshot in spy.job_snapshots(job_id)
        if snapshot["status"] == "running"
    ]
    assert progress == list(range(12))
    assert spy.count == 14


def test_coalesced_progress_keeps_counts_and_never_persists_raw_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spy, store, job_id = _run_synthetic_job(
        monkeypatch,
        tmp_path,
        50,
        failing_documents=frozenset({3, 17, 49}),
        progress_interval_seconds=0.5,
        max_progress_writes=5,
    )

    snapshots = spy.job_snapshots(job_id)
    assert len(snapshots) <= 8
    for snapshot in snapshots:
        _assert_consistent_progress(snapshot)
    final = store.get(job_id)
    assert final is not None
    assert final["status"] == "failed"
    assert final["processed_count"] == 47
    assert final["failed_count"] == 3
    assert final["label_histogram"] == {"NAME": 47}
    assert final["error"]["type"] == "RuntimeError"
    for payload in spy.payloads:
        text = payload.decode("utf-8")
        assert "Alex Rivera" not in text
        assert "Synthetic patient" not in text
        assert "synthetic model failure" not in text


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("progress_interval_seconds", -0.1),
        ("max_progress_writes", 0),
    ],
)
def test_job_queue_rejects_invalid_progress_coalescing_limits(
    tmp_path: Path,
    argument: str,
    value: float,
) -> None:
    from types import SimpleNamespace

    store = jobs.LocalJobStore(tmp_path / "jobs.json")
    with pytest.raises(ValueError, match=argument):
        jobs.DeidentifyJobQueue(SimpleNamespace(), store=store, **{argument: value})
    assert not (tmp_path / "jobs.json").exists()


def test_failed_progress_write_leaves_previous_store_without_temp_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from openmed.service.schemas import DeidentifyJobDocument, DeidentifyJobRequest

    ticks = [0.0]
    path = tmp_path / "jobs.json"
    store = jobs.LocalJobStore(path)
    queue = jobs.DeidentifyJobQueue(
        SimpleNamespace(),
        store=store,
        monotonic=lambda: ticks[0],
        progress_interval_seconds=1.0,
    )
    payload = _synthetic_job_payload(6)
    record = queue._new_record(payload)
    store.create(record)
    calls = [0]

    def process(_payload: DeidentifyJobRequest, _document: DeidentifyJobDocument):
        calls[0] += 1
        ticks[0] += 1.0
        return SimpleNamespace(pii_entities=[])

    original_replace = jobs.os.replace
    replacements = [0]

    def fail_second_checkpoint(src: Any, dst: Any) -> None:
        replacements[0] += 1
        if replacements[0] == 3:
            raise OSError("synthetic disk full")
        original_replace(src, dst)

    monkeypatch.setattr(queue, "_deidentify_document", process)
    monkeypatch.setattr(jobs.os, "replace", fail_second_checkpoint)
    try:
        with pytest.raises(OSError, match="synthetic disk full"):
            queue._run_job(jobs._JobWorkItem(record["id"], payload))
    finally:
        queue.shutdown()

    persisted = json.loads(path.read_text(encoding="utf-8"))["jobs"]
    assert persisted == store._records
    assert persisted[record["id"]]["status"] == "running"
    assert persisted[record["id"]]["processed_count"] == 1
    assert calls[0] == 2
    assert list(tmp_path.iterdir()) == [path]


def test_job_store_rollback_restores_only_the_changed_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "jobs.json"
    store = jobs.LocalJobStore(path)
    store.create({"id": "synthetic-a", "status": "queued", "spans": []})
    store.create({"id": "synthetic-b", "status": "queued", "spans": []})
    untouched = store._records["synthetic-b"]
    original_bytes = path.read_bytes()

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("synthetic publication failure")

    with monkeypatch.context() as patched:
        patched.setattr(jobs.os, "replace", fail)
        with pytest.raises(OSError, match="synthetic publication failure"):
            store.update("synthetic-a", status="running", spans=[{"start": 0}])

    assert store._records["synthetic-a"] == {
        "id": "synthetic-a",
        "status": "queued",
        "spans": [],
    }
    # Rollback must not snapshot and restore unrelated records.
    assert store._records["synthetic-b"] is untouched
    assert path.read_bytes() == original_bytes
    assert list(tmp_path.iterdir()) == [path]
