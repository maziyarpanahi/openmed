"""Conformance suite shared by the local, Ray, and Spark shard executors.

Every backend is driven through the same :class:`ShardExecutor` protocol and
must produce the same observable outcome. Ray and PySpark are optional and are
absent from the default test environment, so their adapters run here against
the fakes in :mod:`tests.unit.processing.fixtures.distributed_backend_fakes`.

What this suite proves offline, for all three backends: protocol conformance,
shard-to-result attribution under out-of-order completion, PHI-free failure
reporting, empty-shard settlement, digest preservation across a resume, and
that no driver-owned state is submitted to a worker.

What it cannot prove, because it needs a live cluster: real ``cloudpickle``
serialization, per-worker model loading, atomic rename on a shared filesystem,
executor loss and re-execution, and Ray object-store spilling. The real-backend
tests at the end of this module are **skipped** unless the optional dependency
is installed; they are not evidence of a verified cluster run.

All fixture data is synthetic and generated algorithmically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from openmed.processing import (
    DriverOnlyStateError,
    InMemoryRunManifestStore,
    LocalShardExecutor,
    RayShardExecutor,
    ShardStatus,
    SparkShardExecutor,
    build_run_manifest,
    plan_document_shards,
    run_shard_plan,
)
from openmed.processing.distributed import DocumentShard
from openmed.processing.shard_executor import ShardTask, shard_output_filename
from tests.unit.processing.fixtures.distributed_backend_fakes import (
    FakeRayModule,
    FakeSparkSession,
)

# ---------------------------------------------------------------------------
# Synthetic corpus and handlers
# ---------------------------------------------------------------------------


def synthetic_documents(count: int) -> list[dict[str, str]]:
    """Return *count* algorithmically generated, PHI-free documents."""

    return [
        {"id": f"doc-{index:05d}", "text": f"synthetic clinical note {index}"}
        for index in range(count)
    ]


def deterministic_handler(shard: DocumentShard) -> bytes:
    """Return stable bytes derived only from the shard's own identity."""

    payload = {
        "shard_id": shard.shard_id,
        "document_count": shard.document_count,
        "fingerprint": shard.fingerprint,
    }
    return (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")


def failing_handler(shard: DocumentShard) -> bytes:
    """Fail on shard 1 with a message that quotes document identifiers."""

    if shard.shard_id == 1:
        raise ValueError(f"synthetic failure touching {shard.document_hashes!r}")
    return deterministic_handler(shard)


def drifting_handler(shard: DocumentShard) -> bytes:
    """Return different bytes than :func:`deterministic_handler` for one shard.

    Module level on purpose. ``ShardTask`` requires a picklable handler, and
    the gate is stdlib pickle on two of the three backends:
    ``ProcessPoolExecutor`` uses it, and PySpark ships ``parallelize`` elements
    with ``CPickleSerializer``, which is stdlib pickle too. Only Ray is more
    permissive, serializing with ``cloudpickle``. So a closure here would break
    the local process pool and Spark alike; the fakes enforce the same stdlib
    constraint, which keeps every handler in this suite usable on all three.
    """

    return deterministic_handler(shard) + b"drift\n"


# ---------------------------------------------------------------------------
# Backend matrix
# ---------------------------------------------------------------------------


def make_local() -> Any:
    return LocalShardExecutor()


def make_ray() -> Any:
    # completion_order is set per-test where ordering matters.
    return RayShardExecutor(auto_init=False, ray_module=FakeRayModule())


def make_spark() -> Any:
    return SparkShardExecutor(session=FakeSparkSession())


BACKENDS = {"local": make_local, "ray": make_ray, "spark": make_spark}


@pytest.fixture(params=sorted(BACKENDS))
def backend(request):
    """Yield one executor factory per backend under test."""

    return BACKENDS[request.param]


def run(tmp_path: Path, executor: Any, *, documents=None, handler=None, shards=3):
    """Plan, execute, and return (result, manifest, store) for one run."""

    docs = synthetic_documents(9) if documents is None else documents
    plan = plan_document_shards(docs, shard_count=shards)
    manifest = build_run_manifest(run_id="run-0001", plan=plan)
    store = InMemoryRunManifestStore(manifest)
    result = run_shard_plan(
        plan,
        handler or deterministic_handler,
        manifest=manifest,
        root=tmp_path,
        store=store,
        executor=executor,
    )
    return result, plan, store


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_every_backend_satisfies_the_executor_protocol(backend):
    executor = backend()

    assert hasattr(executor, "execute")
    assert callable(executor.execute)


def test_empty_task_sequence_yields_nothing(backend):
    assert list(backend().execute([])) == []


def test_all_backends_complete_every_shard(tmp_path, backend):
    result, plan, _ = run(tmp_path, backend())

    assert result.is_complete
    assert sorted(result.completed_shards) == [0, 1, 2]
    assert result.failed_shards == ()
    for shard in plan.shards:
        assert (tmp_path / shard_output_filename(shard.shard_id)).exists()


def test_backends_agree_on_digests(tmp_path):
    """The observable output must not depend on which backend produced it."""

    digests = {}
    for name, factory in sorted(BACKENDS.items()):
        root = tmp_path / name
        root.mkdir()
        result, _, _ = run(root, factory())
        digests[name] = {
            execution.shard_id: execution.output_digest
            for execution in result.executions
        }

    assert digests["local"] == digests["ray"] == digests["spark"]


def test_shard_results_are_attributed_by_shard_id_not_arrival_order(tmp_path):
    """Ray completing out of order must not misattribute a result."""

    executor = RayShardExecutor(
        auto_init=False, ray_module=FakeRayModule(completion_order=[2, 0, 1])
    )
    result, _, _ = run(tmp_path, executor)

    by_id = {e.shard_id: e for e in result.executions}
    for shard_id, execution in by_id.items():
        assert execution.shard_id == shard_id
        assert execution.status == ShardStatus.COMPLETED
    assert sorted(by_id) == [0, 1, 2]


# ---------------------------------------------------------------------------
# PHI-free failure reporting
# ---------------------------------------------------------------------------


def test_handler_failure_reports_type_only_and_never_document_text(tmp_path, backend):
    result, plan, _ = run(tmp_path, backend(), handler=failing_handler)

    failed = [e for e in result.executions if not e.succeeded]
    assert [e.shard_id for e in failed] == [1]
    assert failed[0].error_type == "ValueError"
    serialized = json.dumps(result.to_dict())
    for shard in plan.shards:
        for document_hash in shard.document_hashes:
            assert document_hash not in serialized


def test_failure_of_one_shard_leaves_the_others_completed(tmp_path, backend):
    result, _, _ = run(tmp_path, backend(), handler=failing_handler)

    assert sorted(result.completed_shards) == [0, 2]
    assert result.failed_shards == (1,)
    assert not result.is_complete


# ---------------------------------------------------------------------------
# Empty shards and the degenerate single-shard case
# ---------------------------------------------------------------------------


def test_empty_shards_settle_as_completed(tmp_path, backend):
    """A shard with no documents bypasses the handler and still publishes."""

    result, plan, _ = run(
        tmp_path, backend(), documents=synthetic_documents(2), shards=5
    )

    empty = [shard.shard_id for shard in plan.shards if shard.is_empty]
    assert empty, "fixture must produce at least one empty shard"
    assert result.is_complete
    for shard_id in empty:
        execution = next(e for e in result.executions if e.shard_id == shard_id)
        assert execution.status == ShardStatus.COMPLETED
        assert execution.output_digest is not None
        assert (tmp_path / shard_output_filename(shard_id)).exists()


def test_single_shard_plan_completes(tmp_path, backend):
    result, _, _ = run(tmp_path, backend(), shards=1)

    assert result.is_complete
    assert result.completed_shards == (0,)


# ---------------------------------------------------------------------------
# Determinism guard across resume
# ---------------------------------------------------------------------------


def test_expected_digest_is_carried_to_the_worker_on_rerun(tmp_path, backend):
    """A re-executed shard must still be compared against its earlier digest.

    Clearing ``output_digest`` on requeue would silently disable the
    non-determinism guard for exactly the shards a resume recomputes.
    """

    docs = synthetic_documents(9)
    plan = plan_document_shards(docs, shard_count=3)
    manifest = build_run_manifest(run_id="run-0001", plan=plan)
    store = InMemoryRunManifestStore(manifest)
    first = run_shard_plan(
        plan,
        deterministic_handler,
        manifest=manifest,
        root=tmp_path,
        store=store,
        executor=backend(),
    )
    assert first.is_complete

    seeded = store.load()
    for shard_id in (0, 1, 2):
        assert seeded.shard(shard_id).output_digest is not None

    second = run_shard_plan(
        plan,
        deterministic_handler,
        manifest=seeded,
        root=tmp_path,
        store=store,
        executor=backend(),
        force=True,
    )

    assert second.is_complete
    assert [e.digest_mismatch for e in second.executions] == [False] * 3


def test_nondeterministic_rerun_is_reported_and_output_preserved(tmp_path, backend):
    """A shard that stops reproducing its bytes must not destroy good output."""

    from openmed.processing import ShardOutputDigestMismatchError

    docs = synthetic_documents(9)
    plan = plan_document_shards(docs, shard_count=3)
    manifest = build_run_manifest(run_id="run-0001", plan=plan)
    store = InMemoryRunManifestStore(manifest)
    run_shard_plan(
        plan,
        deterministic_handler,
        manifest=manifest,
        root=tmp_path,
        store=store,
        executor=backend(),
    )
    before = {
        shard.shard_id: (tmp_path / shard_output_filename(shard.shard_id)).read_bytes()
        for shard in plan.shards
    }

    with pytest.raises(ShardOutputDigestMismatchError):
        run_shard_plan(
            plan,
            drifting_handler,
            manifest=store.load(),
            root=tmp_path,
            store=store,
            executor=backend(),
            force=True,
        )

    for shard_id, payload in before.items():
        on_disk = (tmp_path / shard_output_filename(shard_id)).read_bytes()
        assert on_disk == payload


# ---------------------------------------------------------------------------
# Driver-only state must never be submitted
# ---------------------------------------------------------------------------


def _store_capturing_handler(store):
    def handler(shard: DocumentShard, _store=store) -> bytes:
        return deterministic_handler(shard)

    return handler


def test_backends_reject_a_handler_that_captures_the_manifest_store(tmp_path, backend):
    docs = synthetic_documents(9)
    plan = plan_document_shards(docs, shard_count=3)
    manifest = build_run_manifest(run_id="run-0001", plan=plan)
    store = InMemoryRunManifestStore(manifest)

    with pytest.raises(DriverOnlyStateError):
        run_shard_plan(
            plan,
            _store_capturing_handler(store),
            manifest=manifest,
            root=tmp_path,
            store=store,
            executor=backend(),
        )


def test_rejection_happens_before_any_task_is_submitted(tmp_path):
    """Ray must reject the batch without submitting a single remote task."""

    ray_module = FakeRayModule()
    executor = RayShardExecutor(auto_init=False, ray_module=ray_module)
    shard = DocumentShard(
        shard_id=0, document_ids=("d",), document_hashes=("h",), fingerprint="f"
    )
    store = InMemoryRunManifestStore()
    task = ShardTask(
        shard=shard,
        root=tmp_path,
        relative_path=shard_output_filename(0),
        handler=_store_capturing_handler(store),
    )

    with pytest.raises(DriverOnlyStateError):
        list(executor.execute([task]))

    assert ray_module.outcomes == []
    assert ray_module.submitted_payloads == []


# ---------------------------------------------------------------------------
# Real-backend tests -- SKIPPED unless the optional dependency is installed.
# These are not evidence of a verified cluster run.
# ---------------------------------------------------------------------------


def test_ray_backend_completes_every_shard_on_a_real_runtime(tmp_path):
    pytest.importorskip(
        "ray",
        reason="real-backend Ray test requires openmed[ray]; "
        "run: uv sync --extra dev --extra ray",
    )
    import ray as ray_module

    ray_module.init(local_mode=True, ignore_reinit_error=True, num_cpus=2)
    try:
        result, _, _ = run(tmp_path, RayShardExecutor(auto_init=False))
        assert result.is_complete
        assert sorted(result.completed_shards) == [0, 1, 2]
    finally:
        ray_module.shutdown()


def test_spark_backend_completes_every_shard_on_a_real_session(tmp_path):
    pytest.importorskip(
        "pyspark",
        reason="real-backend Spark test requires openmed[spark]; "
        "run: uv sync --extra dev --extra spark",
    )
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[2]")
        .appName("openmed-shard-conformance")
        .getOrCreate()
    )
    try:
        result, _, _ = run(tmp_path, SparkShardExecutor(session=session))
        assert result.is_complete
        assert sorted(result.completed_shards) == [0, 1, 2]
    finally:
        session.stop()


# ---------------------------------------------------------------------------
# Duplicate publication (Ray retries, Spark speculation)
# ---------------------------------------------------------------------------


class _RecordingStore:
    """Captures every manifest state the driver persists."""

    def __init__(self, manifest):
        self._manifest = manifest
        self.history: list[dict[int, tuple[str, int]]] = []

    def load(self):
        return self._manifest

    def save(self, manifest) -> None:
        self._manifest = manifest
        self.history.append(
            {s.shard_id: (s.status.value, s.attempts) for s in manifest.shards}
        )


def _losing_writer(monkeypatch, losses: int):
    """Fail the first *losses* publishes per path with ``PermissionError``.

    Windows refuses the atomic replace with a sharing violation while another
    handle holds the destination, which is exactly what a Ray retry or a Spark
    speculative duplicate produces. POSIX never raises here, so the condition
    has to be injected to be exercised at all on this platform.
    """

    import openmed.processing.shard_executor as module

    real = module._atomic_write_bytes
    seen: dict[str, int] = {}
    # The publish backs off between retries. Real sleeps would add seconds here
    # for no extra coverage, so only the retry *count* is exercised.
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    def flaky(path, payload, hook=None):
        count = seen.get(str(path), 0)
        seen[str(path)] = count + 1
        if count < losses:
            raise PermissionError(32, "The process cannot access the file")
        return real(path, payload, hook=hook)

    monkeypatch.setattr(module, "_atomic_write_bytes", flaky)
    return seen


@pytest.mark.parametrize("losses", [0, 1, 3, 11])
def test_duplicate_publication_records_one_attempt_and_one_status(
    tmp_path, backend, monkeypatch, losses
):
    """A retried publish is still one attempt, and never persists a FAILED.

    The adapters make duplicate execution likely -- Ray retries on worker death
    and Spark may run speculative duplicates -- so the guarantee has to hold on
    every backend, not just the local one.
    """

    _losing_writer(monkeypatch, losses)
    plan = plan_document_shards(synthetic_documents(9), shard_count=3)
    manifest = build_run_manifest(run_id="run-0001", plan=plan)
    store = _RecordingStore(manifest)

    result = run_shard_plan(
        plan,
        deterministic_handler,
        manifest=manifest,
        root=tmp_path,
        store=store,
        executor=backend(),
    )

    assert result.is_complete
    final = store.history[-1]
    assert {attempts for _, attempts in final.values()} == {1}
    assert {status for status, _ in final.values()} == {"completed"}
    assert not any(
        status == "failed"
        for snapshot in store.history
        for status, _ in snapshot.values()
    )


def test_publication_retry_is_bounded_and_surfaces_a_real_failure(
    tmp_path, backend, monkeypatch
):
    """A permanently unwritable destination must not be retried forever."""

    import openmed.processing.shard_executor as module

    def always_fail(path, payload, hook=None):
        raise PermissionError(32, "The process cannot access the file")

    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(module, "_atomic_write_bytes", always_fail)
    plan = plan_document_shards(synthetic_documents(9), shard_count=3)
    manifest = build_run_manifest(run_id="run-0001", plan=plan)

    result = run_shard_plan(
        plan,
        deterministic_handler,
        manifest=manifest,
        root=tmp_path,
        executor=backend(),
    )

    assert not result.is_complete
    assert sorted(result.failed_shards) == [0, 1, 2]
    assert {e.error_type for e in result.executions} == {"PermissionError"}
