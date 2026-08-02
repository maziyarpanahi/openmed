"""Adapter-specific contracts for the Ray and Spark shard executors.

The shared behaviour lives in
:mod:`tests.unit.processing.test_distributed_executor_conformance`; this module
covers what is specific to each backend -- lazy imports, remote-error naming,
submission bounding, and the manifest constraints a worker's metadata must
satisfy. All fixture data is synthetic.
"""

from __future__ import annotations

import sys

import pytest

from openmed.processing import RayShardExecutor, ShardStatus, SparkShardExecutor
from openmed.processing.ray_executor import _remote_error_type
from openmed.processing.run_manifest import ShardRecord
from openmed.processing.shard_executor import (
    ShardExecutionError,
    ShardTask,
    _safe_error_type,
)
from openmed.processing.spark_executor import _load_spark_session
from tests.unit.processing.fixtures.distributed_backend_fakes import (
    FakeRayModule,
    FakeSparkSession,
    make_ray_task_error,
)

# ---------------------------------------------------------------------------
# Import isolation
# ---------------------------------------------------------------------------


def test_importing_processing_imports_neither_backend():
    """The headline acceptance criterion for this change."""

    assert "openmed.processing" in sys.modules
    for module in ("ray", "pyspark", "py4j"):
        assert module not in sys.modules


def test_ray_executor_raises_an_actionable_error_without_ray(monkeypatch):
    import openmed.processing.ray_executor as adapter

    def missing(name: str):
        raise ImportError(name)

    monkeypatch.setattr(adapter, "_import_module", missing)

    with pytest.raises(ImportError, match=r"openmed\[ray\]"):
        RayShardExecutor().ensure_available()


def test_spark_executor_raises_an_actionable_error_without_pyspark(monkeypatch):
    import openmed.processing.spark_executor as adapter

    def missing(name: str):
        raise ImportError(name)

    monkeypatch.setattr(adapter, "_import_module", missing)

    with pytest.raises(ImportError, match=r"openmed\[spark\]"):
        _load_spark_session()


def test_missing_backend_raises_from_execute_not_from_the_generator(monkeypatch):
    """The failure lands at a fixed point, with nothing submitted.

    This does **not** protect the manifest: ``run_shard_plan`` has already
    burned an attempt per shard before it calls ``execute``. See
    ``test_missing_backend_still_costs_a_manifest_attempt`` for that, and
    ``ensure_available`` for the part that does prevent it.
    """

    import openmed.processing.ray_executor as adapter

    monkeypatch.setattr(
        adapter, "_import_module", lambda name: (_ for _ in ()).throw(ImportError(name))
    )
    executor = RayShardExecutor()
    task = object()

    with pytest.raises(ImportError):
        executor.execute([task])  # not iterated


# ---------------------------------------------------------------------------
# Remote error naming
# ---------------------------------------------------------------------------


def test_ray_task_error_is_unwrapped_via_the_cause_attribute():
    """Ray stores the original in a ``cause`` attribute, not ``__cause__``.

    Built with :func:`make_ray_task_error`, which reproduces
    ``RayTaskError.as_instanceof_cause`` from ray 2.56.1 rather than
    hand-assigning ``__cause__``. An earlier version of this test assigned
    ``__cause__`` directly and passed against an adapter that could not unwrap
    a single real Ray failure.

    NOTE: the class *name* ``RayTaskError(ValueError)`` remains a synthetic
    reconstruction, since Ray is not installed here. The unwrapping mechanism
    is modelled on Ray's source; the exact rendered name is not verified.
    """

    exc = make_ray_task_error(ValueError("synthetic remote failure"))

    assert exc.__cause__ is None
    assert exc.__context__ is None
    assert _safe_error_type(exc) == "UnknownError"
    assert _remote_error_type(exc) == "ValueError"


def test_unwrapped_error_type_is_accepted_by_the_manifest():
    exc = make_ray_task_error(ValueError("boom"))

    record = ShardRecord(
        shard_id=0,
        fingerprint="a" * 64,
        document_count=1,
        status=ShardStatus.FAILED,
        error_type=_remote_error_type(exc),
    )

    assert record.error_type == "ValueError"


def test_ordinary_chaining_is_still_unwrapped():
    """Wrappers that do chain normally must keep working."""

    exc = type("Weird Wrapper", (RuntimeError,), {})("x")
    exc.__cause__ = OSError("disk")

    assert _remote_error_type(exc) == "OSError"


def test_unwrapping_survives_a_raising_cause_property():
    """This runs while a shard failure is being recorded; it must not crash."""

    class _Hostile(RuntimeError):
        @property
        def cause(self):
            raise RuntimeError("hostile accessor")

    _Hostile.__name__ = "RayTaskError(ValueError)"

    assert _remote_error_type(_Hostile("x")) == "UnknownError"


def test_unnameable_remote_error_falls_back_to_unknown_error():
    weird = type("Ray Task Error", (RuntimeError,), {})

    assert _remote_error_type(weird("x")) == "UnknownError"


@pytest.mark.parametrize("depth", [1, 5, 50, 300])
def test_unwrapping_survives_a_recomputed_cause_chain(depth):
    """Identity memoization must not skip a node whose address was reissued.

    ``seen`` memoizes by ``id()``, which is unique only among *live* objects. A
    wrapper whose ``cause`` is a computed property returns a fresh exception per
    read, so each hop is freed as the walk moves on and CPython reissues the
    address; without holding visited nodes alive the walk stops early and
    reports ``UnknownError`` for a cause that was reachable. Deterministically
    reproduced from depth 5 before the fix. Same class of bug as the memo
    poisoning fixed in ``find_driver_only_state``.
    """

    class _Hop(RuntimeError):
        def __init__(self, remaining: int) -> None:
            super().__init__("wrapped")
            self.remaining = remaining

        @property
        def cause(self):
            if self.remaining <= 0:
                return ValueError("real underlying error")
            return _Hop(self.remaining - 1)

    _Hop.__name__ = "RayTaskError(ValueError)"

    assert _remote_error_type(_Hop(depth)) == "ValueError"


def test_remote_error_unwrapping_terminates_on_a_cause_cycle():
    first = type("A B", (RuntimeError,), {})("x")
    second = type("C D", (RuntimeError,), {})("y")
    first.__cause__ = second
    second.__cause__ = first

    assert _remote_error_type(first) == "UnknownError"


# ---------------------------------------------------------------------------
# Submission shape
# ---------------------------------------------------------------------------


def test_ray_bounds_the_number_of_in_flight_tasks(tmp_path):
    from openmed.processing import (
        build_run_manifest,
        plan_document_shards,
        run_shard_plan,
    )
    from tests.unit.processing.test_distributed_executor_conformance import (
        deterministic_handler,
        synthetic_documents,
    )

    ray_module = FakeRayModule()
    executor = RayShardExecutor(auto_init=False, ray_module=ray_module, max_in_flight=2)
    plan = plan_document_shards(synthetic_documents(12), shard_count=4)
    manifest = build_run_manifest(run_id="run-0001", plan=plan)

    result = run_shard_plan(
        plan, deterministic_handler, manifest=manifest, root=tmp_path, executor=executor
    )

    assert result.is_complete
    assert len(ray_module.submitted_payloads) == 4


def test_spark_parallelizes_one_slice_per_shard_by_default(tmp_path):
    from openmed.processing import (
        build_run_manifest,
        plan_document_shards,
        run_shard_plan,
    )
    from tests.unit.processing.test_distributed_executor_conformance import (
        deterministic_handler,
        synthetic_documents,
    )

    session = FakeSparkSession()
    executor = SparkShardExecutor(session=session)
    plan = plan_document_shards(synthetic_documents(12), shard_count=4)
    manifest = build_run_manifest(run_id="run-0001", plan=plan)

    run_shard_plan(
        plan, deterministic_handler, manifest=manifest, root=tmp_path, executor=executor
    )

    assert session.sparkContext.num_slices == 4


@pytest.mark.parametrize(
    "kwargs",
    [{"max_in_flight": 0}, {"max_in_flight": -1}, {"max_in_flight": True}],
)
def test_ray_rejects_invalid_in_flight_bounds(kwargs):
    with pytest.raises(ShardExecutionError):
        RayShardExecutor(**kwargs)


@pytest.mark.parametrize("kwargs", [{"num_slices": 0}, {"num_slices": True}])
def test_spark_rejects_invalid_slice_counts(kwargs):
    with pytest.raises(ShardExecutionError):
        SparkShardExecutor(**kwargs)


def test_spark_requires_a_session_when_none_is_active(monkeypatch):
    import openmed.processing.spark_executor as adapter

    class _NoActive:
        @staticmethod
        def getActiveSession():
            return None

    monkeypatch.setattr(adapter, "_load_spark_session", lambda: _NoActive)

    with pytest.raises(ShardExecutionError, match="no active SparkSession"):
        SparkShardExecutor().ensure_available()


# ---------------------------------------------------------------------------
# Worker metadata must satisfy the manifest contract
# ---------------------------------------------------------------------------


def test_synthesized_failure_carries_no_non_finite_timing():
    """Non-finite timings are rejected at manifest construction.

    The adapters synthesize an execution for an infrastructure failure without
    a measured duration, so the field must be absent rather than ``inf``.
    """

    from openmed.processing.shard_executor import ShardExecution

    execution = ShardExecution(
        shard_id=0, status=ShardStatus.FAILED, error_type="OSError"
    )

    assert execution.duration_seconds is None
    record = ShardRecord(
        shard_id=0,
        fingerprint="a" * 64,
        document_count=1,
        status=ShardStatus.FAILED,
        error_type=execution.error_type,
    )
    assert record.error_type == "OSError"


def test_shard_task_is_picklable_for_the_process_and_remote_paths(tmp_path):
    """The constraint every backend shares, checked once, explicitly."""

    import pickle

    from openmed.processing.distributed import DocumentShard
    from tests.unit.processing.test_distributed_executor_conformance import (
        deterministic_handler,
    )

    task = ShardTask(
        shard=DocumentShard(
            shard_id=0,
            document_ids=("doc-00000",),
            document_hashes=("a" * 64,),
            fingerprint="b" * 64,
        ),
        root=tmp_path,
        relative_path="shard-00000.jsonl",
        handler=deterministic_handler,
    )

    restored = pickle.loads(pickle.dumps(task))

    assert restored.shard.shard_id == 0
    assert restored.handler is deterministic_handler


# ---------------------------------------------------------------------------
# What eager resolution does and does not protect
# ---------------------------------------------------------------------------


def _plan_and_manifest(count=9, shards=3):
    from openmed.processing import build_run_manifest, plan_document_shards
    from tests.unit.processing.test_distributed_executor_conformance import (
        synthetic_documents,
    )

    plan = plan_document_shards(synthetic_documents(count), shard_count=shards)
    return plan, build_run_manifest(run_id="run-0001", plan=plan)


@pytest.mark.parametrize("backend", ["ray", "spark"])
def test_missing_backend_still_costs_a_manifest_attempt(tmp_path, monkeypatch, backend):
    """The honest bound on eager resolution.

    ``run_shard_plan`` marks every shard RUNNING and increments ``attempts``
    before it calls ``execute`` at all, so resolving the backend eagerly inside
    ``execute`` cannot prevent the attempt being burned. Pinning the real
    behaviour keeps the docstrings from drifting back to the stronger claim.
    """

    from openmed.processing import InMemoryRunManifestStore, run_shard_plan
    from tests.unit.processing.test_distributed_executor_conformance import (
        deterministic_handler,
    )

    plan, manifest = _plan_and_manifest()
    store = InMemoryRunManifestStore(manifest)

    if backend == "ray":
        import openmed.processing.ray_executor as adapter

        monkeypatch.setattr(
            adapter,
            "_import_module",
            lambda name: (_ for _ in ()).throw(ImportError(name)),
        )
        executor = RayShardExecutor()
        expected = ImportError
    else:
        import openmed.processing.spark_executor as adapter

        class _NoActive:
            @staticmethod
            def getActiveSession():
                return None

        monkeypatch.setattr(adapter, "_load_spark_session", lambda: _NoActive)
        executor = SparkShardExecutor()
        expected = ShardExecutionError

    with pytest.raises(expected):
        run_shard_plan(
            plan,
            deterministic_handler,
            manifest=manifest,
            root=tmp_path,
            store=store,
            executor=executor,
        )

    saved = store.load()
    assert [saved.shard(i).attempts for i in (0, 1, 2)] == [1, 1, 1]
    assert all(saved.shard(i).status == ShardStatus.RUNNING for i in (0, 1, 2))


@pytest.mark.parametrize("backend", ["ray", "spark"])
def test_ensure_available_fails_before_any_attempt_is_burned(
    tmp_path, monkeypatch, backend
):
    """The part that actually protects the manifest, when called first."""

    from openmed.processing import InMemoryRunManifestStore

    _, manifest = _plan_and_manifest()
    store = InMemoryRunManifestStore(manifest)

    if backend == "ray":
        import openmed.processing.ray_executor as adapter

        monkeypatch.setattr(
            adapter,
            "_import_module",
            lambda name: (_ for _ in ()).throw(ImportError(name)),
        )
        executor, expected = RayShardExecutor(), ImportError
    else:
        import openmed.processing.spark_executor as adapter

        class _NoActive:
            @staticmethod
            def getActiveSession():
                return None

        monkeypatch.setattr(adapter, "_load_spark_session", lambda: _NoActive)
        executor, expected = SparkShardExecutor(), ShardExecutionError

    with pytest.raises(expected):
        executor.ensure_available()

    saved = store.load()
    assert [saved.shard(i).attempts for i in (0, 1, 2)] == [0, 0, 0]


def test_ray_init_failure_is_raised_by_ensure_available():
    """A dead cluster is likelier than a missing import for this class."""

    class _DeadRay:
        @staticmethod
        def is_initialized():
            return False

        @staticmethod
        def init(**kwargs):
            raise ConnectionError("cluster unreachable")

    executor = RayShardExecutor(ray_module=_DeadRay())

    with pytest.raises(ConnectionError):
        executor.ensure_available()


def test_ray_init_failure_is_raised_from_execute_not_the_generator():
    class _DeadRay:
        @staticmethod
        def is_initialized():
            return False

        @staticmethod
        def init(**kwargs):
            raise ConnectionError("cluster unreachable")

    executor = RayShardExecutor(ray_module=_DeadRay())

    with pytest.raises(ConnectionError):
        executor.execute([object()])  # not iterated


# ---------------------------------------------------------------------------
# Infrastructure-failure branch
# ---------------------------------------------------------------------------


def test_ray_worker_death_is_recorded_as_the_underlying_error_type(tmp_path):
    """Drives the adapter's remote-failure branch with Ray's real error shape."""

    from openmed.processing import InMemoryRunManifestStore, run_shard_plan
    from tests.unit.processing.test_distributed_executor_conformance import (
        deterministic_handler,
    )

    plan, manifest = _plan_and_manifest()
    store = InMemoryRunManifestStore(manifest)
    ray_module = FakeRayModule(
        remote_failures={1: make_ray_task_error(OSError("worker died"))}
    )
    executor = RayShardExecutor(auto_init=False, ray_module=ray_module)

    result = run_shard_plan(
        plan,
        deterministic_handler,
        manifest=manifest,
        root=tmp_path,
        store=store,
        executor=executor,
    )

    failed = [e for e in result.executions if not e.succeeded]
    assert [e.shard_id for e in failed] == [1]
    assert failed[0].error_type == "OSError"
    assert sorted(result.completed_shards) == [0, 2]
    assert store.load().shard(1).error_type == "OSError"


def test_spark_job_failure_marks_every_shard_failed(tmp_path):
    """``collect`` is all-or-nothing, so no shard can have reported."""

    from openmed.processing import InMemoryRunManifestStore, run_shard_plan
    from tests.unit.processing.test_distributed_executor_conformance import (
        deterministic_handler,
    )

    plan, manifest = _plan_and_manifest()
    store = InMemoryRunManifestStore(manifest)
    job_error = type("Py4JJavaError", (RuntimeError,), {})("executor lost")
    session = FakeSparkSession(job_error=job_error)
    executor = SparkShardExecutor(session=session)

    result = run_shard_plan(
        plan,
        deterministic_handler,
        manifest=manifest,
        root=tmp_path,
        store=store,
        executor=executor,
    )

    assert result.completed_shards == ()
    assert sorted(result.failed_shards) == [0, 1, 2]
    assert {e.error_type for e in result.executions} == {"Py4JJavaError"}


def test_ray_bounded_window_completes_out_of_order(tmp_path):
    """The one realistic Ray configuration: a throttle plus reordering."""

    from openmed.processing import run_shard_plan
    from tests.unit.processing.test_distributed_executor_conformance import (
        deterministic_handler,
    )

    plan, manifest = _plan_and_manifest(count=12, shards=4)
    ray_module = FakeRayModule(completion_order=[1, 0, 3, 2])
    executor = RayShardExecutor(auto_init=False, ray_module=ray_module, max_in_flight=2)

    result = run_shard_plan(
        plan, deterministic_handler, manifest=manifest, root=tmp_path, executor=executor
    )

    assert result.is_complete
    assert sorted(result.completed_shards) == [0, 1, 2, 3]


def test_spark_function_serializer_hook_matches_the_ray_fake(tmp_path):
    """Symmetry: both fakes can gate the mapped function, not just the payload."""

    import cloudpickle  # noqa: F401  - availability checked below

    from openmed.processing import run_shard_plan
    from tests.unit.processing.test_distributed_executor_conformance import (
        deterministic_handler,
    )

    cloudpickle = pytest.importorskip("cloudpickle")
    plan, manifest = _plan_and_manifest()
    session = FakeSparkSession(function_serializer=cloudpickle)
    executor = SparkShardExecutor(session=session)

    result = run_shard_plan(
        plan, deterministic_handler, manifest=manifest, root=tmp_path, executor=executor
    )

    assert result.is_complete


def test_ray_forwards_resource_options_to_remote(tmp_path):
    """``num_cpus`` and extra remote args reach ``ray.remote``."""

    from openmed.processing import run_shard_plan
    from tests.unit.processing.test_distributed_executor_conformance import (
        deterministic_handler,
    )

    ray_module = FakeRayModule()
    executor = RayShardExecutor(
        auto_init=False, ray_module=ray_module, num_cpus=2, memory=1024
    )
    plan, manifest = _plan_and_manifest()

    run_shard_plan(
        plan, deterministic_handler, manifest=manifest, root=tmp_path, executor=executor
    )

    assert ray_module.remote_options == [{"memory": 1024, "num_cpus": 2}]
