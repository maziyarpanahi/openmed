"""Self-tests for the distributed backend fakes.

The Ray and Spark shard-executor adapters are proved offline through the fakes
in :mod:`tests.unit.processing.fixtures.distributed_backend_fakes`, so those
fakes must themselves be shown to reproduce the cluster behaviours they stand
in for. Everything here is synthetic and generated algorithmically.
"""

from __future__ import annotations

import pickle
import re
import threading

import pytest

from tests.unit.processing.fixtures.distributed_backend_fakes import (
    DriverOnlyStateError,
    FakeRayModule,
    FakeRayTaskError,
    FakeSparkContext,
    FakeSparkSession,
    FakeSparkTaskError,
    SerializationError,
    find_driver_only_state,
    round_trip,
    synthetic_shard_payloads,
)

# Mirrors ``_ERROR_TYPE_PATTERN`` shipped by the run manifest (#1069). Replace
# this local copy with the imported constant once that module is on this branch.
ERROR_TYPE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


def _shard_summary(payload: dict[str, object]) -> dict[str, object]:
    """PHI-free worker return value used across these tests."""

    return {
        "shard_id": payload["shard_id"],
        "document_count": payload["document_count"],
    }


# ---------------------------------------------------------------------------
# Serialization boundary
# ---------------------------------------------------------------------------


def test_synthetic_shard_payloads_round_trip_through_stdlib_pickle():
    payloads = synthetic_shard_payloads(5)

    assert [round_trip(payload) for payload in payloads] == list(payloads)


def test_round_trip_rejects_a_payload_holding_a_lock():
    payload = {"shard_id": 0, "handle": threading.Lock()}

    with pytest.raises(SerializationError) as excinfo:
        round_trip(payload)

    assert "TypeError" in str(excinfo.value)


def test_round_trip_failure_message_never_echoes_the_payload():
    payload = {"note": "synthetic-marker-value", "handle": threading.Lock()}

    with pytest.raises(SerializationError) as excinfo:
        round_trip(payload)

    assert "synthetic-marker-value" not in str(excinfo.value)


def test_ray_submission_rejects_an_unserializable_argument():
    ray = FakeRayModule()
    remote_function = ray.remote(_shard_summary)

    with pytest.raises(SerializationError):
        remote_function.remote({"shard_id": 0, "handle": threading.Lock()})

    assert ray.outcomes == []


def test_spark_parallelize_rejects_an_unserializable_element():
    context = FakeSparkContext()

    with pytest.raises(SerializationError):
        context.parallelize([{"shard_id": 0, "handle": threading.Lock()}])


def test_cloudpickle_accepts_closures_that_stdlib_pickle_rejects():
    """Functions must not be gated on stdlib pickle.

    Ray and Spark serialize task functions with ``cloudpickle``, so a locally
    defined closure is legal on a real cluster. This pins the reason the fakes
    only apply the stdlib-pickle gate to task *arguments*.
    """

    cloudpickle = pytest.importorskip(
        "cloudpickle",
        reason="cloudpickle fidelity check requires the optional cloudpickle package",
    )
    offset = 7

    def add_offset(value: int) -> int:
        return value + offset

    with pytest.raises(SerializationError):
        round_trip(add_offset, serializer=pickle)

    assert round_trip(add_offset, serializer=cloudpickle)(1) == 8


# ---------------------------------------------------------------------------
# Driver-only state must never reach a worker
# ---------------------------------------------------------------------------


class _StandInManifestStore:
    """Stands in for a RunManifestStore: picklable, but driver-owned.

    Swap this for the real ``RunManifestStore`` implementations once #1069 is
    on this branch; the hazard it encodes is theirs, verified against
    ``InMemoryRunManifestStore`` and ``LocalFileRunManifestStore``.
    """

    def __init__(self) -> None:
        self.manifest: dict[str, str] = {}

    def save(self, manifest: dict[str, str]) -> None:
        self.manifest = dict(manifest)


def test_a_manifest_store_pickles_without_error_but_yields_a_copy():
    """The reason the structural guard below exists.

    Serialization raises nothing here: each worker would silently mutate its
    own copy while the driver's manifest stayed behind.
    """

    store = _StandInManifestStore()
    store.save({"shard-0": "PENDING"})

    worker_copy = round_trip(store)
    worker_copy.save({"shard-0": "COMPLETED"})

    assert worker_copy is not store
    assert store.manifest == {"shard-0": "PENDING"}
    assert worker_copy.manifest == {"shard-0": "COMPLETED"}


def test_find_driver_only_state_reports_nested_capture_paths():
    payload = {"shard_id": 0, "context": {"store": _StandInManifestStore()}}

    found = find_driver_only_state(payload, (_StandInManifestStore,))

    assert found == ("payload['context']['store']: _StandInManifestStore",)


def test_find_driver_only_state_accepts_a_clean_payload():
    payloads = synthetic_shard_payloads(3)

    assert find_driver_only_state(payloads, (_StandInManifestStore,)) == ()


def test_ray_submission_rejects_a_captured_manifest_store():
    ray = FakeRayModule(forbidden_types=(_StandInManifestStore,))
    remote_function = ray.remote(_shard_summary)

    with pytest.raises(DriverOnlyStateError, match="_StandInManifestStore"):
        remote_function.remote({"shard_id": 0, "store": _StandInManifestStore()})

    assert ray.outcomes == []


def test_spark_submission_rejects_a_captured_manifest_store():
    session = FakeSparkSession(forbidden_types=(_StandInManifestStore,))

    with pytest.raises(DriverOnlyStateError, match="_StandInManifestStore"):
        session.sparkContext.parallelize(
            [{"shard_id": 0, "store": _StandInManifestStore()}], 1
        )


# ---------------------------------------------------------------------------
# Out-of-order completion
# ---------------------------------------------------------------------------


def test_ray_get_preserves_submission_order_under_out_of_order_completion():
    payloads = synthetic_shard_payloads(4)
    ray = FakeRayModule(completion_order=[3, 1, 0, 2])
    remote_function = ray.remote(_shard_summary)

    refs = [remote_function.remote(payload) for payload in payloads]
    results = ray.get(refs)

    assert [result["shard_id"] for result in results] == [0, 1, 2, 3]


def test_ray_wait_yields_the_configured_completion_order():
    payloads = synthetic_shard_payloads(4)
    ray = FakeRayModule(completion_order=[3, 1, 0, 2])
    remote_function = ray.remote(_shard_summary)
    refs = [remote_function.remote(payload) for payload in payloads]

    seen: list[int] = []
    pending = list(refs)
    while pending:
        ready, pending = ray.wait(pending, num_returns=1)
        seen.extend(ref.index for ref in ready)

    assert seen == [3, 1, 0, 2]


def test_spark_collect_returns_element_order_despite_shuffled_execution():
    payloads = synthetic_shard_payloads(4)
    session = FakeSparkSession(execution_order=[2, 0, 3, 1])
    context = session.sparkContext

    results = context.parallelize(payloads, 4).map(_shard_summary).collect()

    assert context.invocation_order == [2, 0, 3, 1]
    assert [result["shard_id"] for result in results] == [0, 1, 2, 3]


def test_spark_parallelize_records_the_requested_slice_count():
    payloads = synthetic_shard_payloads(6)
    context = FakeSparkContext()

    context.parallelize(payloads, len(payloads))

    assert context.num_slices == 6


def test_spark_map_partitions_matches_map_results():
    payloads = synthetic_shard_payloads(3)
    context = FakeSparkContext()

    def summarize_partition(partition):
        return [_shard_summary(payload) for payload in partition]

    mapped = context.parallelize(payloads, 3).map(_shard_summary).collect()
    partitioned = (
        FakeSparkContext().parallelize(payloads, 3).mapPartitions(summarize_partition)
    ).collect()

    assert mapped == partitioned


# ---------------------------------------------------------------------------
# Per-task failure stays PHI-free
# ---------------------------------------------------------------------------


def _fail_on_shard_two(payload: dict[str, object]) -> dict[str, object]:
    if payload["shard_id"] == 2:
        raise ValueError(f"synthetic failure for {payload['document_ids']!r}")
    return _shard_summary(payload)


def test_ray_task_failure_surfaces_only_the_error_type():
    payloads = synthetic_shard_payloads(4)
    ray = FakeRayModule()
    remote_function = ray.remote(_fail_on_shard_two)

    refs = [remote_function.remote(payload) for payload in payloads]

    assert ray.outcomes[2].error_type == "ValueError"
    with pytest.raises(FakeRayTaskError) as excinfo:
        ray.get(refs[2])
    assert excinfo.value.error_type == "ValueError"
    assert "doc-002" not in str(excinfo.value)


def test_ray_failure_of_one_task_leaves_siblings_retrievable():
    payloads = synthetic_shard_payloads(4)
    ray = FakeRayModule()
    remote_function = ray.remote(_fail_on_shard_two)
    refs = [remote_function.remote(payload) for payload in payloads]

    survivors = [ray.get(ref) for index, ref in enumerate(refs) if index != 2]

    assert [result["shard_id"] for result in survivors] == [0, 1, 3]


def test_spark_task_failure_surfaces_only_the_error_type():
    payloads = synthetic_shard_payloads(4)
    context = FakeSparkContext()
    rdd = context.parallelize(payloads, 4).map(_fail_on_shard_two)

    with pytest.raises(FakeSparkTaskError) as excinfo:
        rdd.collect()

    assert excinfo.value.error_type == "ValueError"
    assert "doc-002" not in str(excinfo.value)


@pytest.mark.parametrize(
    "error_type",
    ["ValueError", "OSError", "SerializationError", "openmed.ShardWriteError"],
)
def test_reported_error_types_satisfy_the_manifest_contract(error_type):
    """#1069 rejects any ``error_type`` that is not a bare dotted identifier."""

    assert ERROR_TYPE_PATTERN.match(error_type)


def test_exception_messages_are_not_valid_error_types():
    """Guards the failure mode the manifest contract exists to prevent."""

    assert not ERROR_TYPE_PATTERN.match("ValueError: patient Jane Roe not found")


# ---------------------------------------------------------------------------
# Decorator forms and lifecycle
# ---------------------------------------------------------------------------


def test_ray_remote_supports_bare_and_optioned_decorator_forms():
    ray = FakeRayModule()
    payload = synthetic_shard_payloads(1)[0]

    bare = ray.remote(_shard_summary)
    optioned = ray.remote(num_cpus=2)(_shard_summary)

    assert ray.get(bare.remote(payload)) == _shard_summary(payload)
    assert ray.get(optioned.remote(payload)) == _shard_summary(payload)
    assert ray.remote_options == [{"num_cpus": 2}]


def test_ray_lifecycle_tracks_initialization():
    ray = FakeRayModule()

    assert not ray.is_initialized()
    ray.init(ignore_reinit_error=True)
    assert ray.is_initialized()
    ray.shutdown()
    assert not ray.is_initialized()
