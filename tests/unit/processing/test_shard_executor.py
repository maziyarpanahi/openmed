"""Tests for the local shard executor and idempotent shard writes.

Every document, identifier, and payload below is generated algorithmically and
is entirely synthetic.
"""

from __future__ import annotations

import functools
import json
import multiprocessing
import os
import pickle
import tempfile
import threading
from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path

import pytest

from openmed.processing import (
    DriverOnlyStateError,
    LocalShardExecutor,
    ShardExecutionError,
    ShardTask,
    find_driver_only_state,
    plan_document_shards,
    run_shard_plan,
    shard_output_filename,
    write_shard_output,
)
from openmed.processing.distributed import DocumentShard, ShardPlan
from openmed.processing.run_manifest import (
    InMemoryRunManifestStore,
    LocalFileRunManifestStore,
    ShardOutputDigestMismatchError,
    ShardStatus,
    build_run_manifest,
    shard_output_digest,
    validate_shard_outputs,
)

DOCUMENT_ID_PREFIX = "note-"
SECRET_ERROR_TOKEN = "unredacted-handler-detail"
CRASH_PHASES = (
    "after_write",
    "after_fsync",
    "before_replace",
    "after_replace",
    "after_directory_fsync",
)


# --- Synthetic corpus -------------------------------------------------------


def _documents(count: int = 12) -> list[dict[str, str]]:
    """Build a synthetic corpus with algorithmically generated ids and text."""
    return [
        {
            "id": f"{DOCUMENT_ID_PREFIX}{index:04d}",
            "text": f"Synthetic subject {index:04d} reached extension 555-{index:04d}.",
        }
        for index in range(count)
    ]


def _plan(count: int = 12, shard_count: int = 5) -> ShardPlan:
    return plan_document_shards(_documents(count), shard_count=shard_count)


def _manifest(plan: ShardPlan, run_id: str = "run-0001"):
    return build_run_manifest(plan, run_id=run_id)


# --- Module-level handlers (picklable for the process-pool path) ------------


def deterministic_shard_handler(shard: DocumentShard) -> bytes:
    """Serialize a shard to stable bytes derived only from document hashes."""
    lines = [
        json.dumps(
            {"shard_id": shard.shard_id, "document_hash": document_hash},
            sort_keys=True,
            separators=(",", ":"),
        )
        for document_hash in shard.document_hashes
    ]
    return "".join(f"{line}\n" for line in lines).encode("utf-8")


def alternate_shard_handler(shard: DocumentShard) -> bytes:
    """Produce different but still deterministic bytes for the same shard."""
    return deterministic_shard_handler(shard) + b"revised\n"


def failing_shard_handler(shard: DocumentShard) -> bytes:
    """Always fail with a message that must never reach the manifest."""
    raise RuntimeError(
        f"{SECRET_ERROR_TOKEN}: shard {shard.shard_id} ids={shard.document_ids}"
    )


def rejects_empty_shards_handler(shard: DocumentShard) -> bytes:
    """Fail loudly if the executor ever hands an empty shard to a handler."""
    if shard.document_count == 0:
        raise AssertionError("handler must not be invoked for an empty shard")
    return deterministic_shard_handler(shard)


def non_bytes_shard_handler(shard: DocumentShard) -> bytes:
    """Return an unsupported payload type to exercise the write guard."""
    return f"shard-{shard.shard_id}"  # type: ignore[return-value]


# --- Helpers ----------------------------------------------------------------


def _probe_inode_support() -> bool:
    """Whether this filesystem reports distinct, non-zero inode numbers.

    Windows fills ``st_ino`` from the NTFS file index, but it is 0 on some
    filesystems. That breaks inode comparison in both directions at once:
    "the inode changed" becomes trivially false, and "the inode is unchanged"
    becomes trivially true and silently passes. Measure the property rather
    than guessing from the platform name.
    """
    with tempfile.TemporaryDirectory() as raw_directory:
        directory = Path(raw_directory)
        first = directory / "first"
        second = directory / "second"
        first.write_bytes(b"first")
        second.write_bytes(b"second")
        first_inode = first.stat().st_ino
        second_inode = second.stat().st_ino
    return bool(first_inode) and bool(second_inode) and first_inode != second_inode


#: Inode identity is a *corroborating* signal only. Every test below also
#: proves its point portably -- by instrumenting the atomic-write hook, or by
#: comparing bytes -- so skipping the inode assertion loses no coverage of the
#: product contract, only of the mechanism.
_INODES_DISTINGUISH_FILES = _probe_inode_support()


def _inodes(root: Path) -> dict[str, int]:
    return {
        path.name: path.stat().st_ino for path in sorted(root.glob("shard-*.jsonl"))
    }


def _payloads(root: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(root.glob("shard-*.jsonl"))}


def _iter_serialized_strings(value: object) -> Iterator[str]:
    """Yield every key and value in a serialized payload as a string."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _iter_serialized_strings(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_serialized_strings(item)
    elif isinstance(value, str):
        yield value
    else:
        yield str(value)


def _assert_free_of(payload: object, forbidden: tuple[str, ...]) -> None:
    observed = list(_iter_serialized_strings(payload))
    assert observed, "PHI scan inspected nothing"
    for token in forbidden:
        for text in observed:
            assert token not in text, f"{token!r} leaked into {text!r}"


# --- Digest parity ----------------------------------------------------------


def test_threaded_execution_matches_sequential_digests(tmp_path: Path) -> None:
    plan = _plan()
    sequential_root = tmp_path / "sequential"
    threaded_root = tmp_path / "threaded"

    sequential = run_shard_plan(
        plan,
        deterministic_shard_handler,
        manifest=_manifest(plan),
        root=sequential_root,
        executor=LocalShardExecutor(max_workers=1),
    )
    threaded = run_shard_plan(
        plan,
        deterministic_shard_handler,
        manifest=_manifest(plan),
        root=threaded_root,
        executor=LocalShardExecutor(max_workers=4),
    )

    assert sequential.is_complete and threaded.is_complete
    assert _payloads(sequential_root) == _payloads(threaded_root)
    assert [record.output_digest for record in sequential.manifest.shards] == [
        record.output_digest for record in threaded.manifest.shards
    ]
    assert sequential.completed_shards == threaded.completed_shards


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="requires a fork start method to pickle module-level handlers",
)
def test_process_pool_execution_matches_sequential_digests(tmp_path: Path) -> None:
    plan = _plan(count=6, shard_count=3)
    # This fixture contains an empty shard, so the run also proves the
    # empty-shard bypass survives a process boundary: the substituted handler
    # is module-level and therefore picklable. Asserted so the coverage is a
    # deliberate property of the fixture rather than an accident of hashing.
    assert len(plan.non_empty_shards) < plan.shard_count
    sequential_root = tmp_path / "sequential"
    process_root = tmp_path / "processes"

    sequential = run_shard_plan(
        plan,
        deterministic_shard_handler,
        manifest=_manifest(plan),
        root=sequential_root,
    )
    forked = run_shard_plan(
        plan,
        deterministic_shard_handler,
        manifest=_manifest(plan),
        root=process_root,
        executor=LocalShardExecutor(
            max_workers=2,
            use_processes=True,
            mp_context=multiprocessing.get_context("fork"),
        ),
    )

    assert forked.is_complete
    assert _payloads(sequential_root) == _payloads(process_root)
    assert [record.output_digest for record in sequential.manifest.shards] == [
        record.output_digest for record in forked.manifest.shards
    ]


def test_manifest_digest_matches_shard_output_digest_helper(tmp_path: Path) -> None:
    plan = _plan()
    result = run_shard_plan(
        plan,
        deterministic_shard_handler,
        manifest=_manifest(plan),
        root=tmp_path,
    )

    for record in result.manifest.shards:
        assert record.output_path == shard_output_filename(record.shard_id)
        assert record.output_digest == shard_output_digest(
            tmp_path / record.output_path
        )
        assert record.output_bytes == (tmp_path / record.output_path).stat().st_size
    assert validate_shard_outputs(result.manifest, root=tmp_path).all_valid


# --- Idempotent overwrite ---------------------------------------------------


def test_reexecuting_completed_shard_overwrites_same_bytes(tmp_path: Path) -> None:
    plan = _plan()
    store = InMemoryRunManifestStore()
    first = run_shard_plan(
        plan,
        deterministic_shard_handler,
        manifest=_manifest(plan),
        root=tmp_path,
        store=store,
    )
    first_payloads = _payloads(tmp_path)
    first_inodes = _inodes(tmp_path)

    replaced: list[str] = []

    def counting_hook(phase: str, path: Path) -> None:
        if phase == "after_replace":
            replaced.append(path.name)

    second = run_shard_plan(
        plan,
        deterministic_shard_handler,
        manifest=first.manifest,
        root=tmp_path,
        store=store,
        force=True,
        executor=LocalShardExecutor(hook=counting_hook),
    )

    # The bytes are unchanged...
    assert _payloads(tmp_path) == first_payloads
    assert len(first_payloads) == plan.shard_count
    # ...but the overwrite path really ran: every shard was atomically replaced
    # by a fresh temporary file, so every inode is new.
    assert sorted(replaced) == sorted(first_payloads)
    second_inodes = _inodes(tmp_path)
    assert second_inodes.keys() == first_inodes.keys()
    if _INODES_DISTINGUISH_FILES:
        assert all(second_inodes[name] != first_inodes[name] for name in first_inodes)
    # Re-running appends nothing.
    for name, payload in _payloads(tmp_path).items():
        assert len(payload) == len(first_payloads[name])
    assert second.skipped_shards == ()
    assert all(record.attempts == 2 for record in second.manifest.shards)


def test_matching_outputs_are_skipped_without_rewriting(tmp_path: Path) -> None:
    plan = _plan()
    first = run_shard_plan(
        plan,
        deterministic_shard_handler,
        manifest=_manifest(plan),
        root=tmp_path,
    )
    first_inodes = _inodes(tmp_path)

    def forbidden_hook(phase: str, path: Path) -> None:
        raise AssertionError("a skipped shard must not be rewritten")

    second = run_shard_plan(
        plan,
        deterministic_shard_handler,
        manifest=first.manifest,
        root=tmp_path,
        executor=LocalShardExecutor(hook=forbidden_hook),
    )

    assert second.executions == ()
    assert second.skipped_shards == tuple(range(plan.shard_count))
    if _INODES_DISTINGUISH_FILES:
        assert _inodes(tmp_path) == first_inodes
    assert all(record.attempts == 1 for record in second.manifest.shards)
    assert second.is_complete


# --- Crash safety -----------------------------------------------------------


def _crash_run(tmp_path: Path, phase: str) -> tuple[bytes, object]:
    """Seed a shard output, then crash a forced re-run at ``phase``."""
    root = tmp_path / phase
    plan = _plan(count=4, shard_count=1)
    first = run_shard_plan(
        plan,
        deterministic_shard_handler,
        manifest=_manifest(plan),
        root=root,
    )
    target_name = shard_output_filename(0)
    previous = (root / target_name).read_bytes()

    def crash_hook(hook_phase: str, path: Path) -> None:
        if path.name == target_name and hook_phase == phase:
            raise RuntimeError("simulated power loss")

    # A *fresh* manifest, so the run carries no recorded digest to compare
    # against and the write is genuinely attempted. Reusing the completed
    # manifest would trip the digest guard and skip the write entirely, which
    # would make every crash phase observe identical bytes.
    del first
    result = run_shard_plan(
        plan,
        alternate_shard_handler,
        manifest=_manifest(plan),
        root=root,
        executor=LocalShardExecutor(hook=crash_hook),
    )
    return previous, (root, result, target_name)


@pytest.mark.parametrize("phase", CRASH_PHASES)
def test_crash_during_shard_write_never_leaves_a_partial_output(
    tmp_path: Path,
    phase: str,
) -> None:
    previous, (root, result, target_name) = _crash_run(tmp_path, phase)
    expected_new = alternate_shard_handler(_plan(count=4, shard_count=1).shards[0])

    persisted = (root / target_name).read_bytes()
    # Either the old bytes or the complete new bytes -- never a partial write.
    assert persisted in {previous, expected_new}
    # No temporary file is left behind for anyone to mistake for an output.
    assert not list(root.glob(f".{target_name}.*.tmp"))
    assert not list(root.glob("*.tmp"))
    # The interrupted shard is not counted as completed.
    record = result.manifest.shard(0)
    assert record.status == ShardStatus.FAILED
    assert record.error_type == "RuntimeError"
    assert record.output_digest is None
    assert result.failed_shards == (0,)
    assert not result.is_complete
    assert validate_shard_outputs(result.manifest, root=root).all_valid


def test_crash_phase_observations_differ_across_the_rename_boundary(
    tmp_path: Path,
) -> None:
    """The hook must actually interrupt where it claims to.

    A hook that silently no-opped, or one that always fired too late, would
    produce identical on-disk state for every phase. Crossing ``os.replace`` has
    to be observable: before it the previous bytes survive, after it the new
    bytes are already durable.
    """
    observed: dict[str, bytes] = {}
    previous_bytes: set[bytes] = set()
    for phase in CRASH_PHASES:
        previous, (root, _result, target_name) = _crash_run(tmp_path, phase)
        previous_bytes.add(previous)
        observed[phase] = (root / target_name).read_bytes()

    assert len(previous_bytes) == 1
    previous = previous_bytes.pop()
    expected_new = alternate_shard_handler(_plan(count=4, shard_count=1).shards[0])
    assert previous != expected_new

    assert observed["after_write"] == previous
    assert observed["after_fsync"] == previous
    assert observed["before_replace"] == previous
    assert observed["after_replace"] == expected_new
    assert observed["after_directory_fsync"] == expected_new
    assert len(set(observed.values())) == 2


def test_duplicate_concurrent_writers_never_interleave(tmp_path: Path) -> None:
    """Two writers publishing the same shard leave one complete payload.

    Duplicate execution is expected under at-least-once delivery. Each writer
    gets its own temporary file and the rename is atomic, so the loser's bytes
    are discarded whole rather than interleaved with the winner's.
    """
    path = tmp_path / shard_output_filename(0)
    # Distinct, equal-length payloads: a torn or interleaved write would still
    # have the right size but would not equal any single writer's bytes.
    payloads = [
        b"".join(
            f"writer {writer_index} line {line_index:05d}\n".encode("ascii")
            for line_index in range(5000)
        )
        for writer_index in range(8)
    ]
    assert len({len(payload) for payload in payloads}) == 1
    assert len(set(payloads)) == 8

    barrier = threading.Barrier(len(payloads))
    results: list[tuple[str, int]] = []

    def writer(payload: bytes) -> None:
        barrier.wait()
        results.append(write_shard_output(path, payload))

    threads = [threading.Thread(target=writer, args=(payload,)) for payload in payloads]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == len(payloads)
    persisted = path.read_bytes()
    # Exactly one writer's bytes survive, whole.
    assert persisted in payloads
    assert shard_output_digest(path) in {digest for digest, _ in results}
    assert not list(tmp_path.glob("*.tmp"))
    assert sorted(entry.name for entry in tmp_path.iterdir()) == [path.name]


def test_planted_temporary_file_is_never_counted_as_a_shard_output(
    tmp_path: Path,
) -> None:
    plan = _plan()
    target_name = shard_output_filename(0)
    planted = tmp_path / f".{target_name}.abc123.tmp"
    tmp_path.mkdir(parents=True, exist_ok=True)
    planted.write_bytes(b"partially written shard\n")

    result = run_shard_plan(
        plan,
        deterministic_shard_handler,
        manifest=_manifest(plan),
        root=tmp_path,
    )

    assert planted.read_bytes() == b"partially written shard\n"
    output_paths = [record.output_path for record in result.manifest.shards]
    assert all(path is not None and not path.endswith(".tmp") for path in output_paths)
    assert planted.name not in output_paths
    assert validate_shard_outputs(result.manifest, root=tmp_path).all_valid
    assert (tmp_path / target_name).read_bytes() != planted.read_bytes()


# --- Worker failures --------------------------------------------------------


def test_worker_failure_records_error_type_without_leaking_detail(
    tmp_path: Path,
) -> None:
    plan = _plan()
    # Empty shards bypass the handler and complete, so "every shard fails"
    # only holds while the fixture has none. Assert the shape rather than
    # trusting the hash distribution to keep it.
    assert len(plan.non_empty_shards) == plan.shard_count
    store = LocalFileRunManifestStore(tmp_path / "manifest.json")
    result = run_shard_plan(
        plan,
        failing_shard_handler,
        manifest=_manifest(plan),
        root=tmp_path / "outputs",
        store=store,
    )

    assert result.completed_shards == ()
    assert result.failed_shards == tuple(range(plan.shard_count))
    assert not result.is_complete
    for record in result.manifest.shards:
        assert record.status == ShardStatus.FAILED
        assert record.error_type == "RuntimeError"
        assert record.attempts == 1
        assert record.output_digest is None
    assert not list((tmp_path / "outputs").glob("shard-*.jsonl"))

    forbidden = (SECRET_ERROR_TOKEN, DOCUMENT_ID_PREFIX, "Synthetic subject", "555-")
    _assert_free_of(result.to_dict(), forbidden)
    _assert_free_of(result.manifest.to_dict(), forbidden)
    _assert_free_of(
        json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")),
        forbidden,
    )


def test_completed_manifest_bytes_are_phi_free(tmp_path: Path) -> None:
    """The success path is where ``output_path`` actually reaches the manifest.

    Shard file names must be derived from the shard id alone. A document-derived
    name would put record content into manifest bytes, which downstream PHI
    scanners read.
    """
    plan = _plan()
    store = LocalFileRunManifestStore(tmp_path / "manifest.json")
    outputs = tmp_path / "outputs"
    result = run_shard_plan(
        plan,
        deterministic_shard_handler,
        manifest=_manifest(plan),
        root=outputs,
        store=store,
    )
    assert result.is_complete

    forbidden = (SECRET_ERROR_TOKEN, DOCUMENT_ID_PREFIX, "Synthetic subject", "555-")
    raw_manifest = (tmp_path / "manifest.json").read_text(encoding="utf-8")
    for token in forbidden:
        assert token not in raw_manifest
    _assert_free_of(json.loads(raw_manifest), forbidden)
    _assert_free_of(result.manifest.to_dict(), forbidden)
    _assert_free_of(result.to_dict(), forbidden)

    # Every recorded path is shard-id derived, and no document hash leaks either.
    document_hashes = {
        document_hash
        for shard in plan.shards
        for document_hash in shard.document_hashes
    }
    assert document_hashes
    for record in result.manifest.shards:
        assert record.output_path == f"shard-{record.shard_id:05d}.jsonl"
        assert not any(
            document_hash in raw_manifest for document_hash in document_hashes
        )


def test_failed_shard_is_reexecuted_and_completes(tmp_path: Path) -> None:
    plan = _plan()
    # attempts == 2 below assumes every shard is retried, which an empty shard
    # would not be: it completes on the first run and is skipped afterwards.
    assert len(plan.non_empty_shards) == plan.shard_count
    failed = run_shard_plan(
        plan,
        failing_shard_handler,
        manifest=_manifest(plan),
        root=tmp_path,
    )

    repaired = run_shard_plan(
        plan,
        deterministic_shard_handler,
        manifest=failed.manifest,
        root=tmp_path,
    )

    assert repaired.is_complete
    assert repaired.completed_shards == tuple(range(plan.shard_count))
    assert repaired.skipped_shards == ()
    for record in repaired.manifest.shards:
        assert record.status == ShardStatus.COMPLETED
        assert record.attempts == 2
        assert record.error_type is None
    assert validate_shard_outputs(repaired.manifest, root=tmp_path).all_valid


def test_non_bytes_handler_payload_is_reported_as_a_failure(tmp_path: Path) -> None:
    plan = _plan(count=4, shard_count=2)
    result = run_shard_plan(
        plan,
        non_bytes_shard_handler,
        manifest=_manifest(plan),
        root=tmp_path,
    )

    assert result.completed_shards == ()
    for record in result.manifest.shards:
        assert record.error_type == "ShardExecutionError"


# --- Corruption and determinism --------------------------------------------


def test_corrupt_shard_output_is_reexecuted_and_repaired(tmp_path: Path) -> None:
    plan = _plan()
    first = run_shard_plan(
        plan,
        deterministic_shard_handler,
        manifest=_manifest(plan),
        root=tmp_path,
    )
    original = _payloads(tmp_path)
    corrupted_name = shard_output_filename(1)
    (tmp_path / corrupted_name).write_bytes(b"corrupted shard bytes\n")
    assert not validate_shard_outputs(first.manifest, root=tmp_path).all_valid

    repaired = run_shard_plan(
        plan,
        deterministic_shard_handler,
        manifest=first.manifest,
        root=tmp_path,
    )

    assert repaired.completed_shards == (1,)
    assert repaired.skipped_shards == tuple(
        shard_id for shard_id in range(plan.shard_count) if shard_id != 1
    )
    assert _payloads(tmp_path) == original
    assert validate_shard_outputs(repaired.manifest, root=tmp_path).all_valid


def test_missing_shard_output_is_reexecuted(tmp_path: Path) -> None:
    plan = _plan()
    first = run_shard_plan(
        plan,
        deterministic_shard_handler,
        manifest=_manifest(plan),
        root=tmp_path,
    )
    original = _payloads(tmp_path)
    (tmp_path / shard_output_filename(2)).unlink()

    repaired = run_shard_plan(
        plan,
        deterministic_shard_handler,
        manifest=first.manifest,
        root=tmp_path,
    )

    assert repaired.completed_shards == (2,)
    assert _payloads(tmp_path) == original


def test_nondeterministic_handler_raises_without_destroying_output(
    tmp_path: Path,
) -> None:
    """A changed digest is reported, and the good bytes survive it.

    The comparison happens before the rename, so the existing outputs are never
    replaced, and the error is raised only once every shard has been recorded --
    a run that detects non-determinism must not also lose the data.
    """
    plan = _plan()
    first = run_shard_plan(
        plan,
        deterministic_shard_handler,
        manifest=_manifest(plan),
        root=tmp_path,
    )
    original = _payloads(tmp_path)
    original_inodes = _inodes(tmp_path)

    with pytest.raises(ShardOutputDigestMismatchError, match="left untouched"):
        run_shard_plan(
            plan,
            alternate_shard_handler,
            manifest=first.manifest,
            root=tmp_path,
            force=True,
        )

    # Not one byte changed, and not one file was rewritten.
    assert _payloads(tmp_path) == original
    if _INODES_DISTINGUISH_FILES:
        assert _inodes(tmp_path) == original_inodes
    # The earlier digests are still recorded, so the run stays validatable.
    assert validate_shard_outputs(first.manifest, root=tmp_path).all_valid


def test_failed_attempt_preserves_the_previously_recorded_digest(
    tmp_path: Path,
) -> None:
    """One transient failure must not launder a later mismatch through.

    If a failed attempt cleared ``output_digest``, the next run would have
    nothing to compare against and non-deterministic output would be accepted.
    """
    plan = _plan(count=4, shard_count=1)
    first = run_shard_plan(
        plan, deterministic_shard_handler, manifest=_manifest(plan), root=tmp_path
    )
    recorded = first.manifest.shard(0)
    assert recorded.output_digest is not None

    transient = run_shard_plan(
        plan, failing_shard_handler, manifest=first.manifest, root=tmp_path, force=True
    )
    after_failure = transient.manifest.shard(0)

    assert after_failure.status == ShardStatus.FAILED
    assert after_failure.error_type == "RuntimeError"
    assert after_failure.output_digest == recorded.output_digest
    assert after_failure.output_path == recorded.output_path
    assert after_failure.output_bytes == recorded.output_bytes

    # The digest survived, so a later non-deterministic result is still caught.
    with pytest.raises(ShardOutputDigestMismatchError):
        run_shard_plan(
            plan,
            alternate_shard_handler,
            manifest=transient.manifest,
            root=tmp_path,
            force=True,
        )


# --- Driver-only state must not reach workers -------------------------------


class CustomRunManifestStore:
    """A store that is not one of the concrete types shipped by the manifest."""

    def __init__(self) -> None:
        self.saved: list[object] = []

    def load(self):
        """Return nothing; present only to match the store shape."""
        return None

    def save(self, manifest) -> None:
        """Record a manifest in memory."""
        self.saved.append(manifest)


class StoreCarryingHandler:
    """A picklable callable that captures driver-only state as an attribute."""

    def __init__(self, store: object) -> None:
        self.store = store

    def __call__(self, shard: DocumentShard) -> bytes:
        return deterministic_shard_handler(shard)


def handler_with_store(shard: DocumentShard, store=None, manifest=None) -> bytes:
    """Module-level handler used to build a store-carrying partial."""
    del store, manifest
    return deterministic_shard_handler(shard)


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="requires a fork start method",
)
def test_process_pool_rejects_a_handler_carrying_driver_only_state(
    tmp_path: Path,
) -> None:
    """A worker that received the store would lose its updates silently.

    Both manifest stores pickle without error and round-trip to a *different*
    object, so nothing in pickle or the type system reports this. The executor
    has to refuse it up front.
    """
    plan = _plan(count=4, shard_count=2)
    manifest = _manifest(plan)
    store = InMemoryRunManifestStore()
    leaky = functools.partial(handler_with_store, store=store, manifest=manifest)

    # The leak vector really is picklable -- that is what makes it dangerous.
    pickle.loads(pickle.dumps(leaky))

    with pytest.raises(DriverOnlyStateError) as excinfo:
        run_shard_plan(
            plan,
            leaky,
            manifest=manifest,
            root=tmp_path,
            store=store,
            executor=LocalShardExecutor(
                max_workers=2,
                use_processes=True,
                mp_context=multiprocessing.get_context("fork"),
            ),
        )

    message = str(excinfo.value)
    assert "InMemoryRunManifestStore" in message
    assert "BatchRunManifest" in message
    assert not list(tmp_path.glob("shard-*.jsonl"))


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="requires a fork start method",
)
def test_process_pool_rejects_an_attribute_captured_custom_store(
    tmp_path: Path,
) -> None:
    plan = _plan(count=4, shard_count=2)

    with pytest.raises(DriverOnlyStateError, match="CustomRunManifestStore"):
        run_shard_plan(
            plan,
            StoreCarryingHandler(CustomRunManifestStore()),
            manifest=_manifest(plan),
            root=tmp_path,
            executor=LocalShardExecutor(
                max_workers=2,
                use_processes=True,
                mp_context=multiprocessing.get_context("fork"),
            ),
        )


def test_thread_pool_is_gated_because_shared_memory_does_not_save_the_update(
    tmp_path: Path,
) -> None:
    """Sharing memory does not make a thread worker's manifest write survive.

    ``BatchRunManifest`` is frozen, so the driver keeps re-saving its own local
    reference and anything a worker writes through the shared store is
    overwritten. The invariant is "the driver is the sole writer", not "state
    must not be copied", so the thread path is gated exactly like the process
    path. The first assertion below demonstrates the loss; the second shows the
    guard now prevents it.
    """
    plan = _plan(count=4, shard_count=2)

    # Demonstrate the loss directly: a store shared with a thread, written
    # through by something other than the driver, keeps none of its updates.
    shared = InMemoryRunManifestStore()
    baseline = _manifest(plan)
    shared.save(baseline)
    worker_copy = baseline.with_shard(
        replace(baseline.shard(0), worker_id="pid-1-thread-1")
    )
    shared.save(worker_copy)
    shared.save(baseline)  # the driver re-saving its own frozen reference
    survived = shared.load()
    assert survived is not None
    assert survived.shard(0).worker_id is None, "the worker's update survived"

    leaky = functools.partial(
        handler_with_store, store=InMemoryRunManifestStore(), manifest=baseline
    )
    with pytest.raises(DriverOnlyStateError, match="silently discarded"):
        run_shard_plan(
            plan,
            leaky,
            manifest=baseline,
            root=tmp_path,
            executor=LocalShardExecutor(max_workers=2),
        )
    assert not list(tmp_path.glob("shard-*.jsonl"))


def test_find_driver_only_state_detects_every_capture_shape() -> None:
    manifest = _manifest(_plan(count=2, shard_count=1))
    store = InMemoryRunManifestStore()

    captured_in_closure = lambda shard: store.load()  # noqa: E731
    partial_capture = functools.partial(handler_with_store, store=store)
    attribute_capture = StoreCarryingHandler(store)

    assert find_driver_only_state(captured_in_closure)
    assert find_driver_only_state(partial_capture)
    assert find_driver_only_state(attribute_capture)
    assert find_driver_only_state(manifest)
    assert find_driver_only_state({"nested": [store]})
    assert find_driver_only_state(CustomRunManifestStore())
    assert find_driver_only_state(store.save)


def test_find_driver_only_state_accepts_ordinary_handlers() -> None:
    plan = _plan(count=4, shard_count=2)
    task = ShardTask(
        shard=plan.shards[0],
        root=Path("/tmp/does-not-matter"),
        relative_path=shard_output_filename(0),
        handler=deterministic_shard_handler,
    )

    assert find_driver_only_state(task) == ()
    assert find_driver_only_state(deterministic_shard_handler) == ()
    assert find_driver_only_state(plan) == ()
    assert find_driver_only_state(None) == ()
    assert find_driver_only_state("shard-00000.jsonl") == ()


def test_find_driver_only_state_survives_reference_cycles() -> None:
    cycle: dict[str, object] = {}
    cycle["self"] = cycle
    cycle["store"] = InMemoryRunManifestStore()

    assert find_driver_only_state(cycle)


class SlotsHandler:
    """Captures a store in ``__slots__``, so there is no instance ``__dict__``."""

    __slots__ = ("store",)

    def __init__(self, store: object) -> None:
        self.store = store

    def __call__(self, shard: DocumentShard) -> bytes:
        return deterministic_shard_handler(shard)


class ClassAttributeHandler:
    """Captures a store as a class attribute rather than an instance one."""

    store = None

    def __call__(self, shard: DocumentShard) -> bytes:
        return deterministic_shard_handler(shard)


class PropertyHandler:
    """Exposes the store only through a property."""

    def __init__(self, store: object) -> None:
        self._store = store

    @property
    def store(self) -> object:
        """Return the captured store."""
        return self._store

    def __call__(self, shard: DocumentShard) -> bytes:
        return deterministic_shard_handler(shard)


def _permissive_proxy(returns: object) -> object:
    """A benign object whose ``__getattr__`` answers every name."""

    class Proxy:
        def __getattr__(self, name: str) -> object:
            return returns

        def __call__(self, shard: DocumentShard) -> bytes:
            return deterministic_shard_handler(shard)

    return Proxy()


def test_guard_detects_every_known_smuggle_shape() -> None:
    """Recall over the shapes that evade a naive ``__dict__``-only walker."""
    store = InMemoryRunManifestStore()

    def closure_capture(shard: DocumentShard) -> bytes:
        return deterministic_shard_handler(shard) if store else b""

    def default_argument(shard: DocumentShard, _store: object = store) -> bytes:
        return deterministic_shard_handler(shard)

    class_attribute = ClassAttributeHandler()
    type(class_attribute).store = store
    try:
        shapes = {
            "closure": closure_capture,
            "partial keyword": functools.partial(handler_with_store, store=store),
            "attribute": StoreCarryingHandler(store),
            "__slots__": SlotsHandler(store),
            "class attribute": class_attribute,
            "default argument": default_argument,
            "property": PropertyHandler(store),
            "bound method": store.save,
            "nested container": {"a": [{"b": (store,)}]},
        }
        for label, candidate in shapes.items():
            assert find_driver_only_state(candidate), f"{label} evaded the guard"
    finally:
        type(class_attribute).store = None


def test_guard_reaches_deeply_nested_captures() -> None:
    class Node:
        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)

    node: object = InMemoryRunManifestStore()
    for _ in range(6):
        node = Node(inner=node)

    assert find_driver_only_state(
        functools.partial(handler_with_store, store={"runtime": node})
    )


@pytest.mark.parametrize(
    "returns",
    ["a string", 5, [1, 2], InMemoryRunManifestStore()],
    ids=["str", "int", "list", "store"],
)
def test_guard_never_crashes_on_permissive_getattr_objects(returns: object) -> None:
    """A proxy that answers every name must not abort the submission path.

    Probing for ``__closure__`` or ``__self__`` on a config proxy, a lazy
    wrapper, or an ORM row returns whatever that object invents. Crashing on a
    benign payload at submission time would be worse than not inspecting it.
    """
    result = find_driver_only_state(_permissive_proxy(returns))
    assert isinstance(result, tuple)


def test_guard_never_crashes_on_a_mock_payload() -> None:
    from unittest.mock import MagicMock

    assert isinstance(find_driver_only_state(MagicMock()), tuple)


def test_rejected_batch_does_not_burn_an_attempt(tmp_path: Path) -> None:
    """Nothing was dispatched, so nothing may be marked RUNNING.

    The guard has to run before the dispatch bookkeeping, otherwise a rejected
    batch leaves every shard ``running`` with ``attempts`` already spent.
    """
    plan = _plan()
    manifest = _manifest(plan)
    store = InMemoryRunManifestStore()
    leaky = functools.partial(handler_with_store, store=InMemoryRunManifestStore())

    with pytest.raises(DriverOnlyStateError):
        run_shard_plan(plan, leaky, manifest=manifest, root=tmp_path, store=store)

    for record in manifest.shards:
        assert record.status == ShardStatus.PENDING
        assert record.attempts == 0
        assert record.started_at is None
    assert store.load() is None
    assert not list(tmp_path.glob("shard-*.jsonl"))


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="requires a fork start method",
)
def test_use_processes_runs_out_of_process_even_with_one_worker(
    tmp_path: Path,
) -> None:
    """``use_processes=True`` must not silently execute in the driver.

    Gating a task for process isolation and then running it in the driver's own
    address space would be wrong in both directions at once.
    """
    plan = _plan(count=4, shard_count=2)
    result = run_shard_plan(
        plan,
        deterministic_shard_handler,
        manifest=_manifest(plan),
        root=tmp_path,
        executor=LocalShardExecutor(
            max_workers=1,
            use_processes=True,
            mp_context=multiprocessing.get_context("fork"),
        ),
    )

    assert result.is_complete
    worker_ids = {record.worker_id for record in result.manifest.shards}
    assert worker_ids
    assert all(f"pid-{os.getpid()}-" not in str(w) for w in worker_ids)


def raises_unnameable_error(shard: DocumentShard) -> bytes:
    """Raise an exception whose class name is not a valid identifier."""
    raise type("bad name!(ValueError)", (Exception,), {})("synthetic")


def test_unnameable_exception_type_does_not_crash_the_driver(tmp_path: Path) -> None:
    """Worker-controlled data must not reach a validating constructor.

    Ray synthesizes wrapper class names such as ``RayTaskError(ValueError)``,
    which the manifest rejects. Recording a failure must never itself fail.
    """
    plan = _plan(count=4, shard_count=2)
    result = run_shard_plan(
        plan,
        raises_unnameable_error,
        manifest=_manifest(plan),
        root=tmp_path,
    )

    assert result.completed_shards == ()
    for record in result.manifest.shards:
        assert record.status == ShardStatus.FAILED
        assert record.error_type == "UnknownError"
    _assert_free_of(result.manifest.to_dict(), ("bad name", "synthetic"))


# --- Validation and small units --------------------------------------------


def test_manifest_must_describe_the_shard_plan(tmp_path: Path) -> None:
    plan = _plan()
    other_manifest = _manifest(_plan(count=8, shard_count=5))

    with pytest.raises(ShardExecutionError, match="does not describe this shard plan"):
        run_shard_plan(
            plan,
            deterministic_shard_handler,
            manifest=other_manifest,
            root=tmp_path,
        )


def test_shard_output_filename_is_stable_and_validated() -> None:
    assert shard_output_filename(0) == "shard-00000.jsonl"
    assert shard_output_filename(42) == "shard-00042.jsonl"
    with pytest.raises(ShardExecutionError):
        shard_output_filename(-1)
    with pytest.raises(ShardExecutionError):
        shard_output_filename(True)


def test_executor_configuration_is_validated() -> None:
    with pytest.raises(ShardExecutionError, match="at least 1"):
        LocalShardExecutor(max_workers=0)
    with pytest.raises(ShardExecutionError, match="must be an integer"):
        LocalShardExecutor(max_workers=True)
    with pytest.raises(ShardExecutionError, match="not supported on process pools"):
        LocalShardExecutor(use_processes=True, hook=lambda phase, path: None)
    with pytest.raises(ShardExecutionError, match="requires use_processes"):
        LocalShardExecutor(mp_context=multiprocessing.get_context())


def test_hook_and_custom_executor_are_mutually_exclusive(tmp_path: Path) -> None:
    plan = _plan()

    with pytest.raises(ShardExecutionError, match="default executor"):
        run_shard_plan(
            plan,
            deterministic_shard_handler,
            manifest=_manifest(plan),
            root=tmp_path,
            executor=LocalShardExecutor(),
            hook=lambda phase, path: None,
        )


def test_write_shard_output_rejects_non_bytes_payloads(tmp_path: Path) -> None:
    with pytest.raises(ShardExecutionError, match="must return bytes"):
        write_shard_output(tmp_path / "shard.jsonl", "not bytes")  # type: ignore[arg-type]
    assert not list(tmp_path.iterdir())


def test_write_shard_output_returns_the_prefixed_digest(tmp_path: Path) -> None:
    path = tmp_path / shard_output_filename(3)
    digest, size = write_shard_output(path, b"synthetic shard payload\n")

    assert digest.startswith("sha256:")
    assert digest == shard_output_digest(path)
    assert size == path.stat().st_size


def test_empty_task_list_yields_no_executions(tmp_path: Path) -> None:
    assert list(LocalShardExecutor().execute([])) == []
    assert list(LocalShardExecutor(max_workers=4).execute([])) == []


def test_empty_shards_settle_without_invoking_the_handler(tmp_path: Path) -> None:
    plan = plan_document_shards(_documents(3), shard_count=16)
    empty_ids = tuple(
        shard.shard_id for shard in plan.shards if shard.document_count == 0
    )
    assert empty_ids, "the fixture must produce empty shards"
    assert len(plan.non_empty_shards) == 3

    result = run_shard_plan(
        plan,
        rejects_empty_shards_handler,
        manifest=_manifest(plan),
        root=tmp_path,
    )

    assert result.failed_shards == ()
    assert result.is_complete
    for shard_id in empty_ids:
        record = result.manifest.shard(shard_id)
        assert record.status == ShardStatus.COMPLETED
        assert (tmp_path / record.output_path).read_bytes() == b""
    assert validate_shard_outputs(result.manifest, root=tmp_path).all_valid

    # A second run has nothing left to do: empty shards do not get re-queued.
    second = run_shard_plan(
        plan,
        failing_shard_handler,
        manifest=result.manifest,
        root=tmp_path,
    )
    assert second.executions == ()
    assert second.skipped_shards == tuple(range(plan.shard_count))


def test_persisted_manifest_round_trips_through_the_store(tmp_path: Path) -> None:
    plan = _plan()
    store = LocalFileRunManifestStore(tmp_path / "manifest.json")
    outputs = tmp_path / "outputs"
    result = run_shard_plan(
        plan,
        deterministic_shard_handler,
        manifest=_manifest(plan),
        root=outputs,
        store=store,
    )

    reloaded = store.load()
    assert reloaded is not None
    assert reloaded.to_dict() == result.manifest.to_dict()
    assert validate_shard_outputs(reloaded, root=outputs).all_valid
    for record in reloaded.shards:
        assert isinstance(record.started_at, float)
        assert isinstance(record.completed_at, float)
        assert record.duration_seconds is not None
        assert record.duration_seconds >= 0.0
        assert record.output_digest.startswith("sha256:")
        assert not Path(record.output_path).is_absolute()
        assert ".." not in Path(record.output_path).parts


def test_shard_task_resolves_its_output_path(tmp_path: Path) -> None:
    plan = _plan(count=2, shard_count=1)
    task = ShardTask(
        shard=plan.shards[0],
        root=tmp_path,
        relative_path=shard_output_filename(0),
        handler=deterministic_shard_handler,
    )

    assert task.output_path == tmp_path / "shard-00000.jsonl"
