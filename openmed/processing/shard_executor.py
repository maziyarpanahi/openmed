"""Local executor and idempotent shard writes for distributed batch runs.

The executor splits responsibilities so that a distributed run stays safe to
interrupt and safe to repeat:

* **Workers** run the shard handler and publish the shard output file, then
  return PHI-free metadata only (shard id, status, digest, byte count, duration,
  worker id, and an exception *type* name). Payloads never travel back across a
  process boundary, so no document text can be pickled into the driver.
* **The driver** is the only writer of the run manifest. It increments
  ``attempts`` when it dispatches a shard, records the outcome when the shard
  reports back, and persists the manifest through the store. Single-writer
  manifests mean concurrent workers never race on manifest state.

Idempotence rests on three properties:

* Shard outputs are published with :func:`openmed.processing.batch._atomic_write_bytes`,
  which writes to a same-directory temporary file, fsyncs it, renames it over
  the destination, and fsyncs the directory. A reader therefore observes either
  the previous bytes or the complete new bytes, never a partial write.
* A shard is only marked ``COMPLETED`` after that rename returns, so a crash in
  between leaves the shard ``RUNNING`` and the next run re-executes it.
* Re-executing a shard rewrites the same final path rather than appending, and
  the resulting digest is compared against the digest already recorded in the
  manifest. A handler that is not deterministic is reported rather than
  silently accepted.

Because the driver is the sole manifest writer, a worker must never receive the
manifest or its store, on *any* transport. Across processes the copy is obvious
once stated: both pickle without error and round-trip to a different object. On
threads the state really is shared, and the update is lost anyway --
:class:`BatchRunManifest` is frozen, so the driver keeps re-saving its own local
reference and a worker's writes through the shared store are simply overwritten.
The invariant is the single writer, not the transport, so the executor rejects
tasks carrying driver-only state on every path; see
:func:`find_driver_only_state`.

That guard is **defence in depth, not the primary control**. The structural way
to be safe is the one this module takes internally and that callers should copy:
submit a module-level worker function over plain data -- a shard descriptor and
an output path -- so that no store or manifest can reach a worker by
construction. A walker can always be out-recalled by a sufficiently indirect
capture; a worker that has no reference to smuggle cannot be.

Output roots must be run-scoped. Two runs sharing a ``root`` overwrite each
other's shard files and neither converges, because each keeps finding digests
that do not match its own manifest. Give every run its own directory.

Atomicity assumption: ``os.replace`` is atomic on local POSIX filesystems and on
NTFS. It is not guaranteed atomic on network filesystems such as NFS or SMB, so
shard output roots should be local storage, matching the assumption already made
by the checkpointed batch writer.
"""

from __future__ import annotations

import functools
import inspect
import os
import re
import threading
import time
import types
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Union

from .batch import AtomicWriteHook, _atomic_write_bytes, _sha256_bytes
from .distributed import DocumentShard, ShardPlan
from .run_manifest import (
    BatchRunManifest,
    InMemoryRunManifestStore,
    LocalFileRunManifestStore,
    RunManifestStore,
    ShardOutputDigestMismatchError,
    ShardStatus,
    shards_to_execute,
)

SHARD_OUTPUT_FILENAME_TEMPLATE = "shard-{shard_id:05d}.jsonl"

ShardHandler = Callable[[DocumentShard], bytes]


class ShardExecutionError(ValueError):
    """Raised when a shard cannot be executed or published safely."""


class DriverOnlyStateError(ShardExecutionError):
    """Raised when driver-only state would be copied into a worker process."""


#: State that only the driver may hold. A worker that receives one of these
#: gets a private copy: it pickles without error, mutations apply to the copy,
#: and they are silently lost when the worker exits.
DRIVER_ONLY_TYPES: tuple[type, ...] = (
    BatchRunManifest,
    InMemoryRunManifestStore,
    LocalFileRunManifestStore,
)

_MAX_STATE_DEPTH = 12
_MAX_STATE_NODES = 20_000

_SCALARS = (str, bytes, bytearray, int, float, complex, bool, Path, type(None))


def _safe_getattr(node: Any, name: str) -> Any:
    """Read an attribute without trusting the object to behave.

    A permissive ``__getattr__`` -- a config proxy, a lazy wrapper, an ORM row --
    answers *every* name, so probing for ``__closure__`` or ``__self__`` can
    return arbitrary objects or raise. The guard runs on the submission path, so
    crashing there on a benign payload would be worse than not inspecting it.
    """
    try:
        return getattr(node, name)
    except Exception:
        return None


def _closure_cells(node: Any) -> tuple[Any, ...]:
    """Return real closure cells only, ignoring anything a proxy invents."""
    closure = _safe_getattr(node, "__closure__")
    if not isinstance(closure, (tuple, list)):
        return ()
    return tuple(cell for cell in closure if isinstance(cell, types.CellType))


def _slot_names(node_type: type) -> tuple[str, ...]:
    """Return every ``__slots__`` name declared across the MRO."""
    names: list[str] = []
    try:
        mro = node_type.__mro__
    except Exception:  # pragma: no cover - exotic metaclass
        return ()
    for klass in mro:
        slots = klass.__dict__.get("__slots__")
        if isinstance(slots, str):
            names.append(slots)
        elif isinstance(slots, (tuple, list)):
            names.extend(name for name in slots if isinstance(name, str))
    return tuple(names)


def _looks_like_manifest_store(value: Any) -> bool:
    """Whether ``value`` structurally matches the run-manifest store shape.

    This is a deliberate **heuristic**, matching exactly one shape: a type
    defining both ``load(self)`` and ``save(self, manifest)``. A ``load`` that
    takes no arguments beyond ``self`` is what makes it specific -- loaders that
    take a path or key do not match. It exists so that a *custom* store, which
    by definition is not in :data:`DRIVER_ONLY_TYPES`, is still caught.

    An unrelated captured object exposing that same pair would be rejected too.
    That is the intended direction to be wrong in: a named error at submission
    time is recoverable, silently discarded manifest updates are not. Pass
    ``forbidden_types`` and match on type alone if a caller needs to opt out.
    """
    if isinstance(value, type):
        return False
    load = getattr(type(value), "load", None)
    save = getattr(type(value), "save", None)
    if not callable(load) or not callable(save):
        return False
    try:
        load_params = list(inspect.signature(load).parameters)
        save_params = list(inspect.signature(save).parameters)
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return False
    return len(load_params) == 1 and len(save_params) == 2


def find_driver_only_state(
    value: Any,
    *,
    forbidden_types: Sequence[type] = DRIVER_ONLY_TYPES,
) -> tuple[str, ...]:
    """Return paths at which ``value`` reaches driver-only state.

    This inspects task *arguments* -- including the state a handler carries in
    a ``functools.partial``, a bound ``__self__``, a closure cell, or an
    instance ``__dict__``. It deliberately does not try to pickle anything:
    cluster backends such as Ray and Spark serialize closures that stdlib
    ``pickle`` rejects, so a pickle gate would invent failures that no real
    backend has. The question is what state crosses, not how it is encoded.

    Pickling is also not a usable signal here: a :class:`BatchRunManifest`
    round-trips to an object that compares *equal* but is not *identical*, so
    even a defensive equality check passes at the moment of transport. The copy
    only diverges once the worker mutates it, which is what makes the loss
    invisible without a structural guard.

    Matching is by type against ``forbidden_types``, plus the documented
    heuristic in :func:`_looks_like_manifest_store` for custom stores.
    """
    found: list[str] = []
    seen: set[int] = set()
    # ``seen`` memoizes by ``id()`` to stop cycles and exponential re-walks, but
    # an id is only unique among *live* objects. This walker allocates its own
    # temporaries (``list(partial.args)``, ``dict(partial.keywords)``), and once
    # one is freed CPython hands its address to the next allocation -- so a
    # later node inherits a memoized id and is skipped without ever being
    # inspected, and the guard returns "clean" for a payload it should reject.
    # Holding every visited node alive for the duration of the walk makes the
    # ids unique in practice as well as in principle.
    visited: list[Any] = []
    forbidden = tuple(forbidden_types)
    budget = [_MAX_STATE_NODES]

    def walk_callable_defaults(node: Any, path: str, depth: int) -> None:
        """Inspect a function's captured state without ever calling it."""
        for cell_index, cell in enumerate(_closure_cells(node)):
            try:
                contents = cell.cell_contents
            except ValueError:  # empty cell
                continue
            walk(contents, f"{path}.<closure {cell_index}>", depth + 1)
        defaults = _safe_getattr(node, "__defaults__")
        if isinstance(defaults, (tuple, list)):
            for index, item in enumerate(defaults):
                walk(item, f"{path}.<default {index}>", depth + 1)
        kwdefaults = _safe_getattr(node, "__kwdefaults__")
        if isinstance(kwdefaults, dict):
            for key, item in kwdefaults.items():
                walk(item, f"{path}.<default {key}>", depth + 1)

    def walk(node: Any, path: str, depth: int) -> None:
        if node is None or depth > _MAX_STATE_DEPTH or budget[0] <= 0:
            return
        budget[0] -= 1
        if isinstance(node, _SCALARS):
            return
        marker = id(node)
        if marker in seen:
            return
        seen.add(marker)
        visited.append(node)

        try:
            if isinstance(node, forbidden) or _looks_like_manifest_store(node):
                found.append(f"{path} ({type(node).__name__})")
                return
        except Exception:  # pragma: no cover - hostile __class__/metaclass
            return

        if isinstance(node, Mapping):
            try:
                items = list(node.items())
            except Exception:  # pragma: no cover - hostile mapping
                items = []
            for key, item in items:
                walk(item, f"{path}[{key!r}]", depth + 1)
            return
        if isinstance(node, (list, tuple, set, frozenset)):
            for index, item in enumerate(node):
                walk(item, f"{path}[{index}]", depth + 1)
            return

        if isinstance(node, functools.partial):
            walk(node.func, f"{path}.func", depth + 1)
            walk(list(node.args), f"{path}.args", depth + 1)
            walk(dict(node.keywords), f"{path}.keywords", depth + 1)
            return

        bound_self = _safe_getattr(node, "__self__")
        if bound_self is not None:
            walk(bound_self, f"{path}.__self__", depth + 1)

        walk_callable_defaults(node, path, depth)

        if is_dataclass(node) and not isinstance(node, type):
            for field in fields(node):
                walk(
                    _safe_getattr(node, field.name),
                    f"{path}.{field.name}",
                    depth + 1,
                )
            return

        # Instance state: ``__dict__`` for ordinary objects, and the declared
        # slot descriptors for ``__slots__`` classes, which have no ``__dict__``.
        instance_dict = _safe_getattr(node, "__dict__")
        if isinstance(instance_dict, dict):
            for key, item in list(instance_dict.items()):
                walk(item, f"{path}.{key}", depth + 1)
        for slot in _slot_names(type(node)):
            if slot in {"__dict__", "__weakref__"}:
                continue
            walk(_safe_getattr(node, slot), f"{path}.{slot}", depth + 1)

        # Type-level state: class attributes, and the functions behind
        # properties. Raw class-dict values are read directly so that a
        # ``property`` is inspected structurally rather than invoked -- the
        # guard must never run user code to decide whether to run user code.
        try:
            mro = [klass for klass in type(node).__mro__ if klass is not object]
        except Exception:  # pragma: no cover - exotic metaclass
            mro = []
        for klass in mro:
            for key, item in list(klass.__dict__.items()):
                if key.startswith("__") and key.endswith("__"):
                    continue
                if isinstance(item, property):
                    if item.fget is not None:
                        walk_callable_defaults(
                            item.fget, f"{path}.{key}.fget", depth + 1
                        )
                    continue
                if isinstance(item, (staticmethod, classmethod)):
                    item = item.__func__
                if inspect.isfunction(item):
                    walk_callable_defaults(item, f"{path}.{key}", depth + 1)
                    continue
                walk(item, f"{path}.{key}", depth + 1)

    walk(value, "task", 0)
    return tuple(found)


def reject_driver_only_state(
    tasks: Sequence[ShardTask],
    *,
    forbidden_types: Sequence[type] = DRIVER_ONLY_TYPES,
) -> None:
    """Raise :class:`DriverOnlyStateError` if any task carries driver state."""
    for task in tasks:
        leaked = find_driver_only_state(task, forbidden_types=forbidden_types)
        if leaked:
            raise DriverOnlyStateError(
                "driver-only state would reach a worker and its manifest "
                f"updates would be silently discarded: {', '.join(leaked)}"
            )


def _empty_shard_handler(shard: DocumentShard) -> bytes:
    """Return the canonical payload published for a shard with no documents.

    Shards outside :attr:`~openmed.processing.ShardPlan.non_empty_shards` carry
    no work, so the caller's handler is never invoked for them. They are still
    published, because a completed manifest record requires an output path and
    digest, and because an empty file is what distinguishes "ran, produced
    nothing" from "never ran".
    """
    del shard
    return b""


def shard_output_filename(shard_id: int) -> str:
    """Return the stable, run-relative file name for a shard output."""
    if isinstance(shard_id, bool) or not isinstance(shard_id, int) or shard_id < 0:
        raise ShardExecutionError("shard_id must be a non-negative integer")
    return SHARD_OUTPUT_FILENAME_TEMPLATE.format(shard_id=shard_id)


def current_worker_id() -> str:
    """Return a PHI-free identifier for the current process and thread."""
    return f"pid-{os.getpid()}-thread-{threading.get_ident()}"


#: Manifest labels must be a dotted identifier and bounded in length. Worker
#: output is untrusted here: a handler can raise an exception whose class was
#: built dynamically with an arbitrary ``__name__``, and Ray in particular
#: synthesizes names such as ``RayTaskError(ValueError)``. Passing that
#: straight to a validating constructor would crash the driver *while it is
#: recording the failure*, converting a recoverable shard error into a lost run.
_ERROR_TYPE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")
_MAX_LABEL_LENGTH = 128
UNKNOWN_ERROR_TYPE = "UnknownError"
UNKNOWN_WORKER_ID = "UnknownWorker"


def _safe_error_type(exc: BaseException) -> str:
    """Return a manifest-safe type name for ``exc``, never its message.

    ``error_type`` must be a dotted identifier, so anything else -- a
    dynamically built class, a Ray-style ``RayTaskError(ValueError)`` -- becomes
    a constant rather than crashing the driver as it records the failure.
    """
    name = _safe_getattr(type(exc), "__name__")
    if not isinstance(name, str) or len(name) > _MAX_LABEL_LENGTH:
        return UNKNOWN_ERROR_TYPE
    if not _ERROR_TYPE_PATTERN.match(name):
        return UNKNOWN_ERROR_TYPE
    return name


def _safe_worker_id(value: Any) -> Optional[str]:
    """Return a manifest-safe worker id.

    Worker ids are free-form apart from a length cap and a control-character
    ban, so ``pid-123-thread-456`` passes unchanged; only genuinely invalid
    values from a custom executor are replaced.
    """
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > _MAX_LABEL_LENGTH:
        return UNKNOWN_WORKER_ID
    if _CONTROL_CHARACTERS.search(value):
        return UNKNOWN_WORKER_ID
    return value


def shard_payload_digest(payload: bytes) -> str:
    """Return the ``sha256:`` digest of a shard payload held in memory."""
    return f"sha256:{_sha256_bytes(bytes(payload))}"


#: Windows refuses ``MoveFileEx`` with a sharing violation while another handle
#: holds the destination, so two workers publishing the same shard at the same
#: moment can make the loser's ``os.replace`` fail with ``PermissionError``.
#: POSIX ``rename(2)`` has no such restriction. Duplicate publication is the
#: documented normal path here, not a corner case, so the loser retries rather
#: than failing the shard. The bound matters: a directory that is genuinely
#: unwritable must still surface its error instead of being retried forever.
_REPLACE_RETRY_ATTEMPTS = 12
_REPLACE_RETRY_BACKOFF_SECONDS = 0.01
_MAX_REPLACE_BACKOFF_SECONDS = 0.2


def write_shard_output(
    path: Union[str, Path],
    payload: bytes,
    *,
    hook: Optional[AtomicWriteHook] = None,
) -> tuple[str, int]:
    """Atomically publish a shard payload and return its digest and size.

    The write always replaces the destination rather than appending, so
    re-running a shard converges on byte-identical output. Skipping already
    complete shards is a scheduling decision made by the driver, not by this
    helper.

    A concurrent duplicate writer can make the atomic replace fail with
    ``PermissionError`` on Windows. That is contention, not a fault, so the
    publish is retried with backoff; the payload is republished in full each
    time, from a fresh temporary file, so a retry never appends or leaves a
    partial file behind. If it still fails after the bound, the error is
    raised unchanged -- a genuinely unwritable destination must not be
    mistaken for a lost race.
    """
    if not isinstance(payload, (bytes, bytearray)):
        raise ShardExecutionError("shard handler must return bytes")
    payload_bytes = bytes(payload)
    last_attempt = _REPLACE_RETRY_ATTEMPTS - 1
    for attempt in range(_REPLACE_RETRY_ATTEMPTS):
        try:
            _atomic_write_bytes(Path(path), payload_bytes, hook=hook)
            break
        except PermissionError:
            if attempt == last_attempt:
                raise
            time.sleep(
                min(
                    _REPLACE_RETRY_BACKOFF_SECONDS * (2**attempt),
                    _MAX_REPLACE_BACKOFF_SECONDS,
                )
            )
    return shard_payload_digest(payload_bytes), len(payload_bytes)


@dataclass(frozen=True)
class ShardTask:
    """One unit of shard work handed to an executor.

    ``handler`` must be picklable when the task runs on a process pool, which in
    practice means a module-level function rather than a lambda or closure.
    """

    shard: DocumentShard
    root: Path
    relative_path: str
    handler: ShardHandler
    #: Digest this shard published on an earlier attempt, when there is one.
    #: The worker compares against it *before* replacing the file, so a
    #: non-deterministic handler cannot destroy output that is still good.
    expected_digest: Optional[str] = None

    @property
    def output_path(self) -> Path:
        """Absolute path the worker publishes this shard's output to."""
        return Path(self.root) / self.relative_path


@dataclass(frozen=True)
class ShardExecution:
    """PHI-free outcome reported by a worker for a single shard."""

    shard_id: int
    status: ShardStatus
    output_path: Optional[str] = None
    output_digest: Optional[str] = None
    output_bytes: Optional[int] = None
    worker_id: Optional[str] = None
    duration_seconds: Optional[float] = None
    error_type: Optional[str] = None
    #: Set when the shard recomputed to a different digest than it published
    #: before. The output on disk was left untouched.
    digest_mismatch: bool = False

    @property
    def succeeded(self) -> bool:
        """Whether the shard completed and published its output."""
        return self.status == ShardStatus.COMPLETED

    def to_dict(self) -> dict[str, Any]:
        """Return the PHI-free serialized execution record."""
        return {
            "shard_id": self.shard_id,
            "status": self.status.value,
            "output_path": self.output_path,
            "output_digest": self.output_digest,
            "output_bytes": self.output_bytes,
            "worker_id": self.worker_id,
            "duration_seconds": self.duration_seconds,
            "error_type": self.error_type,
        }


@dataclass(frozen=True)
class ShardRunResult:
    """Run-level reduction of per-shard summaries."""

    manifest: BatchRunManifest
    executions: tuple[ShardExecution, ...] = ()
    skipped_shards: tuple[int, ...] = ()

    @property
    def completed_shards(self) -> tuple[int, ...]:
        """Shard ids that completed during this run."""
        return tuple(
            execution.shard_id for execution in self.executions if execution.succeeded
        )

    @property
    def failed_shards(self) -> tuple[int, ...]:
        """Shard ids that failed during this run."""
        return tuple(
            execution.shard_id
            for execution in self.executions
            if not execution.succeeded
        )

    @property
    def is_complete(self) -> bool:
        """Whether every shard in the manifest has completed successfully."""
        return not self.manifest.pending_shards()

    @property
    def output_bytes(self) -> int:
        """Total bytes published by shards executed during this run."""
        return sum(
            execution.output_bytes or 0
            for execution in self.executions
            if execution.succeeded
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the PHI-free serialized run summary."""
        return {
            "run_id": self.manifest.run_id,
            "shard_count": self.manifest.shard_count,
            "document_count": self.manifest.document_count,
            "plan_fingerprint": self.manifest.plan_fingerprint,
            "completed_shards": list(self.completed_shards),
            "failed_shards": list(self.failed_shards),
            "skipped_shards": list(self.skipped_shards),
            "output_bytes": self.output_bytes,
            "is_complete": self.is_complete,
            "executions": [execution.to_dict() for execution in self.executions],
        }


def execute_shard_task(
    task: ShardTask,
    *,
    hook: Optional[AtomicWriteHook] = None,
) -> ShardExecution:
    """Run one shard and publish its output, returning PHI-free metadata.

    This is the worker-side entry point. It never raises for handler failures:
    the failure is reported as an exception *type* name so that no exception
    message, which may quote document text, can reach the manifest.

    When the task carries an ``expected_digest``, the recomputed payload is
    compared against it *before* the file is replaced. A shard that no longer
    reproduces its earlier bytes is reported and the existing output is left
    exactly as it was, so a non-deterministic handler cannot destroy good data
    on its way to reporting that it is non-deterministic.
    """
    started = time.perf_counter()
    worker_id = current_worker_id()
    try:
        payload = task.handler(task.shard)
        if not isinstance(payload, (bytes, bytearray)):
            raise ShardExecutionError("shard handler must return bytes")
        payload_bytes = bytes(payload)
        digest = shard_payload_digest(payload_bytes)
        if task.expected_digest is not None and digest != task.expected_digest:
            return ShardExecution(
                shard_id=task.shard.shard_id,
                status=ShardStatus.FAILED,
                worker_id=worker_id,
                duration_seconds=time.perf_counter() - started,
                error_type="ShardOutputDigestMismatchError",
                digest_mismatch=True,
            )
        digest, size = write_shard_output(task.output_path, payload_bytes, hook=hook)
    except Exception as exc:  # reported as a PHI-free exception type name
        return ShardExecution(
            shard_id=task.shard.shard_id,
            status=ShardStatus.FAILED,
            worker_id=worker_id,
            duration_seconds=time.perf_counter() - started,
            error_type=_safe_error_type(exc),
        )
    return ShardExecution(
        shard_id=task.shard.shard_id,
        status=ShardStatus.COMPLETED,
        output_path=task.relative_path,
        output_digest=digest,
        output_bytes=size,
        worker_id=worker_id,
        duration_seconds=time.perf_counter() - started,
    )


class ShardExecutor(Protocol):
    """Minimal executor contract used by distributed batch processing."""

    def execute(self, tasks: Sequence[ShardTask]) -> Iterator[ShardExecution]:
        """Run ``tasks`` and yield each PHI-free shard outcome as it finishes."""


class LocalShardExecutor:
    """Local executor running shard tasks sequentially, threaded, or forked.

    With ``use_processes`` unset, ``max_workers=1`` runs in the calling thread,
    which keeps the default path free of pools and makes failures easy to trace,
    and anything above one runs on a thread pool. With ``use_processes`` set,
    every shard runs on a process pool *including* at ``max_workers=1``: asking
    for process isolation and silently getting in-process execution would make
    the setting a lie, and would hide exactly the pickling problems the setting
    exists to surface. Handlers and shards must then be picklable, and ``hook``
    is unsupported.
    """

    def __init__(
        self,
        *,
        max_workers: int = 1,
        use_processes: bool = False,
        hook: Optional[AtomicWriteHook] = None,
        mp_context: Optional[Any] = None,
    ) -> None:
        if isinstance(max_workers, bool) or not isinstance(max_workers, int):
            raise ShardExecutionError("max_workers must be an integer")
        if max_workers < 1:
            raise ShardExecutionError("max_workers must be at least 1")
        if use_processes and hook is not None:
            raise ShardExecutionError(
                "atomic write hooks are not supported on process pools"
            )
        if mp_context is not None and not use_processes:
            raise ShardExecutionError("mp_context requires use_processes=True")
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.hook = hook
        self.mp_context = mp_context

    def execute(self, tasks: Sequence[ShardTask]) -> Iterator[ShardExecution]:
        """Run ``tasks`` and yield each PHI-free shard outcome as it finishes."""
        pending = list(tasks)
        if not pending:
            return iter(())
        # Eager, before any task is submitted, and on every transport. The
        # invariant is "the driver is the sole manifest writer", not "state
        # must not be copied": a thread worker shares the store yet its writes
        # are still discarded, because the manifest is frozen and the driver
        # re-saves its own local reference.
        reject_driver_only_state(pending)
        if self.max_workers == 1 and not self.use_processes:
            return self._execute_sequentially(pending)
        return self._execute_concurrently(pending)

    def _execute_sequentially(
        self,
        tasks: Sequence[ShardTask],
    ) -> Iterator[ShardExecution]:
        for task in tasks:
            yield execute_shard_task(task, hook=self.hook)

    def _execute_concurrently(
        self,
        tasks: Sequence[ShardTask],
    ) -> Iterator[ShardExecution]:
        worker_count = min(self.max_workers, len(tasks))
        if self.use_processes:
            executor: Any = ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=self.mp_context,
            )
        else:
            executor = ThreadPoolExecutor(max_workers=worker_count)
        futures: dict[Any, ShardTask] = {}
        try:
            if self.use_processes:
                futures = {
                    executor.submit(execute_shard_task, task): task for task in tasks
                }
            else:
                futures = {
                    executor.submit(execute_shard_task, task, hook=self.hook): task
                    for task in tasks
                }
            for future in as_completed(futures):
                yield future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)


def run_shard_plan(
    plan: ShardPlan,
    handler: ShardHandler,
    *,
    manifest: BatchRunManifest,
    root: Union[str, Path],
    store: Optional[RunManifestStore] = None,
    executor: Optional[ShardExecutor] = None,
    force: bool = False,
    hook: Optional[AtomicWriteHook] = None,
) -> ShardRunResult:
    """Execute the outstanding shards of ``plan`` and reduce them to a result.

    Only shards that still need work are dispatched, as decided by the manifest
    and the shard outputs already on disk; ``force`` re-runs every shard. The
    driver owns the manifest: it increments ``attempts`` at dispatch, marks the
    shard ``RUNNING``, and records the outcome once the worker reports back.
    Retry policy, straggler detection, and speculative execution are deliberately
    not implemented here -- a shard that fails is simply eligible again on the
    next run.

    Shards with no documents never reach ``handler``; they publish an empty
    output so that they settle as completed instead of being re-queued by every
    later run.

    ``root`` must belong to this run alone. Two runs pointed at one directory
    overwrite each other's shard files and neither ever converges.

    If any shard fails to reproduce a digest it published earlier, the existing
    outputs are left untouched, every shard is still recorded, and
    :class:`~openmed.processing.ShardOutputDigestMismatchError` is raised once
    at the end -- a non-deterministic handler is an error to surface, not a
    reason to destroy good output or to abandon shards that did succeed.
    """
    if manifest.plan_fingerprint != plan.fingerprint:
        raise ShardExecutionError("Run manifest does not describe this shard plan")
    if executor is not None and hook is not None:
        raise ShardExecutionError(
            "hook applies to the default executor; configure it on a custom one"
        )

    resolved_root = Path(root)
    resolved_root.mkdir(parents=True, exist_ok=True)

    planned_shards = {shard.shard_id: shard for shard in plan.shards}
    if force:
        target_ids: tuple[int, ...] = tuple(sorted(planned_shards))
    else:
        target_ids = tuple(shards_to_execute(manifest, root=resolved_root))

    unknown_ids = [
        shard_id for shard_id in target_ids if shard_id not in planned_shards
    ]
    if unknown_ids:
        raise ShardExecutionError(
            f"Run manifest references shard ids absent from the plan: {unknown_ids}"
        )

    executed = set(target_ids)
    skipped = tuple(
        shard_id for shard_id in sorted(planned_shards) if shard_id not in executed
    )

    previous_digests = {
        shard_id: manifest.shard(shard_id).output_digest for shard_id in target_ids
    }

    tasks = [
        ShardTask(
            shard=planned_shards[shard_id],
            root=resolved_root,
            relative_path=shard_output_filename(shard_id),
            handler=(
                _empty_shard_handler if planned_shards[shard_id].is_empty else handler
            ),
            expected_digest=previous_digests.get(shard_id),
        )
        for shard_id in target_ids
    ]

    # Validate before any bookkeeping. A rejected batch must not leave shards
    # marked RUNNING with a burned attempt when nothing was ever dispatched.
    reject_driver_only_state(tasks)

    for task in tasks:
        record = manifest.shard(task.shard.shard_id)
        manifest = manifest.with_shard(
            replace(
                record,
                status=ShardStatus.RUNNING,
                attempts=record.attempts + 1,
                started_at=time.time(),
                completed_at=None,
                error_type=None,
            )
        )
        if store is not None:
            store.save(manifest)

    active_executor = (
        executor if executor is not None else LocalShardExecutor(hook=hook)
    )

    executions: list[ShardExecution] = []
    mismatched: list[int] = []
    for execution in active_executor.execute(tasks):
        executions.append(execution)
        if execution.digest_mismatch:
            mismatched.append(execution.shard_id)
        record = manifest.shard(execution.shard_id)
        # A failed attempt must not erase what an earlier successful attempt
        # published. Clearing the digest here would let one transient IOError
        # launder a later non-deterministic result past the comparison.
        succeeded = execution.succeeded
        manifest = manifest.with_shard(
            replace(
                record,
                status=execution.status,
                completed_at=time.time(),
                output_path=(
                    execution.output_path if succeeded else record.output_path
                ),
                output_digest=(
                    execution.output_digest if succeeded else record.output_digest
                ),
                output_bytes=(
                    execution.output_bytes if succeeded else record.output_bytes
                ),
                worker_id=_safe_worker_id(execution.worker_id),
                error_type=execution.error_type,
            )
        )
        if store is not None:
            store.save(manifest)

    if mismatched:
        # Raised only after every shard has been recorded and persisted, so the
        # manifest is consistent and the shards that did succeed keep their
        # completed records. Aborting mid-loop would discard them along with
        # the good bytes still sitting on disk.
        raise ShardOutputDigestMismatchError(
            "shards did not reproduce their recorded output digests; existing "
            f"outputs were left untouched: {sorted(mismatched)}"
        )

    return ShardRunResult(
        manifest=manifest,
        executions=tuple(sorted(executions, key=lambda item: item.shard_id)),
        skipped_shards=skipped,
    )
