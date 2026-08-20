"""Deterministic local execution for independent trace files.

Trace files are submitted as paths rather than as loaded records.  A worker
creates a fresh store for each file, so mutable pseudonym state cannot be
shared accidentally between independent inputs or inherited from the driver.
The driver places every worker outcome back into its original input slot;
completion timing therefore never changes the returned order.

Process execution uses the ``spawn`` start method by default.  A handler or
store factory that cannot safely cross a process boundary, or a platform where
spawning is unavailable, uses the bounded sequential path instead.  No
exception message, input path, or handler value is copied into the safe run
report.
"""

from __future__ import annotations

import math
import multiprocessing
import os
import pickle
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

DEFAULT_TRACE_SHARD_SIZE = 1
DEFAULT_TRACE_START_METHOD = "spawn"
UNKNOWN_ERROR_TYPE = "UnknownError"

_SAFE_ERROR_TYPES = frozenset(
    {
        "ArithmeticError",
        "AssertionError",
        "AttributeError",
        "EOFError",
        "FileExistsError",
        "FileNotFoundError",
        "ImportError",
        "IndexError",
        "IsADirectoryError",
        "KeyError",
        "LookupError",
        "MemoryError",
        "NotADirectoryError",
        "OSError",
        "OverflowError",
        "PermissionError",
        "RuntimeError",
        "TimeoutError",
        "TypeError",
        "UnicodeError",
        "ValueError",
    }
)
_EXECUTION_MODES = frozenset({"processes", "sequential"})
_FALLBACK_REASONS = frozenset({"unsafe", "unavailable", "startup_failed"})

StoreT = TypeVar("StoreT")
ResultT = TypeVar("ResultT")

TraceHandler = Callable[[Path, StoreT], ResultT]
TraceStoreFactory = Callable[[], StoreT]
TracePath = str | os.PathLike[str]


class TraceExecutionError(RuntimeError):
    """Raised when trace work cannot be reduced safely."""


class TraceInputError(ValueError):
    """Raised when trace execution configuration is invalid."""


class TraceProcessingError(TraceExecutionError):
    """Raised for a failed input without retaining its exception message."""

    def __init__(self, input_index: int, error_type: str | None) -> None:
        self.input_index = input_index
        self.error_type = _safe_error_type_name(error_type)
        super().__init__(f"trace input {input_index} failed with {self.error_type}")


def _safe_error_type_name(value: Any) -> str:
    """Return an allowlisted exception type suitable for operator output."""
    if isinstance(value, str) and value in _SAFE_ERROR_TYPES:
        return value
    return UNKNOWN_ERROR_TYPE


def _safe_error_type(exc: BaseException) -> str:
    """Extract only a validated type name, never an exception message."""
    try:
        name = type(exc).__name__
    except Exception:  # pragma: no cover - hostile exception metaclass
        return UNKNOWN_ERROR_TYPE
    return _safe_error_type_name(name)


def _validate_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TraceInputError(f"{name} must be a positive integer")
    return value


def _coerce_trace_path(value: Any) -> Path:
    try:
        if not isinstance(value, (str, os.PathLike)):
            raise TypeError
        return Path(value)
    except Exception:
        raise TraceInputError("trace inputs must be path-like") from None


@dataclass(frozen=True, repr=False)
class TraceShard:
    """A bounded, order-preserving group of trace-file inputs.

    ``files`` is intentionally retained only for execution.  :meth:`to_dict`
    and :func:`repr` expose input indexes and counts, not paths that may carry
    sensitive naming information.
    """

    shard_id: int
    files: tuple[Path, ...]
    input_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if isinstance(self.shard_id, bool) or not isinstance(self.shard_id, int):
            raise TraceInputError("shard_id must be a non-negative integer")
        if self.shard_id < 0:
            raise TraceInputError("shard_id must be a non-negative integer")

        normalized_files = tuple(_coerce_trace_path(path) for path in self.files)
        indices = tuple(self.input_indices)
        if not normalized_files or len(normalized_files) != len(indices):
            raise TraceInputError("trace shards must contain matching non-empty inputs")
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in indices
        ):
            raise TraceInputError("input indexes must be non-negative integers")
        if tuple(sorted(indices)) != indices:
            raise TraceInputError("input indexes must be ordered")
        object.__setattr__(self, "files", normalized_files)
        object.__setattr__(self, "input_indices", indices)

    @property
    def paths(self) -> tuple[Path, ...]:
        """Return the paths handed to the worker callback."""
        return self.files

    @property
    def file_count(self) -> int:
        """Return the number of files in this bounded shard."""
        return len(self.files)

    def to_dict(self) -> dict[str, Any]:
        """Return PHI-minimized shard metadata."""
        return {
            "shard_id": self.shard_id,
            "file_count": self.file_count,
            "input_indices": list(self.input_indices),
        }

    def __repr__(self) -> str:
        return (
            f"TraceShard(shard_id={self.shard_id}, "
            f"file_count={self.file_count}, input_indices={self.input_indices!r})"
        )


@dataclass(frozen=True, repr=False)
class TraceFileResult(Generic[ResultT]):
    """Result metadata for one input, aligned by its stable input index."""

    input_index: int
    success: bool
    value: ResultT | None = None
    error_type: str | None = None
    duration_seconds: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.input_index, bool)
            or not isinstance(self.input_index, int)
            or self.input_index < 0
        ):
            raise TraceInputError("input_index must be a non-negative integer")
        if not isinstance(self.success, bool):
            raise TraceInputError("success must be a boolean")
        if (
            isinstance(self.duration_seconds, bool)
            or not isinstance(self.duration_seconds, (int, float))
            or not math.isfinite(self.duration_seconds)
            or self.duration_seconds < 0
        ):
            raise TraceInputError("duration_seconds must be a finite number")
        normalized_error = _safe_error_type_name(self.error_type)
        if self.success:
            if self.error_type is not None:
                raise TraceInputError("successful results cannot contain errors")
        else:
            object.__setattr__(self, "error_type", normalized_error)

    def to_dict(self) -> dict[str, Any]:
        """Return safe metadata without the callback value or input path."""
        return {
            "input_index": self.input_index,
            "success": self.success,
            "error_type": self.error_type,
            "duration_seconds": self.duration_seconds,
        }

    def __repr__(self) -> str:
        return (
            f"TraceFileResult(input_index={self.input_index}, "
            f"success={self.success}, error_type={self.error_type!r}, "
            f"duration_seconds={self.duration_seconds!r})"
        )


@dataclass(frozen=True, repr=False)
class TraceRunResult(Generic[ResultT]):
    """Ordered trace results plus a PHI-minimized execution summary."""

    items: tuple[TraceFileResult[ResultT], ...]
    shard_count: int
    worker_count: int
    execution_mode: str
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.execution_mode, str)
            or self.execution_mode not in _EXECUTION_MODES
        ):
            raise TraceInputError("execution_mode is not recognized")
        if (
            isinstance(self.worker_count, bool)
            or not isinstance(self.worker_count, int)
            or self.worker_count < 0
        ):
            raise TraceInputError("worker_count must be non-negative")
        if (
            isinstance(self.shard_count, bool)
            or not isinstance(self.shard_count, int)
            or self.shard_count < 0
        ):
            raise TraceInputError("shard_count must be non-negative")
        if self.fallback_reason not in (None, *_FALLBACK_REASONS):
            raise TraceInputError("fallback_reason is not recognized")
        if self.execution_mode == "processes" and self.fallback_reason is not None:
            raise TraceInputError("process execution cannot have a fallback reason")

    @property
    def file_results(self) -> tuple[TraceFileResult[ResultT], ...]:
        """Return results in the same order as the input files."""
        return self.items

    @property
    def successful_items(self) -> tuple[TraceFileResult[ResultT], ...]:
        """Return successful file results without changing input ordering."""
        return tuple(item for item in self.items if item.success)

    @property
    def failed_items(self) -> tuple[TraceFileResult[ResultT], ...]:
        """Return failed file results without exposing exception messages."""
        return tuple(item for item in self.items if not item.success)

    @property
    def successful_files(self) -> int:
        """Return the number of successful inputs."""
        return len(self.successful_items)

    @property
    def failed_files(self) -> int:
        """Return the number of failed inputs."""
        return len(self.failed_items)

    @property
    def is_complete(self) -> bool:
        """Whether every input completed successfully."""
        return not self.failed_items

    @property
    def ordered_values(self) -> tuple[ResultT | None, ...]:
        """Return callback values in input order, including ``None`` failures."""
        return tuple(item.value for item in self.items)

    def raise_for_errors(self) -> None:
        """Raise a safe error for the first failed input, if any."""
        if self.failed_items:
            first = self.failed_items[0]
            raise TraceProcessingError(first.input_index, first.error_type)

    def to_dict(self) -> dict[str, Any]:
        """Return an operator-safe report without paths, values, or messages."""
        return {
            "file_count": len(self.items),
            "shard_count": self.shard_count,
            "worker_count": self.worker_count,
            "execution_mode": self.execution_mode,
            "fallback_reason": self.fallback_reason,
            "successful_files": self.successful_files,
            "failed_files": self.failed_files,
            "items": [item.to_dict() for item in self.items],
        }

    def __repr__(self) -> str:
        return (
            f"TraceRunResult(file_count={len(self.items)}, "
            f"shard_count={self.shard_count}, worker_count={self.worker_count}, "
            f"execution_mode={self.execution_mode!r}, "
            f"fallback_reason={self.fallback_reason!r}, "
            f"successful_files={self.successful_files}, "
            f"failed_files={self.failed_files})"
        )


@dataclass(frozen=True, repr=False)
class _WorkerOutcome(Generic[ResultT]):
    input_index: int
    success: bool
    value: ResultT | None
    error_type: str | None
    duration_seconds: float


def _new_store() -> dict[str, Any]:
    """Build the default empty per-file store."""
    return {}


def partition_trace_files(
    files: Iterable[TracePath] | TracePath,
    *,
    shard_size: int = DEFAULT_TRACE_SHARD_SIZE,
) -> tuple[TraceShard, ...]:
    """Partition trace paths into bounded shards without reordering them.

    Args:
        files: An iterable of local paths, or one path.
        shard_size: Maximum number of files assigned to one worker task.

    Returns:
        Non-empty shards whose ``input_indices`` cover the original order.

    Raises:
        TraceInputError: If a path or shard size is invalid.
    """
    _validate_positive_int(shard_size, "shard_size")
    if isinstance(files, (str, os.PathLike)):
        raw_files: Iterable[Any] = (files,)
    else:
        try:
            raw_files = iter(files)
        except TypeError:
            raise TraceInputError("files must be an iterable of paths") from None

    normalized = tuple(_coerce_trace_path(path) for path in raw_files)
    shards: list[TraceShard] = []
    for shard_id, start in enumerate(range(0, len(normalized), shard_size)):
        end = min(start + shard_size, len(normalized))
        shards.append(
            TraceShard(
                shard_id=shard_id,
                files=normalized[start:end],
                input_indices=tuple(range(start, end)),
            )
        )
    return tuple(shards)


def _is_process_safe(
    handler: Callable[..., Any],
    store_factory: Callable[..., Any],
) -> bool:
    """Check whether callback references can cross a process boundary."""
    try:
        pickle.dumps((handler, store_factory))
    except Exception:
        return False
    return True


def _process_context(start_method: str | None) -> Any | None:
    """Return a multiprocessing context, or ``None`` when unavailable."""
    method = start_method or DEFAULT_TRACE_START_METHOD
    try:
        if method not in multiprocessing.get_all_start_methods():
            return None
        return multiprocessing.get_context(method)
    except Exception:
        return None


def _execute_trace_shard(
    shard: TraceShard,
    handler: Callable[[Path, Any], Any],
    store_factory: Callable[[], Any],
) -> tuple[_WorkerOutcome[Any], ...]:
    """Run a shard while keeping exception payloads out of the result."""
    outcomes: list[_WorkerOutcome[Any]] = []
    for input_index, path in zip(shard.input_indices, shard.files):
        started = time.perf_counter()
        try:
            store = store_factory()
            value = handler(path, store)
        except Exception as exc:
            outcomes.append(
                _WorkerOutcome(
                    input_index=input_index,
                    success=False,
                    value=None,
                    error_type=_safe_error_type(exc),
                    duration_seconds=time.perf_counter() - started,
                )
            )
        else:
            outcomes.append(
                _WorkerOutcome(
                    input_index=input_index,
                    success=True,
                    value=value,
                    error_type=None,
                    duration_seconds=time.perf_counter() - started,
                )
            )
    return tuple(outcomes)


def _execute_sequentially(
    shards: tuple[TraceShard, ...],
    handler: Callable[[Path, Any], Any],
    store_factory: Callable[[], Any],
) -> tuple[_WorkerOutcome[Any], ...]:
    outcomes: list[_WorkerOutcome[Any]] = []
    for shard in shards:
        outcomes.extend(_execute_trace_shard(shard, handler, store_factory))
    return tuple(outcomes)


def _failed_shard_outcomes(
    shard: TraceShard,
    error_type: str,
) -> tuple[_WorkerOutcome[Any], ...]:
    return tuple(
        _WorkerOutcome(
            input_index=input_index,
            success=False,
            value=None,
            error_type=_safe_error_type_name(error_type),
            duration_seconds=0.0,
        )
        for input_index in shard.input_indices
    )


def _execute_in_processes(
    shards: tuple[TraceShard, ...],
    handler: Callable[[Path, Any], Any],
    store_factory: Callable[[], Any],
    *,
    max_workers: int,
    context: Any,
) -> tuple[_WorkerOutcome[Any], ...] | None:
    """Execute shards in a process pool, returning ``None`` on startup failure."""
    try:
        executor: Any = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=context,
        )
    except Exception:
        return None

    futures: dict[Any, TraceShard] = {}
    shutdown_done = False
    try:
        for shard in shards:
            try:
                future = executor.submit(
                    _execute_trace_shard,
                    shard,
                    handler,
                    store_factory,
                )
            except Exception:
                if futures:
                    raise TraceExecutionError(
                        "parallel trace submission failed after work started"
                    ) from None
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
                shutdown_done = True
                return None
            futures[future] = shard

        outcomes: list[_WorkerOutcome[Any]] = []
        for future in as_completed(futures):
            shard = futures[future]
            try:
                outcomes.extend(future.result())
            except Exception as exc:
                outcomes.extend(_failed_shard_outcomes(shard, _safe_error_type(exc)))
        return tuple(outcomes)
    finally:
        if not shutdown_done:
            try:
                executor.shutdown(wait=True, cancel_futures=True)
            except Exception:
                raise TraceExecutionError("parallel trace shutdown failed") from None


def _materialize_results(
    outcomes: Iterable[_WorkerOutcome[Any]],
    *,
    file_count: int,
) -> tuple[TraceFileResult[Any], ...]:
    ordered: list[TraceFileResult[Any] | None] = [None] * file_count
    for outcome in outcomes:
        index = outcome.input_index
        if isinstance(index, bool) or not isinstance(index, int):
            raise TraceExecutionError("worker returned an invalid input index")
        if index < 0 or index >= file_count or ordered[index] is not None:
            raise TraceExecutionError("worker returned duplicate or unknown input")
        ordered[index] = TraceFileResult(
            input_index=index,
            success=outcome.success,
            value=outcome.value,
            error_type=outcome.error_type,
            duration_seconds=outcome.duration_seconds,
        )
    if any(item is None for item in ordered):
        raise TraceExecutionError("worker results did not cover every input")
    return tuple(item for item in ordered if item is not None)


def run_trace_files(
    files: Iterable[TracePath] | TracePath,
    handler: TraceHandler[StoreT, ResultT],
    *,
    store_factory: TraceStoreFactory[StoreT] | None = None,
    shard_size: int = DEFAULT_TRACE_SHARD_SIZE,
    max_workers: int | None = None,
    use_processes: bool = True,
    start_method: str | None = DEFAULT_TRACE_START_METHOD,
    raise_on_error: bool = False,
) -> TraceRunResult[ResultT]:
    """Process independent trace files deterministically.

    ``handler`` receives ``(path, store)``.  A new store is created for every
    input file, including files handled by the same worker shard.  Handlers
    should return values that are safe for their caller to retain; this module
    never places those values in :meth:`TraceRunResult.to_dict`.

    Process execution is preferred by default.  The handler and store factory
    are checked for picklability before a pool is created.  If that check or
    process startup is unsafe, the function runs all bounded shards in one
    process and records the reason in ``fallback_reason``.  A failure after
    process work has started is reported as a failed input rather than being
    replayed in a second process, avoiding duplicate side effects.

    Args:
        files: Local trace paths in the desired merge order.
        handler: Pure or idempotent callback receiving one path and fresh store.
        store_factory: Callable returning a new per-file store.  The default is
            an empty dictionary.
        shard_size: Maximum number of files in one worker task.
        max_workers: Maximum process count, capped at the shard count.  When
            omitted, the local CPU count is used as the upper bound.
        use_processes: Disable process execution explicitly when false.
        start_method: Multiprocessing start method, defaulting to ``spawn``.
        raise_on_error: Raise :class:`TraceProcessingError` after collecting
            safe failure metadata when true.

    Returns:
        :class:`TraceRunResult` with results aligned to input order.
    """
    if not callable(handler):
        raise TraceInputError("handler must be callable")
    if store_factory is not None and not callable(store_factory):
        raise TraceInputError("store_factory must be callable")
    if not isinstance(use_processes, bool):
        raise TraceInputError("use_processes must be a boolean")
    if not isinstance(raise_on_error, bool):
        raise TraceInputError("raise_on_error must be a boolean")
    if start_method is not None and not isinstance(start_method, str):
        raise TraceInputError("start_method must be a string or None")

    shards = partition_trace_files(files, shard_size=shard_size)
    file_count = sum(shard.file_count for shard in shards)
    if max_workers is None:
        max_workers = os.cpu_count() or 1
    _validate_positive_int(max_workers, "max_workers")
    factory: Callable[[], Any] = (
        store_factory if store_factory is not None else _new_store
    )

    if not shards:
        result: TraceRunResult[ResultT] = TraceRunResult(
            items=(),
            shard_count=0,
            worker_count=0,
            execution_mode="sequential",
        )
        if raise_on_error:
            result.raise_for_errors()
        return result

    bounded_workers = min(max_workers, len(shards))
    fallback_reason: str | None = None
    if not use_processes:
        outcomes = _execute_sequentially(shards, handler, factory)
        execution_mode = "sequential"
        worker_count = 1
    else:
        context = _process_context(start_method)
        if context is None:
            fallback_reason = "unavailable"
            outcomes = _execute_sequentially(shards, handler, factory)
            execution_mode = "sequential"
            worker_count = 1
        elif not _is_process_safe(handler, factory):
            fallback_reason = "unsafe"
            outcomes = _execute_sequentially(shards, handler, factory)
            execution_mode = "sequential"
            worker_count = 1
        else:
            process_outcomes = _execute_in_processes(
                shards,
                handler,
                factory,
                max_workers=bounded_workers,
                context=context,
            )
            if process_outcomes is None:
                fallback_reason = "startup_failed"
                outcomes = _execute_sequentially(shards, handler, factory)
                execution_mode = "sequential"
                worker_count = 1
            else:
                outcomes = process_outcomes
                execution_mode = "processes"
                worker_count = bounded_workers

    items = _materialize_results(outcomes, file_count=file_count)
    result = TraceRunResult(
        items=items,
        shard_count=len(shards),
        worker_count=worker_count,
        execution_mode=execution_mode,
        fallback_reason=fallback_reason,
    )
    if raise_on_error:
        result.raise_for_errors()
    return result


# Name the operation both ways users naturally describe it.  They are aliases,
# not separate implementations, so their ordering and fallback contracts stay
# identical.
run_trace_shards = run_trace_files
process_trace_files = run_trace_files


__all__ = [
    "DEFAULT_TRACE_SHARD_SIZE",
    "DEFAULT_TRACE_START_METHOD",
    "TraceExecutionError",
    "TraceFileResult",
    "TraceHandler",
    "TraceInputError",
    "TracePath",
    "TraceProcessingError",
    "TraceRunResult",
    "TraceShard",
    "TraceStoreFactory",
    "partition_trace_files",
    "process_trace_files",
    "run_trace_files",
    "run_trace_shards",
]
