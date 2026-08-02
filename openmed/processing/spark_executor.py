"""Spark adapter for the distributed batch :class:`ShardExecutor` protocol.

``pyspark`` is imported lazily inside :meth:`SparkShardExecutor.execute`, never
at module scope, so importing :mod:`openmed.processing` never imports Spark. The
optional dependency installs with ``openmed[spark]``.

Driver ownership
----------------
The worker side is :func:`~openmed.processing.shard_executor.execute_shard_task`
-- a module-level function taking a plain :class:`ShardTask` and returning
PHI-free metadata. Nothing else crosses to a Spark executor. That is the
structural control: a manifest store cannot be smuggled to a worker because
nothing that could hold one is ever submitted. :func:`reject_driver_only_state`
runs eagerly before the job is submitted as defence in depth.

As with the Ray adapter, this module must never read a manifest store, or any
other driver-owned object, from a module-level global: a global is re-created by
import in each executor process, so every executor would mutate its own private
copy while the driver's manifest stayed behind, with no payload to inspect.

Shard-to-result attribution
---------------------------
``collect()`` returns partition-ordered results regardless of the order
executors finished in, so results are attributed by the ``shard_id`` each
execution carries rather than by completion order or list position. The shard
plan is parallelized with one slice per shard so a slow shard cannot block an
unrelated one behind it in the same partition.

Retries and duplicate execution
-------------------------------
Neither adapter configures retry policy. Ray retries a task on worker death by
default (``max_retries``), and Spark may run speculative duplicates, so a shard
can execute more than once. That is safe here because the write path is
idempotent: the worker publishes through a temp file and an atomic replace, and
a re-executed shard is compared against ``expected_digest`` before its output is
replaced. A handler that is not deterministic is the one case this does not
cover, and it is reported rather than silently absorbed.

Error reporting
---------------
``execute_shard_task`` already converts handler failures into a ``FAILED``
execution carrying only an exception type name, so a raised exception here is
an infrastructure failure. Spark surfaces those as ``Py4JJavaError`` or
``SparkException``, both of which are valid identifiers and survive the shared
sanitizer intact -- unlike Ray, this backend needs no cause unwrapping. Because
``collect()`` is all-or-nothing, a job that dies yields no partial results, so
the failure is attributed to every shard in the job rather than to a subset.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from importlib import import_module as _import_module
from typing import Any, Optional

from .shard_executor import (
    ShardExecution,
    ShardExecutionError,
    ShardStatus,
    ShardTask,
    _safe_error_type,
    execute_shard_task,
    reject_driver_only_state,
)

__all__ = ["SparkShardExecutor"]


def _load_spark_session() -> Any:
    """Return the ``SparkSession`` class, or raise an actionable ImportError."""

    try:
        module = _import_module("pyspark.sql")
    except ImportError as exc:
        raise ImportError(
            "Spark support requires the optional dependency; install "
            "openmed[spark] to use openmed.processing.spark_executor"
        ) from exc
    return module.SparkSession


class SparkShardExecutor:
    """Run shard tasks as a Spark job, yielding each PHI-free outcome.

    A ``SparkSession`` is never created implicitly: sessions are expensive,
    process-global, and configured by the operator, so building one as a side
    effect of running a shard plan would be the wrong default. Pass ``session``,
    or leave it unset to reuse the active session when one exists.
    """

    def __init__(
        self,
        *,
        session: Optional[Any] = None,
        num_slices: Optional[int] = None,
    ) -> None:
        if num_slices is not None:
            if isinstance(num_slices, bool) or not isinstance(num_slices, int):
                raise ShardExecutionError("num_slices must be an integer")
            if num_slices < 1:
                raise ShardExecutionError("num_slices must be at least 1")
        self.session = session
        self.num_slices = num_slices

    def ensure_available(self) -> None:
        """Resolve the Spark session now, raising if it is unavailable.

        Call this **before** :func:`~openmed.processing.run_shard_plan` to keep
        an unusable backend from costing a manifest attempt. ``run_shard_plan``
        marks every shard ``RUNNING`` and increments ``attempts`` before it
        calls the executor at all, so by the time :meth:`execute` runs the
        bookkeeping is already durable. Resolving eagerly inside :meth:`execute`
        therefore does *not* protect the manifest -- it only guarantees the
        failure surfaces at a fixed point with no job submitted. This method is
        the part that protects the manifest, and only when called first.
        """

        self._resolve_session()

    def _resolve_session(self) -> Any:
        if self.session is not None:
            return self.session
        spark_session = _load_spark_session()
        active = spark_session.getActiveSession()
        if active is None:
            raise ShardExecutionError(
                "no active SparkSession; pass session=... to SparkShardExecutor"
            )
        return active

    def execute(self, tasks: Sequence[ShardTask]) -> Iterator[ShardExecution]:
        """Run ``tasks`` on Spark and yield each PHI-free shard outcome."""

        pending = list(tasks)
        if not pending:
            return iter(())
        # Eager, before the job is submitted, matching the local executor.
        reject_driver_only_state(pending)
        # Also eager: ``_execute`` is a generator, so resolving inside it would
        # defer the failure to the first pull. This does not save the manifest
        # -- ``run_shard_plan`` has already burned an attempt per shard by now
        # -- it only makes the failure land at a fixed point with no job
        # submitted. Call ``ensure_available`` first to protect the manifest.
        session = self._resolve_session()
        return self._execute(pending, session)

    def _execute(
        self,
        tasks: Sequence[ShardTask],
        session: Any,
    ) -> Iterator[ShardExecution]:
        context = session.sparkContext
        slices = self.num_slices or len(tasks)

        try:
            executions = (
                context.parallelize(tasks, slices).map(execute_shard_task).collect()
            )
        except Exception as exc:  # infrastructure failure, not a handler one
            error_type = _safe_error_type(exc)
            for task in tasks:
                yield ShardExecution(
                    shard_id=task.shard.shard_id,
                    status=ShardStatus.FAILED,
                    error_type=error_type,
                )
            return

        yield from executions
