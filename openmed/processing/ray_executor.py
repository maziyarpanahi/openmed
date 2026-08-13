"""Ray adapter for the distributed batch :class:`ShardExecutor` protocol.

``ray`` is imported lazily inside :meth:`RayShardExecutor.execute`, never at
module scope, so importing :mod:`openmed.processing` never imports Ray. The
optional dependency installs with ``openmed[ray]``.

Driver ownership
----------------
The worker side is :func:`~openmed.processing.shard_executor.execute_shard_task`
-- a module-level function taking a plain :class:`ShardTask` and returning
PHI-free metadata. Nothing else crosses to a Ray worker. That is the structural
control: a manifest store cannot be smuggled to a worker because nothing that
could hold one is ever submitted. :func:`reject_driver_only_state` runs eagerly
before the first submission as defence in depth, not as the primary mechanism.

For the same reason this module must never read a manifest store, or any other
driver-owned object, from a module-level global. A global is re-created by
import in each Ray worker process, so every worker would mutate its own private
instance while the driver's manifest stayed behind -- the same silent
divergence, but invisible to any payload inspection.

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
execution carrying only an exception *type* name, so exceptions reaching this
module are infrastructure failures: a dead worker, a lost object, a failed
deserialization. Ray wraps those in a dynamically built ``RayTaskError``
subclass whose ``__name__`` is synthesized from the cause and is not a bare
identifier, which the manifest sanitizer maps to ``UnknownError``.
:func:`_remote_error_type` recovers the underlying type name from the ``cause``
attribute Ray actually populates -- not ``__cause__``, which Ray leaves unset.
Without it every remote failure would record ``UnknownError`` and, since
exception messages are forbidden, the manifest would lose failure-type
discrimination entirely for this backend.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from importlib import import_module as _import_module
from typing import Any, Optional

from .shard_executor import (
    UNKNOWN_ERROR_TYPE,
    ShardExecution,
    ShardExecutionError,
    ShardStatus,
    ShardTask,
    _safe_error_type,
    _safe_getattr,
    execute_shard_task,
    reject_driver_only_state,
)

__all__ = ["RayShardExecutor"]


def _load_ray() -> Any:
    """Return the ``ray`` module, or raise an actionable ImportError."""

    try:
        return _import_module("ray")
    except ImportError as exc:
        raise ImportError(
            "Ray support requires the optional dependency; install "
            "openmed[ray] to use openmed.processing.ray_executor"
        ) from exc


def _remote_error_type(exc: BaseException) -> str:
    """Return a manifest-safe type name for a Ray-side infrastructure failure.

    Ray reports a remote failure as a dynamically built class that subclasses
    both ``RayTaskError`` and the original exception type, named for example
    ``RayTaskError(ValueError)``. That is not a bare identifier, so the shared
    sanitizer maps it to ``UnknownError`` and every remote failure in a run
    would read the same. Since exception messages are forbidden in the
    manifest, ``error_type`` is the only failure signal there is.

    Ray stores the original exception in a plain ``cause`` **attribute**
    (``ray.exceptions.RayTaskError.__init__``), not in ``__cause__``:
    ``__cause__`` is a ``BaseException`` getset descriptor, so it resolves to
    ``None`` rather than falling through the class's ``__getattr__``. The raise
    site is not inside an ``except`` block either, so ``__context__`` is also
    unset. ``cause`` is therefore tried *first*; ``__cause__`` and
    ``__context__`` are kept as fallbacks for wrappers that do chain normally.

    Every hop is read through :func:`_safe_getattr`. This runs while a shard
    failure is being recorded, and an exception object with a raising
    ``__cause__`` property must not take the driver down at exactly that point.
    """

    # ``seen`` memoizes by ``id()`` to stop a cycle looping forever, but an id
    # is only unique among *live* objects. A wrapper whose ``cause`` is a
    # computed property hands back a fresh exception each read, so the previous
    # hop is freed as soon as the walk moves on and CPython reissues its address
    # -- the next hop then inherits a memoized id and the walk stops early,
    # reporting ``UnknownError`` for a chain whose cause was still reachable.
    # That is precisely the diagnostics collapse this function exists to
    # prevent, so every visited node is held alive for the duration of the walk.
    # Mirrors the same fix in ``find_driver_only_state``.
    visited: list[BaseException] = []
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        visited.append(current)
        seen.add(id(current))
        name = _safe_error_type(current)
        if name != UNKNOWN_ERROR_TYPE:
            return name
        nxt = None
        for attribute in ("cause", "__cause__", "__context__"):
            candidate = _safe_getattr(current, attribute)
            if isinstance(candidate, BaseException):
                nxt = candidate
                break
        current = nxt
    return UNKNOWN_ERROR_TYPE


class RayShardExecutor:
    """Run shard tasks as Ray remote tasks, yielding outcomes as they finish.

    With ``auto_init`` left on, a runtime is started or attached to only when
    the caller has not already initialised one. Pass ``auto_init=False`` to make
    that the caller's responsibility entirely, which is what a test wanting no
    implicit runtime should do.
    """

    def __init__(
        self,
        *,
        num_cpus: Optional[float] = None,
        max_in_flight: Optional[int] = None,
        auto_init: bool = True,
        ray_module: Optional[Any] = None,
        **ray_remote_args: Any,
    ) -> None:
        if max_in_flight is not None:
            if isinstance(max_in_flight, bool) or not isinstance(max_in_flight, int):
                raise ShardExecutionError("max_in_flight must be an integer")
            if max_in_flight < 1:
                raise ShardExecutionError("max_in_flight must be at least 1")
        self.num_cpus = num_cpus
        self.max_in_flight = max_in_flight
        self.auto_init = auto_init
        self.ray_remote_args = dict(ray_remote_args)
        self._ray_module = ray_module

    @property
    def ray(self) -> Any:
        """Return the Ray module, importing it lazily on first use."""

        if self._ray_module is None:
            self._ray_module = _load_ray()
        return self._ray_module

    def ensure_available(self) -> None:
        """Import Ray and start or attach to a runtime, raising if either fails.

        :func:`~openmed.processing.run_shard_plan` calls this automatically
        before marking any shard ``RUNNING``. It remains public for operators
        that want to probe a cluster before constructing a run.
        """

        self._prepare()

    def _prepare(self) -> Any:
        """Return a ready Ray module, importing and initialising as configured.

        Initialisation lives here rather than in the generator so a dead or
        unreachable cluster fails at the same point a missing import does. That
        matters more than the import case: anyone reaching for this class has
        Ray installed, so an unavailable runtime is the likelier failure.
        """

        ray = self.ray
        if self.auto_init and not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        return ray

    def _remote_options(self) -> dict[str, Any]:
        options = dict(self.ray_remote_args)
        if self.num_cpus is not None:
            options["num_cpus"] = self.num_cpus
        return options

    def execute(self, tasks: Sequence[ShardTask]) -> Iterator[ShardExecution]:
        """Run ``tasks`` on Ray and yield each PHI-free outcome as it finishes."""

        pending = list(tasks)
        if not pending:
            return iter(())
        # Eager, before any submission, matching the local executor. The
        # invariant is that the driver is the sole manifest writer.
        reject_driver_only_state(pending)
        # Also eager: ``_execute`` is a generator, so importing or initialising
        # inside it would defer the failure to the first pull, part-way through
        # iteration. ``run_shard_plan`` preflights through ``ensure_available``;
        # direct protocol callers still get a fixed failure point here.
        ray = self._prepare()
        return self._execute(pending, ray)

    def _execute(
        self,
        tasks: Sequence[ShardTask],
        ray: Any,
    ) -> Iterator[ShardExecution]:
        options = self._remote_options()
        remote_task = (
            ray.remote(**options)(execute_shard_task)
            if options
            else ray.remote(execute_shard_task)
        )

        limit = self.max_in_flight or len(tasks)
        queued = list(tasks)
        in_flight: dict[Any, ShardTask] = {}

        def submit_next() -> None:
            while queued and len(in_flight) < limit:
                task = queued.pop(0)
                in_flight[remote_task.remote(task)] = task

        submit_next()
        while in_flight:
            ready, _ = ray.wait(list(in_flight), num_returns=1)
            for ref in ready:
                task = in_flight.pop(ref)
                try:
                    yield ray.get(ref)
                except Exception as exc:  # infrastructure failure, not a handler one
                    yield ShardExecution(
                        shard_id=task.shard.shard_id,
                        status=ShardStatus.FAILED,
                        error_type=_remote_error_type(exc),
                    )
            submit_next()
