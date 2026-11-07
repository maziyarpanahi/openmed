"""Backend fakes for the distributed shard-executor conformance suite.

Ray and PySpark are optional backends that are absent from the default test
environment, so the conformance suite drives the adapters through these fakes.
They reproduce the three behaviours that actually break real cluster runs:

* **the serialization boundary** -- every task payload crosses a pickle round
  trip before it "leaves the driver", which is where non-serializable state
  (open handles, locks, live sessions) fails on a real cluster;
* **out-of-order completion** -- workers finish in an arbitrary order while
  each result must still be attributed to the shard that produced it;
* **per-task failure** -- one task raising must surface a PHI-free error type,
  never the payload or the exception message that caused it.

What these fakes deliberately do **not** emulate, because it needs a real
cluster: scheduling and resource placement, object-store spilling, executor
loss and re-execution, and the exact ``cloudpickle`` extensions Ray and Spark
install for their own types.

Serializer fidelity
-------------------
Ray and Spark serialize task *functions* with ``cloudpickle``, which handles
lambdas and locally-defined closures that stdlib ``pickle`` rejects. Applying a
stdlib-``pickle`` gate to functions would therefore invent failures that a real
cluster never sees. These fakes consequently split the two concerns: task
*arguments* always cross a stdlib ``pickle`` round trip (a real constraint on
both backends), while task *functions* are only serialized when the caller
supplies an explicit ``function_serializer``.

Every value produced here is synthetic and generated algorithmically.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from openmed.processing.shard_executor import (
    DRIVER_ONLY_TYPES,
    DriverOnlyStateError,
    find_driver_only_state,
)

__all__ = [
    "DriverOnlyStateError",
    "FakeObjectRef",
    "FakeRDD",
    "FakeRayModule",
    "FakeRayTaskError",
    "FakeSparkContext",
    "FakeSparkSession",
    "FakeSparkTaskError",
    "make_ray_task_error",
    "SerializationError",
    "TaskOutcome",
    "find_driver_only_state",
    "round_trip",
    "synthetic_shard_payloads",
]


class SerializationError(TypeError):
    """Raised when a task payload cannot cross the driver/worker boundary."""


class FakeRayTaskError(RuntimeError):
    """PHI-free analogue of ``ray.exceptions.RayTaskError``.

    Carries only the failing exception's *type name*; the originating message
    is dropped on purpose so no test can accidentally depend on it and no
    payload text can reach a driver-side log.

    This is **not** the shape Ray raises to a driver -- see
    :func:`make_ray_task_error` for that. Use this one only where a task
    failure just needs to be signalled.
    """

    def __init__(self, error_type: str) -> None:
        super().__init__(f"worker task failed with {error_type}")
        self.error_type = error_type


def make_ray_task_error(cause: BaseException) -> BaseException:
    """Return an exception shaped like the one Ray raises to a driver.

    Modelled on ``ray.exceptions.RayTaskError.as_instanceof_cause`` in
    ray 2.56.1, which is what ``ray.get`` re-raises. Three properties matter,
    and an adapter that assumes ordinary chaining gets all three wrong:

    * the class is built dynamically and named ``RayTaskError(<Cause>)``, which
      is **not** a bare identifier, so a manifest sanitizer maps it to
      ``UnknownError``;
    * the original exception is stored in a plain ``cause`` **attribute**, not
      in ``__cause__``;
    * ``__cause__`` and ``__context__`` are both ``None``. ``__cause__`` is a
      ``BaseException`` getset descriptor, so it resolves normally and never
      falls through the class's ``__getattr__``, and the raise site is not
      inside an ``except`` block so nothing chains implicitly.

    Building the real shape here rather than hand-assigning ``__cause__`` is
    the point: a fake that encodes what we believe Ray does can only ever
    confirm that belief.
    """

    cause_cls = type(cause)
    name = f"RayTaskError({cause_cls.__name__})"

    class _RayTaskError(RuntimeError):
        def __init__(self, inner: BaseException) -> None:
            super().__init__("ray worker task failed")
            self.cause = inner

        def __getattr__(self, attribute: str) -> Any:
            return getattr(self.cause, attribute)

    _RayTaskError.__name__ = name
    _RayTaskError.__qualname__ = name
    return _RayTaskError(cause)


class FakeSparkTaskError(RuntimeError):
    """PHI-free analogue of a Spark executor task failure."""

    def __init__(self, error_type: str) -> None:
        super().__init__(f"executor task failed with {error_type}")
        self.error_type = error_type


def round_trip(payload: Any, *, serializer: Any = pickle) -> Any:
    """Return *payload* after a serializer round trip.

    Mirrors the driver-to-worker hop. Raises :class:`SerializationError` with a
    type-name-only message so the failure itself never carries payload text.
    """

    try:
        encoded = serializer.dumps(payload)
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed error below
        raise SerializationError(
            f"task payload is not serializable: {type(exc).__name__}"
        ) from None
    return serializer.loads(encoded)


def synthetic_shard_payloads(
    count: int,
    *,
    documents_per_shard: int = 4,
) -> tuple[dict[str, Any], ...]:
    """Return *count* algorithmically generated, PHI-free shard payloads."""

    payloads: list[dict[str, Any]] = []
    for shard_id in range(count):
        document_ids = tuple(
            f"doc-{shard_id:03d}-{index:03d}" for index in range(documents_per_shard)
        )
        payloads.append(
            {
                "shard_id": shard_id,
                "fingerprint": f"{shard_id:064x}",
                "document_ids": document_ids,
                "document_count": len(document_ids),
            }
        )
    return tuple(payloads)


@dataclass
class TaskOutcome:
    """Result of one faked worker task."""

    index: int
    value: Any = None
    error_type: str | None = None

    @property
    def failed(self) -> bool:
        """Whether the task raised."""
        return self.error_type is not None


def _ordered(order: Sequence[int] | None, total: int) -> tuple[int, ...]:
    if order is None:
        return tuple(range(total))
    if sorted(order) != list(range(total)):
        raise ValueError("order must be a permutation of the task indices")
    return tuple(order)


# ---------------------------------------------------------------------------
# Ray
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeObjectRef:
    """Analogue of ``ray.ObjectRef`` identifying one submitted task."""

    index: int


@dataclass
class FakeRayModule:
    """Minimal stand-in for the ``ray`` module used by the Ray adapter.

    Supports ``@ray.remote``, ``@ray.remote(**options)``, ``ray.get`` (which
    preserves submission order, as real Ray does) and ``ray.wait`` (which
    yields ``completion_order``, letting a test drive out-of-order completion).
    """

    completion_order: Sequence[int] | None = None
    function_serializer: Any = None
    forbidden_types: tuple[type, ...] = DRIVER_ONLY_TYPES
    #: shard_id -> exception ``ray.get`` should raise instead of returning a
    #: result, standing in for a worker death or a lost object.
    remote_failures: dict[int, BaseException] = field(default_factory=dict)
    outcomes: list[TaskOutcome] = field(default_factory=list)
    submitted_payloads: list[tuple[tuple[Any, ...], dict[str, Any]]] = field(
        default_factory=list
    )
    remote_options: list[dict[str, Any]] = field(default_factory=list)
    initialized: bool = False

    # -- lifecycle ---------------------------------------------------------
    def init(self, **kwargs: Any) -> None:
        """Record that the adapter initialised a Ray runtime."""
        self.initialized = True

    def shutdown(self) -> None:
        """Record that the adapter shut the Ray runtime down."""
        self.initialized = False

    def is_initialized(self) -> bool:
        """Whether :meth:`init` has been called without a later shutdown."""
        return self.initialized

    # -- task submission ---------------------------------------------------
    def remote(self, *args: Any, **options: Any) -> Any:
        """Support both ``@ray.remote`` and ``@ray.remote(**options)``."""

        if args and callable(args[0]) and not options:
            return _FakeRemoteFunction(self, args[0], {})

        def decorate(function: Callable[..., Any]) -> _FakeRemoteFunction:
            self.remote_options.append(dict(options))
            return _FakeRemoteFunction(self, function, dict(options))

        return decorate

    def get(self, refs: FakeObjectRef | Iterable[FakeObjectRef]) -> Any:
        """Return task results in submission order, raising on failure."""

        if isinstance(refs, FakeObjectRef):
            return self._resolve(refs)
        return [self._resolve(ref) for ref in refs]

    def wait(
        self,
        refs: Sequence[FakeObjectRef],
        *,
        num_returns: int = 1,
        timeout: float | None = None,
    ) -> tuple[list[FakeObjectRef], list[FakeObjectRef]]:
        """Split *refs* into (ready, pending) following ``completion_order``.

        ``completion_order`` is a *priority hint* over whatever is currently in
        flight, not a required permutation of the whole plan. With a bounded
        ``max_in_flight`` only a window of tasks has been submitted when this is
        first called, so demanding a full permutation would make the one
        realistic Ray configuration -- a bounded window completing out of order
        -- impossible to express. Indices absent from the hint sort last, by id.
        """

        pending = list(refs)
        hint = list(self.completion_order or ())

        def rank(ref: FakeObjectRef) -> tuple[int, int]:
            if ref.index in hint:
                return (0, hint.index(ref.index))
            return (1, ref.index)

        ranked = sorted(pending, key=rank)
        ready = ranked[:num_returns]
        remaining = [ref for ref in pending if ref not in ready]
        return ready, remaining

    def _resolve(self, ref: FakeObjectRef) -> Any:
        outcome = self.outcomes[ref.index]
        shard_id = getattr(outcome.value, "shard_id", None)
        if shard_id in self.remote_failures:
            raise self.remote_failures[shard_id]
        if outcome.error_type is not None:
            raise FakeRayTaskError(outcome.error_type)
        return outcome.value


@dataclass
class _FakeRemoteFunction:
    """Callable returned by :meth:`FakeRayModule.remote`."""

    module: FakeRayModule
    function: Callable[..., Any]
    options: dict[str, Any]

    def remote(self, *args: Any, **kwargs: Any) -> FakeObjectRef:
        """Submit one task, crossing the serialization boundary first."""

        captured = find_driver_only_state(
            (args, kwargs), forbidden_types=self.module.forbidden_types
        )
        if captured:
            raise DriverOnlyStateError(
                "driver-only state captured in a Ray task payload: "
                + ", ".join(captured)
            )
        args = tuple(round_trip(arg) for arg in args)
        kwargs = {key: round_trip(value) for key, value in kwargs.items()}
        self.module.submitted_payloads.append((args, dict(kwargs)))

        function = self.function
        if self.module.function_serializer is not None:
            function = round_trip(function, serializer=self.module.function_serializer)

        index = len(self.module.outcomes)
        try:
            value = function(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - reduced to a PHI-free type name
            self.module.outcomes.append(
                TaskOutcome(index=index, error_type=type(exc).__name__)
            )
        else:
            self.module.outcomes.append(TaskOutcome(index=index, value=value))
        return FakeObjectRef(index=index)


# ---------------------------------------------------------------------------
# Spark
# ---------------------------------------------------------------------------


@dataclass
class FakeRDD:
    """Minimal stand-in for a PySpark RDD of shard payloads."""

    elements: tuple[Any, ...]
    context: FakeSparkContext
    _transform: Callable[[Any], Any] | None = None

    def map(self, function: Callable[[Any], Any]) -> FakeRDD:
        """Register the per-element worker function."""

        return FakeRDD(
            elements=self.elements, context=self.context, _transform=function
        )

    def mapPartitions(  # noqa: N802 - mirrors the PySpark API name
        self,
        function: Callable[[Iterable[Any]], Iterable[Any]],
    ) -> FakeRDD:
        """Register a per-partition worker function."""

        def per_element(element: Any) -> Any:
            (result,) = tuple(function(iter([element])))
            return result

        return FakeRDD(
            elements=self.elements, context=self.context, _transform=per_element
        )

    def collect(self) -> list[Any]:
        """Execute in ``execution_order`` and return results in element order.

        Spark's ``collect()`` returns partition-ordered results regardless of
        the order executors finished in, so a correct adapter must not rely on
        completion order to attribute a result to its shard.
        """

        if self.context.job_error is not None:
            raise self.context.job_error
        if self._transform is None:
            return list(self.elements)

        transform = self._transform
        if self.context.function_serializer is not None:
            transform = round_trip(
                transform, serializer=self.context.function_serializer
            )
        total = len(self.elements)
        order = _ordered(self.context.execution_order, total)
        results: dict[int, Any] = {}
        for index in order:
            self.context.invocation_order.append(index)
            try:
                results[index] = transform(self.elements[index])
            except Exception as exc:  # noqa: BLE001 - reduced to a type name
                raise FakeSparkTaskError(type(exc).__name__) from None
        return [results[index] for index in range(total)]


@dataclass
class FakeSparkContext:
    """Minimal stand-in for ``SparkContext`` used by the Spark adapter."""

    execution_order: Sequence[int] | None = None
    num_slices: int | None = None
    forbidden_types: tuple[type, ...] = DRIVER_ONLY_TYPES
    #: Serializer applied to the mapped function, mirroring
    #: ``FakeRayModule.function_serializer``. PySpark ships the function with
    #: ``CloudPickleSerializer`` while ``parallelize`` ships the *elements* with
    #: ``CPickleSerializer`` (stdlib pickle), so the two are configured apart.
    function_serializer: Any = None
    #: Exception ``collect`` should raise instead of returning results, standing
    #: in for a job-level failure such as a lost executor.
    job_error: BaseException | None = None
    invocation_order: list[int] = field(default_factory=list)

    def parallelize(
        self, elements: Iterable[Any], numSlices: int | None = None
    ) -> FakeRDD:  # noqa: N803 - mirrors the PySpark API name
        """Distribute *elements*, crossing the serialization boundary first."""

        self.num_slices = numSlices
        materialized: list[Any] = []
        for element in elements:
            captured = find_driver_only_state(
                element, forbidden_types=self.forbidden_types
            )
            if captured:
                raise DriverOnlyStateError(
                    "driver-only state captured in a Spark task payload: "
                    + ", ".join(captured)
                )
            materialized.append(round_trip(element))
        return FakeRDD(elements=tuple(materialized), context=self)


@dataclass
class FakeSparkSession:
    """Minimal stand-in for ``SparkSession``."""

    execution_order: Sequence[int] | None = None
    forbidden_types: tuple[type, ...] = DRIVER_ONLY_TYPES
    function_serializer: Any = None
    job_error: BaseException | None = None
    _context: FakeSparkContext | None = None

    @property
    def sparkContext(self) -> FakeSparkContext:  # noqa: N802 - PySpark API name
        """Return the lazily created fake context."""

        if self._context is None:
            self._context = FakeSparkContext(
                execution_order=self.execution_order,
                forbidden_types=self.forbidden_types,
                function_serializer=self.function_serializer,
                job_error=self.job_error,
            )
        return self._context
