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
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, is_dataclass
from dataclasses import fields as dataclass_fields
from typing import Any

__all__ = [
    "DriverOnlyStateError",
    "FakeObjectRef",
    "FakeRDD",
    "FakeRayModule",
    "FakeRayTaskError",
    "FakeSparkContext",
    "FakeSparkSession",
    "FakeSparkTaskError",
    "SerializationError",
    "TaskOutcome",
    "find_driver_only_state",
    "round_trip",
    "synthetic_shard_payloads",
]


class SerializationError(TypeError):
    """Raised when a task payload cannot cross the driver/worker boundary."""


class DriverOnlyStateError(AssertionError):
    """Raised when driver-only state is captured in a worker task payload.

    A manifest store is picklable, so shipping one to a worker raises nothing:
    each worker silently receives a *copy*, mutates it, and the driver's
    manifest never observes the update. That divergence is invisible to the
    serializer, so the fakes reject driver-only state structurally instead.
    """


def find_driver_only_state(
    payload: Any,
    forbidden_types: tuple[type, ...],
    *,
    _path: str = "payload",
    _seen: set[int] | None = None,
) -> tuple[str, ...]:
    """Return the paths at which *payload* holds any of *forbidden_types*."""

    if not forbidden_types:
        return ()
    seen = set() if _seen is None else _seen
    if id(payload) in seen:
        return ()
    seen.add(id(payload))

    if isinstance(payload, forbidden_types):
        return (f"{_path}: {type(payload).__name__}",)

    found: list[str] = []
    children: list[tuple[str, Any]] = []
    if isinstance(payload, Mapping):
        children = [(f"{_path}[{key!r}]", value) for key, value in payload.items()]
    elif isinstance(payload, (list, tuple, set, frozenset)):
        children = [(f"{_path}[{index}]", item) for index, item in enumerate(payload)]
    elif is_dataclass(payload) and not isinstance(payload, type):
        children = [
            (f"{_path}.{f.name}", getattr(payload, f.name, None))
            for f in dataclass_fields(payload)
        ]
    elif hasattr(payload, "__dict__"):
        children = [(f"{_path}.{name}", value) for name, value in vars(payload).items()]

    for child_path, child in children:
        found.extend(
            find_driver_only_state(child, forbidden_types, _path=child_path, _seen=seen)
        )
    return tuple(found)


class FakeRayTaskError(RuntimeError):
    """PHI-free analogue of ``ray.exceptions.RayTaskError``.

    Carries only the failing exception's *type name*; the originating message
    is dropped on purpose so no test can accidentally depend on it and no
    payload text can reach a driver-side log.
    """

    def __init__(self, error_type: str) -> None:
        super().__init__(f"worker task failed with {error_type}")
        self.error_type = error_type


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
    forbidden_types: tuple[type, ...] = ()
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
        """Split *refs* into (ready, pending) following ``completion_order``."""

        pending = list(refs)
        order = _ordered(self.completion_order, len(self.outcomes))
        ranked = sorted(pending, key=lambda ref: order.index(ref.index))
        ready = ranked[:num_returns]
        remaining = [ref for ref in pending if ref not in ready]
        return ready, remaining

    def _resolve(self, ref: FakeObjectRef) -> Any:
        outcome = self.outcomes[ref.index]
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

        captured = find_driver_only_state((args, kwargs), self.module.forbidden_types)
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

        if self._transform is None:
            return list(self.elements)

        total = len(self.elements)
        order = _ordered(self.context.execution_order, total)
        results: dict[int, Any] = {}
        for index in order:
            self.context.invocation_order.append(index)
            try:
                results[index] = self._transform(self.elements[index])
            except Exception as exc:  # noqa: BLE001 - reduced to a type name
                raise FakeSparkTaskError(type(exc).__name__) from None
        return [results[index] for index in range(total)]


@dataclass
class FakeSparkContext:
    """Minimal stand-in for ``SparkContext`` used by the Spark adapter."""

    execution_order: Sequence[int] | None = None
    num_slices: int | None = None
    forbidden_types: tuple[type, ...] = ()
    invocation_order: list[int] = field(default_factory=list)

    def parallelize(
        self, elements: Iterable[Any], numSlices: int | None = None
    ) -> FakeRDD:  # noqa: N803 - mirrors the PySpark API name
        """Distribute *elements*, crossing the serialization boundary first."""

        self.num_slices = numSlices
        materialized: list[Any] = []
        for element in elements:
            captured = find_driver_only_state(element, self.forbidden_types)
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
    forbidden_types: tuple[type, ...] = ()
    _context: FakeSparkContext | None = None

    @property
    def sparkContext(self) -> FakeSparkContext:  # noqa: N802 - PySpark API name
        """Return the lazily created fake context."""

        if self._context is None:
            self._context = FakeSparkContext(
                execution_order=self.execution_order,
                forbidden_types=self.forbidden_types,
            )
        return self._context
