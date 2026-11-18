"""Bounded waits, deadlines, and wait outcomes for Journey metadata stores.

Every metadata-store operation runs under one deadline: the earliest of an
explicit ``deadline=`` argument, the ambient :func:`store_deadline` scope, and
the store's configured server cap (:class:`StoreTimeouts`). The deadline bounds
the in-process checkout of the store's dedicated connection, database lock
waits, and statement execution.

Failures carry controlled codes only. No SQL, parameters, identifiers, paths,
configuration values, or driver diagnostics are attached to these errors.

Example:
    >>> from openmed.structured.store import StoreTimeouts, store_deadline
    >>> timeouts = StoreTimeouts(operation_seconds=10, lock_wait_seconds=2)
    >>> with store_deadline(0.5):
    ...     bound = timeouts.operation_deadline()
    >>> bound.remaining() <= 0.5
    True
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Union

from openmed.core.budget import BudgetClock

from .protocols import StoreResult, StoreState

STORE_TIMEOUT_ENV_VAR = "OPENMED_JOURNEY_STORE_TIMEOUT_SECONDS"
STORE_LOCK_WAIT_ENV_VAR = "OPENMED_JOURNEY_STORE_LOCK_WAIT_SECONDS"
DEFAULT_STORE_TIMEOUT_SECONDS = 30.0
DEFAULT_STORE_LOCK_WAIT_SECONDS = 5.0
MAX_STORE_TIMEOUT_SECONDS = 3600.0

STORE_BUSY_CODE = "store_busy"
STORE_DEADLINE_EXCEEDED_CODE = "store_deadline_exceeded"
STORE_TIMEOUTS_INVALID_CODE = "store_timeouts_invalid"

Clock = Callable[[], float]
StoreDeadlineInput = Union["StoreDeadline", BudgetClock, float, int, None]


class StoreWaitError(RuntimeError):
    """Base class for value-free bounded-wait failures.

    Attributes:
        code: Stable controlled reason code for typed store results.
    """

    code = "store_wait_failed"
    _message = "journey store wait failed"

    def __init__(self) -> None:
        super().__init__(self._message)

    def outcome(self) -> StoreResult[Any]:
        """Return the typed ``failure`` result for this wait failure."""

        return StoreResult.outcome(StoreState.FAILURE, self.code)


class StoreBusyError(StoreWaitError):
    """The dedicated connection or a database lock stayed busy past its bound."""

    code = STORE_BUSY_CODE
    _message = "journey store is busy"


class StoreDeadlineExceededError(StoreWaitError):
    """The operation deadline expired before the operation could finish."""

    code = STORE_DEADLINE_EXCEEDED_CODE
    _message = "journey store deadline exceeded"


class StoreTimeoutsError(ValueError):
    """Store timeout configuration is invalid; values are never echoed."""

    code = STORE_TIMEOUTS_INVALID_CODE


def _finite_seconds(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number of seconds")
    seconds = float(value)
    if not math.isfinite(seconds):
        raise ValueError(f"{name} must be finite")
    return seconds


@dataclass(frozen=True, slots=True)
class StoreDeadline:
    """An absolute monotonic deadline for one store operation.

    Attributes:
        expires_at: Clock reading at which the operation must have finished.
        clock: Monotonic clock in seconds used to measure the time left.
    """

    expires_at: float
    clock: Clock = field(default=time.monotonic, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "expires_at", _finite_seconds(self.expires_at, "expires_at")
        )
        if not callable(self.clock):
            raise TypeError("clock must be callable")

    @classmethod
    def after(cls, seconds: float, *, clock: Clock = time.monotonic) -> StoreDeadline:
        """Return a deadline ``seconds`` from now.

        Args:
            seconds: Time allowed from now. Zero or less is already expired.
            clock: Monotonic clock in seconds.

        Returns:
            The absolute deadline.

        Raises:
            TypeError: If ``seconds`` is not a number.
            ValueError: If ``seconds`` is not finite.
        """

        return cls(clock() + _finite_seconds(seconds, "seconds"), clock)

    @classmethod
    def from_budget(
        cls,
        budget: BudgetClock,
        *,
        clock: Clock = time.monotonic,
    ) -> StoreDeadline | None:
        """Derive a deadline from a started request budget.

        Args:
            budget: A clock returned by ``RequestBudget.start()``.
            clock: Monotonic clock in seconds for the returned deadline.

        Returns:
            The deadline at which the budget's wall time runs out, or ``None``
            when the budget has no wall-time limit.
        """

        limit = budget.budget.max_wall_time
        if limit is None:
            return None
        return cls(clock() + (limit - budget.elapsed), clock)

    def remaining(self) -> float:
        """Return the seconds left; zero or less means the deadline passed."""

        return self.expires_at - self.clock()

    @property
    def expired(self) -> bool:
        """Return whether the deadline has passed."""

        return self.remaining() <= 0

    def check(self) -> None:
        """Raise when the deadline has passed.

        Raises:
            StoreDeadlineExceededError: If no time is left.
        """

        if self.expired:
            raise StoreDeadlineExceededError()

    def milliseconds(self, cap_seconds: float) -> int:
        """Return a positive millisecond bound no later than the deadline or cap.

        Args:
            cap_seconds: Server-side cap for this bound.

        Returns:
            Whole milliseconds, at least one.

        Raises:
            StoreDeadlineExceededError: If no time is left.
        """

        seconds = min(self.remaining(), cap_seconds)
        if seconds <= 0:
            raise StoreDeadlineExceededError()
        return max(1, int(seconds * 1000))


def earliest_deadline(*deadlines: StoreDeadline | None) -> StoreDeadline | None:
    """Return the deadline with the least time left, ignoring ``None``."""

    candidates = [deadline for deadline in deadlines if deadline is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda deadline: deadline.remaining())


def coerce_store_deadline(
    value: StoreDeadlineInput,
    *,
    clock: Clock = time.monotonic,
) -> StoreDeadline | None:
    """Normalize a caller deadline.

    Args:
        value: A ``StoreDeadline``, a started ``BudgetClock``, a number of
            seconds from now, or ``None``.
        clock: Clock for deadlines created from seconds or a budget.

    Returns:
        The deadline, or ``None`` when the caller set no time limit.

    Raises:
        TypeError: If ``value`` has an unsupported type.
        ValueError: If a number of seconds is not finite.
    """

    if value is None or isinstance(value, StoreDeadline):
        return value
    if isinstance(value, BudgetClock):
        return StoreDeadline.from_budget(value, clock=clock)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            "deadline must be a StoreDeadline, BudgetClock, seconds, or None"
        )
    return StoreDeadline.after(value, clock=clock)


_AMBIENT_DEADLINE: ContextVar[StoreDeadline | None] = ContextVar(
    "openmed_journey_store_deadline",
    default=None,
)


def current_store_deadline() -> StoreDeadline | None:
    """Return the ambient caller deadline for this context, if any."""

    return _AMBIENT_DEADLINE.get()


@contextmanager
def store_deadline(deadline: StoreDeadlineInput) -> Iterator[StoreDeadline | None]:
    """Bound every Journey-store operation in this context by a caller deadline.

    Scopes nest and can only shorten the active deadline. The scope is a
    context variable, so it follows ``asyncio`` tasks and ``contextvars``
    copies but not unrelated threads.

    Args:
        deadline: A ``StoreDeadline``, a started ``BudgetClock``, seconds from
            now, or ``None`` to keep the enclosing deadline.

    Yields:
        The effective ambient deadline inside the scope.
    """

    effective = earliest_deadline(
        _AMBIENT_DEADLINE.get(),
        coerce_store_deadline(deadline),
    )
    token = _AMBIENT_DEADLINE.set(effective)
    try:
        yield effective
    finally:
        _AMBIENT_DEADLINE.reset(token)


def _cap_seconds(value: Any, name: str) -> float:
    try:
        seconds = _finite_seconds(value, name)
    except (TypeError, ValueError):
        raise StoreTimeoutsError(f"{name} must be a finite number of seconds") from None
    if not 0 < seconds <= MAX_STORE_TIMEOUT_SECONDS:
        raise StoreTimeoutsError(
            f"{name} must be greater than zero and at most "
            f"{int(MAX_STORE_TIMEOUT_SECONDS)} seconds"
        )
    return seconds


@dataclass(frozen=True, slots=True)
class StoreTimeouts:
    """Server-side caps for Journey metadata-store waits.

    A caller deadline can shorten these caps but never extend them.

    Attributes:
        operation_seconds: Longest time one store operation may take,
            including connection checkout, lock waits, and statement
            execution. It also caps PostgreSQL ``statement_timeout``.
        lock_wait_seconds: Longest wait to check out the store's dedicated
            connection and to acquire a database lock (PostgreSQL
            ``lock_timeout``, SQLite ``busy_timeout``).
        clock: Monotonic clock in seconds; injectable for tests.

    Raises:
        StoreTimeoutsError: If a cap is not in ``(0, 3600]`` seconds.
    """

    operation_seconds: float = DEFAULT_STORE_TIMEOUT_SECONDS
    lock_wait_seconds: float = DEFAULT_STORE_LOCK_WAIT_SECONDS
    clock: Clock = field(default=time.monotonic, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("operation_seconds", "lock_wait_seconds"):
            object.__setattr__(self, name, _cap_seconds(getattr(self, name), name))
        if not callable(self.clock):
            raise TypeError("clock must be callable")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> StoreTimeouts:
        """Read caps from ``OPENMED_JOURNEY_STORE_*`` environment variables.

        Blank or unset variables keep the built-in defaults of 30 seconds per
        operation and 5 seconds per lock wait.

        Args:
            environ: Optional mapping used instead of ``os.environ``.

        Returns:
            The configured caps.

        Raises:
            StoreTimeoutsError: If a variable is not a valid number of seconds.
                The message names the variable but never its value.
        """

        source = os.environ if environ is None else environ
        values: dict[str, float] = {}
        for attribute, name in (
            ("operation_seconds", STORE_TIMEOUT_ENV_VAR),
            ("lock_wait_seconds", STORE_LOCK_WAIT_ENV_VAR),
        ):
            raw = source.get(name, "").strip()
            if not raw:
                continue
            try:
                values[attribute] = _cap_seconds(float(raw), name)
            except ValueError:
                raise StoreTimeoutsError(
                    f"{name} must be a number of seconds in (0, "
                    f"{int(MAX_STORE_TIMEOUT_SECONDS)}]"
                ) from None
        return cls(**values)

    def operation_deadline(self, deadline: StoreDeadlineInput = None) -> StoreDeadline:
        """Return the effective deadline for one operation starting now.

        Args:
            deadline: Optional explicit caller deadline.

        Returns:
            The earliest of the explicit deadline, the ambient
            :func:`store_deadline` scope, and ``operation_seconds`` from now.
        """

        cap = StoreDeadline.after(self.operation_seconds, clock=self.clock)
        effective = earliest_deadline(
            cap,
            coerce_store_deadline(deadline, clock=self.clock),
            current_store_deadline(),
        )
        return cap if effective is None else effective


def resolve_store_timeouts(value: StoreTimeouts | None) -> StoreTimeouts:
    """Return explicit caps, or the environment-configured defaults.

    Args:
        value: Explicit caps, or ``None`` to read :meth:`StoreTimeouts.from_env`.

    Returns:
        The caps a store should apply.

    Raises:
        TypeError: If ``value`` is not ``StoreTimeouts`` or ``None``.
        StoreTimeoutsError: If the environment configuration is invalid.
    """

    if value is None:
        return StoreTimeouts.from_env()
    if not isinstance(value, StoreTimeouts):
        raise TypeError("timeouts must be StoreTimeouts or None")
    return value


class ConnectionCheckout:
    """Reentrant checkout of a store's dedicated connection with a bounded wait.

    It replaces an unbounded ``threading.RLock``. ``with checkout:`` waits at
    most the lock-wait cap (or less when a caller deadline is shorter) and
    raises :class:`StoreBusyError` instead of blocking forever.
    """

    __slots__ = ("_depth", "_lock", "_owner", "_timeouts")

    def __init__(self, timeouts: StoreTimeouts) -> None:
        self._lock = threading.RLock()
        self._timeouts = timeouts
        self._owner: int | None = None
        self._depth = 0

    @property
    def held(self) -> bool:
        """Return whether the current thread holds the checkout."""

        return self._owner == threading.get_ident()

    def acquire(self, deadline: StoreDeadline | None = None) -> None:
        """Check out the connection within the bounded wait.

        Args:
            deadline: Operation deadline; defaults to the configured caps and
                the ambient caller deadline.

        Raises:
            StoreBusyError: If another operation still holds the connection
                when the wait ends.
            StoreDeadlineExceededError: If the deadline already passed.
        """

        if self.held:
            self._lock.acquire()
            self._depth += 1
            return
        bound = self._timeouts.operation_deadline() if deadline is None else deadline
        wait = min(bound.remaining(), self._timeouts.lock_wait_seconds)
        if wait <= 0:
            raise StoreDeadlineExceededError()
        if not self._lock.acquire(timeout=wait):
            raise StoreBusyError()
        self._owner = threading.get_ident()
        self._depth = 1

    def release(self) -> None:
        """Release one level of the checkout."""

        if not self.held:
            raise RuntimeError("connection checkout is not held by this thread")
        self._depth -= 1
        if self._depth == 0:
            self._owner = None
        self._lock.release()

    def __enter__(self) -> ConnectionCheckout:
        self.acquire()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()


__all__ = [
    "DEFAULT_STORE_LOCK_WAIT_SECONDS",
    "DEFAULT_STORE_TIMEOUT_SECONDS",
    "MAX_STORE_TIMEOUT_SECONDS",
    "STORE_BUSY_CODE",
    "STORE_DEADLINE_EXCEEDED_CODE",
    "STORE_LOCK_WAIT_ENV_VAR",
    "STORE_TIMEOUTS_INVALID_CODE",
    "STORE_TIMEOUT_ENV_VAR",
    "ConnectionCheckout",
    "StoreBusyError",
    "StoreDeadline",
    "StoreDeadlineExceededError",
    "StoreDeadlineInput",
    "StoreTimeouts",
    "StoreTimeoutsError",
    "StoreWaitError",
    "coerce_store_deadline",
    "current_store_deadline",
    "earliest_deadline",
    "resolve_store_timeouts",
    "store_deadline",
]
