"""Per-request resource and timeout budgets with cooperative cancellation.

This module provides opt-in budgets that bound a single extraction or
de-identification request. Two independent limits are supported:

- A wall-clock time budget (``max_wall_time`` seconds).
- An input-size budget (``max_input_chars`` characters).

Cancellation is *cooperative*: the pipeline and batch loops call
:meth:`BudgetClock.check` at safe checkpoints (between pipeline stages and
between batch items). When a deadline has passed a :class:`BudgetExceededError`
is raised cleanly, so no thread is killed and no partial state is corrupted.

Budgets are always optional. When no budget is supplied (or ``None`` is passed),
behavior is byte-for-byte identical to the historical unlimited default.

Clock choice: elapsed wall time is measured with :func:`time.perf_counter`, the
highest-resolution clock available for short durations. :func:`time.monotonic`
is not usable here because on Windows it is backed by ``GetTickCount64()`` with
a 15.6 ms granularity until CPython 3.13, which makes any ``max_wall_time``
below one tick unenforceable and quantizes every longer budget to the nearest
tick. Both clocks are monotonic and both have an undefined reference point, so
only differences are meaningful -- which is all :attr:`BudgetClock.elapsed`
computes. ``started_at`` is created and consumed in the same process and is
never serialized or compared across processes.

Privacy: neither the budget object nor :class:`BudgetExceededError` ever
captures raw input text or PHI. Errors carry only counts, limits, and the name
of the checkpoint at which the budget was exceeded.

Example:
    >>> from openmed.core.budget import RequestBudget, BudgetExceededError
    >>> budget = RequestBudget(max_input_chars=8)
    >>> try:
    ...     budget.check_input_length(42)
    ... except BudgetExceededError as exc:
    ...     print(exc.kind, exc.limit, exc.observed)
    input_chars 8 42
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any, Optional

from .errors import BudgetExceededError, InputError

__all__ = [
    "BudgetClock",
    "BudgetExceededError",
    "RequestBudget",
    "coerce_budget",
]


def _validate_positive_number(
    value: Optional[float],
    *,
    name: str,
) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(
            f"{name} must be a number or None. Pass a positive finite number "
            "or omit the limit.",
            details={"argument": name, "expected": "positive finite number or null"},
        )
    normalized = float(value)
    if not isfinite(normalized) or normalized <= 0:
        raise InputError(
            f"{name} must be positive and finite. Pass a value greater than zero.",
            details={"argument": name, "constraint": "positive_finite"},
        )
    return normalized


def _validate_positive_int(
    value: Optional[int],
    *,
    name: str,
) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputError(
            f"{name} must be an integer or None. Pass a positive integer or "
            "omit the limit.",
            details={"argument": name, "expected": "positive integer or null"},
        )
    if value <= 0:
        raise InputError(
            f"{name} must be positive. Pass an integer greater than zero.",
            details={"argument": name, "constraint": "positive"},
        )
    return int(value)


@dataclass(frozen=True)
class RequestBudget:
    """Optional per-request wall-time and input-size budget.

    Both limits are independent and optional. ``None`` for a limit means that
    dimension is unbounded (the historical default). Budgets are cooperative:
    the caller pipeline checks the deadline at safe checkpoints and raises
    :class:`BudgetExceededError` when a limit is breached.

    Args:
        max_wall_time: Maximum wall-clock seconds for the request. ``None``
            means no time limit.
        max_input_chars: Maximum number of input characters. ``None`` means no
            input-length limit.

    Raises:
        TypeError: If a limit is set to a non-numeric value.
        ValueError: If a limit is set to a non-positive value.
    """

    max_wall_time: Optional[float] = None
    max_input_chars: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_wall_time",
            _validate_positive_number(self.max_wall_time, name="max_wall_time"),
        )
        object.__setattr__(
            self,
            "max_input_chars",
            _validate_positive_int(self.max_input_chars, name="max_input_chars"),
        )

    @property
    def is_unlimited(self) -> bool:
        """Return ``True`` when neither limit is set."""
        return self.max_wall_time is None and self.max_input_chars is None

    def start(self) -> "BudgetClock":
        """Return a fresh clock anchored to the current wall-clock time.

        The returned :class:`BudgetClock` shares this budget's limits and is the
        object the pipeline checks at each safe checkpoint.
        """
        return BudgetClock(budget=self, started_at=time.perf_counter())

    def check_input_length(
        self,
        length: int,
        *,
        checkpoint: str = "input_guard",
    ) -> None:
        """Raise :class:`BudgetExceededError` if ``length`` exceeds the budget.

        Args:
            length: Number of input characters. Only the count is inspected;
                the input text itself is never passed in.
            checkpoint: Name of the checkpoint for the error record.

        Raises:
            BudgetExceededError: If ``max_input_chars`` is set and ``length``
                exceeds it.
        """
        if self.max_input_chars is not None and length > self.max_input_chars:
            raise BudgetExceededError(
                kind="input_chars",
                limit=self.max_input_chars,
                observed=length,
                checkpoint=checkpoint,
            )


@dataclass
class BudgetClock:
    """A started budget that checks the wall-time deadline at checkpoints.

    Created via :meth:`RequestBudget.start`. Checking is cooperative: the caller
    invokes :meth:`check` between pipeline stages and batch items. Nothing is
    interrupted preemptively.
    """

    budget: RequestBudget
    started_at: float

    @property
    def elapsed(self) -> float:
        """Seconds elapsed since the clock started."""
        return time.perf_counter() - self.started_at

    def check_input_length(
        self,
        length: int,
        *,
        checkpoint: str = "input_guard",
    ) -> None:
        """Delegate to :meth:`RequestBudget.check_input_length`."""
        self.budget.check_input_length(length, checkpoint=checkpoint)

    def check(self, checkpoint: str) -> None:
        """Raise :class:`BudgetExceededError` if the time budget is exhausted.

        Call at a safe checkpoint (e.g. between pipeline stages or batch items).

        Args:
            checkpoint: Name of the current checkpoint, recorded on the error.

        Raises:
            BudgetExceededError: If ``max_wall_time`` is set and has elapsed.
        """
        max_wall_time = self.budget.max_wall_time
        if max_wall_time is None:
            return
        elapsed = self.elapsed
        if elapsed > max_wall_time:
            raise BudgetExceededError(
                kind="wall_time",
                limit=max_wall_time,
                observed=elapsed,
                checkpoint=checkpoint,
            )


def coerce_budget(
    budget: Optional[RequestBudget | Mapping[str, Any]],
) -> Optional[RequestBudget]:
    """Validate and return a :class:`RequestBudget` or ``None``.

    Accepts an existing :class:`RequestBudget`, ``None`` (unlimited), or a
    mapping with ``max_wall_time`` / ``max_input_chars`` keys. An unlimited
    budget (both limits ``None``) is normalized to ``None`` so the historical
    fast path is preserved.

    Args:
        budget: A :class:`RequestBudget`, a mapping of budget fields, or ``None``.

    Returns:
        A validated :class:`RequestBudget`, or ``None`` when unlimited.

    Raises:
        TypeError: If ``budget`` is not a supported type.
    """
    if budget is None:
        return None
    if isinstance(budget, RequestBudget):
        return None if budget.is_unlimited else budget
    if isinstance(budget, Mapping):
        coerced = RequestBudget(
            max_wall_time=budget.get("max_wall_time"),
            max_input_chars=budget.get("max_input_chars"),
        )
        return None if coerced.is_unlimited else coerced
    raise InputError(
        "budget must be a RequestBudget, a mapping, or None. Pass a validated "
        "RequestBudget, a mapping of budget fields, or omit the argument.",
        details={"argument": "budget", "type": type(budget).__name__},
    )
