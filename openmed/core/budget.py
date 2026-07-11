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
from dataclasses import dataclass
from typing import Optional

__all__ = [
    "BudgetClock",
    "BudgetExceededError",
    "RequestBudget",
    "coerce_budget",
]


class BudgetExceededError(RuntimeError):
    """Raised when a per-request resource or time budget is exceeded.

    The error is intentionally PHI-free: it exposes the budget ``kind`` that was
    exceeded, the configured ``limit``, the ``observed`` value, and the
    ``checkpoint`` name where the breach was detected. It never contains the
    input text, spans, or any identifier surface.

    Attributes:
        kind: Which budget was exceeded (``"wall_time"`` or ``"input_chars"``).
        limit: The configured budget limit (seconds or characters).
        observed: The observed value that breached the limit.
        checkpoint: Name of the safe checkpoint where the breach was detected.
    """

    def __init__(
        self,
        *,
        kind: str,
        limit: float,
        observed: float,
        checkpoint: Optional[str] = None,
    ) -> None:
        self.kind = kind
        self.limit = limit
        self.observed = observed
        self.checkpoint = checkpoint

        if kind == "wall_time":
            detail = f"wall-time budget of {limit:g}s exceeded (elapsed {observed:g}s)"
        elif kind == "input_chars":
            detail = (
                f"input-length budget of {int(limit)} characters exceeded "
                f"(input has {int(observed)} characters)"
            )
        else:  # pragma: no cover - defensive; kind is set by this module
            detail = f"budget '{kind}' exceeded (limit {limit}, observed {observed})"

        if checkpoint:
            detail = f"{detail} at checkpoint '{checkpoint}'"

        super().__init__(f"Request budget exceeded: {detail}")


def _validate_positive_number(
    value: Optional[float],
    *,
    name: str,
) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number or None")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def _validate_positive_int(
    value: Optional[int],
    *,
    name: str,
) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int or None")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
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
        return BudgetClock(budget=self, started_at=time.monotonic())

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
        return time.monotonic() - self.started_at

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


def coerce_budget(budget: Optional[RequestBudget]) -> Optional[RequestBudget]:
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
    if isinstance(budget, dict):
        coerced = RequestBudget(
            max_wall_time=budget.get("max_wall_time"),
            max_input_chars=budget.get("max_input_chars"),
        )
        return None if coerced.is_unlimited else coerced
    raise TypeError(
        "budget must be a RequestBudget, a mapping, or None; "
        f"got {type(budget).__name__}"
    )
