"""Cooperative brief interruption using the existing request budget clock."""

from __future__ import annotations

import inspect
from threading import Event
from typing import Callable

from openmed.core.budget import BudgetClock, BudgetExceededError


class BriefInterrupted(RuntimeError):
    """Value-free terminal interruption with a controlled reason."""

    def __init__(self, *, expired: bool = False) -> None:
        self.reason = "deadline_exceeded" if expired else "cancelled"
        super().__init__(self.reason)


class BriefCancellation:
    """Caller-owned, thread-safe cancellation and optional started deadline.

    Args:
        clock: Existing ``RequestBudget.start()`` clock. Its time allowance
            includes review lookup and all brief stages. Input budgets remain
            owned by extraction; this context checks only the wall-time limit.
    """

    def __init__(self, clock: BudgetClock | None = None) -> None:
        if clock is not None and not isinstance(clock, BudgetClock):
            raise TypeError("invalid brief budget clock")
        self._clock = clock
        self._cancelled = Event()
        self._expired = Event()

    def cancel(self) -> None:
        """Request cancellation without retaining a caller-supplied message."""
        self._cancelled.set()

    def check(self) -> None:
        """Raise at a checkpoint; explicit cancellation wins simultaneous expiry."""
        if self._cancelled.is_set():
            raise BriefInterrupted()
        if self._clock is not None:
            try:
                self._clock.check("clinical_brief")
            except BudgetExceededError:
                self._expired.set()
        if self._expired.is_set():
            raise BriefInterrupted(expired=True)


def check_cancellation(cancellation: BriefCancellation | None) -> None:
    """Check an optional context without changing unlimited calls."""
    if cancellation is not None:
        cancellation.check()


def call_with_cancellation(
    callback: Callable, *args, cancellation: BriefCancellation | None = None, **kwargs
):
    """Pass context to explicitly cooperative callbacks and discard late results.

    Legacy providers keep their signature. They cannot be forcibly stopped;
    the post-call checkpoint runs even if the provider raises an exception.
    """
    check_cancellation(cancellation)
    if cancellation is not None:
        try:
            parameter = inspect.signature(callback).parameters.get("cancellation")
        except (TypeError, ValueError):
            parameter = None
        if parameter is not None and parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            kwargs["cancellation"] = cancellation
    failed: BaseException | None = None
    result = None
    try:
        result = callback(*args, **kwargs)
    except BaseException as error:
        failed = error
    # Outside the handler: interruption tracebacks must not chain provider PHI.
    check_cancellation(cancellation)
    if failed is not None:
        raise failed
    return result
