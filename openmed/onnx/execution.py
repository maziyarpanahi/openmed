"""Cooperative native ONNX cancellation and document-batch deadlines.

ORT observes RunOptions.terminate at runtime cancellation points; this is not an
OS-level process kill and cannot promise preemption of every provider kernel.
https://onnxruntime.ai/docs/api/python/api_summary.html#onnxruntime.RunOptions
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable


class OnnxExecutionCancelled(RuntimeError):
    """Raised when the owner cancels a complete inference batch."""


class OnnxExecutionControl:
    """One absolute deadline/cancellation scope shared by all document windows.

    Args:
        timeout_seconds: Positive wall-clock budget, starting at construction.
        cancel_check: Thread-safe predicate. A coalescing service should cancel
            only when every caller in this batch has cancelled.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        if timeout_seconds is not None and (
            isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive and finite")
        if cancel_check is not None and not callable(cancel_check):
            raise ValueError("cancel_check must be callable")
        self._deadline = (
            None if timeout_seconds is None else time.monotonic() + timeout_seconds
        )
        self._cancel_check = cancel_check
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        """Request cancellation of calls sharing this control."""
        self._cancelled.set()

    def _reason(self) -> str | None:
        cancelled = self._cancelled.is_set()
        if not cancelled and self._cancel_check is not None:
            try:
                cancelled = bool(self._cancel_check())
            except Exception:
                cancelled = True
        if cancelled:
            return "cancelled"
        if self._deadline is not None and time.monotonic() >= self._deadline:
            return "timeout"
        return None

    def check(self) -> None:
        """Raise a PHI-free failure before or after a bounded execution stage."""
        reason = self._reason()
        if reason == "cancelled":
            raise OnnxExecutionCancelled("ONNX execution cancelled")
        if reason == "timeout":
            raise TimeoutError("ONNX execution deadline exceeded")

    def run(self, session: Any, output_names: list[str], feed: dict[str, Any]) -> Any:
        """Run one tensor batch, signalling native ORT when the scope expires.

        Args:
            session: ONNX Runtime InferenceSession.
            output_names: Requested graph outputs.
            feed: Tensor inputs; never logged or retained by this control.

        Returns:
            The session's graph outputs, only if the scope is still valid.
        """
        import onnxruntime as ort

        self.check()
        options = ort.RunOptions()
        stopped = threading.Event()

        def watch() -> None:
            while not stopped.wait(0.02):
                if self._reason() is not None:
                    options.terminate = True
                    return

        watchdog = threading.Thread(target=watch, name="onnx-deadline", daemon=True)
        watchdog.start()
        try:
            try:
                output = session.run(output_names, feed, options)
            except Exception:
                self.check()
                raise
            self.check()
            return output
        finally:
            stopped.set()
            watchdog.join()
