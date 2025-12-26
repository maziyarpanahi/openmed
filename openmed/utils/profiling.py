"""Performance profiling utilities for OpenMed.

This module provides tools for measuring and reporting performance metrics
during model inference and text processing pipelines.
"""

from __future__ import annotations

import functools
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class TimingResult:
    """Result of a timing measurement."""

    name: str
    duration: float  # seconds
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "duration_ms": self.duration * 1000,
            "duration_s": self.duration,
            "metadata": self.metadata or {},
        }


@dataclass
class ProfileReport:
    """Aggregated profiling report."""

    timings: List[TimingResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_duration(self) -> float:
        """Total duration of all timed operations."""
        return sum(t.duration for t in self.timings)

    @property
    def timing_count(self) -> int:
        """Number of timing measurements."""
        return len(self.timings)

    def get_timing(self, name: str) -> Optional[TimingResult]:
        """Get a specific timing by name."""
        for t in self.timings:
            if t.name == name:
                return t
        return None

    def get_timings_by_prefix(self, prefix: str) -> List[TimingResult]:
        """Get all timings that start with a prefix."""
        return [t for t in self.timings if t.name.startswith(prefix)]

    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self.timings:
            return {"total_ms": 0, "count": 0, "timings": []}

        durations = [t.duration * 1000 for t in self.timings]

        return {
            "total_ms": self.total_duration * 1000,
            "count": self.timing_count,
            "min_ms": min(durations),
            "max_ms": max(durations),
            "avg_ms": sum(durations) / len(durations),
            "timings": [t.to_dict() for t in self.timings],
            "metadata": self.metadata,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return self.summary()

    def format_report(self, include_metadata: bool = False) -> str:
        """Generate a human-readable report.

        Args:
            include_metadata: Whether to include metadata in the report.

        Returns:
            Formatted string report.
        """
        lines = [
            "Performance Profile Report",
            "=" * 50,
            f"Total Duration: {self.total_duration * 1000:.2f}ms",
            f"Operations: {self.timing_count}",
            "",
            "Timings:",
            "-" * 50,
        ]

        # Sort by duration descending
        sorted_timings = sorted(self.timings, key=lambda t: t.duration, reverse=True)

        for t in sorted_timings:
            pct = (t.duration / self.total_duration * 100) if self.total_duration > 0 else 0
            lines.append(f"  {t.name:<35} {t.duration * 1000:>8.2f}ms ({pct:>5.1f}%)")

        if include_metadata and self.metadata:
            lines.extend(["", "Metadata:", "-" * 50])
            for key, value in self.metadata.items():
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)


class Profiler:
    """Performance profiler for tracking operation timings.

    Example usage:
        >>> profiler = Profiler()
        >>> with profiler.measure("model_loading"):
        ...     model = load_model("model_name")
        >>> with profiler.measure("inference"):
        ...     result = model(text)
        >>> print(profiler.report().format_report())
    """

    def __init__(self, enabled: bool = True):
        """Initialize profiler.

        Args:
            enabled: Whether profiling is enabled. If False, measurements
                are no-ops for minimal overhead.
        """
        self.enabled = enabled
        self._timings: List[TimingResult] = []
        self._metadata: Dict[str, Any] = {}
        self._start_time: Optional[float] = None

    def start(self) -> None:
        """Start overall profiling session."""
        if self.enabled:
            self._start_time = time.perf_counter()
            self._timings = []
            self._metadata = {}

    def stop(self) -> None:
        """Stop overall profiling session and record total time."""
        if self.enabled and self._start_time is not None:
            total_time = time.perf_counter() - self._start_time
            self._metadata["session_duration_ms"] = total_time * 1000

    @contextmanager
    def measure(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Iterator[None]:
        """Context manager to measure operation duration.

        Args:
            name: Name of the operation being measured.
            metadata: Optional metadata to attach to the timing.

        Yields:
            None (used as context manager).
        """
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self._timings.append(
                TimingResult(name=name, duration=duration, metadata=metadata)
            )

    def add_timing(
        self,
        name: str,
        duration: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Manually add a timing result.

        Args:
            name: Name of the operation.
            duration: Duration in seconds.
            metadata: Optional metadata.
        """
        if self.enabled:
            self._timings.append(
                TimingResult(name=name, duration=duration, metadata=metadata)
            )

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the profile report.

        Args:
            key: Metadata key.
            value: Metadata value.
        """
        if self.enabled:
            self._metadata[key] = value

    def report(self) -> ProfileReport:
        """Generate a profile report.

        Returns:
            ProfileReport with all collected timings.
        """
        return ProfileReport(
            timings=list(self._timings),
            metadata=dict(self._metadata),
        )

    def reset(self) -> None:
        """Reset all collected timings and metadata."""
        self._timings = []
        self._metadata = {}
        self._start_time = None


# Global profiler instance for convenience
_global_profiler: Optional[Profiler] = None


def get_profiler() -> Profiler:
    """Get the global profiler instance (creates one if needed)."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = Profiler(enabled=False)
    return _global_profiler


def enable_profiling() -> Profiler:
    """Enable global profiling and return the profiler."""
    global _global_profiler
    _global_profiler = Profiler(enabled=True)
    _global_profiler.start()
    return _global_profiler


def disable_profiling() -> None:
    """Disable global profiling."""
    global _global_profiler
    if _global_profiler is not None:
        _global_profiler.stop()
        _global_profiler.enabled = False


def get_profile_report() -> ProfileReport:
    """Get the current global profile report."""
    return get_profiler().report()


def profile(name: Optional[str] = None) -> Callable[[F], F]:
    """Decorator to profile a function.

    Args:
        name: Optional name for the timing (defaults to function name).

    Returns:
        Decorated function.

    Example:
        >>> @profile("my_operation")
        ... def slow_function():
        ...     time.sleep(0.1)
    """
    def decorator(func: F) -> F:
        timing_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            profiler = get_profiler()
            with profiler.measure(timing_name):
                return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


@dataclass
class InferenceMetrics:
    """Metrics for a single inference operation."""

    text_length: int
    token_count: Optional[int] = None
    entity_count: int = 0
    inference_time_ms: float = 0.0
    preprocessing_time_ms: float = 0.0
    postprocessing_time_ms: float = 0.0
    total_time_ms: float = 0.0

    @property
    def tokens_per_second(self) -> Optional[float]:
        """Calculate tokens processed per second."""
        if self.token_count and self.inference_time_ms > 0:
            return self.token_count / (self.inference_time_ms / 1000)
        return None

    @property
    def chars_per_second(self) -> float:
        """Calculate characters processed per second."""
        if self.total_time_ms > 0:
            return self.text_length / (self.total_time_ms / 1000)
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text_length": self.text_length,
            "token_count": self.token_count,
            "entity_count": self.entity_count,
            "inference_time_ms": self.inference_time_ms,
            "preprocessing_time_ms": self.preprocessing_time_ms,
            "postprocessing_time_ms": self.postprocessing_time_ms,
            "total_time_ms": self.total_time_ms,
            "tokens_per_second": self.tokens_per_second,
            "chars_per_second": self.chars_per_second,
        }


@dataclass
class BatchMetrics:
    """Aggregated metrics for batch processing."""

    items: List[InferenceMetrics] = field(default_factory=list)
    total_time_ms: float = 0.0

    @property
    def item_count(self) -> int:
        """Number of items processed."""
        return len(self.items)

    @property
    def total_chars(self) -> int:
        """Total characters processed."""
        return sum(m.text_length for m in self.items)

    @property
    def total_entities(self) -> int:
        """Total entities detected."""
        return sum(m.entity_count for m in self.items)

    @property
    def avg_time_per_item_ms(self) -> float:
        """Average processing time per item."""
        if not self.items:
            return 0.0
        return sum(m.total_time_ms for m in self.items) / len(self.items)

    @property
    def throughput_items_per_second(self) -> float:
        """Items processed per second."""
        if self.total_time_ms > 0:
            return self.item_count / (self.total_time_ms / 1000)
        return 0.0

    @property
    def throughput_chars_per_second(self) -> float:
        """Characters processed per second."""
        if self.total_time_ms > 0:
            return self.total_chars / (self.total_time_ms / 1000)
        return 0.0

    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        return {
            "item_count": self.item_count,
            "total_chars": self.total_chars,
            "total_entities": self.total_entities,
            "total_time_ms": self.total_time_ms,
            "avg_time_per_item_ms": self.avg_time_per_item_ms,
            "throughput_items_per_second": self.throughput_items_per_second,
            "throughput_chars_per_second": self.throughput_chars_per_second,
        }

    def format_report(self) -> str:
        """Generate a human-readable report."""
        lines = [
            "Batch Processing Metrics",
            "=" * 40,
            f"Items processed: {self.item_count}",
            f"Total characters: {self.total_chars:,}",
            f"Total entities: {self.total_entities}",
            f"Total time: {self.total_time_ms:.2f}ms",
            f"Avg time/item: {self.avg_time_per_item_ms:.2f}ms",
            f"Throughput: {self.throughput_items_per_second:.2f} items/sec",
            f"Throughput: {self.throughput_chars_per_second:.0f} chars/sec",
        ]
        return "\n".join(lines)


class Timer:
    """Simple timer for measuring elapsed time.

    Example:
        >>> timer = Timer()
        >>> timer.start()
        >>> # do work
        >>> elapsed = timer.stop()
        >>> print(f"Took {elapsed:.2f}ms")
    """

    def __init__(self):
        """Initialize timer."""
        self._start: Optional[float] = None
        self._stop: Optional[float] = None

    def start(self) -> "Timer":
        """Start the timer."""
        self._start = time.perf_counter()
        self._stop = None
        return self

    def stop(self) -> float:
        """Stop the timer and return elapsed time in milliseconds."""
        if self._start is None:
            raise RuntimeError("Timer was not started")
        self._stop = time.perf_counter()
        return self.elapsed_ms

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self._start is None:
            return 0.0
        end = self._stop if self._stop is not None else time.perf_counter()
        return (end - self._start) * 1000

    @property
    def elapsed_s(self) -> float:
        """Get elapsed time in seconds."""
        return self.elapsed_ms / 1000

    def reset(self) -> "Timer":
        """Reset the timer."""
        self._start = None
        self._stop = None
        return self


@contextmanager
def timed(name: str = "operation") -> Iterator[Timer]:
    """Context manager that yields a timer and logs the duration.

    Args:
        name: Name of the operation for logging.

    Yields:
        Timer instance.

    Example:
        >>> with timed("model inference"):
        ...     result = model(text)
        # Logs: "model inference took 123.45ms"
    """
    timer = Timer().start()
    try:
        yield timer
    finally:
        elapsed = timer.stop()
        logger.debug("%s took %.2fms", name, elapsed)
