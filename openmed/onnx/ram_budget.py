"""Fail-closed resident-memory budgets for streaming model loading."""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

RssSampler = Callable[[], int | None]


class RamProbeUnavailable(RuntimeError):
    """Raised when current resident memory cannot be measured safely."""


class RamBudgetExceeded(MemoryError):
    """Raised before or after an operation would exceed its RAM budget."""

    def __init__(
        self,
        *,
        budget_bytes: int,
        observed_bytes: int,
        requested_bytes: int = 0,
        operation: str = "operation",
    ) -> None:
        self.budget_bytes = budget_bytes
        self.observed_bytes = observed_bytes
        self.requested_bytes = requested_bytes
        self.operation = operation
        projected = observed_bytes + requested_bytes
        super().__init__(
            f"RAM budget exceeded while {operation}: projected incremental RSS "
            f"{_format_bytes(projected)} exceeds the configured budget "
            f"{_format_bytes(budget_bytes)} (observed {_format_bytes(observed_bytes)}, "
            f"requested {_format_bytes(requested_bytes)})"
        )


@dataclass(frozen=True)
class RamBudget:
    """Maximum incremental resident memory allowed during one load."""

    max_bytes: int

    def __post_init__(self) -> None:
        if isinstance(self.max_bytes, bool) or self.max_bytes <= 0:
            raise ValueError("RAM budget must be a positive number of bytes")

    @classmethod
    def from_mib(cls, max_mib: float) -> "RamBudget":
        """Build a budget from mebibytes."""

        if isinstance(max_mib, bool) or max_mib <= 0:
            raise ValueError("RAM budget must be a positive number of MiB")
        return cls(max_bytes=int(max_mib * 1024 * 1024))


@dataclass(frozen=True)
class PeakRamReport:
    """Measured process RSS for one budgeted operation."""

    budget_bytes: int
    baseline_rss_bytes: int
    peak_rss_bytes: int

    @property
    def peak_incremental_bytes(self) -> int:
        """Return peak RSS above the operation baseline."""

        return max(self.peak_rss_bytes - self.baseline_rss_bytes, 0)

    @property
    def within_budget(self) -> bool:
        """Return whether the measured peak stayed within the budget."""

        return self.peak_incremental_bytes <= self.budget_bytes


class PeakRamProbe:
    """Measure and enforce an incremental process-RSS budget.

    The probe intentionally fails closed when the platform cannot provide a
    current RSS value. Call :meth:`reserve` immediately before mapping or
    allocating a known amount, and :meth:`checkpoint` after the operation.
    """

    def __init__(
        self,
        budget: RamBudget | int,
        *,
        rss_sampler: RssSampler | None = None,
        poll_interval_seconds: float | None = 0.01,
    ) -> None:
        if poll_interval_seconds is not None and poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive or None")
        self.budget = budget if isinstance(budget, RamBudget) else RamBudget(budget)
        self._rss_sampler = rss_sampler or current_rss_bytes
        self._poll_interval_seconds = poll_interval_seconds
        self._baseline: int | None = None
        self._peak: int | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._monitor: threading.Thread | None = None
        self._monitor_error: RamProbeUnavailable | None = None

    @property
    def started(self) -> bool:
        """Return whether the probe has recorded a baseline."""

        return self._baseline is not None

    @property
    def report(self) -> PeakRamReport:
        """Return the measurements recorded so far."""

        with self._lock:
            if self._baseline is None or self._peak is None:
                raise RuntimeError("RAM probe has not been started")
            return PeakRamReport(
                budget_bytes=self.budget.max_bytes,
                baseline_rss_bytes=self._baseline,
                peak_rss_bytes=self._peak,
            )

    def start(self) -> int:
        """Record and return the current RSS baseline."""

        if self.started:
            raise RuntimeError("RAM probe is already started")
        baseline = self._sample()
        with self._lock:
            self._baseline = baseline
            self._peak = baseline
        if self._poll_interval_seconds is not None:
            self._monitor = threading.Thread(
                target=self._poll,
                name="openmed-streaming-rss-monitor",
                daemon=True,
            )
            self._monitor.start()
        return baseline

    def checkpoint(self, operation: str = "checking memory") -> int:
        """Record current RSS and raise if the budget is already exceeded."""

        if self._baseline is None:
            raise RuntimeError("RAM probe has not been started")
        self._raise_monitor_error()
        current = self._sample()
        self._record(current)
        self._raise_monitor_error()
        report = self.report
        observed = report.peak_incremental_bytes
        if observed > self.budget.max_bytes:
            raise RamBudgetExceeded(
                budget_bytes=self.budget.max_bytes,
                observed_bytes=observed,
                operation=operation,
            )
        return current

    def reserve(self, requested_bytes: int, operation: str) -> None:
        """Fail before an operation whose known working set cannot fit."""

        if isinstance(requested_bytes, bool) or requested_bytes < 0:
            raise ValueError("requested_bytes must be a non-negative integer")
        self.checkpoint(operation)
        observed = self.report.peak_incremental_bytes
        if observed + requested_bytes > self.budget.max_bytes:
            raise RamBudgetExceeded(
                budget_bytes=self.budget.max_bytes,
                observed_bytes=observed,
                requested_bytes=requested_bytes,
                operation=operation,
            )

    def stop(self) -> PeakRamReport:
        """Take a final checked sample and return the peak report."""

        self._stop_monitor()
        self.checkpoint("finalizing the streaming load")
        return self.report

    def __enter__(self) -> "PeakRamProbe":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is None:
            self.stop()
        elif self.started:
            self._stop_monitor()
            try:
                current = self._sample()
            except RamProbeUnavailable:
                pass
            else:
                self._record(current)
        return False

    def _poll(self) -> None:
        if self._poll_interval_seconds is None:
            raise RuntimeError("poll_interval_seconds not set before polling thread started")
        while not self._stop_event.wait(self._poll_interval_seconds):
            try:
                self._record(self._sample())
            except RamProbeUnavailable as exc:
                self._monitor_error = exc
                self._stop_event.set()
                return

    def _record(self, value: int) -> None:
        with self._lock:
            self._peak = value if self._peak is None else max(self._peak, value)

    def _raise_monitor_error(self) -> None:
        if self._monitor_error is not None:
            raise self._monitor_error

    def _stop_monitor(self) -> None:
        self._stop_event.set()
        if self._monitor is not None:
            self._monitor.join(timeout=1.0)
            self._monitor = None

    def _sample(self) -> int:
        try:
            value = self._rss_sampler()
        except Exception as exc:
            raise RamProbeUnavailable(
                "Current RSS measurement failed; refusing to load weights "
                "without enforceable RAM limits"
            ) from exc
        if value is None or isinstance(value, bool) or int(value) < 0:
            raise RamProbeUnavailable(
                "Current RSS is unavailable; refusing to load weights without "
                "enforceable RAM limits"
            )
        return int(value)


def current_rss_bytes() -> int | None:
    """Return current process resident memory using platform-native APIs."""

    if sys.platform.startswith("linux"):
        try:
            resident_pages = int(
                Path("/proc/self/statm").read_text(encoding="ascii").split()[1]
            )
            return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
        except (IndexError, OSError, TypeError, ValueError):
            return None
    if sys.platform == "darwin":
        return _darwin_current_rss_bytes()
    if os.name == "nt":
        return _windows_current_rss_bytes()
    return None


class _TimeValue(ctypes.Structure):
    _fields_ = [("seconds", ctypes.c_int), ("microseconds", ctypes.c_int)]


class _MachTaskBasicInfo(ctypes.Structure):
    _fields_ = [
        ("virtual_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("resident_size_max", ctypes.c_uint64),
        ("user_time", _TimeValue),
        ("system_time", _TimeValue),
        ("policy", ctypes.c_int),
        ("suspend_count", ctypes.c_int),
    ]


def _darwin_current_rss_bytes() -> int | None:
    try:
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        libc.mach_task_self.restype = ctypes.c_uint
        libc.task_info.argtypes = (
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint),
        )
        info = _MachTaskBasicInfo()
        count = ctypes.c_uint(
            ctypes.sizeof(_MachTaskBasicInfo) // ctypes.sizeof(ctypes.c_uint)
        )
        status = libc.task_info(
            libc.mach_task_self(),
            20,
            ctypes.byref(info),
            ctypes.byref(count),
        )
        return int(info.resident_size) if status == 0 else None
    except (AttributeError, OSError, TypeError, ValueError):
        return None


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("page_fault_count", ctypes.c_ulong),
        ("peak_working_set_size", ctypes.c_size_t),
        ("working_set_size", ctypes.c_size_t),
        ("quota_peak_paged_pool_usage", ctypes.c_size_t),
        ("quota_paged_pool_usage", ctypes.c_size_t),
        ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
        ("quota_non_paged_pool_usage", ctypes.c_size_t),
        ("pagefile_usage", ctypes.c_size_t),
        ("peak_pagefile_usage", ctypes.c_size_t),
    ]


def _windows_current_rss_bytes() -> int | None:
    try:
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        windll = getattr(ctypes, "windll")
        get_current_process = windll.kernel32.GetCurrentProcess
        get_current_process.argtypes = ()
        get_current_process.restype = ctypes.c_void_p
        get_process_memory_info = windll.psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(_ProcessMemoryCounters),
            ctypes.c_ulong,
        )
        get_process_memory_info.restype = ctypes.c_int
        process = get_current_process()
        if not process:
            return None
        ok = get_process_memory_info(
            process,
            ctypes.byref(counters),
            counters.cb,
        )
        return int(counters.working_set_size) if ok else None
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _format_bytes(value: int) -> str:
    if value < 1024:
        return f"{value} B"
    return f"{value / (1024 * 1024):.2f} MiB"


__all__ = [
    "PeakRamProbe",
    "PeakRamReport",
    "RamBudget",
    "RamBudgetExceeded",
    "RamProbeUnavailable",
    "current_rss_bytes",
]
