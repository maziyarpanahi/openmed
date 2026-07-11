"""Memory profiler for the inference / de-identification path.

The perf runner in :mod:`openmed.eval.perf` reports peak RSS as a single number.
This module profiles *where* memory goes across the inference path -- model load
vs. the first forward pass vs. a steady-state batch -- so tier-budget overshoots
can be diagnosed with a reproducible breakdown.

Each phase records a baseline RSS, a peak RSS (reusing the resource-metric
helpers in :mod:`openmed.eval.metrics`), and a ``tracemalloc`` top-allocator
snapshot. Allocator entries carry only ``file:lineno`` provenance and byte
counts -- never document text -- so the report stays local-first and free of raw
PHI.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import sys
import threading
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from openmed.eval.metrics import ResourceMetrics, compute_resource_metrics
from openmed.eval.perf import (
    PerfDocument,
    load_perf_documents,
    synthetic_perf_runner,
)

#: Ordered inference-path phases the profiler reports.
PHASE_LOAD = "load"
PHASE_FIRST_FORWARD = "first-forward"
PHASE_STEADY_STATE = "steady-state-batch"
PROFILE_PHASES: tuple[str, ...] = (PHASE_LOAD, PHASE_FIRST_FORWARD, PHASE_STEADY_STATE)

SYNTHETIC_MEMPROFILE_MODEL_NAME = "synthetic-one-page-note-runner"
DEFAULT_TOP_ALLOCATORS = 10
_RSS_POLL_INTERVAL_SECONDS = 0.005

#: A model loader takes the ``model`` handle and returns a runnable callable.
ModelLoader = Callable[[Any], Callable[[PerfDocument], Any]]
#: An RSS sampler returns the current/peak process RSS in bytes (or ``None``).
RssSampler = Callable[[], "int | None"]


@dataclass(frozen=True)
class AllocatorStat:
    """One ``tracemalloc`` top-allocator entry, free of raw PHI.

    Only source provenance (``file``/``lineno``) and byte/count aggregates are
    retained -- never the allocated payload -- so the record is safe to log.
    """

    file: str
    lineno: int
    size_bytes: int
    count: int

    @property
    def size_kib(self) -> float:
        """Return the allocation size in kibibytes."""
        return self.size_bytes / 1024

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable allocator record."""
        return {
            "file": self.file,
            "lineno": self.lineno,
            "size_bytes": self.size_bytes,
            "size_kib": self.size_kib,
            "count": self.count,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


@dataclass(frozen=True)
class PhaseMemory:
    """Memory measurement for a single inference-path phase."""

    phase: str
    baseline_rss_bytes: int | None
    peak_rss_bytes: int | None
    traced_current_bytes: int
    traced_peak_bytes: int
    top_allocators: tuple[AllocatorStat, ...]
    tracemalloc_preexisting: bool = False

    @property
    def resources(self) -> ResourceMetrics:
        """Return the shared resource summary for this phase's peak RSS."""
        return compute_resource_metrics(peak_rss_bytes=self.peak_rss_bytes)

    @property
    def rss_delta_bytes(self) -> int | None:
        """Return peak-minus-baseline RSS growth in bytes, when available."""
        if self.peak_rss_bytes is None or self.baseline_rss_bytes is None:
            return None
        return max(self.peak_rss_bytes - self.baseline_rss_bytes, 0)

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable phase dictionary."""
        return {
            "phase": self.phase,
            "baseline_rss_bytes": self.baseline_rss_bytes,
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_rss_mib": self.resources.peak_rss_mib,
            "rss_delta_bytes": self.rss_delta_bytes,
            "rss_semantics": "current-sampled",
            "traced_current_bytes": self.traced_current_bytes,
            "traced_peak_bytes": self.traced_peak_bytes,
            "tracemalloc_preexisting": self.tracemalloc_preexisting,
            "top_allocators": [stat.to_dict() for stat in self.top_allocators],
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


@dataclass(frozen=True)
class MemoryProfile:
    """Per-phase memory breakdown across the inference path."""

    model_name: str
    document_count: int
    phases: tuple[PhaseMemory, ...]
    generated_at: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def phase(self, name: str) -> PhaseMemory:
        """Return the recorded phase named *name*."""
        for entry in self.phases:
            if entry.phase == name:
                return entry
        raise KeyError(f"unknown profile phase: {name!r}")

    @property
    def peak_rss_bytes(self) -> int | None:
        """Return the maximum peak RSS observed across all phases."""
        values = [
            entry.peak_rss_bytes
            for entry in self.phases
            if entry.peak_rss_bytes is not None
        ]
        return max(values) if values else None

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable profile dictionary."""
        return {
            "document_count": self.document_count,
            "generated_at": self.generated_at,
            "metadata": dict(self.metadata),
            "model_name": self.model_name,
            "peak_rss_bytes": self.peak_rss_bytes,
            "phase_order": [entry.phase for entry in self.phases],
            "phases": {entry.phase: entry.to_dict() for entry in self.phases},
            "schema_version": 1,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the profile to deterministic JSON."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write deterministic JSON to *path*."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    def to_markdown(self) -> str:
        """Serialize the profile to a deterministic Markdown table."""
        lines = [
            "# Memory Profile",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Model | `{self.model_name}` |",
            f"| Documents | {self.document_count} |",
        ]
        if self.generated_at is not None:
            lines.append(f"| Generated At | `{self.generated_at}` |")

        lines.extend(
            [
                "",
                "## Phase Breakdown",
                "",
                "| Phase | Peak RSS (MiB) | RSS Delta (bytes) | Traced Peak (bytes) |",
                "|---|---:|---:|---:|",
            ]
        )
        for entry in self.phases:
            peak_mib = entry.resources.peak_rss_mib
            peak_text = "n/a" if peak_mib is None else f"{peak_mib:.2f}"
            delta_text = (
                "n/a" if entry.rss_delta_bytes is None else str(entry.rss_delta_bytes)
            )
            lines.append(
                f"| `{entry.phase}` | {peak_text} | {delta_text} | "
                f"{entry.traced_peak_bytes} |"
            )

        for entry in self.phases:
            lines.extend(
                [
                    "",
                    f"## Top Allocators: {entry.phase}",
                    "",
                    "| Location | Size (KiB) | Blocks |",
                    "|---|---:|---:|",
                ]
            )
            if not entry.top_allocators:
                lines.append("| _(none)_ | 0.00 | 0 |")
                continue
            for stat in entry.top_allocators:
                lines.append(
                    f"| `{stat.file}:{stat.lineno}` | "
                    f"{stat.size_kib:.2f} | {stat.count} |"
                )

        return "\n".join(lines) + "\n"

    def write_markdown(self, path: str | Path) -> Path:
        """Write deterministic Markdown to *path*."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
        return output_path

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


def profile_memory(
    model: Any,
    docs: Sequence[str | Mapping[str, Any] | PerfDocument] | None = None,
    *,
    loader: ModelLoader | None = None,
    rss_sampler: RssSampler | None = None,
    top_allocators: int = DEFAULT_TOP_ALLOCATORS,
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> MemoryProfile:
    """Profile inference-path memory across load, first-forward, and batch phases.

    ``model`` can be a local model path, a model identifier, or a callable. The
    ``loader`` is invoked once (measured as the ``load`` phase) and must return a
    per-document callable; when omitted, a string identifier is loaded through a
    shared :class:`openmed.core.models.ModelLoader` before inference, while a
    callable model is used directly. The first document runs under the
    ``first-forward`` phase and the remaining documents (or a re-run of the
    single document) under the ``steady-state-batch`` phase.

    Each phase captures a baseline RSS, a peak RSS via the shared resource-metric
    helpers, and a ``tracemalloc`` top-allocator snapshot. Only file/line
    provenance and byte counts are retained -- no document text -- so the returned
    :class:`MemoryProfile` is safe to persist and log.

    Args:
        model: Model path, identifier, or callable to profile.
        docs: Optional workload; defaults to the committed synthetic workload.
        loader: Optional loader returning a per-document callable.
        rss_sampler: Optional current-RSS sampler (bytes); defaults to a
            platform-native process RSS sampler.
        top_allocators: Number of top allocator entries to keep per phase.
        generated_at: Optional ISO timestamp override for deterministic output.
        metadata: Optional extra metadata merged into the profile.

    Returns:
        A :class:`MemoryProfile` with per-phase peak RSS and top allocators for
        the load, first-forward, and steady-state phases, in that order.
    """
    documents = (
        _normalize_documents(docs) if docs is not None else load_perf_documents()
    )
    if not documents:
        raise ValueError("at least one profiling document is required")
    if top_allocators < 1:
        raise ValueError("top_allocators must be at least 1")

    sample_rss = rss_sampler or _current_rss_bytes
    load_model = loader or _default_loader

    phases: list[PhaseMemory] = []

    # Phase 1: model load.
    runnable, load_phase = _measure_phase(
        PHASE_LOAD,
        lambda: load_model(model),
        sample_rss=sample_rss,
        top_allocators=top_allocators,
    )
    phases.append(load_phase)
    if not callable(runnable):
        raise TypeError("loader must return a callable that runs one document")

    # Phase 2: first forward pass over the first document.
    first_document = documents[0]
    _, first_phase = _measure_phase(
        PHASE_FIRST_FORWARD,
        lambda: runnable(first_document),
        sample_rss=sample_rss,
        top_allocators=top_allocators,
    )
    phases.append(first_phase)

    # Phase 3: steady-state batch over the remaining documents (or a re-run of
    # the single document so the phase is always exercised).
    batch = documents[1:] or [first_document]

    def run_batch() -> list[Any]:
        return [runnable(document) for document in batch]

    _, steady_phase = _measure_phase(
        PHASE_STEADY_STATE,
        run_batch,
        sample_rss=sample_rss,
        top_allocators=top_allocators,
    )
    phases.append(steady_phase)

    profile_metadata = {
        "document_hashes": [
            _document_hash(document.document_id) for document in documents
        ],
        "top_allocators": top_allocators,
    }
    profile_metadata.update(metadata or {})

    return MemoryProfile(
        model_name=_model_name(model),
        document_count=len(documents),
        phases=tuple(phases),
        generated_at=generated_at or _utc_now(),
        metadata=profile_metadata,
    )


def synthetic_memprofile_loader(model: Any) -> Callable[[PerfDocument], Any]:
    """Return a deterministic loader over the committed synthetic workload."""

    def run_document(document: PerfDocument) -> Any:
        return synthetic_perf_runner(model, document, "cpu")

    return run_document


def _default_loader(model: Any) -> Callable[[PerfDocument], Any]:
    """Load the model now and return a warmed per-document inference callable."""

    if callable(model):
        return lambda document: model(document.text)

    from openmed.core.models import ModelLoader
    from openmed.core.pii import extract_pii

    model_name = str(model)
    shared_loader = ModelLoader()
    shared_loader.create_pipeline(
        model_name,
        task="token-classification",
        aggregation_strategy="simple",
        use_fast_tokenizer=True,
    )

    def run_document(document: PerfDocument) -> Any:
        return extract_pii(
            document.text,
            model_name=model_name,
            lang=document.language,
            loader=shared_loader,
        )

    return run_document


def _measure_phase(
    phase: str,
    work: Callable[[], Any],
    *,
    sample_rss: RssSampler,
    top_allocators: int,
) -> tuple[Any, PhaseMemory]:
    """Run *work* under tracemalloc and RSS sampling, returning its result."""
    rss_monitor = _RssMonitor(sample_rss)
    baseline_rss = rss_monitor.start()
    started_tracing = not tracemalloc.is_tracing()
    if started_tracing:
        tracemalloc.start()
    baseline_snapshot = tracemalloc.take_snapshot()
    baseline_current, baseline_peak = tracemalloc.get_traced_memory()
    try:
        result = work()
        snapshot = tracemalloc.take_snapshot()
        traced_current, traced_peak = tracemalloc.get_traced_memory()
    finally:
        peak_rss_bytes = rss_monitor.stop()
        if started_tracing:
            tracemalloc.stop()

    allocator_stats = _top_allocators(
        snapshot,
        baseline_snapshot,
        top_allocators,
    )
    phase_current = max(int(traced_current - baseline_current), 0)
    retained_growth = sum(stat.size_bytes for stat in allocator_stats)
    phase_peak = max(
        int(traced_peak - baseline_peak),
        phase_current,
        retained_growth,
        0,
    )
    return result, PhaseMemory(
        phase=phase,
        baseline_rss_bytes=baseline_rss,
        peak_rss_bytes=peak_rss_bytes,
        traced_current_bytes=phase_current,
        traced_peak_bytes=phase_peak,
        top_allocators=allocator_stats,
        tracemalloc_preexisting=not started_tracing,
    )


def _top_allocators(
    snapshot: tracemalloc.Snapshot,
    baseline: tracemalloc.Snapshot,
    limit: int,
) -> tuple[AllocatorStat, ...]:
    """Return the top *limit* allocators as PHI-free provenance records."""
    stats = snapshot.compare_to(baseline, "lineno")
    top: list[AllocatorStat] = []
    for stat in stats:
        if stat.size_diff <= 0 or stat.count_diff <= 0:
            continue
        frame = stat.traceback[0] if stat.traceback else None
        top.append(
            AllocatorStat(
                file=frame.filename if frame is not None else "<unknown>",
                lineno=frame.lineno if frame is not None else 0,
                size_bytes=int(stat.size_diff),
                count=int(stat.count_diff),
            )
        )
        if len(top) >= limit:
            break
    return tuple(top)


class _RssMonitor:
    """Sample current RSS during one phase without retaining payload data."""

    def __init__(self, sampler: RssSampler) -> None:
        self._sampler = sampler
        self._samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> int | None:
        baseline = self._record()
        self._thread = threading.Thread(
            target=self._poll,
            name="openmed-rss-monitor",
            daemon=True,
        )
        self._thread.start()
        return baseline

    def stop(self) -> int | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._record()
        return max(self._samples) if self._samples else None

    def _poll(self) -> None:
        while not self._stop.wait(_RSS_POLL_INTERVAL_SECONDS):
            self._record()

    def _record(self) -> int | None:
        try:
            value = self._sampler()
        except Exception:
            return None
        if value is not None:
            self._samples.append(max(int(value), 0))
            return self._samples[-1]
        return None


def _current_rss_bytes() -> int | None:
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
        process = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            process,
            ctypes.byref(counters),
            counters.cb,
        )
        return int(counters.working_set_size) if ok else None
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _normalize_documents(
    docs: Sequence[str | Mapping[str, Any] | PerfDocument],
) -> list[PerfDocument]:
    documents: list[PerfDocument] = []
    for index, document in enumerate(docs, start=1):
        if isinstance(document, PerfDocument):
            documents.append(document)
        elif isinstance(document, str):
            documents.append(
                PerfDocument(document_id=f"document-{index:03d}", text=document)
            )
        elif isinstance(document, Mapping):
            documents.append(PerfDocument.from_mapping(document))
        else:
            raise TypeError("docs must contain strings, mappings, or PerfDocument")
    return documents


def _model_name(model: Any) -> str:
    if isinstance(model, (str, Path)):
        return str(model)
    name = getattr(model, "__name__", None)
    if name:
        return str(name)
    return model.__class__.__name__


def _document_hash(document_id: str) -> str:
    digest = hashlib.sha256(document_id.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


__all__ = [
    "DEFAULT_TOP_ALLOCATORS",
    "PHASE_FIRST_FORWARD",
    "PHASE_LOAD",
    "PHASE_STEADY_STATE",
    "PROFILE_PHASES",
    "SYNTHETIC_MEMPROFILE_MODEL_NAME",
    "AllocatorStat",
    "MemoryProfile",
    "PhaseMemory",
    "profile_memory",
    "synthetic_memprofile_loader",
]
