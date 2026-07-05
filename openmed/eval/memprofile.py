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

import json
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from openmed.eval.metrics import ResourceMetrics, compute_resource_metrics
from openmed.eval.perf import (
    PerfDocument,
    _peak_rss_bytes,
    default_perf_runner,
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
            "traced_current_bytes": self.traced_current_bytes,
            "traced_peak_bytes": self.traced_peak_bytes,
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
    per-document callable; when omitted, the loader wraps
    :func:`openmed.eval.perf.default_perf_runner` so a string identifier drives
    the existing PII runtime and a callable model is used directly. The first
    document runs under the ``first-forward`` phase and the remaining documents
    (or a re-run of the single document) under the ``steady-state-batch`` phase.

    Each phase captures a baseline RSS, a peak RSS via the shared resource-metric
    helpers, and a ``tracemalloc`` top-allocator snapshot. Only file/line
    provenance and byte counts are retained -- no document text -- so the returned
    :class:`MemoryProfile` is safe to persist and log.

    Args:
        model: Model path, identifier, or callable to profile.
        docs: Optional workload; defaults to the committed synthetic workload.
        loader: Optional loader returning a per-document callable.
        rss_sampler: Optional RSS sampler (bytes); defaults to the process peak.
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

    sample_rss = rss_sampler or _peak_rss_bytes
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
        "document_ids": [document.document_id for document in documents],
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
    """Wrap the perf default runner as a per-document callable."""

    def run_document(document: PerfDocument) -> Any:
        return default_perf_runner(model, document, "cpu")

    return run_document


def _measure_phase(
    phase: str,
    work: Callable[[], Any],
    *,
    sample_rss: RssSampler,
    top_allocators: int,
) -> tuple[Any, PhaseMemory]:
    """Run *work* under tracemalloc and RSS sampling, returning its result."""
    baseline_rss = sample_rss()
    started_tracing = not tracemalloc.is_tracing()
    if started_tracing:
        tracemalloc.start()
    else:
        tracemalloc.clear_traces()
    tracemalloc.reset_peak()
    try:
        result = work()
        snapshot = tracemalloc.take_snapshot()
        traced_current, traced_peak = tracemalloc.get_traced_memory()
    finally:
        if started_tracing:
            tracemalloc.stop()

    peak_rss = sample_rss()
    peak_rss_bytes = _max_optional(baseline_rss, peak_rss)
    return result, PhaseMemory(
        phase=phase,
        baseline_rss_bytes=baseline_rss,
        peak_rss_bytes=peak_rss_bytes,
        traced_current_bytes=int(traced_current),
        traced_peak_bytes=int(traced_peak),
        top_allocators=_top_allocators(snapshot, top_allocators),
    )


def _top_allocators(
    snapshot: tracemalloc.Snapshot,
    limit: int,
) -> tuple[AllocatorStat, ...]:
    """Return the top *limit* allocators as PHI-free provenance records."""
    stats = snapshot.statistics("lineno")
    top: list[AllocatorStat] = []
    for stat in stats[:limit]:
        frame = stat.traceback[0] if stat.traceback else None
        top.append(
            AllocatorStat(
                file=frame.filename if frame is not None else "<unknown>",
                lineno=frame.lineno if frame is not None else 0,
                size_bytes=int(stat.size),
                count=int(stat.count),
            )
        )
    return tuple(top)


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


def _max_optional(*values: int | None) -> int | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


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
