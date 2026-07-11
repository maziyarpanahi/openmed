"""Per-device throughput and latency benchmark runner."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from openmed.eval.metrics import (
    LatencyMetrics,
    ResourceMetrics,
    compute_latency_summary,
    compute_resource_metrics,
)
from openmed.eval.tiers import NANO_SUB_TIER, TIERS

DEFAULT_PERF_WORKLOAD_PATH = Path(__file__).with_name("fixtures") / (
    "mobile_one_page_notes.jsonl"
)
SYNTHETIC_PERF_MODEL_NAME = "synthetic-one-page-note-runner"

TIER_ALIASES: Mapping[str, str] = {
    "tiny": "Tiny",
    "phone": "Tiny",
    "mobile": "Tiny",
    "tablet": "Tiny",
    "nano": "Nano",
    "base": "Base",
    "laptop": "Base",
    "cpu": "Base",
    "large": "Large",
    "workstation": "Large",
    "accurate": "Accurate-XLarge",
    "accurate-xlarge": "Accurate-XLarge",
    "xlarge": "Accurate-XLarge",
    "server": "Accurate-XLarge",
}

PerfRunner = Callable[[Any, "PerfDocument", str], Any]
Clock = Callable[[], float]
RssSampler = Callable[[], int | None]


@dataclass(frozen=True)
class PerfDocument:
    """One synthetic or user-supplied benchmark document."""

    document_id: str
    text: str
    language: str = "en"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PerfDocument":
        """Build a perf document from a JSON-compatible mapping."""
        text = str(data.get("text", ""))
        document_id = str(data.get("id") or data.get("document_id") or "document")
        language = str(data.get("language") or data.get("lang") or "en")
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            metadata = {"value": metadata}
        return cls(
            document_id=document_id,
            text=text,
            language=language,
            metadata=dict(metadata),
        )


@dataclass(frozen=True)
class TierBudget:
    """Canonical section 6.2 device-tier budget."""

    requested_tier: str
    canonical_tier: str
    ram_mb_max: int
    p50_ms_max: int
    p95_ms_max: int
    default_format: str
    parent_tier: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable budget dictionary."""
        result: dict[str, Any] = {
            "canonical_tier": self.canonical_tier,
            "default_format": self.default_format,
            "p50_ms_max": self.p50_ms_max,
            "p95_ms_max": self.p95_ms_max,
            "ram_mb_max": self.ram_mb_max,
            "requested_tier": self.requested_tier,
        }
        if self.parent_tier is not None:
            result["parent_tier"] = self.parent_tier
        return result


@dataclass(frozen=True)
class SloResult:
    """One non-gating SLO comparison result."""

    name: str
    measured: float | None
    limit: float
    unit: str
    passed: bool | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable SLO comparison dictionary."""
        return {
            "limit": self.limit,
            "measured": self.measured,
            "name": self.name,
            "passed": self.passed,
            "unit": self.unit,
        }


@dataclass(frozen=True)
class PerfReport:
    """Throughput, latency, resource, and tier-budget benchmark report."""

    model_name: str
    device: str
    tier: str
    canonical_tier: str
    document_count: int
    docs_per_second: float
    latency: LatencyMetrics
    resources: ResourceMetrics
    tier_budget: TierBudget
    slo_results: Mapping[str, SloResult]
    total_seconds: float
    generated_at: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable report dictionary."""
        return {
            "canonical_tier": self.canonical_tier,
            "device": self.device,
            "docs_per_second": self.docs_per_second,
            "document_count": self.document_count,
            "generated_at": self.generated_at,
            "latency": self.latency.to_dict(),
            "metadata": dict(self.metadata),
            "model_name": self.model_name,
            "resources": self.resources.to_dict(),
            "schema_version": 1,
            "slo_results": {
                key: result.to_dict() for key, result in self.slo_results.items()
            },
            "tier": self.tier,
            "tier_budget": self.tier_budget.to_dict(),
            "total_seconds": self.total_seconds,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the report to deterministic JSON."""
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
        """Serialize the report to deterministic Markdown."""
        lines = [
            "# Perf Report: mobile",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Model | `{self.model_name}` |",
            f"| Device | `{self.device}` |",
            f"| Tier | `{self.tier}` |",
            f"| Canonical Tier | `{self.canonical_tier}` |",
            f"| Documents | {self.document_count} |",
        ]
        if self.generated_at is not None:
            lines.append(f"| Generated At | `{self.generated_at}` |")

        lines.extend(
            [
                "",
                "## Metrics",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| `docs_per_second` | {self.docs_per_second} |",
                f"| `latency.p50_ms` | {self.latency.p50_ms} |",
                f"| `latency.p95_ms` | {self.latency.p95_ms} |",
                f"| `latency.p99_ms` | {self.latency.p99_ms} |",
                f"| `resources.peak_rss_mib` | {self.resources.peak_rss_mib} |",
                f"| `resources.model_size_mib` | {self.resources.model_size_mib} |",
            ]
        )

        lines.extend(["", "## SLO Results", "", "| SLO | Passed | Measured | Limit |"])
        lines.append("|---|---:|---:|---:|")
        for result in self.slo_results.values():
            lines.append(
                f"| `{result.name}` | {result.passed} | "
                f"{result.measured} {result.unit} | {result.limit} {result.unit} |"
            )

        return "\n".join(lines) + "\n"

    def write_markdown(self, path: str | Path) -> Path:
        """Write deterministic Markdown to *path*."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
        return output_path


def load_perf_documents(
    path: str | Path = DEFAULT_PERF_WORKLOAD_PATH,
) -> list[PerfDocument]:
    """Load perf benchmark documents from JSON or JSONL."""
    document_path = Path(path)
    if document_path.suffix.lower() == ".jsonl":
        documents = [
            PerfDocument.from_mapping(json.loads(line))
            for line in document_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(document_path.read_text(encoding="utf-8"))
        rows = payload.get("documents") if isinstance(payload, Mapping) else payload
        if not isinstance(rows, list):
            raise ValueError("perf document JSON must be a list or documents object")
        documents = [PerfDocument.from_mapping(row) for row in rows]
    _validate_documents(documents)
    return documents


def lookup_tier_budget(tier: str) -> TierBudget:
    """Return the canonical device-tier SLO budget for *tier* or an alias."""
    canonical = _canonical_tier(tier)
    if canonical == "Nano":
        return _budget_from_mapping(
            requested_tier=tier,
            canonical_tier=canonical,
            values=NANO_SUB_TIER,
            parent_tier=str(NANO_SUB_TIER["parent_tier"]),
        )
    return _budget_from_mapping(
        requested_tier=tier,
        canonical_tier=canonical,
        values=TIERS[canonical],
    )


def run_perf_benchmark(
    model: Any,
    device: str,
    tier: str,
    docs: Sequence[str | Mapping[str, Any] | PerfDocument] | None = None,
    *,
    runner: PerfRunner | None = None,
    clock: Clock | None = None,
    rss_sampler: RssSampler | None = None,
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PerfReport:
    """Run a per-document throughput and latency benchmark.

    ``model`` can be a local model path, a model identifier, or a callable. A
    string model identifier uses the existing PII runtime by default. Tests and
    offline callers can pass ``runner`` to supply deterministic inference.
    """
    documents = (
        _normalize_documents(docs)
        if docs is not None
        else load_perf_documents(DEFAULT_PERF_WORKLOAD_PATH)
    )
    _validate_documents(documents)

    now = clock or time.perf_counter
    sample_rss = rss_sampler or _peak_rss_bytes
    run_document = runner or default_perf_runner

    total_started = now()
    latencies_ms: list[float] = []
    rss_values: list[int] = []
    initial_rss = sample_rss()
    if initial_rss is not None:
        rss_values.append(initial_rss)

    for document in documents:
        started = now()
        run_document(model, document, device)
        latencies_ms.append((now() - started) * 1000.0)
        current_rss = sample_rss()
        if current_rss is not None:
            rss_values.append(current_rss)

    total_seconds = max(now() - total_started, 0.0)
    docs_per_second = len(documents) / total_seconds if total_seconds > 0 else 0.0
    latency = compute_latency_summary(latencies_ms)
    resources = compute_resource_metrics(
        peak_rss_bytes=max(rss_values) if rss_values else None,
        model_size_bytes=_model_size_bytes(model),
    )
    budget = lookup_tier_budget(tier)
    report_metadata = {
        "document_ids": [document.document_id for document in documents],
        "workload": str(DEFAULT_PERF_WORKLOAD_PATH),
    }
    report_metadata.update(metadata or {})

    return PerfReport(
        model_name=_model_name(model),
        device=device,
        tier=tier,
        canonical_tier=budget.canonical_tier,
        document_count=len(documents),
        docs_per_second=docs_per_second,
        latency=latency,
        resources=resources,
        tier_budget=budget,
        slo_results=_evaluate_slos(latency, resources, budget),
        total_seconds=total_seconds,
        generated_at=generated_at or _utc_now(),
        metadata=report_metadata,
    )


def default_perf_runner(model: Any, document: PerfDocument, device: str) -> Any:
    """Run one perf document through a callable or the PII runtime."""
    if callable(model):
        return model(document.text)

    from openmed.core.pii import extract_pii

    return extract_pii(
        document.text,
        model_name=str(model),
        lang=document.language,
    )


def synthetic_perf_runner(model: Any, document: PerfDocument, device: str) -> Any:
    """Run deterministic lightweight work for the committed synthetic workload."""
    words = document.text.split()
    return {
        "checksum": sum(ord(character) for character in document.text) % 1_000_003,
        "device": device,
        "document_id": document.document_id,
        "model": str(model),
        "word_count": len(words),
    }


def _evaluate_slos(
    latency: LatencyMetrics,
    resources: ResourceMetrics,
    budget: TierBudget,
) -> dict[str, SloResult]:
    peak_rss_mib = resources.peak_rss_mib
    return {
        "p50_latency_ms": SloResult(
            name="p50_latency_ms",
            measured=latency.p50_ms,
            limit=float(budget.p50_ms_max),
            unit="ms",
            passed=latency.p50_ms <= budget.p50_ms_max,
        ),
        "p95_latency_ms": SloResult(
            name="p95_latency_ms",
            measured=latency.p95_ms,
            limit=float(budget.p95_ms_max),
            unit="ms",
            passed=latency.p95_ms <= budget.p95_ms_max,
        ),
        "peak_rss_mib": SloResult(
            name="peak_rss_mib",
            measured=peak_rss_mib,
            limit=float(budget.ram_mb_max),
            unit="MiB",
            passed=(
                None
                if peak_rss_mib is None
                else peak_rss_mib <= float(budget.ram_mb_max)
            ),
        ),
    }


def _budget_from_mapping(
    *,
    requested_tier: str,
    canonical_tier: str,
    values: Mapping[str, Any],
    parent_tier: str | None = None,
) -> TierBudget:
    return TierBudget(
        requested_tier=requested_tier,
        canonical_tier=canonical_tier,
        ram_mb_max=int(values["ram_mb_max"]),
        p50_ms_max=int(values["p50_ms_max"]),
        p95_ms_max=int(values["p95_ms_max"]),
        default_format=str(values["default_format"]),
        parent_tier=parent_tier,
    )


def _canonical_tier(tier: str) -> str:
    key = tier.strip().lower().replace("_", "-").replace("/", "-")
    canonical = TIER_ALIASES.get(key)
    if canonical is None:
        valid = ", ".join(sorted(TIER_ALIASES))
        raise ValueError(f"unknown device tier {tier!r}; expected one of: {valid}")
    return canonical


def _normalize_documents(
    docs: Sequence[str | Mapping[str, Any] | PerfDocument],
) -> list[PerfDocument]:
    documents: list[PerfDocument] = []
    for index, document in enumerate(docs, start=1):
        if isinstance(document, PerfDocument):
            documents.append(document)
        elif isinstance(document, str):
            documents.append(
                PerfDocument(
                    document_id=f"document-{index:03d}",
                    text=document,
                )
            )
        elif isinstance(document, Mapping):
            documents.append(PerfDocument.from_mapping(document))
        else:
            raise TypeError("docs must contain strings, mappings, or PerfDocument")
    return documents


def _validate_documents(documents: Sequence[PerfDocument]) -> None:
    if not documents:
        raise ValueError("at least one benchmark document is required")
    seen: set[str] = set()
    duplicates: list[str] = []
    for document in documents:
        if not document.text:
            raise ValueError(f"benchmark document {document.document_id!r} is empty")
        if document.document_id in seen and document.document_id not in duplicates:
            duplicates.append(document.document_id)
        seen.add(document.document_id)
    if duplicates:
        quoted = ", ".join(repr(value) for value in duplicates)
        raise ValueError(f"duplicate benchmark document id(s): {quoted}")


def _model_name(model: Any) -> str:
    if isinstance(model, (str, Path)):
        return str(model)
    name = getattr(model, "__name__", None)
    if name:
        return str(name)
    return model.__class__.__name__


def _model_size_bytes(model: Any) -> int | None:
    paths: list[Path] = []
    if isinstance(model, (str, Path)):
        paths.append(Path(model))
    for attr in ("model_path", "path"):
        value = getattr(model, attr, None)
        if value is not None:
            paths.append(Path(value))

    for path in paths:
        if path.exists():
            if path.is_file():
                return path.stat().st_size
            if path.is_dir():
                return sum(
                    item.stat().st_size for item in path.rglob("*") if item.is_file()
                )
    return None


def _peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = int(usage.ru_maxrss)
    if sys.platform == "darwin":
        return rss
    return rss * 1024


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


__all__ = [
    "DEFAULT_PERF_WORKLOAD_PATH",
    "SYNTHETIC_PERF_MODEL_NAME",
    "PerfDocument",
    "PerfReport",
    "PerfRunner",
    "SloResult",
    "TierBudget",
    "default_perf_runner",
    "load_perf_documents",
    "lookup_tier_budget",
    "run_perf_benchmark",
    "synthetic_perf_runner",
]
