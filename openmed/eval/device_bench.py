"""Batch-aware device benchmark runner and deterministic result archives."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from openmed.eval import harness
from openmed.eval.metrics import LatencyMetrics, ResourceMetrics
from openmed.eval.perf import (
    DEFAULT_PERF_WORKLOAD_PATH,
    SYNTHETIC_PERF_MODEL_NAME,
    PerfDocument,
    default_perf_runner,
    load_perf_documents,
    lookup_tier_budget,
    synthetic_perf_runner,
)

DEVICE_BENCH_SCHEMA_VERSION = 1
SUPPORTED_DEVICES = ("cpu", "mlx", "coreml")
DEFAULT_SEQUENCE_LENGTHS = (128, 256)
DEFAULT_BATCH_SIZES = (1, 2)
DEFAULT_DEVICE_ARCHIVE_DIR = (
    Path(__file__).resolve().parents[2] / "eval" / "results" / "device"
)
DEFAULT_DEVICE_CORPUS_PATH = DEFAULT_PERF_WORKLOAD_PATH
_RAW_METADATA_KEYS = frozenset(
    {
        "deidentified_text",
        "mention_text",
        "raw_text",
        "source_text",
        "span_text",
        "text",
    }
)

Clock = Callable[[], float]
RssSampler = Callable[[], int | None]
DeviceBenchRunner = Callable[[Any, Sequence[PerfDocument], str, int], Any]


@dataclass(frozen=True)
class DeviceBenchMeasurement:
    """One timed batch in a device benchmark matrix."""

    sequence_length: int
    batch_size: int
    document_count: int
    repeat: int
    latency_ms: float
    docs_per_second: float

    def to_dict(self) -> dict[str, int | float]:
        """Return the measurement without document text or model output."""
        return {
            "batch_size": self.batch_size,
            "docs_per_second": self.docs_per_second,
            "document_count": self.document_count,
            "latency_ms": self.latency_ms,
            "repeat": self.repeat,
            "sequence_length": self.sequence_length,
        }


@dataclass(frozen=True)
class DeviceBenchResult:
    """Aggregate throughput, latency, and memory metrics for one run."""

    repo_id: str
    format: str
    device: str
    tier: str
    canonical_tier: str
    sequence_lengths: tuple[int, ...]
    batch_sizes: tuple[int, ...]
    document_count: int
    batch_count: int
    docs_per_second: float
    latency: LatencyMetrics
    resources: ResourceMetrics
    total_seconds: float
    tier_budget: Mapping[str, Any]
    measurements: tuple[DeviceBenchMeasurement, ...]
    generated_at: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def model_format(self) -> str:
        """Return the format using the descriptive API name."""
        return self.format

    @property
    def archive_key(self) -> str:
        """Return the stable join key for this repository/device result."""
        return _archive_key(self.repo_id, self.format, self.device)

    @property
    def p50_ms(self) -> float:
        """Return aggregate p50 latency in milliseconds."""
        return self.latency.p50_ms

    @property
    def p95_ms(self) -> float:
        """Return aggregate p95 latency in milliseconds."""
        return self.latency.p95_ms

    @property
    def peak_rss_mib(self) -> float | None:
        """Return peak resident memory in MiB when RSS is available."""
        return self.resources.peak_rss_mib

    def to_dict(self) -> dict[str, Any]:
        """Return the stable, archive-safe JSON representation."""
        metrics = {
            "docs_per_second": self.docs_per_second,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "peak_rss_mib": self.peak_rss_mib,
        }
        return {
            "archive_key": self.archive_key,
            "batch_count": self.batch_count,
            "batch_sizes": list(self.batch_sizes),
            "canonical_tier": self.canonical_tier,
            "device": self.device,
            "docs_per_second": self.docs_per_second,
            "document_count": self.document_count,
            "format": self.format,
            "generated_at": self.generated_at,
            "key": {
                "device": self.device,
                "format": self.format,
                "repo_id": self.repo_id,
                "tier": self.canonical_tier,
            },
            "latency": self.latency.to_dict(),
            "measurements": [item.to_dict() for item in self.measurements],
            "metadata": dict(self.metadata),
            "metrics": metrics,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "peak_rss_mib": self.peak_rss_mib,
            "repo_id": self.repo_id,
            "resources": self.resources.to_dict(),
            "schema_version": DEVICE_BENCH_SCHEMA_VERSION,
            "sequence_lengths": list(self.sequence_lengths),
            "tier": self.tier,
            "tier_budget": dict(self.tier_budget),
            "total_seconds": self.total_seconds,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the result with stable key ordering."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write this result as a standalone JSON report."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path


def load_device_corpus(
    path: str | Path = DEFAULT_DEVICE_CORPUS_PATH,
) -> list[PerfDocument]:
    """Load the fixed synthetic corpus used by the device benchmark."""
    return load_perf_documents(path)


def run_device_benchmark(
    model: Any,
    device: str = "cpu",
    tier: str = "base",
    *,
    repo_id: str | None = None,
    model_format: str = "INT8",
    format: str | None = None,
    corpus: str | Path | Sequence[str | Mapping[str, Any] | PerfDocument] | None = None,
    docs: Sequence[str | Mapping[str, Any] | PerfDocument] | None = None,
    sequence_lengths: Sequence[int | str] | None = None,
    batch_sizes: Sequence[int | str] | None = None,
    repeats: int = 1,
    runner: DeviceBenchRunner | None = None,
    clock: Clock | None = None,
    rss_sampler: RssSampler | None = None,
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DeviceBenchResult:
    """Run a model over sequence-length and batch-size combinations.

    The runner receives ``(model, batch, device, sequence_length)``. The
    default runner executes each document locally, while callers can provide a
    backend-specific batch runner for CPU, MLX, or Core ML. Timing and RSS
    functions are injectable so synthetic tests remain deterministic and
    offline.
    """
    if format is not None:
        if model_format != "INT8":
            raise ValueError("use either model_format or format, not both")
        model_format = format
    normalized_device = _normalize_device(device)
    normalized_format = _normalize_format(model_format)
    normalized_tier = str(tier).strip()
    if not normalized_tier:
        raise ValueError("tier must be non-empty")
    budget = lookup_tier_budget(normalized_tier)
    normalized_lengths = _normalize_positive_values(
        sequence_lengths,
        name="sequence_lengths",
        default=DEFAULT_SEQUENCE_LENGTHS,
    )
    normalized_batches = _normalize_positive_values(
        batch_sizes,
        name="batch_sizes",
        default=DEFAULT_BATCH_SIZES,
    )
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if corpus is not None and docs is not None:
        raise ValueError("use either corpus or docs, not both")
    documents = _resolve_documents(corpus if corpus is not None else docs)
    _validate_documents(documents)

    run_batch = runner or default_device_bench_runner
    now = clock or time.perf_counter
    sample_rss = rss_sampler or _peak_rss_bytes
    rss_values: list[int] = []
    initial_rss = sample_rss()
    if initial_rss is not None:
        rss_values.append(initial_rss)

    measurements: list[DeviceBenchMeasurement] = []
    latencies_ms: list[float] = []
    total_seconds = 0.0
    total_documents = 0
    for sequence_length in normalized_lengths:
        prepared_documents = _resize_documents(documents, sequence_length)
        for requested_batch_size in normalized_batches:
            batches = _batches(prepared_documents, requested_batch_size)
            for repeat in range(repeats):
                for batch in batches:
                    started = now()
                    run_batch(model, batch, normalized_device, sequence_length)
                    elapsed_seconds = max(now() - started, 0.0)
                    document_count = len(batch)
                    latency_ms = elapsed_seconds * 1000.0
                    docs_per_second = (
                        document_count / elapsed_seconds if elapsed_seconds > 0 else 0.0
                    )
                    measurements.append(
                        DeviceBenchMeasurement(
                            sequence_length=sequence_length,
                            batch_size=requested_batch_size,
                            document_count=document_count,
                            repeat=repeat,
                            latency_ms=latency_ms,
                            docs_per_second=docs_per_second,
                        )
                    )
                    latencies_ms.append(latency_ms)
                    total_seconds += elapsed_seconds
                    total_documents += document_count
                    current_rss = sample_rss()
                    if current_rss is not None:
                        rss_values.append(current_rss)

    peak_rss_bytes = max(rss_values) if rss_values else None
    latency = harness.compute_latency_summary(latencies_ms)
    resources = harness.compute_resource_metrics(peak_rss_bytes=peak_rss_bytes)
    report_metadata = {
        "corpus_document_count": len(documents),
        "repeats": repeats,
    }
    report_metadata.update(_safe_metadata(metadata))
    identifier = _repo_id(model, repo_id)
    return DeviceBenchResult(
        repo_id=identifier,
        format=normalized_format,
        device=normalized_device,
        tier=normalized_tier,
        canonical_tier=budget.canonical_tier,
        sequence_lengths=normalized_lengths,
        batch_sizes=normalized_batches,
        document_count=total_documents,
        batch_count=len(measurements),
        docs_per_second=(total_documents / total_seconds if total_seconds > 0 else 0.0),
        latency=latency,
        resources=resources,
        total_seconds=total_seconds,
        tier_budget=budget.to_dict(),
        measurements=tuple(measurements),
        generated_at=generated_at,
        metadata=report_metadata,
    )


def default_device_bench_runner(
    model: Any,
    batch: Sequence[PerfDocument],
    device: str,
    sequence_length: int,
) -> list[Any]:
    """Run a non-synthetic model locally over every document in a batch."""
    del sequence_length
    if callable(model):
        return [model(document.text) for document in batch]
    return [default_perf_runner(model, document, device) for document in batch]


def synthetic_device_bench_runner(
    model: Any,
    batch: Sequence[PerfDocument],
    device: str,
    sequence_length: int,
) -> list[Any]:
    """Run deterministic work for the committed synthetic benchmark model."""
    del sequence_length
    return [synthetic_perf_runner(model, document, device) for document in batch]


def archive_path_for(
    result: DeviceBenchResult,
    archive_dir: str | Path = DEFAULT_DEVICE_ARCHIVE_DIR,
) -> Path:
    """Return the stable archive path for a format/device/tier combination."""
    root = Path(archive_dir)
    if root.suffix.lower() == ".json":
        return root
    return root / (
        f"{_path_token(result.format)}-"
        f"{_path_token(result.device)}-"
        f"{_path_token(result.canonical_tier)}.json"
    )


def write_device_benchmark_archive(
    result: DeviceBenchResult,
    archive_dir: str | Path = DEFAULT_DEVICE_ARCHIVE_DIR,
    *,
    archive_path: str | Path | None = None,
) -> Path:
    """Merge *result* into its deterministic per-device JSON archive."""
    output_path = (
        Path(archive_path)
        if archive_path is not None
        else archive_path_for(result, archive_dir)
    )
    payload = _empty_archive(result)
    if output_path.exists():
        payload = _read_archive(output_path, result)
    entries = payload["results"]
    entries[result.repo_id] = result.to_dict()
    payload["results"] = {key: entries[key] for key in sorted(entries)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_device_benchmark_archive(
    path: str | Path,
) -> dict[str, Any]:
    """Load an archive without exposing any corpus text."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), dict):
        raise ValueError("device benchmark archive must contain a results object")
    return payload


def _resolve_documents(
    corpus: str | Path | Sequence[str | Mapping[str, Any] | PerfDocument] | None,
) -> list[PerfDocument]:
    if corpus is None:
        return load_device_corpus()
    if isinstance(corpus, (str, Path)):
        return load_device_corpus(corpus)
    documents: list[PerfDocument] = []
    for index, item in enumerate(corpus, start=1):
        if isinstance(item, PerfDocument):
            documents.append(item)
        elif isinstance(item, str):
            documents.append(
                PerfDocument(document_id=f"document-{index:03d}", text=item)
            )
        elif isinstance(item, Mapping):
            documents.append(PerfDocument.from_mapping(item))
        else:
            raise TypeError("corpus must contain strings, mappings, or PerfDocument")
    return documents


def _resize_documents(
    documents: Sequence[PerfDocument],
    sequence_length: int,
) -> tuple[PerfDocument, ...]:
    resized: list[PerfDocument] = []
    for document in documents:
        tokens = document.text.split()
        if len(tokens) >= sequence_length:
            text = " ".join(tokens[:sequence_length])
        else:
            text = " ".join(
                [*tokens, *(["synthetic-padding"] * (sequence_length - len(tokens)))]
            )
        resized.append(
            PerfDocument(
                document_id=document.document_id,
                text=text,
                language=document.language,
                metadata=document.metadata,
            )
        )
    return tuple(resized)


def _batches(
    documents: Sequence[PerfDocument],
    batch_size: int,
) -> tuple[tuple[PerfDocument, ...], ...]:
    return tuple(
        tuple(documents[index : index + batch_size])
        for index in range(0, len(documents), batch_size)
    )


def _validate_documents(documents: Sequence[PerfDocument]) -> None:
    if not documents:
        raise ValueError("at least one benchmark document is required")
    seen: set[str] = set()
    for document in documents:
        if not document.text:
            raise ValueError(f"benchmark document {document.document_id!r} is empty")
        if document.document_id in seen:
            raise ValueError(
                f"duplicate benchmark document id: {document.document_id!r}"
            )
        seen.add(document.document_id)


def _normalize_positive_values(
    values: Sequence[int | str] | None,
    *,
    name: str,
    default: Sequence[int],
) -> tuple[int, ...]:
    raw_values: Sequence[int | str] = default if values is None else values
    if isinstance(raw_values, (str, bytes)):
        raw_values = [raw_values]  # type: ignore[list-item]
    parsed: set[int] = set()
    for raw_value in raw_values:
        for item in str(raw_value).split(","):
            value = item.strip()
            if not value:
                continue
            try:
                parsed_value = int(value)
            except ValueError as exc:
                raise ValueError(f"{name} must contain positive integers") from exc
            if parsed_value < 1:
                raise ValueError(f"{name} must contain positive integers")
            parsed.add(parsed_value)
    if not parsed:
        raise ValueError(f"{name} must contain at least one positive integer")
    return tuple(sorted(parsed))


def _normalize_device(device: str) -> str:
    normalized = str(device).strip().lower()
    if normalized not in SUPPORTED_DEVICES:
        allowed = ", ".join(SUPPORTED_DEVICES)
        raise ValueError(f"unsupported device {device!r}; expected one of: {allowed}")
    return normalized


def _normalize_format(model_format: str) -> str:
    normalized = str(model_format).strip().upper()
    if not normalized or "/" in normalized or "\\" in normalized:
        raise ValueError("format must be a non-empty filename-safe value")
    return normalized


def _repo_id(model: Any, repo_id: str | None) -> str:
    if repo_id is not None:
        value = str(repo_id).strip()
    elif isinstance(model, (str, Path)):
        value = str(model)
    else:
        value = str(getattr(model, "__name__", "") or model.__class__.__name__)
    if not value:
        raise ValueError("repo_id must be non-empty")
    return value


def _peak_rss_bytes() -> int | None:
    """Reuse the eval harness RSS primitive for platform normalization."""
    return harness._peak_rss_bytes()


def _safe_metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not value:
        return {}
    return _safe_metadata_value(value)


def _safe_metadata_value(value: Any, *, key: str | None = None) -> Any:
    if key is not None and key.lower() in _RAW_METADATA_KEYS:
        digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
        return {f"{key}_sha256": digest}
    if isinstance(value, Mapping):
        return {
            str(item_key): _safe_metadata_value(item, key=str(item_key))
            for item_key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_safe_metadata_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _archive_key(repo_id: str, model_format: str, device: str) -> str:
    return f"{repo_id}|{model_format}|{device}"


def _empty_archive(result: DeviceBenchResult) -> dict[str, Any]:
    return {
        "archive_key": f"{result.format}|{result.device}|{result.canonical_tier}",
        "canonical_tier": result.canonical_tier,
        "device": result.device,
        "format": result.format,
        "key": {
            "device": result.device,
            "format": result.format,
            "tier": result.canonical_tier,
        },
        "results": {},
        "schema_version": DEVICE_BENCH_SCHEMA_VERSION,
        "tier": result.canonical_tier,
    }


def _read_archive(path: Path, result: DeviceBenchResult) -> dict[str, Any]:
    payload = load_device_benchmark_archive(path)
    expected = _empty_archive(result)
    for field_name in ("format", "device", "canonical_tier"):
        if payload.get(field_name) != expected[field_name]:
            raise ValueError(
                f"device benchmark archive identity mismatch for {path}: {field_name}"
            )
    return payload


def _path_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return token or "value"


__all__ = [
    "DEFAULT_BATCH_SIZES",
    "DEFAULT_DEVICE_ARCHIVE_DIR",
    "DEFAULT_DEVICE_CORPUS_PATH",
    "DEFAULT_SEQUENCE_LENGTHS",
    "DEVICE_BENCH_SCHEMA_VERSION",
    "DeviceBenchMeasurement",
    "DeviceBenchResult",
    "DeviceBenchRunner",
    "SUPPORTED_DEVICES",
    "SYNTHETIC_PERF_MODEL_NAME",
    "archive_path_for",
    "default_device_bench_runner",
    "load_device_benchmark_archive",
    "load_device_corpus",
    "run_device_benchmark",
    "synthetic_device_bench_runner",
    "write_device_benchmark_archive",
]
