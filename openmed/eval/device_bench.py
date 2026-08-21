"""Batch-aware device benchmark runner and deterministic result archives."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from openmed.eval import harness
from openmed.eval.metrics import LatencyMetrics, ResourceMetrics
from openmed.eval.perf import (
    DEFAULT_PERF_WORKLOAD_PATH,
    SYNTHETIC_PERF_MODEL_NAME,
    PerfDocument,
    default_perf_runner,
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
_MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
_MAX_ARCHIVE_RESULTS = 10_000
_MAX_BATCH_SIZE = 4_096
_MAX_BATCH_SIZES = 32
_MAX_CORPUS_BYTES = 32 * 1024 * 1024
_MAX_DOCUMENT_CHARS = 1_000_000
_MAX_DOCUMENT_ID_CHARS = 512
_MAX_DOCUMENTS = 4_096
_MAX_FORMAT_CHARS = 64
_MAX_GENERATED_AT_CHARS = 128
_MAX_MATRIX_MEASUREMENTS = 100_000
_MAX_MATRIX_VALUE_CHARS = 4_096
_MAX_METADATA_BYTES = 128 * 1024
_MAX_METADATA_DEPTH = 8
_MAX_METADATA_ITEMS = 1_024
_MAX_METADATA_KEY_CHARS = 64
_MAX_METADATA_STRING_BYTES = 64 * 1024
_MAX_PREPARED_TOKENS = 2_000_000
_MAX_REPEATS = 100
_MAX_REPO_ID_CHARS = 512
_MAX_RESULT_BYTES = 16 * 1024 * 1024
_MAX_SEQUENCE_LENGTH = 32_768
_MAX_SEQUENCE_LENGTHS = 32
_MAX_TIER_CHARS = 64
_METADATA_KEY_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*\Z")
_REPO_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/@+-]*\Z")
_ARCHIVE_FIELDS = frozenset(
    {
        "archive_key",
        "canonical_tier",
        "device",
        "format",
        "key",
        "results",
        "schema_version",
        "tier",
    }
)
_RESULT_FIELDS = frozenset(
    {
        "archive_key",
        "batch_count",
        "batch_sizes",
        "canonical_tier",
        "device",
        "docs_per_second",
        "document_count",
        "format",
        "generated_at",
        "key",
        "latency",
        "measurements",
        "metadata",
        "metrics",
        "p50_ms",
        "p95_ms",
        "peak_rss_mib",
        "repo_id",
        "resources",
        "schema_version",
        "sequence_lengths",
        "tier",
        "tier_budget",
        "total_seconds",
    }
)
_RAW_METADATA_KEYS = frozenset(
    {
        "address",
        "deidentified_text",
        "email",
        "input",
        "mention_text",
        "name",
        "note",
        "output",
        "patient",
        "phone",
        "prompt",
        "raw_text",
        "response",
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
            "metadata": _safe_metadata(self.metadata),
            "metrics": metrics,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "peak_rss_mib": self.peak_rss_mib,
            "repo_id": self.repo_id,
            "resources": self.resources.to_dict(),
            "schema_version": DEVICE_BENCH_SCHEMA_VERSION,
            "sequence_lengths": list(self.sequence_lengths),
            "tier": self.tier,
            "tier_budget": _plain_json(self.tier_budget),
            "total_seconds": self.total_seconds,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the result with stable key ordering."""
        normalized_indent = _normalize_indent(indent)
        payload = json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            indent=normalized_indent,
            sort_keys=True,
        )
        if len(payload.encode("utf-8")) > _MAX_RESULT_BYTES:
            raise ValueError("device benchmark result exceeds the archive size limit")
        return payload

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write this result as a standalone JSON report."""
        output_path = Path(path)
        _atomic_write_text(output_path, self.to_json(indent=indent) + "\n")
        return output_path


def load_device_corpus(
    path: str | Path = DEFAULT_DEVICE_CORPUS_PATH,
) -> list[PerfDocument]:
    """Load the fixed synthetic corpus used by the device benchmark."""
    corpus_path = Path(path)
    encoded = _read_bounded_bytes(corpus_path, _MAX_CORPUS_BYTES, "device corpus")
    try:
        text = encoded.decode("utf-8")
        if corpus_path.suffix.lower() == ".jsonl":
            rows: Any = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            payload = json.loads(text)
            rows = payload.get("documents") if isinstance(payload, dict) else payload
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("device corpus must be valid UTF-8 JSON or JSONL") from exc
    if not isinstance(rows, list):
        raise ValueError("device corpus must contain a list of documents")
    if len(rows) > _MAX_DOCUMENTS:
        raise ValueError("device corpus contains too many documents")
    documents = [_document_from_mapping(row, index) for index, row in enumerate(rows)]
    _validate_documents(documents)
    return documents


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
    offline. Corpus size, matrix dimensions, metadata, and serialized results
    are bounded before materialization; archived metadata never retains raw
    note, prompt, response, or patient fields.
    """
    if not isinstance(model_format, str):
        raise TypeError("model_format must be a string")
    if format is not None:
        if not isinstance(format, str):
            raise TypeError("format must be a string")
        if model_format != "INT8":
            raise ValueError("use either model_format or format, not both")
        model_format = format
    normalized_device = _normalize_device(device)
    normalized_format = _normalize_format(model_format)
    normalized_tier = _bounded_text(tier, "tier", _MAX_TIER_CHARS)
    if not normalized_tier:
        raise ValueError("tier must be non-empty")
    budget = lookup_tier_budget(normalized_tier)
    normalized_lengths = _normalize_positive_values(
        sequence_lengths,
        name="sequence_lengths",
        default=DEFAULT_SEQUENCE_LENGTHS,
        maximum_count=_MAX_SEQUENCE_LENGTHS,
        maximum_value=_MAX_SEQUENCE_LENGTH,
    )
    normalized_batches = _normalize_positive_values(
        batch_sizes,
        name="batch_sizes",
        default=DEFAULT_BATCH_SIZES,
        maximum_count=_MAX_BATCH_SIZES,
        maximum_value=_MAX_BATCH_SIZE,
    )
    if (
        not isinstance(repeats, int)
        or isinstance(repeats, bool)
        or not 1 <= repeats <= _MAX_REPEATS
    ):
        raise ValueError(f"repeats must be between 1 and {_MAX_REPEATS}")
    if corpus is not None and docs is not None:
        raise ValueError("use either corpus or docs, not both")
    documents = _resolve_documents(corpus if corpus is not None else docs)
    _validate_documents(documents)
    _validate_matrix_size(
        document_count=len(documents),
        sequence_lengths=normalized_lengths,
        batch_sizes=normalized_batches,
        repeats=repeats,
    )

    run_batch = runner or default_device_bench_runner
    now = clock or time.perf_counter
    sample_rss = rss_sampler or _peak_rss_bytes
    if not callable(run_batch):
        raise TypeError("runner must be callable")
    if not callable(now):
        raise TypeError("clock must be callable")
    if not callable(sample_rss):
        raise TypeError("rss_sampler must be callable")
    rss_values: list[int] = []
    initial_rss = _sample_rss(sample_rss)
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
                    started = _sample_clock(now)
                    run_batch(model, batch, normalized_device, sequence_length)
                    elapsed_seconds = max(_sample_clock(now) - started, 0.0)
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
                    current_rss = _sample_rss(sample_rss)
                    if current_rss is not None:
                        rss_values.append(current_rss)

    peak_rss_bytes = max(rss_values) if rss_values else None
    latency = harness.compute_latency_summary(latencies_ms)
    resources = harness.compute_resource_metrics(peak_rss_bytes=peak_rss_bytes)
    report_metadata = _safe_metadata(metadata)
    report_metadata.update(
        {
            "corpus_document_count": len(documents),
            "repeats": repeats,
        }
    )
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
        tier_budget=_freeze_json(budget.to_dict()),
        measurements=tuple(measurements),
        generated_at=_normalize_generated_at(generated_at),
        metadata=_freeze_json(report_metadata),
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
    if not isinstance(result, DeviceBenchResult):
        raise TypeError("result must be a DeviceBenchResult")
    _validate_result_identity(result)
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
    if not isinstance(result, DeviceBenchResult):
        raise TypeError("result must be a DeviceBenchResult")
    _validate_result_identity(result)
    output_path = (
        Path(archive_path)
        if archive_path is not None
        else archive_path_for(result, archive_dir)
    )
    payload = _empty_archive(result)
    if output_path.exists():
        payload = _read_archive(output_path, result)
    entries = payload["results"]
    if result.repo_id not in entries and len(entries) >= _MAX_ARCHIVE_RESULTS:
        raise ValueError("device benchmark archive contains too many results")
    entries[result.repo_id] = result.to_dict()
    payload["results"] = {key: entries[key] for key in sorted(entries)}
    payload = _validate_archive_payload(payload)
    serialized = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if len(serialized.encode("utf-8")) > _MAX_ARCHIVE_BYTES:
        raise ValueError("device benchmark archive exceeds the size limit")
    _atomic_write_text(output_path, serialized)
    return output_path


def load_device_benchmark_archive(
    path: str | Path,
) -> dict[str, Any]:
    """Load an archive without exposing any corpus text."""
    encoded = _read_bounded_bytes(Path(path), _MAX_ARCHIVE_BYTES, "device archive")
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("device benchmark archive must be valid UTF-8 JSON") from exc
    return _validate_archive_payload(payload)


def _resolve_documents(
    corpus: str | Path | Sequence[str | Mapping[str, Any] | PerfDocument] | None,
) -> list[PerfDocument]:
    if corpus is None:
        return load_device_corpus()
    if isinstance(corpus, (str, Path)):
        return load_device_corpus(corpus)
    if not isinstance(corpus, Sequence):
        raise TypeError("corpus must be a path or a finite sequence")
    if len(corpus) > _MAX_DOCUMENTS:
        raise ValueError("device corpus contains too many documents")
    documents: list[PerfDocument] = []
    for index, item in enumerate(corpus):
        if isinstance(item, PerfDocument):
            documents.append(item)
        elif isinstance(item, str):
            documents.append(
                PerfDocument(document_id=f"document-{index + 1:03d}", text=item)
            )
        elif isinstance(item, Mapping):
            documents.append(_document_from_mapping(item, index))
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
    if len(documents) > _MAX_DOCUMENTS:
        raise ValueError("device corpus contains too many documents")
    seen: set[str] = set()
    total_bytes = 0
    for index, document in enumerate(documents):
        if not isinstance(document, PerfDocument):
            raise TypeError("device corpus must contain PerfDocument values")
        if not isinstance(document.text, str):
            raise TypeError("benchmark document text must be a string")
        if not document.text.strip():
            raise ValueError(f"benchmark document at position {index} is empty")
        if len(document.text) > _MAX_DOCUMENT_CHARS:
            raise ValueError("benchmark document exceeds the text size limit")
        total_bytes += len(document.text.encode("utf-8"))
        if total_bytes > _MAX_CORPUS_BYTES:
            raise ValueError("device corpus exceeds the in-memory size limit")
        if not isinstance(document.document_id, str) or not document.document_id:
            raise ValueError("benchmark document id must be a non-empty string")
        if len(document.document_id) > _MAX_DOCUMENT_ID_CHARS:
            raise ValueError("benchmark document id exceeds the size limit")
        if not isinstance(document.language, str) or not document.language:
            raise ValueError("benchmark document language must be a non-empty string")
        if document.document_id in seen:
            raise ValueError("device corpus contains a duplicate document id")
        seen.add(document.document_id)


def _normalize_positive_values(
    values: Sequence[int | str] | None,
    *,
    name: str,
    default: Sequence[int],
    maximum_count: int,
    maximum_value: int,
) -> tuple[int, ...]:
    raw_values: Sequence[int | str] = default if values is None else values
    if isinstance(raw_values, (str, bytes)):
        raw_values = [raw_values]  # type: ignore[list-item]
    elif not isinstance(raw_values, Sequence):
        raise TypeError(f"{name} must be a finite sequence")
    if len(raw_values) > maximum_count:
        raise ValueError(f"{name} contains too many values")
    parsed: set[int] = set()
    for raw_value in raw_values:
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, str)):
            raise ValueError(f"{name} must contain positive integers")
        if isinstance(raw_value, int):
            raw_text = str(raw_value) if raw_value.bit_length() <= 63 else ""
        else:
            raw_text = raw_value
        if (
            not raw_text
            or len(raw_text) > _MAX_MATRIX_VALUE_CHARS
            or raw_text.count(",") >= maximum_count
        ):
            raise ValueError(f"{name} must contain positive integers")
        for item in raw_text.split(","):
            value = item.strip()
            if not value:
                continue
            try:
                parsed_value = int(value)
            except ValueError as exc:
                raise ValueError(f"{name} must contain positive integers") from exc
            if not 1 <= parsed_value <= maximum_value:
                raise ValueError(f"{name} must contain positive integers")
            parsed.add(parsed_value)
            if len(parsed) > maximum_count:
                raise ValueError(f"{name} contains too many values")
    if not parsed:
        raise ValueError(f"{name} must contain at least one positive integer")
    return tuple(sorted(parsed))


def _normalize_device(device: str) -> str:
    normalized = _bounded_text(device, "device", 16).lower()
    if normalized not in SUPPORTED_DEVICES:
        allowed = ", ".join(SUPPORTED_DEVICES)
        raise ValueError(f"unsupported device; expected one of: {allowed}")
    return normalized


def _normalize_format(model_format: str) -> str:
    normalized = _bounded_text(
        model_format,
        "format",
        _MAX_FORMAT_CHARS,
    ).upper()
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9_.+-]*", normalized):
        raise ValueError("format must be a non-empty filename-safe value")
    return normalized


def _repo_id(model: Any, repo_id: str | None) -> str:
    if repo_id is not None:
        return _normalize_repo_id(repo_id)
    if isinstance(model, Path):
        return _private_model_identifier(str(model))
    if isinstance(model, str):
        value = _bounded_text(model, "model", 4_096)
        if (
            Path(value).is_absolute()
            or value.startswith(("./", "../"))
            or "\\" in value
            or _REPO_ID_RE.fullmatch(value) is None
        ):
            return _private_model_identifier(value)
        return _normalize_repo_id(value)
    else:
        try:
            candidate = getattr(model, "__name__", None)
        except Exception:
            candidate = None
        value = (
            candidate
            if isinstance(candidate, str) and candidate.strip()
            else model.__class__.__name__
        )
        return _normalize_repo_id(value)


def _normalize_repo_id(value: Any) -> str:
    normalized = _bounded_text(value, "repo_id", _MAX_REPO_ID_CHARS)
    if _REPO_ID_RE.fullmatch(normalized) is None or "//" in normalized:
        raise ValueError("repo_id must be a filename-safe model identifier")
    return normalized


def _private_model_identifier(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"local-sha256-{digest}"


def _peak_rss_bytes() -> int | None:
    """Reuse the eval harness RSS primitive for platform normalization."""
    return harness._peak_rss_bytes()


def _document_from_mapping(data: Any, index: int) -> PerfDocument:
    if not isinstance(data, Mapping):
        raise TypeError("device corpus rows must be JSON objects")
    try:
        text = data.get("text")
        document_id = data.get("id")
        if document_id is None:
            document_id = data.get("document_id")
        language = data.get("language")
        if language is None:
            language = data.get("lang")
        metadata = data.get("metadata", {})
    except Exception as exc:
        raise TypeError(
            "device corpus rows must expose ordinary mapping values"
        ) from exc
    if not isinstance(text, str):
        raise TypeError("benchmark document text must be a string")
    if document_id is None:
        document_id = f"document-{index + 1:03d}"
    if not isinstance(document_id, str):
        raise TypeError("benchmark document id must be a string")
    if language is None:
        language = "en"
    if not isinstance(language, str):
        raise TypeError("benchmark document language must be a string")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise TypeError("benchmark document metadata must be an object")
    if len(metadata) > _MAX_METADATA_ITEMS:
        raise ValueError("benchmark document metadata contains too many fields")
    return PerfDocument(
        document_id=document_id,
        text=text,
        language=language,
        metadata=dict(metadata),
    )


def _validate_matrix_size(
    *,
    document_count: int,
    sequence_lengths: Sequence[int],
    batch_sizes: Sequence[int],
    repeats: int,
) -> None:
    if document_count * max(sequence_lengths) > _MAX_PREPARED_TOKENS:
        raise ValueError("device benchmark prepared corpus exceeds the token limit")
    batch_count = sum(
        (document_count + batch_size - 1) // batch_size for batch_size in batch_sizes
    )
    measurement_count = batch_count * len(sequence_lengths) * repeats
    if measurement_count > _MAX_MATRIX_MEASUREMENTS:
        raise ValueError("device benchmark matrix contains too many measurements")


def _sample_clock(clock: Clock) -> float:
    value = clock()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("clock must return a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("clock must return a finite number")
    return numeric


def _sample_rss(sampler: RssSampler) -> int | None:
    value = sampler()
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= (2**63 - 1)
    ):
        raise ValueError("rss_sampler must return a non-negative integer or None")
    return value


def _bounded_text(value: Any, name: str, maximum_chars: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    if len(normalized) > maximum_chars:
        raise ValueError(f"{name} exceeds the size limit")
    if any(ord(character) < 32 or ord(character) == 127 for character in normalized):
        raise ValueError(f"{name} must not contain control characters")
    return normalized


def _normalize_generated_at(value: Any) -> str | None:
    if value is None:
        return None
    normalized = _bounded_text(value, "generated_at", _MAX_GENERATED_AT_CHARS)
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("generated_at must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("generated_at must include a timezone")
    return normalized


def _safe_metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("metadata must be an object")
    state = [0, 0]
    copied = _copy_json_value(
        value,
        depth=0,
        seen=set(),
        state=state,
        redact_sensitive=True,
        maximum_items=_MAX_METADATA_ITEMS,
        maximum_bytes=_MAX_METADATA_BYTES,
        maximum_string_bytes=_MAX_METADATA_STRING_BYTES,
    )
    if not isinstance(copied, dict):
        raise TypeError("metadata must be an object")
    return copied


def _plain_json(value: Any) -> Any:
    return _copy_json_value(
        value,
        depth=0,
        seen=set(),
        state=[0, 0],
        redact_sensitive=False,
        maximum_items=_MAX_METADATA_ITEMS + 32,
        maximum_bytes=_MAX_METADATA_BYTES + 1_024,
        maximum_string_bytes=_MAX_METADATA_STRING_BYTES,
    )


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _copy_json_value(
    value: Any,
    *,
    depth: int,
    seen: set[int],
    state: list[int],
    redact_sensitive: bool,
    maximum_items: int,
    maximum_bytes: int,
    maximum_string_bytes: int,
) -> Any:
    if depth > _MAX_METADATA_DEPTH:
        raise ValueError("metadata exceeds the nesting limit")
    state[0] += 1
    if state[0] > maximum_items:
        raise ValueError("metadata contains too many values")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value.bit_length() > 63:
            raise ValueError("metadata integer exceeds the supported range")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata numbers must be finite")
        return value
    if isinstance(value, str):
        _consume_json_bytes(
            value,
            state,
            maximum_bytes=maximum_bytes,
            maximum_string_bytes=maximum_string_bytes,
        )
        return value
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in seen:
            raise ValueError("metadata must not contain cycles")
        if len(value) > maximum_items:
            raise ValueError("metadata contains too many fields")
        seen.add(identity)
        try:
            items = list(value.items())
            normalized: list[tuple[str, Any]] = []
            for raw_key, item in items:
                key = _metadata_key(raw_key)
                _consume_json_bytes(
                    key,
                    state,
                    maximum_bytes=maximum_bytes,
                    maximum_string_bytes=maximum_string_bytes,
                )
                normalized.append((key, item))
            result: dict[str, Any] = {}
            for key, item in sorted(normalized, key=lambda pair: pair[0]):
                if (
                    redact_sensitive
                    and _is_sensitive_metadata_key(key)
                    and not _is_sha256_field(key, item)
                ):
                    result[f"{key}_sha256"] = _hash_metadata_value(
                        item,
                        depth=depth + 1,
                        seen=seen,
                        state=state,
                        maximum_items=maximum_items,
                        maximum_bytes=maximum_bytes,
                        maximum_string_bytes=maximum_string_bytes,
                    )
                else:
                    result[key] = _copy_json_value(
                        item,
                        depth=depth + 1,
                        seen=seen,
                        state=state,
                        redact_sensitive=redact_sensitive,
                        maximum_items=maximum_items,
                        maximum_bytes=maximum_bytes,
                        maximum_string_bytes=maximum_string_bytes,
                    )
            return result
        finally:
            seen.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in seen:
            raise ValueError("metadata must not contain cycles")
        if len(value) > maximum_items:
            raise ValueError("metadata contains too many values")
        seen.add(identity)
        try:
            return [
                _copy_json_value(
                    item,
                    depth=depth + 1,
                    seen=seen,
                    state=state,
                    redact_sensitive=redact_sensitive,
                    maximum_items=maximum_items,
                    maximum_bytes=maximum_bytes,
                    maximum_string_bytes=maximum_string_bytes,
                )
                for item in value
            ]
        finally:
            seen.remove(identity)
    raise TypeError("metadata values must be JSON-compatible")


def _metadata_key(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) > _MAX_METADATA_KEY_CHARS
        or _METADATA_KEY_RE.fullmatch(value) is None
    ):
        raise ValueError("metadata keys must be short ASCII identifiers")
    return value


def _consume_json_bytes(
    value: str,
    state: list[int],
    *,
    maximum_bytes: int,
    maximum_string_bytes: int,
) -> None:
    size = len(value.encode("utf-8"))
    if size > maximum_string_bytes:
        raise ValueError("metadata string exceeds the size limit")
    state[1] += size
    if state[1] > maximum_bytes:
        raise ValueError("metadata exceeds the total size limit")


def _is_sensitive_metadata_key(key: str) -> bool:
    tokens = {token for token in re.split(r"[^a-z0-9]+", key.lower()) if token}
    return bool(tokens & _RAW_METADATA_KEYS)


def _is_sha256_field(key: str, value: Any) -> bool:
    return (
        key.lower().endswith("_sha256")
        and isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    )


def _hash_metadata_value(
    value: Any,
    *,
    depth: int,
    seen: set[int],
    state: list[int],
    maximum_items: int,
    maximum_bytes: int,
    maximum_string_bytes: int,
) -> str:
    copied = _copy_json_value(
        value,
        depth=depth,
        seen=seen,
        state=state,
        redact_sensitive=False,
        maximum_items=maximum_items,
        maximum_bytes=maximum_bytes,
        maximum_string_bytes=maximum_string_bytes,
    )
    encoded = json.dumps(
        copied,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _plain_archive_json(value: Any) -> Any:
    return _copy_json_value(
        value,
        depth=0,
        seen=set(),
        state=[0, 0],
        redact_sensitive=False,
        maximum_items=(_MAX_MATRIX_MEASUREMENTS * 8) + 1_024,
        maximum_bytes=_MAX_RESULT_BYTES,
        maximum_string_bytes=_MAX_METADATA_STRING_BYTES,
    )


def _normalize_indent(indent: int) -> int:
    if isinstance(indent, bool) or not isinstance(indent, int) or not 0 <= indent <= 8:
        raise ValueError("indent must be an integer between 0 and 8")
    return indent


def _read_bounded_bytes(path: Path, maximum_bytes: int, label: str) -> bytes:
    with path.open("rb") as handle:
        encoded = handle.read(maximum_bytes + 1)
    if len(encoded) > maximum_bytes:
        raise ValueError(f"{label} exceeds the size limit")
    return encoded


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    descriptor_open = True
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor_open = False
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if descriptor_open:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)


def _validate_result_identity(result: DeviceBenchResult) -> None:
    if _normalize_repo_id(result.repo_id) != result.repo_id:
        raise ValueError("result repo_id is not canonical")
    if _normalize_format(result.format) != result.format:
        raise ValueError("result format is not canonical")
    if _normalize_device(result.device) != result.device:
        raise ValueError("result device is not canonical")
    if _bounded_text(result.canonical_tier, "canonical_tier", _MAX_TIER_CHARS) != (
        result.canonical_tier
    ):
        raise ValueError("result canonical_tier is not canonical")


def _validate_archive_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _ARCHIVE_FIELDS:
        raise ValueError("device benchmark archive has an invalid top-level schema")
    if value.get("schema_version") != DEVICE_BENCH_SCHEMA_VERSION:
        raise ValueError("unsupported device benchmark archive schema")
    raw_format = value.get("format")
    raw_device = value.get("device")
    if not isinstance(raw_format, str) or not isinstance(raw_device, str):
        raise ValueError("device benchmark archive identity must use strings")
    archive_format = _normalize_format(raw_format)
    archive_device = _normalize_device(raw_device)
    canonical_tier = _bounded_text(
        value.get("canonical_tier"),
        "canonical_tier",
        _MAX_TIER_CHARS,
    )
    if value.get("tier") != canonical_tier:
        raise ValueError("device benchmark archive tier identity mismatch")
    expected_archive_key = f"{archive_format}|{archive_device}|{canonical_tier}"
    if value.get("archive_key") != expected_archive_key:
        raise ValueError("device benchmark archive key mismatch")
    expected_key = {
        "device": archive_device,
        "format": archive_format,
        "tier": canonical_tier,
    }
    if value.get("key") != expected_key:
        raise ValueError("device benchmark archive identity object mismatch")
    results = value.get("results")
    if not isinstance(results, dict):
        raise ValueError("device benchmark archive must contain a results object")
    if len(results) > _MAX_ARCHIVE_RESULTS:
        raise ValueError("device benchmark archive contains too many results")

    safe_results: dict[str, Any] = {}
    for raw_repo_id, raw_result in results.items():
        repo_id = _normalize_repo_id(raw_repo_id)
        if repo_id != raw_repo_id:
            raise ValueError("device benchmark archive repo_id is not canonical")
        if not isinstance(raw_result, dict) or set(raw_result) != _RESULT_FIELDS:
            raise ValueError("device benchmark archive contains an invalid result")
        if raw_result.get("schema_version") != DEVICE_BENCH_SCHEMA_VERSION:
            raise ValueError("device benchmark result schema mismatch")
        if raw_result.get("repo_id") != repo_id:
            raise ValueError("device benchmark result repo_id mismatch")
        if raw_result.get("format") != archive_format:
            raise ValueError("device benchmark result format mismatch")
        if raw_result.get("device") != archive_device:
            raise ValueError("device benchmark result device mismatch")
        if raw_result.get("canonical_tier") != canonical_tier:
            raise ValueError("device benchmark result tier mismatch")
        if raw_result.get("archive_key") != _archive_key(
            repo_id,
            archive_format,
            archive_device,
        ):
            raise ValueError("device benchmark result archive key mismatch")
        result_key = raw_result.get("key")
        if result_key != {
            "device": archive_device,
            "format": archive_format,
            "repo_id": repo_id,
            "tier": canonical_tier,
        }:
            raise ValueError("device benchmark result identity object mismatch")
        _validate_loaded_result_payload(raw_result, canonical_tier)
        measurements = raw_result.get("measurements")
        if not isinstance(measurements, list):
            raise ValueError("device benchmark measurements must be a list")
        if len(measurements) > _MAX_MATRIX_MEASUREMENTS:
            raise ValueError("device benchmark result has too many measurements")
        batch_count = raw_result.get("batch_count")
        if (
            isinstance(batch_count, bool)
            or not isinstance(batch_count, int)
            or batch_count != len(measurements)
        ):
            raise ValueError("device benchmark result batch count mismatch")
        safe_result = _plain_archive_json(raw_result)
        safe_result["metadata"] = _safe_metadata(raw_result.get("metadata"))
        safe_results[repo_id] = safe_result

    return {
        "archive_key": expected_archive_key,
        "canonical_tier": canonical_tier,
        "device": archive_device,
        "format": archive_format,
        "key": expected_key,
        "results": {key: safe_results[key] for key in sorted(safe_results)},
        "schema_version": DEVICE_BENCH_SCHEMA_VERSION,
        "tier": canonical_tier,
    }


def _validate_loaded_result_payload(
    result: Mapping[str, Any],
    canonical_tier: str,
) -> None:
    raw_tier = result.get("tier")
    if not isinstance(raw_tier, str):
        raise ValueError("device benchmark result tier must be a string")
    normalized_tier = _bounded_text(raw_tier, "tier", _MAX_TIER_CHARS)
    budget = lookup_tier_budget(normalized_tier)
    if budget.canonical_tier != canonical_tier:
        raise ValueError("device benchmark result tier budget mismatch")
    if result.get("tier_budget") != budget.to_dict():
        raise ValueError("device benchmark result tier budget was modified")
    if _normalize_generated_at(result.get("generated_at")) != result.get(
        "generated_at"
    ):
        raise ValueError("device benchmark generated_at is not canonical")

    sequence_lengths = result.get("sequence_lengths")
    batch_sizes = result.get("batch_sizes")
    if not isinstance(sequence_lengths, list) or not isinstance(batch_sizes, list):
        raise ValueError("device benchmark matrix axes must be lists")
    if (
        list(
            _normalize_positive_values(
                sequence_lengths,
                name="sequence_lengths",
                default=DEFAULT_SEQUENCE_LENGTHS,
                maximum_count=_MAX_SEQUENCE_LENGTHS,
                maximum_value=_MAX_SEQUENCE_LENGTH,
            )
        )
        != sequence_lengths
    ):
        raise ValueError("device benchmark sequence lengths are not canonical")
    if (
        list(
            _normalize_positive_values(
                batch_sizes,
                name="batch_sizes",
                default=DEFAULT_BATCH_SIZES,
                maximum_count=_MAX_BATCH_SIZES,
                maximum_value=_MAX_BATCH_SIZE,
            )
        )
        != batch_sizes
    ):
        raise ValueError("device benchmark batch sizes are not canonical")

    measurements = result.get("measurements")
    if not isinstance(measurements, list):
        raise ValueError("device benchmark measurements must be a list")
    if len(measurements) > _MAX_MATRIX_MEASUREMENTS:
        raise ValueError("device benchmark result has too many measurements")
    measurement_fields = {
        "batch_size",
        "docs_per_second",
        "document_count",
        "latency_ms",
        "repeat",
        "sequence_length",
    }
    measured_documents = 0
    for measurement in measurements:
        if not isinstance(measurement, dict) or set(measurement) != measurement_fields:
            raise ValueError("device benchmark measurement has an invalid schema")
        batch_size = _bounded_integer(
            measurement.get("batch_size"),
            minimum=1,
            maximum=_MAX_BATCH_SIZE,
            name="measurement batch_size",
        )
        document_count = _bounded_integer(
            measurement.get("document_count"),
            minimum=1,
            maximum=batch_size,
            name="measurement document_count",
        )
        _bounded_integer(
            measurement.get("repeat"),
            minimum=0,
            maximum=_MAX_REPEATS - 1,
            name="measurement repeat",
        )
        _bounded_integer(
            measurement.get("sequence_length"),
            minimum=1,
            maximum=_MAX_SEQUENCE_LENGTH,
            name="measurement sequence_length",
        )
        _non_negative_number(
            measurement.get("latency_ms"),
            "measurement latency_ms",
        )
        _non_negative_number(
            measurement.get("docs_per_second"),
            "measurement docs_per_second",
        )
        measured_documents += document_count

    batch_count = _bounded_integer(
        result.get("batch_count"),
        minimum=0,
        maximum=_MAX_MATRIX_MEASUREMENTS,
        name="result batch_count",
    )
    if batch_count != len(measurements):
        raise ValueError("device benchmark result batch count mismatch")
    document_count = _bounded_integer(
        result.get("document_count"),
        minimum=0,
        maximum=_MAX_MATRIX_MEASUREMENTS * _MAX_BATCH_SIZE,
        name="result document_count",
    )
    if document_count != measured_documents:
        raise ValueError("device benchmark result document count mismatch")

    latency = result.get("latency")
    if not isinstance(latency, dict) or set(latency) != {
        "count",
        "p50_ms",
        "p95_ms",
        "p99_ms",
    }:
        raise ValueError("device benchmark latency has an invalid schema")
    if _bounded_integer(
        latency.get("count"),
        minimum=0,
        maximum=_MAX_MATRIX_MEASUREMENTS,
        name="latency count",
    ) != len(measurements):
        raise ValueError("device benchmark latency count mismatch")
    for key in ("p50_ms", "p95_ms", "p99_ms"):
        _non_negative_number(latency.get(key), f"latency {key}")

    resources = result.get("resources")
    if not isinstance(resources, dict) or set(resources) != {
        "model_size_bytes",
        "model_size_mib",
        "peak_rss_bytes",
        "peak_rss_mib",
    }:
        raise ValueError("device benchmark resources have an invalid schema")
    _validate_resource_pair(resources, "peak_rss")
    _validate_resource_pair(resources, "model_size")

    metrics = result.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != {
        "docs_per_second",
        "p50_ms",
        "p95_ms",
        "peak_rss_mib",
    }:
        raise ValueError("device benchmark metrics have an invalid schema")
    docs_per_second = _non_negative_number(
        result.get("docs_per_second"),
        "result docs_per_second",
    )
    _non_negative_number(result.get("total_seconds"), "result total_seconds")
    if metrics.get("docs_per_second") != docs_per_second:
        raise ValueError("device benchmark throughput metric mismatch")
    if (
        result.get("p50_ms") != latency["p50_ms"]
        or metrics.get("p50_ms") != (latency["p50_ms"])
    ):
        raise ValueError("device benchmark p50 metric mismatch")
    if (
        result.get("p95_ms") != latency["p95_ms"]
        or metrics.get("p95_ms") != (latency["p95_ms"])
    ):
        raise ValueError("device benchmark p95 metric mismatch")
    if (
        result.get("peak_rss_mib") != resources["peak_rss_mib"]
        or metrics.get("peak_rss_mib") != resources["peak_rss_mib"]
    ):
        raise ValueError("device benchmark RSS metric mismatch")


def _bounded_integer(
    value: Any,
    *,
    minimum: int,
    maximum: int,
    name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{name} is outside the supported range")
    return value


def _non_negative_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return numeric


def _validate_resource_pair(resources: Mapping[str, Any], prefix: str) -> None:
    bytes_value = resources.get(f"{prefix}_bytes")
    mib_value = resources.get(f"{prefix}_mib")
    if bytes_value is None:
        if mib_value is not None:
            raise ValueError("device benchmark resource units mismatch")
        return
    byte_count = _bounded_integer(
        bytes_value,
        minimum=0,
        maximum=2**63 - 1,
        name=f"{prefix}_bytes",
    )
    mib = _non_negative_number(mib_value, f"{prefix}_mib")
    if mib != byte_count / (1024 * 1024):
        raise ValueError("device benchmark resource units mismatch")


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
