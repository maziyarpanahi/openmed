"""Batch processing utilities for OpenMed.

This module provides efficient batch processing capabilities for analyzing
multiple texts or files with progress reporting and result aggregation.

Dataset redaction supports local ``.csv``, ``.jsonl``/``.ndjson``, and
``.parquet`` files. Callers choose one or more free-text columns explicitly;
all other columns are copied through unchanged.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Union,
)

logger = logging.getLogger(__name__)

BatchOperation = Literal["analyze_text", "extract_pii", "deidentify"]
DatasetFormat = Literal["csv", "jsonl", "parquet"]
_VALID_OPERATIONS = {"analyze_text", "extract_pii", "deidentify"}
_DEFAULT_PII_MODEL = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"
_BATCH_CHECKPOINT_SCHEMA_VERSION = 1
_DEFAULT_CHECKPOINT_INTERVAL = 10


@dataclass
class BatchItem:
    """Represents a single item in a batch processing job."""

    id: str
    text: str
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BatchItemResult:
    """Result for a single batch item."""

    id: str
    result: Optional[Any] = None  # PredictionResult/DeidentificationResult
    error: Optional[str] = None
    processing_time: float = 0.0
    source: Optional[str] = None

    @property
    def success(self) -> bool:
        """Check if the item was processed successfully."""
        return self.error is None and self.result is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        if self.result is None:
            serialized_result = None
        elif hasattr(self.result, "to_dict"):
            serialized_result = self.result.to_dict()
        else:
            serialized_result = self.result
        return {
            "id": self.id,
            "success": self.success,
            "result": serialized_result,
            "error": self.error,
            "processing_time": self.processing_time,
            "source": self.source,
        }


@dataclass(frozen=True)
class BatchProgress:
    """PHI-safe progress snapshot for a batch processing job."""

    completed: int
    total: int
    current_index: int
    elapsed: float


@dataclass
class BatchResult:
    """Aggregate result for a batch processing job."""

    items: List[BatchItemResult] = field(default_factory=list)
    total_processing_time: float = 0.0
    model_name: str = ""
    started_at: str = ""
    completed_at: str = ""

    @property
    def total_items(self) -> int:
        """Total number of items in the batch."""
        return len(self.items)

    @property
    def successful_items(self) -> int:
        """Number of successfully processed items."""
        return sum(1 for item in self.items if item.success)

    @property
    def failed_items(self) -> int:
        """Number of failed items."""
        return sum(1 for item in self.items if not item.success)

    @property
    def success_rate(self) -> float:
        """Success rate as a percentage."""
        if not self.items:
            return 0.0
        return (self.successful_items / self.total_items) * 100

    @property
    def average_processing_time(self) -> float:
        """Average processing time per item."""
        if not self.items:
            return 0.0
        return sum(item.processing_time for item in self.items) / len(self.items)

    def get_successful_results(self) -> List[BatchItemResult]:
        """Get only successful results."""
        return [item for item in self.items if item.success]

    def get_failed_results(self) -> List[BatchItemResult]:
        """Get only failed results."""
        return [item for item in self.items if not item.success]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_items": self.total_items,
            "successful_items": self.successful_items,
            "failed_items": self.failed_items,
            "success_rate": self.success_rate,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": self.average_processing_time,
            "model_name": self.model_name,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "items": [item.to_dict() for item in self.items],
        }

    def summary(self) -> str:
        """Generate a human-readable summary of the batch results."""
        lines = [
            "Batch Processing Summary",
            "========================",
            f"Model: {self.model_name}",
            f"Total items: {self.total_items}",
            f"Successful: {self.successful_items}",
            f"Failed: {self.failed_items}",
            f"Success rate: {self.success_rate:.1f}%",
            f"Total time: {self.total_processing_time:.2f}s",
            f"Average time per item: {self.average_processing_time:.3f}s",
        ]
        return "\n".join(lines)


@dataclass
class DatasetRedactionSummary:
    """Aggregate audit summary for dataset redaction.

    The summary intentionally contains counts and rates only. It does not
    include raw input values, redacted cell values, or entity surfaces.
    """

    input_format: DatasetFormat
    text_columns: List[str]
    total_rows: int = 0
    processed_cells: int = 0
    redacted_cells: int = 0
    total_spans: int = 0
    per_label_counts: Dict[str, int] = field(default_factory=dict)
    residual_span_count: int = 0

    @property
    def residual_leakage_estimate(self) -> float:
        """Estimated residual span leakage rate after redaction."""
        if self.total_spans == 0:
            return 0.0
        return self.residual_span_count / self.total_spans

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a PHI-free dictionary representation."""
        return {
            "input_format": self.input_format,
            "text_columns": list(self.text_columns),
            "total_rows": self.total_rows,
            "processed_cells": self.processed_cells,
            "redacted_cells": self.redacted_cells,
            "total_spans": self.total_spans,
            "per_label_counts": dict(sorted(self.per_label_counts.items())),
            "residual_span_count": self.residual_span_count,
            "residual_leakage_estimate": self.residual_leakage_estimate,
        }


@dataclass
class DatasetRedactionResult:
    """Result returned by :func:`redact_dataset`."""

    output_path: Path
    summary: DatasetRedactionSummary

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "output_path": str(self.output_path),
            "summary": self.summary.to_dict(),
        }


# Type alias for progress callback
ProgressCallback = Callable[[int, int, Optional[BatchItemResult]], None]
BatchProgressCallback = Callable[[BatchProgress], None]


class BatchCheckpointError(ValueError):
    """Raised when a durable batch checkpoint cannot be resumed safely."""


@dataclass(frozen=True)
class _OutputArtifact:
    size: int
    sha256: str


AtomicWriteHook = Callable[[str, Path], None]
ArtifactWriter = Callable[[int, BatchItem, BatchItemResult], Optional[_OutputArtifact]]
ArtifactPathResolver = Callable[[int, BatchItem], Optional[Path]]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _stable_hash(*values: Any) -> str:
    """Hash potentially sensitive values without persisting their plaintext."""
    digest = hashlib.sha256()
    for value in values:
        if isinstance(value, bytes):
            encoded = value
        elif isinstance(value, str):
            encoded = value.encode("utf-8")
        else:
            encoded = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                default=repr,
            ).encode("utf-8")
        digest.update(str(len(encoded)).encode("ascii"))
        digest.update(b":")
        digest.update(encoded)
        digest.update(b"\0")
    return digest.hexdigest()


def _checkpoint_integer(value: Any, field_name: str, *, minimum: int = 0) -> int:
    """Return a strictly typed checkpoint integer or raise a safe error."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BatchCheckpointError(f"Checkpoint {field_name} is invalid")
    return value


def _fsync_directory(path: Path) -> None:
    """Best-effort fsync of a directory after an atomic rename."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        directory_fd = os.open(path, flags)
    except OSError:  # pragma: no cover - platform/filesystem dependent
        return
    try:
        os.fsync(directory_fd)
    except OSError:  # pragma: no cover - platform/filesystem dependent
        pass
    finally:
        os.close(directory_fd)


def _atomic_write_bytes(
    path: Path,
    payload: bytes,
    *,
    hook: Optional[AtomicWriteHook] = None,
) -> None:
    """Durably replace ``path`` with ``payload`` using a same-directory temp."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            if hook:
                hook("after_write", path)
            os.fsync(handle.fileno())
            if hook:
                hook("after_fsync", path)
        if hook:
            hook("before_replace", path)
        os.replace(temporary_path, path)
        if hook:
            hook("after_replace", path)
        _fsync_directory(path.parent)
        if hook:
            hook("after_directory_fsync", path)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _serialize_result_record(result: BatchItemResult) -> bytes:
    return (
        json.dumps(
            result.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def _restore_result_record(record: Mapping[str, Any]) -> BatchItemResult:
    return BatchItemResult(
        id=str(record.get("id", "")),
        result=record.get("result"),
        error=(str(record["error"]) if record.get("error") is not None else None),
        processing_time=float(record.get("processing_time", 0.0)),
        source=(str(record["source"]) if record.get("source") is not None else None),
    )


class _BatchCheckpointSession:
    """Coordinate a PHI-free checkpoint with a durable result journal."""

    def __init__(
        self,
        *,
        items: Sequence[BatchItem],
        checkpoint_path: Path,
        checkpoint_interval: int,
        model_name: str,
        operation: BatchOperation,
        configuration: Mapping[str, Any],
        final_output_path: Optional[Path],
        output_format: str,
        store_full_results: bool,
        artifact_path_resolver: Optional[ArtifactPathResolver],
        atomic_write_hook: Optional[AtomicWriteHook],
    ) -> None:
        self.items = list(items)
        self.checkpoint_path = checkpoint_path
        self.state_path = Path(f"{checkpoint_path}.part")
        self.checkpoint_interval = checkpoint_interval
        self.model_name = model_name
        self.operation = operation
        self.configuration_hash = _stable_hash(configuration)
        self.final_output_path = final_output_path
        self.store_full_results = store_full_results
        self.output_format = output_format
        self.artifact_path_resolver = artifact_path_resolver
        self.atomic_write_hook = atomic_write_hook
        self.input_hash = _stable_hash(
            [
                {
                    "id": item.id,
                    "source": item.source,
                    "text": item.text,
                    "metadata": item.metadata,
                }
                for item in self.items
            ]
        )
        self.started_at = datetime.now().isoformat()
        self.completed_at = ""
        self.is_complete = False
        self.artifacts: Dict[int, _OutputArtifact] = {}

    def start(self) -> None:
        """Create an empty durable journal and initial checkpoint."""
        _atomic_write_bytes(
            self.state_path,
            b"",
            hook=self.atomic_write_hook,
        )
        self._write_checkpoint([], status="in_progress")

    def resume(self) -> List[BatchItemResult]:
        """Load, validate, and return the checkpointed result prefix."""
        try:
            payload = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise BatchCheckpointError(
                f"Checkpoint not found: {self.checkpoint_path}"
            ) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise BatchCheckpointError(
                f"Checkpoint is not readable: {self.checkpoint_path}"
            ) from exc

        self._validate_checkpoint(payload)
        self.checkpoint_interval = _checkpoint_integer(
            payload.get("checkpoint_interval"),
            "interval",
            minimum=1,
        )
        self.started_at = str(payload.get("started_at") or self.started_at)
        self.completed_at = str(payload.get("completed_at") or "")
        self.is_complete = payload.get("status") == "complete"
        committed_count = _checkpoint_integer(
            payload.get("committed_count"),
            "committed count",
        )
        committed_output = payload["committed_output"]
        committed_offset = _checkpoint_integer(
            committed_output.get("offset"),
            "committed output offset",
        )
        expected_hash = str(committed_output["sha256"])

        try:
            durable_bytes = self.state_path.read_bytes()
        except OSError as exc:
            raise BatchCheckpointError(
                f"Committed batch output is not readable: {self.state_path}"
            ) from exc
        if len(durable_bytes) < committed_offset:
            raise BatchCheckpointError(
                "Committed batch output is shorter than the checkpoint offset"
            )

        committed_bytes = durable_bytes[:committed_offset]
        if _sha256_bytes(committed_bytes) != expected_hash:
            raise BatchCheckpointError(
                "Committed batch output does not match the checkpoint hash"
            )

        if len(durable_bytes) > committed_offset:
            _atomic_write_bytes(
                self.state_path,
                committed_bytes,
                hook=self.atomic_write_hook,
            )

        serialized_records: List[Mapping[str, Any]] = []
        record_offsets: List[int] = []
        record_offset = 0
        try:
            for line in committed_bytes.splitlines(keepends=True):
                parsed = json.loads(line)
                if not isinstance(parsed, Mapping):
                    raise TypeError("result record is not an object")
                serialized_records.append(parsed)
                record_offset += len(line)
                record_offsets.append(record_offset)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise BatchCheckpointError(
                "Committed batch output contains an invalid result record"
            ) from exc

        if len(serialized_records) != committed_count:
            raise BatchCheckpointError(
                "Committed result count does not match the checkpoint"
            )

        checkpointed_items = payload.get("items", [])
        if not isinstance(checkpointed_items, list) or len(checkpointed_items) != len(
            serialized_records
        ):
            raise BatchCheckpointError("Checkpoint item metadata is incomplete")

        records: List[BatchItemResult] = []
        for index, (item, serialized_record, metadata, output_offset) in enumerate(
            zip(self.items, serialized_records, checkpointed_items, record_offsets)
        ):
            if not isinstance(metadata, Mapping):
                raise BatchCheckpointError("Checkpoint item metadata is invalid")
            if _checkpoint_integer(metadata.get("index"), "item index") != index:
                raise BatchCheckpointError("Checkpoint item order does not match input")
            if str(metadata.get("item_id_sha256", "")) != _stable_hash(item.id):
                raise BatchCheckpointError("Checkpoint item id does not match input")
            if (
                _checkpoint_integer(
                    metadata.get("committed_output_offset"),
                    "item output offset",
                )
                != output_offset
            ):
                raise BatchCheckpointError(
                    "Checkpoint item output offset does not match the journal"
                )
            if self.store_full_results:
                try:
                    record = _restore_result_record(serialized_record)
                except (TypeError, ValueError) as exc:
                    raise BatchCheckpointError(
                        "Committed batch output contains an invalid result record"
                    ) from exc
            else:
                if (
                    _checkpoint_integer(
                        serialized_record.get("index"),
                        "journal item index",
                    )
                    != index
                ):
                    raise BatchCheckpointError(
                        "Committed journal item order does not match input"
                    )
                journal_status = serialized_record.get("status")
                if journal_status not in {"success", "failed"}:
                    raise BatchCheckpointError(
                        "Committed journal item status is invalid"
                    )
                record_succeeded = journal_status == "success"
                record = BatchItemResult(
                    id=item.id,
                    result={"checkpointed": True} if record_succeeded else None,
                    error=None if record_succeeded else "Previously failed",
                    source=item.source,
                )
            expected_status = "success" if record.success else "failed"
            if metadata.get("status") != expected_status:
                raise BatchCheckpointError("Checkpoint item status is inconsistent")
            self._restore_and_verify_artifact(index, item, metadata)
            records.append(record)

        if payload.get("status") == "complete" and self.final_output_path is not None:
            self._verify_final_output(payload)

        return records

    def commit(
        self,
        results: Sequence[BatchItemResult],
        *,
        status: str,
        completed_at: str = "",
        final_output: Optional[_OutputArtifact] = None,
    ) -> None:
        """Atomically publish a new result journal, then its checkpoint."""
        journal = b"".join(
            self._serialize_journal_record(index, result)
            for index, result in enumerate(results)
        )
        _atomic_write_bytes(
            self.state_path,
            journal,
            hook=self.atomic_write_hook,
        )
        self._write_checkpoint(
            results,
            status=status,
            completed_at=completed_at,
            final_output=final_output,
        )

    def record_artifact(self, index: int, artifact: _OutputArtifact) -> None:
        self.artifacts[index] = artifact

    def _write_checkpoint(
        self,
        results: Sequence[BatchItemResult],
        *,
        status: str,
        completed_at: str = "",
        final_output: Optional[_OutputArtifact] = None,
    ) -> None:
        serialized_records = [
            self._serialize_journal_record(index, result)
            for index, result in enumerate(results)
        ]
        journal = b"".join(serialized_records)
        offsets: List[int] = []
        offset = 0
        for serialized_record in serialized_records:
            offset += len(serialized_record)
            offsets.append(offset)

        item_metadata = []
        for index, (item, result, output_offset) in enumerate(
            zip(self.items, results, offsets)
        ):
            metadata: Dict[str, Any] = {
                "index": index,
                "item_id_sha256": _stable_hash(item.id),
                "status": "success" if result.success else "failed",
                "committed_output_offset": output_offset,
            }
            artifact = self.artifacts.get(index)
            if artifact is not None:
                metadata["artifact_size"] = artifact.size
                metadata["artifact_sha256"] = artifact.sha256
            item_metadata.append(metadata)

        payload: Dict[str, Any] = {
            "schema_version": _BATCH_CHECKPOINT_SCHEMA_VERSION,
            "status": status,
            "operation": self.operation,
            "model_sha256": _stable_hash(self.model_name),
            "configuration_sha256": self.configuration_hash,
            "input_sha256": self.input_hash,
            "total_items": len(self.items),
            "committed_count": len(results),
            "checkpoint_interval": self.checkpoint_interval,
            "journal_format": (
                "full_result" if self.store_full_results else "status_only"
            ),
            "output_format": self.output_format,
            "committed_output": {
                "offset": len(journal),
                "sha256": _sha256_bytes(journal),
            },
            "items": item_metadata,
            "started_at": self.started_at,
            "completed_at": completed_at,
            "contains_raw_input": False,
            "contains_raw_output": False,
        }
        if final_output is not None:
            payload["final_output"] = {
                "size": final_output.size,
                "sha256": final_output.sha256,
                "format": self.output_format,
            }
        checkpoint_bytes = (
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        _atomic_write_bytes(
            self.checkpoint_path,
            checkpoint_bytes,
            hook=self.atomic_write_hook,
        )

    def _serialize_journal_record(
        self,
        index: int,
        result: BatchItemResult,
    ) -> bytes:
        if self.store_full_results:
            return _serialize_result_record(result)
        payload = {
            "index": index,
            "status": "success" if result.success else "failed",
        }
        return (
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")

    def _validate_checkpoint(self, payload: Any) -> None:
        if not isinstance(payload, Mapping):
            raise BatchCheckpointError("Checkpoint must contain a JSON object")
        if (
            _checkpoint_integer(payload.get("schema_version"), "schema version")
            != _BATCH_CHECKPOINT_SCHEMA_VERSION
        ):
            raise BatchCheckpointError("Unsupported batch checkpoint schema version")
        if payload.get("status") not in {"in_progress", "complete"}:
            raise BatchCheckpointError("Checkpoint status is invalid")
        if payload.get("operation") != self.operation:
            raise BatchCheckpointError("Checkpoint operation does not match this batch")
        if payload.get("model_sha256") != _stable_hash(self.model_name):
            raise BatchCheckpointError("Checkpoint model does not match this batch")
        if payload.get("configuration_sha256") != self.configuration_hash:
            raise BatchCheckpointError(
                "Checkpoint configuration does not match this batch"
            )
        if payload.get("input_sha256") != self.input_hash:
            raise BatchCheckpointError("Checkpoint input does not match this batch")
        if _checkpoint_integer(payload.get("total_items"), "item count") != len(
            self.items
        ):
            raise BatchCheckpointError("Checkpoint item count does not match input")
        committed_count = _checkpoint_integer(
            payload.get("committed_count"),
            "committed count",
        )
        if committed_count > len(self.items):
            raise BatchCheckpointError("Checkpoint committed count is invalid")
        _checkpoint_integer(
            payload.get("checkpoint_interval"),
            "interval",
            minimum=1,
        )
        expected_journal_format = (
            "full_result" if self.store_full_results else "status_only"
        )
        if payload.get("journal_format") != expected_journal_format:
            raise BatchCheckpointError(
                "Checkpoint output mode does not match this batch"
            )
        if payload.get("output_format") != self.output_format:
            raise BatchCheckpointError(
                "Checkpoint output format does not match this batch"
            )
        committed_output = payload.get("committed_output")
        if not isinstance(committed_output, Mapping):
            raise BatchCheckpointError("Checkpoint output metadata is missing")
        _checkpoint_integer(
            committed_output.get("offset"),
            "committed output offset",
        )
        committed_hash = committed_output.get("sha256")
        if not isinstance(committed_hash, str) or len(committed_hash) != 64:
            raise BatchCheckpointError("Checkpoint committed output hash is invalid")

    def _restore_and_verify_artifact(
        self,
        index: int,
        item: BatchItem,
        metadata: Mapping[str, Any],
    ) -> None:
        size_value = metadata.get("artifact_size")
        hash_value = metadata.get("artifact_sha256")
        if size_value is None and hash_value is None:
            if not self.store_full_results and metadata.get("status") == "success":
                raise BatchCheckpointError("Checkpoint artifact metadata is incomplete")
            return
        if size_value is None or hash_value is None:
            raise BatchCheckpointError("Checkpoint artifact metadata is incomplete")
        if self.artifact_path_resolver is None:
            raise BatchCheckpointError("Checkpoint artifact cannot be resolved")
        artifact_path = self.artifact_path_resolver(index, item)
        if artifact_path is None:
            raise BatchCheckpointError("Checkpoint artifact path is unavailable")
        try:
            contents = artifact_path.read_bytes()
        except OSError as exc:
            raise BatchCheckpointError(
                "Committed output artifact is missing or unreadable"
            ) from exc
        artifact = _OutputArtifact(size=len(contents), sha256=_sha256_bytes(contents))
        expected_size = _checkpoint_integer(size_value, "artifact size")
        if artifact.size != expected_size or artifact.sha256 != str(hash_value):
            raise BatchCheckpointError(
                "Committed output artifact does not match the checkpoint hash"
            )
        self.artifacts[index] = artifact

    def _verify_final_output(self, payload: Mapping[str, Any]) -> None:
        metadata = payload.get("final_output")
        if not isinstance(metadata, Mapping):
            raise BatchCheckpointError("Completed checkpoint has no final output hash")
        try:
            contents = self.final_output_path.read_bytes()
        except OSError as exc:
            raise BatchCheckpointError(
                "Final batch output is missing or unreadable"
            ) from exc
        expected_size = _checkpoint_integer(
            metadata.get("size"),
            "final output size",
        )
        if len(contents) != expected_size or _sha256_bytes(contents) != str(
            metadata.get("sha256", "")
        ):
            raise BatchCheckpointError(
                "Final batch output does not match the checkpoint hash"
            )


def _serialize_batch_output(batch_result: BatchResult, output_format: str) -> bytes:
    if output_format == "json":
        rendered = json.dumps(batch_result.to_dict(), indent=2)
    elif output_format == "summary":
        rendered = batch_result.summary()
    else:
        raise ValueError("output_format must be 'json' or 'summary'")
    return rendered.encode("utf-8")


class BatchProcessor:
    """Process multiple texts efficiently with progress tracking.

    Example usage:
        >>> from openmed import BatchProcessor, OpenMedConfig
        >>> processor = BatchProcessor(model_name="disease_detection_superclinical")
        >>> texts = ["Patient has diabetes.", "No significant findings."]
        >>> result = processor.process_texts(texts)
        >>> print(result.summary())
    """

    def __init__(
        self,
        model_name: str = "disease_detection_superclinical",
        operation: BatchOperation = "analyze_text",
        batch_size: int = 8,
        config: Optional[Any] = None,
        loader: Optional[Any] = None,
        aggregation_strategy: Optional[str] = "simple",
        confidence_threshold: Optional[float] = None,
        group_entities: bool = False,
        continue_on_error: bool = True,
        checkpoint_interval: int = _DEFAULT_CHECKPOINT_INTERVAL,
        _atomic_write_hook: Optional[AtomicWriteHook] = None,
        **analyze_kwargs: Any,
    ):
        """Initialize batch processor.

        Args:
            model_name: Model registry key or HuggingFace identifier.
            operation: Which function to call per item: ``"analyze_text"``
                (default), ``"extract_pii"`` or ``"deidentify"``.
                Extra kwargs passed via ``**analyze_kwargs`` are passed to
                the selected function.
            batch_size: Number of documents to process together per batch.
            config: Optional OpenMedConfig instance.
            loader: Optional ModelLoader instance to reuse.
            aggregation_strategy: HuggingFace aggregation strategy
                (``analyze_text`` operation only).
            confidence_threshold: Minimum confidence for entities. When not
                provided, defaults match the selected operation:
                ``0.0`` for ``analyze_text``, ``0.5`` for ``extract_pii``,
                and ``0.7`` for ``deidentify``.
            group_entities: Whether to group adjacent entities
                (``analyze_text`` operation only).
            continue_on_error: Continue processing on individual item errors.
            checkpoint_interval: Maximum number of items processed between
                durable checkpoints.
            **analyze_kwargs: Additional arguments passed to the selected function.
        """
        if operation not in _VALID_OPERATIONS:
            allowed = ", ".join(sorted(_VALID_OPERATIONS))
            raise ValueError(
                f"Unsupported batch operation {operation!r}. Use one of: {allowed}"
            )

        from ..utils.validation import validate_batch_size

        self.model_name = model_name
        self.operation = operation
        self.batch_size = validate_batch_size(batch_size)
        self.config = config
        self.loader = loader
        self.aggregation_strategy = aggregation_strategy
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else self._default_confidence_threshold(operation)
        )
        self.group_entities = group_entities
        self.continue_on_error = continue_on_error
        if isinstance(checkpoint_interval, bool) or checkpoint_interval < 1:
            raise ValueError("Checkpoint interval must be positive")
        self.checkpoint_interval = int(checkpoint_interval)
        self._atomic_write_hook = _atomic_write_hook
        self.analyze_kwargs = analyze_kwargs

        self._analyze_text = None
        self._shared_loader = loader
        self._privacy_filter_pipeline_cache: Dict[str, Any] = {}

    @staticmethod
    def _default_confidence_threshold(operation: BatchOperation) -> float:
        """Return the single-call API default for the selected operation."""
        if operation == "extract_pii":
            return 0.5
        if operation == "deidentify":
            return 0.7
        return 0.0

    def _get_analyze_text(self) -> Callable:
        """Lazily import and cache analyze_text function."""
        if self._analyze_text is None:
            from openmed import analyze_text

            self._analyze_text = analyze_text
        return self._analyze_text

    def _get_operation_fn(self) -> Callable:
        """Return the callable for the configured operation."""
        if self.operation == "extract_pii":
            from openmed import extract_pii

            return extract_pii
        elif self.operation == "deidentify":
            from openmed import deidentify

            return deidentify

        return self._get_analyze_text()

    def _get_shared_loader(self) -> Optional[Any]:
        """Return a reusable ModelLoader when the optional dependency exists."""
        if self._shared_loader is not None:
            return self._shared_loader

        try:
            from openmed.core import ModelLoader
        except ImportError:
            return None

        try:
            self._shared_loader = ModelLoader(self.config)
        except ImportError:
            return None
        return self._shared_loader

    def _get_privacy_filter_pipeline(self) -> Optional[Any]:
        """Return a cached privacy-filter pipeline when the model uses one."""
        if self.operation not in {"extract_pii", "deidentify"}:
            return None

        try:
            from openmed.core.backends import create_privacy_filter_pipeline
            from openmed.core.pii import (
                _is_privacy_filter_artifact_path,
                _looks_like_privacy_filter_identifier,
                _resolve_effective_pii_model,
            )
        except ImportError:
            return None

        lang = self.analyze_kwargs.get("lang", "en")
        effective_model = _resolve_effective_pii_model(self.model_name, lang)
        uses_privacy_filter = _looks_like_privacy_filter_identifier(
            effective_model
        ) or _is_privacy_filter_artifact_path(effective_model)
        if not uses_privacy_filter:
            return None

        if effective_model not in self._privacy_filter_pipeline_cache:
            self._privacy_filter_pipeline_cache[effective_model] = (
                create_privacy_filter_pipeline(effective_model)
            )
        return self._privacy_filter_pipeline_cache[effective_model]

    def _iter_chunks(self, items: Sequence[BatchItem]) -> Iterator[List[BatchItem]]:
        """Yield contiguous chunks of batch items."""
        for start in range(0, len(items), self.batch_size):
            yield list(items[start : start + self.batch_size])

    def _create_batch_items(
        self,
        texts: Sequence[str],
        ids: Optional[Sequence[str]] = None,
    ) -> List[BatchItem]:
        """Create BatchItem objects from raw texts."""
        items = []
        for i, text in enumerate(texts):
            item_id = ids[i] if ids and i < len(ids) else f"item_{i}"
            items.append(BatchItem(id=item_id, text=text))
        return items

    def process_texts(
        self,
        texts: Sequence[str],
        ids: Optional[Sequence[str]] = None,
        progress_callback: Optional[ProgressCallback] = None,
        *,
        on_progress: Optional[BatchProgressCallback] = None,
        output_path: Optional[Union[str, Path]] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        resume_from_checkpoint: bool = False,
        checkpoint_interval: Optional[int] = None,
        output_format: str = "json",
    ) -> BatchResult:
        """Process multiple texts.

        Args:
            texts: Sequence of texts to analyze.
            ids: Optional identifiers for each text.
            progress_callback: Optional callback for progress updates.
                Signature: callback(completed_count, total_count, result)
            on_progress: Optional PHI-safe callback that receives a
                BatchProgress record after each completed item.
            output_path: Optional atomically written final result file.
            checkpoint_path: Optional PHI-free durable checkpoint file.
            resume_from_checkpoint: Resume the committed result prefix instead
                of starting a new checkpoint.
            checkpoint_interval: Per-run override for checkpoint frequency.
            output_format: ``"json"`` or ``"summary"`` for ``output_path``.

        Returns:
            BatchResult with all processing results.
        """
        items = self._create_batch_items(texts, ids)
        return self._process_items(
            items,
            progress_callback=progress_callback,
            on_progress=on_progress,
            output_path=output_path,
            checkpoint_path=checkpoint_path,
            resume_from_checkpoint=resume_from_checkpoint,
            checkpoint_interval=checkpoint_interval,
            output_format=output_format,
        )

    def process_files(
        self,
        file_paths: Sequence[Union[str, Path]],
        encoding: str = "utf-8",
        progress_callback: Optional[ProgressCallback] = None,
        *,
        on_progress: Optional[BatchProgressCallback] = None,
        output_path: Optional[Union[str, Path]] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        resume_from_checkpoint: bool = False,
        checkpoint_interval: Optional[int] = None,
        output_format: str = "json",
    ) -> BatchResult:
        """Process multiple files.

        Args:
            file_paths: Paths to text files.
            encoding: File encoding.
            progress_callback: Optional callback for progress updates.
            on_progress: Optional PHI-safe callback that receives a
                BatchProgress record after each completed item.
            output_path: Optional atomically written final result file.
            checkpoint_path: Optional PHI-free durable checkpoint file.
            resume_from_checkpoint: Resume the committed result prefix.
            checkpoint_interval: Per-run override for checkpoint frequency.
            output_format: ``"json"`` or ``"summary"`` for ``output_path``.

        Returns:
            BatchResult with all processing results.
        """
        items = []
        for path in file_paths:
            path = Path(path)
            try:
                text = path.read_text(encoding=encoding)
                items.append(
                    BatchItem(
                        id=path.name,
                        text=text,
                        source=str(path),
                    )
                )
            except (OSError, IOError) as e:
                logger.warning(
                    "Failed to read batch input file: error_type=%s",
                    type(e).__name__,
                )
                if not self.continue_on_error:
                    raise
                items.append(
                    BatchItem(
                        id=path.name,
                        text="",
                        source=str(path),
                        metadata={"read_error": str(e)},
                    )
                )
        return self._process_items(
            items,
            progress_callback=progress_callback,
            on_progress=on_progress,
            output_path=output_path,
            checkpoint_path=checkpoint_path,
            resume_from_checkpoint=resume_from_checkpoint,
            checkpoint_interval=checkpoint_interval,
            output_format=output_format,
        )

    def process_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*.txt",
        recursive: bool = False,
        encoding: str = "utf-8",
        progress_callback: Optional[ProgressCallback] = None,
        *,
        on_progress: Optional[BatchProgressCallback] = None,
        output_path: Optional[Union[str, Path]] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        resume_from_checkpoint: bool = False,
        checkpoint_interval: Optional[int] = None,
        output_format: str = "json",
    ) -> BatchResult:
        """Process all matching files in a directory.

        Args:
            directory: Directory path.
            pattern: Glob pattern for file matching.
            recursive: Whether to search recursively.
            encoding: File encoding.
            progress_callback: Optional callback for progress updates.
            on_progress: Optional PHI-safe callback that receives a
                BatchProgress record after each completed item.
            output_path: Optional atomically written final result file.
            checkpoint_path: Optional PHI-free durable checkpoint file.
            resume_from_checkpoint: Resume the committed result prefix.
            checkpoint_interval: Per-run override for checkpoint frequency.
            output_format: ``"json"`` or ``"summary"`` for ``output_path``.

        Returns:
            BatchResult with all processing results.
        """
        directory = Path(directory)
        if recursive:
            files = list(directory.rglob(pattern))
        else:
            files = list(directory.glob(pattern))

        files.sort()
        return self.process_files(
            files,
            encoding=encoding,
            progress_callback=progress_callback,
            on_progress=on_progress,
            output_path=output_path,
            checkpoint_path=checkpoint_path,
            resume_from_checkpoint=resume_from_checkpoint,
            checkpoint_interval=checkpoint_interval,
            output_format=output_format,
        )

    def process_items(
        self,
        items: Sequence[BatchItem],
        progress_callback: Optional[ProgressCallback] = None,
        *,
        on_progress: Optional[BatchProgressCallback] = None,
        output_path: Optional[Union[str, Path]] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        resume_from_checkpoint: bool = False,
        checkpoint_interval: Optional[int] = None,
        output_format: str = "json",
    ) -> BatchResult:
        """Process a sequence of BatchItem objects.

        Args:
            items: Sequence of BatchItem objects.
            progress_callback: Optional callback for progress updates.
            on_progress: Optional PHI-safe callback that receives a
                BatchProgress record after each completed item.
            output_path: Optional atomically written final result file.
            checkpoint_path: Optional PHI-free durable checkpoint file.
            resume_from_checkpoint: Resume the committed result prefix.
            checkpoint_interval: Per-run override for checkpoint frequency.
            output_format: ``"json"`` or ``"summary"`` for ``output_path``.

        Returns:
            BatchResult with all processing results.
        """
        return self._process_items(
            list(items),
            progress_callback=progress_callback,
            on_progress=on_progress,
            output_path=output_path,
            checkpoint_path=checkpoint_path,
            resume_from_checkpoint=resume_from_checkpoint,
            checkpoint_interval=checkpoint_interval,
            output_format=output_format,
        )

    def resume_from_checkpoint(
        self,
        items: Sequence[BatchItem],
        *,
        checkpoint_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        progress_callback: Optional[ProgressCallback] = None,
        on_progress: Optional[BatchProgressCallback] = None,
        output_format: str = "json",
    ) -> BatchResult:
        """Resume ``items`` from a previously committed batch checkpoint."""
        return self.process_items(
            items,
            progress_callback=progress_callback,
            on_progress=on_progress,
            output_path=output_path,
            checkpoint_path=checkpoint_path,
            resume_from_checkpoint=True,
            output_format=output_format,
        )

    def process_files_to_directory(
        self,
        file_paths: Sequence[Union[str, Path]],
        *,
        input_root: Union[str, Path],
        output_dir: Union[str, Path],
        encoding: str = "utf-8",
        checkpoint_path: Optional[Union[str, Path]] = None,
        resume_from_checkpoint: bool = False,
        checkpoint_interval: Optional[int] = None,
        progress_callback: Optional[ProgressCallback] = None,
        on_progress: Optional[BatchProgressCallback] = None,
    ) -> BatchResult:
        """De-identify files into an atomic, checkpointed output directory.

        Output paths preserve each input's location relative to ``input_root``.
        Committed files are hashed in the PHI-free checkpoint and verified
        before a resumed run skips them.
        """
        if self.operation != "deidentify":
            raise ValueError(
                "process_files_to_directory requires operation='deidentify'"
            )

        resolved_input_root = Path(input_root).resolve()
        resolved_output_dir = Path(output_dir)
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        items: List[BatchItem] = []
        destinations: List[Path] = []

        for raw_path in file_paths:
            path = Path(raw_path)
            relative_path = path.resolve().relative_to(resolved_input_root)
            destinations.append(resolved_output_dir / relative_path)
            try:
                text = path.read_text(encoding=encoding)
                items.append(
                    BatchItem(id=str(relative_path), text=text, source=str(path))
                )
            except (OSError, IOError) as exc:
                logger.warning(
                    "Failed to read batch input file: error_type=%s",
                    type(exc).__name__,
                )
                if not self.continue_on_error:
                    raise
                items.append(
                    BatchItem(
                        id=str(relative_path),
                        text="",
                        source=str(path),
                        metadata={"read_error": str(exc)},
                    )
                )

        def artifact_path_resolver(index: int, item: BatchItem) -> Path:
            del item
            return destinations[index]

        def artifact_writer(
            index: int,
            item: BatchItem,
            item_result: BatchItemResult,
        ) -> Optional[_OutputArtifact]:
            del item
            if not item_result.success:
                return None
            if isinstance(item_result.result, Mapping):
                deidentified_text = item_result.result.get("deidentified_text")
            else:
                deidentified_text = getattr(
                    item_result.result,
                    "deidentified_text",
                    None,
                )
            if not isinstance(deidentified_text, str):
                raise ValueError(
                    "de-identification result does not contain deidentified_text"
                )
            output_bytes = deidentified_text.encode(encoding)
            _atomic_write_bytes(
                destinations[index],
                output_bytes,
                hook=self._atomic_write_hook,
            )
            return _OutputArtifact(
                size=len(output_bytes),
                sha256=_sha256_bytes(output_bytes),
            )

        return self._process_items(
            items,
            progress_callback=progress_callback,
            on_progress=on_progress,
            checkpoint_path=checkpoint_path,
            resume_from_checkpoint=resume_from_checkpoint,
            checkpoint_interval=checkpoint_interval,
            artifact_writer=artifact_writer,
            artifact_path_resolver=artifact_path_resolver,
        )

    def _process_items(
        self,
        items: List[BatchItem],
        progress_callback: Optional[ProgressCallback] = None,
        *,
        on_progress: Optional[BatchProgressCallback] = None,
        output_path: Optional[Union[str, Path]] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        resume_from_checkpoint: bool = False,
        checkpoint_interval: Optional[int] = None,
        output_format: str = "json",
        artifact_writer: Optional[ArtifactWriter] = None,
        artifact_path_resolver: Optional[ArtifactPathResolver] = None,
    ) -> BatchResult:
        """Internal method to process batch items."""
        if output_format not in {"json", "summary"}:
            raise ValueError("output_format must be 'json' or 'summary'")
        if resume_from_checkpoint and checkpoint_path is None:
            raise ValueError("resume_from_checkpoint requires checkpoint_path")
        resolved_interval = (
            self.checkpoint_interval
            if checkpoint_interval is None
            else checkpoint_interval
        )
        if isinstance(resolved_interval, bool) or resolved_interval < 1:
            raise ValueError("Checkpoint interval must be positive")

        session: Optional[_BatchCheckpointSession] = None
        restored_results: List[BatchItemResult] = []
        if checkpoint_path is not None:
            session = _BatchCheckpointSession(
                items=items,
                checkpoint_path=Path(checkpoint_path),
                checkpoint_interval=int(resolved_interval),
                model_name=self.model_name,
                operation=self.operation,
                configuration={
                    "aggregation_strategy": self.aggregation_strategy,
                    "confidence_threshold": self.confidence_threshold,
                    "group_entities": self.group_entities,
                    "continue_on_error": self.continue_on_error,
                    "config": (
                        self.config.to_dict()
                        if callable(getattr(self.config, "to_dict", None))
                        else self.config
                    ),
                    "analyze_kwargs": self.analyze_kwargs,
                },
                final_output_path=(
                    Path(output_path) if output_path is not None else None
                ),
                output_format=output_format,
                store_full_results=artifact_writer is None,
                artifact_path_resolver=artifact_path_resolver,
                atomic_write_hook=self._atomic_write_hook,
            )
            if resume_from_checkpoint:
                restored_results = session.resume()
                resolved_interval = session.checkpoint_interval
            else:
                session.start()

        batch_result = BatchResult(
            items=list(restored_results),
            model_name=self.model_name,
            started_at=(session.started_at if session else datetime.now().isoformat()),
        )

        if session is not None and session.is_complete:
            batch_result.completed_at = session.completed_at
            batch_result.total_processing_time = sum(
                item.processing_time for item in batch_result.items
            )
            return batch_result

        total = len(items)
        batch_start = time.time()

        next_index = len(batch_result.items)
        while next_index < total:
            if session is not None:
                until_checkpoint = int(resolved_interval) - (
                    len(batch_result.items) % int(resolved_interval)
                )
                chunk_size = min(self.batch_size, until_checkpoint, total - next_index)
            else:
                chunk_size = min(self.batch_size, total - next_index)
            chunk = items[next_index : next_index + chunk_size]
            chunk_results = self._process_batch_chunk(chunk)
            for item_result in chunk_results:
                item_index = len(batch_result.items)
                if artifact_writer is not None:
                    artifact = artifact_writer(
                        item_index,
                        items[item_index],
                        item_result,
                    )
                    if artifact is not None and session is not None:
                        session.record_artifact(item_index, artifact)
                batch_result.items.append(item_result)
                self._emit_progress(
                    completed=len(batch_result.items),
                    total=total,
                    started_at=batch_start,
                    item_result=item_result,
                    progress_callback=progress_callback,
                    on_progress=on_progress,
                )
                if session is not None and (
                    len(batch_result.items) % int(resolved_interval) == 0
                ):
                    session.commit(batch_result.items, status="in_progress")
            next_index += len(chunk)

        batch_result.total_processing_time = time.time() - batch_start
        batch_result.completed_at = datetime.now().isoformat()

        if session is not None:
            session.commit(batch_result.items, status="in_progress")

        final_output: Optional[_OutputArtifact] = None
        if output_path is not None:
            output_bytes = _serialize_batch_output(batch_result, output_format)
            _atomic_write_bytes(
                Path(output_path),
                output_bytes,
                hook=self._atomic_write_hook,
            )
            final_output = _OutputArtifact(
                size=len(output_bytes),
                sha256=_sha256_bytes(output_bytes),
            )

        if session is not None:
            session.commit(
                batch_result.items,
                status="complete",
                completed_at=batch_result.completed_at,
                final_output=final_output,
            )

        return batch_result

    def _emit_progress(
        self,
        *,
        completed: int,
        total: int,
        started_at: float,
        item_result: BatchItemResult,
        progress_callback: Optional[ProgressCallback] = None,
        on_progress: Optional[BatchProgressCallback] = None,
    ) -> None:
        """Notify optional progress callbacks after one item completes."""
        if on_progress:
            progress = BatchProgress(
                completed=completed,
                total=total,
                current_index=completed - 1,
                elapsed=time.time() - started_at,
            )
            try:
                on_progress(progress)
            except Exception as e:
                logger.warning(
                    "on_progress callback raised; continuing batch: error_type=%s",
                    type(e).__name__,
                )

        if progress_callback:
            try:
                progress_callback(completed, total, item_result)
            except Exception as e:
                logger.warning(
                    "progress_callback raised; continuing batch: error_type=%s",
                    type(e).__name__,
                )

    def _process_batch_chunk(self, items: List[BatchItem]) -> List[BatchItemResult]:
        """Process a contiguous item chunk for the selected operation."""
        if self.operation == "analyze_text":
            return self._process_analyze_chunk(items)
        if self.operation == "extract_pii":
            return self._process_extract_pii_chunk(items)
        return self._process_deidentify_chunk(items)

    def _empty_item_result(self, item: BatchItem) -> BatchItemResult:
        """Create an error result for empty or unreadable items."""
        read_error = (item.metadata or {}).get("read_error")
        return BatchItemResult(
            id=item.id,
            error=read_error or "Empty text",
            source=item.source,
        )

    def _process_analyze_chunk(self, items: List[BatchItem]) -> List[BatchItemResult]:
        """Process a chunk with analyze_text while reusing one loader."""
        analyze_text = self._get_analyze_text()
        return [self._process_single_item(item, analyze_text) for item in items]

    def _process_extract_pii_chunk(
        self, items: List[BatchItem]
    ) -> List[BatchItemResult]:
        """Process a chunk with batched PII extraction."""
        from openmed.core.pii import _extract_pii_batch

        return self._process_pii_chunk(
            items,
            _extract_pii_batch,
        )

    def _process_deidentify_chunk(
        self, items: List[BatchItem]
    ) -> List[BatchItemResult]:
        """Process a chunk with batched de-identification."""
        from openmed.core.pii import _deidentify_batch

        return self._process_pii_chunk(
            items,
            _deidentify_batch,
        )

    def _process_pii_chunk(
        self,
        items: List[BatchItem],
        batch_fn: Callable[..., List[Any]],
    ) -> List[BatchItemResult]:
        """Run a PII batch helper and map outputs back to item results."""
        results: List[Optional[BatchItemResult]] = [None] * len(items)
        valid_positions = [
            (index, item) for index, item in enumerate(items) if item.text
        ]

        for index, item in enumerate(items):
            if not item.text:
                results[index] = self._empty_item_result(item)

        if not valid_positions:
            return [item for item in results if item is not None]

        valid_items = [item for _, item in valid_positions]
        start_time = time.time()
        try:
            privacy_filter_pipeline = self._get_privacy_filter_pipeline()
            operation_kwargs = {
                "model_name": self.model_name,
                "confidence_threshold": self.confidence_threshold,
                "config": self.config,
                "loader": (
                    None
                    if privacy_filter_pipeline is not None
                    else self._get_shared_loader()
                ),
                "batch_size": self.batch_size,
                **self.analyze_kwargs,
            }
            if privacy_filter_pipeline is not None:
                operation_kwargs["privacy_filter_pipeline"] = privacy_filter_pipeline

            operation_results = batch_fn(
                [item.text for item in valid_items],
                **operation_kwargs,
            )
            if len(operation_results) != len(valid_items):
                raise ValueError(
                    "Batch operation returned "
                    f"{len(operation_results)} results for {len(valid_items)} inputs"
                )
            elapsed = time.time() - start_time
            per_item_time = elapsed / len(valid_items) if valid_items else 0.0

            for (index, item), result in zip(valid_positions, operation_results):
                results[index] = BatchItemResult(
                    id=item.id,
                    result=result,
                    processing_time=per_item_time,
                    source=item.source,
                )

        except Exception as e:
            logger.warning(
                "Error processing batch chunk: error_type=%s",
                type(e).__name__,
            )
            if not self.continue_on_error:
                raise

            fn = self._get_operation_fn()
            for index, item in valid_positions:
                results[index] = self._process_single_item(item, fn)

        return [item for item in results if item is not None]

    def _process_single_item(
        self,
        item: BatchItem,
        fn: Callable,
    ) -> BatchItemResult:
        """Process a single batch item."""
        if not item.text:
            return self._empty_item_result(item)

        start_time = time.time()
        try:
            if self.operation == "analyze_text":
                result = fn(
                    item.text,
                    model_name=self.model_name,
                    config=self.config,
                    loader=self._get_shared_loader(),
                    aggregation_strategy=self.aggregation_strategy,
                    confidence_threshold=self.confidence_threshold,
                    group_entities=self.group_entities,
                    output_format="dict",
                    metadata=item.metadata,
                    **self.analyze_kwargs,
                )
            else:
                result = fn(
                    item.text,
                    model_name=self.model_name,
                    confidence_threshold=self.confidence_threshold,
                    config=self.config,
                    loader=self._get_shared_loader(),
                    **self.analyze_kwargs,
                )
            processing_time = time.time() - start_time

            return BatchItemResult(
                id=item.id,
                result=result,
                processing_time=processing_time,
                source=item.source,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.warning(
                "Error processing batch item: error_type=%s",
                type(e).__name__,
            )

            if not self.continue_on_error:
                raise

            return BatchItemResult(
                id=item.id,
                error=str(e),
                processing_time=processing_time,
                source=item.source,
            )

    def iter_process(
        self,
        texts: Sequence[str],
        ids: Optional[Sequence[str]] = None,
        *,
        on_progress: Optional[BatchProgressCallback] = None,
    ) -> Iterator[BatchItemResult]:
        """Process texts as an iterator, yielding results one at a time.

        This is useful for streaming results or processing very large batches
        where you don't want to hold all results in memory.

        Args:
            texts: Sequence of texts to analyze.
            ids: Optional identifiers for each text.
            on_progress: Optional PHI-safe callback that receives a
                BatchProgress record after each completed item.

        Yields:
            BatchItemResult for each processed text.
        """
        items = self._create_batch_items(texts, ids)
        total = len(items)
        completed = 0
        batch_start = time.time()

        for chunk in self._iter_chunks(items):
            for result in self._process_batch_chunk(chunk):
                completed += 1
                self._emit_progress(
                    completed=completed,
                    total=total,
                    started_at=batch_start,
                    item_result=result,
                    on_progress=on_progress,
                )
                yield result


def process_batch(
    texts: Sequence[str],
    model_name: str = "disease_detection_superclinical",
    ids: Optional[Sequence[str]] = None,
    config: Optional[Any] = None,
    progress_callback: Optional[ProgressCallback] = None,
    on_progress: Optional[BatchProgressCallback] = None,
    **kwargs: Any,
) -> BatchResult:
    """Convenience function for batch processing texts.

    Args:
        texts: Sequence of texts to analyze.
        model_name: Model registry key or HuggingFace identifier.
        ids: Optional identifiers for each text.
        config: Optional OpenMedConfig instance.
        progress_callback: Optional callback for progress updates.
        on_progress: Optional PHI-safe callback that receives a
            BatchProgress record after each completed item.
        **kwargs: Additional arguments passed to BatchProcessor.

    Returns:
        BatchResult with all processing results.

    Example:
        >>> from openmed import process_batch
        >>> texts = ["Patient has diabetes.", "Normal findings."]
        >>> result = process_batch(texts, model_name="disease_detection_superclinical")
        >>> print(f"Processed {result.successful_items}/{result.total_items} texts")
    """
    processor = BatchProcessor(model_name=model_name, config=config, **kwargs)
    return processor.process_texts(
        texts,
        ids,
        progress_callback,
        on_progress=on_progress,
    )


def redact_dataset(
    path: Union[str, Path],
    text_columns: Sequence[str],
    *,
    output_path: Optional[Union[str, Path]] = None,
    policy: Optional[str] = None,
    method: str = "mask",
    model_name: str = _DEFAULT_PII_MODEL,
    confidence_threshold: float = 0.7,
    config: Optional[Any] = None,
    lang: str = "en",
    keep_year: bool = True,
    date_shift_days: Optional[int] = None,
    use_safety_sweep: bool = True,
    encoding: str = "utf-8",
    batch_size: int = 512,
) -> DatasetRedactionResult:
    """Redact selected free-text columns in a local dataset.

    Supported input formats are ``.csv``, ``.jsonl``/``.ndjson``, and
    ``.parquet``. CSV and JSONL files are streamed row-by-row. Parquet files
    are processed in row batches through ``pyarrow`` when that optional
    dependency is installed.

    Args:
        path: Input dataset path.
        text_columns: Free-text column names to pass through ``deidentify``.
        output_path: Destination path. Defaults to ``<stem>.redacted<suffix>``.
        policy: Optional de-identification policy profile name.
        method: De-identification method forwarded to ``deidentify``.
        model_name: PII detection model.
        confidence_threshold: Minimum confidence for redaction.
        config: Optional OpenMed configuration.
        lang: Language hint forwarded to ``deidentify``.
        keep_year: Keep the year in date redaction where applicable.
        date_shift_days: Optional fixed date shift.
        use_safety_sweep: Enable deterministic structured-identifier sweep.
        encoding: Text encoding for CSV/JSONL.
        batch_size: Parquet row batch size.

    Returns:
        DatasetRedactionResult with output path and PHI-free audit summary.
    """
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Dataset path must be a file: {input_path}")

    dataset_format = _infer_dataset_format(input_path)
    normalized_columns = _normalize_text_columns(text_columns)
    destination = (
        Path(output_path)
        if output_path is not None
        else _default_output_path(
            input_path,
            dataset_format,
        )
    )
    if input_path.resolve() == destination.resolve():
        raise ValueError("output_path must not overwrite the input dataset")

    destination.parent.mkdir(parents=True, exist_ok=True)
    summary = DatasetRedactionSummary(
        input_format=dataset_format,
        text_columns=list(normalized_columns),
    )
    deidentify_kwargs = {
        "method": method,
        "model_name": model_name,
        "confidence_threshold": confidence_threshold,
        "keep_year": keep_year,
        "date_shift_days": date_shift_days,
        "config": config,
        "lang": lang,
        "use_safety_sweep": use_safety_sweep,
        "policy": policy,
    }

    if dataset_format == "csv":
        _redact_csv_dataset(
            input_path,
            destination,
            normalized_columns,
            summary,
            deidentify_kwargs,
            encoding,
        )
    elif dataset_format == "jsonl":
        _redact_jsonl_dataset(
            input_path,
            destination,
            normalized_columns,
            summary,
            deidentify_kwargs,
            encoding,
        )
    else:
        _redact_parquet_dataset(
            input_path,
            destination,
            normalized_columns,
            summary,
            deidentify_kwargs,
            batch_size,
        )

    return DatasetRedactionResult(output_path=destination, summary=summary)


def _infer_dataset_format(path: Path) -> DatasetFormat:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".jsonl", ".ndjson"}:
        return "jsonl"
    if suffix == ".parquet":
        return "parquet"
    raise ValueError(
        "Unsupported dataset format. Expected .csv, .jsonl, .ndjson, or .parquet."
    )


def _default_output_path(path: Path, dataset_format: DatasetFormat) -> Path:
    suffix = ".jsonl" if dataset_format == "jsonl" else path.suffix
    return path.with_name(f"{path.stem}.redacted{suffix}")


def _normalize_text_columns(text_columns: Sequence[str]) -> tuple[str, ...]:
    columns: list[str] = []
    seen: set[str] = set()
    for column in text_columns:
        name = str(column).strip()
        if not name:
            continue
        if name not in seen:
            seen.add(name)
            columns.append(name)
    if not columns:
        raise ValueError("At least one text column must be selected")
    return tuple(columns)


def _validate_text_columns(
    available: Sequence[str], text_columns: Sequence[str]
) -> None:
    missing = [column for column in text_columns if column not in available]
    if missing:
        raise ValueError(f"Missing text column(s): {', '.join(missing)}")


def _redact_csv_dataset(
    input_path: Path,
    output_path: Path,
    text_columns: Sequence[str],
    summary: DatasetRedactionSummary,
    deidentify_kwargs: Mapping[str, Any],
    encoding: str,
) -> None:
    with input_path.open("r", encoding=encoding, newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError("CSV input must include a header row")
        _validate_text_columns(fieldnames, text_columns)

        with output_path.open("w", encoding=encoding, newline="") as target:
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                writer.writerow(
                    _redact_dataset_row(row, text_columns, summary, deidentify_kwargs)
                )
                summary.total_rows += 1


def _redact_jsonl_dataset(
    input_path: Path,
    output_path: Path,
    text_columns: Sequence[str],
    summary: DatasetRedactionSummary,
    deidentify_kwargs: Mapping[str, Any],
    encoding: str,
) -> None:
    with (
        input_path.open("r", encoding=encoding) as source,
        output_path.open(
            "w",
            encoding=encoding,
        ) as target,
    ):
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                target.write(line)
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_number} must be an object")
            redacted = _redact_dataset_row(
                row,
                text_columns,
                summary,
                deidentify_kwargs,
            )
            target.write(
                json.dumps(redacted, ensure_ascii=False, separators=(",", ":"))
            )
            target.write("\n")
            summary.total_rows += 1


def _redact_parquet_dataset(
    input_path: Path,
    output_path: Path,
    text_columns: Sequence[str],
    summary: DatasetRedactionSummary,
    deidentify_kwargs: Mapping[str, Any],
    batch_size: int,
) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Parquet dataset redaction requires pyarrow to be installed."
        ) from exc

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    parquet_file = pq.ParquetFile(input_path)
    schema = parquet_file.schema_arrow
    _validate_text_columns(schema.names, text_columns)
    for column in text_columns:
        field_type = schema.field(column).type
        if not (
            pa.types.is_string(field_type)
            or pa.types.is_large_string(field_type)
            or pa.types.is_null(field_type)
        ):
            raise ValueError(f"Parquet text column must be string-typed: {column}")

    writer = None
    try:
        for record_batch in parquet_file.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([record_batch], schema=schema)
            rows = [
                _redact_dataset_row(row, text_columns, summary, deidentify_kwargs)
                for row in table.to_pylist()
            ]
            summary.total_rows += len(rows)
            redacted_table = pa.Table.from_pylist(rows, schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(output_path, schema)
            writer.write_table(redacted_table)

        if writer is None:
            writer = pq.ParquetWriter(output_path, schema)
    finally:
        if writer is not None:
            writer.close()


def _redact_dataset_row(
    row: Mapping[str, Any],
    text_columns: Sequence[str],
    summary: DatasetRedactionSummary,
    deidentify_kwargs: Mapping[str, Any],
) -> Dict[str, Any]:
    output = dict(row)
    _validate_text_columns(tuple(output.keys()), text_columns)

    for column in text_columns:
        value = output[column]
        if value is None:
            continue
        text = value if isinstance(value, str) else str(value)
        if text == "":
            continue

        result = _deidentify_dataset_cell(text, deidentify_kwargs)
        output[column] = result.deidentified_text
        _update_redaction_summary(summary, text, result)

    return output


def _deidentify_dataset_cell(text: str, kwargs: Mapping[str, Any]) -> Any:
    from openmed.core.pii import deidentify

    return deidentify(text, **dict(kwargs))


def _update_redaction_summary(
    summary: DatasetRedactionSummary,
    original_text: str,
    result: Any,
) -> None:
    summary.processed_cells += 1
    if result.deidentified_text != original_text:
        summary.redacted_cells += 1

    label_counts: Counter[str] = Counter()
    residual_spans = 0
    for entity in getattr(result, "pii_entities", []) or []:
        label = str(
            getattr(entity, "label", None)
            or getattr(entity, "entity_type", None)
            or "UNKNOWN"
        )
        label_counts[label] += 1
        surface = getattr(entity, "text", None) or getattr(
            entity, "original_text", None
        )
        if surface and surface in result.deidentified_text:
            residual_spans += 1

    for label, count in label_counts.items():
        summary.per_label_counts[label] = summary.per_label_counts.get(label, 0) + count
    summary.total_spans += sum(label_counts.values())
    summary.residual_span_count += residual_spans
