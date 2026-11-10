"""Object-storage de-identification helpers."""

from __future__ import annotations

import logging
import posixpath
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Callable, Mapping, Optional

from .batch import BatchProgress, BatchProgressCallback

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObjectStorageItemResult:
    """PHI-minimized result for one object-storage input."""

    relative_path: str
    source_uri: str
    output_uri: str
    success: bool
    error: Optional[str] = None
    processing_time: float = 0.0
    span_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "relative_path": self.relative_path,
            "source_uri": self.source_uri,
            "output_uri": self.output_uri,
            "success": self.success,
            "error": self.error,
            "processing_time": self.processing_time,
            "span_count": self.span_count,
        }


@dataclass
class ObjectStorageBatchResult:
    """Aggregate result returned by :func:`deidentify_bucket`."""

    input_uri: str
    output_uri: str
    items: list[ObjectStorageItemResult] = field(default_factory=list)
    total_processing_time: float = 0.0

    @property
    def total_objects(self) -> int:
        """Total object count matched by the input glob."""
        return len(self.items)

    @property
    def successful_objects(self) -> int:
        """Number of objects written successfully."""
        return sum(1 for item in self.items if item.success)

    @property
    def failed_objects(self) -> int:
        """Number of objects that failed to process."""
        return sum(1 for item in self.items if not item.success)

    @property
    def total_spans(self) -> int:
        """Total detected PII spans across processed objects."""
        return sum(item.span_count for item in self.items)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "input_uri": self.input_uri,
            "output_uri": self.output_uri,
            "total_objects": self.total_objects,
            "successful_objects": self.successful_objects,
            "failed_objects": self.failed_objects,
            "total_spans": self.total_spans,
            "total_processing_time": self.total_processing_time,
            "items": [item.to_dict() for item in self.items],
        }


ObjectProgressCallback = Callable[
    [int, int, ObjectStorageItemResult],
    None,
]


def deidentify_bucket(
    uri_in: str,
    uri_out: str,
    *,
    policy: Optional[str] = None,
    glob: str = "**/*.txt",
    concurrency: int = 1,
    encoding: str = "utf-8",
    storage_options: Optional[Mapping[str, Any]] = None,
    input_storage_options: Optional[Mapping[str, Any]] = None,
    output_storage_options: Optional[Mapping[str, Any]] = None,
    progress_callback: Optional[ObjectProgressCallback] = None,
    on_progress: Optional[BatchProgressCallback] = None,
    continue_on_error: bool = True,
    **deidentify_kwargs: Any,
) -> ObjectStorageBatchResult:
    """De-identify text objects from one fsspec URI prefix into another.

    ``uri_in`` and ``uri_out`` may use any fsspec-supported filesystem, including
    ``s3://``, ``gs://``, ``az://``, ``memory://`` and ``file://``. Objects are
    read directly from the source filesystem and written to the destination
    filesystem with the same relative key layout. The output root must be
    distinct from the input root to avoid writing redacted objects into the
    source tree.

    Args:
        uri_in: Input object-storage URI or local file URI.
        uri_out: Output object-storage URI or local file URI.
        policy: Optional de-identification policy profile name.
        glob: fsspec glob pattern relative to ``uri_in``.
        concurrency: Maximum number of objects processed concurrently.
        encoding: Text encoding for object reads and writes.
        storage_options: Common fsspec options for both input and output.
        input_storage_options: fsspec options for the input filesystem.
        output_storage_options: fsspec options for the output filesystem.
        progress_callback: Legacy callback called as ``(completed, total, item)``.
        on_progress: PHI-safe callback receiving ``BatchProgress``.
        continue_on_error: Continue after individual object failures.
        **deidentify_kwargs: Additional keyword arguments forwarded to
            :func:`openmed.deidentify`.

    Returns:
        ObjectStorageBatchResult with per-object status and aggregate counts.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be positive")

    fsspec = _import_fsspec()
    common_options = dict(storage_options or {})
    input_options = {**common_options, **dict(input_storage_options or {})}
    output_options = {**common_options, **dict(output_storage_options or {})}
    input_fs, input_root = fsspec.core.url_to_fs(uri_in, **input_options)
    output_fs, output_root = fsspec.core.url_to_fs(uri_out, **output_options)
    input_root = _normalize_storage_path(input_root)
    output_root = _normalize_storage_path(output_root)
    if not output_root:
        raise ValueError("uri_out must include an output bucket or path")

    _validate_distinct_roots(input_fs, input_root, output_fs, output_root)
    source_paths = _list_source_paths(input_fs, input_root, glob)
    deidentify_options = dict(deidentify_kwargs)
    deidentify_options["policy"] = policy

    started_at = time.time()
    result = ObjectStorageBatchResult(input_uri=uri_in, output_uri=uri_out)
    if not source_paths:
        return result

    if concurrency == 1 or len(source_paths) == 1:
        items: list[ObjectStorageItemResult] = []
        for completed, source_path in enumerate(source_paths, start=1):
            item = _process_object(
                source_path=source_path,
                input_root=input_root,
                input_fs=input_fs,
                output_root=output_root,
                output_fs=output_fs,
                encoding=encoding,
                deidentify_options=deidentify_options,
                continue_on_error=continue_on_error,
            )
            items.append(item)
            _emit_progress(
                completed=completed,
                total=len(source_paths),
                started_at=started_at,
                item_result=item,
                progress_callback=progress_callback,
                on_progress=on_progress,
            )
        result.items = items
    else:
        result.items = _process_objects_concurrently(
            source_paths=source_paths,
            input_root=input_root,
            input_fs=input_fs,
            output_root=output_root,
            output_fs=output_fs,
            encoding=encoding,
            deidentify_options=deidentify_options,
            concurrency=concurrency,
            continue_on_error=continue_on_error,
            started_at=started_at,
            progress_callback=progress_callback,
            on_progress=on_progress,
        )

    result.total_processing_time = time.time() - started_at
    return result


def _import_fsspec() -> Any:
    try:
        return import_module("fsspec")
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            "Object-storage de-identification requires fsspec. "
            "Install with: pip install openmed[cloud]"
        ) from exc


def _process_objects_concurrently(
    *,
    source_paths: list[str],
    input_root: str,
    input_fs: Any,
    output_root: str,
    output_fs: Any,
    encoding: str,
    deidentify_options: Mapping[str, Any],
    concurrency: int,
    continue_on_error: bool,
    started_at: float,
    progress_callback: Optional[ObjectProgressCallback],
    on_progress: Optional[BatchProgressCallback],
) -> list[ObjectStorageItemResult]:
    items: list[Optional[ObjectStorageItemResult]] = [None] * len(source_paths)
    completed = 0
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_index = {
            executor.submit(
                _process_object,
                source_path=source_path,
                input_root=input_root,
                input_fs=input_fs,
                output_root=output_root,
                output_fs=output_fs,
                encoding=encoding,
                deidentify_options=deidentify_options,
                continue_on_error=continue_on_error,
            ): index
            for index, source_path in enumerate(source_paths)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            item = future.result()
            items[index] = item
            completed += 1
            _emit_progress(
                completed=completed,
                total=len(source_paths),
                started_at=started_at,
                item_result=item,
                progress_callback=progress_callback,
                on_progress=on_progress,
            )

    return [item for item in items if item is not None]


def _process_object(
    *,
    source_path: str,
    input_root: str,
    input_fs: Any,
    output_root: str,
    output_fs: Any,
    encoding: str,
    deidentify_options: Mapping[str, Any],
    continue_on_error: bool,
) -> ObjectStorageItemResult:
    started_at = time.time()
    relative_path = _relative_key(source_path, input_root)
    output_path = _join_storage_path(output_root, relative_path)
    source_uri = _format_uri(input_fs, source_path)
    output_uri = _format_uri(output_fs, output_path)

    try:
        with input_fs.open(source_path, "rt", encoding=encoding) as source:
            text = source.read()
        deidentified = _deidentify_text(text, deidentify_options)
        redacted_text = _coerce_deidentified_text(deidentified)
        parent = posixpath.dirname(output_path)
        if parent:
            output_fs.makedirs(parent, exist_ok=True)
        with output_fs.open(output_path, "wt", encoding=encoding) as target:
            target.write(redacted_text)
        return ObjectStorageItemResult(
            relative_path=relative_path,
            source_uri=source_uri,
            output_uri=output_uri,
            success=True,
            processing_time=time.time() - started_at,
            span_count=_count_spans(deidentified),
        )
    except Exception as exc:
        logger.warning(
            "Error processing object-storage item: error_type=%s",
            type(exc).__name__,
        )
        if not continue_on_error:
            raise
        return ObjectStorageItemResult(
            relative_path=relative_path,
            source_uri=source_uri,
            output_uri=output_uri,
            success=False,
            error=str(exc),
            processing_time=time.time() - started_at,
        )


def _deidentify_text(text: str, options: Mapping[str, Any]) -> Any:
    from openmed import deidentify

    return deidentify(text, **dict(options))


def _coerce_deidentified_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if hasattr(result, "deidentified_text"):
        return str(result.deidentified_text)
    raise TypeError("deidentify must return text or an object with deidentified_text")


def _count_spans(result: Any) -> int:
    return len(getattr(result, "pii_entities", []) or [])


def _list_source_paths(fs: Any, root: str, glob_pattern: str) -> list[str]:
    if _is_file(fs, root):
        return [root]

    pattern = _join_storage_path(root, glob_pattern)
    paths = []
    for path in fs.glob(pattern):
        normalized = _normalize_storage_path(path)
        if normalized and not _is_directory(fs, normalized):
            paths.append(normalized)
    return sorted(paths)


def _is_file(fs: Any, path: str) -> bool:
    try:
        return bool(fs.isfile(path))
    except (AttributeError, OSError, ValueError):
        return False


def _is_directory(fs: Any, path: str) -> bool:
    try:
        return bool(fs.isdir(path))
    except (AttributeError, OSError, ValueError):
        return False


def _emit_progress(
    *,
    completed: int,
    total: int,
    started_at: float,
    item_result: ObjectStorageItemResult,
    progress_callback: Optional[ObjectProgressCallback],
    on_progress: Optional[BatchProgressCallback],
) -> None:
    if on_progress:
        progress = BatchProgress(
            completed=completed,
            total=total,
            current_index=completed - 1,
            elapsed=time.time() - started_at,
        )
        try:
            on_progress(progress)
        except Exception as exc:
            logger.warning(
                "on_progress callback raised; continuing object batch: error_type=%s",
                type(exc).__name__,
            )

    if progress_callback:
        try:
            progress_callback(completed, total, item_result)
        except Exception as exc:
            logger.warning(
                "progress_callback raised; continuing object batch: error_type=%s",
                type(exc).__name__,
            )


def _validate_distinct_roots(
    input_fs: Any,
    input_root: str,
    output_fs: Any,
    output_root: str,
) -> None:
    if _protocol(input_fs) != _protocol(output_fs):
        return
    if _is_same_or_nested(input_root, output_root) or _is_same_or_nested(
        output_root,
        input_root,
    ):
        raise ValueError("uri_out must not be equal to or nested with uri_in")


def _is_same_or_nested(parent: str, child: str) -> bool:
    parent = parent.rstrip("/")
    child = child.rstrip("/")
    return bool(parent) and (child == parent or child.startswith(f"{parent}/"))


def _relative_key(path: str, root: str) -> str:
    path = _align_leading_slash(_normalize_storage_path(path), root)
    root = _align_leading_slash(_normalize_storage_path(root), path)
    if path == root:
        return posixpath.basename(path)
    if path.startswith(f"{root}/"):
        return path[len(root) + 1 :]
    relative = posixpath.relpath(path, root)
    if relative == "." or relative.startswith("../"):
        return posixpath.basename(path)
    return relative


def _align_leading_slash(path: str, reference: str) -> str:
    if path.startswith("/") != reference.startswith("/"):
        return path.lstrip("/")
    return path


def _join_storage_path(root: str, child: str) -> str:
    root = _normalize_storage_path(root)
    child = str(child).replace("\\", "/").strip("/")
    if not root:
        return child
    if not child:
        return root
    return f"{root.rstrip('/')}/{child}"


def _normalize_storage_path(path: Any) -> str:
    normalized = posixpath.normpath(str(path).replace("\\", "/"))
    if normalized == ".":
        return ""
    return normalized.rstrip("/")


def _format_uri(fs: Any, path: str) -> str:
    protocol = _protocol(fs)
    if protocol == "file":
        path = _normalize_storage_path(path)
        return f"file://{path}" if path.startswith("/") else path
    return f"{protocol}://{_normalize_storage_path(path).lstrip('/')}"


def _protocol(fs: Any) -> str:
    protocol = getattr(fs, "protocol", "file")
    if isinstance(protocol, (tuple, list)):
        protocol = protocol[0]
    if protocol == "local":
        return "file"
    return str(protocol)
