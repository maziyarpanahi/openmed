"""HTTP sidecar for redacting search-ingest document envelopes."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, status

from openmed.core.pii_i18n import DEFAULT_PII_MODELS
from openmed.processing.batch import process_batch

DEFAULT_SEARCH_INGEST_MODEL = DEFAULT_PII_MODELS["en"]
ProcessBatchCallable = Callable[..., Any]


class SearchIngestProcessorError(RuntimeError):
    """Raised when an ingest document cannot be redacted safely."""


@dataclass(frozen=True)
class _Target:
    field_name: str
    path: tuple[str, ...]
    text: str


def redact_ingest_document(
    document: Mapping[str, Any],
    *,
    fields: Sequence[str],
    policy: str | None = None,
    process_batch_fn: ProcessBatchCallable = process_batch,
) -> dict[str, Any]:
    """Return a redacted copy of a search ingest-document envelope.

    Field names are resolved relative to ``_source`` by default. Paths that
    start with ``_source.`` or another envelope key are also accepted. Missing
    paths, non-string values, and empty strings are preserved without invoking
    the batch processor.

    Args:
        document: Elasticsearch/OpenSearch-style ingest document envelope.
        fields: Top-level or dotted paths to string fields that should be
            redacted.
        policy: Optional OpenMed policy profile for this pipeline invocation.
        process_batch_fn: Batch processor implementation, replaceable for
            offline tests and embedded deployments.

    Returns:
        A deep copy of ``document`` with configured string fields redacted.

    Raises:
        TypeError: If ``document`` is not a mapping.
        ValueError: If a configured field path or policy is invalid.
        SearchIngestProcessorError: If batch redaction fails or returns an
            invalid result. Error messages never include document text.
    """

    if not isinstance(document, Mapping):
        raise TypeError("document must be a mapping")

    normalized_fields = _normalize_fields(fields)
    normalized_policy = _normalize_policy(policy)
    output = copy.deepcopy(dict(document))
    targets = _collect_targets(output, normalized_fields)
    if not targets:
        return output

    kwargs: dict[str, Any] = {
        "model_name": DEFAULT_SEARCH_INGEST_MODEL,
        "ids": [f"ingest:{target.field_name}" for target in targets],
        "operation": "deidentify",
        "batch_size": len(targets),
        "method": "mask",
        "confidence_threshold": 0.7,
        "continue_on_error": False,
        "use_safety_sweep": True,
    }
    if normalized_policy is not None:
        kwargs["policy"] = normalized_policy

    try:
        batch_result = process_batch_fn(
            [target.text for target in targets],
            **kwargs,
        )
        redacted_texts = _extract_redacted_texts(
            batch_result,
            expected=len(targets),
        )
    except Exception:
        raise SearchIngestProcessorError(
            "failed to redact configured ingest document fields"
        ) from None

    for target, redacted_text in zip(targets, redacted_texts):
        _set_path(output, target.path, redacted_text)
    return output


def create_app(
    *,
    process_batch_fn: ProcessBatchCallable = process_batch,
) -> FastAPI:
    """Create the search-ingest redaction sidecar application.

    Args:
        process_batch_fn: Batch processor implementation used by ``/process``.

    Returns:
        A configured FastAPI application.
    """

    app = FastAPI(
        title="OpenMed search ingest redaction processor",
        version="1.0.0",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/process")
    def process_document(payload: dict[str, Any]) -> dict[str, Any]:
        document = payload.get("document")
        fields = payload.get("fields")
        policy = payload.get("policy")

        if not isinstance(document, Mapping):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="document must be an object",
            )
        if not isinstance(fields, list) or any(
            not isinstance(field, str) for field in fields
        ):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="fields must be an array of strings",
            )
        if policy is not None and not isinstance(policy, str):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="policy must be a string or null",
            )

        try:
            return redact_ingest_document(
                document,
                fields=fields,
                policy=policy,
                process_batch_fn=process_batch_fn,
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from None
        except SearchIngestProcessorError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from None

    return app


def _normalize_fields(fields: Sequence[str]) -> tuple[str, ...]:
    if isinstance(fields, (str, bytes)):
        raise TypeError("fields must be a sequence of strings")

    normalized: list[str] = []
    seen: set[str] = set()
    for field in fields:
        if not isinstance(field, str):
            raise TypeError("fields must contain only strings")
        field_name = field.strip()
        if not field_name:
            raise ValueError("field paths must not be empty")
        if any(not part for part in field_name.split(".")):
            raise ValueError("field paths must not contain empty segments")
        if field_name not in seen:
            seen.add(field_name)
            normalized.append(field_name)
    return tuple(normalized)


def _normalize_policy(policy: str | None) -> str | None:
    if policy is None:
        return None
    if not isinstance(policy, str):
        raise TypeError("policy must be a string or null")
    normalized = policy.strip()
    if not normalized:
        raise ValueError("policy must not be empty")
    return normalized


def _collect_targets(
    document: Mapping[str, Any],
    fields: Sequence[str],
) -> list[_Target]:
    targets: list[_Target] = []
    seen_paths: set[tuple[str, ...]] = set()
    for field_name in fields:
        path = _resolve_path(document, field_name)
        if path is None or path in seen_paths:
            continue
        value = _get_path(document, path)
        if isinstance(value, str) and value:
            targets.append(_Target(field_name=field_name, path=path, text=value))
            seen_paths.add(path)
    return targets


def _resolve_path(
    document: Mapping[str, Any],
    field_name: str,
) -> tuple[str, ...] | None:
    parts = tuple(field_name.split("."))
    source = document.get("_source")

    if parts[0] != "_source" and isinstance(source, Mapping):
        if field_name in source:
            return ("_source", field_name)
        source_path = ("_source", *parts)
        if _path_exists(document, source_path):
            return source_path

    if field_name in document:
        return (field_name,)
    if _path_exists(document, parts):
        return parts
    return None


def _path_exists(document: Mapping[str, Any], path: Sequence[str]) -> bool:
    try:
        _get_path(document, path)
    except (KeyError, TypeError):
        return False
    return True


def _get_path(document: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = document
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(part)
        current = current[part]
    return current


def _set_path(document: dict[str, Any], path: Sequence[str], value: str) -> None:
    current = document
    for part in path[:-1]:
        nested = current[part]
        if not isinstance(nested, dict):
            raise SearchIngestProcessorError(
                "configured ingest field path is not an object"
            )
        current = nested
    current[path[-1]] = value


def _extract_redacted_texts(batch_result: Any, *, expected: int) -> list[str]:
    raw_items = getattr(batch_result, "items", batch_result)
    try:
        items = list(raw_items)
    except TypeError as exc:
        raise SearchIngestProcessorError(
            "ingest redaction returned an invalid result"
        ) from exc

    if len(items) != expected:
        raise SearchIngestProcessorError(
            "ingest redaction returned an unexpected result count"
        )

    redacted_texts: list[str] = []
    for item in items:
        if hasattr(item, "success") and not item.success:
            raise SearchIngestProcessorError(
                "ingest redaction failed for one or more configured fields"
            )
        result = getattr(item, "result", item)
        redacted_text = (
            result
            if isinstance(result, str)
            else getattr(result, "deidentified_text", None)
        )
        if not isinstance(redacted_text, str):
            raise SearchIngestProcessorError(
                "ingest redaction returned an invalid field result"
            )
        redacted_texts.append(redacted_text)
    return redacted_texts


app = create_app()
