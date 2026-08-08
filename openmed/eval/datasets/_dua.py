"""Shared helpers for credentialed, eval-only dataset adapters."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .dua_stubs import DUACredentialRequired, require_credentialed_path

_JSON_SUFFIXES = frozenset({".json", ".jsonl", ".ndjson"})
_ROW_CONTAINER_KEYS = (
    "records",
    "documents",
    "examples",
    "fixtures",
    "items",
    "rows",
    "data",
    "train",
    "dev",
    "validation",
    "test",
)
_ROW_KEYS = frozenset(
    {
        "annotations",
        "document",
        "entities",
        "events",
        "gold_label",
        "hypothesis",
        "label",
        "note",
        "premise",
        "relations",
        "sentence1",
        "sentence2",
        "spans",
        "summary",
        "text",
        "tlinks",
    }
)


def source_files(
    root: Path,
    suffixes: set[str] | frozenset[str],
    *,
    dataset: str,
    authority: str,
) -> tuple[Path, ...]:
    """Return deterministic files below an already validated path."""

    normalized_suffixes = {suffix.casefold() for suffix in suffixes}
    if root.is_file():
        files = (root,) if root.suffix.casefold() in normalized_suffixes else tuple()
    else:
        files = tuple(
            path
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.suffix.casefold() in normalized_suffixes
        )
    if not files:
        allowed = ", ".join(sorted(normalized_suffixes))
        raise DUACredentialRequired(
            f"{authority} credentialed {dataset} path contains no supported "
            f"files ({allowed}); no corpus rows were loaded"
        )
    return files


def load_json_rows(
    path: Path,
    *,
    dataset: str,
    authority: str,
) -> list[Mapping[str, Any]]:
    """Load JSON or JSONL rows without writing or caching source content."""

    if path.suffix.casefold() in {".jsonl", ".ndjson"}:
        rows: list[Mapping[str, Any]] = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{dataset} JSONL row {line_number} is invalid: {exc.msg}"
                ) from exc
            rows.extend(_mapping_rows(payload, dataset=dataset))
        return rows

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {dataset} JSON: {exc.msg}") from exc
    rows = _mapping_rows(payload, dataset=dataset)
    if not rows:
        raise DUACredentialRequired(
            f"{authority} {dataset} source contains no rows; no corpus rows were loaded"
        )
    return rows


def fixture_id(dataset: str, source: Path, root: Path, record_id: str) -> str:
    """Build a stable identifier without exposing a source path or row id."""

    relative = _relative_source_path(source, root)
    digest = hashlib.sha256(
        f"{dataset}:{relative}:{record_id}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{dataset}-{digest}"


def source_path_hash(source: Path, root: Path) -> str:
    """Hash source provenance without retaining the credentialed path."""

    return hashlib.sha256(
        _relative_source_path(source, root).encode("utf-8")
    ).hexdigest()


def _mapping_rows(payload: Any, *, dataset: str) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        return [_require_mapping(row, dataset) for row in payload]
    if not isinstance(payload, Mapping):
        raise ValueError(f"{dataset} rows must be JSON objects")

    for key in _ROW_CONTAINER_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            return [_require_mapping(row, dataset) for row in value]
        if isinstance(value, Mapping):
            nested = _mapping_rows(value, dataset=dataset)
            if nested:
                return nested

    if set(payload).intersection(_ROW_KEYS):
        return [payload]

    if payload and all(isinstance(value, Mapping) for value in payload.values()):
        return [{"id": key, **dict(value)} for key, value in payload.items()]
    raise ValueError(
        f"{dataset} JSON must contain a row object or a list under a supported "
        "container key"
    )


def _require_mapping(value: Any, dataset: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{dataset} rows must contain only JSON objects")
    return value


def _relative_source_path(source: Path, root: Path) -> str:
    base = root if root.is_dir() else root.parent
    try:
        return source.relative_to(base).as_posix()
    except ValueError:
        return source.name


__all__ = [
    "fixture_id",
    "load_json_rows",
    "require_credentialed_path",
    "source_files",
    "source_path_hash",
]
