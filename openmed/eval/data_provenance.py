"""PHI-safe dataset provenance and training data manifests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TRAINING_DATA_MANIFEST_SCHEMA_VERSION = "openmed.training_data_manifest.v1"
DATASET_PROVENANCE_SCHEMA_VERSION = "openmed.eval.dataset_provenance.v1"


@dataclass(frozen=True)
class DatasetProvenance:
    """License and content-addressed provenance for an eval dataset source."""

    dataset_id: str
    license_id: str
    source: str
    content_hash: str
    version: str
    split: str
    languages: tuple[str, ...] = ()
    schema_version: str = DATASET_PROVENANCE_SCHEMA_VERSION

    @property
    def source_hash(self) -> str:
        """Return the content hash under the legacy source-hash name."""

        return self.content_hash

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, raw-text-free provenance payload."""

        return {
            "content_hash": self.content_hash,
            "dataset_id": self.dataset_id,
            "languages": list(self.languages),
            "license_id": self.license_id,
            "schema_version": self.schema_version,
            "source": self.source,
            "source_hash": self.source_hash,
            "split": self.split,
            "version": self.version,
        }


def build_dataset_provenance(
    *,
    dataset_id: str,
    license_id: str,
    source: str,
    content_path: str | Path,
    version: str,
    split: str,
    languages: Iterable[str] = (),
) -> DatasetProvenance:
    """Build raw-text-free provenance for a local eval dataset source."""

    return DatasetProvenance(
        dataset_id=_required_string(dataset_id, "dataset_id"),
        license_id=_required_string(license_id, "license_id"),
        source=_required_string(source, "source"),
        content_hash=compute_dataset_content_hash(content_path),
        version=_required_string(version, "version"),
        split=_required_string(split, "split"),
        languages=tuple(
            sorted({_required_string(language, "language") for language in languages})
        ),
    )


def compute_dataset_content_hash(path: str | Path) -> str:
    """Hash a dataset file or directory without retaining its bytes."""

    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"dataset source does not exist: {source_path}")
    if source_path.is_file():
        return _hash_bytes(source_path.read_bytes())

    entries = {
        child.relative_to(source_path).as_posix(): _hash_bytes(child.read_bytes())
        for child in sorted(source_path.rglob("*"))
        if child.is_file()
    }
    return _hash_json({"files": entries})


def build_training_data_manifest(
    fixtures: Iterable[Any],
    *,
    dataset_id: str,
    data_revision: str,
    source: str | None = None,
) -> dict[str, Any]:
    """Build a content-addressed manifest without persisting raw text."""

    entries = sorted(
        (_fixture_manifest_entry(fixture) for fixture in fixtures),
        key=lambda entry: entry["fixture_id"],
    )
    payload: dict[str, Any] = {
        "schema_version": TRAINING_DATA_MANIFEST_SCHEMA_VERSION,
        "data_revision": _required_string(data_revision, "data_revision"),
        "dataset_id": _required_string(dataset_id, "dataset_id"),
        "fixture_count": len(entries),
        "fixtures": entries,
    }
    if source is not None:
        payload["source"] = _required_string(source, "source")
    payload["manifest_hash"] = compute_training_data_manifest_hash(payload)
    return payload


def compute_training_data_manifest_hash(manifest: Mapping[str, Any]) -> str:
    """Return the content hash for a training data manifest."""

    payload = dict(manifest)
    payload.pop("manifest_hash", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def write_training_data_manifest(
    path: str | Path,
    fixtures: Iterable[Any],
    *,
    dataset_id: str,
    data_revision: str,
    source: str | None = None,
) -> Path:
    """Write a PHI-safe training data manifest and return its path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_training_data_manifest(
        fixtures,
        dataset_id=dataset_id,
        data_revision=data_revision,
        source=source,
    )
    output_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path


def _fixture_manifest_entry(fixture: Any) -> dict[str, Any]:
    fixture_id = _fixture_id(fixture)
    text = _fixture_text(fixture)
    spans = tuple(
        _normalise_span(span, source_text=text) for span in _fixture_spans(fixture)
    )
    payload: dict[str, Any] = {
        "fixture_id": fixture_id,
        "span_count": len(spans),
        "spans": list(spans),
        "text_length": len(text),
        "text_sha256": _hash_text(text),
    }
    language = _fixture_language(fixture)
    if language is not None:
        payload["language"] = language

    fixture_hash_payload = dict(payload)
    payload["fixture_hash"] = _hash_json(fixture_hash_payload)
    return payload


def _fixture_id(fixture: Any) -> str:
    if isinstance(fixture, Mapping):
        return _required_string(
            fixture.get("fixture_id") or fixture.get("id"),
            "fixture_id",
        )
    return _required_string(getattr(fixture, "fixture_id", None), "fixture_id")


def _fixture_text(fixture: Any) -> str:
    if isinstance(fixture, Mapping):
        text = fixture.get("text", "")
    else:
        text = getattr(fixture, "text", "")
    return str(text)


def _fixture_language(fixture: Any) -> str | None:
    if isinstance(fixture, Mapping):
        value = fixture.get("language") or fixture.get("lang")
    else:
        value = getattr(fixture, "language", None)
    if value is None:
        return None
    return _required_string(value, "language")


def _fixture_spans(fixture: Any) -> Iterable[Any]:
    if isinstance(fixture, Mapping):
        return (
            fixture.get("gold_spans")
            or fixture.get("spans")
            or fixture.get("entities")
            or []
        )
    return getattr(fixture, "gold_spans", ())


def _normalise_span(span: Any, *, source_text: str) -> dict[str, Any]:
    start = _span_int(span, "start")
    end = _span_int(span, "end")
    if start < 0 or end < start:
        raise ValueError(f"invalid span offsets: {start}:{end}")
    label = _required_string(_span_value(span, "label"), "label")
    span_text = source_text[start:end] if end <= len(source_text) else _span_text(span)
    return {
        "end": end,
        "label": label,
        "length": end - start,
        "start": start,
        "text_sha256": _hash_text(span_text),
    }


def _span_int(span: Any, field: str) -> int:
    value = _span_value(span, field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"span {field} must be an integer")
    return int(value)


def _span_value(span: Any, field: str) -> Any:
    if isinstance(span, Mapping):
        return span.get(field)
    return getattr(span, field, None)


def _span_text(span: Any) -> str:
    value = _span_value(span, "text")
    return "" if value is None else str(value)


def _required_string(value: Any, field: str) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(f"{field} must be a non-empty string")
    return text


def _hash_text(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _hash_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _hash_json(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


__all__ = [
    "DATASET_PROVENANCE_SCHEMA_VERSION",
    "TRAINING_DATA_MANIFEST_SCHEMA_VERSION",
    "DatasetProvenance",
    "build_dataset_provenance",
    "build_training_data_manifest",
    "compute_dataset_content_hash",
    "compute_training_data_manifest_hash",
    "write_training_data_manifest",
]
