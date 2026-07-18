"""Checkpointing primitives for streaming de-identification connectors."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from openmed.__about__ import __version__

CHECKPOINT_SCHEMA_VERSION = 1
DEDUPE_HEADER = "openmed-dedupe-key"

_SENSITIVE_PARAMETER_TOKENS = (
    "key",
    "secret",
    "token",
    "password",
    "text",
    "phi",
)


@dataclass(frozen=True)
class SourcePosition:
    """Broker-agnostic source stream position."""

    topic: str
    partition: str
    offset: int

    def __post_init__(self) -> None:
        if not self.topic.strip():
            raise ValueError("source topic must be non-empty")
        if not self.partition.strip():
            raise ValueError("source partition must be non-empty")
        if self.offset < 0:
            raise ValueError("source offset must be non-negative")

    @property
    def checkpoint_key(self) -> str:
        """Return the stable per-partition key used in checkpoint files."""

        return f"{self.topic}:{self.partition}"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        return {
            "topic": self.topic,
            "partition": self.partition,
            "offset": self.offset,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SourcePosition":
        """Build a source position from a JSON object."""

        return cls(
            topic=str(payload["topic"]),
            partition=str(payload["partition"]),
            offset=int(payload["offset"]),
        )


@dataclass(frozen=True)
class OutputPosition:
    """Broker-agnostic redacted-output stream position."""

    topic: str
    partition: str
    offset: int | str

    def __post_init__(self) -> None:
        if not self.topic.strip():
            raise ValueError("output topic must be non-empty")
        if not self.partition.strip():
            raise ValueError("output partition must be non-empty")
        if isinstance(self.offset, int) and self.offset < 0:
            raise ValueError("output offset must be non-negative")
        if isinstance(self.offset, str) and not self.offset.strip():
            raise ValueError("output offset must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        return {
            "topic": self.topic,
            "partition": self.partition,
            "offset": self.offset,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OutputPosition":
        """Build an output position from a JSON object."""

        offset = payload["offset"]
        return cls(
            topic=str(payload["topic"]),
            partition=str(payload["partition"]),
            offset=int(offset) if isinstance(offset, int) else str(offset),
        )


@dataclass(frozen=True)
class StreamFingerprint:
    """PHI-free policy/model fingerprint pinned to a stream run."""

    policy: str
    model: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-safe representation."""

        return {"policy": self.policy, "model": self.model}


@dataclass(frozen=True)
class CheckpointRecord:
    """Durable checkpoint written after a redacted record is acknowledged."""

    source: SourcePosition
    redacted_output: OutputPosition
    policy_fingerprint: str
    model_fingerprint: str
    dedupe_key: str
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        return {
            "source": self.source.to_dict(),
            "redacted_output": self.redacted_output.to_dict(),
            "policy_fingerprint": self.policy_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "dedupe_key": self.dedupe_key,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CheckpointRecord":
        """Build a checkpoint from a JSON object."""

        return cls(
            source=SourcePosition.from_dict(_mapping(payload["source"], "source")),
            redacted_output=OutputPosition.from_dict(
                _mapping(payload["redacted_output"], "redacted_output")
            ),
            policy_fingerprint=str(payload["policy_fingerprint"]),
            model_fingerprint=str(payload["model_fingerprint"]),
            dedupe_key=str(payload["dedupe_key"]),
            created_at=float(payload["created_at"]),
        )

    @property
    def fingerprint(self) -> StreamFingerprint:
        """Return the policy/model fingerprint pair."""

        return StreamFingerprint(
            policy=self.policy_fingerprint,
            model=self.model_fingerprint,
        )


class CheckpointStore(Protocol):
    """Minimal checkpoint store used by streaming connectors."""

    def load(self, topic: str, partition: str | int) -> CheckpointRecord | None:
        """Return the latest checkpoint for a source topic/partition."""

    def save(self, checkpoint: CheckpointRecord) -> None:
        """Persist ``checkpoint`` atomically after output acknowledgement."""


class InMemoryCheckpointStore:
    """Small checkpoint store useful for tests and embedded runtimes."""

    def __init__(
        self,
        checkpoints: Mapping[str, CheckpointRecord] | None = None,
    ) -> None:
        self._checkpoints = dict(checkpoints or {})

    def load(self, topic: str, partition: str | int) -> CheckpointRecord | None:
        """Return the latest checkpoint for a source topic/partition."""

        return self._checkpoints.get(_checkpoint_key(topic, partition))

    def save(self, checkpoint: CheckpointRecord) -> None:
        """Persist ``checkpoint`` in memory."""

        current = self._checkpoints.get(checkpoint.source.checkpoint_key)
        if current is None or checkpoint.source.offset >= current.source.offset:
            self._checkpoints[checkpoint.source.checkpoint_key] = checkpoint

    def all(self) -> dict[str, CheckpointRecord]:
        """Return all stored checkpoints keyed by source topic/partition."""

        return dict(self._checkpoints)


class LocalFileCheckpointStore:
    """Atomic JSON checkpoint store for local streaming jobs."""

    def __init__(self, path: str | Path, *, fsync: bool = True) -> None:
        self.path = Path(path)
        self.fsync = fsync

    def load(self, topic: str, partition: str | int) -> CheckpointRecord | None:
        """Return the latest checkpoint for a source topic/partition."""

        return self._read_records().get(_checkpoint_key(topic, partition))

    def save(self, checkpoint: CheckpointRecord) -> None:
        """Persist ``checkpoint`` with temp-file plus ``os.replace`` semantics."""

        records = self._read_records()
        current = records.get(checkpoint.source.checkpoint_key)
        if current is not None and current.source.offset > checkpoint.source.offset:
            return
        records[checkpoint.source.checkpoint_key] = checkpoint
        self._write_records(records)

    def _read_records(self) -> dict[str, CheckpointRecord]:
        if not self.path.exists():
            return {}
        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError("checkpoint file must contain a JSON object")
        version = int(payload.get("schema_version", 0))
        if version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                "unsupported checkpoint schema version "
                f"{version}; expected {CHECKPOINT_SCHEMA_VERSION}"
            )
        raw_records = _mapping(payload.get("checkpoints", {}), "checkpoints")
        return {
            str(key): CheckpointRecord.from_dict(_mapping(value, str(key)))
            for key, value in raw_records.items()
        }

    def _write_records(self, records: Mapping[str, CheckpointRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoints": {
                key: checkpoint.to_dict() for key, checkpoint in sorted(records.items())
            },
        }
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=str(self.path.parent),
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                if self.fsync:
                    os.fsync(handle.fileno())
            os.replace(tmp_name, self.path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            raise


def dedupe_key_for_source(source: SourcePosition) -> str:
    """Return a stable dedupe key derived only from source position metadata."""

    digest = hashlib.sha256()
    for value in (
        "openmed.stream.dedupe.v1",
        source.topic,
        source.partition,
        source.offset,
    ):
        encoded = str(value).encode("utf-8")
        digest.update(str(len(encoded)).encode("ascii"))
        digest.update(b":")
        digest.update(encoded)
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def build_stream_fingerprint(
    *,
    policy_name: str,
    deidentify_kwargs: Mapping[str, Any] | None = None,
) -> StreamFingerprint:
    """Build PHI-free policy and model fingerprints for a stream run."""

    options = dict(deidentify_kwargs or {})
    model_name = str(options.get("model_name") or "openmed-default")
    policy_fingerprint = _sha256_json(
        {
            "schema": "openmed.stream.policy.v1",
            "policy": policy_name,
            "openmed_version": __version__,
        }
    )
    model_fingerprint = _sha256_json(
        {
            "schema": "openmed.stream.model.v1",
            "model_name": model_name,
            "parameters": _safe_parameter_identity(options),
        }
    )
    return StreamFingerprint(policy=policy_fingerprint, model=model_fingerprint)


def checkpoint_for_delivery(
    *,
    source: SourcePosition,
    redacted_output: OutputPosition,
    fingerprint: StreamFingerprint,
    dedupe_key: str,
    created_at: float | None = None,
) -> CheckpointRecord:
    """Create the checkpoint record for an acknowledged redacted output."""

    return CheckpointRecord(
        source=source,
        redacted_output=redacted_output,
        policy_fingerprint=fingerprint.policy,
        model_fingerprint=fingerprint.model,
        dedupe_key=dedupe_key,
        created_at=time.time() if created_at is None else created_at,
    )


def _checkpoint_key(topic: str, partition: str | int) -> str:
    return f"{topic}:{partition}"


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_parameter_identity(value: Any, *, key: str | None = None) -> Any:
    if key is not None and _is_sensitive_key(key):
        return _hashed_sensitive_identity(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {
            str(item_key): _safe_parameter_identity(item_value, key=str(item_key))
            for item_key, item_value in sorted(
                value.items(),
                key=lambda item: str(item[0]),
            )
        }
    if isinstance(value, (list, tuple)):
        return [_safe_parameter_identity(item) for item in value]
    if isinstance(value, set):
        return sorted(_safe_parameter_identity(item) for item in value)
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _hashed_sensitive_identity(value: Any) -> dict[str, str]:
    digest = hashlib.sha256(repr(value).encode("utf-8")).hexdigest()
    return {"sha256": digest, "type": type(value).__name__}


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(token in lowered for token in _SENSITIVE_PARAMETER_TOKENS)


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return value


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "DEDUPE_HEADER",
    "CheckpointRecord",
    "CheckpointStore",
    "InMemoryCheckpointStore",
    "LocalFileCheckpointStore",
    "OutputPosition",
    "SourcePosition",
    "StreamFingerprint",
    "build_stream_fingerprint",
    "checkpoint_for_delivery",
    "dedupe_key_for_source",
]
