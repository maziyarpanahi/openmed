"""Canonical span record and schema-version helpers."""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Mapping, Sequence

from openmed.core.labels import CANONICAL_LABELS, policy_label_for

CURRENT_SCHEMA_VERSION = 1
SCHEMA_PACKAGE = "openmed.core.schemas.json"
SCHEMA_NAMES = (
    "span",
    "action",
    "audit",
    "attestation",
    "entity",
    "relation",
    "code",
    "provenance",
)
ACTION_KEEP = "keep"
ACTION_FORMAT_PRESERVE = "format_preserve"
ACTION_VALUES = (
    ACTION_KEEP,
    "redact",
    "replace",
    "mask",
    "hash",
    ACTION_FORMAT_PRESERVE,
)
_TEXT_HASH_RE = re.compile(r"^hmac-sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class OpenMedSpan:
    """Immutable span contract shared across OpenMed privacy layers."""

    doc_id: str
    start: int
    end: int
    text_hash: str
    entity_type: str
    canonical_label: str
    policy_label: str | None = None
    regulatory_tags: Sequence[str] = field(default_factory=tuple)
    score: float | None = None
    detector: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)
    action: str = ACTION_KEEP
    replacement: str | None = None
    reversible_id: str | None = None
    section: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = CURRENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {CURRENT_SCHEMA_VERSION}, "
                f"got {self.schema_version!r}"
            )
        if not self.doc_id:
            raise ValueError("doc_id must be non-empty")
        if not isinstance(self.start, int) or not isinstance(self.end, int):
            raise TypeError("start and end must be integers")
        if self.start < 0 or self.end < self.start:
            raise ValueError("start/end must be non-negative offsets with end >= start")
        if not _TEXT_HASH_RE.fullmatch(self.text_hash):
            raise ValueError("text_hash must be hmac-sha256:<64 lowercase hex chars>")
        if self.canonical_label not in CANONICAL_LABELS:
            raise ValueError(
                f"canonical_label must be in CANONICAL_LABELS: {self.canonical_label!r}"
            )
        if not self.entity_type:
            raise ValueError("entity_type must be non-empty")
        if self.action not in ACTION_VALUES:
            raise ValueError(f"action must be one of {ACTION_VALUES!r}")
        if self.score is not None and not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("score must be between 0.0 and 1.0")

        object.__setattr__(
            self, "regulatory_tags", tuple(map(str, self.regulatory_tags))
        )
        object.__setattr__(self, "evidence", _plain_mapping(self.evidence))
        object.__setattr__(self, "metadata", _plain_mapping(self.metadata))
        if self.policy_label is None:
            object.__setattr__(
                self, "policy_label", policy_label_for(self.canonical_label)
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this span to a JSON-compatible dictionary."""

        return {
            "schema_version": self.schema_version,
            "doc_id": self.doc_id,
            "start": self.start,
            "end": self.end,
            "text_hash": self.text_hash,
            "entity_type": self.entity_type,
            "canonical_label": self.canonical_label,
            "policy_label": self.policy_label,
            "regulatory_tags": list(self.regulatory_tags),
            "score": self.score,
            "detector": self.detector,
            "evidence": copy.deepcopy(dict(self.evidence)),
            "action": self.action,
            "replacement": self.replacement,
            "reversible_id": self.reversible_id,
            "section": self.section,
            "metadata": copy.deepcopy(dict(self.metadata)),
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize this span to deterministic JSON."""

        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OpenMedSpan":
        """Deserialize a span from a JSON-compatible dictionary."""

        data = dict(payload)
        return cls(
            doc_id=str(data["doc_id"]),
            start=int(data["start"]),
            end=int(data["end"]),
            text_hash=str(data["text_hash"]),
            entity_type=str(data["entity_type"]),
            canonical_label=str(data["canonical_label"]),
            policy_label=data.get("policy_label"),
            regulatory_tags=tuple(data.get("regulatory_tags") or ()),
            score=data.get("score"),
            detector=data.get("detector"),
            evidence=data.get("evidence") or {},
            action=str(data.get("action") or ACTION_KEEP),
            replacement=data.get("replacement"),
            reversible_id=data.get("reversible_id"),
            section=data.get("section"),
            metadata=data.get("metadata") or {},
            schema_version=int(data.get("schema_version", CURRENT_SCHEMA_VERSION)),
        )

    @classmethod
    def from_json(cls, text: str) -> "OpenMedSpan":
        """Deserialize a span from JSON text."""

        return cls.from_dict(json.loads(text))


@dataclass(frozen=True)
class SchemaDriftResult:
    """Result of comparing one schema against a saved fingerprint snapshot."""

    schema_name: str
    schema_version: int
    snapshot_version: int | None
    fingerprint: str
    snapshot_fingerprint: str | None
    version_bumped: bool
    breaking_change: bool
    removed_required: tuple[str, ...] = ()
    removed_properties: tuple[str, ...] = ()


def hmac_text_hash(surface: str | bytes, secret: str | bytes) -> str:
    """Return the safe loggable HMAC-SHA256 hash for source surface text."""

    payload = surface.encode("utf-8") if isinstance(surface, str) else surface
    key = secret.encode("utf-8") if isinstance(secret, str) else secret
    digest = hmac.new(key, payload, hashlib.sha256).hexdigest()
    return f"hmac-sha256:{digest}"


def current_schema_version() -> int:
    """Return the current schema version."""

    return CURRENT_SCHEMA_VERSION


def load_schema(name: str) -> dict[str, Any]:
    """Load one bundled JSON Schema by logical name."""

    schema_name = _normalise_schema_name(name)
    resource = resources.files(SCHEMA_PACKAGE).joinpath(f"{schema_name}.schema.json")
    with resource.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    if "schema_version" not in schema:
        raise ValueError(f"{schema_name} schema is missing schema_version")
    return schema


def load_all_schemas() -> dict[str, dict[str, Any]]:
    """Load every bundled JSON Schema."""

    return {name: load_schema(name) for name in SCHEMA_NAMES}


def load_schema_bundle() -> dict[str, Any]:
    """Return current schema_version plus parsed bundled schemas."""

    return {
        "schema_version": current_schema_version(),
        "schemas": load_all_schemas(),
    }


def schema_fingerprint(schema: Mapping[str, Any]) -> str:
    """Return a deterministic SHA256 fingerprint for a JSON Schema."""

    payload = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def build_schema_snapshot(
    schemas: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build the drift snapshot format from parsed schemas."""

    return {
        name: _schema_state(schema)
        for name, schema in sorted(schemas.items(), key=lambda item: item[0])
    }


def load_schema_snapshot(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Load the committed schema fingerprint snapshot."""

    if path is None:
        resource = resources.files(SCHEMA_PACKAGE).joinpath("schema-fingerprints.json")
        with resource.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare_schema_drift(
    schema_name: str,
    current_schema: Mapping[str, Any] | None = None,
    *,
    snapshot: Mapping[str, Mapping[str, Any]] | None = None,
) -> SchemaDriftResult:
    """Compare one schema against a saved snapshot.

    A field removal without a schema_version bump is reported as a breaking
    change. Gate wiring that turns this result into CI failure is owned by a
    later task.
    """

    name = _normalise_schema_name(schema_name)
    schema = dict(current_schema or load_schema(name))
    state = _schema_state(schema)
    snapshot_state = dict((snapshot or load_schema_snapshot()).get(name) or {})
    snapshot_version = snapshot_state.get("schema_version")
    version_bumped = (
        snapshot_version is not None and state["schema_version"] != snapshot_version
    )
    removed_required = tuple(
        sorted(set(snapshot_state.get("required", ())) - set(state["required"]))
    )
    removed_properties = tuple(
        sorted(set(snapshot_state.get("properties", ())) - set(state["properties"]))
    )
    breaking_change = bool(
        not version_bumped
        and (removed_required or removed_properties)
        and snapshot_state
    )
    return SchemaDriftResult(
        schema_name=name,
        schema_version=int(state["schema_version"]),
        snapshot_version=int(snapshot_version)
        if snapshot_version is not None
        else None,
        fingerprint=str(state["fingerprint"]),
        snapshot_fingerprint=snapshot_state.get("fingerprint"),
        version_bumped=version_bumped,
        breaking_change=breaking_change,
        removed_required=removed_required,
        removed_properties=removed_properties,
    )


def compare_all_schema_drift(
    *,
    schemas: Mapping[str, Mapping[str, Any]] | None = None,
    snapshot: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, SchemaDriftResult]:
    """Compare all current schemas against the committed snapshot."""

    current = schemas or load_all_schemas()
    saved = snapshot or load_schema_snapshot()
    return {
        name: compare_schema_drift(name, schema, snapshot=saved)
        for name, schema in current.items()
    }


def _plain_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value), sort_keys=True))


def _normalise_schema_name(name: str) -> str:
    schema_name = name.removesuffix(".schema.json").removesuffix(".schema")
    if schema_name not in SCHEMA_NAMES:
        raise KeyError(f"unknown schema: {name!r}")
    return schema_name


def _schema_state(schema: Mapping[str, Any]) -> dict[str, Any]:
    properties = schema.get("properties") or {}
    return {
        "schema_version": schema.get("schema_version"),
        "fingerprint": schema_fingerprint(schema),
        "required": sorted(str(item) for item in schema.get("required", ())),
        "properties": sorted(str(key) for key in properties),
    }


__all__ = [
    "ACTION_KEEP",
    "ACTION_FORMAT_PRESERVE",
    "ACTION_VALUES",
    "CURRENT_SCHEMA_VERSION",
    "SCHEMA_NAMES",
    "OpenMedSpan",
    "SchemaDriftResult",
    "build_schema_snapshot",
    "compare_all_schema_drift",
    "compare_schema_drift",
    "current_schema_version",
    "hmac_text_hash",
    "load_all_schemas",
    "load_schema",
    "load_schema_bundle",
    "load_schema_snapshot",
    "schema_fingerprint",
]
