"""Versioned, privacy-safe annotation interchange contracts.

The canonical tabular representation is UTF-8 TSV.  It carries offsets and
controlled results, but never source text.  Importers are deliberately strict:
unknown columns, duplicate JSON keys, non-finite numbers, and oversized input
are rejected instead of being silently normalized.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from importlib import resources
from types import MappingProxyType
from typing import Any, Final

from openmed.clinical.journey_contracts import (
    canonical_digest,
    canonical_json,
    derived_opaque_id,
)

ANNOTATION_INTERCHANGE_SCHEMA_VERSION: Final = "1.0.0"
ANNOTATION_INTERCHANGE_COMPATIBILITY: Final = "same_major"
ANNOTATION_INTERCHANGE_SCHEMA_NAMES: Final = (
    "record",
    "envelope",
    "loss_report",
    "page",
)
MAX_ANNOTATION_ROWS: Final = 10_000
MAX_ANNOTATION_INPUT_BYTES: Final = 8 * 1024 * 1024
MAX_ANNOTATION_CELL_CHARS: Final = 65_536
MAX_ANNOTATION_EMBEDDING_DIMENSIONS: Final = 4_096
MAX_ANNOTATION_PAGE_SIZE: Final = 100

_CONTROLLED_RE = re.compile(r"^[a-z][a-z0-9_.:/-]{0,127}$")
_MODEL_VERSION_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
_OPAQUE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}_[A-Za-z0-9_-]{8,128}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_CURSOR_RE = re.compile(r"^[A-Za-z0-9_-]{16,2048}$")
_SENSITIVE_KEYS = frozenset(
    {
        "credential",
        "note",
        "payload",
        "phi",
        "raw",
        "raw_text",
        "secret",
        "source_text",
        "text",
        "token",
    }
)
_TSV_COLUMNS = (
    "schema_version",
    "compatibility_policy",
    "annotation_id",
    "document_id",
    "namespace",
    "annotation_type",
    "coordinate_convention",
    "start",
    "end",
    "state",
    "result_json",
    "metadata_json",
    "embedding_json",
)


class AnnotationInterchangeError(ValueError):
    """Raised when an annotation interchange contract is invalid."""


class AnnotationType(str, Enum):
    """Supported annotation families."""

    ENTITY = "entity"
    RELATION = "relation"
    FACT_CORRECTION = "fact_correction"
    REGISTRY_LABEL = "registry_label"


class CoordinateConvention(str, Enum):
    """Coordinate systems accepted by the tabular envelope."""

    UNICODE_CODEPOINT = "unicode_codepoint"
    UTF8_BYTE = "utf8_byte"
    UTF16_CODE_UNIT = "utf16_code_unit"
    TOKEN_INDEX = "token_index"
    NONE = "none"


class AnnotationState(str, Enum):
    """Non-lossy outcome states shared by interchange operations."""

    SUCCESS = "success"
    PARTIAL = "partial"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"
    UNSUPPORTED = "unsupported"
    DENIED = "denied"
    FAILURE = "failure"


class AnnotationLossKind(str, Enum):
    """Closed reasons for declared interchange loss or review."""

    EMBEDDING_OMITTED = "embedding_omitted"
    METADATA_OMITTED = "metadata_omitted"
    UNSUPPORTED_ANNOTATION = "unsupported_annotation"
    OFFSET_NOT_BOUNDARY = "offset_not_boundary"
    TOKEN_MAP_REQUIRED = "token_map_required"
    REVIEW_REQUIRED = "review_required"


_RESULT_FIELDS: Final[Mapping[AnnotationType, frozenset[str]]] = MappingProxyType(
    {
        AnnotationType.ENTITY: frozenset({"label", "surface_hash"}),
        AnnotationType.RELATION: frozenset(
            {"relation", "source_annotation_id", "target_annotation_id"}
        ),
        AnnotationType.FACT_CORRECTION: frozenset(
            {"evidence_ids", "fact_id", "field", "reason_code", "replacement_code"}
        ),
        AnnotationType.REGISTRY_LABEL: frozenset(
            {"evidence_ids", "label", "record_id", "registry_id"}
        ),
    }
)
_REQUIRED_RESULT_FIELDS: Final[Mapping[AnnotationType, frozenset[str]]] = (
    MappingProxyType(
        {
            AnnotationType.ENTITY: frozenset({"label", "surface_hash"}),
            AnnotationType.RELATION: frozenset(
                {"relation", "source_annotation_id", "target_annotation_id"}
            ),
            AnnotationType.FACT_CORRECTION: frozenset(
                {"fact_id", "field", "reason_code", "replacement_code"}
            ),
            AnnotationType.REGISTRY_LABEL: frozenset(
                {"label", "record_id", "registry_id"}
            ),
        }
    )
)
_METADATA_FIELDS = frozenset(
    {
        "annotator_id",
        "batch_id",
        "model_id",
        "model_version",
        "review_state",
        "source_format",
    }
)


@dataclass(frozen=True, slots=True)
class AnnotationRecord:
    """One source-text-free annotation record."""

    annotation_id: str
    document_id: str
    namespace: str
    annotation_type: AnnotationType
    coordinate_convention: CoordinateConvention
    start: int | None
    end: int | None
    state: AnnotationState
    result: Mapping[str, Any] = field(repr=False)
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)
    embedding: tuple[float, ...] | None = field(default=None, repr=False)
    schema_version: str = ANNOTATION_INTERCHANGE_SCHEMA_VERSION
    compatibility_policy: str = ANNOTATION_INTERCHANGE_COMPATIBILITY

    def __post_init__(self) -> None:
        annotation_type = AnnotationType(self.annotation_type)
        convention = CoordinateConvention(self.coordinate_convention)
        state = AnnotationState(self.state)
        _require_schema(self.schema_version, self.compatibility_policy)
        _opaque(self.annotation_id, "annotation_id")
        _opaque(self.document_id, "document_id")
        _controlled(self.namespace, "namespace")
        _validate_offsets(annotation_type, convention, self.start, self.end)
        result = _validate_result(annotation_type, self.result)
        metadata = _validate_metadata(self.metadata)
        embedding = _validate_embedding(self.embedding)
        if state is not AnnotationState.SUCCESS and embedding is not None:
            raise AnnotationInterchangeError(
                "non-success annotations cannot carry embeddings"
            )
        object.__setattr__(self, "annotation_type", annotation_type)
        object.__setattr__(self, "coordinate_convention", convention)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "result", MappingProxyType(result))
        object.__setattr__(self, "metadata", MappingProxyType(metadata))
        object.__setattr__(self, "embedding", embedding)

    @property
    def record_digest(self) -> str:
        """Return a stable digest over the complete record."""

        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible representation."""

        return {
            "annotation_id": self.annotation_id,
            "annotation_type": self.annotation_type.value,
            "compatibility_policy": self.compatibility_policy,
            "coordinate_convention": self.coordinate_convention.value,
            "document_id": self.document_id,
            "embedding": list(self.embedding) if self.embedding is not None else None,
            "end": self.end,
            "metadata": _plain(self.metadata),
            "namespace": self.namespace,
            "result": _plain(self.result),
            "schema_version": self.schema_version,
            "start": self.start,
            "state": self.state.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AnnotationRecord":
        """Parse one record while rejecting missing and unknown fields."""

        fields = {
            "annotation_id",
            "annotation_type",
            "compatibility_policy",
            "coordinate_convention",
            "document_id",
            "embedding",
            "end",
            "metadata",
            "namespace",
            "result",
            "schema_version",
            "start",
            "state",
        }
        data = _strict_mapping(payload, fields, "annotation record")
        raw_embedding = data["embedding"]
        if raw_embedding is not None and not isinstance(raw_embedding, list):
            raise AnnotationInterchangeError("embedding must be an array or null")
        return cls(
            annotation_id=data["annotation_id"],
            document_id=data["document_id"],
            namespace=data["namespace"],
            annotation_type=AnnotationType(data["annotation_type"]),
            coordinate_convention=CoordinateConvention(data["coordinate_convention"]),
            start=data["start"],
            end=data["end"],
            state=AnnotationState(data["state"]),
            result=data["result"],
            metadata=data["metadata"],
            embedding=(tuple(raw_embedding) if raw_embedding is not None else None),
            schema_version=data["schema_version"],
            compatibility_policy=data["compatibility_policy"],
        )


@dataclass(frozen=True, slots=True)
class AnnotationEnvelope:
    """A deterministic, bounded collection of annotation records."""

    envelope_id: str
    records: tuple[AnnotationRecord, ...]
    schema_version: str = ANNOTATION_INTERCHANGE_SCHEMA_VERSION
    compatibility_policy: str = ANNOTATION_INTERCHANGE_COMPATIBILITY

    def __post_init__(self) -> None:
        _require_schema(self.schema_version, self.compatibility_policy)
        _opaque(self.envelope_id, "envelope_id")
        if not isinstance(self.records, tuple):
            raise AnnotationInterchangeError("records must be a tuple")
        if len(self.records) > MAX_ANNOTATION_ROWS:
            raise AnnotationInterchangeError("annotation row limit exceeded")
        if any(not isinstance(item, AnnotationRecord) for item in self.records):
            raise AnnotationInterchangeError("records must contain annotations")
        ordered = tuple(
            sorted(
                self.records,
                key=lambda item: (
                    item.document_id,
                    item.start if item.start is not None else -1,
                    item.end if item.end is not None else -1,
                    item.annotation_id,
                ),
            )
        )
        identities = [item.annotation_id for item in ordered]
        if len(identities) != len(set(identities)):
            raise AnnotationInterchangeError("annotation IDs must be unique")
        object.__setattr__(self, "records", ordered)

    @property
    def envelope_digest(self) -> str:
        """Return a stable digest over all records."""

        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible representation."""

        return {
            "compatibility_policy": self.compatibility_policy,
            "envelope_id": self.envelope_id,
            "records": [item.to_dict() for item in self.records],
            "schema_version": self.schema_version,
        }

    def to_json(self) -> str:
        """Return canonical JSON for persisted interchange."""

        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AnnotationEnvelope":
        """Parse one strict envelope."""

        data = _strict_mapping(
            payload,
            {"compatibility_policy", "envelope_id", "records", "schema_version"},
            "annotation envelope",
        )
        if not isinstance(data["records"], list):
            raise AnnotationInterchangeError("records must be an array")
        return cls(
            envelope_id=data["envelope_id"],
            records=tuple(AnnotationRecord.from_dict(item) for item in data["records"]),
            schema_version=data["schema_version"],
            compatibility_policy=data["compatibility_policy"],
        )

    @classmethod
    def from_json(cls, payload: str | bytes) -> "AnnotationEnvelope":
        """Parse strict, bounded JSON for persisted interchange."""

        if isinstance(payload, bytes):
            if len(payload) > MAX_ANNOTATION_INPUT_BYTES:
                raise AnnotationInterchangeError("annotation input byte limit exceeded")
            try:
                text = payload.decode("utf-8")
            except UnicodeDecodeError:
                raise AnnotationInterchangeError(
                    "annotation input must be UTF-8"
                ) from None
        elif isinstance(payload, str):
            if len(payload.encode("utf-8")) > MAX_ANNOTATION_INPUT_BYTES:
                raise AnnotationInterchangeError("annotation input byte limit exceeded")
            text = payload
        else:
            raise AnnotationInterchangeError("annotation input must be text or bytes")
        parsed = _strict_json(text, "annotation envelope")
        if not isinstance(parsed, Mapping):
            raise AnnotationInterchangeError("annotation envelope must be an object")
        return cls.from_dict(parsed)


@dataclass(frozen=True, slots=True)
class AnnotationLoss:
    """One declared information-loss or manual-review condition."""

    annotation_id: str
    kind: AnnotationLossKind
    lossy: bool
    code: str

    def __post_init__(self) -> None:
        _opaque(self.annotation_id, "annotation_id")
        object.__setattr__(self, "kind", AnnotationLossKind(self.kind))
        if type(self.lossy) is not bool:
            raise AnnotationInterchangeError("lossy must be boolean")
        _controlled(self.code, "loss code")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic representation."""

        return {
            "annotation_id": self.annotation_id,
            "code": self.code,
            "kind": self.kind.value,
            "lossy": self.lossy,
        }


@dataclass(frozen=True, slots=True)
class AnnotationLossReport:
    """Deterministic report for a tabular import, export, or conversion."""

    operation: str
    state: AnnotationState
    input_digest: str
    output_digest: str
    entries: tuple[AnnotationLoss, ...] = ()
    schema_version: str = ANNOTATION_INTERCHANGE_SCHEMA_VERSION
    compatibility_policy: str = ANNOTATION_INTERCHANGE_COMPATIBILITY

    def __post_init__(self) -> None:
        _require_schema(self.schema_version, self.compatibility_policy)
        _controlled(self.operation, "operation")
        object.__setattr__(self, "state", AnnotationState(self.state))
        _digest(self.input_digest, "input_digest")
        _digest(self.output_digest, "output_digest")
        if not isinstance(self.entries, tuple):
            raise AnnotationInterchangeError("loss entries must be a tuple")
        if any(not isinstance(item, AnnotationLoss) for item in self.entries):
            raise AnnotationInterchangeError("loss entries are invalid")
        ordered = tuple(
            sorted(self.entries, key=lambda item: (item.annotation_id, item.kind.value))
        )
        expected = (
            AnnotationState.PARTIAL
            if any(item.lossy for item in ordered)
            else AnnotationState.SUCCESS
        )
        if self.state is not expected:
            raise AnnotationInterchangeError("loss report state is inconsistent")
        object.__setattr__(self, "entries", ordered)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic representation."""

        return {
            "compatibility_policy": self.compatibility_policy,
            "entries": [item.to_dict() for item in self.entries],
            "input_digest": self.input_digest,
            "operation": self.operation,
            "output_digest": self.output_digest,
            "schema_version": self.schema_version,
            "state": self.state.value,
        }


@dataclass(frozen=True, slots=True)
class AnnotationExport:
    """Canonical TSV plus its declared loss report."""

    text: str = field(repr=False)
    report: AnnotationLossReport


@dataclass(frozen=True, slots=True)
class AnnotationQuery:
    """Bounded, policy-aware query over one immutable annotation envelope."""

    namespace: str = "default"
    purpose: str = "quality"
    role: str = "data_steward"
    consent_state: str = "active"
    export_policy: str = "metadata_only"
    annotation_types: tuple[AnnotationType, ...] = ()
    first: int = 20
    after: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "namespace",
            "purpose",
            "role",
            "consent_state",
            "export_policy",
        ):
            _controlled(getattr(self, name), name)
        if (
            type(self.first) is not int
            or not 1 <= self.first <= MAX_ANNOTATION_PAGE_SIZE
        ):
            raise AnnotationInterchangeError("first exceeds the annotation page limit")
        types = tuple(
            sorted(
                {AnnotationType(item) for item in self.annotation_types},
                key=lambda item: item.value,
            )
        )
        if self.after is not None and _CURSOR_RE.fullmatch(self.after) is None:
            raise AnnotationInterchangeError("annotation cursor is invalid")
        object.__setattr__(self, "annotation_types", types)


@dataclass(frozen=True, slots=True)
class AnnotationAccessPolicy:
    """Minimal access policy for annotation catalog reads."""

    allowed_namespaces: frozenset[str] = frozenset({"default"})
    allowed_purposes: frozenset[str] = frozenset({"quality", "training_review"})
    allowed_roles: frozenset[str] = frozenset({"data_steward", "privacy_officer"})
    allowed_export_policies: frozenset[str] = frozenset({"metadata_only"})

    def __post_init__(self) -> None:
        for name in (
            "allowed_namespaces",
            "allowed_purposes",
            "allowed_roles",
            "allowed_export_policies",
        ):
            raw_values = getattr(self, name)
            if isinstance(raw_values, (str, bytes)):
                raise AnnotationInterchangeError(f"{name} must be a collection")
            values = frozenset(raw_values)
            if not values:
                raise AnnotationInterchangeError(f"{name} cannot be empty")
            for value in values:
                _controlled(value, name)
            object.__setattr__(self, name, values)

    def allows(self, query: AnnotationQuery) -> tuple[bool, str | None]:
        """Return a deterministic allow/deny decision code."""

        checks = (
            (query.namespace in self.allowed_namespaces, "namespace_denied"),
            (query.purpose in self.allowed_purposes, "purpose_denied"),
            (query.role in self.allowed_roles, "role_denied"),
            (query.consent_state == "active", "consent_denied"),
            (
                query.export_policy in self.allowed_export_policies,
                "export_policy_denied",
            ),
        )
        for allowed, code in checks:
            if not allowed:
                return False, code
        return True, None


@dataclass(frozen=True, slots=True)
class AnnotationPage:
    """One bounded annotation page with a cursor bound to query and snapshot."""

    state: AnnotationState
    code: str | None
    records: tuple[AnnotationRecord, ...]
    next_cursor: str | None
    snapshot_digest: str
    schema_version: str = ANNOTATION_INTERCHANGE_SCHEMA_VERSION
    compatibility_policy: str = ANNOTATION_INTERCHANGE_COMPATIBILITY

    def __post_init__(self) -> None:
        _require_schema(self.schema_version, self.compatibility_policy)
        object.__setattr__(self, "state", AnnotationState(self.state))
        if self.code is not None:
            _controlled(self.code, "page code")
        _digest(self.snapshot_digest, "snapshot_digest")
        if not isinstance(self.records, tuple):
            raise AnnotationInterchangeError("page records must be a tuple")
        if len(self.records) > MAX_ANNOTATION_PAGE_SIZE:
            raise AnnotationInterchangeError("annotation page limit exceeded")
        if any(not isinstance(item, AnnotationRecord) for item in self.records):
            raise AnnotationInterchangeError("annotation page records are invalid")
        if (
            self.next_cursor is not None
            and _CURSOR_RE.fullmatch(self.next_cursor) is None
        ):
            raise AnnotationInterchangeError("next cursor is invalid")
        if (
            self.state in {AnnotationState.DENIED, AnnotationState.FAILURE}
            and self.records
        ):
            raise AnnotationInterchangeError("terminal pages cannot carry records")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic representation."""

        return {
            "code": self.code,
            "compatibility_policy": self.compatibility_policy,
            "next_cursor": self.next_cursor,
            "records": [item.to_dict() for item in self.records],
            "schema_version": self.schema_version,
            "snapshot_digest": self.snapshot_digest,
            "state": self.state.value,
        }


class AnnotationCatalog:
    """Read-only catalog with bounded pagination and fail-closed policy checks."""

    def __init__(self, envelope: AnnotationEnvelope) -> None:
        self._envelope = envelope

    def list(
        self,
        query: AnnotationQuery,
        *,
        policy: AnnotationAccessPolicy | None = None,
    ) -> AnnotationPage:
        """Return one policy-authorized page."""

        allowed, code = (policy or AnnotationAccessPolicy()).allows(query)
        if not allowed:
            return self._terminal(AnnotationState.DENIED, code or "policy_denied")
        try:
            offset = self._decode_cursor(query)
        except AnnotationInterchangeError as exc:
            return self._terminal(AnnotationState.FAILURE, str(exc))
        matches = tuple(
            item
            for item in self._envelope.records
            if item.namespace == query.namespace
            and (
                not query.annotation_types
                or item.annotation_type in query.annotation_types
            )
        )
        if offset > len(matches):
            return self._terminal(AnnotationState.FAILURE, "cursor_offset_invalid")
        selected = matches[offset : offset + query.first]
        if not selected:
            return self._terminal(AnnotationState.UNKNOWN, "no_annotations")
        next_offset = offset + len(selected)
        return AnnotationPage(
            state=_aggregate_state(selected),
            code=None,
            records=selected,
            next_cursor=(
                self._cursor(query, next_offset) if next_offset < len(matches) else None
            ),
            snapshot_digest=self._envelope.envelope_digest,
        )

    def _terminal(self, state: AnnotationState, code: str) -> AnnotationPage:
        return AnnotationPage(
            state=state,
            code=code,
            records=(),
            next_cursor=None,
            snapshot_digest=self._envelope.envelope_digest,
        )

    def _cursor_body(self, query: AnnotationQuery, offset: int) -> dict[str, Any]:
        return {
            "annotation_types": [item.value for item in query.annotation_types],
            "consent_state": query.consent_state,
            "export_policy": query.export_policy,
            "namespace": query.namespace,
            "offset": offset,
            "purpose": query.purpose,
            "role": query.role,
            "snapshot_digest": self._envelope.envelope_digest,
            "version": 1,
        }

    def _cursor(self, query: AnnotationQuery, offset: int) -> str:
        body = self._cursor_body(query, offset)
        body["digest"] = canonical_digest(body)
        return (
            base64.urlsafe_b64encode(canonical_json(body).encode("ascii"))
            .decode("ascii")
            .rstrip("=")
        )

    def _decode_cursor(self, query: AnnotationQuery) -> int:
        if query.after is None:
            return 0
        try:
            padding = "=" * (-len(query.after) % 4)
            value = json.loads(
                base64.urlsafe_b64decode(query.after + padding).decode("ascii"),
                object_pairs_hook=_reject_duplicates,
                parse_constant=_reject_constant,
            )
        except (UnicodeError, ValueError, json.JSONDecodeError):
            raise AnnotationInterchangeError("cursor_invalid") from None
        if not isinstance(value, dict):
            raise AnnotationInterchangeError("cursor_invalid")
        digest = value.pop("digest", None)
        if digest != canonical_digest(value):
            raise AnnotationInterchangeError("cursor_integrity_failed")
        offset = value.get("offset")
        if type(offset) is not int or offset < 0:
            raise AnnotationInterchangeError("cursor_offset_invalid")
        expected = self._cursor_body(query, offset)
        if value != expected:
            raise AnnotationInterchangeError("cursor_query_mismatch")
        return offset


def build_annotation_envelope(
    records: Iterable[AnnotationRecord],
) -> AnnotationEnvelope:
    """Build a content-addressed annotation envelope."""

    materialized = tuple(records)
    identity = canonical_digest([item.record_digest for item in materialized])
    return AnnotationEnvelope(
        envelope_id=derived_opaque_id("annotation_envelope", identity),
        records=materialized,
    )


def export_annotation_tsv(
    envelope: AnnotationEnvelope,
    *,
    include_embeddings: bool = True,
) -> AnnotationExport:
    """Serialize an envelope to canonical TSV and declare any omitted data."""

    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=list(_TSV_COLUMNS),
        delimiter="\t",
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    losses: list[AnnotationLoss] = []
    for record in envelope.records:
        embedding = record.embedding if include_embeddings else None
        if record.embedding is not None and not include_embeddings:
            losses.append(
                AnnotationLoss(
                    annotation_id=record.annotation_id,
                    kind=AnnotationLossKind.EMBEDDING_OMITTED,
                    lossy=True,
                    code="export_policy_omitted_embedding",
                )
            )
        writer.writerow(
            {
                "annotation_id": record.annotation_id,
                "annotation_type": record.annotation_type.value,
                "compatibility_policy": record.compatibility_policy,
                "coordinate_convention": record.coordinate_convention.value,
                "document_id": record.document_id,
                "embedding_json": canonical_json(list(embedding))
                if embedding is not None
                else "null",
                "end": "" if record.end is None else str(record.end),
                "metadata_json": canonical_json(_plain(record.metadata)),
                "namespace": record.namespace,
                "result_json": canonical_json(_plain(record.result)),
                "schema_version": record.schema_version,
                "start": "" if record.start is None else str(record.start),
                "state": record.state.value,
            }
        )
    text = output.getvalue()
    output_digest = canonical_digest(text)
    report = AnnotationLossReport(
        operation="export_tsv",
        state=(AnnotationState.PARTIAL if losses else AnnotationState.SUCCESS),
        input_digest=envelope.envelope_digest,
        output_digest=output_digest,
        entries=tuple(losses),
    )
    return AnnotationExport(text=text, report=report)


def import_annotation_tsv(payload: str | bytes) -> AnnotationEnvelope:
    """Parse canonical TSV without accepting source-text-bearing columns."""

    if isinstance(payload, bytes):
        if len(payload) > MAX_ANNOTATION_INPUT_BYTES:
            raise AnnotationInterchangeError("annotation input byte limit exceeded")
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            raise AnnotationInterchangeError("annotation input must be UTF-8") from None
    elif isinstance(payload, str):
        if len(payload.encode("utf-8")) > MAX_ANNOTATION_INPUT_BYTES:
            raise AnnotationInterchangeError("annotation input byte limit exceeded")
        text = payload
    else:
        raise AnnotationInterchangeError("annotation input must be text or bytes")
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter="\t")
    if tuple(reader.fieldnames or ()) != _TSV_COLUMNS:
        raise AnnotationInterchangeError("annotation TSV columns are invalid")
    records: list[AnnotationRecord] = []
    for index, row in enumerate(reader, start=1):
        if index > MAX_ANNOTATION_ROWS:
            raise AnnotationInterchangeError("annotation row limit exceeded")
        if None in row or any(value is None for value in row.values()):
            raise AnnotationInterchangeError("annotation TSV row width is invalid")
        if any(len(value) > MAX_ANNOTATION_CELL_CHARS for value in row.values()):
            raise AnnotationInterchangeError("annotation TSV cell limit exceeded")
        records.append(
            AnnotationRecord(
                annotation_id=row["annotation_id"],
                document_id=row["document_id"],
                namespace=row["namespace"],
                annotation_type=AnnotationType(row["annotation_type"]),
                coordinate_convention=CoordinateConvention(
                    row["coordinate_convention"]
                ),
                start=_optional_int(row["start"], "start"),
                end=_optional_int(row["end"], "end"),
                state=AnnotationState(row["state"]),
                result=_json_object(row["result_json"], "result_json"),
                metadata=_json_object(row["metadata_json"], "metadata_json"),
                embedding=_json_embedding(row["embedding_json"]),
                schema_version=row["schema_version"],
                compatibility_policy=row["compatibility_policy"],
            )
        )
    return build_annotation_envelope(records)


def convert_record_offsets(
    record: AnnotationRecord,
    *,
    source_text: str,
    target: CoordinateConvention,
) -> tuple[AnnotationRecord, AnnotationLossReport]:
    """Convert exact Unicode boundaries without storing or returning source text.

    Token-index conversion is intentionally unsupported without an explicit
    tokenizer map; the function fails rather than guessing token boundaries.
    """

    target = CoordinateConvention(target)
    source = record.coordinate_convention
    if record.start is None or record.end is None:
        raise AnnotationInterchangeError("record does not carry offsets")
    if CoordinateConvention.TOKEN_INDEX in {source, target}:
        raise AnnotationInterchangeError("token_map_required")
    if CoordinateConvention.NONE in {source, target}:
        raise AnnotationInterchangeError("coordinate convention is not convertible")
    boundaries = _coordinate_boundaries(source_text)
    source_map = boundaries[source]
    reverse = {value: index for index, value in enumerate(source_map)}
    if record.start not in reverse or record.end not in reverse:
        raise AnnotationInterchangeError("offset_not_boundary")
    converted = replace(
        record,
        coordinate_convention=target,
        start=boundaries[target][reverse[record.start]],
        end=boundaries[target][reverse[record.end]],
    )
    report = AnnotationLossReport(
        operation="convert_offsets",
        state=AnnotationState.SUCCESS,
        input_digest=record.record_digest,
        output_digest=converted.record_digest,
    )
    return converted, report


def load_annotation_interchange_schema(name: str) -> dict[str, Any]:
    """Load one bundled annotation interchange JSON Schema."""

    normalized = name.removeprefix("annotation_").removesuffix(".schema.json")
    if normalized not in ANNOTATION_INTERCHANGE_SCHEMA_NAMES:
        raise KeyError("unknown annotation interchange schema")
    resource = resources.files("openmed.core.schemas.json").joinpath(
        f"annotation_{normalized}.schema.json"
    )
    with resource.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_offsets(
    annotation_type: AnnotationType,
    convention: CoordinateConvention,
    start: Any,
    end: Any,
) -> None:
    carries_offsets = annotation_type is AnnotationType.ENTITY
    if carries_offsets:
        if convention is CoordinateConvention.NONE:
            raise AnnotationInterchangeError("entity annotation needs coordinates")
        if type(start) is not int or type(end) is not int or start < 0 or start >= end:
            raise AnnotationInterchangeError("annotation offsets are invalid")
        return
    if (
        start is not None
        or end is not None
        or convention is not CoordinateConvention.NONE
    ):
        raise AnnotationInterchangeError(
            "non-entity annotations cannot carry text coordinates"
        )


def _validate_result(
    annotation_type: AnnotationType, value: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AnnotationInterchangeError("annotation result must be an object")
    result = dict(value)
    allowed = _RESULT_FIELDS[annotation_type]
    required = _REQUIRED_RESULT_FIELDS[annotation_type]
    if not required.issubset(result) or not set(result).issubset(allowed):
        raise AnnotationInterchangeError("annotation result fields are invalid")
    for key, item in result.items():
        if key in _SENSITIVE_KEYS:
            raise AnnotationInterchangeError(
                "annotation result contains sensitive data"
            )
        if key == "surface_hash":
            _digest(item, key)
        elif key.endswith("_id"):
            _opaque(item, key)
        elif key == "evidence_ids":
            if not isinstance(item, list) or len(item) > 64:
                raise AnnotationInterchangeError("evidence_ids must be a bounded array")
            for identifier in item:
                _opaque(identifier, "evidence_id")
            result[key] = tuple(item)
        else:
            _controlled(item, key)
    return result


def _validate_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not set(value).issubset(_METADATA_FIELDS):
        raise AnnotationInterchangeError("annotation metadata fields are invalid")
    result = dict(value)
    for key, item in result.items():
        if key.endswith("_id"):
            _opaque(item, key)
        elif key == "model_version":
            if not isinstance(item, str) or _MODEL_VERSION_RE.fullmatch(item) is None:
                raise AnnotationInterchangeError("model_version is invalid")
        else:
            _controlled(item, key)
    return result


def _validate_embedding(value: tuple[float, ...] | None) -> tuple[float, ...] | None:
    if value is None:
        return None
    if (
        not isinstance(value, tuple)
        or not 1 <= len(value) <= MAX_ANNOTATION_EMBEDDING_DIMENSIONS
    ):
        raise AnnotationInterchangeError("embedding dimensions are invalid")
    normalized: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise AnnotationInterchangeError("embedding values must be numbers")
        number = float(item)
        if not math.isfinite(number):
            raise AnnotationInterchangeError("embedding values must be finite")
        normalized.append(number)
    return tuple(normalized)


def _coordinate_boundaries(text: str) -> dict[CoordinateConvention, tuple[int, ...]]:
    codepoints = tuple(range(len(text) + 1))
    utf8 = [0]
    utf16 = [0]
    for character in text:
        utf8.append(utf8[-1] + len(character.encode("utf-8")))
        utf16.append(utf16[-1] + len(character.encode("utf-16-le")) // 2)
    return {
        CoordinateConvention.UNICODE_CODEPOINT: codepoints,
        CoordinateConvention.UTF8_BYTE: tuple(utf8),
        CoordinateConvention.UTF16_CODE_UNIT: tuple(utf16),
    }


def _aggregate_state(records: Sequence[AnnotationRecord]) -> AnnotationState:
    states = {item.state for item in records}
    return next(iter(states)) if len(states) == 1 else AnnotationState.PARTIAL


def _optional_int(value: str, field_name: str) -> int | None:
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        raise AnnotationInterchangeError(f"{field_name} must be an integer") from None


def _json_object(value: str, field_name: str) -> Mapping[str, Any]:
    parsed = _strict_json(value, field_name)
    if not isinstance(parsed, dict):
        raise AnnotationInterchangeError(f"{field_name} must be an object")
    return parsed


def _json_embedding(value: str) -> tuple[float, ...] | None:
    if value == "":
        return None
    parsed = _strict_json(value, "embedding_json")
    if parsed is None:
        return None
    if not isinstance(parsed, list):
        raise AnnotationInterchangeError("embedding_json must be an array")
    return tuple(parsed)


def _strict_json(value: str, field_name: str) -> Any:
    try:
        return json.loads(
            value,
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        raise AnnotationInterchangeError(f"{field_name} is invalid JSON") from None


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite number: {value}")


def _strict_mapping(
    payload: Mapping[str, Any], fields: set[str], name: str
) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise AnnotationInterchangeError(f"{name} fields are invalid")
    return dict(payload)


def _require_schema(version: Any, compatibility: Any) -> None:
    if version != ANNOTATION_INTERCHANGE_SCHEMA_VERSION:
        raise AnnotationInterchangeError("annotation schema version is unsupported")
    if compatibility != ANNOTATION_INTERCHANGE_COMPATIBILITY:
        raise AnnotationInterchangeError("annotation compatibility is unsupported")


def _controlled(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _CONTROLLED_RE.fullmatch(value) is None:
        raise AnnotationInterchangeError(
            f"{field_name} must be a controlled identifier"
        )
    return value


def _opaque(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _OPAQUE_ID_RE.fullmatch(value) is None:
        raise AnnotationInterchangeError(f"{field_name} must be an opaque identifier")
    return value


def _digest(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise AnnotationInterchangeError(f"{field_name} must be a SHA-256 digest")
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


__all__ = [
    "ANNOTATION_INTERCHANGE_COMPATIBILITY",
    "ANNOTATION_INTERCHANGE_SCHEMA_NAMES",
    "ANNOTATION_INTERCHANGE_SCHEMA_VERSION",
    "MAX_ANNOTATION_EMBEDDING_DIMENSIONS",
    "MAX_ANNOTATION_INPUT_BYTES",
    "MAX_ANNOTATION_PAGE_SIZE",
    "MAX_ANNOTATION_ROWS",
    "AnnotationAccessPolicy",
    "AnnotationCatalog",
    "AnnotationEnvelope",
    "AnnotationExport",
    "AnnotationInterchangeError",
    "AnnotationLoss",
    "AnnotationLossKind",
    "AnnotationLossReport",
    "AnnotationPage",
    "AnnotationQuery",
    "AnnotationRecord",
    "AnnotationState",
    "AnnotationType",
    "CoordinateConvention",
    "build_annotation_envelope",
    "convert_record_offsets",
    "export_annotation_tsv",
    "import_annotation_tsv",
    "load_annotation_interchange_schema",
]
