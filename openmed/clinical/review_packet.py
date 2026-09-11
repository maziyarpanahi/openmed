"""Deterministic, PHI-safe packets for human review of clinical output.

The packet is an evidence summary for a human reviewer.  It is deliberately
not a clinical decision, release approval, or compliance certification.  The
module only normalizes caller-provided records; it does not load a model,
contact a service, or make a network request.

Raw source values are accepted only as protected input fields.  They are
represented by a hash and an availability flag in the default rendering.  A
caller must explicitly pass ``include_protected_text=True`` to a local render
operation to include the protected values in the returned artifact.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from openmed.core.audit import hash_text, stable_hash

REVIEW_PACKET_SCHEMA_VERSION = "openmed.clinical.review_packet.v1"
REVIEW_PACKET_ADVISORY = (
    "Human-review packets are assistive evidence summaries, not clinical "
    "decisions, release approvals, or compliance certifications."
)
PROTECTED_TEXT_POLICY = "explicit_local_opt_in"
PROTECTED_TEXT_OMITTED = "[PROTECTED_TEXT_OMITTED]"

_MISSING = object()
_RAW_FIELD_NAMES = frozenset(
    {
        "content",
        "deidentified_text",
        "display",
        "excerpt",
        "message",
        "normalized_value",
        "note",
        "original",
        "original_text",
        "quote",
        "raw",
        "raw_text",
        "raw_value",
        "source_text",
        "surface",
        "suggestion",
        "text",
        "value",
    }
)
_FREE_TEXT_FIELD_NAMES = frozenset(
    {
        "comment",
        "context",
        "description",
        "evidence",
        "message",
        "note",
        "reason_detail",
        "summary",
    }
)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")
_SAFE_REASON_RE = re.compile(r"^[a-z0-9][a-z0-9_.:/-]{0,79}$")
_SAFE_STATUS_RE = re.compile(r"^[a-z0-9][a-z0-9_.:/-]{0,63}$")
_SAFE_METADATA_VALUE_RE = re.compile(r"^[a-z0-9][a-z0-9_.:/-]{0,127}$")
_STRUCTURED_METADATA_FIELDS = frozenset(
    {
        "category",
        "class",
        "code",
        "direction",
        "format",
        "identifier",
        "id",
        "kind",
        "label",
        "language",
        "level",
        "metric",
        "mode",
        "name",
        "policy",
        "reason_code",
        "severity",
        "source",
        "stage",
        "status",
        "system",
        "type",
        "unit",
        "version",
    }
)
_SAFE_REASON_PREFIXES = (
    "blocked",
    "error",
    "failed",
    "gate",
    "high",
    "insufficient",
    "invalid",
    "low",
    "missing",
    "no_",
    "not_",
    "ok",
    "passed",
    "policy",
    "ready",
    "requires",
    "review",
    "threshold",
    "uncertain",
    "unavailable",
    "unsupported",
    "valid",
)


def _canonical_json(value: Any) -> str:
    """Serialize an already sanitized value using the packet contract."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().split())


def _safe_identifier(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    normalized = _normalize_text(value)
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    if len(normalized) > 128:
        return f"identifier:{hash_text(normalized)}"
    return normalized


def _optional_text(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string or None")
    normalized = _normalize_text(value)
    return normalized or None


def _serialize_protected_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return _canonical_json(_plain_json_value(value))
    except (TypeError, ValueError):
        return f"<{type(value).__name__}>"


def _protected_hash(value: str | None) -> str | None:
    return hash_text(value) if value is not None else None


def _plain_json_value(value: object) -> object:
    """Convert a value to a bounded, JSON-compatible value without logging it."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite numbers are not supported")
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json_value(item) for item in value]
    return f"<{type(value).__name__}>"


def _contains_protected(value: str, protected_values: Iterable[str]) -> bool:
    normalized = value.casefold()
    return any(
        candidate and candidate.casefold() in normalized
        for candidate in protected_values
    )


def _sanitize_metadata(
    value: object,
    *,
    protected_values: Iterable[str] = (),
    field_name: str | None = None,
) -> object:
    """Keep structured metadata while dropping free-form source content."""

    protected = tuple(item for item in protected_values if item)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        if _contains_protected(value, protected):
            return PROTECTED_TEXT_OMITTED
        if field_name in _FREE_TEXT_FIELD_NAMES:
            return None
        normalized = _normalize_text(value)
        if len(normalized) > 256:
            return f"hash:{hash_text(normalized)}"
        if (
            field_name not in _STRUCTURED_METADATA_FIELDS
            or not _SAFE_METADATA_VALUE_RE.fullmatch(normalized.casefold())
            or normalized != normalized.casefold()
        ):
            return f"hash:{hash_text(normalized)}"
        return normalized
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for raw_key, raw_item in value.items():
            key = str(raw_key).strip()
            normalized_key = re.sub(r"(?<!^)(?=[A-Z])", "_", key).casefold()
            normalized_key = normalized_key.replace("-", "_")
            if not key or normalized_key.endswith("_hash"):
                pass
            elif normalized_key in _RAW_FIELD_NAMES or any(
                marker in normalized_key
                for marker in (
                    "comment",
                    "content",
                    "context",
                    "description",
                    "evidence",
                    "excerpt",
                    "message",
                    "note",
                    "original",
                    "quote",
                    "raw",
                    "source_text",
                    "surface",
                    "text",
                    "value",
                )
            ):
                continue
            if not _IDENTIFIER_RE.fullmatch(key):
                continue
            item = _sanitize_metadata(
                raw_item,
                protected_values=protected,
                field_name=normalized_key,
            )
            if item is not None:
                result[key] = item
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            sanitized = _sanitize_metadata(item, protected_values=protected)
            if sanitized is not None:
                result.append(sanitized)
        return result
    return None


def _tuple_of_identifiers(value: object, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = (value,)
    else:
        if isinstance(value, (bytes, bytearray)):
            raise ValueError(f"{field_name} must contain identifiers")
        try:
            values = tuple(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError(f"{field_name} must contain identifiers") from exc
    normalized = {
        _safe_identifier(item, field_name=f"{field_name} entry") for item in values
    }
    return tuple(sorted(normalized))


def _optional_finite_float(value: object, *, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number or None")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be a finite number or None")
    return result


def _optional_offset(value: object, *, field_name: str) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        start = value.get("start")
        end = value.get("end")
    else:
        try:
            pair = tuple(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError(f"{field_name} must be a start/end pair") from exc
        if len(pair) != 2:
            raise ValueError(f"{field_name} must be a start/end pair")
        start, end = pair
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end < start
    ):
        raise ValueError(f"{field_name} must contain non-negative offsets")
    return (start, end)


def _first_field(
    value: object, names: Iterable[str], default: object = _MISSING
) -> object:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if not isinstance(value, Mapping):
            candidate = getattr(value, name, _MISSING)
            if candidate is not _MISSING:
                return candidate
    if default is not _MISSING:
        return default
    return None


def _protected_from_fields(value: object, names: Iterable[str]) -> str | None:
    for name in names:
        candidate = _first_field(value, (name,))
        protected = _serialize_protected_value(candidate)
        if protected is not None:
            return protected
    return None


def _safe_reason(
    value: object,
    *,
    protected_values: Iterable[str] = (),
) -> tuple[str, str | None]:
    if value is None:
        return "unspecified", None
    if not isinstance(value, str):
        return "provided", None
    normalized = _normalize_text(value).casefold()
    if not normalized:
        return "unspecified", None
    if _contains_protected(normalized, protected_values):
        return "protected", hash_text(normalized)
    if _SAFE_REASON_RE.fullmatch(normalized) and (
        normalized.startswith(_SAFE_REASON_PREFIXES)
        or normalized in {"ok", "provided", "protected", "unspecified"}
    ):
        return normalized, None
    return "provided", hash_text(normalized)


def _safe_status(
    value: object,
    *,
    default: str,
    protected_values: Iterable[str] = (),
) -> str:
    if not isinstance(value, str) or not value.strip():
        return default
    normalized = _normalize_text(value).casefold().replace(" ", "_")
    if _contains_protected(normalized, protected_values):
        return "protected"
    if _SAFE_STATUS_RE.fullmatch(normalized):
        return normalized
    return f"status:{hash_text(normalized)}"


def _safe_label(
    value: object,
    *,
    default: str,
    protected_values: Iterable[str] = (),
) -> str:
    if not isinstance(value, str) or not value.strip():
        return default
    normalized = _normalize_text(value)
    if _contains_protected(normalized, protected_values):
        return f"label:{hash_text(normalized)}"
    if len(normalized) > 128:
        return f"label:{hash_text(normalized)}"
    return normalized


@dataclass(frozen=True, slots=True)
class ReviewFinding:
    """A typed, PHI-safe finding included in a human-review packet.

    ``protected_text`` (and the compatibility aliases ``source_value``,
    ``text``, and ``value``) are never included by :meth:`to_dict` unless the
    caller explicitly opts in.  ``source_start`` and ``source_end`` are
    offsets, not source text, and are safe to retain for reviewer navigation.
    """

    finding_id: str
    label: str
    confidence: float | None = None
    uncertainty: str | None = None
    status: str = "needs_review"
    citation_ids: tuple[str, ...] = ()
    source_start: int | None = None
    source_end: int | None = None
    source_hash: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict, repr=False)
    protected_text: str | None = field(default=None, repr=False, compare=False)
    source_value: str | None = field(default=None, repr=False, compare=False)
    text: str | None = field(default=None, repr=False, compare=False)
    value: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "finding_id",
            _safe_identifier(self.finding_id, field_name="finding_id"),
        )
        protected = self.protected_text
        if protected is None:
            protected = _serialize_protected_value(self.source_value)
        if protected is None:
            protected = _serialize_protected_value(self.text)
        if protected is None:
            protected = _serialize_protected_value(self.value)
        if protected is not None and not isinstance(protected, str):
            raise ValueError("protected_text must be a string or None")
        object.__setattr__(self, "protected_text", protected)
        protected_values = (protected,) if protected is not None else ()
        object.__setattr__(
            self,
            "label",
            _safe_label(
                self.label,
                default="finding",
                protected_values=protected_values,
            ),
        )
        confidence = _optional_finite_float(
            self.confidence,
            field_name="confidence",
        )
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(
            self,
            "uncertainty",
            (
                None
                if isinstance(self.uncertainty, str)
                and _contains_protected(self.uncertainty, protected_values)
                else _optional_text(self.uncertainty, field_name="uncertainty")
            ),
        )
        object.__setattr__(
            self,
            "status",
            _safe_status(
                self.status,
                default="needs_review",
                protected_values=protected_values,
            ),
        )
        object.__setattr__(
            self,
            "citation_ids",
            _tuple_of_identifiers(self.citation_ids, field_name="citation_ids"),
        )
        offset = _optional_offset(
            (self.source_start, self.source_end)
            if self.source_start is not None or self.source_end is not None
            else None,
            field_name="source offsets",
        )
        object.__setattr__(self, "source_start", offset[0] if offset else None)
        object.__setattr__(self, "source_end", offset[1] if offset else None)
        if self.source_hash is None:
            object.__setattr__(self, "source_hash", _protected_hash(protected))
        elif not isinstance(self.source_hash, str) or not self.source_hash.strip():
            raise ValueError("source_hash must be a non-empty string or None")
        safe_attributes = _sanitize_metadata(
            self.attributes,
            protected_values=protected_values,
        )
        object.__setattr__(
            self,
            "attributes",
            MappingProxyType(
                safe_attributes if isinstance(safe_attributes, dict) else {}
            ),
        )

    @property
    def id(self) -> str:
        """Return the stable finding identifier."""

        return self.finding_id

    def to_dict(self, *, include_protected_text: bool = False) -> dict[str, Any]:
        """Return the finding in its default PHI-safe representation."""

        payload: dict[str, Any] = {
            "finding_id": self.finding_id,
            "label": self.label,
            "status": self.status,
        }
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        if self.uncertainty is not None:
            payload["uncertainty"] = self.uncertainty
        if self.citation_ids:
            payload["citation_ids"] = list(self.citation_ids)
        if self.source_start is not None and self.source_end is not None:
            payload["source_offset"] = {
                "start": self.source_start,
                "end": self.source_end,
            }
        if self.source_hash is not None:
            payload["source_hash"] = self.source_hash
        if self.attributes:
            payload["attributes"] = dict(self.attributes)
        if self.protected_text is not None:
            payload["protected_text_available"] = True
            if include_protected_text:
                payload["protected_text"] = self.protected_text
        return payload


@dataclass(frozen=True, slots=True)
class ReviewCitation:
    """A non-sensitive reference supporting one or more review findings."""

    citation_id: str
    source: str
    locator: str | None = None
    title: str | None = None
    published: str | None = None
    relevance: float | None = None
    source_hash: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)
    protected_text: str | None = field(default=None, repr=False, compare=False)
    excerpt: str | None = field(default=None, repr=False, compare=False)
    quote: str | None = field(default=None, repr=False, compare=False)
    text: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "citation_id",
            _safe_identifier(self.citation_id, field_name="citation_id"),
        )
        protected = self.protected_text
        if protected is None:
            protected = _serialize_protected_value(self.excerpt)
        if protected is None:
            protected = _serialize_protected_value(self.quote)
        if protected is None:
            protected = _serialize_protected_value(self.text)
        object.__setattr__(self, "protected_text", protected)
        protected_values = (protected,) if protected is not None else ()
        object.__setattr__(
            self,
            "source",
            _safe_label(
                self.source,
                default="source",
                protected_values=protected_values,
            ),
        )
        for field_name in ("locator", "title", "published"):
            field_value = getattr(self, field_name)
            if isinstance(field_value, str) and _contains_protected(
                field_value, protected_values
            ):
                field_value = None
            object.__setattr__(
                self,
                field_name,
                _optional_text(field_value, field_name=field_name),
            )
        relevance = _optional_finite_float(self.relevance, field_name="relevance")
        if relevance is not None and not 0.0 <= relevance <= 1.0:
            raise ValueError("relevance must be between 0 and 1")
        object.__setattr__(self, "relevance", relevance)

        if self.source_hash is None:
            object.__setattr__(self, "source_hash", _protected_hash(protected))
        elif not isinstance(self.source_hash, str) or not self.source_hash.strip():
            raise ValueError("source_hash must be a non-empty string or None")
        safe_metadata = _sanitize_metadata(
            self.metadata,
            protected_values=protected_values,
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(safe_metadata if isinstance(safe_metadata, dict) else {}),
        )

    @property
    def id(self) -> str:
        """Return the stable citation identifier."""

        return self.citation_id

    def to_dict(self, *, include_protected_text: bool = False) -> dict[str, Any]:
        """Return the citation without its excerpt or quoted source text."""

        payload: dict[str, Any] = {
            "citation_id": self.citation_id,
            "source": self.source,
        }
        for field_name in ("locator", "title", "published"):
            value = getattr(self, field_name)
            if value is not None:
                payload[field_name] = value
        if self.relevance is not None:
            payload["relevance"] = self.relevance
        if self.source_hash is not None:
            payload["source_hash"] = self.source_hash
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        if self.protected_text is not None:
            payload["protected_text_available"] = True
            if include_protected_text:
                payload["protected_text"] = self.protected_text
        return payload


@dataclass(frozen=True, slots=True)
class ReviewGateResult:
    """A typed policy or quality-gate result attached to a review packet."""

    gate_id: str
    passed: bool
    reason: str = "ok"
    severity: str = "info"
    blocking: bool = False
    citation_ids: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict, repr=False)
    reason_hash: str | None = field(default=None, repr=False, compare=False)
    reason_code: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "gate_id", _safe_identifier(self.gate_id, field_name="gate_id")
        )
        if not isinstance(self.passed, bool):
            raise ValueError("passed must be a boolean")
        object.__setattr__(
            self, "severity", _safe_status(self.severity, default="info")
        )
        reason, generated_hash = _safe_reason(
            self.reason_code if self.reason_code is not None else self.reason
        )
        object.__setattr__(self, "reason", reason)
        if self.reason_hash is None:
            object.__setattr__(self, "reason_hash", generated_hash)
        elif not isinstance(self.reason_hash, str) or not self.reason_hash.strip():
            raise ValueError("reason_hash must be a non-empty string or None")
        if not isinstance(self.blocking, bool):
            raise ValueError("blocking must be a boolean")
        object.__setattr__(
            self,
            "citation_ids",
            _tuple_of_identifiers(self.citation_ids, field_name="citation_ids"),
        )
        safe_details = _sanitize_metadata(self.details)
        object.__setattr__(
            self,
            "details",
            MappingProxyType(safe_details if isinstance(safe_details, dict) else {}),
        )

    @property
    def gate(self) -> str:
        """Return the gate identifier used by common gate-report objects."""

        return self.gate_id

    def to_dict(self) -> dict[str, Any]:
        """Return the gate result with free-form source details removed."""

        payload: dict[str, Any] = {
            "gate_id": self.gate_id,
            "passed": self.passed,
            "reason": self.reason,
            "severity": self.severity,
            "blocking": self.blocking,
        }
        if self.reason_hash is not None:
            payload["reason_hash"] = self.reason_hash
        if self.citation_ids:
            payload["citation_ids"] = list(self.citation_ids)
        if self.details:
            payload["details"] = dict(self.details)
        return payload


def _derive_review_status(gates: tuple[ReviewGateResult, ...]) -> str:
    if not gates:
        return "not_evaluated"
    if any(
        not gate.passed
        and (
            gate.blocking or gate.severity in {"critical", "error", "high", "blocking"}
        )
        for gate in gates
    ):
        return "blocked"
    if any(not gate.passed for gate in gates):
        return "review_required"
    return "ready_for_review"


def _require_unique_ids(
    items: Iterable[object], *, attribute: str, field_name: str
) -> None:
    identifiers: list[str] = []
    for item in items:
        identifier = getattr(item, attribute)
        if identifier in identifiers:
            raise ValueError(f"{field_name} must have unique identifiers")
        identifiers.append(identifier)


@dataclass(frozen=True, slots=True)
class ReviewPacket:
    """A deterministic human-review artifact containing guarded evidence."""

    findings: tuple[ReviewFinding, ...] = ()
    citations: tuple[ReviewCitation, ...] = ()
    gate_results: tuple[ReviewGateResult, ...] = ()
    packet_id: str = ""
    review_status: str | None = None
    schema_version: str = REVIEW_PACKET_SCHEMA_VERSION
    advisory: str = REVIEW_PACKET_ADVISORY

    def __post_init__(self) -> None:
        findings = tuple(
            sorted(self.findings, key=lambda item: (item.finding_id, item.label))
        )
        citations = tuple(
            sorted(self.citations, key=lambda item: (item.citation_id, item.source))
        )
        gates = tuple(
            sorted(self.gate_results, key=lambda item: (item.gate_id, item.severity))
        )
        if not all(isinstance(item, ReviewFinding) for item in findings):
            raise ValueError("findings must contain ReviewFinding records")
        if not all(isinstance(item, ReviewCitation) for item in citations):
            raise ValueError("citations must contain ReviewCitation records")
        if not all(isinstance(item, ReviewGateResult) for item in gates):
            raise ValueError("gate_results must contain ReviewGateResult records")
        _require_unique_ids(findings, attribute="finding_id", field_name="findings")
        _require_unique_ids(citations, attribute="citation_id", field_name="citations")
        _require_unique_ids(gates, attribute="gate_id", field_name="gate_results")
        object.__setattr__(self, "findings", findings)
        object.__setattr__(self, "citations", citations)
        object.__setattr__(self, "gate_results", gates)
        if not isinstance(self.schema_version, str) or not self.schema_version.strip():
            raise ValueError("schema_version must be a non-empty string")
        object.__setattr__(self, "schema_version", _normalize_text(self.schema_version))
        object.__setattr__(self, "advisory", _normalize_text(self.advisory))
        review_status = self.review_status or _derive_review_status(gates)
        object.__setattr__(
            self, "review_status", _safe_status(review_status, default="not_evaluated")
        )
        if self.packet_id:
            object.__setattr__(
                self,
                "packet_id",
                _safe_identifier(self.packet_id, field_name="packet_id"),
            )
        else:
            object.__setattr__(self, "packet_id", stable_hash(self._identity_payload()))

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "review_status": self.review_status,
            "findings": [item.to_dict() for item in self.findings],
            "citations": [item.to_dict() for item in self.citations],
            "gate_results": [item.to_dict() for item in self.gate_results],
        }

    def to_dict(self, *, include_protected_text: bool = False) -> dict[str, Any]:
        """Return the packet, omitting protected source values by default."""

        failed_gates = sum(not gate.passed for gate in self.gate_results)
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "packet_id": self.packet_id,
            "review_status": self.review_status,
            "findings": [
                item.to_dict(include_protected_text=include_protected_text)
                for item in self.findings
            ],
            "citations": [
                item.to_dict(include_protected_text=include_protected_text)
                for item in self.citations
            ],
            "gate_results": [item.to_dict() for item in self.gate_results],
            "summary": {
                "finding_count": len(self.findings),
                "citation_count": len(self.citations),
                "gate_count": len(self.gate_results),
                "failed_gate_count": failed_gates,
                "review_required": True,
            },
            "privacy": {
                "protected_text_policy": PROTECTED_TEXT_POLICY,
                "protected_text_included": bool(include_protected_text),
                "protected_text_available": any(
                    item.protected_text is not None
                    for item in (*self.findings, *self.citations)
                ),
            },
            "advisory": self.advisory,
        }
        return payload

    def to_json(self, *, include_protected_text: bool = False) -> str:
        """Serialize the packet using stable JSON key and record ordering."""

        return _canonical_json(
            self.to_dict(include_protected_text=include_protected_text)
        )

    def to_markdown(self, *, include_protected_text: bool = False) -> str:
        """Render a compact Markdown review packet."""

        lines = [
            "# Human review packet",
            "",
            f"- Packet: `{self.packet_id}`",
            f"- Review status: `{self.review_status}`",
            f"- Findings: {len(self.findings)}",
            f"- Failed gates: {sum(not gate.passed for gate in self.gate_results)}",
            "",
            "## Findings",
            "",
            "| Finding | Label | Status | Confidence | Uncertainty | Source |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        if not self.findings:
            lines.append("| _None_ | | | | | |")
        for finding in self.findings:
            confidence = (
                "" if finding.confidence is None else f"{finding.confidence:.6g}"
            )
            offset = ""
            if finding.source_start is not None and finding.source_end is not None:
                offset = f"{finding.source_start}:{finding.source_end}"
            lines.append(
                "| "
                + " | ".join(
                    _markdown_cell(value)
                    for value in (
                        finding.finding_id,
                        finding.label,
                        finding.status,
                        confidence,
                        finding.uncertainty or "",
                        offset,
                    )
                )
                + " |"
            )
            if include_protected_text and finding.protected_text is not None:
                lines.append(
                    f"  - Protected text: `{_markdown_code(finding.protected_text)}`"
                )

        lines.extend(
            [
                "",
                "## Citations",
                "",
                "| Citation | Source | Locator | Title |",
                "| --- | --- | --- | --- |",
            ]
        )
        if not self.citations:
            lines.append("| _None_ | | | |")
        for citation in self.citations:
            lines.append(
                "| "
                + " | ".join(
                    _markdown_cell(value)
                    for value in (
                        citation.citation_id,
                        citation.source,
                        citation.locator or "",
                        citation.title or "",
                    )
                )
                + " |"
            )
            if include_protected_text and citation.protected_text is not None:
                lines.append(
                    f"  - Protected text: `{_markdown_code(citation.protected_text)}`"
                )

        lines.extend(
            [
                "",
                "## Gate results",
                "",
                "| Gate | Passed | Severity | Reason | Blocking |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        if not self.gate_results:
            lines.append("| _None_ | | | | |")
        for gate in self.gate_results:
            lines.append(
                "| "
                + " | ".join(
                    _markdown_cell(value)
                    for value in (
                        gate.gate_id,
                        "yes" if gate.passed else "no",
                        gate.severity,
                        gate.reason,
                        "yes" if gate.blocking else "no",
                    )
                )
                + " |"
            )

        lines.extend(["", f"> {self.advisory}"])
        return "\n".join(lines) + "\n"


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _markdown_code(value: str) -> str:
    return value.replace("`", "'\"").replace("\n", " ").replace("\r", " ")


def _records(value: object, *, field_name: str) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping) or isinstance(
        value, (ReviewFinding, ReviewCitation, ReviewGateResult)
    ):
        return (value,)
    if isinstance(value, (str, bytes, bytearray)):
        return (value,)
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(
            f"{field_name} must be a record or iterable of records"
        ) from exc


def _reference_ids(value: object, *, field_name: str) -> tuple[str, ...]:
    """Coerce citation references without copying citation content."""

    if value is None:
        return ()
    if isinstance(value, str):
        return _tuple_of_identifiers(value, field_name=field_name)
    records = _records(value, field_name=field_name)
    identifiers: list[str] = []
    for item in records:
        if isinstance(item, str):
            identifiers.append(item)
            continue
        identifier = _first_field(item, ("citation_id", "id", "key"))
        if identifier is None:
            raise ValueError(f"{field_name} entries require identifiers")
        identifiers.append(
            _safe_identifier(identifier, field_name=f"{field_name} entry")
        )
    return _tuple_of_identifiers(identifiers, field_name=field_name)


def _first_span_offset(value: object) -> object:
    spans = _first_field(value, ("source_spans", "spans"))
    if spans is None:
        return None
    for span in _records(spans, field_name="source_spans"):
        start = _first_field(span, ("start", "source_start"))
        end = _first_field(span, ("end", "source_end"))
        if start is not None or end is not None:
            return (start, end)
    return None


def _coerce_finding(value: object) -> ReviewFinding:
    if isinstance(value, ReviewFinding):
        return value
    if isinstance(value, str):
        protected = value
        return ReviewFinding(
            finding_id=f"finding-{hash_text(protected)[-16:]}",
            label="finding",
            protected_text=protected,
        )
    if not isinstance(value, Mapping):
        finding_id = _first_field(value, ("finding_id", "id", "key"))
        label = _first_field(value, ("label", "kind", "type", "concept", "name"))
        suggestion = _first_field(value, ("suggestion",))
        if finding_id is None and label is None and suggestion is not None:
            label = "guarded_suggestion"
        if finding_id is None and label is None:
            raise ValueError("finding records require an identifier or label")
        value = {
            "finding_id": finding_id,
            "label": label,
            "confidence": _first_field(value, ("confidence", "score")),
            "uncertainty": _first_field(value, ("uncertainty", "certainty")),
            "status": _first_field(value, ("status", "state")),
            "citation_ids": _first_field(value, ("citation_ids", "citations")),
            "start": _first_field(value, ("start", "source_start")),
            "end": _first_field(value, ("end", "source_end")),
            "source_hash": _first_field(value, ("source_hash", "text_hash")),
            "attributes": _first_field(value, ("attributes", "metadata"), {}),
            "suggestion": suggestion,
            "source_spans": _first_field(value, ("source_spans", "spans")),
            "text": _first_field(
                value, ("protected_text", "source_value", "raw_text", "text", "value")
            ),
        }
    raw = _protected_from_fields(
        value,
        (
            "protected_text",
            "source_value",
            "raw_text",
            "raw_value",
            "source_text",
            "text",
            "surface",
            "suggestion",
            "value",
        ),
    )
    default_label = (
        "guarded_suggestion"
        if _first_field(value, ("suggestion",)) is not None
        else "finding"
    )
    finding_id_value = _first_field(value, ("finding_id", "id", "key"))
    label_value = _first_field(
        value, ("label", "kind", "type", "concept", "name"), default_label
    )
    label = _safe_label(label_value, default="finding")
    status = _first_field(value, ("status", "state"), "needs_review")
    source_hash = _first_field(value, ("source_hash", "text_hash"))
    if finding_id_value is None:
        finding_id_value = f"finding-{hash_text(_canonical_json({'label': label, 'source_hash': source_hash or _protected_hash(raw)}))[-16:]}"
    offset = _first_field(value, ("source_offset", "offset", "span"))
    if offset is None:
        offset = _first_span_offset(value)
    if offset is None:
        start = _first_field(value, ("source_start", "start"))
        end = _first_field(value, ("source_end", "end"))
        offset = (start, end) if start is not None or end is not None else None
    citation_values = _first_field(value, ("citation_ids",), _MISSING)
    if citation_values is _MISSING:
        citation_values = _first_field(value, ("citations", "evidence_refs"), ())
    return ReviewFinding(
        finding_id=_safe_identifier(finding_id_value, field_name="finding_id"),
        label=label,
        confidence=_first_field(value, ("confidence", "score")),
        uncertainty=_first_field(value, ("uncertainty", "certainty")),
        status=status,
        citation_ids=_reference_ids(citation_values, field_name="citation_ids"),
        source_start=(
            offset[0]
            if isinstance(offset, (tuple, list)) and len(offset) == 2
            else (offset.get("start") if isinstance(offset, Mapping) else None)
        ),
        source_end=(
            offset[1]
            if isinstance(offset, (tuple, list)) and len(offset) == 2
            else (offset.get("end") if isinstance(offset, Mapping) else None)
        ),
        source_hash=source_hash,
        attributes=_first_field(value, ("attributes", "metadata"), {}),
        protected_text=raw,
    )


def _coerce_citation(value: object) -> ReviewCitation:
    if isinstance(value, ReviewCitation):
        return value
    if isinstance(value, str):
        citation_id = f"citation-{hash_text(value)[-16:]}"
        return ReviewCitation(citation_id=citation_id, source="provided", locator=value)
    if not isinstance(value, Mapping):
        value = {
            "citation_id": _first_field(value, ("citation_id", "id", "key")),
            "source": _first_field(value, ("source", "publisher", "organization")),
            "locator": _first_field(value, ("locator", "uri", "url", "doi")),
            "title": _first_field(value, ("title", "name")),
            "published": _first_field(value, ("published", "year", "date")),
            "relevance": _first_field(value, ("relevance", "score")),
            "source_hash": _first_field(value, ("source_hash", "text_hash")),
            "metadata": _first_field(value, ("metadata", "attributes"), {}),
            "excerpt": _first_field(
                value, ("protected_text", "excerpt", "quote", "text")
            ),
        }
    raw = _protected_from_fields(
        value, ("protected_text", "excerpt", "quote", "text", "content")
    )
    source = _first_field(value, ("source", "publisher", "organization"), "source")
    locator = _first_field(value, ("locator", "uri", "url", "doi"))
    title = _first_field(value, ("title", "name"))
    citation_id = _first_field(value, ("citation_id", "id", "key"))
    if citation_id is None:
        identity = {
            "source": source,
            "locator": locator,
            "title": title,
            "source_hash": _protected_hash(raw),
        }
        citation_id = (
            f"citation-{hash_text(_canonical_json(_plain_json_value(identity)))[-16:]}"
        )
    return ReviewCitation(
        citation_id=_safe_identifier(citation_id, field_name="citation_id"),
        source=source,
        locator=locator,
        title=title,
        published=_first_field(value, ("published", "year", "date")),
        relevance=_first_field(value, ("relevance", "score")),
        source_hash=_first_field(value, ("source_hash", "text_hash")),
        metadata=_first_field(value, ("metadata", "attributes"), {}),
        protected_text=raw,
    )


def _coerce_gate(
    value: object,
    *,
    protected_values: Iterable[str] = (),
) -> ReviewGateResult:
    if isinstance(value, ReviewGateResult):
        if not tuple(protected_values):
            return value
        return ReviewGateResult(
            gate_id=value.gate_id,
            passed=value.passed,
            reason=value.reason,
            severity=value.severity,
            blocking=value.blocking,
            citation_ids=value.citation_ids,
            details=_sanitize_metadata(
                value.details, protected_values=protected_values
            ),
            reason_hash=value.reason_hash,
        )
    if not isinstance(value, Mapping):
        value = {
            "gate_id": _first_field(value, ("gate_id", "gate", "id", "name")),
            "passed": _first_field(value, ("passed", "ok", "success")),
            "reason": _first_field(value, ("reason", "reason_code", "message")),
            "severity": _first_field(value, ("severity", "risk")),
            "blocking": _first_field(value, ("blocking", "is_blocking")),
            "citation_ids": _first_field(value, ("citation_ids", "citations")),
            "details": _first_field(value, ("details", "metadata"), {}),
        }
    gate_id = _first_field(value, ("gate_id", "gate", "id", "name"))
    if gate_id is None:
        raise ValueError("gate results require a gate identifier")
    reason, reason_hash = _safe_reason(
        _first_field(value, ("reason_code", "reason", "message")),
        protected_values=protected_values,
    )
    details = _sanitize_metadata(
        _first_field(value, ("details", "metadata"), {}),
        protected_values=protected_values,
    )
    citation_values = _first_field(value, ("citation_ids",), _MISSING)
    if citation_values is _MISSING:
        citation_values = _first_field(value, ("citations",), ())
    return ReviewGateResult(
        gate_id=_safe_identifier(gate_id, field_name="gate_id"),
        passed=_first_field(value, ("passed", "ok", "success"), False),
        reason=reason,
        severity=_first_field(value, ("severity", "risk"), "info"),
        blocking=_first_field(value, ("blocking", "is_blocking"), False),
        citation_ids=_reference_ids(citation_values, field_name="citation_ids"),
        details=details,
        reason_hash=reason_hash,
    )


def _gate_records(value: object) -> tuple[object, ...]:
    nested = _first_field(value, ("gate_results", "gates"))
    if nested is not None:
        return _records(nested, field_name="gate_results")
    return _records(value, field_name="gate_results")


def build_review_packet(
    findings: object = (),
    citations: object = (),
    gate_results: object = (),
    *,
    gates: object | None = None,
    packet_id: str | None = None,
    review_status: str | None = None,
    decision: str | None = None,
) -> ReviewPacket:
    """Build a deterministic review packet from typed or mapping records.

    ``findings``, ``citations``, and ``gate_results`` may contain the typed
    records in this module or mappings with equivalent field names.  A
    ``GateReport``-like object with a ``gate_results`` attribute is accepted for
    convenience.  All records are sorted by stable identifiers before the
    packet hash is computed.
    """

    if gates is not None:
        if gate_results not in ((), None):
            raise ValueError("provide gate_results or gates, not both")
        gate_results = gates
    if review_status is not None and decision is not None:
        raise ValueError("provide review_status or decision, not both")
    requested_status = review_status if review_status is not None else decision

    finding_records = tuple(
        _coerce_finding(item) for item in _records(findings, field_name="findings")
    )
    citation_records = tuple(
        _coerce_citation(item) for item in _records(citations, field_name="citations")
    )
    protected_values = tuple(
        item.protected_text
        for item in (*finding_records, *citation_records)
        if item.protected_text is not None
    )
    gate_records = tuple(
        _coerce_gate(item, protected_values=protected_values)
        for item in _gate_records(gate_results)
    )
    status = requested_status or _derive_review_status(gate_records)
    return ReviewPacket(
        findings=finding_records,
        citations=citation_records,
        gate_results=gate_records,
        packet_id=packet_id or "",
        review_status=status,
    )


def render_review_packet(
    findings: object = (),
    citations: object = (),
    gate_results: object = (),
    *,
    gates: object | None = None,
    format: Literal["json", "markdown", "dict", "text"] = "json",
    include_protected_text: bool = False,
    allow_protected_text: bool = False,
    packet_id: str | None = None,
    review_status: str | None = None,
    decision: str | None = None,
    packet: ReviewPacket | None = None,
) -> str | dict[str, Any]:
    """Render a review packet as stable JSON, Markdown, or a dictionary.

    The default is a PHI-safe JSON string.  ``include_protected_text`` is an
    explicit local opt-in and is intentionally false by default.  The alias
    ``allow_protected_text`` is provided for callers that prefer policy-style
    wording; either flag must be deliberately set by the caller.
    """

    if allow_protected_text:
        include_protected_text = True
    if packet is not None:
        if findings not in ((), None):
            raise ValueError("provide a packet or findings, not both")
        findings = packet
    if isinstance(findings, ReviewPacket):
        if (
            citations not in ((), None)
            or gate_results not in ((), None)
            or gates is not None
        ):
            raise ValueError(
                "a ReviewPacket cannot be combined with additional records"
            )
        packet = findings
    else:
        packet = build_review_packet(
            findings,
            citations,
            gate_results,
            gates=gates,
            packet_id=packet_id,
            review_status=review_status,
            decision=decision,
        )
    normalized_format = format.casefold()
    if normalized_format == "dict":
        return packet.to_dict(include_protected_text=include_protected_text)
    if normalized_format == "json":
        return packet.to_json(include_protected_text=include_protected_text)
    if normalized_format in {"markdown", "text"}:
        return packet.to_markdown(include_protected_text=include_protected_text)
    raise ValueError("format must be json, markdown, dict, or text")


def render_review_packet_json(
    packet: ReviewPacket,
    *,
    include_protected_text: bool = False,
) -> str:
    """Serialize an existing packet as deterministic JSON."""

    return packet.to_json(include_protected_text=include_protected_text)


def render_review_packet_markdown(
    packet: ReviewPacket,
    *,
    include_protected_text: bool = False,
) -> str:
    """Render an existing packet as compact Markdown."""

    return packet.to_markdown(include_protected_text=include_protected_text)


__all__ = [
    "PROTECTED_TEXT_OMITTED",
    "PROTECTED_TEXT_POLICY",
    "REVIEW_PACKET_ADVISORY",
    "REVIEW_PACKET_SCHEMA_VERSION",
    "Citation",
    "Finding",
    "GateResult",
    "HumanReviewPacket",
    "ClinicalFinding",
    "ReviewCitation",
    "ReviewFinding",
    "ReviewGateResult",
    "ReviewPacket",
    "build_review_packet",
    "render_review_packet",
    "render_review_packet_json",
    "render_review_packet_markdown",
]

Citation = ReviewCitation
Finding = ReviewFinding
GateResult = ReviewGateResult
HumanReviewPacket = ReviewPacket
ClinicalFinding = ReviewFinding
