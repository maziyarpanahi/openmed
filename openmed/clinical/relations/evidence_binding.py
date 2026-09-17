"""Mandatory, value-free evidence binding for guarded clinical relations.

Higher-risk relation aids are only reviewable when a reviewer can locate both
endpoints and the linking evidence in one identified source document.  This
module is the boundary for that contract.  It consumes an existing relation
candidate, but never infers a missing assertion state, document identifier, or
evidence span from nearby text.

The bound records are immutable and deliberately allow-list their output:
source offsets, normalized labels, opaque document/span identifiers, optional
content digests, assertion state, confidence, and fixed human-review metadata.
Raw source text and arbitrary candidate metadata are discarded.  Construction
is local and deterministic; no network or model lookup is performed.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from openmed.core.labels import normalize_label

EVIDENCE_BINDING_SCHEMA_VERSION: Final[int] = 1
EVIDENCE_BINDING_ADVISORY: Final[str] = (
    "Guarded relation aids are value-free assistive output for qualified human "
    "review. They are not diagnoses, treatment decisions, or autonomous "
    "clinical decisions."
)

_MISSING = object()
_HASH_RE = re.compile(r"^(?:sha256|hmac-sha256):[0-9a-f]{64}$")
_RELATION_TYPE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:/-]{0,127}$")


class EvidenceBindingError(ValueError):
    """Raised when a guarded relation cannot be bound to source evidence."""


class AssertionState(str, Enum):
    """Controlled assertion states accepted by guarded relation records.

    The first six values match the clinical evidence-table vocabulary.  The
    final four preserve the relation assertion vocabulary already emitted by
    :mod:`openmed.clinical.relations.assertion_filter`.
    """

    AFFIRMED = "affirmed"
    NEGATED = "negated"
    UNCERTAIN = "uncertain"
    HISTORICAL = "historical"
    HYPOTHETICAL = "hypothetical"
    UNKNOWN = "unknown"
    CONFIRMED = "confirmed"
    REFUTED = "refuted"
    CONDITIONAL = "conditional"
    POSSIBLE = "possible"


ASSERTION_STATE_VALUES: Final[tuple[str, ...]] = tuple(
    state.value for state in AssertionState
)


def _string(value: object, field_name: str) -> str:
    """Copy a string safely without allowing subclass errors to echo values."""

    if type(value) is str:
        result = value
    elif isinstance(value, str):
        try:
            result = str.__str__(value)
        except Exception:
            raise EvidenceBindingError(f"{field_name} must be a string") from None
    else:
        raise EvidenceBindingError(f"{field_name} must be a string")
    if not result:
        raise EvidenceBindingError(f"{field_name} must be non-empty")
    return result


def _integer(value: object, field_name: str) -> int:
    if type(value) is int:
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        try:
            return int(value)
        except Exception:
            raise EvidenceBindingError(f"{field_name} must be an integer") from None
    raise EvidenceBindingError(f"{field_name} must be an integer")


def _probability(value: object, field_name: str) -> float:
    if type(value) not in (int, float):
        raise EvidenceBindingError(f"{field_name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise EvidenceBindingError(
            f"{field_name} must be in the closed interval [0, 1]"
        )
    return normalized


def _hash_secret(value: object) -> bytes | None:
    if value is None:
        return None
    if type(value) is bytes:
        if not value:
            raise EvidenceBindingError("hash_secret must be non-empty")
        return value
    if isinstance(value, str):
        try:
            secret = _string(value, "hash_secret").encode("utf-8")
        except UnicodeError:
            raise EvidenceBindingError("hash_secret must be text or bytes") from None
        if not secret:
            raise EvidenceBindingError("hash_secret must be non-empty")
        return secret
    raise EvidenceBindingError("hash_secret must be text or bytes")


def _digest_identifier(
    value: object,
    *,
    namespace: str,
    hash_secret: object = None,
) -> str:
    identifier = _string(value, "identifier")
    if _HASH_RE.fullmatch(identifier) is not None:
        return identifier
    try:
        payload = f"{namespace}\0{identifier}".encode("utf-8")
    except UnicodeError:
        raise EvidenceBindingError("identifier cannot be normalized") from None
    secret = _hash_secret(hash_secret)
    if secret is None:
        digest = hashlib.sha256(payload).hexdigest()
        return f"sha256:{digest}"
    digest = hmac.new(secret, payload, hashlib.sha256).hexdigest()
    return f"hmac-sha256:{digest}"


def hash_document_id(
    document_id: object,
    *,
    hash_secret: str | bytes | None = None,
) -> str:
    """Return an opaque, domain-separated identifier for one source document.

    Existing ``sha256:`` or ``hmac-sha256:`` identifiers are preserved.  Other
    values are hashed before they can enter a bound record.
    """

    return _digest_identifier(
        document_id,
        namespace="relation-document",
        hash_secret=hash_secret,
    )


def _field(value: object, names: Sequence[str]) -> object:
    if isinstance(value, Mapping):
        for name in names:
            try:
                candidate = value.get(name, _MISSING)
            except Exception:
                raise EvidenceBindingError("relation metadata cannot be read") from None
            if candidate is not _MISSING:
                return candidate
        return _MISSING
    for name in names:
        try:
            candidate = getattr(value, name, _MISSING)
        except Exception:
            raise EvidenceBindingError("relation metadata cannot be read") from None
        if candidate is not _MISSING:
            return candidate
    return _MISSING


def _relation_type(value: object) -> str:
    relation_type = _string(value, "relation_type")
    if _RELATION_TYPE_RE.fullmatch(relation_type) is None:
        raise EvidenceBindingError("relation_type must be a bounded code")
    return relation_type


def _safe_label(value: object) -> str:
    label = _string(value, "span label")
    try:
        # Unknown model labels collapse to the fixed taxonomy member OTHER;
        # arbitrary caller strings therefore never enter the artifact.
        return normalize_label(label)
    except Exception:
        raise EvidenceBindingError("span label is unsupported") from None


def _safe_hash(value: object, field_name: str) -> str:
    digest = _string(value, field_name)
    if _HASH_RE.fullmatch(digest) is None:
        raise EvidenceBindingError(f"{field_name} must be a SHA-256 digest")
    return digest


def _document_identifier(
    value: object,
    *,
    hash_secret: object = None,
) -> str:
    if value is _MISSING or value is None:
        raise EvidenceBindingError("document_id is required")
    return hash_document_id(value, hash_secret=hash_secret)


def _offset_pair(value: object, *, field_name: str = "span offsets") -> tuple[int, int]:
    if isinstance(value, Mapping):
        nested = _field(value, ("source_offsets", "offset", "span"))
        if nested is not _MISSING and nested is not value:
            return _offset_pair(nested, field_name=field_name)
        start = _field(value, ("start", "source_start"))
        end = _field(value, ("end", "source_end"))
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        start, end = value
    else:
        nested = _field(value, ("source_offsets", "offset", "span"))
        if nested is not _MISSING and nested is not value:
            return _offset_pair(nested, field_name=field_name)
        start = _field(value, ("start", "source_start"))
        end = _field(value, ("end", "source_end"))
    if start is _MISSING or end is _MISSING:
        raise EvidenceBindingError(f"{field_name} are required")
    normalized_start = _integer(start, f"{field_name} start")
    normalized_end = _integer(end, f"{field_name} end")
    if normalized_start < 0 or normalized_end <= normalized_start:
        raise EvidenceBindingError(f"{field_name} must satisfy 0 <= start < end")
    return normalized_start, normalized_end


def _validate_document_bounds(
    span: "EvidenceSpan",
    document_length: int | None,
) -> None:
    if document_length is not None and span.end > document_length:
        raise EvidenceBindingError("span offsets exceed the source document")


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    """One value-free source span belonging to a relation document.

    ``document_id`` and ``span_id`` are normalized to opaque digests.  The
    optional ``text_hash`` can be supplied when a caller needs content
    correlation, but raw source text is intentionally not accepted as a field.
    """

    document_id: str
    start: int
    end: int
    label: str | None = None
    span_id: str | None = None
    text_hash: str | None = None

    def __post_init__(self) -> None:
        document_id = _document_identifier(self.document_id)
        start, end = _offset_pair((self.start, self.end))
        label = None if self.label is None else _safe_label(self.label)
        span_id = (
            None
            if self.span_id is None
            else _digest_identifier(self.span_id, namespace="relation-span")
        )
        text_hash = (
            None if self.text_hash is None else _safe_hash(self.text_hash, "text_hash")
        )
        object.__setattr__(self, "document_id", document_id)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "span_id", span_id)
        object.__setattr__(self, "text_hash", text_hash)

    @classmethod
    def from_obj(
        cls,
        value: object,
        *,
        document_id: object = _MISSING,
        document_length: int | None = None,
        hash_secret: str | bytes | None = None,
    ) -> "EvidenceSpan":
        """Build a value-free span from a mapping or span-like object.

        The relation-level document identifier is inherited when a span omits
        its own identifier.  If both are present they must identify the same
        document after normalization.
        """

        if isinstance(value, cls):
            if document_id is not _MISSING:
                expected = _document_identifier(
                    document_id,
                    hash_secret=hash_secret,
                )
                if value.document_id != expected:
                    raise EvidenceBindingError(
                        "span document_id does not match relation"
                    )
            _validate_document_bounds(value, document_length)
            return value

        raw_document_id = (
            document_id
            if document_id is not _MISSING
            else _field(value, ("document_id", "document_hash", "doc_id"))
        )
        normalized_document_id = _document_identifier(
            raw_document_id,
            hash_secret=hash_secret,
        )
        if isinstance(value, (tuple, list)):
            start, end = _offset_pair(value)
            label_value = _MISSING
            span_id_value = _MISSING
            text_hash_value = _MISSING
        else:
            start, end = _offset_pair(value)
            label_value = _field(value, ("label", "canonical_label", "entity_type"))
            span_id_value = _field(value, ("span_id", "id", "entity_id"))
            text_hash_value = _field(value, ("text_hash", "source_hash", "value_hash"))
            raw_span_document_id = _field(
                value,
                ("document_id", "document_hash", "doc_id"),
            )
            if raw_span_document_id is not _MISSING:
                span_document_id = _document_identifier(
                    raw_span_document_id,
                    hash_secret=hash_secret,
                )
                if span_document_id != normalized_document_id:
                    raise EvidenceBindingError(
                        "span document_id does not match relation"
                    )
        normalized_span_id = (
            None
            if span_id_value is _MISSING or span_id_value is None
            else _digest_identifier(
                span_id_value,
                namespace="relation-span",
                hash_secret=hash_secret,
            )
        )
        span = cls(
            document_id=normalized_document_id,
            start=start,
            end=end,
            label=(None if label_value is _MISSING else label_value),
            span_id=normalized_span_id,
            text_hash=(None if text_hash_value is _MISSING else text_hash_value),
        )
        _validate_document_bounds(span, document_length)
        return span

    @property
    def offset(self) -> tuple[int, int]:
        """Return the half-open source offset."""

        return self.start, self.end

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible span without source text."""

        payload: dict[str, Any] = {
            "document_id": self.document_id,
            "start": self.start,
            "end": self.end,
        }
        if self.label is not None:
            payload["label"] = self.label
        if self.span_id is not None:
            payload["span_id"] = self.span_id
        if self.text_hash is not None:
            payload["text_hash"] = self.text_hash
        return payload


def _assertion_state(value: object) -> AssertionState:
    if isinstance(value, AssertionState):
        return value
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, Mapping) or not isinstance(value, str):
        direct = _field(
            value,
            ("assertion_state", "assertion_status", "status", "state"),
        )
        if direct is not _MISSING and direct is not value:
            return _assertion_state(direct)
        negation = _field(value, ("negation",))
        temporality = _field(value, ("temporality", "temporality_status"))
        certainty = _field(value, ("certainty", "uncertainty"))
        if negation is not _MISSING and negation is not None:
            negation_text = _string(negation, "assertion negation").casefold()
            if negation_text in {"negated", "refuted", "absent"}:
                return AssertionState.NEGATED
        if temporality is not _MISSING and temporality is not None:
            temporality_text = _string(temporality, "assertion temporality").casefold()
            if temporality_text in {"hypothetical", "conditional"}:
                return AssertionState.HYPOTHETICAL
            if temporality_text == "historical":
                return AssertionState.HISTORICAL
        if certainty is not _MISSING and certainty is not None:
            certainty_text = _string(certainty, "assertion certainty").casefold()
            if certainty_text in {"uncertain", "possible"}:
                return AssertionState.UNCERTAIN
        if any(
            item is not _MISSING and item is not None
            for item in (negation, temporality, certainty)
        ):
            return AssertionState.AFFIRMED
        raise EvidenceBindingError("assertion_state is required")
    if not isinstance(value, str):
        raise EvidenceBindingError("assertion_state is required")
    try:
        return AssertionState(_string(value, "assertion_state").casefold())
    except (TypeError, ValueError):
        raise EvidenceBindingError("assertion_state is unsupported") from None


def _coerce_document_length(value: object) -> int | None:
    if value is None:
        return None
    length = _integer(value, "document_length")
    if length < 0:
        raise EvidenceBindingError("document_length must be non-negative")
    return length


def _document_text_length(
    document_text: object,
    document_length: object,
) -> int | None:
    explicit_length = _coerce_document_length(document_length)
    if document_text is None:
        return explicit_length
    text = _string(document_text, "document_text")
    text_length = len(text)
    if explicit_length is not None and explicit_length != text_length:
        raise EvidenceBindingError("document_length does not match document_text")
    return text_length


def _evidence_items(value: object) -> tuple[object, ...]:
    if value is _MISSING or value is None:
        raise EvidenceBindingError("relation evidence spans are required")
    if isinstance(value, EvidenceSpan):
        return (value,)
    if isinstance(value, Mapping):
        nested = _field(
            value,
            ("evidence_spans", "spans", "offsets", "sentences"),
        )
        if nested is not _MISSING and nested is not value:
            return _evidence_items(nested)
        return (value,)
    if isinstance(value, (str, bytes, bytearray)):
        raise EvidenceBindingError("relation evidence spans must be iterable")
    if isinstance(value, (tuple, list)) and len(value) == 2:
        first, second = value
        if type(first) is int and type(second) is int:
            return (value,)
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except Exception:
        raise EvidenceBindingError("relation evidence spans must be iterable") from None
    if not items:
        raise EvidenceBindingError("relation evidence spans are required")
    return items


def _span_key(span: EvidenceSpan) -> tuple[Any, ...]:
    return (
        span.document_id,
        span.start,
        span.end,
        span.label or "",
        span.span_id or "",
        span.text_hash or "",
    )


@dataclass(frozen=True, slots=True)
class GuardedRelation:
    """An immutable relation record that is safe to enter review workflows.

    Construction requires two endpoint spans, at least one linking evidence
    span, an explicit assertion state, and one document identifier.  The
    ``requires_clinician_review`` and ``autonomous_decision`` fields are fixed
    guardrails rather than caller-controlled policy switches.
    """

    relation_type: str
    document_id: str
    head: EvidenceSpan
    tail: EvidenceSpan
    evidence_spans: tuple[EvidenceSpan, ...]
    assertion_state: AssertionState
    confidence: float | None = None
    requires_clinician_review: bool = True
    autonomous_decision: bool = False
    schema_version: int = EVIDENCE_BINDING_SCHEMA_VERSION
    advisory: str = EVIDENCE_BINDING_ADVISORY

    def __post_init__(self) -> None:
        relation_type = _relation_type(self.relation_type)
        document_id = _document_identifier(self.document_id)
        if type(self.head) is not EvidenceSpan or type(self.tail) is not EvidenceSpan:
            raise EvidenceBindingError("relation endpoint spans are required")
        if self.head.document_id != document_id or self.tail.document_id != document_id:
            raise EvidenceBindingError("relation endpoint document_id does not match")
        if self.head.offset == self.tail.offset:
            raise EvidenceBindingError("relation endpoints must differ")
        if isinstance(self.evidence_spans, (str, bytes, bytearray)):
            raise EvidenceBindingError("relation evidence spans must be iterable")
        try:
            evidence = tuple(self.evidence_spans)
        except Exception:
            raise EvidenceBindingError(
                "relation evidence spans must be iterable"
            ) from None
        if not evidence or any(type(span) is not EvidenceSpan for span in evidence):
            raise EvidenceBindingError("relation evidence spans are required")
        if any(span.document_id != document_id for span in evidence):
            raise EvidenceBindingError("relation evidence document_id does not match")
        if type(self.schema_version) is not int or (
            self.schema_version != EVIDENCE_BINDING_SCHEMA_VERSION
        ):
            raise EvidenceBindingError("unsupported evidence binding schema")
        if self.advisory != EVIDENCE_BINDING_ADVISORY:
            raise EvidenceBindingError("unsupported evidence binding advisory")
        assertion_state = _assertion_state(self.assertion_state)
        confidence = (
            None
            if self.confidence is None
            else _probability(self.confidence, "confidence")
        )
        if type(self.requires_clinician_review) is not bool:
            raise EvidenceBindingError("requires_clinician_review must be a boolean")
        if self.requires_clinician_review is not True:
            raise EvidenceBindingError(
                "guarded relations always require clinician review"
            )
        if type(self.autonomous_decision) is not bool:
            raise EvidenceBindingError("autonomous_decision must be a boolean")
        if self.autonomous_decision is not False:
            raise EvidenceBindingError(
                "guarded relations cannot be autonomous decisions"
            )
        object.__setattr__(self, "relation_type", relation_type)
        object.__setattr__(self, "document_id", document_id)
        object.__setattr__(
            self, "evidence_spans", tuple(sorted(evidence, key=_span_key))
        )
        object.__setattr__(self, "assertion_state", assertion_state)
        object.__setattr__(self, "confidence", confidence)

    @property
    def head_span(self) -> EvidenceSpan:
        """Return the source span for the directed relation head."""

        return self.head

    @property
    def tail_span(self) -> EvidenceSpan:
        """Return the source span for the directed relation tail."""

        return self.tail

    @property
    def evidence(self) -> tuple[EvidenceSpan, ...]:
        """Return the sorted linking evidence spans."""

        return self.evidence_spans

    @property
    def document_hash(self) -> str:
        """Return the opaque document identifier used by the record."""

        return self.document_id

    @property
    def assertion_status(self) -> str:
        """Return the controlled assertion state as a string."""

        return self.assertion_state.value

    def stable_key(self) -> tuple[Any, ...]:
        """Return the deterministic ordering key for this relation."""

        return (
            self.document_id,
            self.relation_type,
            self.head.start,
            self.head.end,
            self.tail.start,
            self.tail.end,
            self.assertion_state.value,
            tuple(_span_key(span) for span in self.evidence_spans),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible, raw-value-free relation record."""

        return {
            "schema_version": self.schema_version,
            "relation_type": self.relation_type,
            "document_id": self.document_id,
            "head": self.head.to_dict(),
            "tail": self.tail.to_dict(),
            "evidence_spans": [span.to_dict() for span in self.evidence_spans],
            "assertion_state": self.assertion_state.value,
            "confidence": self.confidence,
            "requires_clinician_review": self.requires_clinician_review,
            "autonomous_decision": self.autonomous_decision,
            "provenance": {
                "endpoint_span_count": 2,
                "evidence_span_count": len(self.evidence_spans),
                "document_id": self.document_id,
            },
            "advisory": self.advisory,
        }

    def to_json(self) -> str:
        """Return deterministic compact JSON without source values."""

        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )


def _relation_document_id(
    relation: object,
    *,
    explicit: object,
    head: object,
    tail: object,
    hash_secret: object,
) -> str:
    if explicit is not _MISSING:
        return _document_identifier(explicit, hash_secret=hash_secret)
    relation_document = _field(
        relation,
        ("document_id", "document_hash", "doc_id", "source_document_id"),
    )
    if relation_document is _MISSING:
        nested_relation = _field(relation, ("relation", "candidate"))
        if nested_relation is not _MISSING and nested_relation is not relation:
            relation_document = _field(
                nested_relation,
                ("document_id", "document_hash", "doc_id", "source_document_id"),
            )
    if relation_document is not _MISSING and relation_document is not None:
        return _document_identifier(relation_document, hash_secret=hash_secret)
    candidates: list[str] = []
    for value in (head, tail):
        raw = _field(value, ("document_id", "document_hash", "doc_id"))
        if raw is not _MISSING and raw is not None:
            candidates.append(_document_identifier(raw, hash_secret=hash_secret))
    if not candidates:
        raise EvidenceBindingError("document_id is required")
    if len(set(candidates)) != 1:
        raise EvidenceBindingError("relation document identifiers must agree")
    return candidates[0]


def _relation_payload(relation: object) -> object:
    """Return the candidate inside an assertion wrapper, when present."""

    nested_relation = _field(relation, ("relation", "candidate"))
    if nested_relation is not _MISSING and nested_relation is not relation:
        return nested_relation
    return relation


def _relation_field(
    relation: object,
    payload: object,
    names: Sequence[str],
) -> object:
    """Read a field from a wrapper first and its underlying candidate second."""

    value = _field(relation, names)
    if value is _MISSING and payload is not relation:
        value = _field(payload, names)
    return value


def bind_relation_evidence(
    relation: object,
    *,
    document_id: object = _MISSING,
    head: object = _MISSING,
    tail: object = _MISSING,
    evidence_spans: object = _MISSING,
    evidence: object = _MISSING,
    assertion_state: object = _MISSING,
    confidence: object = _MISSING,
    document_text: object = None,
    document_length: object = None,
    hash_secret: str | bytes | None = None,
) -> GuardedRelation:
    """Bind one existing relation candidate to mandatory source evidence.

    Args:
        relation: Existing relation object or mapping.  Its relation type,
            endpoints, assertion state, score, and evidence are read only when
            the corresponding explicit keyword is omitted.
        document_id: Required document identifier, unless present on the
            relation or both endpoint spans.  Raw identifiers are hashed before
            storage; opaque digest identifiers are preserved.
        head: Optional endpoint span override.
        tail: Optional endpoint span override.
        evidence_spans: One or more linking spans or offset pairs.  The
            ``evidence`` alias is accepted for callers using the shorter
            contract name.
        assertion_state: Explicit controlled state, or a state supplied by the
            relation.  Missing state is rejected rather than inferred.
        confidence: Optional finite score in ``[0, 1]``.
        document_text: Optional source used only to validate span bounds.  It
            is never stored or returned.
        document_length: Optional equivalent bound when source text is not
            available.
        hash_secret: Optional HMAC key for raw document identifiers.

    Returns:
        An immutable :class:`GuardedRelation` safe to pass to a summary or
        review workflow.

    Raises:
        EvidenceBindingError: If any mandatory provenance field is missing or
            malformed.
    """

    if isinstance(relation, GuardedRelation):
        if any(
            value is not _MISSING
            for value in (
                document_id,
                head,
                tail,
                evidence_spans,
                evidence,
                assertion_state,
                confidence,
            )
        ):
            raise EvidenceBindingError("bound relation overrides are not supported")
        length = _document_text_length(document_text, document_length)
        validate_guarded_relation(relation, document_length=length)
        return relation

    payload = _relation_payload(relation)
    relation_type = _relation_field(
        relation,
        payload,
        ("relation_type", "relation_label", "type", "label"),
    )
    if relation_type is _MISSING:
        raise EvidenceBindingError("relation_type is required")

    raw_head = _relation_field(
        relation,
        payload,
        ("head", "head_span", "source", "subject"),
    )
    raw_tail = _relation_field(
        relation,
        payload,
        ("tail", "tail_span", "attribute", "target", "object"),
    )
    if head is not _MISSING:
        raw_head = head
    if tail is not _MISSING:
        raw_tail = tail
    if raw_head is _MISSING or raw_tail is _MISSING:
        raise EvidenceBindingError("relation endpoint spans are required")

    normalized_document_id = _relation_document_id(
        relation,
        explicit=document_id,
        head=raw_head,
        tail=raw_tail,
        hash_secret=hash_secret,
    )
    length = _document_text_length(document_text, document_length)
    head_span = EvidenceSpan.from_obj(
        raw_head,
        document_id=normalized_document_id,
        document_length=length,
        hash_secret=hash_secret,
    )
    tail_span = EvidenceSpan.from_obj(
        raw_tail,
        document_id=normalized_document_id,
        document_length=length,
        hash_secret=hash_secret,
    )

    raw_evidence = _relation_field(
        relation,
        payload,
        (
            "evidence_spans",
            "relation_evidence",
            "evidence",
            "supporting_spans",
            "evidence_offsets",
            "evidence_sentence_offsets",
        ),
    )
    if evidence_spans is not _MISSING and evidence is not _MISSING:
        raise EvidenceBindingError(
            "evidence_spans and evidence cannot both be supplied"
        )
    if evidence_spans is not _MISSING:
        raw_evidence = evidence_spans
    elif evidence is not _MISSING:
        raw_evidence = evidence
    normalized_evidence = tuple(
        EvidenceSpan.from_obj(
            item,
            document_id=normalized_document_id,
            document_length=length,
            hash_secret=hash_secret,
        )
        for item in _evidence_items(raw_evidence)
    )

    raw_assertion = _relation_field(
        relation,
        payload,
        ("assertion_state", "assertion_status", "status", "state", "assertion"),
    )
    if assertion_state is not _MISSING:
        raw_assertion = assertion_state
    if raw_assertion is _MISSING or raw_assertion is None:
        raise EvidenceBindingError("assertion_state is required")
    normalized_assertion = _assertion_state(raw_assertion)

    raw_confidence = _relation_field(
        relation,
        payload,
        ("confidence", "score"),
    )
    if confidence is not _MISSING:
        raw_confidence = confidence
    normalized_confidence = (
        None
        if raw_confidence is _MISSING or raw_confidence is None
        else _probability(raw_confidence, "confidence")
    )
    return GuardedRelation(
        relation_type=_relation_type(relation_type),
        document_id=normalized_document_id,
        head=head_span,
        tail=tail_span,
        evidence_spans=normalized_evidence,
        assertion_state=normalized_assertion,
        confidence=normalized_confidence,
    )


def validate_guarded_relation(
    relation: object,
    *,
    document_length: int | None = None,
) -> GuardedRelation:
    """Validate an already-bound relation before a workflow accepts it."""

    if type(relation) is not GuardedRelation:
        raise EvidenceBindingError(
            "summary and review workflows require bound relation records"
        )
    length = _coerce_document_length(document_length)
    _validate_document_bounds(relation.head, length)
    _validate_document_bounds(relation.tail, length)
    for span in relation.evidence_spans:
        _validate_document_bounds(span, length)
    return relation


def require_guarded_relations(
    relations: Iterable[GuardedRelation],
    *,
    workflow: str = "review",
    document_length: int | None = None,
) -> tuple[GuardedRelation, ...]:
    """Reject unbound records before a summary or review workflow.

    The returned tuple is sorted by source document, relation type, endpoint
    offsets, assertion state, and evidence offsets, so changing input order
    cannot change a workflow payload.
    """

    if workflow not in {"summary", "review"}:
        raise EvidenceBindingError("workflow must be summary or review")
    if isinstance(relations, (str, bytes, bytearray, Mapping)):
        raise EvidenceBindingError("workflow relation records must be iterable")
    try:
        records = tuple(relations)
    except Exception:
        raise EvidenceBindingError(
            "workflow relation records must be iterable"
        ) from None
    validated = tuple(
        validate_guarded_relation(record, document_length=document_length)
        for record in records
    )
    return tuple(sorted(validated, key=GuardedRelation.stable_key))


def bind_relation_records(
    relations: Iterable[object],
    *,
    document_id: object = _MISSING,
    evidence_spans: object = _MISSING,
    evidence: object = _MISSING,
    assertion_state: object = _MISSING,
    document_text: object = None,
    document_length: object = None,
    hash_secret: str | bytes | None = None,
) -> tuple[GuardedRelation, ...]:
    """Bind and deterministically order a collection of relation candidates."""

    if isinstance(relations, (str, bytes, bytearray, Mapping)):
        raise EvidenceBindingError("relation records must be iterable")
    try:
        records = tuple(relations)
    except Exception:
        raise EvidenceBindingError("relation records must be iterable") from None
    bound = tuple(
        bind_relation_evidence(
            relation,
            document_id=document_id,
            evidence_spans=evidence_spans,
            evidence=evidence,
            assertion_state=assertion_state,
            document_text=document_text,
            document_length=document_length,
            hash_secret=hash_secret,
        )
        for relation in records
    )
    return tuple(sorted(bound, key=GuardedRelation.stable_key))


def validate_relation_evidence(
    relation: object,
    **kwargs: Any,
) -> GuardedRelation:
    """Validate and bind one candidate using :func:`bind_relation_evidence`."""

    return bind_relation_evidence(relation, **kwargs)


# Names used by downstream relation producers and callers that prefer the
# shorter contract vocabulary.  They are aliases, not separate record types.
RelationEvidenceSpan = EvidenceSpan
RelationEndpoint = EvidenceSpan
RelationEvidence = EvidenceSpan
BoundRelation = GuardedRelation
EvidenceBinding = GuardedRelation
RelationEvidenceBinding = GuardedRelation
RelationEvidenceBindingError = EvidenceBindingError
bind_relation = bind_relation_evidence
bind_relations = bind_relation_records
require_relation_evidence = require_guarded_relations
validate_relation = validate_guarded_relation


__all__ = [
    "ASSERTION_STATE_VALUES",
    "EVIDENCE_BINDING_ADVISORY",
    "EVIDENCE_BINDING_SCHEMA_VERSION",
    "AssertionState",
    "BoundRelation",
    "EvidenceBinding",
    "EvidenceBindingError",
    "EvidenceSpan",
    "GuardedRelation",
    "RelationEndpoint",
    "RelationEvidence",
    "RelationEvidenceBinding",
    "RelationEvidenceBindingError",
    "RelationEvidenceSpan",
    "bind_relation",
    "bind_relation_evidence",
    "bind_relation_records",
    "bind_relations",
    "hash_document_id",
    "require_guarded_relations",
    "require_relation_evidence",
    "validate_guarded_relation",
    "validate_relation",
    "validate_relation_evidence",
]
