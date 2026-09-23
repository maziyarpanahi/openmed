"""Deterministic, value-free provenance for guarded clinical outputs.

This module binds a summary or clinical-NLI decision to the evidence that was
available, the model and policy that produced it, and the human-review state
that governs its use.  The public records are deliberately allow-listed:
generated output, source text, prompts, reviewer notes, paths, and arbitrary
metadata are hashed or omitted before a record is constructed.

The implementation is local-first and uses only the Python standard library.
It does not infer clinical facts, approve a clinical output, or make a network
call.  A record with missing evidence, changed inputs, missing model/policy
identity, or an incomplete review transition remains serializable as a blocked
review artifact so the gap is visible without exposing the protected values.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final

__all__ = [
    "GUARDED_PROVENANCE_DISCLAIMER",
    "GUARDED_PROVENANCE_RECORD_TYPE",
    "GUARDED_PROVENANCE_SCHEMA_VERSION",
    "REVIEW_STATES",
    "EvidenceReference",
    "GuardedProvenanceError",
    "GuardedProvenanceManifest",
    "GuardedProvenanceRecord",
    "IntegritySummary",
    "ModelProvenance",
    "ProvenanceIntegrityReport",
    "ProvenanceManifest",
    "ProvenanceRecord",
    "ReviewState",
    "ReviewStatus",
    "ReviewTransition",
    "build_guarded_provenance",
    "build_guarded_provenance_manifest",
    "build_guarded_provenance_record",
    "build_provenance_manifest",
    "build_provenance_record",
    "check_guarded_provenance",
    "export_guarded_provenance",
    "fingerprint_input",
    "fingerprint_policy",
    "load_guarded_provenance_manifest",
    "validate_guarded_provenance",
    "verify_guarded_provenance",
    "write_guarded_provenance_manifest",
]


GUARDED_PROVENANCE_SCHEMA_VERSION: Final[int] = 1
GUARDED_PROVENANCE_RECORD_TYPE: Final[str] = (
    "openmed.clinical.guarded_output_provenance"
)
GUARDED_PROVENANCE_DISCLAIMER: Final[str] = (
    "Guarded clinical-output provenance is value-free assistive metadata for "
    "human review; it is not a clinical decision, compliance certification, "
    "diagnosis, or treatment recommendation."
)

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_HEX_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+@-]{0,255}$")
_TOKEN_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_MISSING = object()
_SUSPICIOUS_IDENTIFIER_PARTS = frozenset(
    {
        "account",
        "credential",
        "mrn",
        "patient",
        "password",
        "phone",
        "secret",
        "ssn",
        "token",
    }
)


class GuardedProvenanceError(ValueError):
    """Raised when guarded provenance metadata cannot be validated safely."""


class ReviewState(str, Enum):
    """String constants for the guarded human-review state machine."""

    QUEUED = "queued"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"
    REOPENED = "reopened"
    ABSTAINED = "abstained"

    def __str__(self) -> str:
        """Return the stable serialized state value."""

        return self.value


# ``ReviewStatus`` is retained as a readable compatibility name for callers
# that describe the same finite state machine in status terminology.
ReviewStatus = ReviewState


REVIEW_STATES: Final[frozenset[str]] = frozenset(
    {
        ReviewState.QUEUED,
        ReviewState.IN_REVIEW,
        ReviewState.APPROVED,
        ReviewState.REJECTED,
        ReviewState.NEEDS_REVISION,
        ReviewState.REOPENED,
        ReviewState.ABSTAINED,
    }
)

_REVIEW_ALIASES: Final[dict[str, str]] = {
    "pending": ReviewState.QUEUED,
    "review": ReviewState.IN_REVIEW,
    "in-review": ReviewState.IN_REVIEW,
    "needs-review": ReviewState.QUEUED,
    "needs_revision": ReviewState.NEEDS_REVISION,
    "re-opened": ReviewState.REOPENED,
}
_ALLOWED_TRANSITIONS: Final[dict[str, frozenset[str]]] = {
    ReviewState.QUEUED: frozenset({ReviewState.IN_REVIEW, ReviewState.ABSTAINED}),
    ReviewState.IN_REVIEW: frozenset(
        {
            ReviewState.APPROVED,
            ReviewState.REJECTED,
            ReviewState.NEEDS_REVISION,
            ReviewState.ABSTAINED,
        }
    ),
    ReviewState.APPROVED: frozenset({ReviewState.REOPENED}),
    ReviewState.REJECTED: frozenset({ReviewState.REOPENED}),
    ReviewState.NEEDS_REVISION: frozenset({ReviewState.IN_REVIEW}),
    ReviewState.REOPENED: frozenset({ReviewState.IN_REVIEW}),
    ReviewState.ABSTAINED: frozenset({ReviewState.REOPENED}),
}

_OUTPUT_VALUE_FIELDS: Final[tuple[str, ...]] = (
    "generated_text",
    "output_text",
    "summary_text",
    "clinical_text",
    "text",
    "summary",
    "output",
    "hypothesis",
)
_INPUT_VALUE_FIELDS: Final[tuple[str, ...]] = (
    "input_text",
    "source_text",
    "source",
    "document_text",
    "document",
    "premise",
    "input",
)
_EVIDENCE_ID_FIELDS: Final[tuple[str, ...]] = (
    "evidence_ids",
    "evidence_refs",
    "evidence_references",
    "citations",
    "sources",
)


def _safe_text(value: Any) -> str | None:
    """Convert a scalar to text without allowing conversion errors to escape."""

    if value is None:
        return None
    try:
        return str(value)
    except Exception:
        return None


def _qualified_type(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _canonical_value(value: Any, *, depth: int = 0) -> Any:
    """Return a stable hash-only representation of an arbitrary value.

    Unsupported objects are represented by their type, never by ``repr``.
    This matters because an exception or a hash fallback must not accidentally
    copy a custom object's protected ``__repr__`` into a report.
    """

    if depth > 32:
        return ["object", _qualified_type(value), "depth_limit"]
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", int(value)]
    if isinstance(value, float):
        if math.isnan(value):
            return ["float", "nan"]
        if math.isinf(value):
            return ["float", "-inf" if value < 0 else "inf"]
        return ["float", float(value)]
    if isinstance(value, str):
        return ["string", _safe_text(value) or ""]
    if isinstance(value, bytes):
        return ["bytes", bytes(value).hex()]
    if isinstance(value, Mapping):
        items: list[list[Any]] = []
        try:
            pairs = tuple(value.items())
        except Exception:
            return ["mapping", _qualified_type(value), "unreadable"]
        for key, item in pairs:
            key_text = _safe_text(key)
            items.append(
                [
                    "" if key_text is None else key_text,
                    _canonical_value(item, depth=depth + 1),
                ]
            )
        items.sort(key=lambda item: str(item[0]))
        return ["mapping", items]
    if isinstance(value, (list, tuple)):
        return [
            "sequence",
            [_canonical_value(item, depth=depth + 1) for item in value],
        ]
    if isinstance(value, (set, frozenset)):
        items = [_canonical_value(item, depth=depth + 1) for item in value]
        items.sort(key=_canonical_json)
        return ["set", items]
    to_dict = _read(value, ("to_dict",), default=_MISSING)
    if callable(to_dict):
        try:
            converted = to_dict()
        except Exception:
            converted = _MISSING
        if converted is not _MISSING and converted is not value:
            return ["object", _qualified_type(value), _canonical_value(converted)]
    return ["object", _qualified_type(value)]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonical_value(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _hash_payload(value: Any, domain: str) -> str:
    payload = {"domain": domain, "value": _canonical_value(value)}
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def _normalise_digest(value: Any, domain: str) -> str | None:
    if value is None or value is _MISSING:
        return None
    text = _safe_text(value)
    if text is not None:
        candidate = text.strip()
        if _DIGEST_RE.fullmatch(candidate):
            return candidate
        if _HEX_DIGEST_RE.fullmatch(candidate):
            return f"sha256:{candidate}"
    return _hash_payload(value, domain)


def _fingerprint(value: Any, domain: str) -> str | None:
    return _normalise_digest(value, domain)


def fingerprint_input(value: Any) -> str:
    """Return a deterministic SHA-256 fingerprint for local input content.

    The value is consumed only while deriving the digest and is never retained
    by this module.  Callers should prefer passing a precomputed digest when a
    protected source value is already held by a separate local boundary.
    """

    result = _fingerprint(value, "guarded-clinical-input")
    if result is None:  # pragma: no cover - the public function rejects None
        raise GuardedProvenanceError("input fingerprint requires a value")
    return result


def fingerprint_policy(policy: Any) -> str:
    """Return a value-free fingerprint for a local policy descriptor."""

    result = _policy_fingerprint(policy, None)
    if result is None:
        raise GuardedProvenanceError("policy fingerprint requires a policy")
    return result


def _read(value: Any, names: Sequence[str], *, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, Mapping):
        for name in names:
            try:
                if name in value:
                    return value[name]
            except Exception:
                continue
        return default
    for name in names:
        try:
            candidate = getattr(value, name)
        except Exception:
            continue
        if not callable(candidate):
            return candidate
    return default


def _safe_identifier(value: Any) -> str | None:
    text = _safe_text(value)
    if text is None:
        return None
    candidate = text.strip()
    if not _IDENTIFIER_RE.fullmatch(candidate):
        return None
    parts = frozenset(re.split(r"[._:/+@-]+", candidate.casefold()))
    if parts & _SUSPICIOUS_IDENTIFIER_PARTS:
        return None
    return candidate


def _safe_token(value: Any, default: str) -> str:
    text = _safe_text(value)
    if text is None:
        return default
    candidate = text.strip().casefold().replace("-", "_")
    return candidate if _TOKEN_RE.fullmatch(candidate) else default


def _opaque_identifier(value: Any, domain: str) -> str | None:
    if value is None or value is _MISSING:
        return None
    digest = _normalise_digest(value, domain)
    return digest


def _offset_pair(value: Any) -> tuple[int, int] | None:
    if value is None or value is _MISSING:
        return None
    if isinstance(value, Mapping):
        start = _read(value, ("start", "source_start"), default=_MISSING)
        end = _read(value, ("end", "source_end"), default=_MISSING)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise GuardedProvenanceError("source offsets must contain two integers")
        start, end = value
    else:
        start = _read(value, ("start", "source_start"), default=_MISSING)
        end = _read(value, ("end", "source_end"), default=_MISSING)
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end <= start
    ):
        raise GuardedProvenanceError("source offsets must satisfy 0 <= start < end")
    return int(start), int(end)


def _normalise_review_state(value: Any) -> str:
    if isinstance(value, ReviewState):
        return value.value
    text = _safe_text(value)
    if text is None:
        raise GuardedProvenanceError("review state is required")
    state = text.strip().casefold().replace("-", "_")
    state = _REVIEW_ALIASES.get(state, state)
    if state not in REVIEW_STATES:
        raise GuardedProvenanceError("review state is unsupported")
    return state


def _reason_code(value: Any) -> str:
    if value is None or value is _MISSING:
        return "unspecified"
    return _safe_token(value, "unspecified")


@dataclass(frozen=True, slots=True)
class ModelProvenance:
    """Safe model identity and descriptor fingerprint for one output."""

    model_id: str | None = None
    revision: str | None = None
    model_fingerprint: str | None = None
    tokenizer_id: str | None = None
    tokenizer_fingerprint: str | None = None

    def __post_init__(self) -> None:
        model_id = _safe_identifier(self.model_id)
        revision = _safe_identifier(self.revision)
        tokenizer_id = _safe_identifier(self.tokenizer_id)
        fingerprint = _normalise_digest(self.model_fingerprint, "model")
        tokenizer_fingerprint = _normalise_digest(
            self.tokenizer_fingerprint,
            "tokenizer",
        )
        if fingerprint is None and any(
            value is not None for value in (model_id, revision)
        ):
            fingerprint = _hash_payload(
                {"model_id": model_id, "revision": revision},
                "guarded-clinical-model",
            )
        if tokenizer_fingerprint is None and tokenizer_id is not None:
            tokenizer_fingerprint = _hash_payload(
                {"tokenizer_id": tokenizer_id},
                "guarded-clinical-tokenizer",
            )
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "model_fingerprint", fingerprint)
        object.__setattr__(self, "tokenizer_id", tokenizer_id)
        object.__setattr__(self, "tokenizer_fingerprint", tokenizer_fingerprint)

    @property
    def model_hash(self) -> str | None:
        """Alias for the model descriptor fingerprint."""

        return self.model_fingerprint

    @property
    def present(self) -> bool:
        """Return whether a reproducible model identity is available."""

        return self.model_id is not None and self.model_fingerprint is not None

    def to_dict(self) -> dict[str, Any]:
        """Return the fixed, value-free model representation."""

        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "model_fingerprint": self.model_fingerprint,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
        }

    @classmethod
    def from_value(
        cls,
        model: Any = None,
        *,
        model_id: Any = None,
        revision: Any = None,
        model_fingerprint: Any = None,
        tokenizer_id: Any = None,
        tokenizer_fingerprint: Any = None,
    ) -> "ModelProvenance":
        """Build model metadata from a local descriptor without copying it."""

        source = model
        resolved_model_id = (
            model_id
            if model_id is not None
            else _read(source, ("model_id", "id", "name"), default=None)
        )
        if resolved_model_id is None and isinstance(source, str):
            resolved_model_id = source
        resolved_revision = (
            revision
            if revision is not None
            else _read(source, ("revision", "model_revision", "version"), default=None)
        )
        resolved_fingerprint = (
            model_fingerprint
            if model_fingerprint is not None
            else _read(
                source,
                ("model_fingerprint", "model_hash", "weights_hash", "digest"),
                default=None,
            )
        )
        if resolved_fingerprint is None and source is not None:
            resolved_fingerprint = _hash_payload(source, "guarded-clinical-model")
        resolved_tokenizer_id = (
            tokenizer_id
            if tokenizer_id is not None
            else _read(source, ("tokenizer_id", "tokenizer"), default=None)
        )
        resolved_tokenizer_fingerprint = (
            tokenizer_fingerprint
            if tokenizer_fingerprint is not None
            else _read(
                source,
                ("tokenizer_fingerprint", "tokenizer_hash"),
                default=None,
            )
        )
        return cls(
            model_id=resolved_model_id,
            revision=resolved_revision,
            model_fingerprint=resolved_fingerprint,
            tokenizer_id=resolved_tokenizer_id,
            tokenizer_fingerprint=resolved_tokenizer_fingerprint,
        )

    @classmethod
    def from_dict(cls, value: Any) -> "ModelProvenance":
        """Load a model record from its safe serialized representation."""

        if not isinstance(value, Mapping):
            raise GuardedProvenanceError("model provenance must be an object")
        return cls(
            model_id=_read(value, ("model_id",), default=None),
            revision=_read(value, ("revision", "model_revision"), default=None),
            model_fingerprint=_read(
                value,
                ("model_fingerprint", "model_hash"),
                default=None,
            ),
            tokenizer_id=_read(value, ("tokenizer_id",), default=None),
            tokenizer_fingerprint=_read(
                value,
                ("tokenizer_fingerprint", "tokenizer_hash"),
                default=None,
            ),
        )


@dataclass(frozen=True, slots=True)
class EvidenceReference:
    """Opaque identity, offsets, and digest for one evidence item."""

    evidence_id: str
    source_offsets: tuple[int, int] | None = None
    evidence_hash: str | None = None
    kind: str = "source_span"

    def __post_init__(self) -> None:
        evidence_id = _opaque_identifier(self.evidence_id, "guarded-clinical-evidence")
        if evidence_id is None:
            raise GuardedProvenanceError("evidence identifier is required")
        offsets = _offset_pair(self.source_offsets)
        evidence_hash = _normalise_digest(self.evidence_hash, "evidence")
        kind = _safe_token(self.kind, "source_span")
        object.__setattr__(self, "evidence_id", evidence_id)
        object.__setattr__(self, "source_offsets", offsets)
        object.__setattr__(self, "evidence_hash", evidence_hash)
        object.__setattr__(self, "kind", kind)

    @property
    def id(self) -> str:
        """Alias for the opaque evidence identifier."""

        return self.evidence_id

    @property
    def input_hash(self) -> str | None:
        """Alias used by callers that model evidence as input fragments."""

        return self.evidence_hash

    @classmethod
    def from_value(cls, value: Any, *, index: int = 0) -> "EvidenceReference":
        """Convert a source/evidence object while dropping its protected value."""

        if isinstance(value, cls):
            return value
        raw_id: Any
        raw_offsets: Any
        raw_hash: Any
        raw_content: Any
        raw_kind: Any
        if isinstance(value, (str, bytes)):
            raw_id = value
            raw_offsets = _MISSING
            raw_hash = _MISSING
            raw_content = value
            raw_kind = "source_span"
        else:
            raw_id = _read(
                value,
                ("evidence_id", "id", "reference", "source_id", "citation_id"),
                default=_MISSING,
            )
            raw_offsets = _read(
                value,
                ("source_offsets", "source_offset", "offset", "span"),
                default=_MISSING,
            )
            if raw_offsets is _MISSING:
                start = _read(value, ("source_start", "start"), default=_MISSING)
                end = _read(value, ("source_end", "end"), default=_MISSING)
                if start is not _MISSING or end is not _MISSING:
                    raw_offsets = (start, end)
            raw_hash = _read(
                value,
                (
                    "evidence_hash",
                    "input_hash",
                    "content_hash",
                    "text_hash",
                    "value_hash",
                    "source_hash",
                ),
                default=_MISSING,
            )
            raw_content = _read(
                value,
                (
                    "evidence_text",
                    "source_text",
                    "text",
                    "content",
                    "value",
                    "surface",
                    "claim",
                    "premise",
                    "hypothesis",
                ),
                default=_MISSING,
            )
            raw_kind = _read(value, ("kind", "evidence_kind", "type"), default=None)
        if raw_id is _MISSING or raw_id is None:
            raw_id = raw_content if raw_content is not _MISSING else index
        if raw_hash is _MISSING or raw_hash is None:
            raw_hash = (
                raw_content
                if raw_content is not _MISSING and raw_content is not None
                else None
            )
        evidence_hash = _normalise_digest(raw_hash, "guarded-clinical-evidence")
        if evidence_hash is None:
            evidence_hash = _hash_payload(
                {"offsets": raw_offsets, "kind": raw_kind or "source_span"},
                "guarded-clinical-evidence",
            )
        return cls(
            evidence_id=_opaque_identifier(raw_id, "guarded-clinical-evidence")
            or _hash_payload({"index": index}, "guarded-clinical-evidence"),
            source_offsets=None if raw_offsets is _MISSING else raw_offsets,
            evidence_hash=evidence_hash,
            kind=raw_kind or "source_span",
        )

    @classmethod
    def from_dict(cls, value: Any) -> "EvidenceReference":
        """Load one opaque evidence reference from a serialized mapping."""

        if not isinstance(value, Mapping):
            raise GuardedProvenanceError("evidence reference must be an object")
        return cls(
            evidence_id=_read(value, ("evidence_id", "id"), default=None),
            source_offsets=_read(value, ("source_offsets",), default=None),
            evidence_hash=_read(
                value,
                ("evidence_hash", "input_hash", "text_hash"),
                default=None,
            ),
            kind=_read(value, ("kind",), default="source_span"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return only opaque evidence metadata."""

        offsets = self.source_offsets
        return {
            "evidence_id": self.evidence_id,
            "source_offsets": (
                None if offsets is None else {"start": offsets[0], "end": offsets[1]}
            ),
            "evidence_hash": self.evidence_hash,
            "kind": self.kind,
        }


@dataclass(frozen=True, slots=True)
class ReviewTransition:
    """One deterministic, privacy-safe human-review state transition."""

    sequence: int
    from_state: str
    to_state: str
    reviewer_fingerprint: str | None = None
    reason_code: str = "unspecified"

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise GuardedProvenanceError("review transition sequence is invalid")
        if self.sequence < 0:
            raise GuardedProvenanceError("review transition sequence is invalid")
        from_state = _normalise_review_state(self.from_state)
        to_state = _normalise_review_state(self.to_state)
        if to_state not in _ALLOWED_TRANSITIONS.get(from_state, frozenset()):
            raise GuardedProvenanceError("review transition is not allowed")
        reviewer = _opaque_identifier(
            self.reviewer_fingerprint,
            "guarded-clinical-reviewer",
        )
        object.__setattr__(self, "sequence", int(self.sequence))
        object.__setattr__(self, "from_state", from_state)
        object.__setattr__(self, "to_state", to_state)
        object.__setattr__(self, "reviewer_fingerprint", reviewer)
        object.__setattr__(self, "reason_code", _reason_code(self.reason_code))

    @property
    def reviewer_id(self) -> str | None:
        """Alias exposing only the reviewer fingerprint."""

        return self.reviewer_fingerprint

    @classmethod
    def from_value(cls, value: Any, *, sequence: int = 0) -> "ReviewTransition":
        """Build a transition from a mapping without retaining notes or identity."""

        if isinstance(value, cls):
            return value
        from_state = _read(
            value,
            ("from_state", "previous_state", "source_state"),
            default=ReviewState.QUEUED,
        )
        to_state = _read(
            value,
            ("to_state", "next_state", "state", "review_state"),
            default=_MISSING,
        )
        if to_state is _MISSING:
            raise GuardedProvenanceError("review transition target is required")
        raw_sequence = _read(value, ("sequence", "index", "order"), default=sequence)
        reviewer = _read(
            value,
            ("reviewer_fingerprint", "reviewer_id", "reviewer"),
            default=None,
        )
        reason = _read(value, ("reason_code", "reason"), default="unspecified")
        return cls(
            sequence=raw_sequence,
            from_state=from_state,
            to_state=to_state,
            reviewer_fingerprint=reviewer,
            reason_code=reason,
        )

    @classmethod
    def from_dict(cls, value: Any) -> "ReviewTransition":
        """Load a transition from its safe serialized representation."""

        if not isinstance(value, Mapping):
            raise GuardedProvenanceError("review transition must be an object")
        return cls.from_value(value)

    def to_dict(self) -> dict[str, Any]:
        """Return transition state and opaque reviewer metadata."""

        return {
            "sequence": self.sequence,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "reviewer_fingerprint": self.reviewer_fingerprint,
            "reason_code": self.reason_code,
        }


def _normalise_transitions(values: Any) -> tuple[ReviewTransition, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, Mapping, ReviewTransition)):
        candidates: tuple[Any, ...] = (values,)
    else:
        try:
            candidates = tuple(values)
        except Exception:
            raise GuardedProvenanceError(
                "review transitions must be iterable"
            ) from None

    transitions: list[ReviewTransition] = []
    for index, value in enumerate(candidates):
        transitions.append(ReviewTransition.from_value(value, sequence=index))
    transitions.sort(
        key=lambda item: (
            item.sequence,
            item.from_state,
            item.to_state,
            item.reviewer_fingerprint or "",
            item.reason_code,
        )
    )
    if len({item.sequence for item in transitions}) != len(transitions):
        raise GuardedProvenanceError("review transition sequences must be unique")
    current: str = ReviewState.QUEUED.value
    for transition in transitions:
        if transition.from_state != current:
            raise GuardedProvenanceError("review transition chain is not contiguous")
        current = transition.to_state
    return tuple(transitions)


@dataclass(frozen=True, slots=True)
class IntegritySummary:
    """Counts and opaque references describing provenance integrity gaps."""

    missing_evidence_count: int = 0
    missing_evidence_ids: tuple[str, ...] = ()
    changed_evidence_count: int = 0
    changed_evidence_ids: tuple[str, ...] = ()
    input_checked: bool = False
    input_changed: bool = False
    output_checked: bool = False
    output_changed: bool = False
    model_present: bool = False
    policy_present: bool = False
    input_present: bool = False
    output_present: bool = False
    manifest_hash_valid: bool = True
    record_hash_valid: bool = True
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        counts = (
            self.missing_evidence_count,
            self.changed_evidence_count,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts
        ):
            raise GuardedProvenanceError("integrity counts are invalid")
        if any(
            type(value) is not bool
            for value in (
                self.input_checked,
                self.input_changed,
                self.output_checked,
                self.output_changed,
                self.model_present,
                self.policy_present,
                self.input_present,
                self.output_present,
                self.manifest_hash_valid,
                self.record_hash_valid,
            )
        ):
            raise GuardedProvenanceError("integrity flags are invalid")
        missing = tuple(
            sorted(
                {
                    item
                    for item in (
                        _normalise_digest(value, "missing-evidence")
                        for value in self.missing_evidence_ids
                    )
                    if item is not None
                }
            )
        )
        changed = tuple(
            sorted(
                {
                    item
                    for item in (
                        _normalise_digest(value, "changed-evidence")
                        for value in self.changed_evidence_ids
                    )
                    if item is not None
                }
            )
        )
        reason_codes = tuple(
            sorted({_reason_code(value) for value in self.reason_codes})
        )
        object.__setattr__(self, "missing_evidence_ids", missing)
        object.__setattr__(self, "changed_evidence_ids", changed)
        object.__setattr__(self, "reason_codes", reason_codes)

    @property
    def ok(self) -> bool:
        """Return whether the record is complete and internally consistent."""

        return not self.reason_codes

    @property
    def valid(self) -> bool:
        """Alias for :attr:`ok`."""

        return self.ok

    def to_dict(self) -> dict[str, Any]:
        """Return counts, flags, digests, and stable reason codes only."""

        return {
            "missing_evidence_count": self.missing_evidence_count,
            "missing_evidence_ids": list(self.missing_evidence_ids),
            "changed_evidence_count": self.changed_evidence_count,
            "changed_evidence_ids": list(self.changed_evidence_ids),
            "input_checked": self.input_checked,
            "input_changed": self.input_changed,
            "output_checked": self.output_checked,
            "output_changed": self.output_changed,
            "model_present": self.model_present,
            "policy_present": self.policy_present,
            "input_present": self.input_present,
            "output_present": self.output_present,
            "manifest_hash_valid": self.manifest_hash_valid,
            "record_hash_valid": self.record_hash_valid,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "IntegritySummary":
        """Load serialized integrity metadata without trusting its values."""

        if not isinstance(value, Mapping):
            raise GuardedProvenanceError("integrity summary must be an object")
        return cls(
            missing_evidence_count=_read(value, ("missing_evidence_count",), default=0),
            missing_evidence_ids=tuple(
                _read(value, ("missing_evidence_ids",), default=()) or ()
            ),
            changed_evidence_count=_read(value, ("changed_evidence_count",), default=0),
            changed_evidence_ids=tuple(
                _read(value, ("changed_evidence_ids",), default=()) or ()
            ),
            input_checked=_read(value, ("input_checked",), default=False),
            input_changed=_read(value, ("input_changed",), default=False),
            output_checked=_read(value, ("output_checked",), default=False),
            output_changed=_read(value, ("output_changed",), default=False),
            model_present=_read(value, ("model_present",), default=False),
            policy_present=_read(value, ("policy_present",), default=False),
            input_present=_read(value, ("input_present",), default=False),
            output_present=_read(value, ("output_present",), default=False),
            manifest_hash_valid=_read(value, ("manifest_hash_valid",), default=True),
            record_hash_valid=_read(value, ("record_hash_valid",), default=True),
            reason_codes=tuple(_read(value, ("reason_codes",), default=()) or ()),
        )


def _record_reason_codes(
    *,
    missing_evidence_count: int,
    changed_evidence_count: int,
    input_present: bool,
    output_present: bool,
    model_present: bool,
    policy_present: bool,
    input_changed: bool,
    output_changed: bool,
    transitions: Sequence[ReviewTransition],
    review_state: str,
) -> tuple[str, ...]:
    reasons: set[str] = set()
    if missing_evidence_count:
        reasons.add("missing_evidence")
    if changed_evidence_count:
        reasons.add("changed_evidence")
    if not input_present:
        reasons.add("missing_input_hash")
    if not output_present:
        reasons.add("missing_output_hash")
    if not model_present:
        reasons.add("missing_model_identity")
    if not policy_present:
        reasons.add("missing_policy_fingerprint")
    if input_changed:
        reasons.add("changed_input")
    if output_changed:
        reasons.add("changed_output")
    if review_state != ReviewState.QUEUED and not transitions:
        reasons.add("missing_review_transition")
    return tuple(sorted(reasons))


@dataclass(frozen=True, slots=True)
class GuardedProvenanceRecord:
    """One value-free provenance record for a summary or NLI decision."""

    output_kind: str = "clinical_output"
    input_hash: str | None = None
    output_hash: str | None = None
    evidence_ids: tuple[str, ...] = ()
    evidence: tuple[EvidenceReference, ...] = ()
    model: ModelProvenance = field(default_factory=ModelProvenance)
    policy_fingerprint: str | None = None
    review_required: bool = True
    review_status: str = ReviewState.QUEUED
    review_transitions: tuple[ReviewTransition, ...] = ()
    integrity: IntegritySummary | None = None
    record_id: str | None = None
    record_hash: str | None = None
    schema_version: int = GUARDED_PROVENANCE_SCHEMA_VERSION
    disclaimer: str = GUARDED_PROVENANCE_DISCLAIMER

    def __post_init__(self) -> None:
        if self.schema_version != GUARDED_PROVENANCE_SCHEMA_VERSION:
            raise GuardedProvenanceError("unsupported guarded provenance schema")
        if self.disclaimer != GUARDED_PROVENANCE_DISCLAIMER:
            raise GuardedProvenanceError("guarded provenance disclaimer is required")
        if self.review_required is not True:
            raise GuardedProvenanceError(
                "guarded clinical outputs require human review"
            )
        output_kind = _safe_token(self.output_kind, "clinical_output")
        input_hash = _normalise_digest(self.input_hash, "guarded-clinical-input")
        output_hash = _normalise_digest(self.output_hash, "guarded-clinical-output")
        evidence = tuple(
            sorted(
                (
                    item
                    if isinstance(item, EvidenceReference)
                    else EvidenceReference.from_value(item, index=index)
                    for index, item in enumerate(tuple(self.evidence))
                ),
                key=lambda item: item.evidence_id,
            )
        )
        evidence_ids = tuple(
            sorted(
                {
                    item
                    for item in (
                        _opaque_identifier(value, "guarded-clinical-evidence")
                        for value in self.evidence_ids
                    )
                    if item is not None
                }
            )
        )
        if not evidence_ids and evidence:
            evidence_ids = tuple(item.evidence_id for item in evidence)
        model = (
            self.model
            if isinstance(self.model, ModelProvenance)
            else ModelProvenance.from_value(self.model)
        )
        policy_fingerprint = _normalise_digest(
            self.policy_fingerprint,
            "guarded-clinical-policy",
        )
        transitions = _normalise_transitions(self.review_transitions)
        review_status = _normalise_review_state(self.review_status)
        if transitions:
            derived_status = transitions[-1].to_state
            if review_status == ReviewState.QUEUED:
                review_status = derived_status
            elif review_status != derived_status:
                raise GuardedProvenanceError("review status does not match transitions")
        available_ids = {item.evidence_id for item in evidence}
        missing_ids = tuple(sorted(set(evidence_ids) - available_ids))
        missing_count = len(missing_ids) if evidence_ids else 1
        integrity = self.integrity
        if integrity is None:
            integrity = IntegritySummary(
                missing_evidence_count=missing_count,
                missing_evidence_ids=missing_ids,
                model_present=model.present,
                policy_present=policy_fingerprint is not None,
                input_present=input_hash is not None,
                output_present=output_hash is not None,
                reason_codes=_record_reason_codes(
                    missing_evidence_count=missing_count,
                    changed_evidence_count=0,
                    input_present=input_hash is not None,
                    output_present=output_hash is not None,
                    model_present=model.present,
                    policy_present=policy_fingerprint is not None,
                    input_changed=False,
                    output_changed=False,
                    transitions=transitions,
                    review_state=review_status,
                ),
            )
        record_id = self.record_id
        if record_id is None:
            record_id = _hash_payload(
                {
                    "output_kind": output_kind,
                    "input_hash": input_hash,
                    "output_hash": output_hash,
                    "evidence_ids": evidence_ids,
                    "model": model.to_dict(),
                    "policy_fingerprint": policy_fingerprint,
                    "review_status": review_status,
                    "review_transitions": [item.to_dict() for item in transitions],
                },
                "guarded-clinical-record-id",
            )
        else:
            record_id = _opaque_identifier(record_id, "guarded-clinical-record-id")
        if record_id is None:  # pragma: no cover - the generated path is non-empty
            raise GuardedProvenanceError("record identifier is required")
        record_hash = self.record_hash
        object.__setattr__(self, "output_kind", output_kind)
        object.__setattr__(self, "input_hash", input_hash)
        object.__setattr__(self, "output_hash", output_hash)
        object.__setattr__(self, "evidence_ids", evidence_ids)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "policy_fingerprint", policy_fingerprint)
        object.__setattr__(self, "review_status", review_status)
        object.__setattr__(self, "review_transitions", transitions)
        object.__setattr__(self, "integrity", integrity)
        object.__setattr__(self, "record_id", record_id)
        if record_hash is None:
            object.__setattr__(
                self,
                "record_hash",
                _hash_payload(self.hash_material(), "guarded-clinical-record"),
            )
        else:
            object.__setattr__(
                self,
                "record_hash",
                _normalise_digest(record_hash, "guarded-clinical-record"),
            )

    @property
    def review_state(self) -> str:
        """Alias for the current human-review state."""

        return self.review_status

    @property
    def requires_human_review(self) -> bool:
        """Return whether review is still required before clinical use."""

        return self.review_status != ReviewState.APPROVED or not self._integrity().ok

    @property
    def missing_evidence_count(self) -> int:
        """Return the number of evidence references that cannot be resolved."""

        return self._integrity().missing_evidence_count

    def _integrity(self) -> IntegritySummary:
        """Return the normalized integrity summary for this record."""

        if self.integrity is None:  # pragma: no cover - post-init normalizes it
            return IntegritySummary()
        return self.integrity

    def hash_material(self) -> dict[str, Any]:
        """Return the canonical record fields committed by ``record_hash``."""

        return {
            "schema_version": self.schema_version,
            "record_type": GUARDED_PROVENANCE_RECORD_TYPE,
            "record_id": self.record_id,
            "output_kind": self.output_kind,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "evidence_ids": list(self.evidence_ids),
            "evidence": [item.to_dict() for item in self.evidence],
            "model": self.model.to_dict(),
            "policy_fingerprint": self.policy_fingerprint,
            "review_required": self.review_required,
            "review_status": self.review_status,
            "review_transitions": [item.to_dict() for item in self.review_transitions],
            "integrity": self._integrity().to_dict(),
        }

    def compute_record_hash(self) -> str:
        """Recompute the record hash without serializing protected values."""

        return _hash_payload(self.hash_material(), "guarded-clinical-record")

    def verify_hash(self) -> bool:
        """Return whether this record's content hash is intact."""

        return self.record_hash == self.compute_record_hash()

    def to_dict(self) -> dict[str, Any]:
        """Return the fixed-shape, generated-text-free provenance record."""

        review = {
            "required": self.review_required,
            "status": self.review_status,
            "transitions": [item.to_dict() for item in self.review_transitions],
        }
        return {
            "schema_version": self.schema_version,
            "record_type": GUARDED_PROVENANCE_RECORD_TYPE,
            "record_id": self.record_id,
            "output_kind": self.output_kind,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "model": self.model.to_dict(),
            "policy_fingerprint": self.policy_fingerprint,
            "evidence_ids": list(self.evidence_ids),
            "evidence": [item.to_dict() for item in self.evidence],
            "review": review,
            "review_required": self.review_required,
            "review_status": self.review_status,
            "review_transitions": review["transitions"],
            "integrity": self._integrity().to_dict(),
            "record_hash": self.record_hash,
            "disclaimer": self.disclaimer,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "GuardedProvenanceRecord":
        """Load and structurally validate a serialized provenance record."""

        if not isinstance(value, Mapping):
            raise GuardedProvenanceError("provenance record must be an object")
        review = _read(value, ("review",), default={})
        if not isinstance(review, Mapping):
            raise GuardedProvenanceError("review metadata must be an object")
        transitions = _read(
            value,
            ("review_transitions",),
            default=_read(review, ("transitions",), default=()),
        )
        raw_integrity = _read(value, ("integrity",), default=_MISSING)
        return cls(
            output_kind=_read(value, ("output_kind",), default="clinical_output"),
            input_hash=_read(value, ("input_hash", "input_fingerprint"), default=None),
            output_hash=_read(
                value, ("output_hash", "output_fingerprint"), default=None
            ),
            evidence_ids=tuple(_read(value, ("evidence_ids",), default=()) or ()),
            evidence=tuple(
                EvidenceReference.from_dict(item)
                for item in (_read(value, ("evidence",), default=()) or ())
            ),
            model=ModelProvenance.from_dict(_read(value, ("model",), default={})),
            policy_fingerprint=_read(value, ("policy_fingerprint",), default=None),
            review_required=_read(
                value,
                ("review_required",),
                default=_read(review, ("required",), default=True),
            ),
            review_status=_read(
                value,
                ("review_status",),
                default=_read(review, ("status",), default=ReviewState.QUEUED),
            ),
            review_transitions=tuple(transitions or ()),
            integrity=(
                None
                if raw_integrity is _MISSING
                else IntegritySummary.from_dict(raw_integrity)
            ),
            record_id=_read(value, ("record_id",), default=None),
            record_hash=_read(value, ("record_hash",), default=None),
            schema_version=_read(
                value,
                ("schema_version",),
                default=GUARDED_PROVENANCE_SCHEMA_VERSION,
            ),
            disclaimer=_read(
                value,
                ("disclaimer",),
                default=GUARDED_PROVENANCE_DISCLAIMER,
            ),
        )


def _normalise_evidence_collection(value: Any) -> tuple[EvidenceReference, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        field_names = {
            "evidence_id",
            "id",
            "reference",
            "source_offsets",
            "source_start",
            "source_end",
            "text",
            "value",
            "evidence_hash",
        }
        try:
            is_single = bool(field_names.intersection(value.keys()))
        except Exception:
            is_single = True
        if is_single:
            candidates: tuple[Any, ...] = (value,)
        else:
            candidates = tuple(
                {"evidence_id": key, "value": item}
                for key, item in sorted(
                    value.items(),
                    key=lambda pair: _safe_text(pair[0]) or "",
                )
            )
    elif isinstance(value, (str, bytes, EvidenceReference)):
        candidates = (value,)
    else:
        try:
            candidates = tuple(value)
        except Exception:
            raise GuardedProvenanceError("evidence must be iterable") from None
    records = [
        item
        if isinstance(item, EvidenceReference)
        else EvidenceReference.from_value(item, index=index)
        for index, item in enumerate(candidates)
    ]
    unique: dict[str, EvidenceReference] = {}
    for record in records:
        unique.setdefault(record.evidence_id, record)
    return tuple(sorted(unique.values(), key=lambda item: item.evidence_id))


def _normalise_evidence_ids(value: Any) -> tuple[str, ...] | None:
    if value is None or value is _MISSING:
        return None
    if isinstance(value, (str, bytes)):
        values = (value,)
    elif isinstance(value, Mapping):
        named_id = _read(value, ("evidence_id", "id", "reference"), default=_MISSING)
        if named_id is _MISSING:
            values = tuple(value.keys())
        else:
            values = (named_id,)
    else:
        try:
            values = tuple(value)
        except Exception:
            values = (value,)
    result: set[str] = set()
    for item in values:
        if isinstance(item, Mapping):
            item = _read(item, ("evidence_id", "id", "reference"), default=item)
        digest = _opaque_identifier(item, "guarded-clinical-evidence")
        if digest is not None:
            result.add(digest)
    return tuple(sorted(result))


def _policy_fingerprint(policy: Any, explicit: Any) -> str | None:
    if explicit is not None and explicit is not _MISSING:
        return _normalise_digest(explicit, "guarded-clinical-policy")
    if policy is None or policy is _MISSING:
        return None
    direct = _read(policy, ("policy_fingerprint", "fingerprint"), default=_MISSING)
    if direct is not _MISSING and direct is not None:
        return _normalise_digest(direct, "guarded-clinical-policy")
    return _hash_payload(policy, "guarded-clinical-policy")


def _input_value_from_output(output: Any) -> Any:
    value = _read(output, _INPUT_VALUE_FIELDS, default=_MISSING)
    return None if value is _MISSING else value


def _output_value_from_output(output: Any) -> Any:
    if isinstance(output, Mapping):
        value = _read(output, _OUTPUT_VALUE_FIELDS, default=_MISSING)
        if value is not _MISSING:
            return value
    return output


def _model_from_args(
    model: Any,
    output: Any,
    *,
    model_id: Any,
    model_revision: Any,
    model_fingerprint: Any,
    tokenizer_id: Any,
) -> ModelProvenance:
    source = model
    if source is None:
        source = _read(output, ("model",), default=None)
    resolved_model_id = model_id
    if resolved_model_id is None:
        resolved_model_id = _read(output, ("model_id",), default=None)
    resolved_revision = model_revision
    if resolved_revision is None:
        resolved_revision = _read(output, ("model_revision", "revision"), default=None)
    resolved_fingerprint = model_fingerprint
    if resolved_fingerprint is None:
        resolved_fingerprint = _read(
            output,
            ("model_fingerprint", "model_hash", "weights_hash"),
            default=None,
        )
    resolved_tokenizer_id = tokenizer_id
    if resolved_tokenizer_id is None:
        resolved_tokenizer_id = _read(output, ("tokenizer_id",), default=None)
    return ModelProvenance.from_value(
        source,
        model_id=resolved_model_id,
        revision=resolved_revision,
        model_fingerprint=resolved_fingerprint,
        tokenizer_id=resolved_tokenizer_id,
    )


def build_guarded_provenance_record(
    output: Any = None,
    evidence: Any = (),
    *,
    model: Any = None,
    policy: Any = None,
    review_transitions: Any = None,
    output_kind: Any = None,
    record_id: Any = None,
    input_value: Any = _MISSING,
    input_hash: Any = _MISSING,
    expected_input_hash: Any = None,
    output_hash: Any = _MISSING,
    expected_output_hash: Any = None,
    evidence_ids: Any = None,
    review_status: Any = None,
    review_required: bool = True,
    model_id: Any = None,
    model_revision: Any = None,
    model_fingerprint: Any = None,
    tokenizer_id: Any = None,
    policy_fingerprint: Any = None,
) -> GuardedProvenanceRecord:
    """Build one guarded provenance record from a local output boundary.

    ``output`` and ``input_value`` may be protected strings or structured
    objects.  They are used only to derive SHA-256 digests.  If the output is a
    mapping, common ``*_text``/``input``/``evidence_ids`` fields are accepted as
    convenience aliases; all unrecognized fields are ignored by the emitted
    record.  Missing evidence and incomplete metadata produce a blocked,
    review-visible record rather than a record containing the protected value.

    Args:
        output: Generated summary/NLI output or a mapping carrying its safe
            boundary metadata.  The generated value is never serialized.
        evidence: Available evidence records, each with an opaque or local id,
            optional source offsets, and either an explicit digest or a value
            that can be hashed locally.
        model: Model descriptor or model identifier.  Only safe identifiers and
            descriptor fingerprints are emitted.
        policy: Local policy descriptor.  Only its SHA-256 fingerprint is
            emitted; policy values are never copied.
        review_transitions: Ordered mappings or :class:`ReviewTransition`
            values.  A record with no transition starts in ``queued`` state.
        expected_input_hash: Optional prior input digest used to detect a
            changed input at build time.
        expected_output_hash: Optional prior output digest used similarly.

    Returns:
        An immutable, deterministic :class:`GuardedProvenanceRecord`.
    """

    if review_required is not True:
        raise GuardedProvenanceError("guarded clinical outputs require human review")
    output_mapping = output if isinstance(output, Mapping) else None
    resolved_output_kind = (
        _read(
            output_mapping, ("output_kind", "kind", "task"), default="clinical_output"
        )
        if output_kind is None
        else output_kind
    )

    if input_hash is _MISSING:
        raw_input = input_value
        if raw_input is _MISSING:
            raw_input = _read(output_mapping, ("input_hash",), default=_MISSING)
        if raw_input is _MISSING:
            raw_input = _input_value_from_output(output)
        resolved_input_hash = (
            None
            if raw_input is None or raw_input is _MISSING
            else _fingerprint(raw_input, "guarded-clinical-input")
        )
    else:
        resolved_input_hash = _normalise_digest(
            input_hash,
            "guarded-clinical-input",
        )
    if output_hash is _MISSING:
        raw_output = _read(output_mapping, ("output_hash",), default=_MISSING)
        if raw_output is _MISSING:
            raw_output = _output_value_from_output(output)
        resolved_output_hash = (
            None
            if raw_output is None or raw_output is _MISSING
            else _fingerprint(raw_output, "guarded-clinical-output")
        )
    else:
        resolved_output_hash = _normalise_digest(
            output_hash,
            "guarded-clinical-output",
        )

    available_evidence = _normalise_evidence_collection(evidence)
    resolved_evidence_ids = _normalise_evidence_ids(evidence_ids)
    if resolved_evidence_ids is None:
        resolved_evidence_ids = _normalise_evidence_ids(
            _read(output_mapping, _EVIDENCE_ID_FIELDS, default=None)
        )
    if resolved_evidence_ids is None:
        resolved_evidence_ids = tuple(item.evidence_id for item in available_evidence)

    transitions = _normalise_transitions(
        review_transitions
        if review_transitions is not None
        else _read(output_mapping, ("review_transitions", "transitions"), default=())
    )
    requested_status = review_status
    if requested_status is None:
        requested_status = _read(
            output_mapping,
            ("review_status", "review_state"),
            default=ReviewState.QUEUED,
        )
    normalised_status = _normalise_review_state(requested_status)
    if normalised_status == ReviewState.APPROVED and not transitions:
        raise GuardedProvenanceError("approved output requires a review transition")

    model_record = _model_from_args(
        model,
        output_mapping,
        model_id=model_id,
        model_revision=model_revision,
        model_fingerprint=model_fingerprint,
        tokenizer_id=tokenizer_id,
    )
    resolved_policy_fingerprint = _policy_fingerprint(
        policy
        if policy is not None
        else _read(output_mapping, ("policy",), default=None),
        policy_fingerprint
        if policy_fingerprint is not None
        else _read(output_mapping, ("policy_fingerprint",), default=None),
    )

    available_ids = {item.evidence_id for item in available_evidence}
    missing_ids = tuple(sorted(set(resolved_evidence_ids) - available_ids))
    missing_count = len(missing_ids) if resolved_evidence_ids else 1
    input_checked = expected_input_hash is not None
    expected_input = _normalise_digest(
        expected_input_hash,
        "guarded-clinical-input",
    )
    input_changed = bool(input_checked and resolved_input_hash != expected_input)
    output_checked = expected_output_hash is not None
    expected_output = _normalise_digest(
        expected_output_hash,
        "guarded-clinical-output",
    )
    output_changed = bool(output_checked and resolved_output_hash != expected_output)
    reasons = _record_reason_codes(
        missing_evidence_count=missing_count,
        changed_evidence_count=0,
        input_present=resolved_input_hash is not None,
        output_present=resolved_output_hash is not None,
        model_present=model_record.present,
        policy_present=resolved_policy_fingerprint is not None,
        input_changed=input_changed,
        output_changed=output_changed,
        transitions=transitions,
        review_state=normalised_status,
    )
    integrity = IntegritySummary(
        missing_evidence_count=missing_count,
        missing_evidence_ids=missing_ids,
        input_checked=input_checked,
        input_changed=input_changed,
        output_checked=output_checked,
        output_changed=output_changed,
        model_present=model_record.present,
        policy_present=resolved_policy_fingerprint is not None,
        input_present=resolved_input_hash is not None,
        output_present=resolved_output_hash is not None,
        reason_codes=reasons,
    )
    return GuardedProvenanceRecord(
        output_kind=resolved_output_kind,
        input_hash=resolved_input_hash,
        output_hash=resolved_output_hash,
        evidence_ids=resolved_evidence_ids,
        evidence=available_evidence,
        model=model_record,
        policy_fingerprint=resolved_policy_fingerprint,
        review_required=True,
        review_status=normalised_status,
        review_transitions=transitions,
        integrity=integrity,
        record_id=record_id,
    )


@dataclass(frozen=True, slots=True)
class GuardedProvenanceManifest:
    """Versioned deterministic container for one or more output records."""

    records: tuple[GuardedProvenanceRecord, ...] = ()
    manifest_hash: str | None = None
    schema_version: int = GUARDED_PROVENANCE_SCHEMA_VERSION
    disclaimer: str = GUARDED_PROVENANCE_DISCLAIMER

    def __post_init__(self) -> None:
        if self.schema_version != GUARDED_PROVENANCE_SCHEMA_VERSION:
            raise GuardedProvenanceError("unsupported guarded provenance schema")
        if self.disclaimer != GUARDED_PROVENANCE_DISCLAIMER:
            raise GuardedProvenanceError("guarded provenance disclaimer is required")
        records = tuple(
            item
            if isinstance(item, GuardedProvenanceRecord)
            else GuardedProvenanceRecord.from_dict(item)
            for item in self.records
        )
        records = tuple(sorted(records, key=lambda item: item.record_id or ""))
        manifest_hash = self.manifest_hash
        object.__setattr__(self, "records", records)
        if manifest_hash is None:
            object.__setattr__(self, "manifest_hash", self.compute_manifest_hash())
        else:
            object.__setattr__(
                self,
                "manifest_hash",
                _normalise_digest(manifest_hash, "guarded-clinical-manifest"),
            )

    @property
    def record_count(self) -> int:
        """Return the number of guarded output records."""

        return len(self.records)

    @property
    def review_required_count(self) -> int:
        """Return the number of records still requiring human review."""

        return sum(record.requires_human_review for record in self.records)

    @property
    def release_ready(self) -> bool:
        """Return whether every intact record has approved human review."""

        return (
            bool(self.records)
            and self.verify_hash()
            and all(
                record.review_status == ReviewState.APPROVED and record._integrity().ok
                for record in self.records
            )
        )

    @property
    def hash_material(self) -> dict[str, Any]:
        """Return the canonical fields committed by ``manifest_hash``."""

        return {
            "schema_version": self.schema_version,
            "record_type": f"{GUARDED_PROVENANCE_RECORD_TYPE}.manifest",
            "records": [record.to_dict() for record in self.records],
        }

    def compute_manifest_hash(self) -> str:
        """Recompute the manifest hash without protected output values."""

        return _hash_payload(self.hash_material, "guarded-clinical-manifest")

    def verify_hash(self) -> bool:
        """Return whether the manifest and all record hashes are intact."""

        return self.manifest_hash == self.compute_manifest_hash() and all(
            record.verify_hash() for record in self.records
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready manifest with no generated clinical text."""

        return {
            "schema_version": self.schema_version,
            "record_type": f"{GUARDED_PROVENANCE_RECORD_TYPE}.manifest",
            "record_count": self.record_count,
            "review_required_count": self.review_required_count,
            "records": [record.to_dict() for record in self.records],
            "manifest_hash": self.manifest_hash,
            "disclaimer": self.disclaimer,
        }

    def to_json(self) -> str:
        """Serialize the manifest as compact deterministic JSON."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    def to_markdown(self) -> str:
        """Render a counts-and-hashes review view without output text."""

        lines = [
            "# Guarded Clinical-Output Provenance",
            "",
            GUARDED_PROVENANCE_DISCLAIMER,
            "",
            f"Records: {self.record_count}",
            f"Review required: {self.review_required_count}",
            f"Manifest hash: {self.manifest_hash or 'missing'}",
            "",
            "| # | Record | Kind | Review status | Missing evidence | Integrity |",
            "| ---: | --- | --- | --- | ---: | --- |",
        ]
        for index, record in enumerate(self.records, start=1):
            lines.append(
                "| "
                f"{index} | {record.record_id} | {record.output_kind} | "
                f"{record.review_status} | {record.missing_evidence_count} | "
                f"{'ok' if record._integrity().ok else 'blocked'} |"
            )
        return "\n".join(lines) + "\n"

    @classmethod
    def from_dict(cls, value: Any) -> "GuardedProvenanceManifest":
        """Load a manifest from its JSON-ready representation."""

        if not isinstance(value, Mapping):
            raise GuardedProvenanceError("provenance manifest must be an object")
        records = _read(value, ("records",), default=())
        if not isinstance(records, (list, tuple)):
            raise GuardedProvenanceError("provenance manifest records are invalid")
        return cls(
            records=tuple(GuardedProvenanceRecord.from_dict(item) for item in records),
            manifest_hash=_read(value, ("manifest_hash",), default=None),
            schema_version=_read(
                value,
                ("schema_version",),
                default=GUARDED_PROVENANCE_SCHEMA_VERSION,
            ),
            disclaimer=_read(
                value,
                ("disclaimer",),
                default=GUARDED_PROVENANCE_DISCLAIMER,
            ),
        )


ProvenanceRecord = GuardedProvenanceRecord
ProvenanceManifest = GuardedProvenanceManifest


def build_guarded_provenance_manifest(
    records: Any = None,
    *,
    output: Any = None,
    evidence: Any = (),
    model: Any = None,
    policy: Any = None,
    review_transitions: Any = None,
    output_kind: Any = None,
    record_id: Any = None,
    input_value: Any = _MISSING,
    input_hash: Any = _MISSING,
    expected_input_hash: Any = None,
    output_hash: Any = _MISSING,
    expected_output_hash: Any = None,
    evidence_ids: Any = None,
    review_status: Any = None,
    review_required: bool = True,
    model_id: Any = None,
    model_revision: Any = None,
    model_fingerprint: Any = None,
    tokenizer_id: Any = None,
    policy_fingerprint: Any = None,
) -> GuardedProvenanceManifest:
    """Build a deterministic manifest, usually containing one output record."""

    if records is None:
        record = build_guarded_provenance_record(
            output,
            evidence,
            model=model,
            policy=policy,
            review_transitions=review_transitions,
            output_kind=output_kind,
            record_id=record_id,
            input_value=input_value,
            input_hash=input_hash,
            expected_input_hash=expected_input_hash,
            output_hash=output_hash,
            expected_output_hash=expected_output_hash,
            evidence_ids=evidence_ids,
            review_status=review_status,
            review_required=review_required,
            model_id=model_id,
            model_revision=model_revision,
            model_fingerprint=model_fingerprint,
            tokenizer_id=tokenizer_id,
            policy_fingerprint=policy_fingerprint,
        )
        return GuardedProvenanceManifest(records=(record,))
    if isinstance(records, (GuardedProvenanceRecord, Mapping)):
        records = (records,)
    try:
        normalised = tuple(
            item
            if isinstance(item, GuardedProvenanceRecord)
            else GuardedProvenanceRecord.from_dict(item)
            for item in records
        )
    except Exception as exc:
        if isinstance(exc, GuardedProvenanceError):
            raise
        raise GuardedProvenanceError("provenance records are invalid") from None
    return GuardedProvenanceManifest(records=normalised)


def export_guarded_provenance(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Return a JSON-ready guarded provenance manifest."""

    return build_guarded_provenance_manifest(*args, **kwargs).to_dict()


def write_guarded_provenance_manifest(
    path: str | Path,
    manifest: GuardedProvenanceManifest | Mapping[str, Any],
) -> Path:
    """Write one deterministic local manifest and return its path."""

    try:
        resolved = (
            manifest
            if isinstance(manifest, GuardedProvenanceManifest)
            else GuardedProvenanceManifest.from_dict(manifest)
        )
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            resolved.to_json() + "\n",
            encoding="utf-8",
        )
    except GuardedProvenanceError:
        raise
    except Exception:
        raise GuardedProvenanceError(
            "guarded provenance manifest could not be written"
        ) from None
    return destination


def load_guarded_provenance_manifest(path: str | Path) -> GuardedProvenanceManifest:
    """Load a local manifest without echoing file paths or parser details."""

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return GuardedProvenanceManifest.from_dict(value)
    except GuardedProvenanceError:
        raise
    except Exception:
        raise GuardedProvenanceError(
            "guarded provenance manifest could not be loaded"
        ) from None


@dataclass(frozen=True, slots=True)
class _RecordIntegrityReport:
    record_id: str
    review_status: str
    record_hash_valid: bool
    missing_evidence_count: int
    missing_evidence_ids: tuple[str, ...]
    changed_evidence_count: int
    changed_evidence_ids: tuple[str, ...]
    input_checked: bool
    input_changed: bool
    output_checked: bool
    output_changed: bool
    reason_codes: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.reason_codes and self.record_hash_valid

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "review_status": self.review_status,
            "record_hash_valid": self.record_hash_valid,
            "missing_evidence_count": self.missing_evidence_count,
            "missing_evidence_ids": list(self.missing_evidence_ids),
            "changed_evidence_count": self.changed_evidence_count,
            "changed_evidence_ids": list(self.changed_evidence_ids),
            "input_checked": self.input_checked,
            "input_changed": self.input_changed,
            "output_checked": self.output_checked,
            "output_changed": self.output_changed,
            "reason_codes": list(self.reason_codes),
            "ok": self.ok,
        }


@dataclass(frozen=True, slots=True)
class ProvenanceIntegrityReport:
    """Counts-only result of checking a manifest against current local inputs."""

    manifest_hash_valid: bool
    records: tuple[_RecordIntegrityReport, ...] = ()
    malformed_manifest: bool = False
    disclaimer: str = GUARDED_PROVENANCE_DISCLAIMER

    @property
    def record_count(self) -> int:
        """Return the number of records checked."""

        return len(self.records)

    @property
    def missing_evidence_count(self) -> int:
        """Return the aggregate unresolved-evidence count."""

        return sum(item.missing_evidence_count for item in self.records)

    @property
    def changed_evidence_count(self) -> int:
        """Return the aggregate changed-evidence count."""

        return sum(item.changed_evidence_count for item in self.records)

    @property
    def review_required_count(self) -> int:
        """Return the number of records that still require human review."""

        return sum(
            item.review_status != ReviewState.APPROVED
            or not item.record_hash_valid
            or bool(item.reason_codes)
            for item in self.records
        )

    @property
    def input_changed(self) -> bool:
        """Return whether any checked input differs from its recorded digest."""

        return any(item.input_changed for item in self.records)

    @property
    def output_changed(self) -> bool:
        """Return whether any checked output differs from its recorded digest."""

        return any(item.output_changed for item in self.records)

    @property
    def missing_evidence_ids(self) -> tuple[str, ...]:
        """Return opaque unresolved evidence identifiers."""

        return tuple(
            sorted(
                {
                    evidence_id
                    for item in self.records
                    for evidence_id in item.missing_evidence_ids
                }
            )
        )

    @property
    def changed_evidence_ids(self) -> tuple[str, ...]:
        """Return opaque changed evidence identifiers."""

        return tuple(
            sorted(
                {
                    evidence_id
                    for item in self.records
                    for evidence_id in item.changed_evidence_ids
                }
            )
        )

    @property
    def reason_codes(self) -> tuple[str, ...]:
        """Return stable, value-free integrity reason codes."""

        reasons = {reason for item in self.records for reason in item.reason_codes}
        if not self.manifest_hash_valid:
            reasons.add("invalid_manifest_hash")
        if self.malformed_manifest:
            reasons.add("malformed_manifest")
        return tuple(sorted(reasons))

    @property
    def ok(self) -> bool:
        """Return whether the manifest and checked inputs are intact."""

        return (
            self.manifest_hash_valid
            and not self.malformed_manifest
            and all(item.ok for item in self.records)
        )

    @property
    def valid(self) -> bool:
        """Alias for :attr:`ok`."""

        return self.ok

    @property
    def requires_human_review(self) -> bool:
        """Return whether any record is unapproved or integrity-blocked."""

        return any(
            item.review_status != ReviewState.APPROVED
            or item.record_hash_valid is False
            or bool(item.reason_codes)
            for item in self.records
        )

    @property
    def release_ready(self) -> bool:
        """Return whether integrity is valid and every record is approved."""

        return (
            bool(self.records)
            and self.ok
            and all(item.review_status == ReviewState.APPROVED for item in self.records)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return aggregate counts and opaque per-record diagnostics."""

        return {
            "schema_version": GUARDED_PROVENANCE_SCHEMA_VERSION,
            "manifest_hash_valid": self.manifest_hash_valid,
            "malformed_manifest": self.malformed_manifest,
            "record_count": self.record_count,
            "review_required_count": self.review_required_count,
            "missing_evidence_count": self.missing_evidence_count,
            "missing_evidence_ids": list(self.missing_evidence_ids),
            "changed_evidence_count": self.changed_evidence_count,
            "changed_evidence_ids": list(self.changed_evidence_ids),
            "input_changed": self.input_changed,
            "output_changed": self.output_changed,
            "reason_codes": list(self.reason_codes),
            "ok": self.ok,
            "requires_human_review": self.requires_human_review,
            "release_ready": self.release_ready,
            "records": [item.to_dict() for item in self.records],
            "disclaimer": self.disclaimer,
        }

    def to_json(self) -> str:
        """Serialize the report as compact deterministic JSON."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )


def _coerce_manifest(value: Any) -> GuardedProvenanceManifest:
    if isinstance(value, GuardedProvenanceManifest):
        return value
    if isinstance(value, GuardedProvenanceRecord):
        return GuardedProvenanceManifest(records=(value,))
    if isinstance(value, Mapping):
        if "records" in value:
            return GuardedProvenanceManifest.from_dict(value)
        return GuardedProvenanceManifest(
            records=(GuardedProvenanceRecord.from_dict(value),)
        )
    raise GuardedProvenanceError("guarded provenance must be a record or manifest")


def _evidence_for_check(value: Any) -> tuple[EvidenceReference, ...]:
    return _normalise_evidence_collection(value)


def check_guarded_provenance(
    value: GuardedProvenanceManifest | GuardedProvenanceRecord | Mapping[str, Any],
    *,
    current_input: Any = _MISSING,
    current_input_hash: Any = _MISSING,
    current_output: Any = _MISSING,
    current_output_hash: Any = _MISSING,
    current_evidence: Any = _MISSING,
) -> ProvenanceIntegrityReport:
    """Check hashes, evidence availability, and local input drift.

    The returned report contains only counts, reason codes, offsets already
    present in the manifest, and opaque identifiers.  Passing ``current_input``
    or ``current_evidence`` is optional; when omitted, the check verifies the
    persisted manifest and its recorded gaps without making assumptions about
    an external source.
    """

    try:
        manifest = _coerce_manifest(value)
    except GuardedProvenanceError:
        return ProvenanceIntegrityReport(
            manifest_hash_valid=False,
            malformed_manifest=True,
        )

    manifest_hash_valid = manifest.manifest_hash == manifest.compute_manifest_hash()
    current_evidence_records: tuple[EvidenceReference, ...] | None = None
    if current_evidence is not _MISSING:
        try:
            current_evidence_records = _evidence_for_check(current_evidence)
        except GuardedProvenanceError:
            current_evidence_records = ()
    current_evidence_by_id = (
        {}
        if current_evidence_records is None
        else {item.evidence_id: item for item in current_evidence_records}
    )

    reports: list[_RecordIntegrityReport] = []
    for record in manifest.records:
        integrity = record._integrity()
        missing_ids = set(integrity.missing_evidence_ids)
        changed_ids = set(integrity.changed_evidence_ids)
        missing_count = integrity.missing_evidence_count
        changed_count = integrity.changed_evidence_count
        input_checked = integrity.input_checked
        input_changed = integrity.input_changed
        output_checked = integrity.output_checked
        output_changed = integrity.output_changed
        reasons = set(integrity.reason_codes)

        if current_evidence_records is not None:
            missing_ids = set()
            changed_ids = set()
            for evidence_id in record.evidence_ids:
                current = current_evidence_by_id.get(evidence_id)
                if current is None:
                    missing_ids.add(evidence_id)
                    continue
                stored = next(
                    (
                        item
                        for item in record.evidence
                        if item.evidence_id == evidence_id
                    ),
                    None,
                )
                if (
                    stored is not None
                    and stored.evidence_hash is not None
                    and current.evidence_hash != stored.evidence_hash
                ):
                    changed_ids.add(evidence_id)
            missing_count = len(missing_ids) if record.evidence_ids else 1
            changed_count = len(changed_ids)
            reasons.discard("missing_evidence")
            reasons.discard("changed_evidence")
            if missing_count:
                reasons.add("missing_evidence")
            if changed_count:
                reasons.add("changed_evidence")

        if current_input is not _MISSING or current_input_hash is not _MISSING:
            input_checked = True
            candidate = (
                _normalise_digest(current_input_hash, "guarded-clinical-input")
                if current_input_hash is not _MISSING
                else _fingerprint(current_input, "guarded-clinical-input")
            )
            input_changed = record.input_hash is None or candidate != record.input_hash
            reasons.discard("changed_input")
            if input_changed:
                reasons.add("changed_input")

        if current_output is not _MISSING or current_output_hash is not _MISSING:
            output_checked = True
            candidate = (
                _normalise_digest(current_output_hash, "guarded-clinical-output")
                if current_output_hash is not _MISSING
                else _fingerprint(
                    _output_value_from_output(current_output),
                    "guarded-clinical-output",
                )
            )
            output_changed = (
                record.output_hash is None or candidate != record.output_hash
            )
            reasons.discard("changed_output")
            if output_changed:
                reasons.add("changed_output")

        if not manifest_hash_valid:
            reasons.add("invalid_manifest_hash")
        record_hash_valid = record.verify_hash()
        if not record_hash_valid:
            reasons.add("invalid_record_hash")
        reports.append(
            _RecordIntegrityReport(
                record_id=record.record_id or "",
                review_status=record.review_status,
                record_hash_valid=record_hash_valid,
                missing_evidence_count=missing_count,
                missing_evidence_ids=tuple(sorted(missing_ids)),
                changed_evidence_count=changed_count,
                changed_evidence_ids=tuple(sorted(changed_ids)),
                input_checked=input_checked,
                input_changed=input_changed,
                output_checked=output_checked,
                output_changed=output_changed,
                reason_codes=tuple(sorted(reasons)),
            )
        )
    return ProvenanceIntegrityReport(
        manifest_hash_valid=manifest_hash_valid,
        records=tuple(reports),
    )


def validate_guarded_provenance(
    value: GuardedProvenanceManifest | GuardedProvenanceRecord | Mapping[str, Any],
    **kwargs: Any,
) -> ProvenanceIntegrityReport:
    """Return a deterministic integrity report for a guarded manifest."""

    return check_guarded_provenance(value, **kwargs)


def verify_guarded_provenance(
    value: GuardedProvenanceManifest | GuardedProvenanceRecord | Mapping[str, Any],
    **kwargs: Any,
) -> bool:
    """Return whether a guarded manifest passes its structural integrity gate."""

    return check_guarded_provenance(value, **kwargs).ok


build_provenance_record = build_guarded_provenance_record
build_provenance_manifest = build_guarded_provenance_manifest
build_guarded_provenance = build_guarded_provenance_manifest
