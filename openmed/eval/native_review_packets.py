"""Privacy-safe packets for native-language reviewers and decision import.

Language validation needs native experts, but a review package can leak PHI or
expose more institutional text than the review needs. This module builds
minimal, deterministic packets only from items that are explicitly marked as
synthetic or already de-identified, and refuses everything else.

A reviewer-facing :class:`NativeReviewPacket` carries opaque item identifiers,
task instructions, a seeded randomized item order, and a provenance commitment.
The evaluator keeps a separate :class:`NativeReviewManifest` that binds each
opaque item to its private source reference and provenance digest. Reviewer
decisions return through the opaque identifiers; import validates them against
the packet and manifest, rejects reviewer identity and free-text fields, and
records only a closed reviewer-role code.

Everything runs locally with no network call. Exceptions carry field names and
closed codes, never input values.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any

from openmed.clinical.journey_contracts import (
    canonical_digest,
    canonical_json,
    derived_opaque_id,
    sha256_digest,
)
from openmed.eval.annotation.interchange import CoordinateConvention

NATIVE_REVIEW_PACKET_SCHEMA_VERSION = "openmed.eval.native_review.packet.v1"
NATIVE_REVIEW_MANIFEST_SCHEMA_VERSION = "openmed.eval.native_review.manifest.v1"
NATIVE_REVIEW_SUBMISSION_SCHEMA_VERSION = "openmed.eval.native_review.submission.v1"
NATIVE_REVIEW_DECISION_SCHEMA_VERSION = "openmed.eval.native_review.decision.v1"
NATIVE_REVIEW_IMPORT_SCHEMA_VERSION = "openmed.eval.native_review.import.v1"

DEFAULT_DECISION_CODES = ("accept", "reject", "unsure")
DEFAULT_REVIEWER_ROLES = (
    "clinical_linguist",
    "medical_translator",
    "native_clinician",
    "native_speaker",
)
DEFAULT_MAX_ITEM_CHARACTERS = 1_000
MAX_ITEM_CHARACTERS = 4_000
MAX_PACKET_ITEMS = 10_000
MAX_SPANS_PER_ITEM = 64
MAX_TASKS = 16
MAX_SUBMISSION_BYTES = 8 * 1024 * 1024
MAX_PACKET_BYTES = 64 * 1024 * 1024

NATIVE_REVIEW_NOTICE = (
    "Language review only. Every item is synthetic or already de-identified. "
    "Record decisions with the listed codes; do not add names, contact "
    "details, or free-text comments. This review does not make or trigger "
    "clinical decisions."
)

_COORDINATE_CONVENTION = CoordinateConvention.UNICODE_CODEPOINT.value
_PUBLIC_REF_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_CODE_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_LABEL_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,63}")
_LANGUAGE_RE = re.compile(r"[a-z]{2,3}(?:-[A-Za-z0-9]{2,8}){0,3}")
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_ITEM_ID_RE = re.compile(r"nri_[0-9a-f]{32}")
_DECISION_ID_RE = re.compile(r"nrd_[0-9a-f]{32}")
_ORDER_DOMAIN = "openmed.eval.native_review.order.v1"
_PROVENANCE_DOMAIN = "openmed.eval.native_review.provenance.v1"
_PROJECTION_DOMAIN = "openmed.eval.native_review.projection.v1"
_MAX_INSTRUCTION_CHARACTERS = 4_000
_MAX_DECISION_CODES = 16
_MAX_LABEL_CODES = 256
_MAX_REASON_CODES = 32
_MAX_REVIEWER_ROLES = 16
_MAX_SEED = 2**64
_ALLOWED_CONTROL_CHARACTERS = frozenset({"\t", "\n", "\r"})
_ZERO_WIDTH_JOINER = "\u200d"
_IDENTITY_KEYS = frozenset(
    {
        "annotator",
        "annotator_id",
        "contact",
        "institution",
        "organization",
        "reviewer",
        "reviewer_id",
        "user",
        "user_id",
        "username",
    }
)
_IDENTITY_KEY_TOKENS = ("email", "name", "phone", "signature")
_FREE_TEXT_KEYS = frozenset(
    {
        "comment",
        "comments",
        "explanation",
        "free_text",
        "note",
        "notes",
        "rationale",
        "remarks",
        "text",
    }
)
_INVALID = object()


class NativeReviewPacketError(ValueError):
    """Raised when a native-review packet, manifest, or submission is unsafe."""


class NativeReviewIncompleteError(NativeReviewPacketError):
    """Raised when an imported submission does not decide every packet item.

    Attributes:
        completeness: Opaque, value-free completeness summary for the import.
    """

    def __init__(self, completeness: NativeReviewCompleteness) -> None:
        super().__init__("decisions: incomplete")
        self.completeness = completeness


class ReviewPrivacyStatus(str, Enum):
    """Closed privacy states that may enter a native-review packet."""

    SYNTHETIC = "synthetic"
    DEIDENTIFIED = "deidentified"


def _error(field_name: str, code: str) -> NativeReviewPacketError:
    return NativeReviewPacketError(f"{field_name}: {code}")


def _require_ref(value: Any, field_name: str) -> str:
    if type(value) is not str or _PUBLIC_REF_RE.fullmatch(value) is None:
        raise _error(field_name, "invalid_reference")
    return value


def _require_code(value: Any, field_name: str) -> str:
    if type(value) is not str or _CODE_RE.fullmatch(value) is None:
        raise _error(field_name, "invalid_code")
    return value


def _require_label(value: Any, field_name: str) -> str:
    if type(value) is not str or _LABEL_RE.fullmatch(value) is None:
        raise _error(field_name, "invalid_label")
    return value


def _require_digest(value: Any, field_name: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise _error(field_name, "invalid_digest")
    return value


def _require_language(value: Any) -> str:
    if type(value) is not str or _LANGUAGE_RE.fullmatch(value) is None:
        raise _error("language", "invalid_language")
    return value


def _require_item_id(value: Any, field_name: str) -> str:
    if type(value) is not str or _ITEM_ID_RE.fullmatch(value) is None:
        raise _error(field_name, "invalid_item_id")
    return value


def _require_seed(value: Any) -> int:
    if type(value) is not int or not 0 <= value < _MAX_SEED:
        raise _error("seed", "invalid_seed")
    return value


def _require_text(
    value: Any, field_name: str, maximum: int, *, too_long: str = "too_long"
) -> str:
    if type(value) is not str or not value.strip():
        raise _error(field_name, "invalid_text")
    if len(value) > maximum:
        raise _error(field_name, too_long)
    for character in value:
        codepoint = ord(character)
        if (
            codepoint < 32 and character not in _ALLOWED_CONTROL_CHARACTERS
        ) or codepoint == 127:
            raise _error(field_name, "control_character")
        if 0xD800 <= codepoint <= 0xDFFF:
            raise _error(field_name, "invalid_text")
    return value


def _require_privacy_status(value: Any, field_name: str) -> ReviewPrivacyStatus:
    if isinstance(value, ReviewPrivacyStatus):
        return value
    if type(value) is str:
        for status in ReviewPrivacyStatus:
            if status.value == value:
                return status
    raise _error(field_name, "not_reviewable")


def _require_codes(
    values: Any,
    field_name: str,
    *,
    minimum: int,
    maximum: int,
    validator: Callable[[Any, str], str] = _require_code,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise _error(field_name, "invalid_sequence")
    codes = tuple(validator(value, field_name) for value in values)
    if not minimum <= len(codes) <= maximum:
        raise _error(field_name, "invalid_count")
    if len(set(codes)) != len(codes):
        raise _error(field_name, "duplicate_code")
    return codes


def _is_grapheme_boundary(text: str, index: int) -> bool:
    """Return whether ``index`` avoids splitting a common grapheme cluster.

    This conservative check rejects offsets immediately before a combining
    mark (including the spacing vowel signs used by Indic scripts) or an emoji
    modifier, either side of a zero-width joiner, and inside a CRLF pair.
    """
    if index <= 0 or index >= len(text):
        return True
    previous = text[index - 1]
    following = text[index]
    if unicodedata.category(following) in {"Mn", "Mc", "Me"}:
        return False
    if _ZERO_WIDTH_JOINER in (previous, following):
        return False
    if "\U0001f3fb" <= following <= "\U0001f3ff":
        return False
    return not (previous == "\r" and following == "\n")


def _tuple_of(
    values: Any, field_name: str, kind: type, *, minimum: int, maximum: int
) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise _error(field_name, "invalid_sequence")
    normalized = tuple(values)
    if not minimum <= len(normalized) <= maximum or not all(
        isinstance(value, kind) for value in normalized
    ):
        raise _error(field_name, "invalid_sequence")
    return normalized


def _field_names(cls: type) -> frozenset[str]:
    return frozenset(item.name for item in fields(cls))


def _span_tuple(values: Any, field_name: str) -> tuple[NativeReviewSpan, ...]:
    spans = _tuple_of(
        values, field_name, NativeReviewSpan, minimum=0, maximum=MAX_SPANS_PER_ITEM
    )
    ordered = tuple(sorted(spans, key=lambda span: (span.start, span.end, span.label)))
    if len(set(ordered)) != len(ordered):
        raise _error(field_name, "duplicate_span")
    return ordered


def _check_span_offsets(
    text: str, spans: Sequence[NativeReviewSpan], field_name: str
) -> None:
    for span in spans:
        if span.end > len(text):
            raise _error(field_name, "offset_out_of_range")
        if not (
            _is_grapheme_boundary(text, span.start)
            and _is_grapheme_boundary(text, span.end)
        ):
            raise _error(field_name, "split_grapheme")


def _projection_digest(spans: Sequence[NativeReviewSpan]) -> str:
    return canonical_digest(
        {
            "coordinate_convention": _COORDINATE_CONVENTION,
            "domain": _PROJECTION_DOMAIN,
            "spans": [span.to_dict() for span in spans],
        }
    )


def _is_identity_key(key: str) -> bool:
    lowered = key.lower()
    return lowered in _IDENTITY_KEYS or any(
        token in lowered for token in _IDENTITY_KEY_TOKENS
    )


def _strict_fields(
    value: Any,
    field_name: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise _error(field_name, "invalid_object")
    keys = frozenset(value)
    extra = keys - required - optional
    if extra:
        if any(_is_identity_key(key) for key in extra):
            raise _error(field_name, "reviewer_identity_forbidden")
        if any(key.lower() in _FREE_TEXT_KEYS for key in extra):
            raise _error(field_name, "free_text_forbidden")
        raise _error(field_name, "unknown_field")
    if required - keys:
        raise _error(field_name, "missing_field")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError("duplicate key")
        document[key] = value
    return document


def _reject_constant(_constant: str) -> Any:
    raise ValueError("non-finite number")


def _load_document(payload: Any, field_name: str, maximum_bytes: int) -> Any:
    if isinstance(payload, Mapping):
        return payload
    if isinstance(payload, (bytes, bytearray)):
        if len(payload) > maximum_bytes:
            raise _error(field_name, "byte_limit_exceeded")
        try:
            text = bytes(payload).decode("utf-8")
        except UnicodeDecodeError:
            text = None
        if text is None:
            raise _error(field_name, "invalid_encoding")
    elif isinstance(payload, str):
        if len(payload.encode("utf-8", errors="surrogatepass")) > maximum_bytes:
            raise _error(field_name, "byte_limit_exceeded")
        text = payload
    else:
        raise _error(field_name, "invalid_document")
    # Parse outside an exception chain so decoder errors, which hold the raw
    # document, are never attached to the raised error.
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (ValueError, RecursionError):
        parsed = _INVALID
    if parsed is _INVALID:
        raise _error(field_name, "invalid_json")
    return parsed


def _list_field(value: Any, field_name: str, maximum: int) -> list[Any]:
    if not isinstance(value, list) or len(value) > maximum:
        raise _error(field_name, "invalid_sequence")
    return value


@dataclass(frozen=True, slots=True)
class NativeReviewSpan:
    """A proposed or corrected label over Unicode code-point offsets.

    Offsets follow the annotation-interchange ``unicode_codepoint`` convention
    and are half-open: ``start`` is inclusive and ``end`` is exclusive.
    """

    start: int
    end: int
    label: str

    def __post_init__(self) -> None:
        if (
            type(self.start) is not int
            or type(self.end) is not int
            or not 0 <= self.start < self.end
        ):
            raise _error("span", "invalid_offsets")
        _require_label(self.label, "span.label")

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical span document."""
        return {"end": self.end, "label": self.label, "start": self.start}

    @classmethod
    def from_dict(cls, payload: Any) -> NativeReviewSpan:
        """Parse one strict span document."""
        return cls(**_strict_fields(payload, "span", _field_names(cls)))


@dataclass(frozen=True, slots=True)
class NativeReviewTask:
    """Reviewer instructions and closed response vocabularies for one task.

    Attributes:
        task_code: Lowercase controlled task code, such as ``label_check``.
        instructions: Reviewer-facing instructions written by the evaluator.
        decision_codes: Two to sixteen closed decision codes.
        label_codes: Closed label vocabulary for proposed spans and
            corrections. Required when the task carries spans or corrections.
        reason_codes: Optional closed vocabulary for decision reasons.
        correction_decision_code: Decision code that requires a corrected span
            set. ``None`` disables corrections for the task.
    """

    task_code: str
    instructions: str
    decision_codes: tuple[str, ...] = DEFAULT_DECISION_CODES
    label_codes: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    correction_decision_code: str | None = None

    def __post_init__(self) -> None:
        _require_code(self.task_code, "task_code")
        _require_text(self.instructions, "instructions", _MAX_INSTRUCTION_CHARACTERS)
        decision_codes = _require_codes(
            self.decision_codes,
            "decision_codes",
            minimum=2,
            maximum=_MAX_DECISION_CODES,
        )
        label_codes = _require_codes(
            self.label_codes,
            "label_codes",
            minimum=0,
            maximum=_MAX_LABEL_CODES,
            validator=_require_label,
        )
        reason_codes = _require_codes(
            self.reason_codes,
            "reason_codes",
            minimum=0,
            maximum=_MAX_REASON_CODES,
        )
        if self.correction_decision_code is not None:
            _require_code(self.correction_decision_code, "correction_decision_code")
            if self.correction_decision_code not in decision_codes:
                raise _error("correction_decision_code", "unknown_code")
            if not label_codes:
                raise _error("label_codes", "required_for_corrections")
        object.__setattr__(self, "decision_codes", decision_codes)
        object.__setattr__(self, "label_codes", label_codes)
        object.__setattr__(self, "reason_codes", reason_codes)

    def to_dict(self) -> dict[str, Any]:
        """Return the reviewer-facing task document."""
        return {
            "correction_decision_code": self.correction_decision_code,
            "decision_codes": list(self.decision_codes),
            "instructions": self.instructions,
            "label_codes": list(self.label_codes),
            "reason_codes": list(self.reason_codes),
            "task_code": self.task_code,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> NativeReviewTask:
        """Parse one strict task document."""
        data = _strict_fields(payload, "task", _field_names(cls))
        return cls(
            task_code=data["task_code"],
            instructions=data["instructions"],
            decision_codes=_list_field(
                data["decision_codes"], "decision_codes", _MAX_DECISION_CODES
            ),
            label_codes=_list_field(
                data["label_codes"], "label_codes", _MAX_LABEL_CODES
            ),
            reason_codes=_list_field(
                data["reason_codes"], "reason_codes", _MAX_REASON_CODES
            ),
            correction_decision_code=data["correction_decision_code"],
        )


@dataclass(frozen=True, slots=True)
class NativeReviewItem:
    """Private packet input: one synthetic or already de-identified segment.

    Construction refuses any privacy status other than ``synthetic`` or
    ``deidentified``. ``source_ref`` and ``provenance_digest`` stay in the
    evaluator manifest; neither is written to the reviewer packet.

    Attributes:
        source_ref: Caller-owned opaque reference to the source item.
        task_code: Code of the :class:`NativeReviewTask` that applies.
        privacy_status: ``synthetic`` or ``deidentified``.
        provenance_digest: SHA-256 digest of the generator or
            de-identification provenance record for this item.
        text: The minimal reviewer-visible segment.
        spans: Proposed labels to review, as code-point offsets into ``text``.
    """

    source_ref: str
    task_code: str
    privacy_status: ReviewPrivacyStatus
    provenance_digest: str
    text: str = field(repr=False)
    spans: tuple[NativeReviewSpan, ...] = ()

    def __post_init__(self) -> None:
        status = _require_privacy_status(self.privacy_status, "privacy_status")
        _require_ref(self.source_ref, "source_ref")
        _require_code(self.task_code, "task_code")
        _require_digest(self.provenance_digest, "provenance_digest")
        _require_text(
            self.text, "text", MAX_ITEM_CHARACTERS, too_long="exceeds_minimal_context"
        )
        spans = _span_tuple(self.spans, "spans")
        _check_span_offsets(self.text, spans, "spans")
        object.__setattr__(self, "privacy_status", status)
        object.__setattr__(self, "spans", spans)


@dataclass(frozen=True, slots=True)
class NativeReviewPacketItem:
    """One reviewer-visible item identified only by an opaque item ID."""

    item_id: str
    task_code: str
    privacy_status: ReviewPrivacyStatus
    text: str = field(repr=False)
    spans: tuple[NativeReviewSpan, ...] = ()

    def __post_init__(self) -> None:
        _require_item_id(self.item_id, "item_id")
        _require_code(self.task_code, "task_code")
        status = _require_privacy_status(self.privacy_status, "privacy_status")
        _require_text(
            self.text, "text", MAX_ITEM_CHARACTERS, too_long="exceeds_minimal_context"
        )
        spans = _span_tuple(self.spans, "spans")
        _check_span_offsets(self.text, spans, "spans")
        object.__setattr__(self, "privacy_status", status)
        object.__setattr__(self, "spans", spans)

    @property
    def text_digest(self) -> str:
        """Return the SHA-256 digest of the reviewer-visible text."""
        return sha256_digest(self.text)

    @property
    def projection_digest(self) -> str:
        """Return the digest of the proposed span projection."""
        return _projection_digest(self.spans)

    def to_dict(self) -> dict[str, Any]:
        """Return the reviewer-facing item document."""
        return {
            "item_id": self.item_id,
            "privacy_status": self.privacy_status.value,
            "spans": [span.to_dict() for span in self.spans],
            "task_code": self.task_code,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> NativeReviewPacketItem:
        """Parse one strict packet item document."""
        data = _strict_fields(payload, "item", _field_names(cls))
        return cls(
            item_id=data["item_id"],
            task_code=data["task_code"],
            privacy_status=data["privacy_status"],
            text=data["text"],
            spans=tuple(
                NativeReviewSpan.from_dict(span)
                for span in _list_field(data["spans"], "spans", MAX_SPANS_PER_ITEM)
            ),
        )


@dataclass(frozen=True, slots=True)
class NativeReviewPacket:
    """A minimal, reviewer-facing native-language review packet.

    The packet contains no source references, provenance records, seeds, or
    reviewer identity. Items appear in a seeded randomized order.
    """

    packet_ref: str
    language: str
    tasks: tuple[NativeReviewTask, ...]
    items: tuple[NativeReviewPacketItem, ...]
    reviewer_roles: tuple[str, ...]
    provenance_commitment: str
    coordinate_convention: str = _COORDINATE_CONVENTION
    notice: str = NATIVE_REVIEW_NOTICE
    schema_version: str = NATIVE_REVIEW_PACKET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != NATIVE_REVIEW_PACKET_SCHEMA_VERSION:
            raise _error("schema_version", "unsupported_version")
        _require_ref(self.packet_ref, "packet_ref")
        _require_language(self.language)
        _require_digest(self.provenance_commitment, "provenance_commitment")
        if self.coordinate_convention != _COORDINATE_CONVENTION:
            raise _error("coordinate_convention", "unsupported_convention")
        _require_text(self.notice, "notice", _MAX_INSTRUCTION_CHARACTERS)
        roles = tuple(
            sorted(
                _require_codes(
                    self.reviewer_roles,
                    "reviewer_roles",
                    minimum=1,
                    maximum=_MAX_REVIEWER_ROLES,
                )
            )
        )

        tasks = tuple(
            sorted(
                _tuple_of(
                    self.tasks, "tasks", NativeReviewTask, minimum=1, maximum=MAX_TASKS
                ),
                key=lambda task: task.task_code,
            )
        )
        task_by_code = {task.task_code: task for task in tasks}
        if len(task_by_code) != len(tasks):
            raise _error("tasks", "duplicate_code")

        items = _tuple_of(
            self.items,
            "items",
            NativeReviewPacketItem,
            minimum=1,
            maximum=MAX_PACKET_ITEMS,
        )
        if len({item.item_id for item in items}) != len(items):
            raise _error("items", "duplicate_item_id")
        for item in items:
            task = task_by_code.get(item.task_code)
            if task is None:
                raise _error("items", "unknown_task")
            if any(span.label not in task.label_codes for span in item.spans):
                raise _error("spans", "unknown_label")
        if {item.task_code for item in items} != set(task_by_code):
            raise _error("tasks", "unused_task")

        object.__setattr__(self, "reviewer_roles", roles)
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "items", items)

    @property
    def packet_digest(self) -> str:
        """Return the SHA-256 digest of the canonical packet document."""
        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return the reviewer-facing packet document."""
        return {
            "coordinate_convention": self.coordinate_convention,
            "items": [item.to_dict() for item in self.items],
            "language": self.language,
            "notice": self.notice,
            "packet_ref": self.packet_ref,
            "provenance_commitment": self.provenance_commitment,
            "reviewer_roles": list(self.reviewer_roles),
            "schema_version": self.schema_version,
            "tasks": [task.to_dict() for task in self.tasks],
        }

    def to_json(self) -> str:
        """Return canonical JSON for the reviewer packet."""
        return canonical_json(self.to_dict())

    def submission_template(self) -> dict[str, Any]:
        """Return an empty decision submission for the reviewer to complete.

        Every item appears once with a ``null`` decision. The reviewer fills
        ``reviewer_role`` with one of the packet's role codes and each
        ``decision`` with one of the task's decision codes.
        """
        return {
            "decisions": [
                {
                    "corrections": [],
                    "decision": None,
                    "item_id": item.item_id,
                    "reason_code": None,
                }
                for item in self.items
            ],
            "packet_digest": self.packet_digest,
            "packet_ref": self.packet_ref,
            "reviewer_role": None,
            "schema_version": NATIVE_REVIEW_SUBMISSION_SCHEMA_VERSION,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> NativeReviewPacket:
        """Parse and revalidate one strict packet document."""
        data = _strict_fields(payload, "packet", _field_names(cls))
        return cls(
            packet_ref=data["packet_ref"],
            language=data["language"],
            tasks=tuple(
                NativeReviewTask.from_dict(task)
                for task in _list_field(data["tasks"], "tasks", MAX_TASKS)
            ),
            items=tuple(
                NativeReviewPacketItem.from_dict(item)
                for item in _list_field(data["items"], "items", MAX_PACKET_ITEMS)
            ),
            reviewer_roles=tuple(
                _list_field(
                    data["reviewer_roles"], "reviewer_roles", _MAX_REVIEWER_ROLES
                )
            ),
            provenance_commitment=data["provenance_commitment"],
            coordinate_convention=data["coordinate_convention"],
            notice=data["notice"],
            schema_version=data["schema_version"],
        )

    @classmethod
    def from_json(cls, payload: str | bytes) -> NativeReviewPacket:
        """Parse strict, bounded packet JSON."""
        return cls.from_dict(_load_document(payload, "packet", MAX_PACKET_BYTES))


@dataclass(frozen=True, slots=True)
class NativeReviewManifestEntry:
    """Evaluator-private binding from an opaque item to its source."""

    item_id: str
    source_ref: str
    task_code: str
    privacy_status: ReviewPrivacyStatus
    provenance_digest: str
    text_digest: str
    projection_digest: str

    def __post_init__(self) -> None:
        _require_item_id(self.item_id, "item_id")
        _require_ref(self.source_ref, "source_ref")
        _require_code(self.task_code, "task_code")
        status = _require_privacy_status(self.privacy_status, "privacy_status")
        _require_digest(self.provenance_digest, "provenance_digest")
        _require_digest(self.text_digest, "text_digest")
        _require_digest(self.projection_digest, "projection_digest")
        object.__setattr__(self, "privacy_status", status)

    def to_dict(self) -> dict[str, str]:
        """Return the evaluator-private entry document."""
        return {
            "item_id": self.item_id,
            "privacy_status": self.privacy_status.value,
            "projection_digest": self.projection_digest,
            "provenance_digest": self.provenance_digest,
            "source_ref": self.source_ref,
            "task_code": self.task_code,
            "text_digest": self.text_digest,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> NativeReviewManifestEntry:
        """Parse one strict manifest entry document."""
        return cls(**_strict_fields(payload, "manifest_entry", _field_names(cls)))


@dataclass(frozen=True, slots=True)
class NativeReviewManifest:
    """Evaluator-private manifest used to validate returned decisions.

    The manifest maps opaque packet items back to caller-owned source
    references and provenance digests. Store it separately from reviewer
    packets and never send it to reviewers.
    """

    packet_ref: str
    packet_digest: str
    language: str
    seed: int
    entries: tuple[NativeReviewManifestEntry, ...]
    schema_version: str = NATIVE_REVIEW_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != NATIVE_REVIEW_MANIFEST_SCHEMA_VERSION:
            raise _error("schema_version", "unsupported_version")
        _require_ref(self.packet_ref, "packet_ref")
        _require_digest(self.packet_digest, "packet_digest")
        _require_language(self.language)
        _require_seed(self.seed)
        entries = _tuple_of(
            self.entries,
            "entries",
            NativeReviewManifestEntry,
            minimum=1,
            maximum=MAX_PACKET_ITEMS,
        )
        if len({entry.item_id for entry in entries}) != len(entries):
            raise _error("entries", "duplicate_item_id")
        if len({entry.source_ref for entry in entries}) != len(entries):
            raise _error("entries", "duplicate_source_ref")
        object.__setattr__(self, "entries", entries)

    @property
    def manifest_digest(self) -> str:
        """Return the SHA-256 digest of the canonical manifest document."""
        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return the evaluator-private manifest document."""
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "language": self.language,
            "packet_digest": self.packet_digest,
            "packet_ref": self.packet_ref,
            "schema_version": self.schema_version,
            "seed": self.seed,
        }

    def to_json(self) -> str:
        """Return canonical JSON for protected evaluator storage."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Any) -> NativeReviewManifest:
        """Parse and revalidate one strict manifest document."""
        data = _strict_fields(payload, "manifest", _field_names(cls))
        return cls(
            packet_ref=data["packet_ref"],
            packet_digest=data["packet_digest"],
            language=data["language"],
            seed=data["seed"],
            entries=tuple(
                NativeReviewManifestEntry.from_dict(entry)
                for entry in _list_field(data["entries"], "entries", MAX_PACKET_ITEMS)
            ),
            schema_version=data["schema_version"],
        )

    @classmethod
    def from_json(cls, payload: str | bytes) -> NativeReviewManifest:
        """Parse strict, bounded manifest JSON."""
        return cls.from_dict(_load_document(payload, "manifest", MAX_PACKET_BYTES))


@dataclass(frozen=True, slots=True)
class NativeReviewCompleteness:
    """Value-free completeness summary for one imported submission.

    ``missing_item_ids`` lists opaque packet item IDs, which already appear in
    the reviewer packet and reveal no source reference.
    """

    expected_count: int
    decided_count: int
    missing_item_ids: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        """Return whether every manifest item received a decision."""
        return not self.missing_item_ids

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic completeness document."""
        return {
            "complete": self.complete,
            "decided_count": self.decided_count,
            "expected_count": self.expected_count,
            "missing_item_ids": list(self.missing_item_ids),
        }


def _decision_id(packet_digest: str, item_id: str, submission_digest: str) -> str:
    return derived_opaque_id("nrd", packet_digest, item_id, submission_digest)


@dataclass(frozen=True, slots=True)
class NativeReviewDecision:
    """One imported native-review decision with manifest lineage.

    The record carries a reviewer-role code but no reviewer identity, free
    text, or source text. ``decision_id`` is derived from the packet digest,
    opaque item ID, and submission digest, so re-importing the same submission
    is idempotent while a changed resubmission yields a new decision that
    shares the ``(packet_ref, item_id)`` assignment key.

    Attributes:
        decision_id: Deterministic opaque decision identifier.
        packet_ref: Opaque reference of the reviewer assignment packet.
        packet_digest: Digest of the exact packet the reviewer saw.
        submission_digest: Digest of the normalized reviewer submission.
        item_id: Opaque packet item identifier.
        source_ref: Caller-owned source reference resolved from the manifest.
        language: Language tag the packet was reviewed in.
        task_code: Task the decision answers.
        privacy_status: Privacy status of the reviewed item.
        provenance_digest: Provenance digest of the reviewed item.
        text_digest: Digest of the reviewed text.
        projection_digest: Digest of the proposed spans the reviewer saw.
        reviewer_role: Closed reviewer-role code.
        decision: Closed decision code from the task vocabulary.
        reason_code: Optional closed reason code.
        corrections: Corrected span set when the decision requires one.
    """

    decision_id: str
    packet_ref: str
    packet_digest: str
    submission_digest: str
    item_id: str
    source_ref: str
    language: str
    task_code: str
    privacy_status: ReviewPrivacyStatus
    provenance_digest: str
    text_digest: str
    projection_digest: str
    reviewer_role: str
    decision: str
    reason_code: str | None = None
    corrections: tuple[NativeReviewSpan, ...] = ()
    schema_version: str = NATIVE_REVIEW_DECISION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != NATIVE_REVIEW_DECISION_SCHEMA_VERSION:
            raise _error("schema_version", "unsupported_version")
        _require_ref(self.packet_ref, "packet_ref")
        _require_digest(self.packet_digest, "packet_digest")
        _require_digest(self.submission_digest, "submission_digest")
        _require_item_id(self.item_id, "item_id")
        _require_ref(self.source_ref, "source_ref")
        _require_language(self.language)
        _require_code(self.task_code, "task_code")
        status = _require_privacy_status(self.privacy_status, "privacy_status")
        _require_digest(self.provenance_digest, "provenance_digest")
        _require_digest(self.text_digest, "text_digest")
        _require_digest(self.projection_digest, "projection_digest")
        _require_code(self.reviewer_role, "reviewer_role")
        _require_code(self.decision, "decision")
        if self.reason_code is not None:
            _require_code(self.reason_code, "reason_code")
        corrections = _span_tuple(self.corrections, "corrections")
        if type(self.decision_id) is not str or (
            _DECISION_ID_RE.fullmatch(self.decision_id) is None
        ):
            raise _error("decision_id", "invalid_decision_id")
        expected = _decision_id(
            self.packet_digest, self.item_id, self.submission_digest
        )
        if not hmac.compare_digest(self.decision_id, expected):
            raise _error("decision_id", "lineage_mismatch")
        object.__setattr__(self, "privacy_status", status)
        object.__setattr__(self, "corrections", corrections)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, identity-free decision document."""
        return {
            "corrections": [span.to_dict() for span in self.corrections],
            "decision": self.decision,
            "decision_id": self.decision_id,
            "item_id": self.item_id,
            "language": self.language,
            "packet_digest": self.packet_digest,
            "packet_ref": self.packet_ref,
            "privacy_status": self.privacy_status.value,
            "projection_digest": self.projection_digest,
            "provenance_digest": self.provenance_digest,
            "reason_code": self.reason_code,
            "reviewer_role": self.reviewer_role,
            "schema_version": self.schema_version,
            "source_ref": self.source_ref,
            "submission_digest": self.submission_digest,
            "task_code": self.task_code,
            "text_digest": self.text_digest,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> NativeReviewDecision:
        """Parse and revalidate one strict decision document."""
        data = dict(_strict_fields(payload, "decision_record", _field_names(cls)))
        data["corrections"] = tuple(
            NativeReviewSpan.from_dict(span)
            for span in _list_field(
                data["corrections"], "corrections", MAX_SPANS_PER_ITEM
            )
        )
        return cls(**data)


@dataclass(frozen=True, slots=True)
class NativeReviewImport:
    """Validated decisions imported from one reviewer submission."""

    packet_ref: str
    packet_digest: str
    manifest_digest: str
    submission_digest: str
    language: str
    reviewer_role: str
    decisions: tuple[NativeReviewDecision, ...]
    completeness: NativeReviewCompleteness
    schema_version: str = NATIVE_REVIEW_IMPORT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, identity-free import document."""
        return {
            "completeness": self.completeness.to_dict(),
            "decisions": [decision.to_dict() for decision in self.decisions],
            "language": self.language,
            "manifest_digest": self.manifest_digest,
            "packet_digest": self.packet_digest,
            "packet_ref": self.packet_ref,
            "reviewer_role": self.reviewer_role,
            "schema_version": self.schema_version,
            "submission_digest": self.submission_digest,
        }

    def to_json(self) -> str:
        """Return canonical JSON for the imported decisions."""
        return canonical_json(self.to_dict())


def _order_key(seed: int, source_ref: str) -> bytes:
    material = canonical_json(
        {"domain": _ORDER_DOMAIN, "seed": seed, "source_ref": source_ref}
    )
    return hashlib.sha256(material.encode("utf-8")).digest()


def _provenance_commitment(
    packet_ref: str, entries: Sequence[NativeReviewManifestEntry]
) -> str:
    # The commitment binds the private source mapping to the packet, so a
    # swapped or edited manifest entry fails verification. Source references
    # are hashed together with unpublished provenance digests and are not
    # recoverable from the packet.
    return canonical_digest(
        {
            "domain": _PROVENANCE_DOMAIN,
            "items": [
                {
                    "item_id": entry.item_id,
                    "privacy_status": entry.privacy_status.value,
                    "provenance_digest": entry.provenance_digest,
                    "source_ref": entry.source_ref,
                }
                for entry in entries
            ],
            "packet_ref": packet_ref,
        }
    )


def build_native_review_packet(
    items: Sequence[NativeReviewItem],
    *,
    packet_ref: str,
    language: str,
    tasks: Sequence[NativeReviewTask],
    seed: int,
    reviewer_roles: Sequence[str] = DEFAULT_REVIEWER_ROLES,
    max_item_characters: int = DEFAULT_MAX_ITEM_CHARACTERS,
) -> tuple[NativeReviewPacket, NativeReviewManifest]:
    """Build a deterministic native-review packet and its private manifest.

    Items are ordered by a seeded keyed hash of their source reference, so the
    same inputs and seed produce byte-identical packets and manifests in any
    input order. Opaque item IDs are derived from the packet reference, seed,
    position, and reviewed content; they never encode the source reference.
    Only tasks used by at least one item are written to the packet.

    Issue one packet per reviewer assignment, with a distinct ``packet_ref``
    and ``seed``, so returned decisions stay separable without naming the
    reviewer.

    Args:
        items: Synthetic or already de-identified review items.
        packet_ref: Opaque reference for this reviewer assignment.
        language: Language tag the reviewer is native in, such as ``sw`` or
            ``pt-BR``.
        tasks: Task definitions referenced by ``items``.
        seed: Non-negative integer seed for the randomized item order.
        reviewer_roles: Closed reviewer-role codes accepted on import.
        max_item_characters: Upper bound on each item's text, keeping the
            review context minimal. At most ``MAX_ITEM_CHARACTERS``.

    Returns:
        The reviewer-facing packet and the evaluator-private manifest.

    Raises:
        NativeReviewPacketError: If an input is malformed, duplicated, exceeds
            the minimal-context bound, or references an unknown task or label.
    """
    packet_ref = _require_ref(packet_ref, "packet_ref")
    language = _require_language(language)
    seed = _require_seed(seed)
    if (
        type(max_item_characters) is not int
        or not 1 <= max_item_characters <= MAX_ITEM_CHARACTERS
    ):
        raise _error("max_item_characters", "invalid_limit")

    normalized_items = _tuple_of(
        items, "items", NativeReviewItem, minimum=1, maximum=MAX_PACKET_ITEMS
    )
    if len({item.source_ref for item in normalized_items}) != len(normalized_items):
        raise _error("items", "duplicate_source_ref")
    normalized_tasks = _tuple_of(
        tasks, "tasks", NativeReviewTask, minimum=1, maximum=MAX_TASKS
    )
    task_by_code = {task.task_code: task for task in normalized_tasks}
    if len(task_by_code) != len(normalized_tasks):
        raise _error("tasks", "duplicate_code")

    for index, item in enumerate(normalized_items):
        # Re-check the privacy gate even though construction enforces it.
        _require_privacy_status(item.privacy_status, f"items[{index}].privacy_status")
        task = task_by_code.get(item.task_code)
        if task is None:
            raise _error(f"items[{index}].task_code", "unknown_task")
        if len(item.text) > max_item_characters:
            raise _error(f"items[{index}].text", "exceeds_minimal_context")
        if any(span.label not in task.label_codes for span in item.spans):
            raise _error(f"items[{index}].spans", "unknown_label")

    ordered = sorted(
        normalized_items,
        key=lambda item: (_order_key(seed, item.source_ref), item.source_ref),
    )
    packet_items: list[NativeReviewPacketItem] = []
    entries: list[NativeReviewManifestEntry] = []
    for position, item in enumerate(ordered):
        text_digest = sha256_digest(item.text)
        projection_digest = _projection_digest(item.spans)
        item_id = derived_opaque_id(
            "nri",
            packet_ref,
            seed,
            position,
            item.task_code,
            text_digest,
            projection_digest,
        )
        packet_items.append(
            NativeReviewPacketItem(
                item_id=item_id,
                task_code=item.task_code,
                privacy_status=item.privacy_status,
                text=item.text,
                spans=item.spans,
            )
        )
        entries.append(
            NativeReviewManifestEntry(
                item_id=item_id,
                source_ref=item.source_ref,
                task_code=item.task_code,
                privacy_status=item.privacy_status,
                provenance_digest=item.provenance_digest,
                text_digest=text_digest,
                projection_digest=projection_digest,
            )
        )

    used_task_codes = {item.task_code for item in normalized_items}
    packet = NativeReviewPacket(
        packet_ref=packet_ref,
        language=language,
        tasks=tuple(task_by_code[code] for code in sorted(used_task_codes)),
        items=tuple(packet_items),
        reviewer_roles=reviewer_roles,
        provenance_commitment=_provenance_commitment(packet_ref, entries),
    )
    manifest = NativeReviewManifest(
        packet_ref=packet_ref,
        packet_digest=packet.packet_digest,
        language=language,
        seed=seed,
        entries=tuple(entries),
    )
    return packet, manifest


def verify_native_review_manifest(
    packet: NativeReviewPacket, manifest: NativeReviewManifest
) -> None:
    """Verify that a manifest describes exactly the given packet.

    Args:
        packet: The reviewer-facing packet as it was distributed.
        manifest: The evaluator-private manifest built with it.

    Raises:
        NativeReviewPacketError: With a closed code when the packet reference,
            language, digest, item bindings, or provenance commitment differ.
    """
    if not isinstance(packet, NativeReviewPacket):
        raise _error("packet", "invalid_packet")
    if not isinstance(manifest, NativeReviewManifest):
        raise _error("manifest", "invalid_manifest")
    if manifest.packet_ref != packet.packet_ref or manifest.language != packet.language:
        raise _error("manifest", "packet_mismatch")
    if not hmac.compare_digest(manifest.packet_digest, packet.packet_digest):
        raise _error("manifest", "packet_digest_mismatch")
    if len(manifest.entries) != len(packet.items) or not all(
        entry.item_id == item.item_id
        and entry.task_code == item.task_code
        and entry.privacy_status is item.privacy_status
        and entry.text_digest == item.text_digest
        and entry.projection_digest == item.projection_digest
        for entry, item in zip(manifest.entries, packet.items)
    ):
        raise _error("manifest", "item_mismatch")
    expected = _provenance_commitment(manifest.packet_ref, manifest.entries)
    if not hmac.compare_digest(expected, packet.provenance_commitment):
        raise _error("manifest", "provenance_mismatch")


_SUBMISSION_FIELDS = frozenset(
    {"decisions", "packet_digest", "packet_ref", "reviewer_role", "schema_version"}
)
_SUBMISSION_DECISION_REQUIRED = frozenset({"decision", "item_id"})
_SUBMISSION_DECISION_OPTIONAL = frozenset({"corrections", "reason_code"})


@dataclass(frozen=True, slots=True)
class _Answer:
    decision: str
    reason_code: str | None
    corrections: tuple[NativeReviewSpan, ...]

    def to_dict(self, item_id: str) -> dict[str, Any]:
        return {
            "corrections": [span.to_dict() for span in self.corrections],
            "decision": self.decision,
            "item_id": item_id,
            "reason_code": self.reason_code,
        }


def _parse_answer(
    row: Mapping[str, Any], item: NativeReviewPacketItem, task: NativeReviewTask
) -> _Answer | None:
    decision = row["decision"]
    reason_code = row.get("reason_code")
    raw_corrections = row.get("corrections")
    corrections = _span_tuple(
        tuple(
            NativeReviewSpan.from_dict(span)
            for span in _list_field(
                [] if raw_corrections is None else raw_corrections,
                "decisions.corrections",
                MAX_SPANS_PER_ITEM,
            )
        ),
        "decisions.corrections",
    )
    if decision is None:
        if reason_code is not None or corrections:
            raise _error("decisions", "undecided_with_details")
        return None
    if type(decision) is not str or decision not in task.decision_codes:
        raise _error("decisions.decision", "unknown_code")
    if reason_code is not None and (
        type(reason_code) is not str or reason_code not in task.reason_codes
    ):
        raise _error("decisions.reason_code", "unknown_code")
    if any(span.label not in task.label_codes for span in corrections):
        raise _error("decisions.corrections", "unknown_label")
    _check_span_offsets(item.text, corrections, "decisions.corrections")
    if decision == task.correction_decision_code:
        if not corrections:
            raise _error("decisions.corrections", "required")
    elif corrections:
        raise _error("decisions.corrections", "not_allowed")
    return _Answer(decision, reason_code, corrections)


def import_native_review_decisions(
    submission: Mapping[str, Any] | str | bytes,
    *,
    packet: NativeReviewPacket,
    manifest: NativeReviewManifest,
    require_complete: bool = True,
) -> NativeReviewImport:
    """Import one reviewer submission through opaque packet item IDs.

    The submission must name the exact packet, use a reviewer-role code the
    packet allows, and answer items only with closed decision, reason, and
    label codes. Fields that look like reviewer identity or free text are
    rejected, so neither can reach the imported decisions. A ``null``
    decision marks an item as not yet decided.

    Args:
        submission: Submission document as a mapping or strict JSON text.
        packet: The reviewer-facing packet that was distributed.
        manifest: The evaluator-private manifest for that packet.
        require_complete: When true, fail unless every manifest item has a
            decision. When false, return the decided subset with a
            completeness summary.

    Returns:
        Validated, identity-free decisions in packet order plus completeness.

    Raises:
        NativeReviewIncompleteError: If ``require_complete`` is true and any
            manifest item lacks a decision.
        NativeReviewPacketError: With a closed code for any other mismatch,
            unknown or duplicate item, invalid code, forbidden field, or
            invalid correction.
    """
    verify_native_review_manifest(packet, manifest)
    data = _strict_fields(
        _load_document(submission, "submission", MAX_SUBMISSION_BYTES),
        "submission",
        _SUBMISSION_FIELDS,
    )
    if data["schema_version"] != NATIVE_REVIEW_SUBMISSION_SCHEMA_VERSION:
        raise _error("submission.schema_version", "unsupported_version")
    packet_digest = packet.packet_digest
    if (
        data["packet_ref"] != packet.packet_ref
        or data["packet_digest"] != packet_digest
    ):
        raise _error("submission", "packet_mismatch")
    reviewer_role = data["reviewer_role"]
    if type(reviewer_role) is not str or reviewer_role not in packet.reviewer_roles:
        raise _error("submission.reviewer_role", "not_allowed")

    item_by_id = {item.item_id: item for item in packet.items}
    task_by_code = {task.task_code: task for task in packet.tasks}
    answers: dict[str, _Answer | None] = {}
    for raw in _list_field(data["decisions"], "decisions", MAX_PACKET_ITEMS):
        row = _strict_fields(
            raw,
            "decisions",
            _SUBMISSION_DECISION_REQUIRED,
            _SUBMISSION_DECISION_OPTIONAL,
        )
        item_id = row["item_id"]
        if type(item_id) is not str or item_id not in item_by_id:
            raise _error("decisions", "unknown_item")
        if item_id in answers:
            raise _error("decisions", "duplicate_item")
        item = item_by_id[item_id]
        answers[item_id] = _parse_answer(row, item, task_by_code[item.task_code])

    missing = tuple(
        entry.item_id
        for entry in manifest.entries
        if answers.get(entry.item_id) is None
    )
    completeness = NativeReviewCompleteness(
        expected_count=len(manifest.entries),
        decided_count=len(manifest.entries) - len(missing),
        missing_item_ids=missing,
    )
    if require_complete and missing:
        raise NativeReviewIncompleteError(completeness)

    decided = tuple(
        (entry, answer)
        for entry in manifest.entries
        if (answer := answers.get(entry.item_id)) is not None
    )
    # The digest covers the normalized decisions in packet order, so the
    # reviewer's row order and omitted optional fields do not change it.
    submission_digest = canonical_digest(
        {
            "decisions": [answer.to_dict(entry.item_id) for entry, answer in decided],
            "packet_digest": packet_digest,
            "packet_ref": packet.packet_ref,
            "reviewer_role": reviewer_role,
            "schema_version": NATIVE_REVIEW_SUBMISSION_SCHEMA_VERSION,
        }
    )
    decisions = tuple(
        NativeReviewDecision(
            decision_id=_decision_id(packet_digest, entry.item_id, submission_digest),
            packet_ref=packet.packet_ref,
            packet_digest=packet_digest,
            submission_digest=submission_digest,
            item_id=entry.item_id,
            source_ref=entry.source_ref,
            language=manifest.language,
            task_code=entry.task_code,
            privacy_status=entry.privacy_status,
            provenance_digest=entry.provenance_digest,
            text_digest=entry.text_digest,
            projection_digest=entry.projection_digest,
            reviewer_role=reviewer_role,
            decision=answer.decision,
            reason_code=answer.reason_code,
            corrections=answer.corrections,
        )
        for entry, answer in decided
    )
    return NativeReviewImport(
        packet_ref=packet.packet_ref,
        packet_digest=packet_digest,
        manifest_digest=manifest.manifest_digest,
        submission_digest=submission_digest,
        language=manifest.language,
        reviewer_role=reviewer_role,
        decisions=decisions,
        completeness=completeness,
    )


__all__ = [
    "DEFAULT_DECISION_CODES",
    "DEFAULT_MAX_ITEM_CHARACTERS",
    "DEFAULT_REVIEWER_ROLES",
    "MAX_ITEM_CHARACTERS",
    "MAX_PACKET_ITEMS",
    "MAX_SPANS_PER_ITEM",
    "NATIVE_REVIEW_DECISION_SCHEMA_VERSION",
    "NATIVE_REVIEW_IMPORT_SCHEMA_VERSION",
    "NATIVE_REVIEW_MANIFEST_SCHEMA_VERSION",
    "NATIVE_REVIEW_NOTICE",
    "NATIVE_REVIEW_PACKET_SCHEMA_VERSION",
    "NATIVE_REVIEW_SUBMISSION_SCHEMA_VERSION",
    "NativeReviewCompleteness",
    "NativeReviewDecision",
    "NativeReviewImport",
    "NativeReviewIncompleteError",
    "NativeReviewItem",
    "NativeReviewManifest",
    "NativeReviewManifestEntry",
    "NativeReviewPacket",
    "NativeReviewPacketError",
    "NativeReviewPacketItem",
    "NativeReviewSpan",
    "NativeReviewTask",
    "ReviewPrivacyStatus",
    "build_native_review_packet",
    "import_native_review_decisions",
    "verify_native_review_manifest",
]
