"""Explicit adjudication revisions for native-review language labels.

Native-language reviewers return decisions about proposed label projections,
and those decisions can conflict. This module merges imported reviewer
decisions into accepted training and evaluation labels only through an
explicit :class:`AdjudicationPolicy`, and records every merge as an immutable,
content-addressed :class:`AdjudicationRevision`.

The merge is conservative:

* Each reviewer decision is an immutable :class:`DecisionOverlay` carrying a
  reviewer-role code, packet and submission digests, source/projection digests
  and an optional explicit supersession pointer. Reviewer identity and free
  text are refused on input.
* Disagreeing or abstaining reviewers leave an item ``unresolved`` until a
  valid :class:`AdjudicationEvent` bound to the current evidence, text,
  projection and policy is supplied. Conflicting adjudication heads stay
  unresolved too.
* Decisions about a stale projection or changed underlying text, ambiguous
  duplicate decisions, and offsets that no longer land on grapheme boundaries
  invalidate an item, and revoke any acceptance recorded by the parent
  revision.

Accepted labels export with their decision lineage (opaque decision and event
identifiers, reviewer-role codes and digests) but without reviewer identity,
assignment references or source text. Audit reports go further and replace
source references with subject digests. Everything runs locally with no
network call; exceptions carry field names and closed codes, never values.

Decision documents follow the ``openmed.eval.native_review.decision.v1``
contract, so records produced by the native-review packet importer can be
passed in directly, either as mappings or as objects exposing ``to_dict()``.
This module supports label curation only; it makes no clinical decision.
"""

from __future__ import annotations

import hmac
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, TypeVar

from openmed.clinical.journey_contracts import (
    canonical_digest,
    canonical_json,
    derived_opaque_id,
    sha256_digest,
)
from openmed.core.decoding.spans import grapheme_break_checker
from openmed.core.schemas import hmac_text_hash
from openmed.eval.annotation.interchange import (
    AnnotationEnvelope,
    AnnotationRecord,
    AnnotationState,
    AnnotationType,
    CoordinateConvention,
    build_annotation_envelope,
)
from openmed.eval.reviewer_disagreement import DisagreementReason, ReviewerDecision

NATIVE_REVIEW_DECISION_SCHEMA_VERSION = "openmed.eval.native_review.decision.v1"
ADJUDICATION_POLICY_SCHEMA_VERSION = (
    "openmed.training.native_review_adjudication.policy.v1"
)
ADJUDICATION_EVENT_SCHEMA_VERSION = (
    "openmed.training.native_review_adjudication.event.v1"
)
ADJUDICATION_REVISION_SCHEMA_VERSION = (
    "openmed.training.native_review_adjudication.revision.v1"
)
ADJUDICATED_LABELS_SCHEMA_VERSION = (
    "openmed.training.native_review_adjudication.labels.v1"
)
ADJUDICATION_AUDIT_SCHEMA_VERSION = (
    "openmed.training.native_review_adjudication.audit.v1"
)

DEFAULT_REVIEWER_ROLES = (
    "clinical_linguist",
    "medical_translator",
    "native_clinician",
    "native_speaker",
)
MAX_ITEMS = 10_000
MAX_DECISIONS = 100_000
MAX_EVENTS = 100_000
MAX_SPANS_PER_ITEM = 64
MAX_TEXT_CHARACTERS = 4_000
MAX_MINIMUM_REVIEWERS = 16
ANNOTATION_SOURCE_FORMAT = "native_review"

# Blocking conditions and excluded evidence, reported as closed codes.
REASON_ADJUDICATION_DEFERRED = "adjudication_deferred"
REASON_ADJUDICATION_POLICY_MISMATCH = "adjudication_policy_mismatch"
REASON_ADJUDICATION_REQUIRED = "adjudication_required"
REASON_ADJUDICATION_SCOPE_MISMATCH = "adjudication_scope_mismatch"
REASON_ADJUDICATION_STALE = "adjudication_stale"
REASON_ADJUDICATOR_ROLE_NOT_ALLOWED = "adjudicator_role_not_allowed"
REASON_CONFLICTING_ADJUDICATION = "conflicting_adjudication"
REASON_DECISION_SCOPE_MISMATCH = "decision_scope_mismatch"
REASON_DUPLICATE_DECISION = "duplicate_decision"
REASON_INSUFFICIENT_REVIEWS = "insufficient_reviews"
REASON_INVALID_CORRECTION = "invalid_correction"
REASON_OFFSET_OUT_OF_RANGE = "offset_out_of_range"
REASON_REVIEWER_ABSTAINED = "reviewer_abstained"
REASON_REVIEWER_DISAGREEMENT = "reviewer_disagreement"
REASON_REVIEWER_ROLE_NOT_COUNTED = "reviewer_role_not_counted"
REASON_SPLIT_GRAPHEME = "split_grapheme"
REASON_STALE_PROJECTION = "stale_projection"
REASON_SUPERSESSION_SCOPE_MISMATCH = "supersession_scope_mismatch"
REASON_TEXT_CHANGED = "text_changed"
REASON_UNKNOWN_LABEL = "unknown_label"

REASON_CODES = frozenset(
    {
        REASON_ADJUDICATION_DEFERRED,
        REASON_ADJUDICATION_POLICY_MISMATCH,
        REASON_ADJUDICATION_REQUIRED,
        REASON_ADJUDICATION_SCOPE_MISMATCH,
        REASON_ADJUDICATION_STALE,
        REASON_ADJUDICATOR_ROLE_NOT_ALLOWED,
        REASON_CONFLICTING_ADJUDICATION,
        REASON_DECISION_SCOPE_MISMATCH,
        REASON_DUPLICATE_DECISION,
        REASON_INSUFFICIENT_REVIEWS,
        REASON_INVALID_CORRECTION,
        REASON_OFFSET_OUT_OF_RANGE,
        REASON_REVIEWER_ABSTAINED,
        REASON_REVIEWER_DISAGREEMENT,
        REASON_REVIEWER_ROLE_NOT_COUNTED,
        REASON_SPLIT_GRAPHEME,
        REASON_STALE_PROJECTION,
        REASON_SUPERSESSION_SCOPE_MISMATCH,
        REASON_TEXT_CHANGED,
        REASON_UNKNOWN_LABEL,
    }
)

_COORDINATE_CONVENTION = CoordinateConvention.UNICODE_CODEPOINT.value
_PUBLIC_REF_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_CODE_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_LABEL_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,63}")
_LANGUAGE_RE = re.compile(r"[a-z]{2,3}(?:-[A-Za-z0-9]{2,8}){0,3}")
_VERSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,31}")
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_ITEM_ID_RE = re.compile(r"nri_[0-9a-f]{32}")
_DECISION_ID_RE = re.compile(r"nrd_[0-9a-f]{32}")
_EVENT_ID_RE = re.compile(r"nra_[0-9a-f]{32}")
_REVISION_ID_RE = re.compile(r"nrr_[0-9a-f]{32}")
# Shared with the native-review packet contract so that a label set accepted
# unchanged has the same digest as the projection the reviewers saw.
_SPAN_SET_DOMAIN = "openmed.eval.native_review.projection.v1"
_BASIS_DOMAIN = "openmed.training.native_review_adjudication.basis.v1"
_INPUT_DOMAIN = "openmed.training.native_review_adjudication.input.v1"
_SUBJECT_DOMAIN = "openmed.training.native_review_adjudication.subject.v1"
_PRIVACY_STATUSES = frozenset({"deidentified", "synthetic"})
_ALLOWED_CONTROL_CHARACTERS = frozenset({"\t", "\n", "\r"})
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
        "concordance",
        "context",
        "excerpt",
        "explanation",
        "free_text",
        "note",
        "notes",
        "rationale",
        "raw_text",
        "remarks",
        "snippet",
        "source_text",
        "surface",
        "text",
        "window",
    }
)
_DECISION_FIELDS = frozenset(
    {
        "corrections",
        "decision",
        "decision_id",
        "item_id",
        "language",
        "packet_digest",
        "packet_ref",
        "privacy_status",
        "projection_digest",
        "provenance_digest",
        "reason_code",
        "reviewer_role",
        "schema_version",
        "source_ref",
        "submission_digest",
        "task_code",
        "text_digest",
    }
)

_EnumT = TypeVar("_EnumT", bound=Enum)


class NativeReviewAdjudicationError(ValueError):
    """Raised when adjudication inputs or records are malformed or unsafe."""


class LabelState(str, Enum):
    """Closed states for one adjudicated item."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNRESOLVED = "unresolved"
    PENDING = "pending"
    INVALIDATED = "invalidated"


class AdjudicationOutcome(str, Enum):
    """Closed outcomes an adjudication event may record."""

    ACCEPT_PROJECTION = "accept_projection"
    ACCEPT_SPANS = "accept_spans"
    REJECT_PROJECTION = "reject_projection"
    DEFER = "defer"


class Resolution(str, Enum):
    """How an accepted or rejected item was resolved."""

    UNANIMOUS_AGREEMENT = "unanimous_agreement"
    ADJUDICATION_EVENT = "adjudication_event"


class VoteKind(str, Enum):
    """How the policy interprets one reviewer decision code."""

    LABELS = "labels"
    REJECT = "reject"
    ABSTAIN = "abstain"


def _error(field_name: str, code: str) -> NativeReviewAdjudicationError:
    return NativeReviewAdjudicationError(f"{field_name}: {code}")


def _require(value: Any, pattern: re.Pattern[str], field_name: str, code: str) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise _error(field_name, code)
    return value


def _require_ref(value: Any, field_name: str) -> str:
    return _require(value, _PUBLIC_REF_RE, field_name, "invalid_reference")


def _require_code(value: Any, field_name: str) -> str:
    return _require(value, _CODE_RE, field_name, "invalid_code")


def _require_digest(value: Any, field_name: str) -> str:
    return _require(value, _DIGEST_RE, field_name, "invalid_digest")


def _require_language(value: Any) -> str:
    return _require(value, _LANGUAGE_RE, "language", "invalid_language")


def _optional(
    value: Any, validator: Callable[[Any, str], str], field_name: str
) -> str | None:
    return None if value is None else validator(value, field_name)


def _require_enum(value: Any, enum_type: type[_EnumT], field_name: str) -> _EnumT:
    # Look members up without raising so no exception context holds the value.
    if isinstance(value, enum_type):
        return value
    if type(value) is str:
        for member in enum_type:
            if member.value == value:
                return member
    raise _error(field_name, "invalid_code")


def _require_bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise _error(field_name, "invalid_flag")
    return value


def _codes(
    values: Any,
    field_name: str,
    *,
    minimum: int,
    maximum: int,
    validator: Callable[[Any, str], str] = _require_code,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Iterable):
        raise _error(field_name, "invalid_sequence")
    codes = tuple(validator(value, field_name) for value in values)
    if not minimum <= len(codes) <= maximum:
        raise _error(field_name, "invalid_count")
    if len(set(codes)) != len(codes):
        raise _error(field_name, "duplicate_code")
    return tuple(sorted(codes))


def _id_tuple(values: Any, field_name: str, pattern: re.Pattern[str]) -> tuple:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Iterable):
        raise _error(field_name, "invalid_sequence")
    identifiers = tuple(
        _require(value, pattern, field_name, "invalid_identifier") for value in values
    )
    if len(identifiers) > MAX_DECISIONS or len(set(identifiers)) != len(identifiers):
        raise _error(field_name, "invalid_sequence")
    return tuple(sorted(identifiers))


def _bounded(values: Any, field_name: str, maximum: int) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Iterable):
        raise _error(field_name, "invalid_sequence")
    materialized: list[Any] = []
    for value in values:
        if len(materialized) >= maximum:
            raise _error(field_name, "limit_exceeded")
        materialized.append(value)
    return tuple(materialized)


def _list_field(value: Any, field_name: str, maximum: int) -> list[Any]:
    if not isinstance(value, (list, tuple)) or len(value) > maximum:
        raise _error(field_name, "invalid_sequence")
    return list(value)


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


def _field_names(cls: type) -> frozenset[str]:
    return frozenset(item.name for item in fields(cls))


def _span_key(span: ReviewSpan) -> tuple[int, int, str]:
    return span.start, span.end, span.label


def _span_tuple(values: Any, field_name: str) -> tuple[ReviewSpan, ...]:
    spans = _bounded(values, field_name, MAX_SPANS_PER_ITEM)
    if not all(isinstance(span, ReviewSpan) for span in spans):
        raise _error(field_name, "invalid_sequence")
    ordered = tuple(sorted(spans, key=_span_key))
    if len(set(ordered)) != len(ordered):
        raise _error(field_name, "duplicate_span")
    return ordered


def _parse_spans(value: Any, field_name: str) -> tuple[ReviewSpan, ...]:
    return tuple(
        ReviewSpan.from_dict(span)
        for span in _list_field(value, field_name, MAX_SPANS_PER_ITEM)
    )


def _span_set_digest(spans: Sequence[ReviewSpan]) -> str:
    return canonical_digest(
        {
            "coordinate_convention": _COORDINATE_CONVENTION,
            "domain": _SPAN_SET_DOMAIN,
            "spans": [span.to_dict() for span in spans],
        }
    )


def _basis_digest(decision_ids: Iterable[str]) -> str:
    return canonical_digest(
        {"decision_ids": sorted(decision_ids), "domain": _BASIS_DOMAIN}
    )


def _decision_id(packet_digest: str, item_id: str, submission_digest: str) -> str:
    # Matches the native-review import contract, so imported identifiers
    # verify without importing the packet module.
    return derived_opaque_id("nrd", packet_digest, item_id, submission_digest)


def _require_text(value: Any) -> str:
    if type(value) is not str or not value.strip():
        raise _error("text", "invalid_text")
    if len(value) > MAX_TEXT_CHARACTERS:
        raise _error("text", "too_long")
    for character in value:
        codepoint = ord(character)
        if (
            codepoint < 32 and character not in _ALLOWED_CONTROL_CHARACTERS
        ) or codepoint == 127:
            raise _error("text", "control_character")
        if 0xD800 <= codepoint <= 0xDFFF:
            raise _error("text", "invalid_text")
    return value


def _span_problem(
    spans: Sequence[ReviewSpan],
    text: str,
    has_break: Callable[[int], bool],
    label_codes: frozenset[str],
) -> str | None:
    """Return the first closed problem code for ``spans`` over ``text``."""
    for span in spans:
        if span.label not in label_codes:
            return REASON_UNKNOWN_LABEL
        if span.end > len(text):
            return REASON_OFFSET_OUT_OF_RANGE
        if not (has_break(span.start) and has_break(span.end)):
            return REASON_SPLIT_GRAPHEME
    return None


@dataclass(frozen=True, slots=True)
class ReviewSpan:
    """A label over half-open Unicode code-point offsets.

    Offsets follow the annotation-interchange ``unicode_codepoint`` convention:
    ``start`` is inclusive and ``end`` is exclusive.
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
        _require(self.label, _LABEL_RE, "span.label", "invalid_label")

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical span document."""
        return {"end": self.end, "label": self.label, "start": self.start}

    @classmethod
    def from_dict(cls, payload: Any) -> ReviewSpan:
        """Parse one strict span document."""
        return cls(**_strict_fields(payload, "span", _field_names(cls)))


@dataclass(frozen=True, slots=True)
class DecisionOverlay:
    """One immutable reviewer decision layered over a label projection.

    The overlay mirrors an imported native-review decision and adds an
    optional explicit supersession pointer. It carries a closed reviewer-role
    code and digest lineage, never reviewer identity, free text or source
    text. ``decision_id`` must match the native-review derivation from the
    packet digest, opaque item identifier and submission digest.

    Decisions that share the ``(packet_ref, item_id)`` assignment key are
    resubmissions of one assignment. Exactly one of them may be current: a
    newer decision names the one it replaces in ``supersedes_decision_id``.
    Two current decisions for one assignment are treated as duplicates.

    Attributes:
        decision_id: Deterministic ``nrd_`` decision identifier.
        packet_ref: Pseudonymous reviewer-assignment reference. Used only to
            group resubmissions; never exported.
        packet_digest: Digest of the packet the reviewer saw.
        submission_digest: Digest of the normalized reviewer submission.
        item_id: Opaque ``nri_`` packet item identifier.
        source_ref: Caller-owned opaque reference of the reviewed source.
        language: Language tag the item was reviewed in.
        task_code: Task the decision answers.
        privacy_status: ``synthetic`` or ``deidentified``.
        provenance_digest: Provenance digest of the reviewed item.
        text_digest: Digest of the text the reviewer saw.
        projection_digest: Digest of the proposed spans the reviewer saw.
        reviewer_role: Closed reviewer-role code.
        decision: Closed decision code from the task vocabulary.
        reason_code: Optional closed reason code.
        corrections: Corrected span set for revise decisions.
        supersedes_decision_id: Decision this resubmission replaces.
    """

    decision_id: str
    packet_ref: str
    packet_digest: str
    submission_digest: str
    item_id: str
    source_ref: str
    language: str
    task_code: str
    privacy_status: str
    provenance_digest: str
    text_digest: str
    projection_digest: str
    reviewer_role: str
    decision: str
    reason_code: str | None = None
    corrections: tuple[ReviewSpan, ...] = ()
    supersedes_decision_id: str | None = None
    schema_version: str = NATIVE_REVIEW_DECISION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != NATIVE_REVIEW_DECISION_SCHEMA_VERSION:
            raise _error("decision.schema_version", "unsupported_version")
        _require_ref(self.packet_ref, "decision.packet_ref")
        _require_digest(self.packet_digest, "decision.packet_digest")
        _require_digest(self.submission_digest, "decision.submission_digest")
        _require(self.item_id, _ITEM_ID_RE, "decision.item_id", "invalid_item_id")
        _require_ref(self.source_ref, "decision.source_ref")
        _require_language(self.language)
        _require_code(self.task_code, "decision.task_code")
        if self.privacy_status not in _PRIVACY_STATUSES:
            raise _error("decision.privacy_status", "not_reviewable")
        _require_digest(self.provenance_digest, "decision.provenance_digest")
        _require_digest(self.text_digest, "decision.text_digest")
        _require_digest(self.projection_digest, "decision.projection_digest")
        _require_code(self.reviewer_role, "decision.reviewer_role")
        _require_code(self.decision, "decision.decision")
        _optional(self.reason_code, _require_code, "decision.reason_code")
        corrections = _span_tuple(self.corrections, "decision.corrections")
        _require(
            self.decision_id,
            _DECISION_ID_RE,
            "decision.decision_id",
            "invalid_decision_id",
        )
        expected = _decision_id(
            self.packet_digest, self.item_id, self.submission_digest
        )
        if not hmac.compare_digest(self.decision_id, expected):
            raise _error("decision.decision_id", "lineage_mismatch")
        if self.supersedes_decision_id is not None:
            _require(
                self.supersedes_decision_id,
                _DECISION_ID_RE,
                "decision.supersedes_decision_id",
                "invalid_decision_id",
            )
            if self.supersedes_decision_id == self.decision_id:
                raise _error("decision.supersedes_decision_id", "self_supersession")
        object.__setattr__(self, "corrections", corrections)

    @property
    def assignment_key(self) -> tuple[str, str]:
        """Return the ``(packet_ref, item_id)`` resubmission key."""
        return self.packet_ref, self.item_id

    @property
    def overlay_digest(self) -> str:
        """Return the digest of the complete overlay document."""
        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return the evaluator-side overlay document."""
        return {
            "corrections": [span.to_dict() for span in self.corrections],
            "decision": self.decision,
            "decision_id": self.decision_id,
            "item_id": self.item_id,
            "language": self.language,
            "packet_digest": self.packet_digest,
            "packet_ref": self.packet_ref,
            "privacy_status": self.privacy_status,
            "projection_digest": self.projection_digest,
            "provenance_digest": self.provenance_digest,
            "reason_code": self.reason_code,
            "reviewer_role": self.reviewer_role,
            "schema_version": self.schema_version,
            "source_ref": self.source_ref,
            "submission_digest": self.submission_digest,
            "supersedes_decision_id": self.supersedes_decision_id,
            "task_code": self.task_code,
            "text_digest": self.text_digest,
        }

    @classmethod
    def from_decision(
        cls, decision: Any, *, supersedes_decision_id: str | None = None
    ) -> DecisionOverlay:
        """Build an overlay from an imported native-review decision.

        Args:
            decision: A :class:`DecisionOverlay`, a decision mapping, or an
                object whose ``to_dict()`` returns one, such as an imported
                native-review decision record.
            supersedes_decision_id: Optional decision of the same assignment
                that this resubmission replaces.

        Returns:
            The validated, immutable overlay.

        Raises:
            NativeReviewAdjudicationError: With a closed code when the record
                is malformed, carries reviewer identity or free text, or has a
                decision identifier that does not match its lineage.
        """
        if isinstance(decision, DecisionOverlay):
            if supersedes_decision_id is None:
                return decision
            payload: Any = decision.to_dict()
        elif isinstance(decision, Mapping):
            payload = decision
        else:
            to_dict = getattr(decision, "to_dict", None)
            if not callable(to_dict):
                raise _error("decision", "invalid_decision")
            payload = to_dict()
        data = dict(
            _strict_fields(
                payload,
                "decision",
                _DECISION_FIELDS,
                frozenset({"supersedes_decision_id"}),
            )
        )
        if supersedes_decision_id is not None:
            if data.get("supersedes_decision_id") not in (
                None,
                supersedes_decision_id,
            ):
                raise _error("decision.supersedes_decision_id", "conflicting_pointer")
            data["supersedes_decision_id"] = supersedes_decision_id
        data["corrections"] = _parse_spans(data["corrections"], "decision.corrections")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class LabelProjection:
    """The current text and proposed labels for one reviewed source item.

    The projection is the base layer that decision overlays and adjudication
    events refer to by digest. Its text is held in memory only to compute the
    text digest and to revalidate offsets on grapheme boundaries; it is never
    serialized, logged or exported.

    Attributes:
        source_ref: Caller-owned opaque reference, unique per merge.
        language: Language tag of the reviewed text.
        task_code: Review task the decisions answer.
        text: The current reviewed segment.
        spans: Current proposed labels over ``text``.
    """

    source_ref: str
    language: str
    task_code: str
    text: str = field(repr=False)
    spans: tuple[ReviewSpan, ...] = ()

    def __post_init__(self) -> None:
        _require_ref(self.source_ref, "projection.source_ref")
        _require_language(self.language)
        _require_code(self.task_code, "projection.task_code")
        _require_text(self.text)
        spans = _span_tuple(self.spans, "projection.spans")
        if any(span.end > len(self.text) for span in spans):
            raise _error("projection.spans", "offset_out_of_range")
        object.__setattr__(self, "spans", spans)

    @property
    def text_digest(self) -> str:
        """Return the SHA-256 digest of the current text."""
        return sha256_digest(self.text)

    @property
    def projection_digest(self) -> str:
        """Return the digest of the current proposed span set."""
        return _span_set_digest(self.spans)


@dataclass(frozen=True, slots=True)
class AdjudicationPolicy:
    """Explicit rules under which reviewer decisions may become labels.

    Nothing is accepted without a policy. Decision codes are interpreted
    through three disjoint vocabularies; any other code is an abstention.

    Attributes:
        policy_id: Lowercase controlled policy code.
        policy_version: Short version string for the policy text.
        label_codes: Closed label vocabulary for accepted spans. Codes must
            stay distinct when lowercased for annotation interchange.
        adjudicator_roles: Reviewer-role codes allowed to issue adjudication
            events.
        reviewer_roles: Reviewer-role codes whose decisions count as review
            evidence.
        accept_decision_codes: Codes meaning the proposed spans are correct.
        reject_decision_codes: Codes meaning the proposed spans are wrong.
        revise_decision_codes: Codes whose corrections are the reviewer's
            complete label set.
        minimum_reviewers: Distinct counted assignments required before
            unanimous agreement may resolve an item.
        resolve_unanimous: When false, every item needs an adjudication event
            even if reviewers agree.
    """

    policy_id: str
    policy_version: str
    label_codes: tuple[str, ...]
    adjudicator_roles: tuple[str, ...]
    reviewer_roles: tuple[str, ...] = DEFAULT_REVIEWER_ROLES
    accept_decision_codes: tuple[str, ...] = ("accept",)
    reject_decision_codes: tuple[str, ...] = ("reject",)
    revise_decision_codes: tuple[str, ...] = ("revise",)
    minimum_reviewers: int = 2
    resolve_unanimous: bool = True
    schema_version: str = ADJUDICATION_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ADJUDICATION_POLICY_SCHEMA_VERSION:
            raise _error("policy.schema_version", "unsupported_version")
        _require_code(self.policy_id, "policy.policy_id")
        _require(
            self.policy_version, _VERSION_RE, "policy.policy_version", "invalid_version"
        )

        def label_validator(value: Any, field_name: str) -> str:
            return _require(value, _LABEL_RE, field_name, "invalid_label")

        labels = _codes(
            self.label_codes,
            "policy.label_codes",
            minimum=1,
            maximum=256,
            validator=label_validator,
        )
        if len({label.lower() for label in labels}) != len(labels):
            raise _error("policy.label_codes", "case_collision")
        adjudicators = _codes(
            self.adjudicator_roles, "policy.adjudicator_roles", minimum=1, maximum=16
        )
        reviewers = _codes(
            self.reviewer_roles, "policy.reviewer_roles", minimum=1, maximum=16
        )
        accept = _codes(
            self.accept_decision_codes,
            "policy.accept_decision_codes",
            minimum=1,
            maximum=16,
        )
        reject = _codes(
            self.reject_decision_codes,
            "policy.reject_decision_codes",
            minimum=0,
            maximum=16,
        )
        revise = _codes(
            self.revise_decision_codes,
            "policy.revise_decision_codes",
            minimum=0,
            maximum=16,
        )
        if (
            set(accept) & set(reject)
            or set(accept) & set(revise)
            or set(reject) & set(revise)
        ):
            raise _error("policy.decision_codes", "overlapping_codes")
        if (
            type(self.minimum_reviewers) is not int
            or not 1 <= self.minimum_reviewers <= MAX_MINIMUM_REVIEWERS
        ):
            raise _error("policy.minimum_reviewers", "invalid_count")
        _require_bool(self.resolve_unanimous, "policy.resolve_unanimous")
        object.__setattr__(self, "label_codes", labels)
        object.__setattr__(self, "adjudicator_roles", adjudicators)
        object.__setattr__(self, "reviewer_roles", reviewers)
        object.__setattr__(self, "accept_decision_codes", accept)
        object.__setattr__(self, "reject_decision_codes", reject)
        object.__setattr__(self, "revise_decision_codes", revise)

    @property
    def policy_digest(self) -> str:
        """Return the digest that adjudication events must bind to."""
        return canonical_digest(self.to_dict())

    def vote_kind(self, decision: str) -> VoteKind:
        """Return how this policy interprets one decision code."""
        if decision in self.accept_decision_codes or (
            decision in self.revise_decision_codes
        ):
            return VoteKind.LABELS
        if decision in self.reject_decision_codes:
            return VoteKind.REJECT
        return VoteKind.ABSTAIN

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical policy document."""
        return {
            "accept_decision_codes": list(self.accept_decision_codes),
            "adjudicator_roles": list(self.adjudicator_roles),
            "label_codes": list(self.label_codes),
            "minimum_reviewers": self.minimum_reviewers,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "reject_decision_codes": list(self.reject_decision_codes),
            "resolve_unanimous": self.resolve_unanimous,
            "reviewer_roles": list(self.reviewer_roles),
            "revise_decision_codes": list(self.revise_decision_codes),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> AdjudicationPolicy:
        """Parse and revalidate one strict policy document."""
        data = dict(_strict_fields(payload, "policy", _field_names(cls)))
        for name in (
            "accept_decision_codes",
            "adjudicator_roles",
            "label_codes",
            "reject_decision_codes",
            "reviewer_roles",
            "revise_decision_codes",
        ):
            data[name] = tuple(_list_field(data[name], f"policy.{name}", 256))
        return cls(**data)


_EVENT_CONTENT_FIELDS = (
    "adjudicator_role",
    "basis_digest",
    "language",
    "outcome",
    "policy_digest",
    "projection_digest",
    "reason_code",
    "schema_version",
    "source_ref",
    "spans",
    "supersedes_event_id",
    "task_code",
    "text_digest",
)


def _event_content(values: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical event document without its identifier."""
    content = {name: values[name] for name in _EVENT_CONTENT_FIELDS}
    content["outcome"] = content["outcome"].value
    content["spans"] = [span.to_dict() for span in content["spans"]]
    return content


@dataclass(frozen=True, slots=True)
class AdjudicationEvent:
    """One immutable, content-addressed adjudication revision for an item.

    An event binds the adjudicator's outcome to the exact evidence it
    resolved: the policy digest, the basis digest of the current reviewer
    decisions, and the text and projection digests. If any of these change,
    the event no longer applies and the item returns to review. A later event
    replaces an earlier one only by naming it in ``supersedes_event_id``.

    Attributes:
        event_id: Deterministic ``nra_`` identifier derived from the content.
        source_ref: Caller-owned opaque reference of the adjudicated item.
        language: Language tag of the adjudicated item.
        task_code: Task of the adjudicated item.
        policy_digest: Digest of the policy the event applies.
        adjudicator_role: Closed role code of the adjudicator, never identity.
        basis_digest: Basis digest of the decisions the adjudicator resolved.
        text_digest: Digest of the text the adjudicator saw.
        projection_digest: Digest of the projection the adjudicator saw.
        outcome: Closed adjudication outcome.
        spans: Adjudicated span set, required only for ``accept_spans``.
        reason_code: Optional closed reason code.
        supersedes_event_id: Earlier event of the same item this replaces.
    """

    event_id: str
    source_ref: str
    language: str
    task_code: str
    policy_digest: str
    adjudicator_role: str
    basis_digest: str
    text_digest: str
    projection_digest: str
    outcome: AdjudicationOutcome
    spans: tuple[ReviewSpan, ...] = ()
    reason_code: str | None = None
    supersedes_event_id: str | None = None
    schema_version: str = ADJUDICATION_EVENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ADJUDICATION_EVENT_SCHEMA_VERSION:
            raise _error("event.schema_version", "unsupported_version")
        _require_ref(self.source_ref, "event.source_ref")
        _require_language(self.language)
        _require_code(self.task_code, "event.task_code")
        _require_digest(self.policy_digest, "event.policy_digest")
        _require_code(self.adjudicator_role, "event.adjudicator_role")
        _require_digest(self.basis_digest, "event.basis_digest")
        _require_digest(self.text_digest, "event.text_digest")
        _require_digest(self.projection_digest, "event.projection_digest")
        outcome = _require_enum(self.outcome, AdjudicationOutcome, "event.outcome")
        spans = _span_tuple(self.spans, "event.spans")
        if (outcome is AdjudicationOutcome.ACCEPT_SPANS) != bool(spans):
            raise _error("event.spans", "outcome_mismatch")
        _optional(self.reason_code, _require_code, "event.reason_code")
        if self.supersedes_event_id is not None:
            _require(
                self.supersedes_event_id,
                _EVENT_ID_RE,
                "event.supersedes_event_id",
                "invalid_event_id",
            )
        _require(self.event_id, _EVENT_ID_RE, "event.event_id", "invalid_event_id")
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "spans", spans)
        if not hmac.compare_digest(self.event_id, self._derived_id()):
            raise _error("event.event_id", "lineage_mismatch")
        if self.supersedes_event_id == self.event_id:
            raise _error("event.supersedes_event_id", "self_supersession")

    def _derived_id(self) -> str:
        values = {name: getattr(self, name) for name in _EVENT_CONTENT_FIELDS}
        return derived_opaque_id("nra", _event_content(values))

    @classmethod
    def create(
        cls,
        *,
        source_ref: str,
        language: str,
        task_code: str,
        policy_digest: str,
        adjudicator_role: str,
        basis_digest: str,
        text_digest: str,
        projection_digest: str,
        outcome: AdjudicationOutcome | str,
        spans: Sequence[ReviewSpan] = (),
        reason_code: str | None = None,
        supersedes_event_id: str | None = None,
    ) -> AdjudicationEvent:
        """Create an event whose identifier is derived from its content.

        Returns:
            The validated event.

        Raises:
            NativeReviewAdjudicationError: With a closed code for malformed
                fields or spans that do not match the outcome.
        """
        values: dict[str, Any] = {
            "adjudicator_role": adjudicator_role,
            "basis_digest": basis_digest,
            "language": language,
            "outcome": _require_enum(outcome, AdjudicationOutcome, "event.outcome"),
            "policy_digest": policy_digest,
            "projection_digest": projection_digest,
            "reason_code": reason_code,
            "schema_version": ADJUDICATION_EVENT_SCHEMA_VERSION,
            "source_ref": source_ref,
            "spans": _span_tuple(spans, "event.spans"),
            "supersedes_event_id": supersedes_event_id,
            "task_code": task_code,
            "text_digest": text_digest,
        }
        event_id = derived_opaque_id("nra", _event_content(values))
        return cls(event_id=event_id, **values)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical, identity-free event document."""
        values = {name: getattr(self, name) for name in _EVENT_CONTENT_FIELDS}
        return {**_event_content(values), "event_id": self.event_id}

    @classmethod
    def from_dict(cls, payload: Any) -> AdjudicationEvent:
        """Parse and revalidate one strict event document."""
        data = dict(_strict_fields(payload, "event", _field_names(cls)))
        data["spans"] = _parse_spans(data["spans"], "event.spans")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class DecisionVote:
    """How the policy counted one current reviewer decision.

    Attributes:
        decision_id: Opaque decision identifier.
        reviewer_role: Closed reviewer-role code.
        decision: Closed decision code.
        vote: Policy interpretation of ``decision``.
        label_set_digest: Digest of the label set a ``labels`` vote supports.
    """

    decision_id: str
    reviewer_role: str
    decision: str
    vote: VoteKind
    label_set_digest: str | None = None

    def __post_init__(self) -> None:
        _require(
            self.decision_id, _DECISION_ID_RE, "vote.decision_id", "invalid_decision_id"
        )
        _require_code(self.reviewer_role, "vote.reviewer_role")
        _require_code(self.decision, "vote.decision")
        vote = _require_enum(self.vote, VoteKind, "vote.vote")
        _optional(self.label_set_digest, _require_digest, "vote.label_set_digest")
        if (vote is VoteKind.LABELS) != (self.label_set_digest is not None):
            raise _error("vote.label_set_digest", "vote_mismatch")
        object.__setattr__(self, "vote", vote)

    @property
    def signature(self) -> str:
        """Return the value compared when testing reviewer agreement."""
        if self.vote is VoteKind.LABELS:
            return f"labels:{self.label_set_digest}"
        return self.vote.value

    def to_dict(self) -> dict[str, Any]:
        """Return the identity-free vote document."""
        return {
            "decision": self.decision,
            "decision_id": self.decision_id,
            "label_set_digest": self.label_set_digest,
            "reviewer_role": self.reviewer_role,
            "vote": self.vote.value,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> DecisionVote:
        """Parse one strict vote document."""
        return cls(**_strict_fields(payload, "vote", _field_names(cls)))


@dataclass(frozen=True, slots=True)
class ItemAdjudication:
    """The adjudicated state and lineage of one source item in a revision.

    Attributes:
        source_ref: Caller-owned opaque reference of the item.
        language: Language tag of the item.
        task_code: Review task of the item.
        state: Closed item state.
        resolution: How an accepted or rejected item was resolved.
        reason_codes: Closed codes for blocking conditions and excluded
            evidence.
        text_digest: Digest of the current text.
        projection_digest: Digest of the current proposed spans.
        basis_digest: Digest of the counted current decision identifiers.
        labels: Accepted spans; empty unless ``state`` is ``accepted``.
        label_set_digest: Digest of ``labels`` for accepted items.
        votes: Counted current decisions and their interpretation.
        superseded_decision_ids: Decisions replaced by resubmissions.
        excluded_decision_ids: Current decisions excluded as stale or not
            counted under the policy.
        event_id: Adjudication event applied to the item, if any.
        adjudicator_role: Role code of the applied event's adjudicator.
        superseded_event_ids: Events replaced by later events.
        ignored_event_ids: Events that were not applied.
        prior_state: State in the parent revision, if any.
        prior_label_set_digest: Accepted label digest in the parent revision.
        prior_acceptance_revoked: Whether labels accepted in the parent
            revision no longer stand for the same text.
    """

    source_ref: str
    language: str
    task_code: str
    state: LabelState
    resolution: Resolution | None
    reason_codes: tuple[str, ...]
    text_digest: str
    projection_digest: str
    basis_digest: str
    labels: tuple[ReviewSpan, ...] = ()
    label_set_digest: str | None = None
    votes: tuple[DecisionVote, ...] = ()
    superseded_decision_ids: tuple[str, ...] = ()
    excluded_decision_ids: tuple[str, ...] = ()
    event_id: str | None = None
    adjudicator_role: str | None = None
    superseded_event_ids: tuple[str, ...] = ()
    ignored_event_ids: tuple[str, ...] = ()
    prior_state: LabelState | None = None
    prior_label_set_digest: str | None = None
    prior_acceptance_revoked: bool = False

    def __post_init__(self) -> None:
        _require_ref(self.source_ref, "item.source_ref")
        _require_language(self.language)
        _require_code(self.task_code, "item.task_code")
        state = _require_enum(self.state, LabelState, "item.state")
        resolution = (
            None
            if self.resolution is None
            else _require_enum(self.resolution, Resolution, "item.resolution")
        )
        reasons = _codes(
            self.reason_codes,
            "item.reason_codes",
            minimum=0,
            maximum=len(REASON_CODES),
        )
        if not set(reasons) <= REASON_CODES:
            raise _error("item.reason_codes", "unknown_code")
        _require_digest(self.text_digest, "item.text_digest")
        _require_digest(self.projection_digest, "item.projection_digest")
        _require_digest(self.basis_digest, "item.basis_digest")
        labels = _span_tuple(self.labels, "item.labels")
        _optional(self.label_set_digest, _require_digest, "item.label_set_digest")
        votes = _bounded(self.votes, "item.votes", MAX_DECISIONS)
        if not all(isinstance(vote, DecisionVote) for vote in votes):
            raise _error("item.votes", "invalid_sequence")
        votes = tuple(sorted(votes, key=lambda vote: vote.decision_id))
        vote_ids = [vote.decision_id for vote in votes]
        if len(set(vote_ids)) != len(vote_ids):
            raise _error("item.votes", "duplicate_decision")
        if not hmac.compare_digest(self.basis_digest, _basis_digest(vote_ids)):
            raise _error("item.basis_digest", "lineage_mismatch")
        superseded = _id_tuple(
            self.superseded_decision_ids,
            "item.superseded_decision_ids",
            _DECISION_ID_RE,
        )
        excluded = _id_tuple(
            self.excluded_decision_ids, "item.excluded_decision_ids", _DECISION_ID_RE
        )
        _optional(
            self.event_id,
            lambda value, name: _require(value, _EVENT_ID_RE, name, "invalid_event_id"),
            "item.event_id",
        )
        _optional(self.adjudicator_role, _require_code, "item.adjudicator_role")
        if (self.event_id is None) != (self.adjudicator_role is None):
            raise _error("item.adjudicator_role", "event_mismatch")
        superseded_events = _id_tuple(
            self.superseded_event_ids, "item.superseded_event_ids", _EVENT_ID_RE
        )
        ignored_events = _id_tuple(
            self.ignored_event_ids, "item.ignored_event_ids", _EVENT_ID_RE
        )
        prior_state = (
            None
            if self.prior_state is None
            else _require_enum(self.prior_state, LabelState, "item.prior_state")
        )
        _optional(
            self.prior_label_set_digest, _require_digest, "item.prior_label_set_digest"
        )
        _require_bool(self.prior_acceptance_revoked, "item.prior_acceptance_revoked")

        if state is LabelState.ACCEPTED:
            if resolution is None or self.label_set_digest is None:
                raise _error("item.state", "inconsistent_acceptance")
            if not hmac.compare_digest(self.label_set_digest, _span_set_digest(labels)):
                raise _error("item.label_set_digest", "lineage_mismatch")
        elif labels or self.label_set_digest is not None:
            raise _error("item.labels", "labels_without_acceptance")
        elif (state is LabelState.REJECTED) != (resolution is not None):
            raise _error("item.resolution", "inconsistent_resolution")
        if resolution is Resolution.ADJUDICATION_EVENT and self.event_id is None:
            raise _error("item.event_id", "missing_event")
        if (prior_state is not LabelState.ACCEPTED) and (
            self.prior_label_set_digest is not None or self.prior_acceptance_revoked
        ):
            raise _error("item.prior_state", "inconsistent_prior")

        object.__setattr__(self, "state", state)
        object.__setattr__(self, "resolution", resolution)
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "votes", votes)
        object.__setattr__(self, "superseded_decision_ids", superseded)
        object.__setattr__(self, "excluded_decision_ids", excluded)
        object.__setattr__(self, "superseded_event_ids", superseded_events)
        object.__setattr__(self, "ignored_event_ids", ignored_events)
        object.__setattr__(self, "prior_state", prior_state)

    @property
    def subject_digest(self) -> str:
        """Return a digest that stands in for the source reference in audits."""
        return canonical_digest(
            {
                "domain": _SUBJECT_DOMAIN,
                "language": self.language,
                "source_ref": self.source_ref,
                "task_code": self.task_code,
            }
        )

    def _lineage(self) -> dict[str, Any]:
        return {
            "adjudicator_role": self.adjudicator_role,
            "basis_digest": self.basis_digest,
            "decisions": [vote.to_dict() for vote in self.votes],
            "event_id": self.event_id,
            "excluded_decision_ids": list(self.excluded_decision_ids),
            "ignored_event_ids": list(self.ignored_event_ids),
            "prior_label_set_digest": self.prior_label_set_digest,
            "superseded_decision_ids": list(self.superseded_decision_ids),
            "superseded_event_ids": list(self.superseded_event_ids),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the evaluator-side item document."""
        return {
            "adjudicator_role": self.adjudicator_role,
            "basis_digest": self.basis_digest,
            "event_id": self.event_id,
            "excluded_decision_ids": list(self.excluded_decision_ids),
            "ignored_event_ids": list(self.ignored_event_ids),
            "label_set_digest": self.label_set_digest,
            "labels": [span.to_dict() for span in self.labels],
            "language": self.language,
            "prior_acceptance_revoked": self.prior_acceptance_revoked,
            "prior_label_set_digest": self.prior_label_set_digest,
            "prior_state": None if self.prior_state is None else self.prior_state.value,
            "projection_digest": self.projection_digest,
            "reason_codes": list(self.reason_codes),
            "resolution": None if self.resolution is None else self.resolution.value,
            "source_ref": self.source_ref,
            "state": self.state.value,
            "superseded_decision_ids": list(self.superseded_decision_ids),
            "superseded_event_ids": list(self.superseded_event_ids),
            "task_code": self.task_code,
            "text_digest": self.text_digest,
            "votes": [vote.to_dict() for vote in self.votes],
        }

    @classmethod
    def from_dict(cls, payload: Any) -> ItemAdjudication:
        """Parse and revalidate one strict item document."""
        data = dict(_strict_fields(payload, "item", _field_names(cls)))
        data["labels"] = _parse_spans(data["labels"], "item.labels")
        data["votes"] = tuple(
            DecisionVote.from_dict(vote)
            for vote in _list_field(data["votes"], "item.votes", MAX_DECISIONS)
        )
        for name in (
            "excluded_decision_ids",
            "ignored_event_ids",
            "reason_codes",
            "superseded_decision_ids",
            "superseded_event_ids",
        ):
            data[name] = tuple(_list_field(data[name], f"item.{name}", MAX_DECISIONS))
        return cls(**data)

    def to_label_dict(self) -> dict[str, Any]:
        """Return the exported label document for an accepted item.

        Raises:
            NativeReviewAdjudicationError: If the item is not accepted.
        """
        if self.state is not LabelState.ACCEPTED or self.resolution is None:
            raise _error("item.state", "not_accepted")
        return {
            "label_set_digest": self.label_set_digest,
            "language": self.language,
            "lineage": self._lineage(),
            "projection_digest": self.projection_digest,
            "resolution": self.resolution.value,
            "source_ref": self.source_ref,
            "spans": [span.to_dict() for span in self.labels],
            "task_code": self.task_code,
            "text_digest": self.text_digest,
        }

    def to_audit_dict(self) -> dict[str, Any]:
        """Return a value-free audit entry keyed by the subject digest."""
        votes = Counter(vote.vote.value for vote in self.votes)
        return {
            "adjudicator_role": self.adjudicator_role,
            "basis_digest": self.basis_digest,
            "decision_ids": [vote.decision_id for vote in self.votes],
            "event_id": self.event_id,
            "excluded_decision_count": len(self.excluded_decision_ids),
            "ignored_event_count": len(self.ignored_event_ids),
            "label_count": len(self.labels),
            "label_set_digest": self.label_set_digest,
            "language": self.language,
            "prior_acceptance_revoked": self.prior_acceptance_revoked,
            "prior_state": None if self.prior_state is None else self.prior_state.value,
            "projection_digest": self.projection_digest,
            "reason_codes": list(self.reason_codes),
            "resolution": None if self.resolution is None else self.resolution.value,
            "reviewer_roles": sorted({vote.reviewer_role for vote in self.votes}),
            "state": self.state.value,
            "subject_digest": self.subject_digest,
            "superseded_decision_count": len(self.superseded_decision_ids),
            "superseded_event_count": len(self.superseded_event_ids),
            "task_code": self.task_code,
            "text_digest": self.text_digest,
            "vote_counts": {kind.value: votes.get(kind.value, 0) for kind in VoteKind},
        }


def _revision_id(
    parent_revision_id: str | None, policy_digest: str, input_digest: str
) -> str:
    return derived_opaque_id("nrr", parent_revision_id, policy_digest, input_digest)


@dataclass(frozen=True, slots=True)
class AdjudicationRevision:
    """An immutable, content-addressed snapshot of merged review labels.

    Each merge produces a new revision that names its parent, so the chain of
    revisions records how labels moved between ``pending``, ``unresolved``,
    ``accepted``, ``rejected`` and ``invalidated``.

    Attributes:
        revision_id: ``nrr_`` identifier derived from the parent revision,
            policy digest and input digest.
        parent_revision_id: Revision this one supersedes, if any.
        policy_id: Code of the policy applied.
        policy_version: Version of the policy applied.
        policy_digest: Digest of the policy applied.
        input_digest: Digest of the projections, decisions and events merged.
        items: One adjudicated entry per projection, ordered by source.
    """

    revision_id: str
    parent_revision_id: str | None
    policy_id: str
    policy_version: str
    policy_digest: str
    input_digest: str
    items: tuple[ItemAdjudication, ...]
    schema_version: str = ADJUDICATION_REVISION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ADJUDICATION_REVISION_SCHEMA_VERSION:
            raise _error("revision.schema_version", "unsupported_version")
        _optional(
            self.parent_revision_id,
            lambda value, name: _require(
                value, _REVISION_ID_RE, name, "invalid_revision_id"
            ),
            "revision.parent_revision_id",
        )
        _require_code(self.policy_id, "revision.policy_id")
        _require(
            self.policy_version,
            _VERSION_RE,
            "revision.policy_version",
            "invalid_version",
        )
        _require_digest(self.policy_digest, "revision.policy_digest")
        _require_digest(self.input_digest, "revision.input_digest")
        items = _bounded(self.items, "revision.items", MAX_ITEMS)
        if not all(isinstance(item, ItemAdjudication) for item in items):
            raise _error("revision.items", "invalid_sequence")
        items = tuple(sorted(items, key=lambda item: item.source_ref))
        if len({item.source_ref for item in items}) != len(items):
            raise _error("revision.items", "duplicate_source_ref")
        _require(
            self.revision_id,
            _REVISION_ID_RE,
            "revision.revision_id",
            "invalid_revision_id",
        )
        expected = _revision_id(
            self.parent_revision_id, self.policy_digest, self.input_digest
        )
        if not hmac.compare_digest(self.revision_id, expected):
            raise _error("revision.revision_id", "lineage_mismatch")
        object.__setattr__(self, "items", items)

    @property
    def revision_digest(self) -> str:
        """Return the digest of the complete revision document."""
        return canonical_digest(self.to_dict())

    def item(self, source_ref: str) -> ItemAdjudication:
        """Return the adjudicated entry for one source reference.

        Raises:
            KeyError: If the revision has no entry for ``source_ref``.
        """
        for entry in self.items:
            if entry.source_ref == source_ref:
                return entry
        raise KeyError("unknown source reference")

    def state_counts(self) -> dict[str, int]:
        """Return the number of items in each closed state."""
        counts = Counter(item.state.value for item in self.items)
        return {state.value: counts.get(state.value, 0) for state in LabelState}

    def to_dict(self) -> dict[str, Any]:
        """Return the evaluator-side revision document."""
        return {
            "input_digest": self.input_digest,
            "items": [item.to_dict() for item in self.items],
            "parent_revision_id": self.parent_revision_id,
            "policy_digest": self.policy_digest,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "revision_id": self.revision_id,
            "schema_version": self.schema_version,
        }

    def to_json(self) -> str:
        """Return canonical JSON for protected evaluator storage."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Any) -> AdjudicationRevision:
        """Parse and revalidate one strict revision document."""
        data = dict(_strict_fields(payload, "revision", _field_names(cls)))
        data["items"] = tuple(
            ItemAdjudication.from_dict(item)
            for item in _list_field(data["items"], "revision.items", MAX_ITEMS)
        )
        return cls(**data)

    def export_labels(self) -> dict[str, Any]:
        """Return accepted labels with decision lineage.

        Only ``accepted`` items are exported. Each label set keeps its source
        reference, so training jobs can join it to protected text, plus the
        opaque decision and event identifiers, reviewer-role codes and
        digests that produced it. Reviewer identity, assignment references
        and source text are never included.
        """
        return {
            "coordinate_convention": _COORDINATE_CONVENTION,
            "labels": [
                item.to_label_dict()
                for item in self.items
                if item.state is LabelState.ACCEPTED
            ],
            "parent_revision_id": self.parent_revision_id,
            "policy_digest": self.policy_digest,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "revision_id": self.revision_id,
            "schema_version": ADJUDICATED_LABELS_SCHEMA_VERSION,
        }

    def to_audit_dict(self) -> dict[str, Any]:
        """Return a value-free audit report for this revision.

        The report contains closed codes, counts, opaque identifiers and
        digests. Source references are replaced by subject digests, and no
        text, span surface or reviewer identity is included.
        """
        reasons = Counter(code for item in self.items for code in item.reason_codes)
        return {
            "input_digest": self.input_digest,
            "items": sorted(
                (item.to_audit_dict() for item in self.items),
                key=lambda entry: entry["subject_digest"],
            ),
            "parent_revision_id": self.parent_revision_id,
            "policy_digest": self.policy_digest,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "prior_acceptance_revoked_count": sum(
                item.prior_acceptance_revoked for item in self.items
            ),
            "reason_counts": dict(sorted(reasons.items())),
            "revision_digest": self.revision_digest,
            "revision_id": self.revision_id,
            "schema_version": ADJUDICATION_AUDIT_SCHEMA_VERSION,
            "state_counts": self.state_counts(),
        }

    def to_reviewer_decisions(
        self, *, reason: DisagreementReason = DisagreementReason.LABEL_DEFINITION
    ) -> tuple[ReviewerDecision, ...]:
        """Map counted votes to records for the reviewer-disagreement metric.

        Items with fewer than two counted votes and invalidated items are
        skipped. Case identifiers are subject digests and reviewer identifiers
        are opaque decision identifiers, so the metric input carries no
        assignment reference or reviewer identity.

        Args:
            reason: Typed reason applied to every disagreement case.

        Returns:
            Records accepted by
            :func:`openmed.eval.reviewer_disagreement.reviewer_disagreement_report`.
        """
        if not isinstance(reason, DisagreementReason):
            raise _error("reason", "invalid_code")
        records: list[ReviewerDecision] = []
        for item in self.items:
            if item.state is LabelState.INVALIDATED or len(item.votes) < 2:
                continue
            agreed = len({vote.signature for vote in item.votes}) == 1
            adjudicated = (
                not agreed and item.resolution is Resolution.ADJUDICATION_EVENT
            )
            for vote in item.votes:
                records.append(
                    ReviewerDecision(
                        case_id=item.subject_digest,
                        reviewer_id=vote.decision_id,
                        decision=vote.signature,
                        reason=None if agreed else reason,
                        adjudicated=adjudicated,
                    )
                )
        return tuple(records)


def build_adjudication_event(
    item: ItemAdjudication,
    *,
    policy: AdjudicationPolicy,
    adjudicator_role: str,
    outcome: AdjudicationOutcome | str,
    spans: Sequence[ReviewSpan] = (),
    reason_code: str | None = None,
    supersedes_event_id: str | None = None,
) -> AdjudicationEvent:
    """Create an adjudication event bound to an item's current evidence.

    The event takes the item's basis, text and projection digests from the
    revision entry the adjudicator reviewed, so it stops applying as soon as
    any of them change. Whether the event is valid under ``policy`` is
    decided when it is merged.

    Args:
        item: Entry from the revision the adjudicator reviewed.
        policy: Policy the event applies.
        adjudicator_role: Closed role code of the adjudicator.
        outcome: Closed adjudication outcome.
        spans: Adjudicated spans, required only for ``accept_spans``.
        reason_code: Optional closed reason code.
        supersedes_event_id: Earlier event of the same item this replaces.

    Returns:
        The content-addressed adjudication event.
    """
    if not isinstance(item, ItemAdjudication):
        raise _error("item", "invalid_item")
    if not isinstance(policy, AdjudicationPolicy):
        raise _error("policy", "invalid_policy")
    return AdjudicationEvent.create(
        source_ref=item.source_ref,
        language=item.language,
        task_code=item.task_code,
        policy_digest=policy.policy_digest,
        adjudicator_role=adjudicator_role,
        basis_digest=item.basis_digest,
        text_digest=item.text_digest,
        projection_digest=item.projection_digest,
        outcome=outcome,
        spans=spans,
        reason_code=reason_code,
        supersedes_event_id=supersedes_event_id,
    )


def _coerce_event(value: Any) -> AdjudicationEvent:
    if isinstance(value, AdjudicationEvent):
        return value
    if isinstance(value, Mapping):
        return AdjudicationEvent.from_dict(value)
    raise _error("events", "invalid_event")


@dataclass(slots=True)
class _ItemContext:
    projection: LabelProjection
    text_digest: str
    projection_digest: str
    has_break: Callable[[int], bool]
    policy: AdjudicationPolicy
    policy_digest: str
    label_codes: frozenset[str]


def _current_decisions(
    context: _ItemContext,
    overlays: Sequence[DecisionOverlay],
    decision_index: Mapping[str, DecisionOverlay],
    conflicting_ids: frozenset[str],
    reasons: set[str],
) -> tuple[list[DecisionOverlay], set[str], bool]:
    """Resolve resubmissions to one current decision per assignment."""
    invalid = False
    unique: dict[str, DecisionOverlay] = {}
    for overlay in overlays:
        unique.setdefault(overlay.decision_id, overlay)
        if overlay.decision_id in conflicting_ids:
            reasons.add(REASON_DUPLICATE_DECISION)
            invalid = True
    projection = context.projection
    by_assignment: dict[tuple[str, str], list[DecisionOverlay]] = defaultdict(list)
    for overlay in unique.values():
        if (
            overlay.language != projection.language
            or overlay.task_code != projection.task_code
        ):
            reasons.add(REASON_DECISION_SCOPE_MISMATCH)
            invalid = True
        target_id = overlay.supersedes_decision_id
        target = None if target_id is None else decision_index.get(target_id)
        if target is not None and target.assignment_key != overlay.assignment_key:
            reasons.add(REASON_SUPERSESSION_SCOPE_MISMATCH)
            invalid = True
        by_assignment[overlay.assignment_key].append(overlay)

    superseded: set[str] = set()
    current: list[DecisionOverlay] = []
    for group in by_assignment.values():
        pointers = {
            overlay.supersedes_decision_id
            for overlay in group
            if overlay.supersedes_decision_id is not None
        }
        superseded |= pointers
        heads = [overlay for overlay in group if overlay.decision_id not in pointers]
        if len(heads) != 1:
            # Two current answers for one assignment, or a supersession cycle,
            # leave no unambiguous reviewer decision.
            reasons.add(REASON_DUPLICATE_DECISION)
            invalid = True
        current.extend(heads)
    return current, superseded, invalid


def _select_event(
    context: _ItemContext,
    events: Sequence[AdjudicationEvent],
    event_index: Mapping[str, AdjudicationEvent],
    basis_digest: str,
    has_votes: bool,
    reasons: set[str],
) -> tuple[AdjudicationEvent | None, bool, set[str], set[str]]:
    """Return the single valid head event, a conflict flag and history."""
    unique = {event.event_id: event for event in events}
    superseded = {
        event.supersedes_event_id
        for event in unique.values()
        if event.supersedes_event_id is not None
    }
    ignored: set[str] = set()
    valid: list[AdjudicationEvent] = []
    projection = context.projection
    for event in unique.values():
        target_id = event.supersedes_event_id
        target = None if target_id is None else event_index.get(target_id)
        if target is not None and target.source_ref != event.source_ref:
            reasons.add(REASON_SUPERSESSION_SCOPE_MISMATCH)
            ignored.add(event.event_id)
            continue
        if event.event_id in superseded:
            continue
        problem: str | None = None
        if (
            event.language != projection.language
            or event.task_code != projection.task_code
        ):
            problem = REASON_ADJUDICATION_SCOPE_MISMATCH
        elif not hmac.compare_digest(event.policy_digest, context.policy_digest):
            problem = REASON_ADJUDICATION_POLICY_MISMATCH
        elif event.adjudicator_role not in context.policy.adjudicator_roles:
            problem = REASON_ADJUDICATOR_ROLE_NOT_ALLOWED
        elif not (
            hmac.compare_digest(event.text_digest, context.text_digest)
            and hmac.compare_digest(event.projection_digest, context.projection_digest)
            and hmac.compare_digest(event.basis_digest, basis_digest)
        ):
            problem = REASON_ADJUDICATION_STALE
        elif not has_votes:
            problem = REASON_INSUFFICIENT_REVIEWS
        else:
            problem = _span_problem(
                event.spans,
                projection.text,
                context.has_break,
                context.label_codes,
            )
        if problem is None:
            valid.append(event)
        else:
            reasons.add(problem)
            ignored.add(event.event_id)
    conflict = len(valid) > 1 or (bool(unique) and not valid and not ignored)
    if conflict:
        reasons.add(REASON_CONFLICTING_ADJUDICATION)
        ignored |= {event.event_id for event in valid}
        return None, True, superseded, ignored
    return (valid[0] if valid else None), False, superseded, ignored


def _adjudicate_item(
    context: _ItemContext,
    overlays: Sequence[DecisionOverlay],
    events: Sequence[AdjudicationEvent],
    decision_index: Mapping[str, DecisionOverlay],
    event_index: Mapping[str, AdjudicationEvent],
    conflicting_ids: frozenset[str],
    prior: ItemAdjudication | None,
) -> ItemAdjudication:
    projection = context.projection
    policy = context.policy
    reasons: set[str] = set()
    current, superseded_decisions, invalid = _current_decisions(
        context, overlays, decision_index, conflicting_ids, reasons
    )
    projection_problem = _span_problem(
        projection.spans, projection.text, context.has_break, context.label_codes
    )
    if projection_problem is not None:
        reasons.add(projection_problem)
        invalid = True

    excluded: set[str] = set()
    stale = False
    votes: list[DecisionVote] = []
    candidates: dict[str, tuple[ReviewSpan, ...]] = {}
    for overlay in current:
        if not hmac.compare_digest(overlay.text_digest, context.text_digest):
            reasons.add(REASON_TEXT_CHANGED)
            excluded.add(overlay.decision_id)
            stale = True
            continue
        if not hmac.compare_digest(
            overlay.projection_digest, context.projection_digest
        ):
            reasons.add(REASON_STALE_PROJECTION)
            excluded.add(overlay.decision_id)
            stale = True
            continue
        if overlay.reviewer_role not in policy.reviewer_roles:
            reasons.add(REASON_REVIEWER_ROLE_NOT_COUNTED)
            excluded.add(overlay.decision_id)
            continue
        is_revision = overlay.decision in policy.revise_decision_codes
        if is_revision != bool(overlay.corrections):
            reasons.add(REASON_INVALID_CORRECTION)
            invalid = True
        problem = _span_problem(
            overlay.corrections,
            projection.text,
            context.has_break,
            context.label_codes,
        )
        if problem is not None:
            reasons.add(problem)
            invalid = True
        kind = policy.vote_kind(overlay.decision)
        digest = None
        if kind is VoteKind.LABELS:
            spans = overlay.corrections if is_revision else projection.spans
            digest = _span_set_digest(spans)
            candidates[digest] = spans
        votes.append(
            DecisionVote(
                decision_id=overlay.decision_id,
                reviewer_role=overlay.reviewer_role,
                decision=overlay.decision,
                vote=kind,
                label_set_digest=digest,
            )
        )
    if stale and not votes:
        invalid = True

    basis_digest = _basis_digest(vote.decision_id for vote in votes)
    applied: AdjudicationEvent | None = None
    conflict = False
    superseded_events: set[str] = set()
    ignored_events: set[str] = set()
    if invalid:
        ignored_events = {event.event_id for event in events}
    else:
        applied, conflict, superseded_events, ignored_events = _select_event(
            context, events, event_index, basis_digest, bool(votes), reasons
        )

    state: LabelState
    resolution: Resolution | None = None
    labels: tuple[ReviewSpan, ...] = ()
    if invalid:
        state = LabelState.INVALIDATED
    elif conflict:
        state = LabelState.UNRESOLVED
    elif applied is not None:
        if applied.outcome is AdjudicationOutcome.DEFER:
            state = LabelState.UNRESOLVED
            reasons.add(REASON_ADJUDICATION_DEFERRED)
        elif applied.outcome is AdjudicationOutcome.REJECT_PROJECTION:
            state, resolution = LabelState.REJECTED, Resolution.ADJUDICATION_EVENT
        else:
            state, resolution = LabelState.ACCEPTED, Resolution.ADJUDICATION_EVENT
            labels = (
                projection.spans
                if applied.outcome is AdjudicationOutcome.ACCEPT_PROJECTION
                else applied.spans
            )
    else:
        signatures = {
            vote.signature for vote in votes if vote.vote is not VoteKind.ABSTAIN
        }
        abstained = any(vote.vote is VoteKind.ABSTAIN for vote in votes)
        if len(signatures) > 1:
            state = LabelState.UNRESOLVED
            reasons.add(REASON_REVIEWER_DISAGREEMENT)
        elif abstained:
            state = LabelState.UNRESOLVED
            reasons.add(REASON_REVIEWER_ABSTAINED)
        elif len(votes) < policy.minimum_reviewers:
            state = LabelState.PENDING
            reasons.add(REASON_INSUFFICIENT_REVIEWS)
        elif not policy.resolve_unanimous:
            state = LabelState.PENDING
            reasons.add(REASON_ADJUDICATION_REQUIRED)
        elif votes[0].vote is VoteKind.REJECT:
            state, resolution = LabelState.REJECTED, Resolution.UNANIMOUS_AGREEMENT
        else:
            state, resolution = LabelState.ACCEPTED, Resolution.UNANIMOUS_AGREEMENT
            labels = candidates[votes[0].label_set_digest or ""]

    label_set_digest = (
        _span_set_digest(labels) if state is LabelState.ACCEPTED else None
    )
    prior_state = None if prior is None else prior.state
    prior_label_set_digest = (
        prior.label_set_digest
        if prior is not None and prior.state is LabelState.ACCEPTED
        else None
    )
    revoked = prior_state is LabelState.ACCEPTED and not (
        state is LabelState.ACCEPTED
        and label_set_digest == prior_label_set_digest
        and prior is not None
        and prior.text_digest == context.text_digest
    )
    return ItemAdjudication(
        source_ref=projection.source_ref,
        language=projection.language,
        task_code=projection.task_code,
        state=state,
        resolution=resolution,
        reason_codes=tuple(sorted(reasons)),
        text_digest=context.text_digest,
        projection_digest=context.projection_digest,
        basis_digest=basis_digest,
        labels=labels,
        label_set_digest=label_set_digest,
        votes=tuple(votes),
        superseded_decision_ids=tuple(sorted(superseded_decisions)),
        excluded_decision_ids=tuple(sorted(excluded)),
        event_id=None if applied is None else applied.event_id,
        adjudicator_role=None if applied is None else applied.adjudicator_role,
        superseded_event_ids=tuple(sorted(superseded_events)),
        ignored_event_ids=tuple(sorted(ignored_events)),
        prior_state=prior_state,
        prior_label_set_digest=prior_label_set_digest,
        prior_acceptance_revoked=revoked,
    )


def merge_native_review_labels(
    projections: Sequence[LabelProjection],
    decisions: Iterable[Any] = (),
    *,
    policy: AdjudicationPolicy,
    events: Iterable[AdjudicationEvent | Mapping[str, Any]] = (),
    previous: AdjudicationRevision | None = None,
) -> AdjudicationRevision:
    """Merge reviewer decisions into labels under an explicit policy.

    For each projection the merge resolves resubmissions to one current
    decision per assignment, excludes decisions about other text or another
    projection, and then resolves the item in this order:

    1. Ambiguous duplicates, scope mismatches, invalid corrections, and
       offsets that are out of range or split a grapheme cluster invalidate
       the item. So do decisions that are all stale.
    2. A single valid adjudication event decides the item. Conflicting
       event heads keep it unresolved.
    3. Without an event, disagreeing or abstaining reviewers leave the item
       unresolved; too few reviewers leave it pending; unanimous reviewers
       resolve it only when the policy allows.

    When ``previous`` is given, the new revision names it as parent and flags
    every item whose previously accepted labels no longer stand for the same
    text.

    Args:
        projections: Current text and proposed labels, one per source.
        decisions: Imported decisions as :class:`DecisionOverlay` records,
            mappings, or objects exposing ``to_dict()``.
        policy: The explicit adjudication policy.
        events: Adjudication events or their documents.
        previous: Parent revision. Every one of its items must still have a
            projection.

    Returns:
        A new immutable revision.

    Raises:
        NativeReviewAdjudicationError: With a closed code when inputs are
            malformed, duplicated, unbounded, or refer to unknown sources.
    """
    if not isinstance(policy, AdjudicationPolicy):
        raise _error("policy", "invalid_policy")
    if previous is not None and not isinstance(previous, AdjudicationRevision):
        raise _error("previous", "invalid_revision")
    projection_list = _bounded(projections, "projections", MAX_ITEMS)
    if not projection_list or not all(
        isinstance(item, LabelProjection) for item in projection_list
    ):
        raise _error("projections", "invalid_sequence")
    by_source: dict[str, LabelProjection] = {}
    for projection in projection_list:
        if projection.source_ref in by_source:
            raise _error("projections", "duplicate_source_ref")
        by_source[projection.source_ref] = projection

    overlays = tuple(
        DecisionOverlay.from_decision(decision)
        for decision in _bounded(decisions, "decisions", MAX_DECISIONS)
    )
    event_list = tuple(
        _coerce_event(event) for event in _bounded(events, "events", MAX_EVENTS)
    )
    decision_index: dict[str, DecisionOverlay] = {}
    conflicting: set[str] = set()
    decisions_by_source: dict[str, list[DecisionOverlay]] = defaultdict(list)
    for overlay in overlays:
        if overlay.source_ref not in by_source:
            raise _error("decisions", "unknown_source")
        # The same identifier with different content is a conflicting copy;
        # every source holding either copy is invalidated.
        if decision_index.setdefault(overlay.decision_id, overlay) != overlay:
            conflicting.add(overlay.decision_id)
        decisions_by_source[overlay.source_ref].append(overlay)
    event_index: dict[str, AdjudicationEvent] = {}
    events_by_source: dict[str, list[AdjudicationEvent]] = defaultdict(list)
    for event in event_list:
        if event.source_ref not in by_source:
            raise _error("events", "unknown_source")
        event_index.setdefault(event.event_id, event)
        events_by_source[event.source_ref].append(event)

    prior_items: dict[str, ItemAdjudication] = {}
    if previous is not None:
        prior_items = {item.source_ref: item for item in previous.items}
        if not set(prior_items) <= set(by_source):
            raise _error("previous", "missing_projection")

    policy_digest = policy.policy_digest
    label_codes = frozenset(policy.label_codes)
    frozen_conflicts = frozenset(conflicting)
    items: list[ItemAdjudication] = []
    projection_summaries: list[dict[str, str]] = []
    for source_ref in sorted(by_source):
        projection = by_source[source_ref]
        context = _ItemContext(
            projection=projection,
            text_digest=projection.text_digest,
            projection_digest=projection.projection_digest,
            has_break=grapheme_break_checker(projection.text),
            policy=policy,
            policy_digest=policy_digest,
            label_codes=label_codes,
        )
        projection_summaries.append(
            {
                "language": projection.language,
                "projection_digest": context.projection_digest,
                "source_ref": source_ref,
                "task_code": projection.task_code,
                "text_digest": context.text_digest,
            }
        )
        items.append(
            _adjudicate_item(
                context,
                decisions_by_source.get(source_ref, ()),
                events_by_source.get(source_ref, ()),
                decision_index,
                event_index,
                frozen_conflicts,
                prior_items.get(source_ref),
            )
        )

    input_digest = canonical_digest(
        {
            "decisions": sorted({overlay.overlay_digest for overlay in overlays}),
            "domain": _INPUT_DOMAIN,
            "events": sorted(event_index),
            "projections": projection_summaries,
        }
    )
    parent_revision_id = None if previous is None else previous.revision_id
    return AdjudicationRevision(
        revision_id=_revision_id(parent_revision_id, policy_digest, input_digest),
        parent_revision_id=parent_revision_id,
        policy_id=policy.policy_id,
        policy_version=policy.policy_version,
        policy_digest=policy_digest,
        input_digest=input_digest,
        items=tuple(items),
    )


def native_review_document_id(source_ref: str) -> str:
    """Return the opaque annotation-interchange document ID for a source.

    Args:
        source_ref: Caller-owned opaque source reference.

    Returns:
        A deterministic identifier that does not reveal ``source_ref``.
    """
    return derived_opaque_id("native_review_doc", _require_ref(source_ref, "source"))


def export_adjudicated_annotations(
    revision: AdjudicationRevision,
    projections: Sequence[LabelProjection],
    *,
    hash_secret: str | bytes,
    namespace: str = "default",
) -> AnnotationEnvelope:
    """Export accepted labels as a source-text-free annotation envelope.

    Every accepted span becomes an ``entity`` record in the annotation
    interchange format. Offsets are revalidated against the projection text,
    which must still match the revision's text digest. Surface hashes are
    keyed with ``hash_secret``; labels are lowercased into the interchange's
    controlled vocabulary; metadata carries the resolution code and the
    revision identifier as lineage, never annotator identity.

    Args:
        revision: Revision whose accepted labels are exported.
        projections: Current projections for the accepted items.
        hash_secret: Caller-held key for surface hashes.
        namespace: Controlled annotation namespace.

    Returns:
        A content-addressed annotation envelope.

    Raises:
        NativeReviewAdjudicationError: With a closed code when a projection is
            missing, its text changed since the revision, or an offset no
            longer lands on a grapheme boundary.
    """
    if not isinstance(revision, AdjudicationRevision):
        raise _error("revision", "invalid_revision")
    if not isinstance(hash_secret, (str, bytes)) or not hash_secret:
        raise _error("hash_secret", "invalid_key")
    by_source: dict[str, LabelProjection] = {}
    for projection in _bounded(projections, "projections", MAX_ITEMS):
        if not isinstance(projection, LabelProjection):
            raise _error("projections", "invalid_sequence")
        if projection.source_ref in by_source:
            raise _error("projections", "duplicate_source_ref")
        by_source[projection.source_ref] = projection

    records: list[AnnotationRecord] = []
    for item in revision.items:
        if item.state is not LabelState.ACCEPTED or item.resolution is None:
            continue
        projection = by_source.get(item.source_ref)
        if projection is None:
            raise _error("projections", "missing_projection")
        text = projection.text
        if not hmac.compare_digest(sha256_digest(text), item.text_digest):
            raise _error("projections", "text_changed")
        has_break = grapheme_break_checker(text)
        document_id = native_review_document_id(item.source_ref)
        for span in item.labels:
            if span.end > len(text):
                raise _error("labels", REASON_OFFSET_OUT_OF_RANGE)
            if not (has_break(span.start) and has_break(span.end)):
                raise _error("labels", REASON_SPLIT_GRAPHEME)
            records.append(
                AnnotationRecord(
                    annotation_id=derived_opaque_id(
                        "native_review_label",
                        revision.revision_id,
                        item.source_ref,
                        span.to_dict(),
                    ),
                    document_id=document_id,
                    namespace=namespace,
                    annotation_type=AnnotationType.ENTITY,
                    coordinate_convention=CoordinateConvention.UNICODE_CODEPOINT,
                    start=span.start,
                    end=span.end,
                    state=AnnotationState.SUCCESS,
                    result={
                        "label": span.label.lower(),
                        "surface_hash": sha256_digest(
                            hmac_text_hash(text[span.start : span.end], hash_secret)
                        ),
                    },
                    metadata={
                        "batch_id": revision.revision_id,
                        "review_state": item.resolution.value,
                        "source_format": ANNOTATION_SOURCE_FORMAT,
                    },
                )
            )
    # Build from the envelope's own record order so the envelope identifier
    # survives a canonical TSV round trip.
    records.sort(
        key=lambda record: (
            record.document_id,
            record.start,
            record.end,
            record.annotation_id,
        )
    )
    return build_annotation_envelope(records)


__all__ = [
    "ADJUDICATED_LABELS_SCHEMA_VERSION",
    "ADJUDICATION_AUDIT_SCHEMA_VERSION",
    "ADJUDICATION_EVENT_SCHEMA_VERSION",
    "ADJUDICATION_POLICY_SCHEMA_VERSION",
    "ADJUDICATION_REVISION_SCHEMA_VERSION",
    "ANNOTATION_SOURCE_FORMAT",
    "DEFAULT_REVIEWER_ROLES",
    "MAX_DECISIONS",
    "MAX_EVENTS",
    "MAX_ITEMS",
    "MAX_SPANS_PER_ITEM",
    "MAX_TEXT_CHARACTERS",
    "NATIVE_REVIEW_DECISION_SCHEMA_VERSION",
    "REASON_CODES",
    "AdjudicationEvent",
    "AdjudicationOutcome",
    "AdjudicationPolicy",
    "AdjudicationRevision",
    "DecisionOverlay",
    "DecisionVote",
    "ItemAdjudication",
    "LabelProjection",
    "LabelState",
    "NativeReviewAdjudicationError",
    "Resolution",
    "ReviewSpan",
    "VoteKind",
    "build_adjudication_event",
    "export_adjudicated_annotations",
    "merge_native_review_labels",
    "native_review_document_id",
]
