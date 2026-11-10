"""Keep handwriting alternatives unverified until exact-content human review."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum

from openmed.multimodal.evidence import EvidenceNode, NodeKind
from openmed.multimodal.vision.providers.base import Confidence

HANDWRITING_VERSION = "openmed.handwriting.review.v1"
HANDWRITING_NOTICE = (
    "Handwriting review confirms a transcription only, not clinical validity. "
    "Do not automatically trigger clinical decisions from this output."
)
_OPAQUE = re.compile(r"[0-9a-f]{32}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_MAX_TEXT = 4096
_MAX_ALTERNATIVES = 16
_MAX_BATCH = 128
_MAX_BATCH_TEXT = 262_144


class HandwritingReviewError(ValueError):
    """Controlled failure without recognized text, source paths or reviewer names."""


class HandwritingState(str, Enum):
    """Transcription state, distinct from printed OCR and clinical verification."""

    UNVERIFIED = "unverified"
    REVIEWED_TRANSCRIPTION = "reviewed_transcription"
    REJECTED = "rejected"
    ABSTAINED = "abstained"


class ReviewAction(str, Enum):
    """Explicit reviewer choice; never inferred from an alternative's score."""

    SELECT = "select_alternative"
    CORRECT = "correct_transcription"
    REJECT = "reject"
    ABSTAIN = "abstain"


def _require(condition: bool) -> None:
    if not condition:
        raise HandwritingReviewError("invalid handwriting review declaration")


def _opaque(value: str) -> None:
    _require(type(value) is str and _OPAQUE.fullmatch(value) is not None)


def _digest(value: str) -> None:
    _require(type(value) is str and _DIGEST.fullmatch(value) is not None)


def _text(value: str) -> None:
    _require(type(value) is str and 0 < len(value) <= _MAX_TEXT)
    valid = True
    try:
        value.encode("utf-8")
    except UnicodeError:
        valid = False
    _require(valid and bool(value.strip()))


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class TranscriptionAlternative:
    """One protected recognition hypothesis with an uncalibrated score or unknown.

    Args:
        alternative_id: Opaque 128-bit identifier, independent of recognized text.
        text: Unmodified recognition hypothesis, at most 4096 Unicode code points.
        confidence: Provider score in [0, 1], or explicit unknown. Not probability.
    """

    alternative_id: str
    text: str = field(repr=False)
    confidence: Confidence = field(default_factory=Confidence)

    def __post_init__(self) -> None:
        _opaque(self.alternative_id)
        _text(self.text)
        _require(type(self.confidence) is Confidence)


@dataclass(frozen=True, slots=True, kw_only=True)
class HandwritingCandidate:
    """Unverified handwritten region; confidence cannot change its state.

    Args:
        candidate_id: Opaque candidate identifier, never a patient or field label.
        evidence: Original PIXELS node with source digest, page extent and bounds.
        alternatives: Zero to sixteen hypotheses in caller-supplied order.
        recognizer_digest: Optional SHA-256 identifying recognition provenance.

    A region with no readable hypothesis can still be reviewed from its pixels.
    This class records the caller's handwriting classification; it is not a
    handwriting detector, OCR engine or clinical field-grounding algorithm.
    """

    candidate_id: str
    evidence: EvidenceNode
    alternatives: tuple[TranscriptionAlternative, ...] = field(repr=False)
    recognizer_digest: str | None = None

    def __post_init__(self) -> None:
        _opaque(self.candidate_id)
        _require(
            type(self.evidence) is EvidenceNode
            and self.evidence.kind is NodeKind.PIXELS
        )
        _require(
            type(self.alternatives) is tuple
            and len(self.alternatives) <= _MAX_ALTERNATIVES
        )
        _require(
            all(type(item) is TranscriptionAlternative for item in self.alternatives)
        )
        _require(
            len({item.alternative_id for item in self.alternatives})
            == len(self.alternatives)
        )
        if self.recognizer_digest is not None:
            _digest(self.recognizer_digest)

    @property
    def state(self) -> HandwritingState:
        """Always unverified, including hypotheses with confidence equal to one."""
        return HandwritingState.UNVERIFIED

    @property
    def digest(self) -> str:
        """Bind all alternatives, scores, order, original evidence and provenance."""
        return _hash({"schema_version": HANDWRITING_VERSION, "candidate": asdict(self)})

    def to_report(self) -> dict[str, object]:
        """Return evidence and hypothesis metadata without any recognized text."""
        return {
            "schema_version": HANDWRITING_VERSION,
            "source_kind": "handwriting",
            "candidate_id": self.candidate_id,
            "candidate_digest": self.digest,
            "state": self.state.value,
            "evidence": asdict(self.evidence),
            "recognizer_digest": self.recognizer_digest,
            "alternatives": [
                {
                    "alternative_id": item.alternative_id,
                    "confidence": item.confidence.score,
                }
                for item in self.alternatives
            ],
            "notice": HANDWRITING_NOTICE,
        }

    def to_json(self) -> str:
        """Serialize the content-free report, never the hypotheses."""
        return json.dumps(
            self.to_report(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class HandwritingReview:
    """A candidate-bound reviewer decision, pending until explicitly confirmed.

    Args:
        review_id: Opaque review-event identifier.
        reviewer_id: Opaque authenticated-reviewer reference managed by the caller.
        decision_digest: SHA-256 of externally retained approval evidence.
        candidate_id: Exact candidate being reviewed.
        candidate_digest: Complete candidate digest at review time.
        action: Select one alternative, provide a correction, reject or abstain.
        alternative_id: Required only for SELECT; no score-based default exists.
        corrected_text: Protected reviewed transcription, required only for CORRECT.
        reviewer_confirmed: Explicit caller assertion of human confirmation.

    The application authenticates the reviewer and retains the approval. Hash
    binding detects stale decisions; it does not authenticate fabricated reviews.
    """

    review_id: str
    reviewer_id: str
    decision_digest: str
    candidate_id: str
    candidate_digest: str
    action: ReviewAction
    alternative_id: str | None = None
    corrected_text: str | None = field(default=None, repr=False)
    reviewer_confirmed: bool = False

    def __post_init__(self) -> None:
        for reference in (self.review_id, self.reviewer_id, self.candidate_id):
            _opaque(reference)
        for digest in (self.decision_digest, self.candidate_digest):
            _digest(digest)
        _require(type(self.action) is ReviewAction)
        _require(type(self.reviewer_confirmed) is bool)
        if self.action is ReviewAction.SELECT:
            _require(self.alternative_id is not None and self.corrected_text is None)
            assert self.alternative_id is not None
            _opaque(self.alternative_id)
        elif self.action is ReviewAction.CORRECT:
            _require(self.alternative_id is None and self.corrected_text is not None)
            assert self.corrected_text is not None
            _text(self.corrected_text)
        else:
            _require(self.alternative_id is None and self.corrected_text is None)

    @property
    def digest(self) -> str:
        """Bind the exact selected/corrected value and the explicit review event."""
        return _hash({"schema_version": HANDWRITING_VERSION, "review": asdict(self)})


@dataclass(frozen=True, slots=True)
class ReviewedHandwriting:
    """Protected reviewed transcription or abstention with immutable visual lineage.

    This is an inspection result, not an export authorization. The export builder
    always revalidates original candidates and decisions instead of trusting it.
    """

    candidate_id: str
    candidate_digest: str
    evidence: EvidenceNode
    state: HandwritingState
    review_id: str
    reviewer_id: str
    review_digest: str
    decision_digest: str
    action: ReviewAction
    selected_alternative_id: str | None
    confidence: Confidence
    text: str | None = field(repr=False)

    def to_report(self) -> dict[str, object]:
        """Return lineage and selection metadata, without the protected value."""
        return {
            "source_kind": "handwriting",
            "candidate_id": self.candidate_id,
            "candidate_digest": self.candidate_digest,
            "evidence": asdict(self.evidence),
            "state": self.state.value,
            "review_id": self.review_id,
            "reviewer_id": self.reviewer_id,
            "review_digest": self.review_digest,
            "decision_digest": self.decision_digest,
            "action": self.action.value,
            "selected_alternative_id": self.selected_alternative_id,
            "confidence": self.confidence.score,
            "notice": HANDWRITING_NOTICE,
        }

    def to_json(self) -> str:
        """Serialize reviewed metadata without transcript text or alternatives."""
        return json.dumps(
            self.to_report(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )


def review_handwriting(
    candidate: HandwritingCandidate, review: HandwritingReview
) -> ReviewedHandwriting:
    """Resolve an exact-content, confirmed reviewer decision without auto-selection.

    Args:
        candidate: Current unverified candidate, including all alternatives.
        review: Explicit decision bound to the complete current candidate digest.

    Returns:
        Reviewed transcription, rejection or abstention with original coordinates.

    Raises:
        HandwritingReviewError: For stale, mismatched, unconfirmed or invalid choices.
    """
    _require(
        type(candidate) is HandwritingCandidate and type(review) is HandwritingReview
    )
    if (
        review.candidate_id != candidate.candidate_id
        or review.candidate_digest != candidate.digest
    ):
        raise HandwritingReviewError(
            "handwriting review does not match current candidate"
        )
    if not review.reviewer_confirmed:
        raise HandwritingReviewError(
            "handwriting review requires explicit confirmation"
        )
    text: str | None = None
    confidence = Confidence()
    if review.action is ReviewAction.SELECT:
        selected = next(
            (
                item
                for item in candidate.alternatives
                if item.alternative_id == review.alternative_id
            ),
            None,
        )
        if selected is None:
            raise HandwritingReviewError("handwriting alternative is not available")
        text, confidence = selected.text, selected.confidence
        state = HandwritingState.REVIEWED_TRANSCRIPTION
    elif review.action is ReviewAction.CORRECT:
        text = review.corrected_text
        state = HandwritingState.REVIEWED_TRANSCRIPTION
    elif review.action is ReviewAction.REJECT:
        state = HandwritingState.REJECTED
    else:
        state = HandwritingState.ABSTAINED
    return ReviewedHandwriting(
        candidate.candidate_id,
        candidate.digest,
        candidate.evidence,
        state,
        review.review_id,
        review.reviewer_id,
        review.digest,
        review.decision_digest,
        review.action,
        review.alternative_id,
        confidence,
        text,
    )


@dataclass(frozen=True, slots=True)
class HandwritingExport:
    """Protected structured JSON and content-free reviewed lineage.

    data_json contains reviewed values and belongs in an authorized export path,
    not logs. There is no file write, network transfer or implicit clinical use.
    """

    data_json: str = field(repr=False)
    items: tuple[ReviewedHandwriting, ...]

    def to_report_json(self) -> str:
        """Export audit metadata only; no recognized or corrected values."""
        return json.dumps(
            {
                "schema_version": HANDWRITING_VERSION,
                "notice": HANDWRITING_NOTICE,
                "items": [item.to_report() for item in self.items],
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


def export_handwriting(
    candidates: tuple[HandwritingCandidate, ...],
    reviews: tuple[HandwritingReview, ...],
) -> HandwritingExport:
    """Build an all-or-nothing structured export after fresh exact-candidate review.

    Args:
        candidates: One to 128 current unverified candidates, in export order.
        reviews: Exactly one explicit review per candidate; order is immaterial.

    Returns:
        Protected JSON with original source coordinates and review lineage. Values
        remain transcriptions, without automatic date/dose parsing or field mapping.

    Raises:
        HandwritingReviewError: If any candidate lacks a current confirmed accepted
            review, IDs conflict, or the bounded batch is exceeded. No partial export.
    """
    _require(type(candidates) is tuple and 1 <= len(candidates) <= _MAX_BATCH)
    _require(type(reviews) is tuple and len(reviews) == len(candidates))
    _require(all(type(item) is HandwritingCandidate for item in candidates))
    _require(all(type(item) is HandwritingReview for item in reviews))
    candidate_ids = {item.candidate_id for item in candidates}
    _require(len(candidate_ids) == len(candidates))
    _require({item.candidate_id for item in reviews} == candidate_ids)
    _require(len({item.review_id for item in reviews}) == len(reviews))
    _require(
        sum(
            len(alternative.text)
            for candidate in candidates
            for alternative in candidate.alternatives
        )
        + sum(len(review.corrected_text or "") for review in reviews)
        <= _MAX_BATCH_TEXT
    )
    by_candidate = {item.candidate_id: item for item in reviews}
    resolved = tuple(
        review_handwriting(item, by_candidate[item.candidate_id]) for item in candidates
    )
    if any(
        item.state is not HandwritingState.REVIEWED_TRANSCRIPTION for item in resolved
    ):
        raise HandwritingReviewError(
            "handwriting export requires every candidate to be accepted"
        )
    payload = {
        "schema_version": HANDWRITING_VERSION,
        "notice": HANDWRITING_NOTICE,
        "items": [{**item.to_report(), "value": item.text} for item in resolved],
    }
    return HandwritingExport(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False),
        resolved,
    )
