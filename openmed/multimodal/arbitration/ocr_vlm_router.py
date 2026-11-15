"""Compare OCR/VLM observations without choosing a fluent or synthesized value."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from enum import Enum
from fractions import Fraction

from openmed.clinical.review_packet import (
    ReviewCitation,
    ReviewFinding,
    ReviewGateResult,
    ReviewPacket,
)
from openmed.multimodal.evidence import EvidenceNode, NodeKind
from openmed.multimodal.vision.providers.base import Confidence

ROUTING_VERSION = "openmed.ocr-vlm-routing.v1"
ROUTING_NOTICE = "Agreement between observations is not clinical verification; retain downstream transcription, association and clinical review gates."
_OPAQUE = re.compile(r"[0-9a-f]{32}\Z")
_MAX_PAIRS = 128
_MAX_TEXT = 4096
_MAX_TOTAL_TEXT = 262_144


class ArbitrationError(ValueError):
    """Controlled invalid-input or review-required failure without source values."""


class ReadingSource(str, Enum):
    """Independent observation source; never a relative trust ranking."""

    OCR = "ocr"
    VLM = "vlm"


class SemanticType(str, Enum):
    """Declared field semantics; no type inference or clinical interpretation."""

    TEXT = "text"
    IDENTIFIER = "identifier"
    NUMBER = "number"
    DATE = "date"
    DOSE = "dose"
    CONTROL = "control"
    UNKNOWN = "unknown"


class TextNormalization(str, Enum):
    """Comparison-only operations that never replace a protected source string."""

    EXACT = "exact"
    NFC_WHITESPACE = "nfc_whitespace"


class RoutingState(str, Enum):
    """Agreement admission is independent of downstream clinical approval."""

    AGREEMENT = "policy_permitted_agreement"
    REVIEW_REQUIRED = "review_required"


class Disagreement(str, Enum):
    """Closed reasons; multiple independent conflicts are retained together."""

    FIELD = "field_mismatch"
    SOURCE = "source_mismatch"
    PAGE = "page_mismatch"
    GEOMETRY = "geometry_mismatch"
    TEXT = "text_mismatch"
    SEMANTIC_TYPE = "semantic_type_mismatch"
    UNKNOWN_TYPE = "semantic_type_unknown"
    CONFIDENCE_UNKNOWN = "confidence_unknown"
    CONFIDENCE_LOW = "confidence_below_policy"
    TYPE_POLICY = "semantic_type_requires_review"
    HANDWRITING = "handwriting_requires_review"


def _require(condition: bool) -> None:
    if not condition:
        raise ArbitrationError("invalid OCR/VLM arbitration input")


def _opaque(value: str) -> None:
    _require(type(value) is str and _OPAQUE.fullmatch(value) is not None)


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class ReadingCandidate:
    """Protected original reading with explicit field, visual evidence and uncertainty.

    Args:
        candidate_id: Opaque identity for this source observation.
        field_id: Opaque identity of the field the caller intends to compare.
        source: OCR or VLM, independent of confidence.
        text: Original recognized string, excluded from representations and reports.
        semantic_type: Declared semantics, without automatic conversion.
        evidence: Original OCR span or VLM pixel region.
        confidence: Uncalibrated recognition score, or unknown.
        handwritten: Explicit observation classification; always routed to review.
    """

    candidate_id: str
    field_id: str
    source: ReadingSource
    text: str = field(repr=False)
    semantic_type: SemanticType
    evidence: EvidenceNode
    confidence: Confidence = field(default_factory=Confidence)
    handwritten: bool = False

    def __post_init__(self) -> None:
        _opaque(self.candidate_id)
        _opaque(self.field_id)
        _require(type(self.source) is ReadingSource)
        _require(type(self.semantic_type) is SemanticType)
        _require(type(self.confidence) is Confidence and type(self.handwritten) is bool)
        _require(type(self.text) is str and 0 < len(self.text) <= _MAX_TEXT)
        _require(bool(self.text.strip()))
        valid = True
        try:
            self.text.encode("utf-8")
        except UnicodeError:
            valid = False
        _require(valid)
        _require(type(self.evidence) is EvidenceNode)
        if self.source is ReadingSource.OCR:
            _require(self.evidence.kind is NodeKind.OCR)
            assert self.evidence.span is not None
            _require(len(self.text) == self.evidence.span[1] - self.evidence.span[0])
        else:
            _require(self.evidence.kind is NodeKind.PIXELS)


@dataclass(frozen=True, slots=True, kw_only=True)
class ReadingPair:
    """An explicit comparison, not an inferred pairing of unrelated OCR fragments.

    Args:
        pair_id: Unique opaque comparison identifier.
        ocr: Original OCR observation.
        vlm: Original VLM observation.

    Wrong field/source/page pairings remain reviewable conflicts rather than
    silently becoming equivalent because their strings happen to match.
    """

    pair_id: str
    ocr: ReadingCandidate = field(repr=False)
    vlm: ReadingCandidate = field(repr=False)

    def __post_init__(self) -> None:
        _opaque(self.pair_id)
        _require(
            type(self.ocr) is ReadingCandidate and type(self.vlm) is ReadingCandidate
        )
        _require(
            self.ocr.source is ReadingSource.OCR
            and self.vlm.source is ReadingSource.VLM
        )
        _require(self.ocr.candidate_id != self.vlm.candidate_id)
        _require(self.ocr.evidence.node_id != self.vlm.evidence.node_id)


@dataclass(frozen=True, slots=True, kw_only=True)
class AgreementPolicy:
    """Explicit agreement permissions and bounded review capacity.

    Args:
        version: Non-sensitive policy version reference.
        allowed_types: Semantic types permitted to pass this agreement gate.
        normalization: Exact or NFC/Unicode-whitespace comparison only.
        minimum_geometry_iou: Positive minimum original-box intersection over union.
        minimum_confidence: Required score for both observations, not their average.
        preferred_source: Which original string to retain for permitted agreements.
        maximum_review_items: Maximum complete review queue; overflow fails closed.

    No policy defaults or clinical thresholds are supplied. UNKNOWN types and
    handwritten observations cannot be allowed through this agreement gate.
    """

    version: str
    allowed_types: tuple[SemanticType, ...]
    normalization: TextNormalization
    minimum_geometry_iou: float
    minimum_confidence: float
    preferred_source: ReadingSource
    maximum_review_items: int

    def __post_init__(self) -> None:
        _require(
            type(self.version) is str
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", self.version)
            is not None
        )
        _require(
            type(self.allowed_types) is tuple
            and len(self.allowed_types) < len(SemanticType)
        )
        _require(
            all(
                type(kind) is SemanticType and kind is not SemanticType.UNKNOWN
                for kind in self.allowed_types
            )
        )
        _require(len(set(self.allowed_types)) == len(self.allowed_types))
        _require(type(self.normalization) is TextNormalization)
        _require(type(self.preferred_source) is ReadingSource)
        _require(
            type(self.minimum_geometry_iou) in (int, float)
            and 0 < self.minimum_geometry_iou <= 1
        )
        _require(
            type(self.minimum_confidence) in (int, float)
            and 0 <= self.minimum_confidence <= 1
        )
        _require(
            type(self.maximum_review_items) is int
            and 0 <= self.maximum_review_items <= _MAX_PAIRS
        )

    @property
    def digest(self) -> str:
        """Bind every policy setting and the routing algorithm version."""
        return _hash({"version": ROUTING_VERSION, "policy": asdict(self)})


@dataclass(frozen=True, slots=True)
class ReadingComparison:
    """Content-free decision retaining both original visual citations and scores."""

    pair_id: str
    candidate_ids: tuple[str, str]
    field_ids: tuple[str, str]
    state: RoutingState
    reasons: tuple[Disagreement, ...]
    normalized_equal: bool
    geometry_iou: float | None
    confidence: tuple[float | None, float | None]
    semantic_types: tuple[SemanticType, SemanticType]
    evidence: tuple[EvidenceNode, EvidenceNode]
    selected_candidate_id: str | None


@dataclass(frozen=True, slots=True)
class RoutingReport:
    """Complete bounded results with no candidate text or synthesized readings."""

    input_digest: str
    policy_digest: str
    comparisons: tuple[ReadingComparison, ...]

    @property
    def review_queue(self) -> tuple[ReadingComparison, ...]:
        """Return every review-required comparison; never truncate conflicts."""
        return tuple(
            item
            for item in self.comparisons
            if item.state is RoutingState.REVIEW_REQUIRED
        )

    @property
    def review_citation_map(self) -> dict[str, EvidenceNode]:
        """Map the review renderer's normalized citation IDs to original boxes."""
        return {
            ReviewCitation(citation_id=node.node_id, source="source").citation_id: node
            for item in self.review_queue
            for node in item.evidence
        }

    def to_json(self) -> str:
        """Return decisions, source coordinates and input bindings without values."""
        return json.dumps(
            {
                **asdict(self),
                "schema_version": ROUTING_VERSION,
                "notice": ROUTING_NOTICE,
                "review_citation_ids": {
                    key: node.node_id for key, node in self.review_citation_map.items()
                },
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def to_review_packet(self) -> ReviewPacket:
        """Render unresolved comparisons with blocking gates and both source citations.

        Candidate strings stay in the caller's protected input store. Original
        visual coordinates are available through review_citation_map. Rendering
        never resolves a disagreement or grants clinical approval.
        """
        citations = tuple(
            ReviewCitation(
                citation_id=node.node_id,
                source="source",
                source_hash=f"sha256:{node.source_digest}",
            )
            for item in self.review_queue
            for node in item.evidence
        )
        findings = tuple(
            ReviewFinding(
                finding_id=item.pair_id,
                label="finding",
                status="needs_review",
                citation_ids=tuple(node.node_id for node in item.evidence),
            )
            for item in self.review_queue
        )
        gates = tuple(
            ReviewGateResult(
                gate_id=f"review-{item.pair_id}",
                passed=False,
                reason="review_required",
                severity="warning",
                blocking=True,
                citation_ids=tuple(node.node_id for node in item.evidence),
            )
            for item in self.review_queue
        )
        return ReviewPacket(findings=findings, citations=citations, gate_results=gates)


def _normalized(text: str, mode: TextNormalization) -> str:
    if mode is TextNormalization.EXACT:
        return text
    return " ".join(unicodedata.normalize("NFC", text).split())


def _iou(a: EvidenceNode, b: EvidenceNode) -> float:
    assert a.region is not None and b.region is not None
    # Rational arithmetic keeps tiny valid boxes from underflowing to zero area.
    x0, y0, x1, y1 = map(Fraction, a.region)
    u0, v0, u1, v1 = map(Fraction, b.region)
    intersection = max(Fraction(0), min(x1, u1) - max(x0, u0)) * max(
        Fraction(0), min(y1, v1) - max(y0, v0)
    )
    union = (x1 - x0) * (y1 - y0) + (u1 - u0) * (v1 - v0) - intersection
    return float(intersection / union)


def _compare(pair: ReadingPair, policy: AgreementPolicy) -> ReadingComparison:
    a, b = pair.ocr, pair.vlm
    reasons: list[Disagreement] = []
    if a.field_id != b.field_id:
        reasons.append(Disagreement.FIELD)
    same_source = a.evidence.source_digest == b.evidence.source_digest
    if not same_source:
        reasons.append(Disagreement.SOURCE)
    assert a.evidence.geometry is not None and b.evidence.geometry is not None
    same_page = a.evidence.geometry.page == b.evidence.geometry.page
    if not same_page:
        reasons.append(Disagreement.PAGE)
    same_geometry = a.evidence.geometry == b.evidence.geometry
    overlap = (
        _iou(a.evidence, b.evidence)
        if same_source and same_page and same_geometry
        else None
    )
    if not same_geometry or (
        overlap is not None and overlap < policy.minimum_geometry_iou
    ):
        reasons.append(Disagreement.GEOMETRY)
    equal = _normalized(a.text, policy.normalization) == _normalized(
        b.text, policy.normalization
    )
    if not equal:
        reasons.append(Disagreement.TEXT)
    if a.semantic_type is not b.semantic_type:
        reasons.append(Disagreement.SEMANTIC_TYPE)
    if SemanticType.UNKNOWN in (a.semantic_type, b.semantic_type):
        reasons.append(Disagreement.UNKNOWN_TYPE)
    if any(item.semantic_type not in policy.allowed_types for item in (a, b)):
        reasons.append(Disagreement.TYPE_POLICY)
    scores = (a.confidence.score, b.confidence.score)
    if None in scores:
        reasons.append(Disagreement.CONFIDENCE_UNKNOWN)
    if any(score is not None and score < policy.minimum_confidence for score in scores):
        reasons.append(Disagreement.CONFIDENCE_LOW)
    if a.handwritten or b.handwritten:
        reasons.append(Disagreement.HANDWRITING)
    selected = (
        (a if policy.preferred_source is ReadingSource.OCR else b).candidate_id
        if not reasons
        else None
    )
    return ReadingComparison(
        pair.pair_id,
        (a.candidate_id, b.candidate_id),
        (a.field_id, b.field_id),
        RoutingState.REVIEW_REQUIRED if reasons else RoutingState.AGREEMENT,
        tuple(reasons),
        equal,
        overlap,
        scores,
        (a.semantic_type, b.semantic_type),
        (a.evidence, b.evidence),
        selected,
    )


def route_ocr_vlm(
    pairs: tuple[ReadingPair, ...], policy: AgreementPolicy
) -> RoutingReport:
    """Compare every pair and return agreements plus a complete bounded review queue.

    Args:
        pairs: One to 128 explicit OCR/VLM field comparisons with unique observations.
        policy: Explicit agreement permissions, thresholds and review capacity.

    Returns:
        Value-free decisions and unchanged original visual citations.

    Raises:
        ArbitrationError: For invalid inputs or review-capacity overflow; no partial report.
    """
    _require(type(policy) is AgreementPolicy)
    _require(type(pairs) is tuple and 1 <= len(pairs) <= _MAX_PAIRS)
    _require(all(type(pair) is ReadingPair for pair in pairs))
    _require(len({pair.pair_id for pair in pairs}) == len(pairs))
    candidates = tuple(item for pair in pairs for item in (pair.ocr, pair.vlm))
    _require(len({item.candidate_id for item in candidates}) == len(candidates))
    _require(len({item.evidence.node_id for item in candidates}) == len(candidates))
    # A field may occur once per source, preventing duplicate/conflicting exports.
    for source in ReadingSource:
        field_ids = [item.field_id for item in candidates if item.source is source]
        _require(len(set(field_ids)) == len(field_ids))
    _require(sum(len(item.text) for item in candidates) <= _MAX_TOTAL_TEXT)
    comparisons = tuple(_compare(pair, policy) for pair in pairs)
    if (
        sum(item.state is RoutingState.REVIEW_REQUIRED for item in comparisons)
        > policy.maximum_review_items
    ):
        raise ArbitrationError("OCR/VLM review capacity exceeded")
    binding = _hash(
        {
            "version": ROUTING_VERSION,
            "pairs": [asdict(pair) for pair in pairs],
            "policy": policy.digest,
        }
    )
    return RoutingReport(binding, policy.digest, comparisons)


@dataclass(frozen=True, slots=True)
class AgreedValues:
    """Protected agreement data and a separate content-free routing audit."""

    data_json: str = field(repr=False)
    audit_json: str

    def to_report_json(self) -> str:
        """Return source lineage and decisions without recognized strings."""
        return self.audit_json


def prepare_agreed_values(
    pairs: tuple[ReadingPair, ...], policy: AgreementPolicy
) -> AgreedValues:
    """Freshly enforce agreement permissions before preparing original source values.

    Args:
        pairs: Current complete original observations, not a previously approved report.
        policy: Current explicit agreement policy.

    Returns:
        Original selected strings with both source citations. This intermediate
        data does not approve form association, handwriting or clinical use.

    Raises:
        ArbitrationError: If any comparison requires review; no partial data is returned.
    """
    report = route_ocr_vlm(pairs, policy)
    if report.review_queue:
        raise ArbitrationError("OCR/VLM disagreements require review")
    rows = []
    for pair, comparison in zip(pairs, report.comparisons):
        candidate = (
            pair.ocr if policy.preferred_source is ReadingSource.OCR else pair.vlm
        )
        rows.append({**asdict(comparison), "value": candidate.text})
    return AgreedValues(
        json.dumps(
            {
                "schema_version": ROUTING_VERSION,
                "input_digest": report.input_digest,
                "policy_digest": report.policy_digest,
                "notice": ROUTING_NOTICE,
                "items": rows,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        report.to_json(),
    )
