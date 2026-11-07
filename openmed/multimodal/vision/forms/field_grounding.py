"""Associate bounded form regions geometrically without inventing field values."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, replace
from enum import Enum

from openmed.clinical.review_packet import (
    ReviewCitation,
    ReviewFinding,
    ReviewGateResult,
    ReviewPacket,
)
from openmed.multimodal.box_normalization import BoxNormalizationError, normalize_box
from openmed.multimodal.evidence import EvidenceNode, NodeKind, PageGeometry
from openmed.multimodal.vision.handwriting.uncertainty import (
    HandwritingCandidate,
    HandwritingReview,
    HandwritingReviewError,
    HandwritingState,
    ReviewedHandwriting,
    review_handwriting,
)
from openmed.multimodal.vision.providers.base import Confidence

GROUNDING_VERSION = "openmed.visual-form-grounding.v1"
GROUNDING_NOTICE = "Visual association and transcription review only; no clinical validation or automatic clinical decisions."
_OPAQUE = re.compile(r"[0-9a-f]{32}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_MAX_REGIONS = 128
_MAX_TEXT = 4096
_MAX_TOTAL_TEXT = 262_144


class FormGroundingError(ValueError):
    """Controlled invalid-input or blocked-export reason without source values."""


class ValueKind(str, Enum):
    """Separate printed text, handwriting and explicit visual control states."""

    TEXT = "printed_text"
    HANDWRITING = "handwriting"
    CHECKBOX = "checkbox"
    RADIO = "radio"


class ControlState(str, Enum):
    """Unknown is never converted to unchecked."""

    CHECKED = "checked"
    UNCHECKED = "unchecked"
    UNKNOWN = "unknown"


class LabelRelation(str, Enum):
    """Expected position of the value/control relative to its label."""

    RIGHT = "value_right_of_label"
    BELOW = "value_below_label"
    LEFT = "value_left_of_label"


class GroundingState(str, Enum):
    """A unique proposal still requires reviewer confirmation."""

    REVIEW_REQUIRED = "review_required"
    ABSTAINED = "abstained"


def _require(condition: bool) -> None:
    if not condition:
        raise FormGroundingError("invalid visual form declaration")


def _opaque(value: str) -> None:
    _require(type(value) is str and _OPAQUE.fullmatch(value) is not None)


def _digest(value: str) -> None:
    _require(type(value) is str and _DIGEST.fullmatch(value) is not None)


def _text(value: str) -> None:
    _require(type(value) is str and 0 < len(value) <= _MAX_TEXT and bool(value.strip()))
    valid = True
    try:
        value.encode("utf-8")
    except UnicodeError:
        valid = False
    _require(valid)


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _visual(node: EvidenceNode) -> None:
    _require(
        type(node) is EvidenceNode and node.kind in (NodeKind.PIXELS, NodeKind.OCR)
    )


def _same_page(a: EvidenceNode, b: EvidenceNode) -> bool:
    assert a.geometry is not None and b.geometry is not None
    return a.source_digest == b.source_digest and a.geometry == b.geometry


def _contains(outer: EvidenceNode, inner: EvidenceNode) -> bool:
    if not _same_page(outer, inner):
        return False
    assert outer.region is not None and inner.region is not None
    a, b = outer.region, inner.region
    return a[0] <= b[0] < b[2] <= a[2] and a[1] <= b[1] < b[3] <= a[3]


@dataclass(frozen=True, slots=True)
class FormSection:
    """Explicit section extent in original source coordinates; never inferred text."""

    evidence: EvidenceNode

    def __post_init__(self) -> None:
        _require(
            type(self.evidence) is EvidenceNode
            and self.evidence.kind is NodeKind.PIXELS
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class FormLabel:
    """Protected field label and independently declared section/layout expectation.

    Args:
        evidence: Original label box and optional OCR source span.
        text: Recognized label text, excluded from representations and reports.
        section_id: Explicit parent section's evidence node ID.
        relation: Expected value position, without guessing a form convention.
        accepts: Value kinds accepted by this declared field.
        confidence: Uncalibrated label-recognition confidence or unknown.
    """

    evidence: EvidenceNode
    text: str = field(repr=False)
    section_id: str
    relation: LabelRelation
    accepts: tuple[ValueKind, ...]
    confidence: Confidence = field(default_factory=Confidence)

    def __post_init__(self) -> None:
        _visual(self.evidence)
        _text(self.text)
        _opaque(self.section_id)
        _require(
            type(self.relation) is LabelRelation and type(self.confidence) is Confidence
        )
        _require(
            type(self.accepts) is tuple and 1 <= len(self.accepts) <= len(ValueKind)
        )
        _require(all(type(kind) is ValueKind for kind in self.accepts))
        _require(len(set(self.accepts)) == len(self.accepts))
        if self.evidence.kind is NodeKind.OCR:
            assert self.evidence.span is not None
            _require(len(self.text) == self.evidence.span[1] - self.evidence.span[0])


@dataclass(frozen=True, slots=True, kw_only=True)
class FormValue:
    """A value region whose data kind cannot be silently changed during export.

    Args:
        evidence: Original value or control box; OCR spans describe printed text only.
        kind: Printed text, handwriting, checkbox or radio control.
        text: Required only for printed text; never interpreted as a dose/date.
        handwriting: Required only for handwriting, with exactly matching evidence.
        control_state: Required only for controls; UNKNOWN blocks value export.
        confidence: Uncalibrated recognition confidence; handwriting uses unknown here.
    """

    evidence: EvidenceNode
    kind: ValueKind
    text: str | None = field(default=None, repr=False)
    handwriting: HandwritingCandidate | None = field(default=None, repr=False)
    control_state: ControlState | None = field(default=None, repr=False)
    confidence: Confidence = field(default_factory=Confidence)

    def __post_init__(self) -> None:
        _visual(self.evidence)
        _require(type(self.kind) is ValueKind and type(self.confidence) is Confidence)
        if self.kind is ValueKind.TEXT:
            _require(
                self.text is not None
                and self.handwriting is None
                and self.control_state is None
            )
            assert self.text is not None
            _text(self.text)
            if self.evidence.kind is NodeKind.OCR:
                assert self.evidence.span is not None
                _require(
                    len(self.text) == self.evidence.span[1] - self.evidence.span[0]
                )
        elif self.kind is ValueKind.HANDWRITING:
            _require(
                type(self.handwriting) is HandwritingCandidate
                and self.text is None
                and self.control_state is None
            )
            assert self.handwriting is not None
            _require(
                self.handwriting.evidence == self.evidence
                and self.confidence.score is None
            )
        else:
            _require(
                self.text is None
                and self.handwriting is None
                and type(self.control_state) is ControlState
            )
            _require(self.evidence.kind is NodeKind.PIXELS)


@dataclass(frozen=True, slots=True, kw_only=True)
class VisualForm:
    """Bounded immutable recognized form observations, without reading-order inference.

    Args:
        sections: One to 32 declared source section boxes.
        labels: One to 128 recognized field labels, explicitly contained in a section.
        values: One to 128 typed candidate values or controls.

    Values can be missing a section or overlap sections; those are abstentions.
    Inconsistent source-page geometry or duplicate role IDs are malformed inputs.
    """

    sections: tuple[FormSection, ...]
    labels: tuple[FormLabel, ...] = field(repr=False)
    values: tuple[FormValue, ...] = field(repr=False)

    def __post_init__(self) -> None:
        for items, cls, limit in (
            (self.sections, FormSection, 32),
            (self.labels, FormLabel, _MAX_REGIONS),
            (self.values, FormValue, _MAX_REGIONS),
        ):
            _require(type(items) is tuple and 1 <= len(items) <= limit)
            _require(all(type(item) is cls for item in items))
        nodes = self.evidence
        _require(len({node.node_id for node in nodes}) == len(nodes))
        pages: dict[tuple[str | None, int], PageGeometry] = {}
        for node in nodes:
            assert node.geometry is not None
            key = (node.source_digest, node.geometry.page)
            _require(key not in pages or pages[key] == node.geometry)
            pages[key] = node.geometry
        sections = {item.evidence.node_id: item.evidence for item in self.sections}
        for label in self.labels:
            _require(
                label.section_id in sections
                and _contains(sections[label.section_id], label.evidence)
            )
        handwriting = [
            item.handwriting for item in self.values if item.handwriting is not None
        ]
        _require(len({item.candidate_id for item in handwriting}) == len(handwriting))
        _require(self.protected_text_size <= _MAX_TOTAL_TEXT)

    @property
    def evidence(self) -> tuple[EvidenceNode, ...]:
        """Return unchanged source nodes for section, label and value roles."""
        return (
            tuple(item.evidence for item in self.sections)
            + tuple(item.evidence for item in self.labels)
            + tuple(item.evidence for item in self.values)
        )

    @property
    def protected_text_size(self) -> int:
        """Count protected code points for bounded form/export admission."""
        return (
            sum(len(label.text) for label in self.labels)
            + sum(len(value.text or "") for value in self.values)
            + sum(
                len(alt.text)
                for value in self.values
                if value.handwriting is not None
                for alt in value.handwriting.alternatives
            )
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class GroundingPolicy:
    """Explicit sample-independent geometric thresholds, not clinical confidence.

    Args:
        version: Opaque version identifier, without protected content.
        maximum_gap: Maximum directional gap as a fraction of page width/height.
        minimum_axis_overlap: Minimum overlap divided by the shorter aligned side.
        minimum_score: Minimum overlap-times-distance score for a candidate link.
        ambiguity_margin: Top scores this close require abstention, including ties.
    """

    version: str
    maximum_gap: float
    minimum_axis_overlap: float
    minimum_score: float
    ambiguity_margin: float

    def __post_init__(self) -> None:
        _require(
            type(self.version) is str
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", self.version)
            is not None
        )
        for value in (
            self.maximum_gap,
            self.minimum_axis_overlap,
            self.minimum_score,
            self.ambiguity_margin,
        ):
            _require(type(value) in (int, float) and 0 <= value <= 1)
        _require(self.maximum_gap > 0)

    @property
    def digest(self) -> str:
        """Bind all geometric thresholds and the association algorithm version."""
        return _hash({"version": GROUNDING_VERSION, "policy": asdict(self)})


@dataclass(frozen=True, slots=True)
class FieldAssociation:
    """One eligible visual association, with score and original evidence references."""

    association_id: str
    value_id: str
    label_id: str
    section_id: str
    relation: LabelRelation
    geometry_score: float
    confidence: float | None


@dataclass(frozen=True, slots=True)
class ValueGrounding:
    """All eligible associations and controlled reasons for abstention or review."""

    value_id: str
    kind: ValueKind
    state: GroundingState
    associations: tuple[FieldAssociation, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FormGroundingReport:
    """Content-free input binding, alternatives and unchanged original coordinates."""

    grounding_digest: str
    policy_digest: str
    values: tuple[ValueGrounding, ...]
    evidence: tuple[EvidenceNode, ...]

    def to_json(self) -> str:
        """Return aggregate association metadata without labels or values."""
        return json.dumps(
            {
                **asdict(self),
                "schema_version": GROUNDING_VERSION,
                "notice": GROUNDING_NOTICE,
                "review_citation_ids": {
                    reference: node.node_id
                    for reference, node in self.review_citation_map.items()
                },
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @property
    def review_citation_map(self) -> dict[str, EvidenceNode]:
        """Map the review renderer's normalized IDs back to original visual nodes."""
        return {
            ReviewCitation(citation_id=node.node_id, source="source").citation_id: node
            for node in self.evidence
        }

    def to_review_packet(self) -> ReviewPacket:
        """Adapt findings and citation IDs to the existing protected review renderer.

        Original page boxes remain accessible through review_citation_map. The
        existing renderer normalizes identifiers; its privacy policy is preserved.
        The packet never contains source text and is not an export approval.
        """
        citations = tuple(
            ReviewCitation(
                citation_id=node.node_id,
                source="source",
                source_hash=f"sha256:{node.source_digest}",
            )
            for node in self.evidence
        )
        findings = tuple(
            ReviewFinding(
                finding_id=item.value_id,
                label="finding",
                status="uncertain"
                if item.state is GroundingState.ABSTAINED
                else "needs_review",
                citation_ids=tuple(
                    sorted(
                        {item.value_id}
                        | {
                            ref
                            for link in item.associations
                            for ref in (link.label_id, link.section_id)
                        }
                    )
                ),
            )
            for item in self.values
        )
        gates = tuple(
            ReviewGateResult(
                gate_id=f"review-{item.value_id}",
                passed=False,
                reason="review_required",
                severity="warning",
                blocking=True,
                citation_ids=(item.value_id,),
            )
            for item in self.values
        )
        return ReviewPacket(findings=findings, citations=citations, gate_results=gates)


def _box(node: EvidenceNode) -> tuple[float, float, float, float]:
    assert node.geometry is not None and node.region is not None
    try:
        return normalize_box(
            node.region,
            unit=node.geometry.unit.value,
            page_size=(node.geometry.width, node.geometry.height),
            origin="top-left",
            page=node.geometry.page,
            source_ref=node.node_id,
        ).bbox
    except BoxNormalizationError:
        pass
    raise FormGroundingError("form geometry cannot be normalized")


def _score(label: FormLabel, value: FormValue, policy: GroundingPolicy) -> float | None:
    a, b = _box(label.evidence), _box(value.evidence)
    if label.relation is LabelRelation.BELOW:
        gap = b[1] - a[3]
        overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) / min(
            a[2] - a[0], b[2] - b[0]
        )
    else:
        gap = b[0] - a[2] if label.relation is LabelRelation.RIGHT else a[0] - b[2]
        overlap = max(0.0, min(a[3], b[3]) - max(a[1], b[1])) / min(
            a[3] - a[1], b[3] - b[1]
        )
    if not 0 <= gap <= policy.maximum_gap or overlap < policy.minimum_axis_overlap:
        return None
    score = round(overlap * (1 - gap / policy.maximum_gap), 12)
    return score if score >= policy.minimum_score else None


def ground_form_fields(
    form: VisualForm, policy: GroundingPolicy
) -> FormGroundingReport:
    """Compute eligible geometry links, preserve alternatives and abstain on ambiguity.

    Args:
        form: Original source regions, explicit sections/labels and protected values.
        policy: Versioned geometric candidate and ambiguity thresholds.

    Returns:
        Review-required proposals or abstentions. No automatic field export or
        confidence-based transcription selection is performed.
    """
    _require(type(form) is VisualForm and type(policy) is GroundingPolicy)
    results = []
    for value in form.values:
        members = {
            section.evidence.node_id
            for section in form.sections
            if _contains(section.evidence, value.evidence)
        }
        links = []
        for label in form.labels:
            if (
                label.section_id not in members
                or value.kind not in label.accepts
                or not _same_page(label.evidence, value.evidence)
            ):
                continue
            score = _score(label, value, policy)
            if score is None:
                continue
            label_score, value_score = label.confidence.score, value.confidence.score
            confidence = (
                min(label_score, value_score)
                if label_score is not None and value_score is not None
                else None
            )
            identity = (
                value.evidence.node_id,
                label.evidence.node_id,
                label.section_id,
                label.relation.value,
            )
            links.append(
                FieldAssociation(
                    _hash(identity)[:32],
                    *identity[:3],
                    label.relation,
                    score,
                    confidence,
                )
            )
        links.sort(key=lambda item: (-item.geometry_score, item.association_id))
        reasons = []
        if len(members) != 1:
            reasons.append("section_missing" if not members else "section_ambiguous")
        if not links:
            reasons.append("association_missing")
        elif (
            len(links) > 1
            and links[0].geometry_score - links[1].geometry_score
            <= policy.ambiguity_margin
        ):
            reasons.append("association_ambiguous")
        if value.control_state is ControlState.UNKNOWN:
            reasons.append("control_state_unknown")
        state = GroundingState.ABSTAINED if reasons else GroundingState.REVIEW_REQUIRED
        results.append(
            ValueGrounding(
                value.evidence.node_id,
                value.kind,
                state,
                tuple(links),
                tuple(reasons or ["review_required"]),
            )
        )
    # Independent nearest-neighbour choices cannot silently assign a field twice.
    claims: dict[str, list[int]] = {}
    for index, result in enumerate(results):
        if result.associations:
            claims.setdefault(result.associations[0].label_id, []).append(index)
    for indices in claims.values():
        if len(indices) > 1:
            for index in indices:
                item = results[index]
                results[index] = replace(
                    item,
                    state=GroundingState.ABSTAINED,
                    reasons=tuple(sorted(set(item.reasons) | {"field_competition"})),
                )
    nodes = form.evidence
    binding = _hash(
        {
            "version": GROUNDING_VERSION,
            "form": asdict(form),
            "policy": policy.digest,
            "results": [asdict(item) for item in results],
        }
    )
    return FormGroundingReport(binding, policy.digest, tuple(results), nodes)


@dataclass(frozen=True, slots=True, kw_only=True)
class FormAssociationReview:
    """Explicit reviewer-selected association bound to the complete current form.

    Args:
        value_id: Exact value/control node ID.
        association_id: One eligible association listed by the current report.
        grounding_digest: Digest of the complete current form and policy report.
        review_id: Opaque review-event ID.
        reviewer_id: Opaque authenticated-reviewer reference owned by the application.
        decision_digest: Digest of retained approval evidence.
        handwriting_review_digest: Required for handwriting, binding the exact reading.
        reviewer_confirmed: Must be explicitly true before export.
    """

    value_id: str
    association_id: str
    grounding_digest: str
    review_id: str
    reviewer_id: str
    decision_digest: str
    handwriting_review_digest: str | None = None
    reviewer_confirmed: bool = False

    def __post_init__(self) -> None:
        for value in (
            self.value_id,
            self.association_id,
            self.review_id,
            self.reviewer_id,
        ):
            _opaque(value)
        for value in (self.grounding_digest, self.decision_digest):
            _digest(value)
        if self.handwriting_review_digest is not None:
            _digest(self.handwriting_review_digest)
        _require(type(self.reviewer_confirmed) is bool)


@dataclass(frozen=True, slots=True)
class GroundedFormExport:
    """Protected structured field data and a separate value-free audit JSON."""

    data_json: str = field(repr=False)
    audit_json: str

    def to_report_json(self) -> str:
        """Return visual citations and review decisions without label/value text."""
        return self.audit_json


def _handwriting(
    candidate: HandwritingCandidate, review: HandwritingReview
) -> ReviewedHandwriting:
    try:
        result = review_handwriting(candidate, review)
    except HandwritingReviewError:
        pass
    else:
        if result.state is HandwritingState.REVIEWED_TRANSCRIPTION:
            return result
    raise FormGroundingError(
        "form export requires an accepted current handwriting review"
    )


def export_grounded_fields(
    form: VisualForm,
    policy: GroundingPolicy,
    reviews: tuple[FormAssociationReview, ...],
    *,
    handwriting_reviews: tuple[HandwritingReview, ...] = (),
) -> GroundedFormExport:
    """Freshly ground and review every field before an all-or-nothing structured export.

    Args:
        form: Current original source observations and protected values.
        policy: Explicit geometric policy, recomputed before export.
        reviews: Exactly one confirmed eligible association choice per value.
        handwriting_reviews: Exactly one accepted transcription review per handwritten value.

    Returns:
        Protected field JSON with original visual citations. An explicit reviewer
        choice can resolve a listed ambiguous association, but cannot invent a link,
        reuse a label twice or reinterpret an unknown control as unchecked.

    Raises:
        FormGroundingError: For incomplete/stale/conflicting approvals or unresolved values.
    """
    report = ground_form_fields(form, policy)
    _require(type(reviews) is tuple and len(reviews) == len(form.values))
    _require(all(type(item) is FormAssociationReview for item in reviews))
    _require(
        {item.value_id for item in reviews}
        == {item.evidence.node_id for item in form.values}
    )
    _require(len({item.review_id for item in reviews}) == len(reviews))
    handwritten = {
        item.handwriting.candidate_id
        for item in form.values
        if item.handwriting is not None
    }
    _require(
        type(handwriting_reviews) is tuple
        and len(handwriting_reviews) == len(handwritten)
    )
    _require(all(type(item) is HandwritingReview for item in handwriting_reviews))
    _require({item.candidate_id for item in handwriting_reviews} == handwritten)
    _require(
        len({item.review_id for item in handwriting_reviews})
        == len(handwriting_reviews)
    )
    _require(
        form.protected_text_size
        + sum(len(item.corrected_text or "") for item in handwriting_reviews)
        <= _MAX_TOTAL_TEXT
    )
    decisions = {item.value_id: item for item in reviews}
    transcriptions = {item.candidate_id: item for item in handwriting_reviews}
    labels = {item.evidence.node_id: item for item in form.labels}
    sections = {item.evidence.node_id: item for item in form.sections}
    selected_labels: set[str] = set()
    protected_rows = []
    audit_rows = []
    for value, finding in zip(form.values, report.values):
        decision = decisions[value.evidence.node_id]
        if (
            decision.grounding_digest != report.grounding_digest
            or not decision.reviewer_confirmed
        ):
            raise FormGroundingError(
                "form export requires current explicit association review"
            )
        link = next(
            (
                item
                for item in finding.associations
                if item.association_id == decision.association_id
            ),
            None,
        )
        if link is None or link.label_id in selected_labels:
            raise FormGroundingError(
                "form association choice is unavailable or conflicting"
            )
        selected_labels.add(link.label_id)
        transcription: ReviewedHandwriting | None = None
        payload_value: str | bool | None = value.text
        if value.handwriting is not None:
            handwriting_review = transcriptions[value.handwriting.candidate_id]
            if decision.handwriting_review_digest != handwriting_review.digest:
                raise FormGroundingError(
                    "form association review does not bind this handwriting decision"
                )
            transcription = _handwriting(value.handwriting, handwriting_review)
            payload_value = transcription.text
        else:
            _require(decision.handwriting_review_digest is None)
            if value.kind in (ValueKind.CHECKBOX, ValueKind.RADIO):
                if value.control_state is ControlState.UNKNOWN:
                    raise FormGroundingError("form control state requires verification")
                payload_value = value.control_state is ControlState.CHECKED
        label = labels[link.label_id]
        section = sections[link.section_id]
        audit = {
            "association": asdict(link),
            "review": asdict(decision),
            "value_kind": value.kind.value,
            "state": "reviewed_association",
            "evidence": [
                asdict(item)
                for item in (section.evidence, label.evidence, value.evidence)
            ],
            "handwriting_review": transcription.to_report()
            if transcription is not None
            else None,
        }
        audit_rows.append(audit)
        protected_rows.append({**audit, "label": label.text, "value": payload_value})
    common = {
        "schema_version": GROUNDING_VERSION,
        "grounding_digest": report.grounding_digest,
        "notice": GROUNDING_NOTICE,
    }
    return GroundedFormExport(
        json.dumps(
            {**common, "fields": protected_rows},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        json.dumps(
            {**common, "fields": audit_rows},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
    )
