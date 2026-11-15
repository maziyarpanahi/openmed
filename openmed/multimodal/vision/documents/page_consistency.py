"""Bounded, value-free consistency assessment for locally processed page packets.

Callers supply qualified local OCR spans and rendered RGB pixels. Consistency is
not patient identity certification, PHI detection, or clinical verification.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, TypeVar

from ...evidence.fused_graph import EvidenceNode, NodeKind, PageGeometry, SpatialUnit
from ...pdf_geometry import PdfGeometryStatus, read_pdf_geometry
from ...privacy.mask_lineage import (
    MAX_MASKS,
    MAX_TRANSFORMS,
    CoordinateSpace,
    MaskLineageResult,
    TransformStep,
    propagate_masks,
)

SCHEMA_VERSION = "openmed.multimodal.page-consistency.v1"
MAX_PAGES = 256
MAX_OBSERVATIONS = 32
MAX_TEXT_LENGTH = 2_000_000
MAX_PAGE_PIXELS = 4_000_000
T = TypeVar("T")

__all__ = [
    "ConsistencyCode",
    "ConsistencyPolicy",
    "DigestPurpose",
    "GroupBoundary",
    "IdentityKind",
    "IdentityObservation",
    "PacketConsistency",
    "PageConsistencyError",
    "PageFinding",
    "PageGroup",
    "PageNumber",
    "PageObservation",
    "TextDigest",
    "anchor_from_span",
    "assess_page_consistency",
    "assess_pdf_page_consistency",
    "capture_page",
    "continuation_from_span",
    "identity_from_span",
    "numbering_from_span",
    "pdf_page_sources",
    "require_consistent_group",
]


class PageConsistencyError(ValueError):
    """Fixed-message refusal without caller text, keys or exception context."""


class IdentityKind(str, Enum):
    """Typed identity observations; only PATIENT_ID can anchor automatic splits."""

    PATIENT_ID = "patient_id"
    NAME = "name"
    DATE_OF_BIRTH = "date_of_birth"


class DigestPurpose(str, Enum):
    """Prevent accidental reuse of an identity token as an anchor or continuation."""

    IDENTITY = "identity"
    ANCHOR = "anchor"
    CONTINUATION = "continuation"


class GroupBoundary(str, Enum):
    """Value-free reason for retaining a distinct candidate page group."""

    PACKET_START = "packet_start"
    IDENTITY_CHANGE = "identity_change"
    IDENTITY_UNCERTAIN = "identity_uncertain"


class ConsistencyCode(str, Enum):
    """Closed finding vocabulary; missing evidence never implies consistency."""

    IDENTITY_UNKNOWN = "identity_unknown"
    IDENTITY_CONFLICT = "identity_conflict"
    KEY_SCOPE_MISMATCH = "key_scope_mismatch"
    DUPLICATE_PAGE = "duplicate_page"
    NONCONTIGUOUS_IDENTITY = "noncontiguous_identity"
    NUMBER_UNKNOWN = "page_number_unknown"
    NUMBER_SEQUENCE = "page_number_sequence"
    NUMBER_TOTAL = "page_total_conflict"
    INCOMPLETE_GROUP = "incomplete_page_group"
    LAYOUT_UNKNOWN = "layout_unknown"
    LAYOUT_CHANGED = "layout_changed"
    CONTINUATION_MISSING = "continuation_endpoint_missing"
    CONTINUATION_CONFLICT = "continuation_conflict"
    CONTINUATION_CROSSES_IDENTITY = "continuation_crosses_identity"
    MASK_REJECTED = "mask_lineage_rejected"


def _require(condition: bool) -> None:
    if not condition:
        raise PageConsistencyError("page_consistency_rejected")


def _safe(call: Callable[[], T]) -> T:
    failed = False
    result = None
    try:
        result = call()
    except Exception:
        failed = True
    if failed:
        raise PageConsistencyError("page_consistency_rejected")
    return result  # type: ignore[return-value]


def _hex(value: str, size: int = 64) -> None:
    _require(
        type(value) is str
        and len(value) == size
        and all(character in "0123456789abcdef" for character in value)
    )


def _integer(value: int, low: int, high: int) -> None:
    _require(type(value) is int and low <= value <= high)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _node(node: EvidenceNode, kind: NodeKind) -> None:
    _require(type(node) is EvidenceNode and node.kind is kind)
    node.__post_init__()
    _require(type(node.geometry) is PageGeometry)
    assert node.geometry is not None
    node.geometry.__post_init__()


@dataclass(frozen=True, slots=True)
class TextDigest:
    """Packet-keyed digest of an actual selected OCR span, with source coordinates."""

    purpose: DigestPurpose
    scope_digest: str
    value_digest: str
    evidence: EvidenceNode

    def __post_init__(self) -> None:
        _require(type(self.purpose) is DigestPurpose)
        _hex(self.scope_digest)
        _hex(self.value_digest)
        _node(self.evidence, NodeKind.OCR)


@dataclass(frozen=True, slots=True)
class IdentityObservation:
    """Typed identity digest in an opaque identifier-system namespace."""

    kind: IdentityKind
    namespace_ref: str
    token: TextDigest

    def __post_init__(self) -> None:
        _require(type(self.kind) is IdentityKind and type(self.token) is TextDigest)
        _hex(self.namespace_ref, 32)
        self.token.__post_init__()
        _require(self.token.purpose is DigestPurpose.IDENTITY)


@dataclass(frozen=True, slots=True)
class PageNumber:
    """Observed one-based page counter and total, with its selected OCR evidence."""

    current: int
    total: int
    evidence: EvidenceNode

    def __post_init__(self) -> None:
        _integer(self.total, 1, MAX_PAGES)
        _integer(self.current, 1, self.total)
        _node(self.evidence, NodeKind.OCR)


def _span(text: str, evidence: EvidenceNode) -> str:
    _require(type(text) is str and len(text) <= MAX_TEXT_LENGTH)
    _node(evidence, NodeKind.OCR)
    assert evidence.span is not None
    start, end = evidence.span
    _require(0 <= start < end <= len(text) and end - start <= 4096)
    value = unicodedata.normalize("NFC", text[start:end].strip())
    _require(bool(value))
    return value


def _token(text, evidence, key, purpose, domain=None):
    _require(type(key) is bytes and 32 <= len(key) <= 128)
    value = _span(text, evidence)
    scope = hmac.new(key, (SCHEMA_VERSION + ":scope").encode(), "sha256").hexdigest()
    digest = hmac.new(
        key, _json([SCHEMA_VERSION, purpose.value, domain, value]).encode(), "sha256"
    ).hexdigest()
    return TextDigest(purpose, scope, digest, evidence)


def identity_from_span(
    text: str,
    evidence: EvidenceNode,
    *,
    key: bytes,
    kind: IdentityKind,
    namespace_ref: str,
) -> IdentityObservation:
    """Hash an actual protected identity span without retaining its text or key.

    Args:
        text: Protected OCR text in which evidence.span is expressed.
        evidence: Qualified local OCR location, including source/page and region.
        key: Caller-managed random packet key; never persisted by this module.
        kind: Typed identity field selected by the caller's qualified detector.
        namespace_ref: Opaque identifier-system reference, never a patient value.

    NFC and outer-whitespace normalization preserve case and internal spacing.
    There is no automatic identity detection or fuzzy matching.
    """

    def capture():
        _require(type(kind) is IdentityKind)
        _hex(namespace_ref, 32)
        return IdentityObservation(
            kind,
            namespace_ref,
            _token(
                text, evidence, key, DigestPurpose.IDENTITY, [kind.value, namespace_ref]
            ),
        )

    return _safe(capture)


def anchor_from_span(text: str, evidence: EvidenceNode, *, key: bytes) -> TextDigest:
    """Capture a selected template label and its original page-space location.

    Args:
        text: Protected OCR text; no text is retained in the returned observation.
        evidence: Source-bound OCR span selected as a template anchor.
        key: Same caller-managed packet key used for identity observations.
    """
    return _safe(lambda: _token(text, evidence, key, DigestPurpose.ANCHOR))


def continuation_from_span(
    text: str, evidence: EvidenceNode, *, key: bytes
) -> TextDigest:
    """Capture a selected continuation key; equality is only a structural cue.

    Args:
        text: Protected OCR text containing the selected continuation key.
        evidence: Source-bound incoming or outgoing continuation span.
        key: Same caller-managed packet key as the rest of this packet.
    """
    return _safe(lambda: _token(text, evidence, key, DigestPurpose.CONTINUATION))


def numbering_from_span(text: str, evidence: EvidenceNode) -> PageNumber:
    """Read an explicit ``1/3`` or ``Page 1 of 3`` counter from its actual span.

    Args:
        text: Protected OCR text; arbitrary number-like substrings are not searched.
        evidence: Exactly selected counter span. Other formats need caller parsing
            into PageNumber with the same qualified source evidence.
    """

    def capture():
        value = _span(text, evidence)
        match = re.fullmatch(
            r"(?:[Pp]age\s+)?([1-9][0-9]{0,3})\s*(?:/|of)\s*([1-9][0-9]{0,3})",
            value,
        )
        _require(match is not None)
        assert match is not None
        return PageNumber(int(match[1]), int(match[2]), evidence)

    return _safe(capture)


@dataclass(frozen=True, slots=True, kw_only=True)
class PageObservation:
    """Value-free page evidence; original page indexes and mask lineage stay intact."""

    source: EvidenceNode
    pixels_digest: str
    size: tuple[int, int]
    identities: tuple[IdentityObservation, ...] = ()
    anchors: tuple[TextDigest, ...] = ()
    numbering: PageNumber | None = None
    incoming: TextDigest | None = None
    outgoing: TextDigest | None = None
    mask_lineage: MaskLineageResult | None = None

    def __post_init__(self) -> None:
        _node(self.source, NodeKind.PAGE)
        assert self.source.geometry is not None
        _hex(self.pixels_digest)
        _require(type(self.size) is tuple and len(self.size) == 2)
        for dimension in self.size:
            _integer(dimension, 1, 65535)
        _require(math.prod(self.size) <= MAX_PAGE_PIXELS)
        for sequence, item_type in (
            (self.identities, IdentityObservation),
            (self.anchors, TextDigest),
        ):
            _require(type(sequence) is tuple and len(sequence) <= MAX_OBSERVATIONS)
            for item in sequence:
                _require(type(item) is item_type)
                item.__post_init__()
        _require(self.numbering is None or type(self.numbering) is PageNumber)
        if self.numbering is not None:
            self.numbering.__post_init__()
        for token in (self.incoming, self.outgoing):
            _require(token is None or type(token) is TextDigest)
            if token is not None:
                token.__post_init__()
                _require(token.purpose is DigestPurpose.CONTINUATION)
        _require(all(token.purpose is DigestPurpose.ANCHOR for token in self.anchors))
        for evidence in self.observations:
            _require(
                evidence.source_digest == self.source.source_digest
                and evidence.geometry == self.source.geometry
            )
        _require(
            len({node.node_id for node in self.observations}) == len(self.observations)
        )
        if self.mask_lineage is not None:
            lineage = self.mask_lineage
            _require(type(lineage) is MaskLineageResult)
            _require(
                type(lineage.steps) is tuple and len(lineage.steps) <= MAX_TRANSFORMS
            )
            _require(
                type(lineage.source_nodes) is tuple
                and 0 < len(lineage.source_nodes) <= MAX_MASKS
            )
            spaces = [lineage.source, lineage.target]
            for step in lineage.steps:
                _require(type(step) is TransformStep)
                step.__post_init__()
                step.affine.__post_init__()
                for rectangle in step.coverage or ():
                    rectangle.__post_init__()
                spaces.extend((step.source, step.target))
            for space in spaces:
                _require(type(space) is CoordinateSpace)
                _require(type(space.geometry) is PageGeometry)
                space.geometry.__post_init__()
                space.__post_init__()
            for node in lineage.source_nodes:
                _require(
                    type(node) is EvidenceNode
                    and node.kind in (NodeKind.PIXELS, NodeKind.OCR)
                )
                _node(node, node.kind)
            actual = propagate_masks(
                lineage.source, lineage.source_nodes, lineage.steps
            )
            _require(actual == lineage)
            _require(
                lineage.source.source_digest == self.source.source_digest
                and lineage.source.geometry == self.source.geometry
            )
            if lineage.accepted:
                _require(
                    lineage.target.geometry.page == self.source.geometry.page
                    and lineage.target.geometry.unit is SpatialUnit.PIXEL
                    and (lineage.target.geometry.width, lineage.target.geometry.height)
                    == self.size
                )

    @property
    def observations(self) -> tuple[EvidenceNode, ...]:
        """Original OCR references supporting the page observations, without values."""
        return (
            tuple(identity.token.evidence for identity in self.identities)
            + tuple(anchor.evidence for anchor in self.anchors)
            + ((self.numbering.evidence,) if self.numbering else ())
            + ((self.incoming.evidence,) if self.incoming else ())
            + ((self.outgoing.evidence,) if self.outgoing else ())
        )

    @property
    def digest(self) -> str:
        """Bind all value-free observations, source geometry and preserved mask lineage."""
        self.__post_init__()
        payload = {
            "source": asdict(self.source),
            "pixels_digest": self.pixels_digest,
            "size": self.size,
            "identities": [asdict(identity) for identity in self.identities],
            "anchors": [asdict(anchor) for anchor in self.anchors],
            "numbering": asdict(self.numbering) if self.numbering else None,
            "incoming": asdict(self.incoming) if self.incoming else None,
            "outgoing": asdict(self.outgoing) if self.outgoing else None,
            "mask_lineage": json.loads(self.mask_lineage.to_json())
            if self.mask_lineage is not None
            else None,
        }
        return _hash([SCHEMA_VERSION, payload])


def capture_page(
    source: EvidenceNode,
    rgb_pixels: bytes,
    size: tuple[int, int],
    *,
    identities: tuple[IdentityObservation, ...] = (),
    anchors: tuple[TextDigest, ...] = (),
    numbering: PageNumber | None = None,
    incoming: TextDigest | None = None,
    outgoing: TextDigest | None = None,
    mask_lineage: MaskLineageResult | None = None,
) -> PageObservation:
    """Bind actual rendered RGB pixels and source-bound observations without storage.

    Args:
        source: Original PAGE evidence from the qualified intake/renderer.
        rgb_pixels: Actual row-major RGB8 pixels, exactly width*height*3 bytes.
            The caller's qualified local renderer owns decoding and source mapping.
        size: Rendered width and height; original geometry remains in source.
        identities: Selected, typed identity digests from this original page.
        anchors: Template labels and their original source-space locations.
        numbering: Observed page counter and total, or unknown.
        incoming: Selected incoming continuation key, when one exists.
        outgoing: Selected outgoing continuation key, when one exists.
        mask_lineage: Existing mask result; recomputed and retained without rewriting.

    Exact duplicate detection compares actual RGB content, independent of image
    container metadata. No pixel bytes, text or keys survive in the result.
    """

    def capture():
        _require(type(size) is tuple and len(size) == 2)
        for dimension in size:
            _integer(dimension, 1, 65535)
        _require(math.prod(size) <= MAX_PAGE_PIXELS)
        _require(type(rgb_pixels) is bytes and len(rgb_pixels) == math.prod(size) * 3)
        digest = hashlib.sha256(_json([SCHEMA_VERSION, "RGB8", size]).encode())
        digest.update(b"\x00")
        digest.update(rgb_pixels)
        return PageObservation(
            source=source,
            pixels_digest=digest.hexdigest(),
            size=size,
            identities=identities,
            anchors=anchors,
            numbering=numbering,
            incoming=incoming,
            outgoing=outgoing,
            mask_lineage=mask_lineage,
        )

    return _safe(capture)


@dataclass(frozen=True, slots=True, kw_only=True)
class ConsistencyPolicy:
    """Explicit identity system and normalized anchor-displacement tolerance.

    Args:
        identity_namespace: Opaque system in which PATIENT_ID anchors grouping.
        anchor_tolerance: Maximum per-coordinate normalized anchor displacement.
            This tolerance needs task qualification; it never relaxes identities.
        max_pages: Packet page-count ceiling; no partial assessment on exhaustion.
    """

    identity_namespace: str
    anchor_tolerance: float = 0.02
    max_pages: int = MAX_PAGES

    def __post_init__(self) -> None:
        _hex(self.identity_namespace, 32)
        _require(
            type(self.anchor_tolerance) in (int, float)
            and math.isfinite(self.anchor_tolerance)
            and 0 <= self.anchor_tolerance <= 0.1
        )
        _integer(self.max_pages, 1, MAX_PAGES)


@dataclass(frozen=True, slots=True)
class PageFinding:
    """Controlled reason attached to original page evidence references."""

    code: ConsistencyCode
    page_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        _require(type(self.code) is ConsistencyCode)
        _require(type(self.page_refs) is tuple and 0 < len(self.page_refs) <= MAX_PAGES)
        for reference in self.page_refs:
            _hex(reference, 32)
        _require(len(set(self.page_refs)) == len(self.page_refs))


@dataclass(frozen=True, slots=True)
class PageGroup:
    """Ordered candidate group; a group with any finding is quarantined."""

    packet_indexes: tuple[int, ...]
    page_refs: tuple[str, ...]
    boundary: GroupBoundary
    findings: tuple[PageFinding, ...]

    def __post_init__(self) -> None:
        _require(type(self.boundary) is GroupBoundary)
        _require(
            type(self.packet_indexes) is tuple
            and 0 < len(self.packet_indexes) <= MAX_PAGES
        )
        for index in self.packet_indexes:
            _integer(index, 0, MAX_PAGES - 1)
        _require(tuple(sorted(set(self.packet_indexes))) == self.packet_indexes)
        _require(
            type(self.page_refs) is tuple
            and len(self.page_refs) == len(self.packet_indexes)
        )
        for reference in self.page_refs:
            _hex(reference, 32)
        _require(len(set(self.page_refs)) == len(self.page_refs))
        _require(
            type(self.findings) is tuple and len(self.findings) <= len(ConsistencyCode)
        )
        for finding in self.findings:
            _require(type(finding) is PageFinding)
            finding.__post_init__()
            _require(set(finding.page_refs) <= set(self.page_refs))
        _require(len({finding.code for finding in self.findings}) == len(self.findings))

    @property
    def quarantined(self) -> bool:
        """Whether this candidate must be withheld from whole-group extraction."""
        self.__post_init__()
        return bool(self.findings)


@dataclass(frozen=True, slots=True)
class PacketConsistency:
    """Deterministic candidate split plan, never permission to disclose clinical data."""

    packet_digest: str
    groups: tuple[PageGroup, ...]

    def __post_init__(self) -> None:
        _hex(self.packet_digest)
        _require(type(self.groups) is tuple and 0 < len(self.groups) <= MAX_PAGES)
        for group in self.groups:
            _require(type(group) is PageGroup)
            group.__post_init__()
        indexes = tuple(
            index for group in self.groups for index in group.packet_indexes
        )
        references = tuple(
            reference for group in self.groups for reference in group.page_refs
        )
        _require(len(indexes) <= MAX_PAGES and indexes == tuple(range(len(indexes))))
        _require(len(set(references)) == len(references))
        _require(self.groups[0].boundary is GroupBoundary.PACKET_START)
        _require(
            all(
                group.boundary is not GroupBoundary.PACKET_START
                for group in self.groups[1:]
            )
        )

    @property
    def whole_packet_consistent(self) -> bool:
        """A single complete consistent group is required for whole-packet extraction."""
        self.__post_init__()
        return len(self.groups) == 1 and not self.groups[0].quarantined

    def to_json(self) -> str:
        """Return only digests, source references, indexes and controlled reasons."""
        self.__post_init__()
        return _json(
            {
                "schema_version": SCHEMA_VERSION,
                "packet_digest": self.packet_digest,
                "whole_packet_consistent": self.whole_packet_consistent,
                "groups": [
                    {**asdict(group), "quarantined": group.quarantined}
                    for group in self.groups
                ],
                "review_required": True,
            }
        )


def _identity_map(page):
    values = defaultdict(set)
    for observation in page.identities:
        values[(observation.kind, observation.namespace_ref)].add(
            (observation.token.scope_digest, observation.token.value_digest)
        )
    return values


def _primary(page, policy):
    values = _identity_map(page).get(
        (IdentityKind.PATIENT_ID, policy.identity_namespace), set()
    )
    return next(iter(values)) if len(values) == 1 else None


def _anchors(page):
    anchors = defaultdict(list)
    for token in page.anchors:
        node = token.evidence
        x0, y0, x1, y1 = node.region
        anchors[(token.scope_digest, token.value_digest)].append(
            (
                x0 / node.geometry.width,
                y0 / node.geometry.height,
                x1 / node.geometry.width,
                y1 / node.geometry.height,
            )
        )
    return {key: sorted(boxes) for key, boxes in anchors.items()}


def _layout_matches(left, right, tolerance):
    if left.keys() != right.keys():
        return False
    for key, boxes in left.items():
        others = right[key]
        if len(boxes) != len(others):
            return False
        if any(
            abs(a - b) > tolerance
            for box, other in zip(boxes, others, strict=True)
            for a, b in zip(box, other, strict=True)
        ):
            return False
    return True


def assess_page_consistency(
    pages: tuple[PageObservation, ...], *, policy: ConsistencyPolicy
) -> PacketConsistency:
    """Compare the entire packet and retain conservative ordered split candidates.

    Args:
        pages: Captured observations in packet order. Original indexes are immutable.
        policy: Required patient-ID namespace, layout tolerance and resource ceiling.

    A primary identity change creates a candidate boundary. Missing identities
    never bridge groups. Conflicting secondary fields quarantine the full group.
    Duplicate content, noncontiguous identities, incomplete numbering, changed
    anchors, orphan continuation cues and rejected masks remain explicit findings.
    """

    def assess():
        _require(type(policy) is ConsistencyPolicy)
        policy.__post_init__()
        _require(type(pages) is tuple and 0 < len(pages) <= policy.max_pages)
        for page in pages:
            _require(type(page) is PageObservation)
            page.__post_init__()
        refs = tuple(page.source.node_id for page in pages)
        _require(len(set(refs)) == len(refs))
        originals = tuple(
            (page.source.source_digest, page.source.geometry.page) for page in pages
        )
        _require(len(set(originals)) == len(originals))
        findings = [set() for _ in pages]

        def flag(indexes, code):
            for index in indexes:
                findings[index].add(code)

        scopes = set()
        duplicates = defaultdict(list)
        primary = []
        for index, page in enumerate(pages):
            primary.append(_primary(page, policy))
            if primary[-1] is None:
                flag((index,), ConsistencyCode.IDENTITY_UNKNOWN)
            for values in _identity_map(page).values():
                if len(values) > 1:
                    flag((index,), ConsistencyCode.IDENTITY_CONFLICT)
            tokens = (
                tuple(identity.token for identity in page.identities) + page.anchors
            )
            tokens += tuple(token for token in (page.incoming, page.outgoing) if token)
            scopes.update(token.scope_digest for token in tokens)
            duplicates[page.pixels_digest].append(index)
            if not page.anchors:
                flag((index,), ConsistencyCode.LAYOUT_UNKNOWN)
            if page.mask_lineage is not None and not page.mask_lineage.accepted:
                flag((index,), ConsistencyCode.MASK_REJECTED)
        if len(scopes) != 1:
            flag(range(len(pages)), ConsistencyCode.KEY_SCOPE_MISMATCH)
        for matches in duplicates.values():
            if len(matches) > 1:
                flag(matches, ConsistencyCode.DUPLICATE_PAGE)

        groups = []
        for index, key in enumerate(primary):
            if index == 0 or key is None or key != primary[index - 1]:
                groups.append([index])
            else:
                groups[-1].append(index)
        seen = defaultdict(list)
        for indexes in groups:
            if primary[indexes[0]] is not None:
                seen[primary[indexes[0]]].append(indexes)
        for repeated in seen.values():
            if len(repeated) > 1:
                flag(
                    (index for group in repeated for index in group),
                    ConsistencyCode.NONCONTIGUOUS_IDENTITY,
                )

        for indexes in groups:
            identities = defaultdict(set)
            for index in indexes:
                for field, values in _identity_map(pages[index]).items():
                    identities[field].update(values)
            if any(len(values) > 1 for values in identities.values()):
                flag(indexes, ConsistencyCode.IDENTITY_CONFLICT)
            counters = [pages[index].numbering for index in indexes]
            if any(counter is None for counter in counters):
                flag(indexes, ConsistencyCode.NUMBER_UNKNOWN)
            else:
                totals = {counter.total for counter in counters}
                if len(totals) != 1:
                    flag(indexes, ConsistencyCode.NUMBER_TOTAL)
                numbers = [counter.current for counter in counters]
                if any(b != a + 1 for a, b in zip(numbers, numbers[1:])):
                    flag(indexes, ConsistencyCode.NUMBER_SEQUENCE)
                if (
                    numbers[0] != 1
                    or any(counter.total != len(indexes) for counter in counters)
                    or numbers[-1] != len(indexes)
                ):
                    flag(indexes, ConsistencyCode.INCOMPLETE_GROUP)
            layouts = [_anchors(pages[index]) for index in indexes]
            # Every pair must agree: comparing only neighbours permits accumulated drift.
            if any(
                not _layout_matches(left, right, policy.anchor_tolerance)
                for offset, left in enumerate(layouts)
                for right in layouts[offset + 1 :]
            ):
                flag(indexes, ConsistencyCode.LAYOUT_CHANGED)

        if pages[0].incoming is not None:
            flag((0,), ConsistencyCode.CONTINUATION_MISSING)
        if pages[-1].outgoing is not None:
            flag((len(pages) - 1,), ConsistencyCode.CONTINUATION_MISSING)
        for index, (left, right) in enumerate(zip(pages, pages[1:])):
            outgoing, incoming = left.outgoing, right.incoming
            if outgoing is None and incoming is None:
                continue
            pair = (index, index + 1)
            if primary[index] is None or primary[index] != primary[index + 1]:
                flag(pair, ConsistencyCode.CONTINUATION_CROSSES_IDENTITY)
            if outgoing is None or incoming is None:
                flag(pair, ConsistencyCode.CONTINUATION_MISSING)
            elif (outgoing.scope_digest, outgoing.value_digest) != (
                incoming.scope_digest,
                incoming.value_digest,
            ):
                flag(pair, ConsistencyCode.CONTINUATION_CONFLICT)

        results = []
        for indexes in groups:
            reasons = tuple(
                PageFinding(
                    code,
                    tuple(refs[index] for index in indexes if code in findings[index]),
                )
                for code in ConsistencyCode
                if any(code in findings[index] for index in indexes)
            )
            first = indexes[0]
            boundary = (
                GroupBoundary.PACKET_START
                if first == 0
                else GroupBoundary.IDENTITY_UNCERTAIN
                if primary[first] is None or primary[first - 1] is None
                else GroupBoundary.IDENTITY_CHANGE
            )
            results.append(
                PageGroup(
                    tuple(indexes),
                    tuple(refs[index] for index in indexes),
                    boundary,
                    reasons,
                )
            )
        return PacketConsistency(
            _hash([SCHEMA_VERSION, asdict(policy), [page.digest for page in pages]]),
            tuple(results),
        )

    return _safe(assess)


def require_consistent_group(
    pages: tuple[PageObservation, ...], group_index: int, *, policy: ConsistencyPolicy
) -> tuple[PageObservation, ...]:
    """Reassess original observations before returning one non-quarantined group.

    Args:
        pages: Original ordered packet observations, not a caller-edited report.
        group_index: Explicit candidate chosen for further human-reviewed processing.
        policy: Current policy; a previous report cannot override it.
    """

    def require():
        report = assess_page_consistency(pages, policy=policy)
        _integer(group_index, 0, len(report.groups) - 1)
        group = report.groups[group_index]
        _require(not group.quarantined)
        return tuple(pages[index] for index in group.packet_indexes)

    return _safe(require)


def pdf_page_sources(
    source: bytes, *, max_pages: int = MAX_PAGES
) -> tuple[EvidenceNode, ...]:
    """Read actual bounded PDF geometry and return deterministic original PAGE refs.

    Args:
        source: Protected original PDF bytes, at most 64 MiB; never a URL or path.
        max_pages: Complete-packet ceiling. No partial geometry is accepted.

    Geometry uses the preflight's displayed crop dimensions in points. Renderer,
    OCR and mask coordinates must explicitly use that same declared source frame.
    No PDF text, images, metadata or content streams are extracted by this helper.
    """

    def read():
        _require(type(source) is bytes and 0 < len(source) <= 64 * 1024 * 1024)
        _integer(max_pages, 1, MAX_PAGES)
        geometry = read_pdf_geometry(source, max_pages=max_pages)
        _require(geometry.status is PdfGeometryStatus.READABLE and bool(geometry.pages))
        digest = hashlib.sha256(source).hexdigest()
        return tuple(
            EvidenceNode(
                node_id=_hash([SCHEMA_VERSION, digest, page.page_index])[:32],
                kind=NodeKind.PAGE,
                source_digest=digest,
                geometry=PageGeometry(
                    page=page.page_index,
                    width=page.width,
                    height=page.height,
                    unit=SpatialUnit.POINT,
                ),
            )
            for page in geometry.pages
        )

    return _safe(read)


def assess_pdf_page_consistency(
    source: bytes, pages: tuple[PageObservation, ...], *, policy: ConsistencyPolicy
) -> PacketConsistency:
    """Require exact PDF page-tree coverage before assessing its captured observations.

    Args:
        source: Exact protected PDF bytes used by the qualified local renderer/OCR.
        pages: One captured observation per original page, in page-tree order.
        policy: Current identity system, layout tolerance and complete-packet limit.

    This verifies source/page geometry and completeness, not OCR correctness or
    that a caller's RGB buffer was rendered faithfully from the PDF.
    """

    def assess():
        _require(type(policy) is ConsistencyPolicy and type(pages) is tuple)
        policy.__post_init__()
        sources = pdf_page_sources(source, max_pages=policy.max_pages)
        _require(tuple(page.source for page in pages) == sources)
        return assess_page_consistency(pages, policy=policy)

    return _safe(assess)
