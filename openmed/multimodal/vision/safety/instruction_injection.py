"""Bounded instruction screening and quarantine for localized document text."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from enum import Enum

from openmed.agent.security.injection_guard import scan_text
from openmed.multimodal.evidence import EvidenceNode, NodeKind, PageGeometry

SCREENING_VERSION = "openmed.visual-instruction-screening.v1"
_MAX_REGIONS = 256
_MAX_REGION_TEXT = 16_384
_MAX_TOTAL_TEXT = 65_536
_MAX_CUES = 1024
_OPAQUE = re.compile(r"[0-9a-f]{32}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_FORM = re.compile(
    r"\A\s*(?:please\s+)?(?:complete|fill\s+in|tick|check|sign|initial|circle|enter|write)\b"
)
_WORKFLOW = re.compile(
    r"\A\s*(?:please\s+)?(?:send|fax|email|upload|forward|schedule|book|refer|route|submit)\b"
)
_EXTRA_CUES = (
    (
        "model_role_delimiter",
        re.compile(
            r"<\|(?:im_start|im_end|start_header_id|end_header_id|eot_id|system|developer|assistant|user)\|>"
        ),
    ),
    (
        "model_role_heading",
        re.compile(r"(?m)^[ \t]*(?:system|developer|assistant)\s*:"),
    ),
    (
        "model_targeted_directive",
        re.compile(
            r"\b(?:assistant|language\s+model|llm|chatbot)\s*[:,]\s*(?:ignore|override|bypass|reveal|output|return|execute|call|change|replace|send)\b"
        ),
    ),
)


class InstructionScreeningError(ValueError):
    """Controlled failure that never contains document text or source paths."""


class RegionClass(str, Enum):
    """Lexical source classification, never a grant of instruction authority."""

    SOURCE_CONTENT = "source_content"
    FORM_INSTRUCTION = "form_instruction"
    WORKFLOW_DIRECTIVE = "workflow_directive"
    SUSPICIOUS = "suspicious_model_instruction"
    UNASSESSED = "unassessed"


class ReleaseAuthority(str, Enum):
    """Caller-recorded approval authority; authentication remains external."""

    POLICY = "policy"
    REVIEWER = "reviewer"


def _require(condition: bool) -> None:
    if not condition:
        raise InstructionScreeningError("invalid document screening declaration")


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _normalized(text: str) -> tuple[str, tuple[int, ...]]:
    output: list[str] = []
    origins: list[int] = []
    for index, character in enumerate(text):
        if unicodedata.category(character) == "Cf":
            continue
        for folded in unicodedata.normalize("NFKC", character).casefold():
            output.append(" " if folded.isspace() and folded != "\n" else folded)
            origins.append(index)
    return "".join(output), tuple(origins)


@dataclass(frozen=True, slots=True, kw_only=True)
class DocumentRegion:
    """Protected recognized text linked to an original OCR span or visual box.

    Args:
        evidence: OCR or PIXELS node in original source coordinates.
        text: Recognized text, or None when a visual region is unassessed.
        recognition_complete: Explicit assertion that recognition covered the region.
        language: Declared language tag; only English lexical rules are supported.

    OCR text length must match the cited Unicode span. PIXELS text is localized
    recognition, not a model instruction. No pixels are decoded or retained here.
    """

    evidence: EvidenceNode
    text: str | None = field(repr=False)
    recognition_complete: bool = False
    language: str = "und"

    def __post_init__(self) -> None:
        _require(
            type(self.evidence) is EvidenceNode
            and self.evidence.kind in (NodeKind.OCR, NodeKind.PIXELS)
        )
        _require(type(self.recognition_complete) is bool)
        _require(
            type(self.language) is str
            and re.fullmatch(r"[a-z]{2,3}(?:-[A-Za-z0-9]{1,8}){0,3}", self.language)
            is not None
        )
        if self.text is None:
            _require(not self.recognition_complete)
        else:
            _require(type(self.text) is str and 0 < len(self.text) <= _MAX_REGION_TEXT)
            if self.evidence.kind is NodeKind.OCR:
                assert self.evidence.span is not None
                _require(
                    len(self.text) == self.evidence.span[1] - self.evidence.span[0]
                )


@dataclass(frozen=True, slots=True, kw_only=True)
class InstructionPolicy:
    """Explicit inclusion policy for inert form/workflow text, never model commands.

    Args:
        version: Short opaque policy identifier without patient information.
        allow_form_instructions: Include recognized form instructions as source data.
        allow_workflow_directives: Include workflow directives as source data.

    Suspicious and unassessed regions always require an exact-content release.
    Incomplete recognition cannot be released through this text-only gateway.
    """

    version: str
    allow_form_instructions: bool = True
    allow_workflow_directives: bool = False

    def __post_init__(self) -> None:
        _require(
            type(self.version) is str
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", self.version)
            is not None
        )
        _require(
            type(self.allow_form_instructions) is bool
            and type(self.allow_workflow_directives) is bool
        )

    @property
    def digest(self) -> str:
        """Bind the policy and detector version for review and replay."""
        return _hash({"version": SCREENING_VERSION, "policy": asdict(self)})


@dataclass(frozen=True, slots=True)
class InstructionCue:
    """Controlled pattern code and local Unicode offsets, without matched text."""

    code: str
    start: int
    end: int


def _cues(text: str) -> tuple[InstructionCue, ...]:
    found = {
        InstructionCue(item.pattern_id, item.start, item.end)
        for item in scan_text(text).findings
    }
    normalized, origins = _normalized(text)
    for code, expression in _EXTRA_CUES:
        for match in expression.finditer(normalized):
            found.add(
                InstructionCue(
                    code, origins[match.start()], origins[match.end() - 1] + 1
                )
            )
    if len(found) > _MAX_CUES:
        raise InstructionScreeningError("document screening resource limit")
    return tuple(sorted(found, key=lambda item: (item.start, item.end, item.code)))


@dataclass(frozen=True, slots=True)
class RegionAssessment:
    """Content-free classification, original evidence and quarantine reasons."""

    evidence: EvidenceNode
    classification: RegionClass
    quarantined: bool
    cues: tuple[InstructionCue, ...]
    reasons: tuple[str, ...]
    related_region_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DocumentScreeningReport:
    """Evidence-only result, bound to every protected input and the policy."""

    screening_digest: str
    policy_digest: str
    regions: tuple[RegionAssessment, ...]

    def to_json(self) -> str:
        """Serialize controlled findings and source coordinates without raw text."""
        return json.dumps(
            {**asdict(self), "schema_version": SCREENING_VERSION},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


def _validate_regions(regions: tuple[DocumentRegion, ...]) -> None:
    _require(type(regions) is tuple and 1 <= len(regions) <= _MAX_REGIONS)
    _require(all(type(region) is DocumentRegion for region in regions))
    _require(len({region.evidence.node_id for region in regions}) == len(regions))
    _require(
        sum(len(region.text or "") for region in regions) + len(regions) - 1
        <= _MAX_TOTAL_TEXT
    )
    pages: dict[tuple[str | None, int], PageGeometry] = {}
    for region in regions:
        node = region.evidence
        assert node.geometry is not None
        key = (node.source_digest, node.geometry.page)
        if key in pages:
            _require(pages[key] == node.geometry)
        pages[key] = node.geometry


def _joined(
    regions: tuple[DocumentRegion, ...],
) -> tuple[str, tuple[tuple[int, int], ...]]:
    ranges = []
    cursor = 0
    for region in regions:
        end = cursor + len(region.text or "")
        ranges.append((cursor, end))
        cursor = end + 1
    return "\n".join(region.text or "" for region in regions), tuple(ranges)


def _classification(
    region: DocumentRegion, cues: tuple[InstructionCue, ...]
) -> RegionClass:
    if cues:
        return RegionClass.SUSPICIOUS
    if (
        not region.recognition_complete
        or region.text is None
        or region.language.split("-")[0] != "en"
    ):
        return RegionClass.UNASSESSED
    normalized, _ = _normalized(region.text)
    if _WORKFLOW.search(normalized):
        return RegionClass.WORKFLOW_DIRECTIVE
    if _FORM.search(normalized):
        return RegionClass.FORM_INSTRUCTION
    return RegionClass.SOURCE_CONTENT


def _overlaps(first: EvidenceNode, second: EvidenceNode) -> bool:
    assert first.geometry is not None and second.geometry is not None
    if (
        first.source_digest != second.source_digest
        or first.geometry.page != second.geometry.page
    ):
        return False
    assert first.region is not None and second.region is not None
    a, b = first.region, second.region
    spatial = max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])
    # Alternate OCR boxes must not bypass a quarantined shared text interval.
    spans = (
        first.span is not None
        and second.span is not None
        and max(first.span[0], second.span[0]) < min(first.span[1], second.span[1])
    )
    return spatial or spans


def screen_document_regions(
    regions: tuple[DocumentRegion, ...],
    policy: InstructionPolicy,
) -> DocumentScreeningReport:
    """Classify localized text and quarantine direct, split and overlapping cues.

    Args:
        regions: Original evidence in the intended prompt reading order.
        policy: Explicit version and permitted inert source categories.

    Returns:
        A deterministic content-free report. Source classification is absence of
        known lexical cues, not proof of safety, clinical truth or OCR completeness.
    """
    _validate_regions(regions)
    _require(type(policy) is InstructionPolicy)
    cues = [set(_cues(region.text or "")) for region in regions]
    related: list[set[str]] = [set() for _ in regions]
    joined, ranges = _joined(regions)
    for cue in _cues(joined):
        involved = [
            i
            for i, (start, end) in enumerate(ranges)
            if max(start, cue.start) < min(end, cue.end)
        ]
        for i in involved:
            start, end = ranges[i]
            cues[i].add(
                InstructionCue(
                    cue.code, max(start, cue.start) - start, min(end, cue.end) - start
                )
            )
            related[i].update(regions[j].evidence.node_id for j in involved if i != j)
    if sum(len(items) for items in cues) > _MAX_CUES:
        raise InstructionScreeningError("document screening resource limit")
    classes = [
        _classification(region, tuple(items)) for region, items in zip(regions, cues)
    ]
    quarantined = [
        kind in (RegionClass.SUSPICIOUS, RegionClass.UNASSESSED)
        or (kind is RegionClass.FORM_INSTRUCTION and not policy.allow_form_instructions)
        or (
            kind is RegionClass.WORKFLOW_DIRECTIVE
            and not policy.allow_workflow_directives
        )
        for kind in classes
    ]
    reason_codes = {
        RegionClass.SUSPICIOUS: "model_instruction_cue",
        RegionClass.UNASSESSED: "recognition_unverified",
        RegionClass.FORM_INSTRUCTION: "form_policy_hold",
        RegionClass.WORKFLOW_DIRECTIVE: "workflow_policy_hold",
    }
    reasons = [
        [reason_codes[kind]] if hold else [] for kind, hold in zip(classes, quarantined)
    ]
    # Connected overlap components inherit holds, including alternate renderings.
    pending = [i for i, held in enumerate(quarantined) if held]
    visited = set(pending)
    while pending:
        i = pending.pop()
        for j, region in enumerate(regions):
            if i == j or not _overlaps(regions[i].evidence, region.evidence):
                continue
            related[j].add(regions[i].evidence.node_id)
            if j not in visited:
                visited.add(j)
                quarantined[j] = True
                reasons[j].append("overlaps_quarantine")
                pending.append(j)
    report_regions = tuple(
        RegionAssessment(
            region.evidence,
            classes[i],
            quarantined[i],
            tuple(sorted(cues[i], key=lambda item: (item.start, item.end, item.code))),
            tuple(reasons[i]),
            tuple(sorted(related[i])),
        )
        for i, region in enumerate(regions)
    )
    binding = _hash(
        {
            "schema_version": SCREENING_VERSION,
            "policy": policy.digest,
            "inputs": [asdict(region) for region in regions],
            "findings": [asdict(region) for region in report_regions],
        }
    )
    return DocumentScreeningReport(binding, policy.digest, report_regions)


@dataclass(frozen=True, slots=True, kw_only=True)
class RegionRelease:
    """Exact-content permission to include held regions as untrusted data.

    Args:
        screening_digest: Digest from a fresh complete-batch screening report.
        region_ids: Explicit quarantined region IDs covered by this decision.
        authority: Application policy or human reviewer; no identity inferred.
        decision_digest: Digest of the externally retained approval evidence.
        approved: Must explicitly be True; construction is not authentication.
    """

    screening_digest: str
    region_ids: tuple[str, ...]
    authority: ReleaseAuthority
    decision_digest: str
    approved: bool = False

    def __post_init__(self) -> None:
        for digest in (self.screening_digest, self.decision_digest):
            _require(type(digest) is str and _DIGEST.fullmatch(digest) is not None)
        _require(
            type(self.region_ids) is tuple and 1 <= len(self.region_ids) <= _MAX_REGIONS
        )
        _require(
            all(
                type(item) is str and _OPAQUE.fullmatch(item) is not None
                for item in self.region_ids
            )
        )
        _require(len(set(self.region_ids)) == len(self.region_ids))
        _require(type(self.authority) is ReleaseAuthority)
        _require(self.approved is True)


@dataclass(frozen=True, slots=True)
class PreparedDocumentPrompt:
    """Protected JSON data plus audit metadata; never a system or tool message.

    data_json is intended for the protected model-input path, not logs or reports.
    It contains recognized text only. No source image or unscreened text is admitted.
    """

    data_json: str = field(repr=False)
    screening: DocumentScreeningReport
    admitted_region_ids: tuple[str, ...]
    released_region_ids: tuple[str, ...]
    quarantined_region_ids: tuple[str, ...]
    release_decisions: tuple[RegionRelease, ...]

    @property
    def decision_digests(self) -> tuple[str, ...]:
        """Return opaque references to the externally retained approval evidence."""
        return tuple(
            sorted(release.decision_digest for release in self.release_decisions)
        )

    def to_report_json(self) -> str:
        """Export decisions and original evidence without protected prompt content."""
        return json.dumps(
            {
                "screening": json.loads(self.screening.to_json()),
                "admitted_region_ids": self.admitted_region_ids,
                "released_region_ids": self.released_region_ids,
                "quarantined_region_ids": self.quarantined_region_ids,
                "decision_digests": self.decision_digests,
                "release_decisions": [
                    asdict(release) for release in self.release_decisions
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


def prepare_document_prompt(
    regions: tuple[DocumentRegion, ...],
    policy: InstructionPolicy,
    *,
    releases: tuple[RegionRelease, ...] = (),
) -> PreparedDocumentPrompt:
    """Build only admitted untrusted text, with fresh screening and exact releases.

    Args:
        regions: Protected recognized text and original evidence, in reading order.
        policy: Category inclusion policy; never promotes source text to instructions.
        releases: Explicit policy/reviewer decisions bound to the complete current scan.

    Returns:
        A structured data payload omitting entire quarantined regions. An empty
        regions array means all content was withheld; no placeholder attack text
        or original pixels are emitted.

    Raises:
        InstructionScreeningError: For stale/inapplicable releases, incomplete
            recognition, resource limits or a newly formed cue after exclusions.
    """
    report = screen_document_regions(regions, policy)
    _require(type(releases) is tuple and len(releases) <= _MAX_REGIONS)
    _require(all(type(release) is RegionRelease for release in releases))
    indexed = {
        region.evidence.node_id: (region, assessment)
        for region, assessment in zip(regions, report.regions)
    }
    released: set[str] = set()
    for release in releases:
        if release.screening_digest != report.screening_digest:
            raise InstructionScreeningError("document release is stale")
        for region_id in release.region_ids:
            _require(region_id in indexed and region_id not in released)
            region, assessment = indexed[region_id]
            _require(assessment.quarantined)
            if region.text is None or not region.recognition_complete:
                raise InstructionScreeningError(
                    "document recognition must be completed before release"
                )
            released.add(region_id)
    admitted = tuple(
        region
        for region, assessment in zip(regions, report.regions)
        if not assessment.quarantined or region.evidence.node_id in released
    )
    joined, ranges = _joined(admitted)
    for cue in _cues(joined):
        involved = [
            region.evidence.node_id
            for region, (start, end) in zip(admitted, ranges)
            if max(start, cue.start) < min(end, cue.end)
        ]
        if not set(involved) <= released:
            # Omitting a held middle region can create a new adjacent directive.
            # Require a new input/review transaction rather than emit partial data.
            raise InstructionScreeningError(
                "document prompt assembly requires additional review"
            )
    admitted_ids = tuple(region.evidence.node_id for region in admitted)
    data_json = json.dumps(
        {
            "schema_version": SCREENING_VERSION,
            "trust": "untrusted_document_content",
            "regions": [
                {
                    "region_id": region.evidence.node_id,
                    "classification": indexed[region.evidence.node_id][
                        1
                    ].classification.value,
                    "content": region.text,
                }
                for region in admitted
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return PreparedDocumentPrompt(
        data_json,
        report,
        admitted_ids,
        tuple(sorted(released)),
        tuple(
            region.evidence.node_id
            for region in regions
            if region.evidence.node_id not in admitted_ids
        ),
        releases,
    )
