"""Cited, screened ambient drafts with current local clinician-review admission."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Protocol, cast

from openmed.clinical.claim_conflicts import (
    AssertionRecord,
    ClaimRecord,
    ClaimReference,
    review_claim_conflicts,
)
from openmed.clinical.nli import NLIBackendLike, nli
from openmed.clinical.reviewed_local_evidence import (
    CurrentLocalSource,
    LocalReviewReceipt,
    ReviewAuthorityVerifier,
    ReviewedLocalEvidence,
    ReviewedLocalReference,
    admit_reviewed_local_evidence,
    reviewed_source_digest,
)

from ..privacy.withholding_buffer import (
    LocalPhiScreener,
    PhiScreenDeclaration,
    ReleasedSpeechSegment,
    WithholdingBuffer,
    WithholdingPolicy,
)
from ..speech.speaker_aliases import ClinicalRole, SpeakerAliases
from ..transcript.revision_ledger import MappedSpan
from .speaker_roles import (
    ReviewedRoleSpan,
    RoleAttributionReview,
    RoleBoundDraft,
    RoleObservation,
    TranscriptSource,
    extract_with_reviewed_roles,
    finalize_role_bound_note,
)

_SCHEMA = "openmed.ambient.note_evidence.v1"
_REF = re.compile(r"[0-9a-f]{32}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class NoteEvidenceError(ValueError):
    """Fixed-message refusal, without source text, note text or model errors."""


class NoteSectionKind(str, Enum):
    """Closed headings; arbitrary private labels cannot enter audit metadata."""

    HISTORY = "history"
    EXAMINATION = "examination"
    MEDICATIONS = "medications"
    ASSESSMENT = "assessment"
    PLAN = "plan"
    OTHER = "other"


class NoteAssertion(str, Enum):
    """Explicit assertion axes retained alongside generated text."""

    AFFIRMED = "affirmed"
    NEGATED = "negated"
    UNCERTAIN = "uncertain"
    HYPOTHETICAL = "hypothetical"
    UNKNOWN = "unknown"


class NoteUncertainty(str, Enum):
    """Extraction uncertainty is not a clinical confidence score."""

    NONE_DECLARED = "none_declared"
    QUALIFIED = "qualified"
    UNRESOLVED = "unresolved"


class ExtractionOperation(str, Enum):
    """Recorded transformation of cited screened evidence."""

    EXTRACT = "extract"
    PARAPHRASE = "paraphrase"
    SYNTHESIZE = "synthesize"


class SupportVerdict(str, Enum):
    """Only sufficiently supported statements may reach clinician finalization."""

    ENTAILED = "entailment"
    CONTRADICTED = "contradiction"
    NEUTRAL = "neutral"
    ABSTAINED = "abstention"


def _require(condition: bool) -> None:
    if not condition:
        raise NoteEvidenceError("invalid_ambient_note_contract")


def _ref(value: object) -> None:
    _require(type(value) is str and _REF.fullmatch(value) is not None)


def _digest(value: object) -> None:
    _require(type(value) is str and _DIGEST.fullmatch(value) is not None)


def _integer(value: object, maximum: int, minimum: int = 0) -> None:
    _require(type(value) is int and minimum <= value <= maximum)


def _text(value: object, maximum: int) -> None:
    valid = False
    try:
        valid = type(value) is str and 0 < len(value) <= maximum
        if valid:
            cast(str, value).encode("utf-8")
    except UnicodeError:
        valid = False
    _require(valid)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class NotePolicy:
    """Explicit assembly ceilings and the local verifier's support threshold."""

    max_sources: int = 16
    max_source_chars: int = 65_536
    max_statements: int = 64
    max_note_chars: int = 32_768
    min_support_ppm: int = 900_000
    screen_latency_ms: int = 60_000

    def __post_init__(self) -> None:
        _integer(self.max_sources, 32, 1)
        _integer(self.max_source_chars, 65_536, 1)
        _integer(self.max_statements, 64, 1)
        _integer(self.max_note_chars, 65_536, 1)
        _integer(self.min_support_ppm, 1_000_000, 1)
        _integer(self.screen_latency_ms, 60_000, 1)


@dataclass(frozen=True, slots=True, kw_only=True)
class NoteProviderDeclaration:
    """Pinned local model/algorithm and decoding policy, without a model name."""

    artifact_digest: str
    policy_digest: str
    requires_network: bool = False
    deterministic: bool = True

    def __post_init__(self) -> None:
        _digest(self.artifact_digest)
        _digest(self.policy_digest)
        _require(self.requires_network is False and self.deterministic is True)


@dataclass(frozen=True, slots=True)
class NoteOffsetMapping:
    """Screened observation offsets mapped to the original current transcript."""

    output_start: int
    output_end: int
    source_start: int
    source_end: int
    redacted: bool


@dataclass(frozen=True, slots=True)
class ScreenedRoleEvidence:
    """Model-visible screened text, reviewed speaker and original lineage."""

    evidence_ref: str
    alias: str
    role: ClinicalRole
    citation: MappedSpan
    mappings: tuple[NoteOffsetMapping, ...]
    text: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class NoteExtractionRequest:
    """Complete screened context; raw transcript text is never exposed here."""

    request_digest: str
    evidence: tuple[ScreenedRoleEvidence, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class NoteCitation:
    """Half-open offsets into one screened observation's text, not raw text."""

    evidence_ref: str
    start: int
    end: int

    def __post_init__(self) -> None:
        _ref(self.evidence_ref)
        _integer(self.start, 65_536)
        _integer(self.end, 65_536)
        _require(self.start < self.end)


@dataclass(frozen=True, slots=True)
class NoteExtractionStep:
    """One model-bound transformation and its complete cited input identities."""

    step_ref: str
    operation: ExtractionOperation
    input_refs: tuple[str, ...]
    artifact_digest: str

    def __post_init__(self) -> None:
        _ref(self.step_ref)
        _require(type(self.operation) is ExtractionOperation)
        _require(type(self.input_refs) is tuple and len(self.input_refs) <= 8)
        for value in self.input_refs:
            _ref(value)
        _require(len(set(self.input_refs)) == len(self.input_refs))
        _digest(self.artifact_digest)


@dataclass(frozen=True, slots=True, kw_only=True)
class NoteStatement:
    """Protected claim with citations, extraction lineage and explicit uncertainty.

    Concept, subject and temporal refs group claims for deterministic conflict
    checks; they are opaque semantic identities supplied by the local extractor.
    Missing scope remains review-blocked. A value digest optionally identifies a
    normalized value; unlike unit strings, its normalization is provider-owned.
    """

    statement_ref: str
    text: str = field(repr=False)
    citations: tuple[NoteCitation, ...]
    step_ref: str
    assertion: NoteAssertion
    uncertainty: NoteUncertainty
    concept_ref: str | None = None
    subject_ref: str | None = None
    temporal_ref: str | None = None
    value_digest: str | None = None

    def __post_init__(self) -> None:
        _ref(self.statement_ref)
        _ref(self.step_ref)
        _text(self.text, 4096)
        _require(type(self.citations) is tuple and len(self.citations) <= 8)
        _require(all(type(item) is NoteCitation for item in self.citations))
        _require(len(set(self.citations)) == len(self.citations))
        _require(type(self.assertion) is NoteAssertion)
        _require(type(self.uncertainty) is NoteUncertainty)
        for value in (self.concept_ref, self.subject_ref, self.temporal_ref):
            if value is not None:
                _ref(value)
        if self.value_digest is not None:
            _digest(self.value_digest)


@dataclass(frozen=True, slots=True)
class NoteSection:
    """One controlled heading with ordered protected statements."""

    kind: NoteSectionKind
    statements: tuple[NoteStatement, ...] = field(repr=False)

    def __post_init__(self) -> None:
        _require(type(self.kind) is NoteSectionKind)
        _require(type(self.statements) is tuple and 1 <= len(self.statements) <= 64)
        _require(all(type(item) is NoteStatement for item in self.statements))


@dataclass(frozen=True, slots=True)
class NoteProposal:
    """Untrusted local extractor response, bound to the full screened request."""

    request_digest: str
    sections: tuple[NoteSection, ...] = field(repr=False)
    steps: tuple[NoteExtractionStep, ...]

    def __post_init__(self) -> None:
        _digest(self.request_digest)
        _require(type(self.sections) is tuple and 1 <= len(self.sections) <= 6)
        _require(all(type(item) is NoteSection for item in self.sections))
        _require(len({item.kind for item in self.sections}) == len(self.sections))
        _require(type(self.steps) is tuple and 1 <= len(self.steps) <= 64)
        _require(all(type(item) is NoteExtractionStep for item in self.steps))
        _require(len({item.step_ref for item in self.steps}) == len(self.steps))


class LocalNoteExtractor(Protocol):
    """Trusted local extractor; no network, hidden context or undeclared copies."""

    @property
    def declaration(self) -> NoteProviderDeclaration:
        """Return pinned model and deterministic decoding identity."""
        ...

    def extract(self, request: NoteExtractionRequest) -> NoteProposal:
        """Create a proposal from the supplied screened evidence only."""
        ...


@dataclass(frozen=True, slots=True)
class StatementVerification:
    """One exact claim paired with its cited, screened premise for local NLI."""

    statement_ref: str
    premise: str = field(repr=False)
    hypothesis: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class NoteVerificationRequest:
    """Bound verification input; statements are protected even after screening."""

    request_digest: str
    statements: tuple[StatementVerification, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class StatementSupport:
    """Current evidence support, not clinical correctness or reviewer approval."""

    statement_ref: str
    verdict: SupportVerdict
    confidence_ppm: int

    def __post_init__(self) -> None:
        _ref(self.statement_ref)
        _require(type(self.verdict) is SupportVerdict)
        _integer(self.confidence_ppm, 1_000_000)


@dataclass(frozen=True, slots=True)
class NoteVerificationResult:
    """Complete ordered response bound to the exact verification request."""

    request_digest: str
    decisions: tuple[StatementSupport, ...]

    def __post_init__(self) -> None:
        _digest(self.request_digest)
        _require(type(self.decisions) is tuple and len(self.decisions) <= 64)
        _require(all(type(item) is StatementSupport for item in self.decisions))
        _require(
            len({item.statement_ref for item in self.decisions}) == len(self.decisions)
        )


class LocalNoteVerifier(Protocol):
    """Trusted independent local evidence verifier; not an approval authority."""

    @property
    def declaration(self) -> NoteProviderDeclaration:
        """Return pinned verification model and policy identity."""
        ...

    def verify(self, request: NoteVerificationRequest) -> NoteVerificationResult:
        """Evaluate every cited claim against its actual screened premise."""
        ...


class NliNoteVerifier:
    """Adapter for an explicitly supplied local OpenMed clinical NLI backend.

    No default registry alias or cloud fallback is selected. The host must
    qualify the chosen backend, pin its declaration and keep it local. The
    existing heuristic backend is useful for synthetic demos, not qualification.
    """

    def __init__(
        self, backend: NLIBackendLike, declaration: NoteProviderDeclaration
    ) -> None:
        _require(type(declaration) is NoteProviderDeclaration)
        _require(callable(backend) or callable(getattr(backend, "predict", None)))
        self._backend, self._declaration = backend, declaration

    @property
    def declaration(self) -> NoteProviderDeclaration:
        """Pinned verifier identity, independent of backend display names."""
        return self._declaration

    def verify(self, request: NoteVerificationRequest) -> NoteVerificationResult:
        """Call existing local NLI for each protected premise/hypothesis pair."""
        _require(type(request) is NoteVerificationRequest)
        _digest(request.request_digest)
        _require(type(request.statements) is tuple and len(request.statements) <= 64)
        for statement in request.statements:
            _require(type(statement) is StatementVerification)
            _ref(statement.statement_ref)
            _text(statement.premise, 66_560)
            _text(statement.hypothesis, 4096)
        results = []
        failed = False
        try:
            for statement in request.statements:
                result = nli(
                    statement.premise, statement.hypothesis, backend=self._backend
                )
                results.append(
                    StatementSupport(
                        statement.statement_ref,
                        SupportVerdict(result["label"]),
                        round(result["score"] * 1_000_000),
                    )
                )
        except Exception:
            failed = True
        if failed:
            raise NoteEvidenceError("ambient_note_verification_failed")
        return NoteVerificationResult(request.request_digest, tuple(results))


@dataclass(frozen=True, slots=True)
class AmbientNoteDraft:
    """Protected note and a separate value-free review packet; not an export permit."""

    text: str = field(repr=False)
    data_json: str = field(repr=False)
    review_packet: ReviewedLocalEvidence
    report_json: str

    def to_json(self) -> str:
        """Return audit metadata only; never include note or transcript text."""
        return self.report_json


@dataclass(frozen=True, slots=True)
class ReviewedAmbientNote:
    """Explicitly clinician-reviewed output; returned locally, never sent to an EHR."""

    text: str = field(repr=False)
    data_json: str = field(repr=False)
    report_json: str

    def to_json(self) -> str:
        """Return value-free current review and evidence metadata."""
        return self.report_json


def _project(
    span: ReviewedRoleSpan, release: ReleasedSpeechSegment
) -> ScreenedRoleEvidence:
    start, end = span.citation.start, span.citation.end
    _require(start is not None and end is not None)
    assert start is not None and end is not None
    pieces, mappings = [], []
    offset = 0
    for item in release.mappings:
        left, right = max(start, item.source_start), min(end, item.source_end)
        if left >= right:
            continue
        if item.redacted:
            text = release.text[item.output_start : item.output_end]
        else:
            out_start = item.output_start + left - item.source_start
            text = release.text[out_start : out_start + right - left]
        pieces.append(text)
        mappings.append(
            NoteOffsetMapping(offset, offset + len(text), left, right, item.redacted)
        )
        offset += len(text)
    _require(
        sum(item.source_end - item.source_start for item in mappings) == end - start
    )
    return ScreenedRoleEvidence(
        span.observation_id,
        span.alias,
        span.role,
        span.citation,
        tuple(mappings),
        "".join(pieces),
    )


def _screen(
    session_ref: str,
    rows: tuple[tuple[str, str], ...],
    screener: LocalPhiScreener,
    policy: NotePolicy,
    clock: Callable[[], float],
) -> tuple[ReleasedSpeechSegment, ...]:
    _require(1 <= len(rows) <= 64 and sum(len(text) for _, text in rows) <= 65_536)
    with WithholdingBuffer(
        session_ref,
        screener,
        policy=WithholdingPolicy(
            max_segments=64,
            max_chars=65_536,
            trailing_segments=0,
            max_latency_ms=policy.screen_latency_ms,
        ),
        clock=clock,
    ) as buffer:
        for ref, text in rows:
            _require(
                buffer.append(ref, text) and buffer.finalize(ref, expected_revision=0)
            )
        released = buffer.finish()
        _require(
            tuple(item.segment_ref for item in released)
            == tuple(ref for ref, _ in rows)
        )
        return released


def _evidence_record(item: ScreenedRoleEvidence) -> dict[str, Any]:
    return {
        "evidence_ref": item.evidence_ref,
        "alias": item.alias,
        "role": item.role,
        "citation": asdict(item.citation),
        "mappings": [asdict(mapping) for mapping in item.mappings],
        "text_digest": _hash(item.text),
    }


def _statements(proposal: NoteProposal) -> tuple[NoteStatement, ...]:
    return tuple(item for section in proposal.sections for item in section.statements)


def _render(proposal: NoteProposal) -> tuple[str, tuple[ReviewedLocalReference, ...]]:
    text = ""
    references = []
    for section in proposal.sections:
        if text:
            text += "\n\n"
        text += section.kind.value + "\n"
        for index, item in enumerate(section.statements):
            if index:
                text += "\n"
            start = len(text)
            text += item.text
            references.append(
                ReviewedLocalReference(
                    "ref:" + _hash(item.statement_ref), start, len(text)
                )
            )
    return text, tuple(references)


class AmbientNoteSession:
    """Compose one protected draft and freshly admit its clinician-reviewed export.

    Args:
        aliases, sources, observations, role_reviews: Complete current original
            transcript and reviewed role context. Original sources must already
            be finalized. The application must serialize access to this context.
        screener: Trusted local PHI screen; invoked on complete source context and
            generated note statements before any generated text is returned.
        extractor: Pinned local note extractor receiving only screened evidence.
        verifier: Independent local NLI/evidence verifier, never a reviewer.
        policy: Explicit resource and support limits.
        clock: Monotonic clock for each bounded PHI screening operation.

    This owns one draft, not a general revision store. Create a fresh session to
    correct a blocked draft; old review does not authorize revised statements.
    Provider declarations are contracts, not sandboxes or model qualification.
    """

    def __init__(
        self,
        aliases: SpeakerAliases,
        sources: tuple[TranscriptSource, ...],
        observations: tuple[RoleObservation, ...],
        role_reviews: tuple[RoleAttributionReview, ...],
        *,
        screener: LocalPhiScreener,
        extractor: LocalNoteExtractor,
        verifier: LocalNoteVerifier,
        policy: NotePolicy = NotePolicy(),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        _require(type(policy) is NotePolicy and callable(clock))
        _require(type(sources) is tuple and 1 <= len(sources) <= policy.max_sources)
        _require(all(type(item) is TranscriptSource for item in sources))
        _require(sum(len(item.text) for item in sources) <= policy.max_source_chars)
        _require(type(observations) is tuple and 1 <= len(observations) <= 64)
        self._aliases, self._sources = aliases, sources
        self._observations, self._reviews = observations, role_reviews
        self._screener, self._extractor, self._verifier = screener, extractor, verifier
        self._policy, self._clock = policy, clock
        self._declarations = self._read_declarations()
        self._closed = self._busy = False
        self._draft: AmbientNoteDraft | None = None
        self._proposal: NoteProposal | None = None
        self._role_draft: RoleBoundDraft | None = None
        self._evaluation_digest: str | None = None

    def __repr__(self) -> str:
        return "AmbientNoteSession(protected_context)"

    def __enter__(self) -> AmbientNoteSession:
        _require(not self._closed)
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """Release owned references; Python string copies are not securely erased."""
        _require(not self._busy)
        self._closed = True
        self._sources, self._observations, self._reviews = (), (), ()
        self._draft = self._proposal = self._role_draft = None

    def _read_declarations(
        self,
    ) -> tuple[PhiScreenDeclaration, NoteProviderDeclaration, NoteProviderDeclaration]:
        declarations = None
        try:
            screen = self._screener.declaration
            extractor = self._extractor.declaration
            verifier = self._verifier.declaration
            _require(type(screen) is PhiScreenDeclaration)
            _require(
                type(extractor) is NoteProviderDeclaration
                and type(verifier) is NoteProviderDeclaration
            )
            screen.__post_init__()
            extractor.__post_init__()
            verifier.__post_init__()
            declarations = (screen, extractor, verifier)
        except Exception:
            pass
        if declarations is None:
            raise NoteEvidenceError("ambient_note_provider_unavailable")
        return declarations

    def _current_declarations(self) -> None:
        _require(self._read_declarations() == self._declarations)

    def _revalidate_roles(self, spans: tuple[ReviewedRoleSpan, ...]) -> None:
        def compare(current: tuple[ReviewedRoleSpan, ...]) -> str:
            _require(current == spans)
            return "current"

        extract_with_reviewed_roles(
            self._aliases, self._sources, self._observations, self._reviews, compare
        )

    def _screen_sources(
        self, spans: tuple[ReviewedRoleSpan, ...]
    ) -> tuple[ScreenedRoleEvidence, ...]:
        self._current_declarations()
        _require(all(item.ledger.to_dict()["finalized"] for item in self._sources))
        session_ref = spans[0].citation.reference.session_ref
        rows = tuple(
            sorted(
                (item.ledger.to_dict()["segment_ref"], item.text)
                for item in self._sources
            )
        )
        released = _screen(session_ref, rows, self._screener, self._policy, self._clock)
        by_id = {item.segment_ref: item for item in released}
        result = tuple(
            _project(span, by_id[span.citation.reference.segment_ref]) for span in spans
        )
        self._current_declarations()
        return result

    def _validate_proposal(
        self, proposal: NoteProposal, request: NoteExtractionRequest
    ) -> None:
        _require(
            type(proposal) is NoteProposal
            and proposal.request_digest == request.request_digest
        )
        proposal.__post_init__()
        statements = _statements(proposal)
        _require(1 <= len(statements) <= self._policy.max_statements)
        _require(len({item.statement_ref for item in statements}) == len(statements))
        _require(
            sum(len(item.text) for item in statements) <= self._policy.max_note_chars
        )
        _require(
            {item.step_ref for item in statements}
            == {item.step_ref for item in proposal.steps}
        )
        evidence_ids = {item.evidence_ref for item in request.evidence}
        steps = {item.step_ref: item for item in proposal.steps}
        for step in proposal.steps:
            _require(step.artifact_digest == self._declarations[1].artifact_digest)
            _require(set(step.input_refs) <= evidence_ids)
        for item in statements:
            item.__post_init__()
            ends: dict[str, int] = {}
            for citation in sorted(
                item.citations, key=lambda value: (value.evidence_ref, value.start)
            ):
                _require(citation.start >= ends.get(citation.evidence_ref, 0))
                ends[citation.evidence_ref] = citation.end
            _require(
                {ref.evidence_ref for ref in item.citations}
                == set(steps[item.step_ref].input_refs)
            )
        _require(len(_render(proposal)[0]) <= self._policy.max_note_chars)

    def _screen_proposal(
        self, proposal: NoteProposal, session_ref: str
    ) -> tuple[NoteProposal, tuple[str, ...]]:
        statements = _statements(proposal)
        released = _screen(
            session_ref,
            tuple((item.statement_ref, item.text) for item in statements),
            self._screener,
            self._policy,
            self._clock,
        )
        by_id = {item.segment_ref: item for item in released}
        changed = tuple(
            item.statement_ref
            for item in statements
            if by_id[item.statement_ref].text != item.text
        )
        sections = tuple(
            replace(
                section,
                statements=tuple(
                    replace(item, text=by_id[item.statement_ref].text)
                    for item in section.statements
                ),
            )
            for section in proposal.sections
        )
        return replace(proposal, sections=sections), changed

    def _evaluate(
        self,
        proposal: NoteProposal,
        evidence: tuple[ScreenedRoleEvidence, ...],
        output_redactions: tuple[str, ...],
    ) -> dict[str, Any]:
        by_id = {item.evidence_ref: item for item in evidence}
        statements = _statements(proposal)
        blockers: dict[str, set[str]] = {
            item.statement_ref: set() for item in statements
        }
        citations: dict[str, list[dict[str, Any]]] = {}
        verification = []
        for item in statements:
            reasons = blockers[item.statement_ref]
            if item.statement_ref in output_redactions:
                reasons.add("generated_phi_redacted")
            if not item.citations:
                reasons.add("uncited_statement")
            if (
                None in (item.concept_ref, item.subject_ref, item.temporal_ref)
                or item.assertion is NoteAssertion.UNKNOWN
            ):
                reasons.add("unresolved_claim_scope")
            if item.uncertainty is NoteUncertainty.UNRESOLVED:
                reasons.add("unresolved_uncertainty")
            if (
                item.assertion in (NoteAssertion.UNCERTAIN, NoteAssertion.HYPOTHETICAL)
                and item.uncertainty is NoteUncertainty.NONE_DECLARED
            ):
                reasons.add("uncertainty_not_preserved")
            mapped, premises = [], []
            for citation in item.citations:
                source = by_id[citation.evidence_ref]
                if citation.end > len(source.text):
                    reasons.add("invalid_citation")
                    continue
                links = [
                    mapping
                    for mapping in source.mappings
                    if mapping.output_start < citation.end
                    and mapping.output_end > citation.start
                ]
                if any(mapping.redacted for mapping in links):
                    reasons.add("redacted_citation")
                    continue
                _require(
                    sum(
                        min(citation.end, m.output_end)
                        - max(citation.start, m.output_start)
                        for m in links
                    )
                    == citation.end - citation.start
                )
                for mapping in links:
                    left, right = (
                        max(citation.start, mapping.output_start),
                        min(citation.end, mapping.output_end),
                    )
                    mapped.append(
                        {
                            "evidence_ref": source.evidence_ref,
                            "alias": source.alias,
                            "role": source.role,
                            "segment_ref": source.citation.reference.segment_ref,
                            "registered_span_ref": source.citation.reference.span_ref,
                            "revision": source.citation.revision,
                            "start": mapping.source_start + left - mapping.output_start,
                            "end": mapping.source_start + right - mapping.output_start,
                        }
                    )
                premises.append(
                    f"{source.alias} [{source.role.value}]: "
                    + source.text[citation.start : citation.end]
                )
            citations[item.statement_ref] = mapped
            if item.citations and not reasons.intersection(
                {"invalid_citation", "redacted_citation"}
            ):
                verification.append(
                    StatementVerification(
                        item.statement_ref, "\n".join(premises), item.text
                    )
                )
        request = NoteVerificationRequest(
            _hash([asdict(item) for item in verification]), tuple(verification)
        )
        result = self._verifier.verify(request)
        _require(
            type(result) is NoteVerificationResult
            and result.request_digest == request.request_digest
        )
        _require(
            tuple(item.statement_ref for item in result.decisions)
            == tuple(item.statement_ref for item in verification)
        )
        for decision in result.decisions:
            decision.__post_init__()
            reasons = blockers[decision.statement_ref]
            if decision.verdict is SupportVerdict.CONTRADICTED:
                reasons.add("contradicted_statement")
            elif decision.verdict is not SupportVerdict.ENTAILED:
                reasons.add("unsupported_statement")
            elif decision.confidence_ppm < self._policy.min_support_ppm:
                reasons.add("insufficient_support")
        # Independently inspect assertions in complete same-subject/concept/time
        # groups; one favorable per-statement NLI result cannot hide a conflict.
        groups: dict[tuple[str | None, ...], list[NoteStatement]] = {}
        for item in statements:
            key = (item.subject_ref, item.concept_ref, item.temporal_ref)
            if None not in key:
                groups.setdefault(key, []).append(item)
        for group in groups.values():
            conflict = review_claim_conflicts(
                (
                    ClaimRecord(
                        group[0].statement_ref,
                        tuple(ClaimReference(item.statement_ref) for item in group),
                    ),
                ),
                assertion_records=tuple(
                    AssertionRecord(item.statement_ref, item.assertion.value)
                    for item in group
                ),
            )
            values = {
                item.value_digest
                for item in group
                if item.assertion is NoteAssertion.AFFIRMED
                and item.value_digest is not None
            }
            if conflict.requires_review or len(values) > 1:
                for item in group:
                    blockers[item.statement_ref].add("conflicting_statements")
        self._current_declarations()
        return {
            "sections": [
                {
                    "kind": section.kind,
                    "statement_refs": [
                        item.statement_ref for item in section.statements
                    ],
                }
                for section in proposal.sections
            ],
            "statements": [
                {
                    "statement_ref": item.statement_ref,
                    "text_digest": _hash(item.text),
                    "step_ref": item.step_ref,
                    "assertion": item.assertion,
                    "uncertainty": item.uncertainty,
                    "citations": citations[item.statement_ref],
                    "blockers": sorted(blockers[item.statement_ref]),
                }
                for item in statements
            ],
            "support": [asdict(item) for item in result.decisions],
            "steps": [asdict(item) for item in proposal.steps],
            "evidence": [_evidence_record(item) for item in evidence],
            "blocked": any(blockers.values()),
        }

    def compose(self, draft_ref: str) -> AmbientNoteDraft:
        """Create one screened, cited draft; blocked statements remain reviewable."""
        _ref(draft_ref)
        _require(not self._closed and not self._busy and self._draft is None)
        self._busy = True
        draft = None
        failed = False
        try:
            state: dict[str, Any] = {}

            def assemble(spans: tuple[ReviewedRoleSpan, ...]) -> str:
                evidence = self._screen_sources(spans)
                request = NoteExtractionRequest(
                    _hash([_evidence_record(item) for item in evidence]), evidence
                )
                self._revalidate_roles(spans)
                proposal = self._extractor.extract(request)
                self._validate_proposal(proposal, request)
                proposal, redactions = self._screen_proposal(
                    proposal, spans[0].citation.reference.session_ref
                )
                self._validate_proposal(proposal, request)
                self._revalidate_roles(spans)
                evaluation = self._evaluate(proposal, evidence, redactions)
                state.update(
                    proposal=proposal, evaluation=evaluation, redactions=redactions
                )
                return _json(asdict(proposal))

            role_draft = extract_with_reviewed_roles(
                self._aliases,
                self._sources,
                self._observations,
                self._reviews,
                assemble,
            )
            self._current_declarations()
            proposal, evaluation = state["proposal"], state["evaluation"]
            text, refs = _render(proposal)
            policy_digest = "sha256:" + _hash(
                [
                    asdict(self._policy),
                    [asdict(item) for item in self._declarations],
                    role_draft.context_digest,
                    role_draft.roles_digest,
                    role_draft.text_digest,
                    evaluation,
                ]
            )
            packet = ReviewedLocalEvidence(
                "source:" + _hash(draft_ref),
                reviewed_source_digest(text),
                len(text),
                policy_digest,
                refs,
            )
            report = {
                "schema": _SCHEMA,
                "draft_ref": draft_ref,
                "state": "draft",
                "clinician_review_required": True,
                "disclaimer": "assistive_draft_not_a_clinical_decision",
                "review_packet": packet.to_dict(),
                "evaluation": evaluation,
            }
            draft = AmbientNoteDraft(text, role_draft.text, packet, _json(report))
            self._proposal, self._role_draft = proposal, role_draft
            self._evaluation_digest = _hash(evaluation)
            self._output_redactions = state["redactions"]
            self._draft = draft
        except Exception:
            failed = True
        finally:
            self._busy = False
        if failed:
            self.close()
            raise NoteEvidenceError("ambient_note_composition_failed")
        assert draft is not None
        return draft

    def finalize(
        self,
        receipt: LocalReviewReceipt,
        *,
        current_source: CurrentLocalSource,
        authority: ReviewAuthorityVerifier,
        clock: Callable[[], float] = time.time,
    ) -> ReviewedAmbientNote:
        """Revalidate evidence and admit the exact clinician-reviewed local note.

        The trusted authority MUST check authenticated clinician permission,
        current review of this exact packet and revocation in its own custody.
        A caller boolean, serialized receipt or model verdict is insufficient.
        The source lookup must return the current review draft's digest. Neither
        source text nor an EHR destination is passed to the review authority.
        """
        _require(not self._closed and not self._busy and self._draft is not None)
        assert (
            self._draft is not None
            and self._proposal is not None
            and self._role_draft is not None
        )
        self._busy = True
        result = None
        failed = False
        try:
            started = clock()
            _require(
                type(started) in (int, float)
                and math.isfinite(started)
                and started >= 0
            )
            draft = self._draft
            proposal = self._proposal
            evaluation: dict[str, Any] = {}

            def recheck(spans: tuple[ReviewedRoleSpan, ...]) -> str:
                nonlocal evaluation
                evidence = self._screen_sources(spans)
                current, new_redactions = self._screen_proposal(
                    proposal, spans[0].citation.reference.session_ref
                )
                _require(current == proposal and not new_redactions)
                self._revalidate_roles(spans)
                evaluation = self._evaluate(proposal, evidence, self._output_redactions)
                _require(
                    not evaluation["blocked"]
                    and _hash(evaluation) == self._evaluation_digest
                )
                return _json(asdict(proposal))

            fresh = extract_with_reviewed_roles(
                self._aliases, self._sources, self._observations, self._reviews, recheck
            )
            _require(
                fresh.context_digest == self._role_draft.context_digest
                and fresh.roles_digest == self._role_draft.roles_digest
                and fresh.text_digest == self._role_draft.text_digest
            )
            packet = replace(draft.review_packet, review_receipt=receipt)
            admitted = admit_reviewed_local_evidence(
                packet,
                source_digest=reviewed_source_digest(draft.text),
                policy_digest=draft.review_packet.policy_digest,
                source=current_source,
                authority=authority,
                clock=clock,
            )
            # Finish external custody lookups before the final original-context
            # check, so callbacks cannot leave changed attribution behind.
            _require(
                current_source.current_digest(packet.source_id) == packet.source_digest
            )
            self._current_declarations()
            finalize_role_bound_note(
                self._role_draft,
                self._aliases,
                self._sources,
                self._observations,
                self._reviews,
            )
            ended = clock()
            _require(type(ended) in (int, float) and math.isfinite(ended))
            _require(
                ended >= started and receipt.issued_at <= ended < receipt.expires_at
            )
            report = json.loads(draft.report_json)
            report.update(
                state="clinician_reviewed",
                clinician_review_required=False,
                review_packet=admitted.to_dict(),
                evaluation=evaluation,
            )
            result = ReviewedAmbientNote(draft.text, draft.data_json, _json(report))
        except Exception:
            failed = True
        finally:
            self._busy = False
        if failed:
            raise NoteEvidenceError("ambient_note_finalization_blocked")
        assert result is not None
        return result
