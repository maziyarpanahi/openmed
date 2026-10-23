"""Local, value-free correction propagation for caller-supplied ambient drafts.

This is an in-memory correction boundary, not a transcript provider or note
assembler. Callers retain protected content separately and supply complete
claim-to-evidence dependencies, keyed digests, and trusted local authority.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, replace
from enum import Enum

ADVISORY = "For clinician review; not a diagnosis or treatment recommendation."
SCHEMA_VERSION = 1
_UUID = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


class ConflictCode(str, Enum):
    """Controlled conflict codes; rejected values are never echoed."""

    INVALID_METADATA = "invalid_metadata"
    STALE_REVISION = "stale_revision"
    STALE_EVIDENCE = "stale_evidence"
    DUPLICATE_EVENT = "duplicate_event"
    UNKNOWN_REFERENCE = "unknown_reference"
    AUTHORITY_REQUIRED = "authority_required"
    VERIFICATION_REQUIRED = "verification_required"
    REVIEW_REQUIRED = "review_required"
    CONFIRMATION_REQUIRED = "confirmation_required"


class ReviewConflict(ValueError):
    """A typed, content-free rejection of an edit, review, or export."""

    def __init__(self, code: ConflictCode) -> None:
        self.code = code
        super().__init__(code.value)


def _require(
    condition: bool, code: ConflictCode = ConflictCode.INVALID_METADATA
) -> None:
    if not condition:
        raise ReviewConflict(code)


def _opaque(value: object) -> None:
    _require(isinstance(value, str) and _UUID.fullmatch(value) is not None)


def _digest(value: object) -> None:
    _require(isinstance(value, str) and _DIGEST.fullmatch(value) is not None)


def _revision(value: object) -> None:
    _require(type(value) is int and 0 <= value < (1 << 53))


class CorrectionKind(str, Enum):
    """Semantic correction categories without corrected source content."""

    MEDICATION = "medication"
    NEGATION = "negation"
    SPEAKER = "speaker"
    TRANSCRIPT = "transcript"


@dataclass(frozen=True)
class EvidenceRevision:
    """An opaque, stable evidence identity at one immutable revision."""

    evidence_id: str
    revision: int
    digest: str

    def __post_init__(self) -> None:
        _opaque(self.evidence_id)
        _revision(self.revision)
        _digest(self.digest)


@dataclass(frozen=True)
class DraftClaim:
    """A stable claim and its complete, caller-supplied evidence dependencies.

    Initial claims are always unverified and unapproved. Only ledger events can
    establish those states; no caller-provided prior approval is trusted.
    """

    claim_id: str
    citations: tuple[EvidenceRevision, ...]

    def __post_init__(self) -> None:
        _opaque(self.claim_id)
        _require(type(self.citations) is tuple and bool(self.citations))
        _require(all(isinstance(item, EvidenceRevision) for item in self.citations))
        _require(
            len({item.evidence_id for item in self.citations}) == len(self.citations)
        )


@dataclass(frozen=True)
class EvidenceChange:
    """Compare-and-set replacement of one evidence revision's keyed digest."""

    before: EvidenceRevision
    after_digest: str

    def __post_init__(self) -> None:
        _require(isinstance(self.before, EvidenceRevision))
        _digest(self.after_digest)
        _require(self.before.digest != self.after_digest)


@dataclass(frozen=True, kw_only=True)
class CorrectionEvent:
    """Versioned atomic corrections bound to draft, transcript, and authority.

    Args:
        event_id: Random opaque UUID, unique across the journal.
        reviewer_id: Opaque authority handle, never a name or account identifier.
        base_revision: Expected current ledger/draft revision.
        transcript_revision: Expected current transcript revision.
        kind: Controlled correction category.
        changes: Complete affected evidence units; content stays with the caller.
        schema_version: Supported event version, currently 1.
    """

    event_id: str
    reviewer_id: str
    base_revision: int
    transcript_revision: int
    kind: CorrectionKind
    changes: tuple[EvidenceChange, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _event_metadata(self)
        _require(isinstance(self.kind, CorrectionKind))
        _require(type(self.changes) is tuple and bool(self.changes))
        _require(all(isinstance(item, EvidenceChange) for item in self.changes))
        _require(
            len({item.before.evidence_id for item in self.changes}) == len(self.changes)
        )


class ReviewAction(str, Enum):
    """Separate trusted verification and explicit human approval steps."""

    VERIFY = "verify"
    APPROVE = "approve"


@dataclass(frozen=True, kw_only=True)
class ClaimReviewEvent:
    """Review a claim against an exact snapshot and claim revision.

    A verify event supplies a digest of actual caller-owned verification
    evidence. An approve event requires explicit human confirmation. The ledger
    never runs a model, fabricates a verification result, or assesses correctness.
    """

    event_id: str
    reviewer_id: str
    base_revision: int
    transcript_revision: int
    claim_id: str
    claim_revision: int
    action: ReviewAction
    verification_digest: str | None = None
    reviewer_confirmed: bool = False
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _event_metadata(self)
        _opaque(self.claim_id)
        _revision(self.claim_revision)
        _require(isinstance(self.action, ReviewAction))
        _require(type(self.reviewer_confirmed) is bool)
        if self.action is ReviewAction.VERIFY:
            _digest(self.verification_digest)
            _require(not self.reviewer_confirmed)
        else:
            _require(self.verification_digest is None)


def _event_metadata(event: CorrectionEvent | ClaimReviewEvent) -> None:
    _opaque(event.event_id)
    _opaque(event.reviewer_id)
    _revision(event.base_revision)
    _revision(event.transcript_revision)
    _require(
        type(event.schema_version) is int and event.schema_version == SCHEMA_VERSION
    )


@dataclass(frozen=True)
class ClaimState:
    """Claim state with revision-bound verification and approval event links."""

    claim: DraftClaim
    revision: int = 0
    verification_event: str | None = None
    approval_event: str | None = None


@dataclass(frozen=True)
class DraftRevision:
    """An immutable graph node; old nodes remain historical after an edit."""

    revision: int
    transcript_revision: int
    parent_revision: int | None
    event_id: str | None
    evidence: tuple[EvidenceRevision, ...]
    claims: tuple[ClaimState, ...]


class AmbientCorrectionLedger:
    """Propagate corrections atomically through an in-memory revision graph.

    Args:
        evidence: Current finalized evidence metadata from the caller.
        claims: Complete direct citation dependencies for a single draft.
        transcript_revision: Initial caller-owned finalized transcript revision.
        authority: Trusted local callback ``(reviewer_id, action) -> bool``.
            Actions are ``correct``, ``verify``, ``approve``, and ``export``.
            Missing grants and callback failures reject without exposing details.

    Instances are single-writer; applications must serialize calls. The ledger
    owns neither protected content nor dependency discovery, ASR, or EHR writes.
    """

    def __init__(
        self,
        evidence: Iterable[EvidenceRevision],
        claims: Iterable[DraftClaim],
        *,
        authority: Callable[[str, str], bool],
        transcript_revision: int = 0,
    ) -> None:
        evidence = tuple(evidence)
        claims = tuple(claims)
        _require(bool(evidence) and bool(claims) and callable(authority))
        _revision(transcript_revision)
        _require(all(isinstance(item, EvidenceRevision) for item in evidence))
        _require(all(isinstance(item, DraftClaim) for item in claims))
        _require(len({item.evidence_id for item in evidence}) == len(evidence))
        _require(len({item.claim_id for item in claims}) == len(claims))
        for claim in claims:
            _require(all(cite in evidence for cite in claim.citations))
        self._authority = authority
        self._graph = [
            DraftRevision(
                0,
                transcript_revision,
                None,
                None,
                tuple(sorted(evidence, key=lambda item: item.evidence_id)),
                tuple(
                    ClaimState(item)
                    for item in sorted(claims, key=lambda c: c.claim_id)
                ),
            )
        ]
        self._events: list[CorrectionEvent | ClaimReviewEvent] = []

    @property
    def current(self) -> DraftRevision:
        """Return the current immutable graph node."""
        return self._graph[-1]

    @property
    def graph(self) -> tuple[DraftRevision, ...]:
        """Return all historical nodes, including invalidated prior approvals."""
        return tuple(self._graph)

    @property
    def events(self) -> tuple[CorrectionEvent | ClaimReviewEvent, ...]:
        """Return immutable replay events without source payloads."""
        return tuple(self._events)

    def _authorize(self, reviewer_id: str, action: str) -> None:
        _opaque(reviewer_id)
        try:
            granted = self._authority(reviewer_id, action) is True
        except Exception:
            raise ReviewConflict(ConflictCode.AUTHORITY_REQUIRED) from None
        _require(granted, ConflictCode.AUTHORITY_REQUIRED)

    def apply(self, event: CorrectionEvent | ClaimReviewEvent) -> DraftRevision:
        """Apply one event atomically, rejecting stale or conflicting edits.

        Args:
            event: A validated correction, verification, or approval event.

        Returns:
            The new immutable graph node.

        Raises:
            ReviewConflict: No state changes occur on a rejected event.
        """
        _require(isinstance(event, (CorrectionEvent, ClaimReviewEvent)))
        _require(
            all(prior.event_id != event.event_id for prior in self._events),
            ConflictCode.DUPLICATE_EVENT,
        )
        current = self.current
        _revision(current.revision + 1)
        _require(
            event.base_revision == current.revision
            and event.transcript_revision == current.transcript_revision,
            ConflictCode.STALE_REVISION,
        )
        self._authorize(
            event.reviewer_id,
            "correct" if isinstance(event, CorrectionEvent) else event.action.value,
        )
        evidence = current.evidence
        claims = current.claims
        transcript_revision = current.transcript_revision
        if isinstance(event, CorrectionEvent):
            replacements = {}
            for change in event.changes:
                _require(
                    any(e.evidence_id == change.before.evidence_id for e in evidence),
                    ConflictCode.UNKNOWN_REFERENCE,
                )
                _require(change.before in evidence, ConflictCode.STALE_EVIDENCE)
                replacements[change.before.evidence_id] = EvidenceRevision(
                    change.before.evidence_id,
                    change.before.revision + 1,
                    change.after_digest,
                )
            evidence = tuple(replacements.get(e.evidence_id, e) for e in evidence)
            changed = []
            for state in claims:
                if any(c.evidence_id in replacements for c in state.claim.citations):
                    _revision(state.revision + 1)
                    claim = replace(
                        state.claim,
                        citations=tuple(
                            replacements.get(c.evidence_id, c)
                            for c in state.claim.citations
                        ),
                    )
                    changed.append(ClaimState(claim, state.revision + 1))
                else:
                    changed.append(state)
            claims = tuple(changed)
            transcript_revision += 1
            _revision(transcript_revision)
        else:
            targets = [
                state for state in claims if state.claim.claim_id == event.claim_id
            ]
            _require(bool(targets), ConflictCode.UNKNOWN_REFERENCE)
            state = targets[0]
            _require(
                state.revision == event.claim_revision, ConflictCode.STALE_REVISION
            )
            if event.action is ReviewAction.VERIFY:
                # Re-verification always clears any approval of the prior result.
                updated = replace(
                    state, verification_event=event.event_id, approval_event=None
                )
            else:
                _require(
                    state.verification_event is not None,
                    ConflictCode.VERIFICATION_REQUIRED,
                )
                _require(event.reviewer_confirmed, ConflictCode.CONFIRMATION_REQUIRED)
                updated = replace(state, approval_event=event.event_id)
            claims = tuple(updated if item is state else item for item in claims)
        node = DraftRevision(
            current.revision + 1,
            transcript_revision,
            current.revision,
            event.event_id,
            evidence,
            claims,
        )
        self._events.append(event)
        self._graph.append(node)
        return node

    def export_reviewed(
        self, revision: int, *, reviewer_id: str, reviewer_confirmed: bool
    ) -> dict:
        """Export only current reviewed metadata with a non-diagnostic notice.

        Args:
            revision: Exact current graph revision, never a historical node.
            reviewer_id: Opaque handle with export authority.
            reviewer_confirmed: Explicit confirmation for this consequential output.

        Returns:
            Content-free current citations and review links with mandatory notice.

        Raises:
            ReviewConflict: Historical, unreviewed, or unconfirmed export.
        """
        _revision(revision)
        _require(revision == self.current.revision, ConflictCode.STALE_REVISION)
        self._authorize(reviewer_id, "export")
        _require(reviewer_confirmed is True, ConflictCode.CONFIRMATION_REQUIRED)
        _require(
            all(s.verification_event and s.approval_event for s in self.current.claims),
            ConflictCode.REVIEW_REQUIRED,
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "advisory": ADVISORY,
            "non_diagnostic": True,
            "reviewer_confirmed": True,
            "draft": asdict(self.current),
        }

    @classmethod
    def replay(
        cls,
        evidence: Iterable[EvidenceRevision],
        claims: Iterable[DraftClaim],
        events: Iterable[CorrectionEvent | ClaimReviewEvent],
        *,
        authority: Callable[[str, str], bool],
        transcript_revision: int = 0,
    ) -> AmbientCorrectionLedger:
        """Rebuild the identical graph, rechecking authority for every event.

        Args:
            evidence: Identical initial evidence metadata.
            claims: Identical initial claims and complete dependencies.
            events: Original ordered correction and review journal.
            authority: Injected trusted replay authority; never bypassed.
            transcript_revision: Identical initial finalized transcript revision.

        Returns:
            A ledger with the same graph for the same inputs and authority grants.
        """
        ledger = cls(
            evidence,
            claims,
            authority=authority,
            transcript_revision=transcript_revision,
        )
        for event in events:
            ledger.apply(event)
        return ledger
