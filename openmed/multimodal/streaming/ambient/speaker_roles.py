"""Keep uncertain clinical roles separate from speaker aliases and gate their use."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from openmed.multimodal.streaming.speech.speaker_aliases import (
    ClinicalRole,
    SpeakerAliases,
)
from openmed.multimodal.streaming.transcript.revision_ledger import (
    MappedSpan,
    RevisionLedger,
    RevisionLedgerError,
    SpanReference,
)

ROLE_SCHEMA_VERSION = "openmed.ambient.speaker-roles.v1"
ROLE_NOTICE = "Reviewed speaker-role attribution only; not speaker identity, clinical correctness, consent or permission for automatic clinical decisions."
_REF = re.compile(r"[0-9a-f]{32}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_MAX_SOURCES = 32
_MAX_OBSERVATIONS = 128
_MAX_REFERENCES = 1024
_MAX_TEXT = 65_536
_MAX_TOTAL_TEXT = 262_144


class SpeakerRoleError(ValueError):
    """Controlled invalid-input or blocked-role-use failure without protected text."""


class RoleState(str, Enum):
    """Hypothesis states independent of any speaker alias or reviewer decision."""

    UNKNOWN = "unknown"
    SINGLE_HYPOTHESIS = "single_hypothesis"
    MULTIPLE_ROLES = "multiple_roles"


def _require(condition: bool) -> None:
    if not condition:
        raise SpeakerRoleError("invalid ambient speaker-role input")


def _ref(value: str) -> None:
    _require(type(value) is str and _REF.fullmatch(value) is not None)


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


@dataclass(frozen=True, slots=True, kw_only=True)
class TranscriptSource:
    """Protected current text paired with its existing mutable revision ledger.

    Args:
        ledger: Original registered span and token lineage for one segment.
        text: Current protected transcript; applications update it with ledger edits.

    The ledger does not store text. Role-review bindings include this exact text
    as well as ledger metadata, preventing same-length text changes reusing review.
    """

    ledger: RevisionLedger = field(repr=False)
    text: str = field(repr=False)

    def __post_init__(self) -> None:
        _require(type(self.ledger) is RevisionLedger)
        _text(self.text)


@dataclass(frozen=True, slots=True, kw_only=True)
class RoleHypothesis:
    """One unverified role guess with explicit uncertainty and transcript evidence.

    Args:
        role: Patient, clinician, caregiver, interpreter or other.
        evidence: Registered transcript spans supporting this guess.
        confidence_ppm: Uncalibrated confidence in parts per million, or unknown.
    """

    role: ClinicalRole
    evidence: tuple[SpanReference, ...]
    confidence_ppm: int | None = None

    def __post_init__(self) -> None:
        _require(type(self.role) is ClinicalRole)
        _require(type(self.evidence) is tuple and 1 <= len(self.evidence) <= 8)
        _require(all(type(item) is SpanReference for item in self.evidence))
        _require(len({item.span_ref for item in self.evidence}) == len(self.evidence))
        _require(
            self.confidence_ppm is None
            or (
                type(self.confidence_ppm) is int
                and 0 <= self.confidence_ppm <= 1_000_000
            )
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RoleObservation:
    """Per-span role hypotheses, never a guessed permanent role for an alias.

    Args:
        observation_id: Unique opaque attribution identity.
        cluster_ref: Current acoustic-cluster reference in the alias registry.
        diarization_revision: Exact revision that produced this attribution.
        citation: Registered primary speech span whose role-sensitive use is gated.
        hypotheses: Zero to five distinct role guesses; zero is explicitly unknown.
    """

    observation_id: str
    cluster_ref: str
    diarization_revision: int
    citation: SpanReference
    hypotheses: tuple[RoleHypothesis, ...]

    def __post_init__(self) -> None:
        _ref(self.observation_id)
        _ref(self.cluster_ref)
        _require(
            type(self.diarization_revision) is int
            and 0 <= self.diarization_revision <= 1_000_000
        )
        _require(type(self.citation) is SpanReference)
        _require(
            type(self.hypotheses) is tuple and len(self.hypotheses) <= len(ClinicalRole)
        )
        _require(all(type(item) is RoleHypothesis for item in self.hypotheses))
        _require(len({item.role for item in self.hypotheses}) == len(self.hypotheses))


@dataclass(frozen=True, slots=True)
class RoleAssessment:
    """Unverified hypotheses and current mapped evidence without speech text."""

    observation_id: str
    cluster_ref: str
    alias: str | None
    state: RoleState
    hypotheses: tuple[RoleHypothesis, ...]
    primary: MappedSpan
    supporting_evidence: tuple[MappedSpan, ...]
    blockers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RoleReport:
    """Current complete attribution binding; this report is never an approval."""

    context_digest: str
    session_ref: str
    diarization_revision: int
    assessments: tuple[RoleAssessment, ...]

    def to_json(self) -> str:
        """Return unverified role states, opaque references and source coordinates."""
        return json.dumps(
            {
                **asdict(self),
                "schema_version": ROLE_SCHEMA_VERSION,
                "review_required": True,
                "notice": ROLE_NOTICE,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


def _remap(ledger: RevisionLedger, reference: SpanReference) -> MappedSpan:
    try:
        return ledger.remap_span(reference)
    except RevisionLedgerError:
        pass
    raise SpeakerRoleError("role evidence is not a registered current-session citation")


def _assess(
    aliases: SpeakerAliases,
    sources: tuple[TranscriptSource, ...],
    observations: tuple[RoleObservation, ...],
) -> tuple[RoleReport, tuple[str, ...]]:
    _require(type(aliases) is SpeakerAliases)
    _require(type(sources) is tuple and 1 <= len(sources) <= _MAX_SOURCES)
    _require(all(type(item) is TranscriptSource for item in sources))
    _require(sum(len(item.text) for item in sources) <= _MAX_TOTAL_TEXT)
    _require(
        type(observations) is tuple and 1 <= len(observations) <= _MAX_OBSERVATIONS
    )
    _require(all(type(item) is RoleObservation for item in observations))
    _require(len({item.observation_id for item in observations}) == len(observations))
    _require(
        sum(1 + sum(len(h.evidence) for h in item.hypotheses) for item in observations)
        <= _MAX_REFERENCES
    )
    alias_state = aliases.to_dict()
    session = alias_state["session_ref"]
    active = {item["cluster_ref"]: item["alias"] for item in alias_state["active"]}
    segments: dict[str, TranscriptSource] = {}
    source_states: list[dict[str, Any]] = []
    for source in sources:
        state = source.ledger.to_dict()
        _require(
            state["session_ref"] == session and state["length"] == len(source.text)
        )
        _require(state["segment_ref"] not in segments)
        segments[state["segment_ref"]] = source
        source_states.append({"ledger": state, "text_digest": _hash(source.text)})

    def mapped(reference: SpanReference) -> MappedSpan:
        _require(reference.session_ref == session and reference.segment_ref in segments)
        return _remap(segments[reference.segment_ref].ledger, reference)

    assessments = []
    texts = []
    ranges: dict[str, list[tuple[int, int]]] = {}
    for observation in observations:
        primary = mapped(observation.citation)
        support = tuple(
            mapped(reference)
            for h in observation.hypotheses
            for reference in h.evidence
        )
        blockers = []
        if alias_state["closed"]:
            blockers.append("session_closed")
        if observation.diarization_revision != alias_state["revision"]:
            blockers.append("diarization_revision_changed")
        alias = active.get(observation.cluster_ref)
        if alias is None:
            blockers.append("speaker_not_active")
        if any(not item.valid for item in (primary, *support)):
            blockers.append("evidence_invalidated")
        if any(not item.finalized for item in (primary, *support)):
            blockers.append("transcript_not_finalized")
        if primary.valid:
            assert primary.start is not None and primary.end is not None
            previous = ranges.setdefault(observation.citation.segment_ref, [])
            if any(
                start < primary.end and primary.start < end for start, end in previous
            ):
                raise SpeakerRoleError("role-sensitive source spans must not overlap")
            previous.append((primary.start, primary.end))
            text = segments[observation.citation.segment_ref].text[
                primary.start : primary.end
            ]
        else:
            text = ""
        role_state = (
            RoleState.UNKNOWN
            if not observation.hypotheses
            else RoleState.SINGLE_HYPOTHESIS
            if len(observation.hypotheses) == 1
            else RoleState.MULTIPLE_ROLES
        )
        assessments.append(
            RoleAssessment(
                observation.observation_id,
                observation.cluster_ref,
                alias,
                role_state,
                observation.hypotheses,
                primary,
                support,
                tuple(blockers),
            )
        )
        texts.append(text)
    source_states.sort(key=lambda item: item["ledger"]["segment_ref"])
    binding = _hash(
        {
            "version": ROLE_SCHEMA_VERSION,
            "aliases": alias_state,
            "sources": source_states,
            "observations": [asdict(item) for item in observations],
        }
    )
    return RoleReport(
        binding, session, alias_state["revision"], tuple(assessments)
    ), tuple(texts)


def assess_speaker_roles(
    aliases: SpeakerAliases,
    sources: tuple[TranscriptSource, ...],
    observations: tuple[RoleObservation, ...],
) -> RoleReport:
    """Assess explicit hypotheses and fresh source lineage without selecting a role.

    Args:
        aliases: Existing current session alias registry.
        sources: Complete protected transcript segments needed by the observations.
        observations: Explicit non-overlapping role-sensitive spans and hypotheses.

    Returns:
        Unknown, single-hypothesis or multiple-role states. Every state requires
        explicit role review; confidence never selects a winner.
    """
    return _assess(aliases, sources, observations)[0]


@dataclass(frozen=True, slots=True, kw_only=True)
class RoleAttributionReview:
    """Explicit per-span role decision bound to all current attribution evidence.

    Args:
        observation_id: Exact observation being reviewed.
        context_digest: Complete current RoleReport binding.
        role: Selected or reviewer-corrected role; no automatic best-score choice.
        review_id: Unique opaque decision-event reference.
        reviewer_id: Authenticated reviewer reference owned by the application.
        decision_digest: Digest of the application's retained approval evidence.
        reviewer_confirmed: Must be explicitly true before role-sensitive use.
    """

    observation_id: str
    context_digest: str
    role: ClinicalRole
    review_id: str
    reviewer_id: str
    decision_digest: str
    reviewer_confirmed: bool = False

    def __post_init__(self) -> None:
        for value in (self.observation_id, self.review_id, self.reviewer_id):
            _ref(value)
        _digest(self.context_digest)
        _digest(self.decision_digest)
        _require(
            type(self.role) is ClinicalRole and type(self.reviewer_confirmed) is bool
        )


@dataclass(frozen=True, slots=True)
class ReviewedRoleSpan:
    """Protected extractor input with explicit reviewed role and source citation."""

    observation_id: str
    alias: str
    cluster_ref: str
    role: ClinicalRole
    citation: MappedSpan
    review: RoleAttributionReview
    text: str = field(repr=False)


def _admit(
    aliases: SpeakerAliases,
    sources: tuple[TranscriptSource, ...],
    observations: tuple[RoleObservation, ...],
    reviews: tuple[RoleAttributionReview, ...],
    *,
    allow_provisional: bool = False,
) -> tuple[RoleReport, tuple[ReviewedRoleSpan, ...], str]:
    report, texts = _assess(aliases, sources, observations)
    _require(type(reviews) is tuple and len(reviews) == len(observations))
    _require(all(type(item) is RoleAttributionReview for item in reviews))
    _require(
        {item.observation_id for item in reviews}
        == {item.observation_id for item in observations}
    )
    _require(len({item.review_id for item in reviews}) == len(reviews))
    if any(
        reason != "transcript_not_finalized" or not allow_provisional
        for item in report.assessments
        for reason in item.blockers
    ):
        raise SpeakerRoleError(
            "role-sensitive use requires current finalized source attribution"
        )
    by_id = {item.observation_id: item for item in reviews}
    admitted = []
    ordered_reviews = []
    for assessment, text in zip(report.assessments, texts):
        decision = by_id[assessment.observation_id]
        if (
            not decision.reviewer_confirmed
            or decision.context_digest != report.context_digest
        ):
            raise SpeakerRoleError(
                "role-sensitive use requires current explicit role review"
            )
        assert assessment.alias is not None
        admitted.append(
            ReviewedRoleSpan(
                assessment.observation_id,
                assessment.alias,
                assessment.cluster_ref,
                decision.role,
                assessment.primary,
                decision,
                text,
            )
        )
        ordered_reviews.append(asdict(decision))
    return report, tuple(admitted), _hash(ordered_reviews)


@dataclass(frozen=True, slots=True)
class RoleBoundDraft:
    """Protected extractor output bound to the exact reviewed role context."""

    text: str = field(repr=False)
    text_digest: str
    context_digest: str
    roles_digest: str
    provisional: bool
    report_json: str

    def to_json(self) -> str:
        """Return attribution provenance without transcript or draft content."""
        return self.report_json


def _provisional(report: RoleReport) -> bool:
    return any(
        not span.finalized
        for item in report.assessments
        for span in (item.primary, *item.supporting_evidence)
    )


def _audit(
    report: RoleReport,
    spans: tuple[ReviewedRoleSpan, ...],
    roles_digest: str,
    text_digest: str,
) -> str:
    return json.dumps(
        {
            "schema_version": ROLE_SCHEMA_VERSION,
            "notice": ROLE_NOTICE,
            "context": asdict(report),
            "roles_digest": roles_digest,
            "text_digest": text_digest,
            "clinical_review_required": True,
            "provisional_sources": _provisional(report),
            "reviews": [asdict(item.review) for item in spans],
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def extract_with_reviewed_roles(
    aliases: SpeakerAliases,
    sources: tuple[TranscriptSource, ...],
    observations: tuple[RoleObservation, ...],
    reviews: tuple[RoleAttributionReview, ...],
    extractor: Callable[[tuple[ReviewedRoleSpan, ...]], str],
    *,
    allow_provisional: bool = False,
) -> RoleBoundDraft:
    """Invoke a trusted local extractor only after every role attribution is reviewed.

    Args:
        aliases: Current session aliases; access must be serialized by the application.
        sources: Current protected text and original registered ledgers.
        observations: Complete required role-sensitive source spans.
        reviews: Exactly one current explicit role decision for each observation.
        extractor: Trusted local callback receiving immutable reviewed roles/text.
        allow_provisional: Opt in to drafts from current unfinalized transcripts;
            these cannot be finalized without fresh finalized-source review/extraction.

    Returns:
        Protected draft and exact role/source binding. Context is checked again
        after the callback; changed attribution prevents returning its result.

    Raises:
        SpeakerRoleError: Missing/stale review, unresolved evidence, callback failure
            or changed context. Callback side effects cannot be rolled back.
    """
    _require(callable(extractor) and type(allow_provisional) is bool)
    report, spans, roles_digest = _admit(
        aliases, sources, observations, reviews, allow_provisional=allow_provisional
    )
    text: str | None = None
    failed = False
    try:
        text = extractor(spans)
    except Exception:
        failed = True
    if failed:
        raise SpeakerRoleError("local role-sensitive extraction failed")
    _require(type(text) is str)
    assert text is not None
    _text(text)
    current, _, current_roles = _admit(
        aliases, sources, observations, reviews, allow_provisional=allow_provisional
    )
    if current.context_digest != report.context_digest or current_roles != roles_digest:
        raise SpeakerRoleError("role attribution changed during extraction")
    text_digest = _hash(text)
    return RoleBoundDraft(
        text,
        text_digest,
        report.context_digest,
        roles_digest,
        _provisional(report),
        _audit(report, spans, roles_digest, text_digest),
    )


@dataclass(frozen=True, slots=True)
class RoleCheckedNote:
    """Protected finalized role-bound note and separate attribution-only report."""

    data_json: str = field(repr=False)
    report_json: str

    def to_json(self) -> str:
        """Return role and source lineage without the note or transcript text."""
        return self.report_json


def finalize_role_bound_note(
    draft: RoleBoundDraft,
    aliases: SpeakerAliases,
    sources: tuple[TranscriptSource, ...],
    observations: tuple[RoleObservation, ...],
    reviews: tuple[RoleAttributionReview, ...],
) -> RoleCheckedNote:
    """Recheck source and role decisions before preparing the protected final note.

    Args:
        draft: Original role-bound extractor output with its unchanged text digest.
        aliases: Current session registry, not a stored prior snapshot.
        sources: Current original protected transcript segments and ledgers.
        observations: Complete current role-sensitive evidence set used for extraction.
        reviews: Current explicit attribution decisions for all observations.

    Returns:
        Role-checked protected note data. This does not certify factuality,
        authorize clinical use or replace consent, privacy and clinical review.

    Raises:
        SpeakerRoleError: Changed draft, source, aliases, review or missing attribution.
    """
    _require(type(draft) is RoleBoundDraft)
    _text(draft.text)
    report, spans, roles_digest = _admit(aliases, sources, observations, reviews)
    if (
        draft.context_digest != report.context_digest
        or draft.roles_digest != roles_digest
        or draft.text_digest != _hash(draft.text)
    ):
        raise SpeakerRoleError(
            "note finalization requires the unchanged reviewed role context"
        )
    audit = _audit(report, spans, roles_digest, draft.text_digest)
    return RoleCheckedNote(
        json.dumps(
            {
                **json.loads(audit),
                "state": "role_attribution_checked",
                "note": draft.text,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        audit,
    )
