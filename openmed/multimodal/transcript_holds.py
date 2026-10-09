"""Local critical-token holds over fixed transcript and draft evidence.

This slice does not assemble notes, run ASR, or maintain a correction ledger.
Source values remain private inputs; records and export permits are value-free.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

from openmed.clinical.dosing_check import DoseRangeTableSource, check_dose_ranges

POLICY_VERSION = 1
TRANSCRIPT_HOLD_NOTICE = (
    "Non-diagnostic transcript review aid. A clinician must independently verify "
    "the cited evidence and confirm before consequential use. No doses or orders "
    "are changed."
)


class CriticalTokenClass(str, Enum):
    """Controlled local safety-critical token categories."""

    NUMBER = "number"
    UNIT = "unit"
    DOSE = "dose"
    NEGATION = "negation"
    UNCERTAINTY = "uncertainty"
    LATERALITY = "laterality"
    MEDICATION = "medication"
    MODIFIER = "modifier"


CONFIDENCE_THRESHOLDS = MappingProxyType(
    {
        kind: 0.90 if kind == CriticalTokenClass.UNCERTAINTY else 0.95
        for kind in CriticalTokenClass
    }
)
_WORD_NUMBERS = frozenset(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty "
    "fifty sixty seventy eighty ninety hundred thousand million half quarter".split()
)
_UNITS = frozenset(
    "mg g kg mcg ug µg μg ml l mmol meq iu unit units tablet tablets capsule "
    "capsules mg/kg mg/ml mcg/kg percent %".split()
)
_NEGATIONS = frozenset("no not never without denies denied neither nor absent".split())
_UNCERTAINTY = frozenset(
    "possible possibly probable probably maybe uncertain suspected".split()
)
_LATERALITY = frozenset(
    "left right bilateral unilateral ipsilateral contralateral".split()
)
_MODIFIERS = frozenset(
    "hypo hyper hypoglycemia hyperglycemia hypotension hypertension".split()
)


class TranscriptHoldError(ValueError):
    """A controlled, content-free hold validation or review conflict."""


def _identity(value: int) -> None:
    if type(value) is not int or value < 0:
        raise TranscriptHoldError("invalid_identity")


def _normalize(text: str) -> str:
    if not isinstance(text, str):
        raise TranscriptHoldError("invalid_token")
    return text.strip().casefold().strip(";:!?").rstrip(".,")


@dataclass(frozen=True, order=True)
class TokenIdentity:
    """Opaque, non-negative segment and token identities, never source strings."""

    segment_id: int
    token_id: int

    def __post_init__(self) -> None:
        _identity(self.segment_id)
        _identity(self.token_id)


@dataclass(frozen=True)
class TranscriptToken:
    """Fixed private token evidence; an empty alternative represents deletion."""

    identity: TokenIdentity
    text: str = field(repr=False)
    confidence: float | None
    alternatives: tuple[str, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.identity, TokenIdentity):
            raise TranscriptHoldError("invalid_identity")
        if not isinstance(self.text, str) or not isinstance(self.alternatives, tuple):
            raise TranscriptHoldError("invalid_token")
        if not all(isinstance(value, str) for value in self.alternatives):
            raise TranscriptHoldError("invalid_token")
        if self.confidence is not None and (
            type(self.confidence) not in (int, float)
            or not math.isfinite(self.confidence)
            or not 0 <= self.confidence <= 1
        ):
            raise TranscriptHoldError("invalid_confidence")


@dataclass(frozen=True)
class DraftTokenCitation:
    """Fixed statement identity and exact token dependencies; no note text."""

    statement_id: int
    tokens: tuple[TokenIdentity, ...]

    def __post_init__(self) -> None:
        _identity(self.statement_id)
        if not isinstance(self.tokens, tuple) or not self.tokens:
            raise TranscriptHoldError("missing_citations")
        if not all(isinstance(ref, TokenIdentity) for ref in self.tokens):
            raise TranscriptHoldError("invalid_identity")
        if len(set(self.tokens)) != len(self.tokens):
            raise TranscriptHoldError("duplicate_citation")


class DoseCheckStatus(str, Enum):
    """Controlled outcomes from the unchanged local dosing checker."""

    IN_RANGE = "in_range"
    FLAGGED = "flagged"
    NOT_CHECKED = "not_checked"


@dataclass(frozen=True)
class TokenDoseEvidence:
    """Value-free dosing status bound to one cited token."""

    identity: TokenIdentity
    status: DoseCheckStatus

    def __post_init__(self) -> None:
        if not isinstance(self.identity, TokenIdentity) or not isinstance(
            self.status, DoseCheckStatus
        ):
            raise TranscriptHoldError("invalid_dose_evidence")


@dataclass(frozen=True, repr=False)
class TranscriptDose:
    """Private caller-extracted dose and all token identities supporting it."""

    tokens: tuple[TokenIdentity, ...]
    drug: str
    route: str
    amount: object
    unit: str

    def __repr__(self) -> str:
        return "TranscriptDose()"


def check_transcript_doses(
    doses: Iterable[TranscriptDose], reference_ranges: DoseRangeTableSource
) -> tuple[TokenDoseEvidence, ...]:
    """Reuse the existing checker, stripping values before the hold boundary.

    Args:
        doses: Caller-extracted doses with complete token dependencies.
        reference_ranges: Caller-owned local ranges; none are bundled.

    Returns:
        Value-free per-token status. Missing or incompatible ranges stay unchecked.

    Raises:
        TranscriptHoldError: On invalid input or checker failure, without payload.
    """
    findings: list[TokenDoseEvidence] = []
    try:
        for dose in doses:
            if (
                not isinstance(dose, TranscriptDose)
                or not dose.tokens
                or not all(isinstance(ref, TokenIdentity) for ref in dose.tokens)
            ):
                raise TranscriptHoldError("invalid_dose_evidence")
            suggestions = check_dose_ranges(
                [
                    {
                        "drug": dose.drug,
                        "route": dose.route,
                        "dose": dose.amount,
                        "unit": dose.unit,
                    }
                ],
                reference_ranges,
            )
            status = DoseCheckStatus.IN_RANGE
            for suggestion in suggestions:
                if suggestion.suggestion["kind"] == "dose_range_flag":
                    status = DoseCheckStatus.FLAGGED
                    break
                status = DoseCheckStatus.NOT_CHECKED
            findings.extend(TokenDoseEvidence(ref, status) for ref in dose.tokens)
    except Exception:
        raise TranscriptHoldError("dose_check_failed") from None
    return tuple(findings)


def classify_critical_token(
    text: str, *, medication_names: Iterable[str] = ()
) -> tuple[CriticalTokenClass, ...]:
    """Classify an English token using deterministic rules and caller lexicons.

    Args:
        text: Private token text (or alternative), including an omission marker.
        medication_names: Caller-owned exact medication-token lexicon.

    Returns:
        Sorted categories. Unknown words do not imply clinical safety.
    """
    value = _normalize(text)
    classes = set()
    parts = value.split("-")
    if value and (
        all(part in _WORD_NUMBERS for part in parts)
        or re.fullmatch(r"[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)(?:/\d+)?%?", value)
    ):
        classes.add(CriticalTokenClass.NUMBER)
    compact_dose = re.fullmatch(
        r"([+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+))\s*([^\d\s]+)", value
    )
    if compact_dose and compact_dose.group(2) in _UNITS:
        classes.update(
            (
                CriticalTokenClass.NUMBER,
                CriticalTokenClass.UNIT,
                CriticalTokenClass.DOSE,
            )
        )
    if value in _UNITS:
        classes.add(CriticalTokenClass.UNIT)
    for words, kind in (
        (_NEGATIONS, CriticalTokenClass.NEGATION),
        (_UNCERTAINTY, CriticalTokenClass.UNCERTAINTY),
        (_LATERALITY, CriticalTokenClass.LATERALITY),
        (_MODIFIERS, CriticalTokenClass.MODIFIER),
    ):
        if value in words:
            classes.add(kind)
    if value and value in {_normalize(name) for name in medication_names}:
        classes.add(CriticalTokenClass.MEDICATION)
    return tuple(sorted(classes, key=lambda kind: kind.value))


@dataclass(frozen=True)
class TokenHoldRecord:
    """Identifiers, controlled classes and scores only; no transcript values."""

    identity: TokenIdentity
    classes: tuple[CriticalTokenClass, ...]
    confidence: float | None
    threshold: float
    disagreement_score: int
    dose_flag_score: float
    policy_version: int = POLICY_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.identity, TokenIdentity)
            or not isinstance(self.classes, tuple)
            or not self.classes
            or not all(isinstance(kind, CriticalTokenClass) for kind in self.classes)
            or self.policy_version != POLICY_VERSION
            or type(self.threshold) not in (int, float)
            or not math.isfinite(self.threshold)
            or not 0 <= self.threshold <= 1
            or type(self.disagreement_score) is not int
            or self.disagreement_score not in (0, 1)
            or type(self.dose_flag_score) not in (int, float)
            or self.dose_flag_score not in (0, 0.5, 1)
        ):
            raise TranscriptHoldError("invalid_hold_record")
        # Reuse the private-token confidence validator without retaining text.
        TranscriptToken(self.identity, "", self.confidence)

    def to_dict(self) -> dict[str, object]:
        """Serialize the value-free hold record for local audit."""
        return {
            "segment_id": self.identity.segment_id,
            "token_id": self.identity.token_id,
            "classes": [kind.value for kind in self.classes],
            "confidence": self.confidence,
            "threshold": self.threshold,
            "disagreement_score": self.disagreement_score,
            "dose_flag_score": self.dose_flag_score,
            "policy_version": self.policy_version,
        }


@dataclass(frozen=True)
class HoldConfirmation:
    """Correction-path confirmation receipt bound to this exact evidence snapshot.

    An external correction workflow supplies reviewer authority. This receipt
    confirms current evidence; replacement text requires rebuilding the gate.
    """

    identity: TokenIdentity
    evidence_digest: str
    reviewer_id: int
    confirmed: bool

    def __post_init__(self) -> None:
        _identity(self.reviewer_id)
        if (
            not isinstance(self.identity, TokenIdentity)
            or not isinstance(self.evidence_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.evidence_digest) is None
            or type(self.confirmed) is not bool
        ):
            raise TranscriptHoldError("invalid_confirmation")


class TranscriptHoldGate:
    """Propagate token holds to fixed citations and gate reviewed export permits."""

    def __init__(
        self,
        tokens: Iterable[TranscriptToken],
        statements: Iterable[DraftTokenCitation],
        *,
        revision: int,
        medication_names: Iterable[str] = (),
        dose_evidence: Iterable[TokenDoseEvidence] = (),
    ) -> None:
        """Build a deterministic snapshot; reject ambiguous or missing references.

        Args:
            tokens: Fixed finalized evidence, including provider omission markers.
            statements: Every statement and its complete token dependencies.
            revision: Caller-owned evidence revision, never a patient identifier.
            medication_names: Local caller-owned medication-token lexicon.
            dose_evidence: Value-free findings from the unchanged dosing checker.
        """
        _identity(revision)
        token_values = tuple(tokens)
        statement_values = tuple(statements)
        dose_values = tuple(dose_evidence)
        if not all(isinstance(token, TranscriptToken) for token in token_values):
            raise TranscriptHoldError("invalid_token")
        if not all(isinstance(item, DraftTokenCitation) for item in statement_values):
            raise TranscriptHoldError("invalid_citation")
        if not all(isinstance(item, TokenDoseEvidence) for item in dose_values):
            raise TranscriptHoldError("invalid_dose_evidence")
        token_map = {token.identity: token for token in token_values}
        self._statements = {item.statement_id: item.tokens for item in statement_values}
        if len(token_map) != len(token_values) or len(self._statements) != len(
            statement_values
        ):
            raise TranscriptHoldError("duplicate_identity")
        if any(
            ref not in token_map for item in statement_values for ref in item.tokens
        ):
            raise TranscriptHoldError("unknown_citation")
        dose_scores: dict[TokenIdentity, float] = {}
        for item in dose_values:
            if item.identity not in token_map:
                raise TranscriptHoldError("unknown_dose_token")
            score = {
                DoseCheckStatus.IN_RANGE: 0.0,
                DoseCheckStatus.NOT_CHECKED: 0.5,
                DoseCheckStatus.FLAGGED: 1.0,
            }[item.status]
            dose_scores[item.identity] = max(dose_scores.get(item.identity, 0.0), score)
        medications = tuple(sorted({_normalize(name) for name in medication_names}))
        classes = {
            ref: set(
                kind
                for value in (token.text, *token.alternatives)
                for kind in classify_critical_token(value, medication_names=medications)
            )
            for ref, token in token_map.items()
        }
        # Adjacent fixed tokens are only a dose-shaped cue, not dose extraction.
        ordered = sorted(token_map)
        for left, right in zip(ordered, ordered[1:]):
            if (
                left.segment_id == right.segment_id
                and right.token_id == left.token_id + 1
            ):
                if (
                    CriticalTokenClass.NUMBER in classes[left]
                    and CriticalTokenClass.UNIT in classes[right]
                ):
                    classes[left].add(CriticalTokenClass.DOSE)
                    classes[right].add(CriticalTokenClass.DOSE)
        records = []
        for ref in ordered:
            token = token_map[ref]
            if ref in dose_scores:
                classes[ref].add(CriticalTokenClass.DOSE)
            if not classes[ref]:
                continue
            threshold = max(CONFIDENCE_THRESHOLDS[kind] for kind in classes[ref])
            disagreement = int(
                any(
                    _normalize(value) != _normalize(token.text)
                    for value in token.alternatives
                )
            )
            dose_score = dose_scores.get(ref, 0.0)
            if (
                token.confidence is None
                or token.confidence < threshold
                or disagreement
                or dose_score
            ):
                records.append(
                    TokenHoldRecord(
                        ref,
                        tuple(sorted(classes[ref], key=lambda kind: kind.value)),
                        token.confidence,
                        threshold,
                        disagreement,
                        dose_score,
                    )
                )
        self._holds = tuple(records)
        self._resolved: set[TokenIdentity] = set()
        self._revision = revision
        # The digest binds private evidence, policy, citation graph and dose status.
        # Only the digest leaves this computation, never the canonical payload.
        payload = [
            POLICY_VERSION,
            revision,
            medications,
            [
                [
                    ref.segment_id,
                    ref.token_id,
                    token_map[ref].text,
                    token_map[ref].confidence,
                    token_map[ref].alternatives,
                    sorted(kind.value for kind in classes[ref]),
                    dose_scores.get(ref),
                ]
                for ref in ordered
            ],
            [
                [key, [[ref.segment_id, ref.token_id] for ref in self._statements[key]]]
                for key in sorted(self._statements)
            ],
        ]
        self._digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def evidence_digest(self) -> str:
        """Return the content-free identity of this complete evidence snapshot."""
        return self._digest

    @property
    def holds(self) -> tuple[TokenHoldRecord, ...]:
        """Return immutable original findings, retained after confirmation."""
        return self._holds

    def statement_holds(self, statement_id: int) -> tuple[TokenHoldRecord, ...]:
        """Return every unresolved hold cited by the requested statement."""
        _identity(statement_id)
        if statement_id not in self._statements:
            raise TranscriptHoldError("unknown_statement")
        refs = self._statements[statement_id]
        return tuple(
            record
            for record in self._holds
            if record.identity in refs and record.identity not in self._resolved
        )

    def resolve(
        self, confirmation: HoldConfirmation, *, authorize: Callable[[int], bool]
    ) -> None:
        """Resolve through a current, explicitly confirmed correction-path receipt.

        Args:
            confirmation: Current-evidence confirmation from a correction workflow.
            authorize: Local reviewer authority lookup; must return exactly True.

        Raises:
            TranscriptHoldError: For stale, unknown, denied or unconfirmed receipts.
        """
        if not isinstance(confirmation, HoldConfirmation):
            raise TranscriptHoldError("invalid_confirmation")
        _identity(confirmation.reviewer_id)
        if confirmation.evidence_digest != self._digest:
            raise TranscriptHoldError("stale_confirmation")
        if confirmation.confirmed is not True or confirmation.identity not in {
            record.identity for record in self._holds
        }:
            raise TranscriptHoldError("invalid_confirmation")
        try:
            allowed = authorize(confirmation.reviewer_id) is True
        except Exception:
            allowed = False
        if not allowed:
            raise TranscriptHoldError("reviewer_denied")
        self._resolved.add(confirmation.identity)

    def export_reviewed(
        self, statement_id: int, *, reviewer_confirmed: bool
    ) -> dict[str, object]:
        """Issue a value-free permit only after holds and explicit review are clear.

        The caller must enforce this gate on its note export path. This permit
        carries no note text and is valid only for this evidence digest/revision.
        """
        if self.statement_holds(statement_id):
            raise TranscriptHoldError("unresolved_token_hold")
        if reviewer_confirmed is not True:
            raise TranscriptHoldError("review_required")
        return {
            "statement_id": statement_id,
            "revision": self._revision,
            "evidence_digest": self._digest,
            "notice": TRANSCRIPT_HOLD_NOTICE,
        }
