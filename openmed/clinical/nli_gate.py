"""Deterministic abstention policy for clinical NLI evidence.

The gate consumes already-scored NLI probabilities and never consumes or
returns a clinical decision.  It turns a sufficiently separated, calibrated
entailment or contradiction score into a typed outcome.  Neutral, low-score,
and close competing scores become ``abstain`` and carry a structured human
review handoff.

Evidence links contain only opaque identifiers, offsets, and content hashes.
The module deliberately does not retain premise, hypothesis, or source text.
It is local-first and has no model loading, network, logging, or persistence
side effects.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

NLI_GATE_SCHEMA_VERSION = 1
NLI_OUTCOMES = ("entailment", "contradiction", "abstain")
NLI_SCORE_LABELS = ("entailment", "contradiction", "neutral")
DEFAULT_NLI_CALIBRATION_ID = "openmed-clinical-nli-v1"
DEFAULT_NLI_CALIBRATION_METHOD = "held_out_probability_threshold"
HUMAN_REVIEW_QUEUE = "clinical-nli-review"

NLIOutcome = Literal["entailment", "contradiction", "abstain"]

NLI_GATE_ADVISORY = (
    "Clinical NLI gate outputs are assistive evidence for human review, not a "
    "clinical decision, diagnosis, treatment decision, or autonomous clinical "
    "judgment."
)
HUMAN_REVIEW_INSTRUCTION = (
    "Review the cited evidence and claim before any downstream clinical use."
)

_REVIEW_REASONS = frozenset(
    {
        "neutral_or_unsupported",
        "below_calibrated_threshold",
        "insufficient_margin",
    }
)


@dataclass(frozen=True)
class EvidenceLink:
    """Typed, PHI-safe traceability for one NLI comparison.

    ``source_id`` and ``claim_id`` are opaque caller-owned references.  They
    must not contain note text or direct identifiers.  ``start`` and ``end``
    are offsets into the source document; ``source_hash`` and ``claim_hash``
    are content hashes when the caller wants content-addressed traceability.
    The gate never stores the text used to create those hashes.
    """

    source_id: str
    claim_id: str
    source_type: str = "clinical_evidence"
    claim_type: str = "clinical_claim"
    start: int | None = None
    end: int | None = None
    source_hash: str | None = None
    claim_hash: str | None = None
    relation: str = "nli"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_id", _required_token(self.source_id, "source_id")
        )
        object.__setattr__(self, "claim_id", _required_token(self.claim_id, "claim_id"))
        object.__setattr__(
            self,
            "source_type",
            _required_token(self.source_type, "source_type"),
        )
        object.__setattr__(
            self,
            "claim_type",
            _required_token(self.claim_type, "claim_type"),
        )
        object.__setattr__(self, "relation", _required_token(self.relation, "relation"))
        _validate_offsets(self.start, self.end)
        if self.source_hash is not None:
            object.__setattr__(
                self,
                "source_hash",
                _required_token(self.source_hash, "source_hash"),
            )
        if self.claim_hash is not None:
            object.__setattr__(
                self,
                "claim_hash",
                _required_token(self.claim_hash, "claim_hash"),
            )

    @property
    def target_id(self) -> str:
        """Return the claim reference under the generic typed-link name."""

        return self.claim_id

    @property
    def target_type(self) -> str:
        """Return the claim type under the generic typed-link name."""

        return self.claim_type

    @classmethod
    def from_text(
        cls,
        *,
        source_id: str,
        claim_id: str,
        source_text: str,
        claim_text: str,
        source_type: str = "clinical_evidence",
        claim_type: str = "clinical_claim",
        start: int | None = None,
        end: int | None = None,
        relation: str = "nli",
    ) -> "EvidenceLink":
        """Build a trace link while discarding source and claim text.

        This helper is intended for callers that have text at inference time.
        Only deterministic SHA-256 digests are retained in the returned link.
        """

        return cls(
            source_id=source_id,
            claim_id=claim_id,
            source_type=source_type,
            claim_type=claim_type,
            start=start,
            end=end,
            source_hash=hash_text(source_text),
            claim_hash=hash_text(claim_text),
            relation=relation,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvidenceLink":
        """Build a link from common source/target mapping aliases."""

        if not isinstance(value, Mapping):
            raise TypeError("evidence link must be an EvidenceLink or mapping")
        start, end = _mapping_offsets(value)
        source_text = value.get("source_text")
        claim_text = value.get("claim_text", value.get("target_text"))
        source_hash = value.get("source_hash")
        claim_hash = value.get("claim_hash", value.get("target_hash"))
        if source_hash is None and isinstance(source_text, str):
            source_hash = hash_text(source_text)
        if claim_hash is None and isinstance(claim_text, str):
            claim_hash = hash_text(claim_text)
        return cls(
            source_id=_first_mapping_value(value, ("source_id", "source_ref")),
            claim_id=_first_mapping_value(
                value,
                ("claim_id", "target_id", "claim_ref", "target_ref"),
            ),
            source_type=value.get(
                "source_type", value.get("source_kind", "clinical_evidence")
            ),
            claim_type=value.get(
                "claim_type",
                value.get("target_type", value.get("target_kind", "clinical_claim")),
            ),
            start=start,
            end=end,
            source_hash=source_hash,
            claim_hash=claim_hash,
            relation=value.get("relation", "nli"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic trace record without raw text."""

        return {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "claim_id": self.claim_id,
            "claim_type": self.claim_type,
            "target_id": self.claim_id,
            "target_type": self.claim_type,
            "start": self.start,
            "end": self.end,
            "source_hash": self.source_hash,
            "claim_hash": self.claim_hash,
            "relation": self.relation,
        }


@dataclass(frozen=True)
class NLIThresholds:
    """Calibrated operating thresholds for selective NLI decisions.

    A candidate entailment or contradiction must meet its class threshold and
    exceed the next-best class by more than ``margin``.  The calibration
    metadata is part of every result so a threshold cannot be mistaken for an
    uncalibrated model score.
    """

    entailment: float = 0.90
    contradiction: float = 0.90
    margin: float = 0.05
    calibration_id: str = DEFAULT_NLI_CALIBRATION_ID
    calibration_method: str = DEFAULT_NLI_CALIBRATION_METHOD
    calibrated: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "entailment",
            _probability(self.entailment, "entailment threshold"),
        )
        object.__setattr__(
            self,
            "contradiction",
            _probability(self.contradiction, "contradiction threshold"),
        )
        object.__setattr__(self, "margin", _probability(self.margin, "margin"))
        object.__setattr__(
            self,
            "calibration_id",
            _required_token(self.calibration_id, "calibration_id"),
        )
        object.__setattr__(
            self,
            "calibration_method",
            _required_token(self.calibration_method, "calibration_method"),
        )
        if self.calibrated is not True:
            raise ValueError("NLI thresholds must be calibrated")

    @property
    def entailment_threshold(self) -> float:
        """Return the entailment threshold under its descriptive name."""

        return self.entailment

    @property
    def contradiction_threshold(self) -> float:
        """Return the contradiction threshold under its descriptive name."""

        return self.contradiction

    @property
    def minimum_margin(self) -> float:
        """Return the minimum calibrated separation between classes."""

        return self.margin

    def to_dict(self) -> dict[str, Any]:
        """Return the pinned calibrated-policy metadata."""

        return {
            "entailment": self.entailment,
            "contradiction": self.contradiction,
            "margin": self.margin,
            "calibration_id": self.calibration_id,
            "calibration_method": self.calibration_method,
            "calibrated": self.calibrated,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NLIThresholds":
        """Build thresholds from short or descriptive mapping keys."""

        if not isinstance(value, Mapping):
            raise TypeError("NLI thresholds must be an NLIThresholds or mapping")
        return cls(
            entailment=value.get(
                "entailment",
                value.get("entailment_threshold", 0.90),
            ),
            contradiction=value.get(
                "contradiction",
                value.get("contradiction_threshold", 0.90),
            ),
            margin=value.get(
                "margin",
                value.get("minimum_margin", value.get("min_margin", 0.05)),
            ),
            calibration_id=value.get("calibration_id", DEFAULT_NLI_CALIBRATION_ID),
            calibration_method=value.get(
                "calibration_method",
                DEFAULT_NLI_CALIBRATION_METHOD,
            ),
            calibrated=value.get("calibrated", True),
        )


@dataclass(frozen=True)
class CalibratedNLIScores:
    """Validated probabilities supplied to :func:`evaluate_nli`."""

    entailment: float
    contradiction: float
    neutral: float = 0.0
    calibration_id: str | None = None
    calibrated: bool = True

    def __post_init__(self) -> None:
        for label in NLI_SCORE_LABELS:
            object.__setattr__(
                self,
                label,
                _probability(getattr(self, label), f"{label} probability"),
            )
        if self.calibration_id is not None:
            object.__setattr__(
                self,
                "calibration_id",
                _required_token(self.calibration_id, "calibration_id"),
            )
        if self.calibrated is not True:
            raise ValueError("NLI scores must be calibrated")

    @property
    def abstain(self) -> float:
        """Return the neutral probability under the public outcome name."""

        return self.neutral

    def to_dict(self) -> dict[str, Any]:
        """Return score metadata without any source or claim text."""

        return {
            "entailment": self.entailment,
            "contradiction": self.contradiction,
            "neutral": self.neutral,
            "calibration_id": self.calibration_id,
            "calibrated": self.calibrated,
        }


@dataclass(frozen=True)
class HumanReviewHandoff:
    """Deterministic queue handoff emitted for an abstained comparison."""

    reason: str
    evidence: EvidenceLink
    queue: str = HUMAN_REVIEW_QUEUE
    status: str = "pending"
    instruction: str = HUMAN_REVIEW_INSTRUCTION

    def __post_init__(self) -> None:
        if self.reason not in _REVIEW_REASONS:
            raise ValueError("unsupported human-review reason")
        if not isinstance(self.evidence, EvidenceLink):
            raise TypeError("human-review evidence must be an EvidenceLink")
        object.__setattr__(self, "queue", _required_token(self.queue, "queue"))
        object.__setattr__(self, "status", _required_token(self.status, "status"))
        object.__setattr__(
            self,
            "instruction",
            _required_token(self.instruction, "instruction"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a PHI-safe review task payload."""

        return {
            "queue": self.queue,
            "status": self.status,
            "reason": self.reason,
            "instruction": self.instruction,
            "evidence": self.evidence.to_dict(),
        }


@dataclass(frozen=True)
class NLIGateResult:
    """Typed outcome and audit-safe metadata for one evidence link."""

    outcome: NLIOutcome
    evidence: EvidenceLink
    calibrated_scores: Mapping[str, float]
    selected_probability: float
    margin: float
    threshold: float | None
    reason: str
    thresholds: NLIThresholds
    human_review: HumanReviewHandoff | None = None
    disclaimer: str = NLI_GATE_ADVISORY
    schema_version: int = NLI_GATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.outcome not in NLI_OUTCOMES:
            raise ValueError(
                "NLI gate outcome must be entailment, contradiction, or abstain"
            )
        if not isinstance(self.evidence, EvidenceLink):
            raise TypeError("NLI gate evidence must be an EvidenceLink")
        if not isinstance(self.thresholds, NLIThresholds):
            raise TypeError("NLI gate thresholds must be NLIThresholds")
        scores = {
            label: _probability(
                self.calibrated_scores.get(label, 0.0), f"{label} probability"
            )
            for label in NLI_SCORE_LABELS
        }
        object.__setattr__(self, "calibrated_scores", MappingProxyType(scores))
        object.__setattr__(
            self,
            "selected_probability",
            _probability(self.selected_probability, "selected probability"),
        )
        object.__setattr__(self, "margin", _probability(self.margin, "margin"))
        if self.threshold is not None:
            object.__setattr__(
                self, "threshold", _probability(self.threshold, "threshold")
            )
        if self.outcome == "abstain":
            if not isinstance(self.human_review, HumanReviewHandoff):
                raise ValueError("abstained NLI results require a human-review handoff")
        elif self.human_review is not None:
            raise ValueError("definitive NLI results cannot carry a review handoff")

    @property
    def status(self) -> NLIOutcome:
        """Return the outcome under the generic status name."""

        return self.outcome

    @property
    def score(self) -> float:
        """Return the selected calibrated probability."""

        return self.selected_probability

    @property
    def calibrated_confidence(self) -> float:
        """Return the selected calibrated probability."""

        return self.selected_probability

    @property
    def evidence_link(self) -> EvidenceLink:
        """Return the trace link under its descriptive name."""

        return self.evidence

    @property
    def requires_human_review(self) -> bool:
        """Return whether the result must be routed to a reviewer."""

        return self.outcome == "abstain"

    @property
    def human_review_required(self) -> bool:
        """Compatibility alias for :attr:`requires_human_review`."""

        return self.requires_human_review

    @property
    def autonomous_decision(self) -> Literal[False]:
        """Clinical NLI gating never makes an autonomous decision."""

        return False

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, raw-text-free result payload."""

        return {
            "schema_version": self.schema_version,
            "outcome": self.outcome,
            "status": self.outcome,
            "calibrated_scores": dict(self.calibrated_scores),
            "selected_probability": self.selected_probability,
            "score": self.selected_probability,
            "margin": self.margin,
            "threshold": self.threshold,
            "reason": self.reason,
            "thresholds": self.thresholds.to_dict(),
            "evidence": self.evidence.to_dict(),
            "evidence_link": self.evidence.to_dict(),
            "requires_human_review": self.requires_human_review,
            "autonomous_decision": False,
            "human_review": (
                None if self.human_review is None else self.human_review.to_dict()
            ),
            "disclaimer": self.disclaimer,
        }

    def to_audit_entry(self) -> dict[str, Any]:
        """Return the result fields suitable for an audit log."""

        payload = self.to_dict()
        payload.pop("evidence_link", None)
        return payload


class ClinicalNLIGate:
    """Reusable deterministic gate configured with calibrated thresholds."""

    def __init__(self, thresholds: NLIThresholds | Mapping[str, Any] | None = None):
        self.thresholds = _coerce_thresholds(thresholds)

    def evaluate(
        self,
        scores: CalibratedNLIScores | Mapping[str, Any],
        evidence: EvidenceLink | Mapping[str, Any],
    ) -> NLIGateResult:
        """Evaluate one score set and typed evidence link."""

        return evaluate_nli(scores, evidence, thresholds=self.thresholds)

    def evaluate_many(
        self,
        records: Iterable[
            tuple[
                CalibratedNLIScores | Mapping[str, Any],
                EvidenceLink | Mapping[str, Any],
            ]
        ],
    ) -> tuple[NLIGateResult, ...]:
        """Evaluate aligned score/link records in deterministic input order."""

        return tuple(self.evaluate(scores, evidence) for scores, evidence in records)


def hash_text(value: str) -> str:
    """Return a content hash without retaining the supplied text."""

    if not isinstance(value, str):
        raise TypeError("text to hash must be a string")
    if not value:
        raise ValueError("text to hash must not be empty")
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def evaluate_nli(
    scores: CalibratedNLIScores | Mapping[str, Any] | EvidenceLink,
    evidence: EvidenceLink | Mapping[str, Any] | CalibratedNLIScores,
    *,
    thresholds: NLIThresholds | Mapping[str, Any] | None = None,
) -> NLIGateResult:
    """Apply calibrated selective NLI policy to one typed evidence link.

    ``scores`` accepts a mapping with calibrated probabilities under
    ``entailment``, ``contradiction``, and ``neutral``.  For compatibility with
    a three-way verifier, a ``{"label": ..., "score": ...}`` mapping is also
    accepted; ``neutral`` becomes ``abstain``.  The function also accepts the
    evidence-first positional order and normalizes it deterministically.

    A result is definitive only when the winning entailment or contradiction
    score meets its calibrated threshold and has the required margin over the
    next-best class.  Every other result is ``abstain`` with a review handoff.
    """

    if _looks_like_evidence(scores) and not _looks_like_evidence(evidence):
        scores, evidence = evidence, scores
    link = _coerce_evidence(evidence)
    policy = _coerce_thresholds(thresholds)
    probabilities = _coerce_scores(scores, policy)

    ranked = sorted(
        NLI_SCORE_LABELS,
        key=lambda label: (-getattr(probabilities, label), _score_priority(label)),
    )
    winner = ranked[0]
    runner_up = ranked[1]
    winner_probability = getattr(probabilities, winner)
    margin = max(0.0, winner_probability - getattr(probabilities, runner_up))

    threshold = (
        policy.entailment
        if winner == "entailment"
        else policy.contradiction
        if winner == "contradiction"
        else None
    )
    definitive = (
        winner in {"entailment", "contradiction"}
        and threshold is not None
        and winner_probability >= threshold
        and margin >= policy.margin
        and not (margin == 0.0 and policy.margin == 0.0)
    )
    if definitive:
        outcome: NLIOutcome = winner  # type: ignore[assignment]
        reason = "threshold_and_margin_met"
        review = None
    else:
        outcome = "abstain"
        reason = _abstention_reason(
            winner, winner_probability, margin, threshold, policy
        )
        review = HumanReviewHandoff(reason=reason, evidence=link)

    return NLIGateResult(
        outcome=outcome,
        evidence=link,
        calibrated_scores={
            label: getattr(probabilities, label) for label in NLI_SCORE_LABELS
        },
        selected_probability=winner_probability,
        margin=margin,
        threshold=threshold,
        reason=reason,
        thresholds=policy,
        human_review=review,
    )


def evaluate_evidence_links(
    records: Iterable[
        tuple[
            CalibratedNLIScores | Mapping[str, Any],
            EvidenceLink | Mapping[str, Any],
        ]
    ],
    *,
    thresholds: NLIThresholds | Mapping[str, Any] | None = None,
) -> tuple[NLIGateResult, ...]:
    """Evaluate aligned score/link records without reordering them."""

    policy = _coerce_thresholds(thresholds)
    return tuple(
        evaluate_nli(scores, evidence, thresholds=policy)
        for scores, evidence in records
    )


def _coerce_thresholds(
    thresholds: NLIThresholds | Mapping[str, Any] | None,
) -> NLIThresholds:
    if thresholds is None:
        return NLIThresholds()
    if isinstance(thresholds, NLIThresholds):
        return thresholds
    return NLIThresholds.from_mapping(thresholds)


def _coerce_evidence(value: EvidenceLink | Mapping[str, Any]) -> EvidenceLink:
    if isinstance(value, EvidenceLink):
        return value
    if isinstance(value, Mapping):
        return EvidenceLink.from_mapping(value)
    raise TypeError("NLI gate evidence must be an EvidenceLink or mapping")


def _coerce_scores(
    value: CalibratedNLIScores | Mapping[str, Any] | EvidenceLink,
    thresholds: NLIThresholds,
) -> CalibratedNLIScores:
    if isinstance(value, CalibratedNLIScores):
        _validate_calibration_id(value.calibration_id, thresholds)
        return value
    if isinstance(value, EvidenceLink) or not isinstance(value, Mapping):
        raise TypeError("NLI scores must be CalibratedNLIScores or mapping")

    nested = value.get("probabilities", value.get("scores"))
    score_mapping = nested if isinstance(nested, Mapping) else value
    calibration_id = value.get("calibration_id")
    calibrated = value.get("calibrated", True)

    if (
        "label" in value
        and "score" in value
        and not any(label in score_mapping for label in NLI_SCORE_LABELS)
    ):
        label = _normalize_score_label(value.get("label"))
        score = _probability(value.get("score"), "NLI score")
        remainder = (1.0 - score) / 2.0
        return _validated_scores(
            label=label,
            score=score,
            remainder=remainder,
            calibration_id=calibration_id,
            calibrated=calibrated,
            thresholds=thresholds,
        )

    _validate_calibration_id(calibration_id, thresholds)
    return CalibratedNLIScores(
        entailment=score_mapping.get("entailment", 0.0),
        contradiction=score_mapping.get("contradiction", 0.0),
        neutral=score_mapping.get(
            "neutral",
            score_mapping.get("abstain", 0.0),
        ),
        calibration_id=calibration_id,
        calibrated=calibrated,
    )


def _validated_scores(
    *,
    label: str,
    score: float,
    remainder: float,
    calibration_id: Any,
    calibrated: Any,
    thresholds: NLIThresholds,
) -> CalibratedNLIScores:
    _validate_calibration_id(calibration_id, thresholds)
    values = {
        "entailment": remainder,
        "contradiction": remainder,
        "neutral": remainder,
    }
    values[label] = score
    return CalibratedNLIScores(
        **values,
        calibration_id=calibration_id,
        calibrated=calibrated,
    )


def _validate_calibration_id(value: str | None, thresholds: NLIThresholds) -> None:
    if value is not None and value != thresholds.calibration_id:
        raise ValueError(
            "NLI score calibration_id does not match calibrated thresholds"
        )


def _normalize_score_label(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("NLI score label must be a string")
    label = value.strip().casefold()
    if label == "abstain":
        return "neutral"
    if label not in NLI_SCORE_LABELS:
        raise ValueError(
            "NLI score label must be entailment, contradiction, or neutral"
        )
    return label


def _abstention_reason(
    winner: str,
    winner_probability: float,
    margin: float,
    threshold: float | None,
    policy: NLIThresholds,
) -> str:
    if winner == "neutral":
        return "neutral_or_unsupported"
    if threshold is not None and winner_probability < threshold:
        return "below_calibrated_threshold"
    if margin < policy.margin or (margin == 0.0 and policy.margin == 0.0):
        return "insufficient_margin"
    return "below_calibrated_threshold"


def _looks_like_evidence(value: object) -> bool:
    if isinstance(value, EvidenceLink):
        return True
    if not isinstance(value, Mapping):
        return False
    return any(
        key in value for key in ("source_id", "source_ref", "claim_id", "target_id")
    ) and not any(key in value for key in ("entailment", "contradiction", "neutral"))


def _first_mapping_value(value: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in value and value[key] is not None:
            return value[key]
    raise ValueError("evidence link is missing a required reference")


def _mapping_offsets(value: Mapping[str, Any]) -> tuple[int | None, int | None]:
    start = value.get("start", value.get("source_start"))
    end = value.get("end", value.get("source_end"))
    if "offset" in value and (start is None or end is None):
        offset = value["offset"]
        if isinstance(offset, Sequence) and not isinstance(offset, (str, bytes)):
            if len(offset) == 2:
                start, end = offset
    return start, end


def _validate_offsets(start: int | None, end: int | None) -> None:
    if (start is None) != (end is None):
        raise ValueError("evidence offsets require both start and end")
    if start is None:
        return
    if isinstance(start, bool) or not isinstance(start, int):
        raise TypeError("evidence start must be a non-negative integer")
    if isinstance(end, bool) or not isinstance(end, int):
        raise TypeError("evidence end must be a non-negative integer")
    if start < 0 or end <= start:
        raise ValueError("evidence offsets must satisfy 0 <= start < end")


def _required_token(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _probability(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field_name} must be numeric")
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{field_name} must be finite and between 0 and 1")
    return probability


def _score_priority(label: str) -> int:
    return {"entailment": 0, "contradiction": 1, "neutral": 2}[label]


NLIAbstentionGate = ClinicalNLIGate
NLIThresholdConfig = NLIThresholds
SourceTrace = EvidenceLink
GateResult = NLIGateResult
gate_nli = evaluate_nli
apply_nli_gate = evaluate_nli
abstention_gate = evaluate_nli


__all__ = [
    "DEFAULT_NLI_CALIBRATION_ID",
    "DEFAULT_NLI_CALIBRATION_METHOD",
    "GateResult",
    "HUMAN_REVIEW_INSTRUCTION",
    "HUMAN_REVIEW_QUEUE",
    "HumanReviewHandoff",
    "NLIAbstentionGate",
    "NLI_GATE_ADVISORY",
    "NLI_GATE_SCHEMA_VERSION",
    "NLIOutcome",
    "NLI_OUTCOMES",
    "NLI_SCORE_LABELS",
    "NLIThresholdConfig",
    "NLIThresholds",
    "CalibratedNLIScores",
    "ClinicalNLIGate",
    "EvidenceLink",
    "NLIGateResult",
    "SourceTrace",
    "abstention_gate",
    "apply_nli_gate",
    "evaluate_evidence_links",
    "evaluate_nli",
    "gate_nli",
    "hash_text",
]
