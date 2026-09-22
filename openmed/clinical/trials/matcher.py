"""Evidence-bound retrieval and reviewable trial eligibility matching."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from importlib import resources
from types import MappingProxyType
from typing import Any, Final

from openmed.clinical.journey import JourneySnapshot
from openmed.clinical.journey_contracts import canonical_digest, derived_opaque_id

from .contracts import TrialContractError, TrialStudyRecord
from .criteria import (
    TRIAL_CRITERIA_PARSER_VERSION,
    ParsedTrialCriteria,
    TrialCriterion,
    TrialCriterionKind,
    TrialCriterionOperator,
    parse_trial_criteria,
)

TRIAL_MATCH_SCHEMA_VERSION: Final = "1.0.0"
TRIAL_MATCH_COMPATIBILITY_POLICY: Final = "same_major"
TRIAL_MATCHER_VERSION: Final = "openmed.trial-matcher/1.0.0"
TRIAL_MATCH_ADVISORY: Final = (
    "Trial matching output is a review aid, not a clinical recommendation, "
    "eligibility determination, enrollment action, or authorization to contact."
)
TRIAL_MATCH_SCHEMA_PACKAGE: Final = "openmed.core.schemas.json"
TRIAL_MATCH_SCHEMA_NAME: Final = "trial_eligibility"

_CONTROLLED_RE = re.compile(r"^[a-z][a-z0-9_.:/-]{0,127}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_OPAQUE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}_[A-Za-z0-9_-]{16,128}$")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_TIMESTAMP_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})$"
)
_NCT_ID_RE = re.compile(r"^NCT[0-9]{8}$")


class TrialCriterionState(str, Enum):
    """Explicit criterion evaluation states."""

    MET = "met"
    NOT_MET = "not_met"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"
    UNSUPPORTED = "unsupported"


class TrialMatchState(str, Enum):
    """Conservative aggregate matching states."""

    ELIGIBLE = "eligible"
    NOT_ELIGIBLE = "not_eligible"
    REVIEW_REQUIRED = "review_required"


@dataclass(frozen=True, slots=True)
class JourneySignal:
    """Transient Journey projection; patient values are never serialized."""

    concept_kind: str
    concept: str
    snapshot_id: str
    snapshot_digest: str
    fact_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    numeric_value: float | None = field(default=None, repr=False)
    present: bool | None = field(default=True, repr=False)
    unit: str | None = None
    observed_at: str | None = None
    conflict_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _controlled(self.concept_kind, "concept_kind")
        _bounded_text(self.concept, "concept", 2048)
        _opaque_id(self.snapshot_id, "snapshot_id")
        _digest(self.snapshot_digest, "snapshot_digest")
        facts = _opaque_ids(self.fact_ids, "fact_ids", minimum=1)
        evidence = _opaque_ids(self.evidence_ids, "evidence_ids", minimum=1)
        conflicts = _opaque_ids(self.conflict_ids, "conflict_ids")
        if self.numeric_value is not None:
            if isinstance(self.numeric_value, bool) or not isinstance(
                self.numeric_value, (int, float)
            ):
                raise TrialContractError("numeric_value must be a finite number")
            numeric = float(self.numeric_value)
            if not math.isfinite(numeric):
                raise TrialContractError("numeric_value must be a finite number")
            object.__setattr__(self, "numeric_value", numeric)
        if self.present is not None and type(self.present) is not bool:
            raise TrialContractError("present must be boolean or null")
        if self.unit is not None:
            _controlled(self.unit.casefold(), "unit")
            object.__setattr__(self, "unit", self.unit.casefold())
        if self.observed_at is not None:
            _timestamp(self.observed_at, "observed_at")
        object.__setattr__(self, "fact_ids", facts)
        object.__setattr__(self, "evidence_ids", evidence)
        object.__setattr__(self, "conflict_ids", conflicts)

    @property
    def normalized_concept(self) -> str:
        """Return matching-only normalized public concept text."""

        return " ".join(_TOKEN_RE.findall(self.concept.casefold()))


@dataclass(frozen=True, slots=True)
class TrialMatchPolicy:
    """Versioned fail-closed aggregate matching policy."""

    policy_id: str = "trial_match_default"
    version: str = "1.0.0"
    unresolved_blocks_eligible: bool = True

    def __post_init__(self) -> None:
        _controlled(self.policy_id, "policy_id")
        if not re.fullmatch(r"[1-9][0-9]*\.[0-9]+\.[0-9]+", self.version):
            raise TrialContractError("policy version must be semantic")
        if type(self.unresolved_blocks_eligible) is not bool:
            raise TrialContractError("unresolved_blocks_eligible must be boolean")
        if not self.unresolved_blocks_eligible:
            raise TrialContractError("trial matching policy must fail closed")

    @property
    def digest(self) -> str:
        """Return the exact matching policy digest."""

        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return the complete policy."""

        return {
            "policy_id": self.policy_id,
            "unresolved_blocks_eligible": self.unresolved_blocks_eligible,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class TrialCandidate:
    """Value-free candidate rank bound to one study version."""

    study_id: str
    study_version_id: str
    study_version_digest: str
    rank: int
    retrieval_score: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.study_id, str)
            or _NCT_ID_RE.fullmatch(self.study_id) is None
        ):
            raise TrialContractError("candidate study_id must be an NCT identifier")
        _opaque_id(self.study_version_id, "study_version_id")
        _digest(self.study_version_digest, "study_version_digest")
        if type(self.rank) is not int or self.rank < 1:
            raise TrialContractError("candidate rank must be positive")
        if (
            isinstance(self.retrieval_score, bool)
            or not isinstance(self.retrieval_score, (int, float))
            or not 0 <= float(self.retrieval_score) <= 1
        ):
            raise TrialContractError("retrieval_score must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        """Return the value-free candidate rank."""

        return {
            "rank": self.rank,
            "retrieval_score": round(float(self.retrieval_score), 8),
            "study_id": self.study_id,
            "study_version_digest": self.study_version_digest,
            "study_version_id": self.study_version_id,
        }


@dataclass(frozen=True, slots=True)
class TrialCriterionEvidence:
    """Opaque Journey evidence and exact snapshot custody."""

    snapshot_id: str
    snapshot_digest: str
    fact_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    conflict_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _opaque_id(self.snapshot_id, "snapshot_id")
        _digest(self.snapshot_digest, "snapshot_digest")
        object.__setattr__(self, "fact_ids", _opaque_ids(self.fact_ids, "fact_ids"))
        object.__setattr__(
            self, "evidence_ids", _opaque_ids(self.evidence_ids, "evidence_ids")
        )
        object.__setattr__(
            self, "conflict_ids", _opaque_ids(self.conflict_ids, "conflict_ids")
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the value-free Journey evidence link."""

        return {
            "conflict_ids": list(self.conflict_ids),
            "evidence_ids": list(self.evidence_ids),
            "fact_ids": list(self.fact_ids),
            "snapshot_digest": self.snapshot_digest,
            "snapshot_id": self.snapshot_id,
        }


@dataclass(frozen=True, slots=True)
class TrialCriterionResult:
    """One patient criterion state without patient values."""

    criterion_id: str
    criterion_kind: TrialCriterionKind
    state: TrialCriterionState
    reason_code: str
    study_version_id: str
    study_version_digest: str
    evidence: TrialCriterionEvidence

    def __post_init__(self) -> None:
        _opaque_id(self.criterion_id, "criterion_id")
        object.__setattr__(self, "criterion_kind", _criterion_kind(self.criterion_kind))
        object.__setattr__(self, "state", _criterion_state(self.state))
        _controlled(self.reason_code, "reason_code")
        _opaque_id(self.study_version_id, "study_version_id")
        _digest(self.study_version_digest, "study_version_digest")
        if not isinstance(self.evidence, TrialCriterionEvidence):
            raise TypeError("evidence must be TrialCriterionEvidence")
        if (
            self.state is TrialCriterionState.CONFLICT
            and not self.evidence.conflict_ids
        ):
            raise TrialContractError("conflict criterion requires conflict evidence")

    @property
    def review_required(self) -> bool:
        """Return whether this criterion cannot support an eligible result."""

        return self.state in {
            TrialCriterionState.UNKNOWN,
            TrialCriterionState.CONFLICT,
            TrialCriterionState.UNSUPPORTED,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the value-free criterion result."""

        return {
            "criterion_id": self.criterion_id,
            "criterion_kind": self.criterion_kind.value,
            "evidence": self.evidence.to_dict(),
            "reason_code": self.reason_code,
            "review_required": self.review_required,
            "state": self.state.value,
            "study_version_digest": self.study_version_digest,
            "study_version_id": self.study_version_id,
        }


@dataclass(frozen=True, slots=True)
class TrialEligibilityResult:
    """Reviewable, fail-closed result for one study and Journey snapshot."""

    result_id: str
    candidate: TrialCandidate
    criteria_parse_digest: str
    criterion_results: tuple[TrialCriterionResult, ...]
    state: TrialMatchState
    policy: TrialMatchPolicy
    evaluated_at: str
    matcher_version: str = TRIAL_MATCHER_VERSION
    schema_version: str = TRIAL_MATCH_SCHEMA_VERSION
    compatibility_policy: str = TRIAL_MATCH_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        _contract(self.schema_version, self.compatibility_policy)
        _opaque_id(self.result_id, "result_id")
        if not isinstance(self.candidate, TrialCandidate):
            raise TypeError("candidate must be TrialCandidate")
        _digest(self.criteria_parse_digest, "criteria_parse_digest")
        results = tuple(self.criterion_results)
        if not results or any(
            not isinstance(item, TrialCriterionResult) for item in results
        ):
            raise TrialContractError("criterion_results must be non-empty")
        if len({item.criterion_id for item in results}) != len(results):
            raise TrialContractError("criterion result identifiers must be unique")
        for item in results:
            if (
                item.study_version_id != self.candidate.study_version_id
                or item.study_version_digest != self.candidate.study_version_digest
            ):
                raise TrialContractError("criterion result study custody differs")
        object.__setattr__(self, "state", _match_state(self.state))
        if not isinstance(self.policy, TrialMatchPolicy):
            raise TypeError("policy must be TrialMatchPolicy")
        _timestamp(self.evaluated_at, "evaluated_at")
        expected = _aggregate_state(results, self.policy)
        if self.state is not expected:
            raise TrialContractError("aggregate trial match state differs")
        expected_id = derived_opaque_id(
            "trialmatch",
            self.candidate.to_dict(),
            self.criteria_parse_digest,
            [item.to_dict() for item in results],
            self.policy.digest,
            self.matcher_version,
            self.evaluated_at,
        )
        if self.result_id != expected_id:
            raise TrialContractError("trial match result identifier differs")
        object.__setattr__(self, "criterion_results", results)

    @property
    def eligible(self) -> bool:
        """Return true only for a complete and explicitly eligible result."""

        return self.state is TrialMatchState.ELIGIBLE

    @property
    def review_required(self) -> bool:
        """Return whether unresolved criteria require human review."""

        return any(item.review_required for item in self.criterion_results)

    @property
    def result_digest(self) -> str:
        """Return the exact value-free result digest."""

        return canonical_digest(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "advisory": TRIAL_MATCH_ADVISORY,
            "candidate": self.candidate.to_dict(),
            "compatibility_policy": self.compatibility_policy,
            "criteria_parse_digest": self.criteria_parse_digest,
            "criterion_results": [item.to_dict() for item in self.criterion_results],
            "eligible": self.eligible,
            "evaluated_at": self.evaluated_at,
            "matcher_version": self.matcher_version,
            "policy": self.policy.to_dict(),
            "policy_digest": self.policy.digest,
            "result_id": self.result_id,
            "review_required": self.review_required,
            "schema_version": self.schema_version,
            "state": self.state.value,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the strict result without patient values."""

        return {**self._payload(), "result_digest": self.result_digest}

    def to_review_packet(self) -> dict[str, Any]:
        """Return the value-free review packet representation."""

        return self.to_dict()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrialEligibilityResult":
        """Parse and verify one strict eligibility result."""

        data = _mapping(value, "trial eligibility result")
        expected = {
            "advisory",
            "candidate",
            "compatibility_policy",
            "criteria_parse_digest",
            "criterion_results",
            "eligible",
            "evaluated_at",
            "matcher_version",
            "policy",
            "policy_digest",
            "result_digest",
            "result_id",
            "review_required",
            "schema_version",
            "state",
        }
        if set(data) != expected:
            raise TrialContractError("trial eligibility result fields differ")
        if data["advisory"] != TRIAL_MATCH_ADVISORY:
            raise TrialContractError("trial eligibility advisory differs")
        candidate_data = _mapping(data["candidate"], "candidate")
        _exact_keys(
            candidate_data,
            {
                "rank",
                "retrieval_score",
                "study_id",
                "study_version_digest",
                "study_version_id",
            },
            "candidate",
        )
        candidate = TrialCandidate(
            study_id=_text(candidate_data["study_id"], "study_id"),
            study_version_id=_text(
                candidate_data["study_version_id"], "study_version_id"
            ),
            study_version_digest=_text(
                candidate_data["study_version_digest"], "study_version_digest"
            ),
            rank=_integer(candidate_data["rank"], "rank"),
            retrieval_score=_number(candidate_data["retrieval_score"], "score"),
        )
        policy_data = _mapping(data["policy"], "policy")
        _exact_keys(
            policy_data,
            {"policy_id", "unresolved_blocks_eligible", "version"},
            "policy",
        )
        policy = TrialMatchPolicy(
            policy_id=_text(policy_data["policy_id"], "policy_id"),
            version=_text(policy_data["version"], "version"),
            unresolved_blocks_eligible=_boolean(
                policy_data["unresolved_blocks_eligible"],
                "unresolved_blocks_eligible",
            ),
        )
        results = tuple(
            _criterion_result_from_dict(_mapping(item, "criterion result"))
            for item in _sequence(data["criterion_results"], "criterion_results")
        )
        result = cls(
            result_id=_text(data["result_id"], "result_id"),
            candidate=candidate,
            criteria_parse_digest=_text(
                data["criteria_parse_digest"], "criteria_parse_digest"
            ),
            criterion_results=results,
            state=_match_state(data["state"]),
            policy=policy,
            evaluated_at=_text(data["evaluated_at"], "evaluated_at"),
            matcher_version=_text(data["matcher_version"], "matcher_version"),
            schema_version=_text(data["schema_version"], "schema_version"),
            compatibility_policy=_text(
                data["compatibility_policy"], "compatibility_policy"
            ),
        )
        if data["policy_digest"] != result.policy.digest:
            raise TrialContractError("trial match policy digest differs")
        if data["eligible"] is not result.eligible:
            raise TrialContractError("trial match eligibility differs")
        if data["review_required"] is not result.review_required:
            raise TrialContractError("trial match review state differs")
        if data["result_digest"] != result.result_digest:
            raise TrialContractError("trial match result digest differs")
        return result


def retrieve_trial_candidates(
    studies: Sequence[TrialStudyRecord],
    signals: Sequence[JourneySignal],
    *,
    limit: int = 20,
) -> tuple[TrialCandidate, ...]:
    """Retrieve and rerank study versions using deterministic lexical overlap."""

    if type(limit) is not int or not 1 <= limit <= 100:
        raise TrialContractError("candidate limit must be between 1 and 100")
    study_values = tuple(studies)
    signal_values = tuple(signals)
    if any(not isinstance(item, TrialStudyRecord) for item in study_values):
        raise TypeError("studies must contain TrialStudyRecord")
    if any(not isinstance(item, JourneySignal) for item in signal_values):
        raise TypeError("signals must contain JourneySignal")
    query_tokens = {
        token for signal in signal_values for token in _tokens(signal.concept)
    }
    ranked: list[tuple[float, TrialStudyRecord]] = []
    for study in study_values:
        public_text = " ".join(
            (
                study.brief_title,
                study.official_title or "",
                *study.conditions,
                *(item.name for item in study.interventions),
            )
        )
        study_tokens = set(_tokens(public_text))
        union = query_tokens | study_tokens
        score = len(query_tokens & study_tokens) / len(union) if union else 0.0
        ranked.append((score, study))
    ranked.sort(key=lambda item: (-item[0], item[1].study_id, item[1].version_id))
    return tuple(
        TrialCandidate(
            study_id=study.study_id,
            study_version_id=study.version_id,
            study_version_digest=study.version_digest,
            rank=rank,
            retrieval_score=score,
        )
        for rank, (score, study) in enumerate(ranked[:limit], start=1)
    )


def evaluate_trial_eligibility(
    study: TrialStudyRecord,
    snapshot: JourneySnapshot,
    signals: Sequence[JourneySignal],
    *,
    candidate: TrialCandidate | None = None,
    criteria: ParsedTrialCriteria | None = None,
    policy: TrialMatchPolicy | None = None,
    evaluated_at: str,
) -> TrialEligibilityResult:
    """Evaluate one public study against a named Journey snapshot."""

    if not isinstance(study, TrialStudyRecord):
        raise TypeError("study must be TrialStudyRecord")
    if not isinstance(snapshot, JourneySnapshot):
        raise TypeError("snapshot must be JourneySnapshot")
    _timestamp(evaluated_at, "evaluated_at")
    signal_values = tuple(signals)
    if any(not isinstance(item, JourneySignal) for item in signal_values):
        raise TypeError("signals must contain JourneySignal")
    canonical_criteria = parse_trial_criteria(study)
    if criteria is not None and not isinstance(criteria, ParsedTrialCriteria):
        raise TypeError("criteria must be ParsedTrialCriteria")
    parsed = criteria or canonical_criteria
    if (
        parsed.study_id != study.study_id
        or parsed.study_version_id != study.version_id
        or parsed.study_version_digest != study.version_digest
        or parsed.parse_digest != canonical_criteria.parse_digest
    ):
        raise TrialContractError("parsed criteria differ from the public study")
    active_candidate = candidate or TrialCandidate(
        study_id=study.study_id,
        study_version_id=study.version_id,
        study_version_digest=study.version_digest,
        rank=1,
        retrieval_score=1.0,
    )
    if (
        active_candidate.study_id != study.study_id
        or active_candidate.study_version_id != study.version_id
        or active_candidate.study_version_digest != study.version_digest
    ):
        raise TrialContractError("candidate belongs to another study version")
    active_policy = policy or TrialMatchPolicy()
    snapshot_digest = canonical_digest(snapshot.to_dict())
    if any(
        item.snapshot_id != snapshot.snapshot_id
        or item.snapshot_digest != snapshot_digest
        for item in signal_values
    ):
        raise TrialContractError("Journey signals belong to another snapshot")
    results = tuple(
        _evaluate_criterion(
            item,
            signal_values,
            snapshot_id=snapshot.snapshot_id,
            snapshot_digest=snapshot_digest,
            study=study,
            evaluated_at=evaluated_at,
        )
        for item in parsed.criteria
    )
    state = _aggregate_state(results, active_policy)
    result_id = derived_opaque_id(
        "trialmatch",
        active_candidate.to_dict(),
        parsed.parse_digest,
        [item.to_dict() for item in results],
        active_policy.digest,
        TRIAL_MATCHER_VERSION,
        evaluated_at,
    )
    return TrialEligibilityResult(
        result_id=result_id,
        candidate=active_candidate,
        criteria_parse_digest=parsed.parse_digest,
        criterion_results=results,
        state=state,
        policy=active_policy,
        evaluated_at=evaluated_at,
    )


def match_trial_candidates(
    studies: Sequence[TrialStudyRecord],
    snapshot: JourneySnapshot,
    signals: Sequence[JourneySignal],
    *,
    evaluated_at: str,
    limit: int = 20,
    policy: TrialMatchPolicy | None = None,
) -> tuple[TrialEligibilityResult, ...]:
    """Retrieve, rerank, and evaluate a bounded set of public studies."""

    study_values = tuple(studies)
    by_version = {item.version_id: item for item in study_values}
    if len(by_version) != len(study_values):
        raise TrialContractError("study versions must be unique")
    candidates = retrieve_trial_candidates(study_values, signals, limit=limit)
    return tuple(
        evaluate_trial_eligibility(
            by_version[item.study_version_id],
            snapshot,
            signals,
            candidate=item,
            policy=policy,
            evaluated_at=evaluated_at,
        )
        for item in candidates
    )


def _evaluate_criterion(
    criterion: TrialCriterion,
    signals: tuple[JourneySignal, ...],
    *,
    snapshot_id: str,
    snapshot_digest: str,
    study: TrialStudyRecord,
    evaluated_at: str,
) -> TrialCriterionResult:
    if not criterion.supported:
        return _result(
            criterion,
            TrialCriterionState.UNSUPPORTED,
            "criterion_unsupported",
            (),
            snapshot_id,
            snapshot_digest,
            study,
        )
    matches = tuple(
        item
        for item in signals
        if item.concept_kind == criterion.concept_kind
        and item.normalized_concept == " ".join(_tokens(criterion.concept or ""))
    )
    if not matches:
        return _result(
            criterion,
            TrialCriterionState.UNKNOWN,
            "journey_evidence_missing",
            (),
            snapshot_id,
            snapshot_digest,
            study,
        )
    if any(item.conflict_ids for item in matches):
        return _result(
            criterion,
            TrialCriterionState.CONFLICT,
            "journey_evidence_conflict",
            matches,
            snapshot_id,
            snapshot_digest,
            study,
        )
    in_window = tuple(
        item
        for item in matches
        if _within_window(item, criterion, evaluated_at=evaluated_at)
    )
    if not in_window:
        return _result(
            criterion,
            TrialCriterionState.UNKNOWN,
            "time_window_evidence_missing",
            matches,
            snapshot_id,
            snapshot_digest,
            study,
        )
    if criterion.operator is TrialCriterionOperator.EXISTS:
        presence = {item.present for item in in_window}
        if None in presence:
            state = TrialCriterionState.UNKNOWN
            reason = "assertion_state_unknown"
        elif len(presence) > 1:
            state = TrialCriterionState.CONFLICT
            reason = "assertion_state_conflict"
        elif presence == {True}:
            state = TrialCriterionState.MET
            reason = "concept_evidence_present"
        else:
            state = TrialCriterionState.NOT_MET
            reason = "concept_evidence_absent"
        return _result(
            criterion,
            state,
            reason,
            in_window,
            snapshot_id,
            snapshot_digest,
            study,
            synthesize_conflict=state is TrialCriterionState.CONFLICT,
        )
    states = tuple(_compare_signal(item, criterion) for item in in_window)
    resolved = {item for item in states if item is not TrialCriterionState.UNKNOWN}
    if len(resolved) > 1:
        state = TrialCriterionState.CONFLICT
        reason = "typed_value_conflict"
    elif TrialCriterionState.UNKNOWN in states:
        state = TrialCriterionState.UNKNOWN
        reason = "typed_value_missing"
    else:
        state = next(iter(resolved))
        reason = (
            "comparison_satisfied"
            if state is TrialCriterionState.MET
            else "comparison_failed"
        )
    return _result(
        criterion,
        state,
        reason,
        in_window,
        snapshot_id,
        snapshot_digest,
        study,
        synthesize_conflict=state is TrialCriterionState.CONFLICT,
    )


def _compare_signal(
    signal: JourneySignal, criterion: TrialCriterion
) -> TrialCriterionState:
    if (
        signal.present is not True
        or signal.numeric_value is None
        or criterion.value is None
    ):
        return TrialCriterionState.UNKNOWN
    if criterion.value.unit is not None and signal.unit != criterion.value.unit:
        return TrialCriterionState.UNKNOWN
    left = signal.numeric_value
    right = float(criterion.value.normalized_value)
    operator = criterion.operator
    if operator is None:
        return TrialCriterionState.UNKNOWN
    operations = {
        TrialCriterionOperator.EQ: left == right,
        TrialCriterionOperator.NE: left != right,
        TrialCriterionOperator.LT: left < right,
        TrialCriterionOperator.LTE: left <= right,
        TrialCriterionOperator.GT: left > right,
        TrialCriterionOperator.GTE: left >= right,
    }
    matched = operations.get(operator)
    if matched is None:
        return TrialCriterionState.UNKNOWN
    return TrialCriterionState.MET if matched else TrialCriterionState.NOT_MET


def _within_window(
    signal: JourneySignal, criterion: TrialCriterion, *, evaluated_at: str
) -> bool:
    if criterion.window is None:
        return True
    if signal.observed_at is None:
        return False
    observed = _time(signal.observed_at)
    evaluated = _time(evaluated_at)
    return (
        evaluated - timedelta(days=criterion.window.within_days)
        <= observed
        <= evaluated
    )


def _result(
    criterion: TrialCriterion,
    state: TrialCriterionState,
    reason: str,
    signals: Sequence[JourneySignal],
    snapshot_id: str,
    snapshot_digest: str,
    study: TrialStudyRecord,
    *,
    synthesize_conflict: bool = False,
) -> TrialCriterionResult:
    facts = tuple(item for signal in signals for item in signal.fact_ids)
    evidence = tuple(item for signal in signals for item in signal.evidence_ids)
    conflicts = tuple(item for signal in signals for item in signal.conflict_ids)
    if synthesize_conflict and not conflicts:
        conflicts = (
            derived_opaque_id("conflict", criterion.criterion_id, *sorted(set(facts))),
        )
    return TrialCriterionResult(
        criterion_id=criterion.criterion_id,
        criterion_kind=criterion.kind,
        state=state,
        reason_code=reason,
        study_version_id=study.version_id,
        study_version_digest=study.version_digest,
        evidence=TrialCriterionEvidence(
            snapshot_id=snapshot_id,
            snapshot_digest=snapshot_digest,
            fact_ids=tuple(sorted(set(facts))),
            evidence_ids=tuple(sorted(set(evidence))),
            conflict_ids=tuple(sorted(set(conflicts))),
        ),
    )


def _aggregate_state(
    results: Sequence[TrialCriterionResult], policy: TrialMatchPolicy
) -> TrialMatchState:
    unresolved = any(item.review_required for item in results)
    if unresolved:
        return TrialMatchState.REVIEW_REQUIRED
    excluded = any(
        item.criterion_kind is TrialCriterionKind.EXCLUSION
        and item.state is TrialCriterionState.MET
        for item in results
    )
    inclusion_failed = any(
        item.criterion_kind is TrialCriterionKind.INCLUSION
        and item.state is TrialCriterionState.NOT_MET
        for item in results
    )
    return (
        TrialMatchState.NOT_ELIGIBLE
        if excluded or inclusion_failed
        else TrialMatchState.ELIGIBLE
    )


def _criterion_result_from_dict(data: Mapping[str, Any]) -> TrialCriterionResult:
    expected = {
        "criterion_id",
        "criterion_kind",
        "evidence",
        "reason_code",
        "review_required",
        "state",
        "study_version_digest",
        "study_version_id",
    }
    if set(data) != expected:
        raise TrialContractError("criterion result fields differ")
    evidence_data = _mapping(data["evidence"], "criterion evidence")
    _exact_keys(
        evidence_data,
        {
            "conflict_ids",
            "evidence_ids",
            "fact_ids",
            "snapshot_digest",
            "snapshot_id",
        },
        "criterion evidence",
    )
    evidence = TrialCriterionEvidence(
        snapshot_id=_text(evidence_data["snapshot_id"], "snapshot_id"),
        snapshot_digest=_text(evidence_data["snapshot_digest"], "snapshot_digest"),
        fact_ids=_text_sequence(evidence_data["fact_ids"], "fact_ids"),
        evidence_ids=_text_sequence(evidence_data["evidence_ids"], "evidence_ids"),
        conflict_ids=_text_sequence(evidence_data["conflict_ids"], "conflict_ids"),
    )
    result = TrialCriterionResult(
        criterion_id=_text(data["criterion_id"], "criterion_id"),
        criterion_kind=_criterion_kind(data["criterion_kind"]),
        state=_criterion_state(data["state"]),
        reason_code=_text(data["reason_code"], "reason_code"),
        study_version_id=_text(data["study_version_id"], "study_version_id"),
        study_version_digest=_text(
            data["study_version_digest"], "study_version_digest"
        ),
        evidence=evidence,
    )
    if data["review_required"] is not result.review_required:
        raise TrialContractError("criterion review state differs")
    return result


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(value.casefold()))


def _contract(schema_version: str, compatibility_policy: str) -> None:
    if compatibility_policy != TRIAL_MATCH_COMPATIBILITY_POLICY:
        raise TrialContractError("unsupported trial match compatibility policy")
    if not isinstance(schema_version, str) or not schema_version.startswith(
        f"{TRIAL_MATCH_SCHEMA_VERSION.split('.', 1)[0]}."
    ):
        raise TrialContractError("unsupported trial match schema version")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TrialContractError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TrialContractError(f"{name} must be an array")
    return value


def _exact_keys(data: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(data) != expected:
        raise TrialContractError(f"{name} fields do not match the contract")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrialContractError(f"{name} must be non-empty text")
    return value


def _text_sequence(value: Any, name: str) -> tuple[str, ...]:
    return tuple(_text(item, name) for item in _sequence(value, name))


def _bounded_text(value: Any, name: str, limit: int) -> str:
    text = _text(value, name)
    if len(text.encode("utf-8")) > limit:
        raise TrialContractError(f"{name} exceeds the byte limit")
    return text


def _controlled(value: Any, name: str) -> str:
    if not isinstance(value, str) or _CONTROLLED_RE.fullmatch(value) is None:
        raise TrialContractError(f"{name} must be controlled text")
    return value


def _opaque_id(value: Any, name: str) -> str:
    if not isinstance(value, str) or _OPAQUE_ID_RE.fullmatch(value) is None:
        raise TrialContractError(f"{name} must be an opaque identifier")
    return value


def _opaque_ids(
    values: Sequence[str], name: str, *, minimum: int = 0
) -> tuple[str, ...]:
    normalized = tuple(sorted(_opaque_id(item, name) for item in values))
    if len(normalized) < minimum or len(normalized) != len(set(normalized)):
        raise TrialContractError(f"{name} count or uniqueness is invalid")
    return normalized


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise TrialContractError(f"{name} must be a normalized SHA-256 digest")
    return value


def _integer(value: Any, name: str) -> int:
    if type(value) is not int:
        raise TrialContractError(f"{name} must be an integer")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrialContractError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise TrialContractError(f"{name} must be finite")
    return result


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise TrialContractError(f"{name} must be boolean")
    return value


def _timestamp(value: Any, name: str) -> str:
    if not isinstance(value, str) or _TIMESTAMP_RE.fullmatch(value) is None:
        raise TrialContractError(f"{name} must be an ISO 8601 timestamp")
    _time(value)
    return value


def _time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TrialContractError("timestamp must be ISO 8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise TrialContractError("timestamp must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _criterion_kind(value: Any) -> TrialCriterionKind:
    try:
        return (
            value
            if isinstance(value, TrialCriterionKind)
            else TrialCriterionKind(value)
        )
    except (TypeError, ValueError):
        raise TrialContractError("criterion kind is unsupported") from None


def _criterion_state(value: Any) -> TrialCriterionState:
    try:
        return (
            value
            if isinstance(value, TrialCriterionState)
            else TrialCriterionState(value)
        )
    except (TypeError, ValueError):
        raise TrialContractError("criterion state is unsupported") from None


def _match_state(value: Any) -> TrialMatchState:
    try:
        return value if isinstance(value, TrialMatchState) else TrialMatchState(value)
    except (TypeError, ValueError):
        raise TrialContractError("trial match state is unsupported") from None


def matcher_provenance() -> Mapping[str, str]:
    """Return immutable parser and matcher version provenance."""

    return MappingProxyType(
        {
            "matcher_version": TRIAL_MATCHER_VERSION,
            "parser_version": TRIAL_CRITERIA_PARSER_VERSION,
        }
    )


def load_trial_eligibility_schema() -> dict[str, Any]:
    """Load the bundled reviewable trial eligibility JSON Schema."""

    resource = resources.files(TRIAL_MATCH_SCHEMA_PACKAGE).joinpath(
        f"{TRIAL_MATCH_SCHEMA_NAME}.schema.json"
    )
    return json.loads(resource.read_text(encoding="utf-8"))
