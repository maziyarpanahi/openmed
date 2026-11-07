"""Small deterministic native measure evaluator over Journey facts."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from openmed.clinical.journey_contracts import (
    ClinicalFact,
    canonical_digest,
    derived_opaque_id,
)
from openmed.structured.store import StoreResult, StoreState

from .contracts import (
    CalculationTraceStep,
    MeasureConflictError,
    MeasureDefinition,
    MeasureEngineIdentity,
    MeasureEvidence,
    MeasureLanguage,
    MeasureRunResult,
    MeasureSubjectResult,
    MeasureTimeWindow,
    MeasureUnsupportedError,
    PopulationResult,
    PopulationState,
)

_CONTROLLED_RE = re.compile(r"^[a-z][a-z0-9_.:/-]{0,127}$")
_OPAQUE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}_[A-Za-z0-9_-]{16,128}$")

NATIVE_MEASURE_ENGINE = MeasureEngineIdentity(
    engine_id="openmed.native.measure",
    version="1.0.0",
    artifact_digest=canonical_digest(
        {
            "algorithm": "fact_type_status_cardinality",
            "contract": "1.0.0",
        }
    ),
    execution_mode="native",
)


@dataclass(frozen=True, slots=True)
class NativePopulationRule:
    """Validated fact-type/status rule for one small native population."""

    population_id: str
    fact_types: tuple[str, ...]
    allowed_statuses: tuple[str, ...]
    unknown_statuses: tuple[str, ...] = ("unknown", "uncertain")
    match_mode: str = "any"
    minimum_matches: int = 1

    def __post_init__(self) -> None:
        _controlled(self.population_id, "population_id")
        fact_types = _controlled_values(self.fact_types, "fact_types", minimum=1)
        allowed = _controlled_values(
            self.allowed_statuses, "allowed_statuses", minimum=1
        )
        unknown = _controlled_values(self.unknown_statuses, "unknown_statuses")
        if set(allowed).intersection(unknown):
            raise MeasureConflictError("allowed and unknown statuses must differ")
        if self.match_mode not in {"all", "any"}:
            raise MeasureUnsupportedError("native match mode is unsupported")
        if type(self.minimum_matches) is not int or self.minimum_matches < 1:
            raise MeasureConflictError("minimum_matches must be a positive integer")
        if self.match_mode == "all" and self.minimum_matches != 1:
            raise MeasureConflictError("all-mode rules cannot override minimum_matches")
        object.__setattr__(self, "fact_types", fact_types)
        object.__setattr__(self, "allowed_statuses", allowed)
        object.__setattr__(self, "unknown_statuses", unknown)

    @property
    def digest(self) -> str:
        """Return the exact native-rule digest."""

        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        """Return deterministic rule metadata."""

        return {
            "allowed_statuses": list(self.allowed_statuses),
            "fact_types": list(self.fact_types),
            "match_mode": self.match_mode,
            "minimum_matches": self.minimum_matches,
            "population_id": self.population_id,
            "unknown_statuses": list(self.unknown_statuses),
        }


def evaluate_native_measure(
    definition: MeasureDefinition,
    rules: Sequence[NativePopulationRule],
    facts: Iterable[ClinicalFact],
    *,
    subject_ids: Sequence[str],
    source_snapshot_id: str,
    source_snapshot_digest: str,
    measurement_period: MeasureTimeWindow,
    evaluated_at: str,
) -> StoreResult[MeasureRunResult]:
    """Evaluate small validated status/cardinality measures deterministically."""

    try:
        if not isinstance(definition, MeasureDefinition):
            raise TypeError("definition must be MeasureDefinition")
        if definition.language is not MeasureLanguage.NATIVE:
            return StoreResult.outcome(
                StoreState.UNSUPPORTED, "native_measure_language_unsupported"
            )
        normalized_rules = tuple(sorted(rules, key=lambda item: item.population_id))
        if any(not isinstance(item, NativePopulationRule) for item in normalized_rules):
            raise TypeError("rules must contain NativePopulationRule")
        definitions = {item.population_id: item for item in definition.populations}
        if {item.population_id for item in normalized_rules} != set(definitions):
            return StoreResult.outcome(
                StoreState.CONFLICT, "native_measure_population_rule_conflict"
            )
        if any(
            item.expression_ref != f"native:{rule.population_id}"
            for rule in normalized_rules
            for item in (definitions[rule.population_id],)
        ):
            return StoreResult.outcome(
                StoreState.UNSUPPORTED, "native_measure_expression_unsupported"
            )
        normalized_subjects = tuple(sorted(set(subject_ids)))
        if len(normalized_subjects) != len(subject_ids) or not normalized_subjects:
            return StoreResult.outcome(
                StoreState.CONFLICT, "native_measure_subject_conflict"
            )
        if any(_OPAQUE_ID_RE.fullmatch(item) is None for item in normalized_subjects):
            return StoreResult.outcome(
                StoreState.CONFLICT, "native_measure_subject_invalid"
            )
        normalized_facts = tuple(sorted(facts, key=lambda item: item.fact_id))
        if any(not isinstance(item, ClinicalFact) for item in normalized_facts):
            raise TypeError("facts must contain ClinicalFact")
        if len({item.fact_id for item in normalized_facts}) != len(normalized_facts):
            return StoreResult.outcome(
                StoreState.CONFLICT, "native_measure_fact_duplicate"
            )
        if any(item.subject_id not in normalized_subjects for item in normalized_facts):
            return StoreResult.outcome(
                StoreState.CONFLICT, "native_measure_fact_subject_conflict"
            )
        projection_digest = canonical_digest(
            {
                "facts": {
                    item.fact_id: item.canonical_hash for item in normalized_facts
                },
                "rules": {item.population_id: item.digest for item in normalized_rules},
                "subjects": list(normalized_subjects),
            }
        )
        by_subject: dict[str, list[ClinicalFact]] = defaultdict(list)
        for fact in normalized_facts:
            by_subject[fact.subject_id].append(fact)
        value_set_digests = {
            item.value_set_id: item.digest for item in definition.value_sets
        }
        results = tuple(
            _evaluate_subject(
                definition,
                normalized_rules,
                subject_id,
                by_subject.get(subject_id, ()),
                source_snapshot_id=source_snapshot_id,
                source_snapshot_digest=source_snapshot_digest,
                measurement_period=measurement_period,
                evaluated_at=evaluated_at,
                projection_digest=projection_digest,
                value_set_digests=value_set_digests,
            )
            for subject_id in normalized_subjects
        )
        run_id = derived_opaque_id(
            "measurerun",
            definition.definition_digest,
            source_snapshot_digest,
            measurement_period.digest,
            NATIVE_MEASURE_ENGINE.digest,
            projection_digest,
            evaluated_at,
        )
        return StoreResult.success(
            MeasureRunResult(
                run_id=run_id,
                definition=definition,
                source_snapshot_id=source_snapshot_id,
                source_snapshot_digest=source_snapshot_digest,
                measurement_period=measurement_period,
                engine=NATIVE_MEASURE_ENGINE,
                input_projection_digest=projection_digest,
                subject_results=results,
                evaluated_at=evaluated_at,
            ),
            created=True,
        )
    except MeasureUnsupportedError:
        return StoreResult.outcome(
            StoreState.UNSUPPORTED, "native_measure_feature_unsupported"
        )
    except MeasureConflictError:
        return StoreResult.outcome(StoreState.CONFLICT, "native_measure_conflict")


def _evaluate_subject(
    definition: MeasureDefinition,
    rules: Sequence[NativePopulationRule],
    subject_id: str,
    facts: Sequence[ClinicalFact],
    *,
    source_snapshot_id: str,
    source_snapshot_digest: str,
    measurement_period: MeasureTimeWindow,
    evaluated_at: str,
    projection_digest: str,
    value_set_digests: dict[str, str],
) -> MeasureSubjectResult:
    definitions = {item.population_id: item for item in definition.populations}
    populations: list[PopulationResult] = []
    trace: list[CalculationTraceStep] = []
    for rule in rules:
        matched = tuple(
            item for item in facts if item.fact_type in set(rule.fact_types)
        )
        evidence = MeasureEvidence(
            fact_ids=tuple(item.fact_id for item in matched),
            evidence_ids=tuple(
                sorted(
                    {
                        evidence_id
                        for item in matched
                        for evidence_id in item.evidence_ids
                    }
                )
            ),
            derivation_digests=tuple(item.derivation_hash for item in matched),
        )
        unknown = any(item.status in rule.unknown_statuses for item in matched)
        accepted = tuple(
            item for item in matched if item.status in rule.allowed_statuses
        )
        if unknown:
            state = PopulationState.UNKNOWN
            reason = "source_status_unknown"
        elif rule.match_mode == "all":
            by_type = {item.fact_type for item in accepted}
            state = (
                PopulationState.MET
                if set(rule.fact_types) <= by_type
                else PopulationState.NOT_MET
            )
            reason = (
                "all_fact_types_met"
                if state is PopulationState.MET
                else "required_fact_type_not_met"
            )
        else:
            state = (
                PopulationState.MET
                if len(accepted) >= rule.minimum_matches
                else PopulationState.NOT_MET
            )
            reason = (
                "minimum_matches_met"
                if state is PopulationState.MET
                else "minimum_matches_not_met"
            )
        population = PopulationResult(
            population_id=rule.population_id,
            kind=definitions[rule.population_id].kind,
            state=state,
            evidence=evidence,
            reason_code=reason,
        )
        populations.append(population)
        trace.append(
            CalculationTraceStep(
                step_id=rule.population_id,
                expression_ref=definitions[rule.population_id].expression_ref,
                state=state,
                input_digest=canonical_digest(
                    {
                        "fact_digests": [item.canonical_hash for item in matched],
                        "rule_digest": rule.digest,
                    }
                ),
                evidence_digest=evidence.digest,
                reason_code=reason,
            )
        )
    result_id = derived_opaque_id(
        "measureresult",
        definition.version_id,
        subject_id,
        source_snapshot_digest,
        measurement_period.digest,
        projection_digest,
        evaluated_at,
    )
    return MeasureSubjectResult(
        result_id=result_id,
        subject_id=subject_id,
        definition_version_id=definition.version_id,
        definition_digest=definition.definition_digest,
        source_snapshot_id=source_snapshot_id,
        source_snapshot_digest=source_snapshot_digest,
        measurement_period=measurement_period,
        engine=NATIVE_MEASURE_ENGINE,
        input_projection_digest=projection_digest,
        value_set_digests=value_set_digests,
        populations=tuple(populations),
        trace=tuple(trace),
        evaluated_at=evaluated_at,
    )


def _controlled(value: str, name: str) -> str:
    if not isinstance(value, str) or _CONTROLLED_RE.fullmatch(value) is None:
        raise MeasureConflictError(f"{name} must be controlled")
    return value


def _controlled_values(
    values: Sequence[str], name: str, *, minimum: int = 0
) -> tuple[str, ...]:
    result = tuple(sorted({_controlled(item, name) for item in values}))
    if len(result) != len(values):
        raise MeasureConflictError(f"{name} must be unique")
    if len(result) < minimum:
        raise MeasureConflictError(f"{name} has too few values")
    return result


__all__ = [
    "NATIVE_MEASURE_ENGINE",
    "NativePopulationRule",
    "evaluate_native_measure",
]
