"""Adapt native measure custody to aggregate, locally signed evidence."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from openmed.clinical.journey_contracts import canonical_digest
from openmed.clinical.measures.contracts import (
    MeasureLanguage,
    MeasureRunResult,
    PopulationKind,
    PopulationState,
)

from .quality_measure_evidence import (
    CalculationTraceStep,
    ExclusionEvidence,
    InputProjectionEvidence,
    MeasureDefinitionEvidence,
    MeasureImplementationEvidence,
    MeasureResultEvidence,
    MeasureTimeBoundaries,
    QualityMeasureEvidenceError,
    QualityMeasureEvidencePacket,
    ValueSetEvidence,
    build_quality_measure_evidence_packet,
)


def build_native_measure_evidence_packet(
    run: MeasureRunResult,
    *,
    input_schema_digest: str,
    key_id: str,
    key_provider: Callable[[str], bytes | bytearray],
) -> QualityMeasureEvidencePacket:
    """Package a native run without copying patient-level identifiers or values.

    Args:
        run: Completed native run with pinned definition, value sets and engine.
        input_schema_digest: Caller-supplied digest of the actual input schema;
            the native run does not retain that schema.
        key_id: Public identifier of the local signing key.
        key_provider: Trusted local callable resolving key_id to an HMAC key.
            The adapter does not persist the key or perform I/O itself.

    Returns:
        A deterministic signed packet containing aggregate counts and digest
        commitments to source custody and population calculation traces.

    Raises:
        QualityMeasureEvidenceError: The run is not native, its raw counts cannot
            be represented, time precision is unsupported, or the provider fails.
            Exactly one denominator and numerator are required. Counts are not
            adjusted for exclusions, exceptions or initial-population membership.
    """

    if type(run) is not MeasureRunResult:
        raise QualityMeasureEvidenceError("invalid_run", "run")
    if run.definition.language is not MeasureLanguage.NATIVE or (
        run.engine.execution_mode != "native"
    ):
        raise QualityMeasureEvidenceError("unsupported_run", "run")
    populations = run.definition.populations
    denominators = [p for p in populations if p.kind is PopulationKind.DENOMINATOR]
    numerators = [p for p in populations if p.kind is PopulationKind.NUMERATOR]
    if len(denominators) != 1 or len(numerators) != 1:
        raise QualityMeasureEvidenceError("unsupported_population_roles", "run")

    counts = run.safe_summary()["population_counts"]
    met = PopulationState.MET.value
    exclusions: list[ExclusionEvidence] = []
    trace: list[CalculationTraceStep] = []
    exclusion_ids = {
        p.population_id
        for p in populations
        if p.kind is PopulationKind.DENOMINATOR_EXCLUSION
    }
    for population in populations:
        population_id = population.population_id
        steps = [
            step
            for subject in run.subject_results
            for step in subject.trace
            if step.step_id == population_id
        ]
        outcomes = [
            result
            for subject in run.subject_results
            for result in subject.populations
            if result.population_id == population_id
        ]
        # Sort commitments as multisets; never export individual patient records.
        output_digest = canonical_digest(
            {
                "counts": counts[population_id],
                "trace_digests": sorted(canonical_digest(s.to_dict()) for s in steps),
                "outcome_digests": sorted(p.digest for p in outcomes),
            }
        )
        definition_digest = canonical_digest(population.to_dict())
        trace.append(
            CalculationTraceStep(
                population_id,
                population.expression_ref,
                definition_digest,
                canonical_digest(sorted(s.input_digest for s in steps)),
                output_digest,
                counts[population_id][met],
            )
        )
        if population.kind in {
            PopulationKind.DENOMINATOR_EXCLUSION,
            PopulationKind.DENOMINATOR_EXCEPTION,
        }:
            exclusions.append(
                ExclusionEvidence(
                    population_id,
                    definition_digest,
                    output_digest,
                    counts[population_id][met],
                )
            )

    excluded_count = sum(
        any(
            p.population_id in exclusion_ids and p.state is PopulationState.MET
            for p in subject.populations
        )
        for subject in run.subject_results
    )
    result = MeasureResultEvidence(
        counts[denominators[0].population_id][met],
        counts[numerators[0].population_id][met],
        excluded_count,
        canonical_digest(counts),
    )
    projection = InputProjectionEvidence(
        input_schema_digest,
        canonical_digest(
            {
                "input_projection_digest": run.input_projection_digest,
                "source_snapshot_digest": run.source_snapshot_digest,
            }
        ),
        len(run.subject_results),
    )
    boundaries = _packet_boundaries(run)
    definition = MeasureDefinitionEvidence(
        run.definition.measure_id,
        run.definition.version,
        run.definition.definition_digest,
    )
    value_sets = tuple(
        ValueSetEvidence(v.value_set_id, v.version, v.digest)
        for v in run.definition.value_sets
    )
    implementation = MeasureImplementationEvidence(
        run.engine.engine_id,
        run.engine.version,
        run.engine.digest,
    )
    # Validate all public metadata before invoking a caller-managed provider.
    QualityMeasureEvidencePacket(
        definition,
        value_sets,
        projection,
        boundaries,
        exclusions,
        implementation,
        trace,
        result,
        key_id,
        "hmac-sha256:" + "0" * 64,
    )
    try:
        signing_key = key_provider(key_id)
    except Exception:
        raise QualityMeasureEvidenceError(
            "key_provider_failed", "key_provider"
        ) from None
    return build_quality_measure_evidence_packet(
        definition,
        value_sets,
        projection,
        boundaries,
        exclusions,
        implementation,
        trace,
        result,
        key_id=key_id,
        signing_key=signing_key,
    )


def _packet_boundaries(run: MeasureRunResult) -> MeasureTimeBoundaries:
    try:
        if any(
            "." in value
            for value in (
                run.measurement_period.start,
                run.measurement_period.end,
            )
        ):
            raise QualityMeasureEvidenceError("unsupported_time_precision", "period")
        start = datetime.fromisoformat(
            run.measurement_period.start.replace("Z", "+00:00")
        ).astimezone(timezone.utc)
        end = datetime.fromisoformat(
            run.measurement_period.end.replace("Z", "+00:00")
        ).astimezone(timezone.utc)
        # Native ends are inclusive; the packet uses an exclusive end. No rounding.
        end += timedelta(seconds=1)
        return MeasureTimeBoundaries(
            start.isoformat(timespec="seconds").replace("+00:00", "Z"),
            end.isoformat(timespec="seconds").replace("+00:00", "Z"),
        )
    except (ValueError, OverflowError) as error:
        if isinstance(error, QualityMeasureEvidenceError):
            raise
        raise QualityMeasureEvidenceError("unsupported_time_range", "period") from None


__all__ = ["build_native_measure_evidence_packet"]
