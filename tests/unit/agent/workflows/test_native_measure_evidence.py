"""Synthetic native-run adapter and comparator parity controls."""

from __future__ import annotations

import json
import traceback
from dataclasses import replace

import pytest

from openmed.agent.workflows import (
    DriftSource,
    QualityMeasureEvidenceError,
    QualityMeasureEvidencePacket,
    build_native_measure_evidence_packet,
    compare_quality_measure_evidence,
)
from openmed.clinical.journey_contracts import ClinicalFact, canonical_digest
from openmed.clinical.measures import (
    MeasureDefinition,
    MeasureLanguage,
    MeasurePopulationDefinition,
    MeasureTimeWindow,
    NativePopulationRule,
    PopulationKind,
    PopulationState,
    ValueSetBinding,
    compare_measure_runs,
    evaluate_native_measure,
)

KEY = b"synthetic-native-evidence-key"
SCHEMA = canonical_digest({"synthetic_input_schema": "1.0.0"})
PERIOD = MeasureTimeWindow("2026-01-01T00:00:00Z", "2026-12-31T23:59:59Z")


def _run(*, numerator_status="active", value="synthetic protected value"):
    roles = (
        ("denominator", PopulationKind.DENOMINATOR),
        ("numerator", PopulationKind.NUMERATOR),
        ("exclusion", PopulationKind.DENOMINATOR_EXCLUSION),
        ("exception", PopulationKind.DENOMINATOR_EXCEPTION),
        ("initial", PopulationKind.INITIAL_POPULATION),
    )
    definition = MeasureDefinition(
        "measure_syntheticmeasure01",
        "1.0.0",
        MeasureLanguage.NATIVE,
        tuple(
            MeasurePopulationDefinition(name, kind, f"native:{name}")
            for name, kind in roles
        ),
        (ValueSetBinding("synthetic.values", "2026.1", canonical_digest([])),),
    )
    subjects = tuple(f"patient_syntheticpatient{i:02d}" for i in range(3))
    facts = []
    for i, subject in enumerate(subjects):
        for j, (name, _) in enumerate(roles):
            status = "active"
            if name == "numerator":
                status = (
                    numerator_status if i == 0 else ("active" if i == 1 else "inactive")
                )
            elif name == "exclusion":
                status = "active" if i == 0 else "inactive"
            elif name == "exception":
                status = "unknown" if i == 2 else "inactive"
            facts.append(
                ClinicalFact(
                    fact_id=f"fact_syntheticfact{i:02d}{j:02d}",
                    subject_id=subject,
                    fact_type=name,
                    value=value,
                    status=status,
                    evidence_ids=(f"evidence_syntheticevidence{i:02d}{j:02d}",),
                    derivation_hash=canonical_digest({"synthetic": [i, j]}),
                )
            )
    outcome = evaluate_native_measure(
        definition,
        tuple(NativePopulationRule(name, (name,), ("active",)) for name, _ in roles),
        facts,
        subject_ids=subjects,
        source_snapshot_id="snapshot_syntheticsnapshot1",
        source_snapshot_digest=canonical_digest({"synthetic_snapshot": 1}),
        measurement_period=PERIOD,
        evaluated_at="2027-01-02T00:00:00Z",
    )
    assert outcome.value is not None
    return outcome.value


def _packet(run, **kwargs):
    options = dict(
        input_schema_digest=SCHEMA,
        key_id="synthetic.quality.key",
        key_provider=lambda _: KEY,
    )
    options.update(kwargs)
    return build_native_measure_evidence_packet(run, **options)


def _rebind(run, **changes):
    """Build internally consistent synthetic custody variants."""
    definition = changes.get("definition", run.definition)
    changes["subject_results"] = tuple(
        replace(
            s,
            definition_version_id=definition.version_id,
            definition_digest=definition.definition_digest,
            value_set_digests={v.value_set_id: v.digest for v in definition.value_sets},
            **{
                k: v
                for k, v in changes.items()
                if k
                in {
                    "engine",
                    "measurement_period",
                    "source_snapshot_digest",
                    "source_snapshot_id",
                    "input_projection_digest",
                    "evaluated_at",
                }
            },
        )
        for s in run.subject_results
    )
    return replace(run, **changes)


def test_native_run_packet_round_trip_and_exact_aggregate_mapping():
    run = _run()
    requests = []
    packet = _packet(run, key_provider=lambda key_id: requests.append(key_id) or KEY)
    assert requests == ["synthetic.quality.key"]
    assert packet.verify(KEY)
    assert QualityMeasureEvidencePacket.from_json(packet.to_json()) == packet
    assert QualityMeasureEvidencePacket.from_dict(packet.to_dict()).verify(KEY)
    assert packet == _packet(run)
    assert packet.result.denominator_count == 3
    assert packet.result.numerator_count == 2
    assert packet.result.exclusion_count == 1
    assert packet.result.result_digest == canonical_digest(
        run.safe_summary()["population_counts"]
    )
    assert packet.input_projection.record_count == 3
    assert packet.input_projection.schema_digest == SCHEMA
    assert packet.input_projection.projection_digest == canonical_digest(
        {
            "input_projection_digest": run.input_projection_digest,
            "source_snapshot_digest": run.source_snapshot_digest,
        }
    )
    assert packet.implementation.implementation_digest == run.engine.digest
    assert packet.time_boundaries.end == "2027-01-01T00:00:00Z"
    assert packet.value_sets[0].expansion_digest == run.definition.value_sets[0].digest
    assert {e.exclusion_id: e.excluded_count for e in packet.exclusions} == {
        "exclusion": 1,
        "exception": 0,
    }
    assert [t.step_id for t in packet.calculation_trace] == [
        p.population_id for p in run.definition.populations
    ]


def test_zero_subject_run_retains_population_trace_and_zero_counts():
    packet = _packet(replace(_run(), subject_results=()))
    assert packet.verify(KEY)
    assert packet.result.denominator_count == packet.result.numerator_count == 0
    assert packet.result.exclusion_count == packet.input_projection.record_count == 0
    assert all(t.output_count == 0 for t in packet.calculation_trace)


@pytest.mark.parametrize(
    "value",
    [
        "Jane Synthetic 555-010-1234 jane.synthetic@example.test",
        "MRN: SYNTHETIC-123; 123 Synthetic Street; 1940-01-02",
        "مصطنع ۱۲۳۴۵۶۷۸۹۰",
        "架空患者 １２３４５６７８９０",
        "नकली १२३४५६७८९०",
    ],
)
def test_packets_and_reports_never_copy_patient_values_or_identifiers(value):
    run = _run(value=value)
    packet = _packet(run)
    public = (
        packet.to_json()
        + repr(packet)
        + compare_quality_measure_evidence(packet, packet).to_json()
    )
    assert value not in public and value not in json.dumps(
        packet.to_dict(), ensure_ascii=False
    )
    forbidden = [run.run_id, run.source_snapshot_id, run.evaluated_at]
    for s in run.subject_results:
        forbidden.extend([s.subject_id, s.result_id])
        for p in s.populations:
            forbidden.extend(p.evidence.fact_ids + p.evidence.evidence_ids)
    assert all(item not in public for item in forbidden)
    assert KEY.decode() not in public
    assert "subject_results" not in public and "population_counts" not in public


@pytest.mark.parametrize(
    "case,reasons,sources,result_changed",
    [
        ("same", set(), set(), False),
        (
            "value_digest",
            {"definition_changed"},
            {"measure_definition", "value_set"},
            False,
        ),
        (
            "value_version",
            {"definition_changed"},
            {"measure_definition", "value_set"},
            False,
        ),
        (
            "value_added",
            {"definition_changed"},
            {"measure_definition", "value_set"},
            False,
        ),
        (
            "value_removed",
            {"definition_changed"},
            {"measure_definition", "value_set"},
            False,
        ),
        ("definition_version", {"definition_changed"}, {"measure_definition"}, False),
        ("engine_version", {"engine_changed"}, {"implementation"}, False),
        ("engine_artifact", {"engine_changed"}, {"implementation"}, False),
        ("period", {"measurement_period_changed"}, {"time_boundaries"}, False),
        ("snapshot", {"source_snapshot_changed"}, {"input_projection"}, False),
        ("projection", {"input_projection_changed"}, {"input_projection"}, False),
        (
            "population_logic",
            {"definition_changed"},
            {"measure_definition", "calculation_logic"},
            False,
        ),
        (
            "exclusion_logic",
            {"definition_changed"},
            {"measure_definition", "calculation_logic", "exclusion_definition"},
            False,
        ),
        (
            "population_state",
            {"input_projection_changed", "population_counts_changed"},
            {"input_projection", "calculation_output"},
            True,
        ),
        (
            "unknown_state",
            {"input_projection_changed", "population_counts_changed"},
            {"input_projection", "calculation_output"},
            True,
        ),
        ("evidence_only", set(), {"calculation_output", "exclusion_evidence"}, False),
        ("run_identity", set(), set(), False),
        ("evaluation_time", set(), set(), False),
        ("snapshot_identity", set(), set(), False),
        ("equivalent_period", {"measurement_period_changed"}, set(), False),
    ],
)
def test_native_and_packet_comparators_have_documented_parity(
    case, reasons, sources, result_changed
):
    baseline = _run()
    candidate = baseline
    definition = baseline.definition
    if case.startswith("value_"):
        bindings = definition.value_sets
        if case == "value_digest":
            bindings = (replace(bindings[0], digest=canonical_digest(["changed"])),)
        elif case == "value_version":
            bindings = (replace(bindings[0], version="2026.2"),)
        elif case == "value_added":
            bindings += (
                ValueSetBinding("synthetic.second", "2026.1", canonical_digest([1])),
            )
        else:
            bindings = ()
        candidate = _rebind(
            baseline, definition=replace(definition, value_sets=bindings)
        )
    elif case == "definition_version":
        candidate = _rebind(baseline, definition=replace(definition, version="1.1.0"))
    elif case.startswith("engine_"):
        change = (
            {"version": "1.1.0"}
            if case == "engine_version"
            else {"artifact_digest": canonical_digest([1])}
        )
        candidate = _rebind(baseline, engine=replace(baseline.engine, **change))
    elif case in {"period", "equivalent_period"}:
        period = (
            MeasureTimeWindow("2026-02-01T00:00:00Z", PERIOD.end)
            if case == "period"
            else MeasureTimeWindow(
                "2026-01-01T01:00:00+01:00", "2027-01-01T00:59:59+01:00"
            )
        )
        candidate = _rebind(baseline, measurement_period=period)
    elif case in {"snapshot", "projection"}:
        name = (
            "source_snapshot_digest"
            if case == "snapshot"
            else "input_projection_digest"
        )
        candidate = _rebind(baseline, **{name: canonical_digest([1])})
    elif case in {"population_logic", "exclusion_logic"}:
        name = "denominator" if case == "population_logic" else "exclusion"
        populations = tuple(
            replace(p, expression_ref="native:changed")
            if p.population_id == name
            else p
            for p in definition.populations
        )
        candidate = _rebind(
            baseline, definition=replace(definition, populations=populations)
        )
    elif case in {"population_state", "unknown_state"}:
        candidate = _run(
            numerator_status="inactive" if case == "population_state" else "unknown"
        )
    elif case == "evidence_only":
        subjects = tuple(
            replace(
                s,
                trace=tuple(
                    replace(t, evidence_digest=canonical_digest(["new evidence"]))
                    if t.step_id == "exclusion"
                    else t
                    for t in s.trace
                ),
            )
            for s in baseline.subject_results
        )
        candidate = replace(baseline, subject_results=subjects)
    elif case == "run_identity":
        candidate = replace(baseline, run_id="measurerun_otheridentifier01")
    elif case == "evaluation_time":
        candidate = _rebind(baseline, evaluated_at="2027-01-03T00:00:00Z")
    elif case == "snapshot_identity":
        candidate = _rebind(baseline, source_snapshot_id="snapshot_otheridentifier01")
    before, after = _packet(baseline), _packet(candidate)
    assert before.verify(KEY) and after.verify(KEY)
    native = compare_measure_runs(baseline, candidate)
    evidence = compare_quality_measure_evidence(before, after)
    assert set(native.reason_codes) == reasons
    assert {d.source.value for d in evidence.drift} == sources
    assert native.result_drift == evidence.result_changed == result_changed
    if case in {"value_digest", "engine_version", "period"}:
        assert before.packet_digest != after.packet_digest
        assert before.signature != after.signature


@pytest.mark.parametrize(
    "field",
    [
        "value_sets",
        "implementation",
        "time_boundaries",
        "input_projection",
        "calculation_trace",
        "result",
    ],
)
def test_tampered_adapter_packet_fails_signature_verification(field):
    packet = _packet(_run())
    data = packet.to_dict()
    if field == "value_sets":
        data[field][0]["expansion_digest"] = canonical_digest([1])
    elif field == "implementation":
        data[field]["version"] = "1.1.0"
    elif field == "time_boundaries":
        data[field]["end"] = "2027-02-01T00:00:00Z"
    elif field == "input_projection":
        data[field]["projection_digest"] = canonical_digest([1])
    elif field == "calculation_trace":
        data[field][0]["output_digest"] = canonical_digest([1])
    else:
        data[field]["numerator_count"] = 1
    with pytest.raises(QualityMeasureEvidenceError, match="signature_mismatch"):
        QualityMeasureEvidencePacket.from_dict(data).verify(KEY)
    with pytest.raises(QualityMeasureEvidenceError, match="signature_mismatch"):
        packet.verify(b"another-synthetic-key")


def test_provider_failure_diagnostics_hide_credentials_and_private_paths():
    secret = "synthetic-secret /private/synthetic/key.pem"

    def failing_provider(_):
        raise RuntimeError(secret)

    with pytest.raises(QualityMeasureEvidenceError) as caught:
        _packet(_run(), key_provider=failing_provider)
    assert caught.value.code == "key_provider_failed"
    assert secret not in str(caught.value)
    assert secret not in "".join(
        traceback.format_exception(caught.type, caught.value, caught.tb)
    )


@pytest.mark.parametrize("key", [b"short", None, "synthetic-secret", 42])
def test_invalid_provider_keys_fail_closed_without_echo(key):
    with pytest.raises(QualityMeasureEvidenceError, match="invalid_signing_key"):
        _packet(_run(), key_provider=lambda _: key)


@pytest.mark.parametrize(
    "kwargs", [{"input_schema_digest": "private path"}, {"key_id": "private key path"}]
)
def test_invalid_public_metadata_is_rejected_before_provider(kwargs):
    requests = []
    with pytest.raises(QualityMeasureEvidenceError):
        _packet(_run(), key_provider=lambda key: requests.append(key) or KEY, **kwargs)
    assert requests == []


@pytest.mark.parametrize(
    "period,code",
    [
        (
            MeasureTimeWindow("2026-01-01T00:00:00.1Z", PERIOD.end),
            "unsupported_time_precision",
        ),
        (
            MeasureTimeWindow(PERIOD.start, "9999-12-31T23:59:59Z"),
            "unsupported_time_range",
        ),
    ],
)
def test_unrepresentable_periods_fail_without_rounding(period, code):
    with pytest.raises(QualityMeasureEvidenceError, match=code):
        _packet(_rebind(_run(), measurement_period=period))


def test_instant_period_becomes_one_second_half_open_interval():
    packet = _packet(
        _rebind(
            _run(), measurement_period=MeasureTimeWindow(PERIOD.start, PERIOD.start)
        )
    )
    assert packet.time_boundaries.start == PERIOD.start
    assert packet.time_boundaries.end == "2026-01-01T00:00:01Z"


def test_unsupported_roles_and_non_native_runs_fail_closed():
    run = replace(_run(), subject_results=())
    for populations in [
        tuple(
            p
            for p in run.definition.populations
            if p.kind is not PopulationKind.NUMERATOR
        ),
        run.definition.populations
        + (
            MeasurePopulationDefinition(
                "denominator2", PopulationKind.DENOMINATOR, "native:denominator2"
            ),
        ),
    ]:
        with pytest.raises(
            QualityMeasureEvidenceError, match="unsupported_population_roles"
        ):
            _packet(
                replace(
                    run, definition=replace(run.definition, populations=populations)
                )
            )
    with pytest.raises(QualityMeasureEvidenceError, match="unsupported_run"):
        _packet(replace(run, engine=replace(run.engine, execution_mode="service")))
    with pytest.raises(QualityMeasureEvidenceError, match="invalid_run"):
        _packet("synthetic private value")


def test_independent_native_counts_are_not_silently_adjusted():
    run = _run()
    subjects = tuple(
        replace(
            s,
            populations=tuple(
                replace(p, state=PopulationState.NOT_MET)
                if p.kind is PopulationKind.DENOMINATOR
                else p
                for p in s.populations
            ),
        )
        for s in run.subject_results
    )
    with pytest.raises(QualityMeasureEvidenceError, match="count_out_of_range"):
        _packet(replace(run, subject_results=subjects))


def test_overlapping_exclusions_count_subjects_once_and_keep_exceptions_separate():
    run = _run()
    extra = MeasurePopulationDefinition(
        "exclusion2", PopulationKind.DENOMINATOR_EXCLUSION, "native:exclusion2"
    )
    definition = replace(
        run.definition, populations=run.definition.populations + (extra,)
    )
    subjects = []
    for s in run.subject_results:
        exclusion = next(p for p in s.populations if p.population_id == "exclusion")
        trace = next(t for t in s.trace if t.step_id == "exclusion")
        subjects.append(
            replace(
                s,
                definition_version_id=definition.version_id,
                definition_digest=definition.definition_digest,
                populations=s.populations
                + (replace(exclusion, population_id="exclusion2"),),
                trace=s.trace
                + (
                    replace(
                        trace, step_id="exclusion2", expression_ref="native:exclusion2"
                    ),
                ),
            )
        )
    bound = replace(run, definition=definition, subject_results=tuple(subjects))
    packet = _packet(bound)
    assert packet.result.exclusion_count == 1
    assert sum(e.excluded_count for e in packet.exclusions) == 2


def test_unknown_to_error_changes_result_even_when_met_counts_stay_equal():
    run = _run()
    subjects = []
    for s in run.subject_results:
        populations = tuple(
            replace(p, state=PopulationState.ERROR, error_code="synthetic_error")
            if p.state is PopulationState.UNKNOWN
            else p
            for p in s.populations
        )
        trace = tuple(
            replace(t, state=PopulationState.ERROR, reason_code="synthetic_error")
            if t.state is PopulationState.UNKNOWN
            else t
            for t in s.trace
        )
        subjects.append(replace(s, populations=populations, trace=trace))
    changed = replace(run, subject_results=tuple(subjects))
    before, after = _packet(run), _packet(changed)
    assert before.result.denominator_count == after.result.denominator_count
    assert before.result.numerator_count == after.result.numerator_count
    assert compare_measure_runs(run, changed).result_drift
    report = compare_quality_measure_evidence(before, after)
    assert report.result_changed
    assert {d.source for d in report.drift} == {
        DriftSource.CALCULATION_OUTPUT,
        DriftSource.EXCLUSION_EVIDENCE,
    }


def test_schema_changes_and_key_rotation_are_not_native_measure_drift():
    run = _run()
    baseline = _packet(run)
    schema_change = _packet(
        run, input_schema_digest=canonical_digest({"new_schema": 1})
    )
    assert {
        d.source
        for d in compare_quality_measure_evidence(baseline, schema_change).drift
    } == {DriftSource.INPUT_PROJECTION}
    assert not compare_quality_measure_evidence(baseline, schema_change).result_changed
    rotated = _packet(
        run,
        key_id="synthetic.rotated.key",
        key_provider=lambda _: b"synthetic-rotated-signing-key",
    )
    assert rotated.verify(b"synthetic-rotated-signing-key")
    assert baseline.packet_digest != rotated.packet_digest
    assert compare_quality_measure_evidence(baseline, rotated).is_equivalent
    assert not compare_measure_runs(run, run).semantic_drift


def test_changed_exclusion_overlap_is_additional_packet_result_drift():
    run = _run()
    extra = MeasurePopulationDefinition(
        "exclusion2", PopulationKind.DENOMINATOR_EXCLUSION, "native:exclusion2"
    )
    definition = replace(
        run.definition, populations=run.definition.populations + (extra,)
    )
    runs = []
    for met_subject in (0, 1):
        subjects = []
        for index, s in enumerate(run.subject_results):
            original = next(p for p in s.populations if p.population_id == "exclusion")
            step = next(t for t in s.trace if t.step_id == "exclusion")
            state = (
                PopulationState.MET if index == met_subject else PopulationState.NOT_MET
            )
            subjects.append(
                replace(
                    s,
                    definition_version_id=definition.version_id,
                    definition_digest=definition.definition_digest,
                    populations=s.populations
                    + (replace(original, population_id="exclusion2", state=state),),
                    trace=s.trace
                    + (
                        replace(
                            step,
                            step_id="exclusion2",
                            expression_ref="native:exclusion2",
                            state=state,
                        ),
                    ),
                )
            )
        runs.append(
            replace(run, definition=definition, subject_results=tuple(subjects))
        )
    native = compare_measure_runs(*runs)
    before, after = [_packet(r) for r in runs]
    assert not native.result_drift and not native.semantic_drift
    assert before.result.result_digest == after.result.result_digest
    assert before.result.exclusion_count == 1 and after.result.exclusion_count == 2
    assert compare_quality_measure_evidence(before, after).result_changed


def test_added_zero_count_population_is_additional_packet_result_drift():
    run = replace(_run(), subject_results=())
    observation = MeasurePopulationDefinition(
        "observation", PopulationKind.MEASURE_OBSERVATION, "native:observation"
    )
    candidate = replace(
        run,
        definition=replace(
            run.definition, populations=run.definition.populations + (observation,)
        ),
    )
    native = compare_measure_runs(run, candidate)
    report = compare_quality_measure_evidence(_packet(run), _packet(candidate))
    assert native.semantic_drift and not native.result_drift
    assert set(native.reason_codes) == {"definition_changed"}
    assert report.result_changed
    assert {d.source for d in report.drift} == {
        DriftSource.MEASURE_DEFINITION,
        DriftSource.CALCULATION_LOGIC,
    }
