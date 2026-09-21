"""Evidence-bound descriptive drug-safety workflow tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema.validators import validator_for

from openmed.clinical.drug_safety import (
    DescriptiveSignal,
    DrugExposure,
    DrugSafetyConflictError,
    DrugSafetyDataset,
    ExposureWindow,
    NormalizedSafetyTerm,
    OpenEventDatasetAdapter,
    SafetyCase,
    SafetySeriousness,
    SignalContingencyTable,
    SignalFilter,
    SignalState,
    SuspectedDrugEventRelation,
    compute_descriptive_signal,
    import_open_event_csv,
    load_drug_safety_signal_schema,
)
from openmed.clinical.journey_contracts import derived_opaque_id, sha256_digest
from openmed.eval.suites.drug_safety import (
    DrugSafetyBenchmarkCase,
    run_drug_safety_benchmark,
)

FIXTURES = Path(__file__).parents[3] / "fixtures" / "clinical" / "drug_safety"


def _dataset() -> DrugSafetyDataset:
    content = (FIXTURES / "synthetic_events.csv").read_bytes()
    return import_open_event_csv(
        FIXTURES / "synthetic_events.csv",
        dataset_id="synthetic_public_events",
        version="2026-09-21",
        source_digest=sha256_digest(content),
        license_id="synthetic-test-data",
    )


def _rows(
    *,
    a: int,
    b: int,
    c: int,
    d: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    cells = (
        (a, "Drug A", "Event X"),
        (b, "Drug A", "Event Y"),
        (c, "Drug B", "Event X"),
        (d, "Drug B", "Event Y"),
    )
    index = 0
    for count, drug, event in cells:
        for _ in range(count):
            index += 1
            rows.append(
                {
                    "report_id": f"S{index:03d}",
                    "drug": drug,
                    "event": event,
                    "seriousness": "serious",
                    "exposure_start_day": "0",
                    "exposure_end_day": "30",
                }
            )
    return rows


def _import_rows(rows: list[dict[str, str]]) -> DrugSafetyDataset:
    return OpenEventDatasetAdapter().import_rows(
        rows,
        dataset_id="synthetic_rows",
        version="1",
        source_digest="sha256:" + "a" * 64,
        license_id="synthetic-test-data",
    )


def test_csv_adapter_deduplicates_rows_and_removes_source_report_ids() -> None:
    dataset = _dataset()
    assert dataset.imported_row_count == 21
    assert dataset.duplicate_row_count == 1
    assert len(dataset.cases) == 20
    serialized = json.dumps([item.to_dict() for item in dataset.cases], sort_keys=True)
    assert "R01" not in serialized
    assert "report_id" not in serialized
    assert all(case.case_id.startswith("safetycase_") for case in dataset.cases)


def test_hand_computed_table_ratios_schema_and_round_trip() -> None:
    signal = compute_descriptive_signal(_dataset(), drug="Drug A", event="Event X")
    assert signal.state is SignalState.COMPUTED
    assert signal.table == SignalContingencyTable(4, 6, 2, 8)
    assert signal.proportional_reporting_ratio == pytest.approx(2.0)
    assert signal.reporting_odds_ratio == pytest.approx(8 / 3)
    assert signal.reason_codes == ()
    assert "signal_is_not_causal" in signal.caveats
    assert signal.license_id == "synthetic-test-data"
    assert signal.filter.version == "1.0.0"
    assert signal.method_id == "report_level_prr_ror"
    assert DescriptiveSignal.from_dict(signal.to_dict()) == signal

    schema = load_drug_safety_signal_schema()
    validator_for(schema).check_schema(schema)
    assert not list(validator_for(schema)(schema).iter_errors(signal.to_dict()))


def test_signal_round_trip_rejects_changed_digest_and_identifier() -> None:
    signal = compute_descriptive_signal(_dataset(), drug="Drug A", event="Event X")
    changed_digest = signal.to_dict()
    changed_digest["signal_digest"] = "sha256:" + "0" * 64
    with pytest.raises(DrugSafetyConflictError, match="signal digest"):
        DescriptiveSignal.from_dict(changed_digest)

    changed_id = signal.to_dict()
    changed_id["signal_id"] = derived_opaque_id("safetysignal", "different")
    with pytest.raises(DrugSafetyConflictError, match="identifier custody"):
        DescriptiveSignal.from_dict(changed_id)

    with pytest.raises(DrugSafetyConflictError, match="statistics differ"):
        replace(signal, proportional_reporting_ratio=99.0)


def test_small_and_zero_cells_are_explicitly_suppressed() -> None:
    signal = compute_descriptive_signal(
        _import_rows(_rows(a=3, b=3, c=0, d=3)),
        drug="Drug A",
        event="Event X",
    )
    assert signal.state is SignalState.SUPPRESSED
    assert signal.proportional_reporting_ratio is None
    assert signal.reporting_odds_ratio is None
    assert signal.reason_codes == (
        "cell_count_below_minimum",
        "zero_cell_ratio_undefined",
    )

    small = compute_descriptive_signal(
        _import_rows(_rows(a=1, b=4, c=2, d=4)),
        drug="Drug A",
        event="Event X",
    )
    assert small.state is SignalState.SUPPRESSED
    assert "pair_count_below_minimum" in small.reason_codes


def test_missing_denominators_and_empty_filters_are_not_computed() -> None:
    missing_comparator = compute_descriptive_signal(
        _import_rows(_rows(a=3, b=3, c=0, d=0)),
        drug="Drug A",
        event="Event X",
    )
    assert missing_comparator.state is SignalState.INSUFFICIENT_DATA
    assert missing_comparator.reason_codes == ("comparator_denominator_missing",)

    missing_drug = compute_descriptive_signal(
        _dataset(), drug="Absent Drug", event="Event X"
    )
    assert missing_drug.state is SignalState.INSUFFICIENT_DATA
    assert missing_drug.reason_codes == ("drug_denominator_missing",)

    empty = compute_descriptive_signal(
        _dataset(),
        drug="Drug A",
        event="Event X",
        signal_filter=SignalFilter(seriousness=(SafetySeriousness.NON_SERIOUS,)),
    )
    assert empty.state is SignalState.INSUFFICIENT_DATA
    assert empty.reason_codes == ("filtered_population_empty",)


def test_source_digest_and_conflicting_seriousness_fail_closed(tmp_path: Path) -> None:
    source = FIXTURES / "synthetic_events.csv"
    with pytest.raises(DrugSafetyConflictError, match="source digest"):
        import_open_event_csv(
            source,
            dataset_id="synthetic_public_events",
            version="2026-09-21",
            source_digest="sha256:" + "0" * 64,
            license_id="synthetic-test-data",
        )

    rows = _rows(a=1, b=1, c=1, d=1)
    rows.append({**rows[0], "event": "Event Z", "seriousness": "non_serious"})
    with pytest.raises(DrugSafetyConflictError, match="conflicting seriousness"):
        _import_rows(rows)

    unreadable = tmp_path / "missing.csv"
    with pytest.raises(ValueError, match="cannot be read"):
        import_open_event_csv(
            unreadable,
            dataset_id="synthetic_public_events",
            version="1",
            source_digest="sha256:" + "0" * 64,
            license_id="synthetic-test-data",
        )

    malformed = tmp_path / "malformed.csv"
    malformed.write_text("report_id,drug\nR1,Drug A\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        import_open_event_csv(
            malformed,
            dataset_id="synthetic_public_events",
            version="1",
            source_digest=sha256_digest(malformed.read_bytes()),
            license_id="synthetic-test-data",
        )


def test_chart_suspicion_has_no_population_statistics() -> None:
    relation = SuspectedDrugEventRelation(
        relation_id=derived_opaque_id("safetyrelation", "relation"),
        snapshot_id=derived_opaque_id("snapshot", "snapshot"),
        snapshot_digest="sha256:" + "1" * 64,
        drug_fact_ids=(derived_opaque_id("fact", "drug"),),
        event_fact_ids=(derived_opaque_id("fact", "event"),),
        evidence_ids=(derived_opaque_id("evidence", "source"),),
        reason_code="temporal_association_suspected",
    )
    payload = relation.to_dict()
    assert payload["artifact_type"] == "suspected_chart_drug_event_relation"
    assert payload["review_required"] is True
    assert payload["schema_version"] == "1.0.0"
    assert "proportional_reporting_ratio" not in payload
    assert "reporting_odds_ratio" not in payload
    assert SuspectedDrugEventRelation.from_dict(payload) == relation


def test_term_normalization_preserves_unicode_letters() -> None:
    assert NormalizedSafetyTerm(kind="drug", code="  MÉTOPROLOL  ").code == "métoprolol"
    assert NormalizedSafetyTerm(kind="event", code="恶心").code == "恶心"


def test_exposure_ordering_accepts_known_and_unknown_windows() -> None:
    drug = NormalizedSafetyTerm(kind="drug", code="Drug A")
    case = SafetyCase(
        case_id=derived_opaque_id("safetycase", "ordering"),
        exposures=(
            DrugExposure(drug=drug),
            DrugExposure(drug=drug, window=ExposureWindow(0, 30)),
        ),
        events=(NormalizedSafetyTerm(kind="event", code="Event X"),),
        seriousness=SafetySeriousness.UNKNOWN,
        evidence=(
            # The fixture importer exercises the concrete evidence contract.
            _dataset().cases[0].evidence[0],
        ),
    )
    assert case.exposures[0].window is not None
    assert case.exposures[1].window is None


def test_frozen_benchmark_reproduces_counts_and_ratios() -> None:
    case = DrugSafetyBenchmarkCase(
        case_id="hand_checked_primary",
        drug="Drug A",
        event="Event X",
        expected_state=SignalState.COMPUTED,
        expected_table=SignalContingencyTable(4, 6, 2, 8),
        expected_prr=2.0,
        expected_ror=8 / 3,
    )
    report = run_drug_safety_benchmark(_dataset(), (case,))
    assert report.case_count == 1
    assert report.exact_table_accuracy == 1.0
    assert report.exact_state_accuracy == 1.0
    assert report.maximum_absolute_ratio_error == 0.0
    assert report.dataset_digest == _dataset().dataset_digest
    assert report.versions["adapter"].startswith("openmed.open-event/")
