"""Tests for the gate-regression to rollback decision function."""

from __future__ import annotations

import inspect
import json
from datetime import datetime, timezone

import pytest

from openmed.core.baseline import (
    BASELINE_SCHEMA_VERSION,
    BaselineMiss,
    baseline_key,
)
from openmed.eval.release_gates import (
    G7_RECALL_DROP_LIMIT,
    QUARANTINED,
    RELEASABLE,
    GateCheck,
    GateReport,
)
from openmed.eval.rollout import (
    DECISION_ADVANCE,
    DECISION_HOLD,
    DECISION_ROLLBACK,
    LEAKAGE_METRIC,
    PHASE_CANARY,
    MissingLastGreenError,
    PhaseState,
    RolloutStateError,
    RolloutStateMachine,
    decide_rollback,
)

_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)
_BASELINE_RECALL = 0.990
_BASELINE_LEAKAGE = 0.0010
_CHAMPION = "OpenMed/pii-tiny-mlx-fp-champion"
_CHALLENGER = "OpenMed/pii-tiny-mlx-fp-challenger"

_FAMILY = "pii"
_TIER = "tiny"
_FORMAT = "mlx-fp"
_KEY = baseline_key(_FAMILY, _TIER, _FORMAT)

_PHASE_GATES = ("G1a", "G1b", "G2", "G3", "G7")


def _gate_report(
    *,
    recall: float = _BASELINE_RECALL,
    leakage: float = _BASELINE_LEAKAGE,
    decision: str = RELEASABLE,
    repo_id: str = _CHALLENGER,
) -> GateReport:
    """Build a signed synthetic gate report for the fixture coordinates."""

    return GateReport(
        repo_id=repo_id,
        family=_FAMILY,
        tier=_TIER,
        param_count=1_000_000,
        format=_FORMAT,
        per_label_recall={"PERSON": recall, "SSN": 0.999},
        per_label_precision={"PERSON": 0.999, "SSN": 0.999},
        critical_leakage_count=0 if decision == RELEASABLE else 3,
        residual_leakage_rate=leakage,
        quant_recall_delta=None,
        p50_ms=10.0,
        p95_ms=20.0,
        ram_mb=128.0,
        eval_set_hash="a" * 8,
        leakage_fixture_hash="b" * 8,
        decision=decision,
        gate_results=tuple(
            GateCheck(gate, True, reason="ok", details={}) for gate in _PHASE_GATES
        ),
    )


def _baseline_store() -> dict[str, object]:
    """Return a committed-shape baseline store for the fixture key."""

    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "entries": {
            _KEY: {
                "key": _KEY,
                "family": _FAMILY,
                "tier": _TIER,
                "format": _FORMAT,
                "reproducibility_hash": "sha256:" + "c" * 64,
                "metrics": {
                    "per_label_recall": {
                        "PERSON": _BASELINE_RECALL,
                        "SSN": 0.999,
                    },
                    "residual_leakage_rate": _BASELINE_LEAKAGE,
                },
            }
        },
    }


def _phase_state(
    *,
    family: str = _FAMILY,
    last_green: str | None = _CHAMPION,
) -> PhaseState:
    """Return a CANARY phase state carrying the given last-green pointer."""

    return PhaseState(
        family=family,
        tier=_TIER,
        format=_FORMAT,
        phase=PHASE_CANARY,
        entered_at=_EPOCH,
        target=_CHALLENGER,
        last_green=last_green,
    )


def _machine(*, last_green: str | None = _CHAMPION) -> RolloutStateMachine:
    """Return a machine holding one CANARY key with the given last-green."""

    machine = RolloutStateMachine(clock=lambda: _EPOCH)
    state = _phase_state(last_green=last_green)
    machine.entries[state.key] = state
    return machine


# -- rollback criterion ------------------------------------------------------


def test_over_tolerance_recall_regression_rolls_back_to_last_green() -> None:
    """A recall drop past G7 tolerance rolls back to the committed last-green."""

    report = _gate_report(recall=_BASELINE_RECALL - (G7_RECALL_DROP_LIMIT * 5))
    decision = decide_rollback(report, _baseline_store(), _machine())

    assert decision.decision == DECISION_ROLLBACK
    assert decision.rolled_back is True
    assert decision.target == _CHAMPION
    assert [row.metric for row in decision.audit_record.regressions] == [
        "per_label_recall.PERSON"
    ]


def test_releasable_report_still_rolls_back_on_regression() -> None:
    """ROLLBACK outranks a RELEASABLE gate: a regressing candidate never advances."""

    report = _gate_report(
        recall=_BASELINE_RECALL - (G7_RECALL_DROP_LIMIT * 5),
        decision=RELEASABLE,
    )
    decision = decide_rollback(report, _baseline_store(), _machine())

    assert report.decision == RELEASABLE
    assert decision.decision == DECISION_ROLLBACK
    assert decision.decision != DECISION_ADVANCE


def test_leakage_increase_rolls_back_with_zero_tolerance() -> None:
    """Residual leakage has no tolerance band, matching the G7 check."""

    report = _gate_report(leakage=_BASELINE_LEAKAGE + 1e-6)
    decision = decide_rollback(report, _baseline_store(), _machine())

    assert decision.decision == DECISION_ROLLBACK
    regression = decision.audit_record.regressions[0]
    assert regression.metric == LEAKAGE_METRIC
    assert regression.tolerance == 0.0


# -- gate criterion ----------------------------------------------------------


def test_clean_pass_advances() -> None:
    """A clean, RELEASABLE candidate advances."""

    decision = decide_rollback(_gate_report(), _baseline_store(), _machine())

    assert decision.decision == DECISION_ADVANCE
    assert decision.target is None
    assert decision.audit_record.regressions == ()


def test_borderline_within_tolerance_advances() -> None:
    """A drop just inside the tolerance is not a regression."""

    report = _gate_report(recall=_BASELINE_RECALL - (G7_RECALL_DROP_LIMIT * 0.9))
    decision = decide_rollback(report, _baseline_store(), _machine())

    assert decision.decision == DECISION_ADVANCE
    assert decision.audit_record.regressions == ()


def test_tolerance_boundary_uses_the_same_operator_as_g7() -> None:
    """The tolerance is a strict ``>``, bit-for-bit with ``_g7_check``.

    ``G7_RECALL_DROP_LIMIT`` is not exactly representable in binary, so a drop
    built by subtracting it can land marginally *above* the limit and count as
    a regression. That is deliberate: the rollback decision reproduces the gate
    check's comparison exactly rather than softening it with an epsilon, so the
    two can never disagree about the same candidate.
    """

    just_over = _BASELINE_RECALL - G7_RECALL_DROP_LIMIT
    drop = _BASELINE_RECALL - just_over
    assert drop > G7_RECALL_DROP_LIMIT  # the float artefact itself

    decision = decide_rollback(
        _gate_report(recall=just_over),
        _baseline_store(),
        _machine(),
    )
    assert decision.decision == DECISION_ROLLBACK
    assert decision.audit_record.regressions[0].drop == drop


def test_recall_tolerance_is_not_caller_configurable() -> None:
    """Callers cannot silently loosen the release-gate tolerance."""

    assert "recall_tolerance" not in inspect.signature(decide_rollback).parameters


def test_advance_requires_a_releasable_report() -> None:
    """A non-releasable report with no regression holds rather than advancing."""

    report = _gate_report(decision=QUARANTINED)
    decision = decide_rollback(report, _baseline_store(), _machine())

    assert decision.decision == DECISION_HOLD
    assert decision.target is None


def test_unknown_gate_decision_is_refused() -> None:
    """An unrecognised gate decision is refused rather than silently held."""

    report = _gate_report()
    report.decision = "MAYBE"

    with pytest.raises(RolloutStateError, match="unknown gate decision"):
        decide_rollback(report, _baseline_store(), _machine())


# -- reproducibility criterion -----------------------------------------------


def test_decision_is_reproducible_from_committed_state() -> None:
    """The same report, baseline and rollout state yield an identical record."""

    report = _gate_report(recall=_BASELINE_RECALL - (G7_RECALL_DROP_LIMIT * 5))
    first = decide_rollback(report, _baseline_store(), _machine())
    second = decide_rollback(report, _baseline_store(), _machine())

    assert first.audit_record.to_dict() == second.audit_record.to_dict()
    assert first.to_dict() == second.to_dict()


def test_decision_reads_baseline_from_a_committed_path(tmp_path) -> None:
    """The baseline resolves from a committed JSON path with no live call."""

    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(_baseline_store()), encoding="utf-8")
    report = _gate_report(recall=_BASELINE_RECALL - (G7_RECALL_DROP_LIMIT * 5))

    decision = decide_rollback(report, None, _machine(), baseline_path=path)

    assert decision.decision == DECISION_ROLLBACK
    assert decision.audit_record.baseline_key == _KEY


def test_decision_does_not_mutate_the_rollout_state() -> None:
    """The function is side-effect-free: no phase change, no audit append."""

    machine = _machine()
    before = machine.phase_state(_FAMILY, _TIER, _FORMAT)
    report = _gate_report(recall=_BASELINE_RECALL - (G7_RECALL_DROP_LIMIT * 5))

    decide_rollback(report, _baseline_store(), machine)

    after = machine.phase_state(_FAMILY, _TIER, _FORMAT)
    assert after == before
    assert after.phase == PHASE_CANARY
    assert after.target == _CHALLENGER
    assert machine.audit_records == []


def test_tampered_report_hash_is_refused() -> None:
    """A report whose evidence no longer matches its hash cannot decide."""

    report = _gate_report()
    report.repro_hash = "sha256:" + "0" * 64

    with pytest.raises(RolloutStateError, match="reproducibility hash is invalid"):
        decide_rollback(report, _baseline_store(), _machine())


# -- rollback target ---------------------------------------------------------


def test_rollback_without_last_green_raises() -> None:
    """A rollback with no committed last-green refuses rather than guessing."""

    report = _gate_report(recall=_BASELINE_RECALL - (G7_RECALL_DROP_LIMIT * 5))

    with pytest.raises(MissingLastGreenError, match="last-green target is missing"):
        decide_rollback(report, _baseline_store(), _machine(last_green=None))


def test_rollback_without_a_seeded_key_raises() -> None:
    """A rollback for an unseeded rollout key refuses."""

    report = _gate_report(recall=_BASELINE_RECALL - (G7_RECALL_DROP_LIMIT * 5))

    with pytest.raises(MissingLastGreenError, match="seeded rollout key"):
        decide_rollback(report, _baseline_store(), RolloutStateMachine())


def test_target_is_returned_verbatim_not_parsed() -> None:
    """The target is the committed pointer string, never parsed from a name."""

    pointer = "OpenMed/OpenMed-PII-Arabic-BigMed-Large-560M-v1"
    report = _gate_report(recall=_BASELINE_RECALL - (G7_RECALL_DROP_LIMIT * 5))

    decision = decide_rollback(
        report,
        _baseline_store(),
        _machine(last_green=pointer),
    )

    assert decision.target == pointer
    assert decision.audit_record.target == pointer


def test_mismatched_phase_state_coordinates_are_refused() -> None:
    """A phase state for another key cannot supply this report's target."""

    state = _phase_state(family="ner")

    with pytest.raises(RolloutStateError, match="do not match"):
        decide_rollback(_gate_report(), _baseline_store(), state)


# -- audit record ------------------------------------------------------------


def test_audit_record_contains_no_raw_phi() -> None:
    """The audit record carries names, numbers, keys and hashes only."""

    report = _gate_report(recall=_BASELINE_RECALL - (G7_RECALL_DROP_LIMIT * 5))
    decision = decide_rollback(report, _baseline_store(), _machine())
    payload = decision.audit_record.to_dict()

    # Every string in the record must be a metric name, a label, a coordinate
    # key, a repo id, a decision, or a hash -- never evaluated text.
    allowed = {
        _KEY,
        _CHAMPION,
        _CHALLENGER,
        DECISION_ROLLBACK,
        LEAKAGE_METRIC,
        "per_label_recall.PERSON",
        "per_label_recall.SSN",
        report.repro_hash,
        "higher_is_better",
        "lower_is_better",
    }
    for value in _strings(payload):
        assert value in allowed, f"unexpected string in audit record: {value!r}"

    assert isinstance(json.dumps(payload), str)


def test_missing_baseline_entry_refuses_rather_than_advancing() -> None:
    """No committed baseline for the key means no decision, not a clean pass."""

    empty = {"schema_version": BASELINE_SCHEMA_VERSION, "entries": {}}

    with pytest.raises(BaselineMiss, match="No last-green baseline"):
        decide_rollback(_gate_report(), empty, _machine())


def test_disjoint_baseline_metrics_are_visible_as_an_empty_comparison() -> None:
    """A baseline tracking other metrics scores nothing, and says so.

    The committed store is not one namespace: gate keys such as
    ``i18n-throughput::hi::pattern-only`` carry throughput metrics that share no
    axis with a release gate's recall and leakage. Such an entry must not read
    as a clean bill of health, so the audit record reports an empty compared
    surface rather than an implicit pass.
    """

    store = _baseline_store()
    store["entries"][_KEY]["metrics"] = {
        "deidentify_spans_per_second": 70.0,
        "segmentation_chars_per_second": 250000.0,
    }

    decision = decide_rollback(_gate_report(), store, _machine())

    assert decision.audit_record.compared_metrics == ()
    assert decision.audit_record.regressions == ()
    assert decision.decision == DECISION_HOLD
    assert "no monitored metrics overlap" in decision.reasons[0]


def test_audit_record_records_the_compared_metric_surface() -> None:
    """The record names every metric actually compared, so a no-op is visible."""

    decision = decide_rollback(_gate_report(), _baseline_store(), _machine())

    assert decision.audit_record.compared_metrics == (
        "per_label_recall.PERSON",
        "per_label_recall.SSN",
        LEAKAGE_METRIC,
    )


def test_audit_record_binds_the_recomputed_report_hash() -> None:
    """The record carries the recomputed hash, not a caller-supplied value."""

    report = _gate_report()
    decision = decide_rollback(report, _baseline_store(), _machine())

    assert decision.audit_record.report_hash == report.recompute_repro_hash()
    assert decision.audit_record.repo_id == _CHALLENGER
    assert decision.audit_record.key == _KEY


def _strings(value: object) -> list[str]:
    """Return every string *value* in a JSON-compatible payload.

    Mapping keys are skipped: they are this module's own schema field names,
    not data carried over from an evaluation.
    """

    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        rows: list[str] = []
        for item in value.values():
            rows.extend(_strings(item))
        return rows
    if isinstance(value, list):
        rows = []
        for item in value:
            rows.extend(_strings(item))
        return rows
    return []
