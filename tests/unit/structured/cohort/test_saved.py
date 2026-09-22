"""Saved-cohort version, execution, persistence, and rerun tests."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from jsonschema.validators import validator_for

from openmed.structured.cohort import (
    CohortExecution,
    CohortMembership,
    CohortSourceSnapshot,
    CriterionMembership,
    LocalSavedCohortStore,
    MembershipEvidence,
    MembershipState,
    PhenotypeDefinition,
    SavedCohortConflictError,
    build_cohort_execution,
    load_saved_cohort_schema,
    save_cohort_definition,
)
from openmed.structured.store import DenyStorageOperations, StoreState

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "cohort" / "phenotypes"


def _definition() -> PhenotypeDefinition:
    return PhenotypeDefinition.load(FIXTURES / "diabetes_on_metformin.json")


def _definition_without_metformin() -> PhenotypeDefinition:
    return PhenotypeDefinition.load(FIXTURES / "diabetes_without_metformin.json")


def _snapshot(suffix: str = "a", *, bundled: bool = False) -> CohortSourceSnapshot:
    return CohortSourceSnapshot(
        snapshot_id=f"snapshot_{suffix * 16}",
        digest="sha256:" + suffix * 64,
        schema_version="journey-synthetic-v1",
        license_tags=("synthetic",),
        bundled_vocabulary=bundled,
    )


def _criterion(
    criterion_id: str,
    state: MembershipState,
    suffix: str,
) -> CriterionMembership:
    evidence = ()
    if state is MembershipState.MET:
        evidence = (
            MembershipEvidence(
                evidence_id=f"evidence_{suffix * 16}",
                fact_id=f"fact_{suffix * 16}",
                time_window_id=f"window_{suffix * 16}",
            ),
        )
    reasons = (
        ("evidence_missing",)
        if state is MembershipState.UNKNOWN
        else (("evidence_conflict",) if state is MembershipState.CONFLICT else ())
    )
    return CriterionMembership(
        criterion_id=criterion_id,
        state=state,
        evidence=evidence,
        reason_codes=reasons,
    )


def _membership(
    patient_suffix: str = "p",
    *,
    first: MembershipState = MembershipState.MET,
    second: MembershipState = MembershipState.MET,
) -> CohortMembership:
    if MembershipState.CONFLICT in {first, second}:
        aggregate = MembershipState.CONFLICT
    elif MembershipState.UNKNOWN in {first, second}:
        aggregate = MembershipState.UNKNOWN
    elif MembershipState.NOT_MET in {first, second}:
        aggregate = MembershipState.NOT_MET
    else:
        aggregate = MembershipState.MET
    return CohortMembership(
        patient_key=f"patient_{patient_suffix * 16}",
        state=aggregate,
        criteria=(
            _criterion("has-diabetes", first, "d"),
            _criterion("has-metformin", second, "m"),
        ),
    )


def _execution(
    memberships: tuple[CohortMembership, ...] | None = None,
    *,
    snapshot: CohortSourceSnapshot | None = None,
) -> CohortExecution:
    result = build_cohort_execution(
        save_cohort_definition(_definition()),
        source_snapshot=snapshot or _snapshot(),
        vocabulary_digest="sha256:" + "b" * 64,
        policy_digest="sha256:" + "c" * 64,
        evaluator_version="cohort-evaluator-1.0",
        memberships=memberships or (_membership(),),
    )
    assert result.state is StoreState.SUCCESS
    assert result.value is not None
    return result.value


def test_versions_and_executions_are_byte_stable_and_schema_valid() -> None:
    version = save_cohort_definition(_definition())
    execution = _execution()

    assert version.to_json_bytes() == version.to_json_bytes()
    assert CohortExecution.from_json(execution.to_json_bytes()) == execution
    assert (
        CohortExecution.from_json(execution.to_json()).to_json() == execution.to_json()
    )

    schema = load_saved_cohort_schema()
    validator = validator_for(schema)
    validator.check_schema(schema)
    assert not tuple(validator(schema).iter_errors(version.to_dict()))
    assert not tuple(validator(schema).iter_errors(execution.to_dict()))


@given(st.permutations(("p", "q", "r")))
def test_membership_order_cannot_change_run_digest(order: list[str]) -> None:
    memberships = tuple(_membership(item) for item in order)
    execution = _execution(memberships)
    canonical = _execution(tuple(reversed(memberships)))

    assert execution.manifest.execution_id == canonical.manifest.execution_id
    assert execution.membership_digest == canonical.membership_digest
    assert execution.to_json_bytes() == canonical.to_json_bytes()


def test_definition_or_snapshot_change_creates_a_different_run() -> None:
    first = _execution()
    second = _execution(snapshot=_snapshot("e"))
    changed = save_cohort_definition(
        replace(_definition(), name="Synthetic revised definition")
    )
    changed_result = build_cohort_execution(
        changed,
        source_snapshot=_snapshot(),
        vocabulary_digest="sha256:" + "b" * 64,
        policy_digest="sha256:" + "c" * 64,
        evaluator_version="cohort-evaluator-1.0",
        memberships=(_membership(),),
    )

    assert changed_result.value is not None
    assert first.manifest.execution_id != second.manifest.execution_id
    assert first.manifest.execution_id != changed_result.value.manifest.execution_id


def test_definition_expression_not_flat_criterion_order_determines_membership() -> None:
    membership = CohortMembership(
        patient_key="patient_nnnnnnnnnnnnnnnn",
        state=MembershipState.MET,
        criteria=(
            _criterion("has-diabetes", MembershipState.MET, "d"),
            _criterion("has-metformin", MembershipState.NOT_MET, "m"),
        ),
    )
    result = build_cohort_execution(
        save_cohort_definition(_definition_without_metformin()),
        source_snapshot=_snapshot(),
        vocabulary_digest="sha256:" + "b" * 64,
        policy_digest="sha256:" + "c" * 64,
        evaluator_version="cohort-evaluator-1.0",
        memberships=(membership,),
    )

    assert result.state is StoreState.SUCCESS
    assert result.value is not None
    assert result.value.memberships[0].eligible


@pytest.mark.parametrize(
    ("state", "reason"),
    [
        (MembershipState.UNKNOWN, "evidence_missing"),
        (MembershipState.CONFLICT, "evidence_conflict"),
    ],
)
def test_unresolved_memberships_require_review_and_cannot_be_eligible(
    state: MembershipState,
    reason: str,
) -> None:
    membership = _membership(second=state)
    execution = _execution((membership,))

    assert execution.review_required
    assert membership.review_required
    assert not membership.eligible
    assert membership.criteria[1].reason_codes == (reason,)

    with pytest.raises(
        SavedCohortConflictError,
        match="unresolved criterion cannot produce eligible membership",
    ):
        replace(membership, state=MembershipState.MET)


@pytest.mark.parametrize("fchmod_available", [True, False])
def test_local_store_is_immutable_idempotent_and_referential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fchmod_available: bool
) -> None:
    if not fchmod_available:
        monkeypatch.delattr(os, "fchmod", raising=False)
    store = LocalSavedCohortStore(tmp_path / "saved-cohorts")
    version = save_cohort_definition(_definition())
    execution = _execution()

    first = store.put_definition(version)
    second = store.put_definition(version)
    saved = store.put_execution(execution)

    assert first.ok and first.created
    assert second.ok and not second.created
    assert saved.ok and saved.created
    assert store.get_definition(version.version_id or "").value == version
    assert store.get_execution(execution.manifest.execution_id or "").value == execution
    if os.name == "posix":
        assert oct((tmp_path / "saved-cohorts").stat().st_mode & 0o777) == "0o700"

    drifted = CohortExecution(
        manifest=execution.manifest,
        memberships=(_membership(second=MembershipState.NOT_MET),),
    )
    conflict = store.put_execution(drifted)
    assert conflict.state is StoreState.CONFLICT
    assert conflict.code == "immutable_record_conflict"

    missing_store = LocalSavedCohortStore(tmp_path / "missing-definition")
    missing = missing_store.put_execution(execution)
    assert missing.state is StoreState.UNKNOWN
    assert missing.code == "definition_not_available"


def test_store_rejects_valid_definition_under_a_different_id(tmp_path: Path) -> None:
    store = LocalSavedCohortStore(tmp_path / "saved-cohorts")
    expected = save_cohort_definition(_definition())
    other = save_cohort_definition(_definition_without_metformin())
    assert store.put_definition(expected).ok
    path = store.root / "definitions" / f"{expected.version_id}.json"
    path.write_bytes(other.to_json_bytes())
    result = store.get_definition(expected.version_id or "")
    assert result.state is StoreState.CONFLICT
    assert result.code == "definition_integrity_failed"


def test_store_rejects_valid_execution_under_a_different_id(tmp_path: Path) -> None:
    store = LocalSavedCohortStore(tmp_path / "saved-cohorts")
    assert store.put_definition(save_cohort_definition(_definition())).ok
    expected = _execution()
    other = _execution(snapshot=_snapshot("b"))
    assert store.put_execution(expected).ok
    path = store.root / "executions" / f"{expected.manifest.execution_id}.json"
    path.write_bytes(other.to_json_bytes())
    result = store.get_execution(expected.manifest.execution_id or "")
    assert result.state is StoreState.CONFLICT
    assert result.code == "execution_integrity_failed"


def test_rerun_proves_reproducibility_and_reports_drift(tmp_path: Path) -> None:
    store = LocalSavedCohortStore(tmp_path / "saved-cohorts")
    version = save_cohort_definition(_definition())
    execution = _execution()
    assert store.put_definition(version).ok
    assert store.put_execution(execution).ok

    stable = store.rerun(
        execution.manifest.execution_id or "",
        lambda context: execution.memberships,
    )
    drifted = store.rerun(
        execution.manifest.execution_id or "",
        lambda context: (_membership(second=MembershipState.NOT_MET),),
    )

    assert stable.state is StoreState.SUCCESS
    assert stable.value is not None
    assert stable.value.membership_digest == execution.membership_digest
    assert drifted.state is StoreState.CONFLICT
    assert drifted.code == "cohort_rerun_digest_mismatch"

    def failed_evaluator(context: object) -> tuple[CohortMembership, ...]:
        raise RuntimeError("synthetic evaluator failure")

    failed = store.rerun(
        execution.manifest.execution_id or "",
        failed_evaluator,
    )
    assert failed.state is StoreState.FAILURE
    assert failed.code == "cohort_rerun_failed"


def test_tampering_is_detected_without_echoing_values() -> None:
    payload = json.loads(_execution().to_json())
    payload["membership_digest"] = "sha256:" + "f" * 64

    with pytest.raises(SavedCohortConflictError, match="membership digest differs"):
        CohortExecution.from_dict(payload)


def test_restricted_vocabulary_content_cannot_be_bundled() -> None:
    with pytest.raises(Exception, match="cannot bundle vocabulary"):
        _snapshot(bundled=True)


def test_only_opaque_patient_and_evidence_keys_are_accepted() -> None:
    with pytest.raises(Exception, match="patient_key must be opaque"):
        replace(_membership(), patient_key="person@example.test")
    with pytest.raises(Exception, match="evidence_id must be opaque"):
        MembershipEvidence(evidence_id="raw chart evidence")


def test_policy_denial_is_typed(tmp_path: Path) -> None:
    denied = LocalSavedCohortStore(
        tmp_path / "denied",
        policy=DenyStorageOperations(frozenset({"write"})),
    )
    result = denied.put_definition(save_cohort_definition(_definition()))

    assert result.state is StoreState.DENIED
    assert result.code == "policy_denied"


def test_persisted_artifacts_exclude_source_text_and_direct_identifiers() -> None:
    serialized = _execution().to_json()

    for forbidden in (
        "person@example.test",
        "synthetic patient name",
        "raw_text",
        "source_text",
    ):
        assert forbidden not in serialized
