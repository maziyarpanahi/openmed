"""Synthetic safety and recovery tests for sealed evaluation sessions."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from openmed.eval.governance import (
    ComparisonCase,
    ConflictOfInterestMetadata,
    FeedbackBudgetLedger,
    FeedbackBudgetPolicy,
    RubricCriterion,
    SourceEvidence,
    commit_holdout_manifests,
)
from openmed.eval.workflows.sealed_manifest import (
    SEALED_MANIFEST_COMPONENTS,
    seal_workflow_manifest,
)
from openmed.eval.workflows.session import (
    EvaluationSession,
    EvaluationSessionError,
    SessionRecord,
    SQLiteSessionStore,
    evaluation_case_digest,
)

CANARY = "SYNTHETIC-PATIENT-ALICE-SECRET-4901"
CANDIDATE = "SYNTHETIC-CANDIDATE-SECRET-1729"


def digest(value):
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


@pytest.fixture
def setup_session(tmp_path):
    components = {name: digest(name) for name in SEALED_MANIFEST_COMPONENTS}
    manifest = seal_workflow_manifest(components)
    cases = tuple(
        ComparisonCase(
            case_ref=f"case-{i}",
            source_evidence=(SourceEvidence(f"source-{i}", f"{CANARY}-{i}"),),
            candidate_outputs={"evaluated": "Placeholder", "reference": CANDIDATE},
        )
        for i in range(2)
    )
    manifests = {
        "case": tuple(evaluation_case_digest(c) for c in cases),
        "label": (digest("synthetic-labels"),),
        "template": (digest("synthetic-template"),),
        "randomization": (digest("synthetic-randomization"),),
    }
    holdout = commit_holdout_manifests("synthetic-epoch-1", manifests)
    policy = FeedbackBudgetPolicy(holdout.commitment_digest, 1, 2, (0.5, 0.8))
    ledger = FeedbackBudgetLedger([policy])
    path = tmp_path / "sessions.sqlite"
    store = SQLiteSessionStore(path)
    events = []

    def runner(evidence):
        events.append("runner")
        assert isinstance(evidence, tuple)
        assert all(isinstance(e, SourceEvidence) for e in evidence)
        return f"{CANDIDATE} output {evidence[0].evidence_ref}"

    def scorer(case, output):
        events.append("scorer")
        return 0.75 if case.case_ref == "case-0" else 0.25

    def review(packets):
        events.append("review")
        assert len(packets) == 2
        assert all("evaluated" not in p.to_json() for p in packets)

    args = dict(
        session_key=digest("session-one"),
        manifest=manifest,
        component_snapshot=lambda: components,
        holdout=holdout,
        holdout_manifests=manifests,
        cases=cases,
        candidate_identity="evaluated",
        candidate_manifests={
            "evaluated": manifest.manifest_digest,
            "reference": digest("reference-manifest"),
        },
        runner=runner,
        scorer=scorer,
        review=review,
        provider_digests={
            name: digest(name) for name in ("runner", "scorer", "review")
        },
        rubric=(RubricCriterion("grounding", "Synthetic grounding?", 1, 5),),
        conflict_of_interest=ConflictOfInterestMetadata("reviewer-1", digest("coi")),
        randomization_key=b"synthetic-randomization-key-000001",
        public_items=("Unrelated public fixture",),
        shadow_items=(CANARY,),
        canaries=(CANARY,),
        ledger=ledger,
        store=store,
        clock=lambda: 17,
    )
    yield args, events, path
    store.close()


def prepare(session):
    session.verify()
    session.run_cases()
    session.run_forensics()
    session.build_adjudication()


def assert_code(code, callback):
    with pytest.raises(EvaluationSessionError) as error:
        callback()
    assert error.value.code == code
    assert str(error.value) == code


def test_order_guards_and_runner_sees_no_labels(setup_session):
    args, events, _ = setup_session
    session = EvaluationSession(**args)
    assert_code("verification_required", session.run_cases)
    assert_code("forensics_required", session.release_feedback)
    assert_code("forensics_required", session.build_adjudication)
    assert not events
    session.verify()
    session.run_cases()
    assert_code("forensics_required", session.release_feedback)
    session.run_forensics()
    assert_code("adjudication_required", session.release_feedback)
    session.build_adjudication()
    feedback = session.release_feedback()
    assert feedback.per_case_scores == (0.75, 0.25)
    assert feedback.decision.detailed_score == 0.5
    assert events == ["runner", "scorer", "runner", "scorer", "review"]
    assert_code("feedback_already_claimed", session.release_feedback)
    assert_code("feedback_already_claimed", lambda: EvaluationSession(**args))


@pytest.mark.parametrize("phase", ["before_verify", "after_verify"])
@pytest.mark.parametrize("component", SEALED_MANIFEST_COMPONENTS)
def test_component_drift_stops_all_case_execution(setup_session, phase, component):
    args, events, _ = setup_session
    components = dict(args["manifest"].component_digests)
    args["component_snapshot"] = lambda: components
    session = EvaluationSession(**args)
    if phase == "after_verify":
        session.verify()
    components[component] = digest("changed")
    assert_code(
        "component_digest_mismatch",
        session.run_cases if phase == "after_verify" else session.verify,
    )
    assert not events


def test_commitment_binds_actual_cases_and_all_four_manifests(setup_session):
    args, events, _ = setup_session
    changed = dict(args["holdout_manifests"])
    changed["label"] = (digest("changed-label"),)
    session = EvaluationSession(**{**args, "holdout_manifests": changed})
    assert_code("holdout_commitment_mismatch", session.verify)
    assert not events


def test_uncommitted_case_never_reaches_runner(setup_session):
    args, events, _ = setup_session
    cases = (
        replace(
            args["cases"][0],
            source_evidence=(
                SourceEvidence("replacement", "Synthetic uncommitted replacement"),
            ),
        ),
        args["cases"][1],
    )
    session = EvaluationSession(**{**args, "cases": cases})
    assert_code("holdout_commitment_mismatch", session.verify)
    assert not events


def test_budget_applies_once_per_session_and_hides_per_case_scores(setup_session):
    args, _, _ = setup_session
    first = EvaluationSession(**args).run()
    second = EvaluationSession(**{**args, "session_key": digest("session-two")}).run()
    third = EvaluationSession(**{**args, "session_key": digest("session-three")}).run()
    assert first.per_case_scores == (0.75, 0.25)
    assert second.per_case_scores is third.per_case_scores is None
    assert second.decision.coarse_band == third.decision.coarse_band == 1
    assert second.decision.detailed_score is None
    fourth = EvaluationSession(**{**args, "session_key": digest("session-four")})
    prepare(fourth)
    assert_code("rerun_budget_exhausted", fourth.release_feedback)
    assert fourth.record.steps[-1][0] == "budget_denied"
    assert_code("feedback_already_claimed", fourth.release_feedback)
    assert (
        args["ledger"]
        .snapshot(args["holdout"].commitment_digest, args["manifest"].manifest_digest)
        .official_attempts
        == 3
    )


def test_record_and_database_never_contain_case_submission_candidate_text(
    setup_session,
):
    args, _, path = setup_session
    session = EvaluationSession(**args)
    feedback = session.run()
    serialized = session.record.to_json()
    assert SessionRecord.from_json(serialized) == session.record
    assert feedback.session_digest == json.loads(serialized)["session_digest"]
    for forbidden in (
        CANARY,
        CANDIDATE,
        "Placeholder",
        "evaluated",
        "reference",
        "reviewer-1",
        "Synthetic grounding?",
        "0.75",
        "0.25",
        "0.5",
    ):
        assert forbidden not in serialized
        assert forbidden.encode() not in path.read_bytes()
    document = json.loads(serialized)
    assert set(document) == {
        "schema_version",
        "session_key",
        "binding_digest",
        "steps",
        "session_digest",
    }
    for phase, step_digest, count, tick, band in document["steps"]:
        assert step_digest.startswith("sha256:")
        assert type(count) is int and tick == 17
        assert band is None


@pytest.mark.parametrize("completed_steps", [0, 1, 2, 3, 4])
def test_reopen_replays_only_digest_verified_checkpoints(
    setup_session, completed_steps
):
    args, _, path = setup_session
    session = EvaluationSession(**args)
    methods = (
        session.verify,
        session.run_cases,
        session.run_forensics,
        session.build_adjudication,
    )
    for method in methods[:completed_steps]:
        method()
    before = session.record
    reopened = SQLiteSessionStore(path)
    try:
        resumed = EvaluationSession(**{**args, "store": reopened})
        feedback = resumed.run()
        assert feedback.per_case_scores == (0.75, 0.25)
        assert resumed.record.steps[:completed_steps] == before.steps
    finally:
        reopened.close()


def test_replay_drift_refuses_forensics_and_release(setup_session):
    args, _, _ = setup_session
    session = EvaluationSession(**args)
    session.verify()
    session.run_cases()
    resumed = EvaluationSession(
        **{**args, "runner": lambda _: "Changed synthetic output"}
    )
    resumed.verify()
    assert_code("checkpoint_mismatch", resumed.run_cases)
    assert_code("forensics_required", resumed.release_feedback)
    assert (
        args["ledger"]
        .snapshot(args["holdout"].commitment_digest, args["manifest"].manifest_digest)
        .official_attempts
        == 0
    )


def test_bound_configuration_changes_cannot_resume(setup_session):
    args, _, _ = setup_session
    EvaluationSession(**args).verify()
    assert_code(
        "checkpoint_mismatch",
        lambda: EvaluationSession(**{**args, "canaries": ("New synthetic canary",)}),
    )


def test_ledger_reset_after_restart_fails_closed(setup_session):
    args, _, _ = setup_session
    EvaluationSession(**args).run()
    ledger = FeedbackBudgetLedger(
        [FeedbackBudgetPolicy(args["holdout"].commitment_digest, 1, 2, (0.5, 0.8))]
    )
    reset = EvaluationSession(
        **{**args, "session_key": digest("reset-session"), "ledger": ledger}
    )
    prepare(reset)
    assert_code("budget_state_unverified", reset.release_feedback)


def test_failure_after_durable_claim_never_releases_again(setup_session):
    args, _, path = setup_session

    class CrashingLedger(FeedbackBudgetLedger):
        def record_official_attempt(self, *args, **kwargs):
            super().record_official_attempt(*args, **kwargs)
            raise RuntimeError(CANARY)

    args["ledger"] = CrashingLedger(
        [FeedbackBudgetPolicy(args["holdout"].commitment_digest, 1, 2, (0.5, 0.8))]
    )
    session = EvaluationSession(**args)
    prepare(session)
    assert_code("feedback_failed", session.release_feedback)
    assert session.record.steps[-1][0] == "release_claimed"
    reopened = SQLiteSessionStore(path)
    try:
        assert_code(
            "feedback_already_claimed",
            lambda: EvaluationSession(**{**args, "store": reopened}),
        )
        other = EvaluationSession(
            **{**args, "session_key": digest("other-session"), "store": reopened}
        )
        prepare(other)
        assert_code("budget_state_unverified", other.release_feedback)
    finally:
        reopened.close()


def test_two_live_instances_cannot_duplicate_release(setup_session):
    args, _, path = setup_session
    one = EvaluationSession(**args)
    prepare(one)
    other_store = SQLiteSessionStore(path)
    try:
        two = EvaluationSession(**{**args, "store": other_store})
        prepare(two)

        def release(session):
            try:
                return session.release_feedback()
            except EvaluationSessionError as error:
                return error.code

        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(release, (one, two)))
        assert sum(not isinstance(r, str) for r in results) == 1
        assert "record_conflict" in results
        assert (
            args["ledger"]
            .snapshot(
                args["holdout"].commitment_digest, args["manifest"].manifest_digest
            )
            .official_attempts
            == 1
        )
    finally:
        other_store.close()


@pytest.mark.parametrize("step", ["runner", "scorer", "review"])
def test_provider_exceptions_do_not_echo_payloads(setup_session, step):
    args, _, _ = setup_session

    def fail(*_):
        raise RuntimeError(CANARY)

    session = EvaluationSession(**{**args, step: fail})
    code = "adjudication_failed" if step == "review" else "execution_failed"
    assert_code(code, session.run)
    assert CANARY not in session.record.to_json()


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1, True, CANARY])
def test_invalid_scores_never_release_feedback(setup_session, score):
    args, _, _ = setup_session
    session = EvaluationSession(**{**args, "scorer": lambda *_: score})
    assert_code("invalid_score", session.run)


def test_tampered_or_extended_record_is_rejected(setup_session):
    args, _, path = setup_session
    session = EvaluationSession(**args)
    session.verify()
    document = json.loads(session.record.to_json())
    document["case_text"] = CANARY
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE sessions SET record = ?", (json.dumps(document),))
    assert_code("record_invalid", lambda: EvaluationSession(**args))


def test_record_rejects_unknown_phase_and_plaintext_digest():
    assert_code(
        "record_invalid",
        lambda: SessionRecord(
            digest("key"), digest("binding"), ((CANARY, digest("step"), 1, 0, None),)
        ),
    )
    assert_code("invalid_digest", lambda: SessionRecord(CANARY, digest("binding")))


def test_changed_budget_policy_cannot_resume_or_reset_watermark(setup_session):
    args, _, _ = setup_session
    original = EvaluationSession(**args)
    original.verify()
    changed = FeedbackBudgetLedger(
        [FeedbackBudgetPolicy(args["holdout"].commitment_digest, 1, 2, (0.1, 0.9))]
    )
    assert_code(
        "checkpoint_mismatch", lambda: EvaluationSession(**{**args, "ledger": changed})
    )
    original.run_cases()
    original.run_forensics()
    original.build_adjudication()
    original.release_feedback()
    changed.record_official_attempt(
        args["holdout"].commitment_digest, args["manifest"].manifest_digest, score=0.5
    )
    other = EvaluationSession(
        **{**args, "session_key": digest("changed-policy"), "ledger": changed}
    )
    prepare(other)
    assert_code("budget_state_unverified", other.release_feedback)


def test_canary_forensics_runs_without_persisting_marker(setup_session):
    args, _, path = setup_session
    session = EvaluationSession(**{**args, "runner": lambda _: CANARY})
    session.run()
    assert session.record.steps[2][2] > 0
    assert CANARY.encode() not in path.read_bytes()


def test_forensics_exception_cannot_be_skipped(setup_session, monkeypatch):
    args, _, _ = setup_session

    def fail(*_, **kwargs):
        raise RuntimeError(CANARY)

    monkeypatch.setattr("openmed.eval.workflows.session.scan_overlap_forensics", fail)
    session = EvaluationSession(**args)
    assert_code("forensics_failed", session.run)
    assert_code("forensics_required", session.release_feedback)


def test_failure_writing_terminal_receipt_never_returns_feedback(
    setup_session, monkeypatch
):
    args, _, _ = setup_session

    def fail(*_):
        raise RuntimeError(CANARY)

    monkeypatch.setattr(args["store"], "finish_release", fail)
    session = EvaluationSession(**args)
    prepare(session)
    assert_code("feedback_failed", session.release_feedback)
    assert_code("feedback_already_claimed", session.release_feedback)
    assert_code("feedback_already_claimed", lambda: EvaluationSession(**args))


def test_provider_cannot_supply_plaintext_error_code(setup_session):
    args, _, _ = setup_session

    def fail(*_):
        raise EvaluationSessionError(CANARY)

    session = EvaluationSession(**{**args, "runner": fail})
    assert_code("component_failed", session.run)


def test_store_requires_a_durable_location():
    assert_code("store_failed", lambda: SQLiteSessionStore(":memory:"))
    assert_code("store_failed", lambda: SQLiteSessionStore(""))


def test_store_rejects_plaintext_budget_keys_and_invalid_transitions(setup_session):
    args, _, path = setup_session
    session = EvaluationSession(**args)
    prepare(session)
    previous = session.record
    claimed = replace(
        previous,
        steps=(*previous.steps, ("release_claimed", digest("snapshot"), 1, 17, None)),
    )
    assert_code(
        "invalid_digest",
        lambda: args["store"].claim_release(
            previous, claimed, CANARY, digest("watermark")
        ),
    )
    assert_code(
        "invalid_digest",
        lambda: args["store"].claim_release(
            previous, claimed, digest("budget"), CANARY
        ),
    )
    changed_key = replace(claimed, session_key=digest("different-session"))
    assert_code(
        "record_invalid",
        lambda: args["store"].claim_release(
            previous, changed_key, digest("budget"), digest("watermark")
        ),
    )
    assert_code("record_invalid", lambda: args["store"].save(claimed, previous))
    assert CANARY.encode() not in path.read_bytes()


def test_record_digest_detects_checkpoint_corruption(setup_session):
    args, _, path = setup_session
    session = EvaluationSession(**args)
    session.verify()
    document = json.loads(session.record.to_json())
    document["steps"][0][1] = digest("corrupted-step")
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE sessions SET record = ?", (json.dumps(document),))
    assert_code("record_invalid", lambda: EvaluationSession(**args))
