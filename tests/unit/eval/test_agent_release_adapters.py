"""Offline evidence-chain controls for executed agent release sources."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from openmed.agent.security.adversarial import (
    AdversarialSuiteReport,
    AdversarialSuiteValidationError,
    AttackClass,
    BoundaryDecision,
)
from openmed.agent.workflows.recovery import (
    RecoveryDecision,
    RecoveryDisposition,
    RecoveryReason,
)
from openmed.eval.suites.agent_release import (
    APPROVAL_BYPASS,
    NOT_READY,
    RECOVERY_CORRECTNESS,
    UNAUTHORIZED_ACTION_ESCAPE,
    UNSAFE_SIDE_EFFECT,
    evaluate_agent_release_gates,
)
from openmed.eval.suites.agent_release_adapters import (
    DEFAULT_ADVERSARIAL_EVIDENCE_CASES,
    SOURCE_REPORT_VERSION,
    AgentReleaseAdapterError,
    adversarial_release_evidence,
    json_release_evidence,
    recovery_release_evidence,
)


def fixtures():
    spec = importlib.util.spec_from_file_location(
        "executed_gate_fixtures", Path("tests/fixtures/eval/agent_release.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recovery(cases=None, **kwargs):
    source = fixtures().executed_recovery_cases() if cases is None else cases
    options = dict(
        expected_case_digests=tuple(case.case_digest for case in source),
        completed=True,
        synthetic=True,
    )
    options.update(kwargs)
    return recovery_release_evidence(source, **options)


def source_report(kind="replay"):
    agreement = kind == "clinician_agreement"
    slices = {
        "replay": ("clean_replay", "resumed_replay"),
        "reference_server": ("read_interactions", "write_interactions"),
        "clinician_agreement": ("evidence_grounding", "workflow_safety"),
    }[kind]
    interval = {"value": 0.9, "ci_lower": 0.85, "ci_upper": 0.95}
    return {
        "schema_version": SOURCE_REPORT_VERSION,
        "kind": kind,
        "status": "completed",
        "synthetic": True,
        "corpus": "default",
        "cases": [
            dict(
                case_digest=fixtures().digest(str(i)),
                slice=ref,
                **({} if agreement else {"success": True}),
            )
            for i, ref in enumerate(slices)
        ],
        "statistics": {
            "method": "ordinal_krippendorff_alpha_stratified_bootstrap_95",
            "overall": interval,
            "slices": {ref: interval for ref in slices},
        }
        if agreement
        else None,
    }


def import_source(source, expected=None):
    return json_release_evidence(
        json.dumps(source),
        expected_kind=source["kind"],
        expected_case_digests=expected
        or tuple(case["case_digest"] for case in source["cases"]),
    )


def test_default_execution_counts_and_digests_are_deterministic():
    f = fixtures()
    first = adversarial_release_evidence(f.executed_adversarial_report())
    assert first == adversarial_release_evidence(f.executed_adversarial_report())
    assert [(item.metric, item.event_count, item.sample_size) for item in first] == [
        (UNAUTHORIZED_ACTION_ESCAPE, 0, 10),
        (UNSAFE_SIDE_EFFECT, 0, 2),
    ]
    assert len(first[0].slices) == 10
    assert first[0].evidence_digest == first[1].evidence_digest
    assert (
        first[0].evidence_digest
        == "sha256:85fd667735abb594338bd37e168c56b5ba461871d8fc3a9a94c7018969a6edc0"
    )
    assert first[0].limitations == (
        "synthetic_only",
        "dispatch_probe_only",
        "default_corpus",
    )
    assert APPROVAL_BYPASS not in {item.metric for item in first}
    actual = recovery()
    assert actual == recovery()
    assert actual[0].event_count == actual[0].sample_size == 6
    assert (
        actual[0].evidence_digest
        == "sha256:10f35b25db29d057d4c75b776c76fd81b78f69dadec7a5b8b2d843a82cbdd84f"
    )
    assert len(actual[0].slices) == 6


def test_missing_case_or_failed_adversarial_run_refuses_evidence():
    report = fixtures().executed_adversarial_report()
    for altered in (
        AdversarialSuiteReport(report.cases[:-1]),
        AdversarialSuiteReport(
            (replace(report.cases[0], passed=False),) + report.cases[1:]
        ),
        AdversarialSuiteReport(
            report.cases[:1]
            + (
                replace(
                    report.cases[1], decision=BoundaryDecision.ALLOW, dispatch_count=1
                ),
            )
            + report.cases[2:]
        ),
    ):
        with pytest.raises(AgentReleaseAdapterError):
            adversarial_release_evidence(altered)
    with pytest.raises(AdversarialSuiteValidationError):
        AdversarialSuiteReport(report.cases + (report.cases[-1],))
    with pytest.raises(AgentReleaseAdapterError, match="duplicate_case"):
        adversarial_release_evidence(
            report,
            expected_cases=DEFAULT_ADVERSARIAL_EVIDENCE_CASES
            + (DEFAULT_ADVERSARIAL_EVIDENCE_CASES[0],),
        )


def test_declared_approval_challenge_is_counted_by_attack_class():
    # An application-owned manifest defines the semantics absent from the generic report.
    cases = tuple(
        replace(case, metrics=(UNAUTHORIZED_ACTION_ESCAPE, APPROVAL_BYPASS))
        if case.attack_class is AttackClass.CONFUSED_DEPUTY_DELEGATION
        else case
        for case in DEFAULT_ADVERSARIAL_EVIDENCE_CASES
    )
    evidence = adversarial_release_evidence(
        fixtures().executed_adversarial_report(), expected_cases=cases
    )
    approval = next(item for item in evidence if item.metric == APPROVAL_BYPASS)
    assert (approval.event_count, approval.sample_size) == (0, 1)
    assert approval.slices[0].slice_ref == "confused_deputy_delegation"
    assert "default_corpus" not in approval.limitations


def test_recovery_negative_control_changes_digest_and_fails_gate():
    cases = fixtures().executed_recovery_cases()
    original = recovery(cases)[0]
    wrong = RecoveryDecision.create(
        disposition=RecoveryDisposition.REVIEW_REQUIRED,
        reason=RecoveryReason.AMBIGUOUS_EFFECT,
        source_checkpoint_digest=cases[1].checkpoint.checkpoint_digest,
    )
    altered = recovery((cases[0], replace(cases[1], decision=wrong)) + cases[2:])[0]
    assert altered.event_count == 5
    assert altered.evidence_digest != original.evidence_digest
    baseline = tuple(
        altered if item.metric == RECOVERY_CORRECTNESS else item
        for item in fixtures().passing_evidence()
    )
    assert (
        evaluate_agent_release_gates(
            baseline, candidate_digest=fixtures().CANDIDATE_DIGEST
        ).decision
        == NOT_READY
    )


def test_recovery_refuses_incomplete_duplicate_failed_and_misbound_runs():
    cases = fixtures().executed_recovery_cases()
    inventory = tuple(case.case_digest for case in cases)
    for source in (cases[:-1], cases + (cases[0],)):
        with pytest.raises(AgentReleaseAdapterError):
            recovery(source, expected_case_digests=inventory)
    with pytest.raises(AgentReleaseAdapterError, match="failed_run"):
        recovery(cases, completed=False)
    with pytest.raises(AgentReleaseAdapterError, match="checkpoint_mismatch"):
        replace(cases[0], decision=cases[1].decision)
    with pytest.raises(AgentReleaseAdapterError, match="duplicate_run"):
        recovery(
            cases + (replace(cases[0], case_digest=fixtures().digest("duplicate")),)
        )


@pytest.mark.parametrize("kind", ["replay", "reference_server", "clinician_agreement"])
def test_json_sources_are_deterministic_and_content_free(kind):
    source = source_report(kind)
    evidence = import_source(source)[0]
    assert (
        evidence
        == json_release_evidence(
            json.dumps(source, sort_keys=True, indent=2),
            expected_kind=kind,
            expected_case_digests=tuple(
                case["case_digest"] for case in source["cases"]
            ),
        )[0]
    )
    assert evidence.sample_size == 2
    assert evidence.limitations == (
        "host_supplied_report",
        "synthetic_only",
        "default_corpus",
    )
    encoded = json.dumps(evidence.to_dict())
    assert all(case["case_digest"] not in encoded for case in source["cases"])
    assert "cases" not in encoded
    canonical = json.dumps(
        source, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    assert (
        evidence.evidence_digest
        == "sha256:" + hashlib.sha256(canonical.encode("ascii")).hexdigest()
    )
    if kind == "clinician_agreement":
        source["statistics"]["overall"] = {
            "value": 0.88,
            "ci_lower": 0.82,
            "ci_upper": 0.94,
        }
    else:
        source["cases"][0]["success"] = False
    assert import_source(source)[0].evidence_digest != evidence.evidence_digest


@pytest.mark.parametrize("kind", ["replay", "reference_server", "clinician_agreement"])
@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "failed", "truncated", "payload", "version"]
)
def test_json_negative_controls(kind, fault):
    source = source_report(kind)
    inventory = tuple(case["case_digest"] for case in source["cases"])
    if fault == "missing":
        source["cases"].pop()
    elif fault == "duplicate":
        source["cases"].append(source["cases"][0])
    elif fault in ("failed", "truncated"):
        source["status"] = fault
    elif fault == "payload":
        source["cases"][0]["payload"] = "SYNTHETIC_PHI_CANARY"
    else:
        source["schema_version"] = "unknown"
    with pytest.raises(AgentReleaseAdapterError) as error:
        import_source(source, inventory)
    assert "SYNTHETIC_PHI_CANARY" not in str(error.value)


def test_json_duplicate_keys_invalid_intervals_and_nonboolean_outcomes():
    source = source_report()
    serialized = json.dumps(source).replace(
        '"status": "completed"', '"status": "failed", "status": "completed"'
    )
    with pytest.raises(AgentReleaseAdapterError):
        json_release_evidence(
            serialized,
            expected_kind="replay",
            expected_case_digests=tuple(
                case["case_digest"] for case in source["cases"]
            ),
        )
    source["cases"][0]["success"] = 1
    with pytest.raises(AgentReleaseAdapterError, match="invalid_outcome"):
        import_source(source)
    clinical = source_report("clinician_agreement")
    clinical["statistics"]["overall"]["ci_lower"] = float("nan")
    with pytest.raises(AgentReleaseAdapterError, match="invalid_interval"):
        import_source(clinical)


def test_absent_adapter_inputs_emit_nothing_and_gates_fail_closed():
    assert adversarial_release_evidence(None) == ()
    assert (
        recovery_release_evidence(
            None, expected_case_digests=(), completed=False, synthetic=True
        )
        == ()
    )
    assert (
        json_release_evidence(None, expected_kind="replay", expected_case_digests=())
        == ()
    )
    report = evaluate_agent_release_gates(
        (), candidate_digest=fixtures().CANDIDATE_DIGEST
    )
    assert report.decision == NOT_READY
    assert all(not gate.passed for gate in report.gate_results)


def test_executed_report_output_contains_no_default_case_payloads():
    evidence = (
        adversarial_release_evidence(fixtures().executed_adversarial_report())
        + recovery()
    )
    rendered = json.dumps([item.to_dict() for item in evidence])
    for canary in (
        "synthetic-secret-canary",
        "untrusted.invalid",
        "/outside/",
        "Ignore previous instructions",
        "case_id",
        "checkpoint",
        "retry_effect_ids",
    ):
        assert canary not in rendered
