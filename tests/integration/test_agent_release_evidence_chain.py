"""Execute offline synthetic suites through adapters and non-compensable gates."""

import importlib.util
from pathlib import Path

import pytest

from openmed.eval.suites.agent_release import (
    APPROVAL_BYPASS,
    NOT_READY,
    evaluate_agent_release_gates,
)
from openmed.eval.suites.agent_release_adapters import (
    adversarial_release_evidence,
    recovery_release_evidence,
)


@pytest.mark.integration
def test_executed_evidence_flows_into_release_report_without_inventing_coverage():
    spec = importlib.util.spec_from_file_location(
        "chain_fixtures", Path("tests/fixtures/eval/agent_release.py")
    )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    cases = fixture.executed_recovery_cases()
    executed = adversarial_release_evidence(
        fixture.executed_adversarial_report()
    ) + recovery_release_evidence(
        cases,
        expected_case_digests=tuple(case.case_digest for case in cases),
        completed=True,
        synthetic=True,
    )
    report = evaluate_agent_release_gates(
        executed, candidate_digest=fixture.CANDIDATE_DIGEST
    )
    assert report.decision == NOT_READY
    assert sum(gate.passed for gate in report.gate_results) == 3
    approval = next(
        gate for gate in report.gate_results if gate.metric == APPROVAL_BYPASS
    )
    assert not approval.passed
    assert (
        report.to_json()
        == evaluate_agent_release_gates(
            tuple(reversed(executed)), candidate_digest=fixture.CANDIDATE_DIGEST
        ).to_json()
    )
