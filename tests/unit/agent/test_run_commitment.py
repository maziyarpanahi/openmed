from __future__ import annotations

import hashlib
import re
from typing import Any

import pytest

from openmed.agent import OutcomeClass, WorkflowOutcome
from openmed.agent.run_commitment import (
    RUN_COMMITMENT_SCHEMA_VERSION,
    CommitmentStatus,
    RunCommitmentError,
    commit_run_summary,
    verify_run_commitment,
)
from openmed.agent.run_summary import RunEvent, RunSummary

_REASONS = {
    OutcomeClass.SUCCESS: "completed",
    OutcomeClass.ABSTAINED: "insufficient_evidence",
    OutcomeClass.REVIEW_REQUIRED: "human_gate",
    OutcomeClass.POLICY_DENIED: "phi_policy",
    OutcomeClass.FAILED: "tool_error",
}
_DIGEST_A = "sha256:" + "a" * 64
_DIGEST_B = "sha256:" + "b" * 64
_DOMAIN = "openmed.agent.run_commitment.v1"


def _event(
    workflow_id: str,
    outcome_class: OutcomeClass = OutcomeClass.SUCCESS,
    *,
    tool_calls: int = 0,
    duration: float = 0.0,
    digests: tuple[str, ...] = (),
) -> RunEvent:
    return RunEvent(
        workflow_id=workflow_id,
        outcome=WorkflowOutcome(outcome_class, _REASONS[outcome_class]),
        tool_call_count=tool_calls,
        duration_seconds=duration,
        artifact_digests=digests,
    )


def _summary(*events: RunEvent) -> RunSummary:
    return RunSummary.from_events(events)


def _expected(summary: RunSummary) -> str:
    digest = hashlib.sha256(
        _DOMAIN.encode("utf-8") + b"\x00" + summary.to_json().encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def test_empty_summary_has_stable_commitment() -> None:
    summary = _summary()

    assert commit_run_summary(summary) == _expected(summary)


def test_identical_summaries_have_identical_commitments() -> None:
    first = _summary(_event("intake", tool_calls=3, digests=(_DIGEST_A,)))
    second = _summary(_event("intake", tool_calls=3, digests=(_DIGEST_A,)))

    assert commit_run_summary(first) == commit_run_summary(second)


def test_every_metadata_field_change_changes_the_digest() -> None:
    baseline = _summary(_event("intake", tool_calls=1, duration=1.0))

    variants = [
        _summary(_event("intake", tool_calls=1, duration=1.0, digests=(_DIGEST_A,))),
        _summary(_event("intake", tool_calls=2, duration=1.0)),
        _summary(_event("intake", tool_calls=1, duration=1.5)),
        _summary(_event("intake", tool_calls=1, duration=1.0), _event("review")),
        _summary(_event("intake", OutcomeClass.ABSTAINED, duration=1.0)),
    ]

    baseline_digest = commit_run_summary(baseline)
    for variant in variants:
        assert commit_run_summary(variant) != baseline_digest


def test_commitment_format_is_sha256() -> None:
    summary = _summary(_event("intake"))

    commitment = commit_run_summary(summary)

    assert re.fullmatch(r"^sha256:[0-9a-f]{64}$", commitment) is not None


def test_matching_commitment_verifies() -> None:
    summary = _summary(_event("intake", digests=(_DIGEST_A,)))

    assert (
        verify_run_commitment(summary, commit_run_summary(summary))
        is CommitmentStatus.MATCH
    )


def test_mismatched_commitment_reports_mismatch() -> None:
    summary = _summary(_event("intake"))

    status = verify_run_commitment(summary, _DIGEST_B)

    assert status is CommitmentStatus.MISMATCH


def test_constant_time_verification_is_domain_sensitive() -> None:
    summary = _summary(_event("intake"))
    commitment = commit_run_summary(summary)

    forged = (
        "sha256:"
        + hashlib.sha256(
            b"not-the-domain\x00" + summary.to_json().encode("utf-8")
        ).hexdigest()
    )

    assert verify_run_commitment(summary, commitment) is CommitmentStatus.MATCH
    assert verify_run_commitment(summary, forged) is CommitmentStatus.MISMATCH


@pytest.mark.parametrize(
    "commitment",
    [
        "sha256:" + "g" * 64,
        "sha256:" + "a" * 63,
        "sha256:" + "a" * 65,
        "SHA256:" + "a" * 64,
        "sha256:" + "A" * 64,
        "a" * 64,
        "sha256:",
        "",
        None,
        123,
        ["sha256:" + "a" * 64],
    ],
)
def test_malformed_commitments_fail_closed_without_echo(
    commitment: Any,
) -> None:
    summary = _summary(_event("intake"))

    with pytest.raises(RunCommitmentError, match=r"^commitment: invalid_digest$"):
        verify_run_commitment(summary, commitment)


@pytest.mark.parametrize(
    "value",
    [
        None,
        "a summary",
        {"workflow_ids": ["intake"]},
        RunEvent("intake", WorkflowOutcome(OutcomeClass.SUCCESS, "completed")),
    ],
)
def test_non_summary_inputs_are_rejected_without_echo(value: Any) -> None:
    with pytest.raises(RunCommitmentError, match=r"^summary: invalid_summary$"):
        commit_run_summary(value)
    with pytest.raises(RunCommitmentError, match=r"^summary: invalid_summary$"):
        verify_run_commitment(value, _DIGEST_A)


def test_errors_never_echo_submitted_values() -> None:
    sentinel = "Synthetic_Patient_Secret_987"
    summary = _summary(_event("intake"))

    for bad_commitment in (f"sha256:{sentinel}", sentinel, None):
        try:
            verify_run_commitment(summary, bad_commitment)
        except RunCommitmentError as exc:
            assert sentinel not in str(exc)

    try:
        commit_run_summary(sentinel)
    except RunCommitmentError as exc:
        assert sentinel not in str(exc)


def test_commitment_is_exported_from_agent_package() -> None:
    import openmed.agent as agent

    assert agent.commit_run_summary is commit_run_summary
    assert agent.verify_run_commitment is verify_run_commitment
    assert agent.CommitmentStatus is CommitmentStatus
    assert agent.RunCommitmentError is RunCommitmentError
    assert agent.RUN_COMMITMENT_SCHEMA_VERSION == RUN_COMMITMENT_SCHEMA_VERSION
