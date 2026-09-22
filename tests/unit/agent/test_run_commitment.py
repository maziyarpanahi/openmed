from __future__ import annotations

import builtins
import hashlib
import hmac
from typing import Any

import pytest

import openmed.agent.run_commitment as run_commitment
from openmed.agent import OutcomeClass, WorkflowOutcome
from openmed.agent.run_commitment import (
    RUN_COMMITMENT_VERSION,
    RunCommitmentError,
    RunCommitmentVerificationResult,
    compute_run_summary_commitment,
    verify_run_summary_commitment,
)
from openmed.agent.run_summary import RunEvent, RunSummary

_DIGEST_A = "sha256:" + "a" * 64
_DIGEST_B = "sha256:" + "b" * 64

_GOLDEN_COMMITMENTS = {
    "empty": "sha256:1868266566d00dcf44ab1807d4ec4127eb67faad32a828eec8cb0097e3a4917e",
    "success": "sha256:8f4c01add2e40211c293c548148207a09fc9b76ea91f41b41fd9b6d10be0e5ef",
    "abstained": "sha256:a8ee8ac0bd0df9eb6e9db8aa3ece365620b399e0481c103b394bbc43af2f1c99",
    "mixed": "sha256:9d2a816c557e3d6c8206c6fe3504b2190d9f403c68344a781e5f12cd9b1cc5fe",
}

_REASONS = {
    OutcomeClass.SUCCESS: "completed",
    OutcomeClass.ABSTAINED: "insufficient_evidence",
    OutcomeClass.REVIEW_REQUIRED: "human_gate",
    OutcomeClass.POLICY_DENIED: "phi_policy",
    OutcomeClass.FAILED: "tool_error",
}


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


def _golden_summaries() -> dict[str, RunSummary]:
    return {
        "empty": RunSummary.from_events([]),
        "success": RunSummary.from_events(
            [
                _event(
                    "workflow-success",
                    tool_calls=2,
                    duration=1.5,
                    digests=(_DIGEST_A,),
                )
            ]
        ),
        "abstained": RunSummary.from_events(
            [
                _event(
                    "workflow-abstained",
                    OutcomeClass.ABSTAINED,
                    duration=0.25,
                )
            ]
        ),
        "mixed": RunSummary.from_events(
            [
                _event(
                    "workflow-alpha",
                    tool_calls=2,
                    duration=1.25,
                    digests=(_DIGEST_A,),
                ),
                _event(
                    "workflow-beta",
                    OutcomeClass.ABSTAINED,
                    tool_calls=1,
                    duration=0.75,
                    digests=(_DIGEST_B,),
                ),
                _event(
                    "workflow-beta",
                    OutcomeClass.FAILED,
                    tool_calls=3,
                    duration=2.0,
                ),
            ]
        ),
    }


@pytest.mark.parametrize("case_id", ["empty", "success", "abstained", "mixed"])
def test_commitment_matches_golden_vector(case_id: str) -> None:
    summary = _golden_summaries()[case_id]

    assert compute_run_summary_commitment(summary) == _GOLDEN_COMMITMENTS[case_id]


def test_identical_summaries_have_identical_commitments() -> None:
    first = RunSummary.from_json(_golden_summaries()["mixed"].to_json())
    second = RunSummary.from_json(_golden_summaries()["mixed"].to_json())

    assert compute_run_summary_commitment(first) == compute_run_summary_commitment(
        second
    )


@pytest.mark.parametrize(
    "changed",
    [
        RunSummary.from_events([_event("workflow-other")]),
        RunSummary.from_events([_event("workflow-base", OutcomeClass.ABSTAINED)]),
        RunSummary.from_events([_event("workflow-base", tool_calls=2)]),
        RunSummary.from_events([_event("workflow-base", duration=0.5)]),
        RunSummary.from_events([_event("workflow-base", digests=(_DIGEST_A,))]),
    ],
    ids=[
        "workflow_ids",
        "outcome_counts",
        "tool_call_count",
        "duration_seconds",
        "artifact_digests",
    ],
)
def test_each_summary_metadata_field_changes_commitment(changed: RunSummary) -> None:
    baseline = RunSummary.from_events([_event("workflow-base")])

    assert compute_run_summary_commitment(changed) != compute_run_summary_commitment(
        baseline
    )


def test_domain_separation_changes_plain_json_digest() -> None:
    summary = _golden_summaries()["empty"]
    plain_digest = (
        "sha256:" + hashlib.sha256(summary.to_json().encode("utf-8")).hexdigest()
    )

    assert compute_run_summary_commitment(summary) != plain_digest
    assert RUN_COMMITMENT_VERSION == "openmed.agent.run_commitment.v1"


def test_matching_commitment_verifies() -> None:
    summary = _golden_summaries()["success"]

    assert (
        verify_run_summary_commitment(summary, compute_run_summary_commitment(summary))
        is RunCommitmentVerificationResult.VERIFIED
    )


def test_well_formed_nonmatching_commitment_is_categorical() -> None:
    result = verify_run_summary_commitment(
        _golden_summaries()["success"],
        _GOLDEN_COMMITMENTS["empty"],
    )

    assert result is RunCommitmentVerificationResult.MISMATCH
    assert result.value == "mismatch"


@pytest.mark.parametrize(
    "commitment",
    [
        None,
        123,
        b"sha256:" + b"0" * 64,
        "",
        "0" * 64,
        "sha256:short",
        "sha256:" + "A" * 64,
        "sha512:" + "0" * 64,
        "sha256:" + "0" * 65,
        "sha256:" + "g" * 64,
    ],
)
def test_malformed_commitments_return_one_value_free_category(
    commitment: Any,
) -> None:
    result = verify_run_summary_commitment(_golden_summaries()["empty"], commitment)

    assert result is RunCommitmentVerificationResult.MALFORMED
    assert result.value == "malformed_commitment"


@pytest.mark.parametrize(
    "summary",
    [
        None,
        {},
        _golden_summaries()["empty"].to_dict(),
        _golden_summaries()["empty"].to_json(),
    ],
)
def test_only_validated_summary_objects_can_be_hashed(summary: Any) -> None:
    with pytest.raises(RunCommitmentError) as exc_info:
        compute_run_summary_commitment(summary)

    assert str(exc_info.value) == "summary: invalid_type"
    assert exc_info.value.__cause__ is None


def test_verification_uses_compare_digest_for_all_commitment_categories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []
    real_compare_digest = hmac.compare_digest

    def recording_compare_digest(left: str, right: str) -> bool:
        calls.append((left, right))
        return real_compare_digest(left, right)

    monkeypatch.setattr(
        run_commitment.hmac,
        "compare_digest",
        recording_compare_digest,
    )
    summary = _golden_summaries()["empty"]

    verify_run_summary_commitment(summary, _GOLDEN_COMMITMENTS["empty"])
    verify_run_summary_commitment(summary, _GOLDEN_COMMITMENTS["success"])
    verify_run_summary_commitment(summary, "malformed-secret-value")

    assert len(calls) == 3
    assert all(len(left) == len(right) == 71 for left, right in calls)


def test_commitment_computation_performs_no_file_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_open(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("commitment computation attempted file I/O")

    monkeypatch.setattr(builtins, "open", forbidden_open)

    assert (
        compute_run_summary_commitment(_golden_summaries()["empty"])
        == _GOLDEN_COMMITMENTS["empty"]
    )


def test_submitted_values_never_enter_errors_or_results() -> None:
    sentinel = "Synthetic_Patient_Secret_987"

    with pytest.raises(RunCommitmentError) as exc_info:
        compute_run_summary_commitment({"summary": sentinel})  # type: ignore[arg-type]
    result = verify_run_summary_commitment(
        _golden_summaries()["empty"],
        sentinel,
    )

    assert sentinel not in str(exc_info.value)
    assert sentinel not in repr(result)


def test_commitment_api_is_exported_from_agent_package() -> None:
    import openmed.agent as agent

    assert agent.RUN_COMMITMENT_VERSION == RUN_COMMITMENT_VERSION
    assert agent.RunCommitmentError is RunCommitmentError
    assert agent.RunCommitmentVerificationResult is RunCommitmentVerificationResult
    assert agent.compute_run_summary_commitment is compute_run_summary_commitment
    assert agent.verify_run_summary_commitment is verify_run_summary_commitment
