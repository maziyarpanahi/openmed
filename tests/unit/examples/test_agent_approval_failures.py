"""Tests for deterministic human-approval failure examples."""

from __future__ import annotations

import importlib.util

import pytest

from examples import agent_approval_failures as example


def _approval_contract_available() -> bool:
    try:
        return importlib.util.find_spec(example.APPROVAL_MODULE) is not None
    except ModuleNotFoundError:
        return False


def test_failure_matrix_is_closed_and_stable() -> None:
    assert example.EXPECTED_FAILURES == (
        ("expiry", "expired"),
        ("replay", "replayed"),
        ("wrong_action_digest", "action_mismatch"),
        ("wrong_reviewer_role", "reviewer_role_mismatch"),
        ("unsupported_schema_version", "unsupported_schema_version"),
    )


@pytest.mark.skipif(
    not _approval_contract_available(),
    reason="requires the approval-token contract from issue #2768",
)
def test_examples_run_offline_with_one_stable_reason_each(capsys) -> None:
    first = example.main()
    first_output = capsys.readouterr().out
    second = example.main()
    second_output = capsys.readouterr().out

    expected = tuple(
        example.FailureResult(scenario=scenario, reason=reason)
        for scenario, reason in example.EXPECTED_FAILURES
    )
    assert first == expected
    assert second == expected
    assert first_output == second_output
    assert first_output.splitlines() == [
        f"{scenario}: {reason}" for scenario, reason in example.EXPECTED_FAILURES
    ]
    assert "nonce_" not in first_output
    assert "sha256:" not in first_output
    assert example.REVIEWER_ROLE not in first_output
    assert example.OTHER_REVIEWER_ROLE not in first_output
