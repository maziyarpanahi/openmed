"""Focused tests for the deterministic deletion impact planner."""

from __future__ import annotations

import socket
from hashlib import sha256

import pytest

from openmed.risk import (
    ConfirmationRequiredError,
    DeletionExecutionError,
    DeletionPlanError,
    execute_deletion_plan,
    plan_deletion_impact,
)


def _digest(label: str) -> str:
    return f"sha256:{sha256(label.encode('utf-8')).hexdigest()}"


def _manifest() -> tuple[dict[str, object], ...]:
    cache = _digest("synthetic-cache")
    mapping = _digest("synthetic-map")
    evidence = _digest("synthetic-evidence")
    return (
        {
            "artifact_hash": evidence,
            "kind": "evidence",
            "retention_class": "audit",
            "dependencies": [mapping],
        },
        {
            "artifact_hash": cache,
            "kind": "cache",
            "retention_class": "short",
            "dependencies": [],
        },
        {
            "artifact_hash": mapping,
            "kind": "map",
            "retention_class": "standard",
            "dependencies": [cache],
        },
    )


def test_plan_follows_reverse_dependencies_and_is_deterministic() -> None:
    cache = _digest("synthetic-cache")
    first = plan_deletion_impact(_manifest(), [cache])
    second = plan_deletion_impact(tuple(reversed(_manifest())), targets=cache)

    assert first.to_json() == second.to_json()
    assert first.target_count == 1
    assert first.affected_count == 3
    assert first.owned_affected_count == 3
    assert first.counts_by_kind == {"cache": 1, "evidence": 1, "map": 1}
    assert first.counts_by_retention_class == {
        "audit": 1,
        "short": 1,
        "standard": 1,
    }
    assert first.dependency_edge_count == 2
    assert first.unresolved_dependency_count == 0
    assert first.affected_hashes == tuple(sorted(first.affected_hashes))


def test_dry_run_report_is_counts_only_and_does_not_leak_opaque_input() -> None:
    raw_canary = "synthetic-sensitive-value-canary"
    manifest = [
        {
            "artifact_hash": raw_canary,
            "kind": "cache",
            "retention_class": "short",
            "resource_path": "/private/synthetic-sensitive-value-canary",
        }
    ]

    plan = plan_deletion_impact(manifest, raw_canary)
    report = plan.to_json() + plan.to_markdown()

    assert raw_canary not in report
    assert "/private/" not in report
    assert plan.to_dict()["raw_values_included"] is False
    assert plan.to_dict()["target_count"] == 1


def test_planning_works_when_outbound_sockets_are_blocked(monkeypatch) -> None:
    def fail_connect(*_args, **_kwargs):
        raise AssertionError("deletion planning attempted network access")

    monkeypatch.setattr(socket.socket, "connect", fail_connect)
    plan = plan_deletion_impact(_manifest(), _digest("synthetic-cache"))

    assert plan.affected_count == 3


def test_execution_requires_exact_confirmation_before_callback() -> None:
    plan = plan_deletion_impact(_manifest(), _digest("synthetic-cache"))
    calls: list[str] = []

    def executor(artifact) -> None:
        calls.append(artifact.artifact_hash)

    with pytest.raises(ConfirmationRequiredError):
        execute_deletion_plan(plan, executor=executor)
    with pytest.raises(ConfirmationRequiredError):
        execute_deletion_plan(plan, confirmation="confirm:wrong", executor=executor)
    assert calls == []

    result = execute_deletion_plan(
        plan,
        confirmation=plan.confirmation_token,
        executor=executor,
    )

    assert result.deleted_count == 1
    assert result.failed_count == 0
    assert calls == [_digest("synthetic-cache")]


def test_unowned_targets_and_unresolved_links_fail_closed() -> None:
    target = _digest("synthetic-owned-target")
    dependent = _digest("synthetic-dependent")
    missing = _digest("synthetic-missing")
    manifest = [
        {
            "artifact_hash": target,
            "kind": "cache",
            "retention_class": "short",
            "owned": False,
        },
        {
            "artifact_hash": dependent,
            "kind": "evidence",
            "retention_class": "audit",
            "dependencies": [target, missing],
        },
    ]
    plan = plan_deletion_impact(manifest, target)
    assert plan.blocked_target_count == 1
    assert plan.unresolved_dependency_count == 1

    with pytest.raises(DeletionPlanError):
        execute_deletion_plan(
            plan,
            confirmation=plan.confirmation_token,
            executor=lambda _artifact: None,
        )


def test_invalid_manifest_errors_do_not_echo_sensitive_values() -> None:
    canary = "synthetic-sensitive-retention-canary"

    with pytest.raises(DeletionPlanError) as exc_info:
        plan_deletion_impact(
            [
                {
                    "artifact_hash": _digest("synthetic-artifact"),
                    "retention_class": canary + "/private",
                }
            ],
            _digest("synthetic-artifact"),
        )

    assert canary not in str(exc_info.value)


def test_callback_failures_are_content_free() -> None:
    plan = plan_deletion_impact(_manifest(), _digest("synthetic-cache"))

    def failing_executor(_artifact) -> None:
        raise RuntimeError("synthetic-sensitive-callback-value")

    with pytest.raises(DeletionExecutionError) as exc_info:
        execute_deletion_plan(
            plan,
            confirmation=plan.confirmation_token,
            executor=failing_executor,
        )

    assert "synthetic-sensitive-callback-value" not in str(exc_info.value)


def test_non_dry_run_is_not_accepted() -> None:
    with pytest.raises(DeletionPlanError):
        plan_deletion_impact(_manifest(), _digest("synthetic-cache"), dry_run=False)
