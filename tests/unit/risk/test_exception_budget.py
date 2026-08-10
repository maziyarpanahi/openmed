"""Focused tests for the privacy exception budget gate."""

from __future__ import annotations

import json
from datetime import date

import pytest

from openmed.risk import (
    ExceptionBudget,
    ExceptionBudgetExceeded,
    PrivacyException,
    check_exception_budget,
    evaluate_exception_budget,
    fingerprint_policy,
    scope_fingerprint,
)


def _bounded_exception(
    *,
    severity: str = "high",
    scope: str = "release",
    policy: str = "policy-a",
    expiry: str = "2026-09-01",
    count: int = 1,
) -> dict[str, object]:
    return {
        "severity": severity,
        "scope": scope,
        "expires_on": expiry,
        "policy_fingerprint": policy,
        "count": count,
        "finding_text": "synthetic-finding-canary-must-not-escape",
    }


def test_report_counts_all_dimensions_deterministically_and_hashes_scope() -> None:
    budget = ExceptionBudget(
        max_total=3,
        max_by_severity={"high": 2, "medium": 2},
        max_by_scope={"release": 2, "model": 2},
        max_by_policy_fingerprint={"policy-a": 2, "policy-b": 2},
    )
    exceptions = [
        _bounded_exception(),
        _bounded_exception(severity="medium", scope="model", policy="policy-b"),
    ]

    first = evaluate_exception_budget(exceptions, budget)
    second = evaluate_exception_budget(tuple(reversed(exceptions)), budget)

    assert first.allowed is True
    assert first.within_budget is True
    assert first.total_count == 2
    assert first.counts_by_severity == {"high": 1, "medium": 1}
    assert first.counts_by_scope == {
        scope_fingerprint("model"): 1,
        scope_fingerprint("release"): 1,
    }
    assert first.counts_by_expiry == {"bounded": 2}
    assert first.counts_by_policy_fingerprint == {
        fingerprint_policy("policy-a"): 1,
        fingerprint_policy("policy-b"): 1,
    }
    assert first.to_dict() == second.to_dict()
    encoded = json.dumps(first.to_dict(), sort_keys=True)
    assert "synthetic-finding-canary-must-not-escape" not in encoded
    assert "release" not in encoded
    assert json.loads(encoded) == first.to_dict()


def test_total_and_dimension_caps_are_reported_without_raw_metadata() -> None:
    budget = ExceptionBudget(
        max_total=1,
        max_by_severity={"high": 1},
        max_by_scope={"release": 1},
        max_by_policy_fingerprint={"policy-a": 1},
    )
    verdict = evaluate_exception_budget(
        [_bounded_exception(), _bounded_exception()],
        budget,
    )

    assert verdict.allowed is False
    assert {violation.metric for violation in verdict.violations} == {
        "total",
        "severity",
        "scope",
        "policy_fingerprint",
    }
    assert all(
        violation.key is None
        or violation.metric == "severity"
        or violation.key.startswith("sha256:")
        for violation in verdict.violations
    )
    assert "synthetic-finding-canary-must-not-escape" not in str(verdict.to_dict())


def test_unbounded_exception_fails_closed_and_strict_error_is_content_free() -> None:
    budget = ExceptionBudget(max_total=5)
    exception = {
        "severity": "high",
        "scope": "release",
        "policy_fingerprint": "policy-a",
        "finding_text": "synthetic-sensitive-finding-canary",
    }

    verdict = evaluate_exception_budget(exception, budget)

    assert verdict.allowed is False
    assert verdict.unbounded_count == 1
    assert verdict.counts_by_expiry == {"unbounded": 1}
    assert [item.metric for item in verdict.violations] == ["unbounded_exception"]

    with pytest.raises(ExceptionBudgetExceeded) as excinfo:
        check_exception_budget(exception, budget)
    assert "unbounded_exception" in str(excinfo.value)
    assert "synthetic-sensitive-finding-canary" not in str(excinfo.value)


def test_unknown_severity_is_rejected_without_echoing_the_label() -> None:
    raw_label = "synthetic-subject-severity-canary"
    verdict = evaluate_exception_budget(
        {
            **_bounded_exception(),
            "severity": raw_label,
        },
        ExceptionBudget(max_total=5),
    )

    assert verdict.allowed is False
    assert verdict.counts_by_severity == {"unknown": 1}
    assert raw_label not in json.dumps(verdict.to_dict(), sort_keys=True)


def test_expiry_is_evaluated_only_against_explicit_date_and_duration_cap() -> None:
    budget = ExceptionBudget(max_total=4, max_expiry_days=30)
    exceptions = [
        _bounded_exception(expiry="2026-08-10"),
        _bounded_exception(expiry="2026-10-01"),
    ]

    active = evaluate_exception_budget(exceptions, budget, as_of=date(2026, 8, 11))
    assert active.allowed is False
    assert active.evaluation_date == "2026-08-11"
    assert active.counts_by_expiry == {"active": 1, "expired": 1}
    assert active.expired_count == 1
    assert {item.metric for item in active.violations} == {
        "expired_exception",
        "expiry_window",
    }

    missing_date = evaluate_exception_budget(exceptions, budget)
    assert missing_date.allowed is False
    assert missing_date.counts_by_expiry == {"bounded": 2}
    assert [item.metric for item in missing_date.violations] == [
        "expiry_evaluation_date"
    ]


def test_privacy_exception_object_stores_only_safe_metadata() -> None:
    exception = PrivacyException(
        "HIGH",
        "synthetic-subject-scope",
        "2026-09-01",
        "policy-a",
    )

    payload = exception.to_dict()

    assert exception.severity == "high"
    assert exception.expires_at == date(2026, 9, 1)
    assert payload["scope_fingerprint"] == scope_fingerprint("synthetic-subject-scope")
    assert payload["policy_fingerprint"] == fingerprint_policy("policy-a")
    assert "synthetic-subject-scope" not in repr(exception)
    assert "policy-a" not in json.dumps(payload, sort_keys=True)
