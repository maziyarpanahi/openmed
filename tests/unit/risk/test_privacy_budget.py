"""Focused tests for aggregate-release privacy budget accounting."""

from __future__ import annotations

import importlib
import json
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from openmed.risk import (
    PrivacyBudget,
    PrivacyBudgetExceeded,
    PrivacyBudgetLedger,
    PrivacyBudgetLedgerExceeded,
    ReleaseContextPrivacyBudget,
)

privacy_budget_module = importlib.import_module("openmed.risk.privacy_budget")


def _ledger() -> PrivacyBudgetLedger:
    return PrivacyBudgetLedger(
        {
            "daily-release": ReleaseContextPrivacyBudget(epsilon=1.0, delta=1e-5),
            "research-release": {"epsilon": 2.0, "delta": 1e-4},
        }
    )


def test_spend_is_tracked_per_release_context() -> None:
    ledger = _ledger()

    daily = ledger.record_release("daily-release", 0.25, 2e-6)
    research = ledger.spend("research-release", 1.25, 3e-5)

    assert daily.allowed is True
    assert research.allowed is True
    assert ledger.spends[0].context == "daily-release"
    assert ledger.spends[1].context == "research-release"
    assert ledger.to_dict()["contexts"]["daily-release"]["spent_epsilon"] == 0.25
    assert ledger.to_dict()["contexts"]["research-release"]["spent_delta"] == 3e-5


def test_check_is_non_mutating_and_rejects_epsilon_or_delta_over_budget() -> None:
    ledger = _ledger()
    ledger.record_release("daily-release", 0.75, 8e-6)
    before = ledger.spends

    decision = ledger.check("daily-release", 0.3, 1e-6)

    assert decision.allowed is False
    assert decision.reason == "epsilon exceeds budget"
    assert decision.projected_epsilon == pytest.approx(1.05)
    assert decision.remaining_epsilon == 0.0
    assert ledger.spends == before

    delta_decision = ledger.check("research-release", 0.1, 2e-4)
    assert delta_decision.allowed is False
    assert delta_decision.reason == "delta exceeds budget"


def test_over_budget_release_raises_before_recording_a_spend() -> None:
    ledger = _ledger()
    ledger.record_release("daily-release", 0.8, 8e-6)
    before = ledger.spends

    with pytest.raises(PrivacyBudgetLedgerExceeded) as excinfo:
        ledger.record_release("daily-release", 0.3, 3e-6)

    assert ledger.spends == before
    assert excinfo.value.decision.allowed is False
    assert ledger.to_dict()["contexts"]["daily-release"]["release_count"] == 1
    assert ledger.to_dict()["contexts"]["daily-release"]["rejected_count"] == 1


def test_counts_only_evidence_is_deterministic_and_payload_free() -> None:
    first = _ledger()
    second = _ledger()
    for ledger in (first, second):
        ledger.record_release("research-release", 0.25, 2e-6)
        ledger.record_release("daily-release", 0.5, 4e-6)

    evidence = first.render_counts_only()
    encoded = first.to_json()

    assert evidence == second.render()
    assert encoded == second.to_json()
    assert json.loads(encoded) == evidence
    assert evidence["release_count"] == 2
    assert evidence["context_count"] == 2
    assert evidence["contexts"]["daily-release"]["attempt_count"] == 1
    assert "spends" not in evidence
    assert "payload" not in encoded
    assert "Synthetic Patient" not in encoded
    assert "123-45-6789" not in encoded


def test_identifier_and_budget_validation_never_echoes_sensitive_values() -> None:
    with pytest.raises(ValueError, match="PHI-shaped") as excinfo:
        PrivacyBudgetLedger({"123-45-6789": {"epsilon": 1, "delta": 1e-6}})
    assert "123-45-6789" not in str(excinfo.value)

    with pytest.raises(ValueError, match="unsupported fields") as excinfo:
        ReleaseContextPrivacyBudget.from_mapping(
            {
                "epsilon": 1,
                "delta": 1e-6,
                "raw_value": "Synthetic Patient",
            }
        )
    assert "Synthetic Patient" not in str(excinfo.value)


def test_public_budget_api_is_available_from_openmed_risk() -> None:
    import openmed.risk as risk

    assert hasattr(risk, "PrivacyBudgetLedger")
    assert "PrivacyBudgetLedgerExceeded" in risk.__all__
    assert PrivacyBudget.__module__ == "openmed.risk.differential_privacy"
    assert PrivacyBudgetExceeded.__module__ == "openmed.risk.differential_privacy"


def test_budget_mapping_aliases_and_numeric_inputs_are_strict() -> None:
    with pytest.raises(ValueError, match="duplicate aliases"):
        ReleaseContextPrivacyBudget.from_mapping(
            {"epsilon": 1.0, "max_epsilon": 1.0, "delta": 1e-6}
        )
    with pytest.raises(ValueError, match="finite number"):
        ReleaseContextPrivacyBudget(epsilon="1.0", delta=1e-6)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite number"):
        ReleaseContextPrivacyBudget(epsilon=10**10_000, delta=1e-6)


def test_budget_configuration_is_not_mutable_through_public_access() -> None:
    ledger = _ledger()

    with pytest.raises(TypeError):
        ledger.budgets["daily-release"] = ReleaseContextPrivacyBudget(  # type: ignore[index]
            epsilon=10.0,
            delta=1e-5,
        )
    assert ledger.budget_for("daily-release").epsilon == 1.0


def test_threaded_charges_cannot_overrun_one_context_budget() -> None:
    ledger = PrivacyBudgetLedger({"batch-release": {"epsilon": 1.0, "delta": 0.0}})

    def charge() -> bool:
        try:
            ledger.record_release("batch-release", epsilon=0.02, delta=0.0)
        except PrivacyBudgetLedgerExceeded:
            return False
        return True

    with ThreadPoolExecutor(max_workers=16) as executor:
        accepted = list(executor.map(lambda _: charge(), range(100)))

    evidence = ledger.to_dict()
    assert sum(accepted) == 50
    assert evidence["release_count"] == 50
    assert evidence["rejected_count"] == 50
    assert evidence["spent_epsilon"] == 1.0


def test_context_and_spend_limits_are_enforced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(privacy_budget_module, "MAX_PRIVACY_BUDGET_CONTEXTS", 1)
    with pytest.raises(ValueError, match="item limit"):
        PrivacyBudgetLedger(
            {
                "first-release": {"epsilon": 1.0, "delta": 0.0},
                "second-release": {"epsilon": 1.0, "delta": 0.0},
            }
        )

    monkeypatch.setattr(privacy_budget_module, "MAX_PRIVACY_BUDGET_CONTEXTS", 512)
    monkeypatch.setattr(privacy_budget_module, "MAX_PRIVACY_BUDGET_SPENDS", 2)
    ledger = PrivacyBudgetLedger({"daily-release": {"epsilon": 10.0, "delta": 0.0}})
    ledger.record_release("daily-release", 1.0, 0.0)
    ledger.record_release("daily-release", 1.0, 0.0)
    with pytest.raises(OverflowError, match="spend limit"):
        ledger.record_release("daily-release", 1.0, 0.0)


class _ExplodingMapping(Mapping[str, Any]):
    def __getitem__(self, key: str) -> Any:
        raise RuntimeError("synthetic-sensitive-value")

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError("synthetic-sensitive-value")

    def __len__(self) -> int:
        raise RuntimeError("synthetic-sensitive-value")

    def items(self) -> Any:
        raise RuntimeError("synthetic-sensitive-value")


def test_hostile_budget_mapping_fails_without_echoing_values() -> None:
    with pytest.raises(ValueError, match="could not be read") as error:
        PrivacyBudgetLedger(_ExplodingMapping())
    assert "synthetic-sensitive-value" not in str(error.value)
