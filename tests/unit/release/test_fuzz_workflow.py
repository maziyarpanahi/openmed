"""Regression tests for the dedicated property-based fuzz workflow."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
import yaml
from hypothesis import settings
from hypothesis.errors import InvalidArgument

from tests.fuzz import conftest as fuzz_conftest

ROOT = Path(__file__).resolve().parents[3]
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
FUZZ_WORKFLOW = ROOT / ".github" / "workflows" / "fuzz.yml"


def _load_workflow(path: Path) -> dict[str, object]:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_expensive_fuzz_profile_has_a_dedicated_event_gate():
    fuzz = _load_workflow(FUZZ_WORKFLOW)
    ci = _load_workflow(CI_WORKFLOW)

    assert set(fuzz["on"]) == {"schedule", "workflow_dispatch"}
    assert fuzz["on"]["schedule"] == [{"cron": "17 3 * * *"}]
    assert fuzz["permissions"] == {"contents": "read"}

    job = fuzz["jobs"]["fuzz-nightly"]
    assert job["if"] == (
        "github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'"
    )
    assert any(
        step.get("env", {}).get("HYPOTHESIS_PROFILE") == "fuzz-nightly"
        and "pytest tests/fuzz -q -m fuzz" in step.get("run", "")
        for step in job["steps"]
    )

    assert set(ci["on"]) == {"push", "pull_request"}
    assert "fuzz-nightly" not in ci["jobs"]


def test_explicit_unknown_hypothesis_profile_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HYPOTHESIS_PROFILE", "not-a-real-profile")

    with pytest.raises(InvalidArgument, match="not-a-real-profile"):
        fuzz_conftest._load_selected_profile()


@pytest.mark.parametrize("profile", ["default", ""])
def test_explicit_non_harness_profile_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
) -> None:
    """Only the two audited fuzz profiles may be selected through the environment."""
    monkeypatch.setenv("HYPOTHESIS_PROFILE", profile)

    with pytest.raises(InvalidArgument, match="Unsupported HYPOTHESIS_PROFILE"):
        fuzz_conftest._load_selected_profile()


def test_unset_hypothesis_profile_uses_bounded_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HYPOTHESIS_PROFILE", raising=False)

    fuzz_conftest._load_selected_profile()

    assert settings.get_current_profile_name() == "fuzz-default"


def test_fuzz_profile_budgets_are_registered_exactly() -> None:
    """Lock the bounded/default and exploratory/nightly settings to their contract."""
    fuzz_conftest._register_profiles()
    bounded = settings.get_profile("fuzz-default")
    nightly = settings.get_profile("fuzz-nightly")

    assert bounded.max_examples == 40
    assert bounded.deadline == timedelta(milliseconds=400)
    assert bounded.derandomize is True
    assert bounded.database is None

    assert nightly.max_examples == 1000
    assert nightly.deadline == timedelta(milliseconds=1000)
    assert nightly.derandomize is False
    assert nightly.max_examples >= bounded.max_examples * 10
