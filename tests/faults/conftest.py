"""Fixtures for the service fault-injection harness tests."""

from __future__ import annotations

import pytest

from openmed.core.config import PROFILE_ENV_VAR


@pytest.fixture(autouse=True)
def fault_test_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Select the only profile under which fault injectors may arm."""
    monkeypatch.setenv(PROFILE_ENV_VAR, "test")
