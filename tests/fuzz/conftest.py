"""Hypothesis profile registration for the de-identification fuzz harness.

Two profiles keep CI fast by default while allowing a heavier scheduled run:

- ``fuzz-default`` (used by the normal suite): small example budget and a
  short per-example deadline so the fuzz tests add negligible wall-clock time
  to ``pytest tests/``.
- ``fuzz-nightly`` (opted into via ``HYPOTHESIS_PROFILE=fuzz-nightly``): a much
  larger example budget for the scheduled CI job.

Selection order:

1. If ``HYPOTHESIS_PROFILE`` is set, that profile is loaded (so developers and
   the nightly job can override).
2. Otherwise ``fuzz-default`` is loaded.

The profiles are registered at import time and are idempotent, so importing the
fuzz modules in any order (or repeatedly) is safe.
"""

from __future__ import annotations

import os

import pytest

try:  # pragma: no cover - exercised indirectly; skip cleanly if unavailable.
    from hypothesis import HealthCheck, settings
except ImportError:  # pragma: no cover
    settings = None  # type: ignore[assignment]
    HealthCheck = None  # type: ignore[assignment]


_DEFAULT_PROFILE = "fuzz-default"
_NIGHTLY_PROFILE = "fuzz-nightly"
_PROFILES_REGISTERED = False


def _register_profiles() -> None:
    """Register the bounded and nightly Hypothesis profiles once."""
    global _PROFILES_REGISTERED
    if _PROFILES_REGISTERED or settings is None:
        return

    # Bounded default: cheap enough to run in the standard suite on every push.
    settings.register_profile(
        _DEFAULT_PROFILE,
        max_examples=40,
        deadline=400,  # milliseconds per example
        derandomize=True,  # deterministic example stream for a fixed seed
        print_blob=True,
        suppress_health_check=[HealthCheck.too_slow] if HealthCheck else [],
    )

    # Heavier scheduled sweep: broader coverage, still time-bounded per example.
    settings.register_profile(
        _NIGHTLY_PROFILE,
        max_examples=1000,
        deadline=1000,
        derandomize=False,
        print_blob=True,
        suppress_health_check=[HealthCheck.too_slow] if HealthCheck else [],
    )

    _PROFILES_REGISTERED = True


def _load_selected_profile() -> None:
    """Load the profile named by ``HYPOTHESIS_PROFILE`` or the bounded default."""
    if settings is None:
        return
    _register_profiles()
    profile = os.environ.get("HYPOTHESIS_PROFILE", _DEFAULT_PROFILE)
    try:
        settings.load_profile(profile)
    except Exception:  # pragma: no cover - fall back to the bounded profile.
        settings.load_profile(_DEFAULT_PROFILE)


# Register + load at import so the settings apply to every test in this package.
_load_selected_profile()


@pytest.fixture(scope="session", autouse=True)
def _fuzz_profile_loaded() -> None:
    """Ensure the bounded/nightly profile is active for the fuzz package."""
    _load_selected_profile()
