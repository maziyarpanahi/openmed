"""Production guard for the test-only service fault-injection harness.

The harness lives under ``tests/`` and is never packaged into the ``openmed``
wheel. As a second line of defence it refuses to import when the process is
explicitly configured with the production profile, and it refuses to arm an
injector unless the process selected the ``test`` profile. No environment
variable, configuration key or endpoint can *enable* a fault; the profile can
only *deny* one. Errors carry a fixed reason code and never echo values.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from openmed.core.config import PROFILE_ENV_VAR

# ``ServiceRuntime.from_env`` treats an unset profile as production.
DEFAULT_SERVICE_PROFILE = "prod"
PRODUCTION_PROFILES = frozenset({DEFAULT_SERVICE_PROFILE})
PERMITTED_PROFILES = frozenset({"test"})

IMPORT_FORBIDDEN_CODE = "fault_harness_production_profile"
ARM_FORBIDDEN_CODE = "fault_injection_profile_forbidden"
NOT_ARMED_CODE = "fault_injector_not_armed"
ALREADY_ARMED_CODE = "fault_injector_already_armed"


class FaultInjectionForbiddenError(RuntimeError):
    """Raised with a fixed reason code when fault injection is not permitted."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def resolved_profile(environ: Mapping[str, str] | None = None) -> str:
    """Return the service profile the process would run with.

    Args:
        environ: Environment mapping to inspect; defaults to ``os.environ``.

    Returns:
        The stripped, lower-cased profile name, or ``"prod"`` when unset.
    """
    source = os.environ if environ is None else environ
    raw = source.get(PROFILE_ENV_VAR)
    if raw is None or not raw.strip():
        return DEFAULT_SERVICE_PROFILE
    return raw.strip().lower()


def refuse_production_import(environ: Mapping[str, str] | None = None) -> None:
    """Refuse to load the harness into an explicitly production process.

    Args:
        environ: Environment mapping to inspect; defaults to ``os.environ``.

    Raises:
        FaultInjectionForbiddenError: When the profile is explicitly production.
    """
    source = os.environ if environ is None else environ
    raw = source.get(PROFILE_ENV_VAR)
    if raw is not None and raw.strip().lower() in PRODUCTION_PROFILES:
        raise FaultInjectionForbiddenError(IMPORT_FORBIDDEN_CODE)


def require_arm_permitted(environ: Mapping[str, str] | None = None) -> None:
    """Allow arming only under the explicit ``test`` profile.

    Args:
        environ: Environment mapping to inspect; defaults to ``os.environ``.

    Raises:
        FaultInjectionForbiddenError: For production, unset or any other profile.
    """
    if resolved_profile(environ) not in PERMITTED_PROFILES:
        raise FaultInjectionForbiddenError(ARM_FORBIDDEN_CODE)


__all__ = [
    "ALREADY_ARMED_CODE",
    "ARM_FORBIDDEN_CODE",
    "DEFAULT_SERVICE_PROFILE",
    "FaultInjectionForbiddenError",
    "IMPORT_FORBIDDEN_CODE",
    "NOT_ARMED_CODE",
    "PERMITTED_PROFILES",
    "PRODUCTION_PROFILES",
    "refuse_production_import",
    "require_arm_permitted",
    "resolved_profile",
]
