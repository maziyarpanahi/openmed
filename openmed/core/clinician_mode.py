"""Explicit, task-local clinician review scopes for restricted capabilities."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from functools import wraps
from typing import ParamSpec, TypeVar

from openmed.core.disclaimer import (
    NoticeAcknowledgement,
    OpenMedDisclaimer,
    notice_catalog,
)
from openmed.core.intended_use import RegulatoryTier, load_intended_use_registry
from openmed.core.no_phi_telemetry import record_use_error_if_enabled


class ClinicianPurpose(StrEnum):
    """Permitted operator purposes for a restricted review scope."""

    REVIEW = "review"
    QUALITY_ASSURANCE = "quality_assurance"
    RESEARCH = "research"


class ClinicianModeError(ValueError):
    """A restricted operation was refused with a value-free reason code."""


@dataclass(frozen=True, slots=True)
class ClinicianModeReceipt:
    """Codes and version digest describing an explicitly acknowledged scope."""

    purpose: ClinicianPurpose
    notice_id: str
    notice_digest: str


@dataclass(slots=True)
class _Scope:
    receipt: ClinicianModeReceipt
    active: bool = True


_current: ContextVar[_Scope | None] = ContextVar(
    "openmed_clinician_scope", default=None
)


@contextmanager
def clinician_mode(
    *,
    purpose: ClinicianPurpose | str,
    acknowledgement: NoticeAcknowledgement | None = None,
) -> Iterator[ClinicianModeReceipt]:
    """Enable restricted review in this execution context until the block exits."""
    try:
        reviewed_purpose = ClinicianPurpose(purpose)
    except (TypeError, ValueError):
        raise ClinicianModeError("invalid_clinician_purpose") from None
    notice = OpenMedDisclaimer.for_tier(RegulatoryTier.RESTRICTED)
    if isinstance(acknowledgement, NoticeAcknowledgement):
        published = (
            notice_catalog().get(acknowledgement.notice_id)
            if isinstance(acknowledgement.notice_id, str)
            else None
        )
        if published is None or published.tier != RegulatoryTier.RESTRICTED:
            raise ClinicianModeError("invalid_acknowledgement")
        notice = published
    try:
        notice.require_acknowledgement(acknowledgement, capability="clinician_mode")
    except ValueError as exc:
        raise ClinicianModeError(str(exc)) from None

    scope = _Scope(
        ClinicianModeReceipt(reviewed_purpose, notice.notice_id, notice.version_digest)
    )
    token = _current.set(scope)
    try:
        yield scope.receipt
    finally:
        scope.active = False
        _current.reset(token)


def require_clinician_mode(capability: str) -> ClinicianModeReceipt | None:
    """Return the active receipt or refuse a capability that requires clinician mode."""
    declaration = load_intended_use_registry().resolve(capability)
    if "clinician_mode_required" not in declaration.regulatory_tier.obligations:
        return None
    scope = _current.get()
    if scope is None or not scope.active:
        record_use_error_if_enabled(
            "restricted_capability_refused",
            capability,
            declaration.regulatory_tier.value,
        )
        raise ClinicianModeError("clinician_mode_required")
    return scope.receipt


_P = ParamSpec("_P")
_T = TypeVar("_T")


def restricted_entry_point(
    capability: str,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """Enforce the registered capability's scope before invoking a producer."""

    def decorate(func: Callable[_P, _T]) -> Callable[_P, _T]:
        @wraps(func)
        def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            require_clinician_mode(capability)
            return func(*args, **kwargs)

        return guarded

    return decorate
