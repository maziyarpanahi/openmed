"""Deterministic, local-only capability reports for injected adapters.

The probe deliberately knows nothing about optional providers.  Callers declare
an adapter and inject a zero-argument probe that only inspects local state.  A
report contains counts and one-way provider fingerprints, never provider
configuration, credentials, or exception text.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

CAPABILITY_PROBE_SCHEMA_VERSION = "openmed.integrations.capability_probe.v1"
"""Stable schema identifier for :class:`CapabilityProbeReport`."""

_SAFE_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,63}$")
_MISSING_EXTRA_REASONS = frozenset(
    {"missing_dependency", "missing_extra", "not_installed", "unavailable_extra"}
)
_UNAVAILABLE_REASONS = frozenset(
    {
        "disabled",
        "not_available",
        "requires_configuration",
        "unavailable",
    }
)
_PROBE_ERROR_REASONS = frozenset({"error", "exception", "probe_error"})

ProbeCallable: TypeAlias = Callable[[], Any]


@dataclass(frozen=True)
class CapabilityAdapter:
    """Declaration for one locally probed optional capability.

    Args:
        name: Non-sensitive capability identifier, such as ``"pandas"``.
        probe: Zero-argument callable that returns a boolean, a
            :class:`CapabilityCheck`, or a mapping with an ``available``
            boolean.  The callable is supplied by the application and is
            expected to inspect local state only.
        provider: Optional provider identifier.  It is never emitted raw;
            reports contain only its SHA-256 fingerprint.
        extra: Optional OpenMed extra name used to classify a false result as
            a missing optional extra.
        version: Optional provider version, included in the provider
            fingerprint but never emitted raw.
    """

    name: str
    probe: ProbeCallable
    provider: str | None = None
    extra: str | None = None
    version: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("capability adapter name must be a non-empty string")
        if not callable(self.probe):
            raise TypeError("capability adapter probe must be callable")
        for value, field_name in (
            (self.provider, "provider"),
            (self.extra, "extra"),
            (self.version, "version"),
        ):
            if value is not None and not isinstance(value, str):
                raise TypeError(f"capability adapter {field_name} must be a string")


@dataclass(frozen=True)
class CapabilityCheck:
    """Optional structured return value for an adapter probe."""

    available: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.available) is not bool:
            raise TypeError("capability availability must be a boolean")


@dataclass(frozen=True)
class CapabilityStatus:
    """Safe, serializable result for one capability declaration."""

    name: str
    available: bool
    reason: str
    extra: str | None
    provider_fingerprint: str | None

    @property
    def status(self) -> str:
        """Return the stable string form of the availability state."""

        return "available" if self.available else "unavailable"

    @property
    def missing_extra(self) -> bool:
        """Return whether the unavailable result points to a missing extra."""

        return self.reason == "missing_extra"

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe status without raw provider or error values."""

        return {
            "name": self.name,
            "status": self.status,
            "available": self.available,
            "reason": self.reason,
            "extra": self.extra,
            "provider_fingerprint": self.provider_fingerprint,
        }

    to_dict = as_dict


@dataclass(frozen=True)
class CapabilityProbeReport:
    """Deterministic aggregate of locally probed capability statuses."""

    capabilities: tuple[CapabilityStatus, ...]
    schema_version: str = CAPABILITY_PROBE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "capabilities", tuple(self.capabilities))

    @property
    def entries(self) -> tuple[CapabilityStatus, ...]:
        """Alias for callers that use report-entry terminology."""

        return self.capabilities

    @property
    def counts(self) -> dict[str, int]:
        """Return stable total, available, and unavailable counts."""

        available = sum(status.available for status in self.capabilities)
        return {
            "total": len(self.capabilities),
            "available": available,
            "unavailable": len(self.capabilities) - available,
        }

    @property
    def available_count(self) -> int:
        """Return the number of locally available capabilities."""

        return self.counts["available"]

    @property
    def unavailable_count(self) -> int:
        """Return the number of locally unavailable capabilities."""

        return self.counts["unavailable"]

    @property
    def provider_fingerprints(self) -> tuple[str, ...]:
        """Return unique provider fingerprints in stable order."""

        return tuple(
            sorted(
                {
                    status.provider_fingerprint
                    for status in self.capabilities
                    if status.provider_fingerprint is not None
                }
            )
        )

    @property
    def fingerprint(self) -> str:
        """Return a stable fingerprint of the safe report contents."""

        payload = {
            "schema_version": self.schema_version,
            "counts": self.counts,
            "capabilities": [status.as_dict() for status in self.capabilities],
            "provider_fingerprints": list(self.provider_fingerprints),
        }
        return _sha256_json(payload)

    def as_dict(self) -> dict[str, object]:
        """Return the complete JSON-safe report."""

        return {
            "schema_version": self.schema_version,
            "counts": self.counts,
            "capabilities": [status.as_dict() for status in self.capabilities],
            "provider_fingerprints": list(self.provider_fingerprints),
            "fingerprint": self.fingerprint,
        }

    to_dict = as_dict

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the report with stable key ordering."""

        kwargs: dict[str, object] = {
            "allow_nan": False,
            "ensure_ascii": True,
            "sort_keys": True,
        }
        if indent is None:
            kwargs["separators"] = (",", ":")
        else:
            kwargs["indent"] = indent
        return json.dumps(self.as_dict(), **kwargs)


def provider_fingerprint(
    provider: str | None, *, version: str | None = None
) -> str | None:
    """Return a one-way SHA-256 fingerprint for a provider declaration.

    Provider identifiers are normalized for surrounding whitespace and case.
    Empty declarations return ``None``.  The raw identifier and version never
    appear in a report, exception, or log produced by this module.
    """

    if provider is None or not isinstance(provider, str) or not provider.strip():
        return None
    canonical = provider.strip().casefold()
    if version is not None and isinstance(version, str) and version.strip():
        canonical = f"{canonical}\x00{version.strip().casefold()}"
    return _sha256_text(canonical)


def probe_capabilities(
    adapters: (
        Iterable[CapabilityAdapter | Mapping[str, Any] | object]
        | Mapping[str, CapabilityAdapter | Mapping[str, Any] | object]
        | CapabilityAdapter
        | None
    ) = None,
) -> CapabilityProbeReport:
    """Probe injected local adapters and return a deterministic report.

    The function performs no package discovery, credential loading, or network
    operation.  It invokes only the zero-argument callables supplied by the
    caller.  Probe exceptions are intentionally reduced to a fixed safe
    classification; their messages are never retained.

    A mapping with ``name`` and ``probe`` keys is treated as one declaration.
    A mapping without those declaration keys is treated as
    ``{capability_name: probe_or_declaration}``, which is convenient for small
    local registries.
    """

    statuses = [_probe_adapter(adapter) for adapter in _iter_adapters(adapters)]
    statuses.sort(
        key=lambda status: (
            status.name,
            status.provider_fingerprint or "",
            status.extra or "",
            status.reason,
            status.available,
        )
    )
    return CapabilityProbeReport(tuple(statuses))


def probe_capability(
    adapter: CapabilityAdapter | Mapping[str, Any] | object,
) -> CapabilityStatus:
    """Probe one adapter and return its safe status entry."""

    return probe_capabilities((adapter,)).capabilities[0]


def _iter_adapters(adapters: Any) -> Iterable[Any]:
    if adapters is None:
        return ()
    if isinstance(adapters, CapabilityAdapter) or callable(adapters):
        return (adapters,)
    if isinstance(adapters, Mapping):
        declaration_keys = {"name", "capability", "probe", "check", "is_available"}
        if declaration_keys.intersection(adapters):
            return (adapters,)
        return (_with_default_name(value, name) for name, value in adapters.items())
    return adapters


def _with_default_name(value: Any, name: Any) -> Any:
    if isinstance(value, Mapping):
        return {"name": name, **dict(value)}
    return {"name": name, "probe": value}


def _coerce_adapter(raw: Any) -> CapabilityAdapter:
    if isinstance(raw, CapabilityAdapter):
        return raw

    if isinstance(raw, Mapping):
        name = raw.get("name", raw.get("capability", "capability-unknown"))
        probe = raw.get("probe", raw.get("check", raw.get("is_available")))
        if probe is None and type(raw.get("available")) is bool:
            declared_available = raw["available"]
            probe = lambda: declared_available
        return _make_adapter(
            name=name,
            probe=probe,
            provider=raw.get("provider"),
            extra=raw.get("extra"),
            version=raw.get("version"),
        )

    name = _read_attribute(raw, "name") or _read_attribute(raw, "capability")
    probe = _read_attribute(raw, "probe")
    if probe is None:
        probe = _read_attribute(raw, "check")
    if probe is None:
        probe = _read_attribute(raw, "is_available")
    if probe is None and type(_read_attribute(raw, "available")) is bool:
        declared_available = _read_attribute(raw, "available")
        probe = lambda: declared_available
    if probe is None and callable(raw):
        probe = raw
    if name is None and callable(raw):
        name = _read_attribute(raw, "__name__")
    return _make_adapter(
        name=name or "capability-unknown",
        probe=probe,
        provider=_read_attribute(raw, "provider"),
        extra=_read_attribute(raw, "extra"),
        version=_read_attribute(raw, "version"),
    )


def _make_adapter(
    *,
    name: Any,
    probe: Any,
    provider: Any,
    extra: Any,
    version: Any,
) -> CapabilityAdapter:
    safe_name = name if isinstance(name, str) and name.strip() else "capability-unknown"
    safe_provider = provider if isinstance(provider, str) else None
    safe_extra = extra if isinstance(extra, str) else None
    safe_version = version if isinstance(version, str) else None
    if not callable(probe):
        return CapabilityAdapter(
            name=safe_name,
            probe=lambda: CapabilityCheck(False, "invalid_result"),
            provider=safe_provider,
            extra=safe_extra,
            version=safe_version,
        )
    return CapabilityAdapter(
        name=safe_name,
        probe=probe,
        provider=safe_provider,
        extra=safe_extra,
        version=safe_version,
    )


def _probe_adapter(raw: Any) -> CapabilityStatus:
    adapter = _coerce_adapter(raw)
    name = _safe_identifier(adapter.name, prefix="capability")
    extra = _safe_identifier(adapter.extra, prefix="extra")
    fingerprint = provider_fingerprint(adapter.provider, version=adapter.version)

    try:
        result = adapter.probe()
    except ImportError:
        return CapabilityStatus(
            name=name,
            available=False,
            reason="missing_extra" if extra else "unavailable",
            extra=extra,
            provider_fingerprint=fingerprint,
        )
    except Exception:
        return CapabilityStatus(
            name=name,
            available=False,
            reason="probe_error",
            extra=extra,
            provider_fingerprint=fingerprint,
        )

    available, reason = _interpret_result(result, has_extra=extra is not None)
    return CapabilityStatus(
        name=name,
        available=available,
        reason=reason,
        extra=extra,
        provider_fingerprint=fingerprint,
    )


def _interpret_result(result: Any, *, has_extra: bool) -> tuple[bool, str]:
    if isinstance(result, CapabilityCheck):
        available = result.available
        raw_reason = result.reason
    elif type(result) is bool:
        available = result
        raw_reason = None
    elif isinstance(result, Mapping):
        available = result.get("available", result.get("ok"))
        raw_reason = result.get("reason", result.get("status"))
        if type(available) is not bool:
            return False, "invalid_result"
    else:
        available = _read_attribute(result, "available")
        raw_reason = _read_attribute(result, "reason")
        if type(available) is not bool:
            return False, "invalid_result"

    if available:
        return True, "available"
    return False, _safe_reason(raw_reason, has_extra=has_extra)


def _safe_reason(raw_reason: Any, *, has_extra: bool) -> str:
    if isinstance(raw_reason, str):
        normalized = raw_reason.strip().casefold().replace("-", "_").replace(" ", "_")
        if normalized in _MISSING_EXTRA_REASONS:
            return "missing_extra" if has_extra else "unavailable"
        if normalized in _PROBE_ERROR_REASONS:
            return "probe_error"
        if normalized in _UNAVAILABLE_REASONS:
            return "unavailable"
    return "missing_extra" if has_extra else "unavailable"


def _read_attribute(value: Any, name: str) -> Any:
    try:
        return getattr(value, name, None)
    except Exception:
        return None


def _safe_identifier(value: str | None, *, prefix: str) -> str | None:
    if value is None or not isinstance(value, str) or not value.strip():
        return None if prefix == "extra" else f"{prefix}-unknown"
    normalized = value.strip().casefold()
    if _SAFE_IDENTIFIER_RE.fullmatch(normalized):
        return normalized
    digest = _sha256_text(normalized).split(":", 1)[1]
    return f"{prefix}-{digest[:16]}"


def _sha256_text(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return _sha256_text(encoded)


__all__ = [
    "CAPABILITY_PROBE_SCHEMA_VERSION",
    "CapabilityAdapter",
    "CapabilityCheck",
    "CapabilityProbeReport",
    "CapabilityStatus",
    "probe_capabilities",
    "probe_capability",
    "provider_fingerprint",
]
