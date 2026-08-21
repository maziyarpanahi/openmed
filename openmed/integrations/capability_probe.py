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
import unicodedata
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, NoReturn, TypeAlias, cast

CAPABILITY_PROBE_SCHEMA_VERSION = "openmed.integrations.capability_probe.v1"
"""Stable schema identifier for :class:`CapabilityProbeReport`."""

MAX_CAPABILITY_ADAPTERS = 10_000
"""Maximum number of declarations accepted by one bounded report."""

_MAX_SOURCE_TEXT_LENGTH = 16_384
_MAX_REASON_LENGTH = 256

_SAFE_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,63}$")
_GENERATED_IDENTIFIER_RE = re.compile(r"(?:capability|extra)-[0-9a-f]{16}")
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
_SAFE_REASONS = frozenset(
    {"available", "invalid_result", "missing_extra", "probe_error", "unavailable"}
)
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")
_UUID_RE = re.compile(
    r"(?<![a-z0-9])[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}(?![a-z0-9])"
)
_NUMERIC_IDENTIFIER_RE = re.compile(r"(?<![a-z0-9])\d{6,}(?![a-z0-9])")
_NAMED_IDENTIFIER_RE = re.compile(
    r"(?:^|[_.:-])(?:account|case|encounter|member|mrn|patient|record|subject)"
    r"[_.:-]?(?:\d{2,}|[a-f0-9]{8,})(?:$|[_.:-])"
)
_DECLARATION_FIELDS = frozenset(
    {
        "available",
        "capability",
        "check",
        "extra",
        "is_available",
        "name",
        "probe",
        "provider",
        "version",
    }
)
_RESULT_FIELDS = frozenset({"available", "ok", "reason", "status"})

ProbeCallable: TypeAlias = Callable[[], Any]


class CapabilityProbeError(ValueError):
    """Raised when declarations cannot be inspected safely and deterministically."""


def _fail(reason: str) -> NoReturn:
    raise CapabilityProbeError(reason) from None


@dataclass(frozen=True, slots=True, repr=False)
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
        if type(self.name) is not str:
            raise ValueError("capability adapter name must be a non-empty string")
        if len(self.name) > _MAX_SOURCE_TEXT_LENGTH:
            raise ValueError("capability adapter name exceeds the safe length limit")
        if not self.name.strip():
            raise ValueError("capability adapter name must be a non-empty string")
        if not callable(self.probe):
            raise TypeError("capability adapter probe must be callable")
        for value, field_name in (
            (self.provider, "provider"),
            (self.extra, "extra"),
            (self.version, "version"),
        ):
            if value is not None:
                if type(value) is not str:
                    raise TypeError(f"capability adapter {field_name} must be a string")
                if len(value) > _MAX_SOURCE_TEXT_LENGTH:
                    raise ValueError(
                        f"capability adapter {field_name} exceeds the safe length limit"
                    )

    def __repr__(self) -> str:
        """Render the declaration without raw names, providers, or versions."""

        return (
            "CapabilityAdapter(name=<redacted>, probe=<callable>, "
            "provider=<redacted>, extra=<redacted>, version=<redacted>)"
        )


@dataclass(frozen=True, slots=True, repr=False)
class CapabilityCheck:
    """Optional structured return value for an adapter probe."""

    available: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.available) is not bool:
            raise TypeError("capability availability must be a boolean")
        if self.reason is not None:
            if type(self.reason) is not str:
                raise TypeError("capability reason must be a string")
            if len(self.reason) > _MAX_REASON_LENGTH:
                raise ValueError("capability reason exceeds the safe length limit")

    def __repr__(self) -> str:
        """Render availability without retaining a caller-supplied reason."""

        reason = "<classified>" if self.reason is not None else "None"
        return f"CapabilityCheck(available={self.available!r}, reason={reason})"


@dataclass(frozen=True, slots=True)
class CapabilityStatus:
    """Safe, serializable result for one capability declaration."""

    name: str
    available: bool
    reason: str
    extra: str | None
    provider_fingerprint: str | None

    def __post_init__(self) -> None:
        if (
            type(self.name) is not str
            or len(self.name) > 64
            or _SAFE_IDENTIFIER_RE.fullmatch(self.name) is None
            or _safe_identifier(self.name, prefix="capability") != self.name
        ):
            raise ValueError("capability status name must be a safe identifier")
        if type(self.available) is not bool:
            raise TypeError("capability status availability must be a boolean")
        if type(self.reason) is not str or self.reason not in _SAFE_REASONS:
            raise ValueError("capability status reason is not supported")
        if self.available != (self.reason == "available"):
            raise ValueError("capability status availability and reason disagree")
        if self.extra is not None and (
            type(self.extra) is not str
            or len(self.extra) > 64
            or _SAFE_IDENTIFIER_RE.fullmatch(self.extra) is None
            or _safe_identifier(self.extra, prefix="extra") != self.extra
        ):
            raise ValueError("capability status extra must be a safe identifier")
        if self.provider_fingerprint is not None and (
            type(self.provider_fingerprint) is not str
            or len(self.provider_fingerprint) != len("sha256:") + 64
            or _SHA256_RE.fullmatch(self.provider_fingerprint) is None
        ):
            raise ValueError("provider_fingerprint must be a SHA-256 digest")

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


@dataclass(frozen=True, slots=True)
class CapabilityProbeReport:
    """Deterministic aggregate of locally probed capability statuses."""

    capabilities: tuple[CapabilityStatus, ...]
    schema_version: str = CAPABILITY_PROBE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != CAPABILITY_PROBE_SCHEMA_VERSION
        ):
            raise ValueError("capability report schema_version is not supported")
        capabilities = _bounded_tuple(
            self.capabilities,
            label="capability statuses",
            maximum=MAX_CAPABILITY_ADAPTERS,
        )
        if any(type(status) is not CapabilityStatus for status in capabilities):
            raise TypeError("capability reports require CapabilityStatus entries")
        validated = tuple(
            CapabilityStatus(
                name=status.name,
                available=status.available,
                reason=status.reason,
                extra=status.extra,
                provider_fingerprint=status.provider_fingerprint,
            )
            for status in cast(tuple[CapabilityStatus, ...], capabilities)
        )
        ordered = tuple(sorted(validated, key=_status_key))
        names = tuple(status.name for status in ordered)
        if len(names) != len(set(names)):
            raise CapabilityProbeError("capability declarations must have unique names")
        object.__setattr__(self, "capabilities", ordered)

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

        if indent is None:
            return json.dumps(
                self.as_dict(),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        if type(indent) is not int:
            raise TypeError("indent must be an integer or None")
        if not 0 <= indent <= 8:
            raise ValueError("indent must be between 0 and 8")
        return json.dumps(
            self.as_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=indent,
            sort_keys=True,
        )


def provider_fingerprint(
    provider: str | None, *, version: str | None = None
) -> str | None:
    """Return a one-way SHA-256 fingerprint for a provider declaration.

    Provider identifiers are normalized for surrounding whitespace and case.
    Empty declarations return ``None``.  The raw identifier and version never
    appear in a report, exception, or log produced by this module.
    """

    if provider is None:
        return None
    if type(provider) is not str:
        raise TypeError("provider must be a string or None")
    if len(provider) > _MAX_SOURCE_TEXT_LENGTH:
        _fail("provider exceeds the safe length limit")
    if not provider.strip():
        return None
    canonical_provider = unicodedata.normalize("NFKC", provider.strip().casefold())
    canonical_version: str | None = None
    if version is not None:
        if type(version) is not str:
            raise TypeError("version must be a string or None")
        if len(version) > _MAX_SOURCE_TEXT_LENGTH:
            _fail("version exceeds the safe length limit")
        if version.strip():
            canonical_version = unicodedata.normalize(
                "NFKC", version.strip().casefold()
            )
    return _sha256_json([canonical_provider, canonical_version])


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

    statuses = [_probe_adapter(adapter) for adapter in _collect_adapters(adapters)]
    return CapabilityProbeReport(tuple(statuses))


def probe_capability(
    adapter: CapabilityAdapter | Mapping[str, Any] | object,
) -> CapabilityStatus:
    """Probe one adapter and return its safe status entry."""

    return probe_capabilities((adapter,)).capabilities[0]


def _status_key(status: CapabilityStatus) -> tuple[str, str, str, str, bool]:
    return (
        status.name,
        status.provider_fingerprint or "",
        status.extra or "",
        status.reason,
        status.available,
    )


def _bounded_tuple(value: Any, *, label: str, maximum: int) -> tuple[Any, ...]:
    try:
        iterator = iter(value)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        _fail(f"{label} must be a bounded iterable")
    collected: list[Any] = []
    for _ in range(maximum + 1):
        try:
            item = next(iterator)
        except StopIteration:
            return tuple(collected)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            _fail(f"{label} iteration failed")
        if len(collected) == maximum:
            _fail(f"{label} exceed the limit of {maximum}")
        collected.append(item)
    raise AssertionError("unreachable")


def _snapshot_mapping(
    value: Mapping[Any, Any],
    *,
    maximum: int,
    allowed: frozenset[str] | None = None,
) -> dict[Any, Any]:
    pairs = _bounded_tuple(value, label="mapping keys", maximum=maximum)
    snapshot: dict[Any, Any] = {}
    for key in pairs:
        if type(key) is not str or len(key) > _MAX_SOURCE_TEXT_LENGTH:
            _fail("mapping contains unsupported fields")
        if allowed is not None and key not in allowed:
            _fail("mapping contains unsupported fields")
        try:
            if key in snapshot:
                _fail("mapping contains duplicate fields")
            snapshot[key] = value[key]
        except CapabilityProbeError:
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            _fail("mapping could not be read safely")
    return snapshot


def _collect_adapters(adapters: Any) -> tuple[Any, ...]:
    if adapters is None:
        return ()
    if isinstance(adapters, CapabilityAdapter) or callable(adapters):
        return (adapters,)
    if isinstance(adapters, Mapping):
        snapshot = _snapshot_mapping(
            adapters,
            maximum=MAX_CAPABILITY_ADAPTERS,
        )
        name_fields = {"capability", "name"}
        probe_fields = {"available", "check", "is_available", "probe"}
        if name_fields.intersection(snapshot) and probe_fields.intersection(snapshot):
            return (snapshot,)
        return tuple(
            _with_default_name(value, name) for name, value in snapshot.items()
        )
    return _bounded_tuple(
        adapters,
        label="capability declarations",
        maximum=MAX_CAPABILITY_ADAPTERS,
    )


def _with_default_name(value: Any, name: Any) -> Any:
    if isinstance(value, Mapping):
        fields = _snapshot_mapping(
            value,
            maximum=len(_DECLARATION_FIELDS),
            allowed=_DECLARATION_FIELDS,
        )
        if "name" not in fields and "capability" not in fields:
            fields["name"] = name
        return fields
    return {"name": name, "probe": value}


def _coerce_adapter(raw: Any) -> CapabilityAdapter:
    if isinstance(raw, CapabilityAdapter):
        try:
            return CapabilityAdapter(
                name=raw.name,
                probe=raw.probe,
                provider=raw.provider,
                extra=raw.extra,
                version=raw.version,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            return _invalid_adapter()

    if isinstance(raw, Mapping):
        try:
            fields = _snapshot_mapping(
                raw,
                maximum=len(_DECLARATION_FIELDS),
                allowed=_DECLARATION_FIELDS,
            )
        except CapabilityProbeError:
            return _invalid_adapter()
        if sum(key in fields for key in ("name", "capability")) > 1:
            return _invalid_adapter()
        if sum(key in fields for key in ("probe", "check", "is_available")) > 1:
            return _invalid_adapter()
        if "available" in fields and any(
            key in fields for key in ("probe", "check", "is_available")
        ):
            return _invalid_adapter()
        name = fields.get("name", fields.get("capability", "capability-unknown"))
        probe = fields.get("probe", fields.get("check", fields.get("is_available")))
        if probe is None and type(fields.get("available")) is bool:
            declared_available = fields["available"]
            probe = lambda: declared_available
        return _make_adapter(
            name=name,
            probe=probe,
            provider=fields.get("provider"),
            extra=fields.get("extra"),
            version=fields.get("version"),
        )

    name = _read_attribute(raw, "name")
    if name is None:
        name = _read_attribute(raw, "capability")
    probe = _read_attribute(raw, "probe")
    if probe is None:
        probe = _read_attribute(raw, "check")
    if probe is None:
        probe = _read_attribute(raw, "is_available")
    declared_available = _read_attribute(raw, "available")
    if probe is None and type(declared_available) is bool:
        probe = lambda: declared_available
    if probe is None and callable(raw):
        probe = raw
    if name is None and callable(raw):
        name = _read_attribute(raw, "__name__")
    return _make_adapter(
        name=name if name is not None else "capability-unknown",
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
    safe_name = (
        name
        if type(name) is str and len(name) <= _MAX_SOURCE_TEXT_LENGTH and name.strip()
        else "capability-unknown"
    )
    safe_provider = provider if type(provider) is str else None
    safe_extra = extra if type(extra) is str else None
    safe_version = version if type(version) is str else None
    if safe_provider is not None and len(safe_provider) > _MAX_SOURCE_TEXT_LENGTH:
        safe_provider = None
    if safe_extra is not None and len(safe_extra) > _MAX_SOURCE_TEXT_LENGTH:
        safe_extra = None
    if safe_version is not None and len(safe_version) > _MAX_SOURCE_TEXT_LENGTH:
        safe_version = None
    if not callable(probe):
        return _invalid_adapter(
            name=safe_name,
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


def _invalid_adapter(
    *,
    name: str = "capability-unknown",
    provider: str | None = None,
    extra: str | None = None,
    version: str | None = None,
) -> CapabilityAdapter:
    return CapabilityAdapter(
        name=name,
        probe=lambda: CapabilityCheck(False, "invalid_result"),
        provider=provider,
        extra=extra,
        version=version,
    )


def _probe_adapter(raw: Any) -> CapabilityStatus:
    adapter = _coerce_adapter(raw)
    name = _safe_identifier(adapter.name, prefix="capability") or "capability-unknown"
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
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        return CapabilityStatus(
            name=name,
            available=False,
            reason="probe_error",
            extra=extra,
            provider_fingerprint=fingerprint,
        )

    try:
        available, reason = _interpret_result(result, has_extra=extra is not None)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        available, reason = False, "invalid_result"
    return CapabilityStatus(
        name=name,
        available=available,
        reason=reason,
        extra=extra,
        provider_fingerprint=fingerprint,
    )


def _interpret_result(result: Any, *, has_extra: bool) -> tuple[bool, str]:
    if type(result) is CapabilityCheck:
        available = result.available
        raw_reason = result.reason
    elif type(result) is bool:
        available = result
        raw_reason = None
    elif isinstance(result, Mapping):
        try:
            fields = _snapshot_mapping(
                result,
                maximum=len(_RESULT_FIELDS),
                allowed=_RESULT_FIELDS,
            )
        except CapabilityProbeError:
            return False, "invalid_result"
        if "available" in fields and "ok" in fields:
            return False, "invalid_result"
        if "reason" in fields and "status" in fields:
            return False, "invalid_result"
        raw_available = fields.get("available", fields.get("ok"))
        raw_reason = fields.get("reason", fields.get("status"))
        if type(raw_available) is not bool:
            return False, "invalid_result"
        available = cast(bool, raw_available)
    else:
        available = _read_attribute(result, "available")
        raw_reason = _read_attribute(result, "reason")
        if type(available) is not bool:
            return False, "invalid_result"

    if available:
        return True, "available"
    return False, _safe_reason(raw_reason, has_extra=has_extra)


def _safe_reason(raw_reason: Any, *, has_extra: bool) -> str:
    if type(raw_reason) is str and len(raw_reason) <= _MAX_REASON_LENGTH:
        normalized = raw_reason.strip().casefold().replace("-", "_").replace(" ", "_")
        if normalized in _MISSING_EXTRA_REASONS:
            return "missing_extra" if has_extra else "unavailable"
        if normalized in _PROBE_ERROR_REASONS:
            return "probe_error"
        if normalized == "invalid_result":
            return "invalid_result"
        if normalized in _UNAVAILABLE_REASONS:
            return "unavailable"
    return "missing_extra" if has_extra else "unavailable"


def _read_attribute(value: Any, name: str) -> Any:
    try:
        return getattr(value, name, None)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        return None


def _safe_identifier(value: Any, *, prefix: str) -> str | None:
    if value is None or type(value) is not str or not value.strip():
        return None if prefix == "extra" else f"{prefix}-unknown"
    normalized = value.strip().casefold()
    if _GENERATED_IDENTIFIER_RE.fullmatch(normalized):
        return normalized
    looks_sensitive = bool(
        _UUID_RE.search(normalized)
        or _NUMERIC_IDENTIFIER_RE.search(normalized)
        or _NAMED_IDENTIFIER_RE.search(normalized)
    )
    if _SAFE_IDENTIFIER_RE.fullmatch(normalized) and not looks_sensitive:
        return normalized
    digest = _sha256_text(normalized).split(":", 1)[1]
    return f"{prefix}-{digest[:16]}"


def _sha256_text(value: str) -> str:
    encoded = value.encode("utf-8", errors="surrogatepass")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _sha256_json(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        _fail("capability report could not be serialized safely")
    return _sha256_text(encoded)


__all__ = [
    "CAPABILITY_PROBE_SCHEMA_VERSION",
    "MAX_CAPABILITY_ADAPTERS",
    "CapabilityAdapter",
    "CapabilityCheck",
    "CapabilityProbeError",
    "CapabilityProbeReport",
    "CapabilityStatus",
    "probe_capabilities",
    "probe_capability",
    "provider_fingerprint",
]
