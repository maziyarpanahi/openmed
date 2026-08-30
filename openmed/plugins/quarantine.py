"""Importless plugin metadata availability and quarantine reports.

This module evaluates caller-supplied, static metadata mappings.  It does not
enumerate installed distributions, resolve entry points, import plugin code,
read credentials, or contact a package index.  The report is therefore safe to
use as a local preflight before a plugin loader is allowed to run.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from .protocols import (
    COMPONENT_ANONYMIZER_PROVIDER,
    COMPONENT_EXPORTER,
    COMPONENT_INTEROP_ADAPTER,
    COMPONENT_LANGUAGE_PACK,
    COMPONENT_RECOGNIZER,
    PLUGIN_COMPONENT_KINDS,
    PLUGIN_SDK_MAJOR,
    PLUGIN_SDK_VERSION,
)

PLUGIN_API_VERSION: Final = PLUGIN_SDK_VERSION
SUPPORTED_PLUGIN_API_MAJOR: Final = PLUGIN_SDK_MAJOR
PLUGIN_API_MAJOR: Final = SUPPORTED_PLUGIN_API_MAJOR

CAPABILITY_RECOGNIZER: Final = COMPONENT_RECOGNIZER
CAPABILITY_ANONYMIZER_PROVIDER: Final = COMPONENT_ANONYMIZER_PROVIDER
CAPABILITY_EXPORTER: Final = COMPONENT_EXPORTER
CAPABILITY_INTEROP_ADAPTER: Final = COMPONENT_INTEROP_ADAPTER
CAPABILITY_LANGUAGE_PACK: Final = COMPONENT_LANGUAGE_PACK

SUPPORTED_CAPABILITIES: Final = PLUGIN_COMPONENT_KINDS

CATEGORY_AVAILABLE: Final = "available"
CATEGORY_DISABLED: Final = "disabled"
CATEGORY_QUARANTINED: Final = "quarantined"

REASON_AVAILABLE: Final = "available"
REASON_DISABLED: Final = "disabled"
REASON_DUPLICATE_NAME: Final = "duplicate_name"
REASON_INVALID_API_VERSION: Final = "invalid_api_version"
REASON_INVALID_CAPABILITIES: Final = "invalid_capabilities"
REASON_INVALID_METADATA: Final = "invalid_metadata"
REASON_MISSING_CAPABILITIES: Final = "missing_capabilities"
REASON_UNSUPPORTED_API_VERSION: Final = "unsupported_api_version"
REASON_UNSUPPORTED_CAPABILITY: Final = "unsupported_capability"

# A plugin name is normally a public distribution identifier.  Names outside
# this compact identifier shape are still accepted, but are represented by a
# digest in reports so a malformed value cannot become a log or report leak.
_SAFE_NAME_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}"
    r"(?::[A-Za-z0-9][A-Za-z0-9._-]{0,63})?$"
)
_SAFE_CAPABILITY_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_API_VERSION_RE = re.compile(
    r"^(?P<major>0|[1-9]\d*)"
    r"(?:\.(?P<minor>0|[1-9]\d*))?"
    r"(?:\.(?P<patch>0|[1-9]\d*))?"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
_MAX_METADATA_RECORDS = 10_000
_MAX_CAPABILITY_DECLARATIONS = 64

_MISSING = object()
_READ_ERROR = object()
_ITERATION_ERROR = object()
_RECORD_LIMIT_EXCEEDED = object()


@dataclass(frozen=True)
class PluginStatus:
    """Safe status for one injected plugin metadata record.

    Only normalized, allow-listed values are retained.  In particular, the
    original metadata mapping and any free-form values are never serialized.
    ``metadata_hash`` is a stable digest used to make duplicate resolution
    independent of input order.
    """

    name: str
    category: str
    reason: str
    message: str
    api_version: str | None = None
    capabilities: tuple[str, ...] = ()
    metadata_hash: str = ""

    @property
    def status(self) -> str:
        """Return the availability category."""

        return self.category

    @property
    def is_available(self) -> bool:
        """Return whether this plugin passed the static preflight."""

        return self.category == CATEGORY_AVAILABLE

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, sensitive-value-free mapping."""

        return {
            "name": self.name,
            "category": self.category,
            "reason": self.reason,
            "message": self.message,
            "api_version": self.api_version,
            "capabilities": list(self.capabilities),
            "metadata_hash": self.metadata_hash,
        }

    def as_dict(self) -> dict[str, Any]:
        """Alias for :meth:`to_dict` used by report consumers."""

        return self.to_dict()


@dataclass(frozen=True)
class PluginQuarantineReport:
    """Deterministic snapshot of plugin availability and quarantine state."""

    records: tuple[PluginStatus, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))

    @property
    def statuses(self) -> tuple[PluginStatus, ...]:
        """Return all statuses in deterministic report order."""

        return self.records

    @property
    def available(self) -> tuple[PluginStatus, ...]:
        """Return plugins accepted by the static preflight."""

        return tuple(
            record for record in self.records if record.category == CATEGORY_AVAILABLE
        )

    @property
    def disabled(self) -> tuple[PluginStatus, ...]:
        """Return plugins explicitly disabled by their metadata."""

        return tuple(
            record for record in self.records if record.category == CATEGORY_DISABLED
        )

    @property
    def quarantined(self) -> tuple[PluginStatus, ...]:
        """Return malformed, incompatible, or duplicate plugin records."""

        return tuple(
            record for record in self.records if record.category == CATEGORY_QUARANTINED
        )

    @property
    def counts(self) -> dict[str, int]:
        """Return category counts in stable key order."""

        return {
            CATEGORY_AVAILABLE: len(self.available),
            CATEGORY_DISABLED: len(self.disabled),
            CATEGORY_QUARANTINED: len(self.quarantined),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible, sensitive-value-free report."""

        return {
            CATEGORY_AVAILABLE: [record.to_dict() for record in self.available],
            CATEGORY_DISABLED: [record.to_dict() for record in self.disabled],
            CATEGORY_QUARANTINED: [record.to_dict() for record in self.quarantined],
        }

    def as_dict(self) -> dict[str, Any]:
        """Alias for :meth:`to_dict`."""

        return self.to_dict()

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the report with stable ordering and formatting."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )


@dataclass(frozen=True)
class _EvaluatedMetadata:
    status: PluginStatus
    name_key: str
    eligible_for_duplicate_check: bool


def build_quarantine_report(
    metadata: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    *,
    supported_api_major: int = SUPPORTED_PLUGIN_API_MAJOR,
    supported_capabilities: Iterable[str] | None = None,
) -> PluginQuarantineReport:
    """Evaluate injected plugin metadata without loading plugin code.

    Args:
        metadata: One metadata mapping or an iterable of mappings. Supported
            fields are ``name`` or the stable ``plugin_id``/``component_id``
            pair, ``api_version`` (with ``sdk_version`` as a compatibility
            alias), ``capabilities`` (with ``kind`` as a compatibility alias),
            and the boolean ``disabled``/``enabled`` state fields.
        supported_api_major: API major accepted by this process.
        supported_capabilities: Optional allow-list.  The OpenMed plugin
            capability kinds are used by default.

    Returns:
        A deterministic report with separate ``available``, ``disabled``, and
        ``quarantined`` categories.  Malformed input is represented as a safe
        status rather than exposing an input value in an exception.

    Raises:
        ValueError: If the evaluator's own support configuration is invalid.

    The function only inspects the supplied static fields.  It never calls
    ``importlib.metadata``, resolves an entry point, imports a plugin, reads a
    credential field, or performs network I/O.
    """

    if (
        isinstance(supported_api_major, bool)
        or not isinstance(supported_api_major, int)
        or supported_api_major < 0
    ):
        raise ValueError("supported_api_major must be a non-negative integer")

    supported = _normalize_supported_capabilities(supported_capabilities)
    items = _metadata_items(metadata)
    evaluated: list[_EvaluatedMetadata] = []
    for item in items:
        try:
            evaluated.append(
                _evaluate_metadata(
                    item,
                    supported_api_major=supported_api_major,
                    supported_capabilities=supported,
                )
            )
        except Exception:  # pragma: no cover - hostile custom Mapping guard
            evaluated.append(
                _fallback_invalid_status("metadata could not be read safely")
            )

    groups: dict[str, list[int]] = {}
    for index, candidate in enumerate(evaluated):
        if candidate.eligible_for_duplicate_check:
            groups.setdefault(candidate.name_key, []).append(index)

    for indices in groups.values():
        if len(indices) < 2:
            continue
        for duplicate_index in indices:
            candidate = evaluated[duplicate_index]
            evaluated[duplicate_index] = _with_reason(
                candidate,
                category=CATEGORY_QUARANTINED,
                reason=REASON_DUPLICATE_NAME,
                message="plugin name is declared by multiple available plugins",
            )

    statuses = tuple(
        candidate.status
        for candidate in sorted(
            evaluated,
            key=lambda candidate: _status_sort_key(candidate.status),
        )
    )
    return PluginQuarantineReport(records=statuses)


def evaluate_plugin_metadata(
    metadata: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    *,
    supported_api_major: int = SUPPORTED_PLUGIN_API_MAJOR,
    supported_capabilities: Iterable[str] | None = None,
) -> PluginQuarantineReport:
    """Alias for :func:`build_quarantine_report`."""

    return build_quarantine_report(
        metadata,
        supported_api_major=supported_api_major,
        supported_capabilities=supported_capabilities,
    )


def quarantine_report(
    metadata: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    *,
    supported_api_major: int = SUPPORTED_PLUGIN_API_MAJOR,
    supported_capabilities: Iterable[str] | None = None,
) -> PluginQuarantineReport:
    """Return a static plugin quarantine report.

    This short alias is convenient for callers that already have injected
    metadata and do not need to name the builder explicitly.
    """

    return build_quarantine_report(
        metadata,
        supported_api_major=supported_api_major,
        supported_capabilities=supported_capabilities,
    )


def _normalize_supported_capabilities(
    capabilities: Iterable[str] | None,
) -> frozenset[str]:
    if capabilities is None:
        return SUPPORTED_CAPABILITIES
    if isinstance(capabilities, (str, bytes, Mapping)):
        raise ValueError("supported_capabilities must be an iterable of strings")
    values = _bounded_values(capabilities, _MAX_CAPABILITY_DECLARATIONS)
    if values is None:
        raise ValueError(
            "supported_capabilities must be a bounded iterable of strings"
        ) from None
    normalized: set[str] = set()
    for value in values:
        try:
            token = _normalize_capability(value)
        except Exception:
            raise ValueError(
                "supported_capabilities must contain valid strings"
            ) from None
        if token is None:
            raise ValueError("supported_capabilities must contain valid strings")
        normalized.add(token)
    return frozenset(normalized)


def _metadata_items(
    metadata: Mapping[str, Any] | Iterable[Mapping[str, Any]],
) -> tuple[Any, ...]:
    if isinstance(metadata, Mapping):
        return (metadata,)
    if metadata is None or isinstance(metadata, (str, bytes)):
        return (metadata,)
    try:
        iterator = iter(metadata)
    except Exception:
        return (_ITERATION_ERROR,)
    items: list[Any] = []
    while len(items) <= _MAX_METADATA_RECORDS:
        try:
            item = next(iterator)
        except StopIteration:
            return tuple(items)
        except Exception:
            return (_ITERATION_ERROR,)
        if len(items) == _MAX_METADATA_RECORDS:
            return (_RECORD_LIMIT_EXCEEDED,)
        items.append(item)
    return (_RECORD_LIMIT_EXCEEDED,)


def _evaluate_metadata(
    item: Any,
    *,
    supported_api_major: int,
    supported_capabilities: frozenset[str],
) -> _EvaluatedMetadata:
    if item is _ITERATION_ERROR:
        return _fallback_invalid_status("metadata records could not be read safely")
    if item is _RECORD_LIMIT_EXCEEDED:
        return _fallback_invalid_status("metadata record limit exceeded")
    if not isinstance(item, Mapping):
        return _fallback_invalid_status("metadata record is not a mapping")

    name_value = _metadata_name(item)
    if name_value is _READ_ERROR:
        return _fallback_invalid_status("metadata could not be read safely")
    name, name_key, valid_name = _name_details(name_value)
    if not valid_name:
        return _make_evaluated(
            name=name,
            name_key=name_key,
            category=CATEGORY_QUARANTINED,
            reason=REASON_INVALID_METADATA,
            message="plugin metadata requires a non-empty string name",
        )

    disabled, state_error = _disabled_state(item)
    if state_error:
        return _make_evaluated(
            name=name,
            name_key=name_key,
            category=CATEGORY_QUARANTINED,
            reason=REASON_INVALID_METADATA,
            message="plugin enabled state must be a boolean",
        )
    if disabled:
        return _make_evaluated(
            name=name,
            name_key=name_key,
            category=CATEGORY_DISABLED,
            reason=REASON_DISABLED,
            message="plugin is disabled",
        )

    api_value = _first_value(item, "api_version", "plugin_api_version", "sdk_version")
    if api_value is _MISSING or api_value is _READ_ERROR:
        return _make_evaluated(
            name=name,
            name_key=name_key,
            category=CATEGORY_QUARANTINED,
            reason=REASON_INVALID_API_VERSION,
            message="plugin API version is missing or malformed",
        )
    api_version, api_major, api_fingerprint = _normalize_api_version(api_value)
    if api_version is None or api_major is None or api_fingerprint is None:
        return _make_evaluated(
            name=name,
            name_key=name_key,
            category=CATEGORY_QUARANTINED,
            reason=REASON_INVALID_API_VERSION,
            message="plugin API version is missing or malformed",
        )
    if api_major != supported_api_major:
        return _make_evaluated(
            name=name,
            name_key=name_key,
            category=CATEGORY_QUARANTINED,
            reason=REASON_UNSUPPORTED_API_VERSION,
            message="plugin API major version is not supported",
            api_version=api_version,
            fingerprint_values=(api_fingerprint,),
        )

    capability_reason, capabilities, capability_fingerprint = _capability_details(
        item,
        supported_capabilities,
    )
    if capability_reason is not None:
        return _make_evaluated(
            name=name,
            name_key=name_key,
            category=CATEGORY_QUARANTINED,
            reason=capability_reason,
            message=_CAPABILITY_MESSAGES[capability_reason],
            api_version=api_version,
            capabilities=capabilities,
            fingerprint_values=(api_fingerprint, *capability_fingerprint),
        )

    return _make_evaluated(
        name=name,
        name_key=name_key,
        category=CATEGORY_AVAILABLE,
        reason=REASON_AVAILABLE,
        message="plugin metadata accepted",
        api_version=api_version,
        capabilities=capabilities,
        fingerprint_values=(api_fingerprint, *capability_fingerprint),
        eligible_for_duplicate_check=True,
    )


_CAPABILITY_MESSAGES: Final = {
    REASON_INVALID_CAPABILITIES: "plugin capabilities are malformed",
    REASON_MISSING_CAPABILITIES: "plugin declares no capabilities",
    REASON_UNSUPPORTED_CAPABILITY: "plugin declares an unsupported capability",
}


def _disabled_state(item: Mapping[str, Any]) -> tuple[bool, bool]:
    disabled = _first_value(item, "disabled")
    enabled = _first_value(item, "enabled")
    state = _first_value(item, "state")
    if any(value is _READ_ERROR for value in (disabled, enabled, state)):
        return False, True

    disabled_value: bool | None = None
    enabled_value: bool | None = None
    if disabled is not _MISSING:
        if not isinstance(disabled, bool):
            return False, True
        disabled_value = disabled
    if enabled is not _MISSING:
        if not isinstance(enabled, bool):
            return False, True
        enabled_value = enabled
    if disabled_value is not None and enabled_value is not None:
        if disabled_value == enabled_value:
            return False, True

    state_disabled: bool | None = None
    if state is not _MISSING:
        if not isinstance(state, str):
            return False, True
        normalized_state = state.strip().casefold()
        if normalized_state not in {"enabled", "disabled"}:
            return False, True
        state_disabled = normalized_state == "disabled"

    derived_disabled = disabled_value
    if derived_disabled is None and enabled_value is not None:
        derived_disabled = not enabled_value
    if (
        state_disabled is not None
        and derived_disabled is not None
        and state_disabled != derived_disabled
    ):
        return False, True
    return bool(
        state_disabled if state_disabled is not None else derived_disabled
    ), False


def _capability_details(
    item: Mapping[str, Any],
    supported_capabilities: frozenset[str],
) -> tuple[str | None, tuple[str, ...], tuple[str, ...]]:
    raw_capabilities = _first_value(item, "capabilities", "capability", "kinds")
    if raw_capabilities is _READ_ERROR:
        return REASON_INVALID_CAPABILITIES, (), ("read_error",)
    if raw_capabilities is _MISSING:
        kind = _first_value(item, "kind")
        if kind is _READ_ERROR:
            return REASON_INVALID_CAPABILITIES, (), ("read_error",)
        if kind is _MISSING:
            return REASON_MISSING_CAPABILITIES, (), ()
        raw_capabilities = (kind,)

    if isinstance(raw_capabilities, (str, bytes, Mapping)):
        return REASON_INVALID_CAPABILITIES, (), ("invalid_shape",)
    values = _bounded_values(raw_capabilities, _MAX_CAPABILITY_DECLARATIONS)
    if values is None:
        return REASON_INVALID_CAPABILITIES, (), ("unreadable",)
    if not values:
        return REASON_MISSING_CAPABILITIES, (), ()

    normalized: list[str] = []
    for value in values:
        token = _normalize_capability(value)
        if token is None:
            return REASON_INVALID_CAPABILITIES, (), ("invalid_value",)
        normalized.append(token)

    unique = tuple(sorted(set(normalized)))
    known = tuple(token for token in unique if token in supported_capabilities)
    if len(known) != len(unique):
        return REASON_UNSUPPORTED_CAPABILITY, known, unique
    return None, known, unique


def _normalize_capability(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    token = value.strip().casefold().replace("-", "_").replace(" ", "_")
    if _SAFE_CAPABILITY_RE.fullmatch(token) is None:
        return None
    return token


def _bounded_values(value: Iterable[Any], maximum: int) -> tuple[Any, ...] | None:
    """Materialize at most ``maximum`` values, returning ``None`` on failure."""

    try:
        iterator = iter(value)
    except Exception:
        return None
    values: list[Any] = []
    while len(values) <= maximum:
        try:
            item = next(iterator)
        except StopIteration:
            return tuple(values)
        except Exception:
            return None
        if len(values) == maximum:
            return None
        values.append(item)
    return None


def _normalize_api_version(
    value: Any,
) -> tuple[str | None, int | None, str | None]:
    if isinstance(value, bool):
        return None, None, None
    if isinstance(value, int):
        if value < 0:
            return None, None, None
        normalized = f"{value}.0.0"
        return normalized, value, _sha256_text(normalized)
    if not isinstance(value, str):
        return None, None, None
    text = value.strip()
    match = _API_VERSION_RE.fullmatch(text)
    if match is None:
        return None, None, None
    major = int(match.group("major"))
    minor = int(match.group("minor") or 0)
    patch = int(match.group("patch") or 0)
    return f"{major}.{minor}.{patch}", major, _sha256_text(text)


def _metadata_name(item: Mapping[str, Any]) -> Any:
    """Prefer stable plugin/component identifiers over a display name."""

    plugin_id = _first_value(item, "plugin_id", "plugin")
    component_id = _first_value(item, "component_id")
    if plugin_id is _READ_ERROR or component_id is _READ_ERROR:
        return _READ_ERROR
    if plugin_id is not _MISSING and component_id is not _MISSING:
        if not isinstance(plugin_id, str) or not isinstance(component_id, str):
            return _READ_ERROR
        plugin_text = plugin_id.strip()
        component_text = component_id.strip()
        if not plugin_text or not component_text:
            return _READ_ERROR
        return f"{plugin_text}:{component_text}"
    if plugin_id is not _MISSING:
        return plugin_id
    return _first_value(item, "name", "plugin_name", "id", "component_id")


def _name_details(value: Any) -> tuple[str, str, bool]:
    if value is _MISSING or not isinstance(value, str):
        return "unknown", "", False
    raw = value.strip()
    if not raw:
        return "unknown", "", False
    key = raw.casefold()
    if _SAFE_NAME_RE.fullmatch(raw):
        return raw, key, True
    return f"redacted:{_sha256_text(raw)[:16]}", key, True


def _first_value(item: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        try:
            if key not in item:
                continue
            return item[key]
        except Exception:
            return _READ_ERROR
    return _MISSING


def _make_evaluated(
    *,
    name: str,
    name_key: str,
    category: str,
    reason: str,
    message: str,
    api_version: str | None = None,
    capabilities: tuple[str, ...] = (),
    fingerprint_values: tuple[str, ...] = (),
    eligible_for_duplicate_check: bool = False,
) -> _EvaluatedMetadata:
    metadata_hash = _metadata_hash(
        name_key=name_key,
        category=category,
        reason=reason,
        api_version=api_version,
        capabilities=capabilities,
        fingerprint_values=fingerprint_values,
    )
    return _EvaluatedMetadata(
        status=PluginStatus(
            name=name,
            category=category,
            reason=reason,
            message=message,
            api_version=api_version,
            capabilities=tuple(capabilities),
            metadata_hash=metadata_hash,
        ),
        name_key=name_key,
        eligible_for_duplicate_check=eligible_for_duplicate_check,
    )


def _fallback_invalid_status(message: str) -> _EvaluatedMetadata:
    return _make_evaluated(
        name="unknown",
        name_key="",
        category=CATEGORY_QUARANTINED,
        reason=REASON_INVALID_METADATA,
        message=message,
    )


def _with_reason(
    candidate: _EvaluatedMetadata,
    *,
    category: str,
    reason: str,
    message: str,
) -> _EvaluatedMetadata:
    status = candidate.status
    return _EvaluatedMetadata(
        status=PluginStatus(
            name=status.name,
            category=category,
            reason=reason,
            message=message,
            api_version=status.api_version,
            capabilities=status.capabilities,
            metadata_hash=_metadata_hash(
                name_key=candidate.name_key,
                category=category,
                reason=reason,
                api_version=status.api_version,
                capabilities=status.capabilities,
                fingerprint_values=(status.metadata_hash,),
            ),
        ),
        name_key=candidate.name_key,
        eligible_for_duplicate_check=False,
    )


def _metadata_hash(
    *,
    name_key: str,
    category: str,
    reason: str,
    api_version: str | None,
    capabilities: tuple[str, ...],
    fingerprint_values: tuple[str, ...],
) -> str:
    payload = {
        "name": name_key,
        "category": category,
        "reason": reason,
        "api_version": api_version,
        "capabilities": list(capabilities),
        "fingerprint": list(fingerprint_values),
    }
    serialized = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    return f"sha256:{_sha256_text(serialized)}"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _status_sort_key(status: PluginStatus) -> tuple[Any, ...]:
    category_order = {
        CATEGORY_AVAILABLE: 0,
        CATEGORY_DISABLED: 1,
        CATEGORY_QUARANTINED: 2,
    }
    return (
        status.name.casefold(),
        category_order.get(status.category, 3),
        status.reason,
        status.metadata_hash,
        status.name,
    )


__all__ = [
    "CAPABILITY_ANONYMIZER_PROVIDER",
    "CAPABILITY_EXPORTER",
    "CAPABILITY_INTEROP_ADAPTER",
    "CAPABILITY_LANGUAGE_PACK",
    "CAPABILITY_RECOGNIZER",
    "CATEGORY_AVAILABLE",
    "CATEGORY_DISABLED",
    "CATEGORY_QUARANTINED",
    "PLUGIN_API_MAJOR",
    "PLUGIN_API_VERSION",
    "REASON_AVAILABLE",
    "REASON_DISABLED",
    "REASON_DUPLICATE_NAME",
    "REASON_INVALID_API_VERSION",
    "REASON_INVALID_CAPABILITIES",
    "REASON_INVALID_METADATA",
    "REASON_MISSING_CAPABILITIES",
    "REASON_UNSUPPORTED_API_VERSION",
    "REASON_UNSUPPORTED_CAPABILITY",
    "SUPPORTED_CAPABILITIES",
    "SUPPORTED_PLUGIN_API_MAJOR",
    "PluginQuarantineReport",
    "PluginStatus",
    "build_quarantine_report",
    "evaluate_plugin_metadata",
    "quarantine_report",
]
