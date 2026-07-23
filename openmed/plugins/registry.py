"""Lazy entry-point registry for OpenMed plugin components."""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from openmed.core.labels import CANONICAL_LABELS

from .protocols import (
    COMPONENT_RECOGNIZER,
    PLUGIN_COMPONENT_KINDS,
    PLUGIN_SDK_MAJOR,
    PluginComponentMetadata,
)

PLUGIN_ENTRY_POINT_GROUP = "openmed.plugins"

REASON_DUPLICATE_COMPONENT = "duplicate_component"
REASON_ENTRY_POINT_ENUMERATION_FAILED = "entry_point_enumeration_failed"
REASON_INVALID_LABEL = "invalid_label"
REASON_INVALID_METADATA = "invalid_metadata"
REASON_LOAD_ERROR = "load_error"
REASON_MISSING_LABELS = "missing_labels"
REASON_NETWORK_EGRESS_OPT_IN_REQUIRED = "network_egress_opt_in_required"
REASON_NON_PERMISSIVE_LICENSE_OPT_IN_REQUIRED = "non_permissive_license_opt_in_required"
REASON_PROTOCOL_VERSION_MISMATCH = "protocol_version_mismatch"
REASON_UNKNOWN_COMPONENT_KIND = "unknown_component_kind"

PERMISSIVE_LICENSES = frozenset(
    {
        "0BSD",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "CC0-1.0",
        "ISC",
        "MIT",
        "Unlicense",
        "Zlib",
    }
)

_LICENSE_TOKEN_RE = re.compile(r"[A-Za-z0-9.+-]+")


@dataclass(frozen=True)
class PluginDiscoveryPolicy:
    """Policy gates applied while discovering plugin entry points.

    Args:
        allow_network_egress: Allow plugins that declare network egress.
        allow_non_permissive_licenses: Allow plugins with restricted licenses.
        opt_in_plugins: Plugin ids or qualified component ids explicitly
            allowed through both policy gates.
    """

    allow_network_egress: bool = False
    allow_non_permissive_licenses: bool = False
    opt_in_plugins: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "opt_in_plugins",
            tuple(str(item).strip() for item in self.opt_in_plugins if str(item)),
        )


@dataclass(frozen=True)
class PluginRegistration:
    """A plugin component accepted by the registry."""

    entry_point_name: str
    metadata: PluginComponentMetadata
    component: Any
    loaded_by_policy_opt_in: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible registration metadata."""

        return {
            "entry_point_name": self.entry_point_name,
            "metadata": self.metadata.to_dict(),
            "loaded_by_policy_opt_in": self.loaded_by_policy_opt_in,
        }


@dataclass(frozen=True)
class PluginQuarantineRecord:
    """Structured reason a plugin component was skipped."""

    entry_point_name: str
    reason: str
    message: str
    plugin_id: str = ""
    component_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def qualified_id(self) -> str:
        """Return ``plugin_id:component_id`` when both are known."""

        if not self.plugin_id or not self.component_id:
            return ""
        return f"{self.plugin_id}:{self.component_id}"

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible quarantine details."""

        return {
            "entry_point_name": self.entry_point_name,
            "reason": self.reason,
            "message": self.message,
            "plugin_id": self.plugin_id,
            "component_id": self.component_id,
            "qualified_id": self.qualified_id,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PluginDiscoveryResult:
    """Snapshot returned after one discovery pass."""

    registrations: tuple[PluginRegistration, ...]
    quarantined: tuple[PluginQuarantineRecord, ...]

    def registrations_for_kind(self, kind: str) -> tuple[PluginRegistration, ...]:
        """Return registrations whose metadata kind matches *kind*."""

        normalized = str(kind or "").strip().lower()
        return tuple(
            registration
            for registration in self.registrations
            if registration.metadata.kind == normalized
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible discovery results."""

        return {
            "registrations": [
                registration.to_dict() for registration in self.registrations
            ],
            "quarantined": [record.to_dict() for record in self.quarantined],
        }


class PluginRegistry:
    """Lazy registry for third-party OpenMed plugin entry points."""

    def __init__(
        self,
        *,
        policy: PluginDiscoveryPolicy | None = None,
    ) -> None:
        self.policy = policy or PluginDiscoveryPolicy()
        self._lock = RLock()
        self._discovered = False
        self._registrations: dict[str, PluginRegistration] = {}
        self._quarantined: list[PluginQuarantineRecord] = []

    def discover(self, *, force: bool = False) -> PluginDiscoveryResult:
        """Discover entry points once and return the registry snapshot."""

        with self._lock:
            if self._discovered and not force:
                return self.snapshot()
            self._discovered = True
            self._registrations.clear()
            self._quarantined.clear()

        try:
            entry_points = _entry_points_for_group(PLUGIN_ENTRY_POINT_GROUP)
        except Exception as exc:  # pragma: no cover - importlib defensive guard
            self._quarantine(
                entry_point_name=PLUGIN_ENTRY_POINT_GROUP,
                reason=REASON_ENTRY_POINT_ENUMERATION_FAILED,
                message=f"failed to enumerate entry points: {exc.__class__.__name__}",
            )
            return self.snapshot()

        for entry_point in entry_points:
            self._load_entry_point(entry_point)

        return self.snapshot()

    def registrations(
        self,
        *,
        kind: str | None = None,
    ) -> tuple[PluginRegistration, ...]:
        """Return accepted registrations, optionally filtered by kind."""

        result = self.discover()
        if kind is None:
            return result.registrations
        return result.registrations_for_kind(kind)

    def quarantined(self) -> tuple[PluginQuarantineRecord, ...]:
        """Return structured records for skipped plugin components."""

        return self.discover().quarantined

    def snapshot(self) -> PluginDiscoveryResult:
        """Return the current registry snapshot without triggering discovery."""

        with self._lock:
            return PluginDiscoveryResult(
                registrations=tuple(
                    sorted(
                        self._registrations.values(),
                        key=lambda item: item.metadata.qualified_id,
                    )
                ),
                quarantined=tuple(self._quarantined),
            )

    def _load_entry_point(self, entry_point: Any) -> None:
        entry_name = str(getattr(entry_point, "name", "<unknown>"))
        try:
            loaded = entry_point.load()
        except Exception as exc:
            self._quarantine(
                entry_point_name=entry_name,
                reason=REASON_LOAD_ERROR,
                message=f"failed to load entry point: {exc.__class__.__name__}",
            )
            return

        try:
            components = _coerce_components(loaded)
        except Exception as exc:
            self._quarantine(
                entry_point_name=entry_name,
                reason=REASON_INVALID_METADATA,
                message=str(exc),
            )
            return

        for component in components:
            self._register_component(entry_name, component)

    def _register_component(self, entry_name: str, component: Any) -> None:
        try:
            metadata = _metadata_for_component(component)
        except Exception as exc:
            self._quarantine(
                entry_point_name=entry_name,
                reason=REASON_INVALID_METADATA,
                message=str(exc),
            )
            return

        rejection = _metadata_rejection(metadata, self.policy)
        if rejection is not None:
            reason, message = rejection
            self._quarantine_metadata(entry_name, metadata, reason, message)
            return

        key = metadata.qualified_id
        with self._lock:
            if key in self._registrations:
                self._quarantined.append(
                    _record_for_metadata(
                        entry_name,
                        metadata,
                        REASON_DUPLICATE_COMPONENT,
                        f"duplicate plugin component {key!r}",
                    )
                )
                return
            self._registrations[key] = PluginRegistration(
                entry_point_name=entry_name,
                metadata=metadata,
                component=component,
                loaded_by_policy_opt_in=_has_policy_opt_in(metadata, self.policy),
            )

    def _quarantine_metadata(
        self,
        entry_point_name: str,
        metadata: PluginComponentMetadata,
        reason: str,
        message: str,
    ) -> None:
        with self._lock:
            self._quarantined.append(
                _record_for_metadata(entry_point_name, metadata, reason, message)
            )

    def _quarantine(
        self,
        *,
        entry_point_name: str,
        reason: str,
        message: str,
    ) -> None:
        with self._lock:
            self._quarantined.append(
                PluginQuarantineRecord(
                    entry_point_name=entry_point_name,
                    reason=reason,
                    message=message,
                )
            )


_DEFAULT_REGISTRY = PluginRegistry()


def discover_plugins(
    *,
    allow_network_egress: bool = False,
    allow_non_permissive_licenses: bool = False,
    opt_in_plugins: Sequence[str] = (),
    force: bool = False,
) -> PluginDiscoveryResult:
    """Discover third-party plugin components.

    Default discovery is cached and excludes plugins that declare network
    egress or non-permissive licenses. Passing any opt-in option uses a fresh
    registry so callers must make that policy choice explicitly.
    """

    policy = PluginDiscoveryPolicy(
        allow_network_egress=allow_network_egress,
        allow_non_permissive_licenses=allow_non_permissive_licenses,
        opt_in_plugins=opt_in_plugins,
    )
    if policy == PluginDiscoveryPolicy():
        return _DEFAULT_REGISTRY.discover(force=force)
    return PluginRegistry(policy=policy).discover(force=True)


def iter_plugins(
    kind: str | None = None,
    *,
    allow_network_egress: bool = False,
    allow_non_permissive_licenses: bool = False,
    opt_in_plugins: Sequence[str] = (),
) -> tuple[PluginRegistration, ...]:
    """Return accepted plugin registrations, optionally filtered by kind."""

    result = discover_plugins(
        allow_network_egress=allow_network_egress,
        allow_non_permissive_licenses=allow_non_permissive_licenses,
        opt_in_plugins=opt_in_plugins,
    )
    if kind is None:
        return result.registrations
    return result.registrations_for_kind(kind)


def quarantined_plugins() -> tuple[PluginQuarantineRecord, ...]:
    """Return quarantine records from default plugin discovery."""

    return _DEFAULT_REGISTRY.quarantined()


def is_permissive_license(license_expression: str) -> bool:
    """Return whether *license_expression* is allowed for auto-load."""

    tokens = tuple(_LICENSE_TOKEN_RE.findall(str(license_expression or "")))
    if not tokens:
        return False
    license_tokens = [
        token for token in tokens if token.upper() not in {"AND", "OR", "WITH"}
    ]
    return bool(license_tokens) and all(
        token in PERMISSIVE_LICENSES for token in license_tokens
    )


def _entry_points_for_group(group: str) -> Sequence[Any]:
    try:
        return tuple(importlib_metadata.entry_points(group=group))
    except TypeError:
        entry_points = importlib_metadata.entry_points()
        if hasattr(entry_points, "select"):
            return tuple(entry_points.select(group=group))
        return tuple(entry_points.get(group, ()))


def _coerce_components(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if _looks_like_component(value):
        return (value,)
    if callable(value):
        return _coerce_components(value())
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
        components: list[Any] = []
        for item in value:
            components.extend(_coerce_components(item))
        return tuple(components)
    raise TypeError("openmed.plugins entry points must return components with metadata")


def _looks_like_component(value: Any) -> bool:
    return hasattr(value, "metadata")


def _metadata_for_component(component: Any) -> PluginComponentMetadata:
    raw_metadata = getattr(component, "metadata", None)
    if callable(raw_metadata):
        raw_metadata = raw_metadata()
    if isinstance(raw_metadata, PluginComponentMetadata):
        return raw_metadata
    if isinstance(raw_metadata, Mapping):
        return PluginComponentMetadata.from_mapping(raw_metadata)
    raise TypeError("plugin component must expose PluginComponentMetadata metadata")


def _metadata_rejection(
    metadata: PluginComponentMetadata,
    policy: PluginDiscoveryPolicy,
) -> tuple[str, str] | None:
    if not metadata.plugin_id:
        return REASON_INVALID_METADATA, "plugin_id must be non-empty"
    if not metadata.component_id:
        return REASON_INVALID_METADATA, "component_id must be non-empty"
    if metadata.kind not in PLUGIN_COMPONENT_KINDS:
        return (
            REASON_UNKNOWN_COMPONENT_KIND,
            f"component kind must be one of {', '.join(sorted(PLUGIN_COMPONENT_KINDS))}",
        )

    major = _sdk_major(metadata.sdk_version)
    if major != PLUGIN_SDK_MAJOR:
        return (
            REASON_PROTOCOL_VERSION_MISMATCH,
            f"plugin targets SDK {metadata.sdk_version!r}; "
            f"OpenMed supports major {PLUGIN_SDK_MAJOR}",
        )

    label_error = _label_rejection(metadata)
    if label_error is not None:
        return label_error

    if metadata.network_egress and not (
        policy.allow_network_egress or _has_policy_opt_in(metadata, policy)
    ):
        return (
            REASON_NETWORK_EGRESS_OPT_IN_REQUIRED,
            "plugin declares network egress and requires explicit opt-in",
        )

    if not is_permissive_license(metadata.license) and not (
        policy.allow_non_permissive_licenses or _has_policy_opt_in(metadata, policy)
    ):
        return (
            REASON_NON_PERMISSIVE_LICENSE_OPT_IN_REQUIRED,
            "plugin license is not auto-loadable and requires explicit opt-in",
        )

    return None


def _label_rejection(
    metadata: PluginComponentMetadata,
) -> tuple[str, str] | None:
    if metadata.kind == COMPONENT_RECOGNIZER and not metadata.labels:
        return (
            REASON_MISSING_LABELS,
            "recognizer plugins must declare at least one canonical label",
        )
    for label in metadata.labels:
        if label not in CANONICAL_LABELS:
            return (
                REASON_INVALID_LABEL,
                f"label {label!r} is not in the canonical OpenMed label schema",
            )
    return None


def _sdk_major(version: str) -> int:
    major_text = str(version or "").split(".", 1)[0]
    if not major_text.isdigit():
        return -1
    return int(major_text)


def _has_policy_opt_in(
    metadata: PluginComponentMetadata,
    policy: PluginDiscoveryPolicy,
) -> bool:
    opt_ins = set(policy.opt_in_plugins)
    return metadata.plugin_id in opt_ins or metadata.qualified_id in opt_ins


def _record_for_metadata(
    entry_point_name: str,
    metadata: PluginComponentMetadata,
    reason: str,
    message: str,
) -> PluginQuarantineRecord:
    return PluginQuarantineRecord(
        entry_point_name=entry_point_name,
        plugin_id=metadata.plugin_id,
        component_id=metadata.component_id,
        reason=reason,
        message=message,
        metadata=metadata.to_dict(),
    )


def _reset_plugin_registry_for_tests() -> None:
    global _DEFAULT_REGISTRY

    _DEFAULT_REGISTRY = PluginRegistry()


__all__ = [
    "PERMISSIVE_LICENSES",
    "PLUGIN_ENTRY_POINT_GROUP",
    "PluginDiscoveryPolicy",
    "PluginDiscoveryResult",
    "PluginQuarantineRecord",
    "PluginRegistration",
    "PluginRegistry",
    "discover_plugins",
    "is_permissive_license",
    "iter_plugins",
    "quarantined_plugins",
]
