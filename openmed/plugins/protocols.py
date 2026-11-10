"""Versioned contracts for third-party OpenMed plugin components.

The protocols in this module describe the stable boundary exposed to plugin
packages. They intentionally depend only on the canonical :class:`OpenMedSpan`
schema and standard-library typing primitives; importing them does not discover
plugins or import any optional plugin dependency.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

from openmed.core.schemas.span import OpenMedSpan

PLUGIN_SDK_VERSION = "1.0.0"
PLUGIN_SDK_MAJOR = 1

COMPONENT_RECOGNIZER = "recognizer"
COMPONENT_ANONYMIZER_PROVIDER = "anonymizer_provider"
COMPONENT_EXPORTER = "exporter"
COMPONENT_INTEROP_ADAPTER = "interop_adapter"
COMPONENT_LANGUAGE_PACK = "language_pack"

PLUGIN_COMPONENT_KINDS = frozenset(
    {
        COMPONENT_RECOGNIZER,
        COMPONENT_ANONYMIZER_PROVIDER,
        COMPONENT_EXPORTER,
        COMPONENT_INTEROP_ADAPTER,
        COMPONENT_LANGUAGE_PACK,
    }
)


def _string_sequence(value: object, field_name: str) -> tuple[str, ...]:
    """Return a detached tuple of strings for one metadata field."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence of strings")
    values = tuple(value)
    if any(not isinstance(item, str) for item in values):
        raise TypeError(f"{field_name} must contain only strings")
    return tuple(item.strip() for item in values if item.strip())


def _metadata_text(value: object, field_name: str) -> str:
    """Return normalized text for one strictly typed metadata field."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    return value.strip()


@dataclass(frozen=True)
class PluginComponentMetadata:
    """Metadata every OpenMed plugin component declares for validation.

    Args:
        plugin_id: Stable package or distribution identifier.
        component_id: Stable identifier unique within the plugin package.
        kind: One of :data:`PLUGIN_COMPONENT_KINDS`.
        sdk_version: Plugin SDK semantic version targeted by the component.
        license: SPDX license expression for the package or component.
        network_egress: Whether the component may make network calls.
        labels: Canonical OpenMed labels emitted or handled by the component.
        languages: Language codes covered by the component, or ``"*"``.
        name: Human-readable display name.
        description: Human-readable component summary.
        metadata: Extra static, non-PHI machine-readable metadata.
    """

    plugin_id: str
    component_id: str
    kind: str
    sdk_version: str = PLUGIN_SDK_VERSION
    license: str = "Apache-2.0"
    network_egress: bool = False
    labels: Sequence[str] = field(default_factory=tuple)
    languages: Sequence[str] = field(default_factory=lambda: ("*",))
    name: str = ""
    description: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.network_egress, bool):
            raise TypeError("network_egress must be a boolean")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")

        labels = _string_sequence(self.labels, "labels")
        languages = tuple(
            language.lower().replace("_", "-")
            for language in _string_sequence(self.languages, "languages")
        ) or ("*",)

        object.__setattr__(
            self, "plugin_id", _metadata_text(self.plugin_id, "plugin_id")
        )
        object.__setattr__(
            self,
            "component_id",
            _metadata_text(self.component_id, "component_id"),
        )
        object.__setattr__(self, "kind", _metadata_text(self.kind, "kind").lower())
        object.__setattr__(
            self,
            "sdk_version",
            _metadata_text(self.sdk_version, "sdk_version"),
        )
        object.__setattr__(self, "license", _metadata_text(self.license, "license"))
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "languages", languages)
        object.__setattr__(self, "name", _metadata_text(self.name, "name"))
        object.__setattr__(
            self,
            "description",
            _metadata_text(self.description, "description"),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def qualified_id(self) -> str:
        """Return the stable plugin/component identifier."""

        return f"{self.plugin_id}:{self.component_id}"

    def to_dict(self) -> dict[str, Any]:
        """Return detached, JSON-compatible metadata."""

        return {
            "plugin_id": self.plugin_id,
            "component_id": self.component_id,
            "qualified_id": self.qualified_id,
            "kind": self.kind,
            "sdk_version": self.sdk_version,
            "license": self.license,
            "network_egress": self.network_egress,
            "labels": list(self.labels),
            "languages": list(self.languages),
            "name": self.name,
            "description": self.description,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PluginComponentMetadata":
        """Build metadata from a mapping exposed by a plugin package.

        Args:
            payload: Static component metadata supplied by the entry point.

        Returns:
            Normalized, immutable plugin metadata.
        """

        return cls(
            plugin_id=payload.get("plugin_id") or payload.get("plugin") or "",
            component_id=payload.get("component_id") or payload.get("id") or "",
            kind=payload.get("kind") or "",
            sdk_version=payload.get("sdk_version") or PLUGIN_SDK_VERSION,
            license=payload.get("license") or payload.get("license_id") or "",
            network_egress=payload.get("network_egress", False),
            labels=payload.get("labels") or (),
            languages=payload.get("languages") or ("*",),
            name=payload.get("name") or "",
            description=payload.get("description") or "",
            metadata=payload.get("metadata") or {},
        )


class PluginComponent(Protocol):
    """Base protocol shared by all OpenMed plugin components."""

    metadata: PluginComponentMetadata | Mapping[str, Any]


class RecognizerPlugin(PluginComponent, Protocol):
    """Recognizer contract for emitting canonical source-text spans."""

    def recognize(self, text: str, **kwargs: Any) -> Sequence[OpenMedSpan]:
        """Return canonical spans whose offsets refer to ``text``.

        Args:
            text: Source text processed locally by the recognizer.
            **kwargs: Implementation-specific, non-breaking options.

        Returns:
            Canonical spans with offsets into ``text``.
        """


class AnonymizerProviderPlugin(PluginComponent, Protocol):
    """Anonymizer-provider contract for replacing one canonical span."""

    def replacement_for(
        self,
        span: OpenMedSpan,
        surface: str,
        **kwargs: Any,
    ) -> str:
        """Return a replacement for ``span`` without persisting ``surface``.

        Args:
            span: Canonical span being anonymized.
            surface: Source substring covered by ``span``.
            **kwargs: Implementation-specific, non-breaking options.

        Returns:
            Replacement text for the covered source substring.
        """


class ExporterPlugin(PluginComponent, Protocol):
    """Exporter contract for serializing canonical spans."""

    def export(
        self,
        spans: Sequence[OpenMedSpan],
        **kwargs: Any,
    ) -> str | bytes | Mapping[str, Any] | Sequence[Mapping[str, Any]]:
        """Serialize canonical ``spans`` without adding source-text surfaces.

        Args:
            spans: Canonical spans to export.
            **kwargs: Implementation-specific, non-breaking options.

        Returns:
            Text, bytes, or structured records derived from the spans.
        """


class InteropAdapterPlugin(PluginComponent, Protocol):
    """Interop-adapter contract for translating external records."""

    def to_openmed_spans(
        self,
        payload: Any,
        **kwargs: Any,
    ) -> Sequence[OpenMedSpan]:
        """Translate an external payload into canonical spans.

        Args:
            payload: External representation supplied by the caller.
            **kwargs: Implementation-specific, non-breaking options.

        Returns:
            Canonical spans preserving the external record's offsets.
        """

    def from_openmed_spans(
        self,
        spans: Sequence[OpenMedSpan],
        **kwargs: Any,
    ) -> Any:
        """Translate canonical spans into an external representation.

        Args:
            spans: Canonical spans to translate.
            **kwargs: Implementation-specific, non-breaking options.

        Returns:
            Adapter-specific external representation.
        """


class LanguagePackPlugin(PluginComponent, Protocol):
    """Language-pack contract for local routing and span capabilities."""

    def language_code(self) -> str:
        """Return the normalized language code supplied by the package."""

    def canonical_labels(self) -> Sequence[str]:
        """Return canonical labels supported for spans in this language."""


__all__ = [
    "COMPONENT_ANONYMIZER_PROVIDER",
    "COMPONENT_EXPORTER",
    "COMPONENT_INTEROP_ADAPTER",
    "COMPONENT_LANGUAGE_PACK",
    "COMPONENT_RECOGNIZER",
    "PLUGIN_COMPONENT_KINDS",
    "PLUGIN_SDK_MAJOR",
    "PLUGIN_SDK_VERSION",
    "AnonymizerProviderPlugin",
    "ExporterPlugin",
    "InteropAdapterPlugin",
    "LanguagePackPlugin",
    "PluginComponent",
    "PluginComponentMetadata",
    "RecognizerPlugin",
]
