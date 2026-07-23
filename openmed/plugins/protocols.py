"""Versioned contracts for third-party OpenMed plugin components."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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
        metadata: Extra machine-readable metadata reserved for consumers.
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
        object.__setattr__(self, "plugin_id", str(self.plugin_id).strip())
        object.__setattr__(self, "component_id", str(self.component_id).strip())
        object.__setattr__(self, "kind", str(self.kind).strip().lower())
        object.__setattr__(self, "sdk_version", str(self.sdk_version).strip())
        object.__setattr__(self, "license", str(self.license).strip())
        object.__setattr__(self, "network_egress", bool(self.network_egress))
        object.__setattr__(
            self,
            "labels",
            tuple(str(label).strip() for label in self.labels if str(label).strip()),
        )
        object.__setattr__(
            self,
            "languages",
            tuple(
                str(language).strip().lower().replace("_", "-")
                for language in self.languages
                if str(language).strip()
            )
            or ("*",),
        )
        object.__setattr__(self, "name", str(self.name or "").strip())
        object.__setattr__(
            self,
            "description",
            str(self.description or "").strip(),
        )
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def qualified_id(self) -> str:
        """Return the stable plugin/component identifier."""

        return f"{self.plugin_id}:{self.component_id}"

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible metadata."""

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
        """Build metadata from a mapping exposed by a plugin package."""

        return cls(
            plugin_id=str(payload.get("plugin_id") or payload.get("plugin") or ""),
            component_id=str(payload.get("component_id") or payload.get("id") or ""),
            kind=str(payload.get("kind") or ""),
            sdk_version=str(payload.get("sdk_version") or PLUGIN_SDK_VERSION),
            license=str(payload.get("license") or payload.get("license_id") or ""),
            network_egress=bool(payload.get("network_egress", False)),
            labels=tuple(payload.get("labels") or ()),
            languages=tuple(payload.get("languages") or ("*",)),
            name=str(payload.get("name") or ""),
            description=str(payload.get("description") or ""),
            metadata=dict(payload.get("metadata") or {}),
        )


class PluginComponent(Protocol):
    """Base protocol shared by all OpenMed plugin components."""

    metadata: PluginComponentMetadata | Mapping[str, Any]


class RecognizerPlugin(PluginComponent, Protocol):
    """Recognizer contract.

    Implementations return :class:`OpenMedSpan` records with offsets into the
    input text, canonical labels from ``openmed.core.labels.CANONICAL_LABELS``,
    safe evidence/metadata, and no raw PHI persisted outside the source text.
    """

    def recognize(self, text: str, **kwargs: Any) -> Sequence[OpenMedSpan]:
        """Return canonical OpenMed spans for *text*."""


class AnonymizerProviderPlugin(PluginComponent, Protocol):
    """Anonymizer provider contract for replacing one canonical span."""

    def replacement_for(
        self,
        span: OpenMedSpan,
        surface: str,
        **kwargs: Any,
    ) -> str:
        """Return a replacement string for *span* without logging raw PHI."""


class ExporterPlugin(PluginComponent, Protocol):
    """Exporter contract for serializing canonical OpenMed spans."""

    def export(
        self,
        spans: Sequence[OpenMedSpan],
        **kwargs: Any,
    ) -> str | bytes | Mapping[str, Any] | Sequence[Mapping[str, Any]]:
        """Return a serialized representation of canonical *spans*."""


class InteropAdapterPlugin(PluginComponent, Protocol):
    """Interop adapter contract for translating external records."""

    def adapt(self, payload: Any, **kwargs: Any) -> Any:
        """Translate *payload* to or from OpenMed canonical structures."""


class LanguagePackPlugin(PluginComponent, Protocol):
    """Language-pack contract for local lexicons and label metadata."""

    def language_code(self) -> str:
        """Return the BCP-47-ish language code provided by this package."""


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
