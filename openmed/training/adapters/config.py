"""Offline language-family taxonomy and clinical adapter transfer graph.

This module contains metadata only. Importing or querying it never downloads
model weights, starts training, or contacts an external service.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from openmed.core.pii_i18n import SUPPORTED_LANGUAGES

DEFAULT_BACKBONE_MODEL_ID = "OpenMed/privacy-filter-multilingual"

PERMISSIVE_ADAPTER_LICENSES = frozenset(
    {
        "apache-2.0",
        "bsd-2-clause",
        "bsd-3-clause",
        "mit",
    }
)

CLINICAL_ADAPTER_DISCLAIMER = (
    "Family-transfer adapters are clinical decision-support components only; "
    "outputs require validation by qualified users and must not be used as the "
    "sole basis for diagnosis, treatment, or patient identification."
)


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def normalize_language_code(language: str) -> str:
    """Return a normalized base language code for transfer metadata.

    Args:
        language: ISO language code, optionally followed by a region or script.

    Returns:
        A case-folded base language code such as ``"te"``.

    Raises:
        TypeError: If ``language`` is not a string.
        ValueError: If ``language`` is empty.
    """

    normalized = _require_text(language, "language").replace("_", "-").casefold()
    return normalized.split("-", 1)[0]


def _normalize_codes(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable of language codes")
    normalized = tuple(normalize_language_code(value) for value in values)
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must not contain duplicates")
    return normalized


def _normalize_names(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable of strings")
    normalized = tuple(_require_text(value, field_name) for value in values)
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must not contain duplicates")
    return normalized


@dataclass(frozen=True, slots=True)
class LanguageFamily:
    """One operational language family used for adapter transfer decisions.

    ``scripts`` lists every script represented by the family's language
    records. The taxonomy uses broad operational groups (for example,
    ``indic``) where cross-family transfer is intentionally planned.
    """

    family_id: str
    display_name: str
    languages: tuple[str, ...]
    scripts: tuple[str, ...]
    high_resource_languages: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        """Normalize collection fields and reject incomplete records."""

        family_id = _require_text(self.family_id, "family_id").casefold()
        display_name = _require_text(self.display_name, "display_name")
        languages = _normalize_codes(self.languages, "languages")
        scripts = _normalize_names(self.scripts, "scripts")
        if isinstance(self.high_resource_languages, (str, bytes)):
            raise TypeError(
                "high_resource_languages must be an iterable of language codes"
            )
        high_resource = tuple(
            normalize_language_code(language)
            for language in self.high_resource_languages
        )
        if len(set(high_resource)) != len(high_resource):
            raise ValueError("high_resource_languages must not contain duplicates")
        missing = set(high_resource) - set(languages)
        if missing:
            raise ValueError(
                f"{family_id}: high_resource_languages not in family: {sorted(missing)}"
            )
        if not isinstance(self.notes, str):
            raise TypeError("notes must be a string")

        object.__setattr__(self, "family_id", family_id)
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "languages", languages)
        object.__setattr__(self, "scripts", scripts)
        object.__setattr__(self, "high_resource_languages", high_resource)
        object.__setattr__(self, "notes", self.notes.strip())


@dataclass(frozen=True, slots=True)
class AdapterMetadata:
    """Provenance and packaging metadata for a future family adapter."""

    adapter_id: str
    backbone_model_id: str = DEFAULT_BACKBONE_MODEL_ID
    license: str = "apache-2.0"
    provenance: str = "OpenMed built-in family-transfer planning metadata"
    disclaimer: str = CLINICAL_ADAPTER_DISCLAIMER
    offline_runnable: bool = True

    def __post_init__(self) -> None:
        """Normalize metadata strings without resolving any remote artifact."""

        adapter_id = _require_text(self.adapter_id, "adapter_id")
        backbone_model_id = _require_text(
            self.backbone_model_id,
            "backbone_model_id",
        )
        license_name = _require_text(self.license, "license").casefold()
        provenance = _require_text(self.provenance, "provenance")
        disclaimer = _require_text(self.disclaimer, "disclaimer")
        if not isinstance(self.offline_runnable, bool):
            raise TypeError("offline_runnable must be a boolean")

        object.__setattr__(self, "adapter_id", adapter_id)
        object.__setattr__(self, "backbone_model_id", backbone_model_id)
        object.__setattr__(self, "license", license_name)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "disclaimer", disclaimer)


@dataclass(frozen=True, slots=True)
class TransferEdge:
    """Directed donor-to-target relationship for future adapter transfer."""

    target_language: str
    donor_language: str
    family_id: str
    adapter: AdapterMetadata
    priority: int = 1
    mode: str = "zero_shot_or_adapter_init"
    expected_f1_floor: float | None = None

    def __post_init__(self) -> None:
        """Normalize an edge and enforce its local invariants."""

        target = normalize_language_code(self.target_language)
        donor = normalize_language_code(self.donor_language)
        family_id = _require_text(self.family_id, "family_id").casefold()
        mode = _require_text(self.mode, "mode")
        if target == donor:
            raise ValueError(f"{target}: donor_language must differ from target")
        if not isinstance(self.adapter, AdapterMetadata):
            raise TypeError("adapter must be AdapterMetadata")
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise TypeError("priority must be an integer")
        if self.priority < 1:
            raise ValueError(f"{target}: priority must be >= 1")
        if self.expected_f1_floor is not None:
            floor = self.expected_f1_floor
            if isinstance(floor, bool) or not isinstance(floor, (int, float)):
                raise TypeError("expected_f1_floor must be a number or None")
            if not math.isfinite(floor) or not 0.0 <= floor <= 1.0:
                raise ValueError(
                    f"{target}: expected_f1_floor must be a finite probability"
                )

        object.__setattr__(self, "target_language", target)
        object.__setattr__(self, "donor_language", donor)
        object.__setattr__(self, "family_id", family_id)
        object.__setattr__(self, "mode", mode)
        if self.expected_f1_floor is not None:
            object.__setattr__(
                self,
                "expected_f1_floor",
                float(self.expected_f1_floor),
            )


@dataclass(frozen=True, slots=True)
class FamilyTransferResolution:
    """Resolved taxonomy and ordered donor metadata for one target language."""

    language: str
    family: LanguageFamily
    donor_edges: tuple[TransferEdge, ...]

    @property
    def primary_edge(self) -> TransferEdge | None:
        """Return the highest-priority donor edge, if one exists."""

        return self.donor_edges[0] if self.donor_edges else None

    @property
    def primary_donor_language(self) -> str | None:
        """Return the highest-priority donor language, if one exists."""

        edge = self.primary_edge
        return edge.donor_language if edge is not None else None


@dataclass(frozen=True, slots=True)
class FamilyTransferConfig:
    """Validated language-family taxonomy plus directed transfer graph."""

    families: Mapping[str, LanguageFamily]
    transfer_graph: Mapping[str, tuple[TransferEdge, ...]]

    def __post_init__(self) -> None:
        """Copy and validate input mappings at construction time."""

        families: dict[str, LanguageFamily] = {}
        for family in self.families.values():
            if not isinstance(family, LanguageFamily):
                raise TypeError("families values must be LanguageFamily records")
            if family.family_id in families:
                raise ValueError(f"duplicate family_id {family.family_id!r}")
            families[family.family_id] = family

        graph: dict[str, tuple[TransferEdge, ...]] = {}
        for raw_target, raw_edges in self.transfer_graph.items():
            target = normalize_language_code(raw_target)
            if target in graph:
                raise ValueError(f"duplicate transfer target {target!r}")
            edges = tuple(raw_edges)
            if any(not isinstance(edge, TransferEdge) for edge in edges):
                raise TypeError("transfer_graph values must contain TransferEdge")
            graph[target] = tuple(sorted(edges, key=lambda edge: edge.priority))

        object.__setattr__(self, "families", MappingProxyType(families))
        object.__setattr__(self, "transfer_graph", MappingProxyType(graph))
        self.validate()

    @property
    def languages(self) -> tuple[str, ...]:
        """Return all language codes covered by the taxonomy."""

        return tuple(sorted(self._language_to_family()))

    def family_for_language(self, language: str) -> LanguageFamily | None:
        """Return the family record for ``language``, when covered."""

        return self._language_to_family().get(normalize_language_code(language))

    def donor_edges_for(self, language: str) -> tuple[TransferEdge, ...]:
        """Return deterministic donor edges for ``language`` by priority."""

        return self.transfer_graph.get(normalize_language_code(language), ())

    def donor_languages_for(self, language: str) -> tuple[str, ...]:
        """Return deterministic donor language codes for ``language``."""

        return tuple(edge.donor_language for edge in self.donor_edges_for(language))

    def primary_donor_for(self, language: str) -> str | None:
        """Return the primary donor code for ``language``, if configured."""

        edges = self.donor_edges_for(language)
        return edges[0].donor_language if edges else None

    def resolve(self, language: str) -> FamilyTransferResolution | None:
        """Resolve taxonomy and donor metadata for a supported language."""

        normalized = normalize_language_code(language)
        family = self.family_for_language(normalized)
        if family is None:
            return None
        return FamilyTransferResolution(
            language=normalized,
            family=family,
            donor_edges=self.donor_edges_for(normalized),
        )

    def validate(self) -> None:
        """Validate coverage, donor references, licenses, and graph acyclicity."""

        language_to_family = self._language_to_family()
        missing_supported = set(SUPPORTED_LANGUAGES) - set(language_to_family)
        if missing_supported:
            raise ValueError(
                "supported languages missing from transfer families: "
                f"{sorted(missing_supported)}"
            )

        adapter_ids: set[str] = set()
        adjacency: dict[str, tuple[str, ...]] = {}
        for target, edges in self.transfer_graph.items():
            if target not in language_to_family:
                raise ValueError(f"{target}: transfer target has no language family")
            priorities: set[int] = set()
            donors: list[str] = []
            for edge in edges:
                if edge.target_language != target:
                    raise ValueError(
                        f"{target}: edge target mismatch {edge.target_language!r}"
                    )
                self._validate_edge(edge, language_to_family)
                if edge.priority in priorities:
                    raise ValueError(f"{target}: duplicate donor priority")
                if edge.adapter.adapter_id in adapter_ids:
                    raise ValueError(
                        f"duplicate adapter_id {edge.adapter.adapter_id!r}"
                    )
                priorities.add(edge.priority)
                adapter_ids.add(edge.adapter.adapter_id)
                donors.append(edge.donor_language)
            adjacency[target] = tuple(donors)
        self._reject_cycles(adjacency)

    def _language_to_family(self) -> dict[str, LanguageFamily]:
        family_by_language: dict[str, LanguageFamily] = {}
        for family in self.families.values():
            for language in family.languages:
                existing = family_by_language.get(language)
                if existing is not None:
                    raise ValueError(
                        f"{language}: present in both {existing.family_id} "
                        f"and {family.family_id}"
                    )
                family_by_language[language] = family
        return family_by_language

    @staticmethod
    def _validate_edge(
        edge: TransferEdge,
        language_to_family: Mapping[str, LanguageFamily],
    ) -> None:
        if edge.donor_language not in language_to_family:
            raise ValueError(
                f"{edge.target_language}: donor {edge.donor_language!r} "
                "has no language family"
            )
        target_family = language_to_family[edge.target_language]
        donor_family = language_to_family[edge.donor_language]
        if edge.family_id != target_family.family_id:
            raise ValueError(
                f"{edge.target_language}: edge family {edge.family_id!r} "
                f"does not match target family {target_family.family_id!r}"
            )
        if donor_family.family_id != target_family.family_id:
            raise ValueError(
                f"{edge.target_language}: donor {edge.donor_language!r} "
                f"is in family {donor_family.family_id!r}, not "
                f"{target_family.family_id!r}"
            )
        if edge.adapter.license not in PERMISSIVE_ADAPTER_LICENSES:
            raise ValueError(
                f"{edge.adapter.adapter_id}: adapter license "
                f"{edge.adapter.license!r} is not permissive"
            )
        if not edge.adapter.offline_runnable:
            raise ValueError(
                f"{edge.adapter.adapter_id}: adapter metadata must be offline-runnable"
            )

    @staticmethod
    def _reject_cycles(adjacency: Mapping[str, tuple[str, ...]]) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(language: str, path: tuple[str, ...]) -> None:
            if language in visiting:
                cycle = " -> ".join((*path, language))
                raise ValueError(f"transfer graph contains a cycle: {cycle}")
            if language in visited:
                return
            visiting.add(language)
            for donor in adjacency.get(language, ()):
                visit(donor, (*path, language))
            visiting.remove(language)
            visited.add(language)

        for language in adjacency:
            visit(language, ())


DEFAULT_LANGUAGE_FAMILIES: Mapping[str, LanguageFamily] = MappingProxyType(
    {
        "germanic": LanguageFamily(
            family_id="germanic",
            display_name="Germanic",
            languages=("da", "de", "en", "nl", "no", "sv"),
            scripts=("Latin",),
            high_resource_languages=("de", "en"),
        ),
        "romance": LanguageFamily(
            family_id="romance",
            display_name="Romance",
            languages=("es", "fr", "it", "pt", "ro"),
            scripts=("Latin",),
            high_resource_languages=("es", "fr"),
        ),
        "indic": LanguageFamily(
            family_id="indic",
            display_name="Indic transfer group",
            languages=("as", "bn", "hi", "mr", "or", "ta", "te"),
            scripts=("Bengali", "Devanagari", "Odia", "Tamil", "Telugu"),
            high_resource_languages=("bn", "hi"),
            notes=(
                "Operational South Asian transfer group spanning Indo-Aryan and "
                "Dravidian languages."
            ),
        ),
        "semitic": LanguageFamily(
            family_id="semitic",
            display_name="Semitic",
            languages=("am", "ar", "he"),
            scripts=("Arabic", "Ethiopic", "Hebrew"),
            high_resource_languages=("ar",),
        ),
        "slavic": LanguageFamily(
            family_id="slavic",
            display_name="Slavic",
            languages=("cs", "ru", "uk"),
            scripts=("Cyrillic", "Latin"),
            high_resource_languages=("ru",),
        ),
        "bantu": LanguageFamily(
            family_id="bantu",
            display_name="Bantu",
            languages=("sw", "xh", "zu"),
            scripts=("Latin",),
            high_resource_languages=("sw",),
        ),
        "hellenic": LanguageFamily(
            family_id="hellenic",
            display_name="Hellenic",
            languages=("el",),
            scripts=("Greek",),
        ),
        "austronesian": LanguageFamily(
            family_id="austronesian",
            display_name="Austronesian",
            languages=("id",),
            scripts=("Latin",),
        ),
        "japonic": LanguageFamily(
            family_id="japonic",
            display_name="Japonic",
            languages=("ja",),
            scripts=("Han", "Hiragana/Katakana"),
            high_resource_languages=("ja",),
        ),
        "koreanic": LanguageFamily(
            family_id="koreanic",
            display_name="Koreanic",
            languages=("ko",),
            scripts=("Hangul",),
            high_resource_languages=("ko",),
        ),
        "sinitic": LanguageFamily(
            family_id="sinitic",
            display_name="Sinitic",
            languages=("zh",),
            scripts=("Han",),
            high_resource_languages=("zh",),
        ),
        "tai-kadai": LanguageFamily(
            family_id="tai-kadai",
            display_name="Tai-Kadai",
            languages=("th",),
            scripts=("Thai",),
        ),
        "turkic": LanguageFamily(
            family_id="turkic",
            display_name="Turkic",
            languages=("tr",),
            scripts=("Latin",),
            high_resource_languages=("tr",),
        ),
    }
)


def _edge(
    target: str,
    donor: str,
    family_id: str,
    priority: int,
    *,
    expected_f1_floor: float | None = None,
) -> TransferEdge:
    adapter_id = f"family-transfer/{family_id}-{donor}-to-{target}"
    provenance = (
        f"OpenMed built-in planning metadata for {donor}-to-{target} clinical "
        "PII transfer; no model weights are bundled."
    )
    return TransferEdge(
        target_language=target,
        donor_language=donor,
        family_id=family_id,
        adapter=AdapterMetadata(
            adapter_id=adapter_id,
            provenance=provenance,
        ),
        priority=priority,
        expected_f1_floor=expected_f1_floor,
    )


DEFAULT_TRANSFER_GRAPH: Mapping[str, tuple[TransferEdge, ...]] = MappingProxyType(
    {
        "am": (_edge("am", "ar", "semitic", 1),),
        "as": (_edge("as", "bn", "indic", 1),),
        "cs": (
            _edge("cs", "uk", "slavic", 1),
            _edge("cs", "ru", "slavic", 2),
        ),
        "da": (
            _edge("da", "no", "germanic", 1),
            _edge("da", "sv", "germanic", 2),
        ),
        "he": (_edge("he", "ar", "semitic", 1),),
        "it": (
            _edge("it", "es", "romance", 1, expected_f1_floor=0.80),
            _edge("it", "fr", "romance", 2),
        ),
        "mr": (_edge("mr", "hi", "indic", 1),),
        "nl": (
            _edge("nl", "de", "germanic", 1, expected_f1_floor=0.80),
            _edge("nl", "en", "germanic", 2),
        ),
        "no": (_edge("no", "sv", "germanic", 1),),
        "or": (
            _edge("or", "hi", "indic", 1),
            _edge("or", "bn", "indic", 2),
        ),
        "pt": (
            _edge("pt", "es", "romance", 1, expected_f1_floor=0.80),
            _edge("pt", "fr", "romance", 2),
            _edge("pt", "it", "romance", 3),
        ),
        "ro": (
            _edge("ro", "it", "romance", 1),
            _edge("ro", "es", "romance", 2),
            _edge("ro", "fr", "romance", 3),
        ),
        "ta": (
            _edge("ta", "te", "indic", 1),
            _edge("ta", "hi", "indic", 2),
        ),
        "te": (_edge("te", "hi", "indic", 1, expected_f1_floor=0.80),),
        "uk": (_edge("uk", "ru", "slavic", 1),),
        "xh": (
            _edge("xh", "zu", "bantu", 1),
            _edge("xh", "sw", "bantu", 2),
        ),
        "zu": (_edge("zu", "sw", "bantu", 1),),
    }
)

DEFAULT_FAMILY_TRANSFER_CONFIG = FamilyTransferConfig(
    families=DEFAULT_LANGUAGE_FAMILIES,
    transfer_graph=DEFAULT_TRANSFER_GRAPH,
)


def get_family_transfer_config() -> FamilyTransferConfig:
    """Return the committed, immutable, offline family-transfer config."""

    return DEFAULT_FAMILY_TRANSFER_CONFIG


__all__ = [
    "CLINICAL_ADAPTER_DISCLAIMER",
    "DEFAULT_BACKBONE_MODEL_ID",
    "DEFAULT_FAMILY_TRANSFER_CONFIG",
    "DEFAULT_LANGUAGE_FAMILIES",
    "DEFAULT_TRANSFER_GRAPH",
    "PERMISSIVE_ADAPTER_LICENSES",
    "AdapterMetadata",
    "FamilyTransferConfig",
    "FamilyTransferResolution",
    "LanguageFamily",
    "TransferEdge",
    "get_family_transfer_config",
    "normalize_language_code",
]
