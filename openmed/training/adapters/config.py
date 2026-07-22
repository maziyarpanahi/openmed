"""Language-family transfer graph for clinical adapter selection.

The objects in this module are intentionally static metadata. They let
language-pack and registry code reason about family transfer without downloading
weights, launching training, or contacting external services.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

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


def normalize_language_code(language: str) -> str:
    """Return the normalized base language code used by transfer metadata."""

    normalized = language.strip().replace("_", "-").casefold()
    if not normalized:
        raise ValueError("language must not be empty")
    if normalized in SUPPORTED_LANGUAGES:
        return normalized
    return normalized.split("-", 1)[0]


@dataclass(frozen=True)
class LanguageFamily:
    """One family grouping used for adapter transfer decisions."""

    family_id: str
    display_name: str
    languages: tuple[str, ...]
    scripts: tuple[str, ...]
    high_resource_languages: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        """Normalize tuple fields and reject incomplete family definitions."""

        family_id = self.family_id.strip().casefold()
        languages = tuple(normalize_language_code(lang) for lang in self.languages)
        scripts = tuple(script.strip() for script in self.scripts if script.strip())
        high_resource = tuple(
            normalize_language_code(lang) for lang in self.high_resource_languages
        )
        if not family_id:
            raise ValueError("family_id must not be empty")
        if not self.display_name.strip():
            raise ValueError(f"{family_id}: display_name must not be empty")
        if not languages:
            raise ValueError(f"{family_id}: languages must not be empty")
        if not scripts:
            raise ValueError(f"{family_id}: scripts must not be empty")
        missing = set(high_resource) - set(languages)
        if missing:
            raise ValueError(
                f"{family_id}: high_resource_languages not in family: {sorted(missing)}"
            )
        object.__setattr__(self, "family_id", family_id)
        object.__setattr__(self, "languages", languages)
        object.__setattr__(self, "scripts", scripts)
        object.__setattr__(self, "high_resource_languages", high_resource)


@dataclass(frozen=True)
class AdapterMetadata:
    """Provenance and packaging metadata for a future family adapter artifact."""

    adapter_id: str
    backbone_model_id: str = DEFAULT_BACKBONE_MODEL_ID
    license: str = "apache-2.0"
    provenance: str = "synthetic family-transfer planning metadata"
    disclaimer: str = CLINICAL_ADAPTER_DISCLAIMER
    offline_runnable: bool = True

    def __post_init__(self) -> None:
        """Normalize adapter identifiers and license strings."""

        adapter_id = self.adapter_id.strip()
        license_name = self.license.strip().casefold()
        if not adapter_id:
            raise ValueError("adapter_id must not be empty")
        if not self.backbone_model_id.strip():
            raise ValueError(f"{adapter_id}: backbone_model_id must not be empty")
        if not self.provenance.strip():
            raise ValueError(f"{adapter_id}: provenance must not be empty")
        if not self.disclaimer.strip():
            raise ValueError(f"{adapter_id}: disclaimer must not be empty")
        object.__setattr__(self, "adapter_id", adapter_id)
        object.__setattr__(self, "license", license_name)


@dataclass(frozen=True)
class TransferEdge:
    """Directed donor-to-target relationship for adapter transfer."""

    target_language: str
    donor_language: str
    family_id: str
    adapter: AdapterMetadata
    priority: int = 1
    mode: str = "zero_shot_or_adapter_init"
    expected_f1_floor: float | None = None

    def __post_init__(self) -> None:
        """Normalize edge keys and enforce basic edge invariants."""

        target = normalize_language_code(self.target_language)
        donor = normalize_language_code(self.donor_language)
        family_id = self.family_id.strip().casefold()
        mode = self.mode.strip()
        if target == donor:
            raise ValueError(f"{target}: donor_language must differ from target")
        if not family_id:
            raise ValueError(f"{target}: family_id must not be empty")
        if self.priority < 1:
            raise ValueError(f"{target}: priority must be >= 1")
        if not mode:
            raise ValueError(f"{target}: mode must not be empty")
        if self.expected_f1_floor is not None and not 0 <= self.expected_f1_floor <= 1:
            raise ValueError(f"{target}: expected_f1_floor must be between 0 and 1")
        object.__setattr__(self, "target_language", target)
        object.__setattr__(self, "donor_language", donor)
        object.__setattr__(self, "family_id", family_id)
        object.__setattr__(self, "mode", mode)


@dataclass(frozen=True)
class FamilyTransferResolution:
    """Resolved transfer metadata for one target language."""

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


@dataclass(frozen=True)
class FamilyTransferConfig:
    """Validated language-family taxonomy plus directed transfer graph."""

    families: Mapping[str, LanguageFamily]
    transfer_graph: Mapping[str, tuple[TransferEdge, ...]]

    def __post_init__(self) -> None:
        """Normalize mappings and run validation at construction time."""

        families = {family.family_id: family for family in self.families.values()}
        graph = {
            normalize_language_code(target): tuple(
                sorted(edges, key=lambda edge: edge.priority)
            )
            for target, edges in self.transfer_graph.items()
        }
        object.__setattr__(self, "families", families)
        object.__setattr__(self, "transfer_graph", graph)
        self.validate()

    @property
    def languages(self) -> tuple[str, ...]:
        """Return all languages covered by the taxonomy."""

        return tuple(sorted(self._language_to_family()))

    def family_for_language(self, language: str) -> LanguageFamily | None:
        """Return the family for ``language`` when the taxonomy covers it."""

        return self._language_to_family().get(normalize_language_code(language))

    def donor_edges_for(self, language: str) -> tuple[TransferEdge, ...]:
        """Return ordered donor edges for ``language``."""

        return self.transfer_graph.get(normalize_language_code(language), ())

    def donor_languages_for(self, language: str) -> tuple[str, ...]:
        """Return ordered donor languages for ``language``."""

        return tuple(edge.donor_language for edge in self.donor_edges_for(language))

    def primary_donor_for(self, language: str) -> str | None:
        """Return the primary donor language for ``language`` if configured."""

        edges = self.donor_edges_for(language)
        return edges[0].donor_language if edges else None

    def resolve(self, language: str) -> FamilyTransferResolution | None:
        """Return transfer metadata for ``language`` if it is in the taxonomy."""

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
        """Validate family coverage, transfer edges, and adapter metadata."""

        language_to_family = self._language_to_family()
        missing_supported = SUPPORTED_LANGUAGES - set(language_to_family)
        if missing_supported:
            raise ValueError(
                f"supported languages missing from transfer families: "
                f"{sorted(missing_supported)}"
            )

        seen_priorities: dict[str, set[int]] = {}
        adjacency: dict[str, tuple[str, ...]] = {}
        for target, edges in self.transfer_graph.items():
            if target not in language_to_family:
                raise ValueError(f"{target}: transfer target has no language family")
            seen_priorities[target] = set()
            donors = []
            for edge in edges:
                if edge.target_language != target:
                    raise ValueError(
                        f"{target}: edge target mismatch {edge.target_language!r}"
                    )
                self._validate_edge(edge, language_to_family)
                if edge.priority in seen_priorities[target]:
                    raise ValueError(f"{target}: duplicate donor priority")
                seen_priorities[target].add(edge.priority)
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

    def _validate_edge(
        self,
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
        license_name = edge.adapter.license.casefold()
        if license_name not in PERMISSIVE_ADAPTER_LICENSES:
            raise ValueError(
                f"{edge.adapter.adapter_id}: adapter license {license_name!r} "
                "is not permissive"
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


def _metadata(adapter_id: str, provenance: str) -> AdapterMetadata:
    return AdapterMetadata(adapter_id=adapter_id, provenance=provenance)


DEFAULT_LANGUAGE_FAMILIES: Mapping[str, LanguageFamily] = {
    "germanic": LanguageFamily(
        family_id="germanic",
        display_name="Germanic",
        languages=("en", "de", "nl"),
        scripts=("Latin",),
        high_resource_languages=("en", "de"),
        notes="Clinical PII models share Latin-script formatting and name patterns.",
    ),
    "romance": LanguageFamily(
        family_id="romance",
        display_name="Romance",
        languages=("es", "fr", "it", "pt"),
        scripts=("Latin",),
        high_resource_languages=("es", "fr"),
        notes="Sibling donors share Latin-script demographics and date conventions.",
    ),
    "indic": LanguageFamily(
        family_id="indic",
        display_name="Indic",
        languages=("hi", "te"),
        scripts=("Devanagari", "Telugu"),
        high_resource_languages=("hi",),
        notes="Hindi is the donor for Telugu adapter initialization and fallback.",
    ),
    "semitic": LanguageFamily(
        family_id="semitic",
        display_name="Semitic",
        languages=("ar", "he"),
        scripts=("Arabic", "Hebrew"),
        high_resource_languages=("ar",),
        notes="Right-to-left clinical text requires explicit adapter metadata.",
    ),
    "japonic": LanguageFamily(
        family_id="japonic",
        display_name="Japonic",
        languages=("ja",),
        scripts=("Han", "Hiragana", "Katakana"),
        high_resource_languages=("ja",),
    ),
    "turkic": LanguageFamily(
        family_id="turkic",
        display_name="Turkic",
        languages=("tr",),
        scripts=("Latin",),
        high_resource_languages=("tr",),
    ),
    "austronesian": LanguageFamily(
        family_id="austronesian",
        display_name="Austronesian",
        languages=("id",),
        scripts=("Latin",),
        high_resource_languages=(),
    ),
    "tai-kadai": LanguageFamily(
        family_id="tai-kadai",
        display_name="Tai-Kadai",
        languages=("th",),
        scripts=("Thai",),
        high_resource_languages=(),
    ),
}

DEFAULT_TRANSFER_GRAPH: Mapping[str, tuple[TransferEdge, ...]] = {
    "nl": (
        TransferEdge(
            target_language="nl",
            donor_language="de",
            family_id="germanic",
            adapter=_metadata(
                "family-transfer/germanic-de-to-nl",
                "planned German-to-Dutch clinical PII adapter metadata",
            ),
            priority=1,
            expected_f1_floor=0.80,
        ),
        TransferEdge(
            target_language="nl",
            donor_language="en",
            family_id="germanic",
            adapter=_metadata(
                "family-transfer/germanic-en-to-nl",
                "planned English-to-Dutch clinical PII adapter metadata",
            ),
            priority=2,
        ),
    ),
    "it": (
        TransferEdge(
            target_language="it",
            donor_language="es",
            family_id="romance",
            adapter=_metadata(
                "family-transfer/romance-es-to-it",
                "planned Spanish-to-Italian clinical PII adapter metadata",
            ),
            priority=1,
            expected_f1_floor=0.80,
        ),
        TransferEdge(
            target_language="it",
            donor_language="fr",
            family_id="romance",
            adapter=_metadata(
                "family-transfer/romance-fr-to-it",
                "planned French-to-Italian clinical PII adapter metadata",
            ),
            priority=2,
        ),
    ),
    "pt": (
        TransferEdge(
            target_language="pt",
            donor_language="es",
            family_id="romance",
            adapter=_metadata(
                "family-transfer/romance-es-to-pt",
                "planned Spanish-to-Portuguese clinical PII adapter metadata",
            ),
            priority=1,
            expected_f1_floor=0.80,
        ),
        TransferEdge(
            target_language="pt",
            donor_language="fr",
            family_id="romance",
            adapter=_metadata(
                "family-transfer/romance-fr-to-pt",
                "planned French-to-Portuguese clinical PII adapter metadata",
            ),
            priority=2,
        ),
        TransferEdge(
            target_language="pt",
            donor_language="it",
            family_id="romance",
            adapter=_metadata(
                "family-transfer/romance-it-to-pt",
                "planned Italian-to-Portuguese clinical PII adapter metadata",
            ),
            priority=3,
        ),
    ),
    "te": (
        TransferEdge(
            target_language="te",
            donor_language="hi",
            family_id="indic",
            adapter=_metadata(
                "family-transfer/indic-hi-to-te",
                "planned Hindi-to-Telugu clinical PII adapter metadata",
            ),
            priority=1,
            expected_f1_floor=0.80,
        ),
    ),
    "he": (
        TransferEdge(
            target_language="he",
            donor_language="ar",
            family_id="semitic",
            adapter=_metadata(
                "family-transfer/semitic-ar-to-he",
                "planned Arabic-to-Hebrew clinical PII adapter metadata",
            ),
            priority=1,
        ),
    ),
}

DEFAULT_FAMILY_TRANSFER_CONFIG = FamilyTransferConfig(
    families=DEFAULT_LANGUAGE_FAMILIES,
    transfer_graph=DEFAULT_TRANSFER_GRAPH,
)


def get_family_transfer_config() -> FamilyTransferConfig:
    """Return the committed default family-transfer config."""

    return DEFAULT_FAMILY_TRANSFER_CONFIG
