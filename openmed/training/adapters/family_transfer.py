"""Public accessors for offline language-family transfer metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .config import (
    DEFAULT_FAMILY_TRANSFER_CONFIG,
    AdapterMetadata,
    FamilyTransferConfig,
    FamilyTransferResolution,
    TransferEdge,
    normalize_language_code,
)


class UnsupportedFamilyTransferLanguageError(ValueError):
    """Raised when adapter routing is requested for an unknown language."""


class FamilyTransferAdapterUnavailableError(LookupError):
    """Raised when neither a target nor compatible donor adapter is available."""


@dataclass(frozen=True, slots=True)
class FamilyAdapterFallback:
    """Scored provenance for a zero-shot donor-adapter selection.

    ``score`` is routing metadata, not runtime confidence. It uses the transfer
    edge's expected F1 floor when configured; otherwise it is a deterministic
    reciprocal-rank score derived from donor priority. ``provenance`` records
    which source supplied that score.
    """

    donor: str
    target: str
    score: float
    provenance: str


@dataclass(frozen=True, slots=True)
class FamilyAdapterRoute:
    """Resolved backbone and adapter metadata for a requested language."""

    target_language: str
    adapter_language: str
    family_id: str
    backbone_model_id: str
    adapter: AdapterMetadata
    mode: str
    fallback: FamilyAdapterFallback | None = None


class FamilyTransferRouter:
    """Resolve target adapters or zero-shot family donors without network I/O."""

    def __init__(
        self,
        available_adapters: Mapping[str, AdapterMetadata],
        *,
        config: FamilyTransferConfig = DEFAULT_FAMILY_TRANSFER_CONFIG,
    ) -> None:
        """Create an adapter router from an offline availability snapshot.

        Args:
            available_adapters: Installed adapter metadata keyed by the
                adapter's language. Keys may include region or script suffixes.
            config: Validated family taxonomy and ordered donor graph.

        Raises:
            TypeError: If the catalog is not a mapping or contains non-adapter
                metadata.
            ValueError: If catalog keys normalize to the same language.
        """

        if not isinstance(available_adapters, Mapping):
            raise TypeError("available_adapters must be a mapping")
        if not isinstance(config, FamilyTransferConfig):
            raise TypeError("config must be a FamilyTransferConfig")

        adapters: dict[str, AdapterMetadata] = {}
        for raw_language, adapter in available_adapters.items():
            language = normalize_language_code(raw_language)
            if language in adapters:
                raise ValueError(
                    f"available_adapters contains duplicate language {language!r}"
                )
            if not isinstance(adapter, AdapterMetadata):
                raise TypeError(
                    "available_adapters values must be AdapterMetadata records"
                )
            adapters[language] = adapter

        self._available_adapters = MappingProxyType(adapters)
        self._config = config

    @property
    def available_languages(self) -> tuple[str, ...]:
        """Return normalized languages with installed adapters."""

        return tuple(sorted(self._available_adapters))

    def route(self, language: str) -> FamilyAdapterRoute:
        """Resolve a request to a target adapter or scored donor fallback.

        Args:
            language: Requested target language, optionally region-qualified.

        Returns:
            Backbone and adapter metadata for direct or zero-shot inference.

        Raises:
            UnsupportedFamilyTransferLanguageError: If the target is outside
                the configured taxonomy.
            FamilyTransferAdapterUnavailableError: If no direct or compatible
                donor adapter is installed.
        """

        resolution = self._config.resolve(language)
        if resolution is None:
            normalized = normalize_language_code(language)
            raise UnsupportedFamilyTransferLanguageError(
                f"unsupported family-transfer language {normalized!r}"
            )

        target_adapter = self._available_adapters.get(resolution.language)
        if target_adapter is not None:
            return FamilyAdapterRoute(
                target_language=resolution.language,
                adapter_language=resolution.language,
                family_id=resolution.family.family_id,
                backbone_model_id=target_adapter.backbone_model_id,
                adapter=target_adapter,
                mode="target_adapter",
            )

        for edge in resolution.donor_edges:
            donor_adapter = self._available_adapters.get(edge.donor_language)
            if donor_adapter is None:
                continue
            if donor_adapter.backbone_model_id != edge.adapter.backbone_model_id:
                continue
            fallback = _fallback_metadata(edge, donor_adapter)
            return FamilyAdapterRoute(
                target_language=resolution.language,
                adapter_language=edge.donor_language,
                family_id=resolution.family.family_id,
                backbone_model_id=donor_adapter.backbone_model_id,
                adapter=donor_adapter,
                mode="zero_shot_fallback",
                fallback=fallback,
            )

        donors = tuple(edge.donor_language for edge in resolution.donor_edges)
        raise FamilyTransferAdapterUnavailableError(
            f"no target or compatible donor adapter is available for "
            f"{resolution.language!r}; configured donors: {donors!r}"
        )


def _fallback_metadata(
    edge: TransferEdge,
    donor_adapter: AdapterMetadata,
) -> FamilyAdapterFallback:
    if edge.expected_f1_floor is not None:
        score = edge.expected_f1_floor
        score_source = "configured expected_f1_floor"
    else:
        score = 1.0 / (edge.priority + 1)
        score_source = f"reciprocal rank for donor priority {edge.priority}"
    provenance = (
        f"{edge.adapter.provenance} Donor adapter provenance: "
        f"{donor_adapter.provenance}. Routing score source: {score_source}."
    )
    return FamilyAdapterFallback(
        donor=edge.donor_language,
        target=edge.target_language,
        score=score,
        provenance=provenance,
    )


def route_family_adapter(
    language: str,
    available_adapters: Mapping[str, AdapterMetadata],
    *,
    config: FamilyTransferConfig = DEFAULT_FAMILY_TRANSFER_CONFIG,
) -> FamilyAdapterRoute:
    """Resolve one offline target request through :class:`FamilyTransferRouter`.

    Args:
        language: Requested target language.
        available_adapters: Installed adapters keyed by language.
        config: Validated family taxonomy and donor graph.

    Returns:
        Direct target-adapter or scored zero-shot donor route.
    """

    return FamilyTransferRouter(available_adapters, config=config).route(language)


def resolve_family_transfer(
    language: str,
    *,
    config: FamilyTransferConfig = DEFAULT_FAMILY_TRANSFER_CONFIG,
) -> FamilyTransferResolution | None:
    """Resolve family and donor metadata for ``language``."""

    return config.resolve(language)


def donor_languages_for(
    language: str,
    *,
    config: FamilyTransferConfig = DEFAULT_FAMILY_TRANSFER_CONFIG,
) -> tuple[str, ...]:
    """Return ordered donor language codes for ``language``."""

    return config.donor_languages_for(language)


def primary_donor_for(
    language: str,
    *,
    config: FamilyTransferConfig = DEFAULT_FAMILY_TRANSFER_CONFIG,
) -> str | None:
    """Return the highest-priority donor language, if configured."""

    return config.primary_donor_for(language)


def adapter_metadata_for(
    language: str,
    *,
    config: FamilyTransferConfig = DEFAULT_FAMILY_TRANSFER_CONFIG,
) -> AdapterMetadata | None:
    """Return metadata for a target's primary future adapter, if configured."""

    edges = config.donor_edges_for(language)
    return edges[0].adapter if edges else None


__all__ = [
    "FamilyAdapterFallback",
    "FamilyAdapterRoute",
    "FamilyTransferAdapterUnavailableError",
    "FamilyTransferRouter",
    "UnsupportedFamilyTransferLanguageError",
    "adapter_metadata_for",
    "donor_languages_for",
    "primary_donor_for",
    "resolve_family_transfer",
    "route_family_adapter",
]
