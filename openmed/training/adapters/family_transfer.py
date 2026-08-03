"""Public accessors for offline language-family transfer metadata."""

from __future__ import annotations

from .config import (
    DEFAULT_FAMILY_TRANSFER_CONFIG,
    AdapterMetadata,
    FamilyTransferConfig,
    FamilyTransferResolution,
)


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
    "adapter_metadata_for",
    "donor_languages_for",
    "primary_donor_for",
    "resolve_family_transfer",
]
