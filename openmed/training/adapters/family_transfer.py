"""Public helpers for language-family adapter transfer metadata."""

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
    """Resolve family-transfer metadata for ``language``."""

    return config.resolve(language)


def donor_languages_for(
    language: str,
    *,
    config: FamilyTransferConfig = DEFAULT_FAMILY_TRANSFER_CONFIG,
) -> tuple[str, ...]:
    """Return ordered donor languages for ``language``."""

    return config.donor_languages_for(language)


def primary_donor_for(
    language: str,
    *,
    config: FamilyTransferConfig = DEFAULT_FAMILY_TRANSFER_CONFIG,
) -> str | None:
    """Return the highest-priority donor language for ``language``."""

    return config.primary_donor_for(language)


def adapter_metadata_for(
    language: str,
    *,
    config: FamilyTransferConfig = DEFAULT_FAMILY_TRANSFER_CONFIG,
) -> AdapterMetadata | None:
    """Return metadata for the primary donor adapter for ``language``."""

    edges = config.donor_edges_for(language)
    return edges[0].adapter if edges else None
