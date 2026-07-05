"""Adapter-oriented training and routing metadata."""

from .config import (
    CLINICAL_ADAPTER_DISCLAIMER,
    DEFAULT_BACKBONE_MODEL_ID,
    DEFAULT_FAMILY_TRANSFER_CONFIG,
    PERMISSIVE_ADAPTER_LICENSES,
    AdapterMetadata,
    FamilyTransferConfig,
    FamilyTransferResolution,
    LanguageFamily,
    TransferEdge,
    get_family_transfer_config,
    normalize_language_code,
)
from .family_transfer import (
    adapter_metadata_for,
    donor_languages_for,
    primary_donor_for,
    resolve_family_transfer,
)

__all__ = [
    "CLINICAL_ADAPTER_DISCLAIMER",
    "DEFAULT_BACKBONE_MODEL_ID",
    "DEFAULT_FAMILY_TRANSFER_CONFIG",
    "PERMISSIVE_ADAPTER_LICENSES",
    "AdapterMetadata",
    "FamilyTransferConfig",
    "FamilyTransferResolution",
    "LanguageFamily",
    "TransferEdge",
    "adapter_metadata_for",
    "donor_languages_for",
    "get_family_transfer_config",
    "normalize_language_code",
    "primary_donor_for",
    "resolve_family_transfer",
]
