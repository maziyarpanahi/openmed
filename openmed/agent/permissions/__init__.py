"""Local agent permission contracts."""

from __future__ import annotations

from .grants import (
    CAPABILITY_GRANT_SCHEMA_VERSION,
    CAPABILITY_GRANT_SIGNATURE_ALGORITHM,
    CapabilityGrantConstraint,
    CapabilityGrantError,
    CapabilityGrantExpiredError,
    CapabilityGrantKeyError,
    CapabilityGrantKeyProvider,
    CapabilityGrantManifest,
    CapabilityGrantRequest,
    CapabilityGrantRequiredError,
    CapabilityGrantScopeError,
    CapabilityGrantSignatureError,
    CapabilityGrantSigner,
    CapabilityGrantValidationError,
    CapabilityGrantVerificationError,
    CapabilityGrantVerifier,
    MappingCapabilityGrantKeyProvider,
    StaticCapabilityGrantKeyProvider,
    dispatch_with_capability_grant,
)

__all__ = [
    "CAPABILITY_GRANT_SCHEMA_VERSION",
    "CAPABILITY_GRANT_SIGNATURE_ALGORITHM",
    "CapabilityGrantConstraint",
    "CapabilityGrantError",
    "CapabilityGrantExpiredError",
    "CapabilityGrantKeyError",
    "CapabilityGrantKeyProvider",
    "CapabilityGrantManifest",
    "CapabilityGrantRequest",
    "CapabilityGrantRequiredError",
    "CapabilityGrantScopeError",
    "CapabilityGrantSignatureError",
    "CapabilityGrantSigner",
    "CapabilityGrantValidationError",
    "CapabilityGrantVerificationError",
    "CapabilityGrantVerifier",
    "MappingCapabilityGrantKeyProvider",
    "StaticCapabilityGrantKeyProvider",
    "dispatch_with_capability_grant",
]
