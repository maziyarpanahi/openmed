"""Conservative patient and encounter identity resolution."""

from .contracts import (
    ENTITY_TYPES,
    IDENTITY_SCHEMA_NAMES,
    IDENTITY_SCHEMA_PACKAGE,
    IDENTITY_SCHEMA_VERSION,
    IDENTITY_STATES,
    REVIEW_ACTIONS,
    IdentityContractError,
    IdentityEvidence,
    IdentityLink,
    IdentityResolution,
    IdentityResolutionRequest,
    IdentityReviewDecision,
    SourceIdentityKey,
    load_all_identity_schemas,
    load_identity_schema,
)
from .resolver import (
    CompositeIdentityResolver,
    ExactIdentityResolver,
    IdentityCandidatePlugin,
    IdentityResolver,
    ProbabilisticIdentityCandidate,
)
from .store import (
    IDENTITY_STORE_SCHEMA_VERSION,
    IdentityResolutionStore,
    IdentityStoreCompatibilityError,
    IdentityStoreError,
)

__all__ = [
    "ENTITY_TYPES",
    "IDENTITY_SCHEMA_NAMES",
    "IDENTITY_SCHEMA_PACKAGE",
    "IDENTITY_SCHEMA_VERSION",
    "IDENTITY_STATES",
    "IDENTITY_STORE_SCHEMA_VERSION",
    "REVIEW_ACTIONS",
    "CompositeIdentityResolver",
    "ExactIdentityResolver",
    "IdentityCandidatePlugin",
    "IdentityContractError",
    "IdentityEvidence",
    "IdentityLink",
    "IdentityResolution",
    "IdentityResolutionRequest",
    "IdentityResolutionStore",
    "IdentityResolver",
    "IdentityReviewDecision",
    "IdentityStoreCompatibilityError",
    "IdentityStoreError",
    "ProbabilisticIdentityCandidate",
    "SourceIdentityKey",
    "load_all_identity_schemas",
    "load_identity_schema",
]
