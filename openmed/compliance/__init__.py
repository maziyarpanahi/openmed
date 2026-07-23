"""Local-first compliance reporting and subject-access helpers.

This package hosts local-first, access-logged compliance workflows built on the
surrogate vault, de-identification audit reports, and a tamper-evident audit
chain.
"""

from .audit_chain import (
    AuditRecord,
    AuditSink,
    HashChainAuditLog,
)
from .dsar import (
    DSAR_ADVISORY,
    ERASURE_PREVIEW_EVENT,
    EXPORT_EVENT,
    DsarEntry,
    DsarPackage,
    ErasurePlan,
    SubjectIdentifier,
    assemble_dsar_package,
    plan_erasure,
    render_dsar_summary,
)
from .safe_harbor import (
    SAFE_HARBOR_ATTESTATION_NOTICE,
    SAFE_HARBOR_ATTESTATION_SCHEMA_VERSION,
    SAFE_HARBOR_CATEGORY_LABELS,
    SAFE_HARBOR_CATEGORY_ORDER,
    SAFE_HARBOR_POLICY,
    SafeHarborAttestation,
    SafeHarborCategoryAttestation,
    generate_safe_harbor_attestation,
)

__all__ = [
    "AuditRecord",
    "AuditSink",
    "HashChainAuditLog",
    "DSAR_ADVISORY",
    "EXPORT_EVENT",
    "ERASURE_PREVIEW_EVENT",
    "SubjectIdentifier",
    "DsarEntry",
    "DsarPackage",
    "ErasurePlan",
    "assemble_dsar_package",
    "render_dsar_summary",
    "plan_erasure",
    "SAFE_HARBOR_ATTESTATION_NOTICE",
    "SAFE_HARBOR_ATTESTATION_SCHEMA_VERSION",
    "SAFE_HARBOR_CATEGORY_LABELS",
    "SAFE_HARBOR_CATEGORY_ORDER",
    "SAFE_HARBOR_POLICY",
    "SafeHarborAttestation",
    "SafeHarborCategoryAttestation",
    "generate_safe_harbor_attestation",
]
