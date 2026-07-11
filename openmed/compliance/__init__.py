"""GDPR compliance helpers (subject-access export, erasure companion).

This package hosts local-first, access-logged compliance workflows built on the
surrogate vault and a tamper-evident audit chain.
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
]
