"""Local privacy guards and counts-only audit artifacts."""

from importlib import import_module
from typing import Any

from .audit import (
    ARTIFACT_NAME,
    SCHEMA_VERSION,
    TraceAudit,
    TraceAuditArtifact,
    TraceAuditError,
    TracePrivacyAudit,
    build_trace_audit,
    count_categories,
    fingerprint_file,
    hash_bytes,
    hash_policy,
    render_trace_audit_json,
    render_trace_audit_markdown,
)
from .dataset import (
    BLOCK_ONLY_MODE,
    DEFAULT_MODE,
    REDACT_TO_STAGING_MODE,
    DatasetFileReport,
    DatasetFinding,
    DatasetGuardError,
    DatasetGuardReport,
    DatasetUploadBlockedError,
    DatasetUploadError,
    DatasetUploadGuard,
    DatasetUploadResult,
    guard_dataset_upload,
    inspect_dataset_files,
    redact_text,
    scan_dataset_files,
    scan_text,
)

__all__ = [
    "SessionScrubResult",
    "SessionTraceError",
    "scrub_trace",
    "ARTIFACT_NAME",
    "SCHEMA_VERSION",
    "TraceAudit",
    "TraceAuditArtifact",
    "TraceAuditError",
    "TracePrivacyAudit",
    "build_trace_audit",
    "count_categories",
    "fingerprint_file",
    "hash_bytes",
    "hash_policy",
    "render_trace_audit_json",
    "render_trace_audit_markdown",
    "BLOCK_ONLY_MODE",
    "DEFAULT_MODE",
    "REDACT_TO_STAGING_MODE",
    "DatasetFinding",
    "DatasetFileReport",
    "DatasetGuardError",
    "DatasetGuardReport",
    "DatasetUploadBlockedError",
    "DatasetUploadError",
    "DatasetUploadGuard",
    "DatasetUploadResult",
    "guard_dataset_upload",
    "inspect_dataset_files",
    "redact_text",
    "scan_dataset_files",
    "scan_text",
]


def __getattr__(name: str) -> Any:
    """Load session-hook exports without importing the executable eagerly."""
    if name not in {"SessionScrubResult", "SessionTraceError", "scrub_trace"}:
        raise AttributeError(name)
    module = import_module(".session_hook", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
