"""Bounded, value-free diagnostic and recovery bundles for OpenMed services.

A diagnostic bundle is a small, deterministic JSON document that helps an
operator or maintainer understand a failed or degraded service without
collecting restorable clinical state.  Collection is built from a closed set of
typed collectors, one per evidence category.  Every collector:

* reads only its own typed source (a job-metadata file, a read-only SQLite
  connection, a registry-state file, an audit chain, live in-process objects),
  through a budget that enforces elapsed time, file count, per-file bytes, total
  bytes, directory depth, and directory-entry limits;
* emits controlled states, counts, versions, hour-precision timestamps, and
  domain-separated digests only, never values, paths, payloads, identifiers,
  exception messages, or environment contents;
* never opens a network connection and never writes to the system it inspects.

A second, shape-level value-safety contract is applied to every section after
collection, and again by :func:`verify_diagnostic_bundle`.  The bundle carries a
deterministic manifest with per-entry digests so a recipient can detect missing,
added, reordered, and modified entries offline.  Recovery guidance is derived
from controlled reason codes and findings as action categories (``inspect``,
``retry``, ``restore``, ``quarantine``, ``escalate``) with documentation links.

The bundle is diagnostic metadata only.  It is never a backup and cannot be used
to restore service state.  See ``docs/recovery/diagnostic-bundle.md``.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import sqlite3
import stat
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

BUNDLE_FORMAT: Final = "openmed.diagnostic-bundle"
BUNDLE_SCHEMA_VERSION: Final = 1
TIMESTAMP_PRECISION_SECONDS: Final = 3600
DOCS_PAGE: Final = "docs/recovery/diagnostic-bundle.md"

# Evidence categories in canonical bundle order.
CATEGORIES: Final[tuple[str, ...]] = (
    "runtime",
    "capabilities",
    "configuration",
    "health",
    "readiness",
    "operational_events",
    "jobs",
    "journey_store",
    "model_registry",
    "model_activation",
    "audit_continuity",
    "backup",
    "disk",
)
# Categories whose evidence may come from a caller-registered provider, and
# the only top-level data fields such a provider may set.
PROVIDER_FIELDS: Final[Mapping[str, frozenset[str]]] = {
    "backup": frozenset(
        {
            "state",
            "schema_version",
            "backup_count",
            "verified_count",
            "latest_age_class",
            "latest_verification",
            "latest_manifest_digest",
            "encrypted",
        }
    ),
    "model_activation": frozenset(
        {
            "state",
            "journal_schema_version",
            "journal_entry_count",
            "active_slot_count",
            "pending_activation_count",
            "rollback_count",
            "served_models_explicit",
            "served_model_count",
            "served_model_digests",
            "preload_model_count",
            "resident_model_count",
            "max_resident_models",
        }
    ),
}
PROVIDER_CATEGORIES: Final = frozenset(PROVIDER_FIELDS)

STATUS_COMPLETE: Final = "complete"
STATUS_PARTIAL: Final = "partial"
STATUS_UNAVAILABLE: Final = "unavailable"
STATUS_TIMEOUT: Final = "timeout"
STATUS_CANCELLED: Final = "cancelled"
STATUS_FAILED: Final = "failed"
SECTION_STATUSES: Final = (
    STATUS_COMPLETE,
    STATUS_PARTIAL,
    STATUS_UNAVAILABLE,
    STATUS_TIMEOUT,
    STATUS_CANCELLED,
    STATUS_FAILED,
)

# Stable reasons explaining why a section is partial or missing evidence.
PARTIAL_RESULT_CODES: Final = frozenset(
    {
        "source_not_configured",
        "source_absent",
        "source_unsafe",
        "source_unreadable",
        "source_corrupt",
        "schema_unsupported",
        "provider_not_registered",
        "not_in_process",
        "section_timeout",
        "elapsed_budget_exhausted",
        "collection_cancelled",
        "file_limit_reached",
        "file_too_large",
        "total_bytes_limit_reached",
        "depth_limit_reached",
        "entry_limit_reached",
        "item_limit_reached",
        "section_too_large",
        "collector_failed",
    }
)

# Stable counts of evidence deliberately left out of a section.
OMISSION_REASONS: Final = frozenset(
    {
        "unsafe_value",
        "unsafe_key",
        "unknown_code",
        "unknown_key",
        "unknown_variable",
        "invalid_identifier",
        "unrecognized_entry",
    }
)

# Stable observations about the inspected system.
FINDING_CODES: Final = frozenset(
    {
        "persisted_credential",
        "config_value_invalid",
        "profile_unresolved",
        "service_setting_invalid",
        "served_models_not_explicit",
        "health_check_failed",
        "process_not_ready",
        "process_shutting_down",
        "server_errors_observed",
        "operational_failures_observed",
        "circuit_breaker_open",
        "load_shedding_observed",
        "job_store_corrupt",
        "jobs_failed_present",
        "jobs_incomplete_present",
        "webhook_delivery_failed",
        "journey_store_corrupt",
        "integrity_check_failed",
        "migration_uninitialized",
        "migration_pending",
        "migration_ahead",
        "migration_gap",
        "migration_checksum_mismatch",
        "quarantined_records_present",
        "staged_blobs_present",
        "unexpected_store_entries",
        "registry_state_invalid",
        "registry_migration_required",
        "audit_chain_broken",
        "audit_chain_invalid",
        "backup_missing",
        "backup_stale",
        "backup_verification_failed",
        "disk_space_low",
        "disk_space_critical",
    }
)

RECOVERY_KINDS: Final = ("inspect", "retry", "restore", "quarantine", "escalate")

# Recovery action id -> kind.  Each id has a section in ``DOCS_PAGE``.
RECOVERY_ACTIONS: Final[Mapping[str, str]] = {
    "review_configuration": "inspect",
    "inspect_readiness": "inspect",
    "inspect_failed_jobs": "inspect",
    "inspect_interrupted_writes": "inspect",
    "free_disk_space": "inspect",
    "inspect_backup_schedule": "inspect",
    "rerun_with_larger_limits": "retry",
    "retry_collection": "retry",
    "rerun_migrations": "retry",
    "resubmit_interrupted_jobs": "retry",
    "retry_webhook_delivery": "retry",
    "restore_from_backup": "restore",
    "restore_registry_state": "restore",
    "quarantine_job_store": "quarantine",
    "quarantine_journey_store": "quarantine",
    "review_quarantined_records": "quarantine",
    "rotate_credentials": "escalate",
    "escalate_audit_discontinuity": "escalate",
    "align_release_version": "escalate",
    "escalate_backup_gap": "escalate",
    "report_collector_failure": "escalate",
}

# Reason code or finding -> suggested recovery actions.
RECOVERY_RULES: Final[Mapping[str, tuple[str, ...]]] = {
    "section_timeout": ("rerun_with_larger_limits",),
    "elapsed_budget_exhausted": ("rerun_with_larger_limits",),
    "file_limit_reached": ("rerun_with_larger_limits",),
    "file_too_large": ("rerun_with_larger_limits",),
    "total_bytes_limit_reached": ("rerun_with_larger_limits",),
    "depth_limit_reached": ("rerun_with_larger_limits",),
    "entry_limit_reached": ("rerun_with_larger_limits",),
    "item_limit_reached": ("rerun_with_larger_limits",),
    "section_too_large": ("rerun_with_larger_limits",),
    "collection_cancelled": ("retry_collection",),
    "source_unreadable": ("review_configuration",),
    "source_unsafe": ("review_configuration",),
    "schema_unsupported": ("align_release_version",),
    "collector_failed": ("report_collector_failure",),
    "persisted_credential": ("rotate_credentials",),
    "config_value_invalid": ("review_configuration",),
    "profile_unresolved": ("review_configuration",),
    "service_setting_invalid": ("review_configuration",),
    "served_models_not_explicit": ("review_configuration",),
    "health_check_failed": ("review_configuration",),
    "process_not_ready": ("inspect_readiness",),
    "process_shutting_down": ("inspect_readiness",),
    "server_errors_observed": ("inspect_readiness",),
    "operational_failures_observed": ("inspect_readiness",),
    "circuit_breaker_open": ("inspect_readiness",),
    "load_shedding_observed": ("inspect_readiness",),
    "job_store_corrupt": ("quarantine_job_store", "restore_from_backup"),
    "jobs_failed_present": ("inspect_failed_jobs",),
    "jobs_incomplete_present": ("resubmit_interrupted_jobs",),
    "webhook_delivery_failed": ("retry_webhook_delivery",),
    "journey_store_corrupt": ("quarantine_journey_store", "restore_from_backup"),
    "integrity_check_failed": ("quarantine_journey_store", "restore_from_backup"),
    "migration_uninitialized": ("rerun_migrations",),
    "migration_pending": ("rerun_migrations",),
    "migration_ahead": ("align_release_version",),
    "migration_gap": ("restore_from_backup", "align_release_version"),
    "migration_checksum_mismatch": ("restore_from_backup", "align_release_version"),
    "quarantined_records_present": ("review_quarantined_records",),
    "staged_blobs_present": ("inspect_interrupted_writes",),
    "unexpected_store_entries": ("inspect_interrupted_writes",),
    "registry_state_invalid": ("restore_registry_state",),
    "registry_migration_required": ("rerun_migrations",),
    "audit_chain_broken": ("escalate_audit_discontinuity", "restore_from_backup"),
    "audit_chain_invalid": ("escalate_audit_discontinuity",),
    "backup_missing": ("escalate_backup_gap",),
    "backup_stale": ("inspect_backup_schedule",),
    "backup_verification_failed": ("escalate_backup_gap",),
    "disk_space_low": ("free_disk_space",),
    "disk_space_critical": ("free_disk_space",),
}

REQUEST_ERROR_CODES: Final = frozenset(
    {
        "request_invalid",
        "category_unknown",
        "category_duplicate",
        "category_empty",
        "limit_invalid",
        "limit_below_minimum",
        "limit_above_maximum",
        "provider_invalid",
        "provider_category_unsupported",
        "output_exists",
        "output_unwritable",
    }
)

VERIFICATION_REASON_CODES: Final = frozenset(
    {
        "bundle_valid",
        "bundle_too_large",
        "bundle_not_json",
        "bundle_not_object",
        "format_unsupported",
        "schema_version_unsupported",
        "missing_field",
        "unexpected_field",
        "header_invalid",
        "header_modified",
        "manifest_invalid",
        "manifest_modified",
        "entry_invalid",
        "entry_missing",
        "entry_added",
        "entry_reordered",
        "entry_modified",
        "unsafe_content",
        "recovery_invalid",
        "recovery_inconsistent",
    }
)

# Fixed JSON nesting bound for parsed sources; directory recursion uses limits.
MAX_SOURCE_JSON_NESTING: Final = 32
# Maximum nesting of a section's ``data`` mapping.
MAX_DATA_DEPTH: Final = 8

_ENTRY_DOMAIN: Final = b"openmed.diagnostics.entry.v1\x00"
_HEADER_DOMAIN: Final = b"openmed.diagnostics.header.v1\x00"
_RECOVERY_DOMAIN: Final = b"openmed.diagnostics.recovery.v1\x00"
_MANIFEST_DOMAIN: Final = b"openmed.diagnostics.manifest.v1\x00"
_IDENTIFIER_DOMAIN: Final = b"openmed.diagnostics.identifier.v1\x00"

_TOKEN_RE: Final = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_DIGEST_RE: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_VERSION_RE: Final = re.compile(
    r"[0-9]{1,6}(?:\.[0-9]{1,8}){0,4}(?:(?:a|b|rc)[0-9]{1,6})?"
    r"(?:\.post[0-9]{1,6})?(?:\.dev[0-9]{1,8})?(?:\+[a-z0-9]{1,16}"
    r"(?:\.[a-z0-9]{1,16}){0,3})?\Z"
)
_SLOT_RE: Final = re.compile(
    r"[a-z0-9][a-z0-9_.-]{0,31}(?:::[a-z0-9][a-z0-9_.-]{0,31}){2}\Z"
)
_TIMESTAMP_RE: Final = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:00:00Z\Z")
_PROFILE_NAME_RE: Final = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_BLOB_NAME_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_SHARD_NAME_RE: Final = re.compile(r"[0-9a-f]{2}\Z")
_MAX_SAFE_INT: Final = 2**53

_JOBS_STORE_ENV_VAR: Final = "OPENMED_SERVICE_JOBS_STORE_PATH"
_SERVED_MODELS_ENV_VAR: Final = "OPENMED_SERVICE_SERVED_MODELS"
_PRELOAD_MODELS_ENV_VAR: Final = "OPENMED_SERVICE_PRELOAD_MODELS"
_PROFILE_ENV_VAR: Final = "OPENMED_PROFILE"
_SERVICE_DEFAULT_PROFILE: Final = "prod"
_JOB_STATUSES: Final = ("queued", "running", "done", "failed")

# Recognized service settings by category.  Only presence is ever reported.
_SERVICE_SETTINGS: Final[Mapping[str, tuple[str, ...]]] = {
    "auth": (
        "AUTH_API_KEYS",
        "AUTH_DENY_BY_DEFAULT",
        "AUTH_ENABLED",
        "AUTH_FAILURE_RATE_LIMIT_BURST",
        "AUTH_FAILURE_RATE_LIMIT_KEY",
        "AUTH_FAILURE_RATE_LIMIT_RPS",
        "AUTH_JWKS",
        "AUTH_JWKS_FILE",
        "AUTH_JWT_AUDIENCE",
        "AUTH_JWT_ISSUER",
        "AUTH_JWT_LEEWAY_SECONDS",
        "AUTH_JWT_MAX_LIFETIME_SECONDS",
        "AUTH_ROUTE_SCOPES",
    ),
    "batching": (
        "BATCHING_ENABLED",
        "BATCH_HIGH_WATERMARK",
        "BATCH_LOW_WATERMARK",
        "BATCH_MAX_QUEUE_SIZE",
        "BATCH_MAX_QUEUE_WAIT_MS",
        "BATCH_MAX_SIZE",
        "BATCH_MAX_WAIT_MS",
        "COALESCING_ENABLED",
    ),
    "bulk_data": ("BULK_INPUT_ROOTS", "BULK_OUTPUT_ROOT"),
    "integrations": (
        "EXPERIMENTAL_MULTIMODAL_PREFLIGHT",
        "PRIVACY_GATEWAY_ENDPOINT",
        "SYSTEM_ONE",
    ),
    "jobs": ("JOBS_STORE_PATH", "JOBS_TTL_SECONDS", "JOBS_WORKERS"),
    "limits": ("MAX_REQUEST_BYTES", "MAX_TEXT_LENGTH"),
    "models": (
        "DEFAULT_MODEL_FOOTPRINT_BYTES",
        "KEEP_ALIVE",
        "MAX_RESIDENT_MODELS",
        "MLX_PAGED_KV_CACHE_BUDGET",
        "MLX_PAGED_KV_CACHE_BYTES_PER_TOKEN",
        "MLX_PAGED_KV_CACHE_CHUNK_TOKENS",
        "MLX_PAGED_KV_CACHE_PAGE_TOKENS",
        "MLX_PAGED_KV_CACHE_WINDOW_TOKENS",
        "MODEL_ADMISSION_WAIT_SECONDS",
        "MODEL_MEMORY_BUDGET_BYTES",
        "PRELOAD_MODELS",
        "SERVED_MODELS",
    ),
    "network": (
        "CORS_ORIGINS",
        "MTLS_CA_BUNDLE",
        "MTLS_CLIENT_CERT_HEADER",
        "MTLS_ENABLED",
        "MTLS_PRINCIPALS",
        "MTLS_TRUSTED_PROXIES",
        "TRUSTED_HOSTS",
    ),
    "observability": (
        "LOG_FORMAT",
        "LOG_LEVEL",
        "METRICS_ENABLED",
        "OTLP_ENDPOINT",
        "OTLP_HEADERS",
        "OTLP_TIMEOUT_SECONDS",
        "TRACING_ENABLED",
    ),
    "resilience": (
        "CIRCUIT_BREAKER_FAILURE_THRESHOLD",
        "CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS",
        "RESILIENCE_ENABLED",
        "RETRY_BACKOFF_INITIAL_SECONDS",
        "RETRY_BACKOFF_JITTER_SECONDS",
        "RETRY_BACKOFF_MAX_SECONDS",
        "RETRY_BACKOFF_MULTIPLIER",
        "RETRY_MAX_ATTEMPTS",
        "SHUTDOWN_DRAIN_SECONDS",
    ),
    "throttle": (
        "CONCURRENCY_WAIT_SECONDS",
        "RATE_LIMIT_BURST",
        "RATE_LIMIT_MAX_CONCURRENCY",
        "RATE_LIMIT_RPS",
        "THROTTLE_KEY",
    ),
}
# Settings whose values are credentials or key material.
_SECRET_BEARING_SETTINGS: Final = frozenset(
    {"AUTH_API_KEYS", "AUTH_JWKS", "OTLP_HEADERS"}
)
_SECRET_CONFIG_KEYS: Final = frozenset({"hf_token"})
_SERVICE_PREFIX: Final = "OPENMED_SERVICE_"
_CORE_VARIABLES: Final = frozenset(
    {
        "OPENMED_CONFIG",
        "OPENMED_DEVICE",
        "OPENMED_OFFLINE",
        "OPENMED_PROFILE",
        "OPENMED_TORCH_DEVICE",
        "OPENMED_TORCH_ATTENTION_BACKEND",
        "OPENMED_CHINESE_USER_DICT",
        "OPENMED_CLINICAL_PROTECT",
    }
)

_DEPENDENCIES: Final = (
    ("fastapi", "fastapi"),
    ("grpcio", "grpcio"),
    ("huggingface_hub", "huggingface-hub"),
    ("mlx", "mlx"),
    ("numpy", "numpy"),
    ("onnxruntime", "onnxruntime"),
    ("pydantic", "pydantic"),
    ("starlette", "starlette"),
    ("strawberry_graphql", "strawberry-graphql"),
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("uvicorn", "uvicorn"),
)
_MACHINES: Final = frozenset({"x86_64", "amd64", "arm64", "aarch64", "armv7l"})
_SYSTEMS: Final = frozenset({"darwin", "linux", "windows"})

_DISK_CRITICAL_BYTES: Final = 512 * 1024**2
_DISK_LOW_BYTES: Final = 2 * 1024**3


# ---------------------------------------------------------------------------
# Public request contract
# ---------------------------------------------------------------------------


class DiagnosticRequestError(ValueError):
    """Value-free refusal of an invalid diagnostic request or output.

    Attributes:
        code: A stable code from :data:`REQUEST_ERROR_CODES`.
    """

    def __init__(self, code: str) -> None:
        self.code = code if code in REQUEST_ERROR_CODES else "request_invalid"
        super().__init__(self.code)


_LIMIT_BOUNDS: Final[Mapping[str, tuple[int, int]]] = {
    "max_elapsed_ms": (10, 300_000),
    "section_timeout_ms": (10, 120_000),
    "max_files": (1, 256),
    "max_file_bytes": (1, 64 * 1024**2),
    "max_total_bytes": (1, 256 * 1024**2),
    "max_depth": (1, 12),
    "max_dir_entries": (1, 65_536),
    "max_items": (1, 4_096),
    "max_bundle_bytes": (32 * 1024, 2 * 1024**2),
}


@dataclass(frozen=True)
class DiagnosticLimits:
    """Hard collection limits, validated before collection starts.

    Every limit has a documented default and a documented maximum.  Callers may
    tighten any limit; values above the maximum or below the minimum are
    refused with :class:`DiagnosticRequestError`.

    Attributes:
        max_elapsed_ms: Total collection time budget in milliseconds.
        section_timeout_ms: Per-category time budget in milliseconds.
        max_files: Number of source files that may be opened.
        max_file_bytes: Bytes that may be read from one source file.
        max_total_bytes: Bytes that may be read across all source files.
        max_depth: Directory recursion depth below a store root.
        max_dir_entries: Directory entries that may be visited in total.
        max_items: Items kept in one list of a section.
        max_bundle_bytes: Size of the serialized bundle.
    """

    max_elapsed_ms: int = 60_000
    section_timeout_ms: int = 15_000
    max_files: int = 32
    max_file_bytes: int = 8 * 1024**2
    max_total_bytes: int = 32 * 1024**2
    max_depth: int = 6
    max_dir_entries: int = 4_096
    max_items: int = 256
    max_bundle_bytes: int = 256 * 1024

    def __post_init__(self) -> None:
        for name, (minimum, maximum) in _LIMIT_BOUNDS.items():
            value = getattr(self, name)
            if type(value) is not int:
                raise DiagnosticRequestError("limit_invalid")
            if value < minimum:
                raise DiagnosticRequestError("limit_below_minimum")
            if value > maximum:
                raise DiagnosticRequestError("limit_above_maximum")

    @classmethod
    def maximum(cls) -> "DiagnosticLimits":
        """Return the most permissive limits a request may select."""

        return cls(**{name: bounds[1] for name, bounds in _LIMIT_BOUNDS.items()})

    def to_dict(self) -> dict[str, int]:
        """Return the limits as a deterministic mapping of integers."""

        return {item.name: getattr(self, item.name) for item in fields(self)}


@dataclass(frozen=True)
class DiagnosticRequest:
    """Versioned diagnostic-bundle request.

    Attributes:
        categories: Evidence categories to collect.  They are normalized to the
            canonical :data:`CATEGORIES` order; unknown categories are refused.
            There is no category for arbitrary paths, environment dumps,
            database exports, or payloads.
        limits: Collection limits; defaults apply when omitted.
    """

    categories: tuple[str, ...] = CATEGORIES
    limits: DiagnosticLimits = field(default_factory=DiagnosticLimits)

    def __post_init__(self) -> None:
        raw = self.categories
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise DiagnosticRequestError("request_invalid")
        if not raw:
            raise DiagnosticRequestError("category_empty")
        if any(item not in CATEGORIES for item in raw):
            raise DiagnosticRequestError("category_unknown")
        if len(set(raw)) != len(raw):
            raise DiagnosticRequestError("category_duplicate")
        if not isinstance(self.limits, DiagnosticLimits):
            raise DiagnosticRequestError("request_invalid")
        ordered = tuple(item for item in CATEGORIES if item in raw)
        object.__setattr__(self, "categories", ordered)


@dataclass(frozen=True)
class DiagnosticSources:
    """Typed sources for the file-backed collectors.

    Paths select the input of one typed collector; they are never copied into
    the bundle, logs, or errors.  With ``ambient_defaults`` the documented
    service defaults are used for unset paths (``OPENMED_CONFIG`` or the default
    configuration file, the profiles directory, ``OPENMED_SERVICE_JOBS_STORE_PATH``
    or the default job store, and the source checkout's registry state).  The
    Journey store and audit chain have no default and must be named.

    Attributes:
        config_path: Local OpenMed configuration file.
        profiles_dir: Directory holding custom ``*.toml`` profiles.
        jobs_store_path: Async job metadata store.
        journey_store_root: Local Journey store root (``journey.sqlite3`` and
            ``artifacts/``).
        registry_state_path: Model registry slot state document.
        audit_chain_path: Tamper-evident audit chain document.
        ambient_defaults: Whether unset paths fall back to service defaults.
    """

    config_path: Path | None = field(default=None, repr=False)
    profiles_dir: Path | None = field(default=None, repr=False)
    jobs_store_path: Path | None = field(default=None, repr=False)
    journey_store_root: Path | None = field(default=None, repr=False)
    registry_state_path: Path | None = field(default=None, repr=False)
    audit_chain_path: Path | None = field(default=None, repr=False)
    ambient_defaults: bool = True

    def resolved_config_path(self) -> Path | None:
        """Return the configuration file the configuration collector reads."""

        if self.config_path is not None or not self.ambient_defaults:
            return self.config_path
        from openmed.core.config import resolve_config_path

        return resolve_config_path()

    def resolved_profiles_dir(self) -> Path | None:
        """Return the custom profile directory the configuration collector lists."""

        if self.profiles_dir is not None or not self.ambient_defaults:
            return self.profiles_dir
        from openmed.core.config import PROFILES_DIR

        return PROFILES_DIR

    def resolved_jobs_store_path(self) -> Path | None:
        """Return the async job metadata store the jobs collector reads."""

        if self.jobs_store_path is not None or not self.ambient_defaults:
            return self.jobs_store_path
        raw = os.environ.get(_JOBS_STORE_ENV_VAR)
        if raw is not None and raw.strip():
            return Path(raw).expanduser()
        return Path.home() / ".cache" / "openmed" / "service-jobs.json"

    def resolved_registry_state_path(self) -> Path | None:
        """Return the registry state the model-registry collector reads."""

        if self.registry_state_path is not None or not self.ambient_defaults:
            return self.registry_state_path
        from openmed.core.registry_slots import REGISTRY_STATE_PATH

        return REGISTRY_STATE_PATH


SectionProvider = Callable[["DiagnosticSection"], Mapping[str, Any]]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class DiagnosticContext:
    """In-process seams for live service evidence and deterministic tests.

    Attributes:
        app: A running service application (any object with ``state``); used
            for readiness, metrics, and runtime evidence.
        runtime: A service runtime; defaults to ``app.state.runtime``.
        metrics: A metrics registry; defaults to ``app.state.metrics``.
        providers: Category -> provider for :data:`PROVIDER_CATEGORIES`.  A
            provider returns a mapping; only the fields in
            :data:`PROVIDER_FIELDS` are kept and every value is subject to the
            value-safety contract.  It must let budget exceptions propagate.
        cancel_event: Set to stop collection; later sections are cancelled.
        monotonic: Monotonic clock in seconds used for budgets.
        wall_clock: Aware UTC clock used for the hour-precision timestamp.
    """

    app: Any = field(default=None, repr=False)
    runtime: Any = field(default=None, repr=False)
    metrics: Any = field(default=None, repr=False)
    providers: Mapping[str, SectionProvider] = field(default_factory=dict, repr=False)
    cancel_event: threading.Event | None = field(default=None, repr=False)
    monotonic: Callable[[], float] = field(default=time.monotonic, repr=False)
    wall_clock: Callable[[], datetime] = field(default=_utc_now, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.providers, Mapping):
            raise DiagnosticRequestError("provider_invalid")
        for category, provider in self.providers.items():
            if category not in PROVIDER_CATEGORIES:
                raise DiagnosticRequestError("provider_category_unsupported")
            if not callable(provider):
                raise DiagnosticRequestError("provider_invalid")

    def _state(self) -> Any:
        return getattr(self.app, "state", None) if self.app is not None else None

    def resolved_runtime(self) -> Any:
        """Return the live runtime, if one was supplied directly or via ``app``."""

        if self.runtime is not None:
            return self.runtime
        return getattr(self._state(), "runtime", None)

    def resolved_metrics(self) -> Any:
        """Return the live metrics registry, if one is available."""

        if self.metrics is not None:
            return self.metrics
        return getattr(self._state(), "metrics", None)


# ---------------------------------------------------------------------------
# Budgets and the per-section collector handle
# ---------------------------------------------------------------------------


class _Signal(Exception):
    """Internal control flow raised by budgets; never carries source text."""

    kind = "limit"

    def __init__(self, code: str) -> None:
        self.code = code if code in PARTIAL_RESULT_CODES else "collector_failed"
        super().__init__(self.code)


class _Interrupted(_Signal):
    kind = "interrupted"


class _LimitReached(_Signal):
    kind = "limit"


class _SourceProblem(_Signal):
    kind = "source"


class _SharedBudget:
    def __init__(
        self,
        limits: DiagnosticLimits,
        monotonic: Callable[[], float],
        cancel_event: threading.Event | None,
    ) -> None:
        self.limits = limits
        self.monotonic = monotonic
        self.cancel_event = cancel_event
        self.deadline = monotonic() + limits.max_elapsed_ms / 1000.0
        self._lock = threading.Lock()
        self.files_opened = 0
        self.bytes_read = 0
        self.dir_entries = 0

    def remaining(self) -> float:
        return self.deadline - self.monotonic()

    def cancelled(self) -> bool:
        return self.cancel_event is not None and self.cancel_event.is_set()

    def open_file(self) -> None:
        with self._lock:
            if self.files_opened >= self.limits.max_files:
                raise _LimitReached("file_limit_reached")
            self.files_opened += 1

    def add_bytes(self, count: int) -> None:
        with self._lock:
            if self.bytes_read + count > self.limits.max_total_bytes:
                raise _LimitReached("total_bytes_limit_reached")
            self.bytes_read += count

    def would_exceed_bytes(self, count: int) -> bool:
        with self._lock:
            return self.bytes_read + count > self.limits.max_total_bytes

    def visit_entry(self) -> None:
        with self._lock:
            if self.dir_entries >= self.limits.max_dir_entries:
                raise _LimitReached("entry_limit_reached")
            self.dir_entries += 1


class DiagnosticSection:
    """Bounded collection handle for one evidence category.

    Collectors and registered providers read sources only through this handle,
    which enforces the request limits, and record controlled codes on it.
    Exceptions raised by these methods must propagate to the collector runner.

    Attributes:
        category: The evidence category being collected.
        data: Mapping that becomes the section's ``data`` after the value-safety
            contract is applied.
    """

    def __init__(self, category: str, shared: _SharedBudget, deadline: float):
        self.category = category
        self.data: dict[str, Any] = {}
        self._shared = shared
        self._deadline = deadline
        self._abort = threading.Event()
        self._lock = threading.Lock()
        self._reasons: set[str] = set()
        self._findings: set[str] = set()
        self._omissions: dict[str, int] = {}
        self._unavailable: str | None = None

    @property
    def limits(self) -> DiagnosticLimits:
        """Return the request limits that apply to this section."""

        return self._shared.limits

    def remaining_seconds(self) -> float:
        """Return the seconds left before this section times out."""

        return min(self._deadline, self._shared.deadline) - self._shared.monotonic()

    def should_stop(self) -> bool:
        """Return whether collection must stop now (timeout or cancellation)."""

        return (
            self._abort.is_set()
            or self._shared.cancelled()
            or self.remaining_seconds() <= 0
        )

    def check(self) -> None:
        """Raise when the section timed out or collection was cancelled."""

        if self._shared.cancelled():
            raise _Interrupted("collection_cancelled")
        if self._abort.is_set() or self.remaining_seconds() <= 0:
            raise _Interrupted("section_timeout")

    def reason(self, code: str) -> None:
        """Record a partial-result code from :data:`PARTIAL_RESULT_CODES`."""

        with self._lock:
            self._reasons.add(
                code if code in PARTIAL_RESULT_CODES else "collector_failed"
            )

    def finding(self, code: str) -> None:
        """Record an observation from :data:`FINDING_CODES`; others are omitted."""

        if code in FINDING_CODES:
            with self._lock:
                self._findings.add(code)
        else:
            self.omit("unknown_code")

    def omit(self, reason: str, count: int = 1) -> None:
        """Count evidence deliberately left out, by :data:`OMISSION_REASONS`."""

        key = reason if reason in OMISSION_REASONS else "unsafe_value"
        if type(count) is int and count > 0:
            with self._lock:
                self._omissions[key] = self._omissions.get(key, 0) + count

    def unavailable(self, code: str) -> None:
        """Mark the section as unavailable with a partial-result code."""

        self.reason(code)
        with self._lock:
            self._unavailable = code

    def read_bytes(self, path: Path) -> bytes:
        """Read one regular file within the file, byte, and time budgets.

        Args:
            path: The typed source file to read.

        Returns:
            The file content.  It must never be copied into section data.
        """

        self.check()
        shared = self._shared
        limits = shared.limits
        shared.open_file()
        try:
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
        except FileNotFoundError:
            raise _SourceProblem("source_absent") from None
        except OSError:
            raise _SourceProblem("source_unreadable") from None
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise _SourceProblem("source_unsafe")
            if info.st_size > limits.max_file_bytes:
                raise _LimitReached("file_too_large")
            if shared.would_exceed_bytes(info.st_size):
                raise _LimitReached("total_bytes_limit_reached")
            chunks: list[bytes] = []
            total = 0
            while True:
                self.check()
                try:
                    chunk = os.read(descriptor, 65_536)
                except OSError:
                    raise _SourceProblem("source_unreadable") from None
                if not chunk:
                    break
                total += len(chunk)
                if total > limits.max_file_bytes:
                    raise _LimitReached("file_too_large")
                shared.add_bytes(len(chunk))
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(descriptor)

    def read_json(self, path: Path) -> Any:
        """Read and parse one JSON source with a fixed nesting bound.

        Args:
            path: The typed source file to read.

        Returns:
            The decoded JSON value.  It must never be copied into section data.
        """

        payload = self.read_bytes(path)
        try:
            document = json.loads(
                payload.decode("utf-8"), parse_constant=_reject_constant
            )
        except (UnicodeDecodeError, ValueError, RecursionError):
            raise _SourceProblem("source_corrupt") from None
        self._check_nesting(document)
        return document

    def _check_nesting(self, document: Any) -> None:
        stack: list[tuple[Any, int]] = [(document, 1)]
        visited = 0
        while stack:
            node, depth = stack.pop()
            visited += 1
            if visited % 4096 == 0:
                self.check()
            if depth > MAX_SOURCE_JSON_NESTING:
                raise _LimitReached("depth_limit_reached")
            if isinstance(node, dict):
                stack.extend((value, depth + 1) for value in node.values())
            elif isinstance(node, list):
                stack.extend((value, depth + 1) for value in node)

    def scan_tree(
        self, root: Path, *, max_depth: int | None = None
    ) -> Iterator[tuple[tuple[str, ...], str]]:
        """Yield ``(relative parts, kind)`` below *root* without reading files.

        Symlinks are reported but never followed.  Directories at the request's
        ``max_depth`` are reported as ``dir_too_deep``, recorded as
        ``depth_limit_reached``, and not entered.  A collector's own structural
        ``max_depth`` stops descent without recording a limit.

        Args:
            root: Directory to scan.
            max_depth: Optional structural depth for this collector.

        Yields:
            Relative name parts and one of ``dir``, ``dir_too_deep``, ``file``,
            ``symlink``, or ``other``.  Names must never enter section data.
        """

        structural = max_depth if max_depth is not None else self.limits.max_depth + 1
        stack: list[tuple[Path, tuple[str, ...]]] = [(root, ())]
        while stack:
            self.check()
            directory, parts = stack.pop()
            try:
                iterator = os.scandir(directory)
            except FileNotFoundError:
                if not parts:
                    raise _SourceProblem("source_absent") from None
                continue
            except OSError:
                raise _SourceProblem("source_unreadable") from None
            with iterator:
                for entry in iterator:
                    self.check()
                    self._shared.visit_entry()
                    relative = (*parts, entry.name)
                    try:
                        if entry.is_symlink():
                            yield relative, "symlink"
                        elif entry.is_dir(follow_symlinks=False):
                            if len(relative) >= structural:
                                yield relative, "dir"
                            elif len(relative) >= self.limits.max_depth:
                                self.reason("depth_limit_reached")
                                yield relative, "dir_too_deep"
                            else:
                                stack.append((Path(entry.path), relative))
                                yield relative, "dir"
                        elif entry.is_file(follow_symlinks=False):
                            yield relative, "file"
                        else:
                            yield relative, "other"
                    except OSError:
                        yield relative, "other"

    def open_sqlite(self, path: Path) -> sqlite3.Connection:
        """Open a SQLite store read-only without creating or changing files.

        When both WAL sidecars already exist (a live or uncleanly stopped
        service), the database is opened with ``mode=ro`` so committed WAL
        frames are visible.  Otherwise it is opened ``immutable`` so no sidecar
        is created.  Long queries are interrupted at the section deadline.
        """

        self.check()
        try:
            info = os.stat(path)
        except FileNotFoundError:
            raise _SourceProblem("source_absent") from None
        except OSError:
            raise _SourceProblem("source_unreadable") from None
        if not stat.S_ISREG(info.st_mode):
            raise _SourceProblem("source_unsafe")
        self._shared.open_file()
        resolved = Path(path).resolve()
        sidecars = all(
            Path(f"{resolved}{suffix}").is_file() for suffix in ("-wal", "-shm")
        )
        query = "mode=ro" if sidecars else "mode=ro&immutable=1"
        try:
            connection = sqlite3.connect(
                f"{resolved.as_uri()}?{query}",
                uri=True,
                timeout=0.25,
                isolation_level=None,
                check_same_thread=False,
            )
        except sqlite3.Error:
            raise _SourceProblem("source_unreadable") from None
        connection.set_progress_handler(lambda: 1 if self.should_stop() else 0, 1_000)
        self.data.setdefault("wal_state", "present" if sidecars else "absent")
        return connection

    def _abort_now(self) -> None:
        self._abort.set()

    def _snapshot(self) -> tuple[set[str], set[str], dict[str, int], str | None]:
        with self._lock:
            return (
                set(self._reasons),
                set(self._findings),
                dict(self._omissions),
                self._unavailable,
            )


def _reject_constant(_: str) -> Any:
    raise ValueError("non-finite JSON constant")


@dataclass(frozen=True)
class SectionResult:
    """Collector result for one evidence category.

    Attributes:
        category: Evidence category.
        status: One of :data:`SECTION_STATUSES`.
        reason_codes: Sorted partial-result codes.
        findings: Sorted observation codes.
        omissions: Omission reason -> count.
        data: Value-free evidence.
    """

    category: str
    status: str
    reason_codes: tuple[str, ...] = ()
    findings: tuple[str, ...] = ()
    omissions: Mapping[str, int] = field(default_factory=dict)
    data: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the entry body without sequence or digest."""

        return {
            "category": self.category,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "findings": list(self.findings),
            "omissions": dict(sorted(self.omissions.items())),
            "data": _plain(self.data),
        }


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class BundleVerification:
    """Offline verification outcome for a diagnostic bundle.

    Attributes:
        valid: Whether the bundle passed every check.
        reason_code: A stable code from :data:`VERIFICATION_REASON_CODES`.
        entry_index: Index of the first failing entry, when applicable.
        entry_count: Number of manifest entries, when readable.
        root_digest: Recomputed manifest root digest, when readable.
    """

    valid: bool
    reason_code: str
    entry_index: int | None = None
    entry_count: int | None = None
    root_digest: str | None = None

    def __bool__(self) -> bool:
        return self.valid

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable, value-free verification summary."""

        return {
            "valid": self.valid,
            "reason_code": self.reason_code,
            "entry_index": self.entry_index,
            "entry_count": self.entry_count,
            "root_digest": self.root_digest,
        }


@dataclass(frozen=True)
class DiagnosticBundle:
    """A collected, self-verified diagnostic bundle.

    Attributes:
        document: The bundle document (format, header, entries, recovery,
            manifest).
        sections: Collector results in canonical order.
    """

    document: Mapping[str, Any] = field(repr=False)
    sections: tuple[SectionResult, ...] = field(repr=False)

    @property
    def root_digest(self) -> str:
        """Return the manifest root digest."""

        return str(self.document["manifest"]["root_digest"])

    @property
    def collection_status(self) -> str:
        """Return ``complete`` when every section is complete, else ``partial``."""

        return str(self.document["header"]["collection"]["status"])

    @property
    def recovery_actions(self) -> tuple[str, ...]:
        """Return the suggested recovery action ids in bundle order."""

        return tuple(item["action"] for item in self.document["recovery"]["actions"])

    def to_bytes(self) -> bytes:
        """Return the canonical, deterministic serialization of the bundle."""

        return _canonical(self.document).encode("ascii") + b"\n"

    def summary(self) -> dict[str, Any]:
        """Return a short, value-free summary for command-line output."""

        counts = self.document["header"]["collection"]["status_counts"]
        return {
            "format": BUNDLE_FORMAT,
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "entry_count": len(self.sections),
            "collection_status": self.collection_status,
            "status_counts": dict(counts),
            "recovery_actions": list(self.recovery_actions),
            "root_digest": self.root_digest,
        }


# ---------------------------------------------------------------------------
# Digests, canonical JSON, and the value-safety contract
# ---------------------------------------------------------------------------


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _digest(domain: bytes, value: Any) -> str:
    payload = domain + _canonical(value).encode("ascii")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def identifier_digest(kind: str, value: str) -> str:
    """Return a domain-separated digest for a public identifier.

    Diagnostic bundles use these digests in place of model repository ids and
    integrity anchors.  They are unkeyed and must never be computed over PHI,
    credentials, or other guessable sensitive values.

    Args:
        kind: A short token naming the identifier family.
        value: The identifier.

    Returns:
        A ``sha256:``-prefixed digest.
    """

    payload = _IDENTIFIER_DOMAIN + kind.encode("ascii") + b"\x00"
    return "sha256:" + hashlib.sha256(payload + value.encode("utf-8")).hexdigest()


def _is_safe_string(value: str) -> bool:
    return any(
        pattern.fullmatch(value) is not None
        for pattern in (_TOKEN_RE, _DIGEST_RE, _VERSION_RE, _SLOT_RE, _TIMESTAMP_RE)
    )


_OMIT = object()


class _CleanState:
    def __init__(self, max_items: int) -> None:
        self.max_items = max_items
        self.omissions: dict[str, int] = {}
        self.truncated = False

    def omit(self, reason: str) -> object:
        self.omissions[reason] = self.omissions.get(reason, 0) + 1
        return _OMIT


def _clean(value: Any, depth: int, state: _CleanState) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if -_MAX_SAFE_INT <= value <= _MAX_SAFE_INT:
            return value
        return state.omit("unsafe_value")
    if isinstance(value, str):
        return value if _is_safe_string(value) else state.omit("unsafe_value")
    if depth >= MAX_DATA_DEPTH:
        return state.omit("unsafe_value")
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key in sorted(value, key=lambda item: (type(item).__name__, str(item))):
            if not isinstance(key, str) or _TOKEN_RE.fullmatch(key) is None:
                state.omit("unsafe_key")
                continue
            item = _clean(value[key], depth + 1, state)
            if item is not _OMIT:
                cleaned[key] = item
        return cleaned
    if isinstance(value, (list, tuple)):
        items = []
        for raw in value:
            item = _clean(raw, depth + 1, state)
            if item is not _OMIT:
                items.append(item)
        if len(items) > state.max_items:
            state.truncated = True
            items = items[: state.max_items]
        return items
    return state.omit("unsafe_value")


def _clean_data(
    data: Mapping[str, Any], max_items: int
) -> tuple[dict[str, Any], dict[str, int], bool]:
    state = _CleanState(max_items)
    cleaned = _clean(data, 0, state)
    if not isinstance(cleaned, dict):
        cleaned = {}
    return cleaned, state.omissions, state.truncated


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Inputs:
    request: DiagnosticRequest
    sources: DiagnosticSources
    context: DiagnosticContext
    environ: Mapping[str, str]


def _collect_runtime(section: DiagnosticSection, inputs: _Inputs) -> None:
    from openmed.__about__ import __version__

    version = sys.version_info
    implementation = sys.implementation.name.lower()
    system = platform.system().lower()
    machine = platform.machine().lower()
    section.data["openmed_version"] = _version_or_omit(section, __version__)
    section.data["python"] = {
        "version": f"{version.major}.{version.minor}.{version.micro}",
        "implementation": (
            implementation if implementation in {"cpython", "pypy"} else "other"
        ),
    }
    section.data["platform"] = {
        "system": system if system in _SYSTEMS else "other",
        "machine": machine if machine in _MACHINES else "other",
    }
    dependencies: dict[str, Any] = {}
    for key, distribution in _DEPENDENCIES:
        section.check()
        try:
            dependencies[key] = _version_or_omit(
                section, importlib.metadata.version(distribution)
            )
        except importlib.metadata.PackageNotFoundError:
            dependencies[key] = "absent"
        except Exception:
            dependencies[key] = "unreadable"
    section.data["dependencies"] = dependencies


def _version_or_omit(section: DiagnosticSection, value: Any) -> str:
    if isinstance(value, str) and _VERSION_RE.fullmatch(value) is not None:
        return value
    section.omit("unsafe_value")
    return "unrecognized"


def _collect_capabilities(section: DiagnosticSection, inputs: _Inputs) -> None:
    from openmed.core.capabilities import backend_status, registered_backends

    backends: dict[str, Any] = {}
    for name in registered_backends():
        section.check()
        status = backend_status(name)
        backends[name] = {"available": bool(status.available), "extra": status.extra}
    section.data["backends"] = backends
    section.data["registered_count"] = len(backends)
    section.data["available_count"] = sum(
        1 for item in backends.values() if item["available"]
    )


def _collect_configuration(section: DiagnosticSection, inputs: _Inputs) -> None:
    from openmed.core.config import PROFILE_PRESETS, OpenMedConfig
    from openmed.core.config_provenance import (
        ConfigurationResolutionError,
        resolve_configuration,
    )

    environ = inputs.environ
    known_keys = {item.name for item in fields(OpenMedConfig)}

    # Profile: only built-in preset names are reported.
    raw_profile = environ.get(_PROFILE_ENV_VAR)
    profile_name = raw_profile or _SERVICE_DEFAULT_PROFILE
    profile: dict[str, Any] = {
        "source_class": "environment" if raw_profile else "default",
    }
    profiles_dir = inputs.sources.resolved_profiles_dir()
    if profile_name in PROFILE_PRESETS:
        profile["class"] = "preset"
        profile["name"] = profile_name
    elif (
        _PROFILE_NAME_RE.fullmatch(profile_name) is not None
        and profiles_dir is not None
        and (profiles_dir / f"{profile_name}.toml").is_file()
    ):
        profile["class"] = "custom_file"
    else:
        profile["class"] = "unresolved"
        section.finding("profile_unresolved")
    section.data["profile"] = profile

    if profiles_dir is not None:
        custom = 0
        try:
            for parts, kind in section.scan_tree(profiles_dir, max_depth=1):
                if len(parts) == 1 and kind == "file" and parts[0].endswith(".toml"):
                    custom += 1
        except _SourceProblem as problem:
            if problem.code != "source_absent":
                section.reason(problem.code)
        except _LimitReached as limit:
            section.reason(limit.code)
        section.data["custom_profile_count"] = custom

    # Configuration file: keys only; values stay in memory for comparison.
    file_values: dict[str, Any] = {}
    config_file: dict[str, Any] = {"state": "not_configured"}
    config_path = inputs.sources.resolved_config_path()
    if config_path is not None:
        try:
            content = section.read_bytes(config_path)
        except _SourceProblem as problem:
            config_file["state"] = (
                "absent" if problem.code == "source_absent" else "unreadable"
            )
            if problem.code != "source_absent":
                section.reason(problem.code)
        except _LimitReached as limit:
            config_file["state"] = "not_read"
            section.reason(limit.code)
        else:
            parsed, unknown = _parse_config_keys(content, known_keys)
            if parsed is None:
                config_file["state"] = "corrupt"
                section.reason("source_corrupt")
                section.finding("config_value_invalid")
            else:
                config_file["state"] = "present"
                file_values = parsed
                if unknown:
                    section.omit("unknown_key", unknown)
                config_file["known_key_count"] = len(parsed)
                config_file["unknown_key_count"] = unknown
                if parsed.get("hf_token"):
                    section.finding("persisted_credential")
    section.data["config_file"] = config_file

    try:
        resolution = resolve_configuration(
            file_config=file_values,
            environment=dict(environ),
            known_keys=known_keys,
        )
    except (ConfigurationResolutionError, TypeError, ValueError):
        section.finding("config_value_invalid")
    else:
        keys: dict[str, Any] = {}
        for entry in resolution.entries:
            if entry.key not in known_keys or entry.sources == ("default",):
                continue
            keys[entry.key] = {
                "source_class": entry.source_class,
                "sources": list(entry.sources),
                # Never compare credential values across sources.
                "conflict_category": (
                    "not_reported"
                    if entry.key in _SECRET_CONFIG_KEYS
                    else entry.conflict_category
                ),
            }
        section.data["keys"] = keys
        section.data["precedence"] = ["default", "file", "environment", "cli"]

    # Service settings: presence by category, never values.
    by_name = {
        f"{_SERVICE_PREFIX}{name}": (category, name.lower())
        for category, names in _SERVICE_SETTINGS.items()
        for name in names
    }
    env_aliases = {f"OPENMED_{key.upper()}" for key in known_keys}
    present: dict[str, list[str]] = {}
    secret_bearing: list[str] = []
    unknown_variables = 0
    for name in sorted(environ):
        if name in by_name:
            category, setting = by_name[name]
            present.setdefault(category, []).append(setting)
            if name.removeprefix(_SERVICE_PREFIX) in _SECRET_BEARING_SETTINGS:
                secret_bearing.append(setting)
        elif name.startswith("OPENMED_") and not (
            name in _CORE_VARIABLES or name in env_aliases
        ):
            unknown_variables += 1
    if unknown_variables:
        section.omit("unknown_variable", unknown_variables)
    section.data["service_settings"] = present
    section.data["secret_bearing_settings_present"] = sorted(secret_bearing)
    section.data["unrecognized_openmed_variable_count"] = unknown_variables


def _parse_config_keys(
    content: bytes, known_keys: set[str]
) -> tuple[dict[str, Any] | None, int]:
    from openmed.core.config import _parse_value, _strip_toml_comment

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return None, 0
    values: dict[str, Any] = {}
    unknown = 0
    for raw_line in text.splitlines():
        line = _strip_toml_comment(raw_line).strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        if key in known_keys:
            values[key] = _parse_value(value.strip())
        else:
            unknown += 1
    return values, unknown


def _collect_health(section: DiagnosticSection, inputs: _Inputs) -> None:
    from openmed.core.doctor import value_free_health_checks

    checks: dict[str, Any] = {}
    for check in value_free_health_checks(inputs.environ):
        section.check()
        name = str(check.get("name", ""))
        if _TOKEN_RE.fullmatch(name) is None:
            section.omit("unknown_key")
            continue
        status = str(check.get("status", "")).lower()
        if status == "fail":
            section.finding("health_check_failed")
        checks[name] = {"status": status, "code": check.get("code")}
    section.data["checks"] = checks


def _service_setting_checks() -> tuple[tuple[str, Callable[[], Any]], ...]:
    from . import runtime as rt
    from .keep_alive import parse_keep_alive
    from .metrics import parse_metrics_enabled
    from .warm_pool import (
        parse_default_model_footprint_bytes,
        parse_max_resident_models,
        parse_memory_admission_wait_seconds,
        parse_model_memory_budget_bytes,
    )

    getenv = os.environ.get

    def models() -> None:
        rt.parse_preload_models(getenv(rt.SERVICE_PRELOAD_ENV_VAR))
        parse_max_resident_models(getenv(rt.SERVICE_MAX_RESIDENT_ENV_VAR))
        parse_model_memory_budget_bytes(getenv(rt.SERVICE_MODEL_MEMORY_BUDGET_ENV_VAR))
        parse_default_model_footprint_bytes(
            getenv(rt.SERVICE_DEFAULT_MODEL_FOOTPRINT_ENV_VAR)
        )
        parse_memory_admission_wait_seconds(
            getenv(rt.SERVICE_MODEL_ADMISSION_WAIT_ENV_VAR)
        )
        parse_keep_alive(getenv(rt.SERVICE_KEEP_ALIVE_ENV_VAR))

    def jobs() -> None:
        from .jobs import (
            SERVICE_JOBS_TTL_ENV_VAR,
            SERVICE_JOBS_WORKERS_ENV_VAR,
            parse_jobs_ttl_seconds,
            parse_jobs_workers,
        )

        parse_jobs_ttl_seconds(getenv(SERVICE_JOBS_TTL_ENV_VAR))
        parse_jobs_workers(getenv(SERVICE_JOBS_WORKERS_ENV_VAR))

    return (
        ("batching", rt.parse_service_batching_config),
        ("coalescing", rt.parse_service_coalescing_config),
        ("jobs", jobs),
        (
            "metrics",
            lambda: parse_metrics_enabled(getenv("OPENMED_SERVICE_METRICS_ENABLED")),
        ),
        ("models", models),
        ("paged_kv_cache", rt.parse_service_mlx_paged_kv_cache_config),
        ("resilience", rt.parse_service_resilience_config),
        (
            "shutdown",
            lambda: rt.parse_shutdown_drain_seconds(
                getenv(rt.SERVICE_SHUTDOWN_DRAIN_ENV_VAR)
            ),
        ),
        ("throttle", rt.parse_service_throttle_config),
    )


def _collect_readiness(section: DiagnosticSection, inputs: _Inputs) -> None:
    state = inputs.context._state()
    if state is None:
        section.reason("not_in_process")
        section.data["process"] = {"state": "not_in_process"}
    else:
        if bool(getattr(state, "shutting_down", False)):
            process_state = "shutting_down"
            section.finding("process_shutting_down")
        elif bool(getattr(state, "ready", False)):
            process_state = "ready"
        else:
            process_state = "not_ready"
            section.finding("process_not_ready")
        inflight = getattr(state, "inflight", 0)
        section.data["process"] = {
            "state": process_state,
            "inflight_requests": inflight if type(inflight) is int else None,
        }

    groups: dict[str, str] = {}
    try:
        checks = _service_setting_checks()
    except ImportError:
        section.data["service_settings"] = {}
        section.reason("source_unreadable")
        return
    for name, check in checks:
        section.check()
        try:
            check()
        except ImportError:
            groups[name] = "unavailable"
        except Exception:
            groups[name] = "invalid"
            section.finding("service_setting_invalid")
        else:
            groups[name] = "valid"
    section.data["service_settings"] = groups


def _collect_operational_events(section: DiagnosticSection, inputs: _Inputs) -> None:
    from .operational import OPERATIONS_BY_CATEGORY, OperationalState

    metrics = inputs.context.resolved_metrics()
    snapshot_method = getattr(metrics, "diagnostic_snapshot", None)
    if metrics is None or not callable(snapshot_method):
        section.unavailable("not_in_process")
        return
    snapshot = snapshot_method()
    if not isinstance(snapshot, Mapping):
        section.reason("source_corrupt")
        return
    states = {item.value for item in OperationalState}
    operations = {
        category.value: allowed for category, allowed in OPERATIONS_BY_CATEGORY.items()
    }
    events: dict[str, Any] = {}
    failures = 0
    raw_operational = snapshot.get("operational", {})
    if isinstance(raw_operational, Mapping):
        for category, by_operation in raw_operational.items():
            if category not in operations or not isinstance(by_operation, Mapping):
                section.omit("unknown_code")
                continue
            for operation, by_state in by_operation.items():
                if operation not in operations[category] or not isinstance(
                    by_state, Mapping
                ):
                    section.omit("unknown_code")
                    continue
                for state_name, count in by_state.items():
                    if state_name not in states or type(count) is not int:
                        section.omit("unknown_code")
                        continue
                    events.setdefault(category, {}).setdefault(operation, {})[
                        state_name
                    ] = count
                    if state_name not in {"success", "empty"}:
                        failures += count
    section.data["operational"] = events
    if failures:
        section.finding("operational_failures_observed")
    counters: dict[str, Any] = {}
    for key in (
        "inflight_requests",
        "orphaned_work",
        "batch_shed_total",
        "admission_shed_total",
        "model_rejection_total",
        "model_load_total",
        "model_eviction_total",
    ):
        value = snapshot.get(key)
        counters[key] = value if type(value) is int else None
    section.data["counters"] = counters
    if (counters.get("batch_shed_total") or 0) + (
        counters.get("admission_shed_total") or 0
    ):
        section.finding("load_shedding_observed")
    statuses = snapshot.get("request_status_classes", {})
    classes: dict[str, int] = {}
    if isinstance(statuses, Mapping):
        for key, value in statuses.items():
            if (
                key
                in {
                    "status_1xx",
                    "status_2xx",
                    "status_3xx",
                    "status_4xx",
                    "status_5xx",
                    "status_other",
                }
                and type(value) is int
            ):
                classes[key] = value
            else:
                section.omit("unknown_code")
    section.data["request_status_classes"] = classes
    if classes.get("status_5xx"):
        section.finding("server_errors_observed")
    breaker = snapshot.get("circuit_breaker", {})
    if isinstance(breaker, Mapping):
        section.data["circuit_breaker"] = {
            key: breaker.get(key) if type(breaker.get(key)) is int else None
            for key in ("closed", "open", "half_open")
        }
        if (section.data["circuit_breaker"].get("open") or 0) > 0:
            section.finding("circuit_breaker_open")


def _collect_jobs(section: DiagnosticSection, inputs: _Inputs) -> None:
    path = inputs.sources.resolved_jobs_store_path()
    if path is None:
        section.unavailable("source_not_configured")
        return
    try:
        document = section.read_json(path)
    except _SourceProblem as problem:
        if problem.code == "source_absent":
            section.unavailable("source_absent")
            return
        if problem.code == "source_corrupt":
            section.finding("job_store_corrupt")
        raise
    jobs = document.get("jobs") if isinstance(document, dict) else None
    if not isinstance(jobs, dict) or any(
        not isinstance(record, dict) for record in jobs.values()
    ):
        section.finding("job_store_corrupt")
        section.reason("source_corrupt")
        return
    raw_version = document.get("schema_version")
    section.data["schema_version"] = raw_version if type(raw_version) is int else None
    by_status = {status: 0 for status in (*_JOB_STATUSES, "other")}
    with_error = 0
    webhook_failed = 0
    for index, record in enumerate(jobs.values()):
        if index % 1024 == 0:
            section.check()
        status = record.get("status")
        by_status[status if status in _JOB_STATUSES else "other"] += 1
        if record.get("error") is not None:
            with_error += 1
        delivery = record.get("webhook_delivery")
        if isinstance(delivery, dict) and delivery.get("success") is False:
            webhook_failed += 1
    if by_status["other"]:
        section.omit("unknown_code", by_status["other"])
    section.data["record_count"] = len(jobs)
    section.data["by_status"] = by_status
    section.data["records_with_error"] = with_error
    section.data["webhook_delivery_failed"] = webhook_failed
    if by_status["failed"]:
        section.finding("jobs_failed_present")
    if by_status["queued"] or by_status["running"]:
        section.finding("jobs_incomplete_present")
    if webhook_failed:
        section.finding("webhook_delivery_failed")


def _journey_tables() -> tuple[frozenset[str], Mapping[int, tuple[str, str]], int]:
    from openmed.structured.store.migrations import (
        LATEST_MIGRATION_VERSION,
        MIGRATIONS,
    )

    pattern = re.compile(r"CREATE TABLE (?:IF NOT EXISTS )?([a-z_]+)")
    tables = frozenset(
        match
        for migration in MIGRATIONS
        for statement in migration.statements
        for match in pattern.findall(statement)
    )
    expected = {item.version: (item.name, item.checksum) for item in MIGRATIONS}
    return tables, expected, int(LATEST_MIGRATION_VERSION)


def _guard_sql(section: DiagnosticSection, query: Callable[[], Any]) -> Any:
    try:
        return query()
    except sqlite3.Error as exc:
        section.check()
        message = str(exc).lower()
        if "not a database" in message or "malformed" in message:
            raise _SourceProblem("source_corrupt") from None
        raise _SourceProblem("source_unreadable") from None


def _collect_journey_store(section: DiagnosticSection, inputs: _Inputs) -> None:
    root = inputs.sources.journey_store_root
    if root is None:
        section.unavailable("source_not_configured")
        return
    database = Path(root) / "journey.sqlite3"
    if not database.exists():
        section.unavailable("source_absent")
        return
    known_tables, expected, latest = _journey_tables()
    section.data["latest_supported_version"] = latest
    try:
        connection = section.open_sqlite(database)
    except _SourceProblem as problem:
        if problem.code == "source_corrupt":
            section.finding("journey_store_corrupt")
        raise
    try:
        _inspect_journey_database(section, connection, known_tables, expected, latest)
    except _SourceProblem as problem:
        if problem.code == "source_corrupt":
            section.finding("journey_store_corrupt")
        section.reason(problem.code)
    finally:
        connection.close()
    _inspect_journey_blobs(section, Path(root) / "artifacts")


def _inspect_journey_database(
    section: DiagnosticSection,
    connection: sqlite3.Connection,
    known_tables: frozenset[str],
    expected: Mapping[int, tuple[str, str]],
    latest: int,
) -> None:
    rows = _guard_sql(
        section,
        lambda: connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall(),
    )
    present = {str(row[0]) for row in rows}
    unknown = sorted(
        name
        for name in present
        if name not in known_tables
        and name != "schema_migrations"
        and not name.startswith("sqlite_")
    )
    if unknown:
        section.omit("unrecognized_entry", len(unknown))
    section.data["unrecognized_table_count"] = len(unknown)

    migrations: list[dict[str, Any]] = []
    if "schema_migrations" not in present:
        section.data["schema_version"] = 0
        section.data["migration_state"] = "uninitialized"
        section.finding("migration_uninitialized")
    else:
        applied = _guard_sql(
            section,
            lambda: connection.execute(
                "SELECT version, name, checksum FROM schema_migrations ORDER BY version"
            ).fetchall(),
        )
        versions: list[int] = []
        mismatch = False
        for version, name, checksum in applied:
            if type(version) is not int:
                section.omit("unsafe_value")
                continue
            versions.append(version)
            if version not in expected:
                state = "unknown_version"
            elif (name, checksum) == expected[version]:
                state = "match"
            else:
                state = "checksum_mismatch"
                mismatch = True
            migrations.append({"version": version, "state": state})
        current = max(versions) if versions else 0
        section.data["schema_version"] = current
        if mismatch:
            migration_state = "checksum_mismatch"
            section.finding("migration_checksum_mismatch")
        elif current > latest:
            migration_state = "ahead"
            section.finding("migration_ahead")
        elif versions != list(range(1, current + 1)):
            migration_state = "gap"
            section.finding("migration_gap")
        elif current < latest:
            migration_state = "pending"
            section.finding("migration_pending")
        else:
            migration_state = "current"
        section.data["migration_state"] = migration_state
    section.data["migrations"] = migrations

    counts: dict[str, int] = {}
    for table in sorted(known_tables & present):
        section.check()
        if re.fullmatch(r"[a-z_]+", table) is None:
            continue
        row = _guard_sql(
            section,
            lambda table=table: connection.execute(
                f'SELECT COUNT(*) FROM "{table}"'
            ).fetchone(),
        )
        counts[table] = int(row[0])
    section.data["record_counts"] = counts

    if {"ingestion_quarantine_results", "ingestion_quarantine_promotions"} <= present:
        row = _guard_sql(
            section,
            lambda: connection.execute(
                "SELECT COUNT(*) FROM ingestion_quarantine_results WHERE "
                "quarantine_id NOT IN (SELECT quarantine_id FROM "
                "ingestion_quarantine_promotions)"
            ).fetchone(),
        )
        unpromoted = int(row[0])
        section.data["quarantine"] = {
            "results": counts.get("ingestion_quarantine_results", 0),
            "promotions": counts.get("ingestion_quarantine_promotions", 0),
            "unpromoted": unpromoted,
        }
        if unpromoted:
            section.finding("quarantined_records_present")

    if "store_revisions" in present:
        row = _guard_sql(
            section,
            lambda: connection.execute(
                "SELECT MAX(revision) FROM store_revisions"
            ).fetchone(),
        )
        section.data["latest_revision"] = row[0] if type(row[0]) is int else None

    result = _guard_sql(
        section,
        lambda: connection.execute("SELECT * FROM pragma_quick_check").fetchall(),
    )
    healthy = len(result) == 1 and str(result[0][0]).lower() == "ok"
    section.data["integrity"] = "ok" if healthy else "failed"
    if not healthy:
        section.finding("integrity_check_failed")


def _inspect_journey_blobs(section: DiagnosticSection, artifacts: Path) -> None:
    blobs = staged = unexpected = 0
    blobs_root = ("blobs", "sha256")
    directories = {"dir", "dir_too_deep"}
    try:
        for parts, kind in section.scan_tree(artifacts, max_depth=4):
            if parts == blobs_root[: len(parts)] and kind in directories:
                continue
            if (
                len(parts) == 3
                and parts[:2] == blobs_root
                and kind in directories
                and _SHARD_NAME_RE.fullmatch(parts[2]) is not None
            ):
                continue
            if len(parts) == 4 and parts[:2] == blobs_root and kind == "file":
                if parts[3].startswith(".journey-blob-"):
                    staged += 1
                    continue
                if (
                    _BLOB_NAME_RE.fullmatch(parts[3]) is not None
                    and parts[3][:2] == parts[2]
                ):
                    blobs += 1
                    continue
            unexpected += 1
    except _SourceProblem as problem:
        if problem.code != "source_absent":
            section.reason(problem.code)
        section.data["blobs"] = {"state": "absent"}
        return
    except _LimitReached as limit:
        section.reason(limit.code)
    if unexpected:
        section.omit("unrecognized_entry", unexpected)
        section.finding("unexpected_store_entries")
    if staged:
        section.finding("staged_blobs_present")
    section.data["blobs"] = {
        "state": "present",
        "blob_count": blobs,
        "staged_count": staged,
        "unexpected_entry_count": unexpected,
    }


def _collect_model_registry(section: DiagnosticSection, inputs: _Inputs) -> None:
    from openmed.core.registry_errors import RegistryError
    from openmed.core.registry_slots import (
        REGISTRY_POINTER_NAMES,
        REGISTRY_STATE_SCHEMA_VERSION,
        pointer_targets,
    )

    path = inputs.sources.resolved_registry_state_path()
    if path is None:
        section.unavailable("source_not_configured")
        return
    try:
        state = section.read_json(path)
    except _SourceProblem as problem:
        if problem.code == "source_absent":
            section.unavailable("source_absent")
            return
        if problem.code == "source_corrupt":
            section.finding("registry_state_invalid")
        raise
    section.data["supported_schema_version"] = REGISTRY_STATE_SCHEMA_VERSION
    raw_version = state.get("schema_version") if isinstance(state, dict) else None
    section.data["schema_version"] = raw_version if type(raw_version) is int else None
    if type(raw_version) is int and raw_version < REGISTRY_STATE_SCHEMA_VERSION:
        section.data["state"] = "migration_required"
        section.finding("registry_migration_required")
        section.reason("schema_unsupported")
        return
    try:
        pointers = pointer_targets(state)
    except (RegistryError, TypeError, ValueError, KeyError, AttributeError):
        section.data["state"] = "invalid"
        section.finding("registry_state_invalid")
        section.reason("source_corrupt")
        return
    section.data["state"] = "valid"
    slots_out: list[dict[str, Any]] = []
    omitted = 0
    for slot, targets in pointers.items():
        section.check()
        entry = state["slots"][slot]
        checkpoints = entry.get("checkpoints", {})
        latest = targets.get("latest")
        item: dict[str, Any] = {
            "slot_digest": identifier_digest("registry_slot", slot),
            "checkpoint_count": len(checkpoints),
            "lineage_count": len(entry.get("lineage", [])),
            "pointers": {
                name: (
                    None
                    if targets.get(name) is None
                    else identifier_digest("model", str(targets[name]))
                )
                for name in REGISTRY_POINTER_NAMES
            },
            "latest_version": (
                checkpoints.get(latest)
                if latest is not None and isinstance(checkpoints.get(latest), str)
                else None
            ),
        }
        if _SLOT_RE.fullmatch(slot) is not None:
            item["slot"] = slot
        else:
            omitted += 1
        slots_out.append(item)
    if omitted:
        section.omit("invalid_identifier", omitted)
    if len(slots_out) > section.limits.max_items:
        section.reason("item_limit_reached")
        slots_out = slots_out[: section.limits.max_items]
    section.data["slot_count"] = len(pointers)
    section.data["slots"] = slots_out


def _collect_model_activation(section: DiagnosticSection, inputs: _Inputs) -> None:
    runtime = inputs.context.resolved_runtime()
    if runtime is not None:
        served = tuple(runtime.served_model_ids())
        policy = runtime.model_policy_report()
        explicit = bool(policy.get("explicit"))
        for code in policy.get("findings", []):
            section.finding(str(code))
        preload = tuple(getattr(runtime, "preload_models", ()) or ())
        try:
            loaded = runtime.loaded_models()
        except Exception:
            loaded = {}
        models = loaded.get("models") if isinstance(loaded, Mapping) else None
        resident = len(models) if isinstance(models, Mapping) else None
        max_resident = getattr(runtime, "max_resident_models", None)
        section.data["source"] = "in_process"
    else:
        section.reason("not_in_process")
        environ = inputs.environ
        explicit = _SERVED_MODELS_ENV_VAR in environ
        served = tuple(
            sorted(
                {
                    item.strip()
                    for item in environ.get(_SERVED_MODELS_ENV_VAR, "").split(",")
                    if item.strip()
                }
            )
        )
        profile = environ.get(_PROFILE_ENV_VAR) or _SERVICE_DEFAULT_PROFILE
        if profile == "prod" and not explicit:
            section.finding("served_models_not_explicit")
        preload = tuple(
            item.strip()
            for item in environ.get(_PRELOAD_MODELS_ENV_VAR, "").split(",")
            if item.strip()
        )
        resident = None
        max_resident = None
        section.data["source"] = "environment"
    digests = sorted(identifier_digest("model", str(item)) for item in served)
    if len(digests) > section.limits.max_items:
        section.reason("item_limit_reached")
        digests = digests[: section.limits.max_items]
    section.data["served_models_explicit"] = explicit
    section.data["served_model_count"] = len(served)
    section.data["served_model_digests"] = digests
    section.data["preload_model_count"] = len(preload)
    section.data["resident_model_count"] = resident
    section.data["max_resident_models"] = (
        max_resident if type(max_resident) is int else None
    )


_CHAIN_REASONS: Final = (
    ("chain is valid", "chain_valid"),
    ("insertion or deletion detected", "sequence_gap"),
    ("insertion detected", "insertion_detected"),
    ("deletion or mutation detected", "head_mismatch"),
    ("deletion detected", "deletion_detected"),
    ("reordering or mutation detected", "link_broken"),
    ("reordering detected", "reordering_detected"),
    ("mutation detected", "mutation_detected"),
)


def _collect_audit_continuity(section: DiagnosticSection, inputs: _Inputs) -> None:
    from openmed.core.audit_chain import CHAIN_VERSION, AuditChain

    path = inputs.sources.audit_chain_path
    if path is None:
        section.unavailable("source_not_configured")
        return
    try:
        document = section.read_json(path)
    except _SourceProblem as problem:
        if problem.code == "source_absent":
            section.unavailable("source_absent")
            return
        if problem.code == "source_corrupt":
            section.finding("audit_chain_invalid")
        raise
    section.data["supported_format_version"] = CHAIN_VERSION
    try:
        if not isinstance(document, dict):
            raise TypeError("audit chain must be an object")
        chain = AuditChain.from_dict(document)
        result = chain.verify()
    except (TypeError, ValueError, KeyError, AttributeError):
        section.data["state"] = "invalid"
        section.data["result"] = "format_invalid"
        section.finding("audit_chain_invalid")
        return
    code = "verification_failed"
    for prefix, mapped in _CHAIN_REASONS:
        if result.reason.startswith(prefix):
            code = mapped
            break
    declared = document.get("entry_count")
    head = document.get("head_hash")
    section.data["state"] = "valid" if result.valid else "broken"
    section.data["result"] = code
    section.data["declared_entry_count"] = declared if type(declared) is int else None
    section.data["checked_entries"] = int(result.checked_entries)
    section.data["failed_entry_index"] = (
        result.entry_index if type(result.entry_index) is int else None
    )
    section.data["head_digest"] = (
        identifier_digest("audit_head", head) if isinstance(head, str) else None
    )
    if not result.valid:
        section.finding("audit_chain_broken")


def _collect_backup(section: DiagnosticSection, inputs: _Inputs) -> None:
    section.unavailable("provider_not_registered")


def _disk_class(path: Path) -> str | None:
    probe = path
    for _ in range(64):
        if probe.exists():
            break
        if probe.parent == probe:
            return None
        probe = probe.parent
    else:
        return None
    try:
        usage = shutil.disk_usage(probe)
    except OSError:
        return None
    ratio = usage.free / usage.total if usage.total else 0.0
    if usage.free < _DISK_CRITICAL_BYTES or ratio < 0.02:
        return "critical"
    if usage.free < _DISK_LOW_BYTES or ratio < 0.10:
        return "low"
    return "ok"


def _collect_disk(section: DiagnosticSection, inputs: _Inputs) -> None:
    sources = inputs.sources
    probes: dict[str, Path | None] = {
        "job_store": sources.resolved_jobs_store_path(),
        "journey_store": sources.journey_store_root,
        "audit_chain": sources.audit_chain_path,
    }
    volumes: dict[str, Any] = {}
    for name, path in sorted(probes.items()):
        section.check()
        if path is None:
            continue
        target = Path(path)
        if name != "journey_store":
            target = target.parent
        disk_class = _disk_class(target)
        volumes[name] = disk_class or "unknown"
        if disk_class == "critical":
            section.finding("disk_space_critical")
        elif disk_class == "low":
            section.finding("disk_space_low")
    if not volumes:
        section.unavailable("source_not_configured")
        return
    section.data["volumes"] = volumes


_COLLECTORS: Final[Mapping[str, Callable[[DiagnosticSection, _Inputs], None]]] = {
    "runtime": _collect_runtime,
    "capabilities": _collect_capabilities,
    "configuration": _collect_configuration,
    "health": _collect_health,
    "readiness": _collect_readiness,
    "operational_events": _collect_operational_events,
    "jobs": _collect_jobs,
    "journey_store": _collect_journey_store,
    "model_registry": _collect_model_registry,
    "model_activation": _collect_model_activation,
    "audit_continuity": _collect_audit_continuity,
    "backup": _collect_backup,
    "disk": _collect_disk,
}


def _provider_collector(
    provider: SectionProvider,
) -> Callable[[DiagnosticSection, _Inputs], None]:
    def collect(section: DiagnosticSection, inputs: _Inputs) -> None:
        result = provider(section)
        if not isinstance(result, Mapping):
            raise TypeError("provider must return a mapping")
        allowed = PROVIDER_FIELDS[section.category]
        written = dict(section.data)
        section.data.clear()
        for key, value in (*written.items(), *result.items()):
            if key in allowed:
                section.data[key] = value
            else:
                section.omit("unknown_key")
        section.data["source"] = "provider"

    return collect


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _interrupted_result(category: str, status: str, code: str) -> SectionResult:
    return SectionResult(category=category, status=status, reason_codes=(code,))


def _run_category(
    category: str, shared: _SharedBudget, inputs: _Inputs
) -> SectionResult:
    if shared.cancelled():
        return _interrupted_result(category, STATUS_CANCELLED, "collection_cancelled")
    remaining = shared.remaining()
    if remaining <= 0:
        return _interrupted_result(category, STATUS_TIMEOUT, "elapsed_budget_exhausted")
    timeout = min(remaining, shared.limits.section_timeout_ms / 1000.0)
    section = DiagnosticSection(category, shared, shared.monotonic() + timeout)
    provider = inputs.context.providers.get(category)
    collector = (
        _provider_collector(provider) if provider is not None else _COLLECTORS[category]
    )
    outcome: dict[str, Any] = {}

    def target() -> None:
        try:
            collector(section, inputs)
        except _Signal as signal:
            outcome["signal"] = signal
        except BaseException:
            # Never let the thread excepthook print a message or traceback.
            outcome["error"] = True

    worker = threading.Thread(
        target=target, name="openmed-diagnostics-section", daemon=True
    )
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        section._abort_now()
        if shared.cancelled():
            return _interrupted_result(
                category, STATUS_CANCELLED, "collection_cancelled"
            )
        return _interrupted_result(category, STATUS_TIMEOUT, "section_timeout")
    if outcome.get("error"):
        return _interrupted_result(category, STATUS_FAILED, "collector_failed")
    signal = outcome.get("signal")
    if isinstance(signal, _Interrupted):
        status = (
            STATUS_CANCELLED
            if signal.code == "collection_cancelled"
            else STATUS_TIMEOUT
        )
        return _interrupted_result(category, status, signal.code)
    if isinstance(signal, _Signal):
        if signal.code == "source_absent" and not section.data:
            section.unavailable(signal.code)
        else:
            section.reason(signal.code)

    reasons, findings, omissions, unavailable = section._snapshot()
    data, cleaned_omissions, truncated = _clean_data(
        section.data, shared.limits.max_items
    )
    for reason, count in cleaned_omissions.items():
        omissions[reason] = omissions.get(reason, 0) + count
    if truncated:
        reasons.add("item_limit_reached")
    if unavailable is not None:
        status = STATUS_UNAVAILABLE
    elif reasons:
        status = STATUS_PARTIAL
    else:
        status = STATUS_COMPLETE
    return SectionResult(
        category=category,
        status=status,
        reason_codes=tuple(sorted(reasons)),
        findings=tuple(sorted(findings)),
        omissions=dict(sorted(omissions.items())),
        data=data,
    )


def _recovery(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    triggers: dict[str, set[str]] = {}
    sections: dict[str, set[str]] = {}
    for entry in entries:
        for code in (*entry["reason_codes"], *entry["findings"]):
            for action in RECOVERY_RULES.get(code, ()):
                triggers.setdefault(action, set()).add(code)
                sections.setdefault(action, set()).add(entry["category"])
    kind_order = {kind: index for index, kind in enumerate(RECOVERY_KINDS)}
    ordered = sorted(
        triggers, key=lambda action: (kind_order[RECOVERY_ACTIONS[action]], action)
    )
    return {
        "actions": [
            {
                "action": action,
                "kind": RECOVERY_ACTIONS[action],
                "triggers": sorted(triggers[action]),
                "sections": sorted(sections[action], key=CATEGORIES.index),
                "doc": f"{DOCS_PAGE}#{action}",
            }
            for action in ordered
        ]
    }


def _hour(moment: datetime) -> str:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:00:00Z")


def _collection_summary(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = {status: 0 for status in SECTION_STATUSES}
    for entry in entries:
        counts[entry["status"]] += 1
    complete = counts[STATUS_COMPLETE] == len(entries)
    return {
        "status": STATUS_COMPLETE if complete else STATUS_PARTIAL,
        "status_counts": counts,
    }


def _manifest(
    header: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
    recovery: Mapping[str, Any],
) -> dict[str, Any]:
    listing = [
        {
            "sequence": entry["sequence"],
            "category": entry["category"],
            "digest": entry["digest"],
        }
        for entry in entries
    ]
    header_digest = _digest(_HEADER_DOMAIN, header)
    recovery_digest = _digest(_RECOVERY_DOMAIN, recovery)
    root = _manifest_root(header_digest, recovery_digest, listing)
    return {
        "entry_count": len(listing),
        "entries": listing,
        "header_digest": header_digest,
        "recovery_digest": recovery_digest,
        "root_digest": root,
    }


def _manifest_root(
    header_digest: str, recovery_digest: str, listing: Sequence[Mapping[str, Any]]
) -> str:
    return _digest(
        _MANIFEST_DOMAIN,
        {
            "format": BUNDLE_FORMAT,
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "header_digest": header_digest,
            "recovery_digest": recovery_digest,
            "entries": list(listing),
        },
    )


def _entry(sequence: int, result: SectionResult) -> dict[str, Any]:
    body = {"sequence": sequence, **result.to_dict()}
    body["digest"] = _digest(_ENTRY_DOMAIN, body)
    return body


def _assemble(
    results: Sequence[SectionResult],
    request: DiagnosticRequest,
    generated_at: str,
) -> dict[str, Any]:
    from openmed.__about__ import __version__

    entries = [_entry(index, result) for index, result in enumerate(results)]
    generator_version = (
        __version__ if _VERSION_RE.fullmatch(__version__) is not None else "0.0.0"
    )
    header = {
        "generator": {"name": "openmed", "version": generator_version},
        "generated_at": generated_at,
        "timestamp_precision_seconds": TIMESTAMP_PRECISION_SECONDS,
        "categories": list(request.categories),
        "limits": request.limits.to_dict(),
        "collection": _collection_summary(entries),
        "restorable_state": False,
    }
    recovery = _recovery(entries)
    return {
        "format": BUNDLE_FORMAT,
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "header": header,
        "entries": entries,
        "recovery": recovery,
        "manifest": _manifest(header, entries, recovery),
    }


def _shrink(result: SectionResult) -> SectionResult:
    reasons = tuple(sorted({*result.reason_codes, "section_too_large"}))
    status = result.status
    if status == STATUS_COMPLETE:
        status = STATUS_PARTIAL
    return SectionResult(
        category=result.category,
        status=status,
        reason_codes=reasons,
        findings=result.findings,
        omissions=result.omissions,
        data={},
    )


def collect_diagnostic_bundle(
    request: DiagnosticRequest | None = None,
    *,
    sources: DiagnosticSources | None = None,
    context: DiagnosticContext | None = None,
) -> DiagnosticBundle:
    """Collect a bounded, value-free diagnostic and recovery bundle.

    Collection runs one typed collector per requested category, sequentially,
    each under the section and total time budgets.  A collector that exceeds
    its budget is abandoned and reported with a stable partial-result code; the
    remaining categories still run while time remains.  No collector opens a
    network connection or writes to the inspected system, and nothing is
    logged.

    Args:
        request: Categories and limits.  Defaults collect every category.
        sources: Typed file-backed sources.  Defaults use service defaults.
        context: Live in-process seams, providers, cancellation, and clocks.

    Returns:
        The collected bundle, already verified with
        :func:`verify_diagnostic_bundle`.

    Raises:
        DiagnosticRequestError: If the request, sources, or context are invalid.
    """

    request = DiagnosticRequest() if request is None else request
    sources = DiagnosticSources() if sources is None else sources
    context = DiagnosticContext() if context is None else context
    if not isinstance(request, DiagnosticRequest):
        raise DiagnosticRequestError("request_invalid")
    if not isinstance(sources, DiagnosticSources):
        raise DiagnosticRequestError("request_invalid")
    if not isinstance(context, DiagnosticContext):
        raise DiagnosticRequestError("request_invalid")

    shared = _SharedBudget(request.limits, context.monotonic, context.cancel_event)
    inputs = _Inputs(
        request=request,
        sources=sources,
        context=context,
        environ=dict(os.environ),
    )
    results = [
        _run_category(category, shared, inputs) for category in request.categories
    ]
    generated_at = _hour(context.wall_clock())
    document = _assemble(results, request, generated_at)
    limit = request.limits.max_bundle_bytes
    while len(_canonical(document)) + 1 > limit:
        candidates = [
            (len(_canonical(result.data)), -index, index)
            for index, result in enumerate(results)
            if result.data
        ]
        if not candidates:
            raise DiagnosticRequestError("limit_below_minimum")
        _, _, index = max(candidates)
        results[index] = _shrink(results[index])
        document = _assemble(results, request, generated_at)
    verification = verify_diagnostic_bundle(document)
    if not verification.valid:  # pragma: no cover - defensive self-check
        raise RuntimeError("diagnostic bundle failed self-verification")
    return DiagnosticBundle(document=document, sections=tuple(results))


# ---------------------------------------------------------------------------
# Offline verification
# ---------------------------------------------------------------------------

_TOP_LEVEL_KEYS: Final = frozenset(
    {"format", "schema_version", "header", "entries", "recovery", "manifest"}
)
_HEADER_KEYS: Final = frozenset(
    {
        "generator",
        "generated_at",
        "timestamp_precision_seconds",
        "categories",
        "limits",
        "collection",
        "restorable_state",
    }
)
_ENTRY_KEYS: Final = frozenset(
    {
        "sequence",
        "category",
        "status",
        "reason_codes",
        "findings",
        "omissions",
        "data",
        "digest",
    }
)
_MANIFEST_KEYS: Final = frozenset(
    {"entry_count", "entries", "header_digest", "recovery_digest", "root_digest"}
)
_MAX_VERIFY_BYTES: Final = _LIMIT_BOUNDS["max_bundle_bytes"][1]


def _fail(
    code: str,
    *,
    index: int | None = None,
    count: int | None = None,
    root: str | None = None,
) -> BundleVerification:
    return BundleVerification(
        False, code, entry_index=index, entry_count=count, root_digest=root
    )


def _valid_header(header: Any) -> bool:
    if not isinstance(header, dict) or set(header) != _HEADER_KEYS:
        return False
    generator = header["generator"]
    if (
        not isinstance(generator, dict)
        or set(generator) != {"name", "version"}
        or generator["name"] != "openmed"
        or not isinstance(generator["version"], str)
        or _VERSION_RE.fullmatch(generator["version"]) is None
    ):
        return False
    if (
        not isinstance(header["generated_at"], str)
        or _TIMESTAMP_RE.fullmatch(header["generated_at"]) is None
        or header["timestamp_precision_seconds"] != TIMESTAMP_PRECISION_SECONDS
        or header["restorable_state"] is not False
    ):
        return False
    categories = header["categories"]
    if not isinstance(categories, list) or not categories:
        return False
    if [item for item in CATEGORIES if item in categories] != categories:
        return False
    limits = header["limits"]
    if not isinstance(limits, dict) or set(limits) != set(_LIMIT_BOUNDS):
        return False
    try:
        DiagnosticLimits(**limits)
    except DiagnosticRequestError:
        return False
    collection = header["collection"]
    if (
        not isinstance(collection, dict)
        or set(collection) != {"status", "status_counts"}
        or collection["status"] not in {STATUS_COMPLETE, STATUS_PARTIAL}
        or not isinstance(collection["status_counts"], dict)
        or set(collection["status_counts"]) != set(SECTION_STATUSES)
        or any(type(value) is not int for value in collection["status_counts"].values())
    ):
        return False
    return True


def _valid_entry_shape(entry: Any) -> bool:
    if not isinstance(entry, dict) or set(entry) != _ENTRY_KEYS:
        return False
    if type(entry["sequence"]) is not int or entry["category"] not in CATEGORIES:
        return False
    if entry["status"] not in SECTION_STATUSES:
        return False
    for key, vocabulary in (
        ("reason_codes", PARTIAL_RESULT_CODES),
        ("findings", FINDING_CODES),
    ):
        values = entry[key]
        if (
            not isinstance(values, list)
            or any(item not in vocabulary for item in values)
            or values != sorted(set(values))
        ):
            return False
    omissions = entry["omissions"]
    if not isinstance(omissions, dict) or any(
        key not in OMISSION_REASONS or type(value) is not int or value < 1
        for key, value in omissions.items()
    ):
        return False
    if not isinstance(entry["data"], dict):
        return False
    return (
        isinstance(entry["digest"], str)
        and _DIGEST_RE.fullmatch(entry["digest"]) is not None
    )


def _valid_manifest_shape(manifest: Any) -> bool:
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        return False
    listing = manifest["entries"]
    if not isinstance(listing, list) or manifest["entry_count"] != len(listing):
        return False
    for key in ("header_digest", "recovery_digest", "root_digest"):
        value = manifest[key]
        if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
            return False
    for item in listing:
        if (
            not isinstance(item, dict)
            or set(item) != {"sequence", "category", "digest"}
            or type(item["sequence"]) is not int
            or item["category"] not in CATEGORIES
            or not isinstance(item["digest"], str)
            or _DIGEST_RE.fullmatch(item["digest"]) is None
        ):
            return False
    return True


def _entry_digest(entry: Mapping[str, Any]) -> str:
    return _digest(_ENTRY_DOMAIN, {key: entry[key] for key in entry if key != "digest"})


def verify_diagnostic_bundle(
    bundle: Mapping[str, Any] | bytes | str,
) -> BundleVerification:
    """Verify a diagnostic bundle offline.

    The verifier checks the schema, the header and recovery digests, the
    manifest root, the order and digest of every entry, the value-safety
    contract of every section, and that recovery guidance matches the entries.
    It detects missing, added, reordered, and modified entries.  The digests are
    unkeyed integrity checks, not signatures.

    Args:
        bundle: A bundle document, or its serialized JSON bytes or text.

    Returns:
        A :class:`BundleVerification` with a stable reason code.
    """

    if isinstance(bundle, (bytes, str)):
        size = len(bundle.encode("utf-8") if isinstance(bundle, str) else bundle)
        if size > _MAX_VERIFY_BYTES + 1:
            return _fail("bundle_too_large")
        try:
            document = json.loads(bundle, parse_constant=_reject_constant)
        except (ValueError, RecursionError, UnicodeDecodeError):
            return _fail("bundle_not_json")
    else:
        document = bundle
    if not isinstance(document, Mapping):
        return _fail("bundle_not_object")
    document = dict(document)
    keys = set(document)
    if keys - _TOP_LEVEL_KEYS:
        return _fail("unexpected_field")
    if _TOP_LEVEL_KEYS - keys:
        return _fail("missing_field")
    if document["format"] != BUNDLE_FORMAT:
        return _fail("format_unsupported")
    if document["schema_version"] != BUNDLE_SCHEMA_VERSION:
        return _fail("schema_version_unsupported")
    header = document["header"]
    if not _valid_header(header):
        return _fail("header_invalid")
    manifest = document["manifest"]
    if not _valid_manifest_shape(manifest):
        return _fail("manifest_invalid")
    listing = manifest["entries"]
    count = len(listing)
    root = _manifest_root(
        manifest["header_digest"], manifest["recovery_digest"], listing
    )
    if root != manifest["root_digest"]:
        return _fail("manifest_modified", count=count, root=root)
    if _digest(_HEADER_DOMAIN, header) != manifest["header_digest"]:
        return _fail("header_modified", count=count, root=root)
    recovery = document["recovery"]
    if (
        not isinstance(recovery, dict)
        or set(recovery) != {"actions"}
        or not isinstance(recovery["actions"], list)
    ):
        return _fail("recovery_invalid", count=count, root=root)
    if _digest(_RECOVERY_DOMAIN, recovery) != manifest["recovery_digest"]:
        return _fail("recovery_inconsistent", count=count, root=root)
    if [item["sequence"] for item in listing] != list(range(count)):
        return _fail("manifest_invalid", count=count, root=root)
    if [item["category"] for item in listing] != header["categories"]:
        return _fail("manifest_invalid", count=count, root=root)

    entries = document["entries"]
    if not isinstance(entries, list):
        return _fail("entry_invalid", count=count, root=root)
    for index, entry in enumerate(entries):
        if not _valid_entry_shape(entry):
            return _fail("entry_invalid", index=index, count=count, root=root)
    if len(entries) < count:
        return _fail("entry_missing", count=count, root=root)
    if len(entries) > count:
        return _fail("entry_added", count=count, root=root)
    expected_digests = [item["digest"] for item in listing]
    actual = [_entry_digest(entry) for entry in entries]
    for index, entry in enumerate(entries):
        listed = listing[index]
        if (
            entry["sequence"] == listed["sequence"]
            and entry["category"] == listed["category"]
            and actual[index] == entry["digest"] == listed["digest"]
        ):
            continue
        if (
            entry["category"] != listed["category"]
            or entry["sequence"] != index
            or entry["digest"] != listed["digest"]
        ) and sorted(actual) == sorted(expected_digests):
            return _fail("entry_reordered", index=index, count=count, root=root)
        return _fail("entry_modified", index=index, count=count, root=root)
    max_items = header["limits"]["max_items"]
    for index, entry in enumerate(entries):
        cleaned, omissions, truncated = _clean_data(entry["data"], max_items)
        if cleaned != entry["data"] or omissions or truncated:
            return _fail("unsafe_content", index=index, count=count, root=root)
    if _recovery(entries) != recovery:
        return _fail("recovery_inconsistent", count=count, root=root)
    if _collection_summary(entries) != header["collection"]:
        return _fail("header_invalid", count=count, root=root)
    return BundleVerification(True, "bundle_valid", entry_count=count, root_digest=root)


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------


def write_diagnostic_bundle(
    bundle: DiagnosticBundle, path: str | Path, *, overwrite: bool = False
) -> None:
    """Atomically write a bundle as an owner-only (0600) file.

    Args:
        bundle: The bundle to write.
        path: Destination file.  Its parent directory must exist.
        overwrite: Replace an existing file instead of refusing.

    Raises:
        DiagnosticRequestError: ``output_exists`` when the file exists and
            ``overwrite`` is false, or ``output_unwritable`` on any write error.
    """

    destination = Path(path)
    payload = bundle.to_bytes()
    try:
        descriptor, temp_name = tempfile.mkstemp(
            prefix=".openmed-diagnostics-", dir=destination.parent
        )
    except OSError:
        raise DiagnosticRequestError("output_unwritable") from None
    try:
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError:
            raise DiagnosticRequestError("output_unwritable") from None
        try:
            if overwrite:
                os.replace(temp_name, destination)
            else:
                os.link(temp_name, destination)
        except FileExistsError:
            raise DiagnosticRequestError("output_exists") from None
        except OSError:
            raise DiagnosticRequestError("output_unwritable") from None
    finally:
        try:
            os.unlink(temp_name)
        except OSError:
            pass


def read_diagnostic_bundle(path: str | Path) -> bytes:
    """Read a bundle file for verification without exceeding the size bound.

    Args:
        path: The bundle file.

    Returns:
        The file bytes, or at most one byte more than the maximum bundle size so
        :func:`verify_diagnostic_bundle` reports ``bundle_too_large``.

    Raises:
        OSError: If the file cannot be opened or is not a regular file.
    """

    descriptor = os.open(Path(path), os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("bundle source is not a regular file")
        chunks: list[bytes] = []
        total = 0
        while total <= _MAX_VERIFY_BYTES + 1:
            chunk = os.read(descriptor, 65_536)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        return b"".join(chunks)[: _MAX_VERIFY_BYTES + 2]
    finally:
        os.close(descriptor)


__all__ = [
    "BUNDLE_FORMAT",
    "BUNDLE_SCHEMA_VERSION",
    "BundleVerification",
    "CATEGORIES",
    "DOCS_PAGE",
    "DiagnosticBundle",
    "DiagnosticContext",
    "DiagnosticLimits",
    "DiagnosticRequest",
    "DiagnosticRequestError",
    "DiagnosticSection",
    "DiagnosticSources",
    "FINDING_CODES",
    "MAX_DATA_DEPTH",
    "MAX_SOURCE_JSON_NESTING",
    "OMISSION_REASONS",
    "PARTIAL_RESULT_CODES",
    "PROVIDER_CATEGORIES",
    "PROVIDER_FIELDS",
    "RECOVERY_ACTIONS",
    "RECOVERY_KINDS",
    "RECOVERY_RULES",
    "REQUEST_ERROR_CODES",
    "SECTION_STATUSES",
    "SectionProvider",
    "SectionResult",
    "TIMESTAMP_PRECISION_SECONDS",
    "VERIFICATION_REASON_CODES",
    "collect_diagnostic_bundle",
    "identifier_digest",
    "read_diagnostic_bundle",
    "verify_diagnostic_bundle",
    "write_diagnostic_bundle",
]
