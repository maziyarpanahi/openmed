"""Deterministic execution plans for signed portable model bundles.

The resolver turns three inputs into one value-free result:

* a portable bundle that the caller already verified with
  :func:`openmed.core.portable_bundle.verify_portable_bundle`,
* a normalized :class:`~openmed.core.device_capabilities.DeviceCapabilityProfile`,
* caller :class:`ExecutionConstraints` (privacy, memory, precision, energy,
  sequence length, evidence, revocation, step bound, and deadline).

The result is either an ordered plan of ``(runtime, provider)`` steps taken
only from the bundle's signed fallback graph, or an explicit ``unsupported``
result with a reason code. There is no cloud, remote, or network step, and no
step that the signed bundle did not declare: when nothing qualifies, the
resolver says so instead of substituting a different model or service.

Swift, Kotlin, and TypeScript kits implement the same rules and must produce
byte-identical plans for the golden vectors in
``tests/fixtures/device_capabilities/``. Resolution is local and pure: it never
opens a network connection, never loads or runs a model, and reports only
controlled codes, runtime identifiers, counts, and digests.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from importlib import resources
from typing import Any

from .device_capabilities import (
    MAX_OPERATOR_SET,
    MAX_SEQUENCE_LENGTH,
    MODEL_TIERS,
    PLATFORM_PROVIDERS,
    PROVIDER_ACCELERATORS,
    PROVIDERS,
    DeviceCapabilityProfile,
    model_tier_rank,
)
from .portable_bundle import (
    MAX_SAFE_INTEGER,
    PRECISIONS,
    PortableBundleError,
    PortableBundleManifest,
    PortableBundleVerification,
    canonical_json_bytes,
    content_id,
)

EXECUTION_PLAN_SCHEMA = "openmed.execution_plan"
EXECUTION_PLAN_SCHEMA_VERSION = 1
EXECUTION_CONSTRAINTS_SCHEMA = "openmed.execution_constraints"
EXECUTION_CONSTRAINTS_SCHEMA_VERSION = 1
RESOLVER_VERSION = 1
EXECUTION_EXTENSION = "org.openmed.execution"
"""Manifest extension that declares providers and qualifications per runtime."""
EXECUTION_EXTENSION_VERSION = 1

FORMAT_PROVIDERS: Mapping[str, frozenset[str]] = {
    "coreml": frozenset({"coreml", "cpu"}),
    "mlx": frozenset({"cpu", "metal"}),
    "onnx": frozenset({"coreml", "cpu", "nnapi", "qnn", "wasm", "webgpu"}),
    "ort": frozenset({"coreml", "cpu", "nnapi", "qnn"}),
    "pytorch": frozenset({"cpu"}),
    "transformersjs": frozenset({"wasm", "webgpu"}),
}
"""Providers that can execute each bundle runtime format."""

DEFAULT_PROVIDERS: Mapping[str, tuple[str, ...]] = {
    "coreml": ("coreml",),
    "mlx": ("metal",),
    "onnx": ("cpu", "wasm"),
    "ort": ("cpu",),
    "pytorch": ("cpu",),
    "transformersjs": ("wasm",),
}
"""Baseline providers for a runtime the execution extension does not describe.

Only the providers valid on the profile platform are used, so ``onnx`` means
``wasm`` on the web and ``cpu`` everywhere else; a default never adds an
accelerator or a fallback step.
"""

ENERGY_POSTURES = frozenset({"balanced", "low_power"})
DEFAULT_MAX_STEPS = 4
MAX_STEPS_LIMIT = 16
MAX_DEADLINE_MS = 86_400_000
SCHEMA_PACKAGE = "openmed.core.schemas.json"
SCHEMA_NAMES = (
    "device_capability_profile",
    "execution_constraints",
    "execution_extension",
    "execution_plan",
)


class ExecutionPlanReason(str, Enum):
    """Stable, value-free reason codes, in evaluation order.

    Plan-level codes come first, then runtime-level codes in the order each
    runtime is checked, then provider-level codes, then the two codes for
    qualified steps that the plan does not use.
    """

    REMOTE_EXECUTION_UNSUPPORTED = "remote_execution_unsupported"
    BUNDLE_REVOKED = "bundle_revoked"
    BUNDLE_UNVERIFIED = "bundle_unverified"
    BUNDLE_FILES_UNVERIFIED = "bundle_files_unverified"
    SDK_INCOMPATIBLE = "sdk_incompatible"
    EXECUTION_METADATA_INVALID = "execution_metadata_invalid"
    NO_ELIGIBLE_RUNTIME = "no_eligible_runtime"
    RUNTIME_NOT_ELIGIBLE = "runtime_not_eligible"
    ARTIFACT_REVOKED = "artifact_revoked"
    FORMAT_UNSUPPORTED = "format_unsupported"
    PLATFORM_UNSUPPORTED = "platform_unsupported"
    OS_VERSION_UNKNOWN = "os_version_unknown"
    OS_VERSION_UNSUPPORTED = "os_version_unsupported"
    PRECISION_UNSUPPORTED = "precision_unsupported"
    PRECISION_DISALLOWED = "precision_disallowed"
    MEMORY_INSUFFICIENT = "memory_insufficient"
    STORAGE_INSUFFICIENT = "storage_insufficient"
    MODEL_TIER_EXCEEDED = "model_tier_exceeded"
    SEQUENCE_LIMIT_EXCEEDED = "sequence_limit_exceeded"
    SEQUENCE_LIMIT_UNDECLARED = "sequence_limit_undeclared"
    OPERATOR_SET_UNKNOWN = "operator_set_unknown"
    OPERATOR_SET_UNSUPPORTED = "operator_set_unsupported"
    FEATURE_UNAVAILABLE = "feature_unavailable"
    EVIDENCE_MISSING = "evidence_missing"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_DISALLOWED = "provider_disallowed"
    ENERGY_POLICY_EXCLUDED = "energy_policy_excluded"
    NOT_DECLARED_FALLBACK = "not_declared_fallback"
    FALLBACK_BOUND_REACHED = "fallback_bound_reached"


REASON_CODES: tuple[str, ...] = tuple(reason.value for reason in ExecutionPlanReason)
"""Every plan reason code, in evaluation order."""

STEP_ROLES: tuple[str, ...] = ("selected", "provider_fallback", "artifact_fallback")
ASSUMPTIONS: tuple[str, ...] = (
    "device_sequence_limit_unknown",
    "memory_budget_unknown",
    "model_tier_limit_unknown",
    "storage_budget_unknown",
)
"""Limits a plan could not check because the profile does not know them."""

INVALIDATION_COMPONENTS: tuple[str, ...] = (
    "bundle",
    "constraints",
    "profile",
    "resolver",
    "verification",
)


class ExecutionConstraintsError(ValueError):
    """Raised for malformed caller constraints; the message is a code only."""

    def __init__(self) -> None:
        """Use the controlled ``constraints_invalid`` code as the message."""

        super().__init__("constraints_invalid")


_R = ExecutionPlanReason
_TOKEN = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_CONSTRAINT_KEYS = frozenset(
    {
        "allowed_precisions",
        "deadline_ms",
        "disallowed_providers",
        "energy",
        "local_only",
        "max_memory_bytes",
        "max_steps",
        "min_sequence_length",
        "require_verified_files",
        "required_evidence",
        "revoked_artifacts",
        "schema",
        "schema_version",
    }
)
_EXTENSION_KEYS = frozenset({"runtimes", "schema_version"})
_RUNTIME_META_KEYS = frozenset(
    {
        "evidence",
        "features",
        "max_sequence_length",
        "min_operator_set",
        "providers",
        "tier",
    }
)


def _is_uint(value: object, maximum: int = MAX_SAFE_INTEGER) -> bool:
    return type(value) is int and 0 <= value <= maximum


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _string_set(value: object, valid: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise ExecutionConstraintsError()
    items = list(value)
    if not all(isinstance(item, str) and valid(item) for item in items):
        raise ExecutionConstraintsError()
    return tuple(sorted(set(items)))


@dataclass(frozen=True)
class ExecutionConstraints:
    """Caller policy applied on top of the bundle and the device profile.

    Args:
        local_only: Must be ``True``. Version 1 plans never contain a remote or
            cloud step, so ``False`` resolves to
            ``remote_execution_unsupported`` instead of a silent fallback.
        allowed_precisions: Precisions a step may use, or ``None`` for any.
        disallowed_providers: Providers no step may use.
        max_memory_bytes: Memory ceiling for one step, or ``None``.
        min_sequence_length: Token length every step must support, or
            ``None``.
        energy: ``balanced`` or ``low_power``; ``low_power`` excludes GPU
            providers.
        required_evidence: Evidence kinds that must qualify every step.
        revoked_artifacts: ``sha256:`` file digests no step may use.
        max_steps: Upper bound on plan steps (1 to 16).
        deadline_ms: Wall-clock budget the executor must apply across steps,
            or ``None``. The resolver records it and never benchmarks.
        require_verified_files: Whether the bundle file tree must have been
            verified, not only the manifest and signatures.
    """

    local_only: bool = True
    allowed_precisions: tuple[str, ...] | None = None
    disallowed_providers: tuple[str, ...] = ()
    max_memory_bytes: int | None = None
    min_sequence_length: int | None = None
    energy: str = "balanced"
    required_evidence: tuple[str, ...] = ()
    revoked_artifacts: tuple[str, ...] = ()
    max_steps: int = DEFAULT_MAX_STEPS
    deadline_ms: int | None = None
    require_verified_files: bool = True

    def __post_init__(self) -> None:
        """Validate and normalize every member."""

        if (
            not isinstance(self.local_only, bool)
            or not isinstance(self.require_verified_files, bool)
            or self.energy not in ENERGY_POSTURES
            or (
                self.max_memory_bytes is not None
                and not _is_uint(self.max_memory_bytes)
            )
            or (
                self.min_sequence_length is not None
                and not (
                    _is_uint(self.min_sequence_length, MAX_SEQUENCE_LENGTH)
                    and self.min_sequence_length > 0
                )
            )
            or not _is_uint(self.max_steps, MAX_STEPS_LIMIT)
            or self.max_steps < 1
            or (
                self.deadline_ms is not None
                and not (
                    _is_uint(self.deadline_ms, MAX_DEADLINE_MS) and self.deadline_ms > 0
                )
            )
        ):
            raise ExecutionConstraintsError()
        if self.allowed_precisions is not None:
            object.__setattr__(
                self,
                "allowed_precisions",
                _string_set(self.allowed_precisions, PRECISIONS.__contains__),
            )
        object.__setattr__(
            self,
            "disallowed_providers",
            _string_set(self.disallowed_providers, PROVIDERS.__contains__),
        )
        object.__setattr__(
            self,
            "required_evidence",
            _string_set(self.required_evidence, lambda item: _TOKEN.fullmatch(item)),
        )
        object.__setattr__(
            self,
            "revoked_artifacts",
            _string_set(self.revoked_artifacts, lambda item: _SHA256.fullmatch(item)),
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ExecutionConstraints:
        """Parse a constraints document (``execution_constraints``).

        Args:
            data: Constraints mapping with every member present.

        Returns:
            The validated constraints.

        Raises:
            ExecutionConstraintsError: For unknown or missing members, a
                foreign schema, or invalid values.
        """

        if (
            not isinstance(data, Mapping)
            or set(data) != _CONSTRAINT_KEYS
            or data["schema"] != EXECUTION_CONSTRAINTS_SCHEMA
            or type(data["schema_version"]) is not int
            or data["schema_version"] != EXECUTION_CONSTRAINTS_SCHEMA_VERSION
        ):
            raise ExecutionConstraintsError()
        members = {
            key: data[key] for key in _CONSTRAINT_KEYS - {"schema", "schema_version"}
        }
        return cls(**members)

    def to_dict(self) -> dict[str, Any]:
        """Return the constraints document (``execution_constraints``)."""

        return {
            "allowed_precisions": (
                None
                if self.allowed_precisions is None
                else list(self.allowed_precisions)
            ),
            "deadline_ms": self.deadline_ms,
            "disallowed_providers": list(self.disallowed_providers),
            "energy": self.energy,
            "local_only": self.local_only,
            "max_memory_bytes": self.max_memory_bytes,
            "max_steps": self.max_steps,
            "min_sequence_length": self.min_sequence_length,
            "require_verified_files": self.require_verified_files,
            "required_evidence": list(self.required_evidence),
            "revoked_artifacts": list(self.revoked_artifacts),
            "schema": EXECUTION_CONSTRAINTS_SCHEMA,
            "schema_version": EXECUTION_CONSTRAINTS_SCHEMA_VERSION,
        }

    def digest(self) -> str:
        """Return the ``sha256:`` digest of the canonical constraints document."""

        return _digest(self.to_dict())


@dataclass(frozen=True)
class PlanStep:
    """One ``(runtime, provider)`` attempt in an execution plan.

    Attributes:
        runtime_id: Bundle runtime identifier.
        runtime_index: Index of the runtime in the manifest.
        format: Runtime format.
        precision: Runtime precision.
        provider: Normalized execution provider.
        accelerator: Accelerator kind behind ``provider``.
        artifact_sha256: Digest of the runtime entrypoint file.
        min_memory_bytes: Declared memory requirement.
        max_sequence_length: Declared sequence limit, or ``None``.
        tier: Declared model tier, or ``None``.
        evidence: Evidence kinds that qualify the runtime; the executor must
            only rely on them once the file tree is verified.
        role: ``selected``, ``provider_fallback``, or ``artifact_fallback``.
    """

    runtime_id: str
    runtime_index: int
    format: str
    precision: str
    provider: str
    accelerator: str
    artifact_sha256: str
    min_memory_bytes: int
    max_sequence_length: int | None
    tier: str | None
    evidence: tuple[str, ...]
    role: str

    def to_dict(self) -> dict[str, Any]:
        """Return the step document."""

        return {
            "accelerator": self.accelerator,
            "artifact_sha256": self.artifact_sha256,
            "evidence": list(self.evidence),
            "format": self.format,
            "max_sequence_length": self.max_sequence_length,
            "min_memory_bytes": self.min_memory_bytes,
            "precision": self.precision,
            "provider": self.provider,
            "role": self.role,
            "runtime_id": self.runtime_id,
            "runtime_index": self.runtime_index,
            "tier": self.tier,
        }


@dataclass(frozen=True)
class PlanSkip:
    """A runtime or ``(runtime, provider)`` pair the plan does not use.

    Attributes:
        runtime_id: Bundle runtime identifier.
        provider: Provider, or ``None`` when the whole runtime was excluded.
        reason: One of :data:`REASON_CODES`.
    """

    runtime_id: str
    provider: str | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return the skip document."""

        return {
            "provider": self.provider,
            "reason": self.reason,
            "runtime_id": self.runtime_id,
        }


@dataclass(frozen=True)
class ExecutionPlan:
    """Value-free resolver result (``execution_plan``).

    Attributes:
        status: ``planned`` or ``unsupported``.
        reason: ``None`` when planned, else the plan-level reason code.
        bundle_content_id: Content identity of the manifest bytes.
        profile_digest: :meth:`DeviceCapabilityProfile.digest`.
        constraints_digest: :meth:`ExecutionConstraints.digest`.
        verification_digest: Digest of the verification result.
        plan_key: Invalidation key over every input digest and the resolver
            version; a cached plan is stale when the key changes.
        root_runtime: Root of the declared fallback chain the plan follows.
        steps: Ordered attempts.
        skipped: Every runtime or pair not used, with its reason.
        assumptions: Limits that were unknown and therefore not enforced.
        max_steps: Step bound applied.
        deadline_ms: Deadline the executor must apply, or ``None``.
        max_memory_bytes: Effective memory ceiling, or ``None`` when unknown.
    """

    status: str
    reason: str | None
    bundle_content_id: str
    profile_digest: str
    constraints_digest: str
    verification_digest: str
    plan_key: str
    root_runtime: str | None
    steps: tuple[PlanStep, ...]
    skipped: tuple[PlanSkip, ...]
    assumptions: tuple[str, ...]
    max_steps: int
    deadline_ms: int | None
    max_memory_bytes: int | None

    @property
    def supported(self) -> bool:
        """Return whether the plan has at least one step."""

        return self.status == "planned"

    def to_dict(self) -> dict[str, Any]:
        """Return the plan document (``execution_plan``)."""

        return {
            "assumptions": list(self.assumptions),
            "bounds": {
                "deadline_ms": self.deadline_ms,
                "max_memory_bytes": self.max_memory_bytes,
                "max_steps": self.max_steps,
            },
            "bundle_content_id": self.bundle_content_id,
            "constraints_digest": self.constraints_digest,
            "local_only": True,
            "plan_key": self.plan_key,
            "profile_digest": self.profile_digest,
            "reason": self.reason,
            "resolver_version": RESOLVER_VERSION,
            "root_runtime": self.root_runtime,
            "schema": EXECUTION_PLAN_SCHEMA,
            "schema_version": EXECUTION_PLAN_SCHEMA_VERSION,
            "skipped": [item.to_dict() for item in self.skipped],
            "status": self.status,
            "steps": [item.to_dict() for item in self.steps],
            "verification_digest": self.verification_digest,
        }

    def canonical_bytes(self) -> bytes:
        """Return the RFC 8785 canonical encoding of :meth:`to_dict`."""

        return canonical_json_bytes(self.to_dict())


def verification_digest(verification: PortableBundleVerification) -> str:
    """Return the digest of the verification facts a plan depends on.

    Args:
        verification: Result of the portable-bundle verifier.

    Returns:
        ``sha256:`` digest over ``ok``, ``reason``, ``content_id``,
        ``files_verified``, and ``eligible_runtimes``.
    """

    return _digest(
        {
            "content_id": verification.content_id,
            "eligible_runtimes": list(verification.eligible_runtimes),
            "files_verified": verification.files_verified,
            "ok": verification.ok,
            "reason": verification.reason,
        }
    )


def compute_plan_key(
    bundle_content_id: str,
    profile_digest: str,
    constraints_digest: str,
    verification_digest_value: str,
    resolver_version: int = RESOLVER_VERSION,
) -> str:
    """Return the plan invalidation key for a set of input digests.

    Args:
        bundle_content_id: Manifest content identity.
        profile_digest: Device profile digest.
        constraints_digest: Caller constraints digest.
        verification_digest_value: Verification result digest.
        resolver_version: Resolver contract version.

    Returns:
        The ``sha256:`` plan key.
    """

    return _digest(
        {
            "bundle_content_id": bundle_content_id,
            "constraints_digest": constraints_digest,
            "profile_digest": profile_digest,
            "resolver_version": resolver_version,
            "verification_digest": verification_digest_value,
        }
    )


@dataclass(frozen=True)
class _RuntimeMeta:
    providers: tuple[str, ...]
    features: tuple[str, ...]
    evidence: tuple[str, ...]
    max_sequence_length: int | None
    min_operator_set: int | None
    tier: str | None
    declared: bool = True

    def providers_on(self, platform: str) -> tuple[str, ...]:
        if self.declared:
            return self.providers
        valid = PLATFORM_PROVIDERS[platform]
        return tuple(item for item in self.providers if item in valid)


class _MetadataInvalid(Exception):
    pass


def _parse_extension(manifest: Mapping[str, Any]) -> dict[str, _RuntimeMeta]:
    runtimes = manifest["runtimes"]
    evidence_kinds = {str(item["kind"]) for item in manifest["evidence"]}
    by_id = {str(item["id"]): item for item in runtimes}
    raw = manifest["extensions"].get(EXECUTION_EXTENSION)
    declared: Mapping[str, Any] = {}
    if raw is not None:
        if (
            not isinstance(raw, dict)
            or set(raw) != _EXTENSION_KEYS
            or type(raw["schema_version"]) is not int
            or raw["schema_version"] != EXECUTION_EXTENSION_VERSION
            or not isinstance(raw["runtimes"], dict)
            or not set(raw["runtimes"]) <= set(by_id)
        ):
            raise _MetadataInvalid()
        declared = raw["runtimes"]
    metas = {}
    for runtime_id, runtime in by_id.items():
        runtime_format = str(runtime["format"])
        entry = declared.get(runtime_id)
        if entry is None:
            metas[runtime_id] = _RuntimeMeta(
                DEFAULT_PROVIDERS[runtime_format], (), (), None, None, None, False
            )
            continue
        metas[runtime_id] = _parse_runtime_meta(entry, runtime_format, evidence_kinds)
    return metas


def _token_list(value: object, maximum: int) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > maximum:
        raise _MetadataInvalid()
    if not all(isinstance(item, str) and _TOKEN.fullmatch(item) for item in value):
        raise _MetadataInvalid()
    if len(set(value)) != len(value):
        raise _MetadataInvalid()
    return tuple(value)


def _optional_bounded(value: object, maximum: int) -> int | None:
    if value is None:
        return None
    if not _is_uint(value, maximum) or value == 0:
        raise _MetadataInvalid()
    assert isinstance(value, int)
    return value


def _parse_runtime_meta(
    entry: object, runtime_format: str, evidence_kinds: set[str]
) -> _RuntimeMeta:
    if not isinstance(entry, dict) or set(entry) != _RUNTIME_META_KEYS:
        raise _MetadataInvalid()
    providers = _token_list(entry["providers"], 8)
    if not providers or not set(providers) <= FORMAT_PROVIDERS[runtime_format]:
        raise _MetadataInvalid()
    features = _token_list(entry["features"], 16)
    evidence = _token_list(entry["evidence"], 16)
    if not set(evidence) <= evidence_kinds:
        raise _MetadataInvalid()
    tier = entry["tier"]
    if tier is not None and tier not in MODEL_TIERS:
        raise _MetadataInvalid()
    return _RuntimeMeta(
        providers=providers,
        features=features,
        evidence=tuple(sorted(evidence)),
        max_sequence_length=_optional_bounded(
            entry["max_sequence_length"], MAX_SEQUENCE_LENGTH
        ),
        min_operator_set=_optional_bounded(entry["min_operator_set"], MAX_OPERATOR_SET),
        tier=tier,
    )


def _version_key(value: str) -> tuple[int, ...]:
    parts = [int(part) for part in value.split(".")]
    return tuple(parts + [0] * (4 - len(parts)))


def _sdk_compatible(
    manifest: Mapping[str, Any], profile: DeviceCapabilityProfile
) -> bool:
    version = _version_key(profile.sdk_version)
    for sdk in manifest["compatibility"]["sdks"]:
        if sdk["name"] != profile.sdk_name:
            continue
        return version >= _version_key(sdk["min"]) and (
            sdk["max_exclusive"] is None or version < _version_key(sdk["max_exclusive"])
        )
    return False


def _effective_memory(
    profile: DeviceCapabilityProfile, constraints: ExecutionConstraints
) -> int | None:
    limits = [
        value
        for value in (profile.memory_budget_bytes, constraints.max_memory_bytes)
        if value is not None
    ]
    return min(limits) if limits else None


def _runtime_reason(
    index: int,
    runtime: Mapping[str, Any],
    meta: _RuntimeMeta,
    digests: Mapping[str, str],
    verification: PortableBundleVerification,
    profile: DeviceCapabilityProfile,
    constraints: ExecutionConstraints,
    memory_limit: int | None,
) -> ExecutionPlanReason | None:
    if index not in verification.eligible_runtimes:
        return _R.RUNTIME_NOT_ELIGIBLE
    paths = [str(runtime["entrypoint"]), *(str(path) for path in runtime["files"])]
    if any(digests[path] in constraints.revoked_artifacts for path in paths):
        return _R.ARTIFACT_REVOKED
    if runtime["format"] not in profile.runtimes:
        return _R.FORMAT_UNSUPPORTED
    platform = next(
        (item for item in runtime["platforms"] if item["platform"] == profile.platform),
        None,
    )
    if platform is None:
        return _R.PLATFORM_UNSUPPORTED
    low, high = platform["min_os"], platform["max_os_exclusive"]
    if low is not None or high is not None:
        if profile.os_version is None:
            return _R.OS_VERSION_UNKNOWN
        version = _version_key(profile.os_version)
        if (low is not None and version < _version_key(low)) or (
            high is not None and version >= _version_key(high)
        ):
            return _R.OS_VERSION_UNSUPPORTED
    if runtime["precision"] not in profile.precisions:
        return _R.PRECISION_UNSUPPORTED
    if (
        constraints.allowed_precisions is not None
        and runtime["precision"] not in constraints.allowed_precisions
    ):
        return _R.PRECISION_DISALLOWED
    if memory_limit is not None and runtime["min_memory_bytes"] > memory_limit:
        return _R.MEMORY_INSUFFICIENT
    if (
        profile.storage_budget_bytes is not None
        and runtime["min_storage_bytes"] > profile.storage_budget_bytes
    ):
        return _R.STORAGE_INSUFFICIENT
    if (
        meta.tier is not None
        and profile.max_model_tier is not None
        and model_tier_rank(meta.tier) > model_tier_rank(profile.max_model_tier)
    ):
        return _R.MODEL_TIER_EXCEEDED
    required = constraints.min_sequence_length
    if required is not None:
        if (
            profile.max_sequence_length is not None
            and profile.max_sequence_length < required
        ):
            return _R.SEQUENCE_LIMIT_EXCEEDED
        if meta.max_sequence_length is None:
            return _R.SEQUENCE_LIMIT_UNDECLARED
        if meta.max_sequence_length < required:
            return _R.SEQUENCE_LIMIT_EXCEEDED
    if meta.min_operator_set is not None:
        supported = profile.operator_sets.get(str(runtime["format"]))
        if supported is None:
            return _R.OPERATOR_SET_UNKNOWN
        if supported < meta.min_operator_set:
            return _R.OPERATOR_SET_UNSUPPORTED
    if not set(meta.features) <= set(profile.features):
        return _R.FEATURE_UNAVAILABLE
    if not set(constraints.required_evidence) <= set(meta.evidence):
        return _R.EVIDENCE_MISSING
    return None


def _provider_reason(
    provider: str,
    profile: DeviceCapabilityProfile,
    constraints: ExecutionConstraints,
) -> ExecutionPlanReason | None:
    if provider not in profile.providers:
        return _R.PROVIDER_UNAVAILABLE
    if provider in constraints.disallowed_providers:
        return _R.PROVIDER_DISALLOWED
    if constraints.energy == "low_power" and PROVIDER_ACCELERATORS[provider] == "gpu":
        return _R.ENERGY_POLICY_EXCLUDED
    return None


def _assumptions(
    runtime: Mapping[str, Any],
    meta: _RuntimeMeta,
    profile: DeviceCapabilityProfile,
    constraints: ExecutionConstraints,
    memory_limit: int | None,
) -> set[str]:
    found = set()
    if memory_limit is None and runtime["min_memory_bytes"] > 0:
        found.add("memory_budget_unknown")
    if profile.storage_budget_bytes is None and runtime["min_storage_bytes"] > 0:
        found.add("storage_budget_unknown")
    if meta.tier is not None and profile.max_model_tier is None:
        found.add("model_tier_limit_unknown")
    if (
        constraints.min_sequence_length is not None
        and profile.max_sequence_length is None
    ):
        found.add("device_sequence_limit_unknown")
    return found


def _parse_manifest(manifest_bytes: bytes) -> Mapping[str, Any] | None:
    try:
        value = json.loads(bytes(manifest_bytes).decode("utf-8"))
        critical = value.get("critical_extensions") if isinstance(value, dict) else None
        understood = (
            [item for item in critical if isinstance(item, str)]
            if isinstance(critical, list)
            else []
        )
        return PortableBundleManifest.from_bytes(
            bytes(manifest_bytes), understood_extensions=understood
        ).to_dict()
    except (PortableBundleError, ValueError, UnicodeDecodeError, RecursionError):
        return None


def resolve_execution_plan(
    manifest_bytes: bytes,
    verification: PortableBundleVerification,
    profile: DeviceCapabilityProfile,
    constraints: ExecutionConstraints | None = None,
) -> ExecutionPlan:
    """Resolve an ordered, value-free execution plan or an explicit refusal.

    Checks run in the order of :class:`ExecutionPlanReason`. The plan follows
    exactly one declared fallback chain: the first root runtime (a runtime no
    other runtime falls back to, in manifest order) whose chain contains a
    qualified step. Within a runtime, providers keep the order the signed
    execution extension declares. No step is ever added that the bundle did
    not declare, and no step is remote.

    Args:
        manifest_bytes: Exact bytes of ``openmed-bundle.json``.
        verification: The verifier result for those bytes.
        profile: Normalized device profile.
        constraints: Caller constraints; defaults to
            :class:`ExecutionConstraints` defaults.

    Returns:
        The execution plan; ``status`` is ``unsupported`` with a reason code
        when nothing qualifies.

    Raises:
        TypeError: If an argument has the wrong type.
    """

    constraints = constraints if constraints is not None else ExecutionConstraints()
    if not isinstance(verification, PortableBundleVerification):
        raise TypeError("verification must be a PortableBundleVerification")
    if not isinstance(profile, DeviceCapabilityProfile):
        raise TypeError("profile must be a DeviceCapabilityProfile")
    if not isinstance(constraints, ExecutionConstraints):
        raise TypeError("constraints must be ExecutionConstraints")
    manifest_bytes = bytes(manifest_bytes)
    bundle_id = content_id(manifest_bytes)
    profile_value = profile.digest()
    constraints_value = constraints.digest()
    verification_value = verification_digest(verification)
    key = compute_plan_key(
        bundle_id, profile_value, constraints_value, verification_value
    )
    memory_limit = _effective_memory(profile, constraints)

    def result(
        reason: ExecutionPlanReason | None,
        root: str | None = None,
        steps: Sequence[PlanStep] = (),
        skipped: Sequence[PlanSkip] = (),
        assumptions: set[str] | None = None,
    ) -> ExecutionPlan:
        return ExecutionPlan(
            status="planned" if reason is None else "unsupported",
            reason=None if reason is None else reason.value,
            bundle_content_id=bundle_id,
            profile_digest=profile_value,
            constraints_digest=constraints_value,
            verification_digest=verification_value,
            plan_key=key,
            root_runtime=root,
            steps=tuple(steps),
            skipped=tuple(skipped),
            assumptions=tuple(sorted(assumptions or ())),
            max_steps=constraints.max_steps,
            deadline_ms=constraints.deadline_ms,
            max_memory_bytes=memory_limit,
        )

    if not constraints.local_only:
        return result(_R.REMOTE_EXECUTION_UNSUPPORTED)
    if not verification.ok:
        if verification.reason == _R.BUNDLE_REVOKED.value:
            return result(_R.BUNDLE_REVOKED)
        return result(_R.BUNDLE_UNVERIFIED)
    manifest = _parse_manifest(manifest_bytes)
    if verification.content_id != bundle_id or manifest is None:
        return result(_R.BUNDLE_UNVERIFIED)
    if constraints.require_verified_files and not verification.files_verified:
        return result(_R.BUNDLE_FILES_UNVERIFIED)
    if not _sdk_compatible(manifest, profile):
        return result(_R.SDK_INCOMPATIBLE)
    try:
        metas = _parse_extension(manifest)
    except _MetadataInvalid:
        return result(_R.EXECUTION_METADATA_INVALID)

    runtimes = list(manifest["runtimes"])
    digests = {str(item["path"]): str(item["sha256"]) for item in manifest["files"]}
    by_id = {str(item["id"]): item for item in runtimes}
    index_of = {str(item["id"]): index for index, item in enumerate(runtimes)}

    exclusions: dict[str, ExecutionPlanReason] = {}
    pair_reasons: dict[tuple[str, str], ExecutionPlanReason] = {}
    qualified: dict[str, list[str]] = {}
    for index, runtime in enumerate(runtimes):
        runtime_id = str(runtime["id"])
        meta = metas[runtime_id]
        reason = _runtime_reason(
            index,
            runtime,
            meta,
            digests,
            verification,
            profile,
            constraints,
            memory_limit,
        )
        if reason is not None:
            exclusions[runtime_id] = reason
            continue
        qualified[runtime_id] = []
        for provider in meta.providers_on(profile.platform):
            provider_reason = _provider_reason(provider, profile, constraints)
            if provider_reason is None:
                qualified[runtime_id].append(provider)
            else:
                pair_reasons[(runtime_id, provider)] = provider_reason

    fallback_targets = {item["fallback"] for item in runtimes if item["fallback"]}
    root: str | None = None
    candidates: list[tuple[str, str]] = []
    for runtime in runtimes:
        runtime_id = str(runtime["id"])
        if runtime_id in fallback_targets:
            continue
        chain = [runtime_id]
        while (following := by_id[chain[-1]]["fallback"]) is not None:
            chain.append(str(following))
        pairs = [
            (item, provider) for item in chain for provider in qualified.get(item, [])
        ]
        if pairs:
            root, candidates = runtime_id, pairs
            break

    used = candidates[: constraints.max_steps]
    for pair in candidates[constraints.max_steps :]:
        pair_reasons[pair] = _R.FALLBACK_BOUND_REACHED
    for runtime_id, providers in qualified.items():
        for provider in providers:
            pair = (runtime_id, provider)
            if pair not in used and pair not in pair_reasons:
                pair_reasons[pair] = _R.NOT_DECLARED_FALLBACK

    skipped: list[PlanSkip] = []
    for runtime in runtimes:
        runtime_id = str(runtime["id"])
        if runtime_id in exclusions:
            skipped.append(PlanSkip(runtime_id, None, exclusions[runtime_id].value))
            continue
        for provider in metas[runtime_id].providers_on(profile.platform):
            if (runtime_id, provider) in pair_reasons:
                skipped.append(
                    PlanSkip(
                        runtime_id, provider, pair_reasons[(runtime_id, provider)].value
                    )
                )

    if not used:
        return result(_R.NO_ELIGIBLE_RUNTIME, skipped=skipped)

    steps: list[PlanStep] = []
    assumptions: set[str] = set()
    for position, (runtime_id, provider) in enumerate(used):
        runtime = by_id[runtime_id]
        meta = metas[runtime_id]
        if position == 0:
            role = "selected"
        elif used[position - 1][0] == runtime_id:
            role = "provider_fallback"
        else:
            role = "artifact_fallback"
        assumptions |= _assumptions(runtime, meta, profile, constraints, memory_limit)
        steps.append(
            PlanStep(
                runtime_id=runtime_id,
                runtime_index=index_of[runtime_id],
                format=str(runtime["format"]),
                precision=str(runtime["precision"]),
                provider=provider,
                accelerator=PROVIDER_ACCELERATORS[provider],
                artifact_sha256=digests[str(runtime["entrypoint"])],
                min_memory_bytes=int(runtime["min_memory_bytes"]),
                max_sequence_length=meta.max_sequence_length,
                tier=meta.tier,
                evidence=meta.evidence,
                role=role,
            )
        )
    return result(None, root, steps, skipped, assumptions)


def plan_invalidation(
    plan: ExecutionPlan,
    manifest_bytes: bytes,
    verification: PortableBundleVerification,
    profile: DeviceCapabilityProfile,
    constraints: ExecutionConstraints | None = None,
) -> tuple[str, ...]:
    """Return which inputs changed since ``plan`` was resolved.

    A cached plan may be reused only when the result is empty. Re-probe the
    device and re-verify the bundle before calling this, so that an OS,
    runtime, provider, revocation, or bundle change shows up here.

    Args:
        plan: A previously resolved plan.
        manifest_bytes: Current manifest bytes.
        verification: Current verification result.
        profile: Current device profile.
        constraints: Current caller constraints.

    Returns:
        Sorted names from :data:`INVALIDATION_COMPONENTS`.
    """

    constraints = constraints if constraints is not None else ExecutionConstraints()
    changed = []
    if plan.bundle_content_id != content_id(bytes(manifest_bytes)):
        changed.append("bundle")
    if plan.constraints_digest != constraints.digest():
        changed.append("constraints")
    if plan.profile_digest != profile.digest():
        changed.append("profile")
    if plan.plan_key != compute_plan_key(
        plan.bundle_content_id,
        plan.profile_digest,
        plan.constraints_digest,
        plan.verification_digest,
    ):
        changed.append("resolver")
    if plan.verification_digest != verification_digest(verification):
        changed.append("verification")
    return tuple(changed)


def load_execution_plan_schema(name: str = "execution_plan") -> dict[str, Any]:
    """Load one bundled device-capability or execution-plan JSON Schema.

    Args:
        name: One of :data:`SCHEMA_NAMES`.

    Returns:
        The parsed JSON Schema.

    Raises:
        KeyError: If ``name`` is not one of these schemas.
    """

    if name not in SCHEMA_NAMES:
        raise KeyError("unknown execution plan schema")
    resource = resources.files(SCHEMA_PACKAGE).joinpath(f"{name}.schema.json")
    with resource.open("r", encoding="utf-8") as handle:
        return json.load(handle)


__all__ = [
    "ASSUMPTIONS",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_PROVIDERS",
    "ENERGY_POSTURES",
    "EXECUTION_CONSTRAINTS_SCHEMA",
    "EXECUTION_CONSTRAINTS_SCHEMA_VERSION",
    "EXECUTION_EXTENSION",
    "EXECUTION_EXTENSION_VERSION",
    "EXECUTION_PLAN_SCHEMA",
    "EXECUTION_PLAN_SCHEMA_VERSION",
    "FORMAT_PROVIDERS",
    "INVALIDATION_COMPONENTS",
    "MAX_DEADLINE_MS",
    "MAX_STEPS_LIMIT",
    "REASON_CODES",
    "RESOLVER_VERSION",
    "SCHEMA_NAMES",
    "STEP_ROLES",
    "ExecutionConstraints",
    "ExecutionConstraintsError",
    "ExecutionPlan",
    "ExecutionPlanReason",
    "PlanSkip",
    "PlanStep",
    "compute_plan_key",
    "load_execution_plan_schema",
    "plan_invalidation",
    "resolve_execution_plan",
    "verification_digest",
]
