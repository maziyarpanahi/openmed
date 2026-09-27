"""Deterministic, content-free manifests for sealed workflow evaluation.

The manifest records only SHA-256 digests for the executable submission
surface. Evaluation-start verification compares those sealed digests with a
fresh local snapshot; it never opens artifacts or performs network requests.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

SEALED_MANIFEST_SCHEMA_VERSION = "openmed.eval.workflow_sealed_manifest.v1"
SEALED_MANIFEST_VERIFICATION_SCHEMA_VERSION = (
    "openmed.eval.workflow_sealed_manifest_verification.v1"
)
SEALED_MANIFEST_COMPONENTS = (
    "container",
    "model",
    "policy",
    "post_processing",
    "prompt",
    "threshold",
    "tokenizer",
    "tool_inventory",
)

_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")
_MANIFEST_KEYS = frozenset({"schema_version", "component_digests", "manifest_digest"})
_COMPONENT_KEYS = frozenset(SEALED_MANIFEST_COMPONENTS)

REASON_ELIGIBLE = "eligible"
REASON_INCOMPLETE_MANIFEST = "incomplete_manifest"
REASON_INVALID_MANIFEST = "invalid_manifest"
REASON_MUTABLE_MANIFEST = "mutable_manifest"
REASON_INCOMPLETE_EVALUATION_SNAPSHOT = "incomplete_evaluation_snapshot"
REASON_INVALID_EVALUATION_SNAPSHOT = "invalid_evaluation_snapshot"
REASON_COMPONENT_DIGEST_MISMATCH = "component_digest_mismatch"


class SealedManifestError(ValueError):
    """Raised when a manifest cannot be sealed safely."""


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _is_sha256(value: Any) -> bool:
    return type(value) is str and _SHA256_RE.fullmatch(value) is not None


def _normalize_component_digests(
    component_digests: Mapping[str, str],
) -> Mapping[str, str]:
    if not isinstance(component_digests, Mapping):
        raise SealedManifestError("component_digests: invalid_mapping")
    if set(component_digests) != _COMPONENT_KEYS:
        raise SealedManifestError("component_digests: invalid_keys")

    normalized: dict[str, str] = {}
    for component in SEALED_MANIFEST_COMPONENTS:
        digest = component_digests[component]
        if not _is_sha256(digest):
            raise SealedManifestError(f"component_digests.{component}: invalid_digest")
        normalized[component] = digest
    return MappingProxyType(normalized)


def _manifest_payload(component_digests: Mapping[str, str]) -> dict[str, Any]:
    return {
        "component_digests": dict(component_digests),
        "schema_version": SEALED_MANIFEST_SCHEMA_VERSION,
    }


def _payload_digest(component_digests: Mapping[str, str]) -> str:
    payload = _canonical_json(_manifest_payload(component_digests)).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


@dataclass(frozen=True, slots=True)
class SealedWorkflowManifest:
    """An immutable digest-only workflow submission manifest.

    Construct manifests with :meth:`seal` or :func:`seal_workflow_manifest` so
    their content hash is derived from the canonical payload.

    Args:
        component_digests: Exactly one SHA-256 digest for each governed
            submission component.
        manifest_digest: SHA-256 digest of the canonical manifest payload.
        schema_version: Closed manifest schema identifier.
    """

    component_digests: Mapping[str, str]
    manifest_digest: str
    schema_version: str = SEALED_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SEALED_MANIFEST_SCHEMA_VERSION:
            raise SealedManifestError("schema_version: unsupported_version")
        normalized = _normalize_component_digests(self.component_digests)
        if not _is_sha256(self.manifest_digest):
            raise SealedManifestError("manifest_digest: invalid_digest")
        if self.manifest_digest != _payload_digest(normalized):
            raise SealedManifestError("manifest_digest: payload_mismatch")
        object.__setattr__(self, "component_digests", normalized)

    @classmethod
    def seal(cls, component_digests: Mapping[str, str]) -> "SealedWorkflowManifest":
        """Freeze a complete set of component digests into a sealed manifest."""
        normalized = _normalize_component_digests(component_digests)
        return cls(
            component_digests=normalized,
            manifest_digest=_payload_digest(normalized),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible manifest document."""
        return {
            "component_digests": dict(self.component_digests),
            "manifest_digest": self.manifest_digest,
            "schema_version": self.schema_version,
        }

    def to_json(self) -> str:
        """Return stable compact JSON suitable for submission storage."""
        return _canonical_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class SealedManifestVerification:
    """Content-free eligibility result produced at evaluation start."""

    eligible_for_sealed_results: bool
    reason_codes: tuple[str, ...]
    affected_components: tuple[str, ...] = ()
    schema_version: str = SEALED_MANIFEST_VERIFICATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic report without submitted digest values."""
        return {
            "affected_components": list(self.affected_components),
            "eligible_for_sealed_results": self.eligible_for_sealed_results,
            "reason_codes": list(self.reason_codes),
            "schema_version": self.schema_version,
        }


def seal_workflow_manifest(
    component_digests: Mapping[str, str],
) -> SealedWorkflowManifest:
    """Freeze component digests into an immutable canonical manifest."""
    return SealedWorkflowManifest.seal(component_digests)


def _as_manifest_mapping(
    manifest: SealedWorkflowManifest | Mapping[str, Any],
) -> Mapping[str, Any] | None:
    if isinstance(manifest, SealedWorkflowManifest):
        return manifest.to_dict()
    if isinstance(manifest, Mapping):
        return manifest
    return None


def verify_at_evaluation_start(
    manifest: SealedWorkflowManifest | Mapping[str, Any],
    evaluation_component_digests: Mapping[str, str],
) -> SealedManifestVerification:
    """Verify a submission's seal and current component digests offline.

    Incomplete, malformed, changed, or non-canonical inputs return an
    ineligible result instead of raising. The report contains only closed
    reason codes and component names, never caller-provided values.

    Args:
        manifest: Sealed manifest instance or its parsed JSON mapping.
        evaluation_component_digests: Fresh local digest snapshot taken at
            evaluation start.

    Returns:
        A deterministic eligibility report for sealed-result publication.
    """
    document = _as_manifest_mapping(manifest)
    reasons: list[str] = []
    affected: set[str] = set()
    sealed_components: Mapping[str, Any] | None = None

    if document is None:
        reasons.append(REASON_INVALID_MANIFEST)
    else:
        document_keys = set(document)
        missing_manifest_keys = _MANIFEST_KEYS - document_keys
        if missing_manifest_keys:
            reasons.append(REASON_INCOMPLETE_MANIFEST)
        if document_keys - _MANIFEST_KEYS:
            reasons.append(REASON_INVALID_MANIFEST)

        if document.get("schema_version") != SEALED_MANIFEST_SCHEMA_VERSION:
            reasons.append(REASON_INVALID_MANIFEST)

        components = document.get("component_digests")
        if not isinstance(components, Mapping):
            reasons.append(REASON_INCOMPLETE_MANIFEST)
        else:
            sealed_components = components
            component_keys = set(components)
            missing_components = _COMPONENT_KEYS - component_keys
            affected.update(missing_components)
            if missing_components:
                reasons.append(REASON_INCOMPLETE_MANIFEST)
            extra_components = component_keys - _COMPONENT_KEYS
            if extra_components:
                reasons.append(REASON_INVALID_MANIFEST)
            invalid_components = {
                component
                for component in _COMPONENT_KEYS & component_keys
                if not _is_sha256(components[component])
            }
            affected.update(invalid_components)
            if invalid_components:
                reasons.append(REASON_INVALID_MANIFEST)

        manifest_digest = document.get("manifest_digest")
        if manifest_digest is None:
            reasons.append(REASON_INCOMPLETE_MANIFEST)
        elif not _is_sha256(manifest_digest):
            reasons.append(REASON_INVALID_MANIFEST)
        elif (
            sealed_components is not None
            and set(sealed_components) == _COMPONENT_KEYS
            and all(_is_sha256(sealed_components[key]) for key in _COMPONENT_KEYS)
            and manifest_digest != _payload_digest(sealed_components)
        ):
            reasons.append(REASON_MUTABLE_MANIFEST)

    if not isinstance(evaluation_component_digests, Mapping):
        reasons.append(REASON_INVALID_EVALUATION_SNAPSHOT)
    else:
        evaluation_keys = set(evaluation_component_digests)
        missing_evaluation = _COMPONENT_KEYS - evaluation_keys
        affected.update(missing_evaluation)
        if missing_evaluation:
            reasons.append(REASON_INCOMPLETE_EVALUATION_SNAPSHOT)
        extra_evaluation = evaluation_keys - _COMPONENT_KEYS
        if extra_evaluation:
            reasons.append(REASON_INVALID_EVALUATION_SNAPSHOT)
        invalid_evaluation = {
            component
            for component in _COMPONENT_KEYS & evaluation_keys
            if not _is_sha256(evaluation_component_digests[component])
        }
        affected.update(invalid_evaluation)
        if invalid_evaluation:
            reasons.append(REASON_INVALID_EVALUATION_SNAPSHOT)

        if sealed_components is not None:
            mismatches = {
                component
                for component in _COMPONENT_KEYS & evaluation_keys
                if _is_sha256(evaluation_component_digests[component])
                and _is_sha256(sealed_components.get(component))
                and evaluation_component_digests[component]
                != sealed_components[component]
            }
            affected.update(mismatches)
            if mismatches:
                reasons.append(REASON_COMPONENT_DIGEST_MISMATCH)

    reason_codes = tuple(dict.fromkeys(reasons)) or (REASON_ELIGIBLE,)
    return SealedManifestVerification(
        eligible_for_sealed_results=not reasons,
        reason_codes=reason_codes,
        affected_components=tuple(sorted(affected)),
    )


__all__ = [
    "REASON_COMPONENT_DIGEST_MISMATCH",
    "REASON_ELIGIBLE",
    "REASON_INCOMPLETE_EVALUATION_SNAPSHOT",
    "REASON_INCOMPLETE_MANIFEST",
    "REASON_INVALID_EVALUATION_SNAPSHOT",
    "REASON_INVALID_MANIFEST",
    "REASON_MUTABLE_MANIFEST",
    "SEALED_MANIFEST_COMPONENTS",
    "SEALED_MANIFEST_SCHEMA_VERSION",
    "SEALED_MANIFEST_VERIFICATION_SCHEMA_VERSION",
    "SealedManifestError",
    "SealedManifestVerification",
    "SealedWorkflowManifest",
    "seal_workflow_manifest",
    "verify_at_evaluation_start",
]
