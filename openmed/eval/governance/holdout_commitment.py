"""Versioned, privacy-safe commitments for sealed benchmark holdouts.

This module commits to digest-only case, label, template, and randomization
manifests. It does not read holdout artifacts, log caller input, or perform
network requests.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence, cast

HOLDOUT_COMMITMENT_SCHEMA_VERSION = "openmed.eval.holdout_commitment.v1"
HOLDOUT_INCLUSION_PROOF_SCHEMA_VERSION = (
    "openmed.eval.holdout_commitment.inclusion_proof.v1"
)
HOLDOUT_VERIFICATION_SCHEMA_VERSION = "openmed.eval.holdout_commitment.verification.v1"
HOLDOUT_MANIFEST_KINDS = ("case", "label", "template", "randomization")

REASON_VALID = "valid"
REASON_INVALID_COMMITMENT = "invalid_commitment"
REASON_INVALID_PROOF = "invalid_proof"
REASON_COMMITMENT_MISMATCH = "commitment_mismatch"
REASON_ITEM_NOT_INCLUDED = "item_not_included"

_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")
_VERSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_MANIFEST_KINDS = frozenset(HOLDOUT_MANIFEST_KINDS)
_COMMITMENT_KEYS = frozenset(
    {
        "schema_version",
        "holdout_version",
        "manifest_roots",
        "manifest_item_counts",
        "commitment_digest",
    }
)
_PROOF_KEYS = frozenset(
    {
        "schema_version",
        "commitment_digest",
        "manifest_kind",
        "item_index",
        "item_digest",
        "sibling_digests",
    }
)
_LEAF_DOMAIN = b"openmed.eval.holdout_commitment.leaf.v1\0"
_NODE_DOMAIN = b"openmed.eval.holdout_commitment.node.v1\0"


class HoldoutCommitmentError(ValueError):
    """Raised when a private manifest cannot be committed safely."""


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _digest(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _is_sha256(value: Any) -> bool:
    return type(value) is str and _SHA256_RE.fullmatch(value) is not None


def _is_public_version(value: Any) -> bool:
    return type(value) is str and _VERSION_RE.fullmatch(value) is not None


def _leaf_digest(manifest_kind: str, item_index: int, item_digest: str) -> str:
    payload = b"\0".join(
        (
            _LEAF_DOMAIN + manifest_kind.encode("ascii"),
            str(item_index).encode("ascii"),
            item_digest.encode("ascii"),
        )
    )
    return _digest(payload)


def _node_digest(left: str, right: str) -> str:
    return _digest(_NODE_DOMAIN + left.encode("ascii") + b"\0" + right.encode("ascii"))


def _normalize_manifests(
    manifests: Mapping[str, Sequence[str]],
) -> Mapping[str, tuple[str, ...]]:
    if not isinstance(manifests, Mapping) or set(manifests) != _MANIFEST_KINDS:
        raise HoldoutCommitmentError("manifests: invalid_keys")

    normalized: dict[str, tuple[str, ...]] = {}
    for manifest_kind in HOLDOUT_MANIFEST_KINDS:
        items = manifests[manifest_kind]
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
            raise HoldoutCommitmentError(f"manifests.{manifest_kind}: invalid_sequence")
        item_digests = tuple(items)
        if not item_digests:
            raise HoldoutCommitmentError(f"manifests.{manifest_kind}: empty")
        if not all(_is_sha256(item_digest) for item_digest in item_digests):
            raise HoldoutCommitmentError(
                f"manifests.{manifest_kind}: invalid_item_digest"
            )
        normalized[manifest_kind] = item_digests
    return MappingProxyType(normalized)


def _leaf_level(manifest_kind: str, item_digests: Sequence[str]) -> list[str]:
    return [
        _leaf_digest(manifest_kind, item_index, item_digest)
        for item_index, item_digest in enumerate(item_digests)
    ]


def _next_level(level: Sequence[str]) -> list[str]:
    return [
        _node_digest(
            level[index], level[index + 1] if index + 1 < len(level) else level[index]
        )
        for index in range(0, len(level), 2)
    ]


def _merkle_root(manifest_kind: str, item_digests: Sequence[str]) -> str:
    level = _leaf_level(manifest_kind, item_digests)
    while len(level) > 1:
        level = _next_level(level)
    return level[0]


def _commitment_payload(
    holdout_version: str,
    manifest_roots: Mapping[str, str],
    manifest_item_counts: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "holdout_version": holdout_version,
        "manifest_item_counts": dict(manifest_item_counts),
        "manifest_roots": dict(manifest_roots),
        "schema_version": HOLDOUT_COMMITMENT_SCHEMA_VERSION,
    }


def _commitment_digest(
    holdout_version: str,
    manifest_roots: Mapping[str, str],
    manifest_item_counts: Mapping[str, int],
) -> str:
    payload = _commitment_payload(holdout_version, manifest_roots, manifest_item_counts)
    return _digest(_canonical_json(payload).encode("utf-8"))


@dataclass(frozen=True, slots=True)
class HoldoutCommitment:
    """A public, content-free commitment to four private holdout manifests."""

    holdout_version: str
    manifest_roots: Mapping[str, str]
    manifest_item_counts: Mapping[str, int]
    commitment_digest: str
    schema_version: str = HOLDOUT_COMMITMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != HOLDOUT_COMMITMENT_SCHEMA_VERSION:
            raise HoldoutCommitmentError("schema_version: unsupported_version")
        if not _is_public_version(self.holdout_version):
            raise HoldoutCommitmentError("holdout_version: invalid")
        roots = _normalize_roots(self.manifest_roots)
        counts = _normalize_counts(self.manifest_item_counts)
        if not _is_sha256(self.commitment_digest):
            raise HoldoutCommitmentError("commitment_digest: invalid_digest")
        expected = _commitment_digest(self.holdout_version, roots, counts)
        if not hmac.compare_digest(self.commitment_digest, expected):
            raise HoldoutCommitmentError("commitment_digest: payload_mismatch")
        object.__setattr__(self, "manifest_roots", roots)
        object.__setattr__(self, "manifest_item_counts", counts)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible public commitment."""
        return {
            "commitment_digest": self.commitment_digest,
            "holdout_version": self.holdout_version,
            "manifest_item_counts": dict(self.manifest_item_counts),
            "manifest_roots": dict(self.manifest_roots),
            "schema_version": self.schema_version,
        }

    def to_json(self) -> str:
        """Return stable compact JSON suitable for pre-publication storage."""
        return _canonical_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class HoldoutInclusionProof:
    """A digest-only proof that one indexed item was committed."""

    commitment_digest: str
    manifest_kind: str
    item_index: int
    item_digest: str
    sibling_digests: tuple[str, ...]
    schema_version: str = HOLDOUT_INCLUSION_PROOF_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != HOLDOUT_INCLUSION_PROOF_SCHEMA_VERSION:
            raise HoldoutCommitmentError("schema_version: unsupported_version")
        if not _is_sha256(self.commitment_digest):
            raise HoldoutCommitmentError("commitment_digest: invalid_digest")
        if self.manifest_kind not in _MANIFEST_KINDS:
            raise HoldoutCommitmentError("manifest_kind: invalid")
        if type(self.item_index) is not int or self.item_index < 0:
            raise HoldoutCommitmentError("item_index: invalid")
        if not _is_sha256(self.item_digest):
            raise HoldoutCommitmentError("item_digest: invalid_digest")
        siblings = tuple(self.sibling_digests)
        if not all(_is_sha256(sibling) for sibling in siblings):
            raise HoldoutCommitmentError("sibling_digests: invalid_digest")
        object.__setattr__(self, "sibling_digests", siblings)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible audit proof."""
        return {
            "commitment_digest": self.commitment_digest,
            "item_digest": self.item_digest,
            "item_index": self.item_index,
            "manifest_kind": self.manifest_kind,
            "schema_version": self.schema_version,
            "sibling_digests": list(self.sibling_digests),
        }


@dataclass(frozen=True, slots=True)
class HoldoutCommitmentVerification:
    """A content-free result for commitment or inclusion-proof checks."""

    valid: bool
    reason_codes: tuple[str, ...]
    schema_version: str = HOLDOUT_VERIFICATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic report containing no caller values."""
        return {
            "reason_codes": list(self.reason_codes),
            "schema_version": self.schema_version,
            "valid": self.valid,
        }


def _normalize_roots(roots: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(roots, Mapping) or set(roots) != _MANIFEST_KINDS:
        raise HoldoutCommitmentError("manifest_roots: invalid_keys")
    normalized = {kind: roots[kind] for kind in HOLDOUT_MANIFEST_KINDS}
    if not all(_is_sha256(root) for root in normalized.values()):
        raise HoldoutCommitmentError("manifest_roots: invalid_digest")
    return MappingProxyType(normalized)


def _normalize_counts(counts: Mapping[str, int]) -> Mapping[str, int]:
    if not isinstance(counts, Mapping) or set(counts) != _MANIFEST_KINDS:
        raise HoldoutCommitmentError("manifest_item_counts: invalid_keys")
    normalized = {kind: counts[kind] for kind in HOLDOUT_MANIFEST_KINDS}
    if not all(type(count) is int and count > 0 for count in normalized.values()):
        raise HoldoutCommitmentError("manifest_item_counts: invalid_count")
    return MappingProxyType(normalized)


def commit_holdout_manifests(
    holdout_version: str,
    manifests: Mapping[str, Sequence[str]],
) -> HoldoutCommitment:
    """Commit to ordered private manifests using only their item digests.

    Args:
        holdout_version: Public, non-sensitive version identifier.
        manifests: Exactly four ordered sequences of lowercase ``sha256:``
            item digests, keyed by :data:`HOLDOUT_MANIFEST_KINDS`.

    Returns:
        A public commitment containing roots, counts, and a bundle digest.
    """
    if not _is_public_version(holdout_version):
        raise HoldoutCommitmentError("holdout_version: invalid")
    normalized = _normalize_manifests(manifests)
    roots = {
        kind: _merkle_root(kind, normalized[kind]) for kind in HOLDOUT_MANIFEST_KINDS
    }
    counts = {kind: len(normalized[kind]) for kind in HOLDOUT_MANIFEST_KINDS}
    return HoldoutCommitment(
        holdout_version=holdout_version,
        manifest_roots=roots,
        manifest_item_counts=counts,
        commitment_digest=_commitment_digest(holdout_version, roots, counts),
    )


def create_inclusion_proof(
    commitment: HoldoutCommitment,
    manifest_kind: str,
    item_index: int,
    item_digests: Sequence[str],
) -> HoldoutInclusionProof:
    """Create a selective proof from one evaluator-held private manifest.

    The supplied manifest must reproduce the public commitment before any
    proof is returned. Only the selected item digest and Merkle siblings are
    disclosed.
    """
    if not isinstance(commitment, HoldoutCommitment):
        raise HoldoutCommitmentError("commitment: invalid")
    if manifest_kind not in _MANIFEST_KINDS:
        raise HoldoutCommitmentError("manifest_kind: invalid")
    if not isinstance(item_digests, Sequence) or isinstance(item_digests, (str, bytes)):
        raise HoldoutCommitmentError("item_digests: invalid_sequence")
    normalized = tuple(item_digests)
    if not normalized or not all(_is_sha256(item) for item in normalized):
        raise HoldoutCommitmentError("item_digests: invalid")
    if type(item_index) is not int or not 0 <= item_index < len(normalized):
        raise HoldoutCommitmentError("item_index: invalid")
    if len(normalized) != commitment.manifest_item_counts[
        manifest_kind
    ] or not hmac.compare_digest(
        _merkle_root(manifest_kind, normalized),
        commitment.manifest_roots[manifest_kind],
    ):
        raise HoldoutCommitmentError("item_digests: commitment_mismatch")

    siblings: list[str] = []
    level = _leaf_level(manifest_kind, normalized)
    level_index = item_index
    while len(level) > 1:
        sibling_index = level_index - 1 if level_index % 2 else level_index + 1
        siblings.append(
            level[sibling_index] if sibling_index < len(level) else level[level_index]
        )
        level = _next_level(level)
        level_index //= 2

    return HoldoutInclusionProof(
        commitment_digest=commitment.commitment_digest,
        manifest_kind=manifest_kind,
        item_index=item_index,
        item_digest=normalized[item_index],
        sibling_digests=tuple(siblings),
    )


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, (HoldoutCommitment, HoldoutInclusionProof)):
        return value.to_dict()
    if isinstance(value, Mapping):
        return value
    return None


def _validated_commitment_document(
    commitment: HoldoutCommitment | Mapping[str, Any],
) -> Mapping[str, Any] | None:
    document = _as_mapping(commitment)
    if document is None or set(document) != _COMMITMENT_KEYS:
        return None
    if document.get("schema_version") != HOLDOUT_COMMITMENT_SCHEMA_VERSION:
        return None
    version_value = document.get("holdout_version")
    roots_value = document.get("manifest_roots")
    counts_value = document.get("manifest_item_counts")
    digest_value = document.get("commitment_digest")
    if not _is_public_version(version_value) or not _is_sha256(digest_value):
        return None
    version = cast(str, version_value)
    roots = cast(Mapping[str, str], roots_value)
    counts = cast(Mapping[str, int], counts_value)
    digest = cast(str, digest_value)
    try:
        normalized_roots = _normalize_roots(roots)
        normalized_counts = _normalize_counts(counts)
    except (HoldoutCommitmentError, TypeError):
        return None
    expected = _commitment_digest(version, normalized_roots, normalized_counts)
    if not hmac.compare_digest(digest, expected):
        return None
    return MappingProxyType(
        {
            "commitment_digest": digest,
            "holdout_version": version,
            "manifest_item_counts": normalized_counts,
            "manifest_roots": normalized_roots,
            "schema_version": HOLDOUT_COMMITMENT_SCHEMA_VERSION,
        }
    )


def verify_holdout_commitment(
    commitment: HoldoutCommitment | Mapping[str, Any],
) -> HoldoutCommitmentVerification:
    """Verify a parsed public commitment without raising or echoing values."""
    valid = _validated_commitment_document(commitment) is not None
    return HoldoutCommitmentVerification(
        valid=valid,
        reason_codes=(REASON_VALID if valid else REASON_INVALID_COMMITMENT,),
    )


def _proof_depth(item_count: int) -> int:
    depth = 0
    while item_count > 1:
        item_count = (item_count + 1) // 2
        depth += 1
    return depth


def verify_inclusion_proof(
    commitment: HoldoutCommitment | Mapping[str, Any],
    proof: HoldoutInclusionProof | Mapping[str, Any],
) -> HoldoutCommitmentVerification:
    """Verify a selective audit proof offline and fail closed on bad input."""
    commitment_document = _validated_commitment_document(commitment)
    if commitment_document is None:
        return HoldoutCommitmentVerification(False, (REASON_INVALID_COMMITMENT,))

    proof_document = _as_mapping(proof)
    if proof_document is None or set(proof_document) != _PROOF_KEYS:
        return HoldoutCommitmentVerification(False, (REASON_INVALID_PROOF,))
    if proof_document.get("schema_version") != HOLDOUT_INCLUSION_PROOF_SCHEMA_VERSION:
        return HoldoutCommitmentVerification(False, (REASON_INVALID_PROOF,))

    commitment_digest = proof_document.get("commitment_digest")
    manifest_kind = proof_document.get("manifest_kind")
    item_index = proof_document.get("item_index")
    item_digest = proof_document.get("item_digest")
    siblings = proof_document.get("sibling_digests")
    if (
        not _is_sha256(commitment_digest)
        or manifest_kind not in _MANIFEST_KINDS
        or type(item_index) is not int
        or item_index < 0
        or not _is_sha256(item_digest)
        or not isinstance(siblings, Sequence)
        or isinstance(siblings, (str, bytes))
        or not all(_is_sha256(sibling) for sibling in siblings)
    ):
        return HoldoutCommitmentVerification(False, (REASON_INVALID_PROOF,))
    if not hmac.compare_digest(
        commitment_digest, commitment_document["commitment_digest"]
    ):
        return HoldoutCommitmentVerification(False, (REASON_COMMITMENT_MISMATCH,))

    item_count = commitment_document["manifest_item_counts"][manifest_kind]
    if item_index >= item_count or len(siblings) != _proof_depth(item_count):
        return HoldoutCommitmentVerification(False, (REASON_INVALID_PROOF,))

    current = _leaf_digest(manifest_kind, item_index, cast(str, item_digest))
    level_index = item_index
    level_size = item_count
    for sibling in siblings:
        if level_index % 2:
            current = _node_digest(sibling, current)
        else:
            if level_index + 1 >= level_size and not hmac.compare_digest(
                sibling, current
            ):
                return HoldoutCommitmentVerification(False, (REASON_INVALID_PROOF,))
            current = _node_digest(current, sibling)
        level_index //= 2
        level_size = (level_size + 1) // 2

    included = hmac.compare_digest(
        current, commitment_document["manifest_roots"][manifest_kind]
    )
    return HoldoutCommitmentVerification(
        valid=included,
        reason_codes=(REASON_VALID if included else REASON_ITEM_NOT_INCLUDED,),
    )


__all__ = [
    "HOLDOUT_COMMITMENT_SCHEMA_VERSION",
    "HOLDOUT_INCLUSION_PROOF_SCHEMA_VERSION",
    "HOLDOUT_MANIFEST_KINDS",
    "HOLDOUT_VERIFICATION_SCHEMA_VERSION",
    "REASON_COMMITMENT_MISMATCH",
    "REASON_INVALID_COMMITMENT",
    "REASON_INVALID_PROOF",
    "REASON_ITEM_NOT_INCLUDED",
    "REASON_VALID",
    "HoldoutCommitment",
    "HoldoutCommitmentError",
    "HoldoutCommitmentVerification",
    "HoldoutInclusionProof",
    "commit_holdout_manifests",
    "create_inclusion_proof",
    "verify_holdout_commitment",
    "verify_inclusion_proof",
]
