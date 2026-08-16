"""Deterministic pre-install integrity verification for portable skill bundles.

A skill bundle is a directory containing a ``manifest.json`` that declares the
bundle identifier, entry points, per-file SHA-256 digests, and an optional
HMAC-SHA256 signature over the canonical manifest. Verification is purely
local: it performs no network calls and never logs raw file contents, full
hashes, manifest JSON, signature bytes, or signature keys.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUPPORTED_MANIFEST_VERSIONS = frozenset({"1.0"})
HASH_ALGORITHM = "sha256"
HASH_PREFIX_LENGTH = 12

REASON_MANIFEST_MALFORMED = "manifest_malformed"
REASON_MANIFEST_VERSION_UNSUPPORTED = "manifest_version_unsupported"
REASON_FILE_MISSING = "file_missing"
REASON_HASH_MISMATCH = "hash_mismatch"
REASON_ENTRY_POINT_MISSING = "entry_point_missing"
REASON_ENTRY_POINT_NOT_DECLARED = "entry_point_not_declared"
REASON_SIGNATURE_REQUIRED = "signature_required"
REASON_SIGNATURE_INVALID = "signature_invalid"
REASON_SIGNATURE_SCHEME_UNSUPPORTED = "signature_scheme_unsupported"

_LOGGER = logging.getLogger(__name__)
_HEX64_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_SIGNATURE_SCHEMES = frozenset({"none", "hmac-sha256"})


def _hash_prefix(digest: str) -> str:
    """Return the first :data:`HASH_PREFIX_LENGTH` characters of ``digest``."""

    return digest[:HASH_PREFIX_LENGTH]


@dataclass(frozen=True)
class SkillBundleManifest:
    """Declared metadata for a portable skill bundle.

    Args:
        manifest_version: Manifest schema version declared by the bundle.
        bundle_id: Stable, non-empty bundle identifier.
        entry_points: Relative paths of executable entry points in the bundle.
        files: Mapping of relative file path to declared SHA-256 hex digest.
        signature_scheme: Signature scheme; ``"none"`` or ``"hmac-sha256"``.
        signature: Hex-encoded signature tag; empty when scheme is ``"none"``.
    """

    manifest_version: str
    bundle_id: str
    entry_points: tuple[str, ...]
    files: Mapping[str, str]
    signature_scheme: str = "none"
    signature: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.manifest_version, str):
            raise ValueError("manifest_version must be a string")
        if not isinstance(self.bundle_id, str) or not self.bundle_id.strip():
            raise ValueError("bundle_id must be a non-empty string")

        if isinstance(self.entry_points, (str, bytes)) or not isinstance(
            self.entry_points, Sequence
        ):
            raise ValueError("entry_points must be a sequence of strings")
        entry_points = tuple(self.entry_points)
        if any(not isinstance(item, str) or not item.strip() for item in entry_points):
            raise ValueError("entry_points must contain only non-empty strings")
        entry_points = tuple(item.strip() for item in entry_points if item.strip())

        if not isinstance(self.files, Mapping):
            raise ValueError("files must be a mapping")
        files: dict[str, str] = {}
        for path, digest in self.files.items():
            if not isinstance(path, str) or not path:
                raise ValueError("files must have non-empty string paths")
            if not isinstance(digest, str) or _HEX64_RE.fullmatch(digest) is None:
                raise ValueError("files must map to 64-character hex digests")
            files[path] = digest.lower()

        if self.signature_scheme not in _SIGNATURE_SCHEMES:
            raise ValueError("signature_scheme must be 'none' or 'hmac-sha256'")
        if not isinstance(self.signature, str):
            raise ValueError("signature must be a string")

        object.__setattr__(self, "entry_points", entry_points)
        object.__setattr__(self, "files", files)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SkillBundleManifest":
        """Parse and validate a manifest mapping.

        Args:
            data: Raw manifest mapping, typically loaded from ``manifest.json``.

        Returns:
            A validated, immutable :class:`SkillBundleManifest`.

        Raises:
            ValueError: If the mapping is malformed or fails validation.
        """

        if not isinstance(data, Mapping):
            raise ValueError("manifest must be a mapping")
        return cls(
            manifest_version=data.get("manifest_version", ""),
            bundle_id=data.get("bundle_id", ""),
            entry_points=tuple(data.get("entry_points", ())),
            files=dict(data.get("files", {})),
            signature_scheme=data.get("signature_scheme", "none"),
            signature=data.get("signature", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible dict.

        Never includes raw file contents; only declared metadata and hashes.
        """

        return {
            "manifest_version": self.manifest_version,
            "bundle_id": self.bundle_id,
            "entry_points": list(self.entry_points),
            "files": dict(self.files),
            "signature_scheme": self.signature_scheme,
            "signature": self.signature,
        }


@dataclass(frozen=True)
class BundleFileResult:
    """Verification result for a single declared file.

    Args:
        path: Relative path of the declared file.
        declared_hash: SHA-256 hex digest declared in the manifest.
        actual_hash: SHA-256 hex digest computed from the file contents.
        matched: Whether the declared and actual digests are equal.
    """

    path: str
    declared_hash: str
    actual_hash: str
    matched: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a dict with hash prefixes only; never full hashes or contents."""

        return {
            "path": self.path,
            "declared_hash_prefix": _hash_prefix(self.declared_hash),
            "actual_hash_prefix": _hash_prefix(self.actual_hash),
            "matched": self.matched,
        }


@dataclass(frozen=True)
class BundleVerificationResult:
    """Deterministic result of verifying a skill bundle.

    Args:
        bundle_id: Identifier of the verified bundle.
        manifest_version: Manifest schema version declared by the bundle.
        valid: Whether all verification checks passed.
        reason: Failure category label; empty when ``valid`` is true.
        message: Human-readable failure detail; empty when ``valid`` is true.
        files: Per-file verification results.
        entry_points_checked: Entry points examined during verification.
        signature_verified: Whether the signature check passed.
    """

    bundle_id: str
    manifest_version: str
    valid: bool
    reason: str
    message: str
    files: tuple[BundleFileResult, ...]
    entry_points_checked: tuple[str, ...]
    signature_verified: bool

    @property
    def failure_category(self) -> str:
        """Return the reason string, or empty when valid."""

        return self.reason

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible dict with hash prefixes only."""

        return {
            "bundle_id": self.bundle_id,
            "manifest_version": self.manifest_version,
            "valid": self.valid,
            "reason": self.reason,
            "message": self.message,
            "files": [item.to_dict() for item in self.files],
            "entry_points_checked": list(self.entry_points_checked),
            "signature_verified": self.signature_verified,
        }


def _build_failure(
    bundle_id: str,
    manifest_version: str,
    reason: str,
    message: str,
    files: tuple[BundleFileResult, ...] = (),
    entry_points_checked: tuple[str, ...] = (),
    signature_verified: bool = False,
) -> BundleVerificationResult:
    """Assemble a failure result with the given category and context."""

    return BundleVerificationResult(
        bundle_id=bundle_id,
        manifest_version=manifest_version,
        valid=False,
        reason=reason,
        message=message,
        files=files,
        entry_points_checked=entry_points_checked,
        signature_verified=signature_verified,
    )


class BundleVerifier:
    """Deterministic pre-install integrity checker for portable skill bundles.

    Performs no network calls. Logs only operational telemetry (counts, hash
    prefixes, category labels) -- never raw bundle contents, manifest values,
    or signature bytes.
    """

    def __init__(
        self,
        *,
        supported_versions: frozenset[str] = SUPPORTED_MANIFEST_VERSIONS,
    ) -> None:
        """Create a verifier accepting ``supported_versions``.

        Args:
            supported_versions: Manifest versions accepted by this verifier.

        Raises:
            TypeError: If ``supported_versions`` is not a frozenset of strings.
        """

        if not isinstance(supported_versions, frozenset):
            raise TypeError("supported_versions must be a frozenset")
        if any(not isinstance(v, str) for v in supported_versions):
            raise TypeError("supported_versions must contain only strings")
        self._supported_versions = supported_versions

    def verify(
        self,
        bundle_dir: str | Path,
        *,
        signature_key: bytes | None = None,
    ) -> BundleVerificationResult:
        """Verify a skill bundle directory against its manifest.

        Args:
            bundle_dir: Path to the bundle directory containing ``manifest.json``.
            signature_key: Optional HMAC-SHA256 key. Required when the manifest
                declares signature_scheme ``"hmac-sha256"``. Ignored when
                ``"none"``.

        Returns:
            Deterministic verification result. Failures are represented in the
            result and never raised, except for filesystem errors reading the
            manifest itself.

        Raises:
            FileNotFoundError: If ``manifest.json`` does not exist in
                ``bundle_dir``.
        """

        bundle_path = Path(bundle_dir)
        manifest_path = bundle_path / "manifest.json"
        raw_bytes = manifest_path.read_bytes()

        try:
            raw_manifest = json.loads(raw_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            _LOGGER.info(
                "bundle verification failed: category=%s",
                REASON_MANIFEST_MALFORMED,
            )
            return _build_failure(
                "",
                "",
                REASON_MANIFEST_MALFORMED,
                "manifest.json is not valid JSON",
            )

        try:
            manifest = SkillBundleManifest.from_mapping(raw_manifest)
        except ValueError:
            _LOGGER.info(
                "bundle verification failed: category=%s",
                REASON_MANIFEST_MALFORMED,
            )
            return _build_failure(
                "",
                "",
                REASON_MANIFEST_MALFORMED,
                "manifest failed validation",
            )

        if manifest.manifest_version not in self._supported_versions:
            _LOGGER.info(
                "bundle=%s verification failed: category=%s",
                manifest.bundle_id,
                REASON_MANIFEST_VERSION_UNSUPPORTED,
            )
            return _build_failure(
                manifest.bundle_id,
                manifest.manifest_version,
                REASON_MANIFEST_VERSION_UNSUPPORTED,
                f"manifest version {manifest.manifest_version!r} is not supported",
            )

        file_results: list[BundleFileResult] = []
        for rel_path, declared_hash in manifest.files.items():
            file_path = bundle_path / rel_path
            if not file_path.is_file():
                _LOGGER.info(
                    "bundle=%s verification failed: category=%s file_count=%d",
                    manifest.bundle_id,
                    REASON_FILE_MISSING,
                    len(file_results),
                )
                return _build_failure(
                    manifest.bundle_id,
                    manifest.manifest_version,
                    REASON_FILE_MISSING,
                    f"declared file is missing: {rel_path}",
                    files=tuple(file_results),
                )
            actual_hash = self._compute_file_hash(file_path)
            matched = hmac.compare_digest(actual_hash, declared_hash)
            file_results.append(
                BundleFileResult(
                    path=rel_path,
                    declared_hash=declared_hash,
                    actual_hash=actual_hash,
                    matched=matched,
                )
            )
            _LOGGER.debug(
                "bundle=%s file checked: path=%s declared_prefix=%s "
                "actual_prefix=%s matched=%s",
                manifest.bundle_id,
                rel_path,
                _hash_prefix(declared_hash),
                _hash_prefix(actual_hash),
                matched,
            )

        if any(not result.matched for result in file_results):
            _LOGGER.info(
                "bundle=%s verification failed: category=%s file_count=%d",
                manifest.bundle_id,
                REASON_HASH_MISMATCH,
                len(file_results),
            )
            return _build_failure(
                manifest.bundle_id,
                manifest.manifest_version,
                REASON_HASH_MISMATCH,
                "one or more declared file hashes did not match",
                files=tuple(file_results),
            )

        entry_points_checked: list[str] = []
        for entry_point in manifest.entry_points:
            entry_points_checked.append(entry_point)
            entry_path = bundle_path / entry_point
            if not entry_path.is_file():
                _LOGGER.info(
                    "bundle=%s verification failed: category=%s entry_point_count=%d",
                    manifest.bundle_id,
                    REASON_ENTRY_POINT_MISSING,
                    len(entry_points_checked),
                )
                return _build_failure(
                    manifest.bundle_id,
                    manifest.manifest_version,
                    REASON_ENTRY_POINT_MISSING,
                    f"entry point is missing: {entry_point}",
                    files=tuple(file_results),
                    entry_points_checked=tuple(entry_points_checked),
                )
            if entry_point not in manifest.files:
                _LOGGER.info(
                    "bundle=%s verification failed: category=%s entry_point_count=%d",
                    manifest.bundle_id,
                    REASON_ENTRY_POINT_NOT_DECLARED,
                    len(entry_points_checked),
                )
                return _build_failure(
                    manifest.bundle_id,
                    manifest.manifest_version,
                    REASON_ENTRY_POINT_NOT_DECLARED,
                    f"entry point is not declared in files: {entry_point}",
                    files=tuple(file_results),
                    entry_points_checked=tuple(entry_points_checked),
                )

        signature_verified = False
        if manifest.signature_scheme == "none":
            signature_verified = True
        elif manifest.signature_scheme == "hmac-sha256":
            if signature_key is None:
                _LOGGER.info(
                    "bundle=%s verification failed: category=%s",
                    manifest.bundle_id,
                    REASON_SIGNATURE_REQUIRED,
                )
                return _build_failure(
                    manifest.bundle_id,
                    manifest.manifest_version,
                    REASON_SIGNATURE_REQUIRED,
                    "signature key required for hmac-sha256 scheme",
                    files=tuple(file_results),
                    entry_points_checked=tuple(entry_points_checked),
                )
            canonical = self._canonical_manifest_bytes(manifest)
            expected = hmac.new(signature_key, canonical, hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected, manifest.signature):
                _LOGGER.info(
                    "bundle=%s verification failed: category=%s",
                    manifest.bundle_id,
                    REASON_SIGNATURE_INVALID,
                )
                return _build_failure(
                    manifest.bundle_id,
                    manifest.manifest_version,
                    REASON_SIGNATURE_INVALID,
                    "signature did not match canonical manifest",
                    files=tuple(file_results),
                    entry_points_checked=tuple(entry_points_checked),
                )
            signature_verified = True
        else:  # pragma: no cover - validated by SkillBundleManifest.from_mapping
            _LOGGER.info(
                "bundle=%s verification failed: category=%s",
                manifest.bundle_id,
                REASON_SIGNATURE_SCHEME_UNSUPPORTED,
            )
            return _build_failure(
                manifest.bundle_id,
                manifest.manifest_version,
                REASON_SIGNATURE_SCHEME_UNSUPPORTED,
                f"signature scheme {manifest.signature_scheme!r} is not supported",
                files=tuple(file_results),
                entry_points_checked=tuple(entry_points_checked),
            )

        _LOGGER.info(
            "bundle=%s verified: valid=True file_count=%d entry_point_count=%d",
            manifest.bundle_id,
            len(file_results),
            len(entry_points_checked),
        )
        return BundleVerificationResult(
            bundle_id=manifest.bundle_id,
            manifest_version=manifest.manifest_version,
            valid=True,
            reason="",
            message="",
            files=tuple(file_results),
            entry_points_checked=tuple(entry_points_checked),
            signature_verified=signature_verified,
        )

    def _compute_file_hash(self, path: Path) -> str:
        """Return SHA-256 hex digest of file contents.

        Args:
            path: File to hash.

        Returns:
            Lowercase SHA-256 hex digest of the file contents.
        """

        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _canonical_manifest_bytes(self, manifest: SkillBundleManifest) -> bytes:
        """Return canonical JSON encoding for signature verification.

        Uses sorted keys and no extra whitespace for deterministic output.
        Excludes the signature field itself from the canonical form.

        Args:
            manifest: Manifest to canonicalize.

        Returns:
            UTF-8 encoded canonical JSON bytes excluding the signature field.
        """

        payload = {
            "manifest_version": manifest.manifest_version,
            "bundle_id": manifest.bundle_id,
            "entry_points": list(manifest.entry_points),
            "files": dict(manifest.files),
            "signature_scheme": manifest.signature_scheme,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )


def verify_bundle(
    bundle_dir: str | Path,
    *,
    signature_key: bytes | None = None,
    supported_versions: frozenset[str] = SUPPORTED_MANIFEST_VERSIONS,
) -> BundleVerificationResult:
    """Verify a skill bundle and return a deterministic result.

    Convenience wrapper around :meth:`BundleVerifier.verify`. Performs no
    network calls and never logs raw bundle contents.

    Args:
        bundle_dir: Path to the bundle directory containing ``manifest.json``.
        signature_key: Optional HMAC-SHA256 key for signed bundles.
        supported_versions: Override the set of accepted manifest versions.

    Returns:
        Deterministic verification result with structured failure categories.
    """

    return BundleVerifier(supported_versions=supported_versions).verify(
        bundle_dir,
        signature_key=signature_key,
    )


__all__ = [
    "HASH_ALGORITHM",
    "HASH_PREFIX_LENGTH",
    "REASON_ENTRY_POINT_MISSING",
    "REASON_ENTRY_POINT_NOT_DECLARED",
    "REASON_FILE_MISSING",
    "REASON_HASH_MISMATCH",
    "REASON_MANIFEST_MALFORMED",
    "REASON_MANIFEST_VERSION_UNSUPPORTED",
    "REASON_SIGNATURE_INVALID",
    "REASON_SIGNATURE_REQUIRED",
    "REASON_SIGNATURE_SCHEME_UNSUPPORTED",
    "SUPPORTED_MANIFEST_VERSIONS",
    "BundleFileResult",
    "BundleVerificationResult",
    "BundleVerifier",
    "SkillBundleManifest",
    "verify_bundle",
]
