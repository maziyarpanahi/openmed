"""Allowlisted content-addressed artifact storage through ``fsspec``.

The backend is local-only unless a caller explicitly allowlists another
protocol.  Object keys are derived exclusively from verified SHA-256 digests;
callers never supply relative object paths.
"""

from __future__ import annotations

import os
import posixpath
import re
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

from openmed.clinical.journey_contracts import ClinicalArtifact, sha256_digest

from .protocols import (
    AllowAllStoragePolicy,
    StoragePolicy,
    StoreResult,
    StoreState,
)

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_PROTOCOL_RE = re.compile(r"^[a-z][a-z0-9+.-]{0,31}$")


@dataclass(frozen=True, slots=True)
class ObjectStoreNamespace:
    """One bounded object namespace and its explicitly allowed protocols.

    ``storage_options`` is intentionally excluded from representations because
    callers may place backend credentials there.  OpenMed never logs or
    persists those values.
    """

    root_uri: str
    allowed_protocols: frozenset[str] = field(
        default_factory=lambda: frozenset({"file"})
    )
    storage_options: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.root_uri, str) or not self.root_uri.strip():
            raise ValueError("object namespace root must be a non-empty URI")
        protocols = frozenset(
            protocol.casefold() for protocol in self.allowed_protocols
        )
        if not protocols or any(
            _PROTOCOL_RE.fullmatch(protocol) is None for protocol in protocols
        ):
            raise ValueError("allowed object protocols must be controlled names")
        actual = _uri_protocol(self.root_uri)
        if actual not in protocols:
            raise ValueError("object namespace protocol is not explicitly allowed")
        parsed = urlsplit(self.root_uri)
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("credentials must not be embedded in an object URI")
        if parsed.query or parsed.fragment:
            raise ValueError("object namespace URI cannot contain query or fragment")
        path = parsed.path if parsed.scheme else self.root_uri
        if ".." in PurePosixPath(path.replace("\\", "/")).parts:
            raise ValueError("object namespace cannot contain parent traversal")
        if not isinstance(self.storage_options, Mapping):
            raise ValueError("object storage options must be a mapping")
        object.__setattr__(self, "allowed_protocols", protocols)
        object.__setattr__(
            self,
            "storage_options",
            MappingProxyType(dict(self.storage_options)),
        )

    @property
    def protocol(self) -> str:
        """Return the normalized configured protocol."""

        return _uri_protocol(self.root_uri)


class FsspecArtifactStore:
    """Content-addressed artifacts inside one allowlisted object namespace."""

    def __init__(
        self,
        namespace: ObjectStoreNamespace | str | Path,
        *,
        allowed_protocols: frozenset[str] | None = None,
        storage_options: Mapping[str, Any] | None = None,
        policy: StoragePolicy | None = None,
    ) -> None:
        if isinstance(namespace, ObjectStoreNamespace):
            if allowed_protocols is not None or storage_options is not None:
                raise ValueError(
                    "namespace options must be configured on ObjectStoreNamespace"
                )
            configured = namespace
        else:
            configured = ObjectStoreNamespace(
                root_uri=str(namespace),
                allowed_protocols=(
                    frozenset({"file"})
                    if allowed_protocols is None
                    else allowed_protocols
                ),
                storage_options={} if storage_options is None else storage_options,
            )
        self.namespace = configured
        self.policy = policy or AllowAllStoragePolicy()
        try:
            fsspec = _import_fsspec()
            filesystem, root = fsspec.core.url_to_fs(
                configured.root_uri,
                **dict(configured.storage_options),
            )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError):
            raise RuntimeError("object artifact store cannot be initialized") from None

        actual_protocol = _filesystem_protocol(filesystem)
        if actual_protocol not in configured.allowed_protocols:
            raise ValueError("resolved object protocol is not explicitly allowed")
        self._filesystem = filesystem
        self._protocol = actual_protocol
        self._root = self._normalize_root(str(root))
        try:
            self._filesystem.makedirs(self._root, exist_ok=True)
            if self._protocol == "file":
                os.chmod(self._root, 0o700)
        except (OSError, RuntimeError, ValueError):
            raise RuntimeError("object namespace cannot be initialized") from None

    @property
    def protocol(self) -> str:
        """Return the resolved backend protocol."""

        return self._protocol

    def put_bytes(
        self,
        artifact: ClinicalArtifact,
        content: bytes,
    ) -> StoreResult[ClinicalArtifact]:
        """Persist verified bytes under a deterministic digest-derived key."""

        if not self.policy.allows("write", "artifact"):
            return StoreResult.outcome(StoreState.DENIED, "policy_denied")
        if not isinstance(content, bytes):
            return StoreResult.outcome(StoreState.FAILURE, "invalid_content_type")
        if len(content) != artifact.byte_size:
            return StoreResult.outcome(StoreState.CONFLICT, "artifact_size_mismatch")
        if sha256_digest(content) != artifact.content_hash:
            return StoreResult.outcome(StoreState.CONFLICT, "artifact_hash_mismatch")

        target_result = self._key_for_digest(artifact.content_hash)
        if target_result is None:
            return StoreResult.outcome(StoreState.FAILURE, "invalid_content_hash")
        existing = self._read_verified(artifact.content_hash)
        if existing.ok:
            return StoreResult.success(artifact, created=False)
        if existing.state is not StoreState.UNKNOWN:
            return StoreResult.outcome(
                existing.state,
                existing.code or "artifact_read_failed",
            )

        parent = posixpath.dirname(target_result)
        temporary = self._safe_join(
            parent,
            f".openmed-object-{secrets.token_hex(16)}.tmp",
        )
        if temporary is None:
            return StoreResult.outcome(StoreState.FAILURE, "artifact_path_unsafe")
        try:
            self._filesystem.makedirs(parent, exist_ok=True)
            if self._protocol == "file":
                if not self._local_path_is_safe(parent):
                    return StoreResult.outcome(
                        StoreState.FAILURE,
                        "artifact_path_unsafe",
                    )
                self._secure_local_directories(parent)
            with self._filesystem.open(temporary, "wb") as stream:
                stream.write(content)
            if self._protocol == "file":
                os.chmod(temporary, 0o600)
            with self._filesystem.open(temporary, "rb") as stream:
                staged = stream.read()
            if sha256_digest(staged) != artifact.content_hash:
                self._remove_quietly(temporary)
                return StoreResult.outcome(
                    StoreState.FAILURE,
                    "artifact_verify_failed",
                )
            if self._filesystem.exists(target_result):
                self._remove_quietly(temporary)
                verified = self._read_verified(artifact.content_hash)
                if not verified.ok:
                    return StoreResult.outcome(
                        verified.state,
                        verified.code or "artifact_verify_failed",
                    )
                return StoreResult.success(artifact, created=False)
            self._filesystem.mv(temporary, target_result)
            if self._protocol == "file":
                os.chmod(target_result, 0o600)
        except (OSError, RuntimeError, TypeError, ValueError):
            self._remove_quietly(temporary)
            return StoreResult.outcome(StoreState.FAILURE, "artifact_write_failed")

        verified = self._read_verified(artifact.content_hash)
        if not verified.ok:
            self._remove_quietly(target_result)
            return StoreResult.outcome(StoreState.FAILURE, "artifact_verify_failed")
        return StoreResult.success(artifact, created=True)

    def get_bytes(self, content_hash: str) -> StoreResult[bytes]:
        """Read and verify one digest-derived object."""

        if not self.policy.allows("read", "artifact"):
            return StoreResult.outcome(StoreState.DENIED, "policy_denied")
        return self._read_verified(content_hash)

    def discard_if_created(self, content_hash: str) -> None:
        """Remove a just-created object after its metadata transaction fails."""

        target = self._key_for_digest(content_hash)
        if target is not None:
            self._remove_quietly(target)

    def _read_verified(self, content_hash: str) -> StoreResult[bytes]:
        target = self._key_for_digest(content_hash)
        if target is None:
            return StoreResult.outcome(StoreState.FAILURE, "invalid_content_hash")
        if self._protocol == "file" and not self._local_path_is_safe(target):
            return StoreResult.outcome(StoreState.FAILURE, "artifact_path_unsafe")
        try:
            if not self._filesystem.exists(target):
                return StoreResult.outcome(StoreState.UNKNOWN, "artifact_not_found")
            with self._filesystem.open(target, "rb") as stream:
                content = stream.read()
        except (OSError, RuntimeError, TypeError, ValueError):
            return StoreResult.outcome(StoreState.FAILURE, "artifact_read_failed")
        if not isinstance(content, bytes):
            return StoreResult.outcome(StoreState.FAILURE, "artifact_read_failed")
        if sha256_digest(content) != content_hash:
            return StoreResult.outcome(StoreState.FAILURE, "artifact_integrity_failed")
        return StoreResult.success(content)

    def _normalize_root(self, root: str) -> str:
        if self._protocol == "file":
            resolved = Path(root).expanduser().resolve(strict=False)
            if resolved == Path(resolved.anchor):
                raise ValueError("local object namespace root must be bounded")
            return str(resolved)
        normalized = posixpath.normpath(root.replace("\\", "/"))
        if normalized in {"", ".", "/"}:
            raise ValueError("object namespace root must be bounded")
        if normalized == ".." or normalized.startswith("../"):
            raise ValueError("object namespace cannot escape its configured root")
        return normalized.rstrip("/")

    def _key_for_digest(self, content_hash: str) -> str | None:
        if (
            not isinstance(content_hash, str)
            or _DIGEST_RE.fullmatch(content_hash) is None
        ):
            return None
        digest = content_hash.removeprefix("sha256:")
        return self._safe_join(self._root, "blobs", "sha256", digest[:2], digest)

    def _safe_join(self, root: str, *parts: str) -> str | None:
        normalized_root = posixpath.normpath(root.replace("\\", "/"))
        candidate = posixpath.normpath(posixpath.join(normalized_root, *parts))
        try:
            if posixpath.commonpath((normalized_root, candidate)) != normalized_root:
                return None
        except ValueError:
            return None
        return candidate

    def _remove_quietly(self, key: str) -> None:
        try:
            if self._filesystem.exists(key):
                self._filesystem.rm(key)
        except (OSError, RuntimeError, TypeError, ValueError):
            return

    def _secure_local_directories(self, directory: str) -> None:
        root = Path(self._root)
        target = Path(directory)
        try:
            relative = target.relative_to(root)
        except ValueError:
            raise RuntimeError("object directory escaped its namespace") from None
        current = root
        os.chmod(current, 0o700)
        for part in relative.parts:
            current /= part
            os.chmod(current, 0o700)

    def _local_path_is_safe(self, key: str) -> bool:
        root = Path(self._root)
        candidate = Path(key)
        try:
            relative = candidate.relative_to(root)
        except ValueError:
            return False
        current = root
        if current.is_symlink():
            return False
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                return False
        return True


def _uri_protocol(uri: str) -> str:
    parsed = urlsplit(uri)
    if not parsed.scheme or (len(parsed.scheme) == 1 and uri[1:3] in {":/", ":\\"}):
        return "file"
    return parsed.scheme.casefold()


def _filesystem_protocol(filesystem: Any) -> str:
    value = getattr(filesystem, "protocol", None)
    if isinstance(value, (tuple, list)):
        value = value[0] if value else None
    if not isinstance(value, str) or _PROTOCOL_RE.fullmatch(value.casefold()) is None:
        raise ValueError("resolved object filesystem has no controlled protocol")
    protocol = value.casefold()
    return "file" if protocol in {"file", "local"} else protocol


def _import_fsspec() -> Any:
    try:
        return import_module("fsspec")
    except ImportError as exc:
        raise ImportError(
            "Object artifact storage requires the optional fsspec dependency."
        ) from exc


__all__ = ["FsspecArtifactStore", "ObjectStoreNamespace"]
