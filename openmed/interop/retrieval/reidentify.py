"""Encrypted mapping storage and authorized gateway re-identification.

The mapping vault is deliberately separate from the redacted retrieval index.
Only :class:`AuthorizedReidentifier` opens mappings, and it passes them to an
explicit privacy-gateway proxy after checking the principal allow-list.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import tempfile
import threading
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

VAULT_SCHEMA_VERSION = 1
VAULT_ENCRYPTION_SCHEME = "hmac-sha256-stream-xor+hmac-sha256"
DOCUMENT_KEY_SCHEME = "hmac-sha256"
PLACEHOLDER_PATTERN = re.compile(r"<<OPENMED_PHI_[A-Z0-9_]+_[0-9A-F]{8,64}_[0-9]{6,}>>")

_DOCUMENT_KEY_PATTERN = re.compile(r"^hmac-sha256:[0-9a-f]{64}$")
_PLACEHOLDER_CANDIDATE_PATTERN = re.compile(r"<<OPENMED_PHI_[^<>\s]+>>")
_PLACEHOLDER_FRAGMENT_PATTERN = re.compile(r"OPENMED[_-]PHI", re.IGNORECASE)
_NONCE_BYTES = 16
_ZERO_HASH = "0" * 64


class GatewayProxy(Protocol):
    """Explicit privacy-gateway proxy consumed by re-identification."""

    def reidentify(
        self,
        text: str,
        *,
        mapping: Mapping[str, str],
        principal: str,
        request_id: str,
    ) -> str:
        """Resolve approved placeholders inside the gateway boundary."""

        ...


class ReidentificationError(ValueError):
    """Base class for fail-closed re-identification errors."""

    reason_code = "reidentification_failed"


class UnauthorizedPrincipalError(ReidentificationError):
    """Raised when a principal is absent from the configured allow-list."""

    reason_code = "principal_not_allowed"


class UnknownDocumentError(ReidentificationError):
    """Raised when an encrypted mapping does not exist for a document key."""

    reason_code = "mapping_not_found"


class InvalidDocumentKeyError(ReidentificationError):
    """Raised when a caller supplies a non-HMAC document reference."""

    reason_code = "invalid_document_key"


class UnknownPlaceholderError(ReidentificationError):
    """Raised when text contains a placeholder outside the selected documents."""

    reason_code = "unknown_placeholder"


class PrivacyGatewayProxyError(RuntimeError):
    """Raised when the explicit privacy-gateway proxy fails safely."""

    reason_code = "privacy_gateway_proxy_failed"


class MappingVaultIntegrityError(ValueError):
    """Raised when an encrypted mapping vault fails validation."""


class EncryptedMappingVault:
    """Store document-scoped reversible mappings with authenticated encryption.

    Document identifiers are HMACed before they become vault or index keys. The
    optional file contains ciphertext, nonces, authentication tags, and safe
    document keys only. Plaintext mappings exist solely in process memory.
    """

    def __init__(
        self,
        secret: str | bytes,
        *,
        path: str | Path | None = None,
    ) -> None:
        self._root_key = _derive_root_key(secret)
        self._encryption_key = hmac.new(
            self._root_key,
            b"openmed-retrieval-vault:encryption:v1",
            hashlib.sha256,
        ).digest()
        self._path = Path(path) if path is not None else None
        self._mappings: dict[str, dict[str, str]] = {}
        self._lock = threading.RLock()
        if self._path is not None and self._path.exists():
            self._load()

    @classmethod
    def in_memory(cls, secret: str | bytes) -> "EncryptedMappingVault":
        """Create a vault whose encrypted mappings are not persisted."""

        return cls(secret)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        secret: str | bytes,
    ) -> "EncryptedMappingVault":
        """Open or create an encrypted file-backed vault."""

        return cls(secret, path=path)

    def document_key(self, document_id: str) -> str:
        """Return the stable, PHI-safe HMAC key for a caller document id."""

        if not isinstance(document_id, str) or not document_id:
            raise ValueError("document_id must be a non-empty string")
        digest = hmac.new(
            self._root_key,
            b"openmed-retrieval-vault:document:v1\x00" + document_id.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"{DOCUMENT_KEY_SCHEME}:{digest}"

    def store(self, document_id: str, mapping: Mapping[str, str]) -> str:
        """Encrypt and store a placeholder mapping for one document."""

        document_key = self.document_key(document_id)
        clean_mapping = _validate_mapping(mapping)
        _validate_mapping_document_binding(document_key, clean_mapping)
        with self._lock:
            self._mappings[document_key] = clean_mapping
            self._save()
        return document_key

    def contains(self, document_key: str) -> bool:
        """Return whether a mapping exists without exposing its contents."""

        _validate_document_key(document_key)
        with self._lock:
            return document_key in self._mappings

    def hash_principal(self, principal: str) -> str:
        """Return a stable HMAC reference suitable for PHI-free audit records."""

        if not isinstance(principal, str):
            raise TypeError("principal must be a string")
        digest = hmac.new(
            self._root_key,
            b"openmed-retrieval-vault:principal:v1\x00" + principal.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"{DOCUMENT_KEY_SCHEME}:{digest}"

    def _mapping_for_gateway(self, document_key: str) -> dict[str, str]:
        """Return one mapping for the authorized gateway path only."""

        _validate_document_key(document_key)
        with self._lock:
            try:
                return dict(self._mappings[document_key])
            except KeyError as exc:
                raise UnknownDocumentError(
                    "No reversible mapping exists for the requested document"
                ) from exc

    def _save(self) -> None:
        if self._path is None:
            return
        payload = {
            "schema_version": VAULT_SCHEMA_VERSION,
            "encryption_scheme": VAULT_ENCRYPTION_SCHEME,
            "entries": [
                self._encrypted_entry(document_key, mapping)
                for document_key, mapping in sorted(self._mappings.items())
            ],
        }
        _atomic_write_json(self._path, payload)

    def _load(self) -> None:
        assert self._path is not None
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise TypeError("vault payload must be an object")
            if set(payload) != {
                "schema_version",
                "encryption_scheme",
                "entries",
            }:
                raise ValueError("vault payload contains unsupported fields")
            if payload["schema_version"] != VAULT_SCHEMA_VERSION:
                raise ValueError("unsupported vault schema version")
            if payload["encryption_scheme"] != VAULT_ENCRYPTION_SCHEME:
                raise ValueError("unsupported vault encryption scheme")
            entries = payload["entries"]
            if not isinstance(entries, list):
                raise TypeError("vault entries must be a list")
            loaded: dict[str, dict[str, str]] = {}
            for entry in entries:
                document_key, mapping = self._decrypt_entry(entry)
                if document_key in loaded:
                    raise ValueError("duplicate document key")
                loaded[document_key] = mapping
        except (
            KeyError,
            TypeError,
            ValueError,
            UnicodeError,
            json.JSONDecodeError,
        ) as exc:
            raise MappingVaultIntegrityError(
                "Encrypted mapping vault failed validation"
            ) from exc
        self._mappings = loaded

    def _encrypted_entry(
        self,
        document_key: str,
        mapping: Mapping[str, str],
    ) -> dict[str, str]:
        aad = _entry_aad(document_key)
        plaintext = json.dumps(
            dict(sorted(mapping.items())),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        nonce = os.urandom(_NONCE_BYTES)
        stream = _keystream(self._encryption_key, nonce, len(plaintext))
        ciphertext = _xor_bytes(plaintext, stream)
        tag = _authentication_tag(self._encryption_key, aad, nonce, ciphertext)
        return {
            "document_key": document_key,
            "ciphertext": _b64encode(ciphertext),
            "nonce": _b64encode(nonce),
            "tag": tag,
        }

    def _decrypt_entry(self, raw_entry: Any) -> tuple[str, dict[str, str]]:
        if not isinstance(raw_entry, Mapping):
            raise TypeError("vault entry must be an object")
        if set(raw_entry) != {"document_key", "ciphertext", "nonce", "tag"}:
            raise ValueError("vault entry contains unsupported fields")
        document_key = str(raw_entry["document_key"])
        _validate_document_key(document_key)
        ciphertext = _b64decode(str(raw_entry["ciphertext"]))
        nonce = _b64decode(str(raw_entry["nonce"]))
        if len(nonce) != _NONCE_BYTES:
            raise ValueError("vault nonce has an invalid length")
        expected_tag = _authentication_tag(
            self._encryption_key,
            _entry_aad(document_key),
            nonce,
            ciphertext,
        )
        if not hmac.compare_digest(str(raw_entry["tag"]), expected_tag):
            raise ValueError("vault entry authentication failed")
        stream = _keystream(self._encryption_key, nonce, len(ciphertext))
        plaintext = _xor_bytes(ciphertext, stream).decode("utf-8")
        mapping = json.loads(plaintext)
        if not isinstance(mapping, Mapping):
            raise TypeError("decrypted mapping must be an object")
        clean_mapping = _validate_mapping(mapping)
        _validate_mapping_document_binding(document_key, clean_mapping)
        return document_key, clean_mapping

    def __len__(self) -> int:
        with self._lock:
            return len(self._mappings)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(<{len(self)} encrypted documents>)"


@dataclass(frozen=True)
class ReidentificationAuditRecord:
    """PHI-free record for one allowed, denied, or failed attempt."""

    request_id: str
    timestamp: str
    principal_hash: str
    document_keys: tuple[str, ...]
    status: str
    reason_code: str | None
    placeholder_count: int
    previous_hash: str
    record_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable PHI-free record."""

        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "principal_hash": self.principal_hash,
            "document_keys": list(self.document_keys),
            "status": self.status,
            "reason_code": self.reason_code,
            "placeholder_count": self.placeholder_count,
            "previous_hash": self.previous_hash,
            "record_hash": self.record_hash,
        }


class ReidentificationAuditTrail:
    """Thread-safe, hash-chained audit trail containing no plaintext values."""

    def __init__(self) -> None:
        self._records: list[ReidentificationAuditRecord] = []
        self._lock = threading.RLock()

    def append(
        self,
        *,
        request_id: str,
        principal_hash: str,
        document_keys: Sequence[str],
        status: str,
        reason_code: str | None,
        placeholder_count: int,
    ) -> ReidentificationAuditRecord:
        """Append exactly one PHI-free audit event for an attempted operation."""

        _validate_document_key(principal_hash)
        for document_key in document_keys:
            _validate_document_key(document_key)
        try:
            parsed_request_id = uuid.UUID(request_id)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("audit request_id must be a UUID") from exc
        if str(parsed_request_id) != request_id:
            raise ValueError("audit request_id must use canonical UUID form")
        if status not in {"denied", "failed", "rejected", "succeeded"}:
            raise ValueError("audit status is unsupported")
        if reason_code is not None and re.fullmatch(r"[a-z0-9_]+", reason_code) is None:
            raise ValueError("audit reason_code is unsupported")
        if placeholder_count < 0:
            raise ValueError("audit placeholder_count must be non-negative")
        with self._lock:
            previous_hash = (
                self._records[-1].record_hash if self._records else _ZERO_HASH
            )
            payload = {
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "principal_hash": principal_hash,
                "document_keys": list(document_keys),
                "status": status,
                "reason_code": reason_code,
                "placeholder_count": int(placeholder_count),
                "previous_hash": previous_hash,
            }
            record_hash = hashlib.sha256(
                json.dumps(
                    payload,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            record = ReidentificationAuditRecord(
                request_id=request_id,
                timestamp=str(payload["timestamp"]),
                principal_hash=principal_hash,
                document_keys=tuple(document_keys),
                status=status,
                reason_code=reason_code,
                placeholder_count=int(placeholder_count),
                previous_hash=previous_hash,
                record_hash=record_hash,
            )
            self._records.append(record)
            return record

    @property
    def records(self) -> tuple[ReidentificationAuditRecord, ...]:
        """Return an immutable snapshot of audit records."""

        with self._lock:
            return tuple(self._records)

    def to_json(self) -> str:
        """Serialize records for inspection without plaintext identifiers."""

        return json.dumps(
            [record.to_dict() for record in self.records],
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    def contains_plaintext(self, values: Sequence[str]) -> bool:
        """Return whether a supplied plaintext value appears in serialized audit."""

        payload = self.to_json()
        return any(value and value in payload for value in values)

    def verify(self) -> bool:
        """Return whether the complete audit hash chain is intact."""

        previous_hash = _ZERO_HASH
        for record in self.records:
            if record.previous_hash != previous_hash:
                return False
            payload = record.to_dict()
            persisted_hash = str(payload.pop("record_hash"))
            expected_hash = hashlib.sha256(
                json.dumps(
                    payload,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            if not hmac.compare_digest(persisted_hash, expected_hash):
                return False
            previous_hash = persisted_hash
        return True


class AuthorizedReidentifier:
    """Resolve placeholders through a gateway proxy for allow-listed principals."""

    def __init__(
        self,
        *,
        vault: EncryptedMappingVault,
        gateway_proxy: GatewayProxy,
        allowed_principals: Sequence[str],
        audit_trail: ReidentificationAuditTrail | None = None,
    ) -> None:
        if isinstance(allowed_principals, (str, bytes)):
            raise ValueError("allowed_principals must be a sequence of strings")
        principals = frozenset(allowed_principals)
        if not principals or any(
            not isinstance(principal, str) or not principal for principal in principals
        ):
            raise ValueError("allowed_principals must contain non-empty strings")
        self.vault = vault
        self.gateway_proxy = gateway_proxy
        self.allowed_principals = principals
        self.audit_trail = audit_trail or ReidentificationAuditTrail()

    def reidentify(
        self,
        text: str,
        *,
        document_keys: Sequence[str],
        principal: str,
    ) -> str:
        """Authorize, resolve through the proxy, and audit exactly one attempt."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        request_id = str(uuid.uuid4())
        principal_hash = self.vault.hash_principal(principal)
        unique_document_keys, invalid_document_key = _safe_document_keys(document_keys)
        placeholder_count = len(_PLACEHOLDER_CANDIDATE_PATTERN.findall(text))
        status = "denied"
        reason_code: str | None = UnauthorizedPrincipalError.reason_code

        try:
            if principal not in self.allowed_principals:
                raise UnauthorizedPrincipalError(
                    "Principal is not authorized for re-identification"
                )
            if invalid_document_key:
                status = "rejected"
                reason_code = InvalidDocumentKeyError.reason_code
                raise InvalidDocumentKeyError(
                    "Document keys must be OpenMed HMAC references"
                )
            mapping = self._combined_mapping(unique_document_keys)
            candidates = _PLACEHOLDER_CANDIDATE_PATTERN.findall(text)
            unknown = [
                token
                for token in candidates
                if PLACEHOLDER_PATTERN.fullmatch(token) is None or token not in mapping
            ]
            without_candidates = _PLACEHOLDER_CANDIDATE_PATTERN.sub("", text)
            if _PLACEHOLDER_FRAGMENT_PATTERN.search(without_candidates):
                unknown.append("<mangled-placeholder>")
            if unknown:
                status = "rejected"
                reason_code = UnknownPlaceholderError.reason_code
                raise UnknownPlaceholderError(
                    "Text contains a placeholder outside the selected documents"
                )
            result = self._call_gateway_proxy(
                text,
                mapping=mapping,
                principal=principal,
                request_id=request_id,
            )
            if _PLACEHOLDER_FRAGMENT_PATTERN.search(result):
                raise PrivacyGatewayProxyError(
                    "Privacy-gateway proxy left unresolved placeholders"
                )
            status = "succeeded"
            reason_code = None
            return result
        except (UnauthorizedPrincipalError, UnknownPlaceholderError):
            raise
        except UnknownDocumentError:
            status = "rejected"
            reason_code = UnknownDocumentError.reason_code
            raise
        except ReidentificationError as exc:
            status = "rejected"
            reason_code = exc.reason_code
            raise
        except PrivacyGatewayProxyError:
            status = "failed"
            reason_code = PrivacyGatewayProxyError.reason_code
            raise
        except Exception as exc:
            status = "failed"
            reason_code = PrivacyGatewayProxyError.reason_code
            raise PrivacyGatewayProxyError(
                "Privacy-gateway proxy failed during re-identification"
            ) from exc
        finally:
            self.audit_trail.append(
                request_id=request_id,
                principal_hash=principal_hash,
                document_keys=unique_document_keys,
                status=status,
                reason_code=reason_code,
                placeholder_count=placeholder_count,
            )

    def _combined_mapping(self, document_keys: Sequence[str]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for document_key in document_keys:
            document_mapping = self.vault._mapping_for_gateway(document_key)
            for placeholder, original in document_mapping.items():
                current = mapping.get(placeholder)
                if current is not None and current != original:
                    raise ReidentificationError(
                        "Selected documents contain conflicting placeholders"
                    )
                mapping[placeholder] = original
        return mapping

    def _call_gateway_proxy(
        self,
        text: str,
        *,
        mapping: Mapping[str, str],
        principal: str,
        request_id: str,
    ) -> str:
        method = getattr(self.gateway_proxy, "reidentify", None)
        if not callable(method):
            raise PrivacyGatewayProxyError("gateway_proxy must expose reidentify()")
        result = method(
            text,
            mapping=mapping,
            principal=principal,
            request_id=request_id,
        )
        if not isinstance(result, str):
            raise PrivacyGatewayProxyError("Privacy-gateway proxy must return a string")
        return result


def _validate_mapping(mapping: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(mapping, Mapping):
        raise TypeError("mapping must be a mapping")
    clean: dict[str, str] = {}
    for placeholder, original in mapping.items():
        if (
            not isinstance(placeholder, str)
            or PLACEHOLDER_PATTERN.fullmatch(placeholder) is None
        ):
            raise ValueError("mapping keys must be OpenMed retrieval placeholders")
        if not isinstance(original, str) or not original:
            raise ValueError("mapping values must be non-empty strings")
        clean[placeholder] = original
    return clean


def _validate_mapping_document_binding(
    document_key: str,
    mapping: Mapping[str, str],
) -> None:
    digest = document_key.rsplit(":", 1)[-1][:16].upper()
    marker = f"_{digest}_"
    if any(marker not in placeholder for placeholder in mapping):
        raise ValueError("mapping placeholder is not bound to its document key")


def _validate_document_key(document_key: str) -> None:
    if (
        not isinstance(document_key, str)
        or _DOCUMENT_KEY_PATTERN.fullmatch(document_key) is None
    ):
        raise ValueError("document_key must be an OpenMed HMAC reference")


def _safe_document_keys(
    document_keys: Sequence[str],
) -> tuple[tuple[str, ...], bool]:
    if isinstance(document_keys, (str, bytes)):
        return (), True
    safe: list[str] = []
    seen: set[str] = set()
    invalid = False
    for document_key in document_keys:
        if (
            not isinstance(document_key, str)
            or _DOCUMENT_KEY_PATTERN.fullmatch(document_key) is None
        ):
            invalid = True
            continue
        if document_key not in seen:
            safe.append(document_key)
            seen.add(document_key)
    return tuple(safe), invalid


def _derive_root_key(secret: str | bytes) -> bytes:
    if isinstance(secret, str):
        material = secret.encode("utf-8")
    elif isinstance(secret, bytes):
        material = secret
    else:
        raise TypeError("vault secret must be a string or bytes")
    if len(material) < 16:
        raise ValueError("vault secret must contain at least 16 bytes")
    return hmac.new(
        material,
        b"openmed-retrieval-vault:root:v1",
        hashlib.sha256,
    ).digest()


def _entry_aad(document_key: str) -> bytes:
    return json.dumps(
        {
            "document_key": document_key,
            "encryption_scheme": VAULT_ENCRYPTION_SCHEME,
            "schema_version": VAULT_SCHEMA_VERSION,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _authentication_tag(
    key: bytes,
    aad: bytes,
    nonce: bytes,
    ciphertext: bytes,
) -> str:
    mac_key = hmac.new(
        key,
        b"openmed-retrieval-vault:authentication:v1",
        hashlib.sha256,
    ).digest()
    return hmac.new(mac_key, aad + nonce + ciphertext, hashlib.sha256).hexdigest()


def _keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    blocks: list[bytes] = []
    generated = 0
    counter = 0
    while generated < length:
        counter += 1
        block = hmac.new(
            key,
            b"openmed-retrieval-vault:stream:v1" + nonce + counter.to_bytes(8, "big"),
            hashlib.sha256,
        ).digest()
        blocks.append(block)
        generated += len(block)
    return b"".join(blocks)[:length]


def _xor_bytes(left: bytes, right: bytes) -> bytes:
    return bytes(first ^ second for first, second in zip(left, right))


def _b64encode(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _b64decode(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"), validate=True)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            os.chmod(temporary_path, 0o600)
            json.dump(
                payload,
                handle,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        os.chmod(path, 0o600)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


__all__ = [
    "AuthorizedReidentifier",
    "DOCUMENT_KEY_SCHEME",
    "EncryptedMappingVault",
    "GatewayProxy",
    "InvalidDocumentKeyError",
    "MappingVaultIntegrityError",
    "PLACEHOLDER_PATTERN",
    "PrivacyGatewayProxyError",
    "ReidentificationAuditRecord",
    "ReidentificationAuditTrail",
    "ReidentificationError",
    "UnauthorizedPrincipalError",
    "UnknownDocumentError",
    "UnknownPlaceholderError",
    "VAULT_ENCRYPTION_SCHEME",
    "VAULT_SCHEMA_VERSION",
]
