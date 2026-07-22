"""Stable surrogate vaults for cross-document pseudonymization.

The vault stores mappings from ``(canonical_label, lang, text_hash)`` to the
selected surrogate value. ``text_hash`` is an HMAC-SHA256 digest of the source
surface; raw source identifiers are never stored. Persisted vault payloads
encrypt surrogate values under a versioned epoch key so the file at rest does
not reveal replacement identifiers.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import tempfile
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import AbstractSet, Any, Callable

from .labels import normalize_label
from .schemas.span import hmac_text_hash

LEGACY_SCHEMA_VERSION = 1
SCHEMA_VERSION = 2
HMAC_SCHEME = "hmac-sha256"
ENCRYPTION_SCHEME = "hmac-sha256-stream-xor+hmac-sha256"

_EPOCH_PREFIX = "epoch"
_KEY_ID_BYTES = 8
_NONCE_BYTES = 16


@dataclass(frozen=True, order=True)
class SurrogateKey:
    """Privacy-safe key for one source identifier."""

    canonical_label: str
    lang: str
    text_hash: str

    def __post_init__(self) -> None:
        if not self.canonical_label:
            raise ValueError("canonical_label must be non-empty")
        if not self.lang:
            raise ValueError("lang must be non-empty")
        if not self.text_hash.startswith(f"{HMAC_SCHEME}:"):
            raise ValueError(f"text_hash must use {HMAC_SCHEME}")

    def to_dict(self) -> dict[str, str]:
        """Serialize this key to JSON-compatible fields."""

        return {
            "canonical_label": self.canonical_label,
            "lang": self.lang,
            "text_hash": self.text_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SurrogateKey":
        """Deserialize a key from persisted JSON fields."""

        return cls(
            canonical_label=str(payload["canonical_label"]),
            lang=str(payload["lang"]),
            text_hash=str(payload["text_hash"]),
        )


@dataclass(frozen=True)
class SurrogateEntry:
    """Persisted vault entry decrypted into memory."""

    key: SurrogateKey
    surrogate: str
    key_id: str = ""

    def __post_init__(self) -> None:
        if not self.surrogate:
            raise ValueError("surrogate must be non-empty")
        if self.key_id is not None and not isinstance(self.key_id, str):
            raise TypeError("key_id must be a string")

    def to_dict(self) -> dict[str, str]:
        """Serialize this entry to deterministic JSON-compatible fields."""

        payload = {
            **self.key.to_dict(),
            "surrogate": self.surrogate,
        }
        if self.key_id:
            payload["key_id"] = self.key_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SurrogateEntry":
        """Deserialize one legacy plaintext vault entry."""

        allowed = {"canonical_label", "lang", "text_hash", "surrogate", "key_id"}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(
                "surrogate vault entries may not contain fields: "
                + ", ".join(sorted(map(str, unknown)))
            )
        return cls(
            key=SurrogateKey.from_dict(payload),
            surrogate=str(payload["surrogate"]),
            key_id=str(payload.get("key_id") or ""),
        )


@dataclass(frozen=True)
class SurrogateSource:
    """Source descriptor used only in memory for rotation verification."""

    source_text: str = field(repr=False)
    label: str
    lang: str = "en"

    def __post_init__(self) -> None:
        if not self.source_text:
            raise ValueError("source_text must be non-empty")
        if not self.label:
            raise ValueError("label must be non-empty")
        if not self.lang:
            object.__setattr__(self, "lang", "en")


@dataclass(frozen=True)
class VaultConsistencyReport:
    """Privacy-safe result from checking source-to-surrogate consistency."""

    checked: int
    matched: int
    missing: tuple[str, ...] = ()
    mismatched: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        """Return whether every checked source preserved its surrogate."""

        return not self.missing and not self.mismatched and self.checked == self.matched

    def to_dict(self) -> dict[str, Any]:
        """Serialize this report without raw source surfaces."""

        return {
            "checked": self.checked,
            "matched": self.matched,
            "missing": list(self.missing),
            "mismatched": list(self.mismatched),
            "passed": self.passed,
        }


@dataclass(frozen=True)
class VaultRotationResult:
    """Result from an epoch rotation or revocation migration."""

    previous_key_id: str
    current_key_id: str
    migrated_entries: int
    revoked_key_ids: tuple[str, ...] = ()
    consistency: VaultConsistencyReport | None = None


@dataclass(frozen=True)
class _EpochKey:
    sequence: int
    key_id: str
    linkage_key: bytes = field(repr=False)
    encryption_key: bytes = field(repr=False)


class _EpochManager:
    """Derive one-way epoch keys from the caller-supplied local secret."""

    def __init__(
        self,
        secret: str | bytes,
        *,
        current_sequence: int = 1,
        revoked_key_ids: Iterable[str] = (),
    ) -> None:
        self._root_secret = _secret_bytes(secret)
        if current_sequence < 1:
            raise ValueError("current_sequence must be >= 1")
        self.current_sequence = int(current_sequence)
        self.revoked_key_ids = set(map(str, revoked_key_ids))

    @property
    def current_key(self) -> _EpochKey:
        return self.derive(self.current_sequence)

    def derive(self, sequence: int) -> _EpochKey:
        """Return derived material for an epoch sequence."""

        if sequence < 1:
            raise ValueError("sequence must be >= 1")

        seed = hmac.new(
            self._root_secret,
            b"openmed-surrogate-vault:epoch-seed:1",
            hashlib.sha256,
        ).digest()
        for step in range(2, sequence + 1):
            seed = hmac.new(
                seed,
                f"openmed-surrogate-vault:epoch-next:{step}".encode("ascii"),
                hashlib.sha256,
            ).digest()

        key_id_digest = hmac.new(
            seed,
            b"openmed-surrogate-vault:key-id",
            hashlib.sha256,
        ).hexdigest()[: _KEY_ID_BYTES * 2]
        linkage_key = (
            self._root_secret
            if sequence == 1
            else hmac.new(
                seed,
                b"openmed-surrogate-vault:linkage",
                hashlib.sha256,
            ).digest()
        )
        encryption_key = hmac.new(
            seed,
            b"openmed-surrogate-vault:encryption",
            hashlib.sha256,
        ).digest()
        return _EpochKey(
            sequence=sequence,
            key_id=f"{_EPOCH_PREFIX}-{sequence:04d}-{key_id_digest}",
            linkage_key=linkage_key,
            encryption_key=encryption_key,
        )

    def payload(self) -> dict[str, Any]:
        """Return persisted epoch metadata without key material."""

        current = self.current_key
        return {
            "current_epoch": {
                "sequence": current.sequence,
                "key_id": current.key_id,
            },
            "revoked_key_ids": sorted(self.revoked_key_ids),
        }


class InMemorySurrogateStore:
    """In-memory store for surrogate vault entries."""

    def __init__(
        self,
        entries: Iterable[SurrogateEntry] = (),
        *,
        epoch_manager: _EpochManager | None = None,
    ) -> None:
        self._entries: dict[SurrogateKey, SurrogateEntry] = {}
        self._lock = RLock()
        self._epoch_manager = epoch_manager
        for entry in entries:
            self.set(entry.key, entry.surrogate, key_id=entry.key_id)

    @property
    def epoch_manager(self) -> _EpochManager | None:
        """Return the store's epoch manager, if encrypted persistence is active."""

        return self._epoch_manager

    def set_epoch_manager(self, epoch_manager: _EpochManager) -> None:
        """Attach epoch metadata and key IDs to existing in-memory entries."""

        with self._lock:
            self._epoch_manager = epoch_manager
            current_key_id = epoch_manager.current_key.key_id
            self._entries = {
                key: SurrogateEntry(
                    entry.key, entry.surrogate, entry.key_id or current_key_id
                )
                for key, entry in self._entries.items()
            }

    def get(self, key: SurrogateKey) -> str | None:
        """Return the surrogate for ``key``, if present."""

        with self._lock:
            entry = self._entries.get(key)
            return entry.surrogate if entry is not None else None

    def set(
        self,
        key: SurrogateKey,
        surrogate: str,
        *,
        key_id: str | None = None,
    ) -> None:
        """Store ``surrogate`` for ``key``."""

        if not surrogate:
            raise ValueError("surrogate must be non-empty")
        with self._lock:
            current = self._entries.get(key)
            if current is not None and current.surrogate != surrogate:
                raise ValueError("surrogate key already maps to a different value")
            effective_key_id = key_id
            if effective_key_id is None and self._epoch_manager is not None:
                effective_key_id = self._epoch_manager.current_key.key_id
            self._entries[key] = SurrogateEntry(
                key,
                surrogate,
                str(effective_key_id or ""),
            )

    def replace_entries(self, entries: Iterable[SurrogateEntry]) -> None:
        """Replace all entries atomically inside the in-memory store."""

        next_entries: dict[SurrogateKey, SurrogateEntry] = {}
        for entry in entries:
            current = next_entries.get(entry.key)
            if current is not None and current.surrogate != entry.surrogate:
                raise ValueError("surrogate key already maps to a different value")
            next_entries[entry.key] = entry
        with self._lock:
            self._entries = next_entries

    def entries(self) -> tuple[SurrogateEntry, ...]:
        """Return entries sorted by their privacy-safe key."""

        with self._lock:
            return tuple(self._entries[key] for key in sorted(self._entries))

    def used_surrogates(self, *, canonical_label: str, lang: str) -> AbstractSet[str]:
        """Return surrogates already used for the label/language bucket."""

        with self._lock:
            return {
                entry.surrogate
                for key, entry in self._entries.items()
                if key.canonical_label == canonical_label and key.lang == lang
            }

    def to_payload(self) -> dict[str, Any]:
        """Serialize the store without any raw source surfaces."""

        if self._epoch_manager is None:
            return {
                "schema_version": LEGACY_SCHEMA_VERSION,
                "hmac_scheme": HMAC_SCHEME,
                "entries": [entry.to_dict() for entry in self.entries()],
            }
        current_key = self._epoch_manager.current_key
        payload = {
            "schema_version": SCHEMA_VERSION,
            "hmac_scheme": HMAC_SCHEME,
            "encryption_scheme": ENCRYPTION_SCHEME,
            **self._epoch_manager.payload(),
            "entries": [],
        }
        entries_payload: list[dict[str, str]] = []
        for entry in self.entries():
            key_id = entry.key_id or current_key.key_id
            if key_id != current_key.key_id:
                raise ValueError(
                    "surrogate vault contains entries outside the current epoch"
                )
            entries_payload.append(_encrypted_entry_payload(entry, current_key))
        payload["entries"] = entries_payload
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        hmac_secret: str | bytes | None = None,
    ) -> "InMemorySurrogateStore":
        """Load a store from a JSON-compatible payload."""

        schema_version = int(payload.get("schema_version", 0))
        if schema_version == LEGACY_SCHEMA_VERSION:
            return cls._from_legacy_payload(payload, hmac_secret=hmac_secret)
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported surrogate vault schema_version {schema_version!r}"
            )

        hmac_scheme = str(payload.get("hmac_scheme", ""))
        if hmac_scheme != HMAC_SCHEME:
            raise ValueError(f"unsupported surrogate vault hmac_scheme {hmac_scheme!r}")
        encryption_scheme = str(payload.get("encryption_scheme", ""))
        if encryption_scheme != ENCRYPTION_SCHEME:
            raise ValueError(
                f"unsupported surrogate vault encryption_scheme {encryption_scheme!r}"
            )
        if hmac_secret is None:
            raise ValueError("hmac_secret is required to decrypt surrogate vault")

        current_epoch = payload.get("current_epoch")
        if not isinstance(current_epoch, Mapping):
            raise ValueError("surrogate vault current_epoch must be an object")
        manager = _EpochManager(
            hmac_secret,
            current_sequence=int(current_epoch.get("sequence", 0)),
            revoked_key_ids=payload.get("revoked_key_ids") or (),
        )
        current_key = manager.current_key
        persisted_key_id = str(current_epoch.get("key_id", ""))
        if persisted_key_id != current_key.key_id:
            raise ValueError("surrogate vault key_id does not match hmac_secret")

        entries_payload = payload.get("entries", [])
        if not isinstance(entries_payload, list):
            raise ValueError("surrogate vault entries must be a list")

        entries = [
            _decrypt_entry_payload(entry, current_key) for entry in entries_payload
        ]
        return cls(entries, epoch_manager=manager)

    @classmethod
    def _from_legacy_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        hmac_secret: str | bytes | None,
    ) -> "InMemorySurrogateStore":
        hmac_scheme = str(payload.get("hmac_scheme", ""))
        if hmac_scheme != HMAC_SCHEME:
            raise ValueError(f"unsupported surrogate vault hmac_scheme {hmac_scheme!r}")

        entries_payload = payload.get("entries", [])
        if not isinstance(entries_payload, list):
            raise ValueError("surrogate vault entries must be a list")

        manager = _EpochManager(hmac_secret) if hmac_secret is not None else None
        current_key_id = manager.current_key.key_id if manager is not None else ""
        entries = [
            SurrogateEntry.from_dict(entry_payload) for entry_payload in entries_payload
        ]
        if current_key_id:
            entries = [
                SurrogateEntry(
                    entry.key, entry.surrogate, entry.key_id or current_key_id
                )
                for entry in entries
            ]
        return cls(entries, epoch_manager=manager)


class JsonFileSurrogateStore(InMemorySurrogateStore):
    """JSON file-backed surrogate store with deterministic atomic writes."""

    def __init__(
        self,
        path: str | Path,
        entries: Iterable[SurrogateEntry] = (),
        *,
        autosave: bool = True,
        epoch_manager: _EpochManager | None = None,
    ) -> None:
        self.path = Path(path)
        self.autosave = bool(autosave)
        self._hydrating = True
        super().__init__(entries, epoch_manager=epoch_manager)
        self._hydrating = False

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        create: bool = True,
        autosave: bool = True,
        hmac_secret: str | bytes | None = None,
    ) -> "JsonFileSurrogateStore":
        """Load a JSON-backed store from ``path``."""

        vault_path = Path(path)
        if not vault_path.exists():
            if not create:
                raise FileNotFoundError(vault_path)
            manager = _EpochManager(hmac_secret) if hmac_secret is not None else None
            return cls(vault_path, autosave=autosave, epoch_manager=manager)

        try:
            payload = json.loads(vault_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corrupt surrogate vault {vault_path}: {exc}") from exc
        loaded = InMemorySurrogateStore.from_payload(
            payload,
            hmac_secret=hmac_secret,
        )
        return cls(
            vault_path,
            loaded.entries(),
            autosave=autosave,
            epoch_manager=loaded.epoch_manager,
        )

    def set(
        self,
        key: SurrogateKey,
        surrogate: str,
        *,
        key_id: str | None = None,
    ) -> None:
        """Store an entry and save it atomically when autosave is enabled."""

        super().set(key, surrogate, key_id=key_id)
        if getattr(self, "autosave", False) and not getattr(
            self,
            "_hydrating",
            False,
        ):
            self.save()

    def save(self) -> None:
        """Atomically write the deterministic JSON vault payload."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(
            self.to_payload(),
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=str(self.path.parent),
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(data)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
                tmp_path = Path(handle.name)
            os.replace(tmp_path, self.path)
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()


SurrogateFactory = Callable[[int], str]
ConsistencySnapshot = Mapping[str, str]


def _contains_source_surface(candidate: str, source_text: str) -> bool:
    folded_candidate = unicodedata.normalize("NFKC", candidate).casefold()
    folded_source = unicodedata.normalize("NFKC", source_text).casefold()
    if folded_candidate == folded_source:
        return True
    return len(folded_source) >= 4 and folded_source in folded_candidate


def _matches_required_script(candidate: str, required_script: str | None) -> bool:
    if required_script is None:
        return True
    if required_script.casefold() != "latin":
        raise ValueError("required_script currently supports only 'Latin'")
    letters = [char for char in candidate if char.isalpha()]
    return bool(letters) and all(
        "LATIN" in unicodedata.name(char, "") for char in letters
    )


class SurrogateVault:
    """Stable cross-document surrogate mapper.

    Args:
        hmac_secret: Secret used to HMAC source surfaces and derive local epoch
            keys. The secret is required to reopen a vault consistently and is
            never persisted.
        store: Optional backing store. Defaults to an in-memory store.
    """

    def __init__(
        self,
        hmac_secret: str | bytes,
        *,
        store: InMemorySurrogateStore | None = None,
    ) -> None:
        _secret_bytes(hmac_secret)
        self.hmac_secret = hmac_secret
        if store is None:
            store = InMemorySurrogateStore(
                epoch_manager=_EpochManager(hmac_secret),
            )
        elif store.epoch_manager is None:
            store.set_epoch_manager(_EpochManager(hmac_secret))
        self.store = store
        if self.store.epoch_manager is None:
            raise ValueError("surrogate vault store is missing epoch metadata")
        self._epoch_manager = self.store.epoch_manager

    @classmethod
    def in_memory(cls, hmac_secret: str | bytes) -> "SurrogateVault":
        """Create a vault backed by memory only."""

        return cls(hmac_secret)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        hmac_secret: str | bytes,
        create: bool = True,
        autosave: bool = True,
    ) -> "SurrogateVault":
        """Create a vault backed by a deterministic encrypted JSON file."""

        return cls(
            hmac_secret,
            store=JsonFileSurrogateStore.load(
                path,
                create=create,
                autosave=autosave,
                hmac_secret=hmac_secret,
            ),
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"entries={len(self.entries())}, hmac_scheme={HMAC_SCHEME!r}, "
            f"current_key_id={self.current_key_id!r})"
        )

    @property
    def current_key_id(self) -> str:
        """Return the active key epoch identifier."""

        return self._epoch_manager.current_key.key_id

    @property
    def current_epoch_sequence(self) -> int:
        """Return the active epoch sequence."""

        return self._epoch_manager.current_sequence

    @property
    def revoked_key_ids(self) -> tuple[str, ...]:
        """Return revoked epoch identifiers."""

        return tuple(sorted(self._epoch_manager.revoked_key_ids))

    def text_hash(self, source_text: str | bytes) -> str:
        """Return the current-epoch vault HMAC for ``source_text``."""

        return hmac_text_hash(source_text, self._epoch_manager.current_key.linkage_key)

    def key_for(
        self,
        source_text: str,
        *,
        label: str,
        lang: str = "en",
    ) -> SurrogateKey:
        """Build a privacy-safe vault key for one source identifier."""

        return self._key_for_epoch(
            _source(source_text=source_text, label=label, lang=lang),
            self._epoch_manager.current_key,
        )

    def get(
        self,
        source_text: str,
        *,
        label: str,
        lang: str = "en",
    ) -> str | None:
        """Return an existing surrogate for a source identifier."""

        source = _source(source_text=source_text, label=label, lang=lang)
        key = self._key_for_epoch(source, self._epoch_manager.current_key)
        existing = self.store.get(key)
        if existing is not None:
            return existing
        legacy_key = self._legacy_key_for_epoch(source, self._epoch_manager.current_key)
        if legacy_key == key:
            return None
        return self.store.get(legacy_key)

    def get_or_create(
        self,
        source_text: str,
        *,
        label: str,
        lang: str = "en",
        create_surrogate: SurrogateFactory,
        required_script: str | None = None,
    ) -> str:
        """Return a stable surrogate, optionally constrained to one script run."""

        source = _source(source_text=source_text, label=label, lang=lang)
        key = self._key_for_epoch(source, self._epoch_manager.current_key)
        existing = self.store.get(key)
        if existing is not None:
            if not _matches_required_script(existing, required_script):
                raise ValueError("existing surrogate does not satisfy required_script")
            return existing
        legacy_key = self._legacy_key_for_epoch(source, self._epoch_manager.current_key)
        if legacy_key != key:
            existing = self.store.get(legacy_key)
            if existing is not None:
                if not _matches_required_script(existing, required_script):
                    raise ValueError(
                        "existing legacy surrogate does not satisfy required_script"
                    )
                try:
                    self.store.set(key, existing, key_id=self.current_key_id)
                except ValueError:
                    normalized_existing = self.store.get(key)
                    if normalized_existing is None:
                        raise
                    if not _matches_required_script(
                        normalized_existing,
                        required_script,
                    ):
                        raise ValueError(
                            "existing surrogate does not satisfy required_script"
                        )
                    return normalized_existing
                return existing

        used = self.store.used_surrogates(
            canonical_label=key.canonical_label,
            lang=key.lang,
        )
        surrogate = ""
        for attempt in range(10):
            candidate = create_surrogate(attempt)
            if not candidate:
                continue
            if (
                _contains_source_surface(candidate, source_text)
                or candidate in used
                or not _matches_required_script(candidate, required_script)
            ):
                continue
            surrogate = candidate
            break

        if not surrogate:
            suffix = key.text_hash.rsplit(":", 1)[-1][:8]
            base = create_surrogate(0) or key.canonical_label
            if (
                _contains_source_surface(base, source_text)
                or base in used
                or not _matches_required_script(base, required_script)
            ):
                base = key.canonical_label
            surrogate = f"{base}_{suffix}"
            if not _matches_required_script(surrogate, required_script):
                raise ValueError("unable to create a surrogate in required_script")

        self.store.set(key, surrogate, key_id=self.current_key_id)
        return surrogate

    def entries(self) -> tuple[SurrogateEntry, ...]:
        """Return the current sorted vault entries."""

        return self.store.entries()

    def consistency_snapshot(
        self,
        sources: Iterable[SurrogateSource | Mapping[str, Any] | tuple[Any, ...]],
    ) -> dict[str, str]:
        """Capture a privacy-safe source-to-surrogate proof for verification."""

        snapshot: dict[str, str] = {}
        for source in _sources(sources):
            surrogate = self.get(
                source.source_text,
                label=source.label,
                lang=source.lang,
            )
            if surrogate is not None:
                snapshot[self._source_proof(source)] = self._surrogate_proof(surrogate)
        return snapshot

    def verify_consistency(
        self,
        snapshot: ConsistencySnapshot,
        sources: Iterable[SurrogateSource | Mapping[str, Any] | tuple[Any, ...]],
    ) -> VaultConsistencyReport:
        """Verify that sources still resolve to their snapshotted surrogates."""

        missing: list[str] = []
        mismatched: list[str] = []
        matched = 0
        checked = 0
        for source in _sources(sources):
            checked += 1
            proof = self._source_proof(source)
            expected = snapshot.get(proof)
            current = self.get(
                source.source_text,
                label=source.label,
                lang=source.lang,
            )
            if expected is None or current is None:
                missing.append(proof)
            elif self._surrogate_proof(current) != expected:
                mismatched.append(proof)
            else:
                matched += 1
        return VaultConsistencyReport(
            checked=checked,
            matched=matched,
            missing=tuple(missing),
            mismatched=tuple(mismatched),
        )

    def rotate(
        self,
        sources: Iterable[SurrogateSource | Mapping[str, Any] | tuple[Any, ...]] = (),
        *,
        target_sequence: int | None = None,
        revoke_previous: bool = False,
    ) -> VaultRotationResult:
        """Rotate to a new epoch, re-keying and re-encrypting all entries.

        Existing vaults do not persist raw source surfaces, so non-empty vaults
        require a caller-supplied source catalog to rederive current and target
        epoch HMAC keys. The catalog is consumed in memory only and is never
        serialized.
        """

        source_items = _sources(sources)
        previous_sequence = self._epoch_manager.current_sequence
        previous_key = self._epoch_manager.current_key
        next_sequence = target_sequence or previous_sequence + 1
        if next_sequence < previous_sequence:
            raise ValueError("target_sequence cannot move the vault backward")
        if next_sequence == previous_sequence:
            if revoke_previous:
                raise ValueError("revoking the active epoch requires a new epoch")
            snapshot = self.consistency_snapshot(source_items)
            return VaultRotationResult(
                previous_key_id=previous_key.key_id,
                current_key_id=previous_key.key_id,
                migrated_entries=0,
                consistency=self.verify_consistency(snapshot, source_items),
            )

        current_entries = self.entries()
        snapshot = self.consistency_snapshot(source_items)
        target_key = self._epoch_manager.derive(next_sequence)
        remapped = self._remap_entries(
            current_entries,
            sources=source_items,
            from_key=previous_key,
            to_key=target_key,
        )
        old_sequence = self._epoch_manager.current_sequence
        old_revoked = set(self._epoch_manager.revoked_key_ids)

        try:
            self._epoch_manager.current_sequence = next_sequence
            if revoke_previous:
                self._epoch_manager.revoked_key_ids.add(previous_key.key_id)
            self.store.replace_entries(remapped)
            consistency = self.verify_consistency(snapshot, source_items)
            if not consistency.passed:
                raise ValueError("surrogate consistency verification failed")
            self._save_if_file_backed()
        except Exception:
            self._epoch_manager.current_sequence = old_sequence
            self._epoch_manager.revoked_key_ids = old_revoked
            self.store.replace_entries(current_entries)
            raise

        return VaultRotationResult(
            previous_key_id=previous_key.key_id,
            current_key_id=target_key.key_id,
            migrated_entries=len(current_entries),
            revoked_key_ids=tuple(sorted(self._epoch_manager.revoked_key_ids)),
            consistency=consistency,
        )

    def revoke_current_epoch(
        self,
        sources: Iterable[SurrogateSource | Mapping[str, Any] | tuple[Any, ...]] = (),
        *,
        target_sequence: int | None = None,
    ) -> VaultRotationResult:
        """Mark the active epoch compromised and re-encrypt into a new epoch."""

        return self.rotate(
            sources,
            target_sequence=target_sequence,
            revoke_previous=True,
        )

    def revoke_epoch(self, key_id: str) -> VaultRotationResult:
        """Mark an inactive epoch revoked without changing current entries."""

        key_id = str(key_id)
        if key_id == self.current_key_id:
            raise ValueError("use revoke_current_epoch() for the active epoch")
        self._epoch_manager.revoked_key_ids.add(key_id)
        self._save_if_file_backed()
        return VaultRotationResult(
            previous_key_id=self.current_key_id,
            current_key_id=self.current_key_id,
            migrated_entries=0,
            revoked_key_ids=tuple(sorted(self._epoch_manager.revoked_key_ids)),
        )

    def to_payload(self) -> dict[str, Any]:
        """Serialize the vault store without persisting the HMAC secret."""

        return self.store.to_payload()

    def save(self) -> None:
        """Save the backing store if it supports persistence."""

        save = getattr(self.store, "save", None)
        if save is None:
            raise TypeError("this surrogate vault store is not file-backed")
        save()

    def _key_for_epoch(self, source: SurrogateSource, epoch: _EpochKey) -> SurrogateKey:
        effective_lang = str(source.lang or "en")
        canonical_label = normalize_label(str(source.label), effective_lang)
        source_text = _linkage_source_text(source.source_text, canonical_label)
        return SurrogateKey(
            canonical_label=canonical_label,
            lang=effective_lang,
            text_hash=hmac_text_hash(source_text, epoch.linkage_key),
        )

    def _legacy_key_for_epoch(
        self, source: SurrogateSource, epoch: _EpochKey
    ) -> SurrogateKey:
        effective_lang = str(source.lang or "en")
        canonical_label = normalize_label(str(source.label), effective_lang)
        return SurrogateKey(
            canonical_label=canonical_label,
            lang=effective_lang,
            text_hash=hmac_text_hash(source.source_text, epoch.linkage_key),
        )

    def _source_proof(self, source: SurrogateSource) -> str:
        effective_lang = str(source.lang or "en")
        canonical_label = normalize_label(str(source.label), effective_lang)
        source_text = _linkage_source_text(source.source_text, canonical_label)
        material = json.dumps(
            {
                "canonical_label": canonical_label,
                "lang": effective_lang,
                "source_text": source_text,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest = hmac.new(
            _secret_bytes(self.hmac_secret),
            b"openmed-surrogate-vault:consistency-proof:" + material,
            hashlib.sha256,
        ).hexdigest()
        return f"{HMAC_SCHEME}:{digest}"

    def _surrogate_proof(self, surrogate: str) -> str:
        digest = hmac.new(
            _secret_bytes(self.hmac_secret),
            b"openmed-surrogate-vault:surrogate-proof:" + surrogate.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"{HMAC_SCHEME}:{digest}"

    def _remap_entries(
        self,
        entries: tuple[SurrogateEntry, ...],
        *,
        sources: tuple[SurrogateSource, ...],
        from_key: _EpochKey,
        to_key: _EpochKey,
    ) -> tuple[SurrogateEntry, ...]:
        source_by_key: dict[SurrogateKey, SurrogateSource] = {}
        for source in sources:
            source_by_key[self._key_for_epoch(source, from_key)] = source
            source_by_key.setdefault(
                self._legacy_key_for_epoch(source, from_key), source
            )
        remapped: dict[SurrogateKey, SurrogateEntry] = {}
        missing: list[str] = []
        for entry in entries:
            matched_source = source_by_key.get(entry.key)
            if matched_source is None:
                missing.append(entry.key.text_hash)
                continue
            next_key = self._key_for_epoch(matched_source, to_key)
            next_entry = SurrogateEntry(next_key, entry.surrogate, to_key.key_id)
            current = remapped.get(next_key)
            if current is not None and current.surrogate != next_entry.surrogate:
                raise ValueError("rotation would collapse two surrogates into one key")
            remapped[next_key] = next_entry
        if missing:
            raise ValueError(
                "source catalog is missing entries for text hashes: "
                + ", ".join(sorted(missing))
            )
        return tuple(remapped[key] for key in sorted(remapped))

    def _save_if_file_backed(self) -> None:
        save = getattr(self.store, "save", None)
        if save is not None:
            save()


def _encrypted_entry_payload(
    entry: SurrogateEntry, epoch_key: _EpochKey
) -> dict[str, str]:
    aad = _entry_aad(entry.key, epoch_key.key_id)
    plaintext = entry.surrogate.encode("utf-8")
    nonce = hmac.new(
        epoch_key.encryption_key,
        b"openmed-surrogate-vault:nonce:" + aad,
        hashlib.sha256,
    ).digest()[:_NONCE_BYTES]
    stream = _keystream(epoch_key.encryption_key, nonce, len(plaintext))
    ciphertext = _xor_bytes(plaintext, stream)
    tag = _tag(epoch_key.encryption_key, aad, nonce, ciphertext)
    return {
        **entry.key.to_dict(),
        "key_id": epoch_key.key_id,
        "surrogate_ciphertext": _b64encode(ciphertext),
        "surrogate_nonce": _b64encode(nonce),
        "surrogate_tag": f"{HMAC_SCHEME}:{tag}",
    }


def _decrypt_entry_payload(
    payload: Mapping[str, Any],
    epoch_key: _EpochKey,
) -> SurrogateEntry:
    allowed = {
        "canonical_label",
        "lang",
        "text_hash",
        "key_id",
        "surrogate_ciphertext",
        "surrogate_nonce",
        "surrogate_tag",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(
            "encrypted surrogate vault entries may not contain fields: "
            + ", ".join(sorted(map(str, unknown)))
        )
    key = SurrogateKey.from_dict(payload)
    key_id = str(payload["key_id"])
    if key_id != epoch_key.key_id:
        raise ValueError("surrogate vault entry key_id is not the current epoch")
    aad = _entry_aad(key, key_id)
    ciphertext = _b64decode(str(payload["surrogate_ciphertext"]))
    nonce = _b64decode(str(payload["surrogate_nonce"]))
    persisted_tag = str(payload["surrogate_tag"])
    expected_tag = (
        f"{HMAC_SCHEME}:{_tag(epoch_key.encryption_key, aad, nonce, ciphertext)}"
    )
    if not hmac.compare_digest(persisted_tag, expected_tag):
        raise ValueError("surrogate vault entry authentication failed")
    stream = _keystream(epoch_key.encryption_key, nonce, len(ciphertext))
    surrogate = _xor_bytes(ciphertext, stream).decode("utf-8")
    return SurrogateEntry(key, surrogate, key_id)


def _entry_aad(key: SurrogateKey, key_id: str) -> bytes:
    return json.dumps(
        {
            **key.to_dict(),
            "encryption_scheme": ENCRYPTION_SCHEME,
            "hmac_scheme": HMAC_SCHEME,
            "key_id": key_id,
            "schema_version": SCHEMA_VERSION,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _tag(key: bytes, aad: bytes, nonce: bytes, ciphertext: bytes) -> str:
    mac_key = hmac.new(
        key,
        b"openmed-surrogate-vault:mac",
        hashlib.sha256,
    ).digest()
    return hmac.new(
        mac_key,
        aad + b"\x00" + nonce + b"\x00" + ciphertext,
        hashlib.sha256,
    ).hexdigest()


def _keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    stream_key = hmac.new(
        key,
        b"openmed-surrogate-vault:stream",
        hashlib.sha256,
    ).digest()
    blocks: list[bytes] = []
    counter = 0
    while sum(len(block) for block in blocks) < length:
        blocks.append(
            hmac.new(
                stream_key,
                nonce + counter.to_bytes(8, "big"),
                hashlib.sha256,
            ).digest()
        )
        counter += 1
    return b"".join(blocks)[:length]


def _xor_bytes(left: bytes, right: bytes) -> bytes:
    return bytes(left_byte ^ right_byte for left_byte, right_byte in zip(left, right))


def _b64encode(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _b64decode(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"), validate=True)


def _secret_bytes(secret: str | bytes) -> bytes:
    if isinstance(secret, str):
        if not secret:
            raise ValueError("hmac_secret must be non-empty")
        return secret.encode("utf-8")
    if isinstance(secret, bytes):
        if not secret:
            raise ValueError("hmac_secret must be non-empty")
        return secret
    raise TypeError("hmac_secret must be str or bytes")


def _source(*, source_text: str, label: str, lang: str = "en") -> SurrogateSource:
    return SurrogateSource(
        source_text=str(source_text), label=str(label), lang=str(lang or "en")
    )


def _linkage_source_text(source_text: str, canonical_label: str) -> str:
    """Return cross-script linkage material without persisting source text."""

    if canonical_label != "PERSON":
        return source_text

    # Import lazily so the lightweight vault module does not initialize the
    # broader processing package during module import.
    from ..processing.transliteration import INDIC_SCRIPTS, transliteration_key
    from .script_detect import detect_script

    script = detect_script(source_text)
    if script not in {*INDIC_SCRIPTS, "Latin"}:
        return source_text
    return transliteration_key(source_text)


def _sources(
    sources: Iterable[SurrogateSource | Mapping[str, Any] | tuple[Any, ...]],
) -> tuple[SurrogateSource, ...]:
    return tuple(_coerce_source(source) for source in sources)


def _coerce_source(
    source: SurrogateSource | Mapping[str, Any] | tuple[Any, ...],
) -> SurrogateSource:
    if isinstance(source, SurrogateSource):
        return source
    if isinstance(source, Mapping):
        text = source.get("source_text", source.get("text"))
        if text is None:
            raise ValueError("source mappings must include source_text or text")
        return _source(
            source_text=str(text),
            label=str(source["label"]),
            lang=str(source.get("lang") or "en"),
        )
    if isinstance(source, tuple):
        if len(source) == 2:
            source_text, label = source
            return _source(source_text=str(source_text), label=str(label), lang="en")
        if len(source) == 3:
            source_text, label, lang = source
            return _source(
                source_text=str(source_text),
                label=str(label),
                lang=str(lang or "en"),
            )
    raise TypeError("sources must be SurrogateSource, mapping, or tuple values")


__all__ = [
    "ENCRYPTION_SCHEME",
    "HMAC_SCHEME",
    "LEGACY_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "InMemorySurrogateStore",
    "JsonFileSurrogateStore",
    "SurrogateEntry",
    "SurrogateKey",
    "SurrogateSource",
    "SurrogateVault",
    "VaultConsistencyReport",
    "VaultRotationResult",
]
